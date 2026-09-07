from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from .contracts import Artifact, ArtifactBundle, Provenance
from .interfaces import LayoutEnricher
from .parsers import PyMuPdfParser
from .renderers import (
    MicrosoftExcelChartSubprocessRenderer,
    MicrosoftOfficeSubprocessRenderer,
    MicrosoftOfficeVisualSubprocessRenderer,
    MicrosoftWordVisualSubprocessRenderer,
)


def _normalized_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = re.sub(r"\|?\s*:?-{2,}:?\s*", "", value)
    return re.sub(r"[\s|`]", "", value).lower()


def _segment(artifact: Artifact) -> dict[str, object]:
    provenance = artifact.provenance
    return {
        "page": provenance.page,
        "bbox": provenance.bbox,
        "bbox_original": provenance.bbox_original,
        "locator": provenance.locator,
    }


def _apply_segments(artifact: Artifact, matches: list[Artifact]) -> None:
    if not matches:
        return
    segments = [_segment(item) for item in matches]
    artifact.metadata["layout_segments"] = segments
    artifact.metadata["logical_locator"] = artifact.provenance.locator
    first_page = matches[0].provenance.page
    same_page = [item for item in matches if item.provenance.page == first_page and item.provenance.bbox]
    if same_page:
        boxes = [item.provenance.bbox for item in same_page if item.provenance.bbox]
        artifact.provenance.bbox = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
    artifact.provenance.page = first_page
    artifact.provenance.bbox_original = matches[0].provenance.bbox_original
    artifact.provenance.coord_origin = "TOPLEFT"
    artifact.provenance.locator = f"{artifact.metadata['logical_locator']}|page:{first_page}"


def _image_hash(path: str, size: int = 16) -> int:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((size, size)).getdata())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def _hash_distance(left: int, right: int, bits: int = 256) -> float:
    return (left ^ right).bit_count() / bits


def _best_text_window(
    target: str,
    rendered_text: list[Artifact],
    starts: range,
) -> tuple[float, int, int] | None:
    best: tuple[float, int, int] | None = None
    for start in starts:
        combined = ""
        for end in range(start, min(len(rendered_text), start + 10)):
            combined += _normalized_text(rendered_text[end].text)
            score = SequenceMatcher(None, target, combined).ratio()
            if target in combined or combined in target:
                score = max(score, min(len(target), len(combined)) / max(len(target), len(combined)))
            if best is None or score > best[0]:
                best = (score, start, end + 1)
            if len(combined) > len(target) * 1.8:
                break
    return best


def align_layout(native: ArtifactBundle, rendered: ArtifactBundle) -> ArtifactBundle:
    """Attach rendered PDF coordinates to native semantic artifacts in place."""

    rendered_text = [item for item in rendered.artifacts if item.kind == "text" and item.text]
    cursor = 0
    for artifact in native.artifacts:
        if artifact.kind not in {"text", "table"} or not artifact.text:
            continue
        target = _normalized_text(artifact.text)
        if not target:
            continue
        start_limit = min(len(rendered_text), cursor + 16)
        best = _best_text_window(target, rendered_text, range(max(0, cursor - 1), start_limit))
        # Native Office traversal can place chart-internal text after the rest
        # of a slide. Fall back to a whole-document lookup when reading-order
        # matching cannot find it, or when the global match is clearly better.
        global_best = _best_text_window(target, rendered_text, range(len(rendered_text)))
        if global_best and (
            best is None
            or best[0] < 0.52
            or global_best[0] >= best[0] + 0.08
        ):
            best = global_best
        if best and best[0] >= 0.52:
            _, start, end = best
            _apply_segments(artifact, rendered_text[start:end])
            artifact.metadata["layout_match_score"] = round(best[0], 4)
            cursor = end

    rendered_images = [item for item in rendered.artifacts if item.kind == "image" and item.asset_path]
    unused = set(range(len(rendered_images)))
    for artifact in native.artifacts:
        if artifact.kind != "image" or not artifact.asset_path or not unused:
            continue
        if (
            artifact.metadata.get("rendered_from_layout")
            or artifact.metadata.get("rendered_from_excel_com")
            or artifact.metadata.get("rendered_from_word_com")
        ):
            continue
        try:
            native_hash = _image_hash(artifact.asset_path)
            ranked = sorted(
                (_hash_distance(native_hash, _image_hash(rendered_images[index].asset_path or "")), index)
                for index in unused
            )
        except Exception as exc:
            native.warnings.append(f"Image layout matching failed for {artifact.block_id}: {exc}")
            continue
        distance, index = ranked[0]
        if distance <= 0.28:
            match = rendered_images[index]
            _apply_segments(artifact, [match])
            artifact.metadata["layout_image_distance"] = round(distance, 4)
            unused.remove(index)
    return native


def materialize_visual_tasks(bundle: ArtifactBundle, pdf_path: Path, output_dir: Path) -> None:
    """Rasterize chart/SmartArt tasks from a rendered Office PDF using native bbox."""

    import pymupdf

    image_dir = output_dir / "raw-images"
    image_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    try:
        for artifact in bundle.artifacts:
            if artifact.kind != "visual_task":
                continue
            provenance = artifact.provenance
            if not provenance.page or not provenance.bbox:
                bundle.warnings.append(f"Visual task has no renderable coordinates: {artifact.block_id}")
                continue
            page_index = provenance.page - 1
            if page_index < 0 or page_index >= document.page_count:
                bundle.warnings.append(f"Visual task page is out of range: {artifact.block_id}")
                continue
            page = document.load_page(page_index)
            x0, y0, x1, y1 = provenance.bbox
            clip = pymupdf.Rect(
                x0 * page.rect.width,
                y0 * page.rect.height,
                x1 * page.rect.width,
                y1 * page.rect.height,
            )
            if clip.is_empty or clip.width < 1 or clip.height < 1:
                bundle.warnings.append(f"Visual task has an empty bbox: {artifact.block_id}")
                continue
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False)
            destination = image_dir / f"rendered-{artifact.block_id.replace(':', '-')}.png"
            pixmap.save(destination)
            artifact.kind = "image"
            artifact.asset_path = str(destination.resolve())
            artifact.metadata["rendered_from_layout"] = True
            artifact.metadata["render_source"] = str(pdf_path.resolve())
    finally:
        document.close()


def materialize_word_visuals(
    bundle: ArtifactBundle,
    pdf_path: Path,
    visuals: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Crop Word pictures, icons, charts and text shapes using COM page bounds."""
    import pymupdf

    if not visuals:
        bundle.warnings.append("Word visual inspector produced no objects")
        return
    candidates = [
        artifact for artifact in bundle.artifacts if artifact.kind in {"image", "visual_task"}
    ]
    image_dir = output_dir / "raw-images"
    image_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    added: list[Artifact] = []
    try:
        for index, visual in enumerate(visuals):
            page_number = int(visual.get("page") or 0)
            bounds = visual.get("bounds_points")
            if page_number < 1 or page_number > document.page_count:
                bundle.warnings.append(f"Word visual page is out of range: {index}")
                continue
            if not isinstance(bounds, list) or len(bounds) != 4:
                bundle.warnings.append(f"Word visual has no bounds: {index}")
                continue
            page = document.load_page(page_number - 1)
            clip = pymupdf.Rect(*(float(value) for value in bounds)) & page.rect
            if clip.is_empty or clip.width < 1 or clip.height < 1:
                bundle.warnings.append(f"Word visual has an empty bbox: {index}")
                continue
            if index < len(candidates):
                artifact = candidates[index]
            else:
                artifact = Artifact(
                    block_id=f"{bundle.document_id[:16]}:word-visual-{index:06d}",
                    kind="image",
                    provenance=Provenance(bundle.source_path),
                    reading_order=max(
                        (item.reading_order for item in bundle.artifacts), default=0
                    ) + index + 1,
                )
                added.append(artifact)
            native_path = Path(str(visual.get("path") or ""))
            if native_path.is_file():
                destination = native_path
                render_method = str(visual.get("render_method") or "word-copy-as-picture")
            else:
                destination = image_dir / f"word-visual-{index:06d}.png"
                page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False).save(
                    destination
                )
                render_method = "page-crop-fallback"
            artifact.kind = "image"
            artifact.asset_path = str(destination.resolve())
            artifact.provenance.page = page_number
            artifact.provenance.bbox_original = [float(value) for value in bounds]
            artifact.provenance.bbox = [
                clip.x0 / page.rect.width,
                clip.y0 / page.rect.height,
                clip.x1 / page.rect.width,
                clip.y1 / page.rect.height,
            ]
            artifact.provenance.locator = f"word-visual:{index}"
            artifact.metadata["rendered_from_word_com"] = True
            # Word has already identified the exact visual object.  Do not let
            # nearby-text enrichment attach an unrelated adjacent shape/label.
            artifact.metadata["context_locked"] = True
            artifact.metadata["word_object_type"] = visual.get("object_type")
            artifact.metadata["word_collection"] = visual.get("collection")
            artifact.metadata["office_render_method"] = render_method
            text = str(visual.get("text") or "").strip()
            if text:
                artifact.context = text
                added.append(
                    Artifact(
                        block_id=f"{artifact.block_id}:text",
                        kind="text",
                        provenance=Provenance(
                            source_file=Path(bundle.source_path).name,
                            page=page_number,
                            bbox=list(artifact.provenance.bbox),
                            bbox_original=list(artifact.provenance.bbox_original),
                            locator=f"word-visual-text:{index}",
                        ),
                        text=text,
                        reading_order=artifact.reading_order,
                        parent_block_id=artifact.block_id,
                        metadata={"structure_origin": "word-com-visual"},
                    )
                )
    finally:
        document.close()
    bundle.artifacts.extend(added)


def _new_visual_artifact(bundle: ArtifactBundle, family: str, index: int) -> Artifact:
    return Artifact(
        block_id=f"{bundle.document_id[:16]}:{family}-visual-{index:06d}",
        kind="image",
        provenance=Provenance(source_file=Path(bundle.source_path).name),
        reading_order=max((item.reading_order for item in bundle.artifacts), default=0) + index + 1,
    )


def _office_visual_type(visual: dict[str, object]) -> str:
    hint = str(visual.get("visual_type_hint") or "").lower()
    if hint in {"image", "chart", "diagram", "icon"}:
        return hint
    object_type = int(
        visual.get("effective_object_type")
        or visual.get("contained_object_type")
        or visual.get("object_type")
        or 0
    )
    if object_type == 3:
        return "chart"
    if object_type in {21, 24}:
        return "diagram"
    if object_type in {28, 29}:
        return "icon"
    return "image"


def _normalized_powerpoint_bounds(visual: dict[str, object]) -> list[float] | None:
    bounds = visual.get("bounds_points")
    size = visual.get("container_size_points")
    if (
        not isinstance(bounds, list) or len(bounds) != 4
        or not isinstance(size, list) or len(size) != 2
    ):
        return None
    try:
        width, height = (float(value) for value in size)
        left, top, right, bottom = (float(value) for value in bounds)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return [left / width, top / height, right / width, bottom / height]


def _bbox_iou(left: list[float], right: list[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _powerpoint_candidate_match_score(artifact: Artifact, target: list[float]) -> float:
    box = artifact.provenance.bbox
    if not isinstance(box, list) or len(box) != 4:
        return 0.0
    try:
        normalized = [float(value) for value in box]
    except (TypeError, ValueError):
        return 0.0
    flipped = [normalized[0], 1.0 - normalized[3], normalized[2], 1.0 - normalized[1]]
    return max(_bbox_iou(normalized, target), _bbox_iou(flipped, target))


def _bbox_containment(child: list[float], parent: list[float]) -> float:
    width = max(0.0, child[2] - child[0])
    height = max(0.0, child[3] - child[1])
    area = width * height
    if area <= 0:
        return 0.0
    intersection_width = max(0.0, min(child[2], parent[2]) - max(child[0], parent[0]))
    intersection_height = max(0.0, min(child[3], parent[3]) - max(child[1], parent[1]))
    return intersection_width * intersection_height / area


def _attach_powerpoint_visual_hierarchy(artifacts: list[Artifact]) -> None:
    groups = [
        item for item in artifacts
        if item.metadata.get("office_effective_object_type") in {6, 20, 24}
        and isinstance(item.provenance.bbox, list)
    ]
    for child in artifacts:
        child_box = child.provenance.bbox
        if not isinstance(child_box, list) or len(child_box) != 4:
            continue
        child_area = max(0.0, child_box[2] - child_box[0]) * max(0.0, child_box[3] - child_box[1])
        parents: list[tuple[float, Artifact]] = []
        for parent in groups:
            parent_box = parent.provenance.bbox
            if (
                parent is child
                or parent.provenance.slide != child.provenance.slide
                or not isinstance(parent_box, list) or len(parent_box) != 4
            ):
                continue
            parent_area = max(0.0, parent_box[2] - parent_box[0]) * max(0.0, parent_box[3] - parent_box[1])
            if parent_area <= child_area * 1.05:
                continue
            if _bbox_containment(child_box, parent_box) >= 0.92:
                parents.append((parent_area, parent))
        if not parents:
            continue
        parent = min(parents, key=lambda item: item[0])[1]
        child.metadata["parent_visual_id"] = parent.block_id
        parent.metadata.setdefault("child_visual_ids", []).append(child.block_id)


def _recognize_powerpoint_visuals(
    bundle: ArtifactBundle,
    visuals: list[dict[str, object]],
    candidates: list[Artifact],
) -> list[Artifact]:
    """Merge COM shapes with parser pictures by slide and geometry, never order."""
    added: list[Artifact] = []
    recognized: list[Artifact] = []
    unused = set(range(len(candidates)))
    for index, visual in enumerate(visuals):
        slide = int(visual.get("slide") or 0)
        normalized_bounds = _normalized_powerpoint_bounds(visual)
        ranked: list[tuple[float, int]] = []
        if slide and normalized_bounds is not None:
            for candidate_index in unused:
                candidate = candidates[candidate_index]
                candidate_slide = candidate.provenance.slide or candidate.provenance.page
                if candidate_slide != slide:
                    continue
                score = _powerpoint_candidate_match_score(candidate, normalized_bounds)
                if score > 0:
                    ranked.append((score, candidate_index))
        ranked.sort(reverse=True)
        if ranked and ranked[0][0] >= 0.62:
            match_score, candidate_index = ranked[0]
            artifact = candidates[candidate_index]
            unused.remove(candidate_index)
            artifact.metadata["office_geometry_match_iou"] = round(match_score, 4)
        else:
            artifact = _new_visual_artifact(bundle, "ppt", index)
            added.append(artifact)

        artifact.kind = "visual"
        artifact.visual_type = _office_visual_type(visual)
        artifact.asset_path = None
        artifact.provenance.slide = slide or artifact.provenance.slide
        artifact.provenance.page = artifact.provenance.slide
        artifact.provenance.locator = (
            f"powerpoint-shape:{artifact.provenance.slide}:"
            f"{visual.get('shape_index') or index}"
        )
        bounds = visual.get("bounds_points")
        if isinstance(bounds, list) and len(bounds) == 4:
            artifact.provenance.bbox_original = [float(value) for value in bounds]
        if normalized_bounds is not None:
            artifact.provenance.bbox = normalized_bounds
            artifact.provenance.coord_origin = "TOPLEFT"
        artifact.metadata.update({
            "structure_origin": "office-native-visual",
            "office_object_type": visual.get("object_type"),
            "office_contained_object_type": visual.get("contained_object_type"),
            "office_effective_object_type": visual.get("effective_object_type"),
            "office_collection": visual.get("collection"),
            "office_shape_index": visual.get("shape_index"),
            "office_container_size_points": visual.get("container_size_points"),
            "materialization_status": "deferred-to-index",
        })
        text = str(visual.get("text") or "").strip()
        if text:
            artifact.context = text
            artifact.metadata["context_locked"] = True
        recognized.append(artifact)

    _attach_powerpoint_visual_hierarchy(recognized)
    bundle.artifacts.extend(added)
    return recognized


def recognize_office_visuals(
    bundle: ArtifactBundle, visuals: list[dict[str, object]]
) -> None:
    """Attach Office-native visual identity/location without exporting pixels."""
    candidates = [item for item in bundle.artifacts if item.kind in {"image", "visual_task", "visual"}]
    if bundle.document_type in {"ppt", "pptx"}:
        _recognize_powerpoint_visuals(bundle, visuals, candidates)
        return
    added: list[Artifact] = []
    family = {"doc": "word", "docx": "word", "ppt": "ppt", "pptx": "ppt", "xls": "excel", "xlsx": "excel"}[bundle.document_type]
    for index, visual in enumerate(visuals):
        artifact = candidates[index] if index < len(candidates) else _new_visual_artifact(bundle, family, index)
        if index >= len(candidates):
            added.append(artifact)
        artifact.kind = "visual"
        artifact.visual_type = _office_visual_type(visual)
        artifact.asset_path = None
        artifact.metadata.update({
            "structure_origin": "office-native-visual",
            "office_object_type": visual.get("object_type"),
            "office_contained_object_type": visual.get("contained_object_type"),
            "office_effective_object_type": visual.get("effective_object_type"),
            "office_collection": visual.get("collection"),
            "office_shape_index": visual.get("shape_index") or visual.get("collection_index"),
            "materialization_status": "deferred-to-index",
        })
        bounds = visual.get("bounds_points")
        if isinstance(bounds, list) and len(bounds) == 4:
            artifact.provenance.bbox_original = [float(value) for value in bounds]
        if family == "word":
            artifact.provenance.page = int(visual.get("page") or 0) or artifact.provenance.page
        elif family == "ppt":
            artifact.provenance.slide = int(visual.get("slide") or 0) or artifact.provenance.slide
            artifact.provenance.page = artifact.provenance.slide
        else:
            artifact.provenance.sheet = str(visual.get("sheet") or "") or artifact.provenance.sheet
            artifact.provenance.cell_range = str(visual.get("cell_range") or "") or artifact.provenance.cell_range
        artifact.provenance.locator = f"{family}-visual:{index}"
        text = str(visual.get("text") or "").strip()
        if text:
            artifact.context = text
            artifact.metadata["context_locked"] = True
    bundle.artifacts.extend(added)


def _add_visual_text(
    bundle: ArtifactBundle,
    artifact: Artifact,
    text: str,
    locator: str,
    added: list[Artifact],
) -> None:
    if not text:
        return
    artifact.context = text
    added.append(
        Artifact(
            block_id=f"{artifact.block_id}:text",
            kind="text",
            provenance=Provenance(
                source_file=Path(bundle.source_path).name,
                page=artifact.provenance.page,
                slide=artifact.provenance.slide,
                sheet=artifact.provenance.sheet,
                cell_range=artifact.provenance.cell_range,
                bbox=artifact.provenance.bbox,
                bbox_original=artifact.provenance.bbox_original,
                locator=locator,
            ),
            text=text,
            reading_order=artifact.reading_order,
            parent_block_id=artifact.block_id,
            metadata={"structure_origin": "office-native-visual"},
        )
    )


def materialize_powerpoint_visuals(
    bundle: ArtifactBundle,
    pdf_path: Path,
    visuals: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Use native PowerPoint shape exports, with slide cropping as fallback."""
    import pymupdf

    candidates = [item for item in bundle.artifacts if item.kind in {"image", "visual_task"}]
    added: list[Artifact] = []
    image_dir = output_dir / "raw-images"
    image_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    try:
        for index, visual in enumerate(visuals):
            slide = int(visual.get("slide") or 0)
            artifact = candidates[index] if index < len(candidates) else _new_visual_artifact(bundle, "ppt", index)
            if index >= len(candidates):
                added.append(artifact)
            native_path = Path(str(visual.get("path") or ""))
            bounds = visual.get("bounds_points")
            render_method = str(visual.get("render_method") or "powerpoint-shape-export")
            if native_path.is_file():
                destination = native_path
            elif 1 <= slide <= document.page_count and isinstance(bounds, list) and len(bounds) == 4:
                page = document.load_page(slide - 1)
                clip = pymupdf.Rect(*(float(value) for value in bounds)) & page.rect
                if clip.is_empty:
                    bundle.warnings.append(f"PowerPoint visual has an empty bbox: {index}")
                    continue
                destination = image_dir / f"ppt-visual-{index:06d}.png"
                page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False).save(destination)
                render_method = "page-crop-fallback"
            else:
                bundle.warnings.append(f"PowerPoint visual could not be rendered: {index}")
                continue
            artifact.kind = "image"
            artifact.asset_path = str(destination.resolve())
            artifact.provenance.slide = slide or artifact.provenance.slide
            artifact.provenance.page = slide or artifact.provenance.page
            artifact.provenance.locator = f"powerpoint-shape:{visual.get('shape_index') or index}"
            if isinstance(bounds, list) and len(bounds) == 4:
                artifact.provenance.bbox_original = [float(value) for value in bounds]
            artifact.metadata.update(
                {
                    "rendered_from_powerpoint_com": True,
                    "context_locked": True,
                    "office_render_method": render_method,
                    "powerpoint_shape_index": visual.get("shape_index"),
                    "powerpoint_object_type": visual.get("object_type"),
                }
            )
            _add_visual_text(
                bundle, artifact, str(visual.get("text") or "").strip(),
                f"powerpoint-shape-text:{visual.get('shape_index') or index}", added,
            )
    finally:
        document.close()
    bundle.artifacts.extend(added)


def materialize_excel_visuals(
    bundle: ArtifactBundle,
    visuals: list[dict[str, object]],
) -> None:
    """Use native Excel exports for charts, pictures, SmartArt and groups."""
    candidates = [item for item in bundle.artifacts if item.kind in {"image", "visual_task"}]
    unused = set(range(len(candidates)))
    added: list[Artifact] = []
    for index, visual in enumerate(visuals):
        path = Path(str(visual.get("path") or ""))
        if not path.is_file():
            bundle.warnings.append(f"Excel visual native export failed: {index}")
            continue
        sheet = str(visual.get("sheet") or "")
        same_sheet = [
            candidate_index for candidate_index in unused
            if candidates[candidate_index].provenance.sheet == sheet
        ]
        if same_sheet:
            candidate_index = same_sheet[0]
            artifact = candidates[candidate_index]
            unused.remove(candidate_index)
        elif unused:
            candidate_index = min(unused)
            artifact = candidates[candidate_index]
            unused.remove(candidate_index)
        else:
            artifact = _new_visual_artifact(bundle, "excel", index)
            added.append(artifact)
        artifact.kind = "image"
        artifact.asset_path = str(path.resolve())
        artifact.provenance.sheet = sheet or artifact.provenance.sheet
        artifact.provenance.cell_range = str(visual.get("cell_range") or "") or None
        artifact.provenance.locator = f"excel-shape:{visual.get('shape_index') or index}"
        bounds = visual.get("bounds_points")
        if isinstance(bounds, list) and len(bounds) == 4:
            artifact.provenance.bbox_original = [float(value) for value in bounds]
        artifact.metadata.update(
            {
                "rendered_from_excel_com": True,
                "context_locked": True,
                "office_render_method": str(visual.get("render_method") or "excel-native-export"),
                "excel_shape_index": visual.get("shape_index"),
                "excel_object_type": visual.get("object_type"),
            }
        )
        _add_visual_text(
            bundle, artifact, str(visual.get("text") or "").strip(),
            f"excel-shape-text:{visual.get('shape_index') or index}", added,
        )
    bundle.artifacts.extend(added)


def materialize_excel_charts(
    bundle: ArtifactBundle,
    visuals: list[dict[str, object]],
) -> None:
    """Match Excel chart tasks to images exported directly by Excel COM."""

    unused = set(range(len(visuals)))
    for artifact in bundle.artifacts:
        if artifact.kind != "visual_task":
            continue
        same_sheet = [
            index for index in unused
            if visuals[index].get("sheet") == artifact.provenance.sheet
        ]
        candidates = same_sheet or sorted(unused)
        if not candidates:
            bundle.warnings.append(f"No exported Excel chart for visual task: {artifact.block_id}")
            continue
        index = candidates[0]
        visual = visuals[index]
        path = Path(str(visual["path"]))
        if not path.is_file():
            bundle.warnings.append(f"Exported Excel chart is missing: {path}")
            continue
        artifact.kind = "image"
        artifact.asset_path = str(path.resolve())
        artifact.provenance.sheet = str(visual.get("sheet") or artifact.provenance.sheet or "") or None
        artifact.provenance.cell_range = str(visual.get("cell_range") or "") or None
        bounds = visual.get("bounds_points")
        if isinstance(bounds, list) and len(bounds) == 4:
            artifact.provenance.bbox_original = [float(value) for value in bounds]
        artifact.metadata["rendered_from_excel_com"] = True
        artifact.metadata["excel_chart_index"] = visual.get("chart_index")
        artifact.metadata["excel_sheet_index"] = visual.get("sheet_index")
        artifact.metadata["excel_bounds_points"] = bounds
        unused.remove(index)


class OfficePdfLayoutEnricher(LayoutEnricher):
    name = "microsoft-office-pdf-layout"
    TYPES = {"doc", "docx", "ppt", "pptx", "xls", "xlsx"}

    def __init__(
        self,
        renderer: MicrosoftOfficeSubprocessRenderer,
        pdf_parser: PyMuPdfParser | None = None,
        excel_chart_renderer: MicrosoftExcelChartSubprocessRenderer | None = None,
        word_visual_renderer: MicrosoftWordVisualSubprocessRenderer | None = None,
        office_visual_renderer: MicrosoftOfficeVisualSubprocessRenderer | None = None,
    ) -> None:
        self.renderer = renderer
        self.pdf_parser = pdf_parser or PyMuPdfParser()
        self.excel_chart_renderer = excel_chart_renderer
        self.word_visual_renderer = word_visual_renderer
        self.office_visual_renderer = office_visual_renderer

    def supports(self, document_type: str) -> bool:
        return document_type in self.TYPES

    def enrich(self, bundle: ArtifactBundle, output_dir: Path) -> ArtifactBundle:
        if not self.supports(bundle.document_type):
            return bundle
        render_dir = output_dir / "layout-render"
        outputs = self.renderer.render(Path(bundle.source_path), render_dir)
        if not outputs:
            bundle.warnings.append("Office layout renderer produced no output")
            return bundle
        office_visuals: list[dict[str, object]] | None = None
        if self.office_visual_renderer:
            visual_dir = output_dir / "office-visuals"
            office_visuals = self.office_visual_renderer.render(
                Path(bundle.source_path), visual_dir, export=False
            )
        if office_visuals is not None:
            recognize_office_visuals(bundle, office_visuals)
        if bundle.document_type in {"xls", "xlsx"} and office_visuals is not None:
            pass
        elif bundle.document_type in {"xls", "xlsx"} and self.excel_chart_renderer:
            chart_dir = output_dir / "excel-charts"
            visuals = self.excel_chart_renderer.render(Path(bundle.source_path), chart_dir)
            materialize_excel_charts(bundle, visuals)
        if bundle.document_type in {"doc", "docx"} and office_visuals is not None:
            pass
        elif bundle.document_type in {"doc", "docx"} and self.word_visual_renderer:
            visual_dir = output_dir / "word-visuals"
            visuals = self.word_visual_renderer.render(Path(bundle.source_path), visual_dir)
            materialize_word_visuals(bundle, outputs[0], visuals, output_dir)
        if bundle.document_type in {"ppt", "pptx"} and office_visuals is not None:
            pass
        parsed_dir = render_dir / "parsed"
        rendered = self.pdf_parser.parse(outputs[0], parsed_dir, "pdf")
        return align_layout(bundle, rendered)

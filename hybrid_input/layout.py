from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from .contracts import Artifact, ArtifactBundle
from .interfaces import LayoutEnricher
from .parsers import PyMuPdfParser
from .renderers import MicrosoftExcelChartSubprocessRenderer, MicrosoftOfficeSubprocessRenderer


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
        if artifact.metadata.get("rendered_from_layout") or artifact.metadata.get("rendered_from_excel_com"):
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
    ) -> None:
        self.renderer = renderer
        self.pdf_parser = pdf_parser or PyMuPdfParser()
        self.excel_chart_renderer = excel_chart_renderer

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
        if bundle.document_type in {"xls", "xlsx"} and self.excel_chart_renderer:
            chart_dir = output_dir / "excel-charts"
            visuals = self.excel_chart_renderer.render(Path(bundle.source_path), chart_dir)
            materialize_excel_charts(bundle, visuals)
        materialize_visual_tasks(bundle, outputs[0], output_dir)
        parsed_dir = render_dir / "parsed"
        rendered = self.pdf_parser.parse(outputs[0], parsed_dir, "pdf")
        return align_layout(bundle, rendered)

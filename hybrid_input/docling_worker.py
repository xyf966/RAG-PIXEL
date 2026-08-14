from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hybrid_input.contracts import Artifact, ArtifactBundle, Provenance


def _document_id(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(item: object, document: object, source: Path, order: int, document_type: str) -> Provenance:
    locator = getattr(item, "self_ref", None) or f"item:{order}"
    result = Provenance(source_file=source.name, locator=str(locator))
    values = getattr(item, "prov", None) or []
    if not values:
        return result
    prov = values[0]
    page_no = int(getattr(prov, "page_no", 0) or 0)
    result.page = page_no or None
    if document_type in {"ppt", "pptx"}:
        result.slide = result.page
    if document_type in {"xls", "xlsx"}:
        parent_ref = getattr(item, "parent", None)
        try:
            parent = parent_ref.resolve(document) if parent_ref else None
            result.sheet = getattr(parent, "name", None)
        except Exception:
            pass
    bbox = getattr(prov, "bbox", None)
    if bbox is None:
        return result
    result.bbox_original = [float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)]
    result.coord_origin = str(getattr(bbox, "coord_origin", "TOPLEFT")).split(".")[-1]
    page = getattr(document, "pages", {}).get(page_no)
    size = getattr(page, "size", None)
    if size and size.width and size.height:
        normalized = bbox.to_top_left_origin(size.height).normalized(size)
        result.coord_origin = "TOPLEFT"
        result.bbox = [float(normalized.l), float(normalized.t), float(normalized.r), float(normalized.b)]
    return result


def convert(source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import PictureItem, TableItem, TextItem

    doc_id = _document_id(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "raw-images"
    image_dir.mkdir(parents=True, exist_ok=True)
    document = DocumentConverter().convert(source).document
    artifacts: list[Artifact] = []
    warnings: list[str] = []

    for item, level in document.iterate_items(traverse_pictures=True):
        order = len(artifacts)
        block_id = f"{doc_id[:16]}:{order:06d}"
        provenance = _provenance(item, document, source, order, document_type)
        metadata = {"docling_ref": str(getattr(item, "self_ref", "")), "level": level}
        if isinstance(item, TextItem):
            text = (item.text or "").strip()
            if text:
                artifacts.append(Artifact(block_id, "text", provenance, text=text, reading_order=order, metadata=metadata))
        elif isinstance(item, TableItem):
            text = item.export_to_markdown(doc=document).strip()
            artifacts.append(Artifact(block_id, "table", provenance, text=text, reading_order=order, metadata=metadata))
        elif isinstance(item, PictureItem):
            picture = item.get_image(document)
            context = item.caption_text(document).strip() or None
            if picture is None:
                metadata["render_required"] = True
                artifacts.append(
                    Artifact(
                        block_id,
                        "visual_task",
                        provenance,
                        context=context,
                        reading_order=order,
                        metadata=metadata,
                    )
                )
                continue
            destination = image_dir / f"image-{order:06d}.png"
            picture.convert("RGB").save(destination, format="PNG")
            artifacts.append(Artifact(block_id, "image", provenance, asset_path=str(destination.resolve()), context=context, reading_order=order, metadata=metadata))

    return ArtifactBundle(doc_id, str(source.resolve()), document_type, "docling-subprocess", artifacts, warnings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--document-type", required=True)
    args = parser.parse_args()
    bundle = convert(args.input, args.output, args.document_type)
    args.result.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def _short_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    size = ctypes.windll.kernel32.GetShortPathNameW(str(resolved), None, 0)
    if not size:
        return str(resolved)
    buffer = ctypes.create_unicode_buffer(size)
    ctypes.windll.kernel32.GetShortPathNameW(str(resolved), buffer, size)
    return buffer.value or str(resolved)


def _java_major(executable: Path) -> int:
    result = subprocess.run(
        [str(executable), "-version"], capture_output=True, text=True, errors="replace"
    )
    text = result.stderr or result.stdout
    match = re.search(r'version\s+"([0-9]+)(?:\.([0-9]+))?', text)
    if not match:
        return 0
    first = int(match.group(1))
    return int(match.group(2) or 0) if first == 1 else first


def _discover_java() -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("HYBRID_PDF_JAVA")
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path(r"C:\Program Files\PDFsam Basic\runtime\bin\java.exe"))
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home) / "bin" / "java.exe")
    command = shutil.which("java")
    if command:
        candidates.append(Path(command))
    for candidate in candidates:
        if candidate.is_file() and _java_major(candidate) >= 11:
            return candidate
    raise RuntimeError(
        "OpenDataLoader PDF requires Java 11+; set HYBRID_PDF_JAVA to an existing java.exe"
    )


def _node_text(node: Any) -> str:
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, list):
        return " ".join(filter(None, (_node_text(item) for item in node))).strip()
    if not isinstance(node, dict):
        return ""
    direct = node.get("content")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for key in ("kids", "list items", "cells", "rows"):
        text = _node_text(node.get(key, []))
        if text:
            return text
    return ""


def _table_data(node: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    matrix: list[list[str]] = []
    declared_width = int(node.get("number of columns") or 0)
    for row in node.get("rows", []):
        values: list[str] = [""] * declared_width
        cells: list[dict[str, Any]] = []
        next_column = 1
        for cell in row.get("cells", []):
            text = _node_text(cell).replace("|", "\\|")
            column = int(cell.get("column number") or next_column)
            column_span = max(1, int(cell.get("column span") or 1))
            required_width = column + column_span - 1
            if required_width > len(values):
                values.extend([""] * (required_width - len(values)))
            values[column - 1] = text
            next_column = required_width + 1
            cells.append(
                {
                    "row": cell.get("row number"),
                    "column": column,
                    "row_span": cell.get("row span", 1),
                    "column_span": column_span,
                    "text": text,
                }
            )
        matrix.append(values)
        rows.append({"row": row.get("row number"), "cells": cells})
    if not matrix:
        return "", rows
    width = max(declared_width, *(len(row) for row in matrix))
    matrix = [row + [""] * (width - len(row)) for row in matrix]
    lines = ["| " + " | ".join(row) + " |" for row in matrix]
    lines.insert(1, "| " + " | ".join("---" for _ in range(width)) + " |")
    return "\n".join(lines), rows


def _list_text(node: dict[str, Any]) -> str:
    items = node.get("list items", [])
    return "\n".join(f"- {_node_text(item)}" for item in items if _node_text(item))


def _bbox(
    node: dict[str, Any], page_sizes: dict[int, tuple[float, float]]
) -> tuple[list[float] | None, list[float] | None, list[float] | None]:
    raw = node.get("bounding box")
    page = int(node.get("page number") or 0)
    if not isinstance(raw, list) or len(raw) != 4 or page not in page_sizes:
        return None, None, None
    source_box = [float(value) for value in raw]
    width, height = page_sizes[page]
    left, bottom, right, top = source_box
    top_left = [left, height - top, right, height - bottom]
    normalized = [top_left[0] / width, top_left[1] / height, top_left[2] / width, top_left[3] / height]
    return normalized, top_left, source_box


def _resolve_image(source_value: Any, raw_dir: Path) -> Path | None:
    if not source_value:
        return None
    value = Path(str(source_value))
    candidates = [value, raw_dir / value.name, raw_dir.parent / value]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _crop_visual(document: Any, page_no: int, top_left_box: list[float], destination: Path) -> Path:
    import pymupdf

    page = document.load_page(page_no - 1)
    clip = pymupdf.Rect(*top_left_box)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False)
    pixmap.save(destination)
    return destination.resolve()


def _convert_structure(source: Path, structure: dict[str, Any], output_dir: Path) -> ArtifactBundle:
    import pymupdf

    doc_id = _document_id(source)
    raw_dir = output_dir / "raw-images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(source)
    try:
        page_sizes = {
            index + 1: (float(page.rect.width), float(page.rect.height))
            for index, page in enumerate(document)
        }
        artifacts: list[Artifact] = []
        warnings: list[str] = []
        heading_stack: dict[int, str] = {}
        for node in structure.get("kids", []):
            if not isinstance(node, dict):
                continue
            semantic_type = str(node.get("type") or "unknown").lower()
            order = len(artifacts)
            block_id = f"{doc_id[:16]}:{order:06d}"
            page_no = int(node.get("page number") or 0) or None
            normalized, original, source_box = _bbox(node, page_sizes)
            provenance = Provenance(
                source_file=source.name,
                page=page_no,
                bbox=normalized,
                bbox_original=original,
                coord_origin="TOPLEFT",
                locator=f"opendataloader:{node.get('id', order)}",
            )
            metadata: dict[str, Any] = {
                "semantic_type": semantic_type,
                "pdfua_tag": node.get("pdfua_tag"),
                "opendataloader_id": node.get("id"),
                "bbox_bottomleft": source_box,
            }
            heading_level = int(node.get("heading level") or 0)
            if heading_level:
                metadata["heading_level"] = heading_level
            parent_id = None
            if heading_stack:
                valid = [level for level in heading_stack if not heading_level or level < heading_level]
                if valid:
                    parent_id = heading_stack[max(valid)]

            if semantic_type == "table":
                text, rows = _table_data(node)
                metadata.update(
                    {
                        "rows": node.get("number of rows"),
                        "columns": node.get("number of columns"),
                        "table_cells": rows,
                        "structure_origin": "visual_inference",
                        "span_reliability": "inferred",
                    }
                )
                artifact = Artifact(block_id, "table", provenance, text=text, reading_order=order, parent_block_id=parent_id, metadata=metadata)
            elif semantic_type == "image":
                image_path = _resolve_image(node.get("source"), raw_dir)
                if image_path is None and page_no and original:
                    image_path = _crop_visual(document, page_no, original, raw_dir / f"crop-{order:06d}.png")
                    metadata["cropped_from_pdf"] = True
                artifact = Artifact(block_id, "image", provenance, asset_path=str(image_path) if image_path else None, reading_order=order, parent_block_id=parent_id, metadata=metadata)
                if image_path is None:
                    artifact.kind = "visual_task"
                    metadata["render_required"] = True
                    warnings.append(f"OpenDataLoader image has no extractable asset: {block_id}")
            else:
                text = _list_text(node) if semantic_type == "list" else _node_text(node)
                if not text:
                    continue
                artifact = Artifact(block_id, "text", provenance, text=text, reading_order=order, parent_block_id=parent_id, metadata=metadata)
            artifacts.append(artifact)
            if semantic_type == "heading" and heading_level:
                heading_stack = {level: value for level, value in heading_stack.items() if level < heading_level}
                heading_stack[heading_level] = block_id
        return ArtifactBundle(doc_id, str(source.resolve()), "pdf", "opendataloader-pdf", artifacts, warnings)
    finally:
        document.close()


def convert(source: Path, output_dir: Path) -> ArtifactBundle:
    import opendataloader_pdf

    source = source.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = output_dir / "opendataloader"
    raw_dir = output_dir / "raw-images"
    structure_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    jar = Path(opendataloader_pdf.__file__).resolve().parent / "jar" / "opendataloader-pdf-cli.jar"
    java = _discover_java()
    command = [
        str(java), "-jar", _short_path(jar), _short_path(source),
        "--output-dir", _short_path(structure_dir),
        "--format", "json",
        "--image-output", "external",
        "--image-format", "png",
        "--image-dir", _short_path(raw_dir),
        "--reading-order", "xycut",
        "--table-method", "cluster",
        "--use-struct-tree",
        "--quiet",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"OpenDataLoader PDF failed ({result.returncode}): {detail}")
    candidates = sorted(structure_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("OpenDataLoader PDF produced no structure JSON")
    structure = json.loads(candidates[0].read_text(encoding="utf-8"))
    return _convert_structure(source, structure, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    bundle = convert(args.input, args.output)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detector import SignatureDetector
from .parsers import DoclingSubprocessParser, ImageFileParser, PlainTextParser, PyMuPdfParser
from .pipeline import build_default_pipeline, discover_docling_python
from .renderers import MicrosoftOfficeSubprocessRenderer


def _capabilities() -> dict[str, object]:
    docling_python = discover_docling_python()
    return {
        "docling_python": str(docling_python),
        "docling_available": docling_python.is_file(),
        "supported_extensions": [
            ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
            ".txt", ".md", ".html", ".png", ".jpg", ".jpeg", ".webp",
        ],
        "vision_processor": "deferred",
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Pluggable hybrid document ingestion")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities")
    detect = commands.add_parser("detect")
    detect.add_argument("input", type=Path)
    parse = commands.add_parser("parse")
    parse.add_argument("input", type=Path)
    parse.add_argument("--output", required=True, type=Path)
    parse.add_argument("--parser", choices=["plain-text", "image-file", "pymupdf", "docling-subprocess"])
    parse.add_argument("--docling-python", type=Path)
    render = commands.add_parser("render")
    render.add_argument("input", type=Path)
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--office-python", type=Path)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("input", type=Path)
    ingest.add_argument("--output", required=True, type=Path)
    ingest.add_argument("--docling-python", type=Path)
    args = parser.parse_args()
    if args.command == "capabilities":
        print(json.dumps(_capabilities(), ensure_ascii=False, indent=2))
        return
    detector = SignatureDetector()
    if args.command == "detect":
        print(json.dumps({"path": str(args.input.resolve()), "document_type": detector.detect(args.input)}, ensure_ascii=False, indent=2))
        return
    if args.command == "render":
        renderer = MicrosoftOfficeSubprocessRenderer(args.office_python or discover_docling_python())
        outputs = renderer.render(args.input, args.output)
        print(json.dumps({"provider": renderer.name, "outputs": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))
        return
    if args.command == "parse":
        document_type = detector.detect(args.input)
        parsers = {
            "plain-text": PlainTextParser(),
            "image-file": ImageFileParser(),
            "pymupdf": PyMuPdfParser(),
            "docling-subprocess": DoclingSubprocessParser(args.docling_python or discover_docling_python()),
        }
        default = {
            "pdf": "pymupdf", "txt": "plain-text", "md": "plain-text", "html": "plain-text", "image": "image-file",
        }.get(document_type, "docling-subprocess")
        selected = parsers[args.parser or default]
        if not selected.supports(document_type):
            parser.error(f"Parser {selected.name} does not support {document_type}")
        bundle = selected.parse(args.input.resolve(), args.output.resolve(), document_type)
        result_path = args.output / "artifact-bundle.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))
        return
    result = build_default_pipeline(docling_python=args.docling_python).ingest(args.input, args.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

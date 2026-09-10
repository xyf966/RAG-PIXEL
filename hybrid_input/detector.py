from __future__ import annotations

import zipfile
from pathlib import Path

from .interfaces import DocumentDetector


EXTENSION_TYPES = {
    ".pdf": "pdf",
    ".doc": "doc",
    ".docx": "docx",
    ".docm": "docx",
    ".ppt": "ppt",
    ".pptx": "pptx",
    ".pptm": "pptx",
    ".xls": "xls",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".txt": "txt",
    ".md": "md",
    ".html": "html",
    ".htm": "html",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
}


class SignatureDetector(DocumentDetector):
    """Extension routing with lightweight signature validation."""

    def detect(self, source: Path) -> str:
        if not source.is_file():
            raise FileNotFoundError(source)
        expected = EXTENSION_TYPES.get(source.suffix.lower())
        if expected is None:
            raise ValueError(f"Unsupported document extension: {source.suffix}")
        head = source.read_bytes()[:16]
        if expected == "pdf" and not head.startswith(b"%PDF-"):
            raise ValueError(f"File extension says PDF but signature does not: {source}")
        if expected in {"docx", "pptx", "xlsx"}:
            if not zipfile.is_zipfile(source):
                raise ValueError(f"Invalid OOXML container: {source}")
            with zipfile.ZipFile(source) as archive:
                names = set(archive.namelist())
            marker = {"docx": "word/document.xml", "pptx": "ppt/presentation.xml", "xlsx": "xl/workbook.xml"}[expected]
            if marker not in names:
                raise ValueError(f"OOXML content does not match {expected}: {source}")
        if expected in {"doc", "ppt", "xls"} and not head.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
            raise ValueError(f"Invalid legacy Office container: {source}")
        return expected

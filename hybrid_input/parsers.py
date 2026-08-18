from __future__ import annotations

import atexit
import hashlib
import html
import json
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from .contracts import Artifact, ArtifactBundle, Provenance
from .interfaces import DocumentParser


def document_id(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_id(doc_id: str, order: int) -> str:
    return f"{doc_id[:16]}:{order:06d}"


class PlainTextParser(DocumentParser):
    name = "plain-text"

    def supports(self, document_type: str) -> bool:
        return document_type in {"txt", "md", "html"}

    @staticmethod
    def _read(source: Path) -> str:
        raw = source.read_bytes()
        for encoding in ("utf-8-sig", "utf-16", "gb18030"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
        doc_id = document_id(source)
        content = self._read(source)
        if document_type == "html":
            content = re.sub(r"<script\b[^>]*>.*?</script>", "", content, flags=re.I | re.S)
            content = re.sub(r"<style\b[^>]*>.*?</style>", "", content, flags=re.I | re.S)
            content = html.unescape(re.sub(r"<[^>]+>", "\n", content))
        artifacts: list[Artifact] = []
        lines = content.splitlines()
        start = 0
        buffer: list[str] = []

        def flush(end_line: int) -> None:
            nonlocal start, buffer
            text = "\n".join(buffer).strip()
            if text:
                order = len(artifacts)
                artifacts.append(
                    Artifact(
                        block_id=_block_id(doc_id, order),
                        kind="text",
                        text=text,
                        reading_order=order,
                        provenance=Provenance(
                            source_file=source.name,
                            line_start=start + 1,
                            line_end=end_line,
                            locator=f"lines:{start + 1}-{end_line}",
                        ),
                    )
                )
            buffer = []

        for index, line in enumerate(lines):
            if not buffer:
                start = index
            if not line.strip() and buffer:
                flush(index)
            else:
                buffer.append(line)
        if buffer:
            flush(len(lines))
        return ArtifactBundle(doc_id, str(source.resolve()), document_type, self.name, artifacts)


class ImageFileParser(DocumentParser):
    name = "image-file"

    def supports(self, document_type: str) -> bool:
        return document_type == "image"

    def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
        doc_id = document_id(source)
        assets = output_dir / "raw-images"
        assets.mkdir(parents=True, exist_ok=True)
        destination = assets / source.name
        shutil.copy2(source, destination)
        artifact = Artifact(
            block_id=_block_id(doc_id, 0),
            kind="image",
            asset_path=str(destination.resolve()),
            reading_order=0,
            provenance=Provenance(source_file=source.name, locator="whole-image"),
        )
        return ArtifactBundle(doc_id, str(source.resolve()), document_type, self.name, [artifact])


class PyMuPdfParser(DocumentParser):
    name = "pymupdf"

    def supports(self, document_type: str) -> bool:
        return document_type == "pdf"

    def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError("PyMuPDF parser selected but pymupdf is unavailable") from exc

        doc_id = document_id(source)
        image_dir = output_dir / "raw-images"
        image_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[Artifact] = []
        warnings: list[str] = []
        document = pymupdf.open(source)
        try:
            for page_index, page in enumerate(document):
                page_data = page.get_text("dict", sort=True)
                width = float(page_data.get("width") or page.rect.width or 1)
                height = float(page_data.get("height") or page.rect.height or 1)
                for block in page_data.get("blocks", []):
                    order = len(artifacts)
                    bbox_raw = [float(value) for value in block.get("bbox", (0, 0, width, height))]
                    bbox = [bbox_raw[0] / width, bbox_raw[1] / height, bbox_raw[2] / width, bbox_raw[3] / height]
                    provenance = Provenance(
                        source_file=source.name,
                        page=page_index + 1,
                        bbox=bbox,
                        bbox_original=bbox_raw,
                        locator=f"page:{page_index + 1}/block:{block.get('number', order)}",
                    )
                    if block.get("type") == 0:
                        text = "\n".join(
                            "".join(span.get("text", "") for span in line.get("spans", []))
                            for line in block.get("lines", [])
                        ).strip()
                        if text:
                            artifacts.append(Artifact(_block_id(doc_id, order), "text", provenance, text=text, reading_order=order))
                    elif block.get("type") == 1 and block.get("image"):
                        extension = block.get("ext") or "png"
                        destination = image_dir / f"page-{page_index + 1:04d}-block-{order:06d}.{extension}"
                        destination.write_bytes(block["image"])
                        artifacts.append(
                            Artifact(
                                _block_id(doc_id, order),
                                "image",
                                provenance,
                                asset_path=str(destination.resolve()),
                                reading_order=order,
                            )
                        )
        finally:
            document.close()
        if not artifacts:
            warnings.append("PDF parser produced no text or image blocks")
        return ArtifactBundle(doc_id, str(source.resolve()), document_type, self.name, artifacts, warnings)


class OpenDataLoaderPdfSubprocessParser(DocumentParser):
    name = "opendataloader-pdf-hybrid"

    def __init__(self, python_executable: Path, worker_script: Path | None = None) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = worker_script or Path(__file__).with_name("opendataloader_worker.py")
        self._hybrid_process: subprocess.Popen[str] | None = None
        self._hybrid_url: str | None = None
        atexit.register(self.close)

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def _ensure_hybrid_server(self) -> str:
        if self._hybrid_process and self._hybrid_process.poll() is None and self._hybrid_url:
            return self._hybrid_url
        port = self._free_port()
        command = [
            str(self.python_executable),
            "-X", "utf8",
            "-m", "opendataloader_pdf.hybrid_server",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "error",
            "--device", "cpu",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._hybrid_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=creationflags,
        )
        self._hybrid_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 300
        last_error = "server did not become ready"
        while time.monotonic() < deadline:
            if self._hybrid_process.poll() is not None:
                raise RuntimeError(
                    f"OpenDataLoader Hybrid server exited with status {self._hybrid_process.returncode}"
                )
            try:
                with urllib.request.urlopen(f"{self._hybrid_url}/health", timeout=2) as response:
                    if response.status == 200:
                        return self._hybrid_url
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        self.close()
        raise RuntimeError(f"OpenDataLoader Hybrid server startup timed out: {last_error}")

    def close(self) -> None:
        process = self._hybrid_process
        self._hybrid_process = None
        self._hybrid_url = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def supports(self, document_type: str) -> bool:
        return document_type == "pdf"

    def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
        if not self.python_executable.is_file():
            raise RuntimeError(f"OpenDataLoader Python environment not found: {self.python_executable}")
        hybrid_url = self._ensure_hybrid_server()
        result_path = output_dir / "opendataloader-result.json"
        command = [
            str(self.python_executable), str(self.worker_script),
            "--input", str(source.resolve()),
            "--output", str(output_dir.resolve()),
            "--result", str(result_path.resolve()),
            "--hybrid-url", hybrid_url,
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"OpenDataLoader worker failed ({result.returncode}): {detail}")
        return ArtifactBundle.from_dict(json.loads(result_path.read_text(encoding="utf-8")))


class OpenPyxlSubprocessParser(DocumentParser):
    name = "openpyxl-subprocess"

    def __init__(self, python_executable: Path, worker_script: Path | None = None) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = worker_script or Path(__file__).with_name("openpyxl_worker.py")

    def supports(self, document_type: str) -> bool:
        return document_type == "xlsx"

    def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
        if not self.python_executable.is_file():
            raise RuntimeError(f"openpyxl Python environment not found: {self.python_executable}")
        result_path = output_dir / "openpyxl-result.json"
        command = [
            str(self.python_executable), str(self.worker_script),
            "--input", str(source.resolve()),
            "--output", str(output_dir.resolve()),
            "--result", str(result_path.resolve()),
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"openpyxl worker failed ({result.returncode}): {detail}")
        return ArtifactBundle.from_dict(json.loads(result_path.read_text(encoding="utf-8")))


class FallbackDocumentParser(DocumentParser):
    """Try one parser and fall back without coupling either implementation."""

    name = "fallback-auto"

    def __init__(self, primary: DocumentParser, fallback: DocumentParser, name: str = "pdf-structured-auto") -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = name

    def supports(self, document_type: str) -> bool:
        return self.primary.supports(document_type) and self.fallback.supports(document_type)

    def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
        try:
            return self.primary.parse(source, output_dir, document_type)
        except Exception as exc:
            bundle = self.fallback.parse(source, output_dir, document_type)
            bundle.warnings.insert(0, f"Structured PDF parser failed; used {self.fallback.name}: {exc}")
            return bundle


class DoclingSubprocessParser(DocumentParser):
    name = "docling-subprocess"
    TYPES = {"doc", "docx", "ppt", "pptx", "xls", "xlsx"}

    def __init__(self, python_executable: Path, worker_script: Path | None = None) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = worker_script or Path(__file__).with_name("docling_worker.py")

    def supports(self, document_type: str) -> bool:
        return document_type in self.TYPES

    def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
        if not self.python_executable.is_file():
            raise RuntimeError(f"Docling Python environment not found: {self.python_executable}")
        result_path = output_dir / "docling-result.json"
        command = [
            str(self.python_executable),
            str(self.worker_script),
            "--input",
            str(source.resolve()),
            "--output",
            str(output_dir.resolve()),
            "--result",
            str(result_path.resolve()),
            "--document-type",
            document_type,
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Docling worker failed ({result.returncode}): {detail}")
        return ArtifactBundle.from_dict(json.loads(result_path.read_text(encoding="utf-8")))

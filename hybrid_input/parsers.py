from __future__ import annotations

import atexit
import hashlib
import html
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import TextIO

from .contracts import Artifact, ArtifactBundle, Provenance
from .interfaces import DocumentParser


class ScannedPdfOcrError(RuntimeError):
    """A scan cannot be represented safely because its OCR path failed."""


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
        # Digital and scanned PDFs need different backend settings.  Keep one
        # resident server for each mode so mixed document batches do not
        # repeatedly reload Docling/OCR models.
        self._hybrid_processes: dict[bool, subprocess.Popen[str]] = {}
        self._hybrid_urls: dict[bool, str] = {}
        self._hybrid_log_streams: dict[bool, TextIO] = {}
        self._hybrid_log_paths: dict[bool, Path] = {}
        atexit.register(self.close)

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    @staticmethod
    def _requires_force_ocr(source: Path) -> bool:
        """Return True when a PDF has no useful selectable-text layer."""
        import pymupdf

        document = pymupdf.open(source)
        try:
            if document.page_count == 0:
                return False
            meaningful_pages = 0
            total_characters = 0
            for page in document:
                value = "".join(page.get_text("text").split())
                total_characters += len(value)
                if len(value) >= 20:
                    meaningful_pages += 1
            required_pages = max(1, (document.page_count + 1) // 2)
            return meaningful_pages < required_pages or total_characters < 20 * document.page_count
        finally:
            document.close()

    def _hybrid_server_command(self, port: int, force_ocr: bool) -> list[str]:
        command = [
            str(self.python_executable),
            "-X", "utf8",
            "-m", "hybrid_input.opendataloader_hybrid_server",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "error",
            "--device", "cpu",
        ]
        if force_ocr:
            # Use the bundled RapidOCR models for deterministic Chinese OCR.
            # EasyOCR otherwise tries to initialize a per-user model directory
            # and may fail before recognition starts.
            command.extend(
                [
                    "--force-ocr",
                    "--ocr-engine", "rapidocr",
                    "--ocr-lang", "chinese",
                ]
            )
        return command

    def _ensure_hybrid_server(self, force_ocr: bool = False) -> str:
        process = self._hybrid_processes.get(force_ocr)
        url = self._hybrid_urls.get(force_ocr)
        if process and process.poll() is None and url:
            return url
        port = self._free_port()
        command = self._hybrid_server_command(port, force_ocr)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        package_root = Path(__file__).resolve().parents[1]
        configured_hf_home = os.environ.get("HF_HOME")
        model_root = (
            Path(configured_hf_home).expanduser().resolve()
            if configured_hf_home
            else package_root / "PixelRAG-Studio-Data" / "models"
        )
        huggingface_cache = model_root
        easyocr_cache = model_root / "easyocr"
        huggingface_cache.mkdir(parents=True, exist_ok=True)
        easyocr_cache.mkdir(parents=True, exist_ok=True)
        server_environment = os.environ.copy()
        existing_python_path = server_environment.get("PYTHONPATH", "")
        server_environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(package_root), existing_python_path) if value
        )
        server_environment.update(
            {
                "HF_HOME": str(huggingface_cache),
                "HUGGINGFACE_HUB_CACHE": str(huggingface_cache / "hub"),
                "EASYOCR_MODULE_PATH": str(easyocr_cache),
            }
        )
        studio_log = os.environ.get("PIXELRAG_STUDIO_LOG")
        log_dir = Path(studio_log).resolve().parent if studio_log else model_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (
            "opendataloader-hybrid-ocr.log"
            if force_ocr
            else "opendataloader-hybrid-digital.log"
        )
        log_stream = log_path.open("w", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
            env=server_environment,
        )
        url = f"http://127.0.0.1:{port}"
        self._hybrid_processes[force_ocr] = process
        self._hybrid_urls[force_ocr] = url
        self._hybrid_log_streams[force_ocr] = log_stream
        self._hybrid_log_paths[force_ocr] = log_path
        deadline = time.monotonic() + 300
        last_error = "server did not become ready"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log_stream.flush()
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
                self._close_hybrid_server(force_ocr)
                raise RuntimeError(
                    f"OpenDataLoader Hybrid server exited with status {process.returncode}: "
                    f"{detail or 'no diagnostic output'}"
                )
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                    if response.status == 200:
                        return url
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        self._close_hybrid_server(force_ocr)
        raise RuntimeError(
            f"OpenDataLoader Hybrid server startup timed out: {last_error}; "
            f"log={log_path}"
        )

    def _close_hybrid_server(self, force_ocr: bool) -> None:
        process = self._hybrid_processes.pop(force_ocr, None)
        self._hybrid_urls.pop(force_ocr, None)
        log_stream = self._hybrid_log_streams.pop(force_ocr, None)
        self._hybrid_log_paths.pop(force_ocr, None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if log_stream is not None and not log_stream.closed:
            log_stream.close()

    def close(self) -> None:
        for force_ocr in tuple(self._hybrid_processes):
            self._close_hybrid_server(force_ocr)

    def supports(self, document_type: str) -> bool:
        return document_type == "pdf"

    def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
        if not self.python_executable.is_file():
            raise RuntimeError(f"OpenDataLoader Python environment not found: {self.python_executable}")
        force_ocr = self._requires_force_ocr(source)
        try:
            hybrid_url = self._ensure_hybrid_server(force_ocr)
        except Exception as exc:
            if force_ocr:
                raise ScannedPdfOcrError(f"Scanned PDF OCR server failed: {exc}") from exc
            raise
        result_path = output_dir / "opendataloader-result.json"
        command = [
            str(self.python_executable), str(self.worker_script),
            "--input", str(source.resolve()),
            "--output", str(output_dir.resolve()),
            "--result", str(result_path.resolve()),
            "--hybrid-url", hybrid_url,
        ]
        if force_ocr:
            command.append("--strict-hybrid")
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            if force_ocr:
                raise ScannedPdfOcrError(
                    f"Scanned PDF Hybrid OCR failed ({result.returncode}): {detail or 'no diagnostic output'}"
                )
            raise RuntimeError(f"OpenDataLoader worker failed ({result.returncode}): {detail}")
        bundle = ArtifactBundle.from_dict(json.loads(result_path.read_text(encoding="utf-8")))
        mode = "scanned-force-ocr" if force_ocr else "digital-native-text"
        for artifact in bundle.artifacts:
            artifact.metadata.setdefault("pdf_recognition_mode", mode)
        return bundle


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
            if isinstance(exc, ScannedPdfOcrError):
                raise
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

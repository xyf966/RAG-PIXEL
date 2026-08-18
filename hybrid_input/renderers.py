from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .interfaces import DocumentRenderer


class NoOpRenderer(DocumentRenderer):
    name = "none"

    def supports(self, document_type: str) -> bool:
        return True

    def render(self, source: Path, output_dir: Path) -> list[Path]:
        return []


class MicrosoftOfficeSubprocessRenderer(DocumentRenderer):
    """Office renderer isolated behind a worker and a JSON result contract."""

    name = "microsoft-office-com"
    TYPES = {"doc", "docx", "ppt", "pptx", "xls", "xlsx"}

    def __init__(self, python_executable: Path, worker_script: Path | None = None) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = worker_script or Path(__file__).with_name("office_worker.py")

    def supports(self, document_type: str) -> bool:
        return document_type in self.TYPES

    def render(self, source: Path, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "office-render-result.json"
        command = [
            str(self.python_executable), str(self.worker_script),
            "--input", str(source.resolve()),
            "--output", str(output_dir.resolve()),
            "--result", str(result_path.resolve()),
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Office renderer failed ({result.returncode}): {detail}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return [Path(item) for item in payload.get("outputs", [])]


class MicrosoftExcelChartSubprocessRenderer:
    """Exports native Excel chart objects without relying on print layout."""

    name = "microsoft-excel-chart-export"

    def __init__(self, python_executable: Path, worker_script: Path | None = None) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = worker_script or Path(__file__).with_name("office_worker.py")

    def render(self, source: Path, output_dir: Path) -> list[dict[str, object]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "excel-chart-result.json"
        command = [
            str(self.python_executable), str(self.worker_script),
            "--input", str(source.resolve()),
            "--output", str(output_dir.resolve()),
            "--result", str(result_path.resolve()),
            "--mode", "excel-charts",
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Excel chart renderer failed ({result.returncode}): {detail}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return list(payload.get("visuals", []))


class MicrosoftWordVisualSubprocessRenderer:
    """Exports Word visual objects through an isolated COM worker."""

    name = "microsoft-word-visual-inspector"

    def __init__(self, python_executable: Path, worker_script: Path | None = None) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = worker_script or Path(__file__).with_name("office_worker.py")

    def render(self, source: Path, output_dir: Path) -> list[dict[str, object]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "word-visual-result.json"
        command = [
            str(self.python_executable), str(self.worker_script),
            "--input", str(source.resolve()),
            "--output", str(output_dir.resolve()),
            "--result", str(result_path.resolve()),
            "--mode", "word-visuals",
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Word visual inspector failed ({result.returncode}): {detail}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return list(payload.get("visuals", []))


class MicrosoftOfficeVisualSubprocessRenderer:
    """Unified native visual-object exporter for Word, Excel and PowerPoint."""

    name = "microsoft-office-native-visuals"

    def __init__(self, python_executable: Path, worker_script: Path | None = None) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = worker_script or Path(__file__).with_name("office_worker.py")

    def render(self, source: Path, output_dir: Path, *, export: bool = True) -> list[dict[str, object]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "office-visual-result.json"
        command = [
            str(self.python_executable), str(self.worker_script),
            "--input", str(source.resolve()),
            "--output", str(output_dir.resolve()),
            "--result", str(result_path.resolve()),
            "--mode", "office-visuals" if export else "office-inspect",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Office visual renderer failed ({result.returncode}): {detail}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return list(payload.get("visuals", []))

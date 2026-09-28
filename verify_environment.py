from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "PIL": "Pillow",
    "docling": "Docling",
    "faiss": "FAISS",
    "numpy": "NumPy",
    "openpyxl": "openpyxl",
    "opendataloader_pdf": "OpenDataLoader PDF",
    "pdf2image": "pdf2image",
    "pixelrag_embed": "PixelRAG embedding",
    "pymupdf": "PyMuPDF",
    "torch": "PyTorch",
    "transformers": "Transformers",
    "truststore": "truststore",
    "win32com.client": "Microsoft Office COM support",
}


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _java_version() -> tuple[bool, str]:
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
    old_versions: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        result = subprocess.run(
            [str(candidate), "-version"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
        )
        first_line = (result.stderr or result.stdout).splitlines()
        version_text = first_line[0] if first_line else "unknown version"
        match = re.search(r'version\s+"([0-9]+)(?:\.([0-9]+))?', version_text)
        major = 0
        if match:
            first = int(match.group(1))
            major = int(match.group(2) or 0) if first == 1 else first
        detail = f"{candidate} ({version_text})"
        if result.returncode == 0 and major >= 11:
            return True, detail
        old_versions.append(detail)
    if old_versions:
        return False, "only unsupported Java versions found: " + "; ".join(old_versions)
    return False, "not found; digital PDFs can fall back to PyMuPDF, but scanned-PDF OCR needs Java 11+"


def _office_app_paths() -> dict[str, str | None]:
    if os.name != "nt":
        return {"Word": None, "PowerPoint": None, "Excel": None}
    import winreg

    executables = {
        "Word": "WINWORD.EXE",
        "PowerPoint": "POWERPNT.EXE",
        "Excel": "EXCEL.EXE",
    }
    results: dict[str, str | None] = {}
    registry_views = (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY)
    for label, executable in executables.items():
        value: str | None = None
        key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{executable}"
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for view in registry_views:
                try:
                    with winreg.OpenKey(hive, key_path, 0, winreg.KEY_READ | view) as key:
                        candidate = str(winreg.QueryValue(key, None))
                except OSError:
                    continue
                if Path(candidate).is_file():
                    value = candidate
                    break
            if value:
                break
        results[label] = value
    return results


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError):
                pass

    parser = argparse.ArgumentParser(description="Check the PixelRAG Studio source environment")
    parser.add_argument("--require-java", action="store_true")
    parser.add_argument("--require-office", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    version_ok = sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32
    print(f"[{'OK' if version_ok else 'FAIL'}] Python: {sys.version.split()[0]} ({sys.executable})")
    if not version_ok:
        failures.append("PixelRAG Studio requires 64-bit Python 3.12")

    tkinter_ok = _module_exists("tkinter")
    print(f"[{'OK' if tkinter_ok else 'FAIL'}] Tkinter desktop UI")
    if not tkinter_ok:
        failures.append("Tkinter is missing; reinstall Python with Tcl/Tk support")

    for module, label in REQUIRED_MODULES.items():
        available = _module_exists(module)
        print(f"[{'OK' if available else 'FAIL'}] {label} ({module})")
        if not available:
            failures.append(f"Missing Python dependency: {module}")

    java_ok, java_detail = _java_version()
    print(f"[{'OK' if java_ok else 'WARN'}] Java: {java_detail}")
    if not java_ok:
        message = "Java 11+ is unavailable; scanned-PDF OCR will not work"
        (failures if args.require_java else warnings).append(message)

    office_paths = _office_app_paths()
    missing_office = [name for name, path in office_paths.items() if not path]
    for name, path in office_paths.items():
        print(f"[{'OK' if path else 'WARN'}] Microsoft {name}: {path or 'not detected'}")
    if missing_office:
        message = (
            "Microsoft Office desktop applications were not all detected; "
            "native Office page rendering and visual export will be limited"
        )
        (failures if args.require_office else warnings).append(message)

    print()
    if warnings:
        print("Optional capability warnings:")
        for warning in warnings:
            print(f"- {warning}")
        print()
    if failures:
        print("Environment verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Environment verification passed. PixelRAG Studio can be started from source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

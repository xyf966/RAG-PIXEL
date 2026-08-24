from __future__ import annotations

import base64
import importlib
import json
import multiprocessing
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


APP_NAME = "PixelRAG Studio"
APP_VERSION = "0.2.0"
SUPPORTED = {
    ".pdf", ".doc", ".docx", ".docm", ".ppt", ".pptx", ".pptm",
    ".xls", ".xlsx", ".xlsm", ".png", ".jpg", ".jpeg", ".webp",
    ".md", ".txt", ".html", ".htm",
}
DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def _configure_console_output() -> None:
    """Prevent third-party diagnostics from crashing on a legacy Windows code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (OSError, ValueError):
            pass


def _format_retrieval_location(provenance: dict[str, Any] | None) -> str:
    provenance = provenance or {}
    parts: list[str] = []
    if provenance.get("page") is not None:
        parts.append(f"第 {provenance['page']} 页")
    if provenance.get("slide") is not None:
        parts.append(f"幻灯片 {provenance['slide']}")
    if provenance.get("sheet"):
        parts.append(f"工作表 {provenance['sheet']}")
    if provenance.get("cell_range"):
        parts.append(str(provenance["cell_range"]))
    if not parts and provenance.get("locator"):
        parts.append(str(provenance["locator"]))
    return " / ".join(parts) or "未标注位置"


def _format_retrieval_content(hit: dict[str, Any]) -> str:
    content = hit.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if hit.get("modality") == "text" and isinstance(content.get("text"), str):
            return content["text"]
        if hit.get("modality") == "table":
            for key in ("raw", "markdown", "text"):
                if isinstance(content.get(key), str) and content[key].strip():
                    return content[key]
        return json.dumps(content, ensure_ascii=False, indent=2)
    context = hit.get("context")
    if isinstance(context, str) and context.strip():
        return context
    return json.dumps(content, ensure_ascii=False, indent=2) if content is not None else "无可显示内容"


def _format_retrieval_hit_label(hit: dict[str, Any]) -> str:
    source = Path(str(hit.get("source_path") or "未知文档")).name
    modality = str(hit.get("modality") or "unknown").upper()
    location = _format_retrieval_location(hit.get("provenance"))
    return (
        f"{int(hit.get('rank', 0)):02d}  [{modality:<6}]  "
        f"融合 {float(hit.get('fusion_score', 0)):.5f}  "
        f"原始 {float(hit.get('raw_score', 0)):.5f}  {source}  {location}"
    )


def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _data_root() -> Path:
    override = os.environ.get("PIXELRAG_STUDIO_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return _runtime_root() / "PixelRAG-Studio-Data"


def _default_model() -> str:
    root = _runtime_root()
    candidates = (
        root / "models" / "Qwen3-VL-Embedding-2B",
        root / "model-cache" / "Qwen3-VL-Embedding-2B",
    )
    for bundled in candidates:
        if (bundled / "model.safetensors").exists() and (bundled / "config.json").exists():
            return str(bundled)
    return DEFAULT_MODEL


def _redirect_worker_output() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except (ImportError, RuntimeError):
        pass
    log_path = os.environ.get("PIXELRAG_STUDIO_LOG")
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    stream = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream


def _safe_print(message: str) -> None:
    """Keep worker diagnostics usable even under a legacy Windows code page."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, errors="backslashreplace").decode(encoding))


def _dispatch_module(module_name: str, args: list[str]) -> None:
    _redirect_worker_output()
    module = importlib.import_module(module_name)
    sys.argv = [module_name, *args]
    if not hasattr(module, "main"):
        raise RuntimeError(f"Module {module_name} has no main()")
    module.main()


def _install_pymupdf_pdf_renderer() -> None:
    """Use the bundled Python PDF renderer instead of an external Poppler install."""
    import pymupdf as fitz
    from PIL import Image
    import pdf2image

    def convert_from_path(
        pdf_path: str,
        dpi: int = 200,
        first_page: int | None = None,
        last_page: int | None = None,
        **_kwargs: object,
    ) -> list[Image.Image]:
        document = fitz.open(pdf_path)
        try:
            start = max((first_page or 1) - 1, 0)
            stop = min(last_page or document.page_count, document.page_count)
            scale = dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            images: list[Image.Image] = []
            for page_index in range(start, stop):
                pixmap = document.load_page(page_index).get_pixmap(
                    matrix=matrix,
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                images.append(
                    Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                )
            return images
        finally:
            document.close()

    pdf2image.convert_from_path = convert_from_path


def _worker_index(config_path: str, force: bool) -> None:
    _redirect_worker_output()
    import yaml

    from hybrid_input.indexing import build_hybrid_index

    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    source_dir = Path(config["source"]["path"])
    index_dir = Path(config["output"])
    model = config.get("embed", {}).get("model") or DEFAULT_MODEL
    device = config.get("embed", {}).get("device") or "cpu"
    if force:
        _safe_print("Forced rebuild requested")
    build_hybrid_index(
        source_dir=source_dir,
        artifacts_dir=config_file.parent / "artifacts",
        index_dir=index_dir,
        model=model,
        device=device,
    )


def _worker_serve(
    index_dir: str,
    tiles_dir: str,
    articles_json: str,
    model: str,
    port: str,
) -> None:
    _redirect_worker_output()
    from hybrid_input.indexing import serve_hybrid_index

    del tiles_dir, articles_json
    serve_hybrid_index(Path(index_dir), model, int(port), device="cpu")


def _worker_hybrid(source_dir: str, artifacts_dir: str) -> None:
    _redirect_worker_output()
    from hybrid_input import build_default_pipeline

    pipeline = build_default_pipeline()
    failures: list[str] = []
    sources = [
        path for path in sorted(Path(source_dir).iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED
    ]
    print(f"Hybrid ingestion: {len(sources)} document(s)")
    for source in sources:
        try:
            result = pipeline.ingest(source, Path(artifacts_dir))
            visual_count = sum(item.kind == "visual" for item in result.artifacts)
            print(
                f"OK {source.name}: {len(result.artifacts)} blocks, "
                f"{visual_count} visual regions (Pixel deferred to index)"
            )
        except Exception as exc:
            failures.append(f"{source.name}: {exc}")
            print(f"FAILED {source.name}: {exc}")
    if failures:
        raise RuntimeError("Hybrid ingestion failures: " + "; ".join(failures))


def _run_worker_mode() -> bool:
    if len(sys.argv) >= 3 and sys.argv[1] == "-m":
        _dispatch_module(sys.argv[2], sys.argv[3:])
        return True
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker-index":
        _worker_index(sys.argv[2], "--force" in sys.argv[3:])
        return True
    if len(sys.argv) >= 7 and sys.argv[1] == "--worker-serve":
        _worker_serve(*sys.argv[2:7])
        return True
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker-hybrid":
        _worker_hybrid(sys.argv[2], sys.argv[3])
        return True
    return False


class StudioApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        _configure_console_output()
        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)

        self.data_root = _data_root()
        self.projects_root = self.data_root / "projects"
        self.model_cache = self.data_root / "models"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.model_cache.mkdir(parents=True, exist_ok=True)

        self.project_dir: Path | None = None
        self.build_process: subprocess.Popen | None = None
        self.ingest_process: subprocess.Popen | None = None
        self.server_process: subprocess.Popen | None = None
        self.server_port = 30001
        self.server_is_ready = False
        self.search_engine = None
        self.search_busy = False
        self.preview_image = None
        self._log_offset = 0
        self._last_server_log_offset = 0
        self._ingest_log_offset = 0
        self.search_events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.project_name = tk.StringVar(value="我的视觉知识库")
        self.model_name = tk.StringVar(value=_default_model())
        self.status_text = tk.StringVar(value="就绪：请创建或打开项目")
        self.query_text = tk.StringVar()
        self.top_k = tk.IntVar(value=10)
        self.force_rebuild = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(700, self._poll)

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.root.configure(bg="#f4f6f8")

        header = ttk.Frame(self.root, padding=(18, 14))
        header.pack(fill="x")
        ttk.Label(header, text="PixelRAG Studio", font=("Segoe UI", 18, "bold")).pack(side="left")
        ttk.Label(
            header,
            text="Hybrid 文档解析 / 多模态索引与检索检查",
            foreground="#52606d",
        ).pack(side="left", padx=(16, 0), pady=(5, 0))
        ttk.Button(header, text="打开数据目录", command=self._open_data_root).pack(side="right")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        left = ttk.Frame(body, padding=14)
        right = ttk.Frame(body, padding=10)
        body.add(left, weight=2)
        body.add(right, weight=5)

        project_box = ttk.LabelFrame(left, text="1. 项目", padding=10)
        project_box.pack(fill="x")
        ttk.Label(project_box, text="项目名称").pack(anchor="w")
        ttk.Entry(project_box, textvariable=self.project_name).pack(fill="x", pady=(4, 8))
        row = ttk.Frame(project_box)
        row.pack(fill="x")
        ttk.Button(row, text="创建/打开", command=self._create_project).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="打开目录", command=self._open_project).pack(side="left", padx=(8, 0))

        docs_box = ttk.LabelFrame(left, text="2. 文档", padding=10)
        docs_box.pack(fill="both", expand=True, pady=10)
        self.docs_list = tk.Listbox(docs_box, height=12, activestyle="none")
        self.docs_list.pack(fill="both", expand=True)
        doc_row = ttk.Frame(docs_box)
        doc_row.pack(fill="x", pady=(8, 0))
        ttk.Button(doc_row, text="添加文档", command=self._add_documents).pack(side="left", fill="x", expand=True)
        ttk.Button(doc_row, text="移除", command=self._remove_document).pack(side="left", padx=(8, 0))

        ingest_box = ttk.LabelFrame(left, text="3. Hybrid 输入解析", padding=10)
        ingest_box.pack(fill="x", pady=(0, 10))
        ttk.Label(
            ingest_box,
            text="文字/表格直接解析；图片和图表生成独立资产并保留来源坐标。",
            wraplength=300,
            foreground="#6b7280",
        ).pack(anchor="w")
        self.ingest_button = ttk.Button(ingest_box, text="解析输入文档", command=self._start_hybrid_ingest)
        self.ingest_button.pack(fill="x", pady=(6, 0))

        build_box = ttk.LabelFrame(left, text="4. 构建混合索引", padding=10)
        build_box.pack(fill="x")
        ttk.Label(build_box, text="视觉 Embedding 模型").pack(anchor="w")
        ttk.Entry(build_box, textvariable=self.model_name).pack(fill="x", pady=(4, 6))
        ttk.Label(
            build_box,
            text="首次运行会下载 Qwen3-VL 模型；模型缓存保存在应用数据目录。",
            wraplength=300,
            foreground="#6b7280",
        ).pack(anchor="w")
        ttk.Checkbutton(build_box, text="强制完全重建", variable=self.force_rebuild).pack(anchor="w", pady=(6, 4))
        self.build_button = ttk.Button(build_box, text="开始构建", command=self._start_build)
        self.build_button.pack(fill="x", pady=(4, 0))

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        search_tab = ttk.Frame(self.notebook, padding=12)
        log_tab = ttk.Frame(self.notebook, padding=8)
        about_tab = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(search_tab, text="检索检查器")
        self.notebook.add(log_tab, text="运行日志")
        self.notebook.add(about_tab, text="能力说明")

        search_row = ttk.Frame(search_tab)
        search_row.pack(fill="x")
        self.query_entry = ttk.Entry(search_row, textvariable=self.query_text, font=("Segoe UI", 11))
        self.query_entry.pack(side="left", fill="x", expand=True)
        self.query_entry.bind("<Return>", self._search)
        ttk.Label(search_row, text="Top K").pack(side="left", padx=(10, 4))
        ttk.Spinbox(search_row, from_=1, to=100, textvariable=self.top_k, width=4).pack(side="left")
        self.start_search_button = ttk.Button(
            search_row, text="初始化检索器", command=self._start_server, state="disabled"
        )
        self.start_search_button.pack(side="left", padx=8)
        self.search_button = ttk.Button(
            search_row, text="搜索", command=self._search, state="disabled"
        )
        self.search_button.pack(side="left")
        ttk.Label(
            search_tab,
            text="只检查检索结果，不调用 LLM。首次查询会加载向量模型，之后查询会复用模型。",
            foreground="#6b7280",
        ).pack(anchor="w", pady=(8, 0))

        result_frame = ttk.Panedwindow(search_tab, orient="horizontal")
        result_frame.pack(fill="both", expand=True, pady=(12, 0))
        result_left = ttk.Frame(result_frame)
        result_right = ttk.Frame(result_frame, padding=(12, 0, 0, 0))
        result_frame.add(result_left, weight=2)
        result_frame.add(result_right, weight=3)
        self.results = tk.Listbox(result_left, font=("Consolas", 10), activestyle="dotbox")
        self.results.pack(fill="both", expand=True)
        self.results.bind("<<ListboxSelect>>", self._show_selected_result)
        self.result_payloads: list[dict] = []
        self.preview_container = ttk.Frame(result_right)
        self.preview_container.pack(fill="both", expand=True)
        self.preview = ttk.Label(self.preview_container, text="命中图片时显示预览", anchor="center")
        self.result_text_frame = ttk.Frame(self.preview_container)
        self.result_text = tk.Text(
            self.result_text_frame,
            wrap="word",
            font=("Segoe UI", 10),
            bg="#ffffff",
            fg="#1f2937",
            relief="solid",
            borderwidth=1,
        )
        result_scroll = ttk.Scrollbar(self.result_text_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        result_scroll.pack(side="right", fill="y")
        self.result_text_frame.pack(fill="both", expand=True)
        self._set_result_text("输入问题后，这里会显示命中的文字、表格原始内容或图片。")
        self.result_meta = ttk.Label(result_right, text="", wraplength=520, foreground="#374151")
        self.result_meta.pack(fill="x", pady=(8, 0))

        self.log_text = tk.Text(log_tab, wrap="none", font=("Consolas", 9), bg="#111827", fg="#d1fae5")
        self.log_text.pack(fill="both", expand=True)

        about = (
            "本应用使用 Hybrid 输入与图片专用 Pixel 流程：\n\n"
            "文档 → 原生结构解析 → 文字结构分块 / 表格结构分块 / visual 延迟物化 → "
            "Qwen3-VL-Embedding-2B 统一向量 → 三通道 FAISS 索引快照。\n\n"
            "Office 临时 PDF 只负责页面坐标补全和复杂视觉对象渲染；文字和表格不会进入 Pixel。\n\n"
            "当前构建和检索固定使用 CPU，以保证无独立显卡的 Windows 电脑也能运行。"
            "检索检查器已接入，回答生成层仍未接入。"
        )
        ttk.Label(about_tab, text=about, wraplength=760, justify="left", font=("Segoe UI", 11)).pack(anchor="nw")
        ttk.Label(
            about_tab,
            text="PixelRAG：Apache-2.0；应用数据与模型默认保存在程序旁的 PixelRAG-Studio-Data。",
            foreground="#6b7280",
        ).pack(anchor="sw", pady=(30, 0))

        status = ttk.Label(self.root, textvariable=self.status_text, relief="sunken", anchor="w", padding=(10, 5))
        status.pack(fill="x", side="bottom")

    @staticmethod
    def _safe_name(value: str) -> str:
        value = "".join(c if c.isalnum() or c in "-_ " else "_" for c in value).strip()
        return value or "PixelRAG项目"

    def _create_project(self) -> None:
        name = self._safe_name(self.project_name.get())
        self.project_name.set(name)
        next_project = self.projects_root / name
        if self.project_dir != next_project:
            self._stop_server()
        self.project_dir = next_project
        for sub in ("source", "artifacts", "index", "logs"):
            (self.project_dir / sub).mkdir(parents=True, exist_ok=True)
        self._write_config()
        self._refresh_documents()
        self._set_search_controls(self._index_ready())
        self.status_text.set(f"项目已打开：{self.project_dir}")

    def _ensure_project(self) -> bool:
        if self.project_dir is None:
            self._create_project()
        return self.project_dir is not None

    def _write_config(self) -> Path:
        assert self.project_dir is not None
        config = {
            "source": {"type": "local", "path": str((self.project_dir / "source").resolve())},
            "ingest": {"backend": "cdp", "quality": 90, "tile_height": 8192, "dpi": 200},
            "embed": {"model": self.model_name.get().strip() or DEFAULT_MODEL, "device": "cpu"},
            "output": str((self.project_dir / "index").resolve()),
        }
        path = self.project_dir / "pixelrag.yaml"
        import yaml

        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def _add_documents(self) -> None:
        from tkinter import filedialog, messagebox

        if not self._ensure_project():
            return
        files = filedialog.askopenfilenames(
            title="选择 PDF、Word、PPT、Excel、图片或文本",
            filetypes=[
                ("Hybrid 支持的文档", "*.pdf *.doc *.docx *.ppt *.pptx *.xls *.xlsx *.png *.jpg *.jpeg *.webp *.md *.txt *.html *.htm"),
                ("所有文件", "*.*"),
            ],
        )
        copied = 0
        for raw in files:
            src = Path(raw)
            if src.suffix.lower() not in SUPPORTED:
                continue
            dest = self.project_dir / "source" / src.name
            if dest.exists() and dest.resolve() != src.resolve():
                stamp = datetime.now().strftime("%H%M%S")
                dest = dest.with_name(f"{dest.stem}_{stamp}{dest.suffix}")
            shutil.copy2(src, dest)
            copied += 1
        self._refresh_documents()
        if copied:
            self._stop_server()
            self.status_text.set(f"已添加 {copied} 个文档")
        elif files:
            messagebox.showwarning(APP_NAME, "没有可添加的受支持文件。")

    def _refresh_documents(self) -> None:
        self.docs_list.delete(0, self.tk.END)
        if not self.project_dir:
            return
        for p in sorted((self.project_dir / "source").glob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                self.docs_list.insert(self.tk.END, p.name)

    def _remove_document(self) -> None:
        from tkinter import messagebox

        if not self.project_dir or not self.docs_list.curselection():
            return
        name = self.docs_list.get(self.docs_list.curselection()[0])
        if messagebox.askyesno(APP_NAME, f"从项目中移除 {name}？\n原始文件不会受影响。"):
            (self.project_dir / "source" / name).unlink(missing_ok=True)
            self._refresh_documents()
            self._stop_server()
            self.status_text.set("文档已移除，请重新构建索引")

    def _worker_command(self, *args: str) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, *args]
        return [sys.executable, str(Path(__file__).resolve()), *args]

    def _worker_env(self, log_path: Path) -> dict[str, str]:
        env = os.environ.copy()
        root = _runtime_root()
        poppler_candidates = (root / "poppler", root / "vendor" / "poppler")
        for poppler in poppler_candidates:
            if (poppler / "pdfinfo.exe").exists() and (poppler / "pdftoppm.exe").exists():
                env["PATH"] = str(poppler) + os.pathsep + env.get("PATH", "")
                break
        env["PIXELRAG_STUDIO_LOG"] = str(log_path)
        env["HF_HOME"] = str(self.model_cache)
        env["HF_HUB_CACHE"] = str(self.model_cache / "hub")
        env["TRANSFORMERS_CACHE"] = str(self.model_cache / "transformers")
        env["PYTHONUTF8"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        return env

    def _start_hybrid_ingest(self) -> None:
        from tkinter import messagebox

        if not self._ensure_project():
            return
        if self.docs_list.size() == 0:
            messagebox.showwarning(APP_NAME, "请先添加至少一个文档。")
            return
        if self.ingest_process and self.ingest_process.poll() is None:
            messagebox.showinfo(APP_NAME, "Hybrid 输入正在解析，请查看运行日志。")
            return
        if self.build_process and self.build_process.poll() is None:
            messagebox.showinfo(APP_NAME, "索引正在构建，请等待完成后再单独运行输入解析。")
            return
        assert self.project_dir is not None
        log_path = self.project_dir / "logs" / "ingest.log"
        log_path.write_text(
            f"[{datetime.now().isoformat(timespec='seconds')}] 开始 Hybrid 输入解析\n",
            encoding="utf-8",
        )
        self._ingest_log_offset = 0
        cmd = self._worker_command(
            "--worker-hybrid",
            str(self.project_dir / "source"),
            str(self.project_dir / "artifacts"),
        )
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.ingest_process = subprocess.Popen(
            cmd,
            cwd=str(self.project_dir),
            env=self._worker_env(log_path),
            creationflags=flags,
        )
        self.ingest_button.configure(state="disabled")
        self.status_text.set("正在解析：结构识别 → visual 分类与定位 → HybridDocument（不调用 Pixel）")
        self.notebook.select(1)

    def _start_build(self) -> None:
        from tkinter import messagebox

        if not self._ensure_project():
            return
        if self.docs_list.size() == 0:
            messagebox.showwarning(APP_NAME, "请先添加至少一个支持的文档。")
            return
        if self.build_process and self.build_process.poll() is None:
            messagebox.showinfo(APP_NAME, "索引正在构建，请查看运行日志。")
            return
        if self.ingest_process and self.ingest_process.poll() is None:
            messagebox.showinfo(APP_NAME, "Hybrid 输入正在解析，请等待完成后再构建索引。")
            return
        if self.search_busy:
            messagebox.showinfo(APP_NAME, "检索正在执行，请等待本次查询完成后再重建索引。")
            return
        self._stop_server()
        self._set_search_controls(False)
        config = self._write_config()
        log_path = self.project_dir / "logs" / "build.log"
        log_path.write_text(
            f"[{datetime.now().isoformat(timespec='seconds')}] 开始构建 Hybrid 混合索引\n",
            encoding="utf-8",
        )
        self._log_offset = 0
        cmd = self._worker_command("--worker-index", str(config))
        if self.force_rebuild.get():
            cmd.append("--force")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.build_process = subprocess.Popen(
            cmd,
            cwd=str(self.project_dir),
            env=self._worker_env(log_path),
            creationflags=flags,
        )
        self.build_button.configure(state="disabled")
        self.status_text.set("正在构建：结构识别 → 按模态分块 → 统一 Embedding → schema 2.0 索引快照")
        self.notebook.select(1)

    def _index_ready(self) -> bool:
        if not self.project_dir:
            return False
        try:
            from hybrid_input.indexing import current_snapshot

            snapshot = current_snapshot(self.project_dir / "index")
            manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        except Exception:
            return False
        source_files = [path for path in (self.project_dir / "source").iterdir() if path.is_file()]
        if not source_files:
            return False
        indexed = {entry["path"]: entry for entry in manifest.get("sources", [])}
        return len(indexed) == len(source_files) and all(
            str(path.resolve()) in indexed
            and path.stat().st_size == indexed[str(path.resolve())].get("size")
            and path.stat().st_mtime_ns == indexed[str(path.resolve())].get("modified_ns")
            for path in source_files
        )

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _start_server(self) -> None:
        from tkinter import messagebox

        if not self._index_ready():
            messagebox.showwarning(APP_NAME, "当前项目没有可用的最新索引，请先构建索引。")
            self._set_search_controls(False)
            return
        if self.search_engine is not None:
            self.status_text.set("检索器已经初始化，可以直接输入问题搜索")
            return
        if self.search_busy:
            return
        self.search_busy = True
        self._set_search_controls(True)
        self.status_text.set("正在初始化检索器；向量模型将在首次查询时加载")
        threading.Thread(target=self._initialize_search_thread, daemon=True).start()

    def _initialize_search_thread(self) -> None:
        try:
            from hybrid_input.retrieval import HybridSearchEngine

            assert self.project_dir is not None
            engine = HybridSearchEngine(self.project_dir / "index", device="cpu")
            self.search_events.put(("ready", engine))
        except Exception as exc:
            self.search_events.put(("error", str(exc)))

    def _server_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server_port}{path}"

    def _server_ready(self) -> bool:
        try:
            with urllib.request.urlopen(self._server_url("/status"), timeout=1.0) as response:
                return response.status == 200
        except Exception:
            return False

    def _search(self, _event=None) -> None:
        from tkinter import messagebox

        query = self.query_text.get().strip()
        if not query:
            messagebox.showwarning(APP_NAME, "请输入要检索的内容。")
            self.query_entry.focus_set()
            return
        if self.search_busy:
            return
        if not self._index_ready():
            messagebox.showwarning(APP_NAME, "当前项目没有可用的最新索引，请先构建索引。")
            self._set_search_controls(False)
            return
        try:
            top_k = max(1, min(100, int(self.top_k.get())))
        except (TypeError, ValueError):
            top_k = 10
            self.top_k.set(top_k)
        self.search_busy = True
        self._set_search_controls(True)
        self.status_text.set("正在检索；首次查询可能需要一些时间加载向量模型……")
        threading.Thread(target=self._search_thread, args=(query, top_k), daemon=True).start()

    def _search_thread(self, query: str, top_k: int) -> None:
        created_engine = None
        try:
            from hybrid_input.retrieval_contracts import RetrievalRequest

            engine = self.search_engine
            if engine is None:
                from hybrid_input.retrieval import HybridSearchEngine

                assert self.project_dir is not None
                created_engine = HybridSearchEngine(self.project_dir / "index", device="cpu")
                engine = created_engine
            response = engine.search(
                RetrievalRequest(
                    query_text=query,
                    top_k=top_k,
                    candidate_k=max(30, top_k),
                )
            )
            self.search_events.put(("results", {"engine": created_engine, "response": response.to_dict()}))
        except Exception as exc:
            if created_engine is not None:
                created_engine.close()
            self.search_events.put(("error", str(exc)))

    def _display_hits(self, response: dict[str, Any]) -> None:
        hits = response.get("hits", [])
        self.results.delete(0, self.tk.END)
        self.result_payloads = hits
        for hit in hits:
            self.results.insert(self.tk.END, _format_retrieval_hit_label(hit))
        if hits:
            self.results.selection_set(0)
            self._show_selected_result()
            elapsed = float(response.get("elapsed_ms", 0))
            warnings = response.get("warnings") or []
            suffix = f"；警告 {len(warnings)} 条" if warnings else ""
            self.status_text.set(f"检索完成：返回 {len(hits)} 个结果，耗时 {elapsed:.0f} ms{suffix}")
            self.notebook.select(0)
        else:
            self.status_text.set("没有检索到结果")

    def _show_selected_result(self, _event=None) -> None:
        from PIL import Image, ImageTk

        sel = self.results.curselection()
        if not sel or sel[0] >= len(self.result_payloads):
            return
        hit = self.result_payloads[sel[0]]
        asset_path = hit.get("asset_path")
        if hit.get("modality") == "visual" and asset_path and Path(asset_path).is_file():
            try:
                image = Image.open(asset_path).convert("RGB")
                image.thumbnail((620, 520), Image.LANCZOS)
                self.preview_image = ImageTk.PhotoImage(image)
                self.result_text_frame.pack_forget()
                self.preview.configure(image=self.preview_image, text="")
                self.preview.pack(fill="both", expand=True)
            except Exception as exc:
                self._show_result_text(f"图片预览失败：{exc}\n\n{_format_retrieval_content(hit)}")
        else:
            self._show_result_text(_format_retrieval_content(hit))
        provenance = hit.get("provenance") or {}
        self.result_meta.configure(
            text=(
                f"文档：{hit.get('source_path', '')}\n"
                f"模态：{hit.get('modality', '')}    排名：{hit.get('rank', '')}    "
                f"融合分：{float(hit.get('fusion_score', 0)):.5f}    "
                f"原始分：{float(hit.get('raw_score', 0)):.5f}\n"
                f"位置：{_format_retrieval_location(provenance)}    "
                f"记录 ID：{hit.get('record_id', '')}"
            )
        )

    def _set_result_text(self, value: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", self.tk.END)
        self.result_text.insert("1.0", value)
        self.result_text.configure(state="disabled")

    def _show_result_text(self, value: str) -> None:
        self.preview.pack_forget()
        self.preview.configure(image="", text="")
        self.preview_image = None
        self.result_text_frame.pack(fill="both", expand=True)
        self._set_result_text(value)

    def _set_search_controls(self, index_ready: bool) -> None:
        search_state = "normal" if index_ready and not self.search_busy else "disabled"
        initialize_state = (
            "normal"
            if index_ready and not self.search_busy and self.search_engine is None
            else "disabled"
        )
        self.start_search_button.configure(state=initialize_state)
        self.search_button.configure(state=search_state)
        self.start_search_button.configure(
            text="检索器已初始化" if self.search_engine is not None else "初始化检索器"
        )

    def _append_log_file(self, path: Path, offset_attr: str) -> None:
        if not path.exists():
            return
        offset = getattr(self, offset_attr)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                content = handle.read()
                setattr(self, offset_attr, handle.tell())
            if content:
                self.log_text.insert(self.tk.END, content)
                self.log_text.see(self.tk.END)
        except OSError:
            pass

    def _poll(self) -> None:
        while True:
            try:
                event, payload = self.search_events.get_nowait()
            except queue.Empty:
                break
            if event == "ready":
                self.search_engine = payload
                self.search_busy = False
                self._set_search_controls(True)
                self.status_text.set("检索器已初始化；首次查询将加载向量模型")
            elif event == "results" and isinstance(payload, dict):
                if payload.get("engine") is not None:
                    self.search_engine = payload["engine"]
                self.search_busy = False
                self._set_search_controls(True)
                response = payload.get("response")
                self._display_hits(response if isinstance(response, dict) else {})
            else:
                self.search_busy = False
                self._set_search_controls(self._index_ready())
                self.status_text.set(f"搜索失败：{payload}")
        if self.project_dir:
            self._append_log_file(self.project_dir / "logs" / "ingest.log", "_ingest_log_offset")
            self._append_log_file(self.project_dir / "logs" / "build.log", "_log_offset")
            self._append_log_file(self.project_dir / "logs" / "server.log", "_last_server_log_offset")
        if self.build_process and self.build_process.poll() is not None:
            code = self.build_process.returncode
            self.build_process = None
            self.build_button.configure(state="normal")
            if code == 0 and self._index_ready():
                self._set_search_controls(True)
                self.status_text.set("Hybrid 混合索引构建完成，可以使用检索检查器")
            else:
                self._set_search_controls(False)
                self.status_text.set(f"索引构建失败（退出码 {code}），请查看运行日志")
        if self.ingest_process and self.ingest_process.poll() is not None:
            code = self.ingest_process.returncode
            self.ingest_process = None
            self.ingest_button.configure(state="normal")
            if code == 0:
                self.status_text.set("Hybrid 输入解析完成；结果已保存到项目 artifacts 目录")
            else:
                self.status_text.set(f"Hybrid 输入解析失败（退出码 {code}），请查看运行日志")
        self.root.after(800, self._poll)

    def _open_data_root(self) -> None:
        os.startfile(str(self.data_root))

    def _open_project(self) -> None:
        if self._ensure_project():
            os.startfile(str(self.project_dir))

    def _stop_server(self) -> None:
        if self.search_engine is not None:
            self.search_engine.close()
        self.search_engine = None
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        self.server_process = None
        self.server_is_ready = False
        if hasattr(self, "start_search_button"):
            self._set_search_controls(False)

    def _on_close(self) -> None:
        self._stop_server()
        if self.ingest_process and self.ingest_process.poll() is None:
            self.ingest_process.terminate()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    multiprocessing.freeze_support()
    if _run_worker_mode():
        return
    StudioApp().run()


if __name__ == "__main__":
    main()

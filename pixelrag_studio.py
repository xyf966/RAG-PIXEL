from __future__ import annotations

import base64
import hashlib
import importlib
import json
import multiprocessing
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


APP_NAME = "PixelRAG Studio"
APP_VERSION = "0.2.0"
SUPPORTED = {
    ".pdf", ".doc", ".docx", ".docm", ".ppt", ".pptx", ".pptm",
    ".xls", ".xlsx", ".xlsm", ".png", ".jpg", ".jpeg", ".webp",
    ".md", ".txt", ".html", ".htm",
}
DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


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
        self.preview_image = None
        self._log_offset = 0
        self._last_server_log_offset = 0
        self._ingest_log_offset = 0
        self.search_events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.project_name = tk.StringVar(value="我的视觉知识库")
        self.model_name = tk.StringVar(value=_default_model())
        self.status_text = tk.StringVar(value="就绪：请创建或打开项目")
        self.query_text = tk.StringVar()
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
            text="Hybrid 文档解析 / 图片专用 Pixel / 混合检索",
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
        self.notebook.add(search_tab, text="混合搜索")
        self.notebook.add(log_tab, text="运行日志")
        self.notebook.add(about_tab, text="能力说明")

        search_row = ttk.Frame(search_tab)
        search_row.pack(fill="x")
        ttk.Entry(search_row, textvariable=self.query_text, font=("Segoe UI", 11)).pack(side="left", fill="x", expand=True)
        ttk.Button(search_row, text="启动搜索服务", command=self._start_server).pack(side="left", padx=8)
        ttk.Button(search_row, text="搜索", command=self._search).pack(side="left")
        self.root.bind("<Return>", lambda _event: self._search())

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
        self.preview = ttk.Label(result_right, text="命中图片时显示预览；文字/表格显示原始内容", anchor="center")
        self.preview.pack(fill="both", expand=True)
        self.result_meta = ttk.Label(result_right, text="", wraplength=520, foreground="#374151")
        self.result_meta.pack(fill="x", pady=(8, 0))

        self.log_text = tk.Text(log_tab, wrap="none", font=("Consolas", 9), bg="#111827", fg="#d1fae5")
        self.log_text.pack(fill="both", expand=True)

        about = (
            "本应用使用 Hybrid 输入与图片专用 Pixel 流程：\n\n"
            "文档 → 原生结构解析 → 文字/表格直接保留 → 图片提取或复杂对象裁切 → "
            "Qwen3-VL-Embedding-2B 图片向量 → 结构化索引 + 图片 FAISS → 混合检索。\n\n"
            "Office 临时 PDF 只负责页面坐标补全和复杂视觉对象渲染；文字和表格不会进入 Pixel。\n\n"
            "当前构建固定使用 CPU，以保证无独立显卡的 Windows 电脑也能运行。CPU 首次建库和首次启动搜索服务可能需要较长时间。"
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
        self.project_dir = self.projects_root / name
        for sub in ("source", "artifacts", "index", "logs"):
            (self.project_dir / sub).mkdir(parents=True, exist_ok=True)
        self._write_config()
        self._refresh_documents()
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
        self._stop_server()
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
        self.status_text.set("正在构建：结构识别 → visual 物化 → Pixel 视觉向量 → 语义/FAISS 混合索引")
        self.notebook.select(1)

    def _index_ready(self) -> bool:
        if not self.project_dir:
            return False
        index_dir = self.project_dir / "index"
        required = tuple(
            index_dir / name
            for name in ("hybrid-index.json", "semantic-index.json", "image-metadata.json")
        )
        if not all(path.exists() for path in required):
            return False
        source_files = [path for path in (self.project_dir / "source").iterdir() if path.is_file()]
        if not source_files:
            return False
        index_mtime = min(path.stat().st_mtime for path in required)
        return max(path.stat().st_mtime for path in source_files) <= index_mtime

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _start_server(self) -> None:
        from tkinter import messagebox

        if self.build_process and self.build_process.poll() is None:
            messagebox.showwarning(APP_NAME, "混合索引仍在构建，请等待构建完成后再启动搜索服务。")
            return
        if not self._index_ready():
            messagebox.showwarning(APP_NAME, "索引不存在或已落后于源文档，请先完成重新构建。")
            return
        if self.server_process and self.server_process.poll() is None:
            self.status_text.set(f"搜索服务正在运行：http://127.0.0.1:{self.server_port}")
            return
        assert self.project_dir is not None
        self.server_port = self._find_free_port()
        index_dir = self.project_dir / "index"
        cache_key = hashlib.sha256(str(self.project_dir).encode("utf-8")).hexdigest()[:16]
        safe_index_dir = Path(tempfile.gettempdir()) / "PixelRAGStudio" / cache_key / "index"
        safe_index_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(index_dir, safe_index_dir, dirs_exist_ok=True)
        log_path = self.project_dir / "logs" / "server.log"
        log_path.write_text(
            f"[{datetime.now().isoformat(timespec='seconds')}] 启动 PixelRAG 搜索服务\n",
            encoding="utf-8",
        )
        self._last_server_log_offset = 0
        cmd = self._worker_command(
            "--worker-serve",
            str(safe_index_dir),
            "-",
            "-",
            self.model_name.get().strip() or DEFAULT_MODEL,
            str(self.server_port),
        )
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.server_process = subprocess.Popen(
            cmd,
            cwd=str(self.project_dir),
            env=self._worker_env(log_path),
            creationflags=flags,
        )
        self.server_is_ready = False
        self.status_text.set("混合索引服务正在启动；Pixel 模型将在首次图片检索时加载……")
        self.notebook.select(1)

    def _server_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server_port}{path}"

    def _server_ready(self) -> bool:
        try:
            with urllib.request.urlopen(self._server_url("/status"), timeout=1.0) as response:
                return response.status == 200
        except Exception:
            return False

    def _search(self) -> None:
        from tkinter import messagebox

        query = self.query_text.get().strip()
        if not query:
            return
        if not self.server_process or self.server_process.poll() is not None:
            messagebox.showinfo(APP_NAME, "请先点击“启动搜索服务”，等待模型加载完成。")
            return
        if not self.server_is_ready and not self._server_ready():
            messagebox.showinfo(APP_NAME, "搜索服务仍在加载模型，请稍后再试并查看运行日志。")
            return
        self.server_is_ready = True
        self.status_text.set("正在执行文字/表格与图片混合检索……")
        threading.Thread(target=self._search_thread, args=(query,), daemon=True).start()

    def _search_thread(self, query: str) -> None:
        payload = json.dumps(
            {
                "queries": [{"text": query}],
                "n_docs": 8,
                "include_images": True,
                "min_tile_height": 28,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._server_url("/search"),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
            hits = data.get("results", [{}])[0].get("hits", [])
            self.search_events.put(("results", hits))
        except Exception as exc:
            self.search_events.put(("error", str(exc)))

    def _display_hits(self, hits: list[dict]) -> None:
        self.results.delete(0, self.tk.END)
        self.result_payloads = hits
        for i, hit in enumerate(hits, 1):
            label = Path(hit.get("url") or "未知文档").name
            channel = hit.get("channel", "unknown")
            location = hit.get("location") or "-"
            self.results.insert(
                self.tk.END,
                f"{i:02d}  {hit.get('score', 0):.4f}  [{channel}]  {label}  位置={location}",
            )
        if hits:
            self.results.selection_set(0)
            self._show_selected_result()
            self.status_text.set(f"混合检索完成：返回 {len(hits)} 个文字、表格或图片结果")
            self.notebook.select(0)
        else:
            self.status_text.set("没有检索到结果")

    def _show_selected_result(self, _event=None) -> None:
        from io import BytesIO

        from PIL import Image, ImageTk

        sel = self.results.curselection()
        if not sel or sel[0] >= len(self.result_payloads):
            return
        hit = self.result_payloads[sel[0]]
        raw = hit.get("image_base64")
        if raw:
            image = Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
            image.thumbnail((620, 540), Image.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.preview_image, text="")
        else:
            text_preview = hit.get("text") or hit.get("context") or "该结果没有图片预览"
            self.preview.configure(image="", text=text_preview)
        self.result_meta.configure(
            text=(
                f"文档：{hit.get('url', '')}\n"
                f"通道：{hit.get('channel', '')}    类型：{hit.get('kind', '')}    "
                f"得分：{hit.get('score', 0):.5f}    位置：{hit.get('location', '')}"
            )
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
            if event == "results":
                self._display_hits(payload if isinstance(payload, list) else [])
            else:
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
                self.status_text.set("Hybrid 混合索引构建完成，可以启动搜索服务")
            else:
                self.status_text.set(f"索引构建失败（退出码 {code}），请查看运行日志")
        if self.ingest_process and self.ingest_process.poll() is not None:
            code = self.ingest_process.returncode
            self.ingest_process = None
            self.ingest_button.configure(state="normal")
            if code == 0:
                self.status_text.set("Hybrid 输入解析完成；结果已保存到项目 artifacts 目录")
            else:
                self.status_text.set(f"Hybrid 输入解析失败（退出码 {code}），请查看运行日志")
        if (
            self.server_process
            and self.server_process.poll() is None
            and not self.server_is_ready
            and self._server_ready()
        ):
            self.server_is_ready = True
            current = self.status_text.get()
            if current.startswith("正在加载"):
                self.status_text.set(f"搜索服务就绪：http://127.0.0.1:{self.server_port}")
                self.notebook.select(0)
        self.root.after(800, self._poll)

    def _open_data_root(self) -> None:
        os.startfile(str(self.data_root))

    def _open_project(self) -> None:
        if self._ensure_project():
            os.startfile(str(self.project_dir))

    def _stop_server(self) -> None:
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        self.server_process = None
        self.server_is_ready = False

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

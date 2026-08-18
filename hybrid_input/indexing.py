from __future__ import annotations

import base64
import copy
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Artifact, HybridDocument
from .pipeline import PipelineConfig, build_default_pipeline, discover_docling_python
from .processors import PillowImageProcessor
from .renderers import MicrosoftOfficeSubprocessRenderer, MicrosoftOfficeVisualSubprocessRenderer
from .vision import PixelRAGEmbeddingSession, PixelRAGVisionProcessor


INDEX_MANIFEST = "hybrid-index.json"
SEMANTIC_INDEX = "semantic-index.json"
IMAGE_INDEX = "image-index.faiss"
IMAGE_METADATA = "image-metadata.json"
MIN_IMAGE_SCORE = 0.30


def _artifact_text(artifact: Artifact) -> str:
    if artifact.text:
        return artifact.text
    if artifact.kind == "table" and artifact.metadata:
        return json.dumps(artifact.metadata, ensure_ascii=False, sort_keys=True)
    return ""


def _record(document: HybridDocument, artifact: Artifact) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "source_path": document.source_path,
        "source_file": Path(document.source_path).name,
        "document_type": document.document_type,
        "block_id": artifact.block_id,
        "kind": artifact.kind,
        "visual_type": artifact.visual_type,
        "text": _artifact_text(artifact),
        "context": artifact.context or "",
        "asset_path": artifact.asset_path,
        "reading_order": artifact.reading_order,
        "provenance": asdict(artifact.provenance),
    }


def write_hybrid_index(documents: list[HybridDocument], index_dir: Path) -> dict[str, Any]:
    """Persist separate semantic and image indexes from HybridDocument contracts."""
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)
    semantic: list[dict[str, Any]] = []
    image_metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []

    for document in documents:
        artifacts = {item.block_id: item for item in document.artifacts}
        image_ids = {item.block_id for item in document.artifacts if item.kind == "visual"}
        semantic.extend(
            _record(document, item)
            for item in document.artifacts
            if item.kind in {"text", "table"}
        )
        for result in document.vision_results:
            if result.block_id not in image_ids:
                raise RuntimeError(
                    f"Non-visual block entered Pixel vision results: {result.block_id}"
                )
            if not result.vector:
                continue
            artifact = artifacts[result.block_id]
            vector = np.asarray(result.vector, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                raise RuntimeError(f"Pixel returned a zero vector: {result.block_id}")
            vectors.append(vector / norm)
            image_metadata.append(_record(document, artifact))

    (index_dir / SEMANTIC_INDEX).write_text(
        json.dumps(semantic, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (index_dir / IMAGE_METADATA).write_text(
        json.dumps(image_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image_index_path = index_dir / IMAGE_INDEX
    dimension = int(vectors[0].shape[0]) if vectors else 0
    if vectors:
        if any(vector.shape != (dimension,) for vector in vectors):
            raise RuntimeError("Pixel image vectors have inconsistent dimensions")
        index = faiss.IndexFlatIP(dimension)
        index.add(np.vstack(vectors).astype(np.float32))
        image_index_path.write_bytes(faiss.serialize_index(index).tobytes())
    else:
        image_index_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": "1.1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(documents),
        "semantic_blocks": len(semantic),
        "image_vectors": len(vectors),
        "image_dimension": dimension,
        "pixel_input_kinds": ["visual"],
    }
    (index_dir / INDEX_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _materialize_visuals_for_index(
    document: HybridDocument,
    output_dir: Path,
    office_renderer: MicrosoftOfficeVisualSubprocessRenderer,
    office_pdf_renderer: MicrosoftOfficeSubprocessRenderer | None = None,
) -> list[Artifact]:
    """Create temporary pixel inputs; recognition artifacts remain kind=visual."""
    visuals = [
        item for item in document.artifacts
        if item.kind == "visual" and item.metadata.get("indexable", True)
    ]
    temporary = [copy.deepcopy(item) for item in visuals]
    for item in temporary:
        item.kind = "image"

    source = Path(document.source_path)
    if document.document_type in {"doc", "docx", "ppt", "pptx", "xls", "xlsx"}:
        try:
            exports = office_renderer.render(source, output_dir / "office-native", export=True)
        except Exception as exc:
            exports = []
            print(f"Office native visual export unavailable; using page regions: {exc}", flush=True)
        for item, exported in zip(temporary, exports):
            path = Path(str(exported.get("path") or ""))
            if path.is_file():
                item.asset_path = str(path.resolve())
                item.metadata["index_materialization"] = str(exported.get("render_method") or "office-native-export")
        missing = [item for item in temporary if not item.asset_path or not Path(item.asset_path).is_file()]
        if missing:
            # Clipboard/native object export is not guaranteed in a hidden
            # Windows worker.  Render once, then crop only identified visual
            # regions; text and tables are never sent to Pixel.
            rendered: list[Path] = []
            artifacts_root = next(
                (parent for parent in (output_dir, *output_dir.parents) if parent.name == "artifacts"),
                None,
            )
            if artifacts_root is not None:
                existing_layout = (
                    artifacts_root
                    / f"{source.stem}-{document.document_id[:12]}"
                    / "layout-render"
                )
                rendered = sorted(existing_layout.glob("*.pdf"))
            if not rendered and office_pdf_renderer is not None:
                try:
                    rendered = office_pdf_renderer.render(source, output_dir / "office-layout")
                except Exception as exc:
                    print(f"Office page render unavailable; skipping missing visuals: {exc}", flush=True)
            if rendered:
                import pymupdf

                pdf = pymupdf.open(rendered[0])
                try:
                    fallback_dir = output_dir / "office-regions"
                    fallback_dir.mkdir(parents=True, exist_ok=True)
                    for item in missing:
                        page_no = item.provenance.page or item.provenance.slide
                        bounds = item.provenance.bbox_original
                        if not page_no or not bounds or page_no > pdf.page_count:
                            continue
                        page = pdf.load_page(page_no - 1)
                        clip = pymupdf.Rect(*bounds) & page.rect
                        if clip.is_empty or clip.width < 1 or clip.height < 1:
                            continue
                        destination = fallback_dir / f"{item.block_id.replace(':', '-')}.png"
                        page.get_pixmap(
                            matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False
                        ).save(destination)
                        if destination.is_file():
                            item.asset_path = str(destination.resolve())
                            item.metadata["index_materialization"] = "office-page-region-fallback"
                finally:
                    pdf.close()
    elif document.document_type == "pdf":
        import pymupdf

        pdf = pymupdf.open(source)
        try:
            raw_dir = output_dir / "pdf-regions"
            raw_dir.mkdir(parents=True, exist_ok=True)
            for item in temporary:
                if item.asset_path and Path(item.asset_path).is_file():
                    continue
                page_no = item.provenance.page
                if not page_no or not item.provenance.bbox:
                    continue
                page = pdf.load_page(page_no - 1)
                box = item.provenance.bbox
                if max(box) <= 1.01:
                    clip = pymupdf.Rect(box[0] * page.rect.width, box[1] * page.rect.height, box[2] * page.rect.width, box[3] * page.rect.height)
                else:
                    clip = pymupdf.Rect(*box)
                clip &= page.rect
                if clip.is_empty or clip.width < 1 or clip.height < 1:
                    continue
                destination = raw_dir / f"{item.block_id.replace(':', '-')}.png"
                page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False).save(destination)
                item.asset_path = str(destination.resolve())
                item.metadata["index_materialization"] = "pdf-region-crop"
        finally:
            pdf.close()

    processed = PillowImageProcessor().process(temporary, output_dir)
    by_id = {item.block_id: item for item in processed if item.asset_path}
    for original in visuals:
        materialized = by_id.get(original.block_id)
        if materialized:
            original.asset_path = materialized.asset_path
            original.sha256 = materialized.sha256
            original.metadata["materialization_status"] = "complete-at-index"
    return processed


def build_hybrid_index(
    source_dir: Path,
    artifacts_dir: Path,
    index_dir: Path,
    model: str,
    device: str = "cpu",
) -> dict[str, Any]:
    sources = [path for path in sorted(source_dir.iterdir()) if path.is_file()]
    if not sources:
        raise RuntimeError("No source documents were found; hybrid build was stopped.")

    config = PipelineConfig(vision_processor="deferred")
    pipeline = build_default_pipeline(config)
    pixel = PixelRAGVisionProcessor(model=model, device=device)
    office_renderer = MicrosoftOfficeVisualSubprocessRenderer(discover_docling_python())
    office_pdf_renderer = MicrosoftOfficeSubprocessRenderer(discover_docling_python())
    build_visual_root = artifacts_dir / "index-visuals" / f"build-{os.getpid()}"
    documents: list[HybridDocument] = []
    print(f"Stage 1/3: Hybrid recognition of text/table/visual regions ({len(sources)} documents)")
    try:
        for index, source in enumerate(sources, 1):
            print(f"[{index}/{len(sources)}] Ingesting {source.name}")
            documents.append(pipeline.ingest(source, artifacts_dir))
        print("Stage 2/3: Materializing visual regions and running resident Pixel")
        for document in documents:
            materialized = _materialize_visuals_for_index(
                document,
                build_visual_root / document.document_id[:16],
                office_renderer,
                office_pdf_renderer,
            )
            document.vision_results = pixel.process(materialized)
            document.providers["image_processor"] = "pillow@index"
            document.providers["vision_processor"] = "pixelrag@index"
        print("Stage 3/3: Building semantic and visual indexes")
        manifest = write_hybrid_index(documents, index_dir)
        print(
            "Hybrid index complete: "
            f"{manifest['semantic_blocks']} semantic blocks, "
            f"{manifest['image_vectors']} Pixel image vectors"
        )
        return manifest
    finally:
        pixel.close()


def _tokens(text: str) -> set[str]:
    normalized = text.casefold()
    parts = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", normalized)
    tokens: set[str] = set()
    for part in parts:
        tokens.add(part)
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            tokens.update(part)
            tokens.update(part[i : i + 2] for i in range(len(part) - 1))
    return tokens


class HybridSearchEngine:
    def __init__(self, index_dir: Path, model: str, device: str = "cpu") -> None:
        import faiss

        self.index_dir = index_dir
        self.model = model
        self.device = device
        self.manifest = json.loads((index_dir / INDEX_MANIFEST).read_text(encoding="utf-8"))
        self.semantic = json.loads((index_dir / SEMANTIC_INDEX).read_text(encoding="utf-8"))
        self.images = json.loads((index_dir / IMAGE_METADATA).read_text(encoding="utf-8"))
        image_path = index_dir / IMAGE_INDEX
        self.image_index = (
            faiss.deserialize_index(np.frombuffer(image_path.read_bytes(), dtype=np.uint8))
            if image_path.is_file()
            else None
        )
        self.session: PixelRAGEmbeddingSession | None = None

    def status(self) -> dict[str, Any]:
        return {**self.manifest, "model": self.model, "model_loaded": self.session is not None}

    def _semantic_hits(self, query: str, limit: int) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        hits = []
        for record in self.semantic:
            value = f"{record.get('text', '')} {record.get('context', '')}"
            value_tokens = _tokens(value)
            overlap = len(query_tokens & value_tokens)
            if not overlap:
                continue
            score = overlap / max(len(query_tokens), 1)
            if query.casefold() in value.casefold():
                score += 0.25
            hits.append(self._hit(record, min(score, 1.0), "semantic"))
        return sorted(hits, key=lambda item: item["score"], reverse=True)[:limit]

    def _image_hits(self, query: str, limit: int) -> list[dict[str, Any]]:
        if self.image_index is None or not self.images:
            return []
        if self.session is None:
            self.session = PixelRAGEmbeddingSession(self.model, self.device)
        vector = self.session.embed_texts(
            [query], instruction="Retrieve images relevant to the user's query."
        )
        scores, indices = self.image_index.search(vector.astype(np.float32), min(limit, len(self.images)))
        hits = []
        for score, position in zip(scores[0], indices[0]):
            if position < 0:
                continue
            if float(score) < MIN_IMAGE_SCORE:
                continue
            hits.append(self._hit(self.images[int(position)], max(float(score), 0.0), "image"))
        return hits

    @staticmethod
    def _hit(record: dict[str, Any], score: float, channel: str) -> dict[str, Any]:
        provenance = record.get("provenance") or {}
        location = (
            provenance.get("page")
            or provenance.get("slide")
            or provenance.get("sheet")
            or provenance.get("cell_range")
        )
        return {
            **record,
            "score": score,
            "channel": channel,
            "url": record.get("source_path", ""),
            "tile_index": max(int(provenance.get("page") or provenance.get("slide") or 1) - 1, 0),
            "chunk_index": 0,
            "y_offset": 0,
            "location": location,
        }

    def search(self, query: str, limit: int = 8, include_images: bool = True) -> list[dict[str, Any]]:
        candidates = self._semantic_hits(query, limit * 2) + self._image_hits(query, limit * 2)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        hits = candidates[:limit]
        if include_images:
            for hit in hits:
                asset = hit.get("asset_path")
                if asset and Path(asset).is_file():
                    hit["image_base64"] = base64.b64encode(Path(asset).read_bytes()).decode("ascii")
        return hits

    def close(self) -> None:
        if self.session is not None:
            self.session.close()


def serve_hybrid_index(index_dir: Path, model: str, port: int, device: str = "cpu") -> None:
    """Serve Studio-compatible status and mixed-search endpoints."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    engine = HybridSearchEngine(index_dir, model, device)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            if self.path == "/status":
                self._json(200, engine.status())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
            if self.path != "/search":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                queries = payload.get("queries") or []
                results = []
                for query in queries:
                    text = str(query.get("text") or "").strip()
                    if not text:
                        raise ValueError("Each query must contain text")
                    results.append(
                        {
                            "hits": engine.search(
                                text,
                                limit=max(int(payload.get("n_docs", 8)), 1),
                                include_images=bool(payload.get("include_images", True)),
                            )
                        }
                    )
                self._json(200, {"results": results})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            print(f"Hybrid search: {format % args}")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Hybrid search service ready: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.close()

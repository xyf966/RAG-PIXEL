from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Artifact, HybridDocument
from .index_records import (
    IndexBuildConfig,
    IndexRecord,
    TokenCodec,
    build_table_records,
    build_text_records,
    build_visual_record,
)
from .pipeline import PipelineConfig, build_default_pipeline, discover_docling_python
from .processors import PillowImageProcessor, image_content_sha256
from .renderers import MicrosoftOfficeSubprocessRenderer, MicrosoftOfficeVisualSubprocessRenderer
from .vision import PixelRAGEmbeddingSession, PixelRAGVisionProcessor


INDEX_SCHEMA_VERSION = "2.0"
CURRENT_POINTER = "CURRENT"
SNAPSHOTS_DIR = "snapshots"
MANIFEST_FILE = "manifest.json"
BUILD_REPORT_FILE = "build-report.json"
TEXT_INSTRUCTION = "Represent this document passage for retrieval."
TABLE_INSTRUCTION = "Represent this structured table for retrieval."
VALID_VISUAL_TYPES = {"image", "chart", "icon", "diagram"}
OFFICE_NATIVE_METHODS = {
    "doc": {"word-copy-as-picture", "word-native-clipboard-copy"},
    "docx": {"word-copy-as-picture", "word-native-clipboard-copy"},
    "ppt": {"powerpoint-shape-export"},
    "pptx": {"powerpoint-shape-export"},
    "xls": {"excel-native-export"},
    "xlsx": {"excel-native-export"},
}


@dataclass(slots=True)
class VisualIndexBatch:
    records: list[IndexRecord]
    vectors: np.ndarray
    warnings: list[str]
    source_visuals: int
    filtered_visuals: int
    unique_embeddings: int


class SessionTokenCodec(TokenCodec):
    def __init__(self, session: PixelRAGEmbeddingSession) -> None:
        self.session = session

    def encode(self, text: str) -> list[int]:
        return self.session.encode_text(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.session.decode_text(token_ids)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[IndexRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(6):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (attempt + 1))


def _normalize_vectors(
    vectors: Any,
    expected_count: int,
    expected_dimension: int,
    modality: str,
) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if expected_count == 0:
        return np.empty((0, expected_dimension), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (expected_count, expected_dimension):
        raise RuntimeError(
            f"{modality} vectors have shape {matrix.shape}; "
            f"expected ({expected_count}, {expected_dimension})"
        )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(matrix)) or np.any(norms == 0):
        raise RuntimeError(f"{modality} vectors contain zero or non-finite values")
    return matrix / norms[:, None]


def _embed_text_records(
    session: PixelRAGEmbeddingSession,
    records: list[IndexRecord],
    instruction: str,
    config: IndexBuildConfig,
    modality: str,
) -> np.ndarray:
    batches = []
    for start in range(0, len(records), config.text_batch_size):
        values = [item.embedding_text for item in records[start : start + config.text_batch_size]]
        batches.append(np.asarray(session.embed_texts(values, instruction=instruction), dtype=np.float32))
    raw = np.vstack(batches) if batches else np.empty((0, config.expected_dimension), dtype=np.float32)
    return _normalize_vectors(raw, len(records), config.expected_dimension, modality)


def _write_faiss(path: Path, vectors: np.ndarray, dimension: int) -> None:
    import faiss

    index = faiss.IndexFlatIP(dimension)
    if len(vectors):
        index.add(vectors.astype(np.float32))
    path.write_bytes(faiss.serialize_index(index).tobytes())


def _materialize_visuals_for_index(
    document: HybridDocument,
    output_dir: Path,
    office_renderer: MicrosoftOfficeVisualSubprocessRenderer,
    office_pdf_renderer: MicrosoftOfficeSubprocessRenderer | None = None,
    office_visual_policy: str = "require-native",
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
    if temporary and document.document_type in {"doc", "docx", "ppt", "pptx", "xls", "xlsx"}:
        try:
            exports = office_renderer.render(source, output_dir / "office-native", export=True)
        except Exception as exc:
            if office_visual_policy in {"require-com", "require-native"}:
                raise RuntimeError(
                    f"Office native visual export is required for {source.name}: {exc}"
                ) from exc
            exports = []
            print(f"Office native visual export unavailable; using page regions: {exc}", flush=True)
        for item, exported in zip(temporary, exports):
            path = Path(str(exported.get("path") or ""))
            render_method = str(exported.get("render_method") or "")
            native_method = OFFICE_NATIVE_METHODS[document.document_type]
            is_acceptable = (
                office_visual_policy != "require-native"
                or render_method in native_method
            )
            if path.is_file() and is_acceptable:
                item.asset_path = str(path.resolve())
                item.metadata["index_materialization"] = render_method or "office-native-export"
        missing = [item for item in temporary if not item.asset_path or not Path(item.asset_path).is_file()]
        if missing and office_visual_policy == "require-native":
            missing_ids = ", ".join(item.block_id for item in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise RuntimeError(
                f"Office native visual export missed {len(missing)} visual(s) in "
                f"{source.name}: {missing_ids}{suffix}"
            )
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
                if not page_no:
                    item.metadata["filtered"] = "missing-pdf-page"
                    continue
                if page_no < 1 or page_no > pdf.page_count:
                    item.metadata["filtered"] = "pdf-page-out-of-range"
                    continue
                if not item.provenance.bbox or len(item.provenance.bbox) != 4:
                    item.metadata["filtered"] = "missing-or-invalid-pdf-bbox"
                    continue
                page = pdf.load_page(page_no - 1)
                box = item.provenance.bbox
                if max(box) <= 1.01:
                    clip = pymupdf.Rect(box[0] * page.rect.width, box[1] * page.rect.height, box[2] * page.rect.width, box[3] * page.rect.height)
                else:
                    clip = pymupdf.Rect(*box)
                clip &= page.rect
                if clip.is_empty or clip.width < 1 or clip.height < 1:
                    item.metadata["filtered"] = "empty-pdf-region"
                    continue
                destination = raw_dir / f"{item.block_id.replace(':', '-')}.png"
                page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False).save(destination)
                item.asset_path = str(destination.resolve())
                item.metadata["index_materialization"] = "pdf-region-crop"
        finally:
            pdf.close()

    processed = PillowImageProcessor(retain_duplicates=True).process(temporary, output_dir)
    by_id = {item.block_id: item for item in processed if item.asset_path}
    temporary_by_id = {item.block_id: item for item in temporary}
    for original in visuals:
        materialized = by_id.get(original.block_id)
        if materialized:
            original.asset_path = materialized.asset_path
            original.sha256 = materialized.sha256
            original.metadata["materialization_status"] = "complete-at-index"
        else:
            attempted = temporary_by_id[original.block_id]
            original.metadata["materialization_status"] = "filtered-at-index"
            original.metadata["index_filter_reason"] = attempted.metadata.get(
                "filtered", "materialization-failed"
            )
            if attempted.metadata.get("asset_error"):
                original.metadata["asset_error"] = attempted.metadata["asset_error"]
    return processed


def prepare_visual_index(
    documents: list[HybridDocument],
    output_dir: Path,
    session: PixelRAGEmbeddingSession,
    config: IndexBuildConfig,
    office_renderer: MicrosoftOfficeVisualSubprocessRenderer,
    office_pdf_renderer: MicrosoftOfficeSubprocessRenderer | None = None,
) -> VisualIndexBatch:
    visual_root = output_dir / "visual-assets"
    warnings: list[str] = []
    source_visuals = 0
    filtered_visuals = 0
    global_block_ids: set[str] = set()
    materialized_by_document: list[tuple[HybridDocument, list[Artifact]]] = []

    selected_by_document: list[tuple[HybridDocument, list[Artifact]]] = []
    for document in documents:
        selected = [
            item for item in document.artifacts
            if item.kind == "visual" and item.metadata.get("indexable", True)
        ]
        for item in document.artifacts:
            if item.kind == "visual" and not item.metadata.get("indexable", True):
                item.metadata["materialization_status"] = "skipped-not-indexable"
        source_visuals += len(selected)
        for artifact in selected:
            if not artifact.block_id:
                raise RuntimeError("Visual index input contains an empty block_id")
            if artifact.block_id in global_block_ids:
                raise RuntimeError(f"Duplicate visual block_id: {artifact.block_id}")
            global_block_ids.add(artifact.block_id)
            if artifact.visual_type not in VALID_VISUAL_TYPES:
                raise RuntimeError(
                    f"Unsupported visual_type for {artifact.block_id}: {artifact.visual_type!r}"
                )
        selected_by_document.append((document, selected))

    visual_root.mkdir(parents=True, exist_ok=True)
    for document, selected in selected_by_document:
        materialized = _materialize_visuals_for_index(
            document,
            visual_root / document.document_id[:16],
            office_renderer,
            office_pdf_renderer,
            config.office_visual_policy,
        )
        actual = {item.block_id for item in materialized}
        for artifact in selected:
            if artifact.block_id in actual:
                continue
            filtered_visuals += 1
            reason = artifact.metadata.get("index_filter_reason", "materialization-failed")
            warnings.append(
                f"Visual {document.document_id}/{artifact.block_id} failed the index quality gate: {reason}"
            )
        if any(item.block_id not in global_block_ids for item in materialized):
            raise RuntimeError("A non-visual artifact entered Pixel materialization")
        materialized_by_document.append((document, materialized))

    representatives: dict[str, Artifact] = {}
    for _document, artifacts in materialized_by_document:
        for artifact in artifacts:
            if not artifact.sha256:
                raise RuntimeError(f"Materialized visual lacks SHA-256: {artifact.block_id}")
            representative = representatives.setdefault(artifact.sha256, artifact)
            if representative.block_id != artifact.block_id:
                artifact.metadata["embedding_duplicate_of"] = representative.block_id

    pixel = PixelRAGVisionProcessor(
        model=session.model_name,
        device=session.requested_device,
        session=session,
    )
    unique_results = pixel.process(list(representatives.values()))
    result_by_block = {result.block_id: result for result in unique_results}
    missing_representatives = [
        artifact
        for artifact in representatives.values()
        if result_by_block.get(artifact.block_id) is None
        or not result_by_block[artifact.block_id].vector
    ]
    if missing_representatives:
        print(
            f"Retrying {len(missing_representatives)} visual embedding result(s)",
            flush=True,
        )
        retry_results = pixel.process(missing_representatives)
        result_by_block.update({result.block_id: result for result in retry_results})
    vector_by_hash: dict[str, list[float]] = {}
    for digest, artifact in representatives.items():
        result = result_by_block.get(artifact.block_id)
        if result is None or not result.vector:
            raise RuntimeError(f"Pixel produced no vector for visual {artifact.block_id}")
        vector_by_hash[digest] = result.vector

    records: list[IndexRecord] = []
    raw_vectors: list[list[float]] = []
    for document, artifacts in materialized_by_document:
        for visual_ordinal, artifact in enumerate(artifacts):
            record = build_visual_record(document, artifact, visual_ordinal)
            assert artifact.asset_path is not None and artifact.sha256 is not None
            record.asset_path = (
                Path(artifact.asset_path).resolve().relative_to(output_dir.resolve()).as_posix()
            )
            records.append(record)
            raw_vectors.append(vector_by_hash[artifact.sha256])

    vectors = _normalize_vectors(
        raw_vectors, len(records), config.expected_dimension, "visual"
    )
    return VisualIndexBatch(
        records=records,
        vectors=vectors,
        warnings=warnings,
        source_visuals=source_visuals,
        filtered_visuals=filtered_visuals,
        unique_embeddings=len(representatives),
    )


def build_hybrid_index(
    source_dir: Path,
    artifacts_dir: Path,
    index_dir: Path,
    model: str,
    device: str = "cpu",
    config: IndexBuildConfig | None = None,
) -> dict[str, Any]:
    config = config or IndexBuildConfig()
    sources = [path for path in sorted(source_dir.iterdir()) if path.is_file()]
    if not sources:
        raise RuntimeError("No source documents were found; hybrid build was stopped.")

    pipeline = build_default_pipeline(PipelineConfig(vision_processor="deferred"))
    session = PixelRAGEmbeddingSession(model, device)
    office_renderer = MicrosoftOfficeVisualSubprocessRenderer(discover_docling_python())
    office_pdf_renderer = MicrosoftOfficeSubprocessRenderer(discover_docling_python())
    documents: list[HybridDocument] = []
    print(f"Stage 1/4: Hybrid recognition ({len(sources)} documents)", flush=True)
    try:
        for index, source in enumerate(sources, 1):
            print(f"[{index}/{len(sources)}] Ingesting {source.name}", flush=True)
            documents.append(pipeline.ingest(source, artifacts_dir))
        return build_index_snapshot(
            documents=documents,
            source_files=sources,
            index_dir=index_dir,
            session=session,
            config=config,
            office_renderer=office_renderer,
            office_pdf_renderer=office_pdf_renderer,
        )
    finally:
        session.close()


def _source_entries(source_files: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
            "sha256": _sha256_file(path),
        }
        for path in source_files
    ]


def _validate_snapshot(snapshot: Path, config: IndexBuildConfig) -> dict[str, str]:
    import faiss

    checksums: dict[str, str] = {}
    record_ids: set[str] = set()
    for modality in ("text", "table", "visual"):
        index_path = snapshot / f"{modality}.faiss"
        metadata_path = snapshot / f"{modality}-metadata.jsonl"
        if not index_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Missing {modality} index artifact")
        index = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype=np.uint8))
        metadata_rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        metadata_count = len(metadata_rows)
        if index.ntotal != metadata_count:
            raise RuntimeError(
                f"{modality} index count {index.ntotal} != metadata count {metadata_count}"
            )
        if index.d != config.expected_dimension:
            raise RuntimeError(
                f"{modality} index dimension {index.d} != {config.expected_dimension}"
            )
        if index.ntotal:
            vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
            index.reconstruct_n(0, index.ntotal, vectors)
            norms = np.linalg.norm(vectors, axis=1)
            if not np.all(np.isfinite(vectors)) or np.any(norms == 0):
                raise RuntimeError(f"{modality} index contains zero or non-finite vectors")
            if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"{modality} index contains non-normalized vectors")
        for row in metadata_rows:
            provenance = row.get("provenance") or {}
            if (
                row.get("modality") != modality
                or not row.get("record_id")
                or not row.get("document_id")
                or not row.get("source_block_ids")
                or not row.get("source_path")
                or not provenance.get("source_file")
            ):
                raise RuntimeError(f"{modality} metadata contains an untraceable record")
            record_id = row["record_id"]
            if record_id in record_ids:
                raise RuntimeError(f"Duplicate index record_id: {record_id}")
            record_ids.add(record_id)
            if modality == "visual":
                asset_path = Path(str(row.get("asset_path") or ""))
                if asset_path.is_absolute() or ".." in asset_path.parts:
                    raise RuntimeError("Visual metadata contains an unsafe asset path")
                asset = snapshot / asset_path
                expected_sha256 = (row.get("original_content") or {}).get("sha256")
                if not asset.is_file() or not expected_sha256:
                    raise RuntimeError("Visual metadata points to a missing asset")
                if image_content_sha256(asset) != expected_sha256:
                    raise RuntimeError("Visual asset checksum does not match its metadata")
        checksums[index_path.name] = _sha256_file(index_path)
        checksums[metadata_path.name] = _sha256_file(metadata_path)
    return checksums


def _validate_manifest_files(snapshot: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Index manifest has no file checksums")
    for relative_name, expected_sha256 in files.items():
        relative_path = Path(str(relative_name))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("Index manifest contains an unsafe file path")
        path = snapshot / relative_path
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Index snapshot file failed checksum validation: {relative_name}")


def build_index_snapshot(
    documents: list[HybridDocument],
    source_files: list[Path],
    index_dir: Path,
    session: PixelRAGEmbeddingSession,
    config: IndexBuildConfig,
    office_renderer: MicrosoftOfficeVisualSubprocessRenderer,
    office_pdf_renderer: MicrosoftOfficeSubprocessRenderer | None = None,
) -> dict[str, Any]:
    if any(document.schema_version != "1.1" for document in documents):
        raise RuntimeError("Index schema 2.0 requires HybridDocument schema 1.1")
    index_dir.mkdir(parents=True, exist_ok=True)
    snapshots = index_dir / SNAPSHOTS_DIR
    snapshots.mkdir(parents=True, exist_ok=True)
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    staging = index_dir / f".staging-{build_id}"
    staging.mkdir(parents=True)
    warnings = [
        f"Document {document.document_id}: {warning}"
        for document in documents
        for warning in document.warnings
    ]
    started_at = datetime.now(timezone.utc)
    try:
        codec = SessionTokenCodec(session)
        print("Stage 2/4: Structure-aware text and table chunking", flush=True)
        text_records = [
            record for document in documents
            for record in build_text_records(document, codec, config)
        ]
        table_records = [
            record for document in documents
            for record in build_table_records(document, codec, config)
        ]

        print("Stage 3/4: Visual materialization and unified embeddings", flush=True)
        visual_batch = prepare_visual_index(
            documents,
            staging,
            session,
            config,
            office_renderer,
            office_pdf_renderer,
        )
        warnings.extend(visual_batch.warnings)
        visual_records = visual_batch.records

        text_vectors = _embed_text_records(
            session, text_records, TEXT_INSTRUCTION, config, "text"
        )
        table_vectors = _embed_text_records(
            session, table_records, TABLE_INSTRUCTION, config, "table"
        )
        visual_vectors = visual_batch.vectors

        print("Stage 4/4: Persisting and validating schema 2.0 snapshot", flush=True)
        for modality, records, vectors in (
            ("text", text_records, text_vectors),
            ("table", table_records, table_vectors),
            ("visual", visual_records, visual_vectors),
        ):
            _write_jsonl(staging / f"{modality}-metadata.jsonl", records)
            _write_faiss(staging / f"{modality}.faiss", vectors, config.expected_dimension)

        checksums = _validate_snapshot(staging, config)
        completed_at = datetime.now(timezone.utc)
        counts = {
            "documents": len(documents),
            "source_artifacts": sum(len(document.artifacts) for document in documents),
            "source_text_artifacts": sum(
                item.kind == "text" for document in documents for item in document.artifacts
            ),
            "source_table_artifacts": sum(
                item.kind == "table" for document in documents for item in document.artifacts
            ),
            "source_visual_artifacts": sum(
                item.kind == "visual" for document in documents for item in document.artifacts
            ),
            "text_records": len(text_records),
            "table_records": len(table_records),
            "visual_records": len(visual_records),
            "unique_visual_embeddings": visual_batch.unique_embeddings,
            "filtered_visuals": visual_batch.filtered_visuals,
        }
        report = {
            "build_id": build_id,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": (completed_at - started_at).total_seconds(),
            "counts": counts,
            "warnings": warnings,
        }
        _write_json(staging / BUILD_REPORT_FILE, report)
        checksums[BUILD_REPORT_FILE] = _sha256_file(staging / BUILD_REPORT_FILE)
        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "build_id": build_id,
            "status": "complete",
            "built_at": completed_at.isoformat(),
            "model": session.model_name,
            "device": session.resolved_device or session.requested_device,
            "vector_dimension": config.expected_dimension,
            "index_type": "faiss.IndexFlatIP",
            "normalized_vectors": True,
            "pixel_input_kinds": ["visual"],
            "chunking": asdict(config),
            "sources": _source_entries(source_files),
            "counts": counts,
            "files": checksums,
            "warnings": warnings,
        }
        _write_json(staging / MANIFEST_FILE, manifest)

        destination = snapshots / build_id
        _replace_with_retry(staging, destination)
        pointer = index_dir / f"{CURRENT_POINTER}.tmp"
        pointer.write_text(build_id + "\n", encoding="utf-8")
        _replace_with_retry(pointer, index_dir / CURRENT_POINTER)
        print(
            f"Index snapshot complete: {len(text_records)} text, "
            f"{len(table_records)} table, {len(visual_records)} visual records",
            flush=True,
        )
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def current_snapshot(index_dir: Path) -> Path:
    pointer = index_dir / CURRENT_POINTER
    if not pointer.is_file():
        legacy = index_dir / "hybrid-index.json"
        if legacy.is_file():
            raise RuntimeError("Legacy index schema detected; rebuild the project for schema 2.0")
        raise RuntimeError("No published index snapshot")
    build_id = pointer.read_text(encoding="utf-8").strip()
    snapshot = index_dir / SNAPSHOTS_DIR / build_id
    manifest_path = snapshot / MANIFEST_FILE
    if not build_id or not manifest_path.is_file():
        raise RuntimeError("CURRENT points to a missing index snapshot")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise RuntimeError("Unsupported index schema; rebuild the project")
    office_suffixes = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    has_office_source = any(
        Path(str(source.get("path") or "")).suffix.lower() in office_suffixes
        for source in manifest.get("sources", [])
    )
    has_office_visuals = bool(
        has_office_source
        and (manifest.get("counts") or {}).get("source_visual_artifacts", 0)
    )
    policy = (manifest.get("chunking") or {}).get("office_visual_policy")
    if has_office_visuals and policy not in {"require-com", "require-native"}:
        raise RuntimeError(
            "Office visual index was built without required COM validation; rebuild required"
        )
    _validate_manifest_files(snapshot, manifest)
    dimension = manifest.get("vector_dimension")
    if not isinstance(dimension, int) or dimension < 1:
        raise RuntimeError("Index manifest contains an invalid vector dimension")
    _validate_snapshot(snapshot, IndexBuildConfig(expected_dimension=dimension))
    return snapshot


class HybridSearchEngine:
    """Compatibility constructor; new code should import from hybrid_input.retrieval."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from .retrieval import HybridSearchEngine as RetrievalEngine

        return RetrievalEngine(*args, **kwargs)


def serve_hybrid_index(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("Retrieval layer is disabled while index schema 2.0 is being introduced")

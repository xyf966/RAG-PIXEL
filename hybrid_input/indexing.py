from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Artifact, ArtifactBundle, HybridDocument
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


def _office_visual_identity(document_type: str, value: Any) -> tuple[Any, ...] | None:
    """Build the stable COM identity shared by recognition and export records."""
    if isinstance(value, Artifact):
        metadata = value.metadata
        shape_index = metadata.get("office_shape_index")
        collection = metadata.get("office_collection")
        if document_type in {"ppt", "pptx"}:
            container = value.provenance.slide or value.provenance.page
            return ("ppt", container, shape_index) if container and shape_index else None
        if document_type in {"xls", "xlsx"}:
            container = value.provenance.sheet
            return ("excel", container, shape_index) if container and shape_index else None
        container = value.provenance.page
        return (
            ("word", container, collection, shape_index)
            if container and collection and shape_index else None
        )

    if not isinstance(value, dict):
        return None
    shape_index = value.get("shape_index") or value.get("collection_index")
    if document_type in {"ppt", "pptx"}:
        container = value.get("slide")
        return ("ppt", container, shape_index) if container and shape_index else None
    if document_type in {"xls", "xlsx"}:
        container = value.get("sheet")
        return ("excel", container, shape_index) if container and shape_index else None
    container = value.get("page")
    collection = value.get("collection")
    return (
        ("word", container, collection, shape_index)
        if container and collection and shape_index else None
    )


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
        exports_by_identity = {
            identity: exported
            for exported in exports
            if (identity := _office_visual_identity(document.document_type, exported)) is not None
        }
        recognized_identities = {
            identity for item in temporary
            if (identity := _office_visual_identity(document.document_type, item)) is not None
        }
        remaining_exports = iter(
            exported for exported in exports
            if _office_visual_identity(document.document_type, exported) not in recognized_identities
        )
        for item in temporary:
            identity = _office_visual_identity(document.document_type, item)
            exported = exports_by_identity.get(identity) if identity is not None else None
            if exported is None:
                exported = next(remaining_exports, None)
            if exported is None:
                continue
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
    force: bool = False,
) -> dict[str, Any]:
    config = config or IndexBuildConfig()
    sources = [path for path in sorted(source_dir.iterdir()) if path.is_file()]
    if not sources:
        raise RuntimeError("No source documents were found; hybrid build was stopped.")

    previous_snapshot: Path | None = None
    previous_manifest: dict[str, Any] | None = None
    if not force:
        try:
            candidate = current_snapshot(index_dir)
            candidate_manifest = json.loads(
                (candidate / MANIFEST_FILE).read_text(encoding="utf-8")
            )
            if _compatible_incremental_base(candidate_manifest, model, config):
                previous_snapshot = candidate
                previous_manifest = candidate_manifest
        except (OSError, ValueError, RuntimeError):
            pass

    previous_sources = {
        str(entry.get("path") or ""): entry
        for entry in (previous_manifest or {}).get("sources", [])
        if entry.get("path")
    }
    source_entries = _source_entries(sources, previous_sources)
    entry_by_path = {str(entry["path"]): entry for entry in source_entries}
    exact_same_sources = bool(previous_manifest) and previous_sources == entry_by_path
    reusable_documents: dict[str, HybridDocument] = {}
    changed_sources: list[Path] = []
    for source in sources:
        resolved = str(source.resolve())
        current_entry = entry_by_path[resolved]
        previous_entry = previous_sources.get(resolved)
        digest = str(current_entry["sha256"])
        if previous_entry and previous_entry.get("sha256") == digest:
            cached = _cached_hybrid_document(source, artifacts_dir, digest)
            if cached is not None:
                reusable_documents[resolved] = cached
                continue
        changed_sources.append(source)

    if previous_manifest and exact_same_sources and not changed_sources:
        print("Index is already current; no documents require processing", flush=True)
        return previous_manifest

    pipeline = build_default_pipeline(PipelineConfig(vision_processor="deferred"))
    session = PixelRAGEmbeddingSession(model, device)
    office_renderer = MicrosoftOfficeVisualSubprocessRenderer(discover_docling_python())
    office_pdf_renderer = MicrosoftOfficeSubprocessRenderer(discover_docling_python())
    changed_documents: dict[str, HybridDocument] = {}
    print(
        "Stage 1/4: Hybrid recognition "
        f"({len(changed_sources)} changed, {len(reusable_documents)} reused documents)",
        flush=True,
    )
    try:
        for index, source in enumerate(changed_sources, 1):
            print(f"[{index}/{len(changed_sources)}] Ingesting {source.name}", flush=True)
            changed_documents[str(source.resolve())] = pipeline.ingest(source, artifacts_dir)
        documents = [
            reusable_documents.get(str(source.resolve()))
            or changed_documents[str(source.resolve())]
            for source in sources
        ]
        reused_document_ids = {
            document.document_id for document in reusable_documents.values()
        }
        removed_document_ids = {
            str(entry.get("sha256") or "")
            for entry in previous_sources.values()
            if entry.get("sha256")
        } - {document.document_id for document in documents}
        return build_index_snapshot(
            documents=documents,
            source_files=sources,
            index_dir=index_dir,
            session=session,
            config=config,
            office_renderer=office_renderer,
            office_pdf_renderer=office_pdf_renderer,
            previous_snapshot=previous_snapshot,
            previous_manifest=previous_manifest,
            reused_document_ids=reused_document_ids,
            changed_document_ids={
                document.document_id for document in changed_documents.values()
            },
            removed_document_ids=removed_document_ids,
            source_entries=source_entries,
        )
    finally:
        session.close()


def _source_entries(
    source_files: list[Path],
    reusable_entries: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    reusable_entries = reusable_entries or {}
    for path in source_files:
        resolved = str(path.resolve())
        stat = path.stat()
        previous = reusable_entries.get(resolved) or {}
        digest = (
            str(previous["sha256"])
            if previous.get("sha256")
            and previous.get("size") == stat.st_size
            and previous.get("modified_ns") == stat.st_mtime_ns
            else _sha256_file(path)
        )
        entries.append(
            {
                "path": resolved,
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "sha256": digest,
            }
        )
    return entries


def _cached_hybrid_document(source: Path, artifacts_dir: Path, digest: str) -> HybridDocument | None:
    path = artifacts_dir / f"{source.stem}-{digest[:12]}" / "hybrid-document.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # HybridDocument and ArtifactBundle intentionally share the artifact
        # payload, but only the latter carries a parser field.  Cached ingest
        # output is a HybridDocument, so provide the irrelevant parser value
        # solely to reuse the strict artifact/provenance decoder.
        bundle = ArtifactBundle.from_dict(
            {**payload, "parser": payload.get("parser", "cached")}
        )
    except Exception:
        return None
    if (
        bundle.schema_version != "1.1"
        or bundle.document_id != digest
        or Path(bundle.source_path).resolve() != source.resolve()
    ):
        return None
    return HybridDocument(
        document_id=bundle.document_id,
        source_path=bundle.source_path,
        document_type=bundle.document_type,
        artifacts=bundle.artifacts,
        providers=dict(payload.get("providers") or {}),
        warnings=bundle.warnings,
        schema_version=bundle.schema_version,
    )


def _load_reused_modality(
    snapshot: Path,
    modality: str,
    document_ids: set[str],
    expected_dimension: int,
) -> tuple[list[IndexRecord], np.ndarray]:
    import faiss

    rows = [
        json.loads(line)
        for line in (snapshot / f"{modality}-metadata.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    index = faiss.deserialize_index(
        np.frombuffer((snapshot / f"{modality}.faiss").read_bytes(), dtype=np.uint8)
    )
    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    if index.ntotal:
        index.reconstruct_n(0, index.ntotal, vectors)
    selected = [
        position
        for position, row in enumerate(rows)
        if str(row.get("document_id") or "") in document_ids
    ]
    records = [IndexRecord(**rows[position]) for position in selected]
    if not selected:
        return records, np.empty((0, expected_dimension), dtype=np.float32)
    return records, vectors[selected].copy()


def _copy_reused_visual_assets(
    snapshot: Path,
    staging: Path,
    records: list[IndexRecord],
) -> None:
    copied: set[str] = set()
    for record in records:
        relative = Path(str(record.asset_path or ""))
        relative_key = relative.as_posix()
        if not relative_key or relative_key in copied:
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Reused visual record contains an unsafe asset path")
        source = snapshot / relative
        if not source.is_file():
            raise RuntimeError(f"Reused visual asset is missing: {relative_key}")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        copied.add(relative_key)


def _merge_modality_rows(
    documents: list[HybridDocument],
    reused_records: list[IndexRecord],
    reused_vectors: np.ndarray,
    changed_records: list[IndexRecord],
    changed_vectors: np.ndarray,
    expected_dimension: int,
) -> tuple[list[IndexRecord], np.ndarray]:
    source_order = {
        str(Path(document.source_path).resolve()): position
        for position, document in enumerate(documents)
    }
    pairs = list(zip(reused_records, reused_vectors)) + list(
        zip(changed_records, changed_vectors)
    )
    pairs.sort(
        key=lambda pair: source_order.get(
            str(Path(pair[0].source_path).resolve()), len(source_order)
        )
    )
    if not pairs:
        return [], np.empty((0, expected_dimension), dtype=np.float32)
    return [record for record, _vector in pairs], np.vstack(
        [vector for _record, vector in pairs]
    ).astype(np.float32)


def _compatible_incremental_base(
    manifest: dict[str, Any], model: str, config: IndexBuildConfig
) -> bool:
    previous_model = str(manifest.get("model") or "")
    requested_model = str(model)
    try:
        if Path(previous_model).exists() or Path(requested_model).exists():
            models_match = Path(previous_model).resolve() == Path(requested_model).resolve()
        else:
            models_match = previous_model == requested_model
    except OSError:
        models_match = previous_model == requested_model
    return (
        manifest.get("schema_version") == INDEX_SCHEMA_VERSION
        and models_match
        and manifest.get("vector_dimension") == config.expected_dimension
        and (manifest.get("chunking") or {}) == asdict(config)
    )


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
        if modality == "text":
            text_record_ids = {str(row.get("record_id")) for row in metadata_rows}
            for row in metadata_rows:
                for neighbor_field in ("previous_record_id", "next_record_id"):
                    neighbor = row.get(neighbor_field)
                    if neighbor is not None and neighbor not in text_record_ids:
                        raise RuntimeError(
                            f"Text metadata contains a broken {neighbor_field}: {neighbor}"
                        )
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
    previous_snapshot: Path | None = None,
    previous_manifest: dict[str, Any] | None = None,
    reused_document_ids: set[str] | None = None,
    changed_document_ids: set[str] | None = None,
    removed_document_ids: set[str] | None = None,
    source_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if any(document.schema_version != "1.1" for document in documents):
        raise RuntimeError("Index schema 2.0 requires HybridDocument schema 1.1")
    index_dir.mkdir(parents=True, exist_ok=True)
    snapshots = index_dir / SNAPSHOTS_DIR
    snapshots.mkdir(parents=True, exist_ok=True)
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    staging = index_dir / f".staging-{build_id}"
    staging.mkdir(parents=True)
    reused_document_ids = set(reused_document_ids or ())
    changed_document_ids = set(changed_document_ids or ())
    removed_document_ids = set(removed_document_ids or ())
    changed_documents = [
        document for document in documents
        if not reused_document_ids or document.document_id in changed_document_ids
    ]
    if previous_snapshot is None:
        reused_document_ids.clear()
        changed_documents = list(documents)
        changed_document_ids = {document.document_id for document in documents}
    previous_warnings = list((previous_manifest or {}).get("warnings") or [])
    excluded_document_ids = changed_document_ids | removed_document_ids
    warnings = [
        warning for warning in previous_warnings
        if not any(document_id and document_id in warning for document_id in excluded_document_ids)
    ]
    warnings.extend(
        f"Document {document.document_id}: {warning}"
        for document in changed_documents
        for warning in document.warnings
    )
    started_at = datetime.now(timezone.utc)
    try:
        codec = SessionTokenCodec(session)
        print("Stage 2/4: Structure-aware text and table chunking", flush=True)
        changed_text_records = [
            record for document in changed_documents
            for record in build_text_records(document, codec, config)
        ]
        changed_table_records = [
            record for document in changed_documents
            for record in build_table_records(document, codec, config)
        ]

        print("Stage 3/4: Visual materialization and unified embeddings", flush=True)
        visual_batch = prepare_visual_index(
            changed_documents,
            staging,
            session,
            config,
            office_renderer,
            office_pdf_renderer,
        )
        warnings.extend(visual_batch.warnings)
        changed_visual_records = visual_batch.records

        changed_text_vectors = _embed_text_records(
            session, changed_text_records, TEXT_INSTRUCTION, config, "text"
        )
        changed_table_vectors = _embed_text_records(
            session, changed_table_records, TABLE_INSTRUCTION, config, "table"
        )
        changed_visual_vectors = visual_batch.vectors

        reused: dict[str, tuple[list[IndexRecord], np.ndarray]] = {}
        for modality in ("text", "table", "visual"):
            reused[modality] = (
                _load_reused_modality(
                    previous_snapshot,
                    modality,
                    reused_document_ids,
                    config.expected_dimension,
                )
                if previous_snapshot is not None and reused_document_ids
                else ([], np.empty((0, config.expected_dimension), dtype=np.float32))
            )
        if previous_snapshot is not None:
            _copy_reused_visual_assets(
                previous_snapshot, staging, reused["visual"][0]
            )

        text_records, text_vectors = _merge_modality_rows(
            documents,
            reused["text"][0], reused["text"][1],
            changed_text_records, changed_text_vectors,
            config.expected_dimension,
        )
        table_records, table_vectors = _merge_modality_rows(
            documents,
            reused["table"][0], reused["table"][1],
            changed_table_records, changed_table_vectors,
            config.expected_dimension,
        )
        visual_records, visual_vectors = _merge_modality_rows(
            documents,
            reused["visual"][0], reused["visual"][1],
            changed_visual_records, changed_visual_vectors,
            config.expected_dimension,
        )

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
        source_visual_artifacts = sum(
            item.kind == "visual" for document in documents for item in document.artifacts
        )
        selected_visual_artifacts = sum(
            item.kind == "visual" and item.metadata.get("indexable", True)
            for document in documents
            for item in document.artifacts
        )
        counts = {
            "documents": len(documents),
            "source_artifacts": sum(len(document.artifacts) for document in documents),
            "source_text_artifacts": sum(
                item.kind == "text" for document in documents for item in document.artifacts
            ),
            "source_table_artifacts": sum(
                item.kind == "table" for document in documents for item in document.artifacts
            ),
            "source_visual_artifacts": source_visual_artifacts,
            "text_records": len(text_records),
            "table_records": len(table_records),
            "visual_records": len(visual_records),
            "unique_visual_embeddings": len(
                {record.content_hash for record in visual_records}
            ),
            "filtered_visuals": max(0, selected_visual_artifacts - len(visual_records)),
        }
        incremental = {
            "base_build_id": (
                str((previous_manifest or {}).get("build_id") or "") or None
            ),
            "reused_documents": len(reused_document_ids),
            "processed_documents": len(changed_documents),
            "removed_documents": len(removed_document_ids),
        }
        report = {
            "build_id": build_id,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": (completed_at - started_at).total_seconds(),
            "counts": counts,
            "incremental": incremental,
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
            "sources": (
                source_entries
                if source_entries is not None
                else _source_entries(source_files)
            ),
            "counts": counts,
            "incremental": incremental,
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
        print(
            "Incremental summary: "
            f"{incremental['processed_documents']} processed, "
            f"{incremental['reused_documents']} reused, "
            f"{incremental['removed_documents']} removed documents",
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

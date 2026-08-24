from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from .indexing import CURRENT_POINTER, MANIFEST_FILE, current_snapshot
from .retrieval_contracts import (
    VALID_MODALITIES,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResponse,
)


QUERY_INSTRUCTION = "Represent this query for retrieving relevant document content."
RRF_K = 60.0
DEFAULT_MODALITY_WEIGHTS = MappingProxyType(
    {"text": 1.0, "table": 1.0, "visual": 1.10}
)
DEFAULT_MIN_RAW_SCORES = MappingProxyType(
    {"text": 0.40, "table": 0.40, "visual": 0.40}
)


class QueryEmbedder(Protocol):
    model_name: str

    def embed_query(self, query_text: str) -> Any: ...


class PixelRAGQueryEmbedder:
    def __init__(self, session: Any, instruction: str = QUERY_INSTRUCTION) -> None:
        model_name = getattr(session, "model_name", None)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("PixelRAG query session has no model_name")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Query embedding instruction must not be empty")
        self.session = session
        self.model_name = model_name.strip()
        self.instruction = instruction.strip()

    def embed_query(self, query_text: str) -> Any:
        return self.session.embed_texts([query_text], instruction=self.instruction)


@dataclass(frozen=True, slots=True)
class LoadedSnapshot:
    build_id: str
    model: str
    dimension: int
    snapshot_path: Path
    manifest: Mapping[str, Any]
    indexes: Mapping[str, Any]
    metadata: Mapping[str, tuple[Mapping[str, Any], ...]]


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    modality: str
    modality_rank: int
    index_position: int
    raw_score: float
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    snapshot: LoadedSnapshot
    candidates: Mapping[str, tuple[SearchCandidate, ...]]


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: SearchCandidate
    fusion_score: float


@dataclass(frozen=True, slots=True)
class RankedSearchResult:
    snapshot: LoadedSnapshot
    candidates: tuple[RankedCandidate, ...]


class SnapshotStore:
    """Load and atomically cache the immutable snapshot selected by CURRENT."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir).expanduser().resolve()
        self._lock = threading.RLock()
        self._loaded: LoadedSnapshot | None = None

    @property
    def cached_build_id(self) -> str | None:
        with self._lock:
            return self._loaded.build_id if self._loaded else None

    def _pointer_build_id(self) -> str | None:
        pointer = self.index_dir / CURRENT_POINTER
        if not pointer.is_file():
            return None
        value = pointer.read_text(encoding="utf-8").strip()
        return value or None

    @staticmethod
    def _read_manifest(snapshot_path: Path) -> dict[str, Any]:
        manifest_path = snapshot_path / MANIFEST_FILE
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read index manifest: {manifest_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Index manifest must contain a JSON object")
        return payload

    @staticmethod
    def _read_metadata(path: Path) -> tuple[Mapping[str, Any], ...]:
        rows: list[Mapping[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Invalid metadata JSON at {path.name}:{line_number}"
                        ) from exc
                    if not isinstance(payload, dict):
                        raise RuntimeError(
                            f"Metadata row must be an object at {path.name}:{line_number}"
                        )
                    rows.append(MappingProxyType(payload))
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"Unable to read index metadata: {path}") from exc
        return tuple(rows)

    def _load_snapshot(self, snapshot_path: Path) -> LoadedSnapshot:
        import faiss

        manifest = self._read_manifest(snapshot_path)
        build_id = manifest.get("build_id")
        model = manifest.get("model")
        dimension = manifest.get("vector_dimension")
        if not isinstance(build_id, str) or not build_id.strip():
            raise RuntimeError("Index manifest contains no build_id")
        build_id = build_id.strip()
        if build_id != snapshot_path.name:
            raise RuntimeError("Index manifest build_id does not match its snapshot directory")
        if manifest.get("status") != "complete":
            raise RuntimeError("Index snapshot is not marked complete")
        if not isinstance(model, str) or not model.strip():
            raise RuntimeError("Index manifest contains no embedding model")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise RuntimeError("Index manifest contains an invalid vector dimension")
        if manifest.get("index_type") != "faiss.IndexFlatIP":
            raise RuntimeError("Retrieval supports only faiss.IndexFlatIP snapshots")
        if manifest.get("normalized_vectors") is not True:
            raise RuntimeError("Retrieval requires normalized index vectors")

        indexes: dict[str, Any] = {}
        metadata: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for modality in VALID_MODALITIES:
            index_path = snapshot_path / f"{modality}.faiss"
            metadata_path = snapshot_path / f"{modality}-metadata.jsonl"
            try:
                serialized = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
                index = faiss.deserialize_index(serialized)
            except Exception as exc:
                raise RuntimeError(f"Unable to load {modality} FAISS index") from exc
            rows = self._read_metadata(metadata_path)
            if index.d != dimension:
                raise RuntimeError(
                    f"{modality} index dimension {index.d} != manifest dimension {dimension}"
                )
            if index.ntotal != len(rows):
                raise RuntimeError(
                    f"{modality} index count {index.ntotal} != metadata count {len(rows)}"
                )
            for row_number, row in enumerate(rows):
                if row.get("modality") != modality:
                    raise RuntimeError(
                        f"{modality} metadata row {row_number} has a mismatched modality"
                    )
            indexes[modality] = index
            metadata[modality] = rows

        return LoadedSnapshot(
            build_id=build_id,
            model=model.strip(),
            dimension=dimension,
            snapshot_path=snapshot_path,
            manifest=MappingProxyType(manifest),
            indexes=MappingProxyType(indexes),
            metadata=MappingProxyType(metadata),
        )

    def get_snapshot(self) -> LoadedSnapshot:
        with self._lock:
            selected_build_id = self._pointer_build_id()
            if self._loaded is not None and selected_build_id == self._loaded.build_id:
                return self._loaded

            snapshot_path = current_snapshot(self.index_dir)
            loaded = self._load_snapshot(snapshot_path)
            self._loaded = loaded
            return loaded


def _model_identity(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return str(candidate.resolve()).replace("\\", "/").rstrip("/").casefold()
    return value.strip().replace("\\", "/").rstrip("/").casefold()


def _normalized_query_vector(value: Any, dimension: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Query embedder returned a non-numeric vector") from exc
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise RuntimeError(
            f"Query vector has shape {vector.shape}; expected ({dimension},)"
        )
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("Query vector contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("Query vector has zero or non-finite norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


class ModalitySearcher:
    @staticmethod
    def _matches_filters(metadata: Mapping[str, Any], filters: Any) -> bool:
        if filters is None:
            return True
        if filters.document_ids is not None:
            document_id = metadata.get("document_id")
            if document_id not in filters.document_ids:
                return False
        if filters.document_types is not None:
            document_type = str(metadata.get("document_type") or "").lower()
            if document_type not in filters.document_types:
                return False
        if filters.source_paths is not None:
            source_path = metadata.get("source_path")
            if source_path not in filters.source_paths:
                return False
        return True

    def search(
        self,
        snapshot: LoadedSnapshot,
        query_vector: np.ndarray,
        modality: str,
        limit: int,
        filters: Any = None,
    ) -> tuple[SearchCandidate, ...]:
        if modality not in VALID_MODALITIES:
            raise ValueError(f"Unsupported retrieval modality: {modality}")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Search limit must be a positive integer")
        index = snapshot.indexes[modality]
        if index.ntotal == 0:
            return ()
        rows = snapshot.metadata[modality]
        search_count = min(limit, index.ntotal)
        while True:
            scores, positions = index.search(query_vector.reshape(1, -1), search_count)
            candidates: list[SearchCandidate] = []
            for position, score in zip(positions[0], scores[0]):
                index_position = int(position)
                raw_score = float(score)
                if index_position == -1:
                    continue
                if not 0 <= index_position < len(rows):
                    raise RuntimeError(
                        f"{modality} search returned invalid index position {index_position}"
                    )
                if not math.isfinite(raw_score):
                    raise RuntimeError(f"{modality} search returned a non-finite score")
                metadata = rows[index_position]
                if not self._matches_filters(metadata, filters):
                    continue
                candidates.append(
                    SearchCandidate(
                        modality=modality,
                        modality_rank=len(candidates) + 1,
                        index_position=index_position,
                        raw_score=raw_score,
                        metadata=metadata,
                    )
                )
                if len(candidates) == limit:
                    break
            if len(candidates) >= limit or search_count == index.ntotal:
                return tuple(candidates)
            search_count = min(index.ntotal, max(search_count + 1, search_count * 2))


class VectorRetriever:
    def __init__(
        self,
        snapshot_store: SnapshotStore,
        embedder: QueryEmbedder,
        searcher: ModalitySearcher | None = None,
    ) -> None:
        model_name = getattr(embedder, "model_name", None)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Query embedder has no model_name")
        self.snapshot_store = snapshot_store
        self.embedder = embedder
        self.searcher = searcher or ModalitySearcher()

    def search(self, request: RetrievalRequest) -> VectorSearchResult:
        if not isinstance(request, RetrievalRequest):
            raise TypeError("request must be a RetrievalRequest")
        snapshot = self.snapshot_store.get_snapshot()
        if _model_identity(self.embedder.model_name) != _model_identity(snapshot.model):
            raise RuntimeError(
                "Query embedding model does not match the index snapshot model"
            )
        raw_vector = self.embedder.embed_query(request.query_text)
        query_vector = _normalized_query_vector(raw_vector, snapshot.dimension)
        candidates = {
            modality: self.searcher.search(
                snapshot,
                query_vector,
                modality,
                request.candidate_k,
                request.filters,
            )
            for modality in request.modalities
        }
        return VectorSearchResult(
            snapshot=snapshot,
            candidates=MappingProxyType(candidates),
        )


class CandidateRanker:
    def __init__(
        self,
        modality_weights: Mapping[str, float] | None = None,
        rrf_k: float = RRF_K,
        min_raw_scores: Mapping[str, float] | None = None,
        rank_bonus_weight: float = 1.0,
    ) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, (int, float)):
            raise TypeError("rrf_k must be numeric")
        self.rrf_k = float(rrf_k)
        if not math.isfinite(self.rrf_k) or self.rrf_k < 0:
            raise ValueError("rrf_k must be a finite non-negative value")
        weights = dict(DEFAULT_MODALITY_WEIGHTS)
        for modality, value in (modality_weights or {}).items():
            if modality not in VALID_MODALITIES:
                raise ValueError(f"Unsupported modality weight: {modality}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Modality weights must be numeric")
            weight = float(value)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("Modality weights must be finite and positive")
            weights[modality] = weight
        self.modality_weights = MappingProxyType(weights)
        thresholds = dict(DEFAULT_MIN_RAW_SCORES)
        for modality, value in (min_raw_scores or {}).items():
            if modality not in VALID_MODALITIES:
                raise ValueError(f"Unsupported modality threshold: {modality}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Modality thresholds must be numeric")
            threshold = float(value)
            if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
                raise ValueError("Modality thresholds must be finite values between -1 and 1")
            thresholds[modality] = threshold
        self.min_raw_scores = MappingProxyType(thresholds)
        if isinstance(rank_bonus_weight, bool) or not isinstance(
            rank_bonus_weight, (int, float)
        ):
            raise TypeError("rank_bonus_weight must be numeric")
        self.rank_bonus_weight = float(rank_bonus_weight)
        if not math.isfinite(self.rank_bonus_weight) or self.rank_bonus_weight < 0:
            raise ValueError("rank_bonus_weight must be finite and non-negative")

    @staticmethod
    def _metadata_text(metadata: Mapping[str, Any], field: str) -> str:
        value = metadata.get(field)
        return value.strip() if isinstance(value, str) else ""

    @classmethod
    def _is_duplicate(
        cls,
        candidate: SearchCandidate,
        accepted: list[RankedCandidate],
    ) -> bool:
        metadata = candidate.metadata
        record_id = cls._metadata_text(metadata, "record_id")
        document_id = cls._metadata_text(metadata, "document_id")
        content_hash = cls._metadata_text(metadata, "content_hash")
        source_ids = {
            str(value)
            for value in metadata.get("source_block_ids", [])
            if str(value)
        }
        for ranked in accepted:
            other = ranked.candidate
            other_metadata = other.metadata
            if record_id and record_id == cls._metadata_text(other_metadata, "record_id"):
                return True
            same_scope = (
                candidate.modality == other.modality
                and document_id == cls._metadata_text(other_metadata, "document_id")
            )
            if not same_scope:
                continue
            other_hash = cls._metadata_text(other_metadata, "content_hash")
            if content_hash and content_hash == other_hash:
                return True
            if candidate.modality != "text" or not source_ids:
                continue
            other_source_ids = {
                str(value)
                for value in other_metadata.get("source_block_ids", [])
                if str(value)
            }
            if not other_source_ids:
                continue
            overlap = len(source_ids.intersection(other_source_ids))
            if overlap / min(len(source_ids), len(other_source_ids)) >= 0.8:
                return True
        return False

    def rank(self, result: VectorSearchResult, top_k: int) -> RankedSearchResult:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        ranked = [
            RankedCandidate(
                candidate=candidate,
                fusion_score=(
                    self.modality_weights[modality] * candidate.raw_score
                    + self.rank_bonus_weight
                    / (self.rrf_k + candidate.modality_rank)
                ),
            )
            for modality, candidates in result.candidates.items()
            for candidate in candidates
            if candidate.raw_score >= self.min_raw_scores[modality]
        ]
        modality_order = {value: index for index, value in enumerate(VALID_MODALITIES)}
        ranked.sort(
            key=lambda item: (
                -item.fusion_score,
                -item.candidate.raw_score,
                modality_order[item.candidate.modality],
                self._metadata_text(item.candidate.metadata, "record_id"),
            )
        )
        accepted: list[RankedCandidate] = []
        for item in ranked:
            if self._is_duplicate(item.candidate, accepted):
                continue
            accepted.append(item)
            if len(accepted) == top_k:
                break
        return RankedSearchResult(snapshot=result.snapshot, candidates=tuple(accepted))


class HybridSearchEngine:
    """Public, read-only retrieval entry point for schema 2.0 snapshots."""

    def __init__(
        self,
        index_dir: Path,
        embedder: QueryEmbedder | None = None,
        *,
        device: str = "auto",
        ranker: CandidateRanker | None = None,
    ) -> None:
        self.snapshot_store = SnapshotStore(index_dir)
        self.ranker = ranker or CandidateRanker()
        self._owned_session: Any = None
        if embedder is None:
            from .vision import PixelRAGEmbeddingSession

            snapshot = self.snapshot_store.get_snapshot()
            self._owned_session = PixelRAGEmbeddingSession(snapshot.model, device)
            embedder = PixelRAGQueryEmbedder(self._owned_session)
        self.vector_retriever = VectorRetriever(self.snapshot_store, embedder)
        self._lock = threading.RLock()
        self._closed = False

    @staticmethod
    def _visual_asset_path(snapshot: LoadedSnapshot, relative_value: Any) -> str:
        if not isinstance(relative_value, str) or not relative_value.strip():
            raise RuntimeError("Visual retrieval metadata contains no asset_path")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Visual retrieval metadata contains an unsafe asset_path")
        snapshot_root = snapshot.snapshot_path.resolve()
        asset = (snapshot_root / relative).resolve()
        try:
            asset.relative_to(snapshot_root)
        except ValueError as exc:
            raise RuntimeError("Visual retrieval asset escapes the snapshot directory") from exc
        if not asset.is_file():
            raise RuntimeError("Visual retrieval asset is missing")
        return str(asset)

    @staticmethod
    def _warnings(
        snapshot: LoadedSnapshot,
        searched_modalities: tuple[str, ...],
    ) -> list[str]:
        values: list[str] = []
        manifest_warnings = snapshot.manifest.get("warnings", [])
        if isinstance(manifest_warnings, list):
            values.extend(
                value.strip()
                for value in manifest_warnings
                if isinstance(value, str) and value.strip()
            )
        values.extend(
            f"{modality} index is empty"
            for modality in searched_modalities
            if snapshot.indexes[modality].ntotal == 0
        )
        return list(dict.fromkeys(values))

    @classmethod
    def _hit(
        cls,
        ranked: RankedCandidate,
        rank: int,
        snapshot: LoadedSnapshot,
    ) -> RetrievalHit:
        candidate = ranked.candidate
        metadata = candidate.metadata
        asset_path = None
        if candidate.modality == "visual":
            asset_path = cls._visual_asset_path(snapshot, metadata.get("asset_path"))
        return RetrievalHit(
            rank=rank,
            record_id=metadata.get("record_id", ""),
            document_id=metadata.get("document_id", ""),
            modality=candidate.modality,
            raw_score=candidate.raw_score,
            fusion_score=ranked.fusion_score,
            source_block_ids=list(metadata.get("source_block_ids") or []),
            content=metadata.get("original_content"),
            context=metadata.get("context") or "",
            provenance=dict(metadata.get("provenance") or {}),
            source_path=metadata.get("source_path", ""),
            document_type=metadata.get("document_type", ""),
            asset_path=asset_path,
        )

    def search(self, request: RetrievalRequest) -> RetrievalResponse:
        if not isinstance(request, RetrievalRequest):
            raise TypeError("request must be a RetrievalRequest")
        with self._lock:
            if self._closed:
                raise RuntimeError("HybridSearchEngine is closed")
            started = time.perf_counter()
            vector_result = self.vector_retriever.search(request)
            ranked_result = self.ranker.rank(vector_result, request.top_k)
            hits = [
                self._hit(item, rank, ranked_result.snapshot)
                for rank, item in enumerate(ranked_result.candidates, 1)
            ]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return RetrievalResponse(
                query_text=request.query_text,
                snapshot_build_id=ranked_result.snapshot.build_id,
                hits=hits,
                searched_modalities=list(request.modalities),
                elapsed_ms=elapsed_ms,
                warnings=self._warnings(ranked_result.snapshot, request.modalities),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owned_session is not None:
                self._owned_session.close()

    def __enter__(self) -> "HybridSearchEngine":
        with self._lock:
            if self._closed:
                raise RuntimeError("HybridSearchEngine is closed")
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

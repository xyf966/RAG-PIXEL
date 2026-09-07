from __future__ import annotations

import json
import math
import re
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
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
VECTOR_RRF_WEIGHT = 1.0
KEYWORD_RRF_WEIGHT = 0.35
EXACT_RRF_WEIGHT = 0.75
DEFAULT_MODALITY_WEIGHTS = MappingProxyType(
    {"text": 1.0, "table": 1.0, "visual": 1.10}
)
DEFAULT_MIN_RAW_SCORES = MappingProxyType(
    {"text": 0.40, "table": 0.40, "visual": 0.40}
)


class QueryEmbedder(Protocol):
    model_name: str

    def embed_query(self, query_text: str) -> Any: ...


class JsonQueryClient(Protocol):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        image_paths: tuple[str, ...] = (),
    ) -> dict[str, Any]: ...


class QueryExpander(Protocol):
    def expand(self, query_text: str) -> tuple[str, ...]: ...


class LLMEnglishQueryExpander:
    """Add a faithful English search variant for a CJK query."""

    RESPONSE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {"translated_query": {"type": "string"}},
        "required": ["translated_query"],
    }

    def __init__(self, client: JsonQueryClient, *, max_query_chars: int = 1000) -> None:
        if isinstance(max_query_chars, bool) or not isinstance(max_query_chars, int):
            raise TypeError("max_query_chars must be an integer")
        if max_query_chars < 1:
            raise ValueError("max_query_chars must be positive")
        self.client = client
        self.max_query_chars = max_query_chars

    def expand(self, query_text: str) -> tuple[str, ...]:
        original = str(query_text).strip()
        if not original or not re.search(r"[\u3400-\u9fff]", original):
            return (original,)
        response = self.client.complete_json(
            system_prompt=(
                "你是跨语言检索查询翻译器。只翻译查询，不回答问题。"
                "将中文查询忠实翻译成自然、简洁的英文检索查询；保留型号、编号、单位、"
                "产品名和专有名词，不增加原问题没有的条件。"
            ),
            user_prompt=f"待翻译查询：{original}",
            schema=self.RESPONSE_SCHEMA,
        )
        translated = str(response.get("translated_query") or "").strip()
        if not translated:
            raise RuntimeError("Query translator returned an empty translation")
        translated = translated[: self.max_query_chars].strip()
        if _normalized_lexical_text(translated) == _normalized_lexical_text(original):
            return (original,)
        return (original, translated)


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
    keyword_rank: int | None = None
    keyword_score: float = 0.0
    exact_match: bool = False


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


class TextCandidateReranker(Protocol):
    def rerank(
        self,
        result: RankedSearchResult,
        query_text: str,
        top_k: int,
    ) -> RankedSearchResult: ...


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


def _normalized_lexical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _lexical_tokens(value: str) -> tuple[str, ...]:
    """Tokenize Latin/numeric terms and overlapping CJK unigrams/bigrams."""
    normalized = _normalized_lexical_text(value)
    tokens: list[str] = []
    for segment in re.findall(r"[a-z0-9]+(?:[-_.:/][a-z0-9]+)*|[\u3400-\u9fff]+", normalized):
        if re.fullmatch(r"[\u3400-\u9fff]+", segment):
            tokens.append(segment)
            if len(segment) > 1:
                tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tuple(tokens)


def _lexical_relevance_score(query_text: str, searchable_text: str) -> float:
    """Score exact identifiers, token overlap and small Latin spelling errors."""
    normalized_query = _normalized_lexical_text(query_text)
    normalized_searchable = _normalized_lexical_text(searchable_text)
    if not normalized_query or not normalized_searchable:
        return 0.0
    score = 5.0 if normalized_query in normalized_searchable else 0.0
    searchable_tokens = set(_lexical_tokens(searchable_text))
    latin_searchable = [
        token for token in searchable_tokens
        if len(token) >= 4 and token.isascii()
    ]
    for token in dict.fromkeys(_lexical_tokens(query_text)):
        if token in searchable_tokens:
            score += 3.0 if re.search(r"[a-z]", token) and re.search(r"\d", token) else 1.0
            continue
        if len(token) < 4 or not token.isascii():
            continue
        if any(SequenceMatcher(None, token, candidate).ratio() >= 0.8 for candidate in latin_searchable):
            score += 0.6
    return score


_QUERY_FOCUS_FRAGMENTS = tuple(
    sorted(
        {
            "应该怎么",
            "怎么解决",
            "如何解决",
            "怎么处理",
            "如何处理",
            "怎么操作",
            "如何操作",
            "有哪些",
            "有什么",
            "是什么",
            "怎么办",
            "请问",
            "问题",
            "方法",
            "办法",
        },
        key=len,
        reverse=True,
    )
)


def _query_focus_text(query_text: str) -> str:
    focus = _normalized_lexical_text(query_text)
    for fragment in _QUERY_FOCUS_FRAGMENTS:
        focus = focus.replace(fragment, " ")
    return " ".join(focus.split())


@dataclass(frozen=True, slots=True)
class KeywordMatch:
    index_position: int
    rank: int
    score: float
    exact_match: bool


class KeywordSearcher:
    """Small-snapshot BM25 search with an explicit full-query match channel."""

    BM25_K1 = 1.5
    BM25_B = 0.75

    @staticmethod
    def _searchable_text(metadata: Mapping[str, Any]) -> str:
        values = [metadata.get("embedding_text"), metadata.get("context")]
        original = metadata.get("original_content")
        if isinstance(original, str):
            values.append(original)
        elif isinstance(original, Mapping):
            values.extend(value for value in original.values() if isinstance(value, str))
        return "\n".join(value for value in values if isinstance(value, str))

    def search(
        self,
        rows: tuple[Mapping[str, Any], ...],
        query_text: str,
        limit: int,
        eligible: Any,
    ) -> tuple[KeywordMatch, ...]:
        normalized_query = _normalized_lexical_text(query_text)
        query_tokens = _lexical_tokens(query_text)
        if not normalized_query or not query_tokens:
            return ()

        documents: list[tuple[int, str, tuple[str, ...]]] = []
        for index_position, metadata in enumerate(rows):
            if not eligible(metadata):
                continue
            searchable = self._searchable_text(metadata)
            documents.append(
                (index_position, _normalized_lexical_text(searchable), _lexical_tokens(searchable))
            )
        if not documents:
            return ()

        document_frequency = Counter(
            token for _position, _text, tokens in documents for token in set(tokens)
        )
        average_length = sum(len(tokens) for _position, _text, tokens in documents) / len(documents)
        query_counts = Counter(query_tokens)
        scored: list[tuple[int, float, bool]] = []
        for index_position, normalized_text, tokens in documents:
            term_counts = Counter(tokens)
            length_normalizer = 1.0 - self.BM25_B
            if average_length:
                length_normalizer += self.BM25_B * len(tokens) / average_length
            score = 0.0
            for token, query_frequency in query_counts.items():
                frequency = term_counts.get(token, 0)
                if not frequency:
                    continue
                frequency_weight = (
                    frequency * (self.BM25_K1 + 1.0)
                    / (frequency + self.BM25_K1 * length_normalizer)
                )
                document_count = len(documents)
                frequency_in_documents = document_frequency[token]
                inverse_frequency = math.log(
                    1.0
                    + (document_count - frequency_in_documents + 0.5)
                    / (frequency_in_documents + 0.5)
                )
                score += inverse_frequency * frequency_weight * query_frequency
            exact_match = normalized_query in normalized_text
            if exact_match or score > 0.0:
                scored.append((index_position, score, exact_match))

        scored.sort(key=lambda item: (-int(item[2]), -item[1], item[0]))
        return tuple(
            KeywordMatch(position, rank, score, exact_match)
            for rank, (position, score, exact_match) in enumerate(scored[:limit], 1)
        )


class ModalitySearcher:
    def __init__(self, keyword_searcher: KeywordSearcher | None = None) -> None:
        self.keyword_searcher = keyword_searcher or KeywordSearcher()

    @staticmethod
    def _is_final_evidence_eligible(modality: str, metadata: Mapping[str, Any]) -> bool:
        if modality != "visual":
            return True
        structure = metadata.get("structure")
        structure_metadata = structure.get("metadata") if isinstance(structure, Mapping) else None
        if isinstance(structure_metadata, Mapping):
            if structure_metadata.get("llm_eligible") is False:
                return False
            if structure_metadata.get("coarse_only") is True:
                return False
        # Backward-compatible safety gate for snapshots built before the
        # page_visual metadata existed.
        if str(metadata.get("document_type") or "").lower() != "pdf":
            return True
        provenance = metadata.get("provenance")
        bbox = provenance.get("bbox") if isinstance(provenance, Mapping) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return True
        left, top, right, bottom = (float(value) for value in bbox)
        width = max(0.0, min(right, 1.0) - max(left, 0.0))
        height = max(0.0, min(bottom, 1.0) - max(top, 0.0))
        return not (width >= 0.92 and height >= 0.92 and width * height >= 0.88)

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
        query_text: str,
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
        eligible = lambda metadata: (
            self._matches_filters(metadata, filters)
            and self._is_final_evidence_eligible(modality, metadata)
        )
        keyword_matches = self.keyword_searcher.search(rows, query_text, limit, eligible)
        keyword_by_position = {match.index_position: match for match in keyword_matches}
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
                if not eligible(metadata):
                    continue
                keyword = keyword_by_position.get(index_position)
                candidates.append(
                    SearchCandidate(
                        modality=modality,
                        modality_rank=len(candidates) + 1,
                        index_position=index_position,
                        raw_score=raw_score,
                        metadata=metadata,
                        keyword_rank=keyword.rank if keyword else None,
                        keyword_score=keyword.score if keyword else 0.0,
                        exact_match=keyword.exact_match if keyword else False,
                    )
                )
                if len(candidates) == limit:
                    break
            if len(candidates) >= limit or search_count == index.ntotal:
                break
            search_count = min(index.ntotal, max(search_count + 1, search_count * 2))

        combined = {candidate.index_position: candidate for candidate in candidates}
        for keyword in keyword_matches:
            existing = combined.get(keyword.index_position)
            if existing is not None:
                combined[keyword.index_position] = replace(
                    existing,
                    keyword_rank=keyword.rank,
                    keyword_score=keyword.score,
                    exact_match=keyword.exact_match,
                )
                continue
            vector = np.asarray(index.reconstruct(keyword.index_position), dtype=np.float32)
            raw_score = float(np.dot(query_vector, vector))
            if not math.isfinite(raw_score):
                raise RuntimeError(f"{modality} reconstructed a non-finite score")
            combined[keyword.index_position] = SearchCandidate(
                modality=modality,
                modality_rank=0,
                index_position=keyword.index_position,
                raw_score=raw_score,
                metadata=rows[keyword.index_position],
                keyword_rank=keyword.rank,
                keyword_score=keyword.score,
                exact_match=keyword.exact_match,
            )
        return tuple(
            sorted(
                combined.values(),
                key=lambda candidate: (
                    candidate.modality_rank == 0,
                    candidate.modality_rank or math.inf,
                    candidate.keyword_rank or math.inf,
                    candidate.index_position,
                ),
            )
        )


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

    @staticmethod
    def _merge_query_candidates(
        candidate_groups: list[tuple[SearchCandidate, ...]],
    ) -> tuple[SearchCandidate, ...]:
        merged: dict[int, SearchCandidate] = {}
        for candidates in candidate_groups:
            for candidate in candidates:
                existing = merged.get(candidate.index_position)
                if existing is None:
                    merged[candidate.index_position] = candidate
                    continue
                positive_ranks = [
                    rank
                    for rank in (existing.modality_rank, candidate.modality_rank)
                    if rank > 0
                ]
                keyword_ranks = [
                    rank
                    for rank in (existing.keyword_rank, candidate.keyword_rank)
                    if rank is not None
                ]
                merged[candidate.index_position] = replace(
                    existing,
                    modality_rank=min(positive_ranks) if positive_ranks else 0,
                    raw_score=max(existing.raw_score, candidate.raw_score),
                    keyword_rank=min(keyword_ranks) if keyword_ranks else None,
                    keyword_score=max(existing.keyword_score, candidate.keyword_score),
                    exact_match=existing.exact_match or candidate.exact_match,
                )
        return tuple(
            sorted(
                merged.values(),
                key=lambda candidate: (
                    candidate.modality_rank == 0,
                    candidate.modality_rank or math.inf,
                    -candidate.raw_score,
                    candidate.keyword_rank or math.inf,
                    candidate.index_position,
                ),
            )
        )

    def search(
        self,
        request: RetrievalRequest,
        query_variants: tuple[str, ...] | None = None,
    ) -> VectorSearchResult:
        if not isinstance(request, RetrievalRequest):
            raise TypeError("request must be a RetrievalRequest")
        snapshot = self.snapshot_store.get_snapshot()
        if _model_identity(self.embedder.model_name) != _model_identity(snapshot.model):
            raise RuntimeError(
                "Query embedding model does not match the index snapshot model"
            )
        variants = tuple(
            dict.fromkeys(
                value.strip()
                for value in (query_variants or (request.query_text,))
                if isinstance(value, str) and value.strip()
            )
        )
        if not variants:
            variants = (request.query_text,)
        searches: dict[str, list[tuple[SearchCandidate, ...]]] = {
            modality: [] for modality in request.modalities
        }
        for query_text in variants:
            raw_vector = self.embedder.embed_query(query_text)
            query_vector = _normalized_query_vector(raw_vector, snapshot.dimension)
            for modality in request.modalities:
                searches[modality].append(
                    self.searcher.search(
                        snapshot,
                        query_vector,
                        query_text,
                        modality,
                        request.candidate_k,
                        request.filters,
                    )
                )
        candidates = {
            modality: self._merge_query_candidates(groups)
            for modality, groups in searches.items()
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
        vector_rrf_weight: float = VECTOR_RRF_WEIGHT,
        keyword_rrf_weight: float = KEYWORD_RRF_WEIGHT,
        exact_rrf_weight: float = EXACT_RRF_WEIGHT,
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
        fusion_weights = {
            "vector_rrf_weight": vector_rrf_weight,
            "keyword_rrf_weight": keyword_rrf_weight,
            "exact_rrf_weight": exact_rrf_weight,
        }
        for name, value in fusion_weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, value)

    def _fusion_score(self, candidate: SearchCandidate) -> float:
        score = 0.0
        if candidate.modality_rank > 0:
            score += (
                self.vector_rrf_weight
                * self.modality_weights[candidate.modality]
                / (self.rrf_k + candidate.modality_rank)
            )
        if candidate.keyword_rank is not None:
            score += self.keyword_rrf_weight / (self.rrf_k + candidate.keyword_rank)
        if candidate.exact_match:
            score += self.exact_rrf_weight / (self.rrf_k + 1.0)
        return score

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
                fusion_score=self._fusion_score(candidate),
            )
            for modality, candidates in result.candidates.items()
            for candidate in candidates
            if (
                candidate.raw_score >= self.min_raw_scores[modality]
                or candidate.keyword_rank is not None
            )
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


class QueryAwareTextReranker:
    """Boost query-aligned text without changing table or visual scoring."""

    def __init__(self, weight: float = 0.35, rrf_k: float = RRF_K) -> None:
        for name, value in (("weight", weight), ("rrf_k", rrf_k)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, value)

    @staticmethod
    def _score(query_text: str, candidate: SearchCandidate) -> float:
        if candidate.modality != "text":
            return 0.0
        searchable = KeywordSearcher._searchable_text(candidate.metadata)
        normalized_query = _normalized_lexical_text(query_text)
        normalized_searchable = _normalized_lexical_text(searchable)
        if not normalized_query or not normalized_searchable:
            return 0.0

        focus = _query_focus_text(query_text)
        query_tokens = set(_lexical_tokens(query_text))
        searchable_tokens = set(_lexical_tokens(searchable))
        coverage = (
            len(query_tokens.intersection(searchable_tokens)) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        score = 0.45 * coverage
        if normalized_query in normalized_searchable:
            score += 0.35
        if focus and focus in normalized_searchable:
            score += 0.35
        if candidate.exact_match:
            score += 0.20
        if candidate.keyword_rank is not None:
            score += 0.10 / max(candidate.keyword_rank, 1)
        return min(score, 1.0)

    def rerank(
        self,
        result: RankedSearchResult,
        query_text: str,
        top_k: int,
    ) -> RankedSearchResult:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        rescored: list[tuple[RankedCandidate, int]] = []
        denominator = self.rrf_k + 1.0
        for original_position, item in enumerate(result.candidates):
            rerank_score = self._score(query_text, item.candidate)
            boosted = item
            if rerank_score > 0.0 and self.weight > 0.0:
                boosted = RankedCandidate(
                    candidate=item.candidate,
                    fusion_score=(
                        item.fusion_score + self.weight * rerank_score / denominator
                    ),
                )
            rescored.append((boosted, original_position))
        rescored.sort(
            key=lambda value: (
                -value[0].fusion_score,
                value[1],
            )
        )
        selected = [item for item, _position in rescored[:top_k]]
        vector_text_anchors = sorted(
            (
                item
                for item in result.candidates
                if item.candidate.modality == "text" and item.candidate.modality_rank > 0
            ),
            key=lambda item: (
                -item.candidate.raw_score,
                item.candidate.modality_rank,
            ),
        )[:2]
        selected_ids = {
            (item.candidate.modality, item.candidate.index_position) for item in selected
        }
        missing_anchors = [
            item
            for item in vector_text_anchors
            if (item.candidate.modality, item.candidate.index_position) not in selected_ids
        ]
        if missing_anchors:
            replace_count = min(len(missing_anchors), len(selected))
            selected = missing_anchors[:replace_count] + selected[: top_k - replace_count]
        return RankedSearchResult(
            snapshot=result.snapshot,
            candidates=tuple(selected),
        )


class EvidenceExpander:
    """Attach a bounded text window without judging neighbor relevance."""

    def __init__(self, previous_hops: int = 1, next_hops: int = 2) -> None:
        for name, value in (("previous_hops", previous_hops), ("next_hops", next_hops)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value <= 2:
                raise ValueError(f"{name} must be between 0 and 2")
        self.previous_hops = previous_hops
        self.next_hops = next_hops

    @staticmethod
    def _page(metadata: Mapping[str, Any]) -> int | None:
        provenance = metadata.get("provenance")
        page = provenance.get("page") if isinstance(provenance, Mapping) else None
        return page if isinstance(page, int) and not isinstance(page, bool) and page > 0 else None

    @staticmethod
    def _text(metadata: Mapping[str, Any]) -> str:
        original = metadata.get("original_content")
        if isinstance(original, str):
            return original.strip()
        if isinstance(original, Mapping) and isinstance(original.get("text"), str):
            return original["text"].strip()
        value = metadata.get("embedding_text")
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _same_document(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return bool(left.get("document_id")) and left.get("document_id") == right.get("document_id")

    @classmethod
    def _adjacent_payload(cls, metadata: Mapping[str, Any], relation: str) -> dict[str, Any]:
        return {
            "record_id": metadata.get("record_id", ""),
            "relation": relation,
            "content": metadata.get("original_content"),
            "context": metadata.get("context") or "",
            "provenance": dict(metadata.get("provenance") or {}),
            "source_block_ids": list(metadata.get("source_block_ids") or []),
            "structure": dict(metadata.get("structure") or {}),
        }

    def _text_anchor(
        self,
        snapshot: LoadedSnapshot,
        candidate: SearchCandidate,
        query_text: str,
    ) -> int | None:
        if candidate.modality == "text":
            return candidate.index_position
        document_id = candidate.metadata.get("document_id")
        page = self._page(candidate.metadata)
        if not document_id or page is None:
            return None
        rows = snapshot.metadata["text"]
        query = _normalized_lexical_text(query_text)
        matches = [
            index
            for index, row in enumerate(rows)
            if row.get("document_id") == document_id and self._page(row) == page
        ]
        if not matches:
            return None
        return min(
            matches,
            key=lambda index: (
                query not in _normalized_lexical_text(self._text(rows[index])),
                index,
            ),
        )

    def expand(
        self,
        snapshot: LoadedSnapshot,
        candidate: SearchCandidate,
        query_text: str,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        base_page = self._page(candidate.metadata)
        pages = [base_page] if base_page is not None else []
        if self.previous_hops == 0 and self.next_hops == 0:
            return [], pages
        rows = snapshot.metadata["text"]
        anchor = self._text_anchor(snapshot, candidate, query_text)
        if anchor is None or not 0 <= anchor < len(rows):
            return [], pages

        anchor_row = rows[anchor]
        rows_by_id = {
            str(row.get("record_id")): (index, row)
            for index, row in enumerate(rows)
            if isinstance(row.get("record_id"), str) and row.get("record_id")
        }
        explicit_links = (
            "previous_record_id" in anchor_row or "next_record_id" in anchor_row
        )
        selected: list[tuple[int, str]] = []

        def follow(field: str, relation: str, hops: int) -> list[tuple[int, str]]:
            values: list[tuple[int, str]] = []
            current = anchor_row
            visited = {str(anchor_row.get("record_id") or "")}
            for _ in range(hops):
                neighbor_id = current.get(field)
                if not isinstance(neighbor_id, str) or not neighbor_id:
                    break
                neighbor_entry = rows_by_id.get(neighbor_id)
                if (
                    neighbor_entry is None
                    or neighbor_id in visited
                ):
                    break
                index, neighbor = neighbor_entry
                if not self._same_document(anchor_row, neighbor):
                    break
                values.append((index, relation))
                visited.add(neighbor_id)
                current = neighbor
            return values

        if explicit_links:
            previous = follow("previous_record_id", "previous_context", self.previous_hops)
            previous.reverse()
            selected.extend(previous)
            if candidate.modality != "text":
                selected.append((anchor, "same_page_text"))
            selected.extend(follow("next_record_id", "next_context", self.next_hops))
        else:
            # Schema 2.0 snapshots built before explicit neighbor links remain readable.
            offsets = (
                (-1, 1, 2)
                if candidate.modality == "text"
                else (-1, 0, 1, 2)
            )
            for offset in offsets:
                index = anchor + offset
                if not 0 <= index < len(rows) or not self._same_document(anchor_row, rows[index]):
                    continue
                relation = (
                    "same_page_text"
                    if offset == 0
                    else "previous_context"
                    if offset < 0
                    else "next_context"
                )
                selected.append((index, relation))

        payloads = [self._adjacent_payload(rows[index], relation) for index, relation in selected]
        pages.extend(
            page
            for index, _relation in selected
            if (page := self._page(rows[index])) is not None
        )
        return payloads, list(dict.fromkeys(sorted(pages)))


class EvidenceAssembler:
    """Reassemble retrieved modalities into an ordered, display-only evidence view."""

    def __init__(
        self,
        max_related_records: int = 8,
        max_related_per_container: int = 2,
        max_related_per_modality: int = 1,
    ) -> None:
        if isinstance(max_related_records, bool) or not isinstance(max_related_records, int):
            raise TypeError("max_related_records must be an integer")
        if not 0 <= max_related_records <= 50:
            raise ValueError("max_related_records must be between 0 and 50")
        if (
            isinstance(max_related_per_container, bool)
            or not isinstance(max_related_per_container, int)
        ):
            raise TypeError("max_related_per_container must be an integer")
        if not 0 <= max_related_per_container <= 10:
            raise ValueError("max_related_per_container must be between 0 and 10")
        if (
            isinstance(max_related_per_modality, bool)
            or not isinstance(max_related_per_modality, int)
        ):
            raise TypeError("max_related_per_modality must be an integer")
        if not 0 <= max_related_per_modality <= 10:
            raise ValueError("max_related_per_modality must be between 0 and 10")
        self.max_related_records = max_related_records
        self.max_related_per_container = max_related_per_container
        self.max_related_per_modality = max_related_per_modality

    @staticmethod
    def _container(metadata: Mapping[str, Any]) -> tuple[str, Any] | None:
        provenance = metadata.get("provenance")
        if not isinstance(provenance, Mapping):
            return None
        for field in ("page", "slide", "sheet"):
            value = provenance.get(field)
            if value is not None:
                return field, value
        return None

    @staticmethod
    def _bbox(metadata: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
        provenance = metadata.get("provenance")
        bbox = provenance.get("bbox") if isinstance(provenance, Mapping) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            structure = metadata.get("structure")
            sources = structure.get("source_provenance") if isinstance(structure, Mapping) else None
            if isinstance(sources, list) and sources:
                first = sources[0]
                bbox = first.get("bbox") if isinstance(first, Mapping) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            return tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _block(
        cls,
        metadata: Mapping[str, Any],
        modality: str,
        role: str,
        relation: str = "",
    ) -> dict[str, Any]:
        structure = metadata.get("structure")
        sources = structure.get("source_provenance") if isinstance(structure, Mapping) else None
        regions = (
            [dict(item) for item in sources if isinstance(item, Mapping)]
            if isinstance(sources, list) and sources
            else [dict(metadata.get("provenance") or {})]
        )
        content = metadata.get("original_content")
        embedding_text = metadata.get("embedding_text")
        if modality == "table" and isinstance(embedding_text, str) and embedding_text.strip():
            content = {
                "text": embedding_text.strip(),
                "structure": {
                    key: structure.get(key)
                    for key in (
                        "headers",
                        "rows",
                        "column_indices",
                        "row_start",
                        "row_end",
                        "segment_index",
                        "segment_count",
                    )
                    if isinstance(structure, Mapping) and structure.get(key) is not None
                },
            }
        return {
            "record_id": metadata.get("record_id", ""),
            "modality": modality,
            "role": role,
            "relation": relation,
            "content": content,
            "context": metadata.get("context") or "",
            "provenance": dict(metadata.get("provenance") or {}),
            "source_block_ids": list(metadata.get("source_block_ids") or []),
            "asset_path": metadata.get("asset_path") if modality == "visual" else None,
            "regions": regions,
        }

    @classmethod
    def _sort_key(cls, block: Mapping[str, Any]) -> tuple[Any, ...]:
        provenance = block.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        container_order = 3
        container_value: Any = ""
        for order, field in enumerate(("page", "slide", "sheet")):
            if provenance.get(field) is not None:
                container_order = order
                container_value = provenance[field]
                break
        bbox = cls._bbox({"provenance": provenance})
        top = bbox[1] if bbox else math.inf
        left = bbox[0] if bbox else math.inf
        modality_order = {"text": 0, "table": 1, "visual": 2}
        return (
            container_order,
            str(container_value) if isinstance(container_value, str) else container_value,
            top,
            left,
            modality_order.get(str(block.get("modality")), 9),
            str(block.get("record_id") or ""),
        )

    def assemble(
        self,
        snapshot: LoadedSnapshot,
        candidate: SearchCandidate,
        adjacent_context: list[dict[str, Any]],
        query_text: str,
    ) -> list[dict[str, Any]]:
        document_id = candidate.metadata.get("document_id")
        blocks = [self._block(candidate.metadata, candidate.modality, "core")]
        seen = {str(candidate.metadata.get("record_id") or "")}
        context_metadata = [candidate.metadata, *adjacent_context]
        containers = {
            container for metadata in context_metadata
            if (container := self._container(metadata)) is not None
        }
        reference_boxes: dict[tuple[str, Any], list[tuple[float, float, float, float]]] = {}
        for metadata in context_metadata:
            container = self._container(metadata)
            bbox = self._bbox(metadata)
            if container is not None and bbox is not None:
                reference_boxes.setdefault(container, []).append(bbox)
        for adjacent in adjacent_context:
            record_id = str(adjacent.get("record_id") or "")
            if record_id in seen:
                continue
            seen.add(record_id)
            metadata = {
                "record_id": record_id,
                "original_content": adjacent.get("content"),
                "context": adjacent.get("context"),
                "provenance": adjacent.get("provenance"),
                "source_block_ids": adjacent.get("source_block_ids"),
                "structure": adjacent.get("structure"),
            }
            blocks.append(
                self._block(metadata, "text", "neighbor", str(adjacent.get("relation") or ""))
            )

        related_by_container: dict[tuple[str, Any], list[tuple[str, Mapping[str, Any]]]] = {}
        for modality in ("table", "visual"):
            for metadata in snapshot.metadata[modality]:
                record_id = str(metadata.get("record_id") or "")
                container = self._container(metadata)
                if (
                    record_id in seen
                    or metadata.get("document_id") != document_id
                    or container not in containers
                    or not ModalitySearcher._is_final_evidence_eligible(modality, metadata)
                ):
                    continue
                assert container is not None
                related_by_container.setdefault(container, []).append((modality, metadata))

        related_count = 0
        for container in sorted(related_by_container, key=lambda item: (item[0], str(item[1]))):
            references = reference_boxes.get(container, [])

            def related_key(item: tuple[str, Mapping[str, Any]]) -> tuple[Any, ...]:
                modality, metadata = item
                searchable = _normalized_lexical_text(
                    KeywordSearcher._searchable_text(metadata)
                )
                relevance = _lexical_relevance_score(query_text, searchable)
                bbox = self._bbox(metadata)
                distance = math.inf
                if bbox is not None and references:
                    center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                    distance = min(
                        abs(center[0] - (ref[0] + ref[2]) / 2.0)
                        + abs(center[1] - (ref[1] + ref[3]) / 2.0)
                        for ref in references
                    )
                return (
                    -relevance,
                    distance,
                    0 if modality == "table" else 1,
                    str(metadata.get("record_id") or ""),
                )

            selected_in_container = 0
            modality_counts: dict[str, int] = {}
            for modality, metadata in sorted(
                related_by_container[container],
                key=related_key,
            ):
                if (
                    related_count >= self.max_related_records
                    or selected_in_container >= self.max_related_per_container
                ):
                    break
                if modality_counts.get(modality, 0) >= self.max_related_per_modality:
                    continue
                record_id = str(metadata.get("record_id") or "")
                seen.add(record_id)
                relation = (
                    "text_anchor_same_container"
                    if candidate.modality == "text"
                    else "same_container"
                )
                blocks.append(self._block(metadata, modality, "related", relation))
                related_count += 1
                selected_in_container += 1
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        blocks.sort(key=self._sort_key)
        return blocks


class HybridSearchEngine:
    """Public, read-only retrieval entry point for schema 2.0 snapshots."""

    def __init__(
        self,
        index_dir: Path,
        embedder: QueryEmbedder | None = None,
        *,
        device: str = "auto",
        ranker: CandidateRanker | None = None,
        text_reranker: TextCandidateReranker | None = None,
        evidence_expander: EvidenceExpander | None = None,
        evidence_assembler: EvidenceAssembler | None = None,
    ) -> None:
        self.snapshot_store = SnapshotStore(index_dir)
        self.ranker = ranker or CandidateRanker()
        self.text_reranker = text_reranker or QueryAwareTextReranker()
        self.evidence_expander = evidence_expander or EvidenceExpander()
        self.evidence_assembler = evidence_assembler or EvidenceAssembler()
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
        adjacent_context: list[dict[str, Any]] | None = None,
        context_pages: list[int] | None = None,
        evidence_blocks: list[dict[str, Any]] | None = None,
    ) -> RetrievalHit:
        candidate = ranked.candidate
        metadata = candidate.metadata
        asset_path = None
        if candidate.modality == "visual":
            asset_path = cls._visual_asset_path(snapshot, metadata.get("asset_path"))
        resolved_blocks = []
        for block in evidence_blocks or []:
            resolved = dict(block)
            if resolved.get("modality") == "visual" and resolved.get("asset_path"):
                resolved["asset_path"] = cls._visual_asset_path(
                    snapshot,
                    resolved["asset_path"],
                )
            resolved_blocks.append(resolved)
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
            adjacent_context=adjacent_context or [],
            context_pages=context_pages or [],
            evidence_blocks=resolved_blocks,
        )

    def search(
        self,
        request: RetrievalRequest,
        *,
        query_expander: QueryExpander | None = None,
    ) -> RetrievalResponse:
        if not isinstance(request, RetrievalRequest):
            raise TypeError("request must be a RetrievalRequest")
        with self._lock:
            if self._closed:
                raise RuntimeError("HybridSearchEngine is closed")
            started = time.perf_counter()
            query_variants = (request.query_text,)
            query_warnings: list[str] = []
            if query_expander is not None:
                try:
                    query_variants = query_expander.expand(request.query_text)
                except Exception as exc:
                    query_warnings.append(
                        f"Bilingual query expansion unavailable; used original query: {exc}"
                    )
            vector_result = self.vector_retriever.search(request, query_variants)
            coarse_k = max(
                request.top_k,
                min(request.candidate_k, request.top_k * 3),
            )
            coarse_result = self.ranker.rank(vector_result, coarse_k)
            ranked_result = self.text_reranker.rerank(
                coarse_result,
                request.query_text,
                request.top_k,
            )
            hits = []
            for rank, item in enumerate(ranked_result.candidates, 1):
                adjacent_context, context_pages = self.evidence_expander.expand(
                    ranked_result.snapshot,
                    item.candidate,
                    request.query_text,
                )
                evidence_blocks = self.evidence_assembler.assemble(
                    ranked_result.snapshot,
                    item.candidate,
                    adjacent_context,
                    request.query_text,
                )
                hits.append(
                    self._hit(
                        item,
                        rank,
                        ranked_result.snapshot,
                        adjacent_context,
                        context_pages,
                        evidence_blocks,
                    )
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return RetrievalResponse(
                query_text=request.query_text,
                snapshot_build_id=ranked_result.snapshot.build_id,
                hits=hits,
                searched_modalities=list(request.modalities),
                elapsed_ms=elapsed_ms,
                warnings=(
                    self._warnings(ranked_result.snapshot, request.modalities)
                    + query_warnings
                ),
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

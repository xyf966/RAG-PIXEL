from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


VALID_MODALITIES = ("text", "table", "visual")
MAX_TOP_K = 100
MAX_CANDIDATE_K = 1000


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_values(
    values: Iterable[str] | None,
    field_name: str,
    *,
    lowercase: bool = False,
) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized: set[str] = set()
    for value in values:
        item = _required_text(value, f"{field_name} item")
        normalized.add(item.lower() if lowercase else item)
    return frozenset(normalized) or None


def _modalities(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("modalities must be a collection of strings")
    requested = {
        _required_text(value, "modality").lower()
        for value in values
    }
    unknown = requested.difference(VALID_MODALITIES)
    if unknown:
        raise ValueError(f"Unsupported retrieval modalities: {', '.join(sorted(unknown))}")
    normalized = tuple(value for value in VALID_MODALITIES if value in requested)
    if not normalized:
        raise ValueError("At least one retrieval modality is required")
    return normalized


def _unique_required_values(values: Iterable[str], field_name: str) -> list[str]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _required_text(value, f"{field_name} item")
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(slots=True)
class RetrievalFilter:
    document_ids: frozenset[str] | None = None
    document_types: frozenset[str] | None = None
    source_paths: frozenset[str] | None = None

    def __post_init__(self) -> None:
        self.document_ids = _optional_values(self.document_ids, "document_ids")
        self.document_types = _optional_values(
            self.document_types,
            "document_types",
            lowercase=True,
        )
        self.source_paths = _optional_values(self.source_paths, "source_paths")

    def to_dict(self) -> dict[str, list[str] | None]:
        return {
            "document_ids": sorted(self.document_ids) if self.document_ids else None,
            "document_types": sorted(self.document_types) if self.document_types else None,
            "source_paths": sorted(self.source_paths) if self.source_paths else None,
        }


@dataclass(slots=True)
class RetrievalRequest:
    query_text: str
    top_k: int = 10
    candidate_k: int = 30
    modalities: tuple[str, ...] = VALID_MODALITIES
    filters: RetrievalFilter | None = None

    def __post_init__(self) -> None:
        self.query_text = _required_text(self.query_text, "query_text")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer")
        if isinstance(self.candidate_k, bool) or not isinstance(self.candidate_k, int):
            raise TypeError("candidate_k must be an integer")
        if not 1 <= self.top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        if not self.top_k <= self.candidate_k <= MAX_CANDIDATE_K:
            raise ValueError(
                f"candidate_k must be between top_k and {MAX_CANDIDATE_K}"
            )
        self.modalities = _modalities(self.modalities)
        if self.filters is not None and not isinstance(self.filters, RetrievalFilter):
            raise TypeError("filters must be a RetrievalFilter")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "modalities": list(self.modalities),
            "filters": self.filters.to_dict() if self.filters else None,
        }


@dataclass(slots=True)
class RetrievalHit:
    rank: int
    record_id: str
    document_id: str
    modality: str
    raw_score: float
    fusion_score: float
    source_block_ids: list[str]
    content: Any
    context: str
    provenance: dict[str, Any]
    source_path: str
    document_type: str
    asset_path: str | None = None
    adjacent_context: list[dict[str, Any]] = field(default_factory=list)
    context_pages: list[int] = field(default_factory=list)
    evidence_blocks: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        self.record_id = _required_text(self.record_id, "record_id")
        self.document_id = _required_text(self.document_id, "document_id")
        normalized_modality = _required_text(self.modality, "modality").lower()
        if normalized_modality not in VALID_MODALITIES:
            raise ValueError(f"Unsupported retrieval modality: {normalized_modality}")
        self.modality = normalized_modality
        if isinstance(self.raw_score, bool) or not isinstance(self.raw_score, (int, float)):
            raise TypeError("raw_score must be numeric")
        if isinstance(self.fusion_score, bool) or not isinstance(
            self.fusion_score, (int, float)
        ):
            raise TypeError("fusion_score must be numeric")
        self.raw_score = float(self.raw_score)
        self.fusion_score = float(self.fusion_score)
        if not math.isfinite(self.raw_score) or not math.isfinite(self.fusion_score):
            raise ValueError("Retrieval scores must be finite")
        self.source_block_ids = _unique_required_values(
            self.source_block_ids,
            "source_block_ids",
        )
        if not isinstance(self.context, str):
            raise TypeError("context must be a string")
        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a dictionary")
        self.provenance = dict(self.provenance)
        self.source_path = _required_text(self.source_path, "source_path")
        self.document_type = _required_text(
            self.document_type,
            "document_type",
        ).lower()
        if self.asset_path is not None:
            self.asset_path = _required_text(self.asset_path, "asset_path")
        if not isinstance(self.adjacent_context, list) or not all(
            isinstance(item, dict) for item in self.adjacent_context
        ):
            raise TypeError("adjacent_context must be a list of dictionaries")
        self.adjacent_context = copy.deepcopy(self.adjacent_context)
        if not isinstance(self.context_pages, list) or any(
            isinstance(page, bool) or not isinstance(page, int) or page < 1
            for page in self.context_pages
        ):
            raise TypeError("context_pages must be a list of positive integers")
        self.context_pages = list(dict.fromkeys(sorted(self.context_pages)))
        if not isinstance(self.evidence_blocks, list) or not all(
            isinstance(item, dict) for item in self.evidence_blocks
        ):
            raise TypeError("evidence_blocks must be a list of dictionaries")
        self.evidence_blocks = copy.deepcopy(self.evidence_blocks)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResponse:
    query_text: str
    snapshot_build_id: str
    hits: list[RetrievalHit]
    searched_modalities: list[str]
    elapsed_ms: float
    query_variants: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.query_text = _required_text(self.query_text, "query_text")
        self.snapshot_build_id = _required_text(
            self.snapshot_build_id,
            "snapshot_build_id",
        )
        if not isinstance(self.hits, list) or not all(
            isinstance(hit, RetrievalHit) for hit in self.hits
        ):
            raise TypeError("hits must be a list of RetrievalHit values")
        self.hits = list(self.hits)
        self.searched_modalities = list(_modalities(self.searched_modalities))
        if isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, (int, float)):
            raise TypeError("elapsed_ms must be numeric")
        self.elapsed_ms = float(self.elapsed_ms)
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be a finite non-negative value")
        if not isinstance(self.query_variants, list):
            raise TypeError("query_variants must be a list of strings")
        self.query_variants = _unique_required_values(
            self.query_variants or [self.query_text],
            "query_variants",
        )
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list of strings")
        self.warnings = [
            _required_text(value, "warning")
            for value in self.warnings
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "snapshot_build_id": self.snapshot_build_id,
            "hits": [hit.to_dict() for hit in self.hits],
            "searched_modalities": list(self.searched_modalities),
            "elapsed_ms": self.elapsed_ms,
            "query_variants": list(self.query_variants),
            "warnings": list(self.warnings),
        }

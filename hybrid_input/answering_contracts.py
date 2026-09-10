from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .retrieval_contracts import RetrievalResponse


MAX_EVIDENCE_ITEMS = 50
MAX_EVIDENCE_CHARS = 200_000


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _unique_texts(values: list[str], field_name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _required_text(value, f"{field_name} item")
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


@dataclass(slots=True)
class AnswerRequest:
    query_text: str
    retrieval_response: RetrievalResponse
    max_evidence_items: int = 6
    max_evidence_chars: int = 10_000
    allow_visual: bool = True

    def __post_init__(self) -> None:
        self.query_text = _required_text(self.query_text, "query_text")
        if not isinstance(self.retrieval_response, RetrievalResponse):
            raise TypeError("retrieval_response must be a RetrievalResponse")
        if isinstance(self.max_evidence_items, bool) or not isinstance(
            self.max_evidence_items, int
        ):
            raise TypeError("max_evidence_items must be an integer")
        if not 1 <= self.max_evidence_items <= MAX_EVIDENCE_ITEMS:
            raise ValueError(
                f"max_evidence_items must be between 1 and {MAX_EVIDENCE_ITEMS}"
            )
        if isinstance(self.max_evidence_chars, bool) or not isinstance(
            self.max_evidence_chars, int
        ):
            raise TypeError("max_evidence_chars must be an integer")
        if not 1 <= self.max_evidence_chars <= MAX_EVIDENCE_CHARS:
            raise ValueError(
                f"max_evidence_chars must be between 1 and {MAX_EVIDENCE_CHARS}"
            )
        if not isinstance(self.allow_visual, bool):
            raise TypeError("allow_visual must be a boolean")


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    record_id: str
    document_id: str
    modality: str
    role: str
    content: Any
    context: str
    provenance: dict[str, Any]
    source_path: str
    document_type: str
    source_block_ids: list[str]
    retrieval_rank: int
    raw_score: float
    fusion_score: float
    relation: str = ""
    asset_path: str | None = None

    def __post_init__(self) -> None:
        self.evidence_id = _required_text(self.evidence_id, "evidence_id")
        self.record_id = _required_text(self.record_id, "record_id")
        self.document_id = _required_text(self.document_id, "document_id")
        self.modality = _required_text(self.modality, "modality").lower()
        if self.modality not in {"text", "table", "visual"}:
            raise ValueError(f"Unsupported evidence modality: {self.modality}")
        self.role = _required_text(self.role, "role").lower()
        if self.role not in {"core", "neighbor", "related"}:
            raise ValueError(f"Unsupported evidence role: {self.role}")
        if not isinstance(self.context, str):
            raise TypeError("context must be a string")
        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a dictionary")
        self.provenance = copy.deepcopy(self.provenance)
        self.source_path = _required_text(self.source_path, "source_path")
        self.document_type = _required_text(self.document_type, "document_type").lower()
        self.source_block_ids = _unique_texts(self.source_block_ids, "source_block_ids")
        if isinstance(self.retrieval_rank, bool) or not isinstance(self.retrieval_rank, int):
            raise TypeError("retrieval_rank must be an integer")
        if self.retrieval_rank < 1:
            raise ValueError("retrieval_rank must be at least 1")
        for field_name in ("raw_score", "fusion_score"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            setattr(self, field_name, value)
        if not isinstance(self.relation, str):
            raise TypeError("relation must be a string")
        self.relation = self.relation.strip()
        if self.asset_path is not None:
            self.asset_path = _required_text(self.asset_path, "asset_path")
        self.content = copy.deepcopy(self.content)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceDecision:
    evidence_id: str
    relevant: bool
    support_level: str
    score: float
    rationale: str = ""

    def __post_init__(self) -> None:
        self.evidence_id = _required_text(self.evidence_id, "evidence_id")
        if not isinstance(self.relevant, bool):
            raise TypeError("relevant must be a boolean")
        self.support_level = _required_text(self.support_level, "support_level").lower()
        if self.support_level not in {"direct", "context", "weak", "conflict", "none"}:
            raise ValueError(f"Unsupported support level: {self.support_level}")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        self.score = float(self.score)
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be a finite value between 0 and 1")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")
        self.rationale = self.rationale.strip()


@dataclass(slots=True)
class AnswerClaim:
    text: str
    evidence_ids: list[str]

    def __post_init__(self) -> None:
        self.text = _required_text(self.text, "claim text")
        self.evidence_ids = _unique_texts(self.evidence_ids, "evidence_ids")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Citation:
    evidence_id: str
    record_id: str
    modality: str
    source_path: str
    document_type: str
    provenance: dict[str, Any]
    source_block_ids: list[str]

    def __post_init__(self) -> None:
        self.evidence_id = _required_text(self.evidence_id, "evidence_id")
        self.record_id = _required_text(self.record_id, "record_id")
        self.modality = _required_text(self.modality, "modality").lower()
        self.source_path = _required_text(self.source_path, "source_path")
        self.document_type = _required_text(self.document_type, "document_type").lower()
        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a dictionary")
        self.provenance = copy.deepcopy(self.provenance)
        self.source_block_ids = _unique_texts(self.source_block_ids, "source_block_ids")

    @classmethod
    def from_evidence(cls, evidence: EvidenceItem) -> "Citation":
        return cls(
            evidence_id=evidence.evidence_id,
            record_id=evidence.record_id,
            modality=evidence.modality,
            source_path=evidence.source_path,
            document_type=evidence.document_type,
            provenance=evidence.provenance,
            source_block_ids=evidence.source_block_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResponse:
    query_text: str
    snapshot_build_id: str
    status: AnswerStatus
    answer_text: str
    claims: list[AnswerClaim] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    selected_evidence: list[EvidenceItem] = field(default_factory=list)
    decisions: list[EvidenceDecision] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        self.query_text = _required_text(self.query_text, "query_text")
        self.snapshot_build_id = _required_text(self.snapshot_build_id, "snapshot_build_id")
        if isinstance(self.status, str):
            self.status = AnswerStatus(self.status)
        if not isinstance(self.status, AnswerStatus):
            raise TypeError("status must be an AnswerStatus")
        if not isinstance(self.answer_text, str):
            raise TypeError("answer_text must be a string")
        if not isinstance(self.claims, list) or not all(
            isinstance(item, AnswerClaim) for item in self.claims
        ):
            raise TypeError("claims must be a list of AnswerClaim values")
        if not isinstance(self.citations, list) or not all(
            isinstance(item, Citation) for item in self.citations
        ):
            raise TypeError("citations must be a list of Citation values")
        if not isinstance(self.selected_evidence, list) or not all(
            isinstance(item, EvidenceItem) for item in self.selected_evidence
        ):
            raise TypeError("selected_evidence must be a list of EvidenceItem values")
        if not isinstance(self.decisions, list) or not all(
            isinstance(item, EvidenceDecision) for item in self.decisions
        ):
            raise TypeError("decisions must be a list of EvidenceDecision values")
        self.limitations = _unique_texts(self.limitations, "limitations", allow_empty=True)
        self.warnings = _unique_texts(self.warnings, "warnings", allow_empty=True)
        if isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, (int, float)):
            raise TypeError("elapsed_ms must be numeric")
        self.elapsed_ms = float(self.elapsed_ms)
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be a finite non-negative value")
        if self.status is AnswerStatus.ANSWERED and not self.claims:
            raise ValueError("answered responses must contain at least one claim")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "snapshot_build_id": self.snapshot_build_id,
            "status": self.status.value,
            "answer_text": self.answer_text,
            "claims": [item.to_dict() for item in self.claims],
            "citations": [item.to_dict() for item in self.citations],
            "selected_evidence": [item.to_dict() for item in self.selected_evidence],
            "decisions": [asdict(item) for item in self.decisions],
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "elapsed_ms": self.elapsed_ms,
        }

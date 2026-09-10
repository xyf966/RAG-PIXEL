"""Pluggable hybrid document ingestion.

The package deliberately stops at :class:`HybridDocument`. Search and index
implementations consume that stable contract but are not part of ingestion.
"""

from .contracts import Artifact, ArtifactBundle, HybridDocument, Provenance, VisionResult
from .answering import AnswerEngine, BailianChatClient, OllamaChatClient
from .answering_contracts import (
    AnswerClaim,
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
    Citation,
    EvidenceDecision,
    EvidenceItem,
)
from .index_records import IndexBuildConfig, IndexRecord
from .pipeline import HybridPipeline, PipelineConfig, build_default_pipeline
from .retrieval import HybridSearchEngine
from .retrieval_contracts import (
    RetrievalFilter,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResponse,
)
from .vision import DeferredVisionProcessor, PixelRAGEmbeddingSession, PixelRAGVisionProcessor

__all__ = [
    "Artifact",
    "ArtifactBundle",
    "AnswerClaim",
    "AnswerEngine",
    "AnswerRequest",
    "AnswerResponse",
    "AnswerStatus",
    "BailianChatClient",
    "Citation",
    "EvidenceDecision",
    "EvidenceItem",
    "HybridDocument",
    "HybridPipeline",
    "HybridSearchEngine",
    "IndexBuildConfig",
    "IndexRecord",
    "OllamaChatClient",
    "DeferredVisionProcessor",
    "PixelRAGVisionProcessor",
    "PixelRAGEmbeddingSession",
    "PipelineConfig",
    "Provenance",
    "RetrievalFilter",
    "RetrievalHit",
    "RetrievalRequest",
    "RetrievalResponse",
    "VisionResult",
    "build_default_pipeline",
]

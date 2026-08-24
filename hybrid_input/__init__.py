"""Pluggable hybrid document ingestion.

The package deliberately stops at :class:`HybridDocument`. Search and index
implementations consume that stable contract but are not part of ingestion.
"""

from .contracts import Artifact, ArtifactBundle, HybridDocument, Provenance, VisionResult
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
    "HybridDocument",
    "HybridPipeline",
    "HybridSearchEngine",
    "IndexBuildConfig",
    "IndexRecord",
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

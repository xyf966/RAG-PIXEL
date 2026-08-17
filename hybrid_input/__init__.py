"""Pluggable hybrid document ingestion.

The package deliberately stops at :class:`HybridDocument`. Search and index
implementations consume that stable contract but are not part of ingestion.
"""

from .contracts import Artifact, ArtifactBundle, HybridDocument, Provenance, VisionResult
from .pipeline import HybridPipeline, PipelineConfig, build_default_pipeline
from .vision import DeferredVisionProcessor, PixelRAGEmbeddingSession, PixelRAGVisionProcessor

__all__ = [
    "Artifact",
    "ArtifactBundle",
    "HybridDocument",
    "HybridPipeline",
    "DeferredVisionProcessor",
    "PixelRAGVisionProcessor",
    "PixelRAGEmbeddingSession",
    "PipelineConfig",
    "Provenance",
    "VisionResult",
    "build_default_pipeline",
]

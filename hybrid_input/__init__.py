"""Pluggable hybrid document ingestion.

The package deliberately stops at :class:`HybridDocument`. Search and index
implementations consume that stable contract but are not part of ingestion.
"""

from .contracts import Artifact, ArtifactBundle, HybridDocument, Provenance, VisionResult
from .pipeline import HybridPipeline, PipelineConfig, build_default_pipeline

__all__ = [
    "Artifact",
    "ArtifactBundle",
    "HybridDocument",
    "HybridPipeline",
    "PipelineConfig",
    "Provenance",
    "VisionResult",
    "build_default_pipeline",
]

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .contracts import Artifact, ArtifactBundle, HybridDocument, VisionResult


class DocumentDetector(ABC):
    @abstractmethod
    def detect(self, source: Path) -> str:
        raise NotImplementedError


class DocumentParser(ABC):
    name = "parser"

    @abstractmethod
    def supports(self, document_type: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
        raise NotImplementedError


class DocumentRenderer(ABC):
    name = "renderer"

    @abstractmethod
    def supports(self, document_type: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def render(self, source: Path, output_dir: Path) -> list[Path]:
        raise NotImplementedError


class LayoutEnricher(ABC):
    """Adds physical page/slide coordinates without changing parsed content."""

    name = "layout-enricher"

    @abstractmethod
    def supports(self, document_type: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def enrich(self, bundle: ArtifactBundle, output_dir: Path) -> ArtifactBundle:
        raise NotImplementedError


class ContextEnricher(ABC):
    """Adds nearby semantic context without changing parsed content or assets."""

    name = "context-enricher"

    @abstractmethod
    def enrich(self, artifacts: list[Artifact]) -> list[Artifact]:
        raise NotImplementedError


class ImageProcessor(ABC):
    name = "image-processor"

    @abstractmethod
    def process(self, artifacts: list[Artifact], output_dir: Path) -> list[Artifact]:
        raise NotImplementedError


class VisionProcessor(ABC):
    name = "vision"

    @abstractmethod
    def process(self, artifacts: list[Artifact]) -> list[VisionResult]:
        raise NotImplementedError


class DocumentAssembler(ABC):
    name = "assembler"

    @abstractmethod
    def assemble(
        self,
        bundle: ArtifactBundle,
        artifacts: list[Artifact],
        vision_results: list[VisionResult],
        providers: dict[str, str],
    ) -> HybridDocument:
        raise NotImplementedError

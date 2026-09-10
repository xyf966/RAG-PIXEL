from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .assembler import DefaultAssembler
from .context import NearbyTextContextEnricher
from .contracts import HybridDocument
from .detector import SignatureDetector
from .interfaces import ContextEnricher, DocumentAssembler, DocumentDetector, ImageProcessor, LayoutEnricher, VisionProcessor
from .layout import OfficePdfLayoutEnricher
from .parsers import (
    DoclingSubprocessParser,
    FallbackDocumentParser,
    ImageFileParser,
    OpenDataLoaderPdfSubprocessParser,
    OpenPyxlSubprocessParser,
    PlainTextParser,
    PyMuPdfParser,
)
from .processors import PillowImageProcessor
from .registry import ParserRegistry
from .renderers import (
    MicrosoftExcelChartSubprocessRenderer,
    MicrosoftOfficeSubprocessRenderer,
    MicrosoftOfficeVisualSubprocessRenderer,
    MicrosoftWordVisualSubprocessRenderer,
)
from .vision import DeferredVisionProcessor, PixelRAGVisionProcessor


def _recognition_contract(artifacts: list) -> list:
    """Normalize parser-specific visual kinds without rasterizing or embedding."""
    for artifact in artifacts:
        if artifact.kind not in {"image", "visual_task", "visual"}:
            continue
        semantic_type = str(artifact.metadata.get("semantic_type") or "").lower()
        artifact.kind = "visual"
        artifact.visual_type = artifact.visual_type or (
            "chart" if semantic_type == "chart" else
            str(artifact.metadata.get("visual_type") or "image")
        )
        artifact.metadata.setdefault("materialization_status", "deferred-to-index")
    return artifacts


@dataclass(slots=True)
class PipelineConfig:
    parser_by_type: dict[str, str] = field(
        default_factory=lambda: {
            "pdf": "pdf-structured-auto",
            "doc": "docling-subprocess",
            "docx": "docling-subprocess",
            "ppt": "docling-subprocess",
            "pptx": "docling-subprocess",
            "xls": "docling-subprocess",
            "xlsx": "xlsx-native-auto",
            "txt": "plain-text",
            "md": "plain-text",
            "html": "plain-text",
            "image": "image-file",
        }
    )
    image_processor: str = "pillow"
    vision_processor: str = "deferred"
    vision_model: str | None = None
    vision_device: str = "auto"
    vision_instruction: str = "Represent this document image for visual retrieval."
    assembler: str = "default"
    enable_office_layout: bool = True
    strict_layout: bool = False


class HybridPipeline:
    def __init__(
        self,
        detector: DocumentDetector,
        parsers: ParserRegistry,
        image_processor: ImageProcessor,
        vision_processor: VisionProcessor,
        assembler: DocumentAssembler,
        layout_enricher: LayoutEnricher | None = None,
        context_enricher: ContextEnricher | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.parsers = parsers
        self.image_processor = image_processor
        self.vision_processor = vision_processor
        self.assembler = assembler
        self.layout_enricher = layout_enricher
        self.context_enricher = context_enricher
        self.config = config or PipelineConfig()

    def ingest(self, source: Path, output_root: Path) -> HybridDocument:
        source = source.resolve()
        document_type = self.detector.detect(source)
        parser_name = self.config.parser_by_type.get(document_type)
        if not parser_name:
            raise RuntimeError(f"No parser configured for document type: {document_type}")
        parser = self.parsers.get(parser_name)
        if not parser.supports(document_type):
            raise RuntimeError(f"Parser {parser.name} does not support {document_type}")
        # Keep same-named files from different directories isolated.
        from .parsers import document_id

        staging = output_root / f"{source.stem}-{document_id(source)[:12]}"
        staging.mkdir(parents=True, exist_ok=True)
        bundle = parser.parse(source, staging, document_type)
        if self.config.enable_office_layout and self.layout_enricher and self.layout_enricher.supports(document_type):
            try:
                bundle = self.layout_enricher.enrich(bundle, staging)
            except Exception as exc:
                if self.config.strict_layout:
                    raise
                bundle.warnings.append(f"Layout enrichment failed: {exc}")
        contextualized = (
            self.context_enricher.enrich(bundle.artifacts)
            if self.context_enricher
            else bundle.artifacts
        )
        # Recognition stops at text/table/visual.  It never normalizes pixels
        # and never invokes Pixel; both belong to index construction.
        artifacts = _recognition_contract(contextualized)
        vision_results = []
        providers = {
            "detector": type(self.detector).__name__,
            "parser": bundle.parser,
            "layout_enricher": self.layout_enricher.name if self.layout_enricher else "none",
            "context_enricher": self.context_enricher.name if self.context_enricher else "none",
            "image_processor": "deferred-to-index",
            "vision_processor": "deferred-to-index",
            "assembler": self.assembler.name,
        }
        result = self.assembler.assemble(bundle, artifacts, vision_results, providers)
        result.write(staging / "hybrid-document.json")
        return result


def discover_docling_python() -> Path:
    configured = os.environ.get("HYBRID_DOCLING_PYTHON")
    if configured:
        return Path(configured).expanduser().resolve()
    if importlib.util.find_spec("docling") is not None:
        return Path(sys.executable)
    return Path.home() / "Desktop" / "图片识别" / ".conda-env" / "python.exe"


def build_default_pipeline(config: PipelineConfig | None = None, docling_python: Path | None = None) -> HybridPipeline:
    config = config or PipelineConfig()
    worker_python = docling_python or discover_docling_python()
    pymupdf = PyMuPdfParser()
    opendataloader = OpenDataLoaderPdfSubprocessParser(worker_python)
    docling = DoclingSubprocessParser(worker_python)
    openpyxl = OpenPyxlSubprocessParser(worker_python)
    registry = ParserRegistry(
        [
            PlainTextParser(),
            ImageFileParser(),
            pymupdf,
            opendataloader,
            FallbackDocumentParser(opendataloader, pymupdf, "pdf-structured-auto"),
            openpyxl,
            FallbackDocumentParser(openpyxl, docling, "xlsx-native-auto"),
            docling,
        ]
    )
    if config.vision_processor == "deferred":
        vision_processor: VisionProcessor = DeferredVisionProcessor()
    elif config.vision_processor == "pixelrag":
        vision_processor = PixelRAGVisionProcessor(
            model=config.vision_model,
            device=config.vision_device,
            instruction=config.vision_instruction,
        )
    else:
        raise ValueError(f"Unknown vision processor: {config.vision_processor}")

    return HybridPipeline(
        detector=SignatureDetector(),
        parsers=registry,
        image_processor=PillowImageProcessor(),
        vision_processor=vision_processor,
        assembler=DefaultAssembler(),
        context_enricher=NearbyTextContextEnricher(),
        layout_enricher=OfficePdfLayoutEnricher(
            MicrosoftOfficeSubprocessRenderer(worker_python),
            office_visual_renderer=MicrosoftOfficeVisualSubprocessRenderer(worker_python),
        ),
        config=config,
    )

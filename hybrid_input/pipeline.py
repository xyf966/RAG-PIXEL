from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .assembler import DefaultAssembler
from .contracts import HybridDocument
from .detector import SignatureDetector
from .interfaces import DocumentAssembler, DocumentDetector, ImageProcessor, LayoutEnricher, VisionProcessor
from .layout import OfficePdfLayoutEnricher
from .parsers import DoclingSubprocessParser, ImageFileParser, PlainTextParser, PyMuPdfParser
from .processors import PillowImageProcessor
from .registry import ParserRegistry
from .renderers import MicrosoftExcelChartSubprocessRenderer, MicrosoftOfficeSubprocessRenderer
from .vision import DeferredVisionProcessor


@dataclass(slots=True)
class PipelineConfig:
    parser_by_type: dict[str, str] = field(
        default_factory=lambda: {
            "pdf": "pymupdf",
            "doc": "docling-subprocess",
            "docx": "docling-subprocess",
            "ppt": "docling-subprocess",
            "pptx": "docling-subprocess",
            "xls": "docling-subprocess",
            "xlsx": "docling-subprocess",
            "txt": "plain-text",
            "md": "plain-text",
            "html": "plain-text",
            "image": "image-file",
        }
    )
    image_processor: str = "pillow"
    vision_processor: str = "deferred"
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
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.parsers = parsers
        self.image_processor = image_processor
        self.vision_processor = vision_processor
        self.assembler = assembler
        self.layout_enricher = layout_enricher
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
        artifacts = self.image_processor.process(bundle.artifacts, staging)
        vision_results = self.vision_processor.process(artifacts)
        providers = {
            "detector": type(self.detector).__name__,
            "parser": parser.name,
            "layout_enricher": self.layout_enricher.name if self.layout_enricher else "none",
            "image_processor": self.image_processor.name,
            "vision_processor": self.vision_processor.name,
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
    worker_python = docling_python or discover_docling_python()
    registry = ParserRegistry(
        [
            PlainTextParser(),
            ImageFileParser(),
            PyMuPdfParser(),
            DoclingSubprocessParser(worker_python),
        ]
    )
    return HybridPipeline(
        detector=SignatureDetector(),
        parsers=registry,
        image_processor=PillowImageProcessor(),
        vision_processor=DeferredVisionProcessor(),
        assembler=DefaultAssembler(),
        layout_enricher=OfficePdfLayoutEnricher(
            MicrosoftOfficeSubprocessRenderer(worker_python),
            excel_chart_renderer=MicrosoftExcelChartSubprocessRenderer(worker_python),
        ),
        config=config,
    )

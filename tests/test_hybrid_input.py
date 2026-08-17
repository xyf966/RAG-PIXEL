from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from hybrid_input.contracts import Artifact, ArtifactBundle, Provenance
from hybrid_input.context import NearbyTextContextEnricher
from hybrid_input.detector import SignatureDetector
from hybrid_input.interfaces import DocumentParser
from hybrid_input.layout import align_layout
from hybrid_input.parsers import FallbackDocumentParser
from hybrid_input.pipeline import PipelineConfig, build_default_pipeline
from hybrid_input.vision import PixelRAGVisionProcessor
from hybrid_input.opendataloader_worker import _table_data


class HybridInputSmokeTests(unittest.TestCase):
    def test_nearby_context_enricher_is_decoupled_and_preserves_captions(self) -> None:
        artifacts = [
            Artifact("a", "text", Provenance("sample.docx", page=1), text="Section heading", reading_order=0),
            Artifact("b", "image", Provenance("sample.docx", page=1), reading_order=1),
            Artifact("c", "text", Provenance("sample.docx", page=1), text="Figure caption", reading_order=2),
            Artifact("d", "image", Provenance("sample.docx", page=1), context="Native caption", reading_order=3),
        ]

        result = NearbyTextContextEnricher().enrich(artifacts)

        self.assertEqual(result[1].context, "Section heading\nFigure caption")
        self.assertEqual(result[1].metadata["context_block_ids"], ["a", "c"])
        self.assertEqual(result[3].context, "Native caption")
        self.assertEqual(result[3].metadata["context_source"], "parser_caption")

    def test_nearby_context_prefers_excel_sheet_over_rendered_page(self) -> None:
        artifacts = [
            Artifact(
                "table",
                "table",
                Provenance("sample.xlsx", page=3, sheet="Energy"),
                text="Monthly energy table",
                reading_order=0,
            ),
            Artifact(
                "chart",
                "image",
                Provenance("sample.xlsx", page=2, sheet="Energy"),
                reading_order=1,
            ),
        ]

        result = NearbyTextContextEnricher().enrich(artifacts)

        self.assertEqual(result[1].context, "Monthly energy table")

    def test_opendataloader_table_markdown_preserves_empty_middle_columns(self) -> None:
        node = {
            "number of columns": 4,
            "rows": [
                {
                    "row number": 1,
                    "cells": [
                        {"row number": 1, "column number": 1, "content": "8"},
                        {"row number": 1, "column number": 3, "content": "description"},
                        {"row number": 1, "column number": 4, "content": "note"},
                    ],
                }
            ],
        }

        markdown, rows = _table_data(node)

        self.assertEqual(markdown.splitlines()[0], "| 8 |  | description | note |")
        self.assertEqual([cell["column"] for cell in rows[0]["cells"]], [1, 3, 4])

    def test_default_routes_use_structured_pdf_and_native_xlsx(self) -> None:
        routes = PipelineConfig().parser_by_type
        self.assertEqual(routes["pdf"], "pdf-structured-auto")
        self.assertEqual(routes["xlsx"], "xlsx-native-auto")

    def test_structured_pdf_parser_falls_back_without_coupling(self) -> None:
        class FailingParser(DocumentParser):
            name = "structured"

            def supports(self, document_type: str) -> bool:
                return document_type == "pdf"

            def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
                raise RuntimeError("structure backend unavailable")

        class WorkingParser(DocumentParser):
            name = "fallback"

            def supports(self, document_type: str) -> bool:
                return document_type == "pdf"

            def parse(self, source: Path, output_dir: Path, document_type: str) -> ArtifactBundle:
                return ArtifactBundle("id", str(source), "pdf", self.name, [])

        result = FallbackDocumentParser(FailingParser(), WorkingParser()).parse(
            Path("sample.pdf"), Path("output"), "pdf"
        )
        self.assertEqual(result.parser, "fallback")
        self.assertIn("Structured PDF parser failed", result.warnings[0])

    def test_text_ingest_writes_versioned_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.txt"
            source.write_text("标题\n第一段\n\n第二段", encoding="utf-8")
            result = build_default_pipeline().ingest(source, root / "output")
            self.assertEqual(result.document_type, "txt")
            self.assertEqual([item.kind for item in result.artifacts], ["text", "text"])
            manifest = next((root / "output").glob("*/hybrid-document.json"))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")

    def test_image_ingest_normalizes_and_defers_vision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.png"
            Image.new("RGB", (64, 48), (20, 40, 60)).save(source)
            result = build_default_pipeline().ingest(source, root / "output")
            self.assertEqual(len(result.artifacts), 1)
            self.assertEqual(result.artifacts[0].kind, "image")
            self.assertTrue(Path(result.artifacts[0].asset_path or "").is_file())
            self.assertEqual(result.vision_results[0].metadata["status"], "pending")

    def test_pixelrag_processor_embeds_only_existing_image_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "sample.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            artifacts = [
                Artifact("image", "image", Provenance("sample.png"), asset_path=str(image_path)),
                Artifact("text", "text", Provenance("sample.png"), text="caption"),
            ]
            calls = []

            def fake_embedder(items, model, device, instruction):
                calls.append((items, model, device, instruction))
                return [[0.25, 0.75]]

            processor = PixelRAGVisionProcessor(
                model="local-model",
                device="cpu",
                instruction="test instruction",
                embedder=fake_embedder,
            )
            result = processor.process(artifacts)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].block_id, "image")
            self.assertEqual(result[0].vector, [0.25, 0.75])
            self.assertEqual(result[0].metadata["status"], "complete")
            self.assertEqual(result[0].metadata["dimension"], 2)
            self.assertEqual(calls[0][0], [{"path": str(image_path)}])

    def test_pipeline_can_select_pixelrag_without_loading_the_model(self) -> None:
        pipeline = build_default_pipeline(PipelineConfig(vision_processor="pixelrag"))
        self.assertEqual(pipeline.vision_processor.name, "pixelrag")

    def test_pixelrag_processor_reuses_one_resident_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "sample.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            artifact = Artifact(
                "image", "image", Provenance("sample.png"), asset_path=str(image_path)
            )

            class FakeSession:
                load_count = 1
                resolved_device = "cpu"

                def __init__(self, model, device):
                    self.calls = 0

                def embed_items(self, items, model, device, instruction):
                    self.calls += 1
                    return [[1.0, 0.0]]

                def close(self):
                    return None

            with patch(
                "hybrid_input.vision.PixelRAGEmbeddingSession", side_effect=FakeSession
            ) as factory:
                processor = PixelRAGVisionProcessor(model="local-model", device="cpu")
                first = processor.process([artifact])
                second = processor.process([artifact])

            factory.assert_called_once_with("local-model", "cpu")
            self.assertEqual(processor._session.calls, 2)
            self.assertEqual(first[0].metadata["model_load_count"], 1)
            self.assertEqual(second[0].metadata["model_load_count"], 1)

    def test_bundle_round_trip_and_signature_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "fake.pdf"
            source.write_bytes(b"not a pdf")
            with self.assertRaises(ValueError):
                SignatureDetector().detect(source)
        payload = {
            "document_id": "id",
            "source_path": "source.txt",
            "document_type": "txt",
            "parser": "test",
            "artifacts": [],
        }
        self.assertEqual(ArtifactBundle.from_dict(payload).to_dict()["schema_version"], "1.0")

    def test_layout_enrichment_preserves_logical_and_physical_location(self) -> None:
        native = ArtifactBundle(
            "doc", "sample.docx", "docx", "native",
            [
                Artifact("doc:1", "text", Provenance("sample.docx", locator="#/texts/0"), text="第二段文字"),
                Artifact("doc:2", "text", Provenance("sample.docx", locator="#/texts/1"), text="第一段文字"),
            ],
        )
        rendered = ArtifactBundle(
            "pdf", "sample.pdf", "pdf", "pdf",
            [
                Artifact("pdf:1", "text", Provenance("sample.pdf", page=1, bbox=[0.1, 0.1, 0.8, 0.2], locator="page:1/block:1"), text="第一段文字"),
                Artifact("pdf:2", "text", Provenance("sample.pdf", page=2, bbox=[0.1, 0.2, 0.8, 0.3], locator="page:2/block:1"), text="第二段文字"),
            ],
        )
        result = align_layout(native, rendered)
        self.assertEqual(result.artifacts[0].provenance.page, 2)
        self.assertEqual(result.artifacts[0].provenance.bbox, [0.1, 0.2, 0.8, 0.3])
        self.assertEqual(result.artifacts[0].metadata["logical_locator"], "#/texts/0")
        self.assertEqual(result.artifacts[1].provenance.page, 1)


if __name__ == "__main__":
    unittest.main()

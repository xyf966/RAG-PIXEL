from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from hybrid_input.contracts import Artifact, ArtifactBundle, Provenance
from hybrid_input.context import NearbyTextContextEnricher
from hybrid_input.detector import SignatureDetector
from hybrid_input.interfaces import DocumentParser
from hybrid_input.layout import (
    align_layout,
    materialize_excel_visuals,
    materialize_powerpoint_visuals,
    materialize_word_visuals,
)
from hybrid_input.parsers import FallbackDocumentParser
from hybrid_input.pipeline import PipelineConfig, build_default_pipeline
from hybrid_input.processors import PillowImageProcessor
from hybrid_input.vision import PixelRAGVisionProcessor
from hybrid_input.opendataloader_worker import _add_vector_visual_regions, _table_data
from hybrid_input.office_worker import _word_diagram_texts


class HybridInputSmokeTests(unittest.TestCase):
    def test_missing_index_time_visual_asset_does_not_abort_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Artifact(
                "missing", "image", Provenance("sample.pdf"),
                asset_path=str(Path(temporary) / "not-created.png"),
            )
            result = PillowImageProcessor().process([artifact], Path(temporary) / "output")
            self.assertEqual(result, [])
            self.assertEqual(artifact.metadata["filtered"], "missing-or-unreadable-asset")

    def test_pdf_vector_regions_are_recognized_without_rasterization(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "vectors.pdf"
            document = pymupdf.open()
            page = document.new_page(width=500, height=500)
            page.draw_rect(pymupdf.Rect(50, 50, 300, 200))
            page.draw_polyline([(50, 180), (120, 130), (200, 150), (300, 80)])
            page.draw_rect(pymupdf.Rect(350, 60, 390, 100))
            document.save(source)
            document.close()

            artifacts = [
                Artifact(
                    "chart-text", "text",
                    Provenance("vectors.pdf", page=1, bbox_original=[50, 50, 300, 200]),
                    text="0 1 2 3 series", reading_order=0,
                )
            ]
            document = pymupdf.open(source)
            _add_vector_visual_regions(document, source, "doc-id", artifacts)
            document.close()

            visuals = [item for item in artifacts if item.kind == "visual"]
            self.assertEqual({item.visual_type for item in visuals}, {"chart", "icon"})
            self.assertTrue(all(item.asset_path is None for item in visuals))

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
            self.assertEqual(payload["schema_version"], "1.1")

    def test_image_ingest_stops_at_visual_recognition_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.png"
            Image.new("RGB", (64, 48), (20, 40, 60)).save(source)
            result = build_default_pipeline().ingest(source, root / "output")
            self.assertEqual(len(result.artifacts), 1)
            self.assertEqual(result.artifacts[0].kind, "visual")
            self.assertEqual(result.artifacts[0].visual_type, "image")
            self.assertTrue(Path(result.artifacts[0].asset_path or "").is_file())
            self.assertEqual(result.vision_results, [])
            self.assertEqual(result.providers["vision_processor"], "deferred-to-index")

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
        self.assertEqual(ArtifactBundle.from_dict(payload).to_dict()["schema_version"], "1.1")

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

    def test_word_visual_bounds_materialize_images_and_semantic_text(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "layout.pdf"
            document = pymupdf.open()
            page = document.new_page(width=300, height=400)
            page.draw_rect(pymupdf.Rect(20, 30, 120, 100), color=(0, 0, 0), fill=(1, 1, 0))
            page.insert_text((35, 70), "chart")
            document.save(pdf_path)
            document.close()
            bundle = ArtifactBundle(
                "document-id-123456",
                str(root / "sample.docx"),
                "docx",
                "test",
                [Artifact("visual", "visual_task", Provenance("sample.docx"))],
            )
            materialize_word_visuals(
                bundle,
                pdf_path,
                [
                    {
                        "page": 1,
                        "bounds_points": [20, 30, 120, 100],
                        "object_type": 12,
                        "collection": "InlineShapes",
                        "text": "Quarterly chart",
                    }
                ],
                root,
            )
            image = next(item for item in bundle.artifacts if item.kind == "image")
            text = next(item for item in bundle.artifacts if item.kind == "text")
            self.assertTrue(Path(image.asset_path or "").is_file())
            self.assertEqual(image.provenance.page, 1)
            self.assertTrue(image.metadata["context_locked"])
            self.assertEqual(text.text, "Quarterly chart")
            self.assertEqual(text.provenance.source_file, "sample.docx")

    def test_word_visual_prefers_native_object_export_over_blank_pdf_crop(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "blank.pdf"
            document = pymupdf.open()
            document.new_page(width=300, height=400)
            document.save(pdf_path)
            document.close()
            native = root / "native-icon.png"
            icon = Image.new("RGB", (96, 96), "white")
            for offset in range(20, 76):
                icon.putpixel((offset, 20), (0, 0, 0))
                icon.putpixel((offset, 75), (0, 0, 0))
            icon.save(native)
            bundle = ArtifactBundle(
                "document-id-123456", str(root / "sample.docx"), "docx", "test",
                [Artifact("visual", "visual_task", Provenance("sample.docx"))],
            )
            materialize_word_visuals(
                bundle, pdf_path,
                [{
                    "page": 1, "bounds_points": [20, 30, 120, 100],
                    "object_type": 17, "collection": "InlineShapes",
                    "text": "Alarm outline", "path": str(native),
                    "render_method": "word-copy-as-picture",
                }],
                root,
            )
            visual = next(item for item in bundle.artifacts if item.kind == "image")
            self.assertEqual(Path(visual.asset_path or ""), native.resolve())
            self.assertEqual(visual.metadata["office_render_method"], "word-copy-as-picture")

    def test_word_visual_without_object_text_does_not_borrow_adjacent_text(self) -> None:
        visual = Artifact(
            "chart",
            "image",
            Provenance("sample.docx", page=1),
            reading_order=2,
            metadata={"context_locked": True},
        )
        adjacent = Artifact(
            "label",
            "text",
            Provenance("sample.docx", page=1),
            text="Unrelated adjacent label",
            reading_order=3,
        )
        result = NearbyTextContextEnricher().enrich([visual, adjacent])
        self.assertIsNone(result[0].context)

    def test_black_parser_placeholder_is_filtered_without_hiding_valid_solid_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            black = root / "black.png"
            white = root / "white.png"
            blue = root / "blue.png"
            Image.new("RGB", (64, 64), "black").save(black)
            Image.new("RGB", (64, 64), "white").save(white)
            Image.new("RGB", (64, 64), "blue").save(blue)
            artifacts = [
                Artifact("black", "image", Provenance("sample.docx"), asset_path=str(black)),
                Artifact("white", "image", Provenance("sample.docx"), asset_path=str(white)),
                Artifact("blue", "image", Provenance("sample.docx"), asset_path=str(blue)),
            ]
            result = PillowImageProcessor().process(artifacts, root / "output")
            self.assertEqual([item.block_id for item in result], ["blue"])
            self.assertEqual(artifacts[0].metadata["filtered"], "black-placeholder")
            self.assertEqual(artifacts[1].metadata["filtered"], "white-placeholder")

    def test_transparent_black_line_icon_is_composited_on_white(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "icon.png"
            icon = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            for offset in range(12, 52):
                icon.putpixel((offset, 32), (0, 0, 0, 255))
            icon.save(source)
            artifact = Artifact(
                "icon", "image", Provenance("sample.docx"), asset_path=str(source)
            )
            result = PillowImageProcessor().process([artifact], root / "output")
            self.assertEqual(len(result), 1)
            with Image.open(result[0].asset_path or "") as rendered:
                self.assertEqual(rendered.getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(rendered.getpixel((32, 32)), (0, 0, 0))

    def test_native_powerpoint_and_excel_visuals_become_image_artifacts(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            picture = root / "shape.png"
            Image.new("RGB", (80, 60), "green").save(picture)
            pdf_path = root / "slides.pdf"
            document = pymupdf.open()
            document.new_page(width=720, height=540)
            document.save(pdf_path)
            document.close()
            ppt = ArtifactBundle(
                "ppt-document-id", str(root / "sample.pptx"), "pptx", "test", []
            )
            materialize_powerpoint_visuals(
                ppt, pdf_path,
                [{
                    "path": str(picture), "slide": 1, "shape_index": 2,
                    "object_type": 13, "bounds_points": [10, 20, 90, 80],
                    "text": "Product photo", "render_method": "powerpoint-shape-export",
                }],
                root,
            )
            self.assertTrue(any(item.kind == "image" for item in ppt.artifacts))
            self.assertTrue(any(item.text == "Product photo" for item in ppt.artifacts))

            excel = ArtifactBundle(
                "excel-document-id", str(root / "sample.xlsx"), "xlsx", "test", []
            )
            materialize_excel_visuals(
                excel,
                [{
                    "path": str(picture), "sheet": "Dashboard", "shape_index": 1,
                    "object_type": 3, "cell_range": "A1:H20",
                    "bounds_points": [0, 0, 300, 200], "text": "Revenue chart",
                    "render_method": "excel-native-export",
                }],
            )
            image = next(item for item in excel.artifacts if item.kind == "image")
            self.assertEqual(image.provenance.sheet, "Dashboard")
            self.assertTrue(any(item.text == "Revenue chart" for item in excel.artifacts))

    def test_smartart_text_is_read_from_generic_docx_diagram_xml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "sample.docx"
            xml = (
                '<dgm:dataModel xmlns:dgm="urn:dgm" xmlns:a="urn:a">'
                '<a:t>First label</a:t><a:t>Second label</a:t></dgm:dataModel>'
            )
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("word/diagrams/data1.xml", xml)
            self.assertEqual(
                _word_diagram_texts(source), ["First label Second label"]
            )


if __name__ == "__main__":
    unittest.main()

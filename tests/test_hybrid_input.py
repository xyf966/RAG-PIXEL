from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hybrid_input.contracts import Artifact, ArtifactBundle, Provenance
from hybrid_input.detector import SignatureDetector
from hybrid_input.layout import align_layout
from hybrid_input.pipeline import build_default_pipeline


class HybridInputSmokeTests(unittest.TestCase):
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

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hybrid_input.contracts import Artifact, HybridDocument, Provenance, VisionResult
from hybrid_input.indexing import (
    HybridSearchEngine,
    _materialize_visuals_for_index,
    write_hybrid_index,
)


class FakeQuerySession:
    def embed_texts(self, texts: list[str], instruction: str = "") -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    def close(self) -> None:
        pass


class HybridIndexTests(unittest.TestCase):
    def test_office_visual_falls_back_to_identified_page_region(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.docx"
            source.write_bytes(b"office-placeholder")
            rendered = root / "layout.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page(width=300, height=300)
            page.draw_rect(
                pymupdf.Rect(40, 50, 200, 180), color=(0, 0, 0), fill=(0.7, 0.7, 0.7)
            )
            pdf.save(rendered)
            pdf.close()
            visual = Artifact(
                "office-visual", "visual",
                Provenance("sample.docx", page=1, bbox_original=[40, 50, 200, 180]),
                visual_type="chart",
            )
            document = HybridDocument("doc", str(source), "docx", [visual])

            class FailedNativeRenderer:
                def render(self, source, output_dir, export=True):
                    return [{"path": None, "render_method": "page-crop-fallback"}]

            class PdfRenderer:
                def render(self, source, output_dir):
                    return [rendered]

            result = _materialize_visuals_for_index(
                document, root / "assets", FailedNativeRenderer(), PdfRenderer()
            )

            self.assertEqual(len(result), 1)
            self.assertTrue(Path(result[0].asset_path or "").is_file())
            self.assertEqual(
                result[0].metadata["index_materialization"],
                "office-page-region-fallback",
            )

    def test_pdf_visual_is_materialized_only_at_index_time(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "region.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page(width=200, height=200)
            page.draw_rect(pymupdf.Rect(20, 20, 180, 180), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
            pdf.save(source)
            pdf.close()
            visual = Artifact(
                "visual-1", "visual",
                Provenance("region.pdf", page=1, bbox=[0.1, 0.1, 0.9, 0.9]),
                visual_type="diagram",
            )
            document = HybridDocument("doc", str(source), "pdf", [visual])

            result = _materialize_visuals_for_index(document, root / "index-assets", object())

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].kind, "image")
            self.assertTrue(Path(result[0].asset_path or "").is_file())
            self.assertEqual(document.artifacts[0].kind, "visual")
            self.assertEqual(document.artifacts[0].metadata["materialization_status"], "complete-at-index")

    def _document(self, asset: Path) -> HybridDocument:
        artifacts = [
            Artifact(
                "text-1",
                "text",
                Provenance("报告.docx", page=1),
                text="仓库机器人运行正常",
            ),
            Artifact(
                "table-1",
                "table",
                Provenance("报告.docx", page=2, cell_range="A1:B2"),
                text="设备 数量 机器人 3",
            ),
            Artifact(
                "image-1",
                "visual",
                Provenance("报告.docx", page=3),
                asset_path=str(asset),
                context="机器人现场图",
                visual_type="image",
            ),
        ]
        return HybridDocument(
            document_id="doc-1",
            source_path=str(asset.parent / "报告.docx"),
            document_type="docx",
            artifacts=artifacts,
            vision_results=[VisionResult("image-1", "pixelrag", vector=[1.0, 0.0])],
        )

    def test_separates_semantic_blocks_and_image_faiss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "image.png"
            asset.write_bytes(b"png")
            manifest = write_hybrid_index([self._document(asset)], root / "index")

            self.assertEqual(manifest["semantic_blocks"], 2)
            self.assertEqual(manifest["image_vectors"], 1)
            self.assertEqual(manifest["pixel_input_kinds"], ["visual"])
            self.assertTrue((root / "index" / "image-index.faiss").is_file())

            engine = HybridSearchEngine(root / "index", "fake-model")
            engine.session = FakeQuerySession()
            hits = engine.search("机器人", limit=5, include_images=False)
            self.assertTrue(any(hit["channel"] == "semantic" for hit in hits))
            self.assertTrue(any(hit["channel"] == "image" for hit in hits))

    def test_rejects_non_visual_vision_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "image.png"
            asset.write_bytes(b"png")
            document = self._document(asset)
            document.vision_results.append(
                VisionResult("text-1", "pixelrag", vector=[1.0, 0.0])
            )
            with self.assertRaisesRegex(RuntimeError, "Non-visual block"):
                write_hybrid_index([document], root / "index")


if __name__ == "__main__":
    unittest.main()

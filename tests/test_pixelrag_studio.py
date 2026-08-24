from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hybrid_input.contracts import Artifact, HybridDocument, Provenance, VisionResult
from hybrid_input.index_records import CharacterTokenCodec, IndexBuildConfig, build_table_records, build_text_records
from hybrid_input.indexing import (
    _materialize_visuals_for_index,
    build_index_snapshot,
    current_snapshot,
    prepare_visual_index,
)


class FakeSession:
    model_name = "fake-unified-model"
    requested_device = "cpu"
    resolved_device = "cpu"
    load_count = 0

    def __init__(
        self,
        dimension: int = 4,
        zero: bool = False,
        image_zero: bool = False,
    ) -> None:
        self.dimension = dimension
        self.zero = zero
        self.image_zero = image_zero
        self.image_calls = 0

    def encode_text(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode_text(self, token_ids: list[int]) -> str:
        return "".join(chr(value) for value in token_ids)

    def embed_texts(self, texts: list[str], instruction: str = "") -> np.ndarray:
        if self.zero:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        return np.asarray(
            [[1.0, float(index + 1), 0.5, 0.25] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    def embed_items(self, items, model_name, device="cpu", instruction="") -> np.ndarray:
        self.image_calls += len(items)
        if self.image_zero:
            return np.zeros((len(items), self.dimension), dtype=np.float32)
        return np.asarray(
            [[1.0, *([0.0] * (self.dimension - 1))] for _ in items],
            dtype=np.float32,
        )


class UnusedRenderer:
    def render(self, *args, **kwargs):
        raise AssertionError("Renderer should not be called")


class HybridIndexTests(unittest.TestCase):
    def test_text_chunking_preserves_titles_pages_overlap_and_stable_ids(self) -> None:
        artifacts = [
            Artifact("h1", "text", Provenance("doc", page=1), text="1. 概况", reading_order=1, metadata={"level": 1}),
            Artifact("p1", "text", Provenance("doc", page=1), text="甲" * 24, reading_order=2),
            Artifact("p2", "text", Provenance("doc", page=1), text="乙" * 24, reading_order=3),
            Artifact("p3", "text", Provenance("doc", page=2), text="丙" * 24, reading_order=4),
        ]
        document = HybridDocument("doc-1", "doc.txt", "txt", artifacts)
        config = IndexBuildConfig(
            text_target_tokens=32,
            text_max_tokens=48,
            text_overlap_tokens=8,
            expected_dimension=4,
        )
        codec = CharacterTokenCodec()
        first = build_text_records(document, codec, config)
        second = build_text_records(document, codec, config)

        self.assertGreaterEqual(len(first), 3)
        self.assertEqual([item.record_id for item in first], [item.record_id for item in second])
        self.assertTrue(all(len(codec.encode(item.embedding_text)) <= 48 for item in first))
        self.assertTrue(any(item.overlap_source_block_ids for item in first))
        pages_by_id = {artifact.block_id: artifact.provenance.page for artifact in artifacts}
        self.assertTrue(
            all(len({pages_by_id[block_id] for block_id in item.source_block_ids}) == 1 for item in first)
        )
        self.assertIn("1. 概况", first[0].context)
        self.assertEqual(first[0].embedding_text.count("1. 概况"), 1)
        self.assertEqual(
            [item["block_id"] for item in first[0].structure["source_provenance"]],
            first[0].source_block_ids,
        )

    def test_long_text_respects_hard_limit_and_uses_overlapping_windows(self) -> None:
        artifact = Artifact(
            "long", "text", Provenance("doc", page=1),
            text="甲" * 100, reading_order=1,
        )
        document = HybridDocument("doc-long", "doc.txt", "txt", [artifact])
        config = IndexBuildConfig(
            text_target_tokens=24,
            text_max_tokens=32,
            text_overlap_tokens=8,
            expected_dimension=4,
        )
        codec = CharacterTokenCodec()
        records = build_text_records(document, codec, config)

        self.assertGreater(len(records), 1)
        self.assertTrue(all(len(codec.encode(item.embedding_text)) <= 32 for item in records))
        self.assertTrue(all(item.source_block_ids == ["long"] for item in records))
        self.assertEqual(
            records[0].original_content["text"][-8:],
            records[1].original_content["text"][:8],
        )

    def test_table_chunks_repeat_headers_and_split_wide_tables(self) -> None:
        headers = ["ID", *[f"C{index}" for index in range(1, 14)]]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend(
            "| " + " | ".join([f"R{row}", *[f"V{row}-{column}" for column in range(1, 14)]]) + " |"
            for row in range(25)
        )
        table = Artifact(
            "table-1", "table",
            Provenance("book.xlsx", sheet="Sheet1", cell_range="A1:N26"),
            text="\n".join(lines), context="设备台账",
        )
        document = HybridDocument("doc-table", "book.xlsx", "xlsx", [table])
        config = IndexBuildConfig(
            text_target_tokens=512, text_max_tokens=2000, text_overlap_tokens=96,
            table_max_rows=20, table_max_columns=12, expected_dimension=4,
        )
        records = build_table_records(document, CharacterTokenCodec(), config)
        repeated = build_table_records(document, CharacterTokenCodec(), config)

        self.assertEqual(len(records), 4)
        self.assertEqual(
            [item.record_id for item in records],
            [item.record_id for item in repeated],
        )
        self.assertTrue(all(len(item.structure["headers"]) <= 12 for item in records))
        self.assertTrue(all(item.structure["headers"][0] == "ID" for item in records))
        self.assertTrue(all("Columns: ID" in item.embedding_text for item in records))
        self.assertTrue(all(len(item.structure["rows"]) <= 20 for item in records))
        self.assertTrue(all(item.source_path == "book.xlsx" for item in records))

    def test_native_key_value_table_keeps_first_row_and_cell_metadata(self) -> None:
        metadata = {
            "semantic_type": "worksheet_region",
            "rows": 2,
            "columns": 2,
            "merged_ranges": ["A1:A2"],
            "cells": [
                {"cell": "A1", "row": 1, "column": 1, "value": "项目代号", "row_span": 1, "column_span": 1},
                {"cell": "B1", "row": 1, "column": 2, "value": "青岚", "row_span": 1, "column_span": 1},
                {"cell": "A2", "row": 2, "column": 1, "value": "目标", "row_span": 1, "column_span": 1},
                {"cell": "B2", "row": 2, "column": 2, "value": "=18%", "formula": "=18%", "row_span": 1, "column_span": 1},
            ],
        }
        artifact = Artifact(
            "kv", "table", Provenance("book.xlsx", sheet="概览", cell_range="A1:B2"),
            text="| 项目代号 | 青岚 |\n|---|---|\n| 目标 | =18% |",
            metadata=metadata,
        )
        records = build_table_records(
            HybridDocument("doc-kv", "book.xlsx", "xlsx", [artifact]),
            CharacterTokenCodec(), IndexBuildConfig(expected_dimension=4),
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].structure["headers"], ["Field", "Value"])
        self.assertEqual(records[0].structure["rows"][0], ["项目代号", "青岚"])
        self.assertEqual(records[0].structure["formulas"], [{"cell": "B2", "formula": "=18%"}])
        self.assertEqual(records[0].structure["merged_ranges"], ["A1:A2"])

    def test_structured_pdf_title_row_is_not_used_as_header(self) -> None:
        metadata = {
            "columns": 3,
            "table_cells": [
                {"row": 1, "cells": [{"row": 1, "column": 1, "column_span": 3, "text": "合作权益"}]},
                {"row": 2, "cells": [
                    {"row": 2, "column": 1, "text": "编号"},
                    {"row": 2, "column": 2, "text": "分类"},
                    {"row": 2, "column": 3, "text": "说明"},
                ]},
                {"row": 3, "cells": [
                    {"row": 3, "column": 1, "text": "1"},
                    {"row": 3, "column": 2, "row_span": 2, "text": "品牌"},
                    {"row": 3, "column": 3, "text": "授权"},
                ]},
                {"row": 4, "cells": [
                    {"row": 4, "column": 1, "text": "2"},
                    {"row": 4, "column": 3, "text": "素材"},
                ]},
            ],
        }
        artifact = Artifact(
            "pdf-table", "table", Provenance("rights.pdf", page=1),
            text="| 合作权益 | | |\n|---|---|---|\n|编号|分类|说明|\n|1|品牌|授权|",
            metadata=metadata,
        )
        records = build_table_records(
            HybridDocument("doc-pdf", "rights.pdf", "pdf", [artifact]),
            CharacterTokenCodec(), IndexBuildConfig(expected_dimension=4),
        )

        self.assertEqual(records[0].context, "合作权益")
        self.assertEqual(records[0].structure["headers"], ["编号", "分类", "说明"])
        self.assertEqual(
            records[0].structure["rows"],
            [["1", "品牌", "授权"], ["2", "品牌", "素材"]],
        )

    def test_structured_table_ignores_empty_spanning_row_before_header(self) -> None:
        metadata = {
            "columns": 3,
            "table_cells": [
                {"row": 1, "cells": [
                    {"row": 1, "column": 1, "column_span": 3, "text": ""},
                ]},
                {"row": 2, "cells": [
                    {"row": 2, "column": 1, "text": "季度"},
                    {"row": 2, "column": 2, "text": "收入"},
                    {"row": 2, "column": 3, "text": "利润"},
                ]},
                {"row": 3, "cells": [
                    {"row": 3, "column": 1, "text": "Q1"},
                    {"row": 3, "column": 2, "text": "100"},
                    {"row": 3, "column": 3, "text": "20"},
                ]},
            ],
        }
        artifact = Artifact(
            "blank-first-row",
            "table",
            Provenance("slides.pptx", slide=2),
            metadata=metadata,
        )

        records = build_table_records(
            HybridDocument("doc-slides", "slides.pptx", "pptx", [artifact]),
            CharacterTokenCodec(),
            IndexBuildConfig(expected_dimension=4),
        )

        self.assertEqual(records[0].structure["headers"], ["季度", "收入", "利润"])
        self.assertEqual(records[0].structure["rows"], [["Q1", "100", "20"]])
        self.assertEqual(records[0].structure["header_source_row"], 2)

    def test_escaped_pipe_and_long_cell_are_preserved_in_bounded_segments(self) -> None:
        artifact = Artifact(
            "long-table", "table", Provenance("table.md", line_start=1),
            text="| Key | Description |\n|---|---|\n| A \\| B | " + "长" * 160 + " |",
        )
        config = IndexBuildConfig(
            text_target_tokens=48, text_max_tokens=64, text_overlap_tokens=8,
            expected_dimension=4,
        )
        records = build_table_records(
            HybridDocument("doc-long-table", "table.md", "md", [artifact]),
            CharacterTokenCodec(), config,
        )

        self.assertGreater(len(records), 1)
        self.assertTrue(all(len(item.embedding_text) <= 64 for item in records))
        self.assertTrue(all("Columns: Key | Description" in item.embedding_text for item in records))
        self.assertEqual(records[0].structure["rows"][0][0], "A | B")
        self.assertTrue(all(item.structure["segment_count"] == len(records) for item in records))

    def test_non_tabular_content_uses_explicit_raw_fallback(self) -> None:
        artifact = Artifact(
            "raw-table", "table", Provenance("legacy.doc", page=2),
            text="设备 数量 机器人 3", context="旧格式表格",
        )
        records = build_table_records(
            HybridDocument("doc-raw", "legacy.doc", "doc", [artifact]),
            CharacterTokenCodec(), IndexBuildConfig(expected_dimension=4),
        )

        self.assertEqual(records[0].structure["format"], "raw")
        self.assertIn("旧格式表格", records[0].embedding_text)
        self.assertEqual(records[0].provenance["page"], 2)

    def test_office_visual_falls_back_to_identified_page_region(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.docx"
            source.write_bytes(b"office-placeholder")
            rendered = root / "layout.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page(width=300, height=300)
            page.draw_rect(pymupdf.Rect(40, 50, 200, 180), color=(0, 0, 0), fill=(0.7, 0.7, 0.7))
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
                document,
                root / "assets",
                FailedNativeRenderer(),
                PdfRenderer(),
                "require-com",
            )
            self.assertEqual(len(result), 1)
            self.assertTrue(Path(result[0].asset_path or "").is_file())
            self.assertEqual(result[0].metadata["index_materialization"], "office-page-region-fallback")

    def test_office_visual_strict_mode_rejects_missing_native_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.pptx"
            source.write_bytes(b"office-placeholder")
            document = HybridDocument(
                "ppt", str(source), "pptx",
                [Artifact(
                    "visual", "visual", Provenance(str(source), slide=1),
                    visual_type="image",
                )],
            )
            fallback = root / "fallback.png"
            fallback.write_bytes(b"not-accepted")

            class MissingNativeRenderer:
                def render(self, source, output_dir, export=True):
                    return [{
                        "path": str(fallback),
                        "render_method": "page-crop-fallback",
                    }]

            with self.assertRaisesRegex(RuntimeError, "native visual export missed"):
                _materialize_visuals_for_index(
                    document, root / "assets", MissingNativeRenderer()
                )

    def test_office_document_without_visuals_does_not_start_com(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "table.xlsx"
            source.write_bytes(b"office-placeholder")
            document = HybridDocument(
                "xlsx", str(source), "xlsx",
                [Artifact("table", "table", Provenance(str(source)), text="|A|B|")],
            )

            result = _materialize_visuals_for_index(
                document, root / "assets", UnusedRenderer()
            )

            self.assertEqual(result, [])

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
            result = _materialize_visuals_for_index(document, root / "assets", object())
            self.assertEqual(result[0].kind, "image")
            self.assertEqual(document.artifacts[0].kind, "visual")
            self.assertEqual(document.artifacts[0].metadata["materialization_status"], "complete-at-index")

    def test_visual_contract_skips_other_modalities_and_nonindexable_visuals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "doc.txt"
            source.write_text("source", encoding="utf-8")
            skipped = Artifact(
                "skip", "visual", Provenance(str(source)), visual_type="image",
                metadata={"indexable": False},
            )
            document = HybridDocument(
                "doc", str(source), "txt",
                [
                    Artifact("text", "text", Provenance(str(source)), text="正文"),
                    Artifact("table", "table", Provenance(str(source)), text="|A|B|"),
                    skipped,
                ],
            )
            session = FakeSession()
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                batch = prepare_visual_index(
                    [document], Path("visual-output"), session,
                    IndexBuildConfig(expected_dimension=4), UnusedRenderer(),
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(len(batch.records), 0)
            self.assertEqual(batch.source_visuals, 0)
            self.assertEqual(session.image_calls, 0)
            self.assertEqual(skipped.metadata["materialization_status"], "skipped-not-indexable")

    def test_visual_contract_rejects_unknown_type_and_duplicate_block_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            source.write_bytes(b"not-read")
            invalid = HybridDocument(
                "invalid", str(source), "image",
                [Artifact("visual", "visual", Provenance(str(source)), asset_path=str(source), visual_type="video")],
            )
            with self.assertRaisesRegex(RuntimeError, "Unsupported visual_type"):
                prepare_visual_index(
                    [invalid], root / "invalid-output", FakeSession(),
                    IndexBuildConfig(expected_dimension=4), UnusedRenderer(),
                )

            first = HybridDocument(
                "first", str(source), "image",
                [Artifact("same", "visual", Provenance(str(source)), asset_path=str(source), visual_type="image")],
            )
            second = HybridDocument(
                "second", str(source), "image",
                [Artifact("same", "visual", Provenance(str(source)), asset_path=str(source), visual_type="image")],
            )
            with self.assertRaisesRegex(RuntimeError, "Duplicate visual block_id"):
                prepare_visual_index(
                    [first, second], root / "duplicate-output", FakeSession(),
                    IndexBuildConfig(expected_dimension=4), UnusedRenderer(),
                )
            self.assertFalse((root / "duplicate-output").exists())

    def test_office_native_visual_export_is_preferred(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "slides.pptx"
            source.write_bytes(b"office-placeholder")
            exported = root / "native.png"
            Image.new("RGB", (80, 60), "green").save(exported)
            visual = Artifact(
                "chart", "visual", Provenance(str(source), page=1, slide=1),
                visual_type="chart",
            )
            document = HybridDocument("ppt", str(source), "pptx", [visual])

            class NativeRenderer:
                def render(self, source, output_dir, export=True):
                    return [{"path": str(exported), "render_method": "powerpoint-shape-export"}]

            batch = prepare_visual_index(
                [document], root / "visual-output", FakeSession(),
                IndexBuildConfig(expected_dimension=4), NativeRenderer(),
            )

            self.assertEqual(len(batch.records), 1)
            self.assertEqual(
                batch.records[0].structure["metadata"]["index_materialization"],
                "powerpoint-shape-export",
            )

    def test_visual_zero_vector_is_rejected_before_index_persistence(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "valid.png"
            Image.new("RGB", (64, 64), "red").save(source)
            document = HybridDocument(
                "visual", str(source), "image",
                [Artifact("visual", "visual", Provenance(str(source)), asset_path=str(source), visual_type="image")],
            )
            with self.assertRaisesRegex(RuntimeError, "zero or non-finite"):
                prepare_visual_index(
                    [document], root / "visual-output", FakeSession(image_zero=True),
                    IndexBuildConfig(expected_dimension=4), UnusedRenderer(),
                )
            with self.assertRaisesRegex(RuntimeError, r"expected \(1, 4\)"):
                prepare_visual_index(
                    [document], root / "dimension-output", FakeSession(dimension=3),
                    IndexBuildConfig(expected_dimension=4), UnusedRenderer(),
                )

    def test_missing_pixel_result_is_retried_once(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "valid.png"
            Image.new("RGB", (64, 64), "red").save(source)
            document = HybridDocument(
                "visual", str(source), "image",
                [Artifact(
                    "visual", "visual", Provenance(str(source)),
                    asset_path=str(source), visual_type="image",
                )],
            )

            class RetryProcessor:
                calls = 0

                def __init__(self, **_kwargs):
                    pass

                def process(self, artifacts):
                    self.calls += 1
                    if self.calls == 1:
                        return []
                    return [
                        VisionResult(
                            artifact.block_id, "pixelrag", vector=[1.0, 0.0, 0.0, 0.0]
                        )
                        for artifact in artifacts
                    ]

            processor = RetryProcessor()
            with patch(
                "hybrid_input.indexing.PixelRAGVisionProcessor",
                return_value=processor,
            ):
                batch = prepare_visual_index(
                    [document], root / "visual-output", FakeSession(),
                    IndexBuildConfig(expected_dimension=4), UnusedRenderer(),
                )

            self.assertEqual(processor.calls, 2)
            self.assertEqual(len(batch.records), 1)

    def _publish(self, root: Path, document: HybridDocument, session: FakeSession):
        return build_index_snapshot(
            [document], [Path(document.source_path)], root / "index", session,
            IndexBuildConfig(expected_dimension=4), UnusedRenderer(),
        )

    def test_snapshot_contains_three_valid_indexes_and_current_pointer(self) -> None:
        import faiss

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "doc.txt"
            source.write_text("source", encoding="utf-8")
            document = HybridDocument(
                "doc", str(source), "txt",
                [
                    Artifact("text", "text", Provenance(str(source), line_start=1), text="索引正文"),
                    Artifact("table", "table", Provenance(str(source), line_start=2), text="| A | B |\n|---|---|\n|1|2|"),
                ],
            )
            session = FakeSession()
            manifest = self._publish(root, document, session)
            snapshot = current_snapshot(root / "index")

            self.assertEqual(manifest["schema_version"], "2.0")
            self.assertEqual(manifest["counts"]["text_records"], 1)
            self.assertEqual(manifest["counts"]["table_records"], 1)
            self.assertEqual(faiss.read_index(str(snapshot / "text.faiss")).ntotal, 1)
            self.assertEqual(faiss.read_index(str(snapshot / "table.faiss")).ntotal, 1)
            self.assertEqual(faiss.read_index(str(snapshot / "visual.faiss")).ntotal, 0)
            self.assertEqual(session.image_calls, 0)
            self.assertIn("build-report.json", manifest["files"])

    def test_current_snapshot_rejects_a_file_changed_after_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "doc.txt"
            source.write_text("source", encoding="utf-8")
            document = HybridDocument(
                "doc", str(source), "txt",
                [Artifact("text", "text", Provenance(str(source)), text="正文")],
            )
            self._publish(root, document, FakeSession())
            snapshot = current_snapshot(root / "index")
            with (snapshot / "text-metadata.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")

            with self.assertRaisesRegex(RuntimeError, "checksum validation"):
                current_snapshot(root / "index")

    def test_current_snapshot_rejects_a_changed_visual_asset(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            image = Image.new("RGB", (64, 64), "red")
            image.putpixel((5, 5), (0, 0, 255))
            image.save(source)
            document = HybridDocument(
                "visual-doc", str(source), "image",
                [Artifact(
                    "visual", "visual", Provenance(str(source)),
                    asset_path=str(source), visual_type="image",
                )],
            )
            self._publish(root, document, FakeSession())
            snapshot = current_snapshot(root / "index")
            metadata = json.loads(
                (snapshot / "visual-metadata.jsonl").read_text(encoding="utf-8")
            )
            asset = snapshot / metadata["asset_path"]
            Image.new("RGB", (64, 64), "green").save(asset)

            with self.assertRaisesRegex(RuntimeError, "Visual asset checksum"):
                current_snapshot(root / "index")

    def test_duplicate_visuals_share_one_embedding_but_keep_two_records(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "image.png"
            image = Image.new("RGB", (64, 64), "red")
            image.putpixel((10, 10), (0, 0, 255))
            image.save(source)
            artifacts = [
                Artifact(f"visual-{index}", "visual", Provenance(str(source), page=index), asset_path=str(source), visual_type="image")
                for index in (1, 2)
            ]
            document = HybridDocument("visual-doc", str(source), "image", artifacts)
            session = FakeSession()
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                batch = prepare_visual_index(
                    [document], Path("visual-output"), session,
                    IndexBuildConfig(expected_dimension=4), UnusedRenderer(),
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(len(batch.records), 2)
            self.assertEqual(batch.unique_embeddings, 1)
            self.assertEqual(session.image_calls, 1)
            self.assertEqual(batch.vectors.shape, (2, 4))
            self.assertTrue(np.allclose(np.linalg.norm(batch.vectors, axis=1), 1.0))
            self.assertEqual(batch.records[0].asset_path, batch.records[1].asset_path)
            self.assertEqual(
                batch.records[1].structure["metadata"]["embedding_duplicate_of"],
                "visual-1",
            )

    def test_invalid_visual_is_reported_without_blocking_empty_indexes(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "white.png"
            Image.new("RGB", (64, 64), "white").save(source)
            artifact = Artifact(
                "white", "visual", Provenance(str(source)),
                asset_path=str(source), visual_type="image",
            )
            document = HybridDocument("white-doc", str(source), "image", [artifact])
            session = FakeSession()
            batch = prepare_visual_index(
                [document], root / "visual-output", session,
                IndexBuildConfig(expected_dimension=4), UnusedRenderer(),
            )

            self.assertEqual(len(batch.records), 0)
            self.assertEqual(batch.filtered_visuals, 1)
            self.assertIn("white-placeholder", batch.warnings[0])
            self.assertEqual(session.image_calls, 0)
            self.assertEqual(artifact.metadata["materialization_status"], "filtered-at-index")

    def test_failed_build_preserves_current_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "doc.txt"
            source.write_text("source", encoding="utf-8")
            document = HybridDocument(
                "doc", str(source), "txt",
                [Artifact("text", "text", Provenance(str(source)), text="正文")],
            )
            self._publish(root, document, FakeSession())
            original = (root / "index" / "CURRENT").read_text(encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "zero or non-finite"):
                self._publish(root, document, FakeSession(zero=True))
            self.assertEqual((root / "index" / "CURRENT").read_text(encoding="utf-8"), original)
            self.assertFalse(list((root / "index").glob(".staging-*")))

    def test_legacy_index_requires_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "hybrid-index.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Legacy index schema"):
                current_snapshot(root)

    def test_office_visual_snapshot_requires_strict_native_policy(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "slides.pptx"
            source.write_bytes(b"office-placeholder")
            exported = root / "native.png"
            Image.new("RGB", (64, 64), "red").save(exported)
            document = HybridDocument(
                "ppt", str(source), "pptx",
                [Artifact(
                    "visual", "visual", Provenance(str(source), slide=1),
                    visual_type="image",
                )],
            )

            class NativeRenderer:
                def render(self, source, output_dir, export=True):
                    return [{
                        "path": str(exported),
                        "render_method": "powerpoint-shape-export",
                    }]

            build_index_snapshot(
                [document], [source], root / "index", FakeSession(),
                IndexBuildConfig(expected_dimension=4), NativeRenderer(),
            )
            snapshot = current_snapshot(root / "index")
            manifest_path = snapshot / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["chunking"]["office_visual_policy"] = "allow-page-fallback"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "required COM validation"):
                current_snapshot(root / "index")


if __name__ == "__main__":
    unittest.main()

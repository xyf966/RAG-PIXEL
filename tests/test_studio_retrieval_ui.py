from __future__ import annotations

import unittest

from pixelrag_studio import (
    _format_retrieval_content,
    _format_retrieval_hit_label,
    _format_retrieval_location,
)


class StudioRetrievalFormattingTests(unittest.TestCase):
    def test_location_formats_page_and_spreadsheet_coordinates(self) -> None:
        self.assertEqual(_format_retrieval_location({"page": 3}), "第 3 页")
        self.assertEqual(
            _format_retrieval_location({"sheet": "销售", "cell_range": "B2:F8"}),
            "工作表 销售 / B2:F8",
        )
        self.assertEqual(_format_retrieval_location({}), "未标注位置")

    def test_content_prefers_readable_text_and_table_forms(self) -> None:
        self.assertEqual(
            _format_retrieval_content({"modality": "text", "content": {"text": "正文"}}),
            "正文",
        )
        self.assertEqual(
            _format_retrieval_content(
                {"modality": "table", "content": {"raw": "| A | B |"}}
            ),
            "| A | B |",
        )

    def test_hit_label_exposes_rank_modality_scores_source_and_location(self) -> None:
        label = _format_retrieval_hit_label(
            {
                "rank": 2,
                "modality": "visual",
                "fusion_score": 0.016,
                "raw_score": 0.81234,
                "source_path": "C:/docs/report.pdf",
                "provenance": {"page": 7},
            }
        )
        self.assertIn("02", label)
        self.assertIn("VISUAL", label)
        self.assertIn("0.81234", label)
        self.assertIn("report.pdf", label)
        self.assertIn("第 7 页", label)


if __name__ == "__main__":
    unittest.main()

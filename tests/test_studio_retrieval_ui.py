from __future__ import annotations

import unittest

from pixelrag_studio import (
    _format_answer_citation_label,
    _format_answer_diagnostics,
    _format_answer_response,
    _format_evidence_block_content,
    _format_evidence_block_header,
    _format_retrieval_content,
    _format_retrieval_hit_label,
    _format_retrieval_location,
)


class StudioRetrievalFormattingTests(unittest.TestCase):
    def test_answer_diagnostics_exposes_decisions_claims_and_warnings(self) -> None:
        rendered = _format_answer_diagnostics(
            {
                "query_text": "博世赛车有哪些系列赛",
                "status": "insufficient_evidence",
                "elapsed_ms": 123.0,
                "decisions": [
                    {
                        "evidence_id": "E001",
                        "relevant": False,
                        "support_level": "context",
                        "score": 0.9,
                        "rationale": "只有类别标题",
                    }
                ],
                "claims": [],
                "warnings": ["列表问题的回答没有任何可识别的具体条目"],
            }
        )
        self.assertIn("query='博世赛车有哪些系列赛'", rendered)
        self.assertIn("support=context", rendered)
        self.assertIn("只有类别标题", rendered)
        self.assertIn("没有任何可识别的具体条目", rendered)

    def test_answer_response_exposes_answer_limitations_and_warnings(self) -> None:
        rendered = _format_answer_response(
            {
                "answer_text": "销售额为100万元。[E001]",
                "limitations": ["仅覆盖2025年"],
                "warnings": ["视觉证据未发送"],
            }
        )
        self.assertIn("销售额为100万元。[E001]", rendered)
        self.assertIn("限制：\n- 仅覆盖2025年", rendered)
        self.assertIn("诊断：\n- 视觉证据未发送", rendered)

    def test_answer_citation_label_exposes_id_source_location_and_modality(self) -> None:
        label = _format_answer_citation_label(
            {
                "evidence_id": "E001",
                "source_path": "C:/docs/report.pdf",
                "modality": "table",
                "provenance": {"page": 3},
            }
        )
        self.assertEqual(label, "[E001]  report.pdf  ·  第 3 页  ·  TABLE")

    def test_evidence_blocks_expose_role_modality_location_and_content(self) -> None:
        block = {
            "role": "related",
            "modality": "table",
            "provenance": {"page": 7},
            "content": {"markdown": "| 规格 | 数值 |"},
        }
        self.assertEqual(
            _format_evidence_block_header(block),
            "[同页关联 · TABLE · 第 7 页]",
        )
        self.assertEqual(_format_evidence_block_content(block), "| 规格 | 数值 |")

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

    def test_cross_page_context_and_location_are_visible(self) -> None:
        hit = {
            "rank": 1,
            "modality": "text",
            "fusion_score": 0.05,
            "raw_score": 0.7,
            "source_path": "C:/docs/电视.pdf",
            "provenance": {"page": 4},
            "context_pages": [4, 5],
            "content": {"text": "Pay particular attention to cords at the"},
            "adjacent_context": [
                {
                    "content": {"text": "plug end, at wall outlets"},
                    "provenance": {"page": 5},
                }
            ],
        }

        self.assertIn("上下文 第 4–5 页", _format_retrieval_hit_label(hit))
        content = _format_retrieval_content(hit)
        self.assertIn("Pay particular attention", content)
        self.assertIn("邻接上下文（第 5 页）", content)
        self.assertIn("plug end", content)

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

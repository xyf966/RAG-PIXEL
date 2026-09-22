from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pixelrag_studio import (
    _answer_trace_payload,
    _cached_hybrid_summary,
    _format_answer_citation_label,
    _format_answer_diagnostics,
    _format_answer_response,
    _format_evidence_block_content,
    _format_evidence_block_header,
    _format_retrieval_content,
    _format_retrieval_hit_label,
    _format_retrieval_location,
    _original_page_number,
    _resolve_original_page_source,
    _worker_hybrid,
)
from hybrid_input.parsers import document_id


class StudioRetrievalFormattingTests(unittest.TestCase):
    def test_original_page_number_uses_core_block_and_ppt_slide_fallback(self) -> None:
        self.assertEqual(
            _original_page_number({
                "document_type": "docx",
                "provenance": {},
                "evidence_blocks": [{"role": "core", "provenance": {"page": 3}}],
            }),
            3,
        )
        self.assertEqual(
            _original_page_number({
                "document_type": "pptx",
                "provenance": {"slide": 5},
            }),
            5,
        )

    def test_office_original_page_resolves_cached_layout_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            layout = (
                project / "artifacts" / "report-abcdef123456" / "layout-render"
            )
            layout.mkdir(parents=True)
            rendered = layout / "report.pdf"
            rendered.write_bytes(b"%PDF-test")
            source, page, kind = _resolve_original_page_source(
                project,
                {
                    "document_id": "abcdef1234567890",
                    "document_type": "docx",
                    "source_path": "C:/docs/report.docx",
                    "provenance": {"page": 2},
                },
            )
            self.assertEqual(source, rendered)
        self.assertEqual(page, 2)
        self.assertEqual(kind, "office")
    def test_office_original_page_reports_missing_layout_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "重新执行输入解析"):
                _resolve_original_page_source(
                    Path(directory),
                    {
                        "document_id": "abcdef1234567890",
                        "document_type": "xlsx",
                        "source_path": "C:/docs/report.xlsx",
                        "provenance": {"page": 1},
                    },
                )

    def test_answer_trace_omits_document_content_and_raw_model_response(self) -> None:
        class Value:
            def to_dict(self):
                return {
                    "hits": [{
                        "rank": 1,
                        "record_id": "r1",
                        "document_id": "d1",
                        "modality": "text",
                        "content": {"text": "secret document text"},
                        "context": "secret context",
                        "evidence_blocks": [{"content": "secret block"}],
                    }]
                }

        class Evidence:
            def to_dict(self):
                return {"evidence_id": "E001", "content": "secret", "context": "secret"}

        class Normalizer:
            def normalize(self, request):
                return [Evidence()]

        class Budgeter:
            last_limits = {"mode": "simple"}

        class Generator:
            last_response = {"answerable": True, "claims": [{"text": "raw secret"}]}

        class Engine:
            normalizer = Normalizer()
            budgeter = Budgeter()
            generator = Generator()
            selector = type("Selector", (), {"last_response": {"mode": "local"}})()

        class Request:
            max_evidence_items = 10
            max_evidence_chars = 16_000

        class Answer:
            def to_dict(self):
                return {
                    "claims": [{
                        "text": "answer",
                        "evidence_ids": ["E001"],
                        "supporting_quotes": {"E001": "secret quote"},
                    }],
                    "selected_evidence": [{
                        "evidence_id": "E001", "content": "secret", "context": "secret"
                    }],
                }

        trace = _answer_trace_payload(
            model="qwen3.8-max",
            send_visual_assets=False,
            request=Request(),
            retrieval_response=Value(),
            engine=Engine(),
            response=Answer(),
        )
        serialized = str(trace)
        self.assertEqual(trace["privacy_mode"], "metadata_only")
        self.assertNotIn("secret document text", serialized)
        self.assertNotIn("secret quote", serialized)
        self.assertNotIn("raw secret", serialized)

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


class StudioHybridIngestCacheTests(unittest.TestCase):
    @staticmethod
    def _write_cached_result(source: Path, artifacts: Path) -> str:
        digest = document_id(source)
        destination = artifacts / f"{source.stem}-{digest[:12]}" / "hybrid-document.json"
        destination.parent.mkdir(parents=True)
        destination.write_text(
            json.dumps({
                "document_id": digest,
                "source_path": str(source.resolve()),
                "document_type": source.suffix.lstrip("."),
                "artifacts": [{"kind": "text"}, {"kind": "visual"}],
                "vision_results": [],
                "providers": {},
                "warnings": [],
                "schema_version": "1.1",
            }),
            encoding="utf-8",
        )
        return digest

    def test_cached_summary_rejects_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "document.txt"
            artifacts = root / "artifacts"
            source.write_text("first", encoding="utf-8")
            digest = self._write_cached_result(source, artifacts)

            self.assertEqual(
                _cached_hybrid_summary(source, artifacts, digest),
                (2, 1),
            )
            source.write_text("changed", encoding="utf-8")
            self.assertIsNone(
                _cached_hybrid_summary(source, artifacts, document_id(source))
            )

    def test_worker_ingests_only_new_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "source"
            artifacts = root / "artifacts"
            sources.mkdir()
            artifacts.mkdir()
            existing = sources / "existing.txt"
            added = sources / "new.txt"
            existing.write_text("already parsed", encoding="utf-8")
            added.write_text("new document", encoding="utf-8")
            self._write_cached_result(existing, artifacts)

            calls: list[Path] = []

            class Pipeline:
                def ingest(self, source: Path, output_root: Path):
                    calls.append(source)
                    self.output_root = output_root
                    return SimpleNamespace(artifacts=[SimpleNamespace(kind="text")])

            pipeline = Pipeline()
            with (
                patch("pixelrag_studio._redirect_worker_output"),
                patch("hybrid_input.build_default_pipeline", return_value=pipeline) as build,
                patch("builtins.print"),
            ):
                _worker_hybrid(str(sources), str(artifacts))

            build.assert_called_once_with()
            self.assertEqual(calls, [added])
            self.assertEqual(pipeline.output_root, artifacts)


if __name__ == "__main__":
    unittest.main()

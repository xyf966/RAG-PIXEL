from __future__ import annotations

import json
import math
import unittest

from hybrid_input.retrieval_contracts import (
    RetrievalFilter,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResponse,
)


def _hit(**overrides) -> RetrievalHit:
    values = {
        "rank": 1,
        "record_id": "record-1",
        "document_id": "document-1",
        "modality": "text",
        "raw_score": 0.8,
        "fusion_score": 1.0 / 61.0,
        "source_block_ids": ["block-1"],
        "content": {"text": "正文"},
        "context": "章节",
        "provenance": {"page": 2},
        "source_path": "C:/documents/sample.pdf",
        "document_type": "pdf",
    }
    values.update(overrides)
    return RetrievalHit(**values)


class RetrievalContractTests(unittest.TestCase):
    def test_request_normalizes_query_modalities_and_filters(self) -> None:
        request = RetrievalRequest(
            "  查询内容  ",
            modalities=("VISUAL", "text", "visual"),
            filters=RetrievalFilter(
                document_ids={"doc-b", "doc-a"},
                document_types={"PDF", "DocX"},
                source_paths={"C:/b.pdf", "C:/a.pdf"},
            ),
        )

        self.assertEqual(request.query_text, "查询内容")
        self.assertEqual(request.modalities, ("text", "visual"))
        self.assertEqual(request.filters.document_types, frozenset({"pdf", "docx"}))
        self.assertEqual(
            request.to_dict()["filters"],
            {
                "document_ids": ["doc-a", "doc-b"],
                "document_types": ["docx", "pdf"],
                "source_paths": ["C:/a.pdf", "C:/b.pdf"],
            },
        )

    def test_empty_filter_collections_mean_no_restriction(self) -> None:
        filters = RetrievalFilter(document_ids=set(), document_types=[], source_paths=())
        self.assertIsNone(filters.document_ids)
        self.assertIsNone(filters.document_types)
        self.assertIsNone(filters.source_paths)

    def test_filter_rejects_strings_and_blank_items(self) -> None:
        with self.assertRaisesRegex(TypeError, "collection of strings"):
            RetrievalFilter(document_ids="doc-1")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            RetrievalFilter(document_ids={"  "})

    def test_request_rejects_empty_query(self) -> None:
        for value in ("", " \n\t"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "query_text must not be empty"):
                    RetrievalRequest(value)

    def test_request_rejects_invalid_limits(self) -> None:
        invalid = (
            {"top_k": 0},
            {"top_k": 101, "candidate_k": 101},
            {"top_k": 20, "candidate_k": 19},
            {"candidate_k": 1001},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    RetrievalRequest("query", **values)
        with self.assertRaises(TypeError):
            RetrievalRequest("query", top_k=True)

    def test_request_rejects_invalid_modalities(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported retrieval modalities"):
            RetrievalRequest("query", modalities=("audio",))
        with self.assertRaisesRegex(ValueError, "At least one"):
            RetrievalRequest("query", modalities=())
        with self.assertRaisesRegex(TypeError, "collection of strings"):
            RetrievalRequest("query", modalities="text")

    def test_request_rejects_non_filter_and_unknown_constructor_fields(self) -> None:
        with self.assertRaisesRegex(TypeError, "RetrievalFilter"):
            RetrievalRequest("query", filters={"document_ids": ["doc"]})
        with self.assertRaises(TypeError):
            RetrievalRequest("query", asset_path="C:/unsafe.png")

    def test_hit_normalizes_values_and_defensively_copies_provenance(self) -> None:
        provenance = {"page": 2}
        hit = _hit(
            modality="TABLE",
            document_type="XLSX",
            source_block_ids=["block-1", "block-1", "block-2"],
            provenance=provenance,
        )
        provenance["page"] = 99

        self.assertEqual(hit.modality, "table")
        self.assertEqual(hit.document_type, "xlsx")
        self.assertEqual(hit.source_block_ids, ["block-1", "block-2"])
        self.assertEqual(hit.provenance, {"page": 2})

    def test_hit_normalizes_cross_page_evidence(self) -> None:
        adjacent = [{"content": {"text": "下一页"}, "provenance": {"page": 3}}]
        evidence = [
            {
                "modality": "text",
                "content": {"text": "证据"},
                "provenance": {"page": 2},
            }
        ]
        hit = _hit(
            adjacent_context=adjacent,
            context_pages=[3, 2, 3],
            evidence_blocks=evidence,
        )
        adjacent[0]["content"]["text"] = "changed"
        evidence[0]["content"]["text"] = "changed"

        self.assertEqual(hit.context_pages, [2, 3])
        self.assertEqual(hit.adjacent_context[0]["content"]["text"], "下一页")
        self.assertEqual(hit.evidence_blocks[0]["content"]["text"], "证据")

    def test_hit_rejects_invalid_identity_scores_and_sources(self) -> None:
        cases = (
            ({"rank": 0}, ValueError),
            ({"record_id": " "}, ValueError),
            ({"modality": "audio"}, ValueError),
            ({"raw_score": math.nan}, ValueError),
            ({"fusion_score": math.inf}, ValueError),
            ({"source_block_ids": []}, ValueError),
            ({"provenance": []}, TypeError),
        )
        for values, error in cases:
            with self.subTest(values=values):
                with self.assertRaises(error):
                    _hit(**values)

    def test_response_normalizes_modalities_and_serializes_to_json(self) -> None:
        response = RetrievalResponse(
            query_text=" query ",
            snapshot_build_id=" build-1 ",
            hits=[_hit()],
            searched_modalities=["visual", "TEXT", "visual"],
            elapsed_ms=12,
            warnings=[" empty visual index "],
        )
        payload = response.to_dict()

        self.assertEqual(response.searched_modalities, ["text", "visual"])
        self.assertEqual(response.warnings, ["empty visual index"])
        self.assertEqual(payload["hits"][0]["content"], {"text": "正文"})
        self.assertIn('"snapshot_build_id": "build-1"', json.dumps(payload))

    def test_response_rejects_invalid_hits_elapsed_time_and_warnings(self) -> None:
        base = {
            "query_text": "query",
            "snapshot_build_id": "build-1",
            "hits": [_hit()],
            "searched_modalities": ["text"],
            "elapsed_ms": 1.0,
        }
        cases = (
            ({"hits": [{}]}, TypeError),
            ({"elapsed_ms": -1}, ValueError),
            ({"elapsed_ms": math.nan}, ValueError),
            ({"warnings": "warning"}, TypeError),
            ({"warnings": [" "]}, ValueError),
        )
        for override, error in cases:
            values = dict(base)
            values.update(override)
            with self.subTest(override=override):
                with self.assertRaises(error):
                    RetrievalResponse(**values)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest

from hybrid_input.answering_contracts import (
    AnswerClaim,
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
    Citation,
    EvidenceDecision,
    EvidenceItem,
)
from hybrid_input.retrieval_contracts import RetrievalHit, RetrievalResponse


def _retrieval_response() -> RetrievalResponse:
    return RetrievalResponse(
        query_text="问题",
        snapshot_build_id="build-1",
        hits=[
            RetrievalHit(
                rank=1,
                record_id="record-1",
                document_id="document-1",
                modality="text",
                raw_score=0.8,
                fusion_score=0.04,
                source_block_ids=["block-1"],
                content={"text": "证据"},
                context="章节",
                provenance={"page": 2},
                source_path="C:/docs/report.pdf",
                document_type="pdf",
            )
        ],
        searched_modalities=["text"],
        elapsed_ms=2.0,
    )


def _evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="E001",
        record_id="record-1",
        document_id="document-1",
        modality="text",
        role="core",
        content={"text": "销售额为100万元"},
        context="经营情况",
        provenance={"page": 2},
        source_path="C:/docs/report.pdf",
        document_type="pdf",
        source_block_ids=["block-1"],
        retrieval_rank=1,
        raw_score=0.8,
        fusion_score=0.04,
    )


class AnsweringContractTests(unittest.TestCase):
    def test_request_validates_retrieval_response_and_budgets(self) -> None:
        request = AnswerRequest("  问题  ", _retrieval_response())
        self.assertEqual(request.query_text, "问题")
        self.assertEqual(request.max_evidence_items, 6)
        self.assertEqual(request.max_evidence_chars, 10_000)
        with self.assertRaises(TypeError):
            AnswerRequest("问题", {})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AnswerRequest("问题", _retrieval_response(), max_evidence_items=0)
        with self.assertRaises(TypeError):
            AnswerRequest("问题", _retrieval_response(), allow_visual=1)  # type: ignore[arg-type]

    def test_evidence_defensively_copies_nested_values(self) -> None:
        content = {"text": "原文"}
        provenance = {"page": 2, "bbox": [1, 2, 3, 4]}
        item = _evidence()
        item.content = content
        item.provenance = provenance
        copied = EvidenceItem(**item.to_dict())
        content["text"] = "changed"
        provenance["bbox"][0] = 99
        self.assertEqual(copied.content["text"], "原文")
        self.assertEqual(copied.provenance["bbox"][0], 1)

    def test_decision_and_claim_reject_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceDecision("E001", True, "direct", 1.1)
        with self.assertRaises(ValueError):
            EvidenceDecision("E001", True, "unsupported", 0.5)
        with self.assertRaises(ValueError):
            AnswerClaim("事实", [])

    def test_answer_response_serializes_citations_and_status(self) -> None:
        evidence = _evidence()
        claim = AnswerClaim("销售额为100万元", ["E001"])
        response = AnswerResponse(
            query_text="销售额是多少？",
            snapshot_build_id="build-1",
            status=AnswerStatus.ANSWERED,
            answer_text="销售额为100万元。[E001]",
            claims=[claim],
            citations=[Citation.from_evidence(evidence)],
            selected_evidence=[evidence],
            decisions=[EvidenceDecision("E001", True, "direct", 0.9)],
            elapsed_ms=3.0,
        )
        payload = response.to_dict()
        self.assertEqual(payload["status"], "answered")
        self.assertEqual(payload["citations"][0]["provenance"]["page"], 2)
        self.assertIn('"evidence_id": "E001"', json.dumps(payload))

    def test_answered_response_requires_claims(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one claim"):
            AnswerResponse(
                query_text="问题",
                snapshot_build_id="build-1",
                status=AnswerStatus.ANSWERED,
                answer_text="答案",
            )


if __name__ == "__main__":
    unittest.main()

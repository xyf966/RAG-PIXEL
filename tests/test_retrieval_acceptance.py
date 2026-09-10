from __future__ import annotations

import unittest

from tools.run_retrieval_acceptance import calculate_metrics, evaluate_case


class Hit:
    def __init__(self, rank, modality, page, record_id):
        self.rank = rank
        self.modality = modality
        self.provenance = {"page": page}
        self.record_id = record_id
        self.raw_score = 0.5
        self.fusion_score = 0.01
        self.context = ""


class Response:
    snapshot_build_id = "build"
    elapsed_ms = 10.0
    warnings = []

    def __init__(self, hits):
        self.hits = hits


class RetrievalAcceptanceTests(unittest.TestCase):
    def test_exact_record_id_prevents_same_page_false_positive(self) -> None:
        case = {
            "id": "case",
            "question": "query",
            "expected_evidence": [
                {
                    "label": "figure",
                    "modality": "visual",
                    "page": 6,
                    "record_ids": ["expected"],
                }
            ],
        }
        result = evaluate_case(
            Response(
                [
                    Hit(1, "visual", 6, "wrong"),
                    Hit(2, "visual", 6, "expected"),
                ]
            ),
            case,
        )
        self.assertEqual(result["evidence"][0]["rank"], 2)

    def test_metrics_include_multi_evidence_strict_query_recall(self) -> None:
        results = [
            {
                "elapsed_ms": 100.0,
                "evidence": [
                    {"modality": "visual", "rank": 2},
                    {"modality": "table", "rank": 8},
                ],
            },
            {
                "elapsed_ms": 20.0,
                "evidence": [{"modality": "text", "rank": None}],
            },
        ]
        metrics = calculate_metrics(results, [5, 10])
        self.assertAlmostEqual(metrics["evidence_recall_at_5"], 1 / 3)
        self.assertAlmostEqual(metrics["evidence_recall_at_10"], 2 / 3)
        self.assertEqual(metrics["query_all_recall_at_5"], 0)
        self.assertEqual(metrics["query_all_recall_at_10"], 0.5)
        self.assertAlmostEqual(metrics["mrr"], 0.25)


if __name__ == "__main__":
    unittest.main()

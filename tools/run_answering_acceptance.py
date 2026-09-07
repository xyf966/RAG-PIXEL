from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_input.answering import AnswerEngine
from hybrid_input.answering import LLMEvidenceSelector, LLMAnswerGenerator, OllamaChatClient
from hybrid_input.answering_contracts import AnswerRequest, AnswerStatus
from hybrid_input.retrieval_contracts import RetrievalHit, RetrievalResponse


class RecordingClient:
    def __init__(self, client: OllamaChatClient) -> None:
        self.client = client
        self.responses: list[dict[str, object]] = []

    def complete_json(self, **kwargs):
        response = self.client.complete_json(**kwargs)
        self.responses.append(response)
        return response


def _fixture() -> RetrievalResponse:
    return RetrievalResponse(
        query_text="2025年销售额是多少，是否达到年度目标？",
        snapshot_build_id="ollama-answering-acceptance",
        hits=[
            RetrievalHit(
                rank=1,
                record_id="sales-table",
                document_id="annual-report",
                modality="table",
                raw_score=0.88,
                fusion_score=0.061,
                source_block_ids=["table-page-3"],
                content={
                    "markdown": "| 指标 | 2025实际 | 年度目标 |\n|---|---:|---:|\n| 销售额 | 100万元 | 120万元 |"
                },
                context="2025年度经营指标完成情况",
                provenance={"page": 3, "bbox": [0.1, 0.2, 0.9, 0.6]},
                source_path="C:/acceptance/年度报告.pdf",
                document_type="pdf",
                evidence_blocks=[
                    {
                        "record_id": "sales-table",
                        "modality": "table",
                        "role": "core",
                        "relation": "",
                        "content": {
                            "markdown": "| 指标 | 2025实际 | 年度目标 |\n|---|---:|---:|\n| 销售额 | 100万元 | 120万元 |"
                        },
                        "context": "2025年度经营指标完成情况",
                        "provenance": {"page": 3, "bbox": [0.1, 0.2, 0.9, 0.6]},
                        "source_block_ids": ["table-page-3"],
                    },
                    {
                        "record_id": "unrelated-note",
                        "modality": "text",
                        "role": "neighbor",
                        "relation": "next_context",
                        "content": {"text": "公司计划在2026年更新办公设备。"},
                        "context": "其他计划",
                        "provenance": {"page": 4},
                        "source_block_ids": ["text-page-4"],
                    },
                ],
            )
        ],
        searched_modalities=["text", "table", "visual"],
        elapsed_ms=5.0,
    )


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Run a real Ollama answering acceptance test")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    retrieval = _fixture()
    client = RecordingClient(OllamaChatClient(
        args.model,
        base_url=args.base_url,
        timeout_seconds=600.0,
    ))
    result = AnswerEngine(
        LLMEvidenceSelector(client),
        LLMAnswerGenerator(client),
    ).answer(AnswerRequest(retrieval.query_text, retrieval))
    payload = result.to_dict()
    payload["model_responses"] = client.responses
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if result.status is not AnswerStatus.ANSWERED:
        return 1
    if not result.claims or not result.citations:
        return 2
    valid_ids = {citation.evidence_id for citation in result.citations}
    if any(not set(claim.evidence_ids).issubset(valid_ids) for claim in result.claims):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

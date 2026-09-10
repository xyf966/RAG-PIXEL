from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hybrid_input.answering import (
    AnswerEngine,
    BailianChatClient,
    CitationValidator,
    LLMAnswerGenerator,
    RetrievalEvidenceSelector,
)
from hybrid_input.answering_contracts import EvidenceItem


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class BailianChatClientTests(unittest.TestCase):
    def test_retrieval_selector_does_not_keyword_veto_cross_language_evidence(self) -> None:
        def item(evidence_id: str, rank: int, text: str, source: str) -> EvidenceItem:
            return EvidenceItem(
                evidence_id=evidence_id,
                record_id=evidence_id,
                document_id=evidence_id,
                modality="text",
                role="core",
                content={"text": text},
                context="",
                provenance={"page": 1},
                source_path=source,
                document_type="pdf",
                source_block_ids=[evidence_id],
                retrieval_rank=rank,
                raw_score=0.8,
                fusion_score=0.03,
            )

        evidence = [
            item("E001", 1, "Do not expose the TV to rain or moisture.", "电视.pdf"),
            item("E002", 2, "To avoid condensate damage, mount within +-60°.", "sensor.pdf"),
            item("E003", 6, "如果系统卡纸，请清理并重新处理。", "打印机.pdf"),
        ]
        decisions = RetrievalEvidenceSelector().select(
            "如果安装环境可能产生冷凝水，电视机和压力传感器分别应如何处理？",
            evidence,
        )

        self.assertTrue(all(decision.relevant for decision in decisions))
        self.assertEqual(
            [decision.evidence_id for decision in decisions],
            ["E001", "E002", "E003"],
        )

    def test_generator_accepts_schema_envelope_returned_by_bailian_max(self) -> None:
        class _Client:
            def complete_json(self, **kwargs: object) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "answerable": True,
                        "claims": [{
                            "text": "三者共同范围为10°C至40°C",
                            "evidence_ids": ["E004", "E025", "E028"],
                        }],
                        "limitations": [],
                    },
                }

        draft = LLMAnswerGenerator(_Client()).generate("共同范围？", [], [])

        self.assertIs(draft.answerable, True)
        self.assertEqual(draft.claims[0].evidence_ids, ["E004", "E025", "E028"])

    def test_generator_accepts_string_boolean_from_bailian_json_object_mode(self) -> None:
        class _Client:
            def complete_json(self, **kwargs: object) -> dict:
                return {
                    "answerable": "true",
                    "claims": [{"text": "共同范围为10°C至40°C", "evidence_ids": ["E001"]}],
                    "limitations": [],
                }

        draft = LLMAnswerGenerator(_Client()).generate("共同范围？", [], [])

        self.assertIs(draft.answerable, True)
        self.assertEqual(draft.claims[0].evidence_ids, ["E001"])

    def test_bailian_engine_uses_local_citation_validator(self) -> None:
        engine = AnswerEngine.for_bailian("qwen-plus", api_key="secret-key")

        self.assertIs(type(engine.validator), CitationValidator)
        self.assertIs(type(engine.selector), RetrievalEvidenceSelector)

    def test_complete_json_uses_openai_compatible_endpoint_and_bearer_key(self) -> None:
        response = _Response({"choices": [{"message": {"content": '{"ok":true}'}}]})
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            client = BailianChatClient("qwen-plus", api_key="secret-key")
            result = client.complete_json(
                system_prompt="system",
                user_prompt="user",
                schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            )

        self.assertEqual(result, {"ok": True})
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-key")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen-plus")
        self.assertIs(body["enable_thinking"], False)
        self.assertEqual(body["max_tokens"], 4_096)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_visual_input_is_encoded_as_an_openai_image_data_url(self) -> None:
        response = _Response({"choices": [{"message": {"content": '{"ok":true}'}}]})
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "evidence.png"
            image.write_bytes(b"png-data")
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                client = BailianChatClient("qwen-vl-plus", api_key="secret-key")
                client.complete_json(
                    system_prompt="system",
                    user_prompt="user",
                    schema={"type": "object"},
                    image_paths=[str(image)],
                )

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        user_content = body["messages"][1]["content"]
        self.assertEqual(user_content[0], {"type": "text", "text": "user"})
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_list_models_reads_openai_data_shape(self) -> None:
        response = _Response({"data": [{"id": "qwen-plus"}, {"id": "qwen-max"}]})
        with patch("urllib.request.urlopen", return_value=response):
            client = BailianChatClient("qwen-plus", api_key="secret-key")
            self.assertEqual(client.list_models(), ["qwen-max", "qwen-plus"])


if __name__ == "__main__":
    unittest.main()

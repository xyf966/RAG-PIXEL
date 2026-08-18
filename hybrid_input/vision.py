from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import Artifact, VisionResult
from .interfaces import VisionProcessor


class DeferredVisionProcessor(VisionProcessor):
    """Explicit boundary used until a vision/index provider is selected."""

    name = "deferred"

    def process(self, artifacts: list[Artifact]) -> list[VisionResult]:
        return [
            VisionResult(
                block_id=item.block_id,
                provider=self.name,
                metadata={"status": "pending", "asset_path": item.asset_path},
            )
            for item in artifacts
            if item.kind == "image" and item.asset_path
        ]


DEFAULT_PIXELRAG_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def discover_pixelrag_model() -> str:
    """Prefer an explicitly configured or already-downloaded model."""
    configured = os.environ.get("HYBRID_PIXELRAG_MODEL")
    if configured:
        return str(Path(configured).expanduser().resolve())

    project_root = Path(__file__).resolve().parent.parent
    candidates = (
        project_root / "model-cache" / "Qwen3-VL-Embedding-2B",
        project_root / "PixelRAG-Studio-Data" / "models" / "Qwen3-VL-Embedding-2B",
    )
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return str(candidate)
    return DEFAULT_PIXELRAG_MODEL


class PixelRAGEmbeddingSession:
    """Lazy, process-local PixelRAG model session shared across documents."""

    def __init__(self, model: str, device: str = "auto") -> None:
        self.model_name = model
        self.requested_device = device
        self.resolved_device: str | None = None
        self.load_count = 0
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._lock = threading.RLock()

    def _resolve_device(self, torch: Any) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "PixelRAG embedding dependencies are unavailable; install pixelrag[embed]"
            ) from exc

        device = self._resolve_device(torch)
        dtype = torch.float32 if device == "cpu" else torch.float16
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).eval()
        if device != "cpu":
            model = model.to(device)

        self._torch = torch
        self._processor = processor
        self._model = model
        self.resolved_device = device
        self.load_count += 1

    def embed_items(
        self,
        items: list[dict[str, Any]],
        model_name: str,
        device: str = "auto",
        instruction: str = "",
    ) -> Any:
        if model_name != self.model_name or device != self.requested_device:
            raise ValueError("Embedding request does not match the resident PixelRAG session")
        with self._lock:
            self._ensure_loaded()
            import numpy as np
            from PIL import Image
            from pixelrag_embed.embed_cpu import _clamp_width

            assert self._model is not None
            assert self._processor is not None
            assert self._torch is not None
            assert self.resolved_device is not None
            dim = self._model.config.text_config.hidden_size
            embeddings = np.zeros((len(items), dim), dtype=np.float16)
            prefix = f"Instruct: {instruction}\n" if instruction else ""

            for index, item in enumerate(items):
                print(f"Pixel image embedding {index + 1}/{len(items)}", flush=True)
                with Image.open(item["path"]) as source:
                    image = _clamp_width(source.convert("RGB"))
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prefix + "What is shown in this image?"},
                        ],
                    }
                ]
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self._processor(
                    text=[text], images=[image], return_tensors="pt", padding=True
                )
                if self.resolved_device != "cpu":
                    inputs = {
                        key: value.to(self.resolved_device) if hasattr(value, "to") else value
                        for key, value in inputs.items()
                    }
                with self._torch.no_grad():
                    outputs = self._model(**inputs, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1]
                    last_index = inputs["attention_mask"].sum(dim=1) - 1
                    pooled = last_hidden[0, last_index[0]]
                    pooled = pooled / pooled.norm()
                    embeddings[index] = pooled.cpu().numpy().astype(np.float16)
            return embeddings

    def embed_texts(self, texts: list[str], instruction: str = "") -> Any:
        """Embed text queries in the same space as the resident image model."""
        with self._lock:
            self._ensure_loaded()
            assert self._model is not None
            assert self._processor is not None
            assert self._torch is not None
            assert self.resolved_device is not None

            messages = []
            for value in texts:
                conversation = []
                if instruction:
                    conversation.append(
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": instruction}],
                        }
                    )
                conversation.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": value}],
                    }
                )
                messages.append(conversation)
            rendered = [
                self._processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                for conversation in messages
            ]
            inputs = self._processor(text=rendered, return_tensors="pt", padding=True)
            if self.resolved_device != "cpu":
                inputs = {
                    key: value.to(self.resolved_device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            with self._torch.no_grad():
                outputs = self._model.model(**inputs)
            last_hidden = outputs.last_hidden_state
            last_indices = inputs["attention_mask"].sum(dim=1) - 1
            pooled = last_hidden[
                self._torch.arange(last_hidden.size(0), device=last_hidden.device),
                last_indices,
            ]
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=-1)
            return pooled.cpu().float().numpy()

    def close(self) -> None:
        """Release the resident model before process shutdown when desired."""
        import gc

        with self._lock:
            torch = self._torch
            device = self.resolved_device
            self._model = None
            self._processor = None
            self._torch = None
            self.resolved_device = None
            gc.collect()
            if torch is not None and device == "cuda":
                torch.cuda.empty_cache()


class PixelRAGVisionProcessor(VisionProcessor):
    """Embed normalized image artifacts with PixelRAG's official local backend."""

    name = "pixelrag"

    def __init__(
        self,
        model: str | None = None,
        device: str = "auto",
        instruction: str = "Represent this document image for visual retrieval.",
        embedder: Callable[..., Any] | None = None,
        session: PixelRAGEmbeddingSession | None = None,
    ) -> None:
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported PixelRAG device: {device}")
        self.model = model or discover_pixelrag_model()
        self.device = device
        self.instruction = instruction
        self._embedder = embedder
        self._session = session

    def _load_embedder(self) -> Callable[..., Any]:
        if self._embedder is not None:
            return self._embedder
        if self._session is None:
            self._session = PixelRAGEmbeddingSession(self.model, self.device)
        return self._session.embed_items

    @property
    def model_load_count(self) -> int:
        return self._session.load_count if self._session is not None else 0

    @property
    def resolved_device(self) -> str | None:
        return self._session.resolved_device if self._session is not None else None

    def close(self) -> None:
        if self._session is not None:
            self._session.close()

    def process(self, artifacts: list[Artifact]) -> list[VisionResult]:
        images = [
            item
            for item in artifacts
            if item.kind == "image" and item.asset_path and Path(item.asset_path).is_file()
        ]
        if not images:
            return []

        items = [{"path": item.asset_path} for item in images]
        embeddings = self._load_embedder()(
            items,
            self.model,
            device=self.device,
            instruction=self.instruction,
        )
        if len(embeddings) != len(images):
            raise RuntimeError(
                f"PixelRAG returned {len(embeddings)} embeddings for {len(images)} images"
            )

        results: list[VisionResult] = []
        for artifact, embedding in zip(images, embeddings, strict=True):
            vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            results.append(
                VisionResult(
                    block_id=artifact.block_id,
                    provider=self.name,
                    vector=[float(value) for value in vector],
                    metadata={
                        "status": "complete",
                        "asset_path": artifact.asset_path,
                        "model": self.model,
                        "device": self.resolved_device or self.device,
                        "model_load_count": self.model_load_count,
                        "dimension": len(vector),
                    },
                )
            )
        return results

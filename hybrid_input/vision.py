from __future__ import annotations

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

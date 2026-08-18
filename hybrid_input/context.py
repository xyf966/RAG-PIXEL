from __future__ import annotations

import re

from .contracts import Artifact
from .interfaces import ContextEnricher


def _same_container(visual: Artifact, candidate: Artifact) -> bool:
    left = visual.provenance
    right = candidate.provenance
    # Logical Office containers are more stable than rendered PDF page numbers:
    # Excel sheets may be paginated differently, while slides are already pages.
    if left.sheet is not None:
        return right.sheet == left.sheet
    if left.slide is not None:
        return right.slide == left.slide
    if left.page is not None:
        return right.page == left.page
    return True


def _context_text(artifact: Artifact, limit: int = 280) -> str:
    value = re.sub(r"\s+", " ", artifact.text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


class NearbyTextContextEnricher(ContextEnricher):
    name = "nearby-text"

    def __init__(self, max_order_distance: int = 2) -> None:
        self.max_order_distance = max_order_distance

    def enrich(self, artifacts: list[Artifact]) -> list[Artifact]:
        semantic = [
            item
            for item in artifacts
            if item.kind in {"text", "table"} and item.text and item.text.strip()
        ]
        for visual in artifacts:
            if visual.kind not in {"image", "visual_task", "visual"}:
                continue
            if visual.metadata.get("context_locked"):
                if visual.context and visual.context.strip():
                    visual.metadata.setdefault("context_source", "visual_object")
                continue
            if visual.context and visual.context.strip():
                visual.metadata.setdefault("context_source", "parser_caption")
                continue
            candidates = [
                item
                for item in semantic
                if item.block_id != visual.block_id
                and _same_container(visual, item)
                and abs(item.reading_order - visual.reading_order) <= self.max_order_distance
            ]
            before = [item for item in candidates if item.reading_order < visual.reading_order]
            after = [item for item in candidates if item.reading_order > visual.reading_order]
            selected: list[Artifact] = []
            if before:
                selected.append(max(before, key=lambda item: item.reading_order))
            if after:
                selected.append(min(after, key=lambda item: item.reading_order))
            texts = [_context_text(item) for item in selected]
            texts = [text for text in texts if text]
            if texts:
                visual.context = "\n".join(texts)
                visual.metadata["context_source"] = "nearby_artifacts"
                visual.metadata["context_block_ids"] = [item.block_id for item in selected]
        return artifacts

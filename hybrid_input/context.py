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


def _spatial_text_candidates(visual: Artifact, semantic: list[Artifact]) -> list[Artifact]:
    box = visual.provenance.bbox
    if visual.provenance.slide is None or not isinstance(box, list) or len(box) != 4:
        return []
    selected: list[Artifact] = []
    for candidate in semantic:
        if candidate.provenance.slide != visual.provenance.slide:
            continue
        candidate_box = candidate.provenance.bbox
        if not isinstance(candidate_box, list) or len(candidate_box) != 4:
            continue
        if (
            visual.metadata.get("office_geometry_match_iou") is not None
            and abs(candidate.reading_order - visual.reading_order) > 8
        ):
            continue
        center_x = (candidate_box[0] + candidate_box[2]) / 2.0
        center_y = (candidate_box[1] + candidate_box[3]) / 2.0
        center_inside = (
            box[0] - 0.01 <= center_x <= box[2] + 0.01
            and box[1] - 0.01 <= center_y <= box[3] + 0.01
        )
        horizontal_overlap = max(0.0, min(box[2], candidate_box[2]) - max(box[0], candidate_box[0]))
        candidate_width = max(0.001, candidate_box[2] - candidate_box[0])
        vertical_gap = max(
            0.0,
            candidate_box[1] - box[3],
            box[1] - candidate_box[3],
        )
        vertically_adjacent = horizontal_overlap / candidate_width >= 0.30 and vertical_gap <= 0.08
        if center_inside or vertically_adjacent:
            selected.append(candidate)
    return sorted(
        selected,
        key=lambda item: (
            item.provenance.bbox[1] if item.provenance.bbox else 1.0,
            item.provenance.bbox[0] if item.provenance.bbox else 1.0,
            item.reading_order,
        ),
    )


def _set_context(visual: Artifact, selected: list[Artifact], source: str) -> bool:
    texts: list[str] = []
    remaining = 280
    for item in selected:
        value = _context_text(item, remaining)
        if not value:
            continue
        texts.append(value)
        remaining -= len(value) + 1
        if remaining <= 0:
            break
    if not texts:
        return False
    visual.context = "\n".join(texts)
    visual.metadata["context_source"] = source
    visual.metadata["context_block_ids"] = [item.block_id for item in selected[: len(texts)]]
    return True


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
        visuals = [
            item for item in artifacts
            if item.kind in {"image", "visual_task", "visual"}
        ]
        for visual in visuals:
            if visual.metadata.get("context_locked"):
                if visual.context and visual.context.strip():
                    visual.metadata.setdefault("context_source", "visual_object")
                continue
            if visual.context and visual.context.strip():
                visual.metadata.setdefault("context_source", "parser_caption")
                continue
            if visual.metadata.get("child_visual_ids"):
                continue
            spatial = _spatial_text_candidates(visual, semantic)
            if spatial and _set_context(visual, spatial, "spatial_artifacts"):
                continue
            if (
                visual.provenance.slide is not None
                and isinstance(visual.provenance.bbox, list)
            ):
                continue
            # A child visual inherits the text inside its enclosing PowerPoint
            # group after group contexts have been collected.  Reading-order
            # fallback here would attach unrelated neighboring labels.
            if visual.metadata.get("parent_visual_id"):
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
            _set_context(visual, selected, "nearby_artifacts")

        visual_by_id = {item.block_id: item for item in visuals}
        for parent in visuals:
            if parent.context and parent.context.strip():
                continue
            child_ids = parent.metadata.get("child_visual_ids")
            if not isinstance(child_ids, list):
                continue
            children = [visual_by_id.get(child_id) for child_id in child_ids]
            child_contexts = list(dict.fromkeys(
                child.context.strip()
                for child in children
                if child is not None and child.context and child.context.strip()
            ))
            if not child_contexts:
                continue
            parent.context = "\n".join(child_contexts)[:280]
            parent.metadata["context_source"] = "child_visuals"
            parent.metadata["context_block_ids"] = [
                block_id
                for child in children if child is not None
                for block_id in child.metadata.get("context_block_ids") or []
            ]
        for visual in visuals:
            if visual.context and visual.context.strip():
                continue
            parent_id = visual.metadata.get("parent_visual_id")
            parent = visual_by_id.get(parent_id) if isinstance(parent_id, str) else None
            if parent is None or not parent.context or not parent.context.strip():
                continue
            visual.context = parent.context
            visual.metadata["context_source"] = "parent_visual"
            visual.metadata["context_block_ids"] = list(
                parent.metadata.get("context_block_ids") or []
            )
        return artifacts

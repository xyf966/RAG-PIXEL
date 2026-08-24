from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .contracts import Artifact, HybridDocument


@dataclass(slots=True)
class IndexBuildConfig:
    text_target_tokens: int = 512
    text_max_tokens: int = 768
    text_overlap_tokens: int = 96
    text_batch_size: int = 8
    table_max_rows: int = 20
    table_max_columns: int = 12
    expected_dimension: int = 2048
    office_visual_policy: str = "require-com"

    def __post_init__(self) -> None:
        if not 0 <= self.text_overlap_tokens < self.text_target_tokens <= self.text_max_tokens:
            raise ValueError("Require 0 <= overlap < target <= maximum text tokens")
        if self.text_batch_size < 1 or self.table_max_rows < 1 or self.table_max_columns < 2:
            raise ValueError("Batch size and table limits must be positive")
        if self.office_visual_policy not in {
            "require-com", "require-native", "allow-page-fallback"
        }:
            raise ValueError("Unsupported Office visual policy")


@dataclass(slots=True)
class IndexRecord:
    record_id: str
    document_id: str
    modality: str
    source_block_ids: list[str]
    embedding_text: str
    original_content: Any
    context: str
    structure: dict[str, Any]
    provenance: dict[str, Any]
    content_hash: str
    overlap_source_block_ids: list[str] = field(default_factory=list)
    asset_path: str | None = None
    visual_type: str | None = None
    source_path: str = ""
    document_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TokenCodec(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


class CharacterTokenCodec:
    """Deterministic fallback used by tests and model-free callers."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(value) for value in token_ids)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _content_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _record_id(
    document_id: str,
    modality: str,
    source_block_ids: list[str],
    ordinal: int,
    content_hash: str,
) -> str:
    identity = "\0".join(
        [document_id, modality, *source_block_ids, str(ordinal), content_hash]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _provenance_dict(artifact: Artifact) -> dict[str, Any]:
    return asdict(artifact.provenance)


def _container_key(artifact: Artifact) -> tuple[Any, ...]:
    provenance = artifact.provenance
    if provenance.sheet is not None:
        return ("sheet", provenance.sheet)
    if provenance.slide is not None:
        return ("slide", provenance.slide)
    if provenance.page is not None:
        return ("page", provenance.page)
    return ("document",)


def _is_heading(artifact: Artifact) -> bool:
    text = _normalized_text(artifact.text or "")
    level = artifact.metadata.get("level")
    label = str(artifact.metadata.get("label") or artifact.metadata.get("type") or "").lower()
    if label in {"title", "heading", "section_header", "section-header"}:
        return True
    if isinstance(level, int) and level <= 2 and len(text) <= 160:
        return True
    return bool(
        len(text) <= 120
        and re.match(r"^(?:第[一二三四五六七八九十百]+[章节]|\d+(?:\.\d+)*[.、]\s*)", text)
    )


def _heading_level(artifact: Artifact) -> int:
    level = artifact.metadata.get("level")
    return max(int(level), 1) if isinstance(level, int) else 2


def _split_long_text(value: str, codec: TokenCodec, maximum: int, overlap: int) -> list[str]:
    if len(codec.encode(value)) <= maximum:
        return [value]
    sentences = [part for part in re.split(r"(?<=[。！？!?；;\.])\s*|\n+", value) if part]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current}\n{sentence}"
        if len(codec.encode(candidate)) <= maximum:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        ids = codec.encode(sentence)
        if len(ids) <= maximum:
            current = sentence
            continue
        step = maximum - overlap
        pieces.extend(codec.decode(ids[start : start + maximum]) for start in range(0, len(ids), step))
    if current:
        pieces.append(current)
    return pieces


def _bounded_embedding_text(
    titles: list[str],
    overlap_text: str,
    body: str,
    codec: TokenCodec,
    maximum: int,
) -> str:
    effective_titles = list(titles)
    first_body_line = body.splitlines()[0].strip() if body else ""
    if effective_titles and first_body_line == effective_titles[-1]:
        effective_titles.pop()
    body_ids = codec.encode(body)[:maximum]
    remaining = maximum - len(body_ids)
    prefix = "\n".join([*effective_titles, overlap_text]).strip()
    prefix_ids = codec.encode(prefix)
    if prefix_ids and body_ids:
        remaining = max(remaining - len(codec.encode("\n")), 0)
    if len(prefix_ids) > remaining:
        prefix_ids = prefix_ids[:remaining]
    prefix = codec.decode(prefix_ids).strip()
    return "\n".join(part for part in (prefix, codec.decode(body_ids)) if part).strip()


def build_text_records(
    document: HybridDocument,
    codec: TokenCodec,
    config: IndexBuildConfig,
) -> list[IndexRecord]:
    artifacts = [
        item for item in document.artifacts
        if item.kind == "text" and _normalized_text(item.text or "")
    ]
    artifacts.sort(key=lambda item: item.reading_order)
    title_path: list[str] = []
    units: list[dict[str, Any]] = []
    for artifact in artifacts:
        value = _normalized_text(artifact.text or "")
        if _is_heading(artifact):
            level = _heading_level(artifact)
            title_path = title_path[: level - 1]
            title_path.append(value)
        for piece in _split_long_text(
            value, codec, config.text_max_tokens, config.text_overlap_tokens
        ):
            units.append(
                {
                    "text": piece,
                    "artifact": artifact,
                    "container": _container_key(artifact),
                    "titles": list(title_path),
                }
            )

    records: list[IndexRecord] = []
    ordinal = 0
    current: list[dict[str, Any]] = []
    overlap_text = ""
    overlap_ids: list[str] = []

    def flush() -> None:
        nonlocal ordinal, current, overlap_text, overlap_ids
        if not current:
            return
        titles = current[-1]["titles"]
        body = "\n".join(item["text"] for item in current)
        embedding_text = _bounded_embedding_text(
            titles, overlap_text, body, codec, config.text_max_tokens
        )
        source_ids = list(dict.fromkeys(item["artifact"].block_id for item in current))
        source_provenance = []
        seen_provenance_ids: set[str] = set()
        for item in current:
            artifact = item["artifact"]
            if artifact.block_id in seen_provenance_ids:
                continue
            seen_provenance_ids.add(artifact.block_id)
            source_provenance.append(
                {"block_id": artifact.block_id, **_provenance_dict(artifact)}
            )
        first = current[0]["artifact"]
        original = {"text": body, "title_path": titles, "overlap_text": overlap_text}
        digest = _content_hash(original)
        records.append(
            IndexRecord(
                record_id=_record_id(document.document_id, "text", source_ids, ordinal, digest),
                document_id=document.document_id,
                modality="text",
                source_block_ids=source_ids,
                embedding_text=embedding_text,
                original_content=original,
                context=" / ".join(titles),
                structure={
                    "title_path": titles,
                    "source_provenance": source_provenance,
                },
                provenance=_provenance_dict(first),
                content_hash=digest,
                overlap_source_block_ids=list(overlap_ids),
                source_path=document.source_path,
                document_type=document.document_type,
            )
        )
        ordinal += 1
        combined_ids = codec.encode(body)
        overlap_text = codec.decode(combined_ids[-config.text_overlap_tokens :])
        overlap_ids = source_ids
        current = []

    for unit in units:
        if current and (
            unit["container"] != current[-1]["container"]
            or unit["titles"] != current[-1]["titles"]
        ):
            flush()
            overlap_text = ""
            overlap_ids = []
        candidate = "\n".join(
            [*unit["titles"], overlap_text, *(item["text"] for item in current), unit["text"]]
        ).strip()
        if current and len(codec.encode(candidate)) > config.text_target_tokens:
            flush()
        current.append(unit)
        hard_candidate = "\n".join(
            [*unit["titles"], overlap_text, *(item["text"] for item in current)]
        ).strip()
        if len(codec.encode(hard_candidate)) >= config.text_max_tokens:
            flush()
    flush()
    return records


def _markdown_table(value: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    parsed: list[list[str]] = []
    for line in lines:
        if "|" not in line:
            continue
        cells = [
            cell.strip().replace(r"\|", "|")
            for cell in re.split(r"(?<!\\)\|", line.strip("|"))
        ]
        if cells:
            parsed.append(cells)
    if len(parsed) < 2:
        return None
    separator = all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in parsed[1])
    header = parsed[0]
    rows = parsed[2:] if separator else parsed[1:]
    width = max([len(header), *(len(row) for row in rows)], default=len(header))
    header += [f"column_{index + 1}" for index in range(len(header), width)]
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    return header, normalized_rows


def _matrix_from_native_cells(metadata: dict[str, Any]) -> list[list[str]] | None:
    cells = metadata.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    row_values = [int(cell.get("row") or 0) for cell in cells]
    column_values = [int(cell.get("column") or 0) for cell in cells]
    min_row, min_column = min(row_values), min(column_values)
    height = max(int(metadata.get("rows") or 0), max(row_values) - min_row + 1)
    width = max(int(metadata.get("columns") or 0), max(column_values) - min_column + 1)
    matrix = [[""] * width for _ in range(height)]
    for cell in cells:
        row = int(cell.get("row") or min_row) - min_row
        column = int(cell.get("column") or min_column) - min_column
        value = str(cell.get("value") if cell.get("value") is not None else "")
        row_span = max(int(cell.get("row_span") or 1), 1)
        column_span = max(int(cell.get("column_span") or 1), 1)
        for row_offset in range(row_span):
            for column_offset in range(column_span):
                target_row, target_column = row + row_offset, column + column_offset
                if target_row < height and target_column < width:
                    matrix[target_row][target_column] = value
    return matrix


def _matrix_from_table_cells(metadata: dict[str, Any]) -> list[list[str]] | None:
    table_rows = metadata.get("table_cells")
    if not isinstance(table_rows, list) or not table_rows:
        return None
    all_cells = [cell for row in table_rows for cell in row.get("cells", [])]
    if not all_cells:
        return None
    width = max(
        int(metadata.get("columns") or 0),
        *(
            int(cell.get("column") or 1) + int(cell.get("column_span") or 1) - 1
            for cell in all_cells
        ),
    )
    row_numbers = [int(row.get("row") or index + 1) for index, row in enumerate(table_rows)]
    min_row = min(row_numbers)
    height = max(
        int(cell.get("row") or min_row) + int(cell.get("row_span") or 1) - min_row
        for cell in all_cells
    )
    matrix = [[""] * width for _ in range(height)]
    for row in table_rows:
        for cell in row.get("cells", []):
            row_index = max(int(cell.get("row") or row.get("row") or min_row) - min_row, 0)
            column = max(int(cell.get("column") or 1) - 1, 0)
            row_span = max(int(cell.get("row_span") or 1), 1)
            column_span = max(int(cell.get("column_span") or 1), 1)
            value = str(cell.get("text") or "")
            for row_offset in range(row_span):
                for column_offset in range(column_span):
                    target_row, target_column = row_index + row_offset, column + column_offset
                    if target_row < height and target_column < width:
                        matrix[target_row][target_column] = value
    return matrix


def _looks_like_title_row(row: list[str], width: int, metadata: dict[str, Any]) -> bool:
    nonempty = [value for value in row if value.strip()]
    if not nonempty:
        return False
    table_rows = metadata.get("table_cells") or []
    for table_row in table_rows:
        cells = table_row.get("cells") or []
        populated = [str(cell.get("text") or "").strip() for cell in cells]
        populated = [value for value in populated if value]
        if not populated:
            continue
        if (
            len(cells) == 1
            and int(cells[0].get("column_span") or 1) >= width
            and len(set(nonempty)) == 1
            and nonempty[0] == populated[0]
        ):
            return True
        break
    return len(nonempty) == 1 and width >= 2


def _normalize_table(artifact: Artifact) -> dict[str, Any] | None:
    raw = artifact.text or ""
    metadata = artifact.metadata
    matrix = _matrix_from_native_cells(metadata)
    origin = "native-cells" if matrix is not None else ""
    if matrix is None:
        matrix = _matrix_from_table_cells(metadata)
        origin = "structured-table-cells" if matrix is not None else ""
    if matrix is None:
        parsed = _markdown_table(raw)
        if parsed is None:
            return None
        headers, rows = parsed
        matrix = [headers, *rows]
        origin = "markdown"
    if not matrix or not matrix[0]:
        return None

    width = max(len(row) for row in matrix)
    matrix = [row + [""] * (width - len(row)) for row in matrix]
    leading_empty_rows = 0
    while matrix and not any(value.strip() for value in matrix[0]):
        matrix.pop(0)
        leading_empty_rows += 1
    while matrix and not any(value.strip() for value in matrix[-1]):
        matrix.pop()
    if not matrix:
        return None
    detected_title = ""
    if _looks_like_title_row(matrix[0], width, metadata):
        detected_title = next(value for value in matrix.pop(0) if value.strip())
    if not matrix:
        return None

    is_native_key_value = (
        origin == "native-cells"
        and width == 2
        and all(row[0].strip() for row in matrix)
        and len({row[0] for row in matrix}) == len(matrix)
    )
    if is_native_key_value:
        headers = ["Field", "Value"]
        rows = matrix
        header_source_row = None
    else:
        headers = matrix[0]
        rows = matrix[1:]
        header_source_row = leading_empty_rows + 1 + bool(detected_title)

    formulas = [
        {"cell": cell.get("cell"), "formula": cell.get("formula")}
        for cell in metadata.get("cells") or []
        if cell.get("formula")
    ]
    return {
        "format": "table",
        "origin": origin,
        "detected_title": detected_title,
        "headers": headers,
        "rows": rows,
        "header_source_row": header_source_row,
        "native_cells": metadata.get("cells") or [],
        "table_cells": metadata.get("table_cells") or [],
        "merged_ranges": metadata.get("merged_ranges") or [],
        "formulas": formulas,
    }


def _table_embedding_text(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    location: str,
) -> str:
    lines = []
    if title:
        lines.append(f"Table: {title}")
    lines.append("Columns: " + " | ".join(headers))
    lines.extend("Row: " + " | ".join(row) for row in rows)
    if location:
        lines.append(f"Source: {location}")
    return "\n".join(lines)


def _table_embedding_pieces(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    location: str,
    codec: TokenCodec,
    config: IndexBuildConfig,
) -> list[str]:
    value = _table_embedding_text(title, headers, rows, location)
    if len(codec.encode(value)) <= config.text_max_tokens:
        return [value]
    prefix = _table_embedding_text(title, headers, [], location)
    separator_tokens = len(codec.encode("\n"))
    available = config.text_max_tokens - len(codec.encode(prefix)) - separator_tokens
    if available < 2:
        prefix = codec.decode(codec.encode(prefix)[: config.text_max_tokens // 2])
        available = config.text_max_tokens - len(codec.encode(prefix)) - separator_tokens
    overlap = min(config.text_overlap_tokens, max(available - 1, 0))
    body = "\n".join("Row: " + " | ".join(row) for row in rows)
    body_pieces = _split_long_text(body, codec, available, overlap)
    return [
        "\n".join((prefix, codec.decode(codec.encode(piece)[:available]))).strip()
        for piece in body_pieces
    ]


def build_table_records(
    document: HybridDocument,
    codec: TokenCodec,
    config: IndexBuildConfig,
) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    ordinal = 0
    artifacts = [item for item in document.artifacts if item.kind == "table"]
    artifacts.sort(key=lambda item: item.reading_order)
    for artifact in artifacts:
        raw = artifact.text or json.dumps(artifact.metadata, ensure_ascii=False, sort_keys=True)
        normalized = _normalize_table(artifact)
        provenance = _provenance_dict(artifact)
        location = str(
            artifact.provenance.cell_range
            or artifact.provenance.sheet
            or artifact.provenance.page
            or artifact.provenance.slide
            or ""
        )
        detected_title = normalized.get("detected_title", "") if normalized else ""
        title = _normalized_text(
            artifact.context or str(artifact.metadata.get("title") or detected_title)
        )
        if normalized is None:
            raw_embedding = _table_embedding_text(
                title,
                ["raw_content"],
                [[_normalized_text(raw)]],
                location,
            )
            pieces = _split_long_text(
                raw_embedding, codec, config.text_max_tokens, config.text_overlap_tokens
            )
            for piece in pieces:
                structure = {"format": "raw", "metadata": artifact.metadata}
                digest = _content_hash({"text": piece, "structure": structure})
                records.append(
                    IndexRecord(
                        _record_id(document.document_id, "table", [artifact.block_id], ordinal, digest),
                        document.document_id,
                        "table",
                        [artifact.block_id],
                        piece,
                        raw,
                        title,
                        structure,
                        provenance,
                        digest,
                        source_path=document.source_path,
                        document_type=document.document_type,
                    )
                )
                ordinal += 1
            continue

        headers = normalized["headers"]
        rows = normalized["rows"]
        if len(headers) <= config.table_max_columns:
            column_groups = [list(range(len(headers)))]
        else:
            payload_width = config.table_max_columns - 1
            column_groups = [
                [0, *range(start, min(start + payload_width, len(headers)))]
                for start in range(1, len(headers), payload_width)
            ]

        fitted_groups: list[list[int]] = []
        pending_groups = list(column_groups)
        while pending_groups:
            columns = pending_groups.pop(0)
            selected_headers = [headers[index] for index in columns]
            oversized = any(
                len(
                    codec.encode(
                        _table_embedding_text(
                            title,
                            selected_headers,
                            [[row[index] for index in columns]],
                            location,
                        )
                    )
                )
                > config.text_max_tokens
                for row in rows
            )
            if oversized and len(columns) > 2:
                payload = columns[1:]
                middle = max(len(payload) // 2, 1)
                pending_groups[:0] = [[columns[0], *payload[:middle]], [columns[0], *payload[middle:]]]
            else:
                fitted_groups.append(columns)

        for columns in fitted_groups:
            selected_headers = [headers[index] for index in columns]
            selected_rows = [[row[index] for index in columns] for row in rows]
            row_batches: list[tuple[int, list[list[str]]]] = []
            current_rows: list[list[str]] = []
            current_start = 0
            for row_index, row in enumerate(selected_rows or [[]]):
                candidate_rows = [*current_rows, row]
                candidate = _table_embedding_text(title, selected_headers, candidate_rows, location)
                if current_rows and (
                    len(current_rows) >= config.table_max_rows
                    or len(codec.encode(candidate)) > config.text_max_tokens
                ):
                    row_batches.append((current_start, current_rows))
                    current_rows = []
                    current_start = row_index
                current_rows.append(row)
            if current_rows:
                row_batches.append((current_start, current_rows))

            for row_start, row_batch in row_batches:
                embedding_pieces = _table_embedding_pieces(
                    title, selected_headers, row_batch, location, codec, config
                )
                for segment_index, embedding_text in enumerate(embedding_pieces):
                    structure = {
                        **normalized,
                        "headers": selected_headers,
                        "rows": row_batch,
                        "column_indices": columns,
                        "row_start": row_start,
                        "row_end": row_start + len(row_batch) - 1,
                        "segment_index": segment_index,
                        "segment_count": len(embedding_pieces),
                        "metadata": artifact.metadata,
                    }
                    original = {"raw": raw, "structure": structure}
                    digest = _content_hash({"original": original, "embedding_text": embedding_text})
                    records.append(
                        IndexRecord(
                            _record_id(document.document_id, "table", [artifact.block_id], ordinal, digest),
                            document.document_id,
                            "table",
                            [artifact.block_id],
                            embedding_text,
                            original,
                            title,
                            structure,
                            provenance,
                            digest,
                            source_path=document.source_path,
                            document_type=document.document_type,
                        )
                    )
                    ordinal += 1
    return records


def build_visual_record(
    document: HybridDocument,
    artifact: Artifact,
    ordinal: int,
) -> IndexRecord:
    content = {
        "sha256": artifact.sha256,
        "context": artifact.context or "",
        "visual_type": artifact.visual_type,
    }
    digest = _content_hash(content)
    return IndexRecord(
        record_id=_record_id(document.document_id, "visual", [artifact.block_id], ordinal, digest),
        document_id=document.document_id,
        modality="visual",
        source_block_ids=[artifact.block_id],
        embedding_text=artifact.context or "",
        original_content=content,
        context=artifact.context or "",
        structure={"metadata": artifact.metadata},
        provenance=_provenance_dict(artifact),
        content_hash=digest,
        asset_path=artifact.asset_path,
        visual_type=artifact.visual_type,
        source_path=document.source_path,
        document_type=document.document_type,
    )

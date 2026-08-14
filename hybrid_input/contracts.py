from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class Provenance:
    source_file: str
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    paragraph: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    bbox: list[float] | None = None
    bbox_original: list[float] | None = None
    coord_origin: str = "TOPLEFT"
    locator: str | None = None


@dataclass(slots=True)
class Artifact:
    block_id: str
    kind: str
    provenance: Provenance
    text: str | None = None
    asset_path: str | None = None
    context: str | None = None
    reading_order: int = 0
    parent_block_id: str | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactBundle:
    document_id: str
    source_path: str
    document_type: str
    parser: str
    artifacts: list[Artifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactBundle":
        artifacts = []
        for raw in payload.get("artifacts", []):
            item = dict(raw)
            item["provenance"] = Provenance(**item["provenance"])
            artifacts.append(Artifact(**item))
        return cls(
            document_id=payload["document_id"],
            source_path=payload["source_path"],
            document_type=payload["document_type"],
            parser=payload["parser"],
            artifacts=artifacts,
            warnings=list(payload.get("warnings", [])),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(slots=True)
class VisionResult:
    block_id: str
    provider: str
    vector: list[float] | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HybridDocument:
    document_id: str
    source_path: str
    document_type: str
    artifacts: list[Artifact]
    vision_results: list[VisionResult] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

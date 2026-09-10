from __future__ import annotations

from .contracts import Artifact, ArtifactBundle, HybridDocument, VisionResult
from .interfaces import DocumentAssembler


class DefaultAssembler(DocumentAssembler):
    name = "default"

    def assemble(self, bundle: ArtifactBundle, artifacts: list[Artifact], vision_results: list[VisionResult], providers: dict[str, str]) -> HybridDocument:
        valid_ids = {item.block_id for item in artifacts}
        dangling = [result.block_id for result in vision_results if result.block_id not in valid_ids]
        warnings = list(bundle.warnings)
        if dangling:
            warnings.append(f"Vision results reference missing blocks: {', '.join(dangling)}")
        return HybridDocument(
            document_id=bundle.document_id,
            source_path=bundle.source_path,
            document_type=bundle.document_type,
            artifacts=artifacts,
            vision_results=vision_results,
            providers=providers,
            warnings=warnings,
        )

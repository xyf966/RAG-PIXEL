from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_input.contracts import Artifact, HybridDocument, Provenance
from hybrid_input.index_records import IndexBuildConfig
from hybrid_input.indexing import build_index_snapshot, current_snapshot
from hybrid_input.pipeline import PipelineConfig, build_default_pipeline, discover_docling_python
from hybrid_input.renderers import (
    MicrosoftOfficeSubprocessRenderer,
    MicrosoftOfficeVisualSubprocessRenderer,
)
from hybrid_input.vision import PixelRAGEmbeddingSession


def load_document(
    path: Path,
    source: Path,
    preserve_visual_assets: bool = False,
) -> HybridDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts: list[Artifact] = []
    for raw in payload["artifacts"]:
        values = dict(raw)
        provenance = dict(values.pop("provenance"))
        provenance["source_file"] = str(source.resolve())
        if values.get("kind") == "visual" and not preserve_visual_assets:
            # Force PDF region materialization from the supplied acceptance
            # source instead of reusing assets from an older recognition run.
            values["asset_path"] = None
            values["sha256"] = None
        artifacts.append(Artifact(provenance=Provenance(**provenance), **values))
    return HybridDocument(
        document_id=payload["document_id"],
        source_path=str(source.resolve()),
        document_type=payload["document_type"],
        artifacts=artifacts,
        vision_results=[],
        providers=dict(payload.get("providers", {})),
        warnings=list(payload.get("warnings", [])),
        schema_version=payload.get("schema_version", "1.1"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one real schema 2.0 acceptance snapshot")
    parser.add_argument("--document", type=Path, action="append", default=[])
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--ingest-source", type=Path, action="append", default=[])
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--preserve-visual-assets", action="store_true")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.ingest_source:
        if args.document or args.source or args.artifacts_dir is None:
            parser.error("--ingest-source requires --artifacts-dir and cannot use --document/--source")
        pipeline = build_default_pipeline(PipelineConfig(vision_processor="deferred"))
        documents = []
        for ordinal, source in enumerate(args.ingest_source, 1):
            print(f"Recognition {ordinal}/{len(args.ingest_source)}: {source.name}", flush=True)
            documents.append(pipeline.ingest(source, args.artifacts_dir))
        source_files = args.ingest_source
    else:
        if not args.document or len(args.document) != len(args.source):
            parser.error("snapshot mode requires matching --document and --source pairs")
        documents = [
            load_document(document, source, args.preserve_visual_assets)
            for document, source in zip(args.document, args.source)
        ]
        source_files = args.source

    worker_python = discover_docling_python()
    session = PixelRAGEmbeddingSession(str(args.model.resolve()), args.device)
    try:
        manifest = build_index_snapshot(
            documents=documents,
            source_files=source_files,
            index_dir=args.index_dir,
            session=session,
            config=IndexBuildConfig(),
            office_renderer=MicrosoftOfficeVisualSubprocessRenderer(worker_python),
            office_pdf_renderer=MicrosoftOfficeSubprocessRenderer(worker_python),
        )
        snapshot = current_snapshot(args.index_dir)
        print(
            json.dumps(
                {
                    "snapshot": str(snapshot.resolve()),
                    "build_id": manifest["build_id"],
                    "counts": manifest["counts"],
                    "warnings": manifest["warnings"],
                    "model_load_count": session.load_count,
                    "vector_dimension": manifest["vector_dimension"],
                    "status": manifest["status"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()

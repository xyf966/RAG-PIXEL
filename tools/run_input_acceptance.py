from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_input.pipeline import build_default_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    pipeline = build_default_pipeline()
    for source in args.inputs:
        result = pipeline.ingest(source, args.output)
        visuals = [item for item in result.artifacts if item.kind == "image"]
        tables = [item for item in result.artifacts if item.kind == "table"]
        print(
            f"OK {source.name}: blocks={len(result.artifacts)} "
            f"tables={len(tables)} images={len(visuals)} warnings={len(result.warnings)}"
        )
        for table in tables:
            print(
                "  TABLE "
                f"{table.block_id} origin={table.metadata.get('structure_origin')} "
                f"spans={table.metadata.get('span_reliability')}"
            )
        for visual in visuals:
            provenance = visual.provenance
            asset_exists = bool(visual.asset_path and Path(visual.asset_path).is_file())
            print(
                "  IMAGE "
                f"{visual.block_id} asset={asset_exists} page={provenance.page} "
                f"slide={provenance.slide} sheet={provenance.sheet!r} "
                f"cell_range={provenance.cell_range!r} bbox={bool(provenance.bbox)} "
                f"context={bool(visual.context)}"
            )


if __name__ == "__main__":
    main()

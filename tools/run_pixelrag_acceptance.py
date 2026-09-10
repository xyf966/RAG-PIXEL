from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_input.contracts import ArtifactBundle
from hybrid_input.vision import PixelRAGVisionProcessor


def _load(path: Path) -> tuple[dict[str, Any], ArtifactBundle]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bundle = ArtifactBundle.from_dict(
        {
            "document_id": payload["document_id"],
            "source_path": payload["source_path"],
            "document_type": payload["document_type"],
            "parser": payload.get("providers", {}).get("parser", "unknown"),
            "artifacts": payload.get("artifacts", []),
            "warnings": payload.get("warnings", []),
            "schema_version": payload.get("schema_version", "1.0"),
        }
    )
    return payload, bundle


def _write_report(
    output: Path,
    rows: list[dict[str, Any]],
    model: str,
    device: str,
) -> None:
    total = lambda key: sum(int(row[key]) for row in rows)
    lines = [
        "# PixelRAG 混合文档输入验收",
        "",
        f"- 模型：`{model}`",
        f"- 设备：`{device}`",
        f"- 文档数：{len(rows)}",
        f"- 图片块：{total('images')}，进入 PixelRAG：{total('vision_results')}",
        f"- 文字块：{total('texts')}，进入 PixelRAG：0",
        f"- 表格块：{total('tables')}，进入 PixelRAG：0",
        f"- 非图片误入数：{total('non_image_leaks')}",
        "",
        "| 文档 | 类型 | 文字 | 表格 | 图片 | Pixel 结果 | 非图片误入 | 向量维度 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['document_type']} | {row['texts']} | "
            f"{row['tables']} | {row['images']} | {row['vision_results']} | "
            f"{row['non_image_leaks']} | {row['dimension']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## 验收结论",
            "",
            "通过条件：每个有效图片块恰好对应一个 PixelRAG 结果；文字块和表格块均无 PixelRAG 结果；"
            "所有结果 provider 为 `pixelrag`、status 为 `complete`，且向量维度一致。",
            "",
            "**PASS**" if all(row["status"] == "PASS" for row in rows) else "**FAIL**",
            "",
        ]
    )
    (output / "pixelrag-acceptance-report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PixelRAG image-only routing on mixed documents")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="HybridDocument JSON manifests to validate",
    )
    args = parser.parse_args()

    loaded = [(path, *_load(path)) for path in args.manifests]
    all_artifacts = [artifact for _, _, bundle in loaded for artifact in bundle.artifacts]
    processor = PixelRAGVisionProcessor(model=args.model, device=args.device)
    results = processor.process(all_artifacts)
    result_by_block = {result.block_id: result for result in results}
    if len(result_by_block) != len(results):
        raise RuntimeError("Duplicate block_id values in PixelRAG results")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source_manifest, payload, bundle in loaded:
        image_ids = {
            artifact.block_id
            for artifact in bundle.artifacts
            if artifact.kind == "image" and artifact.asset_path and Path(artifact.asset_path).is_file()
        }
        non_image_ids = {artifact.block_id for artifact in bundle.artifacts if artifact.kind != "image"}
        document_results = [result_by_block[block_id] for block_id in image_ids if block_id in result_by_block]
        leaks = non_image_ids.intersection(result_by_block)
        dimensions = {len(result.vector or []) for result in document_results}
        complete = all(
            result.provider == "pixelrag" and result.metadata.get("status") == "complete"
            for result in document_results
        )
        passed = len(document_results) == len(image_ids) and not leaks and len(dimensions) == 1 and complete
        row = {
            "name": Path(payload["source_path"]).name,
            "document_type": payload["document_type"],
            "texts": sum(artifact.kind == "text" for artifact in bundle.artifacts),
            "tables": sum(artifact.kind == "table" for artifact in bundle.artifacts),
            "images": len(image_ids),
            "vision_results": len(document_results),
            "non_image_leaks": len(leaks),
            "dimension": next(iter(dimensions), 0),
            "status": "PASS" if passed else "FAIL",
        }
        rows.append(row)

        validated = dict(payload)
        validated["vision_results"] = [asdict(result) for result in document_results]
        validated["providers"] = dict(payload.get("providers", {}), vision_processor="pixelrag")
        target = output / source_manifest.parent.name / "hybrid-document.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_report(output, rows, processor.model, processor.device)
    summary = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "documents": rows,
        "report": str(output / "pixelrag-acceptance-report.md"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

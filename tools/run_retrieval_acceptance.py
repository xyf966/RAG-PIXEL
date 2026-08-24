from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_input import HybridSearchEngine, RetrievalRequest


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected an object at {path}:{line_number}")
        rows.append(payload)
    return rows


def normalized_cases(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _read_json(config_path)
    ground_truth = _read_jsonl(config_path.parent / config["source_ground_truth"])
    mapping = config.get("modality_mapping") or {}
    overrides = config.get("overrides") or {}
    cases = []
    for row in ground_truth:
        case_id = row.get("id")
        evidence = overrides.get(case_id)
        if evidence is None:
            modality = mapping.get(row.get("modality"))
            if modality is None:
                raise RuntimeError(f"No retrieval modality mapping for {case_id}")
            evidence = [
                {
                    "label": row.get("locator") or row.get("modality"),
                    "modality": modality,
                    "page": row.get("page"),
                }
            ]
        cases.append(
            {
                "id": case_id,
                "question": row["question"],
                "source_modality": row.get("modality"),
                "locator": row.get("locator"),
                "expected_evidence": evidence,
            }
        )
    return config, cases


def _hit_matches(hit: Any, evidence: dict[str, Any]) -> bool:
    if hit.modality != evidence["modality"]:
        return False
    if evidence.get("page") is not None and hit.provenance.get("page") != evidence["page"]:
        return False
    record_ids = evidence.get("record_ids")
    if record_ids and hit.record_id not in record_ids:
        return False
    return True


def evaluate_case(response: Any, case: dict[str, Any]) -> dict[str, Any]:
    evidence_results = []
    for evidence in case["expected_evidence"]:
        rank = next(
            (
                hit.rank
                for hit in response.hits
                if _hit_matches(hit, evidence)
            ),
            None,
        )
        evidence_results.append({**evidence, "rank": rank})
    return {
        **case,
        "snapshot_build_id": response.snapshot_build_id,
        "elapsed_ms": response.elapsed_ms,
        "warnings": response.warnings,
        "evidence": evidence_results,
        "hits": [
            {
                "rank": hit.rank,
                "record_id": hit.record_id,
                "modality": hit.modality,
                "raw_score": hit.raw_score,
                "fusion_score": hit.fusion_score,
                "page": hit.provenance.get("page"),
                "context": hit.context,
            }
            for hit in response.hits
        ],
    }


def calculate_metrics(results: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    evidence = [item for result in results for item in result["evidence"]]
    if not results or not evidence:
        raise RuntimeError("Retrieval acceptance has no evaluated evidence")
    metrics: dict[str, Any] = {
        "queries": len(results),
        "evidence_units": len(evidence),
    }
    for k in ks:
        matched = sum(item["rank"] is not None and item["rank"] <= k for item in evidence)
        any_queries = sum(
            any(item["rank"] is not None and item["rank"] <= k for item in result["evidence"])
            for result in results
        )
        all_queries = sum(
            all(item["rank"] is not None and item["rank"] <= k for item in result["evidence"])
            for result in results
        )
        metrics[f"evidence_recall_at_{k}"] = matched / len(evidence)
        metrics[f"query_any_recall_at_{k}"] = any_queries / len(results)
        metrics[f"query_all_recall_at_{k}"] = all_queries / len(results)
    reciprocal_ranks = []
    for result in results:
        ranks = [item["rank"] for item in result["evidence"] if item["rank"] is not None]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
    metrics["mrr"] = statistics.fmean(reciprocal_ranks)
    per_modality = {}
    for modality in ("text", "table", "visual"):
        items = [item for item in evidence if item["modality"] == modality]
        if items:
            per_modality[modality] = {
                f"recall_at_{k}": sum(
                    item["rank"] is not None and item["rank"] <= k for item in items
                )
                / len(items)
                for k in ks
            }
            per_modality[modality]["evidence_units"] = len(items)
    metrics["per_modality"] = per_modality
    elapsed = [float(result["elapsed_ms"]) for result in results]
    sorted_elapsed = sorted(elapsed)
    p95_index = min(math.ceil(len(sorted_elapsed) * 0.95) - 1, len(sorted_elapsed) - 1)
    metrics["latency_ms"] = {
        "mean": statistics.fmean(elapsed),
        "median": statistics.median(elapsed),
        "p95": sorted_elapsed[p95_index],
        "cold_first": elapsed[0],
        "warm_mean": statistics.fmean(elapsed[1:]) if len(elapsed) > 1 else elapsed[0],
    }
    return metrics


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _markdown(report: dict[str, Any], ks: list[int]) -> str:
    metrics = report["metrics"]
    lines = [
        "# 检索层真实模型验收结果",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 快照：`{report['snapshot_build_id']}`",
        f"- 模型：`{report['model']}`",
        f"- 查询数：{metrics['queries']}；证据单元：{metrics['evidence_units']}",
        f"- 模型加载次数：{report['model_load_count']}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
    ]
    for k in ks:
        lines.extend(
            [
                f"| Evidence Recall@{k} | {_percent(metrics[f'evidence_recall_at_{k}'])} |",
                f"| Query Any Recall@{k} | {_percent(metrics[f'query_any_recall_at_{k}'])} |",
                f"| Query All Recall@{k} | {_percent(metrics[f'query_all_recall_at_{k}'])} |",
            ]
        )
    lines.append(f"| MRR | {metrics['mrr']:.4f} |")
    latency = metrics["latency_ms"]
    lines.extend(
        [
            "",
            "## 耗时",
            "",
            f"- 首次查询（含模型加载）：{latency['cold_first']:.1f} ms",
            f"- 暖查询平均：{latency['warm_mean']:.1f} ms",
            f"- 中位数：{latency['median']:.1f} ms",
            f"- P95：{latency['p95']:.1f} ms",
            "",
            "## 分模态",
            "",
            "| 模态 | 证据数 | " + " | ".join(f"Recall@{k}" for k in ks) + " |",
            "|---|---:|" + "---:|" * len(ks),
        ]
    )
    for modality, values in metrics["per_modality"].items():
        lines.append(
            f"| {modality} | {values['evidence_units']} | "
            + " | ".join(_percent(values[f"recall_at_{k}"]) for k in ks)
            + " |"
        )
    lines.extend(["", "## 未完全命中的问题", ""])
    failures = [
        result
        for result in report["results"]
        if any(item["rank"] is None or item["rank"] > max(ks) for item in result["evidence"])
    ]
    if not failures:
        lines.append("全部问题在 Top-K 内命中所有标注证据。")
    else:
        for result in failures:
            details = ", ".join(
                f"{item['label']}={item['rank'] or '未命中'}" for item in result["evidence"]
            )
            lines.append(f"- `{result['id']}` {result['question']}（{details}）")
    lines.extend(["", "## 阈值", ""])
    for name, value in report["thresholds"].items():
        actual = metrics.get(name)
        status = "通过" if actual is not None and actual >= value else "未通过"
        lines.append(f"- `{name}`：要求 {_percent(value)}，实际 {_percent(actual or 0.0)}，{status}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate schema 2.0 retrieval with a real model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enforce-thresholds", action="store_true")
    args = parser.parse_args()

    config, cases = normalized_cases(args.config.resolve())
    ks = sorted({int(value) for value in config.get("top_k_values", [5, 10])})
    max_k = max(ks)
    candidate_k = max(int(config.get("candidate_k", 30)), max_k)
    results = []
    with HybridSearchEngine(args.index_dir.resolve(), device=args.device) as engine:
        for ordinal, case in enumerate(cases, 1):
            print(f"[{ordinal}/{len(cases)}] {case['id']}: {case['question']}", flush=True)
            response = engine.search(
                RetrievalRequest(
                    case["question"],
                    top_k=max_k,
                    candidate_k=candidate_k,
                )
            )
            results.append(evaluate_case(response, case))
        session = engine._owned_session
        model_load_count = int(getattr(session, "load_count", 0))
        snapshot = engine.snapshot_store.get_snapshot()

    metrics = calculate_metrics(results, ks)
    thresholds = {
        str(name): float(value)
        for name, value in (config.get("thresholds") or {}).items()
    }
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "testset": config.get("name"),
        "snapshot_build_id": snapshot.build_id,
        "model": snapshot.model,
        "model_load_count": model_load_count,
        "metrics": metrics,
        "thresholds": thresholds,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "retrieval-acceptance-results.json"
    markdown_path = args.output_dir / "检索层真实模型验收结果.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report, ks), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {json_path.resolve()}", flush=True)
    print(f"Markdown: {markdown_path.resolve()}", flush=True)

    failed = [
        name
        for name, required in thresholds.items()
        if metrics.get(name) is None or metrics[name] < required
    ]
    if args.enforce_thresholds and failed:
        raise SystemExit(f"Acceptance thresholds failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

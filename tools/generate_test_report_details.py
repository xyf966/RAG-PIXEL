from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "多模态输入结果_20260814"
START = "<!-- GENERATED_ARTIFACTS_START -->"
END = "<!-- GENERATED_ARTIFACTS_END -->"

REPORTS = [
    (
        REPORT_DIR / "Word识别结果_20260814.md",
        ROOT / "input-acceptance-20260817/01_星云智能仓库运营报告-1756628c427a/hybrid-document.json",
    ),
    (
        REPORT_DIR / "PowerPoint识别结果_20260814.md",
        ROOT / "input-acceptance-20260817/02_北辰物流中心扩容方案-516b8d89aad9/hybrid-document.json",
    ),
    (
        REPORT_DIR / "Excel识别结果_20260814.md",
        ROOT / "input-acceptance-20260817/03_云帆仓配中心能效台账-6463ec510d14/hybrid-document.json",
    ),
    (
        REPORT_DIR / "CTCC_Bosch_Partner_Benefits识别结果_20260814.md",
        ROOT / "PixelRAG-Studio-Data/projects/我的视觉知识库/artifacts/CTCC_2026-2027_Bosch_Partner_Benefits_EN_and_CN-f0d17defce22/hybrid-document.json",
    ),
    (
        REPORT_DIR / "PDF识别结果_20260817.md",
        ROOT / "input-acceptance-20260817/05_天枢冷链中心多模态运营分析报告-6f6021ed4f75/hybrid-document.json",
    ),
    (
        REPORT_DIR / "CTCC_PDF识别结果_20260817.md",
        ROOT / "PixelRAG-Studio-Data/projects/我的视觉知识库/artifacts/2026-2027赛季CTCC中国汽车场地职业联赛 合作伙伴权益 - 博世 - 0812-2bfe292e2f7f/hybrid-document.json",
    ),
]


KIND_LABELS = {
    "text": "文字",
    "table": "表格",
    "image": "图片",
    "visual_task": "视觉任务",
}


def _inline(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("`", "\\`").replace("\n", " ")


def _bbox(value: Any) -> str:
    if not isinstance(value, list):
        return "—"
    return "[" + ", ".join(f"{float(item):.6f}" for item in value) + "]"


def _image_target(asset_path: str, report: Path) -> str:
    relative = os.path.relpath(Path(asset_path), report.parent).replace("\\", "/")
    return f"<{relative}>"


def _artifact_section(payload: dict[str, Any], report: Path) -> str:
    vision_by_block = {
        item["block_id"]: item for item in payload.get("vision_results", [])
    }
    lines = [
        START,
        "",
        "## 完整识别结果（由最终 HybridDocument 生成）",
        "",
        "> 以下内容按 `reading_order` 逐块打印；表格正文就是解析器实际输出的 Markdown，不是人工重写。",
        "",
        f"- 最终解析器：`{_inline(payload.get('providers', {}).get('parser'))}`",
        f"- 内容块总数：{len(payload.get('artifacts', []))}",
        f"- 警告数：{len(payload.get('warnings', []))}",
        "",
    ]
    for index, artifact in enumerate(payload.get("artifacts", []), start=1):
        kind = artifact.get("kind", "unknown")
        provenance = artifact.get("provenance", {})
        metadata = artifact.get("metadata", {})
        semantic = metadata.get("semantic_type")
        title = f"### 内容块 {index}：{KIND_LABELS.get(kind, kind)}"
        if semantic:
            title += f"（{semantic}）"
        lines.extend(
            [
                title,
                "",
                f"- `block_id`：`{_inline(artifact.get('block_id'))}`",
                f"- `reading_order`：{artifact.get('reading_order', index - 1)}",
                f"- 页码：{_inline(provenance.get('page'))}；幻灯片：{_inline(provenance.get('slide'))}；工作表：`{_inline(provenance.get('sheet'))}`",
                f"- 单元格范围：`{_inline(provenance.get('cell_range'))}`",
                f"- `bbox`：`{_bbox(provenance.get('bbox'))}`",
                f"- `locator`：`{_inline(provenance.get('locator'))}`",
            ]
        )
        if artifact.get("parent_block_id"):
            lines.append(f"- 父级：`{_inline(artifact.get('parent_block_id'))}`")
        if kind == "table":
            rows = metadata.get("rows")
            columns = metadata.get("columns")
            if rows is not None or columns is not None:
                lines.append(f"- 表格规模：{_inline(rows)} 行 × {_inline(columns)} 列")
            if metadata.get("structure_origin"):
                lines.append(f"- 结构来源：`{_inline(metadata.get('structure_origin'))}`")
            if metadata.get("span_reliability"):
                lines.append(f"- 合并跨度可靠性：`{_inline(metadata.get('span_reliability'))}`")
            merged = metadata.get("merged_ranges") or []
            if merged:
                lines.append(f"- 合并范围（{len(merged)}）：`{'`, `'.join(map(str, merged))}`")
            lines.extend(["", "#### 实际表格输出", "", artifact.get("text") or "_空表格_", ""])
        elif artifact.get("text"):
            lines.extend(["", "#### 实际文字输出", "", artifact["text"], ""])
        if artifact.get("context"):
            lines.extend(["", f"- 图片上下文：{artifact['context']}"])
        if artifact.get("asset_path"):
            path = str(artifact["asset_path"])
            lines.extend(
                [
                    "",
                    f"![内容块 {index} 图片]({_image_target(path, report)})",
                    "",
                    f"- 图片文件：`{path}`",
                    f"- SHA-256：`{_inline(artifact.get('sha256'))}`",
                ]
            )
        vision = vision_by_block.get(artifact.get("block_id"))
        if vision:
            status = vision.get("metadata", {}).get("status", "completed")
            lines.append(
                f"- 视觉处理：provider=`{_inline(vision.get('provider'))}`，status=`{_inline(status)}`，vector={'有' if vision.get('vector') else '无'}"
            )
        lines.append("")
    if payload.get("warnings"):
        lines.extend(["### 警告", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
        lines.append("")
    lines.append(END)
    return "\n".join(lines).rstrip() + "\n"


def update_report(report: Path, manifest: Path) -> None:
    if not report.is_file():
        raise FileNotFoundError(report)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    current = report.read_text(encoding="utf-8")
    if START in current and END in current:
        before, remainder = current.split(START, 1)
        _, after = remainder.split(END, 1)
        current = before.rstrip() + after.lstrip("\r\n")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    updated = current.rstrip() + "\n\n" + _artifact_section(payload, report)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(report)
    print(f"updated {report.name}: {len(payload.get('artifacts', []))} artifacts")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for report, manifest in REPORTS:
        update_report(report, manifest)


if __name__ == "__main__":
    main()

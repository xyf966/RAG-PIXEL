from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(source: Path, output_dir: Path) -> list[Path]:
    import win32com.client

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    destination = (output_dir / f"{source.stem}.pdf").resolve()
    application = None
    document = None
    close_kind = ""
    try:
        if suffix in {".doc", ".docx", ".docm"}:
            close_kind = "word"
            application = win32com.client.DispatchEx("Word.Application")
            application.Visible = False
            application.DisplayAlerts = 0
            document = application.Documents.Open(str(source.resolve()), ReadOnly=True)
            document.ExportAsFixedFormat(str(destination), 17)
        elif suffix in {".ppt", ".pptx", ".pptm"}:
            close_kind = "powerpoint"
            application = win32com.client.DispatchEx("PowerPoint.Application")
            document = application.Presentations.Open(str(source.resolve()), WithWindow=False)
            document.SaveAs(str(destination), 32)
        elif suffix in {".xls", ".xlsx", ".xlsm"}:
            close_kind = "excel"
            application = win32com.client.DispatchEx("Excel.Application")
            application.Visible = False
            application.DisplayAlerts = False
            document = application.Workbooks.Open(str(source.resolve()), ReadOnly=True)
            document.ExportAsFixedFormat(0, str(destination))
        else:
            raise ValueError(f"Unsupported Office format: {suffix}")
    finally:
        if document is not None:
            if close_kind == "powerpoint":
                document.Close()
            else:
                document.Close(False)
        if application is not None:
            application.Quit()
    if not destination.is_file():
        raise RuntimeError("Microsoft Office produced no PDF")
    return [destination]


def export_excel_charts(source: Path, output_dir: Path) -> list[dict[str, object]]:
    import win32com.client

    output_dir.mkdir(parents=True, exist_ok=True)
    application = win32com.client.DispatchEx("Excel.Application")
    application.Visible = False
    application.DisplayAlerts = False
    workbook = None
    visuals: list[dict[str, object]] = []
    try:
        workbook = application.Workbooks.Open(str(source.resolve()), ReadOnly=True)
        for sheet_index in range(1, workbook.Worksheets.Count + 1):
            worksheet = workbook.Worksheets.Item(sheet_index)
            chart_objects = worksheet.ChartObjects()
            for chart_index in range(1, chart_objects.Count + 1):
                chart_object = chart_objects.Item(chart_index)
                destination = (output_dir / f"sheet-{sheet_index:03d}-chart-{chart_index:03d}.png").resolve()
                exported = bool(chart_object.Chart.Export(str(destination), "PNG"))
                if not exported or not destination.is_file():
                    continue
                # Depending on the generated pywin32 wrapper, Address may be
                # exposed either as a callable COM method or a string property.
                top_left_address = chart_object.TopLeftCell.Address
                bottom_right_address = chart_object.BottomRightCell.Address
                top_left = (
                    top_left_address(False, False)
                    if callable(top_left_address)
                    else str(top_left_address).replace("$", "")
                )
                bottom_right = (
                    bottom_right_address(False, False)
                    if callable(bottom_right_address)
                    else str(bottom_right_address).replace("$", "")
                )
                visuals.append(
                    {
                        "path": str(destination),
                        "sheet": str(worksheet.Name),
                        "sheet_index": sheet_index,
                        "chart_index": chart_index,
                        "cell_range": f"{top_left}:{bottom_right}",
                        "bounds_points": [
                            float(chart_object.Left),
                            float(chart_object.Top),
                            float(chart_object.Left + chart_object.Width),
                            float(chart_object.Top + chart_object.Height),
                        ],
                    }
                )
    finally:
        if workbook is not None:
            workbook.Close(False)
        application.Quit()
    return visuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--mode", choices=["pdf", "excel-charts"], default="pdf")
    args = parser.parse_args()
    if args.mode == "excel-charts":
        payload = {"visuals": export_excel_charts(args.input, args.output)}
    else:
        outputs = render(args.input, args.output)
        payload = {"outputs": [str(path) for path in outputs]}
    args.result.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

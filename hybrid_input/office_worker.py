from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree


def _com_text(value: object, attribute: str) -> str:
    try:
        text = str(getattr(value, attribute) or "").replace("\r", " ").strip()
        return text
    except Exception:
        return ""


def _word_visual_text(item: object) -> str:
    candidates: list[str] = []
    for attribute in ("AlternativeText", "Title"):
        value = _com_text(item, attribute)
        if value:
            candidates.append(value)
    try:
        value = _com_text(item.TextEffect, "Text")
        if value:
            candidates.append(value)
    except Exception:
        pass
    for frame_name in ("TextFrame", "TextFrame2"):
        try:
            frame = getattr(item, frame_name)
            if bool(frame.HasText):
                value = _com_text(frame.TextRange, "Text")
                if value:
                    candidates.append(value)
        except Exception:
            pass
    return " ".join(dict.fromkeys(candidates))


def _word_diagram_texts(source: Path) -> list[str]:
    """Read SmartArt text that Word does not expose as ordinary document text."""
    results: list[str] = []
    try:
        with zipfile.ZipFile(source) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name.startswith("word/diagrams/data") and name.endswith(".xml")
            )
            for name in names:
                root = ElementTree.fromstring(archive.read(name))
                values = [
                    (node.text or "").strip()
                    for node in root.iter()
                    if node.tag.endswith("}t") and (node.text or "").strip()
                ]
                text = " ".join(dict.fromkeys(values))
                if text:
                    results.append(text)
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError):
        pass
    return results


def _save_clipboard_picture(copy_action: object, destination: Path) -> bool:
    """Run an Office CopyAsPicture action and persist a raster clipboard image."""
    import win32clipboard
    from PIL import Image, ImageGrab

    destination.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(3):
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.CloseClipboard()
            copy_action()  # type: ignore[operator]
            for _poll in range(12):
                value = ImageGrab.grabclipboard()
                if isinstance(value, Image.Image):
                    rgba = value.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    background.alpha_composite(rgba)
                    background.convert("RGB").save(destination, "PNG")
                    return destination.is_file()
                time.sleep(0.1)
        except Exception:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass
        time.sleep(0.2)
    return False


def _word_copy_action(application: object, item: object, collection_name: str):
    if collection_name == "InlineShapes":
        def copy_inline() -> None:
            item.Range.Select()
            try:
                application.ActiveWindow.ScrollIntoView(item.Range, True)
            except Exception:
                pass
            item.Range.CopyAsPicture()

        return copy_inline

    def copy_shape() -> None:
        item.Select()
        try:
            application.ActiveWindow.ScrollIntoView(item.Anchor, True)
        except Exception:
            pass
        application.Selection.CopyAsPicture()

    return copy_shape


def _word_plain_copy_action(item: object, collection_name: str):
    """Copy the underlying Office object to the clipboard as a second native path."""
    if collection_name == "InlineShapes":
        return lambda: item.Range.Copy()
    return lambda: item.Copy()


def export_word_visuals(source: Path, output_dir: Path | None) -> list[dict[str, object]]:
    """Export visible Word objects directly; retain page bounds as a fallback."""
    import win32com.client

    application = win32com.client.DispatchEx("Word.Application")
    # CopyAsPicture relies on Word's active window and clipboard.  Keeping
    # the application visible is required in interactive COM sessions;
    # strict indexing never treats a failed copy as a page-crop success.
    application.Visible = True
    application.DisplayAlerts = 0
    document = None
    visuals: list[dict[str, object]] = []
    try:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        document = application.Documents.Open(str(source.resolve()), ReadOnly=True)
        document.Activate()
        try:
            application.ActiveWindow.Activate()
        except Exception:
            pass
        document.Repaginate()
        for collection_name in ("InlineShapes", "Shapes"):
            collection = getattr(document, collection_name)
            for index in range(1, collection.Count + 1):
                item = collection.Item(index)
                item_type = int(item.Type)
                if collection_name == "Shapes" and item_type == 17:
                    continue
                if collection_name == "Shapes" and item_type == 1 and _has_shape_text(item):
                    continue
                anchor = item.Range if collection_name == "InlineShapes" else item.Anchor
                try:
                    page = int(anchor.Information(3))
                    left = float(anchor.Information(5))
                    top = float(anchor.Information(6))
                    width = float(item.Width)
                    height = float(item.Height)
                    if left < -1000 or top < -1000:
                        left = float(getattr(item, "Left"))
                        top = float(getattr(item, "Top"))
                    if page < 1 or width <= 0 or height <= 0:
                        continue
                    destination = (
                        (output_dir / f"word-{collection_name.lower()}-{index:04d}.png").resolve()
                        if output_dir is not None else None
                    )
                    exported = bool(destination) and _save_clipboard_picture(
                        _word_copy_action(application, item, collection_name), destination
                    )
                    render_method = "word-copy-as-picture"
                    if not exported and destination:
                        exported = _save_clipboard_picture(
                            _word_plain_copy_action(item, collection_name), destination
                        )
                        if exported:
                            render_method = "word-native-clipboard-copy"
                    visuals.append(
                        {
                            "collection": collection_name,
                            "collection_index": index,
                            "anchor_start": int(anchor.Start),
                            "page": page,
                            "bounds_points": [left, top, left + width, top + height],
                            "object_type": item_type,
                            "text": _word_visual_text(item),
                            "path": str(destination) if exported and destination else None,
                            "render_method": render_method if exported else "page-crop-fallback",
                        }
                    )
                except Exception:
                    continue
    finally:
        if document is not None:
            document.Close(False)
        application.Quit()
    visuals.sort(key=lambda item: (int(item["anchor_start"]), str(item["collection"])))
    diagrams = iter(_word_diagram_texts(source))
    for visual in visuals:
        if int(visual.get("object_type") or 0) == 15 and not visual.get("text"):
            visual["text"] = next(diagrams, "")
    return visuals


def inspect_word_visuals(source: Path) -> list[dict[str, object]]:
    """Compatibility helper used by callers that only need Word metadata."""
    return export_word_visuals(source, None)


def _shape_text(item: object) -> str:
    candidates: list[str] = []
    for attribute in ("AlternativeText", "Title"):
        value = _com_text(item, attribute)
        if value:
            candidates.append(value)
    return " ".join(dict.fromkeys(candidates))


def _has_shape_text(item: object) -> bool:
    for frame_name in ("TextFrame", "TextFrame2"):
        try:
            frame = getattr(item, frame_name)
            if bool(frame.HasText) and _com_text(frame.TextRange, "Text"):
                return True
        except Exception:
            pass
    return False


def export_powerpoint_visuals(source: Path, output_dir: Path | None) -> list[dict[str, object]]:
    """Export non-text PowerPoint shapes with PowerPoint's native renderer."""
    import win32com.client

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    application = win32com.client.DispatchEx("PowerPoint.Application")
    presentation = None
    visuals: list[dict[str, object]] = []
    visual_types = {1, 3, 6, 11, 13, 16, 18, 20, 22, 24, 28, 30}
    try:
        presentation = application.Presentations.Open(str(source.resolve()), WithWindow=False)
        for slide_index in range(1, presentation.Slides.Count + 1):
            slide = presentation.Slides.Item(slide_index)
            for shape_index in range(1, slide.Shapes.Count + 1):
                shape = slide.Shapes.Item(shape_index)
                shape_type = int(shape.Type)
                if shape_type not in visual_types or (shape_type == 1 and _has_shape_text(shape)):
                    continue
                print(
                    f"PowerPoint native visual export: slide {slide_index}, shape {shape_index}",
                    flush=True,
                )
                destination = ((output_dir / f"slide-{slide_index:04d}-shape-{shape_index:04d}.png").resolve() if output_dir is not None else None)
                exported = False
                shape_width = float(shape.Width)
                shape_height = float(shape.Height)
                # Very thin edge decorations can hang PowerPoint's Export
                # method.  They are not useful visual retrieval objects and
                # remain eligible for the normal page-crop quality gate.
                if destination is not None and shape_width >= 12 and shape_height >= 12:
                    try:
                        shape.Export(str(destination), 2)
                        exported = destination.is_file()
                    except Exception:
                        exported = _save_clipboard_picture(lambda: shape.Copy(), destination)
                visuals.append(
                    {
                        "path": str(destination) if exported else None,
                        "slide": slide_index,
                        "shape_index": shape_index,
                        "object_type": shape_type,
                        "bounds_points": [
                            float(shape.Left), float(shape.Top),
                            float(shape.Left + shape_width), float(shape.Top + shape_height),
                        ],
                        "text": _shape_text(shape),
                        "render_method": "powerpoint-shape-export" if exported else "page-crop-fallback",
                    }
                )
    finally:
        if presentation is not None:
            presentation.Close()
        application.Quit()
    return visuals


def export_excel_visuals(source: Path, output_dir: Path | None) -> list[dict[str, object]]:
    """Export Excel pictures, charts, SmartArt and grouped visual shapes."""
    import win32com.client

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    application = win32com.client.DispatchEx("Excel.Application")
    application.Visible = False
    application.DisplayAlerts = False
    workbook = None
    visuals: list[dict[str, object]] = []
    visual_types = {1, 3, 6, 11, 13, 16, 18, 20, 22, 24, 28, 30}
    try:
        workbook = application.Workbooks.Open(str(source.resolve()), ReadOnly=True)
        for sheet_index in range(1, workbook.Worksheets.Count + 1):
            worksheet = workbook.Worksheets.Item(sheet_index)
            for shape_index in range(1, worksheet.Shapes.Count + 1):
                shape = worksheet.Shapes.Item(shape_index)
                shape_type = int(shape.Type)
                if shape_type not in visual_types or (shape_type == 1 and _has_shape_text(shape)):
                    continue
                destination = ((output_dir / f"sheet-{sheet_index:04d}-shape-{shape_index:04d}.png").resolve() if output_dir is not None else None)
                exported = False
                if destination is not None:
                    try:
                        if shape_type == 3:
                            exported = bool(shape.Chart.Export(str(destination), "PNG"))
                        else:
                            exported = _save_clipboard_picture(
                                lambda current=shape: current.CopyPicture(1, 2), destination
                            )
                    except Exception:
                        exported = False
                top_left_address = shape.TopLeftCell.Address
                bottom_right_address = shape.BottomRightCell.Address
                top_left = top_left_address(False, False) if callable(top_left_address) else str(top_left_address).replace("$", "")
                bottom_right = bottom_right_address(False, False) if callable(bottom_right_address) else str(bottom_right_address).replace("$", "")
                visuals.append(
                    {
                        "path": str(destination) if exported and destination and destination.is_file() else None,
                        "sheet": str(worksheet.Name),
                        "sheet_index": sheet_index,
                        "shape_index": shape_index,
                        "object_type": shape_type,
                        "cell_range": f"{top_left}:{bottom_right}",
                        "bounds_points": [
                            float(shape.Left), float(shape.Top),
                            float(shape.Left + shape.Width), float(shape.Top + shape.Height),
                        ],
                        "text": _shape_text(shape),
                        "render_method": "excel-native-export" if exported else "page-crop-fallback",
                    }
                )
    finally:
        if workbook is not None:
            workbook.Close(False)
        application.Quit()
    return visuals


def export_office_visuals(source: Path, output_dir: Path) -> list[dict[str, object]]:
    suffix = source.suffix.lower()
    if suffix in {".doc", ".docx", ".docm"}:
        return export_word_visuals(source, output_dir)
    if suffix in {".ppt", ".pptx", ".pptm"}:
        return export_powerpoint_visuals(source, output_dir)
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        return export_excel_visuals(source, output_dir)
    raise ValueError(f"Unsupported Office format: {suffix}")


def inspect_office_visuals(source: Path) -> list[dict[str, object]]:
    """Identify native visual objects without exporting bitmaps."""
    suffix = source.suffix.lower()
    if suffix in {".doc", ".docx", ".docm"}:
        return export_word_visuals(source, None)
    if suffix in {".ppt", ".pptx", ".pptm"}:
        return export_powerpoint_visuals(source, None)
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        return export_excel_visuals(source, None)
    raise ValueError(f"Unsupported Office format: {suffix}")


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
    parser.add_argument(
        "--mode", choices=["pdf", "excel-charts", "word-visuals", "office-visuals", "office-inspect"], default="pdf"
    )
    args = parser.parse_args()
    if args.mode == "excel-charts":
        payload = {"visuals": export_excel_charts(args.input, args.output)}
    elif args.mode == "word-visuals":
        payload = {"visuals": export_word_visuals(args.input, args.output)}
    elif args.mode == "office-visuals":
        payload = {"visuals": export_office_visuals(args.input, args.output)}
    elif args.mode == "office-inspect":
        payload = {"visuals": inspect_office_visuals(args.input)}
    else:
        outputs = render(args.input, args.output)
        payload = {"outputs": [str(path) for path in outputs]}
    args.result.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

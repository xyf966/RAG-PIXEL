"""Project launcher for OpenDataLoader Hybrid on Unicode Windows paths.

Docling's native docling-parse backend can fail to open its own resource files
when the Python environment path contains non-ASCII characters.  Pdfium is an
official Docling PDF backend and avoids that native resource-path limitation.
"""

from __future__ import annotations


def main() -> None:
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from opendataloader_pdf import hybrid_server

    original_create_converter = hybrid_server.create_converter

    def create_converter(*args, **kwargs):
        converter = original_create_converter(*args, **kwargs)
        pdf_options = converter.format_to_options[InputFormat.PDF]
        pdf_options.backend = PyPdfiumDocumentBackend
        # Transformers layout inference defaults to torch.compile(), which on
        # Windows invokes cl.exe.  Studio must run on ordinary machines without
        # requiring the Visual C++ build toolchain.
        pdf_options.pipeline_options.layout_options.engine_options.compile_model = False
        return converter

    hybrid_server.create_converter = create_converter
    hybrid_server.main()


if __name__ == "__main__":
    main()

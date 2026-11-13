"""Docling (+ Tesseract for scans) layout parsing into a normalized `ParsedDocument`."""

from __future__ import annotations

import os
import time
from pathlib import Path

from .schemas import BBox, Block, PageInfo, ParsedDocument, Table, TableCell
from .sources import Source, load_page_map, work_pdf

# Docling labels that never carry procedure/warning/parts content.
SKIP_LABELS = {"page_header", "page_footer", "picture", "chart", "formula", "code"}


def _converter(kind: str):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.accelerator_options = AcceleratorOptions(
        num_threads=int(os.environ.get("MANUALTRACE_THREADS", "4")),
        device=AcceleratorDevice.CPU,
    )
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.generate_page_images = False
    if kind == "scanned":
        opts.do_ocr = True
        # Ignore any embedded text and OCR every page image with the tesseract binary.
        opts.ocr_options = TesseractCliOcrOptions(lang=["eng"], force_full_page_ocr=True)
        opts.table_structure_options.do_cell_matching = True
    else:
        opts.do_ocr = False
        opts.table_structure_options.do_cell_matching = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _bbox(prov_bbox, page_height: float, x_offset: float = 0.0) -> BBox:
    b = prov_bbox.to_top_left_origin(page_height=page_height)
    return BBox(
        left=min(b.l, b.r) + x_offset, top=min(b.t, b.b),
        right=max(b.l, b.r) + x_offset, bottom=max(b.t, b.b),
    )


def parse(source: Source, data_dir: Path) -> tuple[ParsedDocument, float]:
    """Run Docling on the working PDF. Returns the parsed document and wall seconds."""
    from docling_core.types.doc import ListItem, TableItem, TextItem

    pdf = work_pdf(data_dir, source.doc_id)
    start = time.perf_counter()
    result = _converter(source.kind).convert(pdf)
    elapsed = time.perf_counter() - start
    doc = result.document

    mapping = load_page_map(data_dir, source.doc_id)
    heights = {no: p.size.height for no, p in doc.pages.items()}
    pages = {}
    for m in mapping:
        pages.setdefault(m["page"], PageInfo(page=m["page"], width=m["width"],
                                             height=m["height"]))

    blocks: list[Block] = []
    tables: list[Table] = []
    for item, _level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        prov = item.prov[0]
        m = mapping[prov.page_no - 1]
        page_no, dx = m["page"], m["x_offset"]
        height = heights[prov.page_no]
        bbox = _bbox(prov.bbox, height, dx)
        label = str(getattr(item.label, "value", item.label))
        if isinstance(item, TableItem):
            cells = []
            for c in item.data.table_cells:
                cells.append(
                    TableCell(
                        row=c.start_row_offset_idx,
                        col=c.start_col_offset_idx,
                        text=" ".join(c.text.split()),
                        bbox=_bbox(c.bbox, height, dx) if c.bbox else None,
                        column_header=bool(c.column_header),
                    )
                )
            tables.append(
                Table(
                    id=f"t{len(tables)}",
                    page=page_no,
                    bbox=bbox,
                    n_rows=item.data.num_rows,
                    n_cols=item.data.num_cols,
                    cells=cells,
                )
            )
            # Keep a placeholder block so reading order knows where the table sits.
            blocks.append(
                Block(id=f"b{len(blocks)}", page=page_no, bbox=bbox, label="table",
                      text=f"[table t{len(tables) - 1}]")
            )
        elif isinstance(item, TextItem):
            text = " ".join(item.text.split())
            # Docling moves list enumerators ("1.", "(a)") into `marker`; keep them in text.
            marker = (getattr(item, "marker", "") or "").strip() if isinstance(
                item, ListItem) else ""
            if marker and marker not in "-•·*" and not text.startswith(marker):
                text = f"{marker} {text}"
            if not text or label in SKIP_LABELS:
                continue
            blocks.append(Block(id=f"b{len(blocks)}", page=page_no, bbox=bbox, label=label,
                                text=text))

    parsed = ParsedDocument(
        doc_id=source.doc_id,
        kind=source.kind,
        title=source.title,
        source_url=source.url,
        parser="docling+tesseract" if source.kind == "scanned" else "docling",
        pages=list(pages.values()),
        blocks=blocks,
        tables=tables,
    )
    return parsed, elapsed

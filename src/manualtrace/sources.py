"""Public-domain maritime manuals used by the harness, and the downloader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

from .schemas import DocKind


@dataclass(frozen=True)
class Source:
    doc_id: str
    kind: DocKind
    title: str
    url: str
    sha256: str
    # 1-based pages of the original PDF that the pipeline processes (None = all pages).
    pages: tuple[int, ...] | None
    # Pages that were fully annotated in gold/<doc_id>.json; P/R is scored on these only.
    gold_pages: tuple[int, ...]
    system_code: str  # S1000D systemCode used for this document's data modules
    notes: str
    # Scans of two-page spreads: cut landscape pages into left/right halves before layout
    # analysis (bboxes are mapped back to the original page afterwards).
    split_spreads: bool = False


SOURCES: dict[str, Source] = {
    s.doc_id: s
    for s in [
        Source(
            doc_id="tm55-1905-217-12",
            kind="digital",
            title="TM 55-1905-217-12 Operator and Organizational Maintenance Manual, "
            "Landing Craft, Mechanized, LCM-8",
            url="https://www.govinfo.gov/content/pkg/GOVPUB-D101-PURL-LPS37193/pdf/"
            "GOVPUB-D101-PURL-LPS37193.pdf",
            sha256="bd8d8b6fd024bafa6588ba44bde74fd02e6816dd61785f797efd544b8fec72a4",
            pages=(2, 75, 76, 84, 85, 86, 89, 90, 205, 207, 208, 209),
            gold_pages=(2, 76, 84, 85, 86, 90, 208, 209),
            system_code="LCM",
            notes="US Army technical manual published on govinfo.gov (US GPO). Digital PDF with "
            "an embedded text layer; Docling reads the text layer, no OCR.",
        ),
        Source(
            doc_id="navships-900524-ib",
            kind="scanned",
            title="Instruction Book for Type CAGQ-66134 Antenna Assembly for General Radio Use "
            "(NAVSHIPS 900,524-IB, 1944)",
            url="https://maritime.org/doc/pdf/sub-vhf-antenna-CAGQ-66134.pdf",
            sha256="993505480f90fd22a4a360cb669779414730701d3f96695eab9ecd06f8a71fe8",
            pages=None,
            gold_pages=(6, 7, 8, 9, 10, 11, 12, 13, 15),
            system_code="ANT",
            split_spreads=True,
            notes="US Navy Bureau of Ships instruction book for a submarine VHF antenna, hosted "
            "by the San Francisco Maritime National Park Association. Image-only scan "
            "(no text layer); every character comes from Tesseract OCR.",
        ),
    ]
}


def raw_path(data_dir: Path, doc_id: str) -> Path:
    return data_dir / "raw" / f"{doc_id}.pdf"


def work_pdf(data_dir: Path, doc_id: str) -> Path:
    """PDF actually handed to Docling (a page subset for large manuals)."""
    return data_dir / "work" / f"{doc_id}.pdf"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(source: Source, data_dir: Path) -> Path:
    """Download the PDF (if missing), verify its checksum, write the working subset."""
    dest = raw_path(data_dir, source.doc_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or sha256_file(dest) != source.sha256:
        tmp = dest.with_suffix(".part")
        with httpx.stream("GET", source.url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        digest = sha256_file(tmp)
        if digest != source.sha256:
            tmp.unlink()
            raise RuntimeError(f"{source.doc_id}: checksum mismatch ({digest})")
        tmp.replace(dest)

    work = work_pdf(data_dir, source.doc_id)
    work.parent.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(dest)
    writer = PdfWriter()
    mapping = []
    pages = list(source.pages) if source.pages else list(range(1, len(reader.pages) + 1))
    for page_no in pages:
        page = reader.pages[page_no - 1]
        box = page.cropbox
        x0, y0, x1, y1 = (float(v) for v in (box.left, box.bottom, box.right, box.top))
        w, h = x1 - x0, y1 - y0
        halves = [(x0, x1)]
        if source.split_spreads and w > 1.2 * h:
            halves = [(x0, x0 + w / 2), (x0 + w / 2, x1)]
        for left, right in halves:
            out = writer.add_page(page)
            rect = RectangleObject([left, y0, right, y1])
            out.mediabox = rect
            out.cropbox = rect
            mapping.append({"page": page_no, "x_offset": left - x0, "width": w, "height": h})
    with work.open("wb") as f:
        writer.write(f)
    page_map_path(data_dir, source.doc_id).write_text(json.dumps(mapping, indent=1))
    return work


def page_map_path(data_dir: Path, doc_id: str) -> Path:
    return data_dir / "work" / f"{doc_id}.pages.json"


def load_page_map(data_dir: Path, doc_id: str) -> list[dict]:
    """For each page of the working PDF: original page number, x offset, original size."""
    return json.loads(page_map_path(data_dir, doc_id).read_text())

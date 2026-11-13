"""Pydantic records shared by parsing, extraction, export and evaluation.

Coordinates: every bounding box is in PDF points of the *original* page, origin at the
top-left corner, as (left, top, right, bottom). Page numbers are 1-based page numbers of the
original downloaded PDF, never of an intermediate subset file.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DocKind = Literal["scanned", "digital"]
AdmonitionKind = Literal["warning", "caution", "note"]


class BBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    left: float
    top: float
    right: float
    bottom: float

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError(f"degenerate bbox {self}")
        return self

    def union(self, other: BBox) -> BBox:
        return BBox(
            left=min(self.left, other.left),
            top=min(self.top, other.top),
            right=max(self.right, other.right),
            bottom=max(self.bottom, other.bottom),
        )

    def as_attr(self) -> str:
        return f"{self.left:.1f},{self.top:.1f},{self.right:.1f},{self.bottom:.1f}"


class Provenance(BaseModel):
    """Where a record came from: one entry per page region it spans."""

    doc_id: str
    page: int = Field(ge=1)
    bbox: BBox
    block_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- parsed layout


class Block(BaseModel):
    """One layout element produced by Docling (text item or table row)."""

    id: str
    page: int = Field(ge=1)
    bbox: BBox
    label: str
    text: str


class TableCell(BaseModel):
    row: int
    col: int
    text: str
    bbox: BBox | None = None
    column_header: bool = False


class Table(BaseModel):
    id: str
    page: int = Field(ge=1)
    bbox: BBox
    n_rows: int
    n_cols: int
    cells: list[TableCell]

    def grid(self) -> list[list[str]]:
        rows = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            if 0 <= c.row < self.n_rows and 0 <= c.col < self.n_cols:
                rows[c.row][c.col] = c.text
        return rows


class PageInfo(BaseModel):
    page: int = Field(ge=1)
    width: float
    height: float


class ParsedDocument(BaseModel):
    doc_id: str
    kind: DocKind
    title: str
    source_url: str
    parser: str
    pages: list[PageInfo]
    blocks: list[Block]
    tables: list[Table]


# ---------------------------------------------------------------- extracted records


class Admonition(BaseModel):
    """A WARNING, CAUTION or NOTE."""

    id: str
    kind: AdmonitionKind
    text: str = Field(min_length=1)
    provenance: list[Provenance] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        return " ".join(v.split())


class ProcedureStep(BaseModel):
    number: str = Field(min_length=1)
    text: str = Field(min_length=1)
    provenance: list[Provenance] = Field(min_length=1)


class Procedure(BaseModel):
    id: str
    title: str = Field(min_length=1)
    steps: list[ProcedureStep] = Field(min_length=2)
    provenance: list[Provenance] = Field(min_length=1)


class Part(BaseModel):
    """One illustrated-parts / repair-parts list line."""

    id: str
    description: str = Field(min_length=1)
    item: str | None = None  # symbol designation / callout, e.g. "H-101 thru H-104"
    nsn: str | None = None  # national stock number
    part_number: str | None = None
    cage: str | None = None  # manufacturer (CAGE/FSCM) code
    quantity: str | None = None
    unit: str | None = None
    smr: str | None = None
    provenance: list[Provenance] = Field(min_length=1)


class ExtractionResult(BaseModel):
    doc_id: str
    kind: DocKind
    title: str
    extractor: str
    admonitions: list[Admonition] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    parts: list[Part] = Field(default_factory=list)

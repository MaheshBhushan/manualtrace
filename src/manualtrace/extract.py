"""Extraction of warnings/cautions/notes, procedures and parts lists from parsed layout.

Both extractors produce the same intermediate *labeling*: JSON that points at parsed
block ids and table rows and carries the record text. `build_result` turns a labeling into
validated Pydantic records whose page/bbox provenance is computed from the referenced
blocks, so coordinates always come from the parser, never from a model.

- ``rules``: deterministic heuristics over Docling's reading order (`label_payload`).
- ``llm``: the same payload is sent to a provider (see providers.py); the reply is
  validated, and records whose text is not grounded in the referenced blocks are dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .schemas import (
    Admonition,
    BBox,
    ExtractionResult,
    ParsedDocument,
    Part,
    Procedure,
    ProcedureStep,
    Provenance,
)

# ---------------------------------------------------------------- payload


OCR_DASH = re.compile(r"(?<=[A-Za-z0-9])\s?(?:=+-?|-?=+|~)\s?(?=[A-Za-z0-9])")


def ocr_fix(text: str) -> str:
    """Typewriter scans OCR hyphens as '=' or '~' ('H=105', 'NT=-539', 'H~141')."""
    return OCR_DASH.sub("-", text)


def make_payload(doc: ParsedDocument) -> dict:
    fix = ocr_fix if doc.kind == "scanned" else (lambda s: s)
    return {
        "doc_id": doc.doc_id,
        "kind": doc.kind,
        "blocks": [
            {"id": b.id, "page": b.page, "label": b.label, "text": fix(b.text),
             "x": round(b.bbox.left, 1), "y": round(b.bbox.top, 1),
             "w": round(b.bbox.right - b.bbox.left, 1)}
            for b in doc.blocks
        ],
        "tables": [{"id": t.id, "page": t.page,
                    "rows": [[fix(c) for c in row] for row in t.grid()]}
                   for t in doc.tables],
    }


LABELING_SCHEMA = {
    "admonitions": [{"kind": "warning|caution|note", "block_ids": ["b12", "b13"],
                     "text": "body text without the WARNING/CAUTION/NOTE heading"}],
    "procedures": [{"title": "procedure title", "steps": [
        {"number": "1", "block_ids": ["b20"], "text": "step text without its number"}]}],
    "parts": [{"table_id": "t0", "row": 3, "description": "...", "item": None, "nsn": None,
               "part_number": None, "cage": None, "quantity": None, "unit": None,
               "smr": None}],
}

SYSTEM_PROMPT = (
    "You extract structured records from technical maintenance manuals. You receive the "
    "layout blocks (id, page, label, text, x/y position in points) and tables (row grids) "
    "of one document, already in reading order. Return ONLY a JSON object with keys "
    "admonitions, procedures and parts, shaped like this example:\n"
    + json.dumps(LABELING_SCHEMA)
    + "\nRules: admonitions are WARNING, CAUTION and NOTE paragraphs; give the body text "
    "only. Procedures are ordered lists of at least two numbered or lettered action steps "
    "(e.g. 'STEP 1.', '1.', '(a)', '(A)'); do not treat section or paragraph headings as "
    "steps. Parts are rows of parts/spares lists: item = symbol/callout designation, nsn = "
    "national stock number, part_number = manufacturer part or drawing number, cage = "
    "5-digit CAGE/FSCM code shown in parentheses, smr = SMR code, unit = unit of issue. Skip "
    "header rows and group headings. Copy text verbatim from the blocks (fix only "
    "hyphenation and line breaks); every record must cite the block ids or table row it "
    "came from. Use null for absent fields."
)


def build_prompt(doc: ParsedDocument) -> tuple[str, str]:
    from .providers import embed_payload

    user = ("Extract the records from this document. The payload is JSON between the "
            "markers.\n" + embed_payload(make_payload(doc)))
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------- rule-based labeler

ADM_HEAD = re.compile(r"^(warning|caution|note)\s*[:.]?$", re.I)
ADM_INLINE = re.compile(r"\b(WARNING|CAUTION|NOTE)\s*:\s*", re.S)
STEP_INLINE = re.compile(r"\bSTEP\s+(\d+)\s*\.\s*")
ADM_LEADING = re.compile(r"^(WARNING|CAUTION|NOTE)\s+[^:\s]")
ADM_SPLIT = re.compile(r"\b(WARNING|CAUTION|NOTE)\b(?!\s*:)")

# (style, regex); the group captures the step number/letter.
STEP_STYLES = [
    ("STEP", re.compile(r"^STEP\s+(\d+)\s*[.:]\s*(.*)$", re.S)),
    ("(1)", re.compile(r"^\((\d+)\)\s*(.*)$", re.S)),
    ("(a)", re.compile(r"^\(([a-z])\)\s*(.*)$", re.S)),
    ("(A)", re.compile(r"^\(([A-Z])\)\s*(.*)$", re.S)),
    ("1.", re.compile(r"^(\d+)\s*[.,]{1,2}\s+(.*)$", re.S)),  # tolerates OCR "4," / "5.,"
    ("a.", re.compile(r"^([a-z])\s*\.\s+(.*)$", re.S)),
]
NOISE = re.compile(r"^(restricted|tm [\d\- ]+|change \d+.*|[\d]+-[\d]+|me [\d\-/ ()]+)$", re.I)
PARA_NUM = re.compile(r"^\d+-\d+\.\s")  # "2-11. Cold Weather Operation" paragraph headings


def _ordinal(style: str, token: str) -> int:
    return int(token) if token.isdigit() else ord(token.lower()) - ord("a") + 1


def step_marker(text: str) -> tuple[str, str, str] | None:
    for style, rx in STEP_STYLES:
        m = rx.match(text)
        if m:
            return style, m.group(1), m.group(2).strip()
    return None


def is_heading_item(style: str, body: str, next_text: str = "") -> bool:
    """'1. GENERAL.' / 'b. Engines.' / 'c. Steering System. Test ...' / '(2) Batteries.'
    followed by '(a) ...' are paragraph headings, not procedure steps."""
    first = re.split(r"(?<=\.)\s", body, maxsplit=1)[0].rstrip(".")
    first = re.sub(r"\s*\([^)]*\)", "", first)  # "(fig 3-1 or 3-2)" is not part of a title
    words = first.split()
    if style == "1.":
        letters = re.sub(r"[^A-Za-z]", "", body)
        return bool(letters) and letters.upper() == letters and len(words) <= 12
    if style == "a.":
        return 0 < len(words) <= 6 and all(w[0].isupper() or not w[0].isalpha() or
                                           w in {"and", "or", "of", "the", "to"} for w in words)
    if style == "(1)":
        nxt = step_marker(next_text.strip())
        return (0 < len(words) <= 4 and body.rstrip().endswith(".") and len(body) < 40
                and nxt is not None and nxt[0] == "(a)")
    return False


@dataclass
class _B:
    id: str
    page: int
    label: str
    text: str
    x: float
    y: float
    w: float


def _blocks(payload: dict) -> list[_B]:
    return [_B(b["id"], b["page"], b["label"], b["text"], b["x"], b["y"], b["w"])
            for b in payload["blocks"]]


def _is_noise(b: _B) -> bool:
    return bool(NOISE.match(b.text.strip())) or b.label in {"footnote"}


def _ends_sentence(t: str) -> bool:
    return t.rstrip().endswith((".", ":", ";", "!", "?", ")"))


def label_admonitions(blocks: list[_B]) -> list[dict]:
    out = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        head = ADM_HEAD.match(b.text.strip())
        if head:
            kind = head.group(1).lower()
            ids, parts = [], []
            j = i + 1
            while j < len(blocks):
                n = blocks[j]
                if n.page != b.page or ADM_HEAD.match(n.text.strip()) or _is_noise(n):
                    break
                if n.label in {"caption", "table"} or step_marker(n.text) or PARA_NUM.match(
                        n.text):
                    break
                # body is indented relative to / aligned with the heading's column;
                # a following paragraph that starts clearly left of the body ends it.
                if parts and n.x < blocks[i + 1].x - 12 and _ends_sentence(parts[-1]):
                    break
                if parts and n.label == "section_header" and n.text.upper() != n.text:
                    break
                ids.append(n.id)
                parts.append(n.text)
                j += 1
            if parts:
                out.append({"kind": kind, "block_ids": ids, "text": " ".join(parts)})
            i = j
            continue
        # several admonitions merged into one block: "WARNING <body> WARNING <body> ..."
        if ADM_LEADING.match(b.text):
            pieces = ADM_SPLIT.split(b.text)
            for kind, body in zip(pieces[1::2], pieces[2::2], strict=True):
                if body.strip():
                    out.append({"kind": kind.lower(), "block_ids": [b.id],
                                "text": body.strip()})
            i += 1
            continue
        # inline "CAUTION: ..." inside a block (e.g. figure step lists)
        for m in ADM_INLINE.finditer(b.text):
            rest = b.text[m.end():]
            stop = STEP_INLINE.search(rest)
            text = rest[: stop.start()] if stop else rest
            if text.strip():
                out.append({"kind": m.group(1).lower(), "block_ids": [b.id],
                            "text": text.strip()})
        i += 1
    return out


def _split_inline_steps(blocks: list[_B]) -> list[_B]:
    """Split blocks containing several 'STEP n.' markers, drop inline admonitions."""
    out = []
    for b in blocks:
        text = b.text
        starts = [m.start() for m in STEP_INLINE.finditer(text)]
        if len(starts) <= 1 and not ADM_INLINE.search(text):
            out.append(b)
            continue
        pieces = []
        bounds = starts + [len(text)]
        if starts and starts[0] > 0:
            pieces.append(text[: starts[0]])
        for s, e in zip(bounds, bounds[1:], strict=False):
            pieces.append(text[s:e])
        if not starts:
            pieces = [text]
        for p in pieces:
            p = ADM_INLINE.split(p)[0].strip()
            if p:
                out.append(_B(b.id, b.page, b.label, p, b.x, b.y, b.w))
    return out


def _heading_flags(blocks: list[_B]) -> list[bool]:
    flags = []
    for i, b in enumerate(blocks):
        text = b.text.strip()
        nxt = next((n.text for n in blocks[i + 1:] if not _is_noise(n)), "")
        mk = step_marker(text)
        if mk:
            flags.append(is_heading_item(mk[0], mk[2], nxt))
        else:
            flags.append((b.label in {"section_header", "title"} and not ADM_HEAD.match(text)
                          and not _is_noise(b)) or bool(PARA_NUM.match(text)))
    return flags


def label_procedures(blocks: list[_B]) -> list[dict]:
    blocks = _split_inline_steps(blocks)
    heads = _heading_flags(blocks)
    adm_ids = set()
    for a in label_admonitions(blocks):
        adm_ids.update(a["block_ids"])
    procedures = []
    for style, _rx in STEP_STYLES:
        run: list[dict] = []

        def close():
            nonlocal run
            if len(run) >= 2:
                procedures.append({"style": style, "steps": run, "start": run[0]["_idx"]})
            run = []

        def next_marker(idx: int):
            for n in blocks[idx + 1:]:
                if not _is_noise(n):
                    return step_marker(n.text.strip())
            return None

        for idx, b in enumerate(blocks):
            text = b.text.strip()
            mk = None if heads[idx] else step_marker(text)
            if mk and mk[0] == style:
                n = _ordinal(style, mk[1])
                expected = _ordinal(style, run[-1]["number"]) + 1 if run else 1
                if n != expected:
                    close()
                if n == expected or n == 1:
                    run.append({"number": mk[1], "block_ids": [b.id], "text": mk[2],
                                "_idx": idx})
                continue
            if not run:
                continue
            last_step = blocks[run[-1]["_idx"]]
            if heads[idx] and not ADM_HEAD.match(text):
                if b.page == last_step.page and b.y < last_step.y:
                    continue  # running header placed late in reading order
                close()
                continue
            if mk or _is_noise(b) or b.id in adm_ids or ADM_HEAD.match(text) \
                    or b.label in {"caption", "table"} or len(text.split()) <= 3:
                continue  # transparent: other step styles, page furniture, figure labels
            prev = run[-1]
            nm = next_marker(idx)
            sandwiched = (nm is not None and nm[0] == style and len(text.split()) <= 15
                          and _ordinal(style, nm[1]) == _ordinal(style, prev["number"]) + 1)
            if not _ends_sentence(prev["text"]) or text[:1].islower() or sandwiched:
                prev["text"] += " " + text
                prev["block_ids"].append(b.id)
                continue
            close()
        close()

    out = []
    for proc in sorted(procedures, key=lambda p: p["start"]):
        title = _procedure_title(blocks, heads, proc)
        out.append({"title": title, "steps": [{k: v for k, v in s.items() if k != "_idx"}
                                              for s in proc["steps"]]})
    return out


def _clean_title(text: str, last_sentence: bool = False) -> str:
    mk = step_marker(text)
    if mk:
        text = mk[2]
    text = re.sub(r"^(figure|fig\.)\s*[\d\-.]+\s*", "", text, flags=re.I)
    text = PARA_NUM.sub("", text)
    text = re.sub(r"\s*\((fig|figure)\b[^)]*\)", "", text, flags=re.I)
    sentences = re.split(r"(?<=[a-z0-9)])[.:]\s", text)
    text = sentences[-1] if last_sentence else sentences[0]
    return text.strip().rstrip(":.").strip()


def _procedure_title(blocks: list[_B], heads: list[bool], proc: dict) -> str:
    """Caption right after the steps, else a lead-in sentence ending in ':', else the
    nearest preceding heading."""
    last = proc["steps"][-1]["_idx"]
    nxt = next((blocks[k] for k in range(last + 1, len(blocks)) if not _is_noise(blocks[k])),
               None)
    if nxt is not None and nxt.page == blocks[last].page and (
            nxt.label == "caption" or re.match(r"(figure|fig\.)\s*\d", nxt.text, re.I)):
        return _clean_title(nxt.text)
    first = proc["start"]
    prev_idx = next((k for k in range(first - 1, -1, -1) if not _is_noise(blocks[k])), None)
    prev = blocks[prev_idx] if prev_idx is not None else None
    if prev is not None and prev.text.rstrip().endswith(":"):
        return _clean_title(prev.text, last_sentence=not heads[prev_idx])
    for k in range(first - 1, max(-1, first - 10), -1):
        if heads[k] and blocks[k].label != "caption":
            cleaned = _clean_title(blocks[k].text)
            if cleaned:
                return cleaned
    return _clean_title(prev.text) if prev is not None else "Procedure"


# ---- parts tables

COLS = [
    ("smr", re.compile(r"\bsmr\b", re.I)),
    ("nsn", re.compile(r"national stock|\bnsn\b|stock number", re.I)),
    ("item", re.compile(r"s\wmbol|illus|item no|^item$", re.I)),
    ("quantity", re.compile(r"\bqty\b|quan", re.I)),
    ("unit", re.compile(r"\bu/?i\b|unit of issue", re.I)),
    ("part_number", re.compile(r"drawing|part n", re.I)),
    ("description", re.compile(r"descr", re.I)),
]
NSN = re.compile(r"\b\d{4,5}-?\d{2,4}-?\d{3}-?\d{3,4}\b|\b\d{4,5}-\d{2,5}-\d{4}\b")
CAGE = re.compile(r"\((\d{5})\)")


def _column_roles(header_rows: list[list[str]]) -> dict[int, str]:
    roles: dict[int, str] = {}
    n_cols = max(len(r) for r in header_rows)
    for c in range(n_cols):
        head = " ".join(r[c] for r in header_rows if c < len(r))
        for role, rx in COLS:
            if rx.search(head) and role not in roles.values():
                if role == "part_number" and re.search(r"descr", head, re.I):
                    continue
                roles[c] = role
                break
    return roles


def _split_description(desc: str) -> dict:
    """'GASKET, Plug 694203 (72582)' -> description, part_number, cage."""
    out: dict = {"description": desc}
    m = None
    for m in CAGE.finditer(desc):
        pass
    if not m:
        return out
    before, after = desc[: m.start()].strip(), desc[m.end():].strip()
    out["cage"] = m.group(1)
    if after and re.search(r"\d", after):
        out["part_number"] = after
        out["description"] = before
        return out
    toks = before.split()
    if len(toks) >= 2 and re.search(r"\d", toks[-1]):
        out["part_number"] = toks[-1].rstrip(",")
        out["description"] = " ".join(toks[:-1]).rstrip(",")
    else:
        out["description"] = before
    if after:
        out["description"] = (out["description"] + " " + after).strip()
    return out


DESIGNATOR = re.compile(r"\b(?:[A-Z0]-\d{3}|thru|and)\b")
CODE = re.compile(r"\b[A-Z0-9]*\d[A-Z0-9]*(?:-[A-Z0-9]+)*\b|\b[A-Z]+(?:-[A-Z0-9]+)+\b")


def _clean_item(s: str) -> str | None:
    toks = DESIGNATOR.findall(s)
    while toks and toks[0] in ("thru", "and"):
        toks.pop(0)
    return " ".join(toks) or None


def _clean_code(s: str) -> str | None:
    """First code-like token of an OCR'd drawing/part-number cell ('. PM-1306' -> PM-1306)."""
    m = CODE.search(s)
    return m.group(0) if m and len(m.group(0)) >= 3 else None


LEADING_CAGE = re.compile(r"^(\(\d{5}\))\s*")
ITEM_NAME = re.compile(r"(?<![(\w])[A-Z]{3,}(?:[ -][A-Z]{3,})*(?=,|\s[A-Z][a-z]|\s[A-Z]+\b)")


def label_parts(payload: dict) -> list[dict]:
    out = []
    for t in payload["tables"]:
        rows = t["rows"]
        if not rows:
            continue
        # header rows: leading rows until one contains a description-like column match
        header_end = 0
        roles: dict[int, str] = {}
        for h in range(min(4, len(rows))):
            r = _column_roles(rows[: h + 1])
            if "description" in r.values() and len(r) >= 2 and len(r) > len(roles):
                roles, header_end = r, h + 1
                if len(r) >= 3:
                    break
        if not roles:
            continue
        raw: list[dict] = []
        for ri in range(header_end, len(rows)):
            row = rows[ri]
            rec: dict = {"table_id": t["id"], "row": ri}
            for c, role in roles.items():
                v = " ".join(row[c].split()) if c < len(row) else ""
                if v and v not in ("-", "—"):
                    rec[role] = v
            raw.append(rec)
        # Table structure sometimes shifts a wrapped "(CAGE)" line into the next row.
        for prev, rec in zip(raw, raw[1:], strict=False):
            m = LEADING_CAGE.match(rec.get("description", ""))
            if m and prev.get("description") and not CAGE.search(prev["description"]):
                prev["description"] += " " + m.group(1)
                rec["description"] = rec["description"][m.end():].strip()
        for rec in raw:
            desc = re.sub(r"^[^A-Za-z0-9\"(]+", "", rec.get("description", ""))
            if not desc:
                continue
            rec["description"] = desc
            if rec.get("item"):
                rec["item"] = _clean_item(rec["item"])
            if "part_number" in roles.values() and rec.get("part_number"):
                rec["part_number"] = _clean_code(rec["part_number"])
            # Drop a group heading merged in front of the item name:
            # "Fuel Filter Secondary ELEMENT, Fuel Filter" -> "ELEMENT, Fuel Filter"
            m = ITEM_NAME.search(desc)
            if m and m.start() > 0 and re.search(r"[a-z]", desc[: m.start()]):
                desc = rec["description"] = desc[m.start():]
            if "cage" not in rec:
                rec.update({k: v for k, v in _split_description(desc).items()
                            if k not in rec or k == "description"})
            if "nsn" in rec:
                m = NSN.search(rec["nsn"])
                rec["nsn"] = m.group(0) if m else rec["nsn"]
            if not (rec.get("nsn") or rec.get("part_number") or rec.get("item")):
                continue  # group headings like "Fuel Filter Secondary"
            if rec.get("quantity"):
                nums = re.findall(r"\d+", rec["quantity"])
                rec["quantity"] = nums[0] if len(nums) == 1 else None
            out.append(rec)
    return out


def label_payload(payload: dict) -> dict:
    blocks = _blocks(payload)
    return {
        "admonitions": label_admonitions(blocks),
        "procedures": label_procedures(blocks),
        "parts": label_parts(payload),
    }


# ---------------------------------------------------------------- labeling -> records


def _provenance(doc: ParsedDocument, block_ids: list[str]) -> list[Provenance]:
    by_id = {b.id: b for b in doc.blocks}
    per_page: dict[int, tuple[BBox, list[str]]] = {}
    for bid in dict.fromkeys(block_ids):
        b = by_id.get(bid)
        if b is None:
            raise KeyError(bid)
        if b.page in per_page:
            box, ids = per_page[b.page]
            per_page[b.page] = (box.union(b.bbox), ids + [bid])
        else:
            per_page[b.page] = (b.bbox, [bid])
    return [Provenance(doc_id=doc.doc_id, page=p, bbox=box, block_ids=ids)
            for p, (box, ids) in per_page.items()]


def _row_provenance(doc: ParsedDocument, table_id: str, row: int) -> list[Provenance]:
    t = next(t for t in doc.tables if t.id == table_id)
    boxes = [c.bbox for c in t.cells if c.row == row and c.bbox is not None]
    if not boxes:
        raise KeyError(f"{table_id} row {row}")
    box = boxes[0]
    for b in boxes[1:]:
        box = box.union(b)
    return [Provenance(doc_id=doc.doc_id, page=t.page, bbox=box,
                       block_ids=[f"{table_id}:r{row}"])]


def _grounded(text: str, doc: ParsedDocument, block_ids: list[str]) -> bool:
    from .evaluate import grounding_score

    by_id = {b.id: b for b in doc.blocks}
    source = " ".join(by_id[i].text for i in block_ids if i in by_id)
    return grounding_score(text, source) >= 0.8


def _strip_number(text: str) -> str:
    mk = step_marker(text.strip())
    return mk[2] if mk else text.strip()


def build_result(doc: ParsedDocument, labeling: dict, extractor: str) -> tuple[
        ExtractionResult, list[str]]:
    """Validate a labeling into records. Returns the result and a list of rejections."""
    rejected: list[str] = []
    adms, procs, parts = [], [], []
    for i, a in enumerate(labeling.get("admonitions", [])):
        try:
            if not _grounded(a["text"], doc, a["block_ids"]):
                raise ValueError("text not grounded in cited blocks")
            adms.append(Admonition(id=f"adm{i + 1}", kind=a["kind"].lower(), text=a["text"],
                                   provenance=_provenance(doc, a["block_ids"])))
        except (KeyError, ValueError, ValidationError, AttributeError) as e:
            rejected.append(f"admonition {i}: {e}")
    for i, p in enumerate(labeling.get("procedures", [])):
        try:
            steps = []
            for s in p["steps"]:
                text = _strip_number(s["text"])
                if not _grounded(text, doc, s["block_ids"]):
                    raise ValueError(f"step {s.get('number')} not grounded")
                steps.append(ProcedureStep(number=str(s["number"]), text=text,
                                           provenance=_provenance(doc, s["block_ids"])))
            all_ids = [bid for s in p["steps"] for bid in s["block_ids"]]
            procs.append(Procedure(id=f"proc{i + 1}", title=p.get("title") or "Procedure",
                                   steps=steps, provenance=_provenance(doc, all_ids)))
        except (KeyError, ValueError, ValidationError, AttributeError, TypeError) as e:
            rejected.append(f"procedure {i}: {e}")
    for i, r in enumerate(labeling.get("parts", [])):
        try:
            fields = {k: (str(r[k]).strip() or None) if r.get(k) is not None else None
                      for k in ("description", "item", "nsn", "part_number", "cage",
                                "quantity", "unit", "smr")}
            parts.append(Part(id=f"part{i + 1}", **fields,
                              provenance=_row_provenance(doc, r["table_id"], int(r["row"]))))
        except (KeyError, ValueError, ValidationError, StopIteration, TypeError) as e:
            rejected.append(f"part {i}: {e}")
    result = ExtractionResult(doc_id=doc.doc_id, kind=doc.kind, title=doc.title,
                              extractor=extractor, admonitions=adms, procedures=procs,
                              parts=parts)
    return result, rejected


def extract_document(doc: ParsedDocument, mode: str, provider=None) -> ExtractionResult:
    if mode == "rules":
        labeling = label_payload(make_payload(doc))
        result, rejected = build_result(doc, labeling, "rules")
    elif mode == "llm":
        system, user = build_prompt(doc)
        labeling = provider.complete_json(system, user)
        result, rejected = build_result(doc, labeling, f"llm-{provider.name}")
    else:
        raise ValueError(mode)
    for r in rejected:
        print(f"  rejected {r}")
    return result

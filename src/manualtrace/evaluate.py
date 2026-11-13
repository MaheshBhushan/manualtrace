"""Evaluation against the hand-annotated gold set.

Three checks:

1. Field-level precision/recall. Extracted records on the annotated pages are aligned
   one-to-one with gold records of the same type on the same page (greedy, by text
   similarity >= ALIGN_MIN). For every field, a true positive is an aligned pair whose
   values agree; precision divides by the non-empty fields the extractor produced on gold
   pages, recall by the non-empty gold fields. Free-text fields agree when their normalized
   similarity is >= TEXT_MATCH (OCR tolerance); codes must agree exactly after
   normalization.
2. Provenance: for aligned records, the provenance page must equal the gold page, and the
   provenance bbox must actually contain the record's text in the source PDF (text layer
   for digital PDFs, Tesseract on the cropped region for scans).
3. Retrieval: BM25 over the extracted records answers the queries in gold/queries.json;
   a hit is a retrieved record aligned with one of the query's relevant gold records.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .schemas import ExtractionResult, Provenance

ALIGN_MIN = 0.5
TEXT_MATCH = 0.9
TITLE_MATCH = 0.8
GROUNDING_MIN = 0.8

TEXT_FIELDS = {"text", "description"}
PART_FIELDS = ["item", "nsn", "part_number", "cage", "quantity", "unit", "smr", "description"]
FIELDS = {
    "admonition": ["kind", "text"],
    "step": ["number", "text"],
    "procedure": ["title"],
    "part": PART_FIELDS,
}


# ---------------------------------------------------------------- normalization


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
    s = s.replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s.strip(" .,;:")


def norm_code(s: str) -> str:
    return re.sub(r"\s+", "", norm_text(s))


def similarity(a: str, b: str) -> float:
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _alts(v) -> list[str]:
    if v is None:
        return []
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def field_agrees(name: str, pred: str | None, gold) -> bool:
    if pred is None or gold is None:
        return False
    if name == "title":
        p = norm_text(pred)
        return any(
            similarity(pred, g) >= TITLE_MATCH or (len(p) > 3 and (p in norm_text(g)
                                                                   or norm_text(g) in p))
            for g in _alts(gold)
        )
    if name in TEXT_FIELDS:
        return any(similarity(pred, g) >= TEXT_MATCH for g in _alts(gold))
    return any(norm_code(pred) == norm_code(g) for g in _alts(gold))


# ---------------------------------------------------------------- flattened records


@dataclass
class Rec:
    rtype: str
    id: str
    page: int
    fields: dict[str, object]
    key: str
    provenance: list[Provenance] = field(default_factory=list)
    parent: str | None = None  # procedure id for steps


def part_key(fields: dict) -> str:
    vals = [_alts(fields.get(k))[:1] for k in ("description", "item", "nsn", "part_number")]
    return " ".join(v[0] for v in vals if v)


def gold_records(gold: dict) -> dict[str, list[Rec]]:
    out: dict[str, list[Rec]] = defaultdict(list)
    for a in gold["admonitions"]:
        out["admonition"].append(Rec("admonition", a["id"], a["page"],
                                     {"kind": a["kind"], "text": a["text"]}, a["text"]))
    for p in gold["procedures"]:
        out["procedure"].append(Rec("procedure", p["id"], p["page"], {"title": p["title"]},
                                    " ".join(s["text"] for s in p["steps"])))
        for i, s in enumerate(p["steps"]):
            out["step"].append(Rec("step", f"{p['id']}.{i}", s["page"],
                                   {"number": s["number"], "text": s["text"]}, s["text"],
                                   parent=p["id"]))
    for r in gold["parts"]:
        fields = {k: r.get(k) for k in PART_FIELDS}
        out["part"].append(Rec("part", r["id"], r["page"], fields, part_key(fields)))
    return out


def pred_records(res: ExtractionResult) -> dict[str, list[Rec]]:
    out: dict[str, list[Rec]] = defaultdict(list)
    for a in res.admonitions:
        out["admonition"].append(Rec("admonition", a.id, a.provenance[0].page,
                                     {"kind": a.kind, "text": a.text}, a.text, a.provenance))
    for p in res.procedures:
        out["procedure"].append(Rec("procedure", p.id, p.steps[0].provenance[0].page,
                                    {"title": p.title}, " ".join(s.text for s in p.steps),
                                    p.provenance))
        for i, s in enumerate(p.steps):
            out["step"].append(Rec("step", f"{p.id}.{i}", s.provenance[0].page,
                                   {"number": s.number, "text": s.text}, s.text,
                                   s.provenance, parent=p.id))
    for r in res.parts:
        fields = {k: getattr(r, k) for k in PART_FIELDS}
        out["part"].append(Rec("part", r.id, r.provenance[0].page, fields, part_key(fields),
                               r.provenance))
    return out


def align(preds: list[Rec], golds: list[Rec]) -> dict[str, str]:
    """Greedy one-to-one alignment pred.id -> gold.id on the same page."""
    pairs = []
    for p in preds:
        for g in golds:
            if p.page == g.page:
                s = similarity(p.key, g.key)
                if s >= ALIGN_MIN:
                    pairs.append((s, p.id, g.id))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), {}
    for _s, pid, gid in pairs:
        if pid not in used_p and gid not in used_g:
            out[pid] = gid
            used_p.add(pid)
            used_g.add(gid)
    return out


def align_procedures(pred: dict[str, list[Rec]], gold: dict[str, list[Rec]],
                     step_map: dict[str, str]) -> dict[str, str]:
    """Procedures align through their steps: pick the gold procedure sharing most steps."""
    gold_parent = {g.id: g.parent for g in gold["step"]}
    votes: dict[str, Counter] = defaultdict(Counter)
    for s in pred["step"]:
        if s.id in step_map:
            votes[s.parent][gold_parent[step_map[s.id]]] += 1
    pairs = sorted(((c, pid, gid) for pid, cnt in votes.items() for gid, c in cnt.items()),
                   reverse=True)
    used_p, used_g, out = set(), set(), {}
    for _c, pid, gid in pairs:
        if pid not in used_p and gid not in used_g:
            out[pid] = gid
            used_p.add(pid)
            used_g.add(gid)
    return out


# ---------------------------------------------------------------- grounding


def _bbox_text_digital(pdf: Path, page: int, prov: Provenance) -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    pg = doc[page - 1]
    h = pg.get_height()
    b = prov.bbox
    tp = pg.get_textpage()
    text = tp.get_text_bounded(left=b.left - 2, bottom=h - b.bottom - 2, right=b.right + 2,
                               top=h - b.top + 2)
    tp.close()
    pg.close()
    doc.close()
    return text


def _bbox_text_ocr(pdf: Path, page: int, prov: Provenance) -> str:
    import pypdfium2 as pdfium

    scale = 300 / 72
    doc = pdfium.PdfDocument(str(pdf))
    img = doc[page - 1].render(scale=scale).to_pil()
    doc.close()
    b = prov.bbox
    crop = img.crop((max(0, int((b.left - 4) * scale)), max(0, int((b.top - 4) * scale)),
                     int((b.right + 4) * scale), int((b.bottom + 4) * scale)))
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        crop.save(f.name)
        r = subprocess.run(["tesseract", f.name, "-", "--psm", "6"], capture_output=True,
                           text=True, check=True)
    return r.stdout


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm_text(s))


def grounding_score(text: str, region_text: str) -> float:
    """Fraction of the record's word tokens found in the text of its provenance region."""
    want = [t for t in tokens(text) if len(t) > 1]
    have = Counter(t for t in tokens(region_text))
    if not want:
        return 1.0
    hit = 0
    for t in want:
        if have[t] > 0:
            have[t] -= 1
            hit += 1
    return hit / len(want)


def record_text(r: Rec) -> str:
    if r.rtype == "part":
        return " ".join(str(v) for k, v in r.fields.items() if v and k != "smr")
    return str(r.fields.get("text", ""))


# ---------------------------------------------------------------- retrieval


STOP = set("a an the of to for and or in on at by with from is are be it its this that "
           "how do i can what which who when where long before after into".split())


def _stem(t: str) -> str:
    for suf in ("ing", "es", "ed", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def bm25_tokens(s: str) -> list[str]:
    return [_stem(t) for t in tokens(s) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = [bm25_tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query: str, k: int) -> list[int]:
        q = bm25_tokens(query)
        scores = []
        for i, (tf, d) in enumerate(zip(self.tf, self.docs, strict=True)):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            scores.append((s, -i))
        scores.sort(reverse=True)
        return [-i for s, i in scores[:k] if s > 0]


def retrieval_units(recs: dict[str, list[Rec]]) -> list[tuple[str, str, str]]:
    """(record type, record id, indexed text) for admonitions, whole procedures and parts."""
    units = [("admonition", r.id, record_text(r)) for r in recs["admonition"]]
    for p in recs["procedure"]:
        steps = [s for s in recs["step"] if s.parent == p.id]
        title = " ".join(_alts(p.fields["title"])[:1])
        units.append(("procedure", p.id, title + " " + " ".join(record_text(s) for s in steps)))
    units += [("part", r.id, record_text(r)) for r in recs["part"]]
    return units


def run_retrieval(queries: list[dict], units: list[tuple[str, str, str, set[str]]],
                  ks=(1, 3, 5)) -> dict:
    """units: (doc_id, rtype, rec_id, text, gold ids it counts as)."""
    index = BM25([u[3] for u in units])
    hits = {k: 0 for k in ks}
    rr = 0.0
    per_query = []
    for q in queries:
        relevant = set(q["relevant"])
        ranked = index.search(q["q"], max(ks))
        rank = next((i + 1 for i, u in enumerate(ranked) if units[u][4] & relevant), None)
        for k in ks:
            hits[k] += int(rank is not None and rank <= k)
        rr += 1 / rank if rank else 0.0
        per_query.append({"q": q["q"], "rank": rank})
    n = len(queries)
    return {**{f"hit@{k}": round(hits[k] / n, 4) for k in ks}, "mrr": round(rr / n, 4),
            "n_queries": n, "per_query": per_query}


# ---------------------------------------------------------------- driver


def _pr(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp,
            "n_pred": n_pred, "n_gold": n_gold}


def evaluate_doc(res: ExtractionResult, gold: dict, raw_pdf: Path | None, kind: str) -> dict:
    gold_pages = set(gold["pages"])
    pred_all = pred_records(res)
    pred = {t: [r for r in rs if r.page in gold_pages] for t, rs in pred_all.items()}
    for t in FIELDS:
        pred.setdefault(t, [])
    grecs = gold_records(gold)
    maps = {t: align(pred[t], grecs[t]) for t in ("admonition", "step", "part")}
    maps["procedure"] = align_procedures(pred, grecs, maps["step"])

    fields_out: dict[str, dict] = {}
    totals = Counter()
    for rtype, names in FIELDS.items():
        gold_by_id = {g.id: g for g in grecs[rtype]}
        for name in names:
            n_pred = sum(1 for r in pred[rtype] if r.fields.get(name) not in (None, ""))
            n_gold = sum(1 for g in grecs[rtype] if g.fields.get(name) not in (None, ""))
            tp = 0
            for r in pred[rtype]:
                gid = maps[rtype].get(r.id)
                if gid and field_agrees(name, r.fields.get(name), gold_by_id[gid].fields.get(name)):
                    tp += 1
            fields_out[f"{rtype}.{name}"] = _pr(tp, n_pred, n_gold)
            totals["tp"] += tp
            totals["n_pred"] += n_pred
            totals["n_gold"] += n_gold
    records = {t: _pr(len(maps[t]), len(pred[t]), len(grecs[t])) for t in FIELDS}

    # provenance: page agreement + bbox grounding for aligned records
    gold_page = {t: {g.id: g.page for g in grecs[t]} for t in FIELDS}
    page_ok = page_n = 0
    grounding = []
    for rtype in ("admonition", "step", "part"):
        for r in pred[rtype]:
            gid = maps[rtype].get(r.id)
            if not gid:
                continue
            page_n += 1
            page_ok += int(r.provenance[0].page == gold_page[rtype][gid])
            if raw_pdf is not None:
                fn = _bbox_text_digital if kind == "digital" else _bbox_text_ocr
                region = " ".join(fn(raw_pdf, p.page, p) for p in r.provenance)
                grounding.append(grounding_score(record_text(r), region))
    prov = {
        "page_accuracy": round(page_ok / page_n, 4) if page_n else 0.0,
        "n_aligned": page_n,
    }
    if grounding:
        prov["grounded_fraction"] = round(
            sum(g >= GROUNDING_MIN for g in grounding) / len(grounding), 4)
        prov["mean_token_coverage"] = round(sum(grounding) / len(grounding), 4)

    return {
        "fields": fields_out,
        "micro": _pr(totals["tp"], totals["n_pred"], totals["n_gold"]),
        "records": records,
        "provenance": prov,
        "alignment": {t: maps[t] for t in FIELDS},
        "_pred": pred_all,
        "_gold": grecs,
    }


def _units_with_gold(doc_id: str, recs: dict[str, list[Rec]], maps: dict[str, dict],
                     gold_steps: dict[str, str] | None) -> list[tuple]:
    out = []
    for rtype, rid, text in retrieval_units(recs):
        if gold_steps is None:  # gold index: ids are gold ids already
            ids = {f"{doc_id}:{rid}"}
        else:
            ids = set()
            if rid in maps[rtype]:
                ids.add(f"{doc_id}:{maps[rtype][rid]}")
            if rtype == "procedure":
                for s in recs["step"]:
                    if s.parent == rid and s.id in maps["step"]:
                        ids.add(f"{doc_id}:{gold_steps[maps['step'][s.id]]}")
        out.append((doc_id, rtype, rid, text, ids))
    return out


def evaluate_all(data_dir: Path, out_dir: Path, extractor: str, doc_ids: list[str],
                 gold_dir: Path, grounding: bool = True) -> dict:
    from .sources import SOURCES, raw_path

    report: dict = {"extractor": extractor, "docs": {}}
    pred_units, gold_units = [], []
    totals = Counter()
    for doc_id in doc_ids:
        res = ExtractionResult.model_validate_json(
            (out_dir / extractor / f"{doc_id}.json").read_text())
        gold = json.loads((gold_dir / f"{doc_id}.json").read_text())
        raw = raw_path(data_dir, doc_id) if grounding else None
        if raw is not None and not raw.exists():
            raw = None
        ev = evaluate_doc(res, gold, raw, SOURCES[doc_id].kind)
        pred_all, grecs = ev.pop("_pred"), ev.pop("_gold")
        gold_steps = {g.id: g.parent for g in grecs["step"]}
        pred_units += _units_with_gold(doc_id, pred_all, ev["alignment"], gold_steps)
        gold_units += _units_with_gold(doc_id, grecs, {}, None)
        report["docs"][doc_id] = ev
        for k in ("tp", "n_pred", "n_gold"):
            totals[k] += ev["micro"][k]
    report["micro"] = _pr(totals["tp"], totals["n_pred"], totals["n_gold"])
    queries = json.loads((gold_dir / "queries.json").read_text())["queries"]
    queries = [q for q in queries if q["relevant"][0].split(":")[0] in doc_ids]
    report["retrieval"] = run_retrieval(queries, pred_units)
    report["retrieval_gold_index"] = run_retrieval(queries, gold_units)
    return report


def format_report(report: dict) -> str:
    lines = [f"extractor: {report['extractor']}"]
    for doc_id, ev in report["docs"].items():
        m = ev["micro"]
        lines.append(f"\n== {doc_id}  micro P={m['precision']:.3f} R={m['recall']:.3f} "
                     f"F1={m['f1']:.3f}")
        lines.append(f"{'field':24s} {'P':>6s} {'R':>6s} {'F1':>6s} {'pred':>5s} {'gold':>5s}")
        for name, v in ev["fields"].items():
            lines.append(f"{name:24s} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} "
                         f"{v['n_pred']:5d} {v['n_gold']:5d}")
        for t, v in ev["records"].items():
            lines.append(f"records.{t:16s} {v['precision']:6.3f} {v['recall']:6.3f} "
                         f"{v['f1']:6.3f} {v['n_pred']:5d} {v['n_gold']:5d}")
        lines.append(f"provenance: {ev['provenance']}")
    m = report["micro"]
    lines.append(f"\nALL micro P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    for name in ("retrieval", "retrieval_gold_index"):
        r = report[name]
        lines.append(f"{name}: hit@1={r['hit@1']:.3f} hit@3={r['hit@3']:.3f} "
                     f"hit@5={r['hit@5']:.3f} MRR={r['mrr']:.3f} (n={r['n_queries']})")
    return "\n".join(lines)

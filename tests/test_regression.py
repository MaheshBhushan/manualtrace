"""Regression thresholds on the real manuals.

Needs the parsed Docling output (`manualtrace fetch && manualtrace parse`). Data directory:
$MANUALTRACE_DATA (default ./data). Tests skip when it is missing, unless
MANUALTRACE_REQUIRE_PIPELINE=1 (set in CI), in which case they fail.

Thresholds sit a little below the values measured when they were set (see README); a
change that lowers extraction quality below them fails CI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from lxml import etree

from manualtrace.cli import GOLD_DIR
from manualtrace.evaluate import evaluate_all
from manualtrace.extract import extract_document
from manualtrace.providers import OfflineProvider
from manualtrace.s1000d import PROV_NS, export_document
from manualtrace.schemas import ParsedDocument
from manualtrace.sources import SOURCES, raw_path

pytestmark = pytest.mark.pipeline

DATA = Path(os.environ.get("MANUALTRACE_DATA", "data"))
DOCS = list(SOURCES)

# per document: (micro precision, micro recall)
FIELD_THRESHOLDS = {
    "tm55-1905-217-12": (0.80, 0.80),
    "navships-900524-ib": (0.60, 0.60),
}
RECORD_RECALL_MIN = {
    "tm55-1905-217-12": {"admonition": 0.8, "step": 0.8, "part": 0.8},
    "navships-900524-ib": {"admonition": 0.6, "step": 0.6, "part": 0.5},
}
RETRIEVAL_HIT3_MIN = 0.6
RETRIEVAL_MRR_MIN = 0.5
PAGE_ACCURACY_MIN = 0.95
GROUNDED_MIN = {"tm55-1905-217-12": 0.9, "navships-900524-ib": 0.7}


def _parsed(doc_id: str) -> ParsedDocument:
    path = DATA / "parsed" / f"{doc_id}.json"
    if not path.exists():
        if os.environ.get("MANUALTRACE_REQUIRE_PIPELINE") == "1":
            pytest.fail(f"{path} missing; run `manualtrace fetch && manualtrace parse`")
        pytest.skip(f"{path} missing; run `manualtrace fetch && manualtrace parse`")
    return ParsedDocument.model_validate_json(path.read_text())


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    results = {}
    for doc_id in DOCS:
        res = extract_document(_parsed(doc_id), "rules")
        (out / "rules").mkdir(exist_ok=True)
        (out / "rules" / f"{doc_id}.json").write_text(res.model_dump_json())
        results[doc_id] = res
    report = evaluate_all(DATA, out, "rules", DOCS, GOLD_DIR,
                          grounding=all(raw_path(DATA, d).exists() for d in DOCS))
    return out, results, report


def test_both_document_kinds_present():
    kinds = {SOURCES[d].kind for d in DOCS}
    assert kinds == {"scanned", "digital"}


@pytest.mark.parametrize("doc_id", DOCS)
def test_records_are_nonempty_and_in_page_bounds(run, doc_id):
    _out, results, _ = run
    res = results[doc_id]
    assert res.admonitions and res.procedures and res.parts
    pages = {p.page: p for p in _parsed(doc_id).pages}
    for rec in [*res.admonitions, *res.procedures, *res.parts]:
        for prov in rec.provenance:
            pg = pages[prov.page]
            assert 0 <= prov.bbox.left < prov.bbox.right <= pg.width + 1
            assert 0 <= prov.bbox.top < prov.bbox.bottom <= pg.height + 1


@pytest.mark.parametrize("doc_id", DOCS)
def test_s1000d_export(run, doc_id, tmp_path):
    _out, results, _ = run
    files = export_document(results[doc_id], SOURCES[doc_id], tmp_path)  # XSD-validated
    assert len(files) == 3
    info_codes = set()
    for f in files:
        root = etree.parse(str(f)).getroot()  # well-formed
        info_codes.add(root.find(".//dmCode").get("infoCode"))
        for el in root.iter():
            if el.tag in ("warning", "caution", "note", "catalogSeqNumber"):
                assert el.get(f"{{{PROV_NS}}}page") and el.get(f"{{{PROV_NS}}}bbox")
    assert info_codes == {"012", "300", "941"}


@pytest.mark.parametrize("doc_id", DOCS)
def test_field_precision_recall(run, doc_id):
    report = run[2]
    micro = report["docs"][doc_id]["micro"]
    p_min, r_min = FIELD_THRESHOLDS[doc_id]
    assert micro["precision"] >= p_min, json.dumps(report["docs"][doc_id]["fields"], indent=1)
    assert micro["recall"] >= r_min, json.dumps(report["docs"][doc_id]["fields"], indent=1)


@pytest.mark.parametrize("doc_id", DOCS)
def test_record_recall(run, doc_id):
    records = run[2]["docs"][doc_id]["records"]
    for rtype, r_min in RECORD_RECALL_MIN[doc_id].items():
        assert records[rtype]["recall"] >= r_min, (rtype, records[rtype])


@pytest.mark.parametrize("doc_id", DOCS)
def test_provenance(run, doc_id):
    prov = run[2]["docs"][doc_id]["provenance"]
    assert prov["page_accuracy"] >= PAGE_ACCURACY_MIN, prov
    if "grounded_fraction" in prov:
        assert prov["grounded_fraction"] >= GROUNDED_MIN[doc_id], prov


def test_retrieval(run):
    r = run[2]["retrieval"]
    assert r["hit@3"] >= RETRIEVAL_HIT3_MIN, r["per_query"]
    assert r["mrr"] >= RETRIEVAL_MRR_MIN, r["per_query"]


@pytest.mark.parametrize("doc_id", DOCS)
def test_offline_llm_path_matches_rules(run, doc_id):
    llm = extract_document(_parsed(doc_id), "llm", OfflineProvider())
    rules = run[1][doc_id]
    assert llm.model_dump(exclude={"extractor"}) == rules.model_dump(exclude={"extractor"})

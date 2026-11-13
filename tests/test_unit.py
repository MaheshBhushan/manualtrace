"""Fast tests on synthetic layouts; no downloads, Docling or OCR needed."""

from __future__ import annotations

import datetime as dt

import pytest
from lxml import etree

from manualtrace.evaluate import BM25, evaluate_doc, grounding_score, similarity
from manualtrace.extract import (
    _split_description,
    build_result,
    extract_document,
    label_parts,
    label_payload,
    make_payload,
    step_marker,
)
from manualtrace.providers import (
    AnthropicProvider,
    OfflineProvider,
    decode_payload,
    embed_payload,
    parse_json_reply,
)
from manualtrace.s1000d import PROV_NS, build_modules, schema
from manualtrace.schemas import BBox, Block, PageInfo, ParsedDocument, Table, TableCell
from manualtrace.sources import SOURCES


def _doc(blocks: list[tuple[int, str, str]], tables: list[Table] | None = None) -> ParsedDocument:
    out = []
    for i, (page, label, text) in enumerate(blocks):
        y = 50 + 20 * i
        out.append(Block(id=f"b{i}", page=page, label=label, text=text,
                         bbox=BBox(left=50, top=y, right=500, bottom=y + 15)))
    return ParsedDocument(doc_id="tm55-1905-217-12", kind="digital", title="Test manual",
                          source_url="https://example.invalid", parser="test",
                          pages=[PageInfo(page=p, width=612, height=792) for p in (1, 2)],
                          blocks=out, tables=tables or [])


PROC_DOC = [
    (1, "section_header", "2-11. Cold Weather Operation"),
    (1, "text", "a. Starting Aid Pressurized Cylinder (fig. 2-40)."),
    (1, "text", "(2) Use the starting aid as follows:"),
    (1, "text", "(a) Pull out knob for one-or two-seconds."),
    (1, "text", "(b) Wait 3 seconds and then push starter button."),
    (1, "section_header", "WARNING"),
    (1, "text", "Do not actuate the starting aid more than once with the engine stopped."),
    (1, "section_header", "CAUTION"),
    (1, "text", "Do not crank the engine more than 30 seconds at a time."),
    (1, "text", "b. Engines."),
    (2, "list_item", "STEP 1. PLACE REVERSE GEAR LEVERS IN NEUTRAL."),
    (2, "list_item", "CAUTION: DO NOT CRANK ENGINE FOR MORE THAN 30 SECONDS."),
    (2, "list_item", "STEP 2. OPEN THROTTLE CONTROLS APPROXIMATELY 25 PER CENT."),
    (2, "caption", "Figure 2-32. Engine starting instructions."),
]


def test_step_markers_tolerate_ocr_noise():
    assert step_marker("STEP 3. START ENGINES")[:2] == ("STEP", "3")
    assert step_marker("(B) Slit the covering")[:2] == ("(A)", "B")
    assert step_marker("4, The shear valve")[:2] == ("1.", "4")
    assert step_marker("5., The cable-braid")[:2] == ("1.", "5")
    assert step_marker("The antenna") is None


def test_rules_labeling_finds_admonitions_and_procedures():
    lab = label_payload(make_payload(_doc(PROC_DOC)))
    kinds = [(a["kind"], a["text"][:20]) for a in lab["admonitions"]]
    assert ("caution", "Do not crank the eng") in kinds
    assert ("warning", "Do not actuate the s") in kinds
    assert any(a["text"].startswith("DO NOT CRANK ENGINE") for a in lab["admonitions"])
    titles = {p["title"]: [s["number"] for s in p["steps"]] for p in lab["procedures"]}
    assert titles["Use the starting aid as follows"] == ["a", "b"]
    assert titles["Engine starting instructions"] == ["1", "2"]
    # "a. Starting Aid ..." and "b. Engines." are headings, not a two-step procedure
    assert all(p["steps"][0]["text"] != "Starting Aid Pressurized Cylinder (fig. 2-40)."
               for p in lab["procedures"])


def test_merged_warning_block_is_split():
    doc = _doc([(1, "text", "WARNING FIRE HAZARD Do not smoke. WARNING ASPHYXIATION DANGER "
                            "Open ventilators.")])
    lab = label_payload(make_payload(doc))
    assert [a["text"] for a in lab["admonitions"]] == [
        "FIRE HAZARD Do not smoke.", "ASPHYXIATION DANGER Open ventilators."]


def test_records_carry_page_and_bbox_provenance():
    doc = _doc(PROC_DOC)
    res = extract_document(doc, "rules")
    for rec in [*res.admonitions, *res.procedures]:
        for p in rec.provenance:
            assert p.page in (1, 2) and p.bbox.right > p.bbox.left
            assert p.block_ids
    multi = next(p for p in res.procedures if p.title == "Engine starting instructions")
    assert {p.page for p in multi.provenance} == {2}


def test_ungrounded_llm_output_is_rejected():
    doc = _doc(PROC_DOC)
    labeling = {"admonitions": [
        {"kind": "warning", "block_ids": ["b6"], "text": "Do not actuate the starting aid "
                                                       "more than once with the engine stopped."},
        {"kind": "warning", "block_ids": ["b6"], "text": "Wear a respirator at all times."},
        {"kind": "warning", "block_ids": ["b999"], "text": "x"},
    ]}
    res, rejected = build_result(doc, labeling, "llm-test")
    assert len(res.admonitions) == 1
    assert len(rejected) == 2


def test_offline_provider_roundtrip_matches_rules():
    doc = _doc(PROC_DOC)
    rules = extract_document(doc, "rules")
    llm = extract_document(doc, "llm", OfflineProvider())
    assert llm.extractor == "llm-offline"
    assert llm.model_dump(exclude={"extractor"}) == rules.model_dump(exclude={"extractor"})


def test_payload_and_reply_parsing():
    payload = {"a": [1, 2], "t": "x"}
    assert decode_payload("hello\n" + embed_payload(payload) + "\nbye") == payload
    assert parse_json_reply('Sure:\n```json\n{"parts": []}\n```') == {"parts": []}


def test_anthropic_provider_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        AnthropicProvider()


def test_split_description_variants():
    assert _split_description("GASKET, Plug 694203 (72582)") == {
        "description": "GASKET, Plug", "part_number": "694203", "cage": "72582"}
    assert _split_description("ANCHOR, Marine, 75 lb (81349) A-1 570718") == {
        "description": "ANCHOR, Marine, 75 lb", "part_number": "A-1 570718", "cage": "81349"}
    assert _split_description("PROPELLER, Ship, LH") == {"description": "PROPELLER, Ship, LH"}


def test_parts_table_fixups():
    rows = [
        ["SMR Code", "NATIONAL STOCK NUMBER", "DESCRIPTION, CAGEC and Part Number", "U/I", "QTY"],
        ["", "", "All Hulls", "EA", "1"],
        ["PACZZ", "2910-00-287-1912", "Fuel Filter Secondary ELEMENT, Fuel Filter MIL-E-20627",
         "EA", "8"],
        ["PACZZ", "5330-00-285-3825", "(81349) GASKET, Cover Nut 2243465 (72582)", "EA", "8"],
    ]
    parts = label_parts({"tables": [{"id": "t0", "page": 1, "rows": rows}]})
    assert [p["row"] for p in parts] == [2, 3]
    assert parts[0]["description"] == "ELEMENT, Fuel Filter"
    assert parts[0]["part_number"] == "MIL-E-20627" and parts[0]["cage"] == "81349"
    assert parts[1]["description"] == "GASKET, Cover Nut" and parts[1]["cage"] == "72582"
    assert parts[1]["smr"] == "PACZZ" and parts[1]["unit"] == "EA" and parts[1]["quantity"] == "8"


def _table_doc():
    rows = [["SMR Code", "NATIONAL STOCK NUMBER", "DESCRIPTION", "U/I", "QTY"],
            ["PACZZ", "5330-00-364-2674", "GASKET, Plug 694203 (72582)", "EA", "4"]]
    cells = [TableCell(row=r, col=c, text=v, bbox=BBox(left=40 + 100 * c, top=300 + 20 * r,
                                                       right=130 + 100 * c, bottom=315 + 20 * r))
             for r, row in enumerate(rows) for c, v in enumerate(row)]
    table = Table(id="t0", page=2, bbox=BBox(left=40, top=300, right=540, bottom=340),
                  n_rows=2, n_cols=5, cells=cells)
    return _doc(PROC_DOC, [table])


def test_s1000d_modules_validate_and_carry_provenance():
    doc = _table_doc()
    res = extract_document(doc, "rules")
    assert len(res.parts) == 1
    mods = build_modules(res, SOURCES["tm55-1905-217-12"], dt.date(2026, 1, 2))
    xsd = schema()
    assert len(mods) == 3
    for name, root in mods.items():
        assert name.startswith("DMC-MANUALTRACE-A-LCM-")
        xsd.assertValid(root)
        etree.fromstring(etree.tostring(root))  # serializes to well-formed XML
    ipc = next(r for n, r in mods.items() if "-941A-" in n)
    csn = ipc.find(".//catalogSeqNumber")
    assert csn.get(f"{{{PROV_NS}}}page") == "2"
    assert csn.find(".//nationalStockNumber").get("nsn") == "5330-00-364-2674"
    proc = next(r for n, r in mods.items() if "-300A-" in n)
    assert proc.findall(".//mainProcedure/proceduralStep/proceduralStep")
    # removing provenance makes the module invalid
    del csn.attrib[f"{{{PROV_NS}}}bbox"]
    assert not xsd.validate(ipc)


def test_evaluation_perfect_and_partial():
    doc = _table_doc()
    res = extract_document(doc, "rules")
    gold = {"pages": [1, 2],
            "admonitions": [{"id": "a1", "kind": a.kind, "page": a.provenance[0].page,
                             "text": a.text} for a in res.admonitions],
            "procedures": [{"id": f"p{i}", "page": p.provenance[0].page, "title": p.title,
                            "steps": [{"number": s.number, "page": s.provenance[0].page,
                                       "text": s.text} for s in p.steps]}
                           for i, p in enumerate(res.procedures)],
            "parts": [{"id": "r1", "page": 2, "smr": "PACZZ", "nsn": "5330-00-364-2674",
                       "description": "GASKET, Plug", "part_number": "694203", "cage": "72582",
                       "unit": "EA", "quantity": "4"}]}
    for i, a in enumerate(gold["admonitions"]):
        a["id"] = f"a{i}"
    ev = evaluate_doc(res, gold, None, "digital")
    assert ev["micro"]["precision"] == 1.0 and ev["micro"]["recall"] == 1.0
    gold["parts"][0]["cage"] = "99999"
    gold["admonitions"].append({"id": "extra", "kind": "note", "page": 1, "text": "Missing."})
    ev = evaluate_doc(res, gold, None, "digital")
    assert ev["fields"]["part.cage"]["precision"] == 0.0
    assert ev["records"]["admonition"]["recall"] < 1.0


def test_similarity_grounding_and_bm25():
    assert similarity("Do not  crank the engine.", "do not crank the engine") == 1.0
    assert grounding_score("crank the engine", "CRANK THE ENGINE for 30 s") == 1.0
    assert grounding_score("wear a respirator", "crank the engine") == 0.0
    index = BM25(["fuel filter element", "propeller ship left hand", "battery electrolyte"])
    assert index.search("left hand propeller", 1) == [1]

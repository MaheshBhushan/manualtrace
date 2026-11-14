# ManualTrace

ManualTrace is a harness for extracting structured records from technical equipment
manuals and checking how well that works. It parses public maritime manuals with Docling,
using Tesseract OCR for scanned pages. It pulls out warnings, cautions and notes,
procedures, and parts-list rows as validated Pydantic records. Each record keeps the source
page and bounding box it came from. The records are written out as S1000D-style XML data
modules, and the extraction is scored against a hand-annotated gold set. Field-level
precision/recall, provenance checks and a retrieval check run as pytest regression
thresholds in GitHub Actions.

## Source documents

Both manuals are US government publications. `manualtrace fetch` downloads them and checks
their SHA-256.

| doc id | kind | document | pages processed |
|---|---|---|---|
| `tm55-1905-217-12` | born-digital (text layer) | TM 55-1905-217-12, *Operator and Organizational Maintenance Manual, Landing Craft, Mechanized LCM-8* (US Army, [govinfo.gov](https://www.govinfo.gov/app/details/GOVPUB-D101-PURL-LPS37193)) | 12 of 240: pp. 2, 75, 76, 84, 85, 86, 89, 90, 205, 207, 208, 209 |
| `navships-900524-ib` | scanned (image only, no text layer) | NAVSHIPS 900,524-IB, *Instruction Book for Type CAGQ-66134 Antenna Assembly* (US Navy Bureau of Ships, 1944, hosted by [maritime.org](https://maritime.org/doc/index.php)) | all 15 |

Most pages of the scan are two-page spreads. `fetch` cuts each landscape page into left and
right halves before layout analysis. Bounding boxes are mapped back to the original page.

## Architecture

```
fetch     sources.py    download + checksum, page subset, split scanned spreads -> data/work/
parse     parse.py      Docling layout + TableFormer; Tesseract (CLI) full-page OCR for the scan
                        -> data/parsed/<doc>.json  (blocks and table cells, page + bbox in PDF points)
extract   extract.py    labeling = {admonitions, procedures, parts} citing block ids / table rows
                          rules: deterministic heuristics over Docling reading order
                          llm:   same payload sent through providers.py (offline | anthropic)
                        build_result(): Pydantic validation, provenance = union of cited boxes,
                        records whose text is not found in the cited blocks are rejected
                        -> out/<extractor>/<doc>.json
export    s1000d.py     3 data modules per document, validated against s1000d_subset.xsd
                          infoCode 012 warnings/cautions/notes (description)
                          infoCode 300 procedures (procedure/mainProcedure/proceduralStep)
                          infoCode 941 parts (illustratedPartsCatalog/catalogSeqNumber)
                        provenance on every element: mt:doc, mt:page, mt:bbox (urn:manualtrace:provenance)
evaluate  evaluate.py   field P/R vs gold/, provenance page + bbox grounding, BM25 retrieval
```

Coordinates are PDF points with a top-left origin on the original page,
`left,top,right,bottom`. Example from the generated parts module for the scan:

```xml
<catalogSeqNumber id="part3" catalogSeqNumberValue="003" item="H-105 and H-106 H-121 thru H-126"
    mt:doc="navships-900524-ib" mt:page="7" mt:bbox="745.7,638.3,1151.7,660.7">
```

**LLM adapter.** A model never supplies coordinates. The LLM extractor sends the parsed
blocks and table grids, and gets back a labeling that cites block ids and table rows.
Provenance is computed locally from those citations. Any record whose text is not in the
cited blocks (token coverage below 0.8) is dropped, which is how hallucinated records get
caught. The `offline` provider is deterministic: it decodes the payload and answers with the
rule-based labeler. Tests and CI use it, and it runs the full prompt → JSON → validation
path. The `anthropic` provider makes a real Claude call when `ANTHROPIC_API_KEY` is set.

**S1000D scope.** The XML uses S1000D Issue 5.0 element names. It is validated against a
local XSD (`src/manualtrace/s1000d_subset.xsd`) that models only what the exporter emits. It
is not validated against the official S1000D schemas.

## Gold set

`gold/` was annotated by hand from the rendered page images. For the scan, the transcription
was made without looking at OCR output. Text is kept as printed, including the source's own
typos.

| doc | annotated pages | admonitions | procedures (steps) | parts rows |
|---|---|---|---|---|
| tm55-1905-217-12 | 2, 76, 84, 85, 86, 90, 208, 209 | 10 | 12 (36) | 25 |
| navships-900524-ib | 6–13, 15 | 6 | 3 (16) | 26 |

`gold/queries.json` holds 25 retrieval queries. Each one names the gold records that answer
it.

What gets scored:
- **Field P/R.** Extracted records on annotated pages are aligned one-to-one with gold
  records of the same type on the same page (greedy, by text similarity ≥ 0.5). A field is
  correct when the aligned values agree. Free text needs normalized similarity ≥ 0.9, which
  tolerates OCR noise. Codes (NSN, part number, CAGE, quantity, unit, SMR, item, step
  number) must match exactly after whitespace/case normalization. Procedure titles match at
  similarity ≥ 0.8 or by containment. Precision is computed over every non-empty field the
  extractor produced on annotated pages, so spurious records count against it.
- **Provenance.** For aligned records, the page must match the gold page. Then the text
  inside the cited bounding box is read back from the source PDF: the text layer for the
  digital document, and Tesseract on a 300-dpi crop for the scan. At least 80 % of the
  record's tokens must be found there.
- **Retrieval.** BM25 over the extracted records (an admonition, a whole procedure, or a
  part row). A query counts as a hit@k if a top-k record is aligned with one of its relevant
  gold records. For reference, the same queries are also run against an index built from
  the gold records themselves.

## Measured results

From the commands in *Quickstart*, run locally on 2026-11-13: Docling 2.150.0 and Tesseract
5.5.2, CPU only, `rules` extractor. `manualtrace evaluate` prints these numbers and writes
them to `out/rules/metrics.json`.

| | digital (TM 55-1905-217-12) | scanned (NAVSHIPS 900,524-IB) | all |
|---|---|---|---|
| field-level micro precision | 0.978 | 0.921 | 0.960 |
| field-level micro recall | 0.996 | 0.886 | 0.960 |
| field-level micro F1 | 0.987 | 0.903 | 0.960 |
| admonition records P / R | 1.000 / 1.000 | 1.000 / 1.000 | |
| procedure-step records P / R | 0.947 / 1.000 | 1.000 / 1.000 | |
| parts records P / R | 1.000 / 1.000 | 1.000 / 1.000 | |
| provenance page accuracy (aligned records) | 1.000 (71) | 1.000 (48) | |
| records whose bbox contains ≥ 80 % of their text | 0.986 | 1.000 | |

Per-field numbers worth noting:
- Scanned parts: `quantity` P 0.923 / R 0.706 (OCR misreads the hand-tallied quantity
  column), `item` 0.840 / 0.840, `part_number` 0.875 / 0.824, `description` 0.923 / 0.923.
- Digital: procedure titles 0.923 / 1.000. The precision loss there comes from one spurious
  two-item list ("3-1 a./b.", reference paragraphs, not a procedure).

Retrieval over 25 queries:

| index | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| extracted records | 0.920 | 0.960 | 1.000 | 0.948 |
| gold records (reference) | 0.920 | 1.000 | 1.000 | 0.960 |

The `llm` extractor with the `offline` provider gives the same numbers as `rules`, as
designed; `tests/test_regression.py` checks that both paths produce identical records.

The same pipeline in CI (GitHub Actions run 43523403813, Ubuntu's Tesseract package) gave
identical digital-document numbers. For the scan it gave micro P 0.930 / R 0.901 /
F1 0.915, and for retrieval hit@1 0.920 / hit@3 0.960 / MRR 0.950. The scan numbers
differ from the local run because OCR output changes with the Tesseract version. The run
log has the full report.

Parse time (`data/parsed/timings.json`, 4 CPU threads, run on a heavily shared machine, so
treat these as rough): 265 s for the 12 digital pages and 202 s for the 15 scanned pages.

Regression thresholds in `tests/test_regression.py` sit below these values. For the scan
the margin is wider, because CI uses Ubuntu's Tesseract package and OCR output changes
between Tesseract versions.

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and the `tesseract` binary (with
English data). Torch is installed from the CPU-only wheel index.

```bash
uv sync
uv run manualtrace fetch                      # ~25 MB download, checksum verified
uv run manualtrace parse                      # Docling (+ Tesseract); first run downloads models
uv run manualtrace extract --extractor rules
uv run manualtrace export  --extractor rules  # out/rules/s1000d/DMC-*.xml
uv run manualtrace evaluate --extractor rules # prints the tables above, writes metrics.json

# same pipeline through the LLM adapter
uv run manualtrace extract --extractor llm --provider offline
uv run manualtrace evaluate --extractor llm-offline
# with a real model (needs ANTHROPIC_API_KEY; model via MANUALTRACE_ANTHROPIC_MODEL)
uv sync --extra anthropic
uv run manualtrace extract --extractor llm --provider anthropic
uv run manualtrace evaluate --extractor llm-anthropic
```

`--data` and `--out` move the data and output directories. `MANUALTRACE_THREADS` sets
Docling's CPU threads.

## Tests

```bash
uv run pytest -m "not pipeline"   # 13 unit tests on synthetic layouts, no models needed
uv run pytest                     # + 14 regression tests; needs `fetch` and `parse` first
```

The regression tests use `$MANUALTRACE_DATA` (default `./data`). They skip if the parsed
output is missing, unless `MANUALTRACE_REQUIRE_PIPELINE=1` is set, as it is in CI. They
check:
- non-empty records with in-bounds boxes;
- XSD-valid, well-formed XML for all three data modules;
- per-document field P/R and record recall thresholds;
- provenance page accuracy and bbox grounding;
- retrieval hit@3 and MRR;
- identical output from the offline LLM path.

CI (`.github/workflows/ci.yml`) does the following:
- installs Tesseract and the locked environment;
- runs ruff and the unit tests;
- caches the Hugging Face model directory and the source PDFs;
- runs fetch → parse → extract → export → evaluate on the real manuals, then the
  regression tests;
- uploads `out/` and `data/parsed/` as an artifact.

## Limitations

- **No real-LLM numbers.** The `anthropic` provider is implemented, but the
  `ANTHROPIC_API_KEY` available while this was built was rejected by the API
  (`401 invalid x-api-key`). All reported numbers are from the rule-based extractor and the
  offline provider.
- **Small gold set.** 17 annotated pages across two documents. Some of the rules (heading
  detection, the parts-table fix-ups) were developed against these same pages, so the
  scores show how well the pipeline works on these manuals. They are not an estimate of
  accuracy on unseen manuals.
- **Unscored pages.** Page 14 of the scan has its parts table rotated 90° on the right half.
  Docling/Tesseract do not read it, and it is not in the gold set. The page-8 "major parts"
  table has its symbol column split across two OCR columns, so the `item` field there is
  incomplete.
- **Coarse boxes on page 2.** Docling returns all three warnings on page 2 of the digital
  manual as one text block. Each of those three records therefore carries the whole block
  as its bounding box.
- **Partial S1000D.** No BREX validation, no applicability model, no ICN/figure handling,
  and the data module codes use a placeholder model identification code (`MANUALTRACE`).
- **CI time.** CI runs Docling on CPU over 27 pages, which takes several minutes. Only the
  model weights and PDFs are cached, not the parse output, so every CI run checks the real
  parser.

## License

MIT (code). The manuals are US government works; they are downloaded at run time and not
redistributed here.

"""Command line entry point: fetch -> parse -> extract -> export -> evaluate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .schemas import ExtractionResult, ParsedDocument
from .sources import SOURCES

DEFAULT_DATA = Path("data")
DEFAULT_OUT = Path("out")
GOLD_DIR = Path(__file__).resolve().parents[2] / "gold"


def _ids(arg: list[str] | None) -> list[str]:
    ids = arg or list(SOURCES)
    unknown = [i for i in ids if i not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown doc ids: {unknown}; known: {list(SOURCES)}")
    return ids


def parsed_path(data_dir: Path, doc_id: str) -> Path:
    return data_dir / "parsed" / f"{doc_id}.json"


def extraction_path(out_dir: Path, doc_id: str, extractor: str) -> Path:
    return out_dir / extractor / f"{doc_id}.json"


def load_parsed(data_dir: Path, doc_id: str) -> ParsedDocument:
    return ParsedDocument.model_validate_json(parsed_path(data_dir, doc_id).read_text())


def load_extraction(out_dir: Path, doc_id: str, extractor: str) -> ExtractionResult:
    return ExtractionResult.model_validate_json(
        extraction_path(out_dir, doc_id, extractor).read_text()
    )


def cmd_fetch(args) -> None:
    from .sources import fetch

    for doc_id in _ids(args.doc):
        path = fetch(SOURCES[doc_id], args.data)
        print(f"{doc_id}: {path}")


def cmd_parse(args) -> None:
    from .parse import parse

    timings = {}
    for doc_id in _ids(args.doc):
        parsed, secs = parse(SOURCES[doc_id], args.data)
        dest = parsed_path(args.data, doc_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(parsed.model_dump_json(indent=1))
        timings[doc_id] = {"pages": len(parsed.pages), "seconds": round(secs, 1)}
        print(f"{doc_id}: {len(parsed.pages)} pages, {len(parsed.blocks)} blocks, "
              f"{len(parsed.tables)} tables in {secs:.1f}s -> {dest}")
    (args.data / "parsed" / "timings.json").write_text(json.dumps(timings, indent=1))


def cmd_extract(args) -> None:
    from .extract import extract_document
    from .providers import get_provider

    provider = get_provider(args.provider) if args.extractor == "llm" else None
    for doc_id in _ids(args.doc):
        parsed = load_parsed(args.data, doc_id)
        result = extract_document(parsed, args.extractor, provider)
        dest = extraction_path(args.out, doc_id, result.extractor)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result.model_dump_json(indent=1))
        print(f"{doc_id}: {len(result.admonitions)} admonitions, "
              f"{len(result.procedures)} procedures, {len(result.parts)} parts -> {dest}")


def cmd_export(args) -> None:
    from .s1000d import export_document

    for doc_id in _ids(args.doc):
        result = load_extraction(args.out, doc_id, args.extractor)
        files = export_document(result, SOURCES[doc_id], args.out / args.extractor / "s1000d")
        print(f"{doc_id}: wrote {len(files)} data modules")
        for f in files:
            print(f"  {f}")


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate_all, format_report

    report = evaluate_all(args.data, args.out, args.extractor, _ids(args.doc), GOLD_DIR)
    dest = args.out / args.extractor / "metrics.json"
    dest.write_text(json.dumps(report, indent=1))
    print(format_report(report))
    print(f"-> {dest}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="manualtrace")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--doc", action="append", help="doc id (repeatable; default: all)")
        sp.set_defaults(fn=fn)
        return sp

    add("fetch", cmd_fetch, "download manuals and build working PDFs")
    add("parse", cmd_parse, "run Docling (and Tesseract for scans)")
    sp = add("extract", cmd_extract, "extract admonitions, procedures and parts")
    sp.add_argument("--extractor", choices=["rules", "llm"], default="rules")
    sp.add_argument("--provider", default="offline", help="llm provider: offline | anthropic")
    for name, fn, help_ in [
        ("export", cmd_export, "write S1000D-style XML data modules"),
        ("evaluate", cmd_evaluate, "field-level P/R, retrieval and provenance checks"),
    ]:
        sp = add(name, fn, help_)
        sp.add_argument("--extractor", default="rules",
                        help="extraction run directory under --out (rules, llm-offline, ...)")
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

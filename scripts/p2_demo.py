"""Person 2 integration demo (VS-P2-04): profile a workbook and retrieve definitions.

Runs `profile_workbook`, `sample_rows` and `retrieve_context` without the Planner or UI.

Usage:
    python scripts/build_index.py      # once
    python scripts/p2_demo.py tests/fixtures/mini_multisheet_fixture.xlsx "What was operating profit last quarter?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verisheet.profile import profile_workbook  # noqa: E402
from verisheet.retrieval import DefinitionRetriever, retrieve_context  # noqa: E402
from verisheet.tools.inspect import sample_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Person 2 profiling + retrieval demo")
    parser.add_argument("workbook", help="path to an .xlsx or .csv file")
    parser.add_argument("question", help="analytical question to retrieve definitions for")
    parser.add_argument("--index-dir", default="artifacts/index")
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()

    profile = profile_workbook(args.workbook)
    print(f"== WorkbookProfile  {profile.workbook_id[:23]}…  ({profile.file_type.value})")
    for sheet in profile.sheets:
        flags = " [hidden]" if sheet.hidden else ""
        print(f"  {sheet.name}{flags}: {sheet.row_count} rows, {len(sheet.columns)} columns, "
              f"{sheet.formula_cell_count} formulas")
        for column in sheet.columns:
            print(f"     - {column.name:<20} {column.dtype:<10} roles={column.candidate_roles}")
    print("== Quality warnings")
    for warning in profile.warnings:
        location = warning.sheet + (f".{warning.column}" if warning.column else "") if warning.sheet else "workbook"
        print(f"  [{warning.severity.value}] {warning.code} @ {location}: {warning.message}")

    first = next((s for s in profile.sheets if s.row_count), None)
    if first:
        print(f"== sample_rows('{first.name}', n={args.rows})")
        print(json.dumps(sample_rows(args.workbook, first.name, n=args.rows), indent=2))

    retriever = DefinitionRetriever(index_dir=args.index_dir)
    context = retrieve_context(args.question, profile, retriever=retriever if retriever.is_ready else None)
    print(f"== RetrievedContext  corpus_version={context.corpus_version}")
    for warning in context.warnings:
        print(f"  WARNING {warning}")
    for rank, item in enumerate(context.items, 1):
        print(f"  {rank}. ({item.score:.3f}) {item.title}  <{item.source_path}>")
        print(f"     {item.content[:160].replace(chr(10), ' ')}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

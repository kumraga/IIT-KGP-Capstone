"""Retrieval evaluation and chunking experiment (Person 2).

Computes Recall@1/3/5 and MRR over a labelled query set for each search mode (dense BGE,
BM25, hybrid RRF) and chunk size. A hit is relevant when its content contains the query's
`must_contain` phrase, which keeps labels valid across chunk sizes.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --chunk-sizes 300 600 1200 --out results/retrieval_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verisheet.retrieval import SEARCH_MODES, DefinitionRetriever, collect_corpus  # noqa: E402

KS = (1, 3, 5)


def evaluate(retriever: DefinitionRetriever, queries: list[dict]) -> dict[str, float]:
    reciprocal_ranks: list[float] = []
    hits_at = {k: 0 for k in KS}
    for case in queries:
        items = retriever.search(case["query"], top_k=max(KS)).items
        needle = case["must_contain"].lower()
        rank = next((i + 1 for i, item in enumerate(items) if needle in item.content.lower()), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in KS:
            hits_at[k] += int(rank is not None and rank <= k)
    n = len(queries)
    metrics = {f"recall@{k}": round(hits_at[k] / n, 3) for k in KS}
    metrics["mrr"] = round(sum(reciprocal_ranks) / n, 3)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes and chunk sizes")
    parser.add_argument("--queries", default="tests/fixtures/retrieval_eval_queries.json")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 600, 1200])
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    encoder = None
    rows = []
    for size in args.chunk_sizes:
        corpus = collect_corpus("docs/DATA_DICTIONARY.md", "docs/tools", "prompts", max_chars=size)
        with tempfile.TemporaryDirectory() as tmp:
            retriever = DefinitionRetriever(model_name=args.model, index_dir=tmp)
            if encoder is not None:
                retriever._encoder = encoder  # reuse the loaded model across runs
            retriever.build_index(corpus)
            encoder = retriever._encoder
            for mode in SEARCH_MODES:
                retriever.mode = mode
                rows.append({"chunk_chars": size, "chunks": len(corpus), "mode": mode,
                             **evaluate(retriever, queries)})

    header = f"{'chunk':>6} {'chunks':>6} {'mode':<7} " + " ".join(f"{m:>9}" for m in
                                                               ["recall@1", "recall@3", "recall@5", "mrr"])
    print(f"{len(queries)} labelled queries, model {args.model}\n{header}")
    for row in rows:
        print(f"{row['chunk_chars']:>6} {row['chunks']:>6} {row['mode']:<7} "
              + " ".join(f"{row[m]:>9.3f}" for m in ["recall@1", "recall@3", "recall@5", "mrr"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

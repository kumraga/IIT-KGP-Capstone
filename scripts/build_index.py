"""Build the approved-documentation retrieval index (Person 2, VS-P2-03).

Usage:
    python scripts/build_index.py                      # defaults below
    python scripts/build_index.py --index-dir artifacts/index --chunk-chars 600

Embeddings come from the shared capstone embedding API when EMBEDDING_API_BASE/KEY are set
(see .env.example), otherwise from the local sentence-transformers model.

Indexes only allowlisted documentation (data dictionary, tool docs, prompt rules). Gold answers,
raw/uploaded workbooks and secrets are never read. Prints the corpus version for traceability.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verisheet.retrieval import CHUNK_MAX_CHARS, DefinitionRetriever, collect_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dictionary", default="docs/DATA_DICTIONARY.md")
    parser.add_argument("--tool-docs", default="docs/tools")
    parser.add_argument("--prompt-docs", default="prompts")
    parser.add_argument("--index-dir", default="artifacts/index")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--chunk-chars", type=int, default=CHUNK_MAX_CHARS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    corpus = collect_corpus(args.data_dictionary, args.tool_docs, args.prompt_docs, args.chunk_chars)
    retriever = DefinitionRetriever(model_name=args.model, index_dir=args.index_dir)
    retriever.build_index(corpus)

    by_source = Counter(item["source_path"].split("#")[0] for item in corpus)
    print(f"Indexed {len(corpus)} chunks with {args.model} "
          f"(embeddings: {retriever.embedding_provider}, vector backend: {retriever.backend})")
    for source, count in sorted(by_source.items()):
        print(f"  {count:3d}  {source}")
    print(f"corpus_version: {retriever.corpus_version}")
    print(f"index dir:      {Path(args.index_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

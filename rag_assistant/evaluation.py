"""Retrieval-quality evaluation: recall@k and MRR over a labelled question set.

Usage: python -m rag_assistant.evaluation [--k 4] [--min-recall 0.8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .pipeline import RAGPipeline, load_corpus

DATA = Path(__file__).resolve().parent.parent / "data"


def evaluate(pipeline: RAGPipeline, dataset: list[dict], k: int = 4) -> dict:
    hits_at_k = 0
    reciprocal_ranks = 0.0
    misses: list[str] = []
    for item in dataset:
        results = pipeline.retriever.search(item["question"], k)
        doc_ids = [h.chunk.doc_id for h in results]
        if item["expected_doc"] in doc_ids:
            hits_at_k += 1
            reciprocal_ranks += 1.0 / (doc_ids.index(item["expected_doc"]) + 1)
        else:
            misses.append(item["question"])
    n = len(dataset)
    return {
        "questions": n,
        f"recall@{k}": round(hits_at_k / n, 3) if n else 0.0,
        "mrr": round(reciprocal_ranks / n, 3) if n else 0.0,
        "misses": misses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-recall", type=float, default=0.0)
    args = parser.parse_args()

    pipeline = RAGPipeline(Settings())
    load_corpus(pipeline, DATA / "corpus")
    dataset = json.loads((DATA / "eval_set.json").read_text(encoding="utf-8"))
    report = evaluate(pipeline, dataset, args.k)
    print(json.dumps(report, indent=2))
    return 0 if report[f"recall@{args.k}"] >= args.min_recall else 1


if __name__ == "__main__":
    sys.exit(main())

import json

from rag_assistant.config import Settings
from rag_assistant.evaluation import DATA, evaluate
from rag_assistant.pipeline import RAGPipeline, load_corpus


def test_sample_corpus_retrieval_quality_has_not_regressed():
    pipeline = RAGPipeline(Settings())
    load_corpus(pipeline, DATA / "corpus")
    dataset = json.loads((DATA / "eval_set.json").read_text(encoding="utf-8"))
    report = evaluate(pipeline, dataset, k=4)
    assert report["recall@4"] >= 0.9, report["misses"]
    assert report["mrr"] >= 0.7

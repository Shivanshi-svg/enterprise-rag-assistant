from rag_assistant.chunking import Chunk
from rag_assistant.retrieval import HybridRetriever


def _retriever():
    r = HybridRetriever()
    r.add([
        Chunk("cats", 0, "Cats are small carnivorous mammals that purr and hunt mice."),
        Chunk("rockets", 0, "Rockets burn propellant to produce thrust and reach orbit."),
        Chunk("bread", 0, "Bread dough rises because yeast ferments sugars into gas."),
    ])
    return r


def test_relevant_document_ranks_first():
    hits = _retriever().search("how do rockets reach orbit", k=2)
    assert hits[0].chunk.doc_id == "rockets"


def test_empty_index_returns_nothing():
    assert HybridRetriever().search("anything") == []


def test_unrelated_query_returns_nothing():
    assert _retriever().search("zzzz qqqq") == []


def test_remove_document():
    r = _retriever()
    assert r.remove_document("rockets") == 1
    assert all(h.chunk.doc_id != "rockets" for h in r.search("rockets orbit thrust"))

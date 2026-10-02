import httpx

from rag_assistant.config import Settings
from rag_assistant.generation import OpenAICompatibleGenerator
from rag_assistant.pipeline import ABSTAIN_MESSAGE, RAGPipeline


def _pipeline(**kwargs):
    p = RAGPipeline(Settings(), **kwargs)
    p.ingest(
        "breach",
        "A breach must be reported within 72 hours. Affected people are told if risk is high.",
    )
    p.ingest(
        "rights",
        "People may request deletion of their data. Responses are due within one month.",
    )
    return p


def _llm(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleGenerator("http://llm.test/v1", "key", "m", client=client)


def test_answer_has_citations():
    answer = _pipeline().ask("How fast must a breach be reported?")
    assert not answer.abstained
    assert answer.citations[0].doc_id == "breach"
    assert "72 hours" in answer.answer


def test_abstains_when_nothing_relevant():
    answer = _pipeline().ask("What is the capital of Australia?")
    assert answer.abstained and answer.answer == ABSTAIN_MESSAGE and answer.citations == []


def test_reingest_replaces_document():
    p = _pipeline()
    p.ingest("breach", "Completely different text about gardening and tomatoes.")
    assert p.documents()["breach"] == 1
    assert p.ask("How fast must a breach be reported?").abstained


def test_delete_unknown_document():
    assert _pipeline().delete("nope") is False


def test_llm_generator_used_and_prompt_contains_context():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": " 72 hours. "}}]})

    gen = _llm(handler)
    answer = _pipeline(generator=gen).ask("How fast must a breach be reported?")
    assert answer.answer == "72 hours."
    assert "within 72 hours" in seen["body"]


def test_llm_failure_falls_back_to_extractive():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    gen = _llm(handler)
    answer = _pipeline(generator=gen).ask("How fast must a breach be reported?")
    assert not answer.abstained and "72 hours" in answer.answer

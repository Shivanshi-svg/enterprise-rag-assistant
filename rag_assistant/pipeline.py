"""RAG pipeline: ingest -> retrieve -> (abstain | generate with citations)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .chunking import chunk_text
from .config import Settings
from .generation import ExtractiveGenerator, Generator, OpenAICompatibleGenerator
from .retrieval import HybridRetriever

log = logging.getLogger(__name__)

ABSTAIN_MESSAGE = "I could not find enough information in the knowledge base to answer that."


@dataclass
class Citation:
    doc_id: str
    chunk_index: int
    snippet: str
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        generator: Generator | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or self._default_generator()
        self._fallback = ExtractiveGenerator()
        self._doc_chunks: dict[str, int] = {}

    def _default_generator(self) -> Generator:
        s = self.settings
        if s.llm_base_url and s.llm_api_key:
            return OpenAICompatibleGenerator(s.llm_base_url, s.llm_api_key, s.llm_model)
        return ExtractiveGenerator()

    def ingest(self, doc_id: str, text: str) -> int:
        """Add or replace a document. Returns the number of chunks indexed."""
        if doc_id in self._doc_chunks:
            self.delete(doc_id)
        chunks = chunk_text(doc_id, text, self.settings.chunk_words, self.settings.chunk_overlap)
        if chunks:
            self.retriever.add(chunks)
            self._doc_chunks[doc_id] = len(chunks)
        return len(chunks)

    def delete(self, doc_id: str) -> bool:
        if doc_id not in self._doc_chunks:
            return False
        self.retriever.remove_document(doc_id)
        del self._doc_chunks[doc_id]
        return True

    def documents(self) -> dict[str, int]:
        return dict(self._doc_chunks)

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retriever.search(question, k or self.settings.top_k)
        confidence = max((h.dense_score for h in hits), default=0.0)
        if not hits or confidence < self.settings.min_confidence:
            return Answer(question, ABSTAIN_MESSAGE, [], confidence, abstained=True)

        contexts = [h.chunk.text for h in hits]
        try:
            text = self.generator.generate(question, contexts)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            log.warning("LLM generation failed (%s); falling back to extractive answer", exc)
            text = self._fallback.generate(question, contexts)
        if not text:
            return Answer(question, ABSTAIN_MESSAGE, [], confidence, abstained=True)

        citations = [
            Citation(h.chunk.doc_id, h.chunk.index, h.chunk.text[:200], round(h.score, 4))
            for h in hits
        ]
        return Answer(question, text, citations, round(confidence, 4))


def load_corpus(pipeline: RAGPipeline, directory: str | Path) -> int:
    """Ingest every ``*.md`` file in ``directory`` using the file stem as document id."""
    total = 0
    for path in sorted(Path(directory).glob("*.md")):
        total += pipeline.ingest(path.stem, path.read_text(encoding="utf-8"))
    return total

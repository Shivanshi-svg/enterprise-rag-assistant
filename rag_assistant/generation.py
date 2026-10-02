"""Answer generators. The default is offline and deterministic; an LLM is optional."""

from __future__ import annotations

from typing import Protocol

import httpx

from .chunking import split_sentences
from .retrieval import tokenize

_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be by with as at from that this it its "
    "what which who how when where why do does did can must should under about".split()
)


class Generator(Protocol):
    def generate(self, question: str, contexts: list[str]) -> str: ...


class ExtractiveGenerator:
    """Return the context sentences that overlap most with the question."""

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[str]) -> str:
        wanted = set(tokenize(question)) - _STOPWORDS
        scored: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        position = 0
        for context in contexts:
            for sentence in split_sentences(context):
                position += 1
                if sentence in seen:
                    continue
                seen.add(sentence)
                overlap = len(wanted & set(tokenize(sentence)))
                if overlap:
                    scored.append((-overlap, position, sentence))
        best = sorted(scored)[: self.max_sentences]
        return " ".join(sentence for _, _, sentence in sorted(best, key=lambda t: t[1]))


class OpenAICompatibleGenerator:
    """Calls any OpenAI-compatible ``/chat/completions`` endpoint."""

    SYSTEM_PROMPT = (
        "Answer using ONLY the provided context. If the context does not contain the answer, "
        "say you do not know. Be concise."
    )

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=30.0)

    def generate(self, question: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

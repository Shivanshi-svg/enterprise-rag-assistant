"""Sentence-aware chunking with word-based overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_SPLIT.split(normalized) if s.strip()]


def _carry_overlap(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    kept: list[str] = []
    words = 0
    for sentence in reversed(sentences):
        if words >= overlap:
            break
        kept.insert(0, sentence)
        words += len(sentence.split())
    return kept, words


def chunk_text(doc_id: str, text: str, max_words: int = 120, overlap: int = 30) -> list[Chunk]:
    """Split ``text`` into chunks of roughly ``max_words`` words that never cut a sentence."""
    if max_words <= 0 or not 0 <= overlap < max_words:
        raise ValueError("require max_words > 0 and 0 <= overlap < max_words")

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if current and count + words > max_words:
            chunks.append(" ".join(current))
            current, count = _carry_overlap(current, overlap)
        current.append(sentence)
        count += words
    if current:
        chunks.append(" ".join(current))
    return [Chunk(doc_id, i, t) for i, t in enumerate(chunks)]

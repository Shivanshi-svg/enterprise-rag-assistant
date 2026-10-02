"""Hybrid retrieval: BM25 (lexical) + TF-IDF cosine, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused RRF score, used for ordering
    dense_score: float  # cosine similarity, used for abstention
    bm25_score: float


class HybridRetriever:
    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._rebuild()

    def remove_document(self, doc_id: str) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        self._rebuild()
        return before - len(self._chunks)

    def _rebuild(self) -> None:
        if not self._chunks:
            self._bm25 = self._vectorizer = self._matrix = None
            return
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks])
        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in self._chunks])

    def _rrf(self, *score_arrays: np.ndarray) -> np.ndarray:
        fused = np.zeros(len(score_arrays[0]))
        for scores in score_arrays:
            ranks = np.empty(len(scores), dtype=int)
            ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
            has_signal = scores > 0  # a signal of zero carries no ranking information
            fused[has_signal] += 1.0 / (self.rrf_k + ranks[has_signal])
        return fused

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self._chunks or self._bm25 is None:
            return []
        tokens = tokenize(query)
        bm25 = np.asarray(self._bm25.get_scores(tokens)) if tokens else np.zeros(len(self._chunks))
        dense = cosine_similarity(self._vectorizer.transform([query]), self._matrix).ravel()
        fused = self._rrf(bm25, dense)
        order = np.argsort(-fused, kind="stable")[:k]
        return [
            Hit(self._chunks[i], float(fused[i]), float(dense[i]), float(bm25[i]))
            for i in order
            if fused[i] > 0
        ]

"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    chunk_words: int = 120
    chunk_overlap: int = 30
    top_k: int = 4
    # Below this cosine similarity the assistant abstains instead of guessing.
    min_confidence: float = 0.08
    # Optional OpenAI-compatible endpoint. If unset, the offline extractive generator is used.
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            chunk_words=int(os.getenv("RAG_CHUNK_WORDS", cls.chunk_words)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(os.getenv("RAG_TOP_K", cls.top_k)),
            min_confidence=float(os.getenv("RAG_MIN_CONFIDENCE", cls.min_confidence)),
            llm_base_url=os.getenv("RAG_LLM_BASE_URL") or None,
            llm_api_key=os.getenv("RAG_LLM_API_KEY") or None,
            llm_model=os.getenv("RAG_LLM_MODEL", cls.llm_model),
        )

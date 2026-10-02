"""FastAPI service exposing ingestion and question answering."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .pipeline import RAGPipeline, load_corpus

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


class DocumentIn(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=1_000)
    top_k: int | None = Field(default=None, ge=1, le=20)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(title="Enterprise RAG Assistant", version=__version__)
    rag = pipeline or RAGPipeline(Settings.from_env())
    if pipeline is None and os.getenv("RAG_LOAD_SAMPLES", "1") == "1":
        load_corpus(rag, CORPUS_DIR)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "documents": len(rag.documents())}

    @app.get("/documents")
    def list_documents() -> dict[str, int]:
        return rag.documents()

    @app.put("/documents/{doc_id}", status_code=201)
    def put_document(doc_id: str, body: DocumentIn) -> dict:
        return {"doc_id": doc_id, "chunks": rag.ingest(doc_id, body.text)}

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> None:
        if not rag.delete(doc_id):
            raise HTTPException(status_code=404, detail="document not found")

    @app.post("/query")
    def query(body: QueryIn) -> dict:
        return asdict(rag.ask(body.question, body.top_k))

    return app


app = create_app()

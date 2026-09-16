"""Shared Chroma connection helper, used by every *_vectors / ingest_policies
style pipeline so they all write to the same persistent store with the same
embedding model."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

CHROMA_DIR = Path(__file__).parent / "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_chroma_collection(name: str):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def replace_chunks_for(collection, match_field: str, match_value: str, ids, docs, metadatas) -> int:
    """Delete any existing chunks matching match_field == match_value, then
    upsert the new ones. Returns how many old chunks were replaced. Shared by
    every ingestion pipeline that re-ingests a revised/updated document."""
    existing = collection.get(where={match_field: match_value})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    return len(existing["ids"])

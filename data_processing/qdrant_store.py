"""Shared Qdrant connection helper - a Qdrant-backed counterpart to
vector_store.py, prepared for moving call_notes/complaints off Chroma onto a
server-backed vector DB (see PRD.md §13's "server-backed vector DB" future
consideration) if concurrent access or scale ever require it.

Not currently imported by any ingest script - vector_store.py/Chroma remains
the store actually used end to end for all 6 collections. Requires a running
Qdrant instance (docker-compose.yml at the repo root: `docker compose up`)
and QDRANT_URL in .env to do anything.

Qdrant has no built-in embedding function like Chroma's
embedding_functions.OpenAIEmbeddingFunction, so embedding is done explicitly
here via the OpenAI SDK, using the same model as vector_store.py
(text-embedding-3-small) to keep retrieval behavior comparable across both
stores if this is wired in later.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

import db

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = OpenAI()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def get_or_create_collection(client: QdrantClient, name: str) -> str:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return name


def replace_points_for(client: QdrantClient, collection_name: str, match_field: str, match_value: str,
                        ids: list[str], texts: list[str], payloads: list[dict]) -> int:
    """Delete any existing points matching match_field == match_value, then
    upsert the new ones. Returns how many old points were replaced. Qdrant
    equivalent of vector_store.py's replace_chunks_for."""
    existing = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=[FieldCondition(key=match_field, match=MatchValue(value=match_value))]),
        limit=10_000,
        with_payload=False,
        with_vectors=False,
    )[0]
    n_existing = len(existing)
    if n_existing:
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key=match_field, match=MatchValue(value=match_value))])
            ),
        )

    vectors = embed_texts(texts)
    points = [
        PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, original_id)), vector=vector, payload=payload)
        for original_id, vector, payload in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return n_existing


def query_points(client: QdrantClient, collection_name: str, query_text: str, n_results: int,
                  where: dict | None = None) -> list[dict]:
    query_filter = None
    if where:
        query_filter = Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()])

    vector = embed_texts([query_text])[0]
    results = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=n_results,
        query_filter=query_filter,
    ).points

    return [{"payload": p.payload, "similarity": round(p.score, 4)} for p in results]


def validate_client_ids(client_ids: set[str]) -> set[str]:
    """Returns the subset of client_ids NOT found in the clients table -
    a defensive check, not an expected-to-fire path (see PLAN.md: every CLxxx
    header in these two PDFs is already confirmed valid)."""
    conn = db.get_connection()
    try:
        known = {row["client_id"] for row in conn.execute("SELECT client_id FROM clients")}
    finally:
        conn.close()
    return client_ids - known

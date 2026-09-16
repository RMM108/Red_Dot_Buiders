"""Shared Chroma connection helper, used by every *_vectors / ingest_policies
style pipeline so they all write to the same persistent store with the same
embedding model."""

from __future__ import annotations

import os
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
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


NUMERIC_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%|\d+/\d+|\d{1,3}(?:,\d{3})+")


def keyword_boosted_query(collection, question: str, n_results: int = 3, where: dict | None = None,
                           min_pool: int = 6) -> list[dict]:
    """Plain vector search under-ranks chunks whose whole relevance is one
    specific number against longer, topically-generic chunks - found twice:
    once with a policy section stating a 20% cap (ranked last of 14 chunks),
    again with a fund's exact minimum-investment figure (didn't make the top
    4 at all, returning four *other* funds' minimum-investment rows instead).
    Any chunk containing a numeric token from the query verbatim (e.g. "20%",
    "250,000") is boosted to the front regardless of embedding similarity.
    Returns raw candidates (id/doc/meta/similarity/keyword_hit) - callers
    format their own output shape from meta."""
    vec_results = collection.query(query_texts=[question], n_results=max(n_results, min_pool), where=where)
    candidates = [
        {"id": id_, "doc": doc, "meta": meta, "similarity": round(1 - dist, 4), "keyword_hit": False}
        for id_, doc, meta, dist in zip(vec_results["ids"][0], vec_results["documents"][0],
                                          vec_results["metadatas"][0], vec_results["distances"][0])
    ]
    seen_ids = {c["id"] for c in candidates}

    query_tokens = set(NUMERIC_TOKEN_RE.findall(question))
    if query_tokens:
        all_chunks = collection.get(where=where)
        for id_, doc, meta in zip(all_chunks["ids"], all_chunks["documents"], all_chunks["metadatas"]):
            if not any(tok in doc for tok in query_tokens):
                continue
            if id_ in seen_ids:
                next(c for c in candidates if c["id"] == id_)["keyword_hit"] = True
            else:
                candidates.append({"id": id_, "doc": doc, "meta": meta, "similarity": None, "keyword_hit": True})
                seen_ids.add(id_)

    candidates.sort(key=lambda c: (not c["keyword_hit"], -(c["similarity"] or 0)))
    return candidates[:n_results]

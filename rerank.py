"""Cross-encoder re-ranking for the RAG retrieval layer (PLAN.md §4.2).

The dense vector store (Chroma, text-embedding-3-small, cosine) returns
chunks ranked by embedding similarity. That is a good first pass but can
under-rank short, specific chunks against longer, topically-generic ones
(the same failure mode ingest_policies.py already works around with its
numeric-token boost). A cross-encoder scores each candidate *pair*
(query, chunk) jointly, which captures lexical overlap and exact-phrase
relevance that a bi-encoder embedding cannot.

This module exposes a single helper, ``rerank``, that takes the query and
the candidate list (each dict with a ``text`` field) and returns the same
list re-sorted by cross-encoder relevance, with the ``similarity`` field
overwritten by the cross-encoder score. It is deliberately dependency-light
and lazy: the model is loaded on first use and cached, so modules that never
call it pay no import cost.

Usage (from any ingest_* query function):
    from rerank import rerank
    candidates = collection.query(query_texts=[question], n_results=candidate_pool, ...)
    results = rerank(question, candidates)[:n_results]
"""

from __future__ import annotations

from typing import Any, Optional

# Default cross-encoder. ms-marco-MiniLM is a strong, small, fast general
# re-ranker; swap for a domain-tuned model if one becomes available.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None
_model_name: Optional[str] = None


def _get_model(model_name: str = DEFAULT_MODEL):
    """Load (and cache) the cross-encoder model on first use."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(model_name)
        _model_name = model_name
    return _model


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    keep: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Re-rank ``candidates`` by cross-encoder relevance to ``query``.

    Each candidate must be a dict with a ``text`` field. The returned list
    is the same dicts re-sorted descending by cross-encoder score, with each
    candidate's ``similarity`` field replaced by that score. If ``keep`` is
    given, only the top ``keep`` are returned.

    Empty candidate lists and empty queries are returned unchanged (no model
    load, no cost).
    """
    if not candidates or not query.strip():
        return candidates

    model = _get_model(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["similarity"] = round(float(score), 4)

    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates[:keep] if keep is not None else candidates
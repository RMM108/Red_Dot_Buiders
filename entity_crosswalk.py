"""Resolve a free-text client name mention to a client_id (PLAN.md §5 step 5).

Every document in this corpus that mentions a client already puts the CLxxx
ID directly next to their name (call notes: "CL006 — Ahmad Faisal bin
Rahman"; complaints: "Client Account: CL002"), so there's no ambiguity to
resolve *within* the document corpus. The actual need is at the query
boundary: an RM types "what about Robert Chua's account" with no ID, and the
agent's query_client_db(client_id) tool needs a client_id, not a name.

Not just a lookup table, because a naive "match by surname" approach breaks
on this dataset: CL005 "Siti Rahman" and CL006 "Ahmad Faisal bin Rahman"
share a surname, so a surname-only match is genuinely ambiguous and must say
so rather than silently guessing one of them.

Usage:
    from entity_crosswalk import resolve_client
    resolve_client("Robert Chua")   -> {"client_id": "CL002", ...}
    resolve_client("Rahman")        -> {"ambiguous": True, "candidates": [...]}
"""

from __future__ import annotations

import difflib
import re

import db

TITLES_RE = re.compile(r"^(mr|mrs|ms|miss|dr|mdm)\.?\s+", re.IGNORECASE)


def _normalize(s: str) -> str:
    s = TITLES_RE.sub("", s.strip())
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def build_client_index(conn=None) -> list[dict]:
    own_conn = conn is None
    conn = conn or db.get_connection()
    try:
        clients = [dict(r) for r in conn.execute("SELECT client_id, name FROM clients")]
    finally:
        if own_conn:
            conn.close()

    index = []
    for c in clients:
        name_parts = c["name"].split()
        index.append({
            "client_id": c["client_id"],
            "name": c["name"],
            "name_norm": _normalize(c["name"]),
            "first_name_norm": _normalize(name_parts[0]) if name_parts else "",
            "last_name_norm": _normalize(name_parts[-1]) if name_parts else "",
        })
    return index


def resolve_client(query: str, conn=None) -> dict:
    query_norm = _normalize(query)
    index = build_client_index(conn)

    # 1. exact full-name match
    exact = [c for c in index if c["name_norm"] == query_norm]
    if len(exact) == 1:
        return {"client_id": exact[0]["client_id"], "name": exact[0]["name"], "match_type": "exact_name"}
    if len(exact) > 1:
        return {"ambiguous": True, "candidates": [{"client_id": c["client_id"], "name": c["name"]} for c in exact]}

    # 2. exact first-name-only or last-name-only match (e.g. "Chua", "Robert")
    partial = [c for c in index if query_norm in (c["first_name_norm"], c["last_name_norm"])]
    if len(partial) == 1:
        return {"client_id": partial[0]["client_id"], "name": partial[0]["name"], "match_type": "partial_name"}
    if len(partial) > 1:
        return {
            "ambiguous": True,
            "candidates": [{"client_id": c["client_id"], "name": c["name"]} for c in partial],
            "reason": f"{len(partial)} clients share the name fragment {query!r}",
        }

    # 3. fuzzy fallback (typos / partial phrasing)
    names_norm = [c["name_norm"] for c in index]
    close = difflib.get_close_matches(query_norm, names_norm, n=2, cutoff=0.75)
    if len(close) == 1:
        match = next(c for c in index if c["name_norm"] == close[0])
        return {"client_id": match["client_id"], "name": match["name"], "match_type": "fuzzy"}
    if len(close) > 1:
        candidates = [c for c in index if c["name_norm"] in close]
        return {"ambiguous": True, "candidates": [{"client_id": c["client_id"], "name": c["name"]} for c in candidates]}

    return {"found": False, "query": query}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Robert Chua"
    print(resolve_client(q))

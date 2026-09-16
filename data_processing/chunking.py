"""Shared "typical chunking" fallback: fixed-size, overlapping text windows.

Same algorithm and constants as ingest_policies.py's private _paragraph_chunks()
(that module's own fallback for a policy PDF without numbered sections), pulled
out into a small, backend-agnostic module so ingest_call_notes.py and
ingest_complaints.py can reuse it as the secondary pass of their chunk-by-client-ID
primary / chunk-by-size secondary strategy, without depending on
ingest_policies.py's Chroma-specific, policy-shaped chunk dicts.
"""

from __future__ import annotations

PARAGRAPH_CHUNK_SIZE = 1200
PARAGRAPH_CHUNK_OVERLAP = 200


def paragraph_chunks(text: str, chunk_size: int = PARAGRAPH_CHUNK_SIZE,
                      overlap: int = PARAGRAPH_CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return chunks

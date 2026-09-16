"""Shared "typical chunking" fallback: fixed-size, overlapping text windows.

Same algorithm and constants as ingest_policies.py's private _paragraph_chunks()
(that module's own fallback for a policy PDF without numbered sections), pulled
out into a small, backend-agnostic module so ingest_call_notes.py and
ingest_complaints.py can reuse it as the secondary pass of their chunk-by-client-ID
primary / chunk-by-size secondary strategy, without depending on
ingest_policies.py's Chroma-specific, policy-shaped chunk dicts.
"""

from __future__ import annotations

import bisect

from pypdf import PdfReader

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


# --------------------------------------------------------------------------
# Page tracking for citations: every ingest_*.py script that chunks a PDF by
# regex-matching section/entry headers on the *joined* full-document text
# (ingest_policies.py, ingest_call_notes.py, ingest_complaints.py,
# ingest_ack_forms.py) can pair that with page_offsets computed here, so each
# chunk's character span maps back to the real PDF page(s) it came from -
# the citation the model shows can then say "p.4" instead of just the file
# name. This only requires the offsets be computed against the exact same
# joined string each script already builds (page texts joined with "\n"),
# which each script's own extract_text() does - see page_for_offset/
# page_range_for_span docstrings for why footer-stripping afterwards doesn't
# invalidate these offsets.
# --------------------------------------------------------------------------

def extract_page_offsets(pdf_path) -> list[int]:
    """Character offset (into the "\\n"-joined full-document text, matching
    each ingest_*.py's own extract_text()) where each page begins.
    offsets[i] is where page i+1 (1-indexed) starts."""
    reader = PdfReader(str(pdf_path))
    offsets = []
    cursor = 0
    for page in reader.pages:
        offsets.append(cursor)
        cursor += len(page.extract_text() or "") + 1  # +1 for the "\n" joiner
    return offsets


def page_for_offset(offsets: list[int], char_offset: int) -> int:
    """1-indexed page number containing `char_offset` in the joined text.
    Valid even if the text was later truncated from the end (e.g. a footer
    disclaimer stripped by FOOTER_RE.sub) - that only removes a suffix, so
    any offset before the cut point still lands on the same page it did in
    the original joined text."""
    idx = bisect.bisect_right(offsets, char_offset) - 1
    return max(idx, 0) + 1


def page_range_for_span(offsets: list[int], start: int, end: int) -> str:
    """Human-readable page locator for a chunk spanning [start, end) of the
    joined text - "p.4" for a single page, "pp.4-5" if the chunk crosses a
    page boundary (e.g. a call-note entry that runs past a page break)."""
    p_start = page_for_offset(offsets, start)
    p_end = page_for_offset(offsets, max(end - 1, start))
    return f"p.{p_start}" if p_start == p_end else f"pp.{p_start}-{p_end}"

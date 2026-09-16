"""Ingest data/rm_call_notes_log.pdf into a `call_notes` Chroma collection,
one chunk per client entry (PLAN.md §5 step 4).

The log has a reliable per-entry header - "CLxxx — Name" immediately
followed by "Date: YYYY-MM-DD" (verified: all 10 "CLxxx" occurrences in the
document are at line-start, i.e. entry headers, no stray in-body mentions of
another client's ID) - so entries are split deterministically, same pattern
as ingest_policies.py's section splitting.

Re-ingesting the same PDF replaces all of its previous chunks (the file, not
the entry, is the natural update unit for a log/register document like this).

Usage:
    python ingest_call_notes.py
    python ingest_call_notes.py --query "portfolio rebalancing delay"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader

# rerank.py lives at the repository root (sibling of data_processing/); make
# sure it's importable regardless of how this script is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking import paragraph_chunks
from vector_store import get_chroma_collection, replace_chunks_for
from rerank import rerank

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_PATH = DATA_DIR / "rm_call_notes_log.pdf"
COLLECTION_NAME = "call_notes"

ENTRY_HEADER_RE = re.compile(r"(?m)^(CL\d{3})\s")
DATE_RE = re.compile(r"Date:\s*(\d{4}-\d{2}-\d{2})")
FOOTER_RE = re.compile(r"This document is a fictitious example.*", re.DOTALL)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def split_entries(text: str) -> list[dict]:
    """One chunk per "CLxxx - Name" entry. Falls back to plain paragraph
    chunking (chunking.py) if a future call-notes log doesn't use this
    header pattern at all, so it still gets ingested rather than skipped -
    same fallback shape as ingest_policies.py's section splitting."""
    text = FOOTER_RE.sub("", text).strip()
    matches = list(ENTRY_HEADER_RE.finditer(text))

    if not matches:
        return [{"client_id": None, "date": None, "has_flag": "FLAG:" in chunk, "text": chunk}
                for chunk in paragraph_chunks(text)]

    entries = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entry_text = text[start:end].strip()
        date_m = DATE_RE.search(entry_text)
        entries.append({
            "client_id": m.group(1),
            "date": date_m.group(1) if date_m else None,
            "has_flag": "FLAG:" in entry_text,
            "text": entry_text,
        })
    return entries


def ingest(pdf_path: Path = PDF_PATH) -> dict:
    collection = get_chroma_collection(COLLECTION_NAME)
    text = extract_text(pdf_path)
    entries = split_entries(text)

    ids, docs, metadatas = [], [], []
    for i, e in enumerate(entries):
        ids.append(f"{pdf_path.stem}::{e['client_id']}_{i}")
        docs.append(e["text"])
        metadatas.append({
            "source_file": pdf_path.name,
            "client_id": e["client_id"] or "",
            "date": e["date"] or "",
            "has_flag": e["has_flag"],
        })

    n_replaced = replace_chunks_for(collection, "source_file", pdf_path.name, ids, docs, metadatas)
    return {"source_file": pdf_path.name, "n_chunks": len(entries), "replaced_existing_chunks": n_replaced}


def query_call_notes(question: str, n_results: int = 3, client_id: str | None = None) -> list[dict]:
    collection = get_chroma_collection(COLLECTION_NAME)
    where = {"client_id": client_id} if client_id else None
    results = collection.query(query_texts=[question], n_results=max(n_results * 3, 6), where=where)

    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({
            "client_id": meta["client_id"],
            "date": meta["date"],
            "has_flag": meta["has_flag"],
            "similarity": round(1 - dist, 4),
            "text": doc,
        })
    return rerank(question, out, keep=n_results)


def main() -> int:
    r = ingest()
    replaced = f", replaced {r['replaced_existing_chunks']} old chunks" if r["replaced_existing_chunks"] else ""
    print(f"{r['source_file']}: {r['n_chunks']} entries{replaced}")
    collection = get_chroma_collection(COLLECTION_NAME)
    print(f"collection '{COLLECTION_NAME}' now has {collection.count()} chunks total")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        for r in query_call_notes(sys.argv[2], n_results=int(sys.argv[3]) if len(sys.argv) > 3 else 3):
            flag = " [FLAG]" if r["has_flag"] else ""
            print(f"[{r['similarity']}] {r['client_id']} {r['date']}{flag}")
            print(f"  {r['text'][:200].replace(chr(10), ' ')}...\n")
        raise SystemExit(0)

    raise SystemExit(main())

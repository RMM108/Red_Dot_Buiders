"""Ingest data/client_complaint_letters.pdf into a `complaints` Chroma
collection, one chunk per letter (PLAN.md §5 step 4).

Each letter has a reliable header - "Complaint Letter — CPL-YYYY-NNN" -
and states its client via "Client Account: CLxxx" in the body, both used
here. The document's opening register-summary table (before the first
letter) is dropped, since its content (ref/client/date/subject/status) is
just a summary of what's already in each individual letter's own text.

Usage:
    python ingest_complaints.py
    python ingest_complaints.py --query "mis-sale structured note"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader

from chunking import paragraph_chunks
from vector_store import get_chroma_collection, replace_chunks_for

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_PATH = DATA_DIR / "client_complaint_letters.pdf"
COLLECTION_NAME = "complaints"

LETTER_HEADER_RE = re.compile(r"(?m)^Complaint Letter\s.\s(CPL-\d{4}-\d+)$")
CLIENT_RE = re.compile(r"Client Account:\s*(CL\d{3})")
FOOTER_RE = re.compile(r"This document is a fictitious example.*", re.DOTALL)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def split_letters(text: str) -> list[dict]:
    """One chunk per "Complaint Letter" entry. Falls back to plain paragraph
    chunking (chunking.py) if a future complaints register doesn't use this
    header pattern at all - same fallback shape as ingest_policies.py."""
    text = FOOTER_RE.sub("", text).strip()
    matches = list(LETTER_HEADER_RE.finditer(text))

    if not matches:
        return [{"complaint_ref": None, "client_id": None, "text": chunk} for chunk in paragraph_chunks(text)]

    letters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        letter_text = text[start:end].strip()
        client_m = CLIENT_RE.search(letter_text)
        letters.append({
            "complaint_ref": m.group(1),
            "client_id": client_m.group(1) if client_m else None,
            "text": letter_text,
        })
    return letters


def ingest(pdf_path: Path = PDF_PATH) -> dict:
    collection = get_chroma_collection(COLLECTION_NAME)
    text = extract_text(pdf_path)
    letters = split_letters(text)

    ids, docs, metadatas = [], [], []
    for i, letter in enumerate(letters):
        ids.append(f"{pdf_path.stem}::{letter['complaint_ref'] or i}")
        docs.append(letter["text"])
        metadatas.append({
            "source_file": pdf_path.name,
            "complaint_ref": letter["complaint_ref"] or "",
            "client_id": letter["client_id"] or "",
        })

    n_replaced = replace_chunks_for(collection, "source_file", pdf_path.name, ids, docs, metadatas)
    return {"source_file": pdf_path.name, "n_chunks": len(letters), "replaced_existing_chunks": n_replaced}


def query_complaints(question: str, n_results: int = 3, client_id: str | None = None) -> list[dict]:
    collection = get_chroma_collection(COLLECTION_NAME)
    where = {"client_id": client_id} if client_id else None
    results = collection.query(query_texts=[question], n_results=n_results, where=where)

    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({
            "complaint_ref": meta["complaint_ref"],
            "client_id": meta["client_id"],
            "similarity": round(1 - dist, 4),
            "text": doc,
        })
    return out


def main() -> int:
    r = ingest()
    replaced = f", replaced {r['replaced_existing_chunks']} old chunks" if r["replaced_existing_chunks"] else ""
    print(f"{r['source_file']}: {r['n_chunks']} letters{replaced}")
    collection = get_chroma_collection(COLLECTION_NAME)
    print(f"collection '{COLLECTION_NAME}' now has {collection.count()} chunks total")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        for r in query_complaints(sys.argv[2], n_results=int(sys.argv[3]) if len(sys.argv) > 3 else 3):
            print(f"[{r['similarity']}] {r['complaint_ref']} ({r['client_id']})")
            print(f"  {r['text'][:200].replace(chr(10), ' ')}...\n")
        raise SystemExit(0)

    raise SystemExit(main())

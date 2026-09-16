"""Ingest data/complex_product_risk_acknowledgement_forms.pdf into an
`ack_forms` Chroma collection (PLAN.md §5 step 4).

The PDF contains one completed example (CL001) and one BLANK TEMPLATE. The
template is tagged is_blank_template=True and excluded from
`query_ack_forms`'s default results, per the §3.2 finding: a naive retriever
could otherwise return the blank template as if it were evidence of a signed
acknowledgement for some other client. Pass include_blank_template=True
explicitly to retrieve it (e.g. to hand a compliance officer the template
itself), never as a substitute for a real signature.

Usage:
    python ingest_ack_forms.py
    python ingest_ack_forms.py --query "signed acknowledgement CL001"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader

from vector_store import get_chroma_collection, replace_chunks_for

DATA_DIR = Path(__file__).parent / "data"
PDF_PATH = DATA_DIR / "complex_product_risk_acknowledgement_forms.pdf"
COLLECTION_NAME = "ack_forms"

RECORD_HEADER_RE = re.compile(r"(?m)^(COMPLETED EXAMPLE.*|BLANK TEMPLATE)$")
CLIENT_RE = re.compile(r"/\s*(CL\d{3})\b")
FOOTER_RE = re.compile(r"This document is a fictitious example.*", re.DOTALL)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def split_records(text: str) -> list[dict]:
    text = FOOTER_RE.sub("", text).strip()
    matches = list(RECORD_HEADER_RE.finditer(text))

    records = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        record_text = text[start:end].strip()
        is_blank = m.group(1).startswith("BLANK")
        client_m = None if is_blank else CLIENT_RE.search(record_text)
        records.append({
            "record_type": "blank_template" if is_blank else "completed",
            "is_blank_template": is_blank,
            "client_id": client_m.group(1) if client_m else None,
            "text": record_text,
        })
    return records


def ingest(pdf_path: Path = PDF_PATH) -> dict:
    collection = get_chroma_collection(COLLECTION_NAME)
    text = extract_text(pdf_path)
    records = split_records(text)

    ids, docs, metadatas = [], [], []
    for i, rec in enumerate(records):
        ids.append(f"{pdf_path.stem}::{rec['record_type']}_{i}")
        docs.append(rec["text"])
        metadatas.append({
            "source_file": pdf_path.name,
            "client_id": rec["client_id"] or "",
            "is_blank_template": rec["is_blank_template"],
        })

    n_replaced = replace_chunks_for(collection, "source_file", pdf_path.name, ids, docs, metadatas)
    return {"source_file": pdf_path.name, "n_chunks": len(records), "replaced_existing_chunks": n_replaced}


def query_ack_forms(question: str, n_results: int = 3, client_id: str | None = None,
                     include_blank_template: bool = False) -> list[dict]:
    collection = get_chroma_collection(COLLECTION_NAME)
    conditions = []
    if client_id:
        conditions.append({"client_id": client_id})
    if not include_blank_template:
        conditions.append({"is_blank_template": False})
    where = conditions[0] if len(conditions) == 1 else ({"$and": conditions} if conditions else None)

    results = collection.query(query_texts=[question], n_results=n_results, where=where)

    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({
            "client_id": meta["client_id"],
            "is_blank_template": meta["is_blank_template"],
            "similarity": round(1 - dist, 4),
            "text": doc,
        })
    return out


def main() -> int:
    r = ingest()
    replaced = f", replaced {r['replaced_existing_chunks']} old chunks" if r["replaced_existing_chunks"] else ""
    print(f"{r['source_file']}: {r['n_chunks']} records{replaced}")
    collection = get_chroma_collection(COLLECTION_NAME)
    print(f"collection '{COLLECTION_NAME}' now has {collection.count()} chunks total")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        for r in query_ack_forms(sys.argv[2], n_results=int(sys.argv[3]) if len(sys.argv) > 3 else 3):
            tag = " [BLANK TEMPLATE]" if r["is_blank_template"] else ""
            print(f"[{r['similarity']}] {r['client_id']}{tag}")
            print(f"  {r['text'][:200].replace(chr(10), ' ')}...\n")
        raise SystemExit(0)

    raise SystemExit(main())

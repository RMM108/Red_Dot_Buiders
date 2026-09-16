"""Ingest policy PDFs into a persistent Chroma vector database for the AI copilot to retrieve.

Design choice: policy text is split deterministically (regex on numbered section
headers), not via an LLM. Both policy documents in this corpus (POL-KYC-004,
POL-INV-011) use a consistent "N. Section Title" numbering with no LLM needed to
find the boundaries - and for compliance text specifically, the exact wording
matters for citations, so nothing should paraphrase or reformat it. One chunk =
one numbered section, which also matches how the golden dataset and PLAN.md cite
these documents (e.g. "policy_investment_suitability.pdf Section 4").

A document that doesn't follow the numbered-section pattern (fewer than 2 header
matches) falls back to fixed-size paragraph chunking with overlap, so a future
policy PDF with a different layout still gets ingested rather than skipped -
each chunk is tagged with its chunk_type ("section" vs "paragraph") so the
retrieval side can tell which kind of citation it's dealing with.

Embeddings: OpenAI text-embedding-3-small. Store: a local persistent Chroma
collection at ./chroma_db (gitignored). Re-ingesting a document (e.g. a revised
policy) deletes and replaces all of that document's existing chunks by
document_code first, so a policy revision doesn't leave stale sections behind
or duplicate the ones that are unchanged.

Usage:
    python ingest_policies.py                              # ingest data/policy_*.pdf
    python ingest_policies.py path/to/new_policy.pdf        # ingest one file
    python ingest_policies.py --query "complex product concentration limit"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader

# rerank.py lives at the repository root (sibling of data_processing/); make
# sure it's importable regardless of how this script is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking import extract_page_offsets, page_range_for_span
from vector_store import CHROMA_DIR, get_chroma_collection, keyword_boosted_query, replace_chunks_for
from rerank import rerank

DATA_DIR = Path(__file__).parent.parent / "data"
COLLECTION_NAME = "policies"

SECTION_HEADER_RE = re.compile(r"(?m)^(\d{1,2})\.\s+(.+)$")
FOOTER_RE = re.compile(r"This document is a fictitious example.*", re.DOTALL)
PARAGRAPH_CHUNK_SIZE = 1200
PARAGRAPH_CHUNK_OVERLAP = 200


# --------------------------------------------------------------------------
# 1. Deterministic parsing: header block + section boundaries
# --------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_header(text: str) -> dict:
    lines = text.split("\n")

    subtitle_idx = next((i for i, l in enumerate(lines) if "Meridian Peak Wealth Partners" in l), 1)
    policy_name = " ".join(l.strip() for l in lines[0:subtitle_idx] if l.strip())

    stripped = [l.strip() for l in lines if l.strip()]
    document_code = classification = None
    if "Document Code" in stripped:
        document_code = stripped[stripped.index("Document Code") + 1]
    if "Classification" in stripped:
        classification = stripped[stripped.index("Classification") + 1]

    # document_code_full keeps "(Rev. N)" for display; document_code is the
    # stable base code (revision stripped out) used as the chunk-identity key -
    # a revised reissue of the same policy must resolve to the same base code
    # so its chunks replace the old revision's instead of coexisting with it.
    document_code_full = document_code
    revision = None
    if document_code:
        m = re.search(r"\(Rev\.\s*(\d+)\)", document_code)
        if m:
            revision = int(m.group(1))
            document_code = document_code[: m.start()].strip()

    return {
        "policy_name": policy_name,
        "document_code": document_code,
        "document_code_full": document_code_full,
        "classification": classification,
        "revision": revision,
    }


def split_into_sections(text: str, page_offsets: list[int] | None = None) -> list[dict]:
    """One chunk per numbered top-level section. Falls back to paragraph
    chunking if the document doesn't have >= 2 recognizable section headers.
    `page_offsets` (from chunking.extract_page_offsets) is optional so this
    stays callable with just `text`, as existing tests do; a chunk's "page"
    is None without it."""
    text = FOOTER_RE.sub("", text).strip()
    matches = list(SECTION_HEADER_RE.finditer(text))

    if len(matches) < 2:
        return _paragraph_chunks(text, page_offsets)

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        chunks.append({
            "chunk_type": "section",
            "section_number": m.group(1),
            "section_title": m.group(2).strip(),
            "page": page_range_for_span(page_offsets, start, end) if page_offsets else None,
            "text": section_text,
        })
    return chunks


def _paragraph_chunks(text: str, page_offsets: list[int] | None = None) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + PARAGRAPH_CHUNK_SIZE, len(text))
        chunks.append({
            "chunk_type": "paragraph",
            "section_number": None,
            "section_title": None,
            "page": page_range_for_span(page_offsets, start, end) if page_offsets else None,
            "text": text[start:end].strip(),
        })
        if end == len(text):
            break
        start = end - PARAGRAPH_CHUNK_OVERLAP
    return chunks


# --------------------------------------------------------------------------
# 2. Vector store
# --------------------------------------------------------------------------

def get_collection():
    return get_chroma_collection(COLLECTION_NAME)


def ingest_policy(pdf_path: Path, collection=None) -> dict:
    pdf_path = Path(pdf_path)
    collection = collection or get_collection()

    text = extract_text(pdf_path)
    page_offsets = extract_page_offsets(pdf_path)
    header = parse_header(text)
    chunks = split_into_sections(text, page_offsets)

    doc_code = header["document_code"] or pdf_path.stem

    ids, docs, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_code}::{chunk['chunk_type']}_{chunk['section_number'] or i}"
        ids.append(chunk_id)
        docs.append(chunk["text"])
        metadatas.append({
            "source_file": pdf_path.name,
            "policy_name": header["policy_name"],
            "document_code": doc_code,
            "document_code_full": header["document_code_full"] or doc_code,
            "classification": header["classification"] or "",
            "revision": header["revision"] if header["revision"] is not None else -1,
            "chunk_type": chunk["chunk_type"],
            "section_number": chunk["section_number"] or "",
            "section_title": chunk["section_title"] or "",
            "page": chunk["page"] or "",
        })

    # replace any existing chunks for this document_code so a revised
    # policy doesn't leave stale/duplicate sections behind
    n_replaced = replace_chunks_for(collection, "document_code", doc_code, ids, docs, metadatas)

    return {
        "source_file": pdf_path.name,
        "policy_name": header["policy_name"],
        "document_code": doc_code,
        "revision": header["revision"],
        "n_chunks": len(chunks),
        "chunk_type": chunks[0]["chunk_type"] if chunks else None,
        "replaced_existing_chunks": n_replaced,
    }


def ingest_all(pdf_paths: list[Path]) -> list[dict]:
    collection = get_collection()
    return [ingest_policy(p, collection) for p in pdf_paths]


# --------------------------------------------------------------------------
# 3. Retrieval (what the chatbot's get_policy_section / search_documents tool calls)
# --------------------------------------------------------------------------

def query_policy(question: str, n_results: int = 3, document_code: str | None = None) -> list[dict]:
    """See vector_store.keyword_boosted_query for why this isn't plain
    vector search - found by testing: querying "20% concentration guideline"
    put POL-INV-011 Section 4 (the section that actually states the 20% cap)
    *last* out of 14 chunks, below "Purpose" and "Client Risk Rating"."""
    collection = get_collection()
    where = {"document_code": document_code} if document_code else None
    candidates = keyword_boosted_query(collection, question, n_results=n_results, where=where)

    out = []
    for c in candidates:
        meta = c["meta"]
        out.append({
            "policy_name": meta["policy_name"],
            "document_code": meta["document_code"],
            "section": f"Section {meta['section_number']}: {meta['section_title']}" if meta["section_number"] else "(paragraph chunk)",
            "page": meta.get("page") or None,
            "similarity": c["similarity"],
            "text": c["doc"],
        })
    # Cross-encoder re-rank within the keyword-boosted candidate set. Chunks
    # containing the exact numeric token stay high (the cross-encoder scores
    # them jointly against the query), while the overall ordering improves.
    return rerank(question, out, keep=n_results)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_paths", type=Path, nargs="*", help="Policy PDF(s) to ingest. Defaults to data/policy_*.pdf")
    parser.add_argument("--query", type=str, help="Run a retrieval smoke test against the vector DB instead of ingesting")
    parser.add_argument("--n-results", type=int, default=3)
    args = parser.parse_args()

    if args.query:
        results = query_policy(args.query, n_results=args.n_results)
        print(f'query: "{args.query}"\n')
        for r in results:
            print(f"[{r['similarity']}] {r['policy_name']} — {r['section']}")
            print(f"  {r['text'][:220].replace(chr(10), ' ')}...\n")
        return 0

    paths = args.pdf_paths or sorted(DATA_DIR.glob("policy_*.pdf"))
    if not paths:
        print("no policy PDFs found", file=sys.stderr)
        return 1

    for p in paths:
        if not p.exists():
            print(f"error: {p} does not exist", file=sys.stderr)
            return 1

    reports = ingest_all(paths)
    for r in reports:
        replaced = f", replaced {r['replaced_existing_chunks']} old chunks" if r["replaced_existing_chunks"] else ""
        print(f"{r['source_file']} -> {r['policy_name']!r} ({r['document_code']}, rev {r['revision']}): "
              f"{r['n_chunks']} {r['chunk_type']} chunks{replaced}")

    collection = get_collection()
    print(f"\ncollection '{COLLECTION_NAME}' now has {collection.count()} chunks total, persisted at {CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

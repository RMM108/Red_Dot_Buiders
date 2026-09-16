"""Ingest data/client_correspondence.json into a `correspondence` Chroma
collection, one chunk per email thread (PLAN.md §5 step 4).

No deterministic-segmentation work needed here, unlike the PDFs - the JSON
is already structured per thread and every thread already carries
related_client_id (verified: all 7 threads have it, none null), so this is
mostly formatting each thread's messages into one readable chunk and
attaching that id as metadata directly.

Usage:
    python ingest_correspondence.py
    python ingest_correspondence.py --query "LRS remittance headroom"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vector_store import get_chroma_collection, replace_chunks_for

DATA_DIR = Path(__file__).parent.parent / "data"
JSON_PATH = DATA_DIR / "client_correspondence.json"
COLLECTION_NAME = "correspondence"


def format_thread(thread: dict) -> str:
    lines = [f"Subject: {thread['subject']}"]
    for m in thread["messages"]:
        lines.append(f"\nFrom: {m['from']}\nTo: {m['to']}\nDate: {m['date']}\n{m['body']}")
    return "\n".join(lines)


def ingest(json_path: Path = JSON_PATH) -> dict:
    collection = get_chroma_collection(COLLECTION_NAME)
    with open(json_path, encoding="utf-8") as f:
        threads = json.load(f)["email_threads"]

    n_replaced_total = 0
    for t in threads:
        dates = [m["date"] for m in t["messages"]]
        metadata = {
            "source_file": json_path.name,
            "thread_id": t["thread_id"],
            "subject": t["subject"],
            "client_id": t.get("related_client_id") or "",
            "date_start": min(dates) if dates else "",
            "date_end": max(dates) if dates else "",
            "n_messages": len(t["messages"]),
        }
        n_replaced_total += replace_chunks_for(
            collection, "thread_id", t["thread_id"],
            [t["thread_id"]], [format_thread(t)], [metadata],
        )

    return {"source_file": json_path.name, "n_chunks": len(threads), "replaced_existing_chunks": n_replaced_total}


def query_correspondence(question: str, n_results: int = 3, client_id: str | None = None) -> list[dict]:
    collection = get_chroma_collection(COLLECTION_NAME)
    where = {"client_id": client_id} if client_id else None
    results = collection.query(query_texts=[question], n_results=n_results, where=where)

    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({
            "thread_id": meta["thread_id"],
            "subject": meta["subject"],
            "client_id": meta["client_id"],
            "date_range": f"{meta['date_start']} to {meta['date_end']}",
            "similarity": round(1 - dist, 4),
            "text": doc,
        })
    return out


def main() -> int:
    r = ingest()
    print(f"{r['source_file']}: {r['n_chunks']} threads")
    collection = get_chroma_collection(COLLECTION_NAME)
    print(f"collection '{COLLECTION_NAME}' now has {collection.count()} chunks total")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        for r in query_correspondence(sys.argv[2], n_results=int(sys.argv[3]) if len(sys.argv) > 3 else 3):
            print(f"[{r['similarity']}] {r['thread_id']} ({r['client_id']}) — {r['subject']} [{r['date_range']}]")
            print(f"  {r['text'][:200].replace(chr(10), ' ')}...\n")
        raise SystemExit(0)

    raise SystemExit(main())

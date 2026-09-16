"""Embed the narrative half of the fund fact sheets into a `fund_factsheets`
Chroma collection (PLAN.md §5 step 3 / §4.1's "both stores, split by field
type").

Reads data/processed/fund_factsheets_structured.json (built by
ingest_factsheet.py) rather than re-parsing the PDFs - that extraction is
already grounded, so this step only has to decide chunk boundaries and
embed, not extract. Numeric/lookup fields (SRI, minimum investment,
key_facts rows) are deliberately NOT duplicated here - they live in the
`funds` / `fund_key_facts` SQL tables (ingest_clients.py) where a suitability
check can join against them. This collection carries only what a semantic
search over "which funds are risky for X reason" or a citation needs
verbatim: the objective, who it's for, the risks, and any extra_information
that didn't fit the structured schema.

One chunk per section per fund (objective / who-for / key-risks / each
extra_information item), tagged with fund_name + section so retrieval can
be scoped to one fund or searched across all of them.

Usage:
    python ingest_fund_vectors.py
"""

from __future__ import annotations

import json
from pathlib import Path

from vector_store import get_chroma_collection, replace_chunks_for

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
FUNDS_JSON = PROCESSED_DIR / "fund_factsheets_structured.json"
COLLECTION_NAME = "fund_factsheets"


def sections_for_fund(fund: dict) -> list[dict]:
    sections = [{
        "section": "objective",
        "text": fund["fund_objective_or_product_description"],
    }]

    if fund["who_is_this_for"]:
        sections.append({
            "section": "who_is_this_for",
            "text": "Who is this fund/product designed for?\n" + "\n".join(f"- {b}" for b in fund["who_is_this_for"]),
        })

    if fund["key_risks"]:
        sections.append({
            "section": "key_risks",
            "text": "Key risks:\n" + "\n".join(f"- {b}" for b in fund["key_risks"]),
        })

    for i, item in enumerate(fund.get("extra_information") or []):
        sections.append({"section": f"extra_information_{i}", "text": item})

    return sections


def ingest_fund(fund: dict, collection) -> dict:
    fund_name = fund["fund_name"]
    sections = sections_for_fund(fund)

    ids, docs, metadatas = [], [], []
    for s in sections:
        ids.append(f"{fund_name}::{s['section']}")
        docs.append(s["text"])
        metadatas.append({
            "fund_name": fund_name,
            "document_code": fund["document_code"] or "",
            "source_file": fund["source_file"],
            "section": s["section"],
            "summary_risk_indicator": fund["summary_risk_indicator"] or -1,
        })

    n_replaced = replace_chunks_for(collection, "fund_name", fund_name, ids, docs, metadatas)
    return {"fund_name": fund_name, "n_chunks": len(sections), "replaced_existing_chunks": n_replaced}


def query_fund_factsheets(question: str, n_results: int = 3, fund_name: str | None = None) -> list[dict]:
    collection = get_chroma_collection(COLLECTION_NAME)
    where = {"fund_name": fund_name} if fund_name else None
    results = collection.query(query_texts=[question], n_results=n_results, where=where)

    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({
            "fund_name": meta["fund_name"],
            "section": meta["section"],
            "summary_risk_indicator": meta["summary_risk_indicator"],
            "similarity": round(1 - dist, 4),
            "text": doc,
        })
    return out


def main() -> int:
    if not FUNDS_JSON.exists():
        print(f"error: {FUNDS_JSON} not found - run ingest_factsheet.py / fund_factsheet_ingestion.ipynb first")
        return 1

    with open(FUNDS_JSON, encoding="utf-8") as f:
        funds = json.load(f)

    collection = get_chroma_collection(COLLECTION_NAME)
    for fund in funds:
        r = ingest_fund(fund, collection)
        replaced = f", replaced {r['replaced_existing_chunks']} old chunks" if r["replaced_existing_chunks"] else ""
        print(f"{r['fund_name']}: {r['n_chunks']} chunks{replaced}")

    print(f"\ncollection '{COLLECTION_NAME}' now has {collection.count()} chunks total")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        results = query_fund_factsheets(sys.argv[2], n_results=int(sys.argv[3]) if len(sys.argv) > 3 else 3)
        for r in results:
            print(f"[{r['similarity']}] {r['fund_name']} (SRI {r['summary_risk_indicator']}) — {r['section']}")
            print(f"  {r['text'][:200].replace(chr(10), ' ')}...\n")
        raise SystemExit(0)

    raise SystemExit(main())

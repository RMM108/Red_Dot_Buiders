"""Load clients, holdings, transactions, and the structured half of the fund
fact sheets into the SQLite store (PLAN.md §5 step 1-2).

Applies the §3.1 data-cleaning findings at load time rather than as a later
pass:
  - risk_profile / kyc_status / pep_status are split into a clean category
    column plus a separate narrative/date column, instead of leaving a
    category and a paragraph mixed in one string.
  - transactions.product_name has 2 rows that are not actually products
    ("Portfolio Rebalancing (...)", "Cash (Investment Top-Up)") - these are
    flagged via is_product_transaction rather than silently treated as
    fund lookups.
  - product_crosswalk cross-references every product name seen in holdings/
    transactions against the funds table, so the "Singapore Government Bond
    Fund has no fact sheet" gap (and any future one like it) is a queryable
    fact, not something a tool has to discover by a failed lookup.

This is a full-refresh load: every run clears and reloads all tables from
the source files, since clients_portfolio.json is the single source of
truth for client data (unlike the fact sheet / policy pipelines, which
ingest one new document at a time).

Usage:
    python ingest_clients.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

import db

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLIENTS_JSON = DATA_DIR / "clients_portfolio.json"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
FUNDS_JSON = PROCESSED_DIR / "fund_factsheets_structured.json"


# --------------------------------------------------------------------------
# §3.1 field cleaning
# --------------------------------------------------------------------------

def split_risk_profile(raw: str) -> tuple[str, Optional[str], Optional[str]]:
    """"Conservative (revised down from Growth on 2026-08-10; ...)" ->
    ("Conservative", "revised down from Growth on 2026-08-10; ...", "2026-08-10")"""
    m = re.match(r"^(.+?)\s*\((.+)\)$", raw)
    if not m:
        return raw.strip(), None, None
    category, note = m.group(1).strip(), m.group(2).strip()
    date_m = re.search(r"\d{4}-\d{2}-\d{2}", note)
    return category, note, (date_m.group(0) if date_m else None)


def split_kyc_status(raw: str) -> tuple[str, Optional[str], Optional[str]]:
    """"Verified - last refreshed 2026-06-30; enhanced due diligence completed (UHNW tier)"
    -> ("Verified", "2026-06-30", "enhanced due diligence completed (UHNW tier)")"""
    m = re.match(r"^(.+?)\s*-\s*last refreshed\s*(\d{4}-\d{2}-\d{2})\s*(?:;\s*(.+))?$", raw)
    if not m:
        return raw.strip(), None, None
    status, date, note = m.group(1).strip(), m.group(2), m.group(3)
    return status, date, (note.strip() if note else None)


def normalize_product_name(name: str) -> str:
    """Match key only, never for display. The fact-sheet-derived fund_name
    (OCR'd/extracted from the PDF title, e.g. "... Note — Series 7") and the
    hand-entered holdings/transactions product_name (e.g. "... Note Series 7",
    no dash at all) disagree on dash characters and on whether a separator
    is present at all - found by the product_crosswalk build below initially
    flagging 3 funds that do have fact sheets as gaps. Stripping all dash-like
    characters before comparing fixes it without touching the display names."""
    s = re.sub(r"[‐-―\-]", " ", name)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def split_pep_status(raw: str) -> tuple[str, Optional[str]]:
    """"Not a PEP" -> ("Not a PEP", None)
    "PEP-adjacent - former member of ..." -> ("PEP-adjacent", "former member of ...")
    Note: the separator must have surrounding whitespace, since "PEP-adjacent"
    itself contains a bare hyphen that must NOT be treated as the split point."""
    if raw.strip().lower() == "not a pep":
        return "Not a PEP", None
    m = re.match(r"^(.+?)\s+-\s+(.+)$", raw)
    if not m:
        return raw.strip(), None
    return m.group(1).strip(), m.group(2).strip()


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def clear_tables(conn) -> None:
    for table in ["product_crosswalk", "fund_asset_allocation", "fund_key_facts",
                  "transactions", "holdings", "funds", "clients"]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def load_clients_and_holdings(conn) -> None:
    with open(CLIENTS_JSON, encoding="utf-8") as f:
        clients = json.load(f)["clients"]

    for c in clients:
        risk_profile, risk_profile_note, risk_profile_changed_date = split_risk_profile(c["risk_profile"])
        kyc_status, kyc_refresh_date, kyc_note = split_kyc_status(c["kyc_status"])
        pep_status, pep_note = split_pep_status(c["pep_status"])

        conn.execute(
            """INSERT INTO clients (
                client_id, name, nationality, residency_country, age, occupation,
                marital_status, net_worth_band, investor_status, base_currency, aum_sgd,
                risk_profile, risk_profile_note, risk_profile_changed_date, risk_score_1_to_10,
                investment_objective, relationship_manager, servicing_branch,
                kyc_status, kyc_refresh_date, kyc_note, pep_status, pep_note,
                source_of_wealth, last_portfolio_review_date, suitability_flag, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                c["client_id"], c["name"], c["nationality"], c["residency_country"], c["age"], c["occupation"],
                c["marital_status"], c["net_worth_band"], c["investor_status"], c["base_currency"], c["aum_sgd"],
                risk_profile, risk_profile_note, risk_profile_changed_date, c["risk_score_1_to_10"],
                c["investment_objective"], c["relationship_manager"], c["servicing_branch"],
                kyc_status, kyc_refresh_date, kyc_note, pep_status, pep_note,
                c["source_of_wealth"], c["last_portfolio_review_date"], c["suitability_flag"], c["notes"],
            ),
        )

        for h in c["portfolio_holdings"]:
            conn.execute(
                """INSERT INTO holdings (client_id, product_name, asset_class, allocation_pct, value_sgd, currency)
                   VALUES (?,?,?,?,?,?)""",
                (c["client_id"], h["product_name"], h["asset_class"], h["allocation_pct"], h["value_sgd"], h["currency"]),
            )

    conn.commit()
    print(f"loaded {len(clients)} clients, "
          f"{sum(len(c['portfolio_holdings']) for c in clients)} holdings")


def load_transactions(conn, known_products: set[str]) -> None:
    with open(TRANSACTIONS_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        is_product = normalize_product_name(r["product_name"]) in known_products
        conn.execute(
            """INSERT INTO transactions (
                transaction_id, client_id, date, transaction_type, product_name,
                is_product_transaction, amount, currency, status, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                r["transaction_id"], r["client_id"], r["date"], r["transaction_type"], r["product_name"],
                int(is_product), float(r["amount"]), r["currency"], r["status"], r["notes"] or None,
            ),
        )

    conn.commit()
    n_non_product = sum(1 for r in rows if normalize_product_name(r["product_name"]) not in known_products)
    print(f"loaded {len(rows)} transactions ({n_non_product} flagged is_product_transaction=0)")


def load_funds(conn) -> None:
    with open(FUNDS_JSON, encoding="utf-8") as f:
        funds = json.load(f)

    for fund in funds:
        conn.execute(
            """INSERT INTO funds (
                fund_name, product_type, document_subtitle, summary_risk_indicator,
                summary_risk_indicator_label, as_of_date, document_code, classification,
                minimum_investment, base_currency, source_file
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fund["fund_name"], fund["product_type"], fund["document_subtitle"],
                fund["summary_risk_indicator"], fund["summary_risk_indicator_label"],
                fund["as_of_date"], fund["document_code"], fund["classification"],
                fund["minimum_investment"], fund["base_currency"], fund["source_file"],
            ),
        )
        for kf in fund["key_facts"]:
            conn.execute(
                "INSERT INTO fund_key_facts (fund_name, label, value) VALUES (?,?,?)",
                (fund["fund_name"], kf["label"], kf["value"]),
            )
        for a in (fund["asset_allocation"] or []):
            conn.execute(
                "INSERT INTO fund_asset_allocation (fund_name, asset_type, pct_of_fund) VALUES (?,?,?)",
                (fund["fund_name"], a["asset_type"], a["pct_of_fund"]),
            )

    conn.commit()
    print(f"loaded {len(funds)} funds, "
          f"{sum(len(f['key_facts']) for f in funds)} key facts, "
          f"{sum(len(f['asset_allocation'] or []) for f in funds)} asset-allocation rows")


def build_product_crosswalk(conn) -> None:
    fund_by_norm = {normalize_product_name(r["fund_name"]): r["fund_name"]
                     for r in conn.execute("SELECT fund_name FROM funds")}

    seen_in: dict[str, set[str]] = {}
    for r in conn.execute("SELECT DISTINCT product_name FROM holdings"):
        seen_in.setdefault(r["product_name"], set()).add("holdings")
    for r in conn.execute("SELECT DISTINCT product_name FROM transactions WHERE is_product_transaction = 1"):
        seen_in.setdefault(r["product_name"], set()).add("transactions")

    gaps = []
    for product_name, sources in seen_in.items():
        matched_fund = fund_by_norm.get(normalize_product_name(product_name))
        conn.execute(
            "INSERT INTO product_crosswalk (product_name, has_factsheet, fund_name, seen_in) VALUES (?,?,?,?)",
            (product_name, int(matched_fund is not None), matched_fund, ",".join(sorted(sources))),
        )
        if matched_fund is None:
            gaps.append(product_name)
    conn.commit()

    print(f"built product_crosswalk: {len(seen_in)} products, {len(gaps)} without a fact sheet")
    for g in gaps:
        print(f"  no fact sheet: {g!r}")


def main() -> int:
    conn = db.init_db()
    clear_tables(conn)

    load_clients_and_holdings(conn)

    if not FUNDS_JSON.exists():
        print(f"warning: {FUNDS_JSON} not found - run fund_factsheet_ingestion.ipynb / ingest_factsheet.py first")
    else:
        load_funds(conn)

    # a transaction counts as a real product if it's held by anyone (holdings)
    # or has a fact sheet (funds) - covers a fully-redeemed position too
    known_products = {normalize_product_name(r["product_name"]) for r in conn.execute("SELECT DISTINCT product_name FROM holdings")}
    known_products |= {normalize_product_name(r["fund_name"]) for r in conn.execute("SELECT fund_name FROM funds")}

    load_transactions(conn, known_products)
    build_product_crosswalk(conn)

    print(f"\ndatabase ready at {db.DB_PATH}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

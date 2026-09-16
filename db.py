"""Shared SQLite connection + schema for the structured store.

Holds client/KYC/risk data, holdings, transactions, and the structured half
of the fund fact sheets (see PLAN.md §4.1 for why fund fact sheets are split
between this SQL store and the vector store rather than living in only one).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "processed" / "wealth_management.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    client_id                   TEXT PRIMARY KEY,
    name                        TEXT NOT NULL,
    nationality                 TEXT,
    residency_country           TEXT,
    age                         INTEGER,
    occupation                  TEXT,
    marital_status              TEXT,
    net_worth_band              TEXT,
    investor_status             TEXT,
    base_currency               TEXT,
    aum_sgd                     REAL,
    risk_profile                TEXT,   -- cleaned category, e.g. "Conservative"
    risk_profile_note           TEXT,   -- narrative remainder, e.g. "revised down from Growth on ...; not yet rebalanced"
    risk_profile_changed_date   TEXT,   -- parsed date if the note mentions one
    risk_score_1_to_10          INTEGER,
    investment_objective        TEXT,
    relationship_manager        TEXT,
    servicing_branch            TEXT,
    kyc_status                  TEXT,   -- cleaned status, e.g. "Verified"
    kyc_refresh_date            TEXT,
    kyc_note                    TEXT,   -- EDD / source-of-funds narrative remainder
    pep_status                  TEXT,   -- cleaned category, e.g. "Not a PEP" / "PEP-adjacent"
    pep_note                    TEXT,   -- narrative remainder for non-trivial PEP status
    source_of_wealth            TEXT,
    last_portfolio_review_date  TEXT,
    suitability_flag            TEXT,
    notes                       TEXT
);

CREATE TABLE IF NOT EXISTS holdings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id       TEXT NOT NULL REFERENCES clients(client_id),
    product_name    TEXT NOT NULL,
    asset_class     TEXT,
    allocation_pct  REAL,
    value_sgd       REAL,
    currency        TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id          TEXT PRIMARY KEY,
    client_id                TEXT NOT NULL REFERENCES clients(client_id),
    date                      TEXT,
    transaction_type          TEXT,
    product_name              TEXT,
    is_product_transaction    INTEGER,  -- 0 for the non-product rows (e.g. "Portfolio Rebalancing (...)")
    amount                    REAL,
    currency                  TEXT,
    status                    TEXT,
    notes                     TEXT
);

CREATE TABLE IF NOT EXISTS funds (
    fund_name                     TEXT PRIMARY KEY,
    product_type                  TEXT,
    document_subtitle             TEXT,
    summary_risk_indicator        INTEGER,
    summary_risk_indicator_label  TEXT,
    as_of_date                    TEXT,
    document_code                 TEXT,
    classification                TEXT,
    minimum_investment            TEXT,
    base_currency                 TEXT,
    source_file                   TEXT
    -- narrative fields (objective, who_is_this_for, key_risks, extra_information)
    -- deliberately live in the `fund_factsheets` vector collection, not here -
    -- see PLAN.md section 4.1 "both stores, split by field type"
);

CREATE TABLE IF NOT EXISTS fund_key_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fund_name   TEXT NOT NULL REFERENCES funds(fund_name),
    label       TEXT NOT NULL,
    value       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fund_asset_allocation (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fund_name     TEXT NOT NULL REFERENCES funds(fund_name),
    asset_type    TEXT NOT NULL,
    pct_of_fund   TEXT
);

CREATE TABLE IF NOT EXISTS product_crosswalk (
    product_name    TEXT PRIMARY KEY,
    has_factsheet   INTEGER NOT NULL,
    fund_name       TEXT REFERENCES funds(fund_name),
    seen_in         TEXT  -- comma-separated: "holdings", "transactions"
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> sqlite3.Connection:
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    return conn

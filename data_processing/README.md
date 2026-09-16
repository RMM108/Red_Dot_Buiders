# Data Processing Pipeline

This folder holds every script that turns the raw files in `../data/` into
the structured/vector stores the AI copilot's agent (`agent.py`) queries.
See `../PLAN.md` for the full design rationale, data-quality findings, and
evaluation results — this README documents how the pipeline actually runs.

## Directory layout

```
hackathon/
├── data/                          raw source files (see data/README.md)
│   └── processed/                 pipeline outputs
│       ├── fund_factsheets_structured.json/.csv   (ingest_factsheet.py output)
│       └── wealth_management.db                   (SQLite, ingest_clients.py output)
├── chroma_db/                     persistent Chroma vector store (all *_vectors/ingest_* scripts write here)
├── data_processing/                <- this folder: every pipeline + the agent
│   ├── db.py                       SQLite schema + connection helper
│   ├── vector_store.py             Chroma connection helper (shared by all vector ingestion scripts)
│   ├── entity_crosswalk.py         client name -> client_id resolver
│   ├── ingest_clients.py           raw client/holdings/transactions/funds -> SQLite
│   ├── ingest_factsheet.py         new fund fact sheet PDF -> structured JSON (VLM)
│   ├── ingest_fund_vectors.py      structured JSON's narrative fields -> Chroma
│   ├── ingest_policies.py          policy PDFs -> Chroma
│   ├── ingest_call_notes.py        RM call notes log PDF -> Chroma
│   ├── ingest_complaints.py        complaint letters PDF -> Chroma
│   ├── ingest_ack_forms.py         risk acknowledgement forms PDF -> Chroma
│   ├── ingest_correspondence.py    email correspondence JSON -> Chroma
│   ├── agent.py                    tool-calling agent orchestrator (the copilot itself)
│   ├── run_evaluation.py           runs the agent against the golden dataset, scores it
│   ├── eda.ipynb                   exploratory analysis of the raw structured data
│   └── fund_factsheet_ingestion.ipynb   original 10-fund fact sheet extraction run
└── .env                             OPENAI_API_KEY (gitignored)
```

Every script locates `../data`, `../chroma_db`, and `../.env` via
`Path(__file__).parent.parent`, since the scripts live one level down in
`data_processing/`. All pipelines were re-run after this reorg to confirm
paths resolve correctly (same row/chunk counts as before the move: 15
clients, 70 holdings, 55 transactions, 10 funds in SQLite; 14+30+10+2+7+2 =
65 chunks across the 6 Chroma collections).

## Data architecture

Two stores, populated by different pipelines, queried by different agent
tools. The split follows one rule: **exact, filterable, joinable facts go
in SQL; narrative text that needs semantic search or verbatim citation goes
in the vector store.** A few sources (client records, fund fact sheets) are
deliberately split across both — see `PLAN.md` §4.1 for why.

### Structured store — SQLite

- **File:** `data/processed/wealth_management.db`
- **Engine:** Python's built-in `sqlite3` (`db.py`), no server
- **Built by:** `ingest_clients.py` (full-refresh reload every run)

| Table | Rows | Source | Contents |
|---|---|---|---|
| `clients` | 15 | `clients_portfolio.json` | KYC/risk fields, **cleaned**: `risk_profile`/`kyc_status`/`pep_status` each split into a clean category column + a separate narrative/date column (e.g. `risk_profile="Conservative"`, `risk_profile_note="revised down from Growth on 2026-08-10..."`) |
| `holdings` | 70 | `clients_portfolio.json` | one row per client holding |
| `transactions` | 55 | `transactions.csv` | ledger rows; `is_product_transaction` flags the 2 rows that aren't actually product trades (e.g. "Portfolio Rebalancing (...)") |
| `funds` | 10 | `fund_factsheets_structured.json` | SRI, minimum investment, base currency, document code — **structured fields only**, no narrative text |
| `fund_key_facts` | 127 | ″ | one row per Key Facts table entry per fund |
| `fund_asset_allocation` | 27 | ″ | one row per allocation line per fund |
| `product_crosswalk` | 11 | derived (join) | every product name seen in holdings/transactions, matched (via dash/whitespace-normalized name matching) against `funds.fund_name`; flags the 1 real gap — "Singapore Government Bond Fund" has no fact sheet |

### Vector store — ChromaDB

- **Directory:** `chroma_db/` (persistent client, `chromadb.PersistentClient`)
- **Embedding model:** OpenAI `text-embedding-3-small`, via `chromadb.utils.embedding_functions.OpenAIEmbeddingFunction`
- **Connection helper:** `vector_store.py` (`get_chroma_collection(name)`, shared by every script below)

| Collection | Chunks | Built by | Chunk boundary |
|---|---|---|---|
| `policies` | 14 | `ingest_policies.py` | one numbered policy section (regex on `"N. Section Title"`, deterministic — not LLM-chunked, to preserve exact compliance wording) |
| `fund_factsheets` | 30 | `ingest_fund_vectors.py` | one chunk per narrative section per fund (objective, who-for, key risks, each `extra_information` item) — the **narrative half** of each fact sheet; SRI/min-investment/etc. live in SQL instead |
| `call_notes` | 10 | `ingest_call_notes.py` | one chunk per RM call-note entry (split on `"CLxxx — Name"` / `"Date:"`) |
| `complaints` | 2 | `ingest_complaints.py` | one chunk per letter (split on `"Complaint Letter — CPL-YYYY-NNN"`) |
| `correspondence` | 7 | `ingest_correspondence.py` | one chunk per email thread |
| `ack_forms` | 2 | `ingest_ack_forms.py` | filled record + blank template, **separately tagged** — the blank template is excluded from `query_ack_forms`'s default results so it can never be mistaken for a real signature |

Every chunk carries metadata (`source_file`, `client_id` where applicable,
`date`, `document_code`/`section`, etc.) used both for citation locators and
for scoped retrieval (e.g. "search only this client's documents").

## Models used

| Purpose | Model | Where |
|---|---|---|
| Fund fact sheet extraction (reads PDF pages as **images**, not text) | `gpt-4o-mini` (vision) via OpenAI Responses API, structured output (Pydantic schema) | `ingest_factsheet.py` |
| Agent reasoning / tool-calling loop | `gpt-4o-mini` via OpenAI Responses API (`responses.parse`, function tools + structured final answer) | `agent.py` |
| Evaluation grading (LLM-as-judge) | `gpt-4o-mini` via OpenAI Responses API | `run_evaluation.py` |
| Embeddings | `text-embedding-3-small` | `vector_store.py`, used by every `ingest_*` vector script |

No other LLM/VLM providers are used. `temperature=0` is pinned for both the
agent loop and the evaluation judge for reproducibility.

## Processing steps, in order

Run from the project root (`hackathon/`), not from inside `data_processing/`
(see the path note above — this matters more now than it did before).

1. **`ingest_factsheet.py`** *(only needed once, or when a new fact sheet PDF arrives)*
   Reads a fund fact sheet PDF as page images (VLM), extracts a structured
   record (fund name, SRI, key facts, asset allocation, objective, who
   it's for, key risks, `extra_information` for anything off-schema),
   grounds every extracted value against the PDF's own text layer, and
   merges it into `data/processed/fund_factsheets_structured.json/.csv`.
   The 10 fact sheets already in this dataset were extracted this way (see
   `fund_factsheet_ingestion.ipynb` for the original batch run).

2. **`ingest_clients.py`**
   Loads `clients_portfolio.json`, `transactions.csv`, and
   `fund_factsheets_structured.json` into SQLite, applying field cleaning
   and building `product_crosswalk`. Full refresh every run — safe to
   re-run any time the source files change.

3. **`ingest_fund_vectors.py`**
   Reads `fund_factsheets_structured.json` (no PDF re-parsing) and embeds
   its narrative fields into the `fund_factsheets` Chroma collection.

4. **`ingest_policies.py`**, **`ingest_call_notes.py`**,
   **`ingest_complaints.py`**, **`ingest_ack_forms.py`**,
   **`ingest_correspondence.py`**
   Each parses its one source document/file into chunks (deterministically
   — regex on known header patterns, no LLM) and embeds them into their own
   Chroma collection. Independent of each other and of steps 1–3; run in
   any order.

5. **`agent.py`**
   Not an ingestion step — this is the copilot itself. Wires 5 tools
   (`query_client_db`, `query_transactions`, `get_fund_factsheet`,
   `get_policy_section`, `search_documents`) over the stores built above
   into a tool-calling loop. Every tool result is tagged with a `ref_id`;
   the model's final answer cites by `ref_id` only, which is resolved back
   to a real locator string from the tool's own metadata — never written
   freehand by the model. Also carries an `abstained` flag + reason for
   when evidence is missing, conflicting, or a policy's scope is
   genuinely ambiguous as applied.

6. **`run_evaluation.py`**
   Runs `agent.py` against the 5 golden questions in
   `data/golden_dataset_for_RAG_evaluation.xlsx`, judges each response with
   an LLM-judge call, and writes
   `data/processed/golden_dataset_for_RAG_evaluation_completed.xlsx`. Last
   run: 4/5 factually correct — see `PLAN.md` §5 for the specific failure
   modes found and fixed along the way (a retrieval defect that buried an
   exact policy threshold, a missed-correspondence reasoning error, and
   three rounds of prompt tuning to get the deliberately-ambiguous CL011
   case to abstain consistently).

## Supporting modules (not run directly)

- **`db.py`** — SQLite schema (`CREATE TABLE` statements) and
  `get_connection()`/`init_db()`. Imported by `ingest_clients.py` and
  `agent.py`.
- **`vector_store.py`** — `get_chroma_collection(name)` and
  `replace_chunks_for(...)` (deletes a document's old chunks before
  upserting new ones, so re-ingesting a revised document doesn't leave
  duplicates). Imported by every `ingest_*` vector script.
- **`entity_crosswalk.py`** — `resolve_client(name)` resolves a free-text
  client name mention (as a user might type it) to a `client_id`, and
  returns an explicit ambiguity result rather than guessing when two
  clients share a name fragment (e.g. two clients here share the surname
  "Rahman"). Used internally by `agent.py`'s `query_client_db` tool.

# Feature: Ingest Persona Documents (Call Notes, Complaints) into Qdrant

The following plan should be complete, but it's important that you validate documentation and
codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.
This feature deliberately does **not** touch structured persona data (the `clients`/`holdings`
tables, or a new "persona structured store") — those already exist in `db.py` and are out of scope
here. This feature touches only the two unstructured PDFs on the persona side of the data layer.

## Feature Description

Today, `data/rm_call_notes_log.pdf` and `data/client_complaint_letters.pdf` are chunked and embedded
into a local Chroma persistent store (`./chroma_db`, collections `call_notes` and `complaints`) by
`ingest_call_notes.py` and `ingest_complaints.py`. This feature ports those two pipelines to write to
Qdrant instead, as the first concrete step toward the `persona_kb` vector store described in
`PRD.md` §13 (Future Considerations) — without pulling in any of the rest of that larger
architecture (no LangGraph, no MySQL, no FastAPI, no RM ACL). Every other Chroma collection
(`policies`, `fund_factsheets`, `correspondence`, `ack_forms`) is untouched and keeps using Chroma.

## User Story

As a developer evolving the persona knowledge base toward the target architecture in `PRD.md`
As I want to ingest `rm_call_notes_log.pdf` and `client_complaint_letters.pdf` into Qdrant, keyed
and filterable by `client_id`
So that the persona-side narrative evidence lives in a server-backed vector store (matching the
`persona_kb` design) while the rest of the working system keeps running unmodified.

## Problem Statement

`ingest_call_notes.py` and `ingest_complaints.py` are hard-wired to `vector_store.py`'s Chroma
helpers (`get_chroma_collection`, `replace_chunks_for`). There is no Qdrant integration anywhere in
the codebase (`requirements.txt` has no `qdrant-client`; no `docker-compose.yml`; no `.env.example`).
Moving just these two persona document types to Qdrant requires a parallel storage helper, a running
Qdrant instance, and careful preservation of the two modules' public functions
(`query_call_notes`, `query_complaints`) since `agent.py` imports and calls them directly
(`agent.py:36`, `agent.py:39`, and inside `tool_search_documents` at `agent.py:249-259`) and expects
an unchanged return shape.

## Solution Statement

1. Add a new `qdrant_store.py` shared helper (mirrors `vector_store.py`'s role) that owns the Qdrant
   client connection, OpenAI-embedding calls (Qdrant has no built-in embedding function like Chroma's
   `embedding_functions.OpenAIEmbeddingFunction`), collection get-or-create, and a
   delete-by-filter-then-upsert "replace" helper equivalent to `replace_chunks_for`.
2. Add a small, real client_id cross-check against the existing `clients` SQLite table (already
   built and populated by `ingest_clients.py` — not new structured ingestion) so a header parsed as
   `CL0XX` that doesn't exist in `clients` is caught at ingest time instead of silently indexed.
3. **Two-tier chunking, per-client-ID first, size-bounded second** (this section's own subsection
   below has the full rationale and measured data): keep the existing, already-verified regex
   header-splitting logic (`ENTRY_HEADER_RE`, `DATE_RE`, `LETTER_HEADER_RE`, `CLIENT_RE`,
   `FOOTER_RE`) completely unchanged as the *primary* chunk boundary — this already produces one
   chunk per client entry/letter, i.e. "chunk by client ID." Add a *secondary* pass, extracted into a
   new shared `chunking.py` module reusing `ingest_policies.py`'s already-tuned fallback constants
   (`PARAGRAPH_CHUNK_SIZE = 1200`, `PARAGRAPH_CHUNK_OVERLAP = 200`), that further splits any single
   client entry/letter whose text exceeds that size into overlapping sub-chunks, all tagged with the
   same parent `client_id`/`date`/`complaint_ref` metadata.
4. Port `ingest_call_notes.py` and `ingest_complaints.py`: swap the storage calls inside `ingest()`
   and `query_call_notes()`/`query_complaints()` to target `qdrant_store.py` instead of
   `vector_store.py`, and apply the new two-tier chunking from point 3. The public function
   signatures are unchanged; the **return dict shape gains two new keys** (`chunk_index`, `n_chunks`)
   needed to disambiguate citations for the (real, measured — see below) case where one letter
   produces multiple sub-chunks. This requires one small, justified edit to `agent.py`'s `ref_id`
   construction (§ below) — everything else about `agent.py` is unchanged.
5. Add local Qdrant via Docker (`docker-compose.yml`, single `qdrant` service) plus a checked-in
   `.env.example` documenting `QDRANT_URL` (default `http://localhost:6333`) and the existing
   `OPENAI_API_KEY`.

## Chunking Strategy (validated against the actual PDFs)

**Rule:** chunk primarily by client ID (one chunk per call-note entry / per complaint letter, which
is what the existing regex splitters already do); if that unit's text exceeds
`PARAGRAPH_CHUNK_SIZE` (1200 chars — reusing `ingest_policies.py:45-46`'s already-shipped fallback
constants, not inventing new ones), further split it with the same overlapping fixed-size chunking
`ingest_policies.py:115-129` already uses as its own fallback, so a "too long" case is never a
special new algorithm, just a second application of a pattern this codebase has already tested.

**Order of operations matters**: parse `client_id` / `date` / `has_flag` / `complaint_ref` from the
**full** entry/letter text first (the regexes rely on the header/body appearing once, near the top —
e.g. `Client Account: CLxxx` in a complaint letter), *then* size-split the text if needed. Every
resulting sub-chunk is tagged with the same parent metadata — do not re-run `CLIENT_RE` etc. against
each sub-chunk independently, since a later sub-chunk (starting mid-letter) will not contain the
header line at all.

**Measured against the real data** (extracted via `pypdf`, chars after footer-stripping, same
extraction path `extract_text()` in both modules already uses):

| Document | Units | Lengths (chars) | vs. 1200-char threshold |
|---|---|---|---|
| `rm_call_notes_log.pdf` | 10 entries | 398–921 (avg 624) | **Never exceeds it** — every entry stays exactly one chunk, so per-client-ID chunking is the *whole* story for call notes with today's data. |
| `client_complaint_letters.pdf` | 2 letters | 1943, 2033 | **Both exceed it** — each letter becomes 2 sub-chunks (chars `[0:1200]` and `[1000:len]`, per `_paragraph_chunks`'s 200-char overlap step), so the size-bounded fallback is not hypothetical here — it fires for 100% of the complaints in this dataset today.

**Consequence this surfaces (found by tracing the data, not assumed):** `agent.py`'s `ref_id`
construction for complaints is `f"complaint_{r['complaint_ref']}"` (`agent.py:257`) — one `ref_id`
per letter, not per chunk. With both letters now yielding 2 sub-chunks each, two `EvidencePool.add()`
calls in one `tool_search_documents` invocation could use the *same* `ref_id` for different chunk
text, and the second call silently overwrites the first in `EvidencePool.entries` (a plain dict keyed
by `ref_id` — see `agent.py:103-113`). Not a crash, but a citation could resolve to the wrong half of
a letter. Resolved by appending a chunk index to the `ref_id` **only** when a parent unit produced
more than one chunk (`agent.py:250-253` and `256-259` — see the dedicated task below); the common
single-chunk case (all call notes, and any future short complaint letter) is byte-identical to
today's `ref_id` format.

## Feature Metadata

**Feature Type**: New Capability (parallel storage backend for 2 of 6 existing vector collections)
**Estimated Complexity**: Medium
**Primary Systems Affected**: `ingest_call_notes.py`, `ingest_complaints.py`, new `qdrant_store.py`;
indirectly exercised (not modified) by `agent.py` and `run_evaluation.py`
**Dependencies**: `qdrant-client` (new), local Qdrant server (Docker), existing `openai` package
(already a dependency, used directly instead of via Chroma's embedding function wrapper)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — IMPORTANT: READ THESE BEFORE IMPLEMENTING

- `vector_store.py` (all 42 lines) — Why: the exact pattern to mirror. `get_chroma_collection()` →
  becomes `get_qdrant_collection()`; `replace_chunks_for()` (lines 33-41: delete matching
  `match_field`/`match_value`, then upsert) → becomes the Qdrant delete-by-filter + upsert
  equivalent. Keep the same "replace by a matching field, return count replaced" contract.
- `ingest_call_notes.py` (all 116 lines) — Why: the module being ported.
  - Lines 32-34: `ENTRY_HEADER_RE`, `DATE_RE`, `FOOTER_RE` — reuse unchanged, do not touch.
  - Lines 42-58 `split_entries()` — the header-splitting logic (finding entry boundaries) is reused
    unchanged; **add** the secondary size-bounded pass at the end of each entry's processing (see
    Chunking Strategy above) — parse `client_id`/`date`/`has_flag` from the full entry text exactly as
    today (lines 51-56), then if `len(entry_text) > PARAGRAPH_CHUNK_SIZE`, call
    `chunking.paragraph_chunks(entry_text)` and emit one dict per sub-chunk instead of one dict for
    the whole entry. With today's data (max 921 chars) this branch never fires for call notes — keep
    it anyway, it's the same safety net `ingest_policies.py` already relies on.
  - Lines 61-78 `ingest()` — the `get_chroma_collection`/`replace_chunks_for` calls (lines 62, 77)
    change to their `qdrant_store.py` equivalents; the id built at line 68 must incorporate a
    chunk index (e.g. `f"{pdf_path.stem}::{e['client_id']}_{i}_{chunk_idx}"`) so multiple sub-chunks
    of one entry get distinct, deterministic Qdrant point IDs.
  - Lines 81-95 `query_call_notes()` — the **return shape** (list of dicts with keys `client_id`,
    `date`, `has_flag`, `similarity`, `text`) is a near-hard contract — `agent.py:250-253` iterates
    this exact shape — **plus two new keys**, `chunk_index` (1-based position among this entry's
    sub-chunks) and `n_chunks` (total sub-chunks for this entry), always present (`1`/`1` for the
    common unsplit case) so `agent.py` can build a collision-free `ref_id` without guessing.
- `ingest_complaints.py` (all 109 lines) — Why: same porting pattern as call notes, and the module
  where the size-bounded branch actually fires today (§ Chunking Strategy: both letters exceed 1200
  chars).
  - Lines 29-31: `LETTER_HEADER_RE`, `CLIENT_RE`, `FOOTER_RE` — reuse unchanged.
  - Lines 39-54 `split_letters()` — same treatment as `split_entries()` above: parse
    `complaint_ref`/`client_id` from the full letter text first (line 48), then size-split into
    `chunking.paragraph_chunks(letter_text)` sub-chunks if `len(letter_text) > PARAGRAPH_CHUNK_SIZE`
    — which, per the measured lengths (1943, 2033 chars), it always will for this dataset's two
    letters, each producing 2 sub-chunks.
  - Lines 57-73 `ingest()` — same swap and chunk-indexed-id treatment as `ingest_call_notes.py`.
  - Lines 76-89 `query_complaints()` — return shape (`complaint_ref`, `client_id`, `similarity`,
    `text`, **plus `chunk_index`/`n_chunks`**) is consumed by `agent.py:256-259`.
- `entity_crosswalk.py` lines 37-56 (`build_client_index`) — Why: shows the existing, correct pattern
  for reading the `clients` table via `db.get_connection()`. Mirror this pattern (not
  `resolve_client()`'s fuzzy matching itself — these two PDFs already embed exact `CLxxx` IDs in
  their headers, so no fuzzy resolution is needed here) for the new client_id existence check.
- `db.py` lines 109-114 (`get_connection()`) — Why: the connection helper to reuse for the client_id
  validation query; do not open SQLite connections any other way.
- `agent.py` lines 34-41 (imports), lines 103-113 (`EvidencePool`, dict-keyed-by-`ref_id`), and lines
  244-281 (`tool_search_documents`) — Why: the **only** consumer of
  `query_call_notes`/`query_complaints`. Confirms the exact return-dict keys this feature must
  preserve, and confirms `ref_id` construction (`f"call_note_{r['client_id']}_{r['date']}"` at line
  251, `f"complaint_{r['complaint_ref']}"` at line 257) happens in `agent.py`, not in the ingestion
  module — do not move that logic elsewhere. **This is the one place in `agent.py` this feature
  touches** (see Chunking Strategy above and the dedicated task below) — every other line of
  `agent.py` is unchanged.
- `ingest_policies.py` lines 45-46 (`PARAGRAPH_CHUNK_SIZE = 1200`, `PARAGRAPH_CHUNK_OVERLAP = 200`)
  and lines 115-129 (`_paragraph_chunks()`) — Why: the exact algorithm and tuned constants to reuse
  (via a new shared `chunking.py`, not by importing this private function directly, since
  `ingest_policies.py` is Chroma-specific and out of scope to modify) for the "content too long"
  fallback this feature's chunking strategy calls for. Do not re-derive different constants — reusing
  the already-shipped 1200/200 values keeps chunk-size behavior consistent across every document type
  in the system.
- `requirements.txt` (all 39 lines) — Why: add `qdrant-client` here; confirms `openai` is already a
  dependency (no new embedding-provider dependency needed, per the embedding-provider decision below).
- `.gitignore` (lines 7-9: `*.chroma/`, `chroma_db/`, `*.db`) — Why: add the new Qdrant local storage
  directory here using the same pattern.
- `PLAN.md` §3.2 and §4.1 — Why: documents the exact chunk-boundary rules (`"CLxxx — Name"` /
  `"Date:"` for call notes, `"Complaint Letter — CPL-YYYY-NNN"` for complaints) these scripts already
  implement correctly — do not redesign chunking, only the storage backend changes.
- `PRD.md` §6, §8, §13 — Why: this feature is explicitly framed there as the first step toward the
  `persona_kb` / Qdrant direction; §13 names `Qdrant` as the eventual vector DB and explicitly notes
  today's system uses Chroma — this plan is the bridge, scoped narrowly per the answered clarifying
  questions (Qdrant for persona docs only; keep OpenAI embeddings; reuse existing parsing; read-only
  reference to the existing `clients` table; local Qdrant via Docker).

### New Files to Create

- `chunking.py` — small, backend-agnostic shared module: `paragraph_chunks(text: str, chunk_size:
  int = 1200, overlap: int = 200) -> list[str]`, a direct extraction of `ingest_policies.py`'s
  `_paragraph_chunks()` algorithm (same constants, same sliding-window logic) generalized to take
  raw text in and chunk strings out, decoupled from that module's Chroma-specific chunk-dict shape.
  This is the "typical chunking" fallback both `ingest_call_notes.py` and `ingest_complaints.py` call
  when a per-client-ID unit is too long. (`ingest_policies.py` itself is not modified to use this —
  its own private `_paragraph_chunks()` is left as-is to keep this feature's footprint small; noted
  as an optional follow-up dedup in NOTES.)
- `qdrant_store.py` — shared Qdrant connection, OpenAI embedding helper, collection get-or-create,
  and delete-by-filter + upsert "replace" helper (mirrors `vector_store.py`). Also owns the small
  client_id-existence check shared by both ingestion scripts.
- `docker-compose.yml` — single `qdrant` service (`qdrant/qdrant` image), ports `6333`/`6334`,
  volume-mounted local storage. Deliberately minimal — not the full stack from `PRD.md` §13.
- `.env.example` — first one in the repo. Documents `OPENAI_API_KEY` (already required today, just
  never templated) and the new `QDRANT_URL` (default `http://localhost:6333`).

### Files to Modify

- `ingest_call_notes.py` — swap storage backend (imports + `ingest()` + `query_call_notes()` bodies)
  **and** add the two-tier chunking pass inside `split_entries()` (§ Chunking Strategy). The
  entry-boundary regexes themselves (`ENTRY_HEADER_RE`, `DATE_RE`) are untouched.
- `ingest_complaints.py` — same two changes, for `split_letters()` / `LETTER_HEADER_RE`/`CLIENT_RE`.
- `agent.py` — **one targeted edit**, `tool_search_documents`'s two `ref_id` lines (250-253 for call
  notes, 256-259 for complaints): append a chunk-index suffix only when `n_chunks > 1` for that
  result. No other line in `agent.py` changes — see Chunking Strategy above for why this is
  necessary (a real citation-collision bug the new chunking would otherwise introduce, confirmed by
  the actual complaint-letter lengths, not a hypothetical).
- `requirements.txt` — add `qdrant-client`.
- `.gitignore` — add the local Qdrant storage directory (if using a bind-mounted volume rather than a
  named Docker volume — see Task 6).

### Relevant Documentation — READ BEFORE IMPLEMENTING

- [Qdrant Python client — Collections](https://python-client.qdrant.tech/qdrant_client.qdrant_client.QdrantClient#qdrant_client.qdrant_client.QdrantClient.create_collection)
  - `create_collection` / `collection_exists` — Why: needed for the get-or-create pattern
    (`vector_store.py`'s `get_or_create_collection` has no direct Qdrant equivalent; must check
    existence first, matching Chroma's idempotent behavior).
- [Qdrant Python client — Points](https://python-client.qdrant.tech/qdrant_client.qdrant_client.QdrantClient#qdrant_client.qdrant_client.QdrantClient.upsert)
  - `upsert` with `PointStruct` — Why: replaces Chroma's `collection.upsert(ids=, documents=,
    metadatas=)`; Qdrant upsert takes explicit vectors (must embed ourselves) and payload dicts.
  - **GOTCHA**: Qdrant point IDs must be an unsigned integer or a valid UUID — arbitrary strings
    (e.g. the current Chroma id `"rm_call_notes_log::CL006_3"`) are rejected. Use
    `uuid.uuid5(uuid.NAMESPACE_URL, original_string_id)` for a deterministic UUID so re-ingesting the
    same PDF produces the same point IDs (required for the "replace" idempotency this codebase relies
    on everywhere else — see `PLAN.md` §4.1 "Re-ingesting a revised policy replaces its old chunks...
    rather than duplicating them"). Keep the original string id in the payload (e.g. `"chunk_id"`) for
    debuggability.
- [Qdrant Python client — Filtering & delete](https://python-client.qdrant.tech/qdrant_client.qdrant_client.QdrantClient#qdrant_client.qdrant_client.QdrantClient.delete)
  - `delete(collection_name, points_selector=FilterSelector(filter=...))` with `Filter`/
    `FieldCondition`/`MatchValue` — Why: this is the Qdrant equivalent of Chroma's
    `collection.get(where={...})` + `collection.delete(ids=...)` used in `replace_chunks_for`.
- [Qdrant Python client — Query points](https://python-client.qdrant.tech/qdrant_client.qdrant_client.QdrantClient#qdrant_client.qdrant_client.QdrantClient.query_points)
  - `query_points(collection_name, query=<vector>, limit=, query_filter=Filter(...))` returning
    `.points` (each with `.score`, `.payload`) — Why: replaces `collection.query(query_texts=,
    n_results=, where=)`. **GOTCHA**: `query_points` (not the older, deprecated `search()`) requires
    `qdrant-client >= 1.10` — pin the requirement accordingly and confirm the installed version
    supports it (`python -c "import qdrant_client; print(qdrant_client.__version__)"`).
  - **GOTCHA**: unlike Chroma's `distances` (lower = more similar, converted via `1 - dist` in the
    current code — see `ingest_call_notes.py:92`), Qdrant's default Cosine `query_points` returns a
    `score` where higher = more similar directly. Do not apply the `1 - dist` conversion; use
    `round(point.score, 4)` directly for the `similarity` field to keep it comparable to the existing
    Chroma-backed collections' output.
- [OpenAI Python SDK — Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
  - Why: Qdrant has no built-in embedding function, unlike
    `chromadb.utils.embedding_functions.OpenAIEmbeddingFunction` (`vector_store.py:22-25`). This
    feature must call `client.embeddings.create(model="text-embedding-3-small", input=[...])`
    directly and pass the resulting vectors to `upsert`/`query_points`. `text-embedding-3-small`
    produces 1536-dimension vectors — this is the `size` for `VectorParams` at collection creation.

### Patterns to Follow

**Naming Conventions:**
`snake_case` for functions/variables, `SCREAMING_SNAKE_CASE` for module-level constants
(`COLLECTION_NAME`, regex patterns), matching every existing `ingest_*.py` file.

**Idempotent re-ingestion pattern** (`vector_store.py:33-41`, used identically by every `ingest_*.py`
module): delete existing chunks matching a stable identifying field (here, `source_file`), then
upsert the fresh set — never append-only. The new Qdrant helper must preserve this exact contract:
"delete matching `match_field`==`match_value`, then upsert; return count replaced."

**Module structure** (every `ingest_*.py` file): `extract_text()` → `split_*()` (pure parsing, no I/O
to the store) → `ingest()` (calls the store) → `query_*()` (calls the store) → `main()` with a
`--query` CLI flag for manual spot-checking, e.g.:
```python
if len(sys.argv) > 1 and sys.argv[1] == "--query":
    for r in query_call_notes(sys.argv[2], n_results=int(sys.argv[3]) if len(sys.argv) > 3 else 3):
        ...
    raise SystemExit(0)
raise SystemExit(main())
```
Preserve this exact CLI pattern in both modified files — it is this project's only manual-testing
mechanism (there is no `pytest` suite; see Testing Strategy).

**Error handling:** this codebase does not use custom exception classes or try/except-heavy
defensive code — see `tool_get_fund_factsheet` in `agent.py:194-226`, which returns an explicit
`{"error": "..."}` JSON string rather than raising. Follow the same style for the new client_id
validation: log a plain `print()` warning for an unresolvable client_id found in a document header
(this is fixed, already-inspected synthetic data — per `PLAN.md` every `CLxxx` header in these two
PDFs is already verified valid — so this is a defensive check for future document revisions, not an
expected-to-fire path today) rather than raising or silently dropping the chunk.

**Logging pattern:** plain `print()` statements summarizing counts after each ingestion run (see
`ingest_call_notes.py:99-104`, `ingest_complaints.py:94-98`) — no logging framework anywhere in this
codebase. Match this.

**Parse-then-chunk ordering (new pattern this feature introduces, but consistent with the codebase's
existing "parse metadata from the full unit" style):** always extract identifying metadata
(`client_id`, `date`, `has_flag`, `complaint_ref`) from a call-note entry's or complaint letter's
**complete** text before any size-based sub-splitting happens. A sub-chunk starting mid-entry or
mid-letter will not itself contain the header line the regexes match on — re-running the regexes
per sub-chunk would silently produce `None`/missing metadata for every sub-chunk after the first.

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation
Stand up the Qdrant server and the shared storage/chunking helpers before touching either ingestion
script.

**Tasks:**
- Add `docker-compose.yml` with a single `qdrant` service
- Add `.env.example` documenting `OPENAI_API_KEY` and `QDRANT_URL`
- Add `qdrant-client` to `requirements.txt`
- Create `chunking.py`: `paragraph_chunks()`, extracted from `ingest_policies.py`'s fallback
- Create `qdrant_store.py`: connection, embedding helper, get-or-create collection, replace helper,
  client_id validation helper

### Phase 2: Core Implementation
Port both ingestion scripts to the new helper, one at a time, validating each independently before
moving to the next.

**Tasks:**
- Port `ingest_call_notes.py`
- Spot-check call notes ingestion end-to-end
- Port `ingest_complaints.py`
- Spot-check complaints ingestion end-to-end

### Phase 3: Integration
Apply the one required `agent.py` edit, then confirm the rest of the system — which was never told
anything else changed — still works.

**Tasks:**
- Update `agent.py`'s two `ref_id` builders in `tool_search_documents` for chunk-index disambiguation
- Run `agent.py` against questions that require `search_documents` to hit `call_notes`/`complaints`
- Run `run_evaluation.py` and confirm no regression versus the documented 4/5 baseline (`PRD.md` §11)

### Phase 4: Testing & Validation
- Idempotency check: re-run both ingestion scripts, confirm chunk counts are stable (no duplication)
- Client_id validation check: confirm the existence check actually runs against real `clients` rows

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently
testable.

### CREATE docker-compose.yml

- **IMPLEMENT**: A single `qdrant` service using the official `qdrant/qdrant` image, exposing
  `6333:6333` (REST) and `6334:6334` (gRPC), with a bind-mounted volume `./qdrant_storage:/qdrant/storage`
  for persistence across restarts (mirrors the local-persistence intent of `CHROMA_DIR` in
  `vector_store.py:16`).
- **PATTERN**: None in-repo (first Docker file) — use the standard Qdrant quickstart compose service
  definition.
- **GOTCHA**: Do not add `backend`/`frontend`/`mysql` services from `PRD.md`'s §13 future
  architecture — this compose file is scoped to Qdrant only, for this feature.
- **VALIDATE**: `docker compose up -d qdrant && curl -s http://localhost:6333/collections` (expect
  a JSON response like `{"result":{"collections":[]},"status":"ok","time":...}`)

### CREATE .env.example

- **IMPLEMENT**: Document `OPENAI_API_KEY=` (already required by `agent.py`/`vector_store.py` today,
  just never templated) and `QDRANT_URL=http://localhost:6333`.
- **PATTERN**: `agent.py:29` / `vector_store.py:12-14` (`load_dotenv()`) show what's already read from
  `.env` today.
- **VALIDATE**: `cat .env.example` shows both keys; confirm neither is a real secret.

### UPDATE requirements.txt

- **IMPLEMENT**: Add `qdrant-client>=1.10` under a new `# --- Qdrant (persona vector store) ---`
  comment block, following the existing grouped-comment style (`# --- RAG / vector search ---` etc.,
  `requirements.txt:18-30`).
- **PATTERN**: `requirements.txt:18-30` (existing grouping style).
- **VALIDATE**: `pip install -r requirements.txt` completes without error.

### CREATE chunking.py

- **IMPLEMENT**: `PARAGRAPH_CHUNK_SIZE = 1200`, `PARAGRAPH_CHUNK_OVERLAP = 200` (module constants,
  same values as `ingest_policies.py:45-46`). `paragraph_chunks(text: str, chunk_size: int =
  PARAGRAPH_CHUNK_SIZE, overlap: int = PARAGRAPH_CHUNK_OVERLAP) -> list[str]` — the same sliding-window
  loop as `ingest_policies.py:115-129`'s `_paragraph_chunks()`, but returning plain strings (not the
  `{"chunk_type": ..., "section_number": ..., ...}` dicts that function builds, since call notes and
  complaints have their own, different metadata shape) so it's reusable by both ingestion modules
  without pulling in policy-specific fields.
- **PATTERN**: `ingest_policies.py:115-129` (`_paragraph_chunks`) — same algorithm, generalized return
  type.
- **IMPORTS**: none beyond the standard library.
- **GOTCHA**: keep the loop's termination condition identical to the source
  (`if end == len(text): break`, then `start = end - overlap`) — an off-by-one here would either drop
  the tail of a long letter or infinite-loop for a very short overflow remainder.
- **VALIDATE**: `python -c "from chunking import paragraph_chunks; cs = paragraph_chunks('x'*2033); print(len(cs), [len(c) for c in cs])"`
  prints `2 [1200, 1033]` — matching the exact complaint-letter split computed during planning.

### CREATE qdrant_store.py

- **IMPLEMENT**:
  - `get_qdrant_client()` — reads `QDRANT_URL` from env (default `http://localhost:6333`), returns a
    `qdrant_client.QdrantClient` instance.
  - `embed_texts(texts: list[str]) -> list[list[float]]` — calls
    `openai.OpenAI().embeddings.create(model="text-embedding-3-small", input=texts)`, returns the
    vectors in input order. `EMBEDDING_MODEL = "text-embedding-3-small"` and
    `EMBEDDING_DIM = 1536` as module constants (mirrors `vector_store.py:17`'s
    `EMBEDDING_MODEL` constant).
  - `get_or_create_collection(name: str)` — `client.collection_exists(name)`; if not,
    `client.create_collection(name, vectors_config=VectorParams(size=EMBEDDING_DIM,
    distance=Distance.COSINE))` (Cosine to match Chroma's `metadata={"hnsw:space": "cosine"}` in
    `vector_store.py:29`, keeping retrieval behavior comparable across both stores). Returns the
    client + collection name (or just the name — collection is stateless in Qdrant, unlike Chroma's
    collection object).
  - `replace_points_for(client, collection_name: str, match_field: str, match_value: str, ids: list,
    texts: list[str], payloads: list[dict]) -> int` — delete-by-filter
    (`Filter(must=[FieldCondition(key=match_field, match=MatchValue(value=match_value))])` via
    `FilterSelector`), then embed `texts` via `embed_texts()` and `client.upsert(collection_name,
    points=[PointStruct(id=..., vector=..., payload=...) for ...])`. Point `id` = deterministic
    `str(uuid.uuid5(uuid.NAMESPACE_URL, original_id))`; store `original_id` in the payload as
    `chunk_id`. Returns the count of points deleted (mirrors `replace_chunks_for`'s return contract in
    `vector_store.py:33-41`).
  - `query_points(client, collection_name: str, query_text: str, n_results: int, where: dict | None)
    -> list[dict]` — embed `query_text` via `embed_texts([query_text])[0]`, build a `Filter` from
    `where` (single-key exact-match dict, e.g. `{"client_id": "CL002"}`, matching the existing
    `where={"client_id": client_id}` usage at `ingest_call_notes.py:83`), call
    `client.query_points(collection_name, query=vector, limit=n_results,
    query_filter=filter_or_None)`, return a list of `{"payload": point.payload, "similarity":
    round(point.score, 4)}` dicts — leave shaping into the final caller-facing dict shape
    (`client_id`/`date`/`has_flag`/`text` etc.) to `ingest_call_notes.py`/`ingest_complaints.py`
    themselves, matching how `vector_store.py` today stays generic and shape-agnostic.
  - `validate_client_ids(client_ids: set[str]) -> set[str]` — open a connection via `db.get_connection()`
    (reuse, don't reimplement — see `entity_crosswalk.py:39` for the exact pattern), query
    `SELECT client_id FROM clients`, return the subset of `client_ids` NOT found in that set (i.e. the
    "unknown" ones, for the caller to warn about).
- **PATTERN**: `vector_store.py` (whole file) for the module's role and the replace-then-upsert
  contract; `entity_crosswalk.py:37-45` for the `db.get_connection()` usage pattern.
- **IMPORTS**: `qdrant_client`, `qdrant_client.models` (`PointStruct`, `VectorParams`, `Distance`,
  `Filter`, `FieldCondition`, `MatchValue`, `FilterSelector`), `openai.OpenAI`, `uuid`, `os`,
  `pathlib.Path`, `dotenv.load_dotenv`, and `db` (for `validate_client_ids`).
- **GOTCHA**: `load_dotenv(dotenv_path=Path(__file__).parent / ".env")` exactly as
  `vector_store.py:14` does — do not rely on `load_dotenv()`'s default CWD-search behavior, for
  consistency with the rest of the codebase.
- **VALIDATE**: `python -c "from qdrant_store import get_qdrant_client, get_or_create_collection; c = get_qdrant_client(); get_or_create_collection(c, 'smoke_test'); print(c.collection_exists('smoke_test'))"`
  prints `True`.

### UPDATE ingest_call_notes.py

- **IMPLEMENT**:
  1. Add `from chunking import PARAGRAPH_CHUNK_SIZE, paragraph_chunks` and replace
     `from vector_store import get_chroma_collection, replace_chunks_for` with
     `from qdrant_store import get_qdrant_client, get_or_create_collection, replace_points_for,
     query_points, validate_client_ids`.
  2. In `split_entries()` (currently lines 42-58): keep the boundary-finding loop (lines 44-49)
     unchanged; after computing `entry_text`, `client_id`, `date`, `has_flag` for one entry (lines
     50-56, unchanged — parse-then-chunk order, see Patterns), branch: if
     `len(entry_text) <= PARAGRAPH_CHUNK_SIZE`, emit one dict as today with added
     `"chunk_index": 1, "n_chunks": 1`; else call `sub_texts = paragraph_chunks(entry_text)` and emit
     one dict per `sub_texts[i]` (`"text": sub_texts[i]`), each carrying the same `client_id`/`date`/
     `has_flag` plus `"chunk_index": i + 1, "n_chunks": len(sub_texts)`.
  3. In `ingest()` (currently lines 61-78): get a Qdrant client, ensure the `call_notes` collection
     exists, build `ids`/`docs`/`metadatas` from the (now possibly-multiple-per-entry) list from step
     2 — the id must include the chunk index for uniqueness, e.g.
     `f"{pdf_path.stem}::{e['client_id']}_{i}_{e['chunk_index']}"`; `metadatas` must include the
     chunk text itself (e.g. `"text": e["text"]`) plus `chunk_index`/`n_chunks`, since Qdrant payloads
     carry both metadata and document body (no separate "documents" list like Chroma's
     `collection.upsert(documents=...)`). Additionally collect the set of parsed `client_id`s and call
     `validate_client_ids()` — `print()` a warning listing any unresolved IDs (do not fail the run),
     then call `replace_points_for(client, "call_notes", "source_file", pdf_path.name, ids, docs,
     metadatas)` in place of the old `replace_chunks_for` call.
  4. In `query_call_notes()` (currently lines 81-95), replace the Chroma `collection.query(...)` call
     with `query_points(client, "call_notes", question, n_results, {"client_id": client_id} if
     client_id else None)`, then reshape each result into the **existing plus two new keys**:
     `{"client_id": payload["client_id"], "date": payload["date"], "has_flag": payload["has_flag"],
     "similarity": ..., "text": payload["text"], "chunk_index": payload["chunk_index"], "n_chunks":
     payload["n_chunks"]}`.
- **PATTERN**: `ingest_call_notes.py:42-58` (`split_entries`, boundary logic unchanged),
  `ingest_call_notes.py:61-95` (structure to preserve), `agent.py:250-253` (the consumer contract —
  now extended with two additive keys, not broken).
- **IMPORTS**: as in step 1 above; no other import changes.
- **GOTCHA**: with today's data every call-note entry takes the `chunk_index=1, n_chunks=1` branch
  (max entry length 921 < 1200) — this is expected, not a sign the sub-chunking code is dead; it's
  the safety net, exercised for real by the complaints module (see below).
- **VALIDATE**: `python ingest_call_notes.py` prints `"rm_call_notes_log.pdf: 10 entries"` (10 chunks
  total — `n_chunks` in the summary output should equal the point count, since no entry splits);
  `python ingest_call_notes.py --query "portfolio rebalancing delay"` returns results in the original
  printed format (`[similarity] client_id date [FLAG]` + snippet).

### UPDATE ingest_complaints.py

- **IMPLEMENT**: Same four-step pattern as `ingest_call_notes.py` above, applied to `split_letters()`
  (currently lines 39-54), `ingest()` (lines 57-73), and `query_complaints()` (lines 76-89). Preserve
  the `{"complaint_ref": ..., "client_id": ..., "similarity": ..., "text": ...}` output shape **plus**
  `chunk_index`/`n_chunks` — this is what `agent.py:256-259` consumes. Point id becomes
  `f"{pdf_path.stem}::{letter['complaint_ref']}_{chunk_index}"`.
- **PATTERN**: `ingest_complaints.py:39-54` (`split_letters`, boundary logic unchanged),
  `ingest_call_notes.py`'s post-port version (just written) as the direct sibling pattern to mirror.
- **GOTCHA (the one that actually fires today)**: both complaint letters (1943, 2033 chars) exceed
  `PARAGRAPH_CHUNK_SIZE`, so **every** ingested complaint produces `n_chunks == 2` — unlike call
  notes, do not assume this branch is a no-op during manual testing; explicitly verify the Qdrant
  `complaints` collection ends up with **4** points (2 letters × 2 sub-chunks), not 2.
- **GOTCHA**: `client_id` can be empty string (`letter["client_id"] or ""` at
  `ingest_complaints.py:69`, for a letter whose body didn't match `CLIENT_RE`) — do not pass an empty
  string into `validate_client_ids()`'s existence check as if it were a real ID; filter it out first.
- **VALIDATE**: `python ingest_complaints.py` prints `"client_complaint_letters.pdf: 4 chunks"` (2
  letters, 2 sub-chunks each — update the summary print at `ingest_complaints.py:94-98` to report
  chunk count, not letter count, since they now differ); `python ingest_complaints.py --query
  "mis-sale structured note"` returns results, each showing its `chunk_index`/`n_chunks` alongside the
  existing `[similarity] complaint_ref (client_id)` line.

### UPDATE agent.py

- **IMPLEMENT**: In `tool_search_documents` (lines 244-281):
  - Line 251: change
    `ref_id = f"call_note_{r['client_id']}_{r['date']}"` to append a suffix only when split:
    `ref_id = f"call_note_{r['client_id']}_{r['date']}"` then
    `if r.get("n_chunks", 1) > 1: ref_id += f"_{r['chunk_index']}"`.
  - Line 257: same treatment —
    `ref_id = f"complaint_{r['complaint_ref']}"` then
    `if r.get("n_chunks", 1) > 1: ref_id += f"_{r['chunk_index']}"`.
- **PATTERN**: the existing `if/else`-free, straight-line style of the surrounding function — this is
  a two-line conditional addition, not a restructure. `r.get("n_chunks", 1)` (not `r["n_chunks"]`)
  keeps this defensive against any other `doc_types` result dict (e.g. `correspondence`,
  `fund_factsheets`) that doesn't carry this key, without needing to touch those branches.
- **GOTCHA**: this is the **only** change to `agent.py` in this entire feature. Do not touch
  `EvidencePool`, `AgentAnswer`, the system prompt, or any other tool function — those are correctly
  described as "untouched" elsewhere in this plan, and this task must stay scoped to exactly these two
  lines.
- **VALIDATE**: `python agent.py "What did Robert Chua's complaint letter say?"` (or another CL002
  question) — inspect the printed citations list; if the complaint's two sub-chunks are both cited in
  one answer, confirm their `ref_id`s differ (e.g. `complaint_CPL-2026-014_1` and
  `complaint_CPL-2026-014_2`), not identical.

### UPDATE .gitignore

- **IMPLEMENT**: Add `qdrant_storage/` near the existing `chroma_db/` entry (`.gitignore:8`).
- **PATTERN**: `.gitignore:7-9`.
- **VALIDATE**: `git status` after running `docker compose up -d` shows `qdrant_storage/` is not
  listed as untracked.

---

## TESTING STRATEGY

This codebase has no `pytest` suite — its established testing pattern (see `PLAN.md` throughout §5)
is targeted manual spot-checks via each ingestion script's `--query` CLI flag, plus end-to-end runs
through `agent.py` and `run_evaluation.py`. Follow that pattern rather than introducing a new test
framework for just this feature.

### Unit-level (manual, via existing `--query` pattern)
- `python ingest_call_notes.py --query "portfolio rebalancing delay"` — expect CL013's entry to rank
  near the top (this is the exact scenario `PLAN.md` §1 names for CL013's de-risking timeline).
- `python ingest_complaints.py --query "mis-sale structured note"` — expect CL002's complaint letter
  (CPL ref) to rank first (the CL002 mis-sale scenario named throughout `PLAN.md`).
- Filtered query: call `query_call_notes("de-risk", client_id="CL013")` from a `python -c` one-liner
  and confirm only CL013 entries are returned (validates the Qdrant `Filter`/`FieldCondition` wiring,
  the direct equivalent of the existing `where={"client_id": ...}` Chroma test path).

### Integration Tests (manual, via existing agent/eval entry points)
- `python agent.py "Has James Sullivan's portfolio been rebalanced since he asked to de-risk?"` —
  this question (`PRD.md` §5 story 3) requires `search_documents` to pull from both the now-Qdrant-backed
  `call_notes` collection and the still-Chroma-backed `correspondence`/`fund_factsheets` collections in
  the same tool call — confirms the two backends coexist correctly inside one `search_documents`
  invocation (`agent.py:244-281`).
- `python run_evaluation.py` — re-run the full golden set and diff the printed summary table against
  the documented baseline in `PRD.md` §11 (4/5 factually correct, including the CL002 row's known
  partial-evidence-trail gap and the CL011 abstention case). The goal is **no regression** — this
  feature is a storage-backend swap, not a retrieval-quality improvement, so scores should be stable
  or better, not worse.

### Edge Cases
- **Idempotent re-ingestion**: run `python ingest_call_notes.py` twice in a row; confirm the second
  run's "replaced N old chunks" count equals the first run's chunk count (10), and the Qdrant
  collection's point count does not grow (`client.count("call_notes")` stays at 10). Repeat for
  `python ingest_complaints.py` — expect a stable count of **4** (2 letters × 2 sub-chunks each), not
  2 — this is the direct Qdrant equivalent of the dedup behavior every other `ingest_*.py` module
  already guarantees via `vector_store.py`'s `replace_chunks_for`, now also proving the chunk-indexed
  point IDs are deterministic across runs (a random-per-run ID scheme would fail this check by
  silently doubling the point count on the second run).
- **Sub-chunk retrieval and citation uniqueness**: query `query_complaints("mis-sale structured
  note")` directly and confirm two distinct results come back for the same `complaint_ref` with
  `chunk_index` 1 and 2; then confirm (via the `agent.py` task's own validation) that citing both in
  one answer produces two distinct `ref_id`s, not a silent overwrite in `EvidencePool`.
- **Unmatched client_id in a document**: temporarily verify the `validate_client_ids()` warning path
  fires by testing it directly against a synthetic ID (e.g.
  `validate_client_ids({"CL999"})` should return `{"CL999"}`) — do not modify the actual PDFs to test
  this, since all real `CLxxx` headers in both PDFs are already confirmed valid (`PLAN.md` §3.2).
- **Empty complaint client_id**: confirm a letter with no `CLIENT_RE` match (if any exist in future
  revisions of the PDF) doesn't get passed into the validation check as a false "unknown ID" warning.
- **Qdrant unreachable**: run an ingestion script with the `qdrant` Docker container stopped; confirm
  it fails with a clear connection error rather than hanging or silently writing nothing — no explicit
  retry/fallback logic is in scope for this feature (matches the codebase's existing no-defensive-code
  style for infra dependencies, e.g. `agent.py`'s `OpenAI()` client has no retry wrapper either).

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness. There is no linter,
formatter, or `pytest` config in this repo today (confirmed: no `pyproject.toml`, `.flake8`,
`ruff.toml`, or `pytest.ini`) — validation is syntax-check + the manual/functional commands below,
consistent with how the rest of the codebase is validated per `PLAN.md`.

### Level 1: Syntax & Style
```
python -m py_compile chunking.py qdrant_store.py ingest_call_notes.py ingest_complaints.py agent.py
```

### Level 2: Unit Tests
```
python ingest_call_notes.py
python ingest_call_notes.py --query "portfolio rebalancing delay"
python ingest_complaints.py
python ingest_complaints.py --query "mis-sale structured note"
```

### Level 3: Integration Tests
```
python agent.py "Has James Sullivan's portfolio been rebalanced since he asked to de-risk?"
python run_evaluation.py
```

### Level 4: Manual Validation
```
docker compose up -d qdrant
curl -s http://localhost:6333/collections/call_notes | python -m json.tool
curl -s http://localhost:6333/collections/complaints | python -m json.tool
```
Confirm `call_notes` reports `"points_count": 10` (one per entry — none split, per the measured
lengths) and `complaints` reports `"points_count": 4` (2 letters × 2 sub-chunks each — **not** 2),
and `"vectors_count"` matches in both (no orphaned/duplicate points from re-running ingestion during
development).

### Level 5: Additional Validation (Optional)
```
python -c "from qdrant_store import get_qdrant_client; c = get_qdrant_client(); print(c.count('call_notes'), c.count('complaints'))"
```

---

## ACCEPTANCE CRITERIA

- [ ] `docker-compose up -d qdrant` brings up a local Qdrant instance reachable at `localhost:6333`
- [ ] `chunking.py` provides `paragraph_chunks()`, reusing `ingest_policies.py`'s exact 1200/200
      constants and sliding-window algorithm
- [ ] `qdrant_store.py` provides connection, embedding, get-or-create, replace, query, and
      client_id-validation helpers, mirroring `vector_store.py`'s contract
- [ ] Both ingestion scripts chunk **primarily by client ID** (one chunk per call-note entry / per
      complaint letter — the existing header-splitting regexes, unchanged) and **secondarily by size**
      (any unit over 1200 chars is further split via `chunking.paragraph_chunks()`), with metadata
      parsed from the full unit *before* any size-splitting
- [ ] `ingest_call_notes.py` produces 10 points total (no entry splits, per measured data);
      `ingest_complaints.py` produces 4 points total (both letters split into 2 each, per measured
      data) — both counts explicitly verified, not assumed
- [ ] `query_call_notes()` and `query_complaints()` return the original dict shape **plus**
      `chunk_index`/`n_chunks` — an additive, non-breaking change
- [ ] `agent.py`'s two `ref_id` builders in `tool_search_documents` are updated to disambiguate
      multi-chunk citations; this is the **only** change to `agent.py` in this feature
- [ ] Re-running either ingestion script is idempotent (no duplicate points; "replaced" count matches
      the prior run's chunk count, including the 4-point complaints case)
- [ ] Every parsed `client_id` is cross-checked against the existing `clients` SQLite table; unknown
      IDs are warned about, not silently indexed or fatally erroring
- [ ] `python run_evaluation.py` shows no regression versus the documented 4/5 baseline
- [ ] All other Chroma collections (`policies`, `fund_factsheets`, `correspondence`, `ack_forms`)
      are completely untouched, and `ingest_policies.py` itself is not modified
- [ ] `requirements.txt`, `.env.example`, `.gitignore`, and `docker-compose.yml` are all updated/added

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task's validation command passed immediately after that task
- [ ] All Level 1-4 validation commands executed successfully
- [ ] Manual testing confirms `agent.py` and `run_evaluation.py` still work end-to-end
- [ ] No regressions in the golden-dataset evaluation score
- [ ] Code follows the existing `ingest_*.py` module structure and style
- [ ] `PRD.md` §8/§13 could be updated in a follow-up to reflect Qdrant now partially in use (left as
      a documentation follow-up, not part of this implementation task list)

---

## NOTES

- **Chunking strategy validation (measured, not assumed)**: extracted actual text from both PDFs via
  `pypdf` during planning. `rm_call_notes_log.pdf`'s 10 entries run 398–921 chars (avg 624) — always
  under the reused 1200-char threshold, so per-client-ID chunking is the complete story for call
  notes with today's data; the size-bounded fallback is a safety net, not something exercised now.
  `client_complaint_letters.pdf`'s 2 letters run 1943 and 2033 chars — **both** exceed 1200 chars, so
  the fallback fires for every complaint in this dataset, each producing exactly 2 sub-chunks
  (`[0:1200]` and `[1000:len]`, per the 200-char overlap step). This asymmetry — one document type
  never needs the fallback, the other always does — is why the plan treats the fallback as a real,
  tested code path rather than defensive code for a hypothetical future document.
- **Why reuse `ingest_policies.py`'s exact 1200/200 constants instead of tuning new ones for these
  document types**: consistency of chunk-size behavior across the whole system was preferred over a
  size tuned specifically to make complaint letters avoid splitting — see the two clarifying
  questions answered before this refinement (reuse the shipped threshold; fix the resulting
  `ref_id`-collision consequence in `agent.py` rather than picking a threshold that dodges it).
- **Why the `agent.py` edit is in scope despite the earlier plan saying "zero changes"**: that
  claim was true under the original (single-chunk-per-unit) chunking assumption. The two-tier
  chunking strategy this refinement adds is a genuine behavior change with a real consequence
  (`ref_id` collisions for the 2 complaint letters, confirmed by measured data, not hypothetical) —
  fixing it is 2 lines in one already-read function, not a scope expansion.
- **Why not touch structured persona data**: the user explicitly scoped this feature to the two PDF
  documents only. The `clients` SQLite table this feature *reads from* (for client_id validation)
  already exists and is fully built (`ingest_clients.py`, done per `PLAN.md` §5 step 1) — this feature
  adds a read-only consumer of it, not a new structured ingestion pipeline, and does not create any
  "persona structured store" concept beyond what already exists.
- **Why `chunking.py` is a new module rather than importing from `ingest_policies.py`**: the source
  function (`_paragraph_chunks`) is private (leading underscore) and returns policy-specific chunk
  dicts (`chunk_type`, `section_number`, `section_title`) that don't apply to call notes/complaints.
  Extracting the reusable part (the string-splitting algorithm itself) into a small shared module
  avoids both an awkward cross-import into a Chroma-specific file and a second, silently-diverging
  copy of the same logic. Refactoring `ingest_policies.py` to also import from `chunking.py` (instead
  of keeping its own private copy) is a reasonable follow-up but is deliberately left out of this
  feature's task list to keep its footprint limited to the persona side, per the original scoping.
- **Why keep OpenAI embeddings instead of switching to Voyage AI now**: avoids introducing a second
  embedding model/dimension into the system while only 2 of 6 vector collections move to Qdrant —
  keeps this feature's blast radius to "storage backend," not "storage backend + embedding model,"
  per the clarifying-question answers. `PRD.md` §13 still names Voyage AI as the target for a later,
  separate decision.
- **Why Chroma stays for the other 4 collections**: `policies`, `fund_factsheets`, `correspondence`,
  and `ack_forms` are explicitly out of scope — moving them is a mechanically identical follow-up
  feature once this one is validated in production use, not bundled in here to keep the change
  reviewable and the rollback surface small.
- **Deterministic point IDs matter**: because `PLAN.md` documents idempotent re-ingestion as a
  load-bearing property of every existing pipeline (re-ingesting a revised policy replaces its old
  chunks "rather than duplicating them"), the `uuid.uuid5` deterministic-ID scheme is not a nice-to-have
  — without it, every re-run of `ingest_call_notes.py`/`ingest_complaints.py` would silently duplicate
  all points in Qdrant, since a random UUID per run would never match a prior run's IDs for the
  delete-by-filter step to catch (the filter is by `source_file`, but a `PointStruct` with a new
  random ID is still a distinct point from Qdrant's perspective if the delete-then-upsert isn't done
  correctly — deterministic IDs are what make the "delete matching source_file, then upsert" pattern
  safe even if implemented slightly differently later).

**Confidence Score**: 8/10 — the porting logic is a direct, well-understood mechanical translation
(Chroma primitives → Qdrant primitives) with no new business logic, and every consumer contract
(`agent.py`'s exact dict-shape expectations, now extended with two additive keys) is pinned down
explicitly, backed by measured character counts rather than assumptions about which branch of the
two-tier chunking strategy actually fires. The main first-attempt risks are (1) the Qdrant client API
surface (`query_points` vs. deprecated `search()`, exact `Filter` construction syntax) not matching
whatever `qdrant-client` version actually installs — worth a quick `pip show qdrant-client` /
doc-version check before writing `qdrant_store.py` — and (2) an off-by-one in the chunk-indexed point
ID or the `ref_id` suffix logic silently breaking idempotent re-ingestion or citation uniqueness for
the complaints collection specifically, since that's the one path exercised by real data rather than
only by the safety-net branch.

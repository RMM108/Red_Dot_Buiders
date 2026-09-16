"""Build the SQL + vector stores from the committed source data.

The SQLite DB (data/processed/wealth_management.db) and the Chroma vector
store (chroma_db/) are gitignored, so they never reach Streamlit Cloud on a
fresh deploy. This module rebuilds both from the committed source files in
data/ (clients_portfolio.json, transactions.csv, fund_factsheets_structured.json,
the policy/call-note/complaint/correspondence/ack-form PDFs & JSON) so the
deployed app is self-sufficient on first launch.

It is idempotent: each ingest pipeline replaces its own chunks/tables, so
re-running is safe. It requires OPENAI_API_KEY (for embeddings), which
app.py ensures is set from Streamlit Secrets before calling this.

Usage (from app.py):
    from bootstrap import ensure_stores
    ensure_stores()
"""

from __future__ import annotations

import sys
from pathlib import Path

# These modules live in data_processing/ and use bare imports (e.g. `import
# db`, `from vector_store import ...`), so this package dir must be on
# sys.path. app.py already inserts it; this guard keeps the module runnable
# standalone too.
_PKG = Path(__file__).parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
# rerank.py lives at the repo root (sibling of data_processing/).
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import db  # noqa: E402
import ingest_ack_forms  # noqa: E402
import ingest_call_notes  # noqa: E402
import ingest_clients  # noqa: E402
import ingest_complaints  # noqa: E402
import ingest_correspondence  # noqa: E402
import ingest_fund_vectors  # noqa: E402
import ingest_policies  # noqa: E402

DB_PATH = db.DB_PATH
CHROMA_PATH = _ROOT / "chroma_db"


def _sql_store_ready() -> bool:
    return DB_PATH.exists()


def _vector_store_ready() -> bool:
    # Chroma creates the dir on first collection access; treat the presence
    # of the policies collection's persisted data as the readiness signal.
    return CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir())


def ensure_stores(force: bool = False) -> list[str]:
    """Build any missing stores. Returns a list of human-readable steps run.

    If ``force`` is True, rebuild everything regardless of current state.
    """
    steps: list[str] = []

    if force or not _sql_store_ready():
        ingest_clients.main()
        steps.append("SQL store (wealth_management.db)")

    if force or not _vector_store_ready():
        # Policies + fund fact sheets use keyword-boosted retrieval; the
        # narrative stores (call notes, complaints, correspondence, ack
        # forms) are plain vector collections. Order doesn't matter.
        ingest_policies.main()
        ingest_fund_vectors.main()
        ingest_call_notes.main()
        ingest_complaints.main()
        ingest_correspondence.main()
        ingest_ack_forms.main()
        steps.append("Vector store (chroma_db)")

    return steps

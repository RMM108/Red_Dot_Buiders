"""Transparent persona construction from the supplied client corpus.

Persona construction is deliberately separate from the agentic RAG pipeline
in ``agent.py``. It summarizes observed client records and portfolio choices;
it does not make investment recommendations or replace policy evaluation.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PERSONA_FIELDS = (
    "age", "nationality", "residency_country", "occupation", "marital_status",
    "net_worth_band", "investor_status", "base_currency", "aum_sgd", "pep_status",
    "source_of_wealth", "investment_objective",
)


def _data_root(root: str | Path) -> Path:
    """Accept either the repository root or the corpus ``data`` directory."""

    root = Path(root)
    if (root / "clients_portfolio.json").exists():
        return root
    data_root = root / "data"
    if (data_root / "clients_portfolio.json").exists():
        return data_root
    # When called from inside data_processing/, the corpus lives in the
    # sibling data/ directory (root.parent / "data").
    sibling_data = root.parent / "data"
    if (sibling_data / "clients_portfolio.json").exists():
        return sibling_data
    raise FileNotFoundError(
        f"Could not find clients_portfolio.json under {root}, {data_root}, or {sibling_data}"
    )


# Repository root, resolved once at import. The persona tool wrappers use this
# as the default data location; callers may override it per call via `root=`.
ROOT = _data_root(Path(__file__).parent)


def _canonical_risk_profile(value: str) -> str:
    return value.split("(", 1)[0].strip()


def _client_transactions(root: Path, client_id: str) -> list[dict[str, str]]:
    with (root / "transactions.csv").open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("client_id") == client_id]


def _client_correspondence(root: Path, client_id: str) -> list[dict[str, Any]]:
    correspondence_path = root / "client_correspondence.json"
    if not correspondence_path.exists():
        return []
    data = json.loads(correspondence_path.read_text(encoding="utf-8"))
    return [
        thread for thread in data.get("email_threads", [])
        if thread.get("related_client_id") == client_id
    ]


def _current_profile_overlay(
    client: dict[str, Any],
    transactions: list[dict[str, str]],
    correspondence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive current, source-backed signals without changing the base record."""

    messages = [
        message for thread in correspondence for message in thread.get("messages", [])
    ]
    dated_events = [
        {
            "date": message.get("date"),
            "source": "client_correspondence.json",
            "locator": f"thread={thread.get('thread_id')}",
            "subject": thread.get("subject"),
            "text": message.get("body", ""),
        }
        for thread in correspondence
        for message in thread.get("messages", [])
    ]
    all_text = " ".join(
        [message.get("body", "") for message in messages]
        + [row.get("notes", "") or "" for row in transactions]
    ).casefold()
    de_risk_signal = any(
        phrase in all_text for phrase in ("de-risk", "de risk", "retiring", "retirement")
    )
    pending_transactions = [
        row for row in transactions if row.get("status", "").casefold() == "pending"
    ]
    realised_loss = any(
        "realised fx loss" in (row.get("notes", "") or "").casefold()
        for row in transactions
    )

    recorded_score = client["risk_score_1_to_10"]
    recorded_profile = _canonical_risk_profile(client["risk_profile"])
    risk_signals: list[str] = []
    if de_risk_signal:
        risk_signals.append(
            "Recent correspondence indicates retirement or an explicit request to de-risk."
        )
    if pending_transactions:
        risk_signals.append(
            "A risk-related or portfolio-related action remains pending and is not an executed holding."
        )
    if realised_loss:
        risk_signals.append("A settled transaction note records a realised FX loss.")
    if client.get("suitability_flag", "").casefold().startswith(
        ("potential mismatch", "review recommended", "suitability exception")
    ):
        risk_signals.append("The base record contains an active suitability review signal.")

    product_activity: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {"settled_count": 0, "pending_count": 0, "settled_amounts": defaultdict(float)}
    )
    for row in transactions:
        product = row.get("product_name", "")
        if not product:
            continue
        status_key = "pending_count" if row.get("status") == "Pending" else "settled_count"
        product_activity[product][status_key] += 1
        if row.get("status") == "Settled":
            try:
                product_activity[product]["settled_amounts"][row.get("currency", "")] += float(
                    row.get("amount", 0)
                )
            except (TypeError, ValueError):
                pass

    return {
        "recorded_risk_profile": recorded_profile,
        "recorded_risk_score": recorded_score,
        "current_risk_profile": recorded_profile,
        "current_risk_score": recorded_score,
        "risk_profile_basis": "clients_portfolio.json/clients_portfolio.csv recorded values; activity is a review signal only",
        "risk_signals": risk_signals,
        "current_objective": (
            "De-risk toward retirement and reduce growth/illiquid exposure."
            if de_risk_signal
            else client.get("investment_objective")
        ),
        "pending_actions": pending_transactions,
        "recent_transactions": sorted(
            transactions, key=lambda row: row.get("date", ""), reverse=True
        ),
        "correspondence_events": sorted(
            dated_events, key=lambda event: event.get("date", ""), reverse=True
        ),
        "product_activity": {
            product: {
                "settled_count": values["settled_count"],
                "pending_count": values["pending_count"],
                "settled_amounts": dict(values["settled_amounts"]),
            }
            for product, values in product_activity.items()
        },
        "product_fit_status": (
            "Observed category-level context only; fund factsheets and product suitability rules "
            "must be retrieved by the agentic RAG pipeline before any advisor conclusion."
        ),
    }


def _advisor_attention_items(
    client: dict[str, Any], overlay: dict[str, Any]
) -> list[dict[str, str]]:
    """Turn source-backed signals into RM review prompts, not advice."""

    items: list[dict[str, str]] = []
    suitability_flag = client.get("suitability_flag") or ""
    if suitability_flag and not suitability_flag.casefold().startswith("no "):
        items.append({
            "priority": "high",
            "type": "suitability_review",
            "message": suitability_flag,
            "source": "clients_portfolio.json",
        })
    if overlay["pending_actions"]:
        items.append({
            "priority": "high",
            "type": "pending_action",
            "message": "Confirm whether pending portfolio or remittance actions were executed before advising on current holdings.",
            "source": "transactions.csv",
        })
    if client.get("pep_status", "").casefold() not in {"", "not a pep"}:
        items.append({
            "priority": "high",
            "type": "kyc_monitoring",
            "message": "Review the recorded PEP-adjacent/monitoring status and latest KYC evidence before client servicing.",
            "source": "clients_portfolio.json",
        })
    if overlay["risk_signals"]:
        items.append({
            "priority": "medium",
            "type": "profile_change",
            "message": "Reconfirm the current risk profile and objective against the recorded correspondence and activity overlay.",
            "source": "client_correspondence.json; transactions.csv",
        })
    if not items:
        items.append({
            "priority": "low",
            "type": "routine_review",
            "message": "No active exception was detected in the supplied client record; continue ordinary suitability and KYC review.",
            "source": "clients_portfolio.json",
        })
    return items


def _advisor_view(client: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    return {
        "client_snapshot": {
            "client_id": client["client_id"],
            "name": client["name"],
            "recorded_risk_profile": overlay["recorded_risk_profile"],
            "recorded_risk_score": overlay["recorded_risk_score"],
            "current_risk_profile": overlay["current_risk_profile"],
            "current_risk_score": overlay["current_risk_score"],
            "investor_status": client.get("investor_status"),
            "investment_objective": client.get("investment_objective"),
            "aum_sgd": client.get("aum_sgd"),
        },
        "attention_items": _advisor_attention_items(client, overlay),
        "next_steps": [
            "Use the agentic RAG tools to retrieve the relevant policy, factsheet, and client documents before making a suitability determination.",
            "Treat pending transactions and correspondence signals as review prompts, not executed changes to the portfolio.",
            "Do not infer a product recommendation from comparable-client portfolio patterns.",
        ],
        "evidence_sources": [
            "clients_portfolio.json",
            "transactions.csv",
            "client_correspondence.json",
        ],
        "scope_note": "This is an advisor review brief, not investment, legal, or compliance advice.",
    }


def _text_similarity(left: Any, right: Any) -> float:
    tokens = lambda value: set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", str(value).lower()))
    left_terms, right_terms = tokens(left), tokens(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms.intersection(right_terms)) / len(left_terms.union(right_terms))


def _age_similarity(input_age: Any, client_age: Any) -> float:
    try:
        difference = abs(float(input_age) - float(client_age))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, 1.0 - difference / 25.0)


def _aum_similarity(input_aum: Any, client_aum: Any) -> float:
    try:
        input_value = float(input_aum)
        client_value = float(client_aum)
    except (TypeError, ValueError):
        return 0.0
    if input_value <= 0 or client_value <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(math.log(input_value / client_value)) / math.log(10))


def _persona_match_score(
    characteristics: dict[str, Any], client: dict[str, Any]
) -> tuple[float, list[str], int]:
    weights = {
        "age": 0.18, "nationality": 0.06, "residency_country": 0.08,
        "occupation": 0.08, "marital_status": 0.08, "net_worth_band": 0.16,
        "investor_status": 0.12, "base_currency": 0.04, "aum_sgd": 0.15,
        "pep_status": 0.05, "source_of_wealth": 0.16, "investment_objective": 0.27,
    }
    similarities: dict[str, float] = {}
    available_weight = 0.0
    exact_fields = {
        "marital_status", "nationality", "residency_country", "occupation",
        "net_worth_band", "investor_status", "base_currency", "pep_status",
    }
    for field in PERSONA_FIELDS:
        input_value, client_value = characteristics.get(field), client.get(field)
        if input_value in (None, "") or client_value in (None, ""):
            continue
        if field == "age":
            similarity = _age_similarity(input_value, client_value)
        elif field == "aum_sgd":
            similarity = _aum_similarity(input_value, client_value)
        elif field in exact_fields:
            similarity = 1.0 if str(input_value).casefold() == str(client_value).casefold() else 0.0
        else:
            similarity = _text_similarity(input_value, client_value)
        similarities[field] = similarity
        available_weight += weights[field]
    if not available_weight:
        return 0.0, [], 0
    score = sum(weights[field] * value for field, value in similarities.items()) / available_weight
    return score, [field for field, value in similarities.items() if value >= 0.5], len(similarities)


def generate_client_persona(
    root: str | Path, characteristics: dict[str, Any], top_k: int = 5
) -> dict[str, Any]:
    """Match supplied characteristics to observed clients and portfolio choices."""

    root = _data_root(root)
    data = json.loads((root / "clients_portfolio.json").read_text(encoding="utf-8"))
    normalized = dict(characteristics)
    scored_clients: list[tuple[float, dict[str, Any], list[str], int]] = []
    for client in data["clients"]:
        score, reasons, fields_used = _persona_match_score(normalized, client)
        if fields_used:
            scored_clients.append((score, client, reasons, fields_used))
    scored_clients.sort(key=lambda item: (-item[0], item[1]["client_id"]))
    matches = scored_clients[: max(1, top_k)]

    product_scores: defaultdict[str, float] = defaultdict(float)
    asset_class_scores: defaultdict[str, float] = defaultdict(float)
    investor_status_scores: defaultdict[str, float] = defaultdict(float)
    pep_status_scores: defaultdict[str, float] = defaultdict(float)
    suitability_flags: list[dict[str, Any]] = []
    current_profile_signals: list[dict[str, Any]] = []
    total_match_score = sum(score for score, *_ in matches)
    if total_match_score == 0:
        return {
            "input_characteristics": normalized, "current_profile_signals": [],
            "matched_client_risk_profiles": [], "investor_status_distribution": [],
            "pep_status_distribution": [], "suitability_flags": [],
            "favoured_portfolios": [], "favoured_asset_classes": [], "matched_clients": [],
            "observed_portfolio_patterns": [],
            "advisor_use": {
                "scope_note": "No comparable client records were found; do not infer a client recommendation."
            },
            "limitations": ["No comparable client records were found for the supplied characteristics."],
        }

    for match_score, client, _, _ in matches:
        overlay = _current_profile_overlay(
            client, _client_transactions(root, client["client_id"]),
            _client_correspondence(root, client["client_id"]),
        )
        current_profile_signals.append({
            "client_id": client["client_id"], "current_profile": overlay["current_risk_profile"],
            "risk_signals": overlay["risk_signals"], "pending_actions": overlay["pending_actions"],
        })
        investor_status_scores[client["investor_status"]] += match_score
        pep_status_scores[client["pep_status"]] += match_score
        suitability_flags.append({
            "client_id": client["client_id"], "flag": client["suitability_flag"],
            "notes": client.get("notes"),
        })
        for holding in client["portfolio_holdings"]:
            allocation = float(holding["allocation_pct"])
            product_scores[holding["product_name"]] += match_score * allocation
            asset_class_scores[holding["asset_class"]] += match_score * allocation

    def ranked_allocations(scores: dict[str, float]) -> list[dict[str, Any]]:
        return [
            {"name": name, "weighted_allocation": round(value / total_match_score, 2)}
            for name, value in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ]

    unavailable_fields = [
        field for field in PERSONA_FIELDS
        if normalized.get(field) not in (None, "")
        and not any(client.get(field) not in (None, "") for _, client, _, _ in matches)
    ]
    limitations = [
        "Portfolio preferences are inferred from similar observed clients, not stated recommendations.",
        "Risk appetite is not calculated here; matched clients' recorded risk fields are provided for advisor verification only.",
    ]
    if "aum_sgd" in normalized and "aum_sgd" in unavailable_fields:
        limitations.append("AUM in SGD was supplied but is absent from the client records, so it was excluded from matching.")

    return {
        "input_characteristics": normalized,
        "current_profile_signals": current_profile_signals,
        "matched_client_risk_profiles": [
            {
                "client_id": client["client_id"],
                "recorded_profile": _canonical_risk_profile(client["risk_profile"]),
                "recorded_score": client["risk_score_1_to_10"],
            }
            for _, client, _, _ in matches
        ],
        "investor_status_distribution": [
            {"status": status, "match_weight": round(weight / total_match_score, 3)}
            for status, weight in sorted(investor_status_scores.items(), key=lambda item: (-item[1], item[0]))
        ],
        "pep_status_distribution": [
            {"status": status, "match_weight": round(weight / total_match_score, 3)}
            for status, weight in sorted(pep_status_scores.items(), key=lambda item: (-item[1], item[0]))
        ],
        "suitability_flags": suitability_flags,
        "favoured_portfolios": ranked_allocations(product_scores)[:top_k],
        "favoured_asset_classes": ranked_allocations(asset_class_scores)[:top_k],
        "observed_portfolio_patterns": ranked_allocations(product_scores)[:top_k],
        "matched_clients": [
            {"client_id": client["client_id"], "name": client["name"],
             "match_score": round(score, 3), "matching_characteristics": reasons}
            for score, client, reasons, _ in matches
        ],
        "limitations": limitations,
        "advisor_use": {
            "purpose": "Use comparable clients to generate discussion context and questions for review, not product recommendations.",
            "review_before_action": [
                "Confirm the supplied characteristics and the client's current recorded profile.",
                "Retrieve current policy and product evidence before discussing suitability.",
                "Check for missing documentation, pending actions, and conflicting correspondence.",
            ],
        },
    }


def lookup_client_persona(root: str | Path, client_id: str) -> dict[str, Any]:
    """Return the stored persona and portfolio for one exact client ID."""

    root = _data_root(root)
    data = json.loads((root / "clients_portfolio.json").read_text(encoding="utf-8"))
    client = next((record for record in data["clients"] if record.get("client_id") == client_id), None)
    if client is None:
        raise ValueError(f"Client ID not found: {client_id}")
    asset_class_allocations: defaultdict[str, float] = defaultdict(float)
    for holding in client["portfolio_holdings"]:
        asset_class_allocations[holding["asset_class"]] += float(holding["allocation_pct"])
    overlay = _current_profile_overlay(
        client, _client_transactions(root, client_id), _client_correspondence(root, client_id)
    )
    return {
        "client_id": client["client_id"], "name": client["name"], "client_record": client,
        "characteristics": {field: client.get(field) for field in PERSONA_FIELDS if client.get(field) is not None},
        "expected_risk_profile": _canonical_risk_profile(client["risk_profile"]),
        "expected_risk_score": client["risk_score_1_to_10"], "current_profile": overlay,
        "advisor_view": _advisor_view(client, overlay),
        "favoured_portfolios": [
            {"name": holding["product_name"], "asset_class": holding["asset_class"],
             "allocation_pct": holding["allocation_pct"], "value_sgd": holding["value_sgd"],
             "currency": holding["currency"]}
            for holding in sorted(client["portfolio_holdings"], key=lambda holding: (-holding["allocation_pct"], holding["product_name"]))
        ],
        "favoured_asset_classes": [
            {"name": name, "allocation_pct": round(allocation, 2)}
            for name, allocation in sorted(asset_class_allocations.items(), key=lambda item: (-item[1], item[0]))
        ],
        "suitability_flag": client.get("suitability_flag"), "source": "clients_portfolio.json",
        "limitations": [
            "This persona reflects the client's recorded profile and holdings; it is not investment advice. "
            "Current risk and product-fit fields are provisional overlays from transactions and correspondence; "
            "fund factsheets and policy evaluation belong to the agentic RAG pipeline."
        ],
    }


def generate_personas(root: str | Path) -> list[dict[str, Any]]:
    """Return exact, non-inferred personas for every client record."""

    root = _data_root(root)
    data = json.loads((root / "clients_portfolio.json").read_text(encoding="utf-8"))
    return [lookup_client_persona(root, client["client_id"]) for client in data["clients"]]


# --------------------------------------------------------------------------
# Function-tool layer (mirrors the tool convention in agent.py)
#
# Each wrapper returns a JSON string so the result can be fed back into an
# LLM tool-calling loop. When an EvidencePool is supplied, every result is
# recorded as a citable evidence entry with a metadata-derived locator, so
# persona output is never re-derived or re-written by the model.
# --------------------------------------------------------------------------

def _tool_result(pool, ref_id: str, locator: str, payload: dict) -> str:
    """Record an evidence entry (if a pool is given) and return a JSON string."""
    if pool is not None:
        pool.add(ref_id, locator, json.dumps(payload))
    return json.dumps({"ref_id": ref_id, "content": json.dumps(payload)})


def tool_lookup_client_persona(pool, client_id: str, root: str | Path | None = None) -> str:
    """Tool wrapper: return the stored persona for one exact client ID."""
    try:
        result = lookup_client_persona(root or ROOT, client_id)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return _tool_result(pool, f"persona_{client_id}", f"clients_portfolio.json ({client_id})", result)


def tool_generate_client_persona(
    pool,
    age: int,
    marital_status: str,
    net_worth_band: str,
    aum_sgd: float,
    source_of_wealth: str,
    investment_objective: str,
    top_k: int = 3,
    root: str | Path | None = None,
) -> str:
    """Tool wrapper: match supplied characteristics to observed clients."""
    characteristics = {
        "age": age,
        "marital_status": marital_status,
        "net_worth_band": net_worth_band,
        "aum_sgd": aum_sgd,
        "source_of_wealth": source_of_wealth,
        "investment_objective": investment_objective,
    }
    result = generate_client_persona(root or ROOT, characteristics, top_k=top_k)
    return _tool_result(
        pool,
        "persona_generated",
        "clients_portfolio.json (comparable-client match)",
        result,
    )


def tool_generate_personas(pool, root: str | Path | None = None) -> str:
    """Tool wrapper: return exact personas for every client record."""
    result = generate_personas(root or ROOT)
    return _tool_result(pool, "personas_all", "clients_portfolio.json (all clients)", {"personas": result})


PERSONA_TOOLS = [
    {
        "type": "function",
        "name": "lookup_client_persona",
        "description": "Return the stored persona and portfolio for one exact client ID (e.g. 'CL002').",
        "parameters": {
            "type": "object",
            "properties": {"client_id": {"type": "string"}},
            "required": ["client_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "generate_client_persona",
        "description": "Match supplied client characteristics to observed clients and surface comparable portfolio patterns. "
                        "Does not make investment recommendations.",
        "parameters": {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
                "marital_status": {"type": "string"},
                "net_worth_band": {"type": "string"},
                "aum_sgd": {"type": "number"},
                "source_of_wealth": {"type": "string"},
                "investment_objective": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": [
                "age", "marital_status", "net_worth_band", "aum_sgd",
                "source_of_wealth", "investment_objective",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "generate_personas",
        "description": "Return exact, non-inferred personas for every client record.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]

PERSONA_DISPATCH = {
    "lookup_client_persona": tool_lookup_client_persona,
    "generate_client_persona": tool_generate_client_persona,
    "generate_personas": tool_generate_personas,
}


def run_persona_tool(name: str, args: dict, pool=None, root: str | Path | None = None) -> str:
    """Invoke a persona tool by name, mirroring agent.py's DISPATCH routing."""
    fn = PERSONA_DISPATCH[name]
    return fn(pool, **args, root=root)
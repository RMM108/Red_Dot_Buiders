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
    raise FileNotFoundError(
        f"Could not find clients_portfolio.json under {root} or {data_root}"
    )


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

    recorded_score = float(client["risk_score_1_to_10"])
    current_score = min(recorded_score, 3.0) if de_risk_signal else recorded_score
    current_profile = (
        "Conservative"
        if de_risk_signal and current_score <= 3
        else _canonical_risk_profile(client["risk_profile"])
    )
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

    viable_product_categories = (
        [
            "Cash and money-market funds",
            "Government and investment-grade fixed income",
            "Balanced income and growth funds",
        ]
        if current_score <= 3
        else ["Diversified growth funds", "Equity funds", "Fixed income for diversification"]
    )
    return {
        "recorded_risk_profile": _canonical_risk_profile(client["risk_profile"]),
        "recorded_risk_score": client["risk_score_1_to_10"],
        "current_risk_profile": current_profile,
        "current_risk_score": int(current_score)
        if current_score.is_integer()
        else round(current_score, 2),
        "risk_profile_basis": "client correspondence and transaction activity overlay",
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
        "viable_product_categories": viable_product_categories,
        "product_fit_status": (
            "Provisional category-level fit only; fund factsheets and product suitability rules "
            "are evaluated by the agentic RAG pipeline, not persona construction."
        ),
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
    risk_scores: defaultdict[str, float] = defaultdict(float)
    current_risk_scores: defaultdict[str, float] = defaultdict(float)
    investor_status_scores: defaultdict[str, float] = defaultdict(float)
    pep_status_scores: defaultdict[str, float] = defaultdict(float)
    suitability_flags: list[dict[str, Any]] = []
    current_profile_signals: list[dict[str, Any]] = []
    weighted_risk_total = weighted_current_risk_total = 0.0
    total_match_score = sum(score for score, *_ in matches)
    if total_match_score == 0:
        return {
            "input_characteristics": normalized, "expected_risk_profile": None,
            "expected_risk_score": None, "current_expected_risk_profile": None,
            "current_expected_risk_score": None, "current_profile_signals": [],
            "risk_profile_distribution": [], "investor_status_distribution": [],
            "pep_status_distribution": [], "suitability_flags": [],
            "favoured_portfolios": [], "favoured_asset_classes": [], "matched_clients": [],
            "limitations": ["No comparable client records were found for the supplied characteristics."],
        }

    for match_score, client, _, _ in matches:
        risk_profile = _canonical_risk_profile(client["risk_profile"])
        risk_scores[risk_profile] += match_score
        overlay = _current_profile_overlay(
            client, _client_transactions(root, client["client_id"]),
            _client_correspondence(root, client["client_id"]),
        )
        current_risk_scores[overlay["current_risk_profile"]] += match_score
        weighted_current_risk_total += match_score * overlay["current_risk_score"]
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
        weighted_risk_total += match_score * client["risk_score_1_to_10"]
        for holding in client["portfolio_holdings"]:
            allocation = float(holding["allocation_pct"])
            product_scores[holding["product_name"]] += match_score * allocation
            asset_class_scores[holding["asset_class"]] += match_score * allocation

    def ranked_allocations(scores: dict[str, float]) -> list[dict[str, Any]]:
        return [
            {"name": name, "weighted_allocation": round(value / total_match_score, 2)}
            for name, value in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ]

    dominant_risk = max(risk_scores, key=risk_scores.get)
    unavailable_fields = [
        field for field in PERSONA_FIELDS
        if normalized.get(field) not in (None, "")
        and not any(client.get(field) not in (None, "") for _, client, _, _ in matches)
    ]
    limitations = [
        "Portfolio preferences are inferred from similar observed clients, not stated recommendations.",
        "Risk is grounded in the matched clients' recorded risk profiles.",
    ]
    if "aum_sgd" in normalized and "aum_sgd" in unavailable_fields:
        limitations.append("AUM in SGD was supplied but is absent from the client records, so it was excluded from matching.")

    return {
        "input_characteristics": normalized,
        "expected_risk_profile": dominant_risk,
        "expected_risk_score": round(weighted_risk_total / total_match_score, 2),
        "current_expected_risk_profile": max(current_risk_scores, key=current_risk_scores.get),
        "current_expected_risk_score": round(weighted_current_risk_total / total_match_score, 2),
        "current_profile_signals": current_profile_signals,
        "risk_profile_distribution": [
            {"profile": profile, "match_weight": round(weight / total_match_score, 3)}
            for profile, weight in sorted(risk_scores.items(), key=lambda item: (-item[1], item[0]))
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
        "matched_clients": [
            {"client_id": client["client_id"], "name": client["name"],
             "match_score": round(score, 3), "matching_characteristics": reasons}
            for score, client, reasons, _ in matches
        ],
        "limitations": limitations,
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
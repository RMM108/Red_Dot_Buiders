"""Deterministic compliance rule-check node - independent of the LLM.

The README's roadmap called this out as the single biggest architectural
gap versus this project's own "right store for the shape of the data"
principle (PRD.md §2 point 4): today a suitability judgment (SRI vs. risk
score, 20% concentration, signed risk acknowledgement) is made by the
agent's own LLM reading SQL/vector facts, so a plausible-but-wrong
arithmetic or reasoning error wouldn't be caught by anything else in the
pipeline. This module computes the same 3 rules in plain Python/SQL
instead, so the agent can cite a verdict that was *computed*, not reasoned
about.

Encodes exactly 3 rules from policy_investment_suitability.pdf Section 4
("Sale of Complex Products - Mandatory Requirements"), each tagged with a
verdict of PASS / FAIL / AMBIGUOUS / N_A and a plain-language citation:

  R1_risk_score            Complex Products (SRI >= 6, per Section 3's
                            definition) may only be recommended to clients
                            with a Risk Score of 7 or above, UNLESS the
                            client is an Accredited/Expert/Institutional
                            Investor "as defined under the SFA".
  R2_concentration         No Complex Product may represent more than 20%
                            of a *Retail Investor's* liquid net worth.
  R3_risk_acknowledgement  A Complex Product sale to a Retail Investor
                            requires a signed risk acknowledgement on file
                            (checked via an exact metadata match against the
                            ack_forms collection - a presence/absence check,
                            not a similarity search, so it's deterministic).

Verdicts found by running this against the actual dataset (not assumed):
CL002 and CL013 both fail R1 and R3 (Retail Investors, Risk Score < 7,
holding a Complex Product with no signed acknowledgement on file) - CL002
is the dataset's known documented case; CL013 failing the same two rules is
a new finding this module surfaces that the narrative case study (framed
around his rebalancing delay) doesn't emphasize. CL001/CL009/CL011 each
hold a Complex Product above 20% as an Accredited Investor - R2 is
literal-text-scoped to "a Retail Investor's liquid net worth", so these are
AMBIGUOUS, not FAIL, mirroring PLAN.md's CL011 finding - and CL011 turns
out to have *two* holdings over 20%, not the one already documented.
CL004/CL006/CL014 are "Professional Investor (per SFO Cap. 571)" - a Hong
Kong classification, not literally one of the SFA-defined terms R1's
exemption names - flagged AMBIGUOUS on R1 whenever their Risk Score alone
doesn't already clear the bar.

Usage:
    from compliance_rules import check_client
    check_client("CL002")  # -> {"client_id", "name", "overall_verdict", "findings": [...]}
"""

from __future__ import annotations

from typing import Optional

from langsmith import traceable

import db
from ingest_clients import normalize_product_name
from vector_store import get_chroma_collection

COMPLEX_SRI_THRESHOLD = 6
CONCENTRATION_CAP_PCT = 20.0
MIN_RISK_SCORE_FOR_COMPLEX = 7

SFA_SOPHISTICATED_PREFIXES = ("accredited investor", "expert investor", "institutional investor")
RETAIL_PREFIX = "retail investor"

CITATION = "POL-INV-011 Section 4 (Sale of Complex Products - Mandatory Requirements)"


def _investor_class(investor_status: str) -> str:
    """"retail" / "sfa_sophisticated" / "ambiguous_classification" (e.g. a
    Hong Kong SFO Cap. 571 "Professional Investor" - not one of the SFA-
    defined terms POL-INV-011's exemption literally names)."""
    s = investor_status.strip().lower()
    if s.startswith(RETAIL_PREFIX):
        return "retail"
    if any(s.startswith(p) for p in SFA_SOPHISTICATED_PREFIXES):
        return "sfa_sophisticated"
    return "ambiguous_classification"


def _has_signed_ack(client_id: str) -> bool:
    """Exact metadata presence check against the ack_forms collection -
    deliberately NOT a similarity search, since presence/absence of a
    specific document is a fact to look up, not a fuzzy match."""
    collection = get_chroma_collection("ack_forms")
    result = collection.get(where={"$and": [{"client_id": client_id}, {"is_blank_template": False}]})
    return len(result["ids"]) > 0


def check_holding(client: dict, holding: dict, fund_sri: int) -> list[dict]:
    """Evaluate R1-R3 for one holding. Returns [] if the holding isn't a
    Complex Product (SRI < 6) - the rules don't apply to it at all."""
    if fund_sri < COMPLEX_SRI_THRESHOLD:
        return []

    investor_class = _investor_class(client["investor_status"])
    product_name = holding["product_name"]
    findings = []

    # R1: Risk Score >= 7, unless an SFA-defined sophisticated investor
    risk_score = client["risk_score_1_to_10"]
    if risk_score >= MIN_RISK_SCORE_FOR_COMPLEX:
        findings.append({
            "rule_id": "R1_risk_score", "verdict": "PASS", "citation": CITATION,
            "explanation": f"Risk Score {risk_score}/10 meets the >=7 threshold for holding {product_name} (SRI {fund_sri}/7).",
        })
    elif investor_class == "sfa_sophisticated":
        findings.append({
            "rule_id": "R1_risk_score", "verdict": "PASS", "citation": CITATION,
            "explanation": f"Risk Score {risk_score}/10 is below 7, but {client['investor_status']} is exempt from the Risk Score requirement under {CITATION}.",
        })
    elif investor_class == "ambiguous_classification":
        findings.append({
            "rule_id": "R1_risk_score", "verdict": "AMBIGUOUS", "citation": CITATION,
            "explanation": (f"Risk Score {risk_score}/10 is below 7. The exemption in {CITATION} applies to "
                             f"'Accredited Investor, Expert Investor, or Institutional Investor as defined under "
                             f"the SFA'; {client['investor_status']} is a different (Hong Kong SFO Cap. 571) "
                             f"classification not literally on that list - requires a human compliance judgment, "
                             f"not an automatic pass or fail."),
        })
    else:
        findings.append({
            "rule_id": "R1_risk_score", "verdict": "FAIL", "citation": CITATION,
            "explanation": (f"Risk Score {risk_score}/10 is below the required 7 for a Retail Investor holding "
                             f"{product_name} (SRI {fund_sri}/7), with no SFA-defined sophisticated-investor exemption."),
        })

    # R2: concentration cap - literal text scopes this to "a Retail Investor's liquid net worth"
    allocation = holding["allocation_pct"]
    if allocation <= CONCENTRATION_CAP_PCT:
        findings.append({
            "rule_id": "R2_concentration", "verdict": "PASS", "citation": CITATION,
            "explanation": f"{product_name} at {allocation}% does not exceed the 20% cap.",
        })
    elif investor_class == "retail":
        findings.append({
            "rule_id": "R2_concentration", "verdict": "FAIL", "citation": CITATION,
            "explanation": (f"{product_name} at {allocation}% exceeds the 20% cap on a Retail Investor's liquid "
                             f"net worth, with no documented senior-management exception on file."),
        })
    else:
        findings.append({
            "rule_id": "R2_concentration", "verdict": "AMBIGUOUS", "citation": CITATION,
            "explanation": (f"{product_name} at {allocation}% exceeds 20%, but the cap's literal wording covers "
                             f"only a 'Retail Investor's liquid net worth' and this client is "
                             f"{client['investor_status']} - not literally in scope, though no documented "
                             f"senior-management exception is on file either. Requires a human compliance "
                             f"judgment call, not an automatic pass or fail."),
        })

    # R3: signed risk acknowledgement, required before first sale to a Retail Investor
    if investor_class == "retail":
        has_ack = _has_signed_ack(client["client_id"])
        findings.append({
            "rule_id": "R3_risk_acknowledgement", "verdict": "PASS" if has_ack else "FAIL", "citation": CITATION,
            "explanation": (f"A signed risk acknowledgement is on file for {client['client_id']}." if has_ack else
                             f"{CITATION} requires a Customer Account Review / Customer Knowledge Assessment and "
                             f"signed risk acknowledgement before selling a Complex Product to a Retail Investor; "
                             f"no such record was found on file for {client['client_id']}."),
        })
    else:
        findings.append({
            "rule_id": "R3_risk_acknowledgement", "verdict": "N_A", "citation": CITATION,
            "explanation": f"This rule is scoped to sales to Retail Investors; {client['investor_status']} is outside its literal scope.",
        })

    for f in findings:
        f["product_name"] = product_name
    return findings


@traceable(name="check_client", run_type="chain")
def check_client(client_id: str) -> dict:
    """Run R1-R3 across every Complex Product holding for one client.
    overall_verdict is FAIL if any finding failed, else AMBIGUOUS if any
    finding was ambiguous, else PASS (including the case of no Complex
    Product holdings at all, trivially)."""
    conn = db.get_connection()
    client_row = conn.execute("SELECT * FROM clients WHERE client_id = ?", (client_id,)).fetchone()
    if not client_row:
        conn.close()
        return {"client_id": client_id, "error": f"no client found matching {client_id!r}"}
    client = dict(client_row)

    holdings = [dict(r) for r in conn.execute("SELECT * FROM holdings WHERE client_id = ?", (client_id,))]
    fund_sri_by_norm = {
        normalize_product_name(r["fund_name"]): r["summary_risk_indicator"]
        for r in conn.execute("SELECT fund_name, summary_risk_indicator FROM funds")
    }
    conn.close()

    all_findings: list[dict] = []
    for h in holdings:
        sri = fund_sri_by_norm.get(normalize_product_name(h["product_name"]))
        if sri is None:
            continue  # no fact sheet on file - product_crosswalk's job to flag, not this node's
        all_findings.extend(check_holding(client, h, sri))

    verdicts = {f["verdict"] for f in all_findings}
    overall = "FAIL" if "FAIL" in verdicts else "AMBIGUOUS" if "AMBIGUOUS" in verdicts else "PASS"

    return {
        "client_id": client_id,
        "name": client["name"],
        "investor_status": client["investor_status"],
        "risk_score_1_to_10": client["risk_score_1_to_10"],
        "overall_verdict": overall,
        "findings": all_findings,
    }


def has_compliance_result(pool, client_id: str) -> bool:
    """True if a compliance_rules_{client_id} entry is already in the pool -
    used by agent_graph.py's compliance_gate to tell "the model already
    called check_compliance_rules itself" apart from "this node needs to
    inject a fresh result and give the model one more turn to see it"."""
    return pool.resolve(f"compliance_rules_{client_id}") is not None


def check_all_clients() -> list[dict]:
    """Batch sweep, e.g. for a standalone compliance report."""
    conn = db.get_connection()
    client_ids = [r["client_id"] for r in conn.execute("SELECT client_id FROM clients ORDER BY client_id")]
    conn.close()
    return [check_client(cid) for cid in client_ids]


def format_findings(result: dict) -> str:
    """Plain-text rendering for an agent tool result / evidence pool entry."""
    if result.get("error"):
        return result["error"]
    if not result["findings"]:
        return f"{result['client_id']} ({result['name']}): no Complex Product holdings - rules R1-R3 do not apply."

    lines = [f"{result['client_id']} ({result['name']}, {result['investor_status']}, "
              f"Risk Score {result['risk_score_1_to_10']}/10) - overall: {result['overall_verdict']}"]
    for f in result["findings"]:
        lines.append(f"  [{f['verdict']}] {f['rule_id']} on {f['product_name']}: {f['explanation']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for cid in sys.argv[1:]:
            print(format_findings(check_client(cid.upper())))
            print()
    else:
        for result in check_all_clients():
            print(format_findings(result))
            print()

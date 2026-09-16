"""LangGraph rewrite of agent.py's single-loop orchestrator into an explicit
multi-node graph (PRD.md §13's "full agentic graph" future consideration):

    START -> rewrite -> agent <-> tools
                          |
                          v (no more tool calls; draft answer produced)
                     compliance_gate
                          |
                          v
                        critic --(invalid ref_id found, retries left)--> agent
                          |
                          v (all citations valid, or retries exhausted)
                         END

- **rewrite**: resolves conversational references (reuses agent.rewrite_query)
  so every downstream node works from a standalone question.
- **agent**: the planner - calls the model with the same TOOLS/DISPATCH as
  agent.py; if it requests tool calls, routes to **tools**; once it produces
  a draft answer with no further tool calls, routes to **compliance_gate**.
- **tools**: executes the requested tool call(s) against the shared
  EvidencePool, then loops back to **agent** - this is where the "parallel
  tool execution" from PRD.md §13 would fan out if more than one call target
  ever needed true concurrency; today's tool set is fast enough sequentially
  and parallelizing would just add complexity for no measured benefit.
- **compliance_gate**: the deterministic node PRD.md §13 calls out as the
  biggest architectural gap versus agent.py's LLM-only reasoning. Scans the
  EvidencePool for every client_id already looked up (via `client_{ID}`
  ref_ids) and, for any not yet compliance-checked in this run, calls
  compliance_rules.check_client directly and injects the result - this
  happens whether or not the model remembered to call the
  check_compliance_rules tool itself, which agent.py's single-loop design
  cannot guarantee (it relies on the system prompt instructing the model to
  call it, not on a graph edge that always runs the check).
- **critic**: a faithfulness check (PRD.md §13) - verifies every ref_id the
  draft answer cites actually resolves in the EvidencePool. An invalid
  ref_id sends the draft back to **agent** with an explicit correction
  instruction (bounded to MAX_CRITIC_RETRIES) rather than silently shipping
  a citation to evidence the model never actually received.

Exposes run_agent_graph(question, ...) with the SAME return shape as
agent.run_agent, so callers (run_evaluation.py, app.py) can switch between
the single-loop and graph orchestrators without changing anything else.
Additive, not a replacement: agent.run_agent is untouched and remains the
default the rest of the codebase uses.

Usage:
    python agent_graph.py "Does Park Ji-hoon's (CL011) 35% allocation breach firm policy?"
"""

from __future__ import annotations

import json
import sys
from typing import Annotated, Any, Optional, TypedDict

from openai import OpenAI

from agent import (
    DISPATCH,
    MAX_TOOL_TURNS,
    MODEL,
    SUPPORTS_TEMPERATURE,
    SYSTEM_PROMPT,
    AgentAnswer,
    EvidencePool,
    rewrite_query,
)
from compliance_rules import check_client, format_findings, has_compliance_result
from langgraph.graph import END, START, StateGraph

MAX_CRITIC_RETRIES = 2


def _merge_evidence_pool(left: EvidencePool, right: EvidencePool) -> EvidencePool:
    """Reducer: both branches share one pool object in practice (no
    parallel fan-out yet), so this just keeps whichever is passed most
    recently - present so the state schema is explicit about the merge
    policy rather than relying on TypedDict's default overwrite silently
    being "correct by accident"."""
    return right if right is not None else left


class GraphState(TypedDict):
    input_list: list
    pool: Annotated[EvidencePool, _merge_evidence_pool]
    tool_call_log: list[dict]
    checked_clients: set[str]
    draft: Optional[AgentAnswer]
    critic_retries: int
    original_question: str
    rewritten_question: str
    n_evidence_before: int
    # Every key a node returns must be declared here - LangGraph's TypedDict-based
    # StateGraph only tracks channels for annotated keys and silently drops any
    # others, which caused a real bug during testing: node_agent's "pending_calls"
    # was invisible to route_after_agent, so it always routed past "tools".
    history: list[dict]
    pending_calls: list
    needs_rerun: bool


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def node_rewrite(state: GraphState) -> dict:
    question = state["original_question"]
    history = state.get("history") or []
    rewritten = rewrite_query(question, history)
    input_list = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-20:]:
        input_list.append({"role": h["role"], "content": h["content"]})
    input_list.append({"role": "user", "content": rewritten})
    return {"input_list": input_list, "rewritten_question": rewritten}


def node_agent(state: GraphState) -> dict:
    client = OpenAI()
    kwargs = {"temperature": 0} if SUPPORTS_TEMPERATURE else {}
    from agent import TOOLS  # local import: avoids a partial-init cycle with agent.py at module load time

    resp = client.responses.parse(
        model=MODEL, input=state["input_list"], tools=TOOLS, text_format=AgentAnswer, **kwargs
    )
    new_input_list = state["input_list"] + resp.output

    function_calls = [o for o in resp.output if o.type == "function_call"]
    if function_calls:
        return {"input_list": new_input_list, "draft": None, "pending_calls": function_calls}
    return {"input_list": new_input_list, "draft": resp.output_parsed, "pending_calls": []}


def route_after_agent(state: GraphState) -> str:
    return "tools" if state.get("pending_calls") else "compliance_gate"


def node_tools(state: GraphState) -> dict:
    pool = state["pool"]
    tool_call_log = list(state["tool_call_log"])
    input_list = list(state["input_list"])

    for call in state["pending_calls"]:
        args = json.loads(call.arguments)
        args = {k: v for k, v in args.items() if v is not None}
        fn = DISPATCH[call.name]
        result = fn(pool, **args)
        tool_call_log.append({"tool": call.name, "args": args})
        input_list.append({"type": "function_call_output", "call_id": call.call_id, "output": result})

    return {"input_list": input_list, "tool_call_log": tool_call_log, "pending_calls": []}


def node_compliance_gate(state: GraphState) -> dict:
    """Deterministic node, not an LLM tool call: for every client_id already
    in the evidence pool (a `client_{ID}` ref_id) that hasn't been
    compliance-checked yet this run, run compliance_rules.check_client
    directly and inject the result - guaranteed to happen regardless of
    whether the model remembered to call check_compliance_rules itself."""
    pool = state["pool"]
    checked = set(state["checked_clients"])
    client_ids = {ref_id.removeprefix("client_") for ref_id in pool.entries if ref_id.startswith("client_")}

    newly_checked = []
    for client_id in sorted(client_ids - checked):
        already_in_pool = has_compliance_result(pool, client_id)
        if not already_in_pool:
            result = check_client(client_id)
            if result.get("error"):
                checked.add(client_id)
                continue
            pool.add(f"compliance_rules_{client_id}", "compliance_rules.py (POL-INV-011 S4, computed)",
                      format_findings(result))
            newly_checked.append(client_id)  # only inject/rerun for results the model hasn't already seen
        checked.add(client_id)

    input_list = list(state["input_list"])
    if newly_checked:
        injected = "\n\n".join(
            f"[compliance_rules_{cid}] {pool.resolve(f'compliance_rules_{cid}')['text']}" for cid in newly_checked
        )
        input_list.append({
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": (
                "Deterministic compliance rule-check results (computed independently of you - "
                "treat as authoritative, do not re-derive or contradict):\n\n" + injected
            )}],
        })
    return {"checked_clients": checked, "input_list": input_list, "needs_rerun": bool(newly_checked)}


def route_after_compliance_gate(state: GraphState) -> str:
    # If the gate injected a compliance result the model hadn't already used,
    # give it one more turn to incorporate it before critiquing the draft.
    return "agent" if state.get("needs_rerun") else "critic"


def node_critic(state: GraphState) -> dict:
    """Faithfulness check: every ref_id the draft cites must resolve in the
    EvidencePool. If not, send it back to the agent with a correction
    instruction instead of shipping a citation to evidence it never received."""
    pool = state["pool"]
    draft: AgentAnswer = state["draft"]
    invalid = [c.ref_id for c in draft.citations if pool.resolve(c.ref_id) is None]

    if not invalid or state["critic_retries"] >= MAX_CRITIC_RETRIES:
        return {}

    input_list = list(state["input_list"])
    input_list.append({
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": (
            f"Faithfulness check failed: your citations {invalid} do not match any ref_id you actually "
            "received from a tool. Re-issue your final answer citing only ref_ids shown in prior tool "
            "results."
        )}],
    })
    return {"input_list": input_list, "draft": None, "critic_retries": state["critic_retries"] + 1}


def route_after_critic(state: GraphState) -> str:
    return "agent" if state["draft"] is None else END


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("rewrite", node_rewrite)
    graph.add_node("agent", node_agent)
    graph.add_node("tools", node_tools)
    graph.add_node("compliance_gate", node_compliance_gate)
    graph.add_node("critic", node_critic)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "compliance_gate": "compliance_gate"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("compliance_gate", route_after_compliance_gate, {"agent": "agent", "critic": "critic"})
    graph.add_conditional_edges("critic", route_after_critic, {"agent": "agent", END: END})
    return graph.compile()


_COMPILED_GRAPH = None


def _get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


# --------------------------------------------------------------------------
# Public entry point - same return shape as agent.run_agent
# --------------------------------------------------------------------------

def run_agent_graph(question: str, history: Optional[list[dict]] = None, pool: Optional[EvidencePool] = None,
                     verbose: bool = False) -> dict:
    pool = pool if pool is not None else EvidencePool()
    n_evidence_before = len(pool.entries)

    initial_state: dict[str, Any] = {
        "input_list": [], "pool": pool, "tool_call_log": [], "checked_clients": set(),
        "draft": None, "critic_retries": 0, "original_question": question,
        "rewritten_question": question, "n_evidence_before": n_evidence_before,
        "history": history or [], "pending_calls": [], "needs_rerun": False,
    }

    final_state = _get_graph().invoke(initial_state, {"recursion_limit": MAX_TOOL_TURNS * 4})

    if verbose:
        if final_state["rewritten_question"] != question:
            print(f"  rewritten query: {final_state['rewritten_question']!r}")
        for t in final_state["tool_call_log"]:
            print(f"  tool call: {t['tool']}({t['args']})")
        if final_state["checked_clients"]:
            print(f"  compliance_gate checked: {sorted(final_state['checked_clients'])}")

    draft: AgentAnswer = final_state["draft"]
    resolved_citations = []
    for c in draft.citations:
        entry = pool.resolve(c.ref_id)
        if entry:
            resolved_citations.append({"ref_id": c.ref_id, "locator": entry["locator"], "text": entry["text"]})
        else:
            resolved_citations.append({"ref_id": c.ref_id, "locator": "INVALID ref_id (model cited a source it never received)", "text": None})

    return {
        "answer": draft.answer,
        "citations": resolved_citations,
        "abstained": draft.abstained,
        "abstention_reason": draft.abstention_reason,
        "tool_calls": final_state["tool_call_log"],
        "n_evidence_retrieved": len(pool.entries) - n_evidence_before,
        "n_evidence_total": len(pool.entries),
        "original_question": question,
        "rewritten_question": final_state["rewritten_question"],
    }


def print_result(result: dict) -> None:
    if result.get("rewritten_question") and result["rewritten_question"] != result.get("original_question"):
        print(f"\nInterpreted as: {result['rewritten_question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nAbstained: {result['abstained']}" + (f" ({result['abstention_reason']})" if result["abstention_reason"] else ""))
    print(f"\nCitations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  [{c['ref_id']}] {c['locator']}")
    print(f"\nTool calls: {[t['tool'] for t in result['tool_calls']]}")
    print(f"Evidence retrieved: {result['n_evidence_retrieved']}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the minimum investment for the APAC Stable Income Money Market Fund?"
    print(f"Question: {q}")
    result = run_agent_graph(q, verbose=True)
    print_result(result)

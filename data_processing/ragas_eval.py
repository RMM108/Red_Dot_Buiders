"""RAGAS-based scoring, swapped in for run_evaluation.py's hand-rolled
context_precision / context_recall / faithfulness ratio math (PRD.md §13:
"swap the custom LLM-judge for ragas, already listed in requirements.txt").

Why this needed a shim: ragas 0.4.3 (and every 0.1-0.4 version tried)
unconditionally imports `langchain_community.chat_models.vertexai.ChatVertexAI`
at package import time, which no longer exists in the langchain-community
version this project's modern langchain/langgraph/langchain-openai stack
requires (a real upstream incompatibility between ragas's still-old
LangChain integration path and the rest of this project's dependencies -
downgrading langchain-community to satisfy ragas breaks langgraph/
langchain-openai instead, verified by trying it). Since that VertexAI class
is never actually used (this project is OpenAI-only), _install_stub()
inserts a placeholder module into sys.modules for that one dead import path
before ragas is imported anywhere, rather than fighting the dependency
resolver further.

Metrics used: Faithfulness, ContextPrecisionWithReference, ContextRecall -
the 3 of ragas's standard 4 that this project already computes a version of
(context_precision, context_recall, faithfulness in run_evaluation.py's
console summary). AnswerRelevancy is not used here since it needs a
separate embeddings wrapper and doesn't map to a metric this project
already reports; ragas.metrics.collections has it available if that's
wanted later.

Usage:
    from ragas_eval import score_triad
    scores = score_triad(
        question="...", response="...", retrieved_contexts=["...", "..."], reference="...",
    )
    # -> {"context_precision": 0.8, "context_recall": 1.0, "faithfulness": 0.9}
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


def _install_vertexai_stub() -> None:
    """Insert a placeholder for the one ragas import path this project's
    dependency stack can't satisfy and never actually needs (OpenAI-only).
    Idempotent - safe to call more than once."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)

    class ChatVertexAI:  # never instantiated; only needs to exist for the import to resolve
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = stub


_install_vertexai_stub()

from openai import AsyncOpenAI  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall, Faithfulness  # noqa: E402

JUDGE_MODEL = "gpt-4o-mini"

_llm = None


def _get_llm():
    """max_tokens=4096 (ragas's own default is much lower) - found by
    testing: the default silently truncated Faithfulness's internal
    statement/verdict generation for any question with several long cited
    contexts (raising instructor.IncompleteOutputException), which is
    exactly the multi-hop, multi-source questions this project cares most
    about (e.g. the CL002/CL013 cases citing 4-7 sources)."""
    global _llm
    if _llm is None:
        _llm = llm_factory(JUDGE_MODEL, client=AsyncOpenAI(), max_tokens=4096)
    return _llm


def score_triad(question: str, response: str, retrieved_contexts: list[str], reference: str) -> dict:
    """Faithfulness, Context Precision, and Context Recall via ragas, for
    one golden question. retrieved_contexts should be the text of the
    citations the agent actually used (agent.py's resolved citation
    "text" fields) - the same evidence run_evaluation.py already has on
    hand from result["citations"]. reference is the golden expected
    answer. Returns {} (no score) for any metric ragas can't compute
    (e.g. faithfulness needs at least one context) rather than raising."""
    if not retrieved_contexts:
        return {}

    llm = _get_llm()
    scores: dict[str, float] = {}

    metrics = {
        "faithfulness": lambda: Faithfulness(llm=llm).score(
            user_input=question, response=response, retrieved_contexts=retrieved_contexts,
        ),
        "context_precision": lambda: ContextPrecisionWithReference(llm=llm).score(
            user_input=question, reference=reference, retrieved_contexts=retrieved_contexts,
        ),
        "context_recall": lambda: ContextRecall(llm=llm).score(
            user_input=question, retrieved_contexts=retrieved_contexts, reference=reference,
        ),
    }
    for name, call in metrics.items():
        try:
            scores[name] = call().value
        except Exception as e:
            # Visible, not silently dropped - a missing score in the summary
            # should be traceable to a specific cause, not just "n/a".
            print(f"  [ragas_eval] {name} failed: {type(e).__name__}: {e}")

    return scores


if __name__ == "__main__":
    result = score_triad(
        question="What is the minimum investment for the APAC Stable Income Money Market Fund?",
        response="The minimum initial investment is SGD 10,000 for retail investors, with SGD 1,000 for subsequent investments.",
        retrieved_contexts=["Minimum Initial Investment: SGD 10,000 (retail); SGD 1,000 subsequent."],
        reference="SGD 10,000 for a retail investor's initial investment, with SGD 1,000 for subsequent investments.",
    )
    print(result)

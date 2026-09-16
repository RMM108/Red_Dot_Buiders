"""LangSmith tracing helper, shared by agent.py and agent_graph.py.

Tracing is opt-in via environment variables (LANGSMITH_TRACING, LANGSMITH_API_KEY,
LANGSMITH_PROJECT - see .env.example) and off by default: wrap_openai() and the
@traceable decorator are both safe to apply unconditionally - with LANGSMITH_TRACING
unset/false, they're documented no-ops (no network calls, negligible overhead), so
this module requires no LangSmith account to run the app normally. Set those three
variables to see full run traces (every tool call, every OpenAI request/response,
every compliance verdict) in the LangSmith dashboard.

Usage:
    from tracing import traced_openai_client
    client = traced_openai_client()
"""

from __future__ import annotations

import warnings

from openai import OpenAI
from langsmith.wrappers import wrap_openai

# wrap_openai() serializes the raw OpenAI SDK response (a large union-typed
# Pydantic model covering every Responses API output variant) to build the
# trace payload, regardless of whether tracing is actually enabled. Most
# union members don't apply to any single response, so Pydantic's fallback
# serializer logs one "unexpected value" line per non-matching member -
# harmless (the real fields still serialize correctly) but drowns real
# output in noise. Filtered globally here rather than at each call site.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:.*", category=UserWarning)


def traced_openai_client() -> OpenAI:
    return wrap_openai(OpenAI())

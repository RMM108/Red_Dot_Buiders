"""Streamlit chat front-end over agent.run_agent() (PLAN.md §5 step 9).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# agent.py and its sibling ingestion modules live in data_processing/ as flat
# (non-package) modules with bare imports (e.g. `import db`), so they're loaded
# by putting that directory on sys.path rather than importing as a package.
sys.path.insert(0, str(Path(__file__).parent / "data_processing"))

st.set_page_config(page_title="Meridian Peak | Wealth Copilot", page_icon="\U0001F3DB️", layout="centered")

DB_PATH = Path("data/processed/wealth_management.db")
CHROMA_PATH = Path("chroma_db")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---- Brand header ---- */
.brand-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #9A7B3F;
    margin-bottom: 0.15rem;
}
.brand-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: #0F2A4A;
    line-height: 1.15;
    margin-bottom: 0.35rem;
}
.brand-rule {
    height: 2px;
    background: linear-gradient(90deg, #0F2A4A 0%, #9A7B3F 55%, rgba(154,123,63,0) 100%);
    border: none;
    margin: 0.6rem 0 1.1rem 0;
}
.brand-caption {
    color: #4B5566;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    border-right: 1px solid #E4E8EF;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5A6478;
}

/* ---- Chat area ---- */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    border: 1px solid #E9ECF2;
    padding: 0.25rem 0.5rem;
    margin-bottom: 0.5rem;
}

/* ---- Expanders (citations / tool calls) ---- */
[data-testid="stExpander"] {
    border: 1px solid #E4E8EF;
    border-radius: 8px;
}
summary {
    font-weight: 500;
    color: #0F2A4A;
}

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 6px;
    font-weight: 500;
}

footer, #MainMenu, [data-testid="stToolbar"] {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def system_ready() -> tuple[bool, list[str]]:
    problems = []
    if not os.getenv("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set (add it to a .env file).")
    if not DB_PATH.exists():
        problems.append(f"SQL store not found at {DB_PATH} - run `python ingest_clients.py` first.")
    if not CHROMA_PATH.exists():
        problems.append(f"Vector store not found at {CHROMA_PATH} - run the ingest_*.py pipelines first.")
    return (len(problems) == 0, problems)


with st.sidebar:
    st.markdown("### System status")
    ready, problems = system_ready()
    if ready:
        st.success("Agent stores are ready.")
    else:
        st.warning("Setup incomplete - the agent will error until these are resolved:")
        for p in problems:
            st.markdown(f"- {p}")

    st.divider()
    st.markdown("### Display options")
    show_tool_calls = st.checkbox("Show tool calls", value=True)
    show_citations = st.checkbox("Show citations", value=True)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.evidence_pool = None
        st.rerun()

st.markdown('<div class="brand-eyebrow">Meridian Peak Wealth Partners</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-title">Wealth Management AI Copilot</div>', unsafe_allow_html=True)
st.markdown('<hr class="brand-rule" />', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-caption">Ask a client, product, or policy question. Answers are grounded in '
    "retrieved evidence, with citations and explicit abstention when evidence is insufficient.</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "evidence_pool" not in st.session_state:
    st.session_state.evidence_pool = None

ASSISTANT_AVATAR = "\U0001F3DB️"

for message in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            result = message.get("result")
            if result:
                rewritten = result.get("rewritten_question")
                if rewritten and rewritten != result.get("original_question"):
                    st.caption(f"Interpreted as: {rewritten}")
                if result["abstained"]:
                    st.warning(f"Abstained: {result['abstention_reason'] or 'insufficient evidence'}")
                if show_citations and result["citations"]:
                    with st.expander(f"Citations ({len(result['citations'])})"):
                        for c in result["citations"]:
                            st.markdown(f"**[{c['ref_id']}]** {c.get('apa') or c['locator']}")
                if show_tool_calls and result["tool_calls"]:
                    with st.expander(f"Tool calls ({len(result['tool_calls'])})"):
                        for t in result["tool_calls"]:
                            st.code(f"{t['tool']}({t['args']})", language="python")

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        ready, problems = system_ready()
        if not ready:
            error_text = "Can't answer yet - " + " ".join(problems)
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text, "result": None})
        else:
            # Conversation so far, excluding the question just asked (agent.rewrite_query
            # uses this to resolve pronouns/references like "him" or "that fund").
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Retrieving evidence..."):
                try:
                    from agent import EvidencePool, run_agent
                    if st.session_state.evidence_pool is None:
                        st.session_state.evidence_pool = EvidencePool()
                    result = run_agent(question, history=history, pool=st.session_state.evidence_pool)
                except Exception as e:
                    result = {
                        "answer": f"Error while running the agent: {e}",
                        "citations": [], "abstained": True, "abstention_reason": str(e),
                        "tool_calls": [], "n_evidence_retrieved": 0,
                        "original_question": question, "rewritten_question": question,
                    }

            rewritten = result.get("rewritten_question")
            if rewritten and rewritten != result.get("original_question"):
                st.caption(f"Interpreted as: {rewritten}")
            st.markdown(result["answer"])
            if result["abstained"]:
                st.warning(f"Abstained: {result['abstention_reason'] or 'insufficient evidence'}")
            if show_citations and result["citations"]:
                with st.expander(f"Citations ({len(result['citations'])})"):
                    for c in result["citations"]:
                        st.markdown(f"**[{c['ref_id']}]** {c.get('apa') or c['locator']}")
            if show_tool_calls and result["tool_calls"]:
                with st.expander(f"Tool calls ({len(result['tool_calls'])})"):
                    for t in result["tool_calls"]:
                        st.code(f"{t['tool']}({t['args']})", language="python")

            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})

"""
app.py
------
Streamlit UI for the Bank Scheme Comparator.

Startup logic (runs once per browser session via pipeline_done guard):
  1. Check if Qdrant collection already exists and data is fresh
  2. If yes  → skip scraping and ingestion entirely
  3. If no   → scrape (with TTL cache) then rebuild Qdrant index
  4. Load agent (cached per provider+model via @st.cache_resource)

Temperature / top_p sliders affect the main agent LLM only.
The summarization tool (Tool 3) uses its own fixed values (0.3 / 0.9).
"""

import os
import sys
import streamlit as st

# ── Make src/ importable ──────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Bank Scheme Comparator",
    page_icon="🏦",
    layout="wide",
)

QDRANT_DIR = os.path.join(ROOT, "data", "qdrant_db")


# ── Helpers ───────────────────────────────────────────────────────────────────

def collection_exists() -> bool:
    """Check if Qdrant collection already has data."""
    from src.ingestion.ingest import collection_exists as _check
    return _check()


def run_pipeline():
    """
    Scrape (TTL-aware) and rebuild Qdrant index.
    Called once per browser session, guarded by pipeline_done.
    """
    from src.scraper.scraper import scrape_all, is_data_fresh
    from src.scraper.sources import SOURCES

    all_fresh = all(is_data_fresh(s) for s in SOURCES)
    db_exists = collection_exists()

    # Best case: nothing to do
    if all_fresh and db_exists:
        st.toast("Data is fresh and index is ready!", icon="⚡")
        return

    # Data fresh but index missing — just re-ingest from cached JSONs
    if all_fresh and not db_exists:
        with st.status("Rebuilding index from cached data...", expanded=True) as status:
            st.write("All scraped data is fresh — skipping scraping")
            st.write("Rebuilding Qdrant index...")
            from src.ingestion.ingest import load_raw_records, chunk_records, build_vectorstore
            records   = load_raw_records()
            documents = chunk_records(records)
            build_vectorstore(documents)
            st.write(f"Ready — {len(documents)} chunks indexed")
            status.update(label="Index ready!", state="complete", expanded=False)
        return

    # Stale data — scrape then re-index
    with st.status("Fetching latest data from banks...", expanded=True) as status:
        st.write("Checking which sources need updating...")
        results = scrape_all()

        skipped = len(results.get("skipped", []))
        scraped = len(results.get("scraped", []))
        failed  = len(results.get("failed",  []))

        if skipped:
            st.write(f"{skipped} source(s) were fresh — skipped")
        if scraped:
            st.write(f"{scraped} source(s) scraped successfully")
        if failed:
            st.warning(f"{failed} source(s) failed — using any cached data")

        st.write("Rebuilding Qdrant index...")
        from src.ingestion.ingest import load_raw_records, chunk_records, build_vectorstore
        records   = load_raw_records()
        documents = chunk_records(records)
        build_vectorstore(documents)
        st.write(f"Ready — {len(documents)} chunks indexed")
        status.update(label="Ready! Ask your questions below.", state="complete", expanded=False)


# ── Run pipeline once per browser session ─────────────────────────────────────
if "pipeline_done" not in st.session_state:
    run_pipeline()
    st.session_state.pipeline_done = True

# Pre-warm the embedding and reranker models so first query isn't slow
if "models_warmed" not in st.session_state:
    with st.spinner("Loading AI models..."):
        from src.agent.tools import _get_embeddings, _get_reranker
        _get_embeddings()
        _get_reranker()
    st.session_state.models_warmed = True


# ── Agent imports — AFTER pipeline so Qdrant is guaranteed built ──────────────
from src.agent.llm_factory import ALL_MODELS, get_llm
from src.agent.graph import build_agent, run_agent


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Bank Scheme Comparator")
    st.caption("Education loans — HDFC · ICICI · SBI · BOB · UnionBank · Axis · Kotak · CUB")
    st.divider()

    st.subheader("LLM Settings")
    provider_display = st.selectbox(
        "Provider",
        options=list(ALL_MODELS.keys()),
        index=0,
    )

    provider_key_map = {
        "Gemini (Google)": "gemini",
        "Groq (Free)":     "groq",
        "Ollama (Local)":  "ollama",
        "Anthropic":       "anthropic",
    }
    provider_key = provider_key_map[provider_display]

    model_name = st.selectbox(
        "Model",
        options=ALL_MODELS[provider_display],
        index=0,
    )

    st.divider()

    # ── Temperature / top_p sliders ───────────────────────────────────────────
    st.subheader("Sampling Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher = more creative. Set to 0 for factual Q&A.",
    )

    st.divider()

    # ── Manual data refresh button ────────────────────────────────────────────
    if st.button("Refresh Data", use_container_width=True):
        del st.session_state.pipeline_done
        st.rerun()

    st.caption("Data auto-refreshes once per session (24hr TTL cache).")
    st.divider()
    st.caption("Built with LangGraph + Qdrant + Streamlit")


# ── Agent (cached per provider + model + temperature ) ─────────────────
@st.cache_resource(show_spinner="Loading LLM and agent...")
def get_agent(provider: str, model: str, temp: float):
    llm = get_llm(provider, model, temperature=temp)
    return build_agent(llm)


try:
    agent = get_agent(provider_key, model_name, temperature)
except Exception as e:
    st.error(f"Failed to load LLM: {e}")
    st.info("Check your API key in the .env file and confirm the provider is reachable.")
    st.stop()


# ── Chat UI ───────────────────────────────────────────────────────────────────
st.title("Bank Education Loan Comparator")
st.caption(
    f"Using **{model_name}** · temp={temperature} · "
    "Ask about education loans from any indexed bank"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Unified message handler ───────────────────────────────────────────────────
def handle_message(user_input: str):
    """
    Full message cycle: add user message → call agent → render + save response.
    Used by both starter buttons and the chat input so both paths behave identically.
    """
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = run_agent(agent, user_input)
            except Exception as e:
                response = f"Error: {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


# ── Starter buttons — only shown before first message ─────────────────────────
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    starters = [
        "Compare interest rates across all banks",
        "What are HDFC's eligibility criteria?",
        "Which bank has the best education loan?",
        "What documents does SBI require?",
        "Compare loan amounts and tenure",
        "What is ICICI's moratorium period?",
    ]
    for i, col in enumerate(cols):
        with col:
            if st.button(starters[i], key=f"s{i}"):
                handle_message(starters[i])
                st.rerun()
            if st.button(starters[i + 3], key=f"s{i+3}"):
                handle_message(starters[i + 3])
                st.rerun()

# ── Render existing chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about education loans..."):
    handle_message(prompt)
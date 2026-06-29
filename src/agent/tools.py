"""
tools.py
--------
Four LangChain tools the LangGraph agent can call:

1. search_bank_policy               — semantic search scoped to ONE bank
2. compare_schemes_across_banks     — fan-out: raw chunks, ALL banks side-by-side
3. summarize_and_compare_banks      — map-reduce: LLM summary per bank, then compare
4. list_available_banks_and_schemes — meta-tool, tells agent what's in the DB

Key design: vectorstore is loaded LAZILY (inside each tool call), NOT at import
time. This prevents file locks during the startup scrape+ingest pipeline.

Retrieval pipeline (Tools 1, 2, 3):
  bi-encoder (bge-large-en-v1.5)    → top-20 candidates from Qdrant
  cross-encoder (bge-reranker-large) → reranked, top-5 (Tool 1) / top-3 (Tool 2)
  Tool 3: top-5 per bank → one LLM summary call per bank → final comparison

Performance: both models cached with lru_cache — loaded once per process.
"""

from functools import lru_cache
from langchain_core.tools import tool
from qdrant_client.models import Filter, FieldCondition, MatchValue
from concurrent.futures import ThreadPoolExecutor, as_completed

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL_NAME  = "BAAI/bge-reranker-base"
QDRANT_DIR           = "data/qdrant_db"
COLLECTION_NAME      = "bank_schemes"
KNOWN_BANKS = ["HDFC", "ICICI", "SBI", "BOB", "UnionBank", "Axis", "Kotak", "CUB"]

# Retrieval knobs — change here to tune globally
BI_ENCODER_K      = 12  # candidates fetched from Qdrant per bank
RERANKER_TOP_N    = 4   # final chunks kept after reranking (Tool 1)
COMPARE_TOP_N     = 3   # final chunks kept per bank in fan-out (Tool 2)
SUMMARIZE_TOP_N   = 4   # chunks per bank fed into summarization LLM (Tool 3)

# Summarization LLM — default to gemini-2.5-flash (free tier, fast)
# Override by setting SUMMARY_PROVIDER / SUMMARY_MODEL env vars
SUMMARY_PROVIDER  = "gemini"
SUMMARY_MODEL     = "gemini-2.5-flash"


def _bank_filter(bank: str) -> Filter:
    """Build a Qdrant payload filter that matches a single bank."""
    return Filter(
        must=[FieldCondition(key="metadata.bank", match=MatchValue(value=bank))]
    )


@lru_cache(maxsize=1)
def _get_embeddings():
    """Load and cache the bi-encoder embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"},
    )


@lru_cache(maxsize=1)
def _get_reranker():
    """Load and cache the cross-encoder reranker model."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL_NAME, device="cpu")


def _get_vectorstore():
    """
    Return a Qdrant vectorstore using the cached embedding model.
    - If QDRANT_URL env var is set → Qdrant Cloud (GitHub Actions / HF Spaces)
    - Otherwise → local file mode (local dev)
    Not cached — QdrantClient is lightweight, and caching causes file locks.
    """
    import os
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    else:
        client = QdrantClient(path=QDRANT_DIR)

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=_get_embeddings(),
    )


def _rerank(query: str, docs: list, top_n: int) -> list:
    """Rerank docs by cross-encoder score, return top_n."""
    if not docs:
        return docs
    reranker = _get_reranker()
    pairs    = [(query, doc.page_content) for doc in docs]
    scores   = reranker.predict(pairs)
    ranked   = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def _get_summary_llm():
    """
    Return a fast/free LLM for per-bank summarization.
    Uses SUMMARY_PROVIDER/SUMMARY_MODEL constants.
    Falls back gracefully if API key is missing.
    """
    from src.agent.llm_factory import get_llm
    return get_llm(
        provider=SUMMARY_PROVIDER,
        model=SUMMARY_MODEL,
        temperature=0.3,   # slight creativity for fluent summaries
    )


# ── Tool 1: Single-bank semantic search ──────────────────────────────────────

@tool
def search_bank_policy(query: str, bank: str) -> str:
    """
    Search for loan policy information from a SPECIFIC bank.
    Use this when the user asks about one particular bank.

    Args:
        query: what to search for, e.g. 'education loan interest rate'
        bank:  which bank — must be one of: HDFC, ICICI, SBI, BOB,
               UnionBank, Axis, Kotak, CUB

    Returns:
        Relevant text chunks from that bank's documents, reranked by relevance.
    """
    bank = bank.upper().strip()
    if bank not in KNOWN_BANKS:
        return f"Unknown bank '{bank}'. Available banks: {', '.join(KNOWN_BANKS)}"

    vs         = _get_vectorstore()
    candidates = vs.similarity_search(query, k=BI_ENCODER_K, filter=_bank_filter(bank))
    results    = _rerank(query, candidates, top_n=RERANKER_TOP_N)

    if not results:
        return f"No relevant information found for '{query}' in {bank} documents."

    chunks = []
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        chunks.append(
            f"[{i}] Bank: {meta.get('bank')} | Type: {meta.get('doc_type', '?')}\n"
            f"{doc.page_content.strip()}"
        )

    return f"=== {bank} — results for: '{query}' ===\n\n" + "\n\n---\n\n".join(chunks)


# ── Tool 2: Cross-bank comparison (fan-out, raw chunks) ──────────────────────

@tool
def compare_schemes_across_banks(query: str) -> str:
    """
    Compare loan schemes across ALL banks for a given query.
    Returns raw reranked chunks per bank — best for specific fact lookups
    (interest rates, processing fees, eligibility criteria).
    Use summarize_and_compare_banks instead for broad comparison questions.

    Args:
        query: what to compare, e.g. 'education loan interest rate'

    Returns:
        Top reranked chunks from each bank presented side by side.
    """
    vs       = _get_vectorstore()
    sections = []

    for bank in KNOWN_BANKS:
        candidates = vs.similarity_search(query, k=BI_ENCODER_K, filter=_bank_filter(bank))
        results    = _rerank(query, candidates, top_n=COMPARE_TOP_N)

        if not results:
            sections.append(f"=== {bank} ===\nNo relevant information found.")
            continue

        chunks = []
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            chunks.append(
                f"  [{i}] ({meta.get('doc_type', '?')}) "
                f"{doc.page_content.strip()}"
            )

        sections.append(f"=== {bank} ===\n" + "\n\n".join(chunks))

    header = f"Cross-bank comparison for: '{query}'\n{'=' * 50}\n\n"
    return header + "\n\n".join(sections)


# ── Tool 3: Summarize and compare (map-reduce) ────────────────────────────────

def _summarize_one_bank(
    bank: str,
    query: str,
    vs,
    llm,
) -> tuple[str, str]:
    """Fetch, rerank, and summarize one bank's chunks — runs in a thread."""
    candidates = vs.similarity_search(
        query, k=BI_ENCODER_K, filter=_bank_filter(bank)
    )
    top_chunks = _rerank(query, candidates, top_n=SUMMARIZE_TOP_N)

    if not top_chunks:
        return bank, "Insufficient information available in the knowledge base."

    context = "\n\n".join(
        f"[Chunk {i}] {doc.page_content.strip()}"
        for i, doc in enumerate(top_chunks, 1)
    )
    prompt = (
        f"You are summarizing {bank} bank's education loan policy.\n"
        f"Question: {query}\n\n"
        f"Source chunks:\n{context}\n\n"
        f"Write a concise 3-5 sentence summary answering the question "
        f"using only the information above. If the information is insufficient, "
        f"say so clearly. Do not invent figures."
    )
    try:
        response = llm.invoke(prompt)
        summary  = response.content.strip()
    except Exception as e:
        summary = f"Summary unavailable: {e}"

    return bank, summary


@tool
def summarize_and_compare_banks(query: str) -> str:
    """
    Compare loan schemes across ALL banks using LLM-generated summaries.
    Better than compare_schemes_across_banks for broad questions like
    'which bank has the best education loan overall' or
    'compare collateral requirements across banks'.
    Flow: fetch top chunks per bank → parallel LLM summary per bank
    → return clean summaries for the agent to synthesize.

    Args:
        query: the comparison question, e.g. 'which bank offers lowest interest rate'
    Returns:
        One LLM-generated summary per bank, clearly labeled.
    """
    vs  = _get_vectorstore()
    llm = _get_summary_llm()

    # Fan out all banks in parallel — each bank gets its own thread
    bank_summaries: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(KNOWN_BANKS), 4)) as executor:
        futures = {
            executor.submit(_summarize_one_bank, bank, query, vs, llm): bank
            for bank in KNOWN_BANKS
        }
        for future in as_completed(futures):
            bank, summary = future.result()
            bank_summaries[bank] = summary

    # Reassemble in KNOWN_BANKS order (not thread completion order)
    results = []
    for bank in KNOWN_BANKS:
        summary = bank_summaries.get(bank, "No information available.")
        results.append(f"=== {bank} ===\n{summary}")

    header = f"Bank-by-bank summary for: '{query}'\n{'=' * 50}\n\n"
    return header + "\n\n".join(results)

# ── Tool 4: Meta — what's in the DB ──────────────────────────────────────────

@tool
def list_available_banks_and_schemes() -> str:
    """
    Lists all banks and document types available in the knowledge base.
    Use this when the user asks what banks or schemes are available,
    or when you need to know what data exists before searching.

    Returns:
        A summary of banks and document types in the database.
    """
    try:
        from qdrant_client import QdrantClient

        client  = QdrantClient(path=QDRANT_DIR)
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            limit=10_000,
        )

        if not records:
            return "The knowledge base appears to be empty."

        summary: dict[str, dict[str, int]] = {}
        for record in records:
            payload  = record.payload or {}
            metadata = payload.get("metadata", {})
            bank     = metadata.get("bank", "Unknown")
            doc_type = metadata.get("doc_type", "unknown")
            summary.setdefault(bank, {})
            summary[bank][doc_type] = summary[bank].get(doc_type, 0) + 1

        lines = ["Available banks and document types in the knowledge base:\n"]
        for bank in sorted(summary):
            lines.append(f"  {bank}:")
            for doc_type, count in sorted(summary[bank].items()):
                lines.append(f"    - {doc_type}: {count} chunks")

        return "\n".join(lines)

    except Exception as e:
        return f"Error reading knowledge base: {e}"


# ── Export all tools as a list (used by graph.py) ────────────────────────────
TOOLS = [
    search_bank_policy,
    compare_schemes_across_banks,
    summarize_and_compare_banks,
    list_available_banks_and_schemes,
]
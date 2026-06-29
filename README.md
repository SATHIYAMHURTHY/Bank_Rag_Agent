# 🏦 Bank-RAG-Agent

An **agentic Hybrid RAG** system that compares education loan schemes across 8 Indian banks using LangGraph, Qdrant, and a bi-encoder + cross-encoder retrieval pipeline.

Built as a portfolio project to demonstrate agentic retrieval-augmented generation with real-world web scraping, vector search, and tool-calling LLMs.

---

## What it does

Ask natural language questions about education loans — the agent decides which tool to call, retrieves relevant chunks from a Qdrant vector store, reranks them, and answers with citations from the actual bank data.

```
"Compare interest rates across all banks"
"What are HDFC's eligibility criteria?"
"Which bank offers the highest loan for studying abroad?"
"What documents does SBI require?"
```

**Banks covered:** HDFC · ICICI · SBI · Bank of Baroda · Union Bank · Axis · Kotak · CUB

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                    │
│         Provider · Model · Temperature sliders              │
└────────────────────────────┬────────────────────────────────┘
                             │ user query
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Agent  (create_react_agent)          │
│    LLM reads query + tool descriptions → decides tool       │
└────┬──────────────────┬───────────────┬────────────────┬────┘
     │                  │               │                │
     ▼                  ▼               ▼                ▼
Tool 1              Tool 2          Tool 3           Tool 4
search_bank_    compare_schemes_ summarize_and_  list_available_
policy()        _across_banks()  compare_banks() banks_and_
                                                 schemes()
Single bank     Fan-out:         Map-reduce:     Scrolls Qdrant
metadata-       8 parallel       8 parallel      metadata for
filtered        Qdrant queries   Qdrant queries  bank/doc counts
search          (Thread-         + 1 LLM call
                PoolExecutor)    per bank →
                                 clean summary
     │                  │               │
     └──────────────────┴───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Retrieval Pipeline                        │
│                                                             │
│   bge-large-en-v1.5 (bi-encoder)                           │
│   Qdrant top-12  →  bge-reranker-base  →  top-4 results   │
│   1024-dim cosine similarity + metadata.bank filter        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Qdrant Vector Store                        │
│   Collection: bank_schemes  │  ~155 chunks                  │
│   Metadata: bank · scheme_name · doc_type · url            │
└─────────────────────────────────────────────────────────────┘
                         ▲
                         │ ingest
┌─────────────────────────────────────────────────────────────┐
│              Scraping + Ingestion Pipeline                  │
│                                                             │
│   sources.py → 20 URLs across 8 banks                      │
│   scraper.py → aiohttp (fast) + Playwright (JS pages)      │
│              → TTL cache (24h), async, bank-specific        │
│                noise cleaning                               │
│   ingest.py  → RecursiveCharacterTextSplitter (600/100)    │
│              → low-value chunk filter                       │
│              → bge-large-en-v1.5 embeddings                 │
│              → Qdrant upsert                                │
└─────────────────────────────────────────────────────────────┘
```

### Why hybrid retrieval?

Plain vector similarity search returns results dominated by whichever bank has the most chunks (ICICI in this dataset). The fan-out pattern — one metadata-filtered Qdrant query per bank — guarantees equal representation regardless of chunk count.

```
Plain search for "interest rate" → [ICICI, ICICI, ICICI, ICICI]   ❌
Fan-out with filter per bank    → [HDFC, ICICI, SBI, BOB, ...]    ✅
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent | LangGraph (`create_react_agent`) |
| LLM | Gemini / Groq / Anthropic / Ollama (switchable) |
| Vector DB | Qdrant (local file mode) |
| Bi-encoder | `BAAI/bge-large-en-v1.5` (1024-dim) |
| Cross-encoder | `BAAI/bge-reranker-base` |
| Scraping | aiohttp + Playwright (async hybrid) |
| Orchestration | LangChain + LangGraph |
| Package manager | uv |

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- At least one LLM API key (Gemini free tier recommended — no card required)

### 1. Clone and create virtual environment

```bash
git clone https://github.com/SATHIYAMHURTHY/Bank_Rag_Agent.git
cd Bank_Rag_Agent
uv venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

Install Playwright browsers (needed for JS-rendered bank pages):

```bash
playwright install chromium
```

### 3. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
# .env

# LLM — add the key for whichever provider you choose
GEMINI_API_KEY=your-key-here        # free at aistudio.google.com
GROQ_API_KEY=your-key-here          # free at console.groq.com
ANTHROPIC_API_KEY=your-key-here     # paid, console.anthropic.com

# Qdrant — leave blank to use local file mode (default)
QDRANT_URL=
QDRANT_API_KEY=
```

### 4. Run

```bash
streamlit run src/ui/app.py
```

On first launch the app will:
1. Scrape all 20 bank sources (async, ~15–20s cold start)
2. Embed and index ~155 chunks into local Qdrant
3. Load bge-large-en-v1.5 and bge-reranker-base (~1.4GB total, cached after first run)

Subsequent launches skip scraping if data is under 24 hours old.

---

## How the 4 tools work

| Tool | When the LLM calls it | What it does |
|---|---|---|
| `search_bank_policy` | Single bank question | Qdrant query filtered to one bank → reranker → top-4 chunks |
| `compare_schemes_across_banks` | Cross-bank comparison | 8 parallel filtered Qdrant queries (ThreadPoolExecutor) → reranked chunks per bank |
| `summarize_and_compare_banks` | "Best bank" / summary questions | Same fan-out + one LLM call per bank to summarize, then final synthesis |
| `list_available_banks_and_schemes` | "What banks do you have?" | Qdrant metadata scroll → distinct bank + scheme names |

The LLM never sees raw Qdrant output directly — all retrieval goes through the cross-encoder reranker first.

---

## LLM Providers

Switch provider and model from the sidebar at runtime. No code changes needed.

| Provider | Free tier | Models |
|---|---|---|
| Gemini (Google) | ✅ Yes | gemini-2.5-flash, gemini-2.5-pro |
| Groq | ✅ Yes | openai/gpt-oss-20b, openai/gpt-oss-120b |
| Ollama | ✅ Local | llama3.2, mistral, phi3 |
| Anthropic | ❌ Paid | claude-sonnet-4-6, claude-opus-4-6 |

**Recommended for getting started:** Gemini — free tier, no credit card, best tool-calling quality.

---

## Key Design Decisions

**Fan-out over plain similarity search** — without per-bank metadata filters, embedding similarity alone heavily favours whichever bank has the most chunks. Fan-out gives every bank equal retrieval opportunity.

**Bi-encoder → cross-encoder pipeline** — `bge-large-en-v1.5` retrieves the top-12 candidates quickly; `bge-reranker-base` re-scores all 12 pairs (query, chunk) and returns the top-4. The reranker is slower but far more accurate than cosine similarity alone.

**`bge-reranker-base` over `large`** — half the size, 2–3x faster on CPU, comparable quality for this domain.

**Lazy vectorstore, cached model loaders** — `_get_vectorstore()` is not cached (avoids Qdrant file lock during startup); the embedding and reranker models are loaded once via `@lru_cache(maxsize=1)`.

**TTL scrape cache** — raw JSON files are kept for 24 hours. Re-launching the app within that window skips all scraping and re-indexes from cache instead.

---

## Retrieval Configuration

Tune at the top of `src/agent/tools.py`:

```python
BI_ENCODER_K  = 12   # candidates retrieved from Qdrant
RERANKER_TOP_N = 4   # kept after cross-encoder reranking (Tools 1 & 2)
COMPARE_TOP_N  = 3   # per-bank chunks for comparison tool
SUMMARIZE_TOP_N = 4  # per-bank chunks fed to summarization LLM
```

---

## Packages

```
# Scraping
aiohttp, aiofiles, playwright, beautifulsoup4, lxml, requests

# LangChain + LangGraph
langchain, langchain-community, langchain-text-splitters
langchain-huggingface, langchain-qdrant==1.1.0

# LLM providers
langchain-google-genai, langchain-anthropic, langchain-groq

# Vector DB + Embeddings
qdrant-client, sentence-transformers

# UI
streamlit, python-dotenv
```

---

## Limitations

- **Kotak / CUB** have fewer indexed chunks due to scraping gaps — answers for these banks may be less detailed than others.
- Interest rates and loan terms change frequently. Data reflects the last scrape and may not be current — always verify directly with the bank.
- First cold start (empty cache) takes 15–20 seconds for scraping and ~60 seconds for model loading on CPU.

---

## License

MIT

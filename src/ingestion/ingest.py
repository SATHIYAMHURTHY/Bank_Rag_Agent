# src/ingestion/ingest.py
import os
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_DIR           = Path(__file__).resolve().parents[2] / "data" / "qdrant_db"
COLLECTION_NAME      = "bank_schemes"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM        = 1024         # must match the model above
RAW_DATA_DIR         = Path(__file__).resolve().parents[2] / "data" / "raw"

CHUNK_SIZE       = 600
CHUNK_OVERLAP    = 100
MIN_CHUNK_LENGTH = 80


def load_raw_records() -> list[dict]:
    """Load every JSON file from data/raw/ into a list of dicts."""
    records = []
    for file_path in RAW_DATA_DIR.glob("*.json"):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        records.append(data)
    return records


def is_low_value_chunk(text: str) -> bool:
    """Heuristic filter for heading/nav-soup chunks."""
    if len(text) < MIN_CHUNK_LENGTH:
        return True
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return True
    short_lines = [line for line in lines if len(line) < 35]
    if len(short_lines) / len(lines) > 0.7:
        return True
    return False


def chunk_records(records: list[dict]) -> list[Document]:
    """Split each record's content into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = []
    skipped   = 0

    for record in records:
        chunks = splitter.split_text(record["content"])
        for i, chunk_text in enumerate(chunks):
            if is_low_value_chunk(chunk_text):
                skipped += 1
                continue
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "bank":        record["bank"],
                    "scheme_type": record["scheme_type"],
                    "scheme_name": record["scheme_name"],
                    "doc_type":    record.get("doc_type", "unknown"),
                    "url":         record["url"],
                    "chunk_index": i,
                },
            )
            documents.append(doc)

    print(f"Skipped {skipped} low-value chunks")
    return documents


def get_qdrant_client() -> QdrantClient:
    """
    Return a Qdrant client.
    - If QDRANT_URL env var is set → connect to Qdrant Cloud (GitHub Actions / HF Spaces)
    - Otherwise → local file mode (local dev)
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
 
    if qdrant_url:
        print(f"Connecting to Qdrant Cloud: {qdrant_url}")
        return QdrantClient(url=qdrant_url, api_key=qdrant_key)
 
    # Local file mode
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_DIR))


def collection_exists() -> bool:
    """Return True if the collection already exists and has vectors."""
    try:
        client   = get_qdrant_client()
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            return False
        info  = client.get_collection(COLLECTION_NAME)
        count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0
        return count > 0
    except Exception:
        return False


def build_vectorstore(documents: list[Document]) -> QdrantVectorStore:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={
            "batch_size": 64,
            "normalize_embeddings": True,
        },
        model_kwargs={"device": "cpu"},
    )

    client = get_qdrant_client()   # was _make_client()

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    print(f"Embedding {len(documents)} chunks into Qdrant at {QDRANT_DIR} ...")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vectorstore.add_documents(documents)
    print("Done. Vector store built and saved to disk.")
    return vectorstore


if __name__ == "__main__":
    records = load_raw_records()
    print(f"Loaded {len(records)} records")
    documents = chunk_records(records)
    print(f"Kept {len(documents)} chunks after filtering")
    build_vectorstore(documents)
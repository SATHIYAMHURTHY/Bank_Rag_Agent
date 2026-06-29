from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def show_results(query: str, results):
    print(f"\n=== Query: '{query}' ===")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.metadata['bank']} | {doc.metadata['scheme_name']}")
        print(doc.page_content[:200])


if __name__ == "__main__":
    store = load_vectorstore()

    # Test 1: plain similarity search, no filtering
    results = store.similarity_search("education loan interest rate", k=4)
    show_results("education loan interest rate", results)

    # Test 2: filtered to just one bank, to prove metadata filtering works
    results = store.similarity_search(
        "eligibility criteria",
        k=3,
        filter={"bank": "ICICI"},
    )
    show_results("eligibility criteria (ICICI only)", results)
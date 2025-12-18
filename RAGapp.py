import os
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    docs = load_all_documents("data")

    store = FaissVectorStore("faiss_store")

    if not os.path.exists("faiss_store/faiss.index"):
        store.build_from_documents(docs)
    else:
        store.load()

    rag_search = RAGSearch(store)
    query = "What is similarity search?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)

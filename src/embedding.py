from typing import List, Any

from sentence_transformers import SentenceTransformer
import numpy as np
from .data_loader import load_all_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 2000, chunk_overlap: int = 500):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Initialized SentenceTransformer model: {model_name}")

    def chunk_documents(self, documents: List[Any])->List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    def embedded_chunks(self, chunks: List[any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings for {len(chunks)} chunks. and embedding dimension is {embeddings.shape}")
        return embeddings
    
if __name__ == "__main__":
    data_directory = "../data/pdf"
    documents = load_all_documents(data_directory)
    
    embedding_pipeline = EmbeddingPipeline()
    chunks = embedding_pipeline.chunk_documents(documents)
    embeddings = embedding_pipeline.embedded_chunks(chunks)

        


import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSManager:
    def __init__(self, chunked_path="chunked_texts.json", vector_path="vector_store/faiss_index.index", meta_path="vector_store/metadata.json"):
        self.chunked_path = chunked_path
        self.vector_path = vector_path
        self.meta_path = meta_path
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        os.makedirs(os.path.dirname(self.vector_path), exist_ok=True)

    def load_chunked_texts(self):
        if not os.path.exists(self.chunked_path):
            print("❌ No chunked texts found! Run chunking.py first.")
            return []
        with open(self.chunked_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_embeddings(self, texts):
        return self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def initialize_faiss(self, embedding_dim):
        return faiss.IndexFlatL2(embedding_dim)

    def process_embeddings(self):
        chunked_data = self.load_chunked_texts()
        if not chunked_data:
            return
        metadata = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        existing_chunks = {entry["chunk"] for entry in metadata}
        new_chunks = [entry for entry in chunked_data if entry["chunk"] not in existing_chunks]
        if not new_chunks:
            print("✅ No new chunks to process. FAISS is up-to-date.")
            return
        new_texts = [entry["chunk"] for entry in new_chunks]
        new_embeddings = self.generate_embeddings(new_texts)
        new_embeddings = np.array(new_embeddings)
        if os.path.exists(self.vector_path):
            index = faiss.read_index(self.vector_path)
        else:
            index = self.initialize_faiss(new_embeddings.shape[1])
        index.add(new_embeddings)
        faiss.write_index(index, self.vector_path)
        metadata.extend(new_chunks)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Processed {len(new_chunks)} new embeddings and updated FAISS!")

    def search_faiss(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        if not os.path.exists(self.vector_path):
            print("❌ No FAISS index found. Run embedding.py first.")
            return []
        index = faiss.read_index(self.vector_path)
        D, I = index.search(query_embedding, top_k)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return [{"chunk": metadata[i]["chunk"], "file": metadata[i]["paper"], "score": D[0][j]} for j, i in enumerate(I[0]) if i < len(metadata)]

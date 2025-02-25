import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Define file paths
CHUNKED_TEXTS_PATH = "chunked_texts.json"
VECTOR_STORE_PATH = "vector_store/faiss_index.index"
METADATA_PATH = "vector_store/metadata.json"

# Ensure vector store directory exists
os.makedirs("vector_store", exist_ok=True)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Change if needed


# Function to load chunked texts
def load_chunked_texts():
    if not os.path.exists(CHUNKED_TEXTS_PATH):
        print("❌ No chunked texts found! Run chunking.py first.")
        return []

    with open(CHUNKED_TEXTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Function to generate embeddings
def generate_embeddings(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)


# Function to initialize FAISS index
def initialize_faiss(embedding_dim):
    index = faiss.IndexFlatL2(embedding_dim)
    return index


# Function to process and store embeddings
def process_embeddings():
    # Load chunked text data
    chunked_data = load_chunked_texts()
    if not chunked_data:
        return

    # Load existing metadata if available
    metadata = []
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    existing_chunks = {entry["chunk"] for entry in metadata}

    new_chunks = []
    for entry in chunked_data:
        chunk_text = entry["chunk"]
        if chunk_text not in existing_chunks:
            new_chunks.append(chunk_text)
            metadata.append(entry)

    if not new_chunks:
        print("✅ No new chunks to process. FAISS is up-to-date.")
        return

    # Generate embeddings for new chunks
    new_embeddings = generate_embeddings(new_chunks)
    new_embeddings = np.array(new_embeddings)

    # Initialize FAISS or load existing index
    if os.path.exists(VECTOR_STORE_PATH):
        index = faiss.read_index(VECTOR_STORE_PATH)
    else:
        index = initialize_faiss(new_embeddings.shape[1])

    index.add(new_embeddings)
    faiss.write_index(index, VECTOR_STORE_PATH)

    # Save updated metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Processed {len(new_chunks)} new embeddings and updated FAISS!")


# Function to search FAISS for relevant chunks
def search_faiss(query, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ No FAISS index found. Run embedding.py first.")
        return []

    index = faiss.read_index(VECTOR_STORE_PATH)
    D, I = index.search(query_embedding, top_k)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    results = [{"chunk": metadata[i]["chunk"], "file": metadata[i]["paper"], "score": D[0][j]}
               for j, i in enumerate(I[0])]

    return results


if __name__ == "__main__":
    process_embeddings()

    # Example search
    query_text = '''In the paper "Characterization of presence and activity of microRNAs in the rumen of cattle hints at possible host‑microbiota
cross‑talk mechanism" whaT was the Network analysis and functional prediction'''
    retrieved_chunks = search_faiss(query_text)
    print("\n🔍 Search Results:\n", retrieved_chunks)

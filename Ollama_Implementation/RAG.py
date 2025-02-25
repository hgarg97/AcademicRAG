import os
import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Paths
CHUNKED_JSON_PATH = "chunked_texts.json"
FAISS_INDEX_PATH = "vector_store/faiss_index.index"
METADATA_PATH = "vector_store/metadata.json"

# Load Embedding Model
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # Adjust to your preferred model
LLM_MODEL = OllamaLLM(model="llama3.2:latest")

# Prompt Template
PROMPT_TEMPLATE = """
You are an expert research assistant with knowledge of Animal Science Research.
Use the provided context to answer the query. If unsure, say you don't know.
Be concise and factual.

Query: {user_query}
Context: {document_context}
Answer:
"""

# Function to Load Chunked Texts
def load_chunked_texts():
    with open(CHUNKED_JSON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

# Function to Compute and Store Embeddings in FAISS
def build_faiss_index():
    chunked_data = load_chunked_texts()
    texts = [item["chunk"] for item in chunked_data]

    embeddings = EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)
    
    # FAISS Index Creation
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS Index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save Metadata (Mapping Index to Text)
    with open(METADATA_PATH, "w", encoding="utf-8") as meta_file:
        json.dump(chunked_data, meta_file, indent=4)

    print("✅ FAISS index built and saved.")

# Function to Load FAISS Index
def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        build_faiss_index()

    return faiss.read_index(FAISS_INDEX_PATH)

# Function to Search for Relevant Chunks
def find_related_chunks(query, top_k=5):
    index = load_faiss_index()
    
    query_embedding = EMBEDDING_MODEL.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    with open(METADATA_PATH, "r", encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)

    results = [metadata[idx]["chunk"] for idx in indices[0] if idx < len(metadata)]
    return results

# Function to Generate LLM Response
def generate_answer(user_query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LLM_MODEL

    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# 🟢 Streamlit UI
st.title("📘 Academic RAG - Chatbot")
st.markdown("### Ask questions based on indexed research papers!")

# User Query Input
user_query = st.chat_input("Enter your research query...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Retrieving relevant information..."):
        relevant_chunks = find_related_chunks(user_query, top_k=10)
        ai_response = generate_answer(user_query, relevant_chunks)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)

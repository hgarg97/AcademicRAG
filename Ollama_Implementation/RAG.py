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
@st.cache_resource
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

    results = []
    references = set()
    unique_files = set()

    for idx in indices[0]:
        if idx < len(metadata):
            chunk_text = metadata[idx]["chunk"]
            paper_info = metadata[idx]["paper"]
            file_name = metadata[idx].get("file_name", None)  # Get filename
            doi = metadata[idx].get("doi", "DOI not available")

            results.append((chunk_text, file_name))  # Store chunk + filename
            references.add(f"{paper_info} (DOI: {doi})")

            if file_name:
                unique_files.add(file_name)  # Track unique filenames

    return results, references, list(unique_files)

# Function to Generate LLM Response
def generate_answer(user_query, context_chunks, references):
    context_text = "\n\n".join(chunk[0] for chunk in context_chunks)
    reference_text = "\n".join(f"- {ref}" for ref in references)

    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LLM_MODEL

    answer = response_chain.invoke({"user_query": user_query, "document_context": context_text})

    # Append References to Answer
    answer += f"\n\n📚 **References:**\n{reference_text}"
    
    return answer

# 🟢 Streamlit UI
st.title("📘 Academic RAG - Chatbot")
st.markdown("### Ask questions based on indexed research papers!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# User Query Input
user_query = st.chat_input("Enter your research query...")

if user_query:
    # Display user query
    with st.chat_message("user"):
        st.write(user_query)

    # Retrieve relevant chunks & references
    with st.spinner("Retrieving relevant information..."):
        retrieved_chunks, references, unique_files = find_related_chunks(user_query, top_k=10)
        ai_response = generate_answer(user_query, retrieved_chunks, references)

    # 🔹 Display Retrieved Chunks with Source Papers
    with st.expander("🔍 Retrieved Context Chunks (Click to Expand)"):
        for idx, (chunk, file_name) in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {idx + 1}:**")
            st.info(chunk)

    with st.expander("📂 Source Papers (Click to Expand)"):
        for file_name in unique_files:
            pdf_path = f"G:/AcademicRAG/Subdataset/{file_name}"
            st.markdown(f"[📄 {file_name}]({pdf_path})", unsafe_allow_html=True)

    # Display AI response
    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)

    # Store conversation in history
    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", ai_response))

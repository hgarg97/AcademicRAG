import os
import json
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from embedding import FAISSManager

class AcademicRAG:
    def __init__(self):
        self.chunked_path = "chunked_texts.json"
        self.faiss_path = "vector_store/faiss_index.index"
        self.metadata_path = "vector_store/metadata.json"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = OllamaLLM(model="llama3.2:latest")
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert research assistant with knowledge of Animal Science Research.
        Use the provided context to answer the query. If unsure, say you don't know.
        Be concise and factual.

        Query: {user_query}
        Context: {document_context}
        Answer:
        """)

    def load_faiss_index(self):
        if not os.path.exists(self.faiss_path):
            FAISSManager().process_embeddings()
        return faiss.read_index(self.faiss_path)

    def find_related_chunks(self, query, top_k=5):
        index = self.load_faiss_index()
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        results, references, files = [], set(), set()
        for idx in indices[0]:
            if idx < len(metadata):
                chunk = metadata[idx]["chunk"]
                file = metadata[idx].get("file_name")
                paper = metadata[idx]["paper"]
                doi = metadata[idx].get("doi", "DOI not available")
                results.append((chunk, file))
                references.add(f"{paper} (DOI: {doi})")
                if file:
                    files.add(file)
        return results, references, list(files)

    def generate_answer(self, query, chunks, references):
        context = "\n\n".join(chunk[0] for chunk in chunks)
        refs = "\n".join(f"- {ref}" for ref in references)
        response = (self.prompt_template | self.llm).invoke({"user_query": query, "document_context": context})
        return f"{response}\n\n📚 **References:**\n{refs}"

    def launch_ui(self):
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
                chunks, refs, files = self.find_related_chunks(user_query, top_k=10)
                response = self.generate_answer(user_query, chunks, refs)

            # 🔹 Display Retrieved Chunks with Source Papers
            with st.expander("🔍 Retrieved Context Chunks (Click to Expand)"):
                for idx, (chunk, file) in enumerate(chunks):
                    st.markdown(f"**Chunk {idx+1}:**")
                    st.info(chunk)
            with st.expander("📂 Source Papers (Click to Expand)"):
                for file in files:
                    path = f"G:/AcademicRAG/Subdataset/{file}"
                    st.markdown(f"[📄 {file}]({path})", unsafe_allow_html=True)

            # Display AI response
            with st.chat_message("assistant", avatar="🤖"):
                st.write(response)

            # Store conversation in history
            st.session_state.chat_history.append(("user", user_query))
            st.session_state.chat_history.append(("assistant", response))

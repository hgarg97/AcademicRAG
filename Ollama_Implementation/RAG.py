import os
import json
import faiss
import streamlit as st
import base64
import config
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from embedding import FAISSManager
from bm25 import BM25Retriever
from graph_retriever import GraphRetriever
from graph_extraction import TripletExtractor

class AcademicRAG:
    def __init__(self, retriever_mode="faiss"):
        self.chunked_path = config.CHUNKED_JSON_PATH
        self.faiss_path = config.FAISS_INDEX_PATH
        self.metadata_path = config.METADATA_PATH
        self.retriever_mode = retriever_mode
        self.bm25 = BM25Retriever() if retriever_mode in ["bm25", "bm25+graphrag", "hybrid"] else None
        self.graph = GraphRetriever(config.GRAPH_PATH) if retriever_mode in ["graphrag", "faiss+graphrag", "bm25+graphrag", "hybrid"] else None
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.llm = OllamaLLM(model=config.LLM_MODEL_NAME)
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert research assistant with knowledge of Animal Science Research.
        Use the provided context to answer the query. If unsure, say you don't know.
        Be concise and factual.

        Query: {user_query}
        Context: {document_context}
        Answer:
        """)

    def load_faiss_index(self):
        vector_dir = os.path.dirname(self.faiss_path)
        os.makedirs(vector_dir, exist_ok=True)
        if not os.path.exists(self.faiss_path):
            FAISSManager().process_embeddings()
        return faiss.read_index(self.faiss_path)

    def find_related_chunks(self, query, top_k=5):
        results, references, files = [], set(), set()

        if "faiss" in self.retriever_mode:
            faiss_index = self.load_faiss_index()
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            distances, indices = faiss_index.search(query_embedding, top_k)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for idx in indices[0]:
                if idx < len(metadata):
                    results.append((metadata[idx]["chunk"], metadata[idx].get("file_name")))
                    references.add(f"{metadata[idx]['paper']} (DOI: {metadata[idx].get('doi', 'N/A')})")
                    files.add(metadata[idx].get("file_name"))

        if self.bm25:
            bm25_chunks = self.bm25.search(query, top_k=top_k)
            for chunk in bm25_chunks:
                results.append((chunk["chunk"], chunk["file_name"]))
                references.add(f"{chunk['paper']} (DOI: {chunk.get('doi', 'N/A')})")
                files.add(chunk["file_name"])

        if self.graph:
            extractor = TripletExtractor(model_name=config.TRIPLET_MODEL_NAME)
            entities = extractor.extract_entities(query)
            for entity in entities:
                graph_chunks = self.graph.query(entity, depth=2)
                for edge in graph_chunks:
                    results.append((edge["sentence"], f"{edge['source']} ↔ {edge['target']}"))
                    references.add(f"Graph Entity: {edge['source']}")

        seen = set()
        deduped = []
        for r in results:
            if r[0] not in seen:
                deduped.append(r)
                seen.add(r[0])

        return deduped[:top_k], references, list(files)

    def generate_answer(self, query, chunks, references):
        context = "\n\n".join(chunk[0] for chunk in chunks)
        refs = "<br>".join(f"- {ref}" for ref in references)
        response = (self.prompt_template | self.llm).invoke({"user_query": query, "document_context": context})
        return f"{response}<br><br><p style='color:#500000;font-size:18px;'><strong>📚 References:</strong><br>{refs}</p>"

    def launch_ui(self):
        with open(config.TAMU_LOGO_PATH, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()

        bg_img = Image.open(config.BACKGROUND_IMAGE_PATH)
        buffered = BytesIO()
        bg_img.save(buffered, format="JPEG")
        bg_base64 = base64.b64encode(buffered.getvalue()).decode()

        st.set_page_config(page_title="Animal Science Chatbot", layout="wide")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{bg_base64}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(255, 255, 255, 0.50);
                padding: 2.5rem;
                border-radius: 20px;
            }}
            h1 {{ font-size: 40px !important; }}
            h4 {{ font-size: 18px !important; color: white !important; }}
            h3, h5 {{ font-size: 24px !important; color: #500000 !important; font-weight: bold !important; }}
            .stMarkdown p {{ font-size: 18px !important; color: #500000 !important; }}
            div[data-testid="stChatMessageContent"] {{ color: #500000 !important; font-size: 18px !important; }}
            details summary {{
                color: #500000 !important;
                font-size: 14px !important;
                font-weight: 600 !important;
                background-color: #f3f3f3 !important;
                padding: 6px 12px !important;
                border-radius: 6px !important;
                margin-bottom: 8px !important;
                cursor: pointer;
            }}
            details p, details span {{
                color: #333 !important;
                font-size: 14px !important;
                background-color: transparent !important;
                padding: 6px 0 !important;
                border-radius: 0 !important;
                margin-top: 6px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f'''
            <div style="background-color:#500000;padding:20px 30px;border-radius:12px;margin-bottom:20px;display:flex;align-items:center;">
                <img src="data:image/png;base64,{logo_base64}" style="height:80px;margin-right:30px">
                <div>
                    <h1 style="color:white;margin:0;">Animal Science RAG Chatbot</h1>
                    <h4 style="color:white;font-size:18px;margin:0;">
                        Explore cutting-edge animal nutrition research powered by Texas A&M and LLaMA 3.2.
                    </h4>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown("""
        ### 🐄 Who Are We?
        <p style='color:#500000; font-size:18px;'>
        We are a research-driven platform dedicated to advancing the accessibility and analysis of animal nutrition science through cutting-edge AI technologies. Our system leverages Retrieval-Augmented Generation (RAG) pipelines powered by Large Language Models (LLMs) like Meta’s LLaMA to provide fast, context-aware answers from a curated library of scientific publications.<br><br>
        This tool enables researchers, students, and industry experts to efficiently search, retrieve, and synthesize insights from thousands of peer-reviewed research papers in the field of animal science. Whether you're exploring new feed formulations, analyzing health outcomes, or reviewing recent advancements in livestock nutrition, this intelligent assistant provides structured, reference-backed responses to support your decision-making and innovation.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        ### 💬 Ask a question below:
        """)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

        user_query = st.chat_input("Enter your research query...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)

            with st.spinner("Retrieving relevant information..."):
                chunks, refs, files = self.find_related_chunks(user_query, top_k=10)
                response = self.generate_answer(user_query, chunks, refs)

            with st.chat_message("assistant", avatar="🦮"):
                st.markdown(response, unsafe_allow_html=True)

            with st.expander("🔍 View Retrieved Context Chunks"):
                for idx, (chunk, file) in enumerate(chunks):
                    st.markdown(f"**Chunk {idx+1} from {file}:**")
                    st.markdown(f"""
                    <div style='border-left: 4px solid #500000; padding-left: 10px; margin: 8px 0;'>
                        <p style='margin: 0; font-size: 14px; color: #333; line-height: 1.5;'>{chunk}</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📂 Source Papers"):
                for file in files:
                    st.markdown(f"[📄 {file}]({config.RAW_PDF_DIR}/{file})")

            st.session_state.chat_history.append(("user", user_query))
            st.session_state.chat_history.append(("assistant", response))
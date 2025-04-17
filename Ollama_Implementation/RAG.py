import os
import json
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from chunking import PDFChunker
from embedding import FAISSManager
from bm25 import BM25Retriever
from graph_retriever import GraphRetriever
from graph_extraction import TripletExtractor
import config
import shutil

class AcademicRAG:
    def __init__(self, retriever_mode="faiss"):
        self.chunked_path = config.CHUNKED_JSON_PATH
        self.faiss_path = config.FAISS_INDEX_PATH
        self.metadata_path = config.METADATA_PATH
        self.retriever_mode = retriever_mode
        self.bm25 = BM25Retriever() if retriever_mode in ["bm25", "bm25+graphrag", "hybrid"] else None
        self.graph = GraphRetriever(config.GRAPH_PATH) if retriever_mode in ["graphrag", "faiss+graphrag", "bm25+graphrag", "hybrid"] else None
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.llm = None  # initialized later
        self.llm_temperature = 0.7
        self.model_name = os.getenv("RAG_MODEL_NAME", config.LLM_MODEL_NAME)
        self.set_temperature(self.llm_temperature)
        self.chat_prompt = ChatPromptTemplate.from_template("""
        You are an expert research assistant with knowledge of Animal Science Research.
        Use the previous conversation (if any) and provided context to answer this query. If unsure, say you don't know.

        Previous Conversation:
        {history_context}

        Current Query: {user_query}
        Context:
        {document_context}

        Answer:
        """)
        self.user_pdf_chunks = []
        self.raw_pdf_dir = config.RAW_PDF_DIR

    def load_faiss_index(self):
        vector_dir = os.path.dirname(self.faiss_path)
        os.makedirs(vector_dir, exist_ok=True)
        if not os.path.exists(self.faiss_path):
            FAISSManager().process_embeddings()
        return faiss.read_index(self.faiss_path)
    
    def set_model(self, model_name: str):
        self.model_name = model_name
        print(f"✅ [RAG] Setting model to: {model_name}")
        self.set_temperature(self.llm_temperature)
    
    def set_temperature(self, temperature: float):
        self.llm_temperature = temperature
        self.llm = OllamaLLM(model=self.model_name, temperature=temperature)

    def summarize_paper(self, paper_title: str):
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        selected_chunks = [item["chunk"] for item in metadata if item["paper"] == paper_title]
        context = "\n\n".join(selected_chunks)
        summary_prompt = ChatPromptTemplate.from_template("""
        You are a scientific summarizer. Given the context from a research paper, generate a clear, concise summary broken down as:
        1. Abstract
        2. Methodology
        3. Results
        4. Conclusion

        Paper Title: {paper_title}
        Context:
        {context}

        Summary:
        """)
        response = (summary_prompt | self.llm).invoke({
            "paper_title": paper_title,
            "context": context
        })
        return response

    def get_all_paper_titles(self):
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return sorted(set(item["paper"] for item in metadata if item.get("paper")))

    def process_uploaded_pdf(self, uploaded_file):
        os.makedirs("uploaded_pdfs", exist_ok=True)
        temp_path = os.path.join("uploaded_pdfs", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        chunker = PDFChunker()
        self.user_pdf_chunks = chunker.process_pdf(temp_path)

    def query_uploaded_pdfs(self, user_query, history_context="", selected_file=None):
        # Filter chunks if a specific file was selected
        chunks_to_search = self.user_pdf_chunks
        if selected_file:
            chunks_to_search = [chunk for chunk in self.user_pdf_chunks if chunk["file_name"] == selected_file]

        if not chunks_to_search:
            return [], {"No chunks found for selected file."}, [], "⚠️ No content available to retrieve."

        # Compute similarity
        query_emb = self.embedding_model.encode([user_query], convert_to_numpy=True)[0]
        chunk_embeddings = self.embedding_model.encode([c["chunk"] for c in chunks_to_search], convert_to_numpy=True)
        sims = cosine_similarity([query_emb], chunk_embeddings)[0]
        top_k_idx = sims.argsort()[::-1][:5]

        # Prepare chunks
        top_chunks = [(chunks_to_search[i]["chunk"], chunks_to_search[i]["file_name"]) for i in top_k_idx]
        refs = {f"Uploaded Document: {selected_file}"} if selected_file else {"Uploaded Document"}
        files = [selected_file] if selected_file else list({chunk["file_name"] for chunk in top_chunks})
        context = "\n\n".join(chunk[0] for chunk in top_chunks)

        # Run LLM
        response = (self.chat_prompt | self.llm).invoke({
            "user_query": user_query,
            "document_context": context,
            "history_context": history_context
        })

        return top_chunks, refs, files, response


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

    def generate_answer(self, query, chunks, references, history_context=""):
        context = "\n\n".join(chunk[0] for chunk in chunks)
        refs = "\n".join(f"- {ref}" for ref in references)
        response = (self.chat_prompt | self.llm).invoke({
            "user_query": query,
            "document_context": context,
            "history_context": history_context
        })
        return f"{response}\n\n📚 **References:**\n{refs}"
    
    # Auto-purge uploaded_pdfs on session end
    def cleanup_uploaded_pdfs(self):
        if os.path.exists("uploaded_pdfs"):
            shutil.rmtree("uploaded_pdfs")

# To preprocess new pdfs
# python main.py --mode preprocess

# To Use BM25
# python main.py --mode preprocess --use_bm25


# To Launch the Chatbot
# FAISS (default)
# streamlit run main.py -- --mode chatbot

# BM25 only
# streamlit run main.py -- --mode chatbot --retriever bm25

# Hybrid (BM25 + FAISS)
# streamlit run main.py -- --mode chatbot --retriever hybrid

import argparse
import subprocess
import socket
from chunking import PDFChunker
from embedding import FAISSManager
from RAG import AcademicRAG
import config

def is_ollama_running():
    try:
        with socket.create_connection(("localhost", 11434), timeout=1):
            return True
    except (OSError, ConnectionRefusedError):
        return False

def start_ollama_if_needed(model_name):
    if not is_ollama_running():
        print(f"🟡 Ollama is not running. Starting Ollama with model '{model_name}'...")
        try:
            subprocess.Popen(["ollama", "run", model_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("🟢 Ollama process started.")
        except FileNotFoundError:
            print("❌ 'ollama' command not found. Please install Ollama CLI.")
    else:
        print("🟢 Ollama is already running.")

def run_preprocessing(use_bm25=False):
    print("🔄 Starting Preprocessing Pipeline...")
    start_ollama_if_needed(config.LLM_MODEL_NAME)

    # Step 1: Chunk PDFs
    chunker = PDFChunker(output_json=config.CHUNKED_JSON_PATH)
    chunker.process_new_pdfs(config.RAW_PDF_DIR)
    print("✅ Chunking completed!")

    # Step 2: FAISS embedding
    embedder = FAISSManager(
        chunked_path=config.CHUNKED_JSON_PATH,
        vector_path=config.FAISS_INDEX_PATH,
        meta_path=config.METADATA_PATH
    )
    embedder.process_embeddings()
    print("✅ FAISS Indexing completed!")

    # Step 3: Optional BM25 indexing
    if use_bm25:
        from bm25 import BM25Retriever
        retriever = BM25Retriever()
        retriever.build_index()
        print("✅ BM25 Indexing completed!")

    print("🚀 Preprocessing Completed! Ready for Retrieval.")

def run_chatbot(retriever_mode="faiss"):
    print(f"💬 Launching Academic RAG Chatbot with {retriever_mode.upper()} retriever...")
    start_ollama_if_needed(config.LLM_MODEL_NAME)
    app = AcademicRAG(retriever_mode=retriever_mode)
    app.launch_ui()

def main():
    parser = argparse.ArgumentParser(description="Academic RAG Pipeline")
    parser.add_argument(
        "--mode",
        choices=["preprocess", "chatbot"],
        required=True,
        help="Choose 'preprocess' to process PDFs or 'chatbot' to launch the UI."
    )
    parser.add_argument(
        "--use_bm25",
        action="store_true",
        help="Also build BM25 index in preprocessing step."
    )
    parser.add_argument(
        "--retriever",
        choices=["faiss", "bm25", "hybrid"],
        default="faiss",
        help="Retriever to use in chatbot mode"
    )


    args = parser.parse_args()

    if args.mode == "preprocess":
        run_preprocessing(use_bm25=args.use_bm25)

    elif args.mode == "chatbot":
        print(f"📥 Using {args.retriever.upper()} retriever mode.")
        run_chatbot(retriever_mode=args.retriever)

if __name__ == "__main__":
    main()

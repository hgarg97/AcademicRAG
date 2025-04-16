# To preprocess new pdfs
# python main.py --mode preprocess

# To Use BM25
# python main.py --mode preprocess --use_bm25

# To Use GraphRAG
# python main.py --mode preprocess --use_graphrag

# Hybrid (Bm25 + GraphRAG)
# python main.py --mode preprocess --use_bm25 --use_graphrag

# To Launch the Chatbot
# FAISS (default)
# streamlit run main.py -- --mode chatbot

# BM25 only
# streamlit run main.py -- --mode chatbot --retriever bm25

# GraphRAG only
# streamlit run main.py -- --mode chatbot --retriever graphrag

# Hybrid (FAISS + BM25)
# streamlit run main.py -- --mode chatbot --retriever hybrid

# FAISS + GraphRAG
# streamlit run main.py -- --mode chatbot --retriever faiss+graphrag

# BM25 + GraphRAG
# streamlit run main.py -- --mode chatbot --retriever bm25+graphrag

# Full Hybrid (FAISS + BM25 + GraphRAG)
# streamlit run main.py -- --mode chatbot --retriever hybrid

import argparse
import subprocess
import socket
from chunking import PDFChunker
from embedding import FAISSManager
from RAG import AcademicRAG
import config
from graph_extraction import TripletExtractor
from graph_builder import GraphBuilder
import os
import subprocess


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


def run_preprocessing(use_bm25=False, use_graphrag=False):
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

    # Step 4: Optional GraphRAG
    if use_graphrag:
        print("🔗 Building GraphRAG...")
        TripletExtractor(model_name=config.TRIPLET_MODEL_NAME).run(
            chunked_path=config.CHUNKED_JSON_PATH,
            output_path=config.TRIPLET_PATH
        )
        builder = GraphBuilder(triplet_file=config.TRIPLET_PATH)
        builder.build_graph()
        builder.save_graph(path=config.GRAPH_PATH)
        print("✅ GraphRAG saved to", config.GRAPH_PATH)

    print("🚀 Preprocessing Completed! Ready for Retrieval.")


def run_chatbot(retriever_mode="faiss"):
    print(f"💬 Launching Academic RAG Chatbot with {retriever_mode.upper()} retriever...")
    start_ollama_if_needed(config.LLM_MODEL_NAME)
    os.environ["RAG_RETRIEVER_MODE"] = retriever_mode
    subprocess.run(["streamlit", "run", "app.py"])


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
        "--use_graphrag",
        action="store_true",
        help="Also run GraphRAG triplet extraction and graph building."
    )
    parser.add_argument(
        "--retriever",
        choices=["faiss", "bm25", "graphrag", "hybrid", "faiss+graphrag", "bm25+graphrag"],
        default="faiss",
        help="Retriever to use in chatbot mode"
    )

    args = parser.parse_args()

    if args.mode == "preprocess":
        run_preprocessing(use_bm25=args.use_bm25, use_graphrag=args.use_graphrag)

    elif args.mode == "chatbot":
        print(f"📥 Using {args.retriever.upper()} retriever mode.")
        run_chatbot(retriever_mode=args.retriever)


if __name__ == "__main__":
    main()

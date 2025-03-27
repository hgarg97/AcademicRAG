# To preprocess new pdfs
# python main.py --mode preprocess

# To Launch the Chatbot
# streamlit run main.py -- --mode chatbot


import argparse
from chunking import PDFChunker
from embedding import FAISSManager
from RAG import AcademicRAG
# from bm25_module import BM25Retriever  # Optional if you add BM25 support

# File paths
RAW_PDF_DIR = "G:/AcademicRAG/Subdataset/"
CHUNKED_JSON_PATH = "chunked_texts.json"
FAISS_INDEX_PATH = "vector_store/faiss_index.index"
# BM25_INDEX_PATH = "vector_store/bm25_index.pkl"

def run_preprocessing():
    print("🔄 Starting Preprocessing Pipeline...")

    # Step 1: Chunk PDFs
    chunker = PDFChunker(output_json=CHUNKED_JSON_PATH)
    chunker.process_new_pdfs(RAW_PDF_DIR)
    print("✅ Chunking completed!")

    # Step 2: FAISS embedding
    embedder = FAISSManager(
        chunked_path=CHUNKED_JSON_PATH,
        vector_path=FAISS_INDEX_PATH
    )
    embedder.process_embeddings()
    print("✅ FAISS Indexing completed!")

    # Step 3: Optional BM25 indexing
    # retriever = BM25Retriever()
    # retriever.build_index(CHUNKED_JSON_PATH, BM25_INDEX_PATH)
    # print("✅ BM25 Indexing completed!")

    print("🚀 Preprocessing Completed! Ready for Retrieval.")

def run_chatbot():
    print("💬 Launching Academic RAG Chatbot...")
    app = AcademicRAG()
    app.launch_ui()

def main():
    parser = argparse.ArgumentParser(description="Academic RAG Pipeline")
    parser.add_argument(
        "--mode",
        choices=["preprocess", "chatbot"],
        required=True,
        help="Choose 'preprocess' to process PDFs or 'chatbot' to launch the UI."
    )
    args = parser.parse_args()

    if args.mode == "preprocess":
        run_preprocessing()
    elif args.mode == "chatbot":
        run_chatbot()

if __name__ == "__main__":
    main()

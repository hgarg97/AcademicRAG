# To preprocess new pdfs
# python main.py --mode preprocess

# To Launch the Chatbot
# streamlit run main.py -- --mode chatbot

import argparse
from chunking import PDFChunker
from embedding import FAISSManager
from RAG import AcademicRAG
import config

def run_preprocessing():
    print("🔄 Starting Preprocessing Pipeline...")

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
    # retriever = BM25Retriever()
    # retriever.build_index(config.CHUNKED_JSON_PATH, config.BM25_INDEX_PATH)
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

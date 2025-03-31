# Configuration module for Academic RAG

# Paths
RAW_PDF_DIR = "G:/AcademicRAG/Subdataset/"
CHUNKED_JSON_PATH = "chunked_texts.json"
FAISS_INDEX_PATH = "vector_store/faiss_index.index"
METADATA_PATH = "vector_store/metadata.json"
# For BM25 support
BM25_INDEX_PATH = "vector_store/bm25_index.pkl"

# Models
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2:latest"

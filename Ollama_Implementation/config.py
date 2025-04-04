# ✅ UPDATED config.py

# === File & Folder Paths ===
RAW_PDF_DIR = "G:/AcademicRAG/Subdataset/"
CHUNKED_JSON_PATH = "files/chunked_texts.json"
FAISS_INDEX_PATH = "files/faiss_index.index"
METADATA_PATH = "files/metadata.json"
BM25_INDEX_PATH = "files/bm25_index.pkl"
GRAPH_PATH = "files/graph_data.gpickle"
TRIPLET_PATH = "files/graph_triplets.json"

# === Model Names ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2:latest"
TRIPLET_MODEL_NAME = "allenai/scibert_scivocab_cased"
NER_MODEL_NAME = "d4data/biomedical-ner-all"

# === Image Assets ===
IMAGE_DIR = "images"
TAMU_LOGO_PATH = f"{IMAGE_DIR}/tamu.jpg"
BACKGROUND_IMAGE_PATH = f"{IMAGE_DIR}/holstein.jpg"

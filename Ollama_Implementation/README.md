# 🧠 Retrieval-Augmented Generation (RAG) Chatbot for Academic PDFs

This project is a fully modular, class-based **RAG (Retrieval-Augmented Generation) system** that allows you to query a collection of academic **PDFs** and get contextual answers using **LLaMA 3.2 via Ollama**.

It integrates:

- Document preprocessing and intelligent **chunking**
- **Dense vector search (FAISS)** for semantic retrieval
- **Sparse keyword search (BM25)** for exact match queries
- **Graph-based retrieval (GraphRAG)** via scientific entity relationships
- A **Streamlit chatbot** interface powered by LLMs

---

## 🚦 End-to-End Workflow Overview

```
[PDFs]
 ↓
chunking.py (semantic splitting)
 ↓
embedding.py → FAISS index
       ↓
   bm25.py → BM25 index (optional)
       ↓
graph_extraction.py + graph_builder.py → GraphRAG (optional)
       ↓
RAG.py (retriever: faiss / bm25 / graphrag / hybrid)
 ↓
LLaMA 3.2 (via Ollama)
 ↓
Answer + References
```

---

## 📂 Folder Structure & Key Files

| File/Folder           | Purpose                                          |
| --------------------- | ------------------------------------------------ |
| `chunking.py`         | Extract and intelligently chunk PDF text         |
| `embedding.py`        | Create vector embeddings using MiniLM            |
| `bm25.py`             | Create a sparse keyword search index             |
| `graph_extraction.py` | Extract triplets (subject, relation, object)     |
| `graph_builder.py`    | Build a directed graph from triplets             |
| `graph_retriever.py`  | Query neighbors/entities from the graph          |
| `RAG.py`              | RAG chatbot logic with retriever + LLM           |
| `config.py`           | Centralized config for paths and model names     |
| `main.py`             | Unified CLI for preprocessing and chatbot launch |
| `files/`              | Stores FAISS, BM25, Graph indexes and metadata   |

---

## 🚀 How to Use

### ✅ 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🧱 2. Preprocess the PDFs

This step:

- Reads all PDFs
- Chunks their text
- Generates embeddings for semantic retrieval (FAISS)
- (Optional) Creates BM25 index for exact keyword search
- (Optional) Extracts triplets and builds a GraphRAG knowledge graph

#### ➤ Run FAISS-only pipeline:

```bash
python main.py --mode preprocess
```

#### ➤ Run FAISS + BM25 indexing:

```bash
python main.py --mode preprocess --use_bm25
```

#### ➤ Run FAISS + GraphRAG indexing:

```bash
python main.py --mode preprocess --use_graphrag
```

#### ➤ Run FAISS + BM25 + GraphRAG:

```bash
python main.py --mode preprocess --use_bm25 --use_graphrag
```

---

### 💬 3. Launch the Chatbot

You can choose **which retrieval mode to use**:

| Mode              | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| `faiss` (default) | Dense semantic retrieval — great for paraphrased or long-form queries |
| `bm25`            | Sparse keyword match — great for exact phrases, chemical names        |
| `graphrag`        | Graph-based — queries related entities via knowledge triplets         |
| `faiss+graphrag`  | Combines FAISS + Graph for semantic + entity-grounded context         |
| `bm25+graphrag`   | Combines BM25 + Graph for keyword + entity retrieval                  |
| `hybrid`          | Combines FAISS + BM25 + Graph — best for comprehensive retrieval      |

#### ➤ Launch default (FAISS):

```bash
streamlit run main.py -- --mode chatbot
```

#### ➤ BM25 only:

```bash
streamlit run main.py -- --mode chatbot --retriever bm25
```

#### ➤ GraphRAG only:

```bash
streamlit run main.py -- --mode chatbot --retriever graphrag
```

#### ➤ FAISS + GraphRAG:

```bash
streamlit run main.py -- --mode chatbot --retriever faiss+graphrag
```

#### ➤ BM25 + GraphRAG:

```bash
streamlit run main.py -- --mode chatbot --retriever bm25+graphrag
```

#### ➤ Full hybrid (FAISS + BM25 + GraphRAG):

```bash
streamlit run main.py -- --mode chatbot --retriever hybrid
```

---

## 🧠 How Retrieval Works

### ✅ FAISS (Dense Semantic Retrieval)

- Uses `all-MiniLM-L6-v2` via `sentence-transformers`
- Captures meaning beyond keywords
- Great for paraphrased or long-form questions

### ✅ BM25 (Sparse Keyword Retrieval)

- Based on token overlap + frequency
- Ideal for exact matches, chemical names, acronyms

### ✅ GraphRAG (Knowledge Graph Retrieval)

- Uses `SciBERT` (or BioBERT) to extract triplets
- Builds graph where nodes = entities, edges = relations from text
- Returns related nodes and sentences using graph traversal

### ✅ Hybrid

- Combines top results from all retrievals
- Deduplicates context
- Ensures semantic + symbolic + lexical coverage

---

## 🔧 Configuration

All file paths and model names are stored in:

```python
config.py
```

You can change:

- `RAW_PDF_DIR` — path to your academic PDFs
- `LLM_MODEL_NAME` — Ollama model to run (e.g. `llama3.2:latest`)
- `EMBEDDING_MODEL_NAME` — transformer model for embeddings
- `TRIPLET_MODEL_NAME` — SciBERT/BioBERT model for triplet extraction
- `TAMU_LOGO_PATH`, `BACKGROUND_IMAGE_PATH` — Streamlit UI images

---

## 🔍 Benefits of the System

- ⚙️ Fully modular (chunking, embedding, BM25, Graph, chatbot)
- 🧠 LLM-powered contextual answers
- 🔌 Easily swappable retrievers (FAISS, BM25, GraphRAG, Hybrid)
- 🚀 GPU-compatible for faster embeddings
- 🖼️ Beautiful Streamlit UI
- 🔄 Automatic Ollama startup if not running
- 🔓 Designed to scale with future improvements

---

## 🙌 Credits & Tools Used

- FAISS — Dense vector search
- RankBM25 — Sparse keyword search
- NetworkX — Graph operations
- LangChain — LLM orchestration
- Ollama — Local LLM serving
- SentenceTransformers — Embeddings
- HuggingFace Transformers — SciBERT for entity triplets

---

Built for deep, document-grounded academic question answering.  
_Query smarter, not harder_ 🤖📘✨

# 🧠 Retrieval-Augmented Generation (RAG) Chatbot for Academic PDFs

This project is a fully modular, class-based **RAG (Retrieval-Augmented Generation) system** that allows you to query a collection of academic **PDFs** and get contextual answers using **LLaMA 3.2 via Ollama**.

It integrates:

- Document preprocessing and intelligent **chunking**
- **Dense vector search (FAISS)** for semantic retrieval
- Optional **sparse keyword search (BM25)** for exact match queries
- A **Streamlit chatbot** interface powered by LLMs

---

## 🚦 End-to-End Workflow Overview

```mermaid
graph TD;
    A[PDFs] --> B[Chunking: chunking.py];
    B --> C[Vector Embeddings: embedding.py];
    B --> D[BM25 Indexing (optional): bm25.py];
    C --> E[FAISS Index];
    D --> E2[BM25 Index];
    E --> F[Chatbot Query: RAG.py];
    E2 --> F;
    F --> G[Answer via LLaMA 3.2 (Ollama)];
```

---

## 📂 Folder Structure & Key Files

| File            | Purpose                                          |
| --------------- | ------------------------------------------------ |
| `chunking.py`   | Extract and intelligently chunk PDF text         |
| `embedding.py`  | Create vector embeddings using MiniLM            |
| `bm25.py`       | Create a sparse keyword search index             |
| `RAG.py`        | RAG chatbot logic with retriever + LLM           |
| `config.py`     | Centralized config for paths and model names     |
| `main.py`       | Unified CLI for preprocessing and chatbot launch |
| `vector_store/` | Stores FAISS + BM25 indexes and metadata         |

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

#### ➤ Run FAISS-only pipeline:

```bash
python main.py --mode preprocess
```

#### ➤ Run FAISS + BM25 indexing:

```bash
python main.py --mode preprocess --use_bm25
```

---

### 💬 3. Launch the Chatbot

You can choose **which retrieval mode to use**:

| Mode              | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| `faiss` (default) | Uses dense semantic retrieval — best for natural language queries   |
| `bm25`            | Uses keyword-based search — good for exact matches or rare terms    |
| `hybrid`          | Combines both FAISS and BM25 — best for balanced recall & precision |

#### ➤ Launch (default = FAISS):

```bash
streamlit run main.py -- --mode chatbot
```

#### ➤ Use BM25 only:

```bash
streamlit run main.py -- --mode chatbot --retriever bm25
```

#### ➤ Use hybrid (FAISS + BM25):

```bash
streamlit run main.py -- --mode chatbot --retriever hybrid
```

---

## 🧠 How Retrieval Works

### ✅ FAISS (Dense Semantic Retrieval)

- Uses `all-MiniLM-L6-v2` via `sentence-transformers`
- Captures meaning beyond keywords
- Great for paraphrased or long-form questions
- Leverages GPU if available

### ✅ BM25 (Sparse Keyword Retrieval)

- Based on token overlap + frequency
- Great for matching **exact phrases**, **chemical names**, or **IDs**
- Especially helpful for niche scientific jargon

### ✅ Hybrid

- Retrieves top-k from both BM25 + FAISS
- De-duplicates and merges
- Ensures **strong recall** and **semantic depth**

---

## 🔧 Configuration

All file paths and model names are stored in:

```python
config.py
```

You can change:

- `RAW_PDF_DIR` — path to your academic PDFs
- `LLM_MODEL_NAME` — Ollama model to run (e.g. `"llama3.2:latest"`)
- `EMBEDDING_MODEL_NAME` — transformer model for embeddings

---

## 🔍 Benefits of the System

- ⚙️ Fully modular (chunking, embedding, BM25, chatbot)
- 🧠 LLM-powered contextual answers
- 🔌 Easily swappable retrievers (FAISS, BM25, Hybrid)
- 🚀 GPU-compatible for faster embeddings
- 🖼️ Interactive UI with Streamlit
- 🔄 Automatic Ollama startup if not already running
- 🔓 Designed to scale and integrate future enhancements (e.g. GraphRAG)

---

## 🙌 Credits & Tools Used

- FAISS — Dense vector search
- RankBM25 — Sparse keyword search
- LangChain — LLM orchestration
- Ollama — Local LLM serving
- SentenceTransformers — Embeddings

---

Built for deep, document-grounded academic question answering.  
_Query smarter, not harder_ 🤖📘✨

# 🧠 Retrieval-Augmented Generation (RAG) Chatbot for Research Papers

This project implements a modular **Retrieval-Augmented Generation (RAG) chatbot** designed to help researchers query large collections of academic PDFs. It uses **FAISS** for vector similarity search and **LLaMA 3.2 (via Ollama)** to generate contextual answers from relevant research papers.

---

## 📐 Project Architecture

```text
📂 data/
├── raw_texts/ (or G:/AcademicRAG/Subdataset/)
├── chunked_texts.json
├── vector_store/
│   ├── faiss_index.index
│   └── metadata.json

🧠 Components:
- 📄 chunking.py → PDF text extraction + semantic chunking
- 📈 embedding.py → Sentence embeddings + FAISS vector index
- 🤖 RAG.py → Query interface (Streamlit) + LLM (Llama3.2 via Ollama)
- 🧩 config.py → Central configuration for paths and models
- 🚀 main.py → CLI entrypoint to run pipeline or chatbot
```

---

## 🔄 Workflow Overview

### 1️⃣ `chunking.py` — PDF Parsing & Semantic Chunking

- Extracts full text from PDFs using `PyMuPDF`.
- Performs hierarchical + semantic chunking.
- Stores the result in `chunked_texts.json`.

### 2️⃣ `embedding.py` — Vector Embeddings + FAISS

- Reads chunked data from JSON.
- Generates dense vectors using `sentence-transformers` (MiniLM).
- Saves vectors in a FAISS index (`vector_store/faiss_index.index`) and stores mapping info in `metadata.json`.

### 3️⃣ `RAG.py` — Streamlit Chatbot with LLaMA 3.2

- Loads FAISS and metadata to retrieve relevant chunks based on query.
- Uses `langchain + Ollama` to query LLaMA 3.2 for concise answers.
- Includes a Streamlit UI that shows:
  - User questions
  - Retrieved chunks
  - Source PDFs
  - AI-generated answers

### 4️⃣ `config.py` — Centralized Configuration

- Contains:
  - All directory/file paths
  - Embedding model and LLM names
- Used in every component to keep things modular and easy to maintain.

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess PDFs (Extract, Chunk, Embed)

```bash
python main.py --mode preprocess
```

### 3. Launch the Chatbot

```bash
streamlit run main.py -- --mode chatbot
```

> 💡 Use `--` before mode when running with `streamlit`.

---

## ✅ Features

- Modular structure with class-based architecture.
- FAISS-based dense retrieval.
- LLM-based answer generation using Ollama + LangChain.
- Configurable file paths, models, and directories.
- GPU-compatible out of the box (MiniLM uses GPU if available).

---

## 📝 Summary

| File           | Responsibility                              |
| -------------- | ------------------------------------------- |
| `chunking.py`  | Extract and chunk PDF texts                 |
| `embedding.py` | Generate embeddings, store in FAISS         |
| `RAG.py`       | Retrieve relevant chunks + generate answers |
| `config.py`    | Centralized config (paths + models)         |
| `main.py`      | Unified CLI to run full pipeline or chatbot |

---

## 🙌 Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com)
- [LangChain](https://www.langchain.com/)

---

Enjoy querying your academic corpus like a pro researcher! 🧪📚🤖

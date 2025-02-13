# **AcademicRAG 🧠📚**

AcademicRAG is a Retrieval-Augmented Generation (RAG) system designed to assist researchers by extracting, retrieving, and summarizing insights from large collections of academic papers. The goal is to experiment with multiple retrieval mechanisms and optimize them for enhanced accuracy and efficiency.

---

## **📌 Project Overview**

This repository explores different RAG implementations, including:

- **Vector Store-based Retrieval** (`withVectorStore_Implementation/`)
- **Local Inference with Ollama** (`Ollama_Implementation/`)
- **Hybrid and Advanced AI Methods** (Planned)

Each implementation uses a shared `.env` configuration and dependencies.

---

## **📂 Folder Structure**

- `Ollama_Implementation/` → Runs a local LLM for retrieval.
- `withVectorStore_Implementation/` → Uses Astra DB for vector-based document retrieval.
- `.env` → Stores API keys and credentials.
- `.gitignore` → Ensures sensitive files are not committed.
- `file-extraction.py` → Extracts and processes research papers.
- `LICENSE` → Open-source license details.
- `README.md` → This document.

---

## **🚀 Getting Started**

### **1️⃣ Setup the Environment**

Ensure you have Python 3.10+ installed. Create and activate a virtual environment:

```bash
conda create -p venv python==3.10
conda activate venv/
```

### **2️⃣ Install Dependencies**

```bash
pip install -r <implementation_folder>/requirements.txt
```

### **3️⃣ Configure API Keys**

Set up environment variables in `.env`:

```ini
ASTRA_DB_APPLICATION_TOKEN=<your_astra_db_token>
OPENAI_API_KEY=<your_openai_api_key>
GROQ_API_KEY=<your_groq_api_key>
```

---

## **📌 Implementations**

| Implementation              | Description                             | Status         |
| --------------------------- | --------------------------------------- | -------------- |
| **Vector Store (Astra DB)** | Uses embeddings and vector search.      | ✅ Implemented |
| **Ollama-Based Local RAG**  | Runs a local LLM for retrieval.         | 🚧 In Progress |
| **Hybrid Search**           | Combines keyword & semantic search.     | 🛠 Planned      |
| **Agentic AI Reasoning**    | Adds multi-step reasoning to retrieval. | 🛠 Planned      |

---

## **🔬 Future Enhancements**

- Optimize retrieval mechanisms for academic datasets.
- Evaluate different embedding models.
- Develop a unified UI for seamless switching between implementations.

💡 **This project is evolving—contributions and suggestions are welcome!** 🚀

---

# Retrieval-Augmented Generation (RAG) Chatbot for Research Papers

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to query research papers efficiently. It extracts text from **PDFs**, processes them into **chunks**, generates **vector embeddings**, stores them in a **FAISS index**, and retrieves relevant information using **Llama 3.2 (via Ollama)** for answering research-related questions.

---

## **Project Workflow**

### **1️⃣ chunking.py (PDF Processing & Text Chunking)**

**Purpose:** Extracts text from **PDF research papers** and breaks it into **smaller chunks** for efficient retrieval.

**Logic Flow:**

- **Load PDFs**: Extracts text using `pdfplumber`.
- **Chunking the Text**: Splits text into **semantic chunks**, ensuring sentences remain intact.
- **Metadata Storage**: Saves each chunk with metadata (filename, title, DOI, etc.) in `chunked_texts.json`.

---

### **2️⃣ embedding.py (Generating Vector Embeddings & Storing in FAISS)**

**Purpose:** Converts text chunks into **vector embeddings** and stores them in a **FAISS index** for fast retrieval.

**Logic Flow:**

- **Load Chunked Data**: Reads `chunked_texts.json`.
- **Convert Text to Embeddings**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to generate **dense vector embeddings**.
- **Build FAISS Index**: Stores embeddings in FAISS and saves them in `faiss_index.index`.
- **Save Metadata**: Maps embeddings to their original text in `metadata.json`.

---

### **3️⃣ RAG.py (Retrieval-Augmented Generation Chatbot)**

**Purpose:** Implements a **RAG-based chatbot** that retrieves relevant research text chunks from FAISS and **generates answers using Llama 3.2**.

**Logic Flow:**

- **Load FAISS Index & Metadata**: Reads `faiss_index.index` and `metadata.json`.
- **User Query Processing**: Accepts a research question from the user via **Streamlit UI**.
- **Retrieve Relevant Chunks**: Embeds the user query, searches FAISS, and fetches **top-k relevant chunks**.
- **Generate AI Answer**: Uses **Llama 3.2 (via Ollama)** to generate a concise answer with references.
- **Streamlit UI**: Displays retrieved chunks, source PDFs, and AI-generated responses interactively.

---

## **How to Run**

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Chunking Script:**

   ```bash
   python chunking.py
   ```

3. **Run Embedding Script:**

   ```bash
   python embedding.py
   ```

4. **Start the RAG Chatbot:**
   ```bash
   streamlit run RAG.py
   ```

---

## **Summary**

- **chunking.py** → Extracts text from PDFs and creates small **semantic chunks**.
- **embedding.py** → Converts chunks into **vector embeddings** and stores them in **FAISS**.
- **RAG.py** → Implements the **RAG chatbot**, retrieving relevant chunks & generating **AI-powered answers**.

This system provides a **fast and accurate research assistant** to help users query thousands of academic papers efficiently. 🚀

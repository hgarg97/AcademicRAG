# Academic RAG - Ollama Implementation

## Overview

The **Academic RAG (Retrieval-Augmented Generation)** system is a research assistant powered by **Ollama**, designed to help researchers efficiently extract relevant insights from uploaded **PDF research papers**. This system processes academic documents, indexes their contents into an **in-memory vector store**, and enables users to query information interactively. It provides factual, context-aware responses using **large language models (LLMs)**.

### Key Features:

- **PDF Upload & Processing**: Extracts and segments content from uploaded research PDFs.
- **Vector Search for Context Retrieval**: Utilizes **Ollama embeddings** to store and retrieve relevant document sections.
- **LLM-powered Responses**: Uses **Llama 3.2** to generate concise, research-focused answers.
- **Streamlit UI**: A simple, interactive web interface for seamless user interaction.

---

## Setup & Installation

### 1️⃣ Prerequisites

Ensure your system has:

- **Python 3.9+** installed
- **Ollama** installed for running LLM models

### 2️⃣ Install Ollama

Ollama is required to run **Llama3.2** and **nomic-embed-text** models locally.

#### Install Ollama:

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

For Windows users, download and install Ollama from [here](https://ollama.com/download).

### 3️⃣ Install Required Python Packages

Navigate to the **Ollama_Implementation** folder and run:

```sh
pip install -r requirements.txt
```

This will install the necessary dependencies:

- **streamlit** (for UI)
- **langchain** components (for document processing and retrieval)
- **pdfplumber** (for PDF parsing)
- **ollama** LLM and embedding models

### 4️⃣ Download and Run Models

#### Pull the Required Models in Ollama:

```sh
ollama pull nomic-embed-text:latest
ollama pull llama3.2:latest
```

This ensures the embedding and LLM models are available for use.

---

## Running the Application

1️⃣ Navigate to the **Ollama_Implementation** folder:

```sh
cd path/to/Ollama_Implementation
```

2️⃣ Start the Streamlit application:

```sh
streamlit run UploadPDF_RAG.py
```

3️⃣ Open the provided **localhost URL** in your browser and begin uploading PDFs to query academic insights.

---

## How It Works

1. **Upload a research paper (PDF).**
2. The document is **chunked and stored** using **Ollama embeddings**.
3. When a user asks a question, the system retrieves **relevant document sections** using **vector similarity search**.
4. The retrieved text is passed into **Llama 3.2**, which generates an accurate and concise answer.
5. The response is displayed in an interactive chat format.

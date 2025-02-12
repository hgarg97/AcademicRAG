# **AcademicRAG 🧠📚**

**AcademicRAG** is a Retrieval-Augmented Generation (RAG)-based chatbot designed to query and extract information from a vast collection of academic research papers. This project leverages **LangChain**, **Astra DB (Serverless Cassandra with Vector Search)**, **OpenAI LLMs**, and **Groq LLAMA 3.3** to build an intelligent research assistant. Additionally, we will be exploring **Agentic AI** methodologies to enhance response accuracy and contextual understanding.

---

## **🗂 Project Structure**

The repository is structured as follows:

- **Main Directory:** Contains shared resources and configurations.
  - `.env` → Stores environment variables like API keys.
  - `requirements.txt` → Lists Python dependencies.
- **Datasets/** → Contains research papers (excluded from Git).
- **Project Codebase:**
  - `create_vector_store.py` → Extracts text from PDFs, splits content, and stores embeddings in Astra DB.
  - `app.py` → Streamlit chatbot interface using LLAMA 3.3 and Astra DB vector retrieval.
  - `file-extraction.py` → Moves all files from subdirectories to the main folder.

---

## **🤝 Initial Setup**

### **1️⃣ Installing Anaconda and Setting Up the Environment**

To set up the Python environment, use Anaconda:

```bash
conda create -p venv python==3.10
```

- **Note**: `-p` creates the environment in the current directory.

Activate the environment:

```bash
conda activate venv/
```

or

```bash
conda activate /path/to/venv
```

---

### **2️⃣ Installing Dependencies**

Ensure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

#### **`requirements.txt` Includes:**

```txt
cassio
datasets
langchain
openai
tiktoken
streamlit
pymupdf
huggingface_hub
sentence-transformers
```

---

### **3️⃣ Setting Up Environment Variables**

#### **Option 1: Using a `.env` File**

Create a `.env` file and add:

```ini
ASTRA_DB_APPLICATION_TOKEN=<your_astra_db_token>
ASTRA_DB_ID=<your_astra_db_id>
OPENAI_API_KEY=<your_openai_api_key>
GROQ_API_KEY=<your_groq_api_key>
```

#### **Option 2: Setting Temporary Environment Variables**

For **Mac/Linux**:

```bash
export ASTRA_DB_APPLICATION_TOKEN="your_astra_db_token"
export ASTRA_DB_ID="your_astra_db_id"
export OPENAI_API_KEY="your_openai_api_key"
export GROQ_API_KEY="your_groq_api_key"
```

For **Windows** (Command Prompt):

```bash
setx ASTRA_DB_APPLICATION_TOKEN "your_astra_db_token"
setx ASTRA_DB_ID "your_astra_db_id"
setx OPENAI_API_KEY "your_openai_api_key"
setx GROQ_API_KEY "your_groq_api_key"
```

For **persistent storage** in Conda:

```bash
conda env config vars set ASTRA_DB_APPLICATION_TOKEN="your_astra_db_token"
conda env config vars set ASTRA_DB_ID="your_astra_db_id"
conda env config vars set OPENAI_API_KEY="your_openai_api_key"
conda env config vars set GROQ_API_KEY="your_groq_api_key"
conda env config vars list
```

---

## **🚀 Running the AI Bot**

### **1️⃣ Extracting and Storing Research Papers in AstraDB**

Run the script to extract text from PDFs and store embeddings:

```bash
python create_vector_store.py
```

### **2️⃣ Running the Streamlit Chatbot**

Navigate to the project directory and execute:

```bash
streamlit run app.py
```

---

## **🔬 Workflow Overview**

1️⃣ **Extract text from PDFs using `fitz` (PyMuPDF).**  
2️⃣ **Split the text into smaller chunks (800 tokens with 200 overlap) for efficient vector search.**  
3️⃣ **Embed text using HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`).**  
4️⃣ **Store embeddings in AstraDB (Serverless Cassandra with vector search).**  
5️⃣ **Retrieve top 4 relevant documents based on similarity search.**  
6️⃣ **Use LLAMA 3.3 via Groq API to generate responses.**  
7️⃣ **Display responses in the Streamlit UI.**

---

## **📌 Future Enhancements**

- 📈 Fine-tune retrieval for better accuracy.
- 🛠️ Experiment with **Agentic AI** for improved contextual reasoning.
- 📺 Enhance UI with a more interactive research chatbot experience.

---

💪 _Built with love for academic research!_

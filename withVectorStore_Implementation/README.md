# **AcademicRAG - Vector Store Implementation 📚🔍**

This implementation of **AcademicRAG** leverages **Astra DB (Serverless Cassandra with Vector Search)** and **LLMs (OpenAI/Groq LLAMA 3.3)** to create an intelligent research assistant.

---

## **🗂 Folder Structure**

- `create_vector_store.py` → Extracts text, generates embeddings, and stores in Astra DB.
- `app.py` → Streamlit chatbot retrieving documents via Astra DB.
- `requirements.txt` → Dependencies for this implementation.

---

## **🤝 Setup Instructions**

### **1️⃣ Create a Virtual Environment**

```bash
conda create -p venv python==3.10
conda activate venv/
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Configure Environment Variables**

Create a `.env` file with:

```ini
ASTRA_DB_APPLICATION_TOKEN=<your_astra_db_token>
ASTRA_DB_ID=<your_astra_db_id>
OPENAI_API_KEY=<your_openai_api_key>
GROQ_API_KEY=<your_groq_api_key>
```

Or set variables manually:

```bash
export ASTRA_DB_APPLICATION_TOKEN="your_astra_db_token"
export OPENAI_API_KEY="your_openai_api_key"
```

---

## **🚀 Running the Chatbot**

### **1️⃣ Store Research Paper Embeddings**

```bash
python create_vector_store.py
```

### **2️⃣ Launch the Chatbot**

```bash
streamlit run app.py
```

---

## **🔬 Workflow Overview**

1️⃣ **Extract text from PDFs using PyMuPDF.**  
2️⃣ **Split text into chunks (800 tokens, 200 overlap) for retrieval.**  
3️⃣ **Generate embeddings using `all-MiniLM-L6-v2`.**  
4️⃣ **Store embeddings in AstraDB for vector-based search.**  
5️⃣ **Retrieve top 4 relevant documents via similarity search.**  
6️⃣ **Generate responses using LLAMA 3.3 (Groq API).**  
7️⃣ **Display responses in a Streamlit chatbot.**

---

## **📌 Future Enhancements**

- Experiment with better chunking methods.
- Optimize retrieval efficiency.
- Compare different embedding models.

💡 **Part of a broader RAG experimentation—stay tuned for more implementations!** 🚀

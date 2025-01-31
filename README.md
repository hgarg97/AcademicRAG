Here’s your updated **README** with the initial setup details included:

---

# **Animal Science AI Bot 🐮🤖**

**Animal Science AI Bot** is a Retrieval-Augmented Generation (RAG)-based chatbot designed to query and extract information from a vast collection of animal science research papers. This project leverages **LangChain**, **Astra DB (Serverless Cassandra with Vector Search)**, and **OpenAI LLMs** to build an intelligent research assistant. Additionally, we will be exploring **Agentic AI** methodologies to enhance response accuracy and contextual understanding.

---

## **📂 Project Structure**

The repository is structured as follows:

- **Main Directory:** Contains shared resources and configurations.
  - `.env` → Stores environment variables like API keys.
  - `requirements.txt` → Lists Python dependencies.
- **Datasets/** → Contains research papers (excluded from Git).
- **Project Codebase:** Includes scripts for data ingestion, vector search, and chatbot interaction.

---

## **🛠 Initial Setup**

### **1️⃣ Installing Anaconda and Setting Up the Environment**

To set up the Python environment, use Anaconda:

```bash
conda create -p venv python==3.12
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
```

---

### **3️⃣ Setting Up Environment Variables**

#### **Option 1: Using a `.env` File**

Create a `.env` file and add:

```ini
ASTRA_DB_APPLICATION_TOKEN=<your_astra_db_token>
ASTRA_DB_ID=<your_astra_db_id>
OPENAI_API_KEY=<your_openai_api_key>
```

#### **Option 2: Setting Temporary Environment Variables**

For **Mac/Linux**:

```bash
export ASTRA_DB_APPLICATION_TOKEN="your_astra_db_token"
export ASTRA_DB_ID="your_astra_db_id"
export OPENAI_API_KEY="your_openai_api_key"
```

For **Windows** (Command Prompt):

```bash
setx ASTRA_DB_APPLICATION_TOKEN "your_astra_db_token"
setx ASTRA_DB_ID "your_astra_db_id"
setx OPENAI_API_KEY "your_openai_api_key"
```

For **persistent storage** in Conda:

```bash
conda env config vars set ASTRA_DB_APPLICATION_TOKEN="your_astra_db_token"
conda env config vars set ASTRA_DB_ID="your_astra_db_id"
conda env config vars set OPENAI_API_KEY="your_openai_api_key"
conda env config vars list
```

---

## **🚀 Running the AI Bot**

Navigate to the project directory and execute:

```bash
python main.py
```

Or, for interactive testing:

```bash
python playground.py
```

---

## **🔬 Workflow Overview**

1️⃣ **Import dependencies**  
2️⃣ **Load and preprocess research data** (vector embeddings, chunking)  
3️⃣ **Initialize the LangChain vector store with Astra DB**  
4️⃣ **Run a Question-Answering loop** (retrieve relevant research snippets)  
5️⃣ **Experiment with Agentic AI techniques**

---

## **📌 Future Enhancements**

🔹 Fine-tune retrieval for better accuracy.  
🔹 Evaluate **Agentic AI** for dynamic interactions.  
🔹 Improve UI for research-based chatbot interaction.

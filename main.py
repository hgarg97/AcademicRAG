from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from groq import Groq

# With CassIO, the engine powering the Astra DB integration in LangChain
import cassio
import fitz

import os


from dotenv import load_dotenv
load_dotenv()

# Read values from environment
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Define the directory containing PDFs
pdf_directory = "Subdataset/"

# List all PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

# Read and concatenate text from all PDFs
raw_text = ''

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    doc = fitz.open(pdf_path)

    print(f"\n--- Reading: {pdf_file} ---\n")

    # Extract text from each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        content = page.get_text("text")  # Extract text in text format
        if content:
            raw_text += content + "\n"  # Append with a newline for readability


cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# llm = OpenAI(openai_api_key=OPENAI_API_KEY)
# embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  


astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="subset_data",
    session=None,
    keyspace=None,
)

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


astra_vector_store.add_texts(texts)

print("Inserted %i headlines." % len(texts))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)


# Initialize Groq API client
groq_client = Groq(api_key=GROQ_API_KEY)

# Streamlit UI
st.set_page_config(page_title="Animal Science AI Chatbot", layout="wide")
st.title("💬 Animal Science AI Chatbot Powered by Groq API")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
query = st.chat_input("Ask me anything about the research papers documents...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    # Query AstraDB vector index
    answer = astra_vector_index.query(query).strip()

    # Retrieve top 4 relevant documents with scores
    relevant_docs = astra_vector_store.similarity_search_with_score(query, k=4)
    context = "\n".join([f"[{score:.4f}] {doc.page_content[:100]}..." for doc, score in relevant_docs])

    # Format prompt for Groq API
    prompt = f"You are an expert with the knowledge of Animal Science and various research in the field, Refer to the Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Call Groq API
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )

    # Get LLM response
    llm_answer = response.choices[0].message.content

    # Display response
    with st.chat_message("assistant"):
        st.markdown(f"**Answer:** {llm_answer}")

    # Display relevant documents
    with st.expander("🔍 Relevant Documents"):
        st.write(context)

    # Save to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": llm_answer})

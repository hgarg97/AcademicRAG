import os
import streamlit as st
from langchain_community.vectorstores import Cassandra
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import cassio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Cassandra session
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="animal_science_sample_db",
    session=None,
    keyspace=None,
)

# Initialize LLM (Groq - LLAMA 3.3)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# Streamlit UI
st.title("🐄 Animal Science AI Chatbot")

# Chat input
query = st.chat_input("Ask me anything about animal science...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve top 4 relevant documents
    relevant_docs = astra_vector_store.similarity_search_with_score(query, k=4)
    context = "\n".join([f"[{score:.4f}] {doc.page_content[:300]}..." for doc, score in relevant_docs])

    # Format the prompt
    prompt = f'''You are an expert in the field of Animal Science and Agriculture.
    Using the given Context:\n{context}\n\
        Answer the following query in a very to the point and professional manner in detail and giving all the facts and references 
        \n{query}'''

    # Get response from Groq
    response = llm.invoke(prompt)

    response_text = response if isinstance(response, str) else response.content


    # Display answer
    with st.chat_message("assistant"):
        st.markdown(f"**Answer:** {response_text}")

    # Show retrieved documents
    with st.expander("🔍 Relevant Documents"):
        st.write(context)
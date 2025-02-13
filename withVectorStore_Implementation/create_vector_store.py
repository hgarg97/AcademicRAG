import os
from langchain_community.vectorstores import AstraDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import cassio
import fitz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the directory containing PDFs
pdf_directory = "../Subdataset/"

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

# Set API keys
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="animal_science_sample_db",
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

print(f"✅ {len(texts)} document chunks added to AstraDB.")
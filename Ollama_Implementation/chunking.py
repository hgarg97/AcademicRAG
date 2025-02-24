import fitz  # PyMuPDF for PDF text extraction
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import requests
from nltk.tokenize import PunktTokenizer

# Initialize Punkt Tokenizer
punkt_tokenizer  = PunktTokenizer()

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# If we need nomic embeddings by OLLAMA
def get_nomic_embeddings(text):
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "nomic-embed-text",
        "prompt": text
    }
    response = requests.post(url, json=payload)
    return response.json()

# If we need to simple chunking
def simple_chunking(text, chunk_size=512, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

# Function to split text hierarchically
def hierarchical_chunking(text):
    # 1️⃣ Split by Headings (Sections) - More robust regex
    section_pattern = re.compile(r'(?m)^(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References)\b.*$', re.IGNORECASE)
    sections = re.split(section_pattern, text)

    structured_chunks = []
    
    for section in sections:
        # 2️⃣ Split into Paragraphs
        paragraphs = [para.strip() for para in section.split("\n\n") if para.strip()]
        
        for para in paragraphs:
            # 3️⃣ Split into Sentences using Punkt Tokenizer
            sentences = punkt_tokenizer.tokenize(para)
            structured_chunks.extend(sentences)

    return structured_chunks

# Function to merge sentences based on semantic similarity
def semantic_chunking(sentences, similarity_threshold=0.75, max_tokens=512):
    chunks = []
    current_chunk = []
    token_count = 0

    embeddings = model.encode(sentences, convert_to_numpy=True)

    for i in range(len(sentences)):
        token_count += len(sentences[i].split())
        current_chunk.append(sentences[i])

        if i < len(sentences) - 1:
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            if sim < similarity_threshold or token_count >= max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                token_count = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Example usage
pdf_text = extract_text_from_pdf("G:/AcademicRAG/Subdataset/143-Kofler-2023.pdf")
sentences = hierarchical_chunking(pdf_text)
final_chunks = semantic_chunking(sentences, max_tokens= 1024)

# Save chunks for embedding
with open("chunked_texts.txt", "w", encoding="utf-8") as f:
    for chunk in final_chunks:
        f.write(chunk + "\n\n")

print(f"✅ Chunking completed! {len(final_chunks)} chunks created.")

import fitz  # PyMuPDF for PDF text extraction
import os
import re
import json
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Punkt Sentence Tokenizer
punkt_tokenizer = PunktSentenceTokenizer()

# JSON file path
OUTPUT_JSON = "chunked_texts.json"

# DOI regex pattern
DOI_PATTERN = r"10\.\d{4,9}/\S*[^.\s]"

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Function to extract DOI from text
def extract_doi(text):
    match = re.search(DOI_PATTERN, text)
    return match.group() if match else None

# Function to extract paper title (assuming title is near the beginning)
def extract_title(text):
    lines = text.split("\n")
    for line in lines[:20]:  # Check first 20 lines for a title
        if len(line.strip()) > 10 and not line.lower().startswith(("abstract", "introduction")):
            return line.strip()
    return "Unknown Title"

# Function to extract paper title (assuming title is in the DOI website)
def get_title_from_doi(doi):
    url = f"https://api.crossref.org/works/{doi}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad responses
        data = response.json()
        return data["message"]["title"][0] if "message" in data and "title" in data["message"] else None
    except Exception as e:
        print(f"❌ Error fetching title for DOI {doi}: {e}")
        return None

# Function to split text hierarchically
def hierarchical_chunking(text):
    # 1️⃣ Split by Headings (Sections)
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
def semantic_chunking(sentences, similarity_threshold=0.75, max_tokens=1024):
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

# Function to process a single PDF
def process_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Extract metadata
    doi = extract_doi(pdf_text)
    doi_url = f"https://doi.org/{doi}" if doi else None
    
    # Fetch title from DOI
    title = get_title_from_doi(doi) if doi else "Unknown Title"

    # Chunking process
    sentences = hierarchical_chunking(pdf_text)
    final_chunks = semantic_chunking(sentences)

    # Add metadata for each chunk
    chunked_data = [{
        "file_name": os.path.basename(pdf_path),
        "paper": title,
        "doi": doi_url,
        "chunk": chunk
    } for chunk in final_chunks]
    
    return chunked_data

# Function to process new PDFs only and update JSON
def process_new_pdfs(folder_path, output_json=OUTPUT_JSON):
    # Load existing data
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        all_chunks = []
    # Get list of already processed papers
    processed_papers = set(chunk["file_name"] for chunk in all_chunks)
    # Get new PDFs
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    new_pdfs = [pdf for pdf in pdf_files if pdf not in processed_papers]

    if not new_pdfs:
        print("✅ No new PDFs to process. Everything is up-to-date.")
        return

    print(f"📂 Found {len(new_pdfs)} new PDFs to process.")

    for pdf in new_pdfs:
        pdf_path = os.path.join(folder_path, pdf)
        print(f"📄 Processing: {pdf}")
        pdf_chunks = process_pdf(pdf_path)
        all_chunks.extend(pdf_chunks)
    # Save updated JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print(f"✅ Processing completed! {len(new_pdfs)} new PDFs added.")

# Usage
folder_path = "G:/AcademicRAG/Subdataset/"
process_new_pdfs(folder_path)

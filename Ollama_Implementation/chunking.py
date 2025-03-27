import os
import re
import json
import fitz
import numpy as np
import requests
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import config

class PDFChunker:
    def __init__(self, output_json=config.CHUNKED_JSON_PATH):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.tokenizer = PunktSentenceTokenizer()
        self.output_json = output_json
        self.doi_pattern = r"10\.\d{4,9}/\S*[^.\s]"

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()

    def extract_doi(self, text):
        match = re.search(self.doi_pattern, text)
        return match.group() if match else None

    def extract_title(self, text):
        lines = text.split("\n")
        for line in lines[:20]:
            if len(line.strip()) > 10 and not line.lower().startswith(("abstract", "introduction")):
                return line.strip()
        return "Unknown Title"

    def get_title_from_doi(self, doi):
        url = f"https://api.crossref.org/works/{doi}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data["message"]["title"][0] if "message" in data and "title" in data["message"] else None
        except Exception as e:
            print(f"❌ Error fetching title for DOI {doi}: {e}")
            return None

    def hierarchical_chunking(self, text):
        
        # 1️⃣ Split by Headings (Sections)
        section_pattern = re.compile(r'(?m)^(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References)\b.*$', re.IGNORECASE)
        sections = re.split(section_pattern, text)
        structured_chunks = []
        for section in sections:
            # 2️⃣ Split into Paragraphs
            paragraphs = [para.strip() for para in section.split("\n\n") if para.strip()]
            for para in paragraphs:
                # 3️⃣ Split into Sentences using Punkt Tokenizer
                sentences = self.tokenizer.tokenize(para)
                structured_chunks.extend(sentences)
        return structured_chunks

    def semantic_chunking(self, sentences, similarity_threshold=0.75, max_tokens=1024):
        chunks = []
        current_chunk = []
        token_count = 0
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
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

    def process_pdf(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        # Extract metadata
        doi = self.extract_doi(text)
        title = self.get_title_from_doi(doi) if doi else "Unknown Title"
        sentences = self.hierarchical_chunking(text)
        final_chunks = self.semantic_chunking(sentences)
        return [{"file_name": os.path.basename(pdf_path), "paper": title, "doi": f"https://doi.org/{doi}" if doi else None, "chunk": chunk} for chunk in final_chunks]

    def process_new_pdfs(self, folder_path):
        if os.path.exists(self.output_json):
            with open(self.output_json, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
        else:
            all_chunks = []
        processed_files = set(chunk["file_name"] for chunk in all_chunks)
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        new_pdfs = [f for f in pdf_files if f not in processed_files]
        if not new_pdfs:
            print("✅ No new PDFs to process. Everything is up-to-date.")
            return
        for pdf in new_pdfs:
            print(f"📄 Processing: {pdf}")
            chunks = self.process_pdf(os.path.join(folder_path, pdf))
            all_chunks.extend(chunks)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=4, ensure_ascii=False)
        print(f"✅ Added {len(new_pdfs)} new PDFs.")

import os
import json
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import config

class BM25Retriever:
    def __init__(self, chunked_path=config.CHUNKED_JSON_PATH, bm25_path=config.BM25_INDEX_PATH):
        self.chunked_path = chunked_path
        self.bm25_path = bm25_path
        self.tokenized_corpus = []
        self.chunks = []
        self.bm25 = None

    def build_index(self):
        if not os.path.exists(self.chunked_path):
            print("❌ No chunked_texts.json found. Run chunking first.")
            return

        with open(self.chunked_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.tokenized_corpus = [word_tokenize(chunk["chunk"].lower()) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

        print(f"✅ BM25 index built and saved to {self.bm25_path}")

    def load_index(self):
        if not os.path.exists(self.bm25_path):
            print("❌ BM25 index not found. Run build_index() first.")
            return False

        with open(self.bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        # 🔁 Load chunks even if preprocessing skipped chunking
        if os.path.exists(self.chunked_path):
            with open(self.chunked_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            print("❌ chunked_texts.json not found!")
            return False

        return True

    def search(self, query, top_k=5):
        if not self.bm25:
            if not self.load_index():
                return []
        if not self.chunks:
            print("⚠️ Warning: No chunks loaded for BM25.")
            return []

        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_indices]

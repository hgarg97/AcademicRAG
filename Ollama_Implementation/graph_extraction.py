# graph_extraction.py
import json
import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

CHUNKED_JSON = "chunked_texts.json"

class TripletExtractor:
    def __init__(self, model_name="allenai/scibert_scivocab_cased"):
        print(f"📦 Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def extract_entities(self, sentence):
        # Simple capitalized noun phrase matcher
        return re.findall(r'\b[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*', sentence)

    def build_triplets(self, sentence):
        entities = self.extract_entities(sentence)
        triplets = []
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                triplets.append((entities[i], "related_to", entities[j], sentence))
        return triplets

    def run(self, chunked_path=CHUNKED_JSON, output_path="graph_triplets.json"):
        print("🔍 Reading chunks...")
        with open(chunked_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        all_triplets = []
        for chunk in chunks:
            text = chunk["chunk"]
            triplets = self.build_triplets(text)
            for subj, rel, obj, sent in triplets:
                all_triplets.append({
                    "subject": subj,
                    "relation": rel,
                    "object": obj,
                    "sentence": sent,
                    "paper": chunk.get("paper"),
                    "file_name": chunk.get("file_name"),
                    "doi": chunk.get("doi")
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_triplets, f, indent=2)
        print(f"✅ Extracted {len(all_triplets)} triplets → Saved to {output_path}")


if __name__ == "__main__":
    extractor = TripletExtractor()
    extractor.run()

import re
import json
from transformers import pipeline
import config
import torch

class TripletExtractor:
    def __init__(self, model_name=config.TRIPLET_MODEL_NAME, ner_model_name=config.NER_MODEL_NAME):
        self.model_name = model_name
        self.ner_model_name = ner_model_name
        device = 0 if torch.cuda.is_available() else -1
        self.ner_pipeline = pipeline("ner", model=ner_model_name, tokenizer=ner_model_name, aggregation_strategy="simple", device=device)

    def extract_entities(self, sentence):
        """
        Transformer-based NER using biomedical-ner-all.
        You can switch to `regex_entities()` for simpler fallback.
        """
        entities = self.ner_pipeline(sentence)
        return list({ent['word'] for ent in entities if ent['score'] > 0.7})

    def regex_entities(self, sentence):
        """
        Simple regex for capitalized noun phrases.
        """
        return re.findall(r'\b[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*', sentence)

    def extract_triplets(self, sentence):
        # Placeholder for actual scientific triplet extraction logic.
        return []

    def run(self, chunked_path, output_path):
        with open(chunked_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        triplets = []
        for item in chunks:
            sentence = item["chunk"]
            # Placeholder: you can replace with real model-driven triplet output
            triplets.append({
                "subject": "Subject",
                "relation": "related_to",
                "object": "Object",
                "sentence": sentence,
                "paper": item.get("paper", "Unknown"),
                "file_name": item.get("file_name", "Unknown"),
                "doi": item.get("doi", "")
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(triplets, f, indent=4)

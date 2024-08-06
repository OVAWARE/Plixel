# FILE: text_embed.py

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from parameters import NUM_CHANNELS

TEXT_EMBED_DIM = 768  # DistilBERT's output dimension

class TextEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # We use the [CLS] token embedding

text_embedder = TextEmbedder()

def get_text_embedding(text):
    with torch.no_grad():
        return text_embedder(text)
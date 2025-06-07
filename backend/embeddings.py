# backend/embeddings.py
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed_text(text):
    return model.encode(text)
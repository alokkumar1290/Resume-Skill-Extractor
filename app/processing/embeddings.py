import os
import json
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional

# Load a lightweight embedding model once
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embedder: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder

def embed_text(text: str) -> List[float]:
    """Return embedding vector (list of floats) for given text."""
    if not text:
        return []
    model = _get_model()
    vec = model.encode([text.strip()])[0]
    return vec.tolist()

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    va = np.array(a)
    vb = np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)

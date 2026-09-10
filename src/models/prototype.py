"""
Prototype Vector Computation & Metric Comparison
SIH Problem Statement 26172
"""

import numpy as np

def compute_prototype(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes a single prototype vector from K enrollment embeddings.
    p = L2_normalize(mean(embeddings))
    """
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]

    mean_vec = np.mean(embeddings, axis=0, keepdims=True)
    norm = np.linalg.norm(mean_vec, axis=1, keepdims=True)
    if norm[0, 0] == 0:
        return mean_vec[0]
    prototype = (mean_vec / norm)[0]
    return prototype.astype(np.float32)

def compute_cosine_similarity(prototype: np.ndarray, query: np.ndarray) -> float:
    """
    Computes cosine similarity between an L2-normalized prototype and query.
    For L2-normalized vectors, cosine similarity is identical to the dot product.
    """
    # Ensure 1D vectors
    p = prototype.flatten()
    q = query.flatten()
    sim = float(np.dot(p, q))
    return max(min(sim, 1.0), -1.0)

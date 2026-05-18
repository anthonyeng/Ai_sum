"""
Sentence-BERT text embedding module.

Encodes transcript chunks, captions, and queries into 384-dim dense vectors
using the pretrained all-MiniLM-L6-v2 model. Used for:
  - Multimodal feature fusion (text signal for importance scoring)
  - Semantic summarization (via sbert_mmr.py)
  - Query-focused search (match user queries against transcript chunks)

Usage:
    from src.features.text_embeddings import encode_texts, encode_query, text_similarity
    embeddings = encode_texts(["chunk1", "chunk2", ...])   # (N, 384)
    query_emb = encode_query("explain thermodynamics")     # (384,)
    scores = text_similarity(embeddings, query_emb)        # (N,)
"""

import numpy as np

_sbert_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"
_SBERT_DIM = 384


def _load_model():
    """Load Sentence-BERT model (lazy, cached)."""
    global _sbert_model
    if _sbert_model is None:
        import os
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TF", "0")
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(_MODEL_NAME)
    return _sbert_model


def encode_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of text strings into dense vectors.

    Args:
        texts: list of strings to encode
        batch_size: encoding batch size

    Returns:
        np.ndarray of shape (len(texts), 384) — L2-normalized
    """
    if not texts:
        return np.empty((0, _SBERT_DIM), dtype=np.float32)

    model = _load_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single query string.

    Args:
        query: search query text

    Returns:
        np.ndarray of shape (384,) — L2-normalized
    """
    return encode_texts([query])[0]


def text_similarity(embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between text embeddings and a query.

    Args:
        embeddings: (N, 384) text embeddings
        query_embedding: (384,) query embedding

    Returns:
        np.ndarray of shape (N,) — similarity scores
    """
    # Already L2-normalized, so dot product = cosine similarity
    return (embeddings @ query_embedding).astype(np.float32)


def get_sbert_dim() -> int:
    """Return Sentence-BERT embedding dimension."""
    return _SBERT_DIM

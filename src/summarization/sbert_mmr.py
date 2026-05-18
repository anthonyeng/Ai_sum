"""
Sentence-BERT + Maximal Marginal Relevance (MMR) extractive summarizer.

Encodes transcript chunks into dense semantic embeddings using a pretrained
Sentence-BERT model, then selects a diverse, relevant subset via MMR ranking.

MMR balances two objectives:
  - Relevance: each selected chunk should be close to the document centroid.
  - Diversity:  each selected chunk should be dissimilar to already-selected chunks.

MMR(i) = λ · sim(chunk_i, centroid) − (1−λ) · max_{j ∈ S} sim(chunk_i, chunk_j)

Usage:
    from src.summarization.sbert_mmr import sbert_mmr_summarize
    result = sbert_mmr_summarize(chunks, num_sentences=5, lambda_param=0.7)
"""

import os
# Prevent transformers from importing tensorflow (causes numpy binary conflict in Anaconda)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
from typing import List, TypedDict

# Lazy-load to avoid slow import at module level
_sbert_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, runs on CPU


class SummaryChunk(TypedDict):
    text: str
    score: float
    start: float
    end: float
    rank: int


class SummaryResult(TypedDict):
    method: str
    selected_chunks: List[SummaryChunk]
    summary_text: str
    num_input_chunks: int
    num_selected: int
    redundancy_score: float
    diversity_score: float
    compression_ratio: float
    embedding_dim: int
    model_name: str
    lambda_param: float


def _load_sbert():
    """Load Sentence-BERT model (lazy, cached)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(_MODEL_NAME)
    return _sbert_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector(s) a and vector(s) b."""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    if a_norm.ndim == 1:
        a_norm = a_norm[np.newaxis, :]
    if b_norm.ndim == 1:
        b_norm = b_norm[np.newaxis, :]
    return a_norm @ b_norm.T


def _mmr_selection(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    num_select: int,
    lambda_param: float = 0.7,
) -> list[int]:
    """
    Select indices using Maximal Marginal Relevance.

    Args:
        embeddings: (N, D) matrix of chunk embeddings
        centroid: (D,) document centroid vector
        num_select: how many chunks to pick
        lambda_param: trade-off between relevance (1.0) and diversity (0.0)

    Returns:
        List of selected indices in MMR order.
    """
    n = len(embeddings)
    if n <= num_select:
        return list(range(n))

    # Relevance: similarity of each chunk to the document centroid
    relevance = _cosine_similarity(embeddings, centroid).squeeze()

    selected = []
    remaining = set(range(n))

    for _ in range(num_select):
        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            rel = relevance[idx]
            if selected:
                sel_embs = embeddings[selected]
                max_sim = _cosine_similarity(
                    embeddings[idx], sel_embs
                ).max()
            else:
                max_sim = 0.0

            mmr = lambda_param * rel - (1 - lambda_param) * max_sim

            if mmr > best_score:
                best_score = mmr
                best_idx = idx

        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def _compute_redundancy(embeddings: np.ndarray, indices: list[int]) -> float:
    """Average pairwise cosine similarity among selected chunks."""
    if len(indices) < 2:
        return 0.0
    sel = embeddings[indices]
    sim_matrix = _cosine_similarity(sel, sel)
    n = len(indices)
    # Extract upper triangle (exclude diagonal)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i, j]
            count += 1
    return round(total / count, 4) if count > 0 else 0.0


def sbert_mmr_summarize(
    chunks: list[dict],
    num_sentences: int = 5,
    lambda_param: float = 0.7,
) -> SummaryResult:
    """
    Sentence-BERT + MMR extractive summarization.

    Args:
        chunks: list of {"text": str, "start": float, "end": float}
        num_sentences: how many chunks to select
        lambda_param: MMR trade-off (0.5=balanced, 0.7=more relevant, 0.3=more diverse)

    Returns:
        SummaryResult with ranked chunks, summary text, and metrics.
    """
    texts = [c["text"] for c in chunks]
    total_input_words = sum(len(t.split()) for t in texts)

    # Filter out very short chunks
    valid = [(i, c) for i, c in enumerate(chunks) if len(c["text"].split()) >= 3]
    if not valid:
        valid = list(enumerate(chunks))

    valid_indices = [i for i, _ in valid]
    valid_texts = [chunks[i]["text"] for i in valid_indices]

    # Encode with Sentence-BERT
    model = _load_sbert()
    embeddings = model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
    embedding_dim = embeddings.shape[1]

    # Document centroid = mean of all chunk embeddings
    centroid = embeddings.mean(axis=0)

    # MMR selection
    mmr_indices = _mmr_selection(
        embeddings, centroid, num_sentences, lambda_param
    )

    # Map back to original chunk indices and preserve document order
    original_indices = [valid_indices[i] for i in mmr_indices]
    ordered = sorted(range(len(original_indices)), key=lambda k: original_indices[k])

    # Compute relevance scores for output
    relevance = _cosine_similarity(embeddings, centroid).squeeze()

    selected_chunks = []
    for rank, k in enumerate(ordered):
        orig_idx = original_indices[k]
        mmr_idx = mmr_indices[k]
        selected_chunks.append(SummaryChunk(
            text=chunks[orig_idx]["text"],
            score=round(float(relevance[mmr_idx]), 4),
            start=chunks[orig_idx].get("start", 0.0),
            end=chunks[orig_idx].get("end", 0.0),
            rank=rank + 1,
        ))

    summary_text = " ".join(c["text"] for c in selected_chunks)
    summary_words = len(summary_text.split())
    redundancy = _compute_redundancy(embeddings, mmr_indices)

    return SummaryResult(
        method="sbert_mmr",
        selected_chunks=selected_chunks,
        summary_text=summary_text,
        num_input_chunks=len(chunks),
        num_selected=len(selected_chunks),
        redundancy_score=redundancy,
        diversity_score=round(1.0 - redundancy, 4),
        compression_ratio=round(summary_words / max(total_input_words, 1), 4),
        embedding_dim=embedding_dim,
        model_name=_MODEL_NAME,
        lambda_param=lambda_param,
    )

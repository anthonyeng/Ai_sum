"""
TextRank extractive summarizer — graph-based baseline.

Builds a sentence similarity graph and applies PageRank to identify
the most central (important) sentences. Uses TF-IDF vectors for
similarity computation (no neural models required).

Reference:
    Mihalcea & Tarau (2004), "TextRank: Bringing Order into Texts"

Usage:
    from src.summarization.textrank import textrank_summarize
    result = textrank_summarize(chunks, num_sentences=5)
"""

import math
import re
from collections import Counter
from typing import List, TypedDict

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None


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


STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'out',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this', 'it',
    'its', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'our', 'their', 'what', 'which',
    'who', 'about', 'also', 'like', 'these', 'those',
})


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'[a-zA-Z]+', text)
            if w.lower() not in STOP_WORDS and len(w) > 2]


def _build_tfidf_vectors(texts: list[str]) -> np.ndarray:
    """Build sparse TF-IDF vectors for cosine similarity."""
    token_lists = [_tokenize(t) for t in texts]

    # Build vocabulary
    vocab = {}
    doc_freq = Counter()
    for tokens in token_lists:
        unique = set(tokens)
        for w in unique:
            if w not in vocab:
                vocab[w] = len(vocab)
            doc_freq[w] += 1

    if not vocab:
        return np.zeros((len(texts), 1))

    n_docs = len(texts)
    n_vocab = len(vocab)
    vectors = np.zeros((n_docs, n_vocab), dtype=np.float32)

    for i, tokens in enumerate(token_lists):
        if not tokens:
            continue
        tf = Counter(tokens)
        for word, count in tf.items():
            j = vocab[word]
            tf_val = count / len(tokens)
            idf_val = math.log((n_docs + 1) / (doc_freq[word] + 1)) + 1
            vectors[i, j] = tf_val * idf_val

    # L2 normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    vectors = vectors / norms
    return vectors


def _pagerank_scores(sim_matrix: np.ndarray, damping: float = 0.85,
                     max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Compute PageRank on a similarity matrix (no networkx dependency)."""
    n = sim_matrix.shape[0]
    if n == 0:
        return np.array([])

    # Build transition matrix: normalize rows
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim_matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * (transition.T @ scores)
        if np.abs(new_scores - scores).sum() < tol:
            break
        scores = new_scores

    return scores


def _compute_redundancy(chunks: list[dict]) -> float:
    """Word-overlap redundancy between selected chunks."""
    if len(chunks) < 2:
        return 0.0
    overlaps = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            w1 = set(chunks[i]["text"].lower().split())
            w2 = set(chunks[j]["text"].lower().split())
            if w1 and w2:
                overlaps.append(len(w1 & w2) / len(w1 | w2))
    return round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0


def textrank_summarize(
    chunks: list[dict],
    num_sentences: int = 5,
    sim_threshold: float = 0.1,
) -> SummaryResult:
    """
    TextRank extractive summarization.

    Builds a similarity graph over sentences and applies PageRank to find
    the most central (important) ones.

    Args:
        chunks: list of {"text": str, "start": float, "end": float}
        num_sentences: how many chunks to select
        sim_threshold: minimum similarity to create an edge

    Returns:
        SummaryResult with ranked chunks, summary text, and metrics.
    """
    texts = [c["text"] for c in chunks]
    total_input_words = sum(len(t.split()) for t in texts)
    n = len(texts)

    if n <= num_sentences:
        selected_chunks = [
            SummaryChunk(text=c["text"], score=1.0,
                         start=c.get("start", 0.0), end=c.get("end", 0.0),
                         rank=i + 1)
            for i, c in enumerate(chunks)
        ]
        summary_text = " ".join(c["text"] for c in selected_chunks)
        return SummaryResult(
            method="textrank",
            selected_chunks=selected_chunks,
            summary_text=summary_text,
            num_input_chunks=n,
            num_selected=n,
            redundancy_score=0.0,
            diversity_score=1.0,
            compression_ratio=1.0,
        )

    # Build TF-IDF vectors and similarity matrix
    vectors = _build_tfidf_vectors(texts)
    sim_matrix = vectors @ vectors.T

    # Apply threshold to create sparse graph
    sim_matrix[sim_matrix < sim_threshold] = 0.0
    np.fill_diagonal(sim_matrix, 0.0)

    # Use networkx if available, otherwise custom PageRank
    if nx is not None:
        graph = nx.from_numpy_array(sim_matrix)
        try:
            scores_dict = nx.pagerank(graph, alpha=0.85, max_iter=100)
            scores = np.array([scores_dict[i] for i in range(n)])
        except nx.PowerIterationFailedConvergence:
            scores = _pagerank_scores(sim_matrix)
    else:
        scores = _pagerank_scores(sim_matrix)

    # Rank and select top sentences, preserve document order
    ranked = np.argsort(scores)[::-1][:num_sentences]
    selected_indices = sorted(ranked)

    selected_chunks = []
    for rank, idx in enumerate(selected_indices):
        selected_chunks.append(SummaryChunk(
            text=chunks[idx]["text"],
            score=round(float(scores[idx]), 4),
            start=chunks[idx].get("start", 0.0),
            end=chunks[idx].get("end", 0.0),
            rank=rank + 1,
        ))

    summary_text = " ".join(c["text"] for c in selected_chunks)
    summary_words = len(summary_text.split())
    redundancy = _compute_redundancy(selected_chunks)

    return SummaryResult(
        method="textrank",
        selected_chunks=selected_chunks,
        summary_text=summary_text,
        num_input_chunks=n,
        num_selected=len(selected_chunks),
        redundancy_score=redundancy,
        diversity_score=round(1.0 - redundancy, 4),
        compression_ratio=round(summary_words / max(total_input_words, 1), 4),
    )

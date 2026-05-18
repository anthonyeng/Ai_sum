"""
TF-IDF extractive summarizer — baseline method.

Extracts from the original pipeline. Scores sentences using term frequency–
inverse document frequency with topic-word boosting and position weighting.

Usage:
    from src.summarization.tfidf_baseline import tfidf_summarize
    result = tfidf_summarize(chunks)
"""

import math
import re
from collections import Counter
from typing import List, TypedDict


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


# ── Stop words ────────────────────────────────────────────────────────────────

STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'out', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this',
    'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'our', 'their', 'what', 'which', 'who',
    'decided', 'went', 'came', 'going', 'got', 'get', 'let', 'make', 'know',
    'think', 'want', 'see', 'look', 'find', 'give', 'tell', 'say', 'said',
    'about', 'also', 'like', 'these', 'those',
})

FILLER_RE = re.compile(
    r'subscribe|thumbs up|like.*video|bell icon|comment.*below|'
    r'click.*subscribe|notification|watch.*next|thank.*watching|'
    r'stay tuned|check.*out|link.*description|appreciated|'
    r'reminder to|our channel|next week',
    re.IGNORECASE,
)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'[a-zA-Z]+', text)
            if w.lower() not in STOP_WORDS and len(w) > 2]


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


def tfidf_summarize(
    chunks: list[dict],
    num_sentences: int = 5,
) -> SummaryResult:
    """
    TF-IDF extractive summarization.

    Args:
        chunks: list of {"text": str, "start": float, "end": float}
        num_sentences: how many chunks to select

    Returns:
        SummaryResult with ranked chunks, summary text, and metrics.
    """
    texts = [c["text"] for c in chunks]
    total_input_words = sum(len(t.split()) for t in texts)

    # Filter filler
    valid_indices = [
        i for i, t in enumerate(texts)
        if not FILLER_RE.search(t) and len(t.split()) >= 5
    ]
    if len(valid_indices) <= num_sentences:
        valid_indices = list(range(len(texts)))

    # Tokenize
    all_words = Counter()
    sent_tokens = {}
    for i in valid_indices:
        tokens = _tokenize(texts[i])
        sent_tokens[i] = tokens
        all_words.update(tokens)

    topic_words = set(w for w, _ in all_words.most_common(20))

    # Document frequency
    doc_freq = Counter()
    for tokens in sent_tokens.values():
        for w in set(tokens):
            doc_freq[w] += 1

    n_docs = len(valid_indices)

    # Score each sentence
    scores = {}
    for i in valid_indices:
        tokens = sent_tokens[i]
        if not tokens:
            scores[i] = 0.0
            continue
        tf = Counter(tokens)
        score = 0.0
        for word, count in tf.items():
            tf_val = count / len(tokens)
            idf_val = math.log((n_docs + 1) / (doc_freq[word] + 1)) + 1
            score += tf_val * idf_val
            if word in topic_words:
                score += 0.5
        if len(tokens) < 5:
            score *= 0.5
        position = valid_indices.index(i) / max(n_docs - 1, 1)
        if 0.1 < position < 0.85:
            score *= 1.2
        scores[i] = score

    # Rank and select
    ranked = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    selected_indices = sorted(ranked[:num_sentences])

    selected_chunks = []
    for rank, idx in enumerate(selected_indices):
        selected_chunks.append(SummaryChunk(
            text=chunks[idx]["text"],
            score=round(scores[idx], 4),
            start=chunks[idx].get("start", 0.0),
            end=chunks[idx].get("end", 0.0),
            rank=rank + 1,
        ))

    summary_text = " ".join(c["text"] for c in selected_chunks)
    summary_words = len(summary_text.split())
    redundancy = _compute_redundancy(selected_chunks)

    return SummaryResult(
        method="tfidf",
        selected_chunks=selected_chunks,
        summary_text=summary_text,
        num_input_chunks=len(chunks),
        num_selected=len(selected_chunks),
        redundancy_score=redundancy,
        diversity_score=round(1.0 - redundancy, 4),
        compression_ratio=round(summary_words / max(total_input_words, 1), 4),
    )

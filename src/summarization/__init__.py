"""
Summarization module — multiple extractive summarization strategies.

Available methods:
  - TF-IDF baseline (original)
  - Sentence-BERT + MMR (semantic)
  - TextRank (graph-based baseline)
"""

from src.summarization.tfidf_baseline import tfidf_summarize
from src.summarization.sbert_mmr import sbert_mmr_summarize
from src.summarization.textrank import textrank_summarize

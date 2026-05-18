"""
Text summarization evaluation metrics.

Evaluates extractive text summaries using:
  - ROUGE-1, ROUGE-2, ROUGE-L (using rouge-score library if available)
  - BERTScore (using bert-score library if available)
  - Fallback implementations if libraries not installed

Usage:
    from src.evaluation.text_metrics import evaluate_text_summary
    metrics = evaluate_text_summary(hypothesis="...", reference="...")
"""

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def rouge_n(hypothesis: str, reference: str, n: int = 1) -> dict:
    """Compute ROUGE-N (n-gram overlap) precision, recall, F1."""
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    hyp_ngrams = Counter(ngrams(hyp_tokens, n))
    ref_ngrams = Counter(ngrams(ref_tokens, n))

    overlap = 0
    for ng, count in hyp_ngrams.items():
        overlap += min(count, ref_ngrams.get(ng, 0))

    hyp_total = sum(hyp_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    precision = overlap / hyp_total if hyp_total > 0 else 0.0
    recall = overlap / ref_total if ref_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def rouge_l(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-L (longest common subsequence)."""
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    m, n = len(hyp_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / m if m > 0 else 0.0
    recall = lcs / n if n > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_rouge_all(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L.
    Uses rouge-score library if available, otherwise falls back to custom impl.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": {
                "precision": round(scores['rouge1'].precision, 4),
                "recall": round(scores['rouge1'].recall, 4),
                "f1": round(scores['rouge1'].fmeasure, 4),
            },
            "rouge2": {
                "precision": round(scores['rouge2'].precision, 4),
                "recall": round(scores['rouge2'].recall, 4),
                "f1": round(scores['rouge2'].fmeasure, 4),
            },
            "rougeL": {
                "precision": round(scores['rougeL'].precision, 4),
                "recall": round(scores['rougeL'].recall, 4),
                "f1": round(scores['rougeL'].fmeasure, 4),
            },
        }
    except ImportError:
        return {
            "rouge1": rouge_n(hypothesis, reference, 1),
            "rouge2": rouge_n(hypothesis, reference, 2),
            "rougeL": rouge_l(hypothesis, reference),
            "_note": "using fallback implementation (install rouge-score for stemmed version)",
        }


def compute_bertscore(hypothesis: str, reference: str) -> dict:
    """Compute BERTScore if bert-score library is available."""
    try:
        import os
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TF", "0")
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [hypothesis], [reference],
            lang="en", verbose=False, rescale_with_baseline=True,
        )
        return {
            "bertscore_precision": round(P.item(), 4),
            "bertscore_recall": round(R.item(), 4),
            "bertscore_f1": round(F1.item(), 4),
        }
    except ImportError:
        return {"bertscore_error": "bert-score not installed"}


def evaluate_text_summary(
    hypothesis: str,
    reference: str,
    compute_bert: bool = True,
) -> dict:
    """
    Full evaluation of a text summary.

    Args:
        hypothesis: generated summary text
        reference: ground-truth reference summary
        compute_bert: whether to compute BERTScore (slower)

    Returns:
        dict with ROUGE-1/2/L and optionally BERTScore
    """
    metrics = compute_rouge_all(hypothesis, reference)

    if compute_bert:
        bert = compute_bertscore(hypothesis, reference)
        metrics["bertscore"] = bert

    # Simple stats
    metrics["hypothesis_words"] = len(hypothesis.split())
    metrics["reference_words"] = len(reference.split())
    metrics["compression_ratio"] = round(
        len(hypothesis.split()) / max(len(reference.split()), 1), 4
    )

    return metrics

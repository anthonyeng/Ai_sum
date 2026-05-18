"""
Video captioning evaluation metrics.

Evaluates the trained Seq2Seq VideoCaptionModel against reference captions.

Metrics:
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4 (n-gram precision)
  - ROUGE-L (longest common subsequence)
  - METEOR (alignment-based, if nltk available)

Usage:
    from src.evaluation.caption_metrics import evaluate_captions
    metrics = evaluate_captions(predictions=["a man walks"], references=[["a person walking"]])
"""

import re
import math
from collections import Counter
from typing import Optional


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\w+', text.lower())


def bleu_score(
    prediction: str,
    references: list[str],
    max_n: int = 4,
) -> dict:
    """
    Compute BLEU-1 through BLEU-N for a single prediction against references.

    Uses smoothed BLEU (add-1 smoothing for n > 1) to handle short captions.
    """
    pred_tokens = _tokenize(prediction)
    ref_token_lists = [_tokenize(r) for r in references]

    if not pred_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    scores = {}
    log_bleu_sum = 0.0

    for n in range(1, max_n + 1):
        pred_ngrams = Counter()
        for i in range(len(pred_tokens) - n + 1):
            ngram = tuple(pred_tokens[i:i + n])
            pred_ngrams[ngram] += 1

        max_ref_ngrams = Counter()
        for ref_tokens in ref_token_lists:
            ref_ngrams = Counter()
            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i + n])
                ref_ngrams[ngram] += 1
            for ngram, count in ref_ngrams.items():
                max_ref_ngrams[ngram] = max(max_ref_ngrams.get(ngram, 0), count)

        clipped = 0
        for ngram, count in pred_ngrams.items():
            clipped += min(count, max_ref_ngrams.get(ngram, 0))

        total = sum(pred_ngrams.values())

        # Smoothing for n > 1
        if n == 1:
            precision = clipped / total if total > 0 else 0.0
        else:
            precision = (clipped + 1) / (total + 1) if total > 0 else 0.0

        scores[f"bleu_{n}"] = round(precision, 4)

        if precision > 0:
            log_bleu_sum += math.log(precision)
        else:
            log_bleu_sum += -float("inf")

    # Brevity penalty
    pred_len = len(pred_tokens)
    ref_lens = [len(r) for r in ref_token_lists]
    closest_ref_len = min(ref_lens, key=lambda r: abs(r - pred_len))

    if pred_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / pred_len) if pred_len > 0 else 0.0

    # Cumulative BLEU (geometric mean)
    for n in range(1, max_n + 1):
        precisions = [scores[f"bleu_{k}"] for k in range(1, n + 1)]
        if all(p > 0 for p in precisions):
            log_avg = sum(math.log(p) for p in precisions) / n
            scores[f"bleu_{n}_cumulative"] = round(bp * math.exp(log_avg), 4)
        else:
            scores[f"bleu_{n}_cumulative"] = 0.0

    return scores


def rouge_l(prediction: str, reference: str) -> dict:
    """
    Compute ROUGE-L (longest common subsequence) F-score.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    # LCS length via DP
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "rouge_l_precision": round(precision, 4),
        "rouge_l_recall": round(recall, 4),
        "rouge_l_f1": round(f1, 4),
    }


def meteor_score(prediction: str, reference: str) -> dict:
    """
    Compute METEOR score using NLTK (if available).
    Falls back to a simple unigram F-score if NLTK is not installed.
    """
    try:
        from nltk.translate.meteor_score import single_meteor_score
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        score = single_meteor_score(ref_tokens, pred_tokens)
        return {"meteor": round(score, 4)}
    except ImportError:
        # Fallback: simple unigram overlap
        pred_set = set(_tokenize(prediction))
        ref_set = set(_tokenize(reference))
        if not pred_set or not ref_set:
            return {"meteor": 0.0}
        overlap = len(pred_set & ref_set)
        p = overlap / len(pred_set)
        r = overlap / len(ref_set)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"meteor": round(f, 4), "meteor_note": "nltk not available, using unigram fallback"}


def evaluate_captions(
    predictions: list[str],
    references: list[list[str]],
) -> dict:
    """
    Evaluate a batch of predicted captions against reference captions.

    Args:
        predictions: list of predicted caption strings
        references: list of reference caption lists (multiple refs per prediction)

    Returns:
        dict with averaged metrics across all predictions
    """
    if not predictions or not references:
        return {}

    all_bleu = {f"bleu_{n}": [] for n in range(1, 5)}
    all_rouge_l = []
    all_meteor = []

    for pred, refs in zip(predictions, references):
        # BLEU
        b = bleu_score(pred, refs)
        for n in range(1, 5):
            all_bleu[f"bleu_{n}"].append(b[f"bleu_{n}"])

        # ROUGE-L (against best reference)
        best_rouge = 0.0
        for ref in refs:
            r = rouge_l(pred, ref)
            if r["rouge_l_f1"] > best_rouge:
                best_rouge = r["rouge_l_f1"]
        all_rouge_l.append(best_rouge)

        # METEOR (against best reference)
        best_meteor = 0.0
        for ref in refs:
            m = meteor_score(pred, ref)
            if m["meteor"] > best_meteor:
                best_meteor = m["meteor"]
        all_meteor.append(best_meteor)

    metrics = {}
    for n in range(1, 5):
        vals = all_bleu[f"bleu_{n}"]
        metrics[f"bleu_{n}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

    metrics["rouge_l_f1"] = round(sum(all_rouge_l) / len(all_rouge_l), 4) if all_rouge_l else 0.0
    metrics["meteor"] = round(sum(all_meteor) / len(all_meteor), 4) if all_meteor else 0.0
    metrics["num_evaluated"] = len(predictions)

    return metrics

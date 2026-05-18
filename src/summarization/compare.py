"""
Compare summarization methods: TF-IDF vs TextRank vs Sentence-BERT + MMR.

Runs all three methods on the same input and reports:
  - Selected sentences
  - ROUGE-1, ROUGE-2, ROUGE-L (if reference summary provided)
  - BERTScore (if reference summary provided)
  - Redundancy and diversity scores
  - Compression ratio

Usage:
    python src/summarization/compare.py --transcript "path/to/transcript.txt"
    python src/summarization/compare.py --transcript "path/to/transcript.txt" --reference "path/to/reference.txt"
    python src/summarization/compare.py --demo
"""

import argparse
import json
import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.summarization.tfidf_baseline import tfidf_summarize
from src.summarization.textrank import textrank_summarize
from src.summarization.sbert_mmr import sbert_mmr_summarize


# ── Chunking ──────────────────────────────────────────────────────────────────

def text_to_chunks(text: str, method: str = "sentence") -> list[dict]:
    """
    Split raw text into chunks with fake timestamps.

    Args:
        text: raw transcript or document text
        method: "sentence" for sentence splitting, "window" for fixed-size windows
    """
    if method == "sentence":
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Fallback for unpunctuated text
        if len(sentences) <= 2 and len(text) > 150:
            words = text.split()
            chunk_size = max(12, min(20, len(words) // 8))
            sentences = [' '.join(words[i:i + chunk_size])
                         for i in range(0, len(words), chunk_size)]
    else:
        words = text.split()
        chunk_size = 20
        sentences = [' '.join(words[i:i + chunk_size])
                     for i in range(0, len(words), chunk_size)]

    chunks = []
    offset = 0.0
    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        duration = max(2.0, len(s.split()) * 0.4)
        chunks.append({"text": s, "start": round(offset, 1), "end": round(offset + duration, 1)})
        offset += duration
    return chunks


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_rouge(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F-scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1_f": round(scores['rouge1'].fmeasure, 4),
            "rouge2_f": round(scores['rouge2'].fmeasure, 4),
            "rougeL_f": round(scores['rougeL'].fmeasure, 4),
        }
    except ImportError:
        return {"rouge_error": "Install rouge-score: pip install rouge-score"}


def compute_bertscore(hypothesis: str, reference: str) -> dict:
    """Compute BERTScore F1."""
    try:
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
        return {"bertscore_error": "Install bert-score: pip install bert-score"}


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_methods(
    chunks: list[dict],
    num_sentences: int = 5,
    reference: str = None,
) -> dict:
    """
    Run all three summarization methods and compare.

    Args:
        chunks: list of {"text": str, "start": float, "end": float}
        num_sentences: how many sentences to extract
        reference: optional reference summary for ROUGE/BERTScore

    Returns:
        dict with per-method results and metrics.
    """
    methods = {
        "tfidf": lambda: tfidf_summarize(chunks, num_sentences),
        "textrank": lambda: textrank_summarize(chunks, num_sentences),
        "sbert_mmr": lambda: sbert_mmr_summarize(chunks, num_sentences),
    }

    results = {}
    for name, fn in methods.items():
        print(f"  Running {name}...", end=" ", flush=True)
        t0 = time.time()
        result = fn()
        elapsed = round(time.time() - t0, 3)
        print(f"done ({elapsed}s)")

        entry = {
            "summary": result["summary_text"],
            "num_selected": result["num_selected"],
            "compression_ratio": result["compression_ratio"],
            "redundancy_score": result["redundancy_score"],
            "diversity_score": result["diversity_score"],
            "inference_time_sec": elapsed,
        }

        if reference:
            entry["rouge"] = compute_rouge(result["summary_text"], reference)
            entry["bertscore"] = compute_bertscore(result["summary_text"], reference)

        results[name] = entry

    return {
        "num_input_chunks": len(chunks),
        "num_sentences_requested": num_sentences,
        "has_reference": reference is not None,
        "methods": results,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

DEMO_TRANSCRIPT = """
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Supervised learning uses labeled datasets to train models that can make predictions.
Common algorithms include linear regression, decision trees, and neural networks.
Deep learning is a specialized form of machine learning using multi-layered neural networks.
Convolutional neural networks are particularly effective for image recognition tasks.
Recurrent neural networks handle sequential data like time series and natural language.
Transfer learning allows models pretrained on large datasets to be fine-tuned for specific tasks.
Reinforcement learning trains agents through reward signals in an environment.
The bias-variance tradeoff is a fundamental concept in model selection and evaluation.
Overfitting occurs when a model memorizes training data instead of learning generalizable patterns.
Regularization techniques like L1 and L2 penalties help prevent overfitting.
Cross-validation provides more reliable estimates of model performance than a single train-test split.
Feature engineering transforms raw data into representations that improve model accuracy.
Dimensionality reduction techniques like PCA compress high-dimensional data while preserving structure.
Ensemble methods combine multiple models to achieve better predictive performance.
Random forests aggregate many decision trees to reduce variance and improve accuracy.
Gradient boosting builds trees sequentially, each correcting errors of the previous one.
Neural architecture search automates the design of neural network architectures.
Attention mechanisms allow models to focus on the most relevant parts of the input.
Transformers revolutionized natural language processing with self-attention mechanisms.
""".strip()

DEMO_REFERENCE = (
    "Machine learning enables systems to learn from data using algorithms like "
    "neural networks and decision trees. Deep learning uses multi-layered networks "
    "for tasks like image recognition. Key concepts include the bias-variance "
    "tradeoff, regularization to prevent overfitting, and ensemble methods like "
    "random forests and gradient boosting. Transformers with attention mechanisms "
    "have revolutionized natural language processing."
)


def run_demo():
    """Run comparison on a demo transcript."""
    print("=" * 70)
    print("SUMMARIZATION METHOD COMPARISON — DEMO")
    print("=" * 70)

    chunks = text_to_chunks(DEMO_TRANSCRIPT)
    print(f"\nInput: {len(chunks)} chunks, {sum(len(c['text'].split()) for c in chunks)} words")
    print(f"Reference summary: {len(DEMO_REFERENCE.split())} words\n")

    results = compare_methods(chunks, num_sentences=5, reference=DEMO_REFERENCE)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for method, data in results["methods"].items():
        print(f"\n{'─' * 40}")
        print(f"  Method: {method.upper()}")
        print(f"{'─' * 40}")
        print(f"  Compression:  {data['compression_ratio']:.3f}")
        print(f"  Redundancy:   {data['redundancy_score']:.4f}")
        print(f"  Diversity:    {data['diversity_score']:.4f}")
        print(f"  Time:         {data['inference_time_sec']}s")

        if "rouge" in data and "rouge_error" not in data["rouge"]:
            r = data["rouge"]
            print(f"  ROUGE-1 F:    {r['rouge1_f']:.4f}")
            print(f"  ROUGE-2 F:    {r['rouge2_f']:.4f}")
            print(f"  ROUGE-L F:    {r['rougeL_f']:.4f}")

        if "bertscore" in data and "bertscore_error" not in data["bertscore"]:
            b = data["bertscore"]
            print(f"  BERTScore F1: {b['bertscore_f1']:.4f}")

        print(f"\n  Summary:")
        words = data["summary"].split()
        for i in range(0, len(words), 15):
            print(f"    {' '.join(words[i:i+15])}")

    print(f"\n{'=' * 70}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare summarization methods")
    parser.add_argument("--transcript", type=str, help="Path to transcript .txt file")
    parser.add_argument("--reference", type=str, help="Path to reference summary .txt file")
    parser.add_argument("--num-sentences", type=int, default=5, help="Number of sentences to extract")
    parser.add_argument("--demo", action="store_true", help="Run demo comparison")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if args.demo:
        results = run_demo()
    elif args.transcript:
        if not os.path.exists(args.transcript):
            print(f"File not found: {args.transcript}")
            sys.exit(1)

        with open(args.transcript) as f:
            text = f.read().strip()

        reference = None
        if args.reference:
            with open(args.reference) as f:
                reference = f.read().strip()

        chunks = text_to_chunks(text)
        print(f"Input: {len(chunks)} chunks, {sum(len(c['text'].split()) for c in chunks)} words\n")
        results = compare_methods(chunks, args.num_sentences, reference)

        for method, data in results["methods"].items():
            print(f"\n{method.upper()}: compression={data['compression_ratio']:.3f}, "
                  f"redundancy={data['redundancy_score']:.4f}, "
                  f"diversity={data['diversity_score']:.4f}, "
                  f"time={data['inference_time_sec']}s")
    else:
        parser.print_help()
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

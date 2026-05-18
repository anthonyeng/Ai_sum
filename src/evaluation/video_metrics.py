"""
Video summarization evaluation metrics.

Evaluates the Video-to-Video pipeline output against human annotations
(TVSum-style frame-level importance scores) and intrinsic quality measures.

Metrics:
  - F-score: overlap between predicted and ground-truth important segments
  - Compression ratio: summary length / original length
  - Redundancy score: pairwise similarity among selected segments
  - Diversity score: 1 - redundancy
  - Coverage score: % of ground-truth important segments captured

Usage:
    from src.evaluation.video_metrics import evaluate_video_summary
    metrics = evaluate_video_summary(selected_indices, scores, total_segments)
"""

import numpy as np
from typing import Optional


def f_score(
    predicted_indices: list[int],
    ground_truth_indices: list[int],
    total_segments: int,
) -> dict:
    """
    Compute precision, recall, and F1-score for segment selection.

    Args:
        predicted_indices: indices of segments selected by the model
        ground_truth_indices: indices of segments marked as important by humans
        total_segments: total number of segments in the video

    Returns:
        dict with precision, recall, f1
    """
    pred_set = set(predicted_indices)
    gt_set = set(ground_truth_indices)

    if not pred_set or not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compression_ratio(selected_count: int, total_count: int) -> float:
    """Ratio of selected segments to total segments."""
    if total_count == 0:
        return 0.0
    return round(selected_count / total_count, 4)


def redundancy_score(features: np.ndarray, selected_indices: list[int]) -> float:
    """
    Average pairwise cosine similarity among selected segments.
    Higher = more redundant (bad). Lower = more diverse (good).
    """
    if len(selected_indices) < 2:
        return 0.0

    selected = features[selected_indices]
    norms = np.linalg.norm(selected, axis=1, keepdims=True) + 1e-10
    normalized = selected / norms
    sim_matrix = normalized @ normalized.T

    n = len(selected_indices)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i, j]
            count += 1

    return round(total / count, 4) if count > 0 else 0.0


def diversity_score(features: np.ndarray, selected_indices: list[int]) -> float:
    """1 - redundancy. Higher = more diverse (good)."""
    return round(1.0 - redundancy_score(features, selected_indices), 4)


def coverage_score(
    selected_indices: list[int],
    importance_scores: np.ndarray,
    threshold_factor: float = 0.5,
) -> float:
    """
    What fraction of the "important" segments (above threshold) were selected.

    Args:
        selected_indices: model's selected segment indices
        importance_scores: ground-truth importance per segment
        threshold_factor: segments above mean + factor*std are "important"
    """
    if len(importance_scores) == 0:
        return 0.0

    threshold = importance_scores.mean() + threshold_factor * importance_scores.std()
    gt_important = set(np.where(importance_scores >= threshold)[0])

    if not gt_important:
        return 1.0  # Nothing was important, so we covered everything

    selected_set = set(selected_indices)
    covered = len(gt_important & selected_set)
    return round(covered / len(gt_important), 4)


def evaluate_video_summary(
    selected_indices: list[int],
    total_segments: int,
    features: Optional[np.ndarray] = None,
    ground_truth_scores: Optional[np.ndarray] = None,
    threshold_factor: float = 0.5,
) -> dict:
    """
    Compute all video summarization metrics.

    Args:
        selected_indices: segment indices chosen by the model
        total_segments: total video segments
        features: (N, D) feature vectors (for redundancy/diversity)
        ground_truth_scores: (N,) human importance annotations
        threshold_factor: threshold for determining "important" segments

    Returns:
        dict with all metrics
    """
    metrics = {
        "compression_ratio": compression_ratio(len(selected_indices), total_segments),
        "num_selected": len(selected_indices),
        "total_segments": total_segments,
    }

    if features is not None and len(features) > 0:
        metrics["redundancy_score"] = redundancy_score(features, selected_indices)
        metrics["diversity_score"] = diversity_score(features, selected_indices)

    if ground_truth_scores is not None:
        gt_threshold = ground_truth_scores.mean() + threshold_factor * ground_truth_scores.std()
        gt_important = list(np.where(ground_truth_scores >= gt_threshold)[0])

        f = f_score(selected_indices, gt_important, total_segments)
        metrics["f_score"] = f
        metrics["coverage"] = coverage_score(
            selected_indices, ground_truth_scores, threshold_factor
        )

    return metrics

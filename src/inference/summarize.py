"""
VIDEO -> VIDEO summarization pipeline.

Takes an input video, extracts ResNet18 features, predicts importance scores
via XGBoost (default) or Temporal Transformer, selects key scenes, and
produces a condensed summary video.

Models:
  - XGBoost: classical ML baseline (default, fast)
  - Temporal Transformer: deep learning with self-attention (if trained)

Run:
    python src/inference/summarize.py <video_path>
    python src/inference/summarize.py <video_path> --model transformer
"""

import os
import sys
import shutil
import subprocess
from typing import List, Tuple

# Ensure project root is on path when run as subprocess
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import joblib
import numpy as np
from PIL import Image

import torch

from src.config.config import (
    XGBOOST_MODEL_PATH, UPLOAD_DIR, TEMP_DIR, SUMMARY_DIR, DEVICE,
    SEGMENT_SECONDS, TEMPORAL_RADIUS, THRESHOLD_FACTOR, MAX_SUMMARY_RATIO,
    MIN_SUMMARY_SEGMENTS, CONTEXT_PAD, SIMILARITY_THRESHOLD, CROSSFADE_SECONDS,
    MODEL_DIR,
)

TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, "temporal_transformer.pt")
from src.features.video_features import (
    load_resnet, extract_feature_from_path, build_temporal_features,
)
from src.utils.helpers import ensure_dir, clean_dir


def resolve_video_path(video_input: str) -> str:
    if os.path.exists(video_input):
        return os.path.abspath(video_input)
    alt = os.path.join(UPLOAD_DIR, video_input)
    if os.path.exists(alt):
        return os.path.abspath(alt)
    raise FileNotFoundError(f"Video not found: {video_input}")


def extract_segment_frames(video_path: str, output_dir: str) -> List[List[str]]:
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Could not read FPS from {video_path}")

    frame_interval = max(1, int(fps * SEGMENT_SECONDS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segments: List[List[str]] = []
    segment_idx = 0

    while True:
        start_frame = segment_idx * frame_interval
        if start_frame >= total_frames:
            break
        sample_offsets = [0, frame_interval // 2, max(0, frame_interval - 1)]
        segment_paths: List[str] = []
        for sample_idx, offset in enumerate(sample_offsets):
            target_frame = min(start_frame + offset, max(0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_path = os.path.join(output_dir, f"segment_{segment_idx:04d}_{sample_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            segment_paths.append(frame_path)
        if segment_paths:
            segments.append(segment_paths)
        segment_idx += 1

    cap.release()
    return segments


def _score_with_xgboost(raw_features: np.ndarray) -> np.ndarray:
    """Score segments using XGBoost with temporal context features."""
    if not os.path.exists(XGBOOST_MODEL_PATH):
        raise FileNotFoundError(f"XGBoost model not found: {os.path.abspath(XGBOOST_MODEL_PATH)}")
    model = joblib.load(XGBOOST_MODEL_PATH)
    X = build_temporal_features(raw_features, radius=TEMPORAL_RADIUS)
    return model.predict(X).astype(np.float32)


def _score_with_transformer(raw_features: np.ndarray) -> np.ndarray:
    """Score segments using the Temporal Transformer with self-attention."""
    if not os.path.exists(TRANSFORMER_MODEL_PATH):
        raise FileNotFoundError(
            f"Transformer model not found: {os.path.abspath(TRANSFORMER_MODEL_PATH)}. "
            f"Train it first: python src/training/train_temporal_transformer.py"
        )

    from src.models.temporal_transformer import TemporalTransformer

    ckpt = torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})

    model = TemporalTransformer(
        input_dim=cfg.get("input_dim", 512),
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("num_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=0.0,  # No dropout at inference
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    feat_tensor = torch.tensor(raw_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, len(raw_features), dtype=torch.bool).to(DEVICE)

    with torch.no_grad():
        scores = model(feat_tensor, mask).squeeze(0).cpu().numpy()

    # Rescale from [0,1] to roughly [1,5] to match XGBoost output range
    scores = scores * 4.0 + 1.0
    return scores.astype(np.float32)


def extract_features_and_scores(video_path: str, scorer: str = "xgboost"):
    """
    Extract ResNet features and predict importance scores for each segment.

    Args:
        video_path: path to video file
        scorer: "xgboost" (default) or "transformer"
    """
    frames_dir = os.path.join(TEMP_DIR, "frames")
    clean_dir(frames_dir)

    try:
        frame_groups = extract_segment_frames(video_path, frames_dir)
        if not frame_groups:
            raise ValueError("No segment frames extracted.")

        resnet, transform = load_resnet()

        features = []
        for segment in frame_groups:
            seg_feats = [extract_feature_from_path(fp) for fp in segment]
            if seg_feats:
                features.append(np.array(seg_feats, dtype=np.float32).mean(axis=0))

        if not features:
            raise ValueError("No features extracted from segments.")

        raw_features = np.array(features, dtype=np.float32)

        if scorer == "transformer":
            scores = _score_with_transformer(raw_features)
        else:
            scores = _score_with_xgboost(raw_features)

        return raw_features, scores
    finally:
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)


def select_important_segments(scores: np.ndarray) -> List[int]:
    """Select segments above an adaptive importance threshold."""
    n = len(scores)
    if n == 0:
        return []

    mean = scores.mean()
    std = scores.std()
    threshold = mean + THRESHOLD_FACTOR * std
    selected = [i for i in range(n) if scores[i] >= threshold]

    if len(selected) < MIN_SUMMARY_SEGMENTS:
        ranked = np.argsort(scores)[::-1]
        selected = sorted(ranked[:MIN_SUMMARY_SEGMENTS].tolist())

    max_segments = max(MIN_SUMMARY_SEGMENTS, int(n * MAX_SUMMARY_RATIO))
    if len(selected) > max_segments:
        scored = [(i, scores[i]) for i in selected]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = sorted([i for i, _ in scored[:max_segments]])

    return selected


def group_into_scenes(indices: List[int], n_segments: int) -> List[Tuple[int, int]]:
    """Group consecutive selected indices into scene ranges with context padding."""
    if not indices:
        return []

    scenes = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx <= prev + 2:
            prev = idx
        else:
            scenes.append((start, prev))
            start = idx
            prev = idx
    scenes.append((start, prev))

    padded = []
    for s, e in scenes:
        s = max(0, s - CONTEXT_PAD)
        e = min(n_segments - 1, e + CONTEXT_PAD)
        padded.append((s, e))

    padded.sort()
    merged = [padded[0]]
    for s, e in padded[1:]:
        ls, le = merged[-1]
        if s <= le + 1:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))

    return merged


def deduplicate_scenes(scenes: List[Tuple[int, int]], features: np.ndarray) -> List[Tuple[int, int]]:
    """Remove scenes that are visually too similar to a previous scene."""
    if len(scenes) <= 1:
        return scenes

    scene_features = []
    for s, e in scenes:
        scene_feat = features[s:e + 1].mean(axis=0)
        norm = np.linalg.norm(scene_feat)
        if norm > 0:
            scene_feat = scene_feat / norm
        scene_features.append(scene_feat)

    kept = [0]
    for i in range(1, len(scenes)):
        is_duplicate = False
        for j in kept:
            sim = np.dot(scene_features[i], scene_features[j])
            if sim > SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(i)

    return [scenes[i] for i in kept]


def cut_scenes(video_path: str, scenes: List[Tuple[int, int]]) -> List[str]:
    """Cut each scene from the video as a separate clip."""
    segments_dir = os.path.join(TEMP_DIR, "segments")
    clean_dir(segments_dir)

    output_files = []
    for i, (start, end) in enumerate(scenes):
        start_time = start * SEGMENT_SECONDS
        duration = (end - start + 1) * SEGMENT_SECONDS
        output_path = os.path.join(segments_dir, f"scene_{i:04d}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        output_files.append(output_path)

    return output_files


def get_clip_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def concat_with_crossfade(clip_files: List[str], output_path: str) -> None:
    """Concatenate clips with crossfade transitions between them."""
    if not clip_files:
        raise ValueError("No clips to concatenate.")

    if len(clip_files) == 1:
        shutil.copy(clip_files[0], output_path)
        return

    durations = [get_clip_duration(f) for f in clip_files]
    fade = min(CROSSFADE_SECONDS, min(durations) / 2)

    inputs = []
    for f in clip_files:
        inputs.extend(["-i", f])

    video_filters = []
    audio_filters = []

    offset = durations[0] - fade
    video_filters.append(f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={offset}[v01]")
    audio_filters.append(f"[0:a][1:a]acrossfade=d={fade}[a01]")
    cumulative_duration = durations[0] + durations[1] - fade

    for i in range(2, len(clip_files)):
        prev_v = f"v{0}{i - 1}" if i == 2 else f"vx{i - 1}"
        prev_a = f"a{0}{i - 1}" if i == 2 else f"ax{i - 1}"
        out_v = f"vx{i}" if i < len(clip_files) - 1 else "vout"
        out_a = f"ax{i}" if i < len(clip_files) - 1 else "aout"

        offset = cumulative_duration - fade
        video_filters.append(f"[{prev_v}][{i}:v]xfade=transition=fade:duration={fade}:offset={offset}[{out_v}]")
        audio_filters.append(f"[{prev_a}][{i}:a]acrossfade=d={fade}[{out_a}]")
        cumulative_duration = cumulative_duration + durations[i] - fade

    if len(clip_files) == 2:
        final_v, final_a = "v01", "a01"
    else:
        final_v, final_a = "vout", "aout"

    filter_complex = ";".join(video_filters + audio_filters)

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{final_v}]", "-map", f"[{final_a}]",
        "-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Crossfade failed, falling back to simple concat: {result.stderr[-200:]}")
        concat_simple(clip_files, output_path)


def concat_simple(clip_files: List[str], output_path: str) -> None:
    """Fallback: simple concat without transitions."""
    concat_file = os.path.join(TEMP_DIR, "concat.txt")
    with open(concat_file, "w") as f:
        for clip in clip_files:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_file, "-c:v", "libx264", "-c:a", "aac",
         output_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def summarize_video(video_input: str, scorer: str = "xgboost") -> str:
    ensure_dir(SUMMARY_DIR)
    ensure_dir(TEMP_DIR)

    video_path = resolve_video_path(video_input)
    video_filename = os.path.basename(video_path)

    print(f"Processing: {video_path}")
    print(f"Scorer: {scorer}")

    try:
        # 1. extract features and predict importance
        features, scores = extract_features_and_scores(video_path, scorer=scorer)
        n_segments = len(scores)

        print(f"Segments: {n_segments} ({n_segments * SEGMENT_SECONDS}s video)")
        print(f"Scores — min: {scores.min():.2f}, max: {scores.max():.2f}, "
              f"mean: {scores.mean():.2f}, std: {scores.std():.2f}")

        # 2. select important segments
        selected = select_important_segments(scores)
        print(f"Selected {len(selected)} important segments")

        # 3. group into scenes
        scenes = group_into_scenes(selected, n_segments)
        print(f"Grouped into {len(scenes)} scenes: {scenes}")

        # 4. deduplicate
        scenes = deduplicate_scenes(scenes, features)
        print(f"After dedup: {len(scenes)} scenes: {scenes}")

        if not scenes:
            raise ValueError("No scenes selected for summary.")

        total_summary_segments = sum(e - s + 1 for s, e in scenes)
        print(f"Summary duration: ~{total_summary_segments * SEGMENT_SECONDS}s "
              f"({total_summary_segments}/{n_segments} segments, "
              f"{total_summary_segments / n_segments * 100:.0f}%)")

        # 5. cut scenes
        clip_files = cut_scenes(video_path, scenes)

        # 6. concatenate with crossfade
        output_name = os.path.splitext(video_filename)[0] + "_summary.mp4"
        output_path = os.path.join(SUMMARY_DIR, output_name)
        concat_with_crossfade(clip_files, output_path)

        # Compute inference metrics
        score_mean = float(scores.mean())
        score_std = float(scores.std())
        threshold = score_mean + THRESHOLD_FACTOR * score_std
        selected_ratio = len(selected) / n_segments
        summary_ratio = total_summary_segments / n_segments

        metrics = {
            "scorer_model": scorer,
            "mse_proxy": round(float(score_std ** 2), 4),
            "r2_confidence": round(1.0 - (score_std / max(score_mean, 0.01)), 4),
            "mean_importance": round(score_mean, 4),
            "score_std": round(score_std, 4),
            "threshold": round(threshold, 4),
            "selected_segments": len(selected),
            "total_segments": n_segments,
            "compression_ratio": round(summary_ratio, 3),
            "scenes_after_dedup": len(scenes),
        }

        # Save metrics to a JSON file alongside the video
        metrics_path = output_path.replace(".mp4", "_metrics.json")
        import json as _json
        with open(metrics_path, "w") as f:
            _json.dump(metrics, f)

        print(f"Summary saved to: {output_path}")
        return output_path

    finally:
        # Clean up temp segments
        segments_dir = os.path.join(TEMP_DIR, "segments")
        if os.path.exists(segments_dir):
            shutil.rmtree(segments_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video-to-Video summarization")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--model", choices=["xgboost", "transformer"], default="xgboost",
                        help="Scoring model: xgboost (default) or transformer")
    args = parser.parse_args()
    summarize_video(args.video_path, scorer=args.model)

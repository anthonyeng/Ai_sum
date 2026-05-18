"""
Multimodal feature fusion module.

Combines multiple signal sources into a single feature vector per video segment:
  - Visual: ResNet18 features (512-dim) — for trained models
  - Visual: CLIP ViT-B/32 features (512-dim) — semantic visual understanding
  - Audio: Energy, spectral centroid, ZCR, energy variance (4-dim)
  - Text: Sentence-BERT transcript embeddings (384-dim)
  - Scene: Scene-change score per segment (1-dim)
  - Motion: Inter-frame motion magnitude (1-dim)

Two fusion modes:
  1. "concat" — concatenate all features into one long vector
  2. "scored" — produce a single importance score from 0 to 1

The concat mode is for training new models (Temporal Transformer).
The scored mode is for direct segment ranking without additional models.

Usage:
    from src.features.multimodal_fusion import extract_multimodal_features, score_segments
    features = extract_multimodal_features(video_path, transcript_chunks)
    scores = score_segments(features)
"""

import numpy as np
import cv2
from typing import Optional

from src.config.config import SEGMENT_SECONDS, DEVICE


def compute_scene_change_scores(video_path: str, segment_seconds: int = None) -> np.ndarray:
    """
    Compute histogram-difference scene-change score per segment.

    Higher score = more visual change from previous segment = likely a scene boundary.

    Returns:
        np.ndarray of shape (num_segments, 1), normalized to [0, 1]
    """
    if segment_seconds is None:
        segment_seconds = SEGMENT_SECONDS

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return np.empty((0, 1), dtype=np.float32)

    frame_interval = max(1, int(fps * segment_seconds))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    seg_idx = 0
    while True:
        start_frame = seg_idx * frame_interval
        if start_frame >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        seg_idx += 1

    cap.release()

    if len(frames) < 2:
        return np.zeros((max(1, len(frames)), 1), dtype=np.float32)

    scores = [0.0]  # First segment has no previous frame
    for i in range(1, len(frames)):
        diff = 0.0
        for ch in range(3):
            h1 = cv2.calcHist([frames[i - 1]], [ch], None, [64], [0, 256]).flatten()
            h2 = cv2.calcHist([frames[i]], [ch], None, [64], [0, 256]).flatten()
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            diff += np.sum(np.abs(h1 - h2))
        scores.append(diff / 3.0)

    scores = np.array(scores, dtype=np.float32)
    max_score = scores.max()
    if max_score > 0:
        scores /= max_score

    return scores.reshape(-1, 1)


def compute_motion_scores(video_path: str, segment_seconds: int = None) -> np.ndarray:
    """
    Compute inter-frame motion magnitude per segment using optical flow.

    Higher score = more visual motion = more dynamic content.

    Returns:
        np.ndarray of shape (num_segments, 1), normalized to [0, 1]
    """
    if segment_seconds is None:
        segment_seconds = SEGMENT_SECONDS

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return np.empty((0, 1), dtype=np.float32)

    frame_interval = max(1, int(fps * segment_seconds))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gray_frames = []

    seg_idx = 0
    while True:
        start_frame = seg_idx * frame_interval
        if start_frame >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downsample for speed
        gray = cv2.resize(gray, (160, 120))
        gray_frames.append(gray)
        seg_idx += 1

    cap.release()

    if len(gray_frames) < 2:
        return np.zeros((max(1, len(gray_frames)), 1), dtype=np.float32)

    scores = [0.0]
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
        motion = np.mean(diff) / 255.0
        scores.append(motion)

    scores = np.array(scores, dtype=np.float32)
    max_score = scores.max()
    if max_score > 0:
        scores /= max_score

    return scores.reshape(-1, 1)


def extract_multimodal_features(
    video_path: str,
    transcript_chunks: Optional[list[dict]] = None,
    use_clip: bool = True,
    use_audio: bool = True,
    segment_seconds: int = None,
) -> dict:
    """
    Extract all feature modalities for a video.

    Args:
        video_path: path to video file
        transcript_chunks: list of {"text": str, "start": float, "end": float}
        use_clip: whether to extract CLIP features (slower but semantic)
        use_audio: whether to extract audio features
        segment_seconds: segment duration

    Returns:
        dict with keys:
          - "resnet": (N, 512) ResNet18 features
          - "clip": (N, 512) CLIP features (if use_clip)
          - "audio": (N, 4) audio features (if use_audio)
          - "text": (N, 384) Sentence-BERT embeddings (if transcript provided)
          - "scene_change": (N, 1) scene-change scores
          - "motion": (N, 1) motion scores
          - "num_segments": int
          - "fused": (N, D) concatenated multimodal feature vector
    """
    if segment_seconds is None:
        segment_seconds = SEGMENT_SECONDS

    result = {}

    # 1. ResNet18 features (always — needed for trained models)
    from src.features.video_features import extract_video_segment_features
    resnet_feats = extract_video_segment_features(video_path, segment_seconds)
    num_segments = len(resnet_feats)
    result["resnet"] = resnet_feats
    result["num_segments"] = num_segments

    if num_segments == 0:
        result["fused"] = np.empty((0, 0), dtype=np.float32)
        return result

    # 2. CLIP features (parallel stream)
    if use_clip:
        try:
            from src.features.clip_embeddings import extract_clip_features
            clip_feats = extract_clip_features(video_path, segment_seconds)
            # Align lengths
            clip_feats = _align_length(clip_feats, num_segments, 512)
            result["clip"] = clip_feats
        except (ImportError, Exception):
            result["clip"] = None

    # 3. Audio features
    if use_audio:
        try:
            from src.features.audio_features import extract_audio_features
            audio_feats = extract_audio_features(video_path, segment_seconds)
            audio_feats = _align_length(audio_feats, num_segments, 4)
            result["audio"] = audio_feats
        except Exception:
            result["audio"] = None

    # 4. Text embeddings (from transcript chunks)
    if transcript_chunks:
        try:
            from src.features.text_embeddings import encode_texts
            # Map transcript chunks to segments by timestamp
            text_per_segment = _map_text_to_segments(
                transcript_chunks, num_segments, segment_seconds
            )
            text_feats = encode_texts(text_per_segment)
            text_feats = _align_length(text_feats, num_segments, 384)
            result["text"] = text_feats
        except Exception:
            result["text"] = None
    else:
        result["text"] = None

    # 5. Scene-change scores
    scene_scores = compute_scene_change_scores(video_path, segment_seconds)
    scene_scores = _align_length(scene_scores, num_segments, 1)
    result["scene_change"] = scene_scores

    # 6. Motion scores
    motion_scores = compute_motion_scores(video_path, segment_seconds)
    motion_scores = _align_length(motion_scores, num_segments, 1)
    result["motion"] = motion_scores

    # 7. Fused vector — concatenate all available modalities
    parts = [resnet_feats]  # Always include ResNet
    if result.get("clip") is not None:
        parts.append(result["clip"])
    if result.get("audio") is not None:
        parts.append(result["audio"])
    if result.get("text") is not None:
        parts.append(result["text"])
    parts.append(scene_scores)
    parts.append(motion_scores)

    result["fused"] = np.concatenate(parts, axis=1).astype(np.float32)
    result["fused_dim"] = result["fused"].shape[1]

    return result


def score_segments(features: dict, weights: Optional[dict] = None) -> np.ndarray:
    """
    Compute a weighted importance score per segment from multimodal features.

    This is a simple heuristic scorer — for production use, train a model
    on the fused features instead.

    Args:
        features: dict from extract_multimodal_features()
        weights: optional weight overrides

    Returns:
        np.ndarray of shape (N,) — importance score per segment
    """
    if weights is None:
        weights = {
            "audio_energy": 0.25,
            "scene_change": 0.20,
            "motion": 0.15,
            "text_density": 0.20,
            "visual_variance": 0.20,
        }

    n = features["num_segments"]
    if n == 0:
        return np.array([], dtype=np.float32)

    scores = np.zeros(n, dtype=np.float32)

    # Audio energy contributes to importance
    if features.get("audio") is not None:
        audio_energy = features["audio"][:, 0]  # RMS energy column
        scores += weights["audio_energy"] * audio_energy

    # Scene changes mark segment boundaries (important moments)
    if features.get("scene_change") is not None:
        scores += weights["scene_change"] * features["scene_change"].squeeze()

    # Motion indicates dynamic/interesting content
    if features.get("motion") is not None:
        scores += weights["motion"] * features["motion"].squeeze()

    # Segments with speech content are more important
    if features.get("text") is not None:
        # Text density = norm of text embedding (higher = more content)
        text_norms = np.linalg.norm(features["text"], axis=1)
        text_norms_max = text_norms.max()
        if text_norms_max > 0:
            text_norms /= text_norms_max
        scores += weights["text_density"] * text_norms

    # Visual variance within ResNet features (diverse content = interesting)
    resnet = features["resnet"]
    mean_feat = resnet.mean(axis=0, keepdims=True)
    visual_dist = np.linalg.norm(resnet - mean_feat, axis=1)
    visual_dist_max = visual_dist.max()
    if visual_dist_max > 0:
        visual_dist /= visual_dist_max
    scores += weights["visual_variance"] * visual_dist

    # Normalize to [0, 1]
    score_max = scores.max()
    if score_max > 0:
        scores /= score_max

    return scores


def _align_length(arr: np.ndarray, target_len: int, feat_dim: int) -> np.ndarray:
    """Pad or truncate array to match target number of segments."""
    if len(arr) == 0:
        return np.zeros((target_len, feat_dim), dtype=np.float32)
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    # Pad with zeros
    pad = np.zeros((target_len - len(arr), feat_dim), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def _map_text_to_segments(
    chunks: list[dict], num_segments: int, segment_seconds: int
) -> list[str]:
    """Map transcript chunks to video segments by timestamp."""
    segment_texts = [""] * num_segments

    for chunk in chunks:
        start = chunk.get("start", 0.0)
        end = chunk.get("end", start + 1.0)
        text = chunk.get("text", "")

        seg_start = int(start / segment_seconds)
        seg_end = int(end / segment_seconds) + 1

        for s in range(max(0, seg_start), min(num_segments, seg_end)):
            if segment_texts[s]:
                segment_texts[s] += " " + text
            else:
                segment_texts[s] = text

    # Fill empty segments with a placeholder
    for i in range(num_segments):
        if not segment_texts[i]:
            segment_texts[i] = "[no speech]"

    return segment_texts

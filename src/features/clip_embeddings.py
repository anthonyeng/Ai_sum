"""
CLIP ViT-B/32 visual embedding extractor (pretrained, no training needed).

Extracts 512-dim semantic visual embeddings from video frames using OpenAI's
CLIP model. Unlike ResNet18 (texture/pattern features), CLIP understands
semantic content — it knows "a person explaining on a whiteboard" vs
"a transition slide".

Used for:
  - Multimodal feature fusion (new modules)
  - Query-focused summarization (semantic search over visual content)
  - Scene semantic similarity

ResNet18 is KEPT for all trained models (XGBoost, BiLSTM, Seq2Seq caption).
CLIP runs as a parallel feature stream — no retraining required.

Usage:
    from src.features.clip_embeddings import extract_clip_features, encode_text_query
    visual_feats = extract_clip_features(video_path)       # (N, 512)
    query_feat = encode_text_query("person at whiteboard")  # (512,)
"""

import numpy as np
import cv2
from PIL import Image

import torch

from src.config.config import DEVICE, SEGMENT_SECONDS

# Lazy-loaded globals
_clip_model = None
_clip_preprocess = None
_clip_tokenize = None
_CLIP_DIM = 512


def _load_clip():
    """Load CLIP ViT-B/32 model (lazy, cached)."""
    global _clip_model, _clip_preprocess, _clip_tokenize
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_tokenize

    try:
        import clip
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        _clip_tokenize = clip.tokenize
        _clip_model.eval()
    except ImportError:
        raise ImportError(
            "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
        )

    return _clip_model, _clip_preprocess, _clip_tokenize


def extract_clip_feature_from_image(image: Image.Image) -> np.ndarray:
    """Extract a 512-dim CLIP feature vector from a PIL image."""
    model, preprocess, _ = _load_clip()
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 normalize
    return feat.squeeze().cpu().numpy().astype(np.float32)


def extract_clip_features(video_path: str, segment_seconds: int = None) -> np.ndarray:
    """
    Extract CLIP ViT-B/32 features for every segment of a video.

    Args:
        video_path: path to video file
        segment_seconds: seconds per segment (default: config SEGMENT_SECONDS)

    Returns:
        np.ndarray of shape (num_segments, 512)
    """
    if segment_seconds is None:
        segment_seconds = SEGMENT_SECONDS

    model, preprocess, _ = _load_clip()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return np.empty((0, _CLIP_DIM), dtype=np.float32)

    frame_interval = max(1, int(fps * segment_seconds))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features = []

    segment_idx = 0
    while True:
        start_frame = segment_idx * frame_interval
        if start_frame >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        features.append(feat.squeeze().cpu().numpy().astype(np.float32))
        segment_idx += 1

    cap.release()
    return np.array(features, dtype=np.float32) if features else np.empty((0, _CLIP_DIM), dtype=np.float32)


def encode_text_query(query: str) -> np.ndarray:
    """
    Encode a text query into CLIP's shared embedding space.

    This enables semantic search: compare this vector against visual features
    to find which video segments match the query.

    Args:
        query: natural language query (e.g., "person writing on whiteboard")

    Returns:
        np.ndarray of shape (512,) — L2-normalized
    """
    model, _, tokenize = _load_clip()
    tokens = tokenize([query]).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze().cpu().numpy().astype(np.float32)


def clip_similarity(visual_features: np.ndarray, text_feature: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between visual segment features and a text query.

    Args:
        visual_features: (N, 512) CLIP visual features
        text_feature: (512,) CLIP text feature

    Returns:
        np.ndarray of shape (N,) — similarity scores per segment
    """
    # Already L2-normalized, so dot product = cosine similarity
    return (visual_features @ text_feature).astype(np.float32)


def get_clip_dim() -> int:
    """Return CLIP embedding dimension."""
    return _CLIP_DIM

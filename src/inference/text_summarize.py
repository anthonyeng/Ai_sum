"""
VIDEO -> TEXT summarization pipeline.

Uses the trained VideoCaptionModel (seq2seq with Bahdanau attention,
trained on MSR-VTT) to generate captions. No LLMs, no API keys.

Run:
    python src/inference/text_summarize.py /path/to/video.mp4
"""

import json
import os
import pickle
import sys

# Ensure project root is on path when run as subprocess
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np
import torch
from PIL import Image

from src.config.config import (
    CAPTION_MODEL_PATH, CAPTION_VOCAB_PATH, DEVICE, BASE_DIR,
    NUM_KEYFRAMES, SCENE_THRESHOLD, TEXT_SIMILARITY_THRESHOLD,
    SEGMENT_SECONDS, MAX_CAPTION_LEN, CAPTION_MAX_SEQ_LEN,
)
from src.features.video_features import load_resnet


# ── Trained caption model loader ─────────────────────────────────────────────

_caption_model = None
_caption_vocab = None


def _load_caption_model():
    global _caption_model, _caption_vocab
    if _caption_model is not None:
        return _caption_model, _caption_vocab

    if not os.path.exists(CAPTION_MODEL_PATH):
        raise FileNotFoundError(
            f"Caption model not found: {CAPTION_MODEL_PATH}\n"
            "Train it first using the Colab notebook or src/training/train_caption.py"
        )
    if not os.path.exists(CAPTION_VOCAB_PATH):
        raise FileNotFoundError(
            f"Caption vocabulary not found: {CAPTION_VOCAB_PATH}\n"
            "Build it first using src/data/build_caption_dataset.py"
        )

    # Ensure project root is on path so pickle can resolve src.data.vocabulary
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    with open(CAPTION_VOCAB_PATH, "rb") as f:
        _caption_vocab = pickle.load(f)

    from src.models.caption_model import VideoCaptionModel

    ckpt = torch.load(CAPTION_MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})

    _caption_model = VideoCaptionModel(
        vocab_size=ckpt.get("vocab_size", len(_caption_vocab)),
        embed_dim=cfg.get("embed_dim", 256),
        encoder_hidden=cfg.get("encoder_hidden", 512),
        decoder_hidden=cfg.get("decoder_hidden", 512),
        attention_dim=cfg.get("attention_dim", 256),
        input_dim=cfg.get("input_dim", 512),
        encoder_layers=cfg.get("encoder_layers", 2),
        dropout=0.0,
    )
    _caption_model.load_state_dict(ckpt["model_state"])
    _caption_model.to(DEVICE)
    _caption_model.eval()
    return _caption_model, _caption_vocab


# ── Frame extraction with scene-change detection ────────────────────────────

def _hist_diff(frame_a, frame_b):
    diff = 0.0
    for ch in range(3):
        h1 = cv2.calcHist([frame_a], [ch], None, [64], [0, 256]).flatten()
        h2 = cv2.calcHist([frame_b], [ch], None, [64], [0, 256]).flatten()
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        diff += np.sum(np.abs(h1 - h2))
    return diff / 3.0


def _extract_frames_and_features(video_path):
    """Extract frames for scene-change selection AND 2-sec segment features
    for the trained caption model."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    resnet, transform = load_resnet()

    # 1) Extract ResNet18 features for every 2-sec segment
    seg_step = int(fps * SEGMENT_SECONDS)
    segment_features = []
    for pos in range(0, total, seg_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = resnet(tensor).squeeze().cpu().numpy()
        segment_features.append(feat)

    # 2) Scene-change detection for key frames
    scene_step = int(fps * 4)
    candidates = []
    for pos in range(0, total, scene_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            candidates.append((pos, frame))
    cap.release()

    if not candidates:
        return [], [], duration, np.array(segment_features, dtype=np.float32)

    selected = [candidates[0]]
    for i in range(1, len(candidates)):
        diff = _hist_diff(selected[-1][1], candidates[i][1])
        if diff > SCENE_THRESHOLD:
            selected.append(candidates[i])

    if len(selected) < 3:
        n = min(NUM_KEYFRAMES, len(candidates))
        indices = np.linspace(0, len(candidates) - 1, n, dtype=int)
        selected = [candidates[int(i)] for i in indices]

    if len(selected) > NUM_KEYFRAMES:
        indices = np.linspace(0, len(selected) - 1, NUM_KEYFRAMES, dtype=int)
        selected = [selected[int(i)] for i in indices]

    frames, timestamps = [], []
    for pos, raw in selected:
        frames.append(Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)))
        m, s = divmod(int(pos / fps), 60)
        timestamps.append(f"{m}:{s:02d}")

    seg_feats = np.array(segment_features, dtype=np.float32) if segment_features else np.empty((0, 512))
    return frames, timestamps, duration, seg_feats


# ── Caption generation using trained model ───────────────────────────────────

def _generate_caption(model, vocab, features):
    """Generate a caption from segment features using the trained seq2seq model."""
    max_seq = CAPTION_MAX_SEQ_LEN
    n = features.shape[0]

    if n > max_seq:
        indices = np.linspace(0, n - 1, max_seq, dtype=int)
        features = features[indices]
    elif n < max_seq:
        pad = np.zeros((max_seq - n, features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)

    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(feat_tensor)
        hidden = hidden.unsqueeze(0)

        sos_idx = vocab.word2idx.get("<sos>", 1)
        eos_idx = vocab.word2idx.get("<eos>", 2)

        input_token = torch.tensor([sos_idx], device=DEVICE)
        decoded_ids = []

        for _ in range(MAX_CAPTION_LEN):
            pred, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
            next_id = pred.argmax(1).item()
            if next_id == eos_idx:
                break
            decoded_ids.append(next_id)
            input_token = torch.tensor([next_id], device=DEVICE)

    return vocab.decode(decoded_ids)


def _generate_segment_captions(model, vocab, all_features, timestamps):
    """Generate captions for each key-moment time window using the caption model."""
    fps_approx = 1.0 / SEGMENT_SECONDS
    captions = []

    for i, ts in enumerate(timestamps):
        parts = ts.split(":")
        sec = int(parts[0]) * 60 + int(parts[1])

        center_seg = int(sec * fps_approx)
        window = 15
        start = max(0, center_seg - window)
        end = min(len(all_features), center_seg + window)

        if start >= len(all_features):
            start = max(0, len(all_features) - 5)
            end = len(all_features)

        window_feats = all_features[start:end]
        if len(window_feats) == 0:
            captions.append(f"Scene {i + 1}")
            continue

        caption = _generate_caption(model, vocab, window_feats)
        captions.append(caption if caption else f"Scene {i + 1}")

    return captions


# ── Deduplication ────────────────────────────────────────────────────────────

def _word_set_similarity(a, b):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _build_summary(captions):
    seen_exact, deduped = set(), []
    for c in captions:
        key = c.strip().lower()
        if key and key not in seen_exact:
            seen_exact.add(key)
            deduped.append(c.strip())

    if len(deduped) > 1:
        unique = [deduped[0]]
        for i in range(1, len(deduped)):
            sims = [_word_set_similarity(deduped[i], u) for u in unique]
            if max(sims) < TEXT_SIMILARITY_THRESHOLD:
                unique.append(deduped[i])
        deduped = unique

    result = [c[0].upper() + c[1:] for c in deduped if c]
    return ". ".join(result) + "." if result else ""


# ── Schema builder ──────────────────────────────────────────────────────────

def _build_schema(captions, timestamps, duration, video_name):
    summary = _build_summary(captions)

    first_words = captions[0].strip().split() if captions else summary.split()
    title = (
        " ".join(first_words[:6]).title()
        if len(first_words) >= 3
        else os.path.basename(video_name)
    )

    key_moments = [
        {
            "timestamp": timestamps[i],
            "label": captions[i].strip().capitalize() or f"Scene {i + 1}",
        }
        for i in range(len(captions))
    ]

    n = len(captions)
    observe = captions[0].strip() if n > 0 else summary
    orient = captions[1].strip() if n > 1 else "—"
    decide = captions[int(n * 0.5)].strip() if n > 2 else "—"
    act = captions[-1].strip() if n > 3 else "—"

    return {
        "title": title,
        "summary": summary,
        "ooda": {
            "observe": observe,
            "orient": orient,
            "decide": decide,
            "act": act,
        },
        "key_moments": key_moments,
        "stats": {
            "scene_count": n,
            "duration_analyzed_sec": round(duration, 1),
            "segments_per_sec": round(n / max(duration, 1), 2),
        },
        "video_file": os.path.basename(video_name),
    }


# ── Public API ──────────────────────────────────────────────────────────────

def text_summarize_video(video_path: str) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames, timestamps, duration, seg_features = _extract_frames_and_features(video_path)

    if not frames:
        raise ValueError("No frames could be extracted from video.")

    # Load trained caption model (required — no fallback LLMs)
    caption_model, vocab = _load_caption_model()

    if len(seg_features) == 0:
        raise ValueError("No segment features extracted from video.")

    # Generate captions using trained seq2seq model
    captions = _generate_segment_captions(caption_model, vocab, seg_features, timestamps)

    return _build_schema(captions, timestamps, duration, video_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference/text_summarize.py <video_path>")
        sys.exit(1)
    result = text_summarize_video(sys.argv[1])
    print(json.dumps(result, indent=2))

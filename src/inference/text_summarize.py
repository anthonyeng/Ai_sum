"""
Generate a structured text summary from a video using BLIP captioning
and Whisper audio transcription (both pretrained, fully local — no API
key needed).

Run:
    python src/inference/text_summarize.py /path/to/video.mp4
"""

import json
import os
import sys

# Disable TF/Flax so transformers doesn't try to import the broken TF install
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

NUM_FRAMES = 8
SCENE_THRESHOLD = 30  # histogram diff threshold for scene-change detection
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
SIMILARITY_THRESHOLD = 0.85  # cosine similarity cutoff for semantic dedup


# ── BLIP loader (cached after first call) ─────────────────────────────────────

_blip_processor = None
_blip_model = None


def _load_blip():
    global _blip_processor, _blip_model
    if _blip_model is None:
        _blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
        _blip_model.eval()
    return _blip_processor, _blip_model


# ── Whisper loader (cached) ──────────────────────────────────────────────────

_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _whisper_model


# ── Frame extraction with scene-change detection ────────────────────────────

def _hist_diff(frame_a, frame_b):
    """Mean absolute histogram difference across BGR channels."""
    diff = 0.0
    for ch in range(3):
        h1 = cv2.calcHist([frame_a], [ch], None, [64], [0, 256]).flatten()
        h2 = cv2.calcHist([frame_b], [ch], None, [64], [0, 256]).flatten()
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        diff += np.sum(np.abs(h1 - h2))
    return diff / 3.0


def _extract_frames(video_path):
    """Extract visually diverse frames using scene-change detection."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    # Sample candidate frames every 4 seconds
    step = int(fps * 4)
    candidates = []
    for pos in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            candidates.append((pos, frame))
    cap.release()

    if not candidates:
        return [], [], duration

    # Always keep the first frame, then pick frames with enough visual change
    selected = [candidates[0]]
    for i in range(1, len(candidates)):
        diff = _hist_diff(selected[-1][1], candidates[i][1])
        if diff > SCENE_THRESHOLD:
            selected.append(candidates[i])

    # If scene detection gave too few, fall back to evenly spaced
    if len(selected) < 3:
        n = min(NUM_FRAMES, len(candidates))
        indices = np.linspace(0, len(candidates) - 1, n, dtype=int)
        selected = [candidates[i] for i in indices]

    # Cap at NUM_FRAMES, evenly spaced from selected
    if len(selected) > NUM_FRAMES:
        indices = np.linspace(0, len(selected) - 1, NUM_FRAMES, dtype=int)
        selected = [selected[i] for i in indices]

    frames, timestamps = [], []
    for pos, raw in selected:
        frames.append(Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)))
        m, s = divmod(int(pos / fps), 60)
        timestamps.append(f"{m}:{s:02d}")

    return frames, timestamps, duration


# ── Audio transcription ──────────────────────────────────────────────────────

def _transcribe_audio(video_path):
    """Return transcribed text from the video's audio track using Whisper."""
    try:
        model = _load_whisper()
        segments, _ = model.transcribe(video_path, beam_size=3)
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()
    except Exception:
        return ""


# ── BLIP captioning ──────────────────────────────────────────────────────────

def _caption_frame(processor, model, image):
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=40, num_beams=3)
    return processor.decode(ids[0], skip_special_tokens=True).strip()


def _word_set_similarity(a, b):
    """Simple word-overlap similarity (Jaccard) — fast, no model needed."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ── Schema builder ───────────────────────────────────────────────────────────

def _build_summary(captions, processor=None, model=None):
    """Deduplicate (exact + semantic) and join per-frame captions."""
    # 1) Exact dedup (case-insensitive)
    seen_exact, deduped = set(), []
    for c in captions:
        key = c.strip().lower()
        if key and key not in seen_exact:
            seen_exact.add(key)
            deduped.append(c.strip())

    # 2) Semantic dedup via word-overlap similarity
    if len(deduped) > 1:
        unique = [deduped[0]]
        for i in range(1, len(deduped)):
            sims = [_word_set_similarity(deduped[i], u) for u in unique]
            if max(sims) < SIMILARITY_THRESHOLD:
                unique.append(deduped[i])
        deduped = unique

    result = []
    for c in deduped:
        result.append(c[0].upper() + c[1:])
    return ". ".join(result) + "." if result else ""


def _build_schema(captions, timestamps, duration, video_name, transcript=""):
    summary = _build_summary(captions)

    # Incorporate transcript into summary if available
    if transcript:
        summary = f"{summary}\n\nTranscript: {transcript}"

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
    observe = captions[0].strip()            if n > 0 else summary
    orient  = captions[1].strip()            if n > 1 else "—"
    decide  = captions[int(n * 0.5)].strip() if n > 2 else "—"
    act     = captions[-1].strip()           if n > 3 else "—"

    return {
        "title": title,
        "summary": summary,
        "transcript": transcript if transcript else None,
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


# ── Public API ───────────────────────────────────────────────────────────────

def text_summarize_video(video_path: str) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    processor, model = _load_blip()
    frames, timestamps, duration = _extract_frames(video_path)

    if not frames:
        raise ValueError("No frames could be extracted from video.")

    captions = [_caption_frame(processor, model, img) for img in frames]
    transcript = _transcribe_audio(video_path)

    return _build_schema(captions, timestamps, duration, video_path,
                         transcript=transcript)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference/text_summarize.py <video_path>")
        sys.exit(1)
    result = text_summarize_video(sys.argv[1])
    print(json.dumps(result, indent=2))

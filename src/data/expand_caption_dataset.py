"""
Expand the caption dataset from 1 caption/video to 20 captions/video.

Uses existing extracted features from dataset.pkl and pairs them with
all 20 captions from MSRVTT_data.json.

Run:
    python src/data/expand_caption_dataset.py
"""

import json
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.vocabulary import Vocabulary

EXISTING_DATASET = "data/processed/msvtt/dataset.pkl"
FULL_ANNO_FILE = "data/raw/msvtt/MSRVTT_data.json"
OUTPUT_DIR = "data/processed/msvtt"
VOCAB_PATH = "outputs/models/caption_vocab.pkl"

MAX_CAPTION_LEN = 30
MIN_WORD_FREQ = 2


def main():
    print("Loading existing dataset (features)...")
    with open(EXISTING_DATASET, "rb") as f:
        existing = pickle.load(f)

    # Build feature lookup: video_id → features array
    feature_map = {d["video_id"]: d["features"] for d in existing}
    print(f"  Videos with extracted features: {len(feature_map)}")

    print("Loading full MSRVTT annotations (200k captions)...")
    with open(FULL_ANNO_FILE, "r") as f:
        data = json.load(f)

    # Group all 20 captions per video, filtering to only videos we have features for
    video_captions: dict = {}
    for sentence in data["sentences"]:
        vid = sentence["video_id"]
        if vid in feature_map:
            video_captions.setdefault(vid, []).append(sentence["caption"].strip())

    total_videos = len(video_captions)
    total_captions = sum(len(v) for v in video_captions.values())
    print(f"  Videos matched: {total_videos}")
    print(f"  Total captions: {total_captions}")
    print(f"  Avg captions/video: {total_captions / max(total_videos, 1):.1f}")

    # Build vocabulary from all matched captions
    print("\nBuilding vocabulary...")
    all_captions = [c for caps in video_captions.values() for c in caps]
    vocab = Vocabulary()
    vocab.build(all_captions, min_freq=MIN_WORD_FREQ)
    print(f"  Vocab size: {len(vocab)}")

    os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    print(f"  Vocab saved → {VOCAB_PATH}")

    # Build expanded dataset: features × 20 captions each
    print("\nBuilding expanded dataset...")
    dataset = []
    for video_id, captions in video_captions.items():
        features = feature_map[video_id]
        for caption in captions:
            encoded = vocab.encode(caption, max_len=MAX_CAPTION_LEN)
            dataset.append({
                "video_id": video_id,
                "features": features,
                "caption": encoded,
                "caption_text": caption,
            })

    print(f"  Total samples: {len(dataset)}")

    save_path = os.path.join(OUTPUT_DIR, "dataset.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"  Dataset saved → {save_path}")
    print(f"\nDone! Expanded from {len(existing)} → {len(dataset)} samples")


if __name__ == "__main__":
    main()

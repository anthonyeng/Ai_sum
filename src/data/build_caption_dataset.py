"""
Build the MSR-VTT caption dataset.

Expected folder layout after downloading MSR-VTT:
    data/raw/msvtt/
        videos/                  ← video7975.mp4, video0000.mp4, ...
        train_val_videodatainfo.json

Run:
    python src/data/build_caption_dataset.py
"""

import os
import json
import pickle
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as transforms

MSVTT_VIDEO_DIR = "data/raw/msvtt/TrainValVideo"
MSVTT_ANNO_FILE = "data/raw/msvtt/train_val_videodatainfo.json"
OUTPUT_DIR = "data/processed/msvtt"
VOCAB_PATH = "outputs/models/caption_vocab.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEGMENT_SECONDS = 2
MAX_CAPTION_LEN = 30
MIN_WORD_FREQ = 2

from src.data.vocabulary import Vocabulary


# ── Feature extraction ────────────────────────────────────────────────────────

def load_resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_video_features(video_path, resnet, transform, segment_seconds=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

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
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = resnet(tensor).squeeze().cpu().numpy()
        features.append(feat)
        segment_idx += 1

    cap.release()
    return np.array(features, dtype=np.float32) if features else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)

    print("Loading MSR-VTT annotations...")
    with open(MSVTT_ANNO_FILE, "r") as f:
        data = json.load(f)

    # video_id → list of captions
    video_captions: dict = {}
    for sentence in data["sentences"]:
        vid = sentence["video_id"]
        cap = sentence["caption"].strip()
        video_captions.setdefault(vid, []).append(cap)

    print(f"Total videos in annotations: {len(video_captions)}")

    # ── Build vocabulary ──
    all_captions = [c for caps in video_captions.values() for c in caps]
    vocab = Vocabulary()
    vocab.build(all_captions, min_freq=MIN_WORD_FREQ)
    print(f"Vocabulary size: {len(vocab)}")

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocab saved → {VOCAB_PATH}")

    # ── Extract features + encode captions ──
    resnet = load_resnet()
    transform = get_transform()

    dataset = []
    skipped = 0

    for video_id in tqdm(sorted(video_captions.keys()), desc="Processing videos"):
        # Kaggle version uses plain numeric names: video0.mp4, video1.mp4, ...
        # The JSON uses ids like "video0", "video1", etc. — match directly
        video_path = os.path.join(MSVTT_VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            skipped += 1
            continue

        features = extract_video_features(video_path, resnet, transform)
        if features is None or len(features) == 0:
            skipped += 1
            continue

        for caption in video_captions[video_id]:
            encoded = vocab.encode(caption, max_len=MAX_CAPTION_LEN)
            dataset.append({
                "video_id": video_id,
                "features": features,
                "caption": encoded,
                "caption_text": caption,
            })

    print(f"Total samples: {len(dataset)}  |  Videos skipped: {skipped}")

    save_path = os.path.join(OUTPUT_DIR, "dataset.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved → {save_path}")


if __name__ == "__main__":
    main()

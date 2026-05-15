"""Data loading utilities for TVSum and MSR-VTT datasets."""

import os
import json
import pickle
import numpy as np
import pandas as pd


def load_tvsum_annotations(anno_file: str) -> dict:
    """Load TVSum annotations. Returns dict: video_id -> np.array of scores."""
    df = pd.read_csv(anno_file, sep="\t", header=None, names=["video_id", "category", "scores"])
    annotations = {}
    for _, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        score_str = str(row["scores"]).strip()
        scores = np.array(
            [float(x) for x in score_str.split(",") if x != ""],
            dtype=np.float32,
        )
        annotations[video_id] = scores
    return annotations


def get_available_tvsum_videos(feature_dir: str, annotations: dict) -> list:
    """Get list of video IDs that have both features and annotations."""
    videos = []
    for fname in os.listdir(feature_dir):
        if not fname.endswith(".npy"):
            continue
        video_id = fname.replace(".npy", "")
        if video_id in annotations:
            videos.append(video_id)
    videos.sort()
    return videos


def load_tvsum_features(video_id: str, feature_dir: str) -> np.ndarray | None:
    """Load pre-extracted ResNet18 features for a TVSum video."""
    path = os.path.join(feature_dir, f"{video_id}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def load_msvtt_annotations(anno_file: str) -> dict:
    """Load MSR-VTT annotations. Returns dict: video_id -> list of captions."""
    with open(anno_file, "r") as f:
        data = json.load(f)

    video_captions = {}
    for sentence in data["sentences"]:
        vid = sentence["video_id"]
        cap = sentence["caption"].strip()
        video_captions.setdefault(vid, []).append(cap)

    return video_captions


def load_caption_dataset(dataset_path: str) -> list:
    """Load the pre-built caption dataset from pickle."""
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def load_vocabulary(vocab_path: str):
    """Load the vocabulary from pickle."""
    with open(vocab_path, "rb") as f:
        return pickle.load(f)

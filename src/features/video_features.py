"""ResNet18 feature extraction module — reusable across both pipelines."""

import numpy as np
import cv2
from PIL import Image
from typing import List

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from src.config.config import DEVICE, FEATURE_DIM, RESNET_MEAN, RESNET_STD


_resnet = None
_transform = None


def load_resnet():
    """Load ResNet18 with classification head removed (512-dim features)."""
    global _resnet, _transform
    if _resnet is None:
        _resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        _resnet.fc = torch.nn.Identity()
        _resnet.eval()
        _resnet.to(DEVICE)
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=RESNET_MEAN, std=RESNET_STD),
        ])
    return _resnet, _transform


def extract_feature_from_image(image: Image.Image) -> np.ndarray:
    """Extract a 512-dim feature vector from a PIL image."""
    model, transform = load_resnet()
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze().cpu().numpy()


def extract_feature_from_path(image_path: str) -> np.ndarray:
    """Extract a 512-dim feature vector from an image file path."""
    image = Image.open(image_path).convert("RGB")
    return extract_feature_from_image(image)


def extract_video_segment_features(video_path: str, segment_seconds: int = 2) -> np.ndarray:
    """
    Extract ResNet18 features for every segment_seconds-second interval of a video.
    Returns: np.ndarray of shape (num_segments, 512)
    """
    model, transform = load_resnet()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

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
            feat = model(tensor).squeeze().cpu().numpy()
        features.append(feat)
        segment_idx += 1

    cap.release()
    return np.array(features, dtype=np.float32) if features else np.empty((0, FEATURE_DIM), dtype=np.float32)


def build_temporal_features(features: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Build temporal context features: [prev, current, next] for each segment.
    For radius=1: input (N, 512) -> output (N, 1536)
    """
    num_segments, feat_dim = features.shape
    temporal_rows = []
    for i in range(num_segments):
        parts = []
        for offset in range(-radius, radius + 1):
            j = max(0, min(num_segments - 1, i + offset))
            parts.append(features[j])
        temporal_rows.append(np.concatenate(parts))
    return np.vstack(temporal_rows).astype(np.float32)

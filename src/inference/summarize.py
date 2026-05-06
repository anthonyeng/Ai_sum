import os
import sys
import shutil
import subprocess
from typing import List

import cv2
import joblib
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

MODEL_PATH = "outputs/models/xgboost_video_split.pkl"
VIDEO_DIR = "uploads"
TEMP_DIR = "outputs/temp_segments"
SUMMARY_DIR = "outputs/summaries"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEGMENT_SECONDS = 2

# quality / behavior tuning
TOP_RATIO = 0.15
MIN_GAP = 4
EXTEND = 2
MAX_SUMMARY_SECONDS = 30


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


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
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def resolve_video_path(video_input: str) -> str:
    if os.path.exists(video_input):
        return os.path.abspath(video_input)

    alt = os.path.join(VIDEO_DIR, video_input)
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

            frame_path = os.path.join(
                output_dir,
                f"segment_{segment_idx:04d}_{sample_idx}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            segment_paths.append(frame_path)

        if segment_paths:
            segments.append(segment_paths)

        segment_idx += 1

    cap.release()
    return segments


def extract_feature(image_path: str, model, transform) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feature = model(tensor)

    return feature.squeeze().cpu().numpy()


def predict_segment_scores(video_path: str):
    frames_dir = os.path.join(TEMP_DIR, "frames")
    clean_dir(frames_dir)

    frame_groups = extract_segment_frames(video_path, frames_dir)

    if not frame_groups:
        raise ValueError("No segment frames extracted.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {os.path.abspath(MODEL_PATH)}")

    resnet = load_resnet()
    transform = get_transform()
    model = joblib.load(MODEL_PATH)

    features = []

    for segment in frame_groups:
        seg_feats = []

        for frame_path in segment:
            feat = extract_feature(frame_path, resnet, transform)
            seg_feats.append(feat)

        if not seg_feats:
            continue

        seg_feats = np.array(seg_feats, dtype=np.float32)
        features.append(seg_feats.mean(axis=0))

    if not features:
        raise ValueError("No features extracted from segments.")

    X = np.array(features, dtype=np.float32)
    preds = model.predict(X).astype(np.float32)

    return preds, len(features)


def select_smart_segments(scores: np.ndarray) -> List[int]:
    n_segments = len(scores)
    if n_segments == 0:
        return []

    ratio_limit = max(1, int(n_segments * TOP_RATIO))
    duration_limit = max(1, int(MAX_SUMMARY_SECONDS // SEGMENT_SECONDS))
    target_count = min(ratio_limit, duration_limit)

    ranked = np.argsort(scores)[::-1]
    selected: List[int] = []

    for idx in ranked:
        idx = int(idx)

        if len(selected) >= target_count:
            break

        if any(abs(idx - s) <= MIN_GAP for s in selected):
            continue

        selected.append(idx)

    return sorted(selected)


def merge_segments(indices: List[int]):
    if not indices:
        return []

    merged = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            merged.append((start, prev))
            start = idx
            prev = idx

    merged.append((start, prev))
    return merged


def extend_segments(ranges, max_len: int):
    extended = []

    for start, end in ranges:
        start = max(0, start - EXTEND)
        end = min(max_len - 1, end + EXTEND)
        extended.append((start, end))

    return extended


def merge_overlapping_ranges(ranges):
    if not ranges:
        return []

    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [ranges[0]]

    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def trim_ranges_to_max_duration(ranges, max_summary_seconds: int):
    if not ranges:
        return []

    max_segments_allowed = max(1, max_summary_seconds // SEGMENT_SECONDS)
    trimmed = []
    used = 0

    for start, end in ranges:
        length = end - start + 1

        if used >= max_segments_allowed:
            break

        remaining = max_segments_allowed - used
        if length <= remaining:
            trimmed.append((start, end))
            used += length
        else:
            trimmed.append((start, start + remaining - 1))
            used += remaining
            break

    return trimmed


def cut_segments(video_path: str, ranges):
    segments_dir = os.path.join(TEMP_DIR, "segments")
    clean_dir(segments_dir)

    output_files = []

    for i, (start, end) in enumerate(ranges):
        start_time = start * SEGMENT_SECONDS
        duration = (end - start + 1) * SEGMENT_SECONDS
        output_path = os.path.join(segments_dir, f"part_{i:04d}.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            output_path,
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        output_files.append(output_path)

    return output_files


def concat_segments(segment_files: List[str], output_path: str) -> None:
    if not segment_files:
        raise ValueError("No segment files to concatenate.")

    concat_file = os.path.join(TEMP_DIR, "concat.txt")

    with open(concat_file, "w") as f:
        for segment_file in segment_files:
            abs_path = os.path.abspath(segment_file)
            f.write(f"file '{abs_path}'\n")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def summarize_video(video_input: str) -> None:
    ensure_dir(SUMMARY_DIR)
    ensure_dir(TEMP_DIR)

    video_path = resolve_video_path(video_input)
    video_filename = os.path.basename(video_path)

    print(f"Processing: {video_path}")

    scores, n_segments = predict_segment_scores(video_path)

    selected = select_smart_segments(scores)
    merged = merge_segments(selected)
    extended = extend_segments(merged, n_segments)
    cleaned = merge_overlapping_ranges(extended)
    final_ranges = trim_ranges_to_max_duration(cleaned, MAX_SUMMARY_SECONDS)

    if not final_ranges:
        raise ValueError("No final ranges selected for summary.")

    print(f"Selected segment indices: {selected}")
    print(f"Merged ranges: {merged}")
    print(f"Extended ranges: {extended}")
    print(f"Final ranges: {final_ranges}")

    segment_files = cut_segments(video_path, final_ranges)

    output_name = os.path.splitext(video_filename)[0] + "_summary.mp4"
    output_path = os.path.join(SUMMARY_DIR, output_name)

    concat_segments(segment_files, output_path)

    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python src/inference/summarize.py <video_path>")

    summarize_video(sys.argv[1])
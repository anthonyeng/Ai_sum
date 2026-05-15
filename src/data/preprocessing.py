"""Video preprocessing and validation utilities."""

import os
import cv2


def validate_video(video_path: str) -> dict:
    """
    Validate a video file and return metadata.
    Returns dict with keys: valid, error, fps, duration, width, height, frame_count.
    """
    result = {
        "valid": False,
        "error": None,
        "fps": 0.0,
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "frame_count": 0,
    }

    if not os.path.exists(video_path):
        result["error"] = f"File not found: {video_path}"
        return result

    ext = os.path.splitext(video_path)[1].lower()
    if ext != ".mp4":
        result["error"] = f"Unsupported format: {ext}. Only .mp4 is supported."
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["error"] = "Could not open video file"
        return result

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        result["error"] = "Could not read FPS from video"
        return result

    if frame_count <= 0:
        result["error"] = "Video has no frames"
        return result

    duration = frame_count / fps

    result.update({
        "valid": True,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    })
    return result


def get_video_metadata(video_path: str) -> dict:
    """Get basic video metadata without full validation."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "duration": frame_count / fps if fps > 0 else 0,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }

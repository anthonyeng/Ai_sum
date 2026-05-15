"""Shared utilities for the AI Video Summarizer."""

import os
import shutil
import subprocess
from contextlib import contextmanager


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


@contextmanager
def temp_directory(path: str):
    """Context manager that creates a temp directory and cleans it up after use."""
    ensure_dir(path)
    try:
        yield path
    finally:
        if os.path.exists(path):
            shutil.rmtree(path)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         video_path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def get_video_fps(video_path: str) -> float:
    """Get video FPS using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1",
         video_path],
        capture_output=True, text=True,
    )
    try:
        num, den = result.stdout.strip().split("/")
        return float(num) / float(den)
    except (ValueError, AttributeError, ZeroDivisionError):
        return 0.0


def get_video_resolution(video_path: str) -> tuple:
    """Get video resolution (width, height) using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x",
         video_path],
        capture_output=True, text=True,
    )
    try:
        w, h = result.stdout.strip().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return 0, 0


def validate_video_file(path: str, max_size_mb: int = 500, max_duration_sec: int = 600) -> str | None:
    """Validate a video file. Returns error message or None if valid."""
    if not os.path.exists(path):
        return "File does not exist"

    ext = os.path.splitext(path)[1].lower()
    if ext != ".mp4":
        return "Only .mp4 files are supported"

    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_size_mb:
        return f"File too large ({size_mb:.0f} MB, max {max_size_mb} MB)"

    duration = get_video_duration(path)
    if duration > max_duration_sec:
        return f"Video too long ({duration:.0f}s, max {max_duration_sec}s)"

    return None

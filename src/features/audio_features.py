"""
Audio feature extraction module.

Extracts per-segment audio features from video files:
  - RMS energy (loudness)
  - Spectral centroid (brightness / voice activity)
  - Zero-crossing rate (speech vs silence indicator)
  - Speech ratio (% of segment with speech, from Whisper timestamps)

These features complement visual embeddings in the multimodal fusion pipeline
for segment importance scoring.

Usage:
    from src.features.audio_features import extract_audio_features
    features = extract_audio_features(video_path, segment_seconds=2)
    # Returns: np.ndarray of shape (num_segments, 4)
    # Columns: [rms_energy, spectral_centroid, zero_crossing_rate, is_speech]
"""

import os
import subprocess
import tempfile

import numpy as np

from src.config.config import SEGMENT_SECONDS


def _extract_audio_wav(video_path: str, output_path: str, sr: int = 16000) -> bool:
    """Extract audio from video as mono WAV using ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(sr), "-ac", "1",
                "-y", output_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio_features(
    video_path: str,
    segment_seconds: int = None,
    sr: int = 16000,
) -> np.ndarray:
    """
    Extract audio features per video segment.

    Args:
        video_path: path to video file
        segment_seconds: duration of each segment (default: config SEGMENT_SECONDS)
        sr: audio sample rate

    Returns:
        np.ndarray of shape (num_segments, 4) with columns:
          [rms_energy, spectral_centroid, zero_crossing_rate, energy_variance]
        All values normalized to [0, 1] range.
        Returns empty array if audio extraction fails.
    """
    if segment_seconds is None:
        segment_seconds = SEGMENT_SECONDS

    # Extract audio to temp WAV
    tmp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(tmp_dir, "audio.wav")

    if not _extract_audio_wav(video_path, wav_path, sr):
        return np.empty((0, 4), dtype=np.float32)

    try:
        import librosa
        audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    except Exception:
        _cleanup(tmp_dir, wav_path)
        return np.empty((0, 4), dtype=np.float32)

    _cleanup(tmp_dir, wav_path)

    samples_per_segment = sr * segment_seconds
    num_segments = max(1, len(audio) // samples_per_segment)
    features = []

    for i in range(num_segments):
        start = i * samples_per_segment
        end = min(start + samples_per_segment, len(audio))
        segment = audio[start:end]

        if len(segment) < sr // 4:  # less than 0.25s of audio
            features.append([0.0, 0.0, 0.0, 0.0])
            continue

        # RMS energy (loudness)
        rms = np.sqrt(np.mean(segment ** 2))

        # Spectral centroid (brightness — higher = more voice-like)
        try:
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            centroid_mean = np.mean(centroid)
        except Exception:
            centroid_mean = 0.0

        # Zero-crossing rate (speech indicator)
        zcr = np.mean(np.abs(np.diff(np.sign(segment)))) / 2.0

        # Energy variance within segment (dynamic content indicator)
        frame_size = sr // 10  # 100ms frames
        frame_energies = []
        for j in range(0, len(segment) - frame_size, frame_size):
            frame = segment[j:j + frame_size]
            frame_energies.append(np.sqrt(np.mean(frame ** 2)))
        energy_var = np.var(frame_energies) if frame_energies else 0.0

        features.append([rms, centroid_mean, zcr, energy_var])

    if not features:
        return np.empty((0, 4), dtype=np.float32)

    features = np.array(features, dtype=np.float32)

    # Normalize each column to [0, 1]
    for col in range(features.shape[1]):
        col_max = features[:, col].max()
        if col_max > 0:
            features[:, col] /= col_max

    return features


def _cleanup(tmp_dir: str, wav_path: str):
    """Remove temp files."""
    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
    except OSError:
        pass


def get_audio_feature_dim() -> int:
    """Return audio feature dimension."""
    return 4


def get_audio_feature_names() -> list[str]:
    """Return names of audio features."""
    return ["rms_energy", "spectral_centroid", "zero_crossing_rate", "energy_variance"]

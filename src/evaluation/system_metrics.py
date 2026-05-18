"""
System performance metrics.

Tracks inference time, memory usage, and processing speed
for profiling and benchmarking the pipelines.

Usage:
    from src.evaluation.system_metrics import Timer, get_memory_usage, SystemProfiler

    with Timer() as t:
        result = summarize_video(path)
    print(f"Took {t.elapsed:.2f}s")

    profiler = SystemProfiler()
    profiler.start("feature_extraction")
    # ... do work ...
    profiler.stop("feature_extraction")
    print(profiler.report())
"""

import os
import time
from typing import Optional


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


def get_memory_usage() -> dict:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        return {
            "rss_mb": round(mem.rss / 1024 / 1024, 1),
            "vms_mb": round(mem.vms / 1024 / 1024, 1),
        }
    except ImportError:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "max_rss_mb": round(usage.ru_maxrss / 1024 / 1024, 1),
        }


class SystemProfiler:
    """
    Profile multiple stages of a pipeline.

    Usage:
        profiler = SystemProfiler()
        profiler.start("stage1")
        # work...
        profiler.stop("stage1")
        profiler.start("stage2")
        # work...
        profiler.stop("stage2")
        print(profiler.report())
    """

    def __init__(self):
        self._timers: dict[str, float] = {}
        self._starts: dict[str, float] = {}
        self._memory_before: Optional[dict] = None

    def start(self, name: str):
        """Start timing a named stage."""
        self._starts[name] = time.perf_counter()

    def stop(self, name: str):
        """Stop timing a named stage."""
        if name in self._starts:
            elapsed = time.perf_counter() - self._starts[name]
            self._timers[name] = elapsed
            del self._starts[name]

    def get(self, name: str) -> float:
        """Get elapsed time for a stage."""
        return self._timers.get(name, 0.0)

    def report(self, video_duration_sec: Optional[float] = None) -> dict:
        """
        Generate a performance report.

        Args:
            video_duration_sec: original video duration (for speed ratio)
        """
        total = sum(self._timers.values())

        result = {
            "stages": {
                name: {
                    "seconds": round(elapsed, 3),
                    "percent": round(elapsed / total * 100, 1) if total > 0 else 0.0,
                }
                for name, elapsed in self._timers.items()
            },
            "total_seconds": round(total, 3),
            "memory": get_memory_usage(),
        }

        if video_duration_sec and video_duration_sec > 0:
            result["processing_ratio"] = round(total / video_duration_sec, 2)
            result["speed"] = f"{round(video_duration_sec / total, 1)}x realtime" if total > 0 else "N/A"

        return result

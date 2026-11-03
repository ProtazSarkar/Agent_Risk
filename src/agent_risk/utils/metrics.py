"""System profiling and runtime telemetry utilities.

Tracks execution latency, CPU usage, and memory overhead during DAG decomposition
and inference shielding.
"""

import time
import os
import psutil
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class ProfileResult:
    """Telemetry report containing runtime overhead metrics."""
    elapsed_time_sec: float
    cpu_percent: float
    start_rss_mb: float
    end_rss_mb: float
    peak_rss_mb: float


@contextmanager
def profile_block(name: Optional[str] = None) -> Generator[ProfileResult, None, None]:
    """Context manager to measure elapsed execution time, CPU utilization, and RAM change."""
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss / (1024 * 1024)
    start_time = time.perf_counter()

    # Pre-measure CPU
    process.cpu_percent(interval=None)

    result = ProfileResult(
        elapsed_time_sec=0.0,
        cpu_percent=0.0,
        start_rss_mb=start_rss,
        end_rss_mb=start_rss,
        peak_rss_mb=start_rss,
    )

    try:
        yield result
    finally:
        end_time = time.perf_counter()
        end_rss = process.memory_info().rss / (1024 * 1024)
        cpu_usage = process.cpu_percent(interval=None)

        result.elapsed_time_sec = end_time - start_time
        result.cpu_percent = cpu_usage
        result.end_rss_mb = end_rss
        result.peak_rss_mb = max(start_rss, end_rss)


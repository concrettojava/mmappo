"""Low-overhead GPU telemetry for long-running training phases.

The monitor deliberately uses ``nvidia-smi`` instead of adding a Python NVML
runtime dependency.  Sampling is infrequent (default 1 Hz), happens on a daemon
thread, and is grouped by caller-defined phases such as collect/replay/update.
PyTorch CUDA peak allocator statistics are captured for the same phase.
"""
from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import threading
import time
from typing import Dict

import numpy as np
import torch


@dataclass
class GPUPhaseStats:
    samples: int = 0
    util_avg_pct: float = 0.0
    util_max_pct: float = 0.0
    vram_avg_mib: float = 0.0
    vram_peak_mib: float = 0.0
    vram_total_mib: float = 0.0
    power_avg_w: float = 0.0
    power_max_w: float = 0.0
    temp_avg_c: float = 0.0
    temp_max_c: float = 0.0
    torch_peak_allocated_mib: float = 0.0
    torch_peak_reserved_mib: float = 0.0

    def as_dict(self, prefix: str) -> Dict[str, float]:
        return {
            f"gpu_{prefix}_samples": float(self.samples),
            f"gpu_{prefix}_util_avg_pct": self.util_avg_pct,
            f"gpu_{prefix}_util_max_pct": self.util_max_pct,
            f"gpu_{prefix}_vram_avg_mib": self.vram_avg_mib,
            f"gpu_{prefix}_vram_peak_mib": self.vram_peak_mib,
            f"gpu_{prefix}_vram_total_mib": self.vram_total_mib,
            f"gpu_{prefix}_power_avg_w": self.power_avg_w,
            f"gpu_{prefix}_power_max_w": self.power_max_w,
            f"gpu_{prefix}_temp_avg_c": self.temp_avg_c,
            f"gpu_{prefix}_temp_max_c": self.temp_max_c,
            f"gpu_{prefix}_torch_peak_allocated_mib": self.torch_peak_allocated_mib,
            f"gpu_{prefix}_torch_peak_reserved_mib": self.torch_peak_reserved_mib,
        }


class GPUPhaseMonitor:
    """Sample one CUDA device while a named training phase is running."""

    def __init__(self, device: str | torch.device, interval_seconds: float = 1.0):
        self.device = torch.device(device)
        self.interval_seconds = max(float(interval_seconds), 0.2)
        self.enabled = bool(
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and shutil.which("nvidia-smi") is not None
        )
        self.device_index = int(self.device.index or 0)

    def _sample_once(self):
        if not self.enabled:
            return None
        cmd = [
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(
                cmd,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).strip().splitlines()[0]
            fields = [item.strip() for item in output.split(",")]
            if len(fields) != 5:
                return None
            return tuple(float(item) for item in fields)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def measure(self, phase: str):
        return _GPUPhaseContext(self, phase)


class _GPUPhaseContext:
    def __init__(self, monitor: GPUPhaseMonitor, phase: str):
        self.monitor = monitor
        self.phase = str(phase)
        self.stats = GPUPhaseStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float, float, float, float]] = []

    def _append_sample(self) -> None:
        sample = self.monitor._sample_once()
        if sample is not None:
            self._samples.append(sample)

    def _worker(self) -> None:
        while not self._stop.wait(self.monitor.interval_seconds):
            self._append_sample()

    def __enter__(self):
        if self.monitor.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.monitor.device)
        if self.monitor.enabled:
            self._append_sample()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.monitor.enabled:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=max(2.0, self.monitor.interval_seconds * 2.0))
            self._append_sample()

        if self.monitor.device.type == "cuda" and torch.cuda.is_available():
            self.stats.torch_peak_allocated_mib = (
                torch.cuda.max_memory_allocated(self.monitor.device) / (1024.0**2)
            )
            self.stats.torch_peak_reserved_mib = (
                torch.cuda.max_memory_reserved(self.monitor.device) / (1024.0**2)
            )

        if self._samples:
            arr = np.asarray(self._samples, dtype=np.float64)
            util, used, total, power, temp = arr.T
            self.stats.samples = int(arr.shape[0])
            self.stats.util_avg_pct = float(util.mean())
            self.stats.util_max_pct = float(util.max())
            self.stats.vram_avg_mib = float(used.mean())
            self.stats.vram_peak_mib = float(used.max())
            self.stats.vram_total_mib = float(total.max())
            self.stats.power_avg_w = float(power.mean())
            self.stats.power_max_w = float(power.max())
            self.stats.temp_avg_c = float(temp.mean())
            self.stats.temp_max_c = float(temp.max())
        return False

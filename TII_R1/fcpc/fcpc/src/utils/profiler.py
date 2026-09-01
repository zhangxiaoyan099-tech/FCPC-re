"""Lightweight resource and communication accounting for FCPC experiments."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Mapping


MIB = 1024.0 * 1024.0


def state_dict_nbytes(state: Mapping[str, object]) -> int:
    """Return the exact tensor payload size of a model state."""
    total = 0
    for value in state.values():
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            total += int(value.numel()) * int(value.element_size())
    return total


@dataclass(frozen=True)
class ResourceStats:
    process_cpu_mean_pct: float = 0.0
    process_cpu_peak_pct: float = 0.0
    rss_peak_mib: float = 0.0
    gpu_util_mean_pct: float = 0.0
    gpu_util_peak_pct: float = 0.0
    gpu_memory_peak_mib: float = 0.0
    gpu_monitor_backend: str = "none"
    gpu_sample_count: int = 0


class ResourceMonitor:
    """Sample process CPU/RSS and CUDA utilization during one FL round."""

    def __init__(self, device: str = "cpu", interval_s: float = 0.1):
        self.device = str(device)
        self.interval_s = max(float(interval_s), 0.02)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_samples: list[float] = []
        self._rss_samples: list[float] = []
        self._gpu_samples: list[float] = []
        self._process = None
        self._torch = None
        self._gpu_index = 0
        self._torch_utilization_available = False
        self._nvidia_smi_path: str | None = None
        self._gpu_monitor_backend = "none"

    def start(self) -> "ResourceMonitor":
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(interval=None)
        except Exception:
            self._process = None

        if self.device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    self._torch = torch
                    parsed_device = torch.device(self.device)
                    self._gpu_index = (
                        torch.cuda.current_device()
                        if parsed_device.index is None
                        else int(parsed_device.index)
                    )
                    torch.cuda.reset_peak_memory_stats(self.device)
                    self._torch_utilization_available = callable(
                        getattr(torch.cuda, "utilization", None)
                    )
                    self._nvidia_smi_path = shutil.which("nvidia-smi")
            except Exception:
                self._torch = None

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> ResourceStats:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.interval_s))
        self._sample_once()

        gpu_memory_peak = 0.0
        if self._torch is not None:
            try:
                gpu_memory_peak = float(
                    self._torch.cuda.max_memory_allocated(self.device)
                ) / MIB
            except Exception:
                pass

        return ResourceStats(
            process_cpu_mean_pct=_mean(self._cpu_samples),
            process_cpu_peak_pct=max(self._cpu_samples, default=0.0),
            rss_peak_mib=max(self._rss_samples, default=0.0),
            gpu_util_mean_pct=_mean(self._gpu_samples),
            gpu_util_peak_pct=max(self._gpu_samples, default=0.0),
            gpu_memory_peak_mib=gpu_memory_peak,
            gpu_monitor_backend=self._gpu_monitor_backend,
            gpu_sample_count=len(self._gpu_samples),
        )

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        if self._process is not None:
            try:
                logical_cpus = max(os.cpu_count() or 1, 1)
                normalized_cpu = float(self._process.cpu_percent(interval=None)) / logical_cpus
                self._cpu_samples.append(normalized_cpu)
                self._rss_samples.append(float(self._process.memory_info().rss) / MIB)
            except Exception:
                pass
        if self._torch is not None:
            if self._torch_utilization_available:
                try:
                    self._gpu_samples.append(
                        float(self._torch.cuda.utilization(self.device))
                    )
                    self._gpu_monitor_backend = "torch.cuda.utilization"
                    return
                except Exception:
                    self._torch_utilization_available = False
            if self._nvidia_smi_path is not None:
                try:
                    completed = subprocess.run(
                        [
                            self._nvidia_smi_path,
                            f"--id={self._gpu_index}",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=max(2.0, 4.0 * self.interval_s),
                    )
                    value = completed.stdout.strip().splitlines()[0]
                    self._gpu_samples.append(float(value))
                    self._gpu_monitor_backend = "nvidia-smi"
                except Exception:
                    self._nvidia_smi_path = None


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

"""Read-only local hardware advisory (NVIDIA GPU, Apple MPS or CPU).

This module only inspects the machine; it never installs packages or reaches remote hosts.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass

HIGH_VRAM_THRESHOLD_MB = 8192


@dataclass(frozen=True)
class HardwareProfile:
    has_gpu: bool
    gpu_type: str  # "cuda" | "mps" | "cpu"
    gpu_name: str
    vram_mb: int | None
    tier: str  # "high" | "limited" | "cpu_only"
    warning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_hardware() -> HardwareProfile:
    """Detect local GPU hardware: NVIDIA first, then Apple Silicon, then CPU only."""
    return _detect_nvidia() or _detect_mps() or _cpu_profile()


def _cpu_profile() -> HardwareProfile:
    return HardwareProfile(
        has_gpu=False,
        gpu_type="cpu",
        gpu_name="CPU only",
        vram_mb=None,
        tier="cpu_only",
        warning="No GPU detected; research constraints that assume a GPU cannot be met locally.",
    )


def _detect_nvidia() -> HardwareProfile | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 2:
        return None
    try:
        vram_mb = int(float(parts[1]))
    except ValueError:
        vram_mb = 0
    high = vram_mb >= HIGH_VRAM_THRESHOLD_MB
    return HardwareProfile(
        has_gpu=True,
        gpu_type="cuda",
        gpu_name=parts[0],
        vram_mb=vram_mb,
        tier="high" if high else "limited",
        warning="" if high else f"Local GPU ({parts[0]}, {vram_mb} MB VRAM) has limited memory.",
    )


def _detect_mps() -> HardwareProfile | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    name = "Apple Silicon GPU"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return HardwareProfile(
        has_gpu=True,
        gpu_type="mps",
        gpu_name=name,
        vram_mb=None,
        tier="limited",
        warning=f"Apple GPU detected ({name}); shared memory and lower throughput than CUDA.",
    )

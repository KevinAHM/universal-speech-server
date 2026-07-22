"""Preflight the selected CrispASR shared library before server startup."""

from __future__ import annotations

import sys
from pathlib import Path

from .config import load_config
from .crisp import CrispBindingError, load_library


OPENBLAS_SONAME = "libopenblas.so.0"
CUDA_12_SONAMES = (
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libnvrtc.so.12",
)


def _load_error_message(exc: CrispBindingError, *, platform: str) -> str:
    detail = str(exc)
    if platform.startswith("linux") and OPENBLAS_SONAME in detail:
        return (
            f"Required Linux system library {OPENBLAS_SONAME} is missing.\n"
            "Install OpenBLAS, then run start_server.sh again:\n"
            "  Ubuntu/Debian: sudo apt install libopenblas0\n"
            "  Fedora/RHEL:   sudo dnf install openblas\n"
            "  Arch Linux:    sudo pacman -S openblas"
        )
    missing_cuda = next(
        (soname for soname in CUDA_12_SONAMES if soname in detail), None
    )
    if platform.startswith("linux") and missing_cuda is not None:
        return (
            f"Required NVIDIA CUDA 12 library {missing_cuda} is missing.\n"
            "This CrispASR runtime requires CUDA Toolkit 12.8 runtime libraries.\n"
            "Choose your Linux distribution and follow NVIDIA's official "
            "installation steps:\n"
            "  https://developer.nvidia.com/cuda-12-8-0-download-archive\n"
            "Then run start_server.sh again."
        )
    return f"CrispASR native runtime could not be loaded: {detail}"


def check_native_runtime(lib_path: Path, *, platform: str = sys.platform) -> None:
    """Load the selected library and explain known missing system dependencies."""

    try:
        load_library(lib_path)
    except CrispBindingError as exc:
        raise CrispBindingError(_load_error_message(exc, platform=platform)) from exc


def main() -> int:
    cfg = load_config()
    if cfg.lib_path is None:
        print("CrispASR native runtime is not installed.", file=sys.stderr)
        return 1
    try:
        check_native_runtime(cfg.lib_path)
    except CrispBindingError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"CrispASR native runtime loaded successfully: {cfg.lib_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

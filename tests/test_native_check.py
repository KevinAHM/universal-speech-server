from pathlib import Path

import pytest

from speech_server import native_check
from speech_server.crisp import CrispBindingError


def test_native_check_explains_missing_openblas(monkeypatch, tmp_path: Path):
    library = tmp_path / "libcrispasr.so"

    def fail_load(_path):
        raise CrispBindingError(
            "cannot load libcrispasr: libopenblas.so.0: "
            "cannot open shared object file: No such file or directory"
        )

    monkeypatch.setattr(native_check, "load_library", fail_load)
    with pytest.raises(CrispBindingError) as failure:
        native_check.check_native_runtime(library, platform="linux")

    message = str(failure.value)
    assert "Required Linux system library libopenblas.so.0 is missing" in message
    assert "sudo apt install libopenblas0" in message
    assert "sudo dnf install openblas" in message
    assert "sudo pacman -S openblas" in message


def test_native_check_preserves_other_loader_errors(monkeypatch, tmp_path: Path):
    library = tmp_path / "libcrispasr.so"

    def fail_load(_path):
        raise CrispBindingError("cannot load libcrispasr: wrong ELF class")

    monkeypatch.setattr(native_check, "load_library", fail_load)
    with pytest.raises(CrispBindingError, match="wrong ELF class") as failure:
        native_check.check_native_runtime(library, platform="linux")
    assert "OpenBLAS" not in str(failure.value)


@pytest.mark.parametrize(
    "soname",
    [
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libnvrtc.so.12",
    ],
)
def test_native_check_explains_missing_cuda_12_runtime(
    monkeypatch, tmp_path: Path, soname: str
):
    library = tmp_path / "libcrispasr.so"

    def fail_load(_path):
        raise CrispBindingError(
            f"cannot load libcrispasr: {soname}: "
            "cannot open shared object file: No such file or directory"
        )

    monkeypatch.setattr(native_check, "load_library", fail_load)
    with pytest.raises(CrispBindingError) as failure:
        native_check.check_native_runtime(library, platform="linux")

    message = str(failure.value)
    assert f"Required NVIDIA CUDA 12 library {soname} is missing" in message
    assert "requires CUDA Toolkit 12.8 runtime libraries" in message
    assert "https://developer.nvidia.com/cuda-12-8-0-download-archive" in message
    assert "sudo apt" not in message
    assert "sudo dnf" not in message
    assert "sudo pacman" not in message


def test_native_check_loads_available_runtime(monkeypatch, tmp_path: Path):
    library = tmp_path / "libcrispasr.so"
    loaded = []
    monkeypatch.setattr(native_check, "load_library", loaded.append)

    native_check.check_native_runtime(library, platform="linux")

    assert loaded == [library]

import io
import json
import subprocess

import pytest

from speech_server import gpu_select
from speech_server.gpu_select import GPUDevice


def test_nvidia_enumeration_uses_reported_device_indices():
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, NVIDIA RTX 4090\n2, NVIDIA RTX 5070 Ti\n",
            stderr="",
        )

    assert gpu_select.enumerate_nvidia_devices(run) == [
        GPUDevice("0", "NVIDIA RTX 4090"),
        GPUDevice("2", "NVIDIA RTX 5070 Ti"),
    ]


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("windows-x86_64-cuda", "cuda"),
        ("nvidia", "cuda"),
        ("linux-x86_64-vulkan", "vulkan"),
        ("amd", "vulkan"),
        ("windows-x86_64-cpu", None),
        ("macos-arm64-metal", None),
    ],
)
def test_backend_is_derived_from_installed_target(target, expected):
    assert gpu_select.backend_for_target(target) == expected


def test_single_gpu_is_selected_without_prompt():
    output = io.StringIO()
    selected = gpu_select.choose_device(
        [GPUDevice("3", "Only GPU")],
        backend="cuda",
        input_fn=lambda: pytest.fail("prompted for a single GPU"),
        output=output,
    )
    assert selected == "3"
    assert "only detected CUDA GPU" in output.getvalue()


def test_multiple_gpus_prompt_until_a_valid_index_is_selected():
    answers = iter(["bad", "1"])
    output = io.StringIO()
    selected = gpu_select.choose_device(
        [GPUDevice("0", "First"), GPUDevice("1", "Second")],
        backend="vulkan",
        interactive=True,
        input_fn=lambda: next(answers),
        output=output,
    )
    assert selected == "1"
    assert "Detected 2 VULKAN GPUs" in output.getvalue()
    assert "Enter one of: 0, 1" in output.getvalue()


def test_multiple_gpus_default_to_first_without_an_interactive_terminal():
    output = io.StringIO()
    selected = gpu_select.choose_device(
        [GPUDevice("0", "First"), GPUDevice("1", "Second")],
        backend="cuda",
        interactive=False,
        output=output,
    )
    assert selected == "0"
    assert "No interactive terminal" in output.getvalue()


def test_requested_device_must_be_present_when_enumeration_succeeds():
    with pytest.raises(ValueError, match="available indices: 0"):
        gpu_select.choose_device(
            [GPUDevice("0", "First")],
            backend="cuda",
            requested="7",
            output=io.StringIO(),
        )


def test_startup_uses_installed_runtime_and_preserves_explicit_visibility(
    tmp_path,
):
    manifest = tmp_path / "installed.json"
    manifest.write_text(
        json.dumps({"target": "windows-x86_64-cuda"}), encoding="utf-8"
    )
    assignment = gpu_select.startup_assignment(
        target="cpu",
        manifest=manifest,
        environ={"CUDA_VISIBLE_DEVICES": "2"},
        output=io.StringIO(),
    )
    assert assignment == "CUDA_VISIBLE_DEVICES=2"


def test_requested_device_overrides_inherited_visibility(tmp_path, monkeypatch):
    manifest = tmp_path / "installed.json"
    manifest.write_text(
        json.dumps({"target": "windows-x86_64-cuda"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        gpu_select,
        "enumerate_nvidia_devices",
        lambda: [GPUDevice("0", "First"), GPUDevice("1", "Second")],
    )
    assignment = gpu_select.startup_assignment(
        target="auto",
        manifest=manifest,
        environ={
            "CUDA_VISIBLE_DEVICES": "1",
            "SPEECH_SERVER_GPU_DEVICE": "0",
        },
        output=io.StringIO(),
    )
    assert assignment == "CUDA_VISIBLE_DEVICES=0"


def test_startup_selects_from_the_actual_vulkan_device_list(tmp_path, monkeypatch):
    manifest = tmp_path / "installed.json"
    manifest.write_text(
        json.dumps({"target": "linux-x86_64-vulkan"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        gpu_select,
        "enumerate_vulkan_devices",
        lambda: [GPUDevice("0", "Integrated"), GPUDevice("1", "Discrete")],
    )
    assignment = gpu_select.startup_assignment(
        target="auto",
        manifest=manifest,
        environ={},
        interactive=True,
        input_fn=lambda: "1",
        output=io.StringIO(),
    )
    assert assignment == "GGML_VK_VISIBLE_DEVICES=1"


def test_explicit_library_backend_takes_precedence_over_installed_manifest(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "installed.json"
    manifest.write_text(
        json.dumps({"target": "windows-x86_64-cuda"}), encoding="utf-8"
    )
    library_dir = tmp_path / "developer-runtime"
    library_dir.mkdir()
    library = library_dir / "crispasr.dll"
    library.write_bytes(b"library")
    (library_dir / "ggml-vulkan.dll").write_bytes(b"plugin")
    monkeypatch.setattr(
        gpu_select,
        "enumerate_vulkan_devices",
        lambda: [GPUDevice("0", "Vulkan GPU")],
    )
    assignment = gpu_select.startup_assignment(
        target="auto",
        manifest=manifest,
        environ={"SPEECH_SERVER_LIB": str(library)},
        output=io.StringIO(),
    )
    assert assignment == "GGML_VK_VISIBLE_DEVICES=0"

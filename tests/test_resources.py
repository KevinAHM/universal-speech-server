import subprocess
from pathlib import Path

from speech_server import resources
from speech_server.config import ModelSpec, ServerConfig
from speech_server.engine import SessionPool
from tests.fakes import FakeSession


def test_missing_nvidia_smi_is_reported_without_failing(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing)
    gpus, error = resources._nvidia_gpus()
    assert gpus == []
    assert error == "nvidia-smi-not-found"


def test_malformed_nvidia_smi_output_is_unavailable(monkeypatch):
    result = subprocess.CompletedProcess([], 0, stdout="not,a,valid,row\n", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)
    gpus, error = resources._nvidia_gpus()
    assert gpus == []
    assert error == "nvidia-smi-malformed"


def test_nvidia_telemetry_queries_only_the_cuda_visible_device(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="2, GPU-selected, NVIDIA RTX 5070 Ti, 16000, 8000, 8000\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    gpus, error = resources._nvidia_gpus({"CUDA_VISIBLE_DEVICES": "2"})
    assert error is None
    assert "--id=2" in commands[0]
    assert len(gpus) == 1
    assert gpus[0]["index"] == 2
    assert gpus[0]["uuid"] == "GPU-selected"
    assert gpus[0]["selected"] is True


def test_nvidia_telemetry_does_not_invent_a_selection(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "0, GPU-first, NVIDIA RTX 3060, 12000, 1000, 11000\n"
                "2, GPU-second, NVIDIA RTX 5070 Ti, 16000, 8000, 8000\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    gpus, error = resources._nvidia_gpus({})
    assert error is None
    assert not any(argument.startswith("--id=") for argument in commands[0])
    assert [gpu["index"] for gpu in gpus] == [0, 2]
    assert not any(gpu["selected"] for gpu in gpus)


def test_load_plan_uses_selected_gpu_instead_of_emptiest_gpu():
    model = ModelSpec(
        id="new",
        backend="omnivoice",
        model_path=Path("new.gguf"),
        sample_rate=24000,
        estimated_ram_mb=100,
        estimated_vram_mb=100,
    )
    cfg = ServerConfig(models={"new": model})
    pool = SessionPool(cfg, session_factory=lambda cfg, spec: FakeSession())
    plan = resources.build_load_plan(
        cfg,
        pool,
        None,
        {
            "sampledAt": "now",
            "ram": {"freeBytes": 1024 * 1024 * 1024},
            "gpus": [
                {
                    "index": 0,
                    "freeBytes": 200 * 1024 * 1024,
                    "selected": True,
                },
                {
                    "index": 1,
                    "freeBytes": 10 * 1024 * 1024 * 1024,
                    "selected": False,
                },
            ],
        },
        model_id="new",
        upscale=False,
        adaptive_batching=False,
    )
    assert plan["gpuIndex"] == 0
    assert plan["requirements"]["vram"]["projectedFreeBytes"] == 200 * 1024 * 1024


def test_load_plan_projects_idle_eviction_and_reports_pinned_wait():
    def model(model_id, size_mb):
        return ModelSpec(
            id=model_id,
            backend="omnivoice",
            model_path=Path(f"{model_id}.gguf"),
            sample_rate=24000,
            estimated_ram_mb=size_mb,
            estimated_vram_mb=size_mb,
        )

    cfg = ServerConfig(
        models={"old": model("old", 500), "new": model("new", 600)},
        resident_limit=1,
    )
    pool = SessionPool(cfg, session_factory=lambda cfg, spec: FakeSession())
    pool.acquire("old")
    sampled = {
        "sampledAt": "now",
        "ram": {"freeBytes": 400 * 1024 * 1024},
        "gpus": [{"index": 0, "freeBytes": 400 * 1024 * 1024}],
    }
    plan = resources.build_load_plan(
        cfg,
        pool,
        None,
        sampled,
        model_id="new",
        upscale=False,
        adaptive_batching=False,
    )
    assert plan["evict"] == [{"kind": "model", "id": "old"}]
    assert plan["fit"]["status"] == "comfortable"
    assert plan["requirements"]["vram"]["reclaimableBytes"] == 500 * 1024 * 1024

    pool.pin("old")
    busy = resources.build_load_plan(
        cfg,
        pool,
        None,
        sampled,
        model_id="new",
        upscale=False,
        adaptive_batching=False,
    )
    assert busy["busy"] == [{"kind": "model", "id": "old"}]
    assert busy["fit"]["status"] == "busy"


def test_load_plan_does_not_claim_fit_when_vram_telemetry_is_missing():
    model = ModelSpec(
        id="new",
        backend="omnivoice",
        model_path=Path("new.gguf"),
        sample_rate=24000,
        estimated_ram_mb=100,
        estimated_vram_mb=100,
    )
    cfg = ServerConfig(models={"new": model})
    pool = SessionPool(cfg, session_factory=lambda cfg, spec: FakeSession())
    plan = resources.build_load_plan(
        cfg,
        pool,
        None,
        {
            "sampledAt": "now",
            "ram": {"freeBytes": 1024 * 1024 * 1024},
            "gpus": [],
        },
        model_id="new",
        upscale=False,
        adaptive_batching=False,
    )
    assert plan["fit"]["status"] == "unknown"
    assert plan["requirements"]["ram"]["ratio"] > 1
    assert plan["requirements"]["vram"]["ratio"] is None

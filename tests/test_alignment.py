from types import SimpleNamespace
import threading
import time

import numpy as np

from speech_server.alignment import AlignmentEngine, AlignmentError
from speech_server.config import AlignerSpec, ServerConfig


def test_alignment_resamples_final_audio_to_16khz_and_tracks_load(tmp_path):
    model = tmp_path / "aligner.gguf"
    model.write_bytes(b"model")
    seen = {}

    def align(model_path, transcript, pcm, **kwargs):
        seen.update(
            model_path=model_path,
            transcript=transcript,
            samples=len(pcm),
            kwargs=kwargs,
        )
        return [
            SimpleNamespace(text="Hello", start=kwargs["t_offset"], end=kwargs["t_offset"] + 0.4),
            SimpleNamespace(text="world", start=kwargs["t_offset"] + 0.9, end=kwargs["t_offset"] + 1.1),
        ]

    cfg = ServerConfig(
        models={},
        aligner=AlignerSpec("canary", "canary-ctc", model, ["en"], n_threads=3),
    )
    engine = AlignmentEngine(cfg, align_func=align)
    words = engine.align("Hello", np.zeros(48000, dtype=np.float32), 48000, t_offset=1.25)
    assert seen["samples"] == 16000
    assert seen["transcript"] == "Hello"
    assert seen["kwargs"]["n_threads"] == 3
    assert words == [
        {"text": "Hello", "start": 1.25, "end": 1.65},
        {"text": "world", "start": 2.15, "end": 2.25},
    ]
    assert engine.loaded is True
    assert engine.supports_language("English") is True


def test_aligner_is_loaded_only_after_success(tmp_path):
    model = tmp_path / "aligner.gguf"
    model.write_bytes(b"model")

    def fail(*args, **kwargs):
        raise RuntimeError("bad model")

    cfg = ServerConfig(
        models={}, aligner=AlignerSpec("canary", "canary-ctc", model, ["en"])
    )
    engine = AlignmentEngine(cfg, align_func=fail)
    try:
        engine.align("Hello", np.zeros(1600, dtype=np.float32), 16000, t_offset=0)
    except AlignmentError:
        pass
    else:
        raise AssertionError("alignment failure was not reported")
    assert engine.loaded is False


def test_nonfinite_aligner_timestamps_are_rejected(tmp_path):
    model = tmp_path / "aligner.gguf"
    model.write_bytes(b"model")
    cfg = ServerConfig(
        models={}, aligner=AlignerSpec("canary", "canary-ctc", model, ["en"])
    )
    engine = AlignmentEngine(
        cfg,
        align_func=lambda *args, **kwargs: [
            SimpleNamespace(text="bad", start=float("nan"), end=0.1)
        ],
    )
    try:
        engine.align("bad", np.zeros(1600, dtype=np.float32), 16000, t_offset=0)
    except AlignmentError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite alignment escaped validation")
    assert engine.loaded is False


def test_process_static_aligner_calls_are_serialized(tmp_path):
    model = tmp_path / "aligner.gguf"
    model.write_bytes(b"model")
    active = 0
    peak = 0
    lock = threading.Lock()

    def align(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return [SimpleNamespace(text="ok", start=0.0, end=0.05)]

    cfg = ServerConfig(
        models={}, aligner=AlignerSpec("canary", "canary-ctc", model, ["en"])
    )
    engine = AlignmentEngine(cfg, align_func=align)
    threads = [
        threading.Thread(
            target=engine.align,
            args=("ok", np.zeros(1600, dtype=np.float32), 16000),
            kwargs={"t_offset": 0},
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 1

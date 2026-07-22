import wave
from pathlib import Path

import numpy as np
import pytest

from speech_server.config import ModelSpec, ServerConfig
from speech_server.engine import EngineBusyError, SessionPool, _prepare_chatterbox_voice
from tests.fakes import FakeSession


def _cfg(limit=1):
    def model(model_id):
        return ModelSpec(
            id=model_id,
            backend="omnivoice",
            model_path=Path(f"{model_id}.gguf"),
            sample_rate=24000,
        )

    return ServerConfig(models={"a": model("a"), "b": model("b")}, resident_limit=limit)


def test_lazy_load_and_reuse():
    made = []
    pool = SessionPool(
        _cfg(), session_factory=lambda cfg, spec: made.append(spec.id) or FakeSession()
    )
    first = pool.acquire("a")
    second = pool.acquire("a")
    assert first is second and made == ["a"]


def test_lru_eviction_closes():
    pool = SessionPool(_cfg(limit=1), session_factory=lambda cfg, spec: FakeSession())
    first = pool.acquire("a")
    pool.acquire("b")
    assert first.raw.closed is True


def test_lru_victim_is_closed_before_replacement_is_constructed():
    first = None

    def factory(cfg, spec):
        nonlocal first
        if spec.id == "b":
            assert first is not None and first.raw.closed is True
        return FakeSession()

    pool = SessionPool(_cfg(limit=1), session_factory=factory)
    first = pool.acquire("a")
    pool.acquire("b")


def test_pinned_session_blocks_replacement_until_unpinned():
    pool = SessionPool(_cfg(limit=1), session_factory=lambda cfg, spec: FakeSession())
    first = pool.acquire("a")
    pool.pin("a")
    with pytest.raises(EngineBusyError) as exc:
        pool.acquire("b")
    assert exc.value.model_ids == ["a"]
    assert first.raw.closed is False
    assert pool.loaded_ids() == ["a"]
    pool.unpin("a")
    second = pool.acquire("b")
    assert first.raw.closed is True
    assert second.raw.closed is False
    assert pool.loaded_ids() == ["b"]


def test_replacement_plan_reports_idle_victims_and_busy_residents():
    pool = SessionPool(_cfg(limit=1), session_factory=lambda cfg, spec: FakeSession())
    pool.acquire("a")
    assert pool.replacement_plan("b") == {
        "loaded": False, "evict": ["a"], "busy": []
    }
    pool.pin("a")
    assert pool.replacement_plan("b") == {
        "loaded": False, "evict": [], "busy": ["a"]
    }
    pool.unpin("a")


def test_unknown_model():
    pool = SessionPool(_cfg(), session_factory=lambda cfg, spec: FakeSession())
    with pytest.raises(KeyError):
        pool.acquire("nope")


def test_synthesize_applies_options_and_caches_voice(tmp_path):
    pool = SessionPool(_cfg(), session_factory=lambda cfg, spec: FakeSession())
    session = pool.acquire("a")
    voice = tmp_path / "v.wav"
    voice.touch()
    session.synthesize(
        "hi", voice_wav=voice, ref_text="r", steps=24, seed=7, guidance=2.0
    )
    session.synthesize(
        "again", voice_wav=voice, ref_text="r", steps=32, seed=0, guidance=2.0
    )
    kinds = [call[0] for call in session.raw.calls]
    assert kinds.count("voice") == 1
    assert ("steps", 24) in session.raw.calls and ("steps", 32) in session.raw.calls
    assert ("seed", 7) in session.raw.calls
    assert ("seed", 0) in session.raw.calls
    assert ("tts_cfg", 2.0) in session.raw.calls


def test_omitted_seed_reseeds_each_generation(tmp_path, monkeypatch):
    generated = iter((100, 101))
    monkeypatch.setattr(
        "speech_server.engine.secrets.randbelow", lambda _limit: next(generated) - 1
    )
    pool = SessionPool(_cfg(), session_factory=lambda cfg, spec: FakeSession())
    session = pool.acquire("a")
    voice = tmp_path / "v.wav"
    voice.touch()

    for text in ("first", "second"):
        session.synthesize(
            text,
            voice_wav=voice,
            ref_text="r",
            steps=None,
            seed=None,
            guidance=None,
        )

    assert [call for call in session.raw.calls if call[0] == "seed"] == [
        ("seed", 100),
        ("seed", 101),
    ]


def test_overwriting_voice_file_invalidates_session_voice_cache(tmp_path):
    pool = SessionPool(_cfg(), session_factory=lambda cfg, spec: FakeSession())
    session = pool.acquire("a")
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"one")
    session.synthesize(
        "hi", voice_wav=voice, ref_text="r", steps=None, seed=None, guidance=None
    )
    voice.write_bytes(b"different-size")
    session.synthesize(
        "again", voice_wav=voice, ref_text="r", steps=None, seed=None, guidance=None
    )
    assert [call[0] for call in session.raw.calls].count("voice") == 2


def test_chatterbox_redecodes_24khz_wav_that_is_not_mono_pcm16(
    tmp_path, monkeypatch
):
    voice = tmp_path / "stereo.wav"
    with wave.open(str(voice), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0" * 16)
    seen = []
    monkeypatch.setattr(
        "speech_server.engine.decode_audio_at_rate",
        lambda path, rate, session: seen.append((path, rate, session))
        or np.zeros(8, dtype=np.float32),
    )
    raw = object()

    prepared = _prepare_chatterbox_voice(raw, voice)

    assert seen == [(voice, 24000, raw)]
    with wave.open(str(prepared), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            1,
            2,
            24000,
        )

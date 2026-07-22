import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from speech_server import crisp


def test_open_session_uses_the_vendored_binding(monkeypatch, tmp_path: Path):
    captured = {}
    sentinel = object()

    def fake_session(model_path, *, lib_path, backend):
        captured.update(model_path=model_path, lib_path=lib_path, backend=backend)
        return sentinel

    monkeypatch.setattr(crisp._crispasr, "Session", fake_session)
    result = crisp.open_session(
        tmp_path / "model.gguf",
        lib_path=tmp_path / "crispasr.dll",
        backend="omnivoice",
    )
    assert result is sentinel
    assert captured == {
        "model_path": str(tmp_path / "model.gguf"),
        "lib_path": str(tmp_path / "crispasr.dll"),
        "backend": "omnivoice",
    }


def test_open_session_normalizes_native_loader_errors(monkeypatch, tmp_path: Path):
    def fail(*args, **kwargs):
        raise OSError("missing dependency")

    monkeypatch.setattr(crisp._crispasr, "Session", fail)
    with pytest.raises(crisp.CrispBindingError, match="missing dependency"):
        crisp.open_session(
            tmp_path / "model.gguf",
            lib_path=tmp_path / "crispasr.dll",
            backend="omnivoice",
        )


def test_align_words_uses_the_vendored_binding(monkeypatch):
    expected = [object()]
    monkeypatch.setattr(crisp._crispasr, "align_words", lambda *args, **kwargs: expected)
    pcm = np.zeros(10, dtype=np.float32)
    assert crisp.align_words("aligner.gguf", "hello", pcm, n_threads=2) is expected


def test_transcription_rejects_malformed_native_timestamps():
    class Session:
        def transcribe(self, pcm, **kwargs):
            return [
                SimpleNamespace(
                    text="bad",
                    start=0.0,
                    end=float("nan"),
                    words=None,
                )
            ]

    with pytest.raises(crisp.CrispBindingError, match="malformed timing"):
        crisp.transcribe_session(
            Session(), np.zeros(160, dtype=np.float32),
            sample_rate=16000, language=None,
        )


def test_session_library_failure_is_explicit():
    with pytest.raises(crisp.CrispBindingError, match="no native library handle"):
        crisp._session_library(object())


def test_synthesize_raw_session_uses_unwatermarked_native_abi():
    samples = (ctypes.c_float * 3)(0.25, -0.5, 0.75)
    freed = []

    class NativeFunction:
        def __init__(self, implementation):
            self.implementation = implementation

        def __call__(self, *args):
            return self.implementation(*args)

    def synthesize(handle, text, count_out):
        assert handle == 123
        assert text == b"hello"
        ctypes.cast(count_out, ctypes.POINTER(ctypes.c_int))[0] = len(samples)
        return ctypes.cast(samples, ctypes.POINTER(ctypes.c_float))

    library = SimpleNamespace(
        crispasr_session_synthesize_raw=NativeFunction(synthesize),
        crispasr_pcm_free=NativeFunction(lambda pcm: freed.append(pcm)),
    )
    session = SimpleNamespace(_lib=library, _handle=123)

    result = crisp.synthesize_raw_session(session, "hello")

    np.testing.assert_array_equal(result, np.array([0.25, -0.5, 0.75], np.float32))
    assert len(freed) == 1


def test_synthesize_raw_session_rejects_runtime_without_raw_abi():
    session = SimpleNamespace(_lib=SimpleNamespace(), _handle=123)
    with pytest.raises(crisp.CrispBindingError, match="raw TTS synthesis"):
        crisp.synthesize_raw_session(session, "hello")


def test_audio_decode_rejects_unexpected_output_rate(tmp_path: Path):
    samples = (ctypes.c_float * 2)(0.25, -0.25)

    class NativeFunction:
        def __init__(self, implementation):
            self.implementation = implementation

        def __call__(self, *args):
            return self.implementation(*args)

    def load(path, requested_rate, pcm_out, count_out, rate_out):
        ctypes.cast(
            pcm_out, ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        )[0] = ctypes.cast(samples, ctypes.POINTER(ctypes.c_float))
        ctypes.cast(count_out, ctypes.POINTER(ctypes.c_int))[0] = 2
        ctypes.cast(rate_out, ctypes.POINTER(ctypes.c_int))[0] = requested_rate + 1
        return 0

    class Library:
        crispasr_audio_load_at_rate = NativeFunction(load)
        crispasr_audio_free = NativeFunction(lambda pcm: None)

    session = type("Session", (), {"_lib": Library()})()
    with pytest.raises(crisp.CrispBindingError, match="not the requested"):
        crisp.decode_audio_at_rate(tmp_path / "voice.wav", 24000, session=session)


def test_registry_default_bundle_reads_the_public_native_abi(monkeypatch, tmp_path: Path):
    class NativeFunction:
        def __init__(self, implementation):
            self.implementation = implementation

        def __call__(self, *args):
            return self.implementation(*args)

    def info(backend, canonical, canonical_cap, license_name, license_cap, accepts):
        assert backend == b"omnivoice"
        canonical.value = b"omnivoice"
        license_name.value = b"MIT"
        ctypes.cast(accepts, ctypes.POINTER(ctypes.c_int32))[0] = 0
        return 2

    rows = [
        (0, b"omnivoice-f16.gguf", b"https://example/model", b"~1.2 GB"),
        (1, b"codec-f16.gguf", b"https://example/codec", b"~400 MB"),
    ]

    def artifact(backend, index, kind, filename, filename_cap, url, url_cap, size, size_cap):
        row = rows[index]
        ctypes.cast(kind, ctypes.POINTER(ctypes.c_int32))[0] = row[0]
        filename.value = row[1]
        url.value = row[2]
        size.value = row[3]
        return 0

    library = SimpleNamespace(
        crispasr_registry_default_bundle_info_abi=NativeFunction(info),
        crispasr_registry_default_bundle_artifact_abi=NativeFunction(artifact),
    )
    monkeypatch.setattr(crisp, "load_library", lambda path, symbols: library)
    bundle = crisp.registry_default_bundle("omnivoice", lib_path=tmp_path / "lib")
    assert bundle.backend == "omnivoice"
    assert bundle.license == "MIT"
    assert [item.kind for item in bundle.artifacts] == ["primary", "companion"]
    assert bundle.artifacts[0].filename == "omnivoice-f16.gguf"


def test_registry_default_bundle_rejects_a_missing_primary(monkeypatch, tmp_path: Path):
    class NativeFunction:
        def __init__(self, implementation):
            self.implementation = implementation

        def __call__(self, *args):
            return self.implementation(*args)

    def info(backend, canonical, canonical_cap, license_name, license_cap, accepts):
        canonical.value = b"broken"
        return 1

    def artifact(backend, index, kind, filename, filename_cap, url, url_cap, size, size_cap):
        ctypes.cast(kind, ctypes.POINTER(ctypes.c_int32))[0] = 2
        filename.value = b"extra.gguf"
        url.value = b"https://example/extra"
        return 0

    library = SimpleNamespace(
        crispasr_registry_default_bundle_info_abi=NativeFunction(info),
        crispasr_registry_default_bundle_artifact_abi=NativeFunction(artifact),
    )
    monkeypatch.setattr(crisp, "load_library", lambda path, symbols: library)
    with pytest.raises(crisp.CrispBindingError, match="no primary"):
        crisp.registry_default_bundle("broken", lib_path=tmp_path / "lib")

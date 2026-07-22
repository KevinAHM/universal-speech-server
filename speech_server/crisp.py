"""Stable boundary between speech-server code and the pinned CrispASR binding."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ._vendor import crispasr as _crispasr


class CrispBindingError(RuntimeError):
    """The pinned binding and selected native runtime cannot perform an operation."""


@dataclass(frozen=True)
class RegistryArtifact:
    kind: str
    filename: str
    url: str
    approx_size: str


@dataclass(frozen=True)
class RegistryBundle:
    backend: str
    license: str
    requires_acceptance: bool
    artifacts: tuple[RegistryArtifact, ...]


def open_session(model_path: Path, *, lib_path: Path, backend: str):
    """Open a native session through the speech server's pinned binding."""
    try:
        return _crispasr.Session(
            str(model_path), lib_path=str(lib_path), backend=backend
        )
    except (OSError, RuntimeError) as exc:
        raise CrispBindingError(f"cannot open CrispASR session: {exc}") from exc


def align_words(
    model_path: str,
    transcript: str,
    pcm: np.ndarray,
    **kwargs: Any,
):
    """Forward forced alignment through the pinned binding."""
    return _crispasr.align_words(model_path, transcript, pcm, **kwargs)


def transcribe_session(
    session: Any,
    pcm: np.ndarray,
    *,
    sample_rate: int,
    language: str | None,
    hotwords: Iterable[str] = (),
    hotword_boost: float = 2.0,
) -> dict[str, Any]:
    """Transcribe through the pinned binding and normalize unstable details."""
    try:
        terms = [str(term).strip() for term in hotwords if str(term).strip()]
        set_hotwords = getattr(session, "set_hotwords", None)
        if set_hotwords is not None:
            set_hotwords(",".join(terms), hotword_boost)
        elif terms:
            raise CrispBindingError("the selected runtime does not support bias terms")
        segments = session.transcribe(
            np.ascontiguousarray(pcm, dtype=np.float32),
            sample_rate=sample_rate,
            language=language,
        )
        detector = getattr(session, "detected_language", None)
        detected = detector() if callable(detector) else None
    except CrispBindingError:
        raise
    except (AttributeError, OSError, RuntimeError, ValueError) as exc:
        raise CrispBindingError(f"transcription failed: {exc}") from exc

    try:
        def interval(start_value: Any, end_value: Any) -> tuple[float, float]:
            start = float(start_value)
            end = float(end_value)
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError(f"invalid timestamp interval {start!r}..{end!r}")
            return start, end

        normalized_segments = []
        all_words = []
        for segment in segments:
            words = []
            for word in (getattr(segment, "words", None) or ()):
                word_start, word_end = interval(word.start, word.end)
                words.append(
                    {
                        "text": str(word.text),
                        "startTimeSeconds": word_start,
                        "endTimeSeconds": word_end,
                    }
                )
            segment_start, segment_end = interval(segment.start, segment.end)
            all_words.extend(words)
            normalized_segments.append(
                {
                    "text": str(segment.text),
                    "startTimeSeconds": segment_start,
                    "endTimeSeconds": segment_end,
                    "words": words,
                }
            )
        detected = str(detected or "").strip().lower()
    except (AttributeError, TypeError, ValueError) as exc:
        raise CrispBindingError(
            f"transcription returned malformed timing data: {exc}"
        ) from exc
    return {
        "text": " ".join(
            segment["text"].strip()
            for segment in normalized_segments
            if segment["text"].strip()
        ).strip(),
        "detectedLanguage": None if detected in {"", "unknown", "auto"} else detected,
        "segments": normalized_segments,
        "words": all_words,
    }


def load_library(lib_path: Path, required_symbols: Iterable[str] = ()):
    """Load a selected runtime and verify its required public C symbols."""
    try:
        library = ctypes.CDLL(str(lib_path))
    except OSError as exc:
        raise CrispBindingError(f"cannot load libcrispasr: {exc}") from exc
    missing = [name for name in required_symbols if not hasattr(library, name)]
    if missing:
        raise CrispBindingError(
            "loaded libcrispasr is missing required API: " + ", ".join(missing)
        )
    return library


def registry_default_bundle(backend: str, *, lib_path: Path) -> RegistryBundle | None:
    """Resolve CrispASR's exact canonical ``-m auto`` artifact bundle."""
    if not backend:
        return None
    library = load_library(
        lib_path,
        (
            "crispasr_registry_default_bundle_info_abi",
            "crispasr_registry_default_bundle_artifact_abi",
        ),
    )
    info = library.crispasr_registry_default_bundle_info_abi
    info.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    info.restype = ctypes.c_int
    canonical = ctypes.create_string_buffer(256)
    license_name = ctypes.create_string_buffer(2048)
    requires_acceptance = ctypes.c_int32()
    try:
        count = info(
            backend.encode("utf-8"),
            canonical,
            len(canonical),
            license_name,
            len(license_name),
            ctypes.byref(requires_acceptance),
        )
    except (AttributeError, OSError, ValueError) as exc:
        raise CrispBindingError(f"cannot resolve CrispASR registry bundle: {exc}") from exc
    if count == 0:
        return None
    if count < 0:
        raise CrispBindingError(f"CrispASR registry bundle lookup failed (rc={count})")
    if count > 128:
        raise CrispBindingError("CrispASR registry returned an implausibly large bundle")

    artifact_fn = library.crispasr_registry_default_bundle_artifact_abi
    artifact_fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    artifact_fn.restype = ctypes.c_int
    kinds = {0: "primary", 1: "companion", 2: "extra"}
    artifacts = []
    for index in range(count):
        kind = ctypes.c_int32()
        filename = ctypes.create_string_buffer(512)
        url = ctypes.create_string_buffer(4096)
        approx_size = ctypes.create_string_buffer(128)
        try:
            rc = artifact_fn(
                backend.encode("utf-8"),
                index,
                ctypes.byref(kind),
                filename,
                len(filename),
                url,
                len(url),
                approx_size,
                len(approx_size),
            )
        except (AttributeError, OSError, ValueError) as exc:
            raise CrispBindingError(
                f"cannot read CrispASR registry artifact {index}: {exc}"
            ) from exc
        if rc != 0 or kind.value not in kinds:
            raise CrispBindingError(
                f"CrispASR registry artifact {index} is invalid "
                f"(rc={rc}, kind={kind.value})"
            )
        artifacts.append(
            RegistryArtifact(
                kind=kinds[kind.value],
                filename=filename.value.decode("utf-8"),
                url=url.value.decode("utf-8"),
                approx_size=approx_size.value.decode("utf-8"),
            )
        )
    if not artifacts or artifacts[0].kind != "primary":
        raise CrispBindingError("CrispASR registry bundle has no primary artifact")
    return RegistryBundle(
        backend=canonical.value.decode("utf-8"),
        license=license_name.value.decode("utf-8"),
        requires_acceptance=requires_acceptance.value != 0,
        artifacts=tuple(artifacts),
    )


def _session_library(session: Any):
    """Confine the sole dependency on the upstream binding's private handle."""
    library = getattr(session, "_lib", None)
    if library is None:
        raise CrispBindingError("the CrispASR session has no native library handle")
    return library


def synthesize_raw_session(session: Any, text: str) -> np.ndarray:
    """Synthesize unwatermarked PCM for server-side audio processing.

    CrispASR's default session synthesis entry point embeds a watermark in the
    returned samples.  The speech server may still concatenate, resample, or
    upscale that PCM, so it must use the native raw entry point instead.
    """
    native_method = getattr(session, "synthesize_raw", None)
    if callable(native_method):
        try:
            pcm = native_method(text)
        except (OSError, RuntimeError, ValueError) as exc:
            raise CrispBindingError(f"synthesis failed: {exc}") from exc
        result = np.ascontiguousarray(pcm, dtype=np.float32)
        if result.ndim != 1 or result.size == 0:
            raise CrispBindingError("synthesis returned no mono audio")
        return result

    library = _session_library(session)
    handle = getattr(session, "_handle", None)
    if not handle:
        raise CrispBindingError("the CrispASR session has no native session handle")
    if not hasattr(library, "crispasr_session_synthesize_raw") or not hasattr(
        library, "crispasr_pcm_free"
    ):
        raise CrispBindingError(
            "the selected CrispASR runtime does not support raw TTS synthesis"
        )

    synthesize = library.crispasr_session_synthesize_raw
    synthesize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    synthesize.restype = ctypes.POINTER(ctypes.c_float)
    free = library.crispasr_pcm_free
    free.argtypes = [ctypes.POINTER(ctypes.c_float)]
    free.restype = None

    count = ctypes.c_int()
    try:
        pcm = synthesize(handle, text.encode("utf-8"), ctypes.byref(count))
    except (ctypes.ArgumentError, OSError, ValueError) as exc:
        raise CrispBindingError(f"synthesis failed: {exc}") from exc
    if not pcm or count.value <= 0:
        if pcm:
            free(pcm)
        raise CrispBindingError("synthesis returned no audio")
    try:
        return np.ascontiguousarray(
            np.ctypeslib.as_array(pcm, shape=(count.value,)).copy(),
            dtype=np.float32,
        )
    finally:
        free(pcm)


def decode_audio_at_rate(
    path: Path,
    sample_rate: int,
    *,
    lib_path: Path | None = None,
    session: Any | None = None,
) -> np.ndarray:
    """Decode an audio file to contiguous float32 mono PCM with libcrispasr."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if session is not None:
        library = _session_library(session)
    elif lib_path is not None and Path(lib_path).is_file():
        library = load_library(
            Path(lib_path),
            ("crispasr_audio_load_at_rate", "crispasr_audio_free"),
        )
    else:
        raise CrispBindingError("libcrispasr is unavailable for audio decoding")

    if not hasattr(library, "crispasr_audio_load_at_rate") or not hasattr(
        library, "crispasr_audio_free"
    ):
        raise CrispBindingError("CrispASR audio decoder is unavailable")
    load = library.crispasr_audio_load_at_rate
    load.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    load.restype = ctypes.c_int
    free = library.crispasr_audio_free
    free.argtypes = [ctypes.POINTER(ctypes.c_float)]
    free.restype = None

    pcm = ctypes.POINTER(ctypes.c_float)()
    n_samples = ctypes.c_int()
    decoded_rate = ctypes.c_int()
    rc = load(
        str(path).encode("utf-8"),
        sample_rate,
        ctypes.byref(pcm),
        ctypes.byref(n_samples),
        ctypes.byref(decoded_rate),
    )
    if rc != 0 or not pcm or n_samples.value <= 0:
        if pcm:
            free(pcm)
        raise CrispBindingError(f"failed to decode audio {path} (rc={rc})")
    try:
        if decoded_rate.value != sample_rate:
            raise CrispBindingError(
                f"CrispASR decoded {path} at {decoded_rate.value} Hz, "
                f"not the requested {sample_rate} Hz"
            )
        return np.ascontiguousarray(
            np.ctypeslib.as_array(pcm, shape=(n_samples.value,)).copy(),
            dtype=np.float32,
        )
    finally:
        free(pcm)

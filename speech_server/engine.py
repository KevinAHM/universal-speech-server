"""CrispASR session pool using the vendored ctypes bindings."""

import os
import secrets
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import ModelSpec, ServerConfig
from .crisp import (
    CrispBindingError,
    decode_audio_at_rate,
    open_session,
    synthesize_raw_session,
    transcribe_session,
)


class EngineError(RuntimeError):
    pass


class EngineBusyError(EngineError):
    """A replacement model cannot load until pinned residents finish."""

    def __init__(self, model_ids: list[str]):
        self.model_ids = list(model_ids)
        joined = ", ".join(self.model_ids) or "resident model"
        super().__init__(f"waiting for active model(s) to finish: {joined}")


_DLL_DIR_HANDLES: list[object] = []
_REGISTERED_DLL_DIRS: set[str] = set()


def _register_windows_dll_dirs(cfg: ServerConfig) -> None:
    """Make native and CUDA dependencies visible to Python's DLL loader."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates: list[Path] = []
    if cfg.lib_path is not None:
        candidates.append(Path(cfg.lib_path).resolve().parent)
    cuda_root = os.getenv("CUDA_PATH", "").strip()
    if cuda_root:
        candidates.extend((Path(cuda_root) / "bin", Path(cuda_root) / "bin" / "x64"))

    for directory in candidates:
        key = os.path.normcase(str(directory.resolve()))
        if directory.is_dir() and key not in _REGISTERED_DLL_DIRS:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
            _REGISTERED_DLL_DIRS.add(key)


def _is_24khz_mono_pcm16(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            return (
                wav.getframerate() == 24000
                and wav.getnchannels() == 1
                and wav.getsampwidth() == 2
                and wav.getcomptype() == "NONE"
            )
    except (EOFError, OSError, wave.Error):
        return False


def _prepare_chatterbox_voice(raw, voice_wav: Path) -> Path:
    """Cache a 24 kHz PCM16 copy using CrispASR's native audio decoder."""
    if _is_24khz_mono_pcm16(voice_wav):
        return voice_wav

    stat = voice_wav.stat()
    cache_dir = voice_wav.parent / ".speech_server_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{voice_wav.stem}.{stat.st_size}.{stat.st_mtime_ns}.24000.wav"
    if cached.is_file() and _is_24khz_mono_pcm16(cached):
        return cached

    try:
        audio = decode_audio_at_rate(voice_wav, 24000, session=raw)
    except CrispBindingError as exc:
        raise EngineError(str(exc)) from exc

    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    with tempfile.NamedTemporaryFile(
        prefix=f".{voice_wav.stem}.", suffix=".wav", dir=cache_dir, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with wave.open(str(temporary_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm16)
        os.replace(temporary_path, cached)
    finally:
        temporary_path.unlink(missing_ok=True)
    return cached


def _open_real_session(cfg: ServerConfig, spec: ModelSpec):
    _register_windows_dll_dirs(cfg)
    if cfg.lib_path is None or not Path(cfg.lib_path).is_file():
        raise EngineError(f"libcrispasr not found (SPEECH_SERVER_LIB={cfg.lib_path})")
    if not spec.installed:
        missing = [str(path) for path in spec.component_paths if not path.is_file()]
        raise EngineError(f"model component missing: {', '.join(missing)}")
    try:
        session = open_session(
            spec.model_path, lib_path=Path(cfg.lib_path), backend=spec.backend
        )
        if spec.codec_path:
            session.set_codec_path(str(spec.codec_path))
        return session
    except (CrispBindingError, RuntimeError) as exc:
        raise EngineError(str(exc)) from exc


class ModelSession:
    def __init__(self, spec: ModelSpec, raw):
        self.spec = spec
        self.raw = raw
        self.lock = threading.Lock()
        self._voice_key: Optional[tuple[str, int, int, Optional[str]]] = None
        self._control_defaults = {
            control.id: control.default for control in spec.controls
        }
        self._steps_overridden = False
        self._guidance_overridden = False
        self._exaggeration_overridden = False

    def _set_voice(self, voice_wav: Path, ref_text: Optional[str]) -> None:
        # A voice ID may be overwritten in place through the REST API.  Path and
        # transcript alone would then incorrectly reuse the previous encoded
        # reference for the lifetime of this session.
        stat = voice_wav.stat()
        voice_key = (str(voice_wav), stat.st_size, stat.st_mtime_ns, ref_text)
        if voice_key == self._voice_key:
            return
        prepared_voice = (
            _prepare_chatterbox_voice(self.raw, voice_wav)
            if self.spec.backend == "chatterbox"
            else voice_wav
        )
        try:
            self.raw.set_voice(str(prepared_voice), ref_text)
        except RuntimeError as exc:
            if "rc=-3" in str(exc):
                raise EngineError(
                    "the selected CrispASR runtime can open backend "
                    f"{self.spec.backend!r} but does not wire it into the unified "
                    "voice-cloning API; select a newer runtime build"
                ) from exc
            raise
        self._voice_key = voice_key

    def prepare_voice(self, voice_wav: Path, ref_text: Optional[str]) -> None:
        """Apply a reference without synthesis, populating backend disk caches."""
        with self.lock:
            try:
                self._set_voice(voice_wav, ref_text)
            except (AttributeError, OSError, RuntimeError, ValueError) as exc:
                raise EngineError(str(exc)) from exc

    def synthesize(
        self,
        text: str,
        *,
        voice_wav: Path,
        ref_text: Optional[str],
        steps: Optional[int],
        seed: Optional[int],
        guidance: Optional[float],
        exaggeration: Optional[float] = None,
    ) -> np.ndarray:
        with self.lock:
            try:
                self._set_voice(voice_wav, ref_text)
                steps_explicit = steps is not None
                if steps is None and self._steps_overridden:
                    steps = self._control_defaults.get("numSteps")
                if steps is not None:
                    self.raw.set_tts_steps(int(steps))
                    self._steps_overridden = steps_explicit
                seed_explicit = seed is not None
                # OmniVoice's native default (and its interpretation of seed 0)
                # is the fixed seed 42. Generate a nonzero uint32-compatible seed
                # for ordinary requests so repeated text is not deterministic;
                # an explicit client seed remains exactly reproducible.
                effective_seed = (
                    int(seed)
                    if seed_explicit
                    else secrets.randbelow(0xFFFFFFFF) + 1
                )
                try:
                    self.raw.set_tts_seed(effective_seed)
                except (AttributeError, RuntimeError):
                    if seed_explicit:
                        raise
                guidance_explicit = guidance is not None
                if guidance is None and self._guidance_overridden:
                    guidance = self._control_defaults.get("guidanceScale")
                if guidance is not None:
                    if self.spec.backend == "omnivoice":
                        set_cfg = getattr(self.raw, "set_tts_cfg_scale", None)
                        if set_cfg:
                            set_cfg(float(guidance))
                        else:
                            os.environ["CRISPASR_OMNIVOICE_GUIDANCE"] = str(guidance)
                    elif self.spec.backend == "chatterbox":
                        self.raw.set_cfg_weight(float(guidance))
                    else:
                        try:
                            self.raw.set_cfg_weight(float(guidance))
                        except (RuntimeError, AttributeError):
                            pass
                    self._guidance_overridden = guidance_explicit
                exaggeration_explicit = exaggeration is not None
                if exaggeration is None and self._exaggeration_overridden:
                    exaggeration = self._control_defaults.get("exaggeration")
                if exaggeration is not None:
                    self.raw.set_exaggeration(float(exaggeration))
                    self._exaggeration_overridden = exaggeration_explicit
                return synthesize_raw_session(self.raw, text)
            except (AttributeError, OSError, RuntimeError, ValueError) as exc:
                raise EngineError(str(exc)) from exc

    def close(self) -> None:
        close = getattr(self.raw, "close", None)
        if close:
            close()

    def restore(self, audio: np.ndarray, input_sample_rate: int) -> np.ndarray:
        """Run an audio-to-audio backend while holding its session lock."""
        with self.lock:
            try:
                self.raw.set_pcm_sample_rate(int(input_sample_rate))
                restored, _ = self.raw.speech_to_speech(audio)
                restored = np.asarray(restored, dtype=np.float32)
                if restored.size == 0:
                    raise EngineError("upscaler produced no audio")
                return restored
            except (AttributeError, RuntimeError, ValueError) as exc:
                raise EngineError(str(exc)) from exc

    def transcribe(
        self,
        pcm: np.ndarray,
        *,
        sample_rate: int,
        language: str | None,
        bias_terms: list[str],
    ) -> dict:
        if self.spec.task != "asr" or self.spec.asr is None:
            raise EngineError(f"model {self.spec.id!r} is not an ASR model")
        with self.lock:
            started = time.perf_counter()
            try:
                result = transcribe_session(
                    self.raw,
                    pcm,
                    sample_rate=sample_rate,
                    language=language,
                    hotwords=bias_terms,
                    hotword_boost=self.spec.asr.hotword_boost,
                )
            except CrispBindingError as exc:
                raise EngineError(str(exc)) from exc
            result["inferenceMs"] = (time.perf_counter() - started) * 1000.0
            return result


class SessionPool:
    def __init__(
        self,
        cfg: ServerConfig,
        session_factory: Optional[
            Callable[[ServerConfig, ModelSpec], object]
        ] = None,
    ):
        self._cfg = cfg
        self._factory = session_factory or _open_real_session
        self._sessions: OrderedDict[str, ModelSession] = OrderedDict()
        self._upscaler: Optional[ModelSession] = None
        self._pins: dict[str, int] = {}
        self._lock = threading.Lock()

    def _evict_idle_locked(self, protected_id: str | None = None) -> None:
        while len(self._sessions) > max(1, self._cfg.resident_limit):
            victim_id = next(
                (
                    model_id
                    for model_id in self._sessions
                    if self._pins.get(model_id, 0) == 0
                    and model_id != protected_id
                ),
                None,
            )
            if victim_id is None:
                # Concurrent requests may temporarily exceed the resident
                # limit. Their final unpin performs the deferred eviction.
                return
            old = self._sessions.pop(victim_id)
            old.close()

    def _replacement_plan_locked(self, model_id: str) -> dict:
        if model_id in self._sessions:
            return {"loaded": True, "evict": [], "busy": []}
        evictions_needed = max(
            0,
            len(self._sessions) + 1 - max(1, self._cfg.resident_limit),
        )
        idle = [
            resident_id
            for resident_id in self._sessions
            if self._pins.get(resident_id, 0) == 0
        ]
        victims = idle[:evictions_needed]
        remaining = evictions_needed - len(victims)
        busy = (
            [
                resident_id
                for resident_id in self._sessions
                if self._pins.get(resident_id, 0) > 0
            ][:remaining]
            if remaining > 0
            else []
        )
        return {"loaded": False, "evict": victims, "busy": busy}

    def replacement_plan(self, model_id: str) -> dict:
        self.spec(model_id)
        with self._lock:
            return self._replacement_plan_locked(model_id)

    def stack_replacement_plan(self, model_ids: list[str]) -> dict:
        desired = list(dict.fromkeys(model_ids))
        for model_id in desired:
            self.spec(model_id)
        with self._lock:
            if len(desired) > max(1, self._cfg.resident_limit):
                return {
                    "capacity": False,
                    "loaded": [mid for mid in desired if mid in self._sessions],
                    "load": [mid for mid in desired if mid not in self._sessions],
                    "evict": [],
                    "busy": [],
                }
            obsolete = [mid for mid in self._sessions if mid not in desired]
            busy = [mid for mid in obsolete if self._pins.get(mid, 0) > 0]
            return {
                "capacity": True,
                "loaded": [mid for mid in desired if mid in self._sessions],
                "load": [mid for mid in desired if mid not in self._sessions],
                "evict": [mid for mid in obsolete if mid not in busy],
                "busy": busy,
            }

    def evict_except(self, model_ids: list[str]) -> list[str]:
        desired = set(model_ids)
        with self._lock:
            busy = [
                model_id
                for model_id in self._sessions
                if model_id not in desired and self._pins.get(model_id, 0) > 0
            ]
            if busy:
                raise EngineBusyError(busy)
            evicted = []
            for model_id in list(self._sessions):
                if model_id in desired:
                    continue
                self._sessions.pop(model_id).close()
                evicted.append(model_id)
            return evicted

    def pin(self, model_id: str) -> None:
        """Prevent a request's model session from being evicted between chunks."""
        with self._lock:
            self._pins[model_id] = self._pins.get(model_id, 0) + 1

    def unpin(self, model_id: str) -> None:
        with self._lock:
            count = self._pins.get(model_id, 0)
            if count <= 1:
                self._pins.pop(model_id, None)
            else:
                self._pins[model_id] = count - 1
            self._evict_idle_locked()

    def spec(self, model_id: str) -> ModelSpec:
        return self._cfg.models[model_id]

    def acquire(self, model_id: str) -> ModelSession:
        spec = self.spec(model_id)
        with self._lock:
            if model_id in self._sessions:
                self._sessions.move_to_end(model_id)
                return self._sessions[model_id]
            plan = self._replacement_plan_locked(model_id)
            if plan["busy"]:
                raise EngineBusyError(plan["busy"])
            # Evict before constructing the replacement so model swaps never
            # require both complete TTS sessions to fit on the device.
            for victim_id in plan["evict"]:
                old = self._sessions.pop(victim_id)
                old.close()
            session = ModelSession(spec, self._factory(self._cfg, spec))
            self._sessions[model_id] = session
            return session

    def loaded_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def model_residency(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "kind": "model",
                    "id": model_id,
                    "task": session.spec.task,
                    "loaded": True,
                    "busy": self._pins.get(model_id, 0) > 0,
                    "evictable": self._pins.get(model_id, 0) == 0,
                    "sticky": False,
                    "resources": session.spec.resource_requirements(),
                }
                for model_id, session in self._sessions.items()
            ]

    def acquire_upscaler(self) -> ModelSession:
        spec = self._cfg.upscaler
        if spec is None:
            raise EngineError("upscaling is not configured")
        with self._lock:
            if self._upscaler is None:
                self._upscaler = ModelSession(spec, self._factory(self._cfg, spec))
            return self._upscaler

    def upscaler_loaded(self) -> bool:
        with self._lock:
            return self._upscaler is not None

    def upscaler_busy(self) -> bool:
        with self._lock:
            return bool(self._upscaler and self._upscaler.lock.locked())

    def upscaler_residency(self) -> dict | None:
        with self._lock:
            if self._upscaler is None:
                return None
            return {
                "kind": "upscaler",
                "id": self._upscaler.spec.id,
                "loaded": True,
                "busy": self._upscaler.lock.locked(),
                "evictable": False,
                "sticky": True,
                "resources": self._upscaler.spec.resource_requirements(),
            }

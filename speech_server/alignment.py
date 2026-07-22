"""Lazy CrispASR forced-alignment adapter."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .audio import load_audio_mono, resample_mono
from .config import ServerConfig
from .crisp import align_words


class AlignmentError(RuntimeError):
    pass


class AlignmentEngine:
    def __init__(
        self,
        cfg: ServerConfig,
        align_func: Optional[Callable] = None,
    ):
        self._cfg = cfg
        self._align_func = align_func
        self._loaded = False
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._cfg.aligner is not None and self._cfg.aligner.installed

    @property
    def loaded(self) -> bool:
        with self._state_lock:
            return self._loaded

    @property
    def busy(self) -> bool:
        return self._inference_lock.locked()

    @property
    def spec(self):
        return self._cfg.aligner

    def supports_language(self, language: str) -> bool:
        if not self._cfg.aligner:
            return False
        requested = (language or "").lower().replace("_", "-")
        requested = requested.split("-", 1)[0]
        aliases = {"english": "en"}
        requested = aliases.get(requested, requested)
        supported = {value.lower() for value in self._cfg.aligner.languages}
        return requested in supported or "*" in supported

    def _resolve_func(self) -> Callable:
        if self._align_func is not None:
            return self._align_func
        self._align_func = align_words
        return self._align_func

    def align(
        self,
        transcript: str,
        pcm: np.ndarray,
        sample_rate: int,
        *,
        t_offset: float,
    ) -> list[dict]:
        spec = self._cfg.aligner
        if spec is None or not spec.installed:
            raise AlignmentError("forced alignment is not configured")
        pcm_16k = resample_mono(pcm, sample_rate, 16000)
        # CrispASR currently owns a process-static aligner cache. Serialize the
        # load/inference call so concurrent WebSocket requests cannot race that
        # cache or mutate its CUDA graph state simultaneously.
        with self._inference_lock:
            try:
                words = self._resolve_func()(
                    str(spec.model_path),
                    transcript,
                    pcm_16k,
                    t_offset=t_offset,
                    n_threads=spec.n_threads,
                    lib_path=str(self._cfg.lib_path) if self._cfg.lib_path else None,
                )
            except Exception as exc:
                raise AlignmentError("forced alignment failed") from exc
        if not words:
            raise AlignmentError("forced alignment returned no words")
        lower = float(t_offset)
        upper = lower + len(pcm) / sample_rate
        output = []
        previous_start = lower
        previous_end = lower
        for word in words:
            raw_start = float(word.start)
            raw_end = float(word.end)
            if not np.isfinite(raw_start) or not np.isfinite(raw_end):
                raise AlignmentError("forced alignment returned non-finite timestamps")
            start = max(previous_start, min(upper, max(lower, raw_start)))
            end = max(start, previous_end, min(upper, max(lower, raw_end)))
            output.append({"text": str(word.text), "start": start, "end": end})
            previous_start = start
            previous_end = end
        with self._state_lock:
            self._loaded = True
        return output

    def warmup_reference(self, audio_path: Path, transcript: str) -> list[dict]:
        """Load and validate the aligner using a real transcribed reference."""
        try:
            pcm = load_audio_mono(
                audio_path,
                sample_rate=16000,
                lib_path=self._cfg.lib_path,
            )
        except Exception as exc:
            raise AlignmentError("failed to decode alignment warmup reference") from exc
        return self.align(transcript, pcm, 16000, t_offset=0.0)

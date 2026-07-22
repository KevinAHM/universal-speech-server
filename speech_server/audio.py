"""PCM conversion and file-loading helpers."""

import itertools
import math
from pathlib import Path

import numpy as np

from .crisp import decode_audio_at_rate


def load_audio_mono(
    path: Path, *, sample_rate: int, lib_path: Path | None
) -> np.ndarray:
    """Decode an audio file through CrispASR into float32 mono PCM."""
    return decode_audio_at_rate(path, sample_rate, lib_path=lib_path)


def pcm16_bytes(f32: np.ndarray) -> bytes:
    return (np.clip(f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def silence_pcm16(ms: float, sample_rate: int) -> bytes:
    sample_count = int(ms / 1000.0 * sample_rate)
    return np.zeros(sample_count, dtype=np.int16).tobytes()


def trim_omnivoice_output_silence(
    f32: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Match OmniVoice's default generated-audio silence-removal stage.

    The Torch implementation slides a 500 ms RMS window in 10 ms steps at
    -50 dBFS, keeps 100 ms at the outer edges, and uses 500 ms of silence on
    each side of an internal cut. Detection and output both use quantized
    PCM16, as OmniVoice's pydub path does.

    An all-silent result is left intact.  Torch retries generation without its
    post-processor in that case; preserving the native output is the equivalent
    fail-safe without paying for a second inference.
    """
    audio = np.ascontiguousarray(f32, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("OmniVoice output must be mono")
    if audio.size == 0:
        return audio
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    pcm16 = np.clip(audio * 32768.0, -32768.0, 32767.0).astype(np.int16)
    silence_rms = 32768.0 * 10.0 ** (-50.0 / 20.0)
    total_ms = round(1000.0 * pcm16.size / sample_rate)

    def sample_at(ms: int) -> int:
        return min(pcm16.size, max(0, int(ms * (sample_rate / 1000.0))))

    squared = pcm16.astype(np.float64)
    np.square(squared, out=squared)
    energy = np.empty(squared.size + 1, dtype=np.float64)
    energy[0] = 0.0
    np.cumsum(squared, out=energy[1:])

    def rms_between(start_ms: int, end_ms: int) -> int:
        start = sample_at(start_ms)
        end = sample_at(end_ms)
        if end <= start:
            return 0
        mean_square = (energy[end] - energy[start]) / (end - start)
        return int(math.sqrt(max(0.0, mean_square)))

    min_silence_ms = 500
    seek_ms = 10
    silence_starts: list[int] = []
    if total_ms >= min_silence_ms:
        last_start = total_ms - min_silence_ms
        starts: list[int] | itertools.chain[int] = list(
            range(0, last_start + 1, seek_ms)
        )
        if last_start % seek_ms:
            starts = itertools.chain(starts, [last_start])
        silence_starts = [
            start
            for start in starts
            if rms_between(start, start + min_silence_ms) <= silence_rms
        ]

    silent_ranges: list[list[int]] = []
    if silence_starts:
        previous = silence_starts[0]
        range_start = previous
        for start in silence_starts[1:]:
            continuous = start == previous + seek_ms
            has_gap = start > previous + min_silence_ms
            if not continuous and has_gap:
                silent_ranges.append([range_start, previous + min_silence_ms])
                range_start = start
            previous = start
        silent_ranges.append([range_start, previous + min_silence_ms])

    if not silent_ranges:
        non_silent_ranges = [[0, total_ms]]
    elif silent_ranges[0] == [0, total_ms]:
        return audio
    else:
        non_silent_ranges: list[list[int]] = []
        previous_end = 0
        for start, end in silent_ranges:
            non_silent_ranges.append([previous_end, start])
            previous_end = end
        if previous_end != total_ms:
            non_silent_ranges.append([previous_end, total_ms])
        if non_silent_ranges and non_silent_ranges[0] == [0, 0]:
            non_silent_ranges.pop(0)

    keep_middle_ms = 500
    output_ranges = [
        [start - keep_middle_ms, end + keep_middle_ms]
        for start, end in non_silent_ranges
    ]
    for left, right in itertools.pairwise(output_ranges):
        if right[0] < left[1]:
            midpoint = (left[1] + right[0]) // 2
            left[1] = midpoint
            right[0] = midpoint
    chunks = [
        pcm16[sample_at(max(start, 0)) : sample_at(min(end, total_ms))]
        for start, end in output_ranges
    ]
    processed = np.concatenate(chunks) if len(chunks) > 1 else chunks[0].copy()

    def leading_silence_ms(samples: np.ndarray) -> int:
        length_ms = round(1000.0 * samples.size / sample_rate)
        trim_ms = 0
        while trim_ms < length_ms:
            begin = min(samples.size, int(trim_ms * sample_rate / 1000.0))
            end = min(samples.size, int((trim_ms + 10) * sample_rate / 1000.0))
            block = samples[begin:end].astype(np.float64)
            rms = int(math.sqrt(np.mean(block * block))) if block.size else 0
            if rms != 0 and 20.0 * math.log10(rms / 32768.0) >= -50.0:
                break
            trim_ms += 10
        return min(trim_ms, length_ms)

    leading = max(0, leading_silence_ms(processed) - 100)
    begin = min(processed.size, int(leading * sample_rate / 1000.0))
    processed = processed[begin:]
    reversed_pcm = processed[::-1]
    trailing = max(0, leading_silence_ms(reversed_pcm) - 100)
    begin = min(reversed_pcm.size, int(trailing * sample_rate / 1000.0))
    processed = reversed_pcm[begin:][::-1]
    return np.ascontiguousarray(processed.astype(np.float32) / 32768.0)


def resample_mono(f32: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Dependency-free linear resampling suitable for CTC preprocessing."""
    audio = np.asarray(f32, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if audio.size == 0 or source_rate == target_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)
    output_size = max(1, round(audio.size * target_rate / source_rate))
    source_positions = np.arange(output_size, dtype=np.float64) * (
        source_rate / target_rate
    )
    source_positions = np.minimum(source_positions, audio.size - 1)
    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, audio.size - 1)
    fraction = source_positions - left
    output = audio[left] * (1.0 - fraction) + audio[right] * fraction
    return np.ascontiguousarray(output, dtype=np.float32)

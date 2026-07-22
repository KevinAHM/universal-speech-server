"""Model-aware text duration estimation and safe physical chunk planning."""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import SegmentationSpec
from .textproc import NormalizedText, normalize_unit


@dataclass(frozen=True)
class DurationEstimate:
    seconds: float
    source: str
    confidence: str


@dataclass(frozen=True)
class UnitPart:
    id: str
    normalized: NormalizedText
    part: int = 0


@dataclass(frozen=True)
class PhysicalChunk:
    units: tuple[UnitPart, ...]
    estimate: DurationEstimate


class SegmentationError(ValueError):
    """Text cannot be partitioned beneath the configured predicted maximum."""


def _measure(text: str) -> tuple[int, int]:
    return sum(1 for char in text if not char.isspace()), len(text.split())


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate > 0 else None
    except (OSError, EOFError, wave.Error):
        return None


def estimate_duration(
    text: str,
    policy: SegmentationSpec,
    *,
    voice_wav: Path,
    ref_text: str | None,
    profile: str | None = None,
    tag_map: Mapping[str, str] | None = None,
) -> DurationEstimate:
    chars, words = _measure(text)
    ref_duration = _wav_duration(voice_wav)
    if ref_text and ref_duration and ref_duration > 0:
        comparable_ref = (
            normalize_unit(ref_text, profile, tag_map).tts_text if profile else ref_text
        )
        ref_chars, ref_words = _measure(comparable_ref)
        estimates = []
        if ref_chars:
            estimates.append(chars / (ref_chars / ref_duration))
        if ref_words:
            estimates.append(words / (ref_words / ref_duration))
        if estimates:
            return DurationEstimate(
                max(estimates) * policy.safety_factor,
                "reference-rate",
                "medium",
            )
    return DurationEstimate(
        max(
            chars / policy.fallback_characters_per_second,
            words / policy.fallback_words_per_second,
        )
        * policy.safety_factor,
        "registry-fallback",
        "low",
    )


def _split_candidates(text: str) -> list[str]:
    for pattern in (r"(?<=[.!?])\s+", r"(?<=[,;:])\s+"):
        parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
        if len(parts) > 1:
            return parts
    tokens = re.findall(r"\[[^\]]+\]|\S+", text)
    return tokens or [text]


def _split_oversized(
    unit: UnitPart,
    policy: SegmentationSpec,
    *,
    profile: str,
    tag_map: Mapping[str, str],
    voice_wav: Path,
    ref_text: str | None,
) -> list[UnitPart]:
    if estimate_duration(
        unit.normalized.tts_text,
        policy,
        voice_wav=voice_wav,
        ref_text=ref_text,
        profile=profile,
        tag_map=tag_map,
    ).seconds <= policy.max_seconds:
        return [unit]
    pieces = _split_candidates(unit.normalized.tts_text)
    if len(pieces) == 1:
        raise SegmentationError(
            f"unit {unit.id!r} contains an indivisible span predicted at "
            f"{estimate_duration(unit.normalized.tts_text, policy, voice_wav=voice_wav, ref_text=ref_text, profile=profile, tag_map=tag_map).seconds:.2f}s, "
            f"above the {policy.max_seconds:.2f}s maximum"
        )
    output: list[UnitPart] = []
    current: list[str] = []
    part_index = 0
    for piece in pieces:
        candidate = " ".join([*current, piece])
        candidate_estimate = estimate_duration(
            candidate,
            policy,
            voice_wav=voice_wav,
            ref_text=ref_text,
            profile=profile,
            tag_map=tag_map,
        ).seconds
        if current and candidate_estimate > policy.max_seconds:
            normalized = normalize_unit(" ".join(current), profile, tag_map)
            child = UnitPart(unit.id, normalized, part_index)
            output.extend(
                _split_oversized(
                    child,
                    policy,
                    profile=profile,
                    tag_map=tag_map,
                    voice_wav=voice_wav,
                    ref_text=ref_text,
                )
            )
            part_index += 1
            current = [piece]
        else:
            current.append(piece)
    if current:
        normalized = normalize_unit(" ".join(current), profile, tag_map)
        child = UnitPart(unit.id, normalized, part_index)
        if len(current) == 1:
            child_estimate = estimate_duration(
                child.normalized.tts_text,
                policy,
                voice_wav=voice_wav,
                ref_text=ref_text,
                profile=profile,
                tag_map=tag_map,
            )
            if child_estimate.seconds > policy.max_seconds:
                raise SegmentationError(
                    f"unit {unit.id!r} contains an indivisible span predicted at "
                    f"{child_estimate.seconds:.2f}s, above the {policy.max_seconds:.2f}s maximum"
                )
            output.append(child)
        else:
            output.extend(
                _split_oversized(
                    child,
                    policy,
                    profile=profile,
                    tag_map=tag_map,
                    voice_wav=voice_wav,
                    ref_text=ref_text,
                )
            )
    return output


def plan_chunks(
    units: list[UnitPart],
    policy: SegmentationSpec | None,
    *,
    profile: str,
    tag_map: Mapping[str, str] | None = None,
    voice_wav: Path,
    ref_text: str | None,
) -> list[PhysicalChunk]:
    if not units:
        return []
    accepted_tags = tag_map or {}
    if policy is None:
        return [
            PhysicalChunk(
                tuple(units), DurationEstimate(0.0, "unavailable", "unavailable")
            )
        ]

    def estimate(parts: list[UnitPart]) -> DurationEstimate:
        return estimate_duration(
            " ".join(part.normalized.tts_text for part in parts),
            policy,
            voice_wav=voice_wav,
            ref_text=ref_text,
            profile=profile,
            tag_map=accepted_tags,
        )

    total = estimate(units)
    if total.seconds <= policy.max_seconds:
        return [PhysicalChunk(tuple(units), total)]

    safe_parts: list[UnitPart] = []
    for unit in units:
        safe_parts.extend(
            _split_oversized(
                unit,
                policy,
                profile=profile,
                tag_map=accepted_tags,
                voice_wav=voice_wav,
                ref_text=ref_text,
            )
        )

    chunks: list[PhysicalChunk] = []
    current: list[UnitPart] = []
    for part in safe_parts:
        candidate = [*current, part]
        candidate_estimate = estimate(candidate)
        if current and candidate_estimate.seconds > policy.target_seconds:
            current_estimate = estimate(current)
            # Choose the side closest to target while never exceeding max.
            # This avoids pathological 1s + 18s -> [1s], [18s] partitions.
            include_candidate = (
                candidate_estimate.seconds <= policy.max_seconds
                and (
                    current_estimate.seconds < policy.min_seconds
                    or abs(candidate_estimate.seconds - policy.target_seconds)
                    <= abs(current_estimate.seconds - policy.target_seconds)
                )
            )
            if include_candidate:
                current = candidate
                chunks.append(PhysicalChunk(tuple(current), candidate_estimate))
                current = []
            else:
                chunks.append(PhysicalChunk(tuple(current), current_estimate))
                current = [part]
        else:
            current = candidate
    if current:
        chunks.append(PhysicalChunk(tuple(current), estimate(current)))
    return chunks

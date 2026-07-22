"""Synchronous synthesis loop bridging WebSocket messages to the engine."""

from __future__ import annotations

import json
import math
import queue
import random
import time
from collections import OrderedDict
from typing import Callable, Union

import numpy as np
from pydantic import ValidationError

from .alignment import AlignmentEngine, AlignmentError
from .audio import pcm16_bytes, silence_pcm16, trim_omnivoice_output_silence
from .engine import EngineBusyError, EngineError, SessionPool
from .protocol import SegmentRequest, StartRequest, error_event
from .segmentation import SegmentationError, UnitPart, estimate_duration, plan_chunks
from .textproc import normalize_unit
from .voices import VoiceError, VoiceStore

Emit = Callable[[Union[dict, bytes]], None]
MODEL_CONTROL_IDS = {"numSteps", "firstSegmentSteps", "guidanceScale", "exaggeration"}


def _rtfx(audio_seconds: float, elapsed_ms: float) -> float | None:
    """Audio duration divided by work duration; higher is faster."""
    return audio_seconds / (elapsed_ms / 1000.0) if elapsed_ms > 0 else None


def _print_chunk_debug(payload: dict) -> None:
    prefix = "[speech-server debug] "
    try:
        print(
            prefix
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
    except UnicodeEncodeError:
        # Keep non-UTF-8 Windows consoles usable without hiding the diagnostic.
        try:
            print(
                prefix + json.dumps(payload, separators=(",", ":")),
                flush=True,
            )
        except (OSError, UnicodeError, ValueError):
            pass
    except (OSError, ValueError):
        # Debug output must never turn successful synthesis into a failed request.
        pass


def _validate_options(start: StartRequest, pool: SessionPool) -> str | None:
    try:
        spec = pool.spec(start.model)
    except KeyError:
        return None
    advertised = {control.id: control for control in spec.controls}
    supplied = start.options.model_fields_set & MODEL_CONTROL_IDS
    unsupported = supplied - advertised.keys()
    if unsupported:
        return f"model {start.model!r} does not support {sorted(unsupported)[0]!r}"
    for control_id in supplied:
        value = getattr(start.options, control_id)
        if value is None:
            continue
        control = advertised[control_id]
        if not math.isfinite(float(value)):
            return f"{control_id} must be finite"
        if not control.minimum <= float(value) <= control.maximum:
            return (
                f"{control_id} must be between {control.minimum:g} "
                f"and {control.maximum:g}"
            )
        increments = (float(value) - control.minimum) / control.step
        if not math.isclose(increments, round(increments), abs_tol=1e-7):
            return f"{control_id} must use increments of {control.step:g}"
    return None


def _estimated_unit_timings(
    units: list[UnitPart],
    *,
    policy,
    voice_wav,
    ref_text,
    profile: str,
    tag_map,
    start_seconds: float,
    duration_seconds: float,
) -> OrderedDict[str, dict]:
    weights: list[float] = []
    for unit in units:
        if policy:
            weight = estimate_duration(
                unit.normalized.tts_text,
                policy,
                voice_wav=voice_wav,
                ref_text=ref_text,
                profile=profile,
                tag_map=tag_map,
            ).seconds
        else:
            weight = max(1.0, len(unit.normalized.tts_text))
        weights.append(max(weight, 0.001))
    total = sum(weights) or 1.0
    output: OrderedDict[str, dict] = OrderedDict()
    cursor = start_seconds
    for unit, weight in zip(units, weights):
        end = cursor + duration_seconds * weight / total
        existing = output.get(unit.id)
        if existing:
            existing["endTimeSeconds"] = end
        else:
            output[unit.id] = {
                "id": unit.id,
                "startTimeSeconds": cursor,
                "endTimeSeconds": end,
                "source": "duration-estimate",
                "confidence": "low",
            }
        cursor = end
    return output


def _merge_alignment(
    unit_timings: OrderedDict[str, dict],
    units: tuple[UnitPart, ...],
    words: list[dict],
) -> list[dict]:
    """Map sequential aligner words to explicit unit IDs, never raw source text."""
    cursor = 0
    part_ranges: list[dict] = []
    for index, unit in enumerate(units):
        count = len(unit.normalized.alignment_tokens)
        begin = cursor
        end = min(len(words), begin + count)
        if index + 1 == len(units) and end < len(words):
            end = len(words)
        part_ranges.append({"unit": unit, "begin": begin, "end": end})
        if end > begin:
            timing = unit_timings[unit.id]
            start = words[begin]["start"]
            finish = words[end - 1]["end"]
            if timing.get("source") == "ctc":
                timing["startTimeSeconds"] = min(timing["startTimeSeconds"], start)
                timing["endTimeSeconds"] = max(timing["endTimeSeconds"], finish)
            else:
                timing.update(
                    {
                        "startTimeSeconds": start,
                        "endTimeSeconds": finish,
                        "source": "ctc",
                        "confidence": "measured",
                    }
                )
        cursor = end
    return part_ranges


def _tag_events(part_ranges: list[dict], words: list[dict], unit_timings) -> list[dict]:
    events: list[dict] = []
    for part_range in part_ranges:
        unit = part_range["unit"]
        begin = part_range["begin"]
        end = part_range["end"]
        fallback = unit_timings[unit.id]
        for anchor in unit.normalized.audio_tags:
            position = min(begin + anchor.before_word, end)
            if begin == end:
                timestamp = fallback["startTimeSeconds"]
                source = "duration-estimate"
            elif position <= begin:
                timestamp = max(fallback["startTimeSeconds"], words[begin]["start"] - 0.15)
                source = "estimated-gap"
            elif position >= end:
                timestamp = words[end - 1]["end"]
                source = "estimated-gap"
            else:
                timestamp = (words[position - 1]["end"] + words[position]["start"]) / 2
                source = "estimated-gap"
            events.append(
                {
                    "unitId": unit.id,
                    "tag": anchor.tag,
                    "timeSeconds": timestamp,
                    "source": source,
                }
            )
    return events


def _run_request_impl(
    start: StartRequest,
    segments: queue.Queue,
    emit: Emit,
    *,
    pool: SessionPool,
    voices: VoiceStore,
    aligner: AlignmentEngine | None = None,
    rng: random.Random | None = None,
    debug: bool = False,
) -> None:
    rng = rng or random.Random()
    options_error = _validate_options(start, pool)
    if options_error:
        emit(error_event("bad_request", options_error))
        return

    already_loaded = start.model in pool.loaded_ids()
    load_started = time.perf_counter()
    try:
        session = pool.acquire(start.model)
    except KeyError:
        emit(error_event("model_not_found", f"unknown model {start.model!r}"))
        return
    except EngineBusyError as exc:
        emit(error_event("model_busy", str(exc)))
        return
    except EngineError as exc:
        emit(error_event("model_load_failed", str(exc)))
        return
    model_load_ms = (time.perf_counter() - load_started) * 1000

    upscaler = None
    if start.options.upscale:
        upscale_load_started = time.perf_counter()
        try:
            upscaler = pool.acquire_upscaler()
        except EngineError as exc:
            emit(error_event("upscaler_unavailable", str(exc)))
            return
        model_load_ms += (time.perf_counter() - upscale_load_started) * 1000

    sample_rate = upscaler.spec.sample_rate if upscaler else session.spec.sample_rate
    options = start.options
    emit(
        {
            "type": "started",
            "requestId": start.requestId,
            "sampleRate": sample_rate,
            "mode": "block",
            "upscaled": upscaler is not None,
            "modelLoadMs": model_load_ms,
            "cold": not already_loaded,
        }
    )
    byte_offset = 0
    first_physical = True
    first_segment = True
    aborted = False
    seen_unit_ids: set[str] = set()
    prepared_voices: set[tuple[str, str]] = set()

    while True:
        message = segments.get()
        message_type = message.get("type")
        if message_type == "abort":
            aborted = True
            break
        if message_type == "end":
            break
        if message_type != "segment":
            emit(error_event("bad_request", f"unexpected message {message_type!r}"))
            return
        try:
            segment = SegmentRequest.model_validate(message)
        except ValidationError as exc:
            emit(error_event("bad_request", str(exc)))
            return

        raw_units = (
            [(unit.id, unit.text) for unit in segment.units]
            if segment.units is not None
            else [(f"segment-{segment.idx}", segment.text or "")]
        )
        duplicate = next((unit_id for unit_id, _ in raw_units if unit_id in seen_unit_ids), None)
        if duplicate is not None:
            emit(error_event("bad_request", f"duplicate request unit ID {duplicate!r}"))
            return
        seen_unit_ids.update(unit_id for unit_id, _ in raw_units)

        normalized_units = [
            UnitPart(
                unit_id,
                normalize_unit(
                    text,
                    session.spec.text_profile,
                    session.spec.paralinguistic_tag_map,
                ),
            )
            for unit_id, text in raw_units
        ]
        units = [unit for unit in normalized_units if unit.normalized.tts_text]
        if not units:
            stream_seconds = byte_offset / (sample_rate * 2)
            emit(
                {
                    "type": "segment_start",
                    "idx": segment.idx,
                    "byteOffset": byte_offset,
                    "unitIds": [unit.id for unit in normalized_units],
                }
            )
            if segment.units is not None:
                emit(
                    {
                        "type": "timing",
                        "idx": segment.idx,
                        "status": "skipped",
                        "wordAlignment": {
                            "words": [],
                            "wordStartTimeSeconds": [],
                            "wordEndTimeSeconds": [],
                        },
                        "units": [
                            {
                                "id": unit.id,
                                "startTimeSeconds": stream_seconds,
                                "endTimeSeconds": stream_seconds,
                                "source": "duration-estimate",
                                "confidence": "low",
                            }
                            for unit in normalized_units
                        ],
                        "events": [],
                    }
                )
            emit(
                {
                    "type": "segment_done",
                    "idx": segment.idx,
                    "bytes": 0,
                    "audioSeconds": 0.0,
                    "synthesisMs": 0.0,
                    "upscaleMs": 0.0,
                    "alignmentMs": 0.0,
                    "processingMs": 0.0,
                    "throughputX": None,
                    "physicalChunks": 0,
                    "estimatedSeconds": 0.0,
                    "estimateSource": "unavailable",
                    "alignmentStatus": "skipped",
                    "limitExceeded": False,
                }
            )
            continue

        voice_id = segment.voiceId or start.voiceId
        try:
            voice_wav, ref_text = voices.resolve(voice_id)
        except (KeyError, VoiceError):
            emit(error_event("voice_not_found", voice_id))
            return
        transcript_policy = session.spec.voice_reference.transcript
        if transcript_policy == "required" and not ref_text:
            emit(
                error_event(
                    "voice_transcript_required",
                    f"model {session.spec.id!r} requires a reference transcript",
                )
            )
            return
        if transcript_policy == "unused":
            ref_text = None

        try:
            physical_chunks = plan_chunks(
                units,
                session.spec.segmentation,
                profile=session.spec.text_profile,
                tag_map=session.spec.paralinguistic_tag_map,
                voice_wav=voice_wav,
                ref_text=ref_text,
            )
        except SegmentationError as exc:
            emit(error_event("bad_request", str(exc)))
            return
        if not first_segment and options.silence.maxMs > 0:
            silence_ms = rng.uniform(options.silence.minMs, options.silence.maxMs)
            silence = silence_pcm16(silence_ms, sample_rate)
            emit(silence)
            byte_offset += len(silence)

        processing_started = time.perf_counter()
        synthesis_ms = 0.0
        upscale_ms = 0.0
        alignment_ms = 0.0
        segment_audio_parts: list[np.ndarray] = []
        physical_records: list[
            tuple[tuple[UnitPart, ...], float, float, list[dict]]
        ] = []
        all_words: list[dict] = []
        all_events: list[dict] = []
        alignment_failed = False
        debug_generations: list[dict] = []
        should_align = start.options.timing == "word" or (
            start.options.timing == "auto" and len(units) > 1
        )
        can_align = bool(
            should_align
            and aligner
            and aligner.available
            and aligner.supports_language(start.language)
        )
        segment_start_seconds = byte_offset / (sample_rate * 2)
        prepared_voice_key = (session.spec.id, voice_id)

        for physical_index, physical in enumerate(physical_chunks):
            if physical_index > 0 and options.silence.maxMs > 0:
                silence_ms = rng.uniform(options.silence.minMs, options.silence.maxMs)
                segment_audio_parts.append(
                    np.zeros(round(silence_ms / 1000 * sample_rate), dtype=np.float32)
                )
            text = " ".join(unit.normalized.tts_text for unit in physical.units)
            steps = (
                options.firstSegmentSteps
                if first_physical and options.firstSegmentSteps is not None
                else options.numSteps
            )
            chunk_started = time.perf_counter()
            stage_started = chunk_started
            try:
                audio = session.synthesize(
                    text,
                    voice_wav=voice_wav,
                    ref_text=ref_text,
                    steps=steps,
                    seed=options.seed,
                    guidance=options.guidanceScale,
                    exaggeration=options.exaggeration,
                )
            except EngineError as exc:
                emit(error_event("synthesis_failed", str(exc)))
                return
            chunk_synthesis_ms = (time.perf_counter() - stage_started) * 1000
            synthesis_ms += chunk_synthesis_ms
            if (
                prepared_voice_key not in prepared_voices
                and session.spec.voice_reference.preparation_mode == "persistent"
            ):
                try:
                    voices.mark_prepared(
                        voice_id,
                        model_id=session.spec.id,
                        revision=session.spec.voice_preparation_revision,
                        inputs=session.spec.voice_reference.preparation_inputs,
                    )
                    prepared_voices.add(prepared_voice_key)
                except (KeyError, OSError, ValueError, VoiceError):
                    # Preparation metadata is advisory; successful audio must not
                    # be discarded because a cache marker could not be written.
                    pass
            first_physical = False

            audio = np.asarray(audio, dtype=np.float32)
            if session.spec.backend == "omnivoice":
                audio = trim_omnivoice_output_silence(
                    audio, session.spec.sample_rate
                )
            chunk_upscale_ms = 0.0
            if upscaler is not None:
                stage_started = time.perf_counter()
                try:
                    audio = upscaler.restore(audio, session.spec.sample_rate)
                except EngineError as exc:
                    emit(error_event("upscaling_failed", str(exc)))
                    return
                chunk_upscale_ms = (time.perf_counter() - stage_started) * 1000
                upscale_ms += chunk_upscale_ms
            audio = np.asarray(audio, dtype=np.float32)
            chunk_offset = segment_start_seconds + sum(
                len(part) / sample_rate for part in segment_audio_parts
            )
            segment_audio_parts.append(audio)
            chunk_words: list[dict] = []
            chunk_alignment_ms = 0.0
            if can_align and not alignment_failed:
                transcript = " ".join(
                    unit.normalized.alignment_text
                    for unit in physical.units
                    if unit.normalized.alignment_text
                )
                if transcript:
                    stage_started = time.perf_counter()
                    try:
                        chunk_words = aligner.align(
                            transcript, audio, sample_rate, t_offset=chunk_offset
                        )
                    except AlignmentError:
                        alignment_failed = True
                        chunk_words = []
                    chunk_alignment_ms = (time.perf_counter() - stage_started) * 1000
                    alignment_ms += chunk_alignment_ms
            if debug:
                chunk_audio_seconds = len(audio) / sample_rate
                chunk_processing_ms = (time.perf_counter() - chunk_started) * 1000
                debug_generations.append(
                    {
                        "index": physical_index,
                        "unitIds": [unit.id for unit in physical.units],
                        "text": text,
                        "steps": steps,
                        "samples": len(audio),
                        "bytes": len(audio) * 2,
                        "audioSeconds": chunk_audio_seconds,
                        "sampleRate": sample_rate,
                        "synthesisMs": chunk_synthesis_ms,
                        "synthesisRtfX": _rtfx(
                            chunk_audio_seconds, chunk_synthesis_ms
                        ),
                        "upscaleMs": chunk_upscale_ms,
                        "alignmentMs": chunk_alignment_ms,
                        "processingMs": chunk_processing_ms,
                        "processingRtfX": _rtfx(
                            chunk_audio_seconds, chunk_processing_ms
                        ),
                    }
                )
            physical_records.append(
                (physical.units, chunk_offset, len(audio) / sample_rate, chunk_words)
            )

        segment_audio = (
            np.concatenate(segment_audio_parts)
            if segment_audio_parts
            else np.zeros(0, dtype=np.float32)
        )
        audio_seconds = len(segment_audio) / sample_rate
        unit_timings: OrderedDict[str, dict] = OrderedDict()
        for chunk_units, chunk_start, chunk_duration, _ in physical_records:
            estimated = _estimated_unit_timings(
                list(chunk_units),
                policy=session.spec.segmentation,
                voice_wav=voice_wav,
                ref_text=ref_text,
                profile=session.spec.text_profile,
                tag_map=session.spec.paralinguistic_tag_map,
                start_seconds=chunk_start,
                duration_seconds=chunk_duration,
            )
            for unit_id, timing in estimated.items():
                if unit_id in unit_timings:
                    unit_timings[unit_id]["endTimeSeconds"] = timing["endTimeSeconds"]
                else:
                    unit_timings[unit_id] = timing
        # A partial alignment cannot safely preserve unit-to-word mapping. Return
        # duration-weighted boundaries for the whole segment and label it failed.
        if alignment_failed:
            all_words = []
        if not alignment_failed:
            # Keep imperfect CTC output local to its physical chunk. Otherwise
            # one missing/extra word could shift every later unit across a
            # server-enforced split even though each CTC call was independent.
            for chunk_units, _, _, chunk_words in physical_records:
                if not chunk_words:
                    for unit in chunk_units:
                        for anchor in unit.normalized.audio_tags:
                            all_events.append(
                                {
                                    "unitId": unit.id,
                                    "tag": anchor.tag,
                                    "timeSeconds": unit_timings[unit.id][
                                        "startTimeSeconds"
                                    ],
                                    "source": "duration-estimate",
                                }
                            )
                    continue
                ranges = _merge_alignment(unit_timings, chunk_units, chunk_words)
                all_events.extend(_tag_events(ranges, chunk_words, unit_timings))
                all_words.extend(chunk_words)
        if not all_words:
            all_events = []
            for unit in units:
                for anchor in unit.normalized.audio_tags:
                    all_events.append(
                        {
                            "unitId": unit.id,
                            "tag": anchor.tag,
                            "timeSeconds": unit_timings[unit.id]["startTimeSeconds"],
                            "source": "duration-estimate",
                        }
                    )

        if can_align and alignment_failed:
            alignment_status = "failed"
        elif can_align and all_words:
            alignment_status = "aligned"
        elif should_align:
            alignment_status = "unavailable"
        else:
            alignment_status = "skipped"

        pcm = pcm16_bytes(segment_audio)
        processing_ms = (time.perf_counter() - processing_started) * 1000
        emit(
            {
                "type": "segment_start",
                "idx": segment.idx,
                "byteOffset": byte_offset,
                "unitIds": [unit.id for unit in units],
            }
        )
        if debug:
            _print_chunk_debug(
                {
                    "event": "synthesis_chunk_sent",
                    "requestId": start.requestId,
                    "segmentIndex": segment.idx,
                    "model": session.spec.id,
                    "backend": session.spec.backend,
                    "voiceId": voice_id,
                    "generation": debug_generations,
                    "output": {
                        "bytes": len(pcm),
                        "samples": len(segment_audio),
                        "audioSeconds": audio_seconds,
                        "sampleRate": sample_rate,
                        "synthesisMs": synthesis_ms,
                        "synthesisRtfX": _rtfx(audio_seconds, synthesis_ms),
                        "upscaleMs": upscale_ms,
                        "alignmentMs": alignment_ms,
                        "processingMs": processing_ms,
                        "processingRtfX": _rtfx(audio_seconds, processing_ms),
                        "physicalChunks": len(physical_chunks),
                        "alignmentStatus": alignment_status,
                    },
                }
            )
        emit(pcm)
        byte_offset += len(pcm)

        if should_align or segment.units is not None:
            emit(
                {
                    "type": "timing",
                    "idx": segment.idx,
                    "status": alignment_status,
                    "wordAlignment": {
                        "words": [word["text"] for word in all_words],
                        "wordStartTimeSeconds": [word["start"] for word in all_words],
                        "wordEndTimeSeconds": [word["end"] for word in all_words],
                    },
                    "units": list(unit_timings.values()),
                    "events": all_events,
                }
            )

        estimate_seconds = sum(chunk.estimate.seconds for chunk in physical_chunks)
        estimate_sources = {chunk.estimate.source for chunk in physical_chunks}
        emit(
            {
                "type": "segment_done",
                "idx": segment.idx,
                "bytes": len(pcm),
                "audioSeconds": audio_seconds,
                "synthesisMs": synthesis_ms,
                "upscaleMs": upscale_ms,
                "alignmentMs": alignment_ms,
                "processingMs": processing_ms,
                "throughputX": (
                    audio_seconds / (synthesis_ms / 1000) if synthesis_ms > 0 else None
                ),
                "physicalChunks": len(physical_chunks),
                "estimatedSeconds": estimate_seconds,
                "estimateSource": (
                    next(iter(estimate_sources))
                    if len(estimate_sources) == 1
                    else "mixed"
                ),
                "alignmentStatus": alignment_status,
                "limitExceeded": bool(
                    session.spec.segmentation
                    and any(
                        duration > session.spec.segmentation.max_seconds
                        for _, _, duration, _ in physical_records
                    )
                ),
            }
        )
        first_segment = False

    emit(
        {
            "type": "done",
            "requestId": start.requestId,
            "totalBytes": byte_offset,
            "aborted": aborted,
        }
    )


def run_request(
    start: StartRequest,
    segments: queue.Queue,
    emit: Emit,
    *,
    pool: SessionPool,
    voices: VoiceStore,
    aligner: AlignmentEngine | None = None,
    rng: random.Random | None = None,
    debug: bool = False,
) -> None:
    """Pin the selected model for the lifetime of a multi-segment request."""
    pool.pin(start.model)
    try:
        _run_request_impl(
            start,
            segments,
            emit,
            pool=pool,
            voices=voices,
            aligner=aligner,
            rng=rng,
            debug=debug,
        )
    finally:
        pool.unpin(start.model)

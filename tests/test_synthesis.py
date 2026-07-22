import json
import queue
import random
import wave
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from speech_server.config import (
    ControlSpec,
    ModelSpec,
    ParalinguisticTagSpec,
    SegmentationSpec,
    ServerConfig,
    VoiceReferenceSpec,
)
from speech_server.engine import SessionPool
from speech_server.protocol import SegmentRequest, StartRequest
from speech_server.synthesis import run_request
from speech_server.voices import VoiceStore
from tests.fakes import FakeSession


def _setup(tmp_path, *, with_upscaler=False, backend="omnivoice"):
    controls = (
        [
            ControlSpec("numSteps", "integer", 1, 30, 1, 6),
            ControlSpec("guidanceScale", "number", 0, 2, 0.05, 0.5),
            ControlSpec("exaggeration", "number", 0, 2, 0.05, 0.5),
        ]
        if backend == "chatterbox"
        else [
            ControlSpec("numSteps", "integer", 8, 64, 4, 32),
            ControlSpec("firstSegmentSteps", "integer", 8, 64, 4, 32),
            ControlSpec("guidanceScale", "number", 0, 10, 0.1, 2),
        ]
    )
    spec = ModelSpec(
        id="omnivoice",
        backend=backend,
        model_path=Path("x.gguf"),
        sample_rate=24000,
        controls=controls,
        text_profile="omnivoice" if backend == "omnivoice" else "plain",
        paralinguistic_tags=(
            [
                ParalinguisticTagSpec(
                    "[laughter]",
                    "Inserts natural laughter.",
                    ("[laugh]", "[laughs]", "[laughing]"),
                )
            ]
            if backend == "omnivoice"
            else []
        ),
        voice_reference=(
            VoiceReferenceSpec(
                "required", "persistent", ("audio", "transcript")
            )
            if backend == "omnivoice"
            else VoiceReferenceSpec("unused", "lazy", ("audio",))
        ),
    )
    upscaler = (
        ModelSpec(
            id="voxcpm2-vae",
            backend="voxcpm2-vae",
            model_path=Path("voxcpm2-vae.gguf"),
            sample_rate=48000,
            task="audio-to-audio",
            cloning=False,
        )
        if with_upscaler
        else None
    )
    cfg = ServerConfig(
        models={"omnivoice": spec},
        upscaler=upscaler,
        resident_limit=1,
        voice_dir=tmp_path,
    )
    pool = SessionPool(
        cfg, session_factory=lambda cfg, model: FakeSession(model.backend)
    )
    store = VoiceStore(tmp_path)
    with wave.open(str(tmp_path / "default__Seb.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00\x00\x00")
    (tmp_path / "default__Seb.txt").write_text("ref", encoding="utf-8")
    return pool, store


def _drive(pool, store, messages, options=None, aligner=None, *, debug=False):
    events = []
    segment_queue = queue.Queue()
    for message in messages:
        segment_queue.put(message)
    start = StartRequest(
        requestId="r1",
        model="omnivoice",
        voiceId="default__Seb",
        language="English",
        options=options or {},
    )
    run_request(
        start,
        segment_queue,
        events.append,
        pool=pool,
        voices=store,
        aligner=aligner,
        rng=random.Random(0),
        debug=debug,
    )
    return events


def test_debug_prints_exact_generation_text_and_chunk_rtfx(tmp_path, capsys):
    pool, store = _setup(tmp_path)
    _drive(
        pool,
        store,
        [
            {
                "type": "segment",
                "idx": 7,
                "units": [
                    {"id": "s1", "text": "[laugh] I knew it."},
                    {"id": "s2", "text": "[unsupported] Really!"},
                ],
            },
            {"type": "end"},
        ],
        options={"silence": {"minMs": 0, "maxMs": 0}},
        debug=True,
    )
    line = capsys.readouterr().out.strip()
    prefix = "[speech-server debug] "
    assert line.startswith(prefix)
    diagnostic = json.loads(line[len(prefix) :])
    assert diagnostic["event"] == "synthesis_chunk_sent"
    assert diagnostic["requestId"] == "r1"
    assert diagnostic["segmentIndex"] == 7
    assert diagnostic["model"] == "omnivoice"
    assert diagnostic["voiceId"] == "default__Seb"
    generation = diagnostic["generation"]
    assert [item["text"] for item in generation] == [
        "[laughter] I knew it. Really!"
    ]
    assert generation[0]["unitIds"] == ["s1", "s2"]
    assert generation[0]["samples"] == 2400
    assert generation[0]["bytes"] == 4800
    assert generation[0]["audioSeconds"] > 0
    assert generation[0]["synthesisMs"] > 0
    assert generation[0]["synthesisRtfX"] > 0
    assert generation[0]["processingRtfX"] > 0
    assert diagnostic["output"]["bytes"] > 0
    assert diagnostic["output"]["samples"] == 2400
    assert diagnostic["output"]["physicalChunks"] == 1
    assert diagnostic["output"]["synthesisRtfX"] > 0


def test_debug_output_is_disabled_by_default(tmp_path, capsys):
    pool, store = _setup(tmp_path)
    _drive(
        pool,
        store,
        [{"type": "segment", "idx": 0, "text": "Quiet."}, {"type": "end"}],
    )
    assert capsys.readouterr().out == ""


def test_two_segments_event_order_and_offsets(tmp_path):
    pool, store = _setup(tmp_path)
    events = _drive(
        pool,
        store,
        [
            {"type": "segment", "idx": 0, "text": "One."},
            {"type": "segment", "idx": 1, "text": "Two."},
            {"type": "end"},
        ],
        options={"silence": {"minMs": 100, "maxMs": 100}},
    )
    types = [event["type"] if isinstance(event, dict) else "pcm" for event in events]
    assert types == [
        "started",
        "segment_start",
        "pcm",
        "segment_done",
        "pcm",
        "segment_start",
        "pcm",
        "segment_done",
        "done",
    ]
    segment_bytes = 2400 * 2
    silence_bytes = int(0.1 * 24000) * 2
    starts = [
        event
        for event in events
        if isinstance(event, dict) and event["type"] == "segment_start"
    ]
    assert starts[0]["byteOffset"] == 0
    assert starts[1]["byteOffset"] == segment_bytes + silence_bytes
    done = next(
        event
        for event in events
        if isinstance(event, dict) and event["type"] == "done"
    )
    assert done["totalBytes"] == 2 * segment_bytes + silence_bytes


def test_required_reference_transcript_fails_before_synthesis(tmp_path):
    pool, store = _setup(tmp_path)
    (tmp_path / "default__Seb.txt").unlink()
    events = _drive(
        pool,
        store,
        [{"type": "segment", "idx": 0, "text": "One."}, {"type": "end"}],
    )
    error = next(
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "error"
    )
    assert error["code"] == "voice_transcript_required"
    assert not any(call[0] == "synth" for call in pool.acquire("omnivoice").raw.calls)


def test_persistent_marker_is_written_once_per_voice_per_request(tmp_path):
    pool, store = _setup(tmp_path)
    calls = []
    original = store.mark_prepared

    def mark(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    store.mark_prepared = mark
    _drive(
        pool,
        store,
        [
            {"type": "segment", "idx": 0, "text": "One."},
            {"type": "segment", "idx": 1, "text": "Two."},
            {"type": "end"},
        ],
    )
    assert len(calls) == 1


def test_first_segment_steps(tmp_path):
    pool, store = _setup(tmp_path)
    _drive(
        pool,
        store,
        [
            {"type": "segment", "idx": 0, "text": "One."},
            {"type": "segment", "idx": 1, "text": "Two."},
            {"type": "end"},
        ],
        options={"numSteps": 32, "firstSegmentSteps": 24},
    )
    calls = pool.acquire("omnivoice").raw.calls
    assert [call[1] for call in calls if call[0] == "steps"] == [24, 32]


def test_first_segment_only_override_resets_for_later_segments(tmp_path):
    pool, store = _setup(tmp_path)
    _drive(
        pool,
        store,
        [
            {"type": "segment", "idx": 0, "text": "One."},
            {"type": "segment", "idx": 1, "text": "Two."},
            {"type": "end"},
        ],
        options={"firstSegmentSteps": 24},
    )
    calls = pool.acquire("omnivoice").raw.calls
    assert [call[1] for call in calls if call[0] == "steps"] == [24, 32]


def test_omitted_controls_preserve_backend_defaults(tmp_path):
    pool, store = _setup(tmp_path)
    _drive(
        pool,
        store,
        [{"type": "segment", "idx": 0, "text": "One."}, {"type": "end"}],
    )
    calls = pool.acquire("omnivoice").raw.calls
    assert not [call for call in calls if call[0] in {"steps", "tts_cfg", "cfg"}]


def test_omitted_controls_reset_values_from_previous_request(tmp_path):
    pool, store = _setup(tmp_path)
    segment = [{"type": "segment", "idx": 0, "text": "One."}, {"type": "end"}]
    _drive(pool, store, segment, options={"numSteps": 24, "guidanceScale": 3})
    _drive(pool, store, segment)
    calls = pool.acquire("omnivoice").raw.calls
    assert [call for call in calls if call[0] == "steps"][-1] == ("steps", 32)
    assert [call for call in calls if call[0] == "tts_cfg"][-1] == (
        "tts_cfg",
        2.0,
    )


def test_omitted_seed_replaces_previous_override_with_random_seed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("speech_server.engine.secrets.randbelow", lambda _limit: 98)
    pool, store = _setup(tmp_path)
    segment = [{"type": "segment", "idx": 0, "text": "One."}, {"type": "end"}]
    _drive(pool, store, segment, options={"seed": 7})
    _drive(pool, store, segment)
    calls = pool.acquire("omnivoice").raw.calls
    assert [call for call in calls if call[0] == "seed"] == [
        ("seed", 7),
        ("seed", 99),
    ]


def test_nonfinite_options_are_rejected(tmp_path):
    pool, store = _setup(tmp_path)
    events = _drive(
        pool,
        store,
        [{"type": "end"}],
        options={"guidanceScale": float("nan")},
    )
    assert events[0]["code"] == "bad_request"
    assert "finite" in events[0]["message"]


@pytest.mark.parametrize(
    "options",
    [
        {"numSteps": "24"},
        {"guidanceScale": True},
        {"seed": False},
        {"seed": -1},
        {"seed": 0x10000000000000000},
        {"upscale": "false"},
        {"silence": {"minMs": "0", "maxMs": 100}},
        {"silence": {"minMs": 0, "maxMs": 100, "extra": 1}},
    ],
)
def test_option_types_are_not_coerced(options):
    with pytest.raises(ValidationError):
        StartRequest(
            requestId="r",
            model="omnivoice",
            voiceId="default__Seb",
            options=options,
        )


def test_unit_segment_validation_and_legacy_compatibility():
    assert SegmentRequest(type="segment", idx=0, text="legacy").text == "legacy"
    segment = SegmentRequest(
        type="segment",
        idx=1,
        units=[{"id": "s1", "text": "One."}, {"id": "s2", "text": "Two."}],
    )
    assert [unit.id for unit in segment.units] == ["s1", "s2"]
    for invalid in (
        {"type": "segment", "idx": 0},
        {"type": "segment", "idx": 0, "text": "x", "units": [{"id": "s", "text": "x"}]},
        {"type": "segment", "idx": 0, "units": [{"id": "s", "text": "x"}, {"id": "s", "text": "y"}]},
        {"type": "segment", "idx": 0, "units": [{"id": f"s{i}", "text": "x"} for i in range(33)]},
    ):
        with pytest.raises(ValidationError):
            SegmentRequest.model_validate(invalid)


def test_chatterbox_shared_steps_cfg_and_exaggeration(tmp_path):
    pool, store = _setup(tmp_path, backend="chatterbox")
    events = _drive(
        pool,
        store,
        [
            {"type": "segment", "idx": 0, "text": "One."},
            {"type": "segment", "idx": 1, "text": "Two."},
            {"type": "end"},
        ],
        options={"numSteps": 6, "guidanceScale": 0.5, "exaggeration": 0.75},
    )
    assert events[0]["type"] == "started"
    calls = pool.acquire("omnivoice").raw.calls
    assert [call[1] for call in calls if call[0] == "steps"] == [6, 6]
    assert ("cfg", 0.5) in calls
    assert ("exaggeration", 0.75) in calls


def test_unsupported_and_out_of_range_controls_are_rejected(tmp_path):
    pool, store = _setup(tmp_path, backend="chatterbox")
    unsupported = _drive(
        pool, store, [{"type": "end"}], options={"firstSegmentSteps": 24}
    )
    assert unsupported[0]["code"] == "bad_request"
    out_of_range = _drive(
        pool, store, [{"type": "end"}], options={"numSteps": 31}
    )
    assert out_of_range[0]["code"] == "bad_request"


def test_voice_not_found(tmp_path):
    pool, store = _setup(tmp_path)
    events = _drive(
        pool,
        store,
        [
            {"type": "segment", "idx": 0, "text": "Hi.", "voiceId": "missing"},
            {"type": "end"},
        ],
    )
    errors = [
        event
        for event in events
        if isinstance(event, dict) and event["type"] == "error"
    ]
    assert errors and errors[0]["code"] == "voice_not_found"


def test_abort_stops_before_next_segment(tmp_path):
    pool, store = _setup(tmp_path)
    events = _drive(
        pool,
        store,
        [{"type": "segment", "idx": 0, "text": "One."}, {"type": "abort"}],
    )
    types = [event["type"] if isinstance(event, dict) else "pcm" for event in events]
    assert types[-1] == "done"
    done = next(event for event in events if isinstance(event, dict) and event["type"] == "done")
    assert done["aborted"] is True


def test_unknown_model(tmp_path):
    pool, store = _setup(tmp_path)
    events = []
    segment_queue = queue.Queue()
    segment_queue.put({"type": "end"})
    start = StartRequest(
        requestId="r", model="nope", voiceId="v", language="English", options={}
    )
    run_request(start, segment_queue, events.append, pool=pool, voices=store)
    assert events[0]["type"] == "error" and events[0]["code"] == "model_not_found"


def test_upscale_restores_each_segment_and_announces_48khz(tmp_path):
    pool, store = _setup(tmp_path, with_upscaler=True)
    events = _drive(
        pool,
        store,
        [
            {"type": "segment", "idx": 0, "text": "Restore me."},
            {"type": "end"},
        ],
        options={"upscale": True, "silence": {"minMs": 0, "maxMs": 0}},
    )
    started = events[0]
    assert started["sampleRate"] == 48000
    assert started["upscaled"] is True
    pcm = next(event for event in events if isinstance(event, bytes))
    assert len(pcm) == 4800 * 2
    calls = pool.acquire_upscaler().raw.calls
    assert ("pcm_sample_rate", 24000) in calls
    assert ("s2s", 2400) in calls


def test_omnivoice_trims_native_silence_before_upscaling(tmp_path):
    pool, store = _setup(tmp_path, with_upscaler=True)
    session = pool.acquire("omnivoice")
    tone = np.full(2400, 0.1, dtype=np.float32)
    native = np.concatenate(
        [
            np.zeros(7200, dtype=np.float32),
            tone,
            np.zeros(36000, dtype=np.float32),
            tone,
            np.zeros(9600, dtype=np.float32),
        ]
    )
    session.raw.synthesize_raw = lambda text: native

    events = _drive(
        pool,
        store,
        [{"type": "segment", "idx": 0, "text": "Trim me."}, {"type": "end"}],
        options={"upscale": True, "silence": {"minMs": 0, "maxMs": 0}},
    )

    calls = pool.acquire_upscaler().raw.calls
    assert ("s2s", 33600) in calls
    pcm = next(event for event in events if isinstance(event, bytes))
    assert len(pcm) == 67200 * 2


def test_upscale_rejected_when_not_configured(tmp_path):
    pool, store = _setup(tmp_path)
    events = _drive(
        pool,
        store,
        [{"type": "end"}],
        options={"upscale": True},
    )
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "upscaler_unavailable"


class _FakeAligner:
    available = True

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def supports_language(self, language):
        return language == "English"

    def align(self, transcript, pcm, sample_rate, *, t_offset):
        from speech_server.alignment import AlignmentError

        self.calls.append((transcript, len(pcm), sample_rate, t_offset))
        if self.fail:
            raise AlignmentError("forced failure")
        tokens = transcript.split()
        return [
            {
                "text": token,
                "start": t_offset + index * 0.01,
                "end": t_offset + (index + 1) * 0.01,
            }
            for index, token in enumerate(tokens)
        ]


def test_multi_unit_auto_alignment_uses_tag_free_text_and_unit_ids(tmp_path):
    pool, store = _setup(tmp_path)
    aligner = _FakeAligner()
    events = _drive(
        pool,
        store,
        [
            {
                "type": "segment",
                "idx": 1,
                "units": [
                    {"id": "s1", "text": "[laugh] I knew it."},
                    {"id": "s2", "text": "You cannot hide."},
                ],
            },
            {"type": "end"},
        ],
        options={"timing": "auto"},
        aligner=aligner,
    )
    assert aligner.calls[0][0] == "I knew it. You cannot hide."
    timing = next(event for event in events if isinstance(event, dict) and event["type"] == "timing")
    assert timing["status"] == "aligned"
    assert [unit["id"] for unit in timing["units"]] == ["s1", "s2"]
    assert timing["units"][0]["source"] == "ctc"
    assert timing["events"] == [
        {
            "unitId": "s1",
            "tag": "laughter",
            "timeSeconds": 0.0,
            "source": "estimated-gap",
        }
    ]


def test_auto_skips_single_unit_but_word_mode_aligns_it(tmp_path):
    pool, store = _setup(tmp_path)
    aligner = _FakeAligner()
    segment = [
        {"type": "segment", "idx": 0, "units": [{"id": "s0", "text": "One."}]},
        {"type": "end"},
    ]
    auto = _drive(pool, store, segment, options={"timing": "auto"}, aligner=aligner)
    assert not aligner.calls
    assert next(event for event in auto if isinstance(event, dict) and event["type"] == "timing")["status"] == "skipped"
    word = _drive(pool, store, segment, options={"timing": "word"}, aligner=aligner)
    assert aligner.calls
    assert next(event for event in word if isinstance(event, dict) and event["type"] == "timing")["status"] == "aligned"


def test_alignment_failure_keeps_pcm_and_returns_estimated_boundaries(tmp_path):
    pool, store = _setup(tmp_path)
    events = _drive(
        pool,
        store,
        [
            {
                "type": "segment",
                "idx": 0,
                "units": [
                    {"id": "s0", "text": "One."},
                    {"id": "s1", "text": "Two."},
                ],
            },
            {"type": "end"},
        ],
        options={"timing": "auto"},
        aligner=_FakeAligner(fail=True),
    )
    assert any(isinstance(event, bytes) and event for event in events)
    timing = next(event for event in events if isinstance(event, dict) and event["type"] == "timing")
    assert timing["status"] == "failed"
    assert timing["wordAlignment"]["words"] == []
    assert all(unit["source"] == "duration-estimate" for unit in timing["units"])
    done = next(event for event in events if isinstance(event, dict) and event["type"] == "segment_done")
    assert done["alignmentStatus"] == "failed"


def test_upscale_precedes_alignment_and_reports_stage_stats(tmp_path):
    pool, store = _setup(tmp_path, with_upscaler=True)
    aligner = _FakeAligner()
    events = _drive(
        pool,
        store,
        [
            {
                "type": "segment",
                "idx": 0,
                "units": [{"id": "s0", "text": "One."}, {"id": "s1", "text": "Two."}],
            },
            {"type": "end"},
        ],
        options={"upscale": True, "timing": "auto"},
        aligner=aligner,
    )
    assert aligner.calls[0][2] == 48000
    done = next(event for event in events if isinstance(event, dict) and event["type"] == "segment_done")
    for key in ("audioSeconds", "synthesisMs", "upscaleMs", "alignmentMs", "processingMs", "throughputX", "physicalChunks"):
        assert key in done


def test_tag_only_unit_that_normalizes_empty_still_completes(tmp_path):
    pool, store = _setup(tmp_path, backend="chatterbox")
    events = _drive(
        pool,
        store,
        [
            {
                "type": "segment",
                "idx": 0,
                "units": [{"id": "s0", "text": "[unsupported]"}],
            },
            {"type": "end"},
        ],
        options={"timing": "auto"},
    )
    assert not any(isinstance(event, bytes) for event in events)
    timing = next(event for event in events if isinstance(event, dict) and event["type"] == "timing")
    assert timing["units"][0]["id"] == "s0"
    assert timing["units"][0]["startTimeSeconds"] == timing["units"][0]["endTimeSeconds"]
    assert any(isinstance(event, dict) and event["type"] == "segment_done" for event in events)


def test_alignment_word_mismatch_does_not_shift_across_physical_chunks(tmp_path):
    pool, store = _setup(tmp_path)
    (tmp_path / "default__Seb.txt").unlink()
    pool.spec("omnivoice").voice_reference = VoiceReferenceSpec(
        "optional", "lazy", ("audio",)
    )
    pool.spec("omnivoice").segmentation = SegmentationSpec(
        estimator="reference-rate",
        min_seconds=0.1,
        target_seconds=0.7,
        max_seconds=1.0,
        fallback_characters_per_second=10,
        fallback_words_per_second=10,
        safety_factor=1.0,
    )

    class MismatchedAligner(_FakeAligner):
        def align(self, transcript, pcm, sample_rate, *, t_offset):
            words = super().align(transcript, pcm, sample_rate, t_offset=t_offset)
            return words[:1] if not self.calls[:-1] else words

    events = _drive(
        pool,
        store,
        [
            {
                "type": "segment",
                "idx": 0,
                "units": [
                    {"id": "s0", "text": "one two"},
                    {"id": "s1", "text": "three four"},
                ],
            },
            {"type": "end"},
        ],
        options={"timing": "auto", "silence": {"minMs": 0, "maxMs": 0}},
        aligner=MismatchedAligner(),
    )
    timing = next(
        event
        for event in events
        if isinstance(event, dict) and event["type"] == "timing"
    )
    by_id = {unit["id"]: unit for unit in timing["units"]}
    assert by_id["s0"]["endTimeSeconds"] <= 0.1
    assert by_id["s1"]["startTimeSeconds"] >= 0.1

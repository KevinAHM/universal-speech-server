import wave
from pathlib import Path

from speech_server.config import SegmentationSpec
from speech_server.segmentation import (
    SegmentationError,
    UnitPart,
    estimate_duration,
    plan_chunks,
)
from speech_server.textproc import normalize_unit


POLICY = SegmentationSpec(
    estimator="reference-rate",
    min_seconds=8,
    target_seconds=20,
    max_seconds=28,
    fallback_characters_per_second=10,
    fallback_words_per_second=2,
    safety_factor=1.15,
)
LAUGHTER_TAGS = {"[laughter]": "[laughter]"}


def _wav(path: Path, seconds: float = 10.0):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(b"\0\0" * round(seconds * 1000))


def test_reference_rate_uses_slower_character_or_word_prediction(tmp_path):
    reference = tmp_path / "reference.wav"
    _wav(reference)
    # Reference rates are 2 chars/s and 0.2 words/s. The target predicts five
    # seconds by characters but ten by words, then applies the safety factor.
    estimate = estimate_duration(
        "abcdefghij two", POLICY, voice_wav=reference, ref_text="abcdefghij klmnopqrst"
    )
    assert estimate.source == "reference-rate"
    assert estimate.confidence == "medium"
    assert estimate.seconds == 11.5


def test_reference_text_uses_the_same_tag_profile_as_target_text(tmp_path):
    reference = tmp_path / "reference.wav"
    _wav(reference)
    estimate = estimate_duration(
        "one two",
        POLICY,
        voice_wav=reference,
        ref_text="[unsupported] one two",
        profile="plain",
    )
    assert estimate.seconds == 11.5


def test_fallback_estimate_when_reference_is_unusable(tmp_path):
    estimate = estimate_duration(
        "one two three four", POLICY, voice_wav=tmp_path / "missing.wav", ref_text=None
    )
    assert estimate.source == "registry-fallback"
    assert estimate.confidence == "low"
    assert estimate.seconds == 2.3


def test_oversized_units_split_without_breaking_bracket_tags(tmp_path):
    reference = tmp_path / "missing.wav"
    text = "[laughter] " + "word " * 70
    unit = UnitPart("s1", normalize_unit(text, "omnivoice", LAUGHTER_TAGS))
    chunks = plan_chunks(
        [unit],
        POLICY,
        profile="omnivoice",
        tag_map=LAUGHTER_TAGS,
        voice_wav=reference,
        ref_text=None,
    )
    assert len(chunks) > 1
    assert all(chunk.estimate.seconds <= POLICY.max_seconds for chunk in chunks)
    assert all(part.id == "s1" for chunk in chunks for part in chunk.units)
    rendered = " ".join(
        part.normalized.tts_text for chunk in chunks for part in chunk.units
    )
    assert "[ laughter" not in rendered
    assert rendered.count("[laughter]") == 1


def test_multiple_units_are_greedily_partitioned_near_target(tmp_path):
    units = [
        UnitPart(f"s{index}", normalize_unit("one two three four five", "plain"))
        for index in range(12)
    ]
    chunks = plan_chunks(
        units, POLICY, profile="plain", voice_wav=tmp_path / "missing.wav", ref_text=None
    )
    assert len(chunks) > 1
    assert [part.id for chunk in chunks for part in chunk.units] == [
        f"s{index}" for index in range(12)
    ]
    assert all(chunk.estimate.seconds <= POLICY.max_seconds for chunk in chunks)


def test_indivisible_span_above_maximum_is_rejected(tmp_path):
    unit = UnitPart("s0", normalize_unit("x" * 400, "plain"))
    try:
        plan_chunks(
            [unit],
            POLICY,
            profile="plain",
            voice_wav=tmp_path / "missing.wav",
            ref_text=None,
        )
    except SegmentationError as exc:
        assert "indivisible" in str(exc)
        assert "maximum" in str(exc)
    else:
        raise AssertionError("an indivisible over-limit span was synthesized")

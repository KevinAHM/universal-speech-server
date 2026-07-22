import base64
import json
import struct

import pytest

from speech_server.voices import VoiceError, VoiceStore


def _wav_bytes() -> bytes:
    data = b"\x00\x00\x00\x00"
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
    )
    return header + data


def test_clone_list_resolve_delete(tmp_path):
    store = VoiceStore(tmp_path)
    record = store.clone(
        display_name="Sebastian Sallow",
        lang_code="EN_US",
        audio_b64=base64.b64encode(_wav_bytes()).decode(),
        ref_text="Hello there.",
        reference_hash="abcd1234",
        tags=["cloned"],
    )
    assert record["voiceId"] == "default__Sebastian Sallow"
    assert (tmp_path / "default__Sebastian Sallow.wav").is_file()
    assert (
        tmp_path / "default__Sebastian Sallow.txt"
    ).read_text(encoding="utf-8") == "Hello there."
    metadata = json.loads(
        (tmp_path / "default__Sebastian Sallow.voice.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["schemaVersion"] == 1
    assert metadata["referenceHash"] == "abcd1234"
    assert [voice["voiceId"] for voice in store.list()] == [
        "default__Sebastian Sallow"
    ]
    audio, transcript = store.resolve("default__Sebastian Sallow")
    assert audio.suffix == ".wav" and transcript == "Hello there."
    assert store.delete("default__Sebastian Sallow") is True
    assert not (tmp_path / "default__Sebastian Sallow.voice.json").exists()
    assert store.list() == []


def test_resolve_missing_raises(tmp_path):
    with pytest.raises(KeyError):
        VoiceStore(tmp_path).resolve("nope")


def test_rejects_bad_audio(tmp_path):
    with pytest.raises(VoiceError):
        VoiceStore(tmp_path).clone(
            display_name="X",
            lang_code="EN_US",
            audio_b64=base64.b64encode(b"not audio data").decode(),
        )


def test_rejects_path_traversal_name(tmp_path):
    with pytest.raises(VoiceError):
        VoiceStore(tmp_path).clone(
            display_name="../evil",
            lang_code="EN_US",
            audio_b64=base64.b64encode(_wav_bytes()).decode(),
        )


def test_reclone_without_transcript_removes_stale_transcript(tmp_path):
    store = VoiceStore(tmp_path)
    kwargs = {
        "display_name": "Sebastian",
        "lang_code": "EN_US",
        "audio_b64": base64.b64encode(_wav_bytes()).decode(),
    }
    store.clone(**kwargs, ref_text="Old transcript")
    record = store.clone(**kwargs)
    assert record["hasTranscript"] is False
    _, transcript = store.resolve(record["voiceId"])
    assert transcript is None


def test_mp3_clone_keeps_its_real_extension_and_replaces_old_audio(tmp_path):
    store = VoiceStore(tmp_path)
    encoded_wav = base64.b64encode(_wav_bytes()).decode()
    encoded_mp3 = base64.b64encode(b"ID3" + b"\x00" * 20).decode()
    store.clone(display_name="Sebastian", audio_b64=encoded_wav)
    record = store.clone(display_name="Sebastian", audio_b64=encoded_mp3)
    audio, _ = store.resolve(record["voiceId"])
    assert audio.suffix == ".mp3"
    assert not (tmp_path / "default__Sebastian.wav").exists()

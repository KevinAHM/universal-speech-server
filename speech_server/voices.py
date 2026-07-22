"""Voice store over the audio_prompts audio/transcript/metadata layout."""

import base64
import binascii
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MAX_SAMPLE_BYTES = 100 * 1024 * 1024
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")
WORKSPACE = "default"
SEP = "__"
VOICE_METADATA_SUFFIX = ".voice.json"
VOICE_METADATA_SCHEMA_VERSION = 1


class VoiceError(ValueError):
    pass


def _sanitize(name: str) -> str:
    if any(sequence in name for sequence in ("/", "\\", "..")):
        raise VoiceError("invalid voice name")
    sanitized = re.sub(r"[^a-zA-Z0-9_\- ]", "", name).strip()[:100]
    if not sanitized:
        raise VoiceError("invalid voice name")
    return sanitized


def _validate_voice_id(voice_id: str) -> None:
    if not voice_id or any(sequence in voice_id for sequence in ("/", "\\", "..")):
        raise VoiceError("invalid voice id")


def _audio_suffix(data: bytes) -> str:
    if len(data) < 12:
        raise VoiceError("audio too short")
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return ".mp3"
    raise VoiceError("unsupported audio format (wav/mp3 only)")


class VoiceStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._audio_hash_cache: dict[Path, tuple[int, int, str]] = {}

    def _audio_hash(self, path: Path) -> str:
        stat = path.stat()
        cached = self._audio_hash_cache.get(path)
        if cached and cached[:2] == (stat.st_size, stat.st_mtime_ns):
            return cached[2]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._audio_hash_cache[path] = (stat.st_size, stat.st_mtime_ns, value)
        return value

    def _path(self, voice_id: str, suffix: str) -> Path:
        _validate_voice_id(voice_id)
        path = (self.root / f"{voice_id}{suffix}").resolve()
        if self.root not in path.parents:
            raise VoiceError("voice id escapes store")
        return path

    def clone(
        self,
        *,
        display_name: str,
        lang_code: str = "EN_US",
        audio_b64: str,
        ref_text: Optional[str] = None,
        reference_hash: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> dict:
        name = _sanitize(display_name)
        try:
            raw = base64.b64decode(audio_b64.split(",", 1)[-1], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise VoiceError("invalid base64 audio") from exc
        if len(raw) > MAX_SAMPLE_BYTES:
            raise VoiceError("sample too large")
        audio_suffix = _audio_suffix(raw)
        voice_id = f"{WORKSPACE}{SEP}{name}"
        audio_hash = hashlib.sha256(raw).hexdigest()
        transcript = ref_text.strip() if ref_text and ref_text.strip() else None
        transcript_hash = (
            hashlib.sha256(transcript.encode("utf-8")).hexdigest()
            if transcript
            else None
        )
        with self._lock:
            old_metadata = self._metadata_for_voice(voice_id)
            old_audio_path = next(
                (
                    self._path(voice_id, suffix)
                    for suffix in _AUDIO_EXTS
                    if self._path(voice_id, suffix).is_file()
                ),
                None,
            )
            old_audio_hash = (
                self._audio_hash(old_audio_path)
                if old_audio_path is not None
                else None
            )
            prepared_models = (
                old_metadata.get("preparedModels", {})
                if old_audio_hash == audio_hash
                else {}
            )
            if not isinstance(prepared_models, dict):
                prepared_models = {}
            audio_path = self._path(voice_id, audio_suffix)
            audio_path.write_bytes(raw)
            stat = audio_path.stat()
            self._audio_hash_cache[audio_path] = (
                stat.st_size,
                stat.st_mtime_ns,
                audio_hash,
            )
            for suffix in _AUDIO_EXTS:
                if suffix != audio_suffix:
                    stale_path = self._path(voice_id, suffix)
                    stale_path.unlink(missing_ok=True)
                    self._audio_hash_cache.pop(stale_path, None)
            transcript_path = audio_path.with_suffix(".txt")
            if transcript:
                transcript_path.write_text(transcript, encoding="utf-8")
            else:
                transcript_path.unlink(missing_ok=True)
            metadata = {
                "schemaVersion": VOICE_METADATA_SCHEMA_VERSION,
                "displayName": display_name,
                "langCode": lang_code,
                "referenceHash": reference_hash,
                "audioHash": audio_hash,
                "transcriptHash": transcript_hash,
                "tags": tags or ["cloned"],
                "preparedModels": prepared_models,
            }
            audio_path.with_suffix(VOICE_METADATA_SUFFIX).write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            return self._record(audio_path)

    def _metadata_for_voice(self, voice_id: str) -> dict:
        path = self._path(voice_id, VOICE_METADATA_SUFFIX)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(value, dict)
                and value.get("schemaVersion") == VOICE_METADATA_SCHEMA_VERSION
            ):
                return value
            return {}
        except (OSError, ValueError):
            return {}

    def _record(self, audio_path: Path) -> dict:
        metadata = self._metadata_for_voice(audio_path.stem)
        prepared_models = metadata.get("preparedModels", {})
        if not isinstance(prepared_models, dict):
            prepared_models = {}
        transcript_path = audio_path.with_suffix(".txt")
        transcript = (
            transcript_path.read_text(encoding="utf-8").strip()
            if transcript_path.is_file()
            else ""
        )
        return {
            "voiceId": audio_path.stem,
            "displayName": metadata.get(
                "displayName", audio_path.stem.split(SEP, 1)[-1]
            ),
            "langCode": metadata.get("langCode", "EN_US"),
            "referenceHash": metadata.get("referenceHash"),
            # Files are the source of truth.  Reporting stored hashes here can
            # leave preparation looking current after a sidecar is edited by an
            # administrator outside the REST API.
            "audioHash": self._audio_hash(audio_path),
            "transcriptHash": (
                hashlib.sha256(transcript.encode("utf-8")).hexdigest()
                if transcript
                else None
            ),
            "tags": metadata.get("tags", ["built-in"]),
            "hasTranscript": bool(transcript),
            "preparedModels": prepared_models,
        }

    def list(self) -> list[dict]:
        with self._lock:
            return [
                self._record(path)
                for path in sorted(self.root.iterdir())
                if path.is_file()
                and path.suffix.lower() in _AUDIO_EXTS
                and not path.name.startswith(".")
            ]

    def resolve(self, voice_id: str) -> tuple[Path, Optional[str]]:
        with self._lock:
            for extension in _AUDIO_EXTS:
                path = self._path(voice_id, extension)
                if path.is_file():
                    transcript_path = path.with_suffix(".txt")
                    transcript = (
                        transcript_path.read_text(encoding="utf-8").strip()
                        if transcript_path.is_file()
                        else None
                    )
                    return path, transcript
        raise KeyError(voice_id)

    def mark_prepared(
        self,
        voice_id: str,
        *,
        model_id: str,
        revision: str,
        inputs: tuple[str, ...],
    ) -> dict:
        """Persist successful model-specific reference preparation."""
        with self._lock:
            audio_path, transcript = self.resolve(voice_id)
            metadata_path = audio_path.with_suffix(VOICE_METADATA_SUFFIX)
            metadata = self._metadata_for_voice(voice_id)
            audio_hash = self._audio_hash(audio_path)
            transcript_hash = (
                hashlib.sha256(transcript.encode("utf-8")).hexdigest()
                if transcript
                else None
            )
            input_hashes = {}
            if "audio" in inputs:
                input_hashes["audio"] = audio_hash
            if "transcript" in inputs:
                input_hashes["transcript"] = transcript_hash
            marker = {
                "revision": revision,
                "inputHashes": input_hashes,
                "preparedAt": datetime.now(timezone.utc).isoformat(),
            }
            prepared = metadata.get("preparedModels")
            if not isinstance(prepared, dict):
                prepared = {}
            prepared[model_id] = marker
            metadata.update(
                {
                    "schemaVersion": VOICE_METADATA_SCHEMA_VERSION,
                    "audioHash": audio_hash,
                    "transcriptHash": transcript_hash,
                    "preparedModels": prepared,
                }
            )
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return marker

    def delete(self, voice_id: str) -> bool:
        with self._lock:
            found = False
            for extension in (*_AUDIO_EXTS, ".txt", VOICE_METADATA_SUFFIX, ".tokens.pt"):
                path = self._path(voice_id, extension)
                if path.is_file():
                    path.unlink()
                    self._audio_hash_cache.pop(path, None)
                    found = True
            return found

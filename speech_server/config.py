"""Server configuration: environment variables plus the models.toml registry."""

import hashlib
import json
import math
import os
import re
import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_MANIFEST = REPO_ROOT / "runtime" / "crispasr" / "installed.json"
DEFAULT_MODEL_MANIFEST = REPO_ROOT / "runtime" / "models" / "installed.json"


ESTIMATE_QUANTUM_BYTES = 256 * 1024 * 1024
SUPPORTED_CONTROL_IDS = {
    "numSteps",
    "firstSegmentSteps",
    "guidanceScale",
    "exaggeration",
}
BACKEND_CONTROL_IDS = {
    "omnivoice": {"numSteps", "firstSegmentSteps", "guidanceScale"},
    "chatterbox": {"numSteps", "guidanceScale", "exaggeration"},
}
TEXT_PROFILES = {"plain", "omnivoice"}
SEGMENTATION_ESTIMATORS = {"reference-rate"}
TRANSCRIPT_POLICIES = {"required", "optional", "unused"}
VOICE_PREPARATION_MODES = {"persistent", "lazy"}
VOICE_PREPARATION_INPUTS = {"audio", "transcript"}
MODEL_TASKS = {"tts", "asr"}
ASR_ENCODINGS = {"pcm_s16le"}
ASR_TIMESTAMP_MODES = {"none", "segment", "word"}
MAX_ASR_BIAS_TERMS = 1024
MAX_ASR_BIAS_TERM_LENGTH = 1024
# Broad BCP-47 shape validation. Detailed language support remains model metadata;
# this only rejects malformed catalog values without excluding uncommon valid tags.
LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,8}(?:-[a-z0-9]{1,8})*$")
REGISTRY_BUNDLE_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
PARALINGUISTIC_TAG_RE = re.compile(r"^\[[a-z0-9][a-z0-9 _-]{0,62}\]$")


def _files_installed(paths: list[Path]) -> bool:
    if not paths:
        return False
    try:
        return all(path.is_file() and path.stat().st_size > 0 for path in paths)
    except OSError:
        return False


@dataclass(frozen=True)
class ControlSpec:
    """A model-specific numeric option exposed on the v2 wire protocol."""

    id: str
    type: str
    minimum: float
    maximum: float
    step: float
    default: float

    def __post_init__(self) -> None:
        values = (self.minimum, self.maximum, self.step, self.default)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"control {self.id!r} values must be finite")
        if self.type not in {"integer", "number"}:
            raise ValueError(f"control {self.id!r} has unsupported type {self.type!r}")
        if self.minimum > self.maximum:
            raise ValueError(f"control {self.id!r} minimum exceeds maximum")
        if self.step <= 0:
            raise ValueError(f"control {self.id!r} step must be positive")
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError(f"control {self.id!r} default is outside its range")
        if self.type == "integer" and any(
            not float(value).is_integer()
            for value in (self.minimum, self.maximum, self.step, self.default)
        ):
            raise ValueError(f"integer control {self.id!r} requires integer values")
        increments = (self.default - self.minimum) / self.step
        if not math.isclose(increments, round(increments), abs_tol=1e-7):
            raise ValueError(
                f"control {self.id!r} default must use increments of {self.step:g}"
            )

    def as_capability(self) -> dict[str, Any]:
        cast = int if self.type == "integer" else float
        return {
            "id": self.id,
            "type": self.type,
            "minimum": cast(self.minimum),
            "maximum": cast(self.maximum),
            "step": cast(self.step),
            "default": cast(self.default),
        }


@dataclass(frozen=True)
class ParalinguisticTagSpec:
    """A model-supported inline non-verbal token and accepted input aliases."""

    token: str
    description: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = (self.token, *self.aliases)
        if any(not PARALINGUISTIC_TAG_RE.fullmatch(value) for value in values):
            raise ValueError("paralinguistic tags must be lowercase bracketed tokens")
        if (
            not self.description.strip()
            or self.description != self.description.strip()
            or len(self.description) > 256
        ):
            raise ValueError(
                "paralinguistic tag descriptions must contain 1 to 256 characters"
            )
        if len(set(values)) != len(values):
            raise ValueError("paralinguistic tag tokens and aliases must be unique")

    def as_capability(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "aliases": list(self.aliases),
            "description": self.description,
        }


@dataclass(frozen=True)
class SegmentationSpec:
    estimator: str
    min_seconds: float
    target_seconds: float
    max_seconds: float
    fallback_characters_per_second: float = 14.0
    fallback_words_per_second: float = 2.7
    safety_factor: float = 1.15

    def __post_init__(self) -> None:
        values = (
            self.min_seconds,
            self.target_seconds,
            self.max_seconds,
            self.fallback_characters_per_second,
            self.fallback_words_per_second,
            self.safety_factor,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("segmentation values must be finite and positive")
        if self.estimator not in SEGMENTATION_ESTIMATORS:
            raise ValueError(f"unsupported segmentation estimator {self.estimator!r}")
        if not self.min_seconds <= self.target_seconds <= self.max_seconds:
            raise ValueError("segmentation requires min_seconds <= target_seconds <= max_seconds")
        if self.safety_factor < 1.0:
            raise ValueError("segmentation safety_factor must be at least 1")

    def as_capability(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "minSeconds": self.min_seconds,
            "targetSeconds": self.target_seconds,
            "maxSeconds": self.max_seconds,
            "fallbackCharactersPerSecond": self.fallback_characters_per_second,
            "fallbackWordsPerSecond": self.fallback_words_per_second,
            "safetyFactor": self.safety_factor,
        }


@dataclass(frozen=True)
class VoiceReferenceSpec:
    transcript: str = "unused"
    preparation_mode: str = "lazy"
    preparation_inputs: tuple[str, ...] = ("audio",)

    def __post_init__(self) -> None:
        if self.transcript not in TRANSCRIPT_POLICIES:
            raise ValueError(
                f"unsupported voice-reference transcript policy {self.transcript!r}"
            )
        if self.preparation_mode not in VOICE_PREPARATION_MODES:
            raise ValueError(
                f"unsupported voice-reference preparation mode {self.preparation_mode!r}"
            )
        if not self.preparation_inputs:
            raise ValueError("voice-reference preparation inputs cannot be empty")
        if len(set(self.preparation_inputs)) != len(self.preparation_inputs):
            raise ValueError("voice-reference preparation inputs must be unique")
        unknown = set(self.preparation_inputs) - VOICE_PREPARATION_INPUTS
        if unknown:
            raise ValueError(
                f"unsupported voice-reference preparation inputs: {sorted(unknown)!r}"
            )
        if "transcript" in self.preparation_inputs and self.transcript == "unused":
            raise ValueError(
                "voice-reference preparation cannot use a transcript that the model ignores"
            )


@dataclass(frozen=True)
class ASRSpec:
    encoding: str = "pcm_s16le"
    sample_rates: tuple[int, ...] = (16000,)
    channels: int = 1
    max_seconds: float = 60.0
    automatic_language_detection: bool = False
    timestamps: tuple[str, ...] = ("none",)
    bias_terms: bool = False
    max_bias_terms: int = 256
    max_bias_term_length: int = 128
    hotword_boost: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.encoding, str) or self.encoding not in ASR_ENCODINGS:
            raise ValueError(f"unsupported ASR encoding {self.encoding!r}")
        if not self.sample_rates or any(
            isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0
            for rate in self.sample_rates
        ):
            raise ValueError("ASR sample_rates must contain positive integers")
        if len(set(self.sample_rates)) != len(self.sample_rates):
            raise ValueError("ASR sample_rates must be unique")
        if isinstance(self.channels, bool) or not isinstance(self.channels, int) or self.channels != 1:
            raise ValueError("only mono ASR input is supported")
        if (
            isinstance(self.max_seconds, bool)
            or not isinstance(self.max_seconds, (int, float))
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("ASR max_seconds must be finite and positive")
        if (
            not isinstance(self.automatic_language_detection, bool)
            or not isinstance(self.bias_terms, bool)
        ):
            raise ValueError("ASR feature flags must be booleans")
        if (
            not self.timestamps
            or any(not isinstance(mode, str) for mode in self.timestamps)
            or "none" not in self.timestamps
        ):
            raise ValueError("ASR timestamps must include 'none'")
        if len(set(self.timestamps)) != len(self.timestamps):
            raise ValueError("ASR timestamps must be unique")
        unknown = set(self.timestamps) - ASR_TIMESTAMP_MODES
        if unknown:
            raise ValueError(f"unsupported ASR timestamp modes: {sorted(unknown)!r}")
        if (
            isinstance(self.max_bias_terms, bool)
            or not isinstance(self.max_bias_terms, int)
            or not 0 < self.max_bias_terms <= MAX_ASR_BIAS_TERMS
        ):
            raise ValueError(
                f"ASR max_bias_terms must be between 1 and {MAX_ASR_BIAS_TERMS}"
            )
        if (
            isinstance(self.max_bias_term_length, bool)
            or not isinstance(self.max_bias_term_length, int)
            or not 0 < self.max_bias_term_length <= MAX_ASR_BIAS_TERM_LENGTH
        ):
            raise ValueError(
                "ASR max_bias_term_length must be between 1 and "
                f"{MAX_ASR_BIAS_TERM_LENGTH}"
            )
        if (
            isinstance(self.hotword_boost, bool)
            or not isinstance(self.hotword_boost, (int, float))
            or not math.isfinite(self.hotword_boost)
            or self.hotword_boost <= 0
        ):
            raise ValueError("ASR hotword_boost must be finite and positive")

    def as_capability(self) -> dict[str, Any]:
        return {
            "audio": {
                "encodings": [self.encoding],
                "sampleRates": list(self.sample_rates),
                "channels": [self.channels],
                "maxSeconds": self.max_seconds,
            },
            "automaticLanguageDetection": self.automatic_language_detection,
            "timestamps": list(self.timestamps),
            "biasTerms": {
                "supported": self.bias_terms,
                "maxCount": self.max_bias_terms if self.bias_terms else 0,
                "maxLength": self.max_bias_term_length if self.bias_terms else 0,
            },
        }


@dataclass
class ModelSpec:
    id: str
    backend: str
    model_path: Path
    sample_rate: int
    codec_path: Optional[Path] = None
    task: str = "tts"
    cloning: bool = True
    languages: list[str] = field(default_factory=lambda: ["en"])
    controls: list[ControlSpec] = field(default_factory=list)
    estimated_ram_mb: Optional[int] = None
    estimated_vram_mb: Optional[int] = None
    text_profile: str = "plain"
    paralinguistic_tags: list[ParalinguisticTagSpec] = field(default_factory=list)
    segmentation: Optional[SegmentationSpec] = None
    voice_reference: VoiceReferenceSpec = field(default_factory=VoiceReferenceSpec)
    asr: Optional[ASRSpec] = None
    registry_bundle: Optional[str] = None
    extra_paths: list[Path] = field(default_factory=list)

    @cached_property
    def paralinguistic_tag_map(self) -> dict[str, str]:
        return {
            accepted.lower(): tag.token
            for tag in self.paralinguistic_tags
            for accepted in (tag.token, *tag.aliases)
        }

    @property
    def audio_tags(self) -> list[str]:
        return [tag.token for tag in self.paralinguistic_tags]

    @property
    def tag_aliases(self) -> dict[str, str]:
        return {
            alias: tag.token
            for tag in self.paralinguistic_tags
            for alias in tag.aliases
        }

    @property
    def component_paths(self) -> list[Path]:
        return [
            path
            for path in (self.model_path, self.codec_path, *self.extra_paths)
            if path is not None
        ]

    @property
    def installed(self) -> bool:
        """Whether every runtime component required by this model exists."""
        return _files_installed(self.component_paths)

    @property
    def installable(self) -> bool:
        """Whether the catalog declares a CrispASR default download bundle."""
        return bool(self.registry_bundle)

    @property
    def component_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.component_paths if path.is_file())

    def _estimate(self, override_mb: Optional[int]) -> dict[str, Any]:
        if override_mb is not None:
            return {
                "estimatedBytes": override_mb * 1024 * 1024,
                "source": "registry",
                "confidence": "declared",
            }
        component_bytes = self.component_bytes
        estimated = (
            math.ceil((component_bytes * 2) / ESTIMATE_QUANTUM_BYTES)
            * ESTIMATE_QUANTUM_BYTES
            if component_bytes
            else None
        )
        return {
            "estimatedBytes": estimated,
            "source": "component-size-heuristic" if estimated is not None else "unavailable",
            "confidence": "low" if estimated is not None else "unavailable",
        }

    def resource_requirements(self) -> dict[str, Any]:
        return {
            "componentBytes": self.component_bytes,
            "ram": self._estimate(self.estimated_ram_mb),
            "vram": self._estimate(self.estimated_vram_mb),
        }

    @cached_property
    def voice_preparation_revision(self) -> str:
        """Startup-stable model-local cache revision.

        Registry discovery is intentionally restart-based.  Cache this value on
        first use so a component changed underneath a running server cannot make
        capabilities disagree with the startup registry revision.
        """
        components = []
        for path in self.component_paths:
            if path.is_file():
                stat = path.stat()
                components.append({"size": stat.st_size, "mtimeNs": stat.st_mtime_ns})
            else:
                components.append({"missing": True})
        payload = {
            "id": self.id,
            "backend": self.backend,
            "transcript": self.voice_reference.transcript,
            "mode": self.voice_reference.preparation_mode,
            "inputs": self.voice_reference.preparation_inputs,
            "components": components,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def voice_reference_capability(self) -> dict[str, Any]:
        return {
            "transcript": self.voice_reference.transcript,
            "preparation": {
                "mode": self.voice_reference.preparation_mode,
                "inputs": list(self.voice_reference.preparation_inputs),
                "revision": self.voice_preparation_revision,
            },
        }


@dataclass
class AlignerSpec:
    id: str
    backend: str
    model_path: Path
    languages: list[str]
    n_threads: int = 4
    estimated_ram_mb: Optional[int] = None
    estimated_vram_mb: Optional[int] = None
    registry_bundle: Optional[str] = None
    extra_paths: list[Path] = field(default_factory=list)

    @property
    def component_paths(self) -> list[Path]:
        return [self.model_path, *self.extra_paths]

    @property
    def installed(self) -> bool:
        return _files_installed(self.component_paths)

    @property
    def installable(self) -> bool:
        return bool(self.registry_bundle)

    @property
    def component_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.component_paths if path.is_file())

    def _estimate(self, override_mb: Optional[int]) -> dict[str, Any]:
        if override_mb is not None:
            return {
                "estimatedBytes": override_mb * 1024 * 1024,
                "source": "registry",
                "confidence": "declared",
            }
        component_bytes = self.component_bytes
        estimated = (
            math.ceil((component_bytes * 2) / ESTIMATE_QUANTUM_BYTES)
            * ESTIMATE_QUANTUM_BYTES
            if component_bytes
            else None
        )
        return {
            "estimatedBytes": estimated,
            "source": "component-size-heuristic" if estimated is not None else "unavailable",
            "confidence": "low" if estimated is not None else "unavailable",
        }

    def resource_requirements(self) -> dict[str, Any]:
        return {
            "componentBytes": self.component_bytes,
            "ram": self._estimate(self.estimated_ram_mb),
            "vram": self._estimate(self.estimated_vram_mb),
        }


@dataclass
class ServerConfig:
    models: dict[str, ModelSpec]
    upscaler: Optional[ModelSpec] = None
    aligner: Optional[AlignerSpec] = None
    resident_limit: int = 1
    auth_token: str = ""
    host: str = "127.0.0.1"
    port: int = 8100
    lib_path: Optional[Path] = None
    voice_dir: Path = REPO_ROOT / "audio_prompts"
    registry_revision: str = ""
    model_manifest_path: Path = DEFAULT_MODEL_MANIFEST
    debug: bool = False


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1, 0, true, false, yes, no, on, or off"
    )


def _library_from_runtime_manifest(manifest_path: Path) -> Path:
    """Resolve an installed runtime library without allowing path traversal."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read CrispASR runtime manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        raise ValueError(f"unsupported CrispASR runtime manifest: {manifest_path}")
    relative = manifest.get("library")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(
            f"CrispASR runtime manifest has no library path: {manifest_path}"
        )
    root = manifest_path.resolve().parent
    library = (root / relative).resolve()
    if not library.is_relative_to(root):
        raise ValueError(
            f"CrispASR runtime library escapes its install directory: {relative}"
        )
    if not library.is_file():
        raise ValueError(f"CrispASR runtime library does not exist: {library}")
    return library


def _default_lib_path() -> Optional[Path]:
    env = os.getenv("SPEECH_SERVER_LIB", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    manifest_override = os.getenv("SPEECH_SERVER_RUNTIME_MANIFEST", "").strip()
    manifest_path = Path(
        manifest_override or DEFAULT_RUNTIME_MANIFEST
    ).expanduser()
    if manifest_path.is_file():
        return _library_from_runtime_manifest(manifest_path)
    if manifest_override:
        raise ValueError(
            f"CrispASR runtime manifest does not exist: {manifest_path.resolve()}"
        )
    # Developer checkouts may contain several historical build directories.
    # Prefer the current-main Release build; build-cuda12-runtime predates the
    # unified OmniVoice/Chatterbox voice-cloning dispatch and can open a model
    # successfully but fail later at set_voice with rc=-3.
    local_libraries = (
        REPO_ROOT / "CrispASR" / "build-main-cuda" / "bin" / "Release" / "crispasr.dll",
        REPO_ROOT / "CrispASR" / "build-main-cuda" / "bin" / "crispasr.dll",
        REPO_ROOT / "CrispASR" / "build-cuda12-runtime" / "bin" / "crispasr.dll",
    )
    for local_library in local_libraries:
        if local_library.is_file():
            return local_library
    for candidate in sorted(REPO_ROOT.glob("libcrispasr-*/bin")):
        for name in ("crispasr.dll", "libcrispasr.so", "libcrispasr.dylib"):
            library = candidate / name
            if library.is_file():
                return library
    return None


def _model_manifest_path() -> Path:
    value = os.getenv("SPEECH_SERVER_MODEL_MANIFEST", "").strip()
    return Path(value or DEFAULT_MODEL_MANIFEST).expanduser().resolve()


def _apply_model_installations(
    manifest_path: Path,
    specs: dict[str, ModelSpec | AlignerSpec],
) -> None:
    """Apply complete, verified installer records to catalog component paths."""
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read model installation manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        raise ValueError(f"unsupported model installation manifest: {manifest_path}")
    installations = manifest.get("installations")
    if not isinstance(installations, dict):
        raise ValueError(
            f"model installation manifest has no installations: {manifest_path}"
        )

    root = manifest_path.parent.resolve()
    for key, spec in specs.items():
        record = installations.get(key)
        if record is None:
            continue
        if not isinstance(record, dict):
            raise ValueError(f"model installation {key!r} must be an object")
        if record.get("registryBundle") != spec.registry_bundle:
            continue
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"model installation {key!r} has no artifacts")
        resolved: dict[str, list[Path]] = {
            "primary": [],
            "companion": [],
            "extra": [],
        }
        complete = True
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValueError(
                    f"model installation {key!r} has an invalid artifact"
                )
            kind = artifact.get("kind")
            filename = artifact.get("filename")
            relative = artifact.get("path")
            sha256 = artifact.get("sha256")
            size = artifact.get("size")
            if kind not in resolved:
                raise ValueError(
                    f"model installation {key!r} has invalid artifact kind"
                )
            if (
                not isinstance(filename, str)
                or not filename
                or PurePosixPath(filename).name != filename
                or PureWindowsPath(filename).name != filename
                or not isinstance(relative, str)
                or not relative
                or not isinstance(sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", sha256)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise ValueError(
                    f"model installation {key!r} has invalid artifact metadata"
                )
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root) or candidate.name != filename:
                raise ValueError(
                    f"model installation {key!r} artifact escapes its install directory"
                )
            if not candidate.is_file() or candidate.stat().st_size != size:
                complete = False
            resolved[kind].append(candidate)
        if len(resolved["primary"]) != 1 or len(resolved["companion"]) > 1:
            raise ValueError(f"model installation {key!r} has invalid artifact roles")
        if (
            isinstance(spec, ModelSpec)
            and spec.codec_path is not None
            and not resolved["companion"]
        ):
            complete = False
        if not complete:
            continue
        spec.model_path = resolved["primary"][0]
        if isinstance(spec, ModelSpec):
            spec.codec_path = (
                resolved["companion"][0] if resolved["companion"] else None
            )
        spec.extra_paths = list(resolved["extra"])


def _resolve_path(base: Path, value: str) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def _number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a number")
    return float(value)


def _integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    return value


def _registry_bundle(value: Any, description: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a nonempty string")
    normalized = value.strip().lower()
    if not REGISTRY_BUNDLE_RE.fullmatch(normalized):
        raise ValueError(f"{description} contains invalid characters")
    return normalized


def _languages(value: Any, description: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(language, str) and language.strip() for language in value
    ):
        raise ValueError(f"{description} must be a list of strings")
    normalized = [language.strip().lower().replace("_", "-") for language in value]
    invalid = [
        language
        for language in normalized
        if language != "*" and not LANGUAGE_TAG_RE.fullmatch(language)
    ]
    if invalid:
        raise ValueError(
            f"{description} contains invalid language tags: {invalid!r}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{description} must not contain duplicates")
    return normalized


def _parse_segmentation(model_id: str, raw: Any) -> Optional[SegmentationSpec]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"model {model_id!r} segmentation must be a table")
    return SegmentationSpec(
        estimator=str(raw.get("estimator", "reference-rate")),
        min_seconds=_number(raw["min_seconds"], f"model {model_id!r} min_seconds"),
        target_seconds=_number(raw["target_seconds"], f"model {model_id!r} target_seconds"),
        max_seconds=_number(raw["max_seconds"], f"model {model_id!r} max_seconds"),
        fallback_characters_per_second=_number(
            raw.get("fallback_characters_per_second", 14.0),
            f"model {model_id!r} fallback_characters_per_second",
        ),
        fallback_words_per_second=_number(
            raw.get("fallback_words_per_second", 2.7),
            f"model {model_id!r} fallback_words_per_second",
        ),
        safety_factor=_number(
            raw.get("safety_factor", 1.15), f"model {model_id!r} safety_factor"
        ),
    )


def _parse_voice_reference(model_id: str, raw: Any) -> VoiceReferenceSpec:
    if raw is None:
        return VoiceReferenceSpec()
    if not isinstance(raw, dict):
        raise ValueError(f"model {model_id!r} voice_reference must be a table")
    inputs = raw.get("preparation_inputs", ["audio"])
    if not isinstance(inputs, list) or not all(
        isinstance(item, str) and item.strip() for item in inputs
    ):
        raise ValueError(
            f"model {model_id!r} voice_reference preparation_inputs must be a list of strings"
        )
    return VoiceReferenceSpec(
        transcript=str(raw.get("transcript", "unused")),
        preparation_mode=str(raw.get("preparation", "lazy")),
        preparation_inputs=tuple(item.strip() for item in inputs),
    )


def _parse_paralinguistic_tags(
    model_id: str, raw: Any
) -> list[ParalinguisticTagSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError(
            f"model {model_id!r} paralinguistic_tags must be an array of tables"
        )
    tags: list[ParalinguisticTagSpec] = []
    accepted: set[str] = set()
    for item in raw:
        token = item.get("token")
        description = item.get("description")
        aliases = item.get("aliases", [])
        if (
            not isinstance(token, str)
            or not isinstance(description, str)
            or not isinstance(aliases, list)
            or any(not isinstance(alias, str) for alias in aliases)
        ):
            raise ValueError(
                f"model {model_id!r} has invalid paralinguistic tag metadata"
            )
        tag = ParalinguisticTagSpec(
            token=token,
            description=description,
            aliases=tuple(aliases),
        )
        collisions = accepted.intersection((tag.token, *tag.aliases))
        if collisions:
            raise ValueError(
                f"model {model_id!r} repeats paralinguistic tag {sorted(collisions)[0]!r}"
            )
        accepted.update((tag.token, *tag.aliases))
        tags.append(tag)
    return tags


def _parse_asr(model_id: str, raw: Any) -> ASRSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"ASR model {model_id!r} requires a transcription table")
    sample_rates = raw.get("sample_rates", [16000])
    if not isinstance(sample_rates, list) or not all(
        isinstance(rate, int) and not isinstance(rate, bool) for rate in sample_rates
    ):
        raise ValueError(f"model {model_id!r} sample_rates must be a list of integers")
    timestamps = raw.get("timestamps", ["none"])
    if not isinstance(timestamps, list) or not all(
        isinstance(mode, str) and mode.strip() for mode in timestamps
    ):
        raise ValueError(f"model {model_id!r} timestamps must be a list of strings")
    automatic = raw.get("automatic_language_detection", False)
    bias_terms = raw.get("bias_terms", False)
    if not isinstance(automatic, bool) or not isinstance(bias_terms, bool):
        raise ValueError(
            f"model {model_id!r} ASR feature flags must be booleans"
        )
    return ASRSpec(
        encoding=str(raw.get("encoding", "pcm_s16le")),
        sample_rates=tuple(sample_rates),
        channels=_integer(raw.get("channels", 1), f"model {model_id!r} channels"),
        max_seconds=_number(
            raw.get("max_seconds", 60), f"model {model_id!r} max_seconds"
        ),
        automatic_language_detection=automatic,
        timestamps=tuple(mode.strip() for mode in timestamps),
        bias_terms=bias_terms,
        max_bias_terms=_integer(
            raw.get("max_bias_terms", 256), f"model {model_id!r} max_bias_terms"
        ),
        max_bias_term_length=_integer(
            raw.get("max_bias_term_length", 128),
            f"model {model_id!r} max_bias_term_length",
        ),
        hotword_boost=_number(
            raw.get("hotword_boost", 2.0), f"model {model_id!r} hotword_boost"
        ),
    )


def _parse_controls(
    model_id: str, backend: str, raw_controls: list[dict[str, Any]]
) -> list[ControlSpec]:
    controls: list[ControlSpec] = []
    seen: set[str] = set()
    for raw in raw_controls:
        control = ControlSpec(
            id=str(raw["id"]),
            type=str(raw["type"]),
            minimum=_number(raw["minimum"], f"control {raw.get('id')!r} minimum"),
            maximum=_number(raw["maximum"], f"control {raw.get('id')!r} maximum"),
            step=_number(raw["step"], f"control {raw.get('id')!r} step"),
            default=_number(raw["default"], f"control {raw.get('id')!r} default"),
        )
        if control.id not in SUPPORTED_CONTROL_IDS:
            raise ValueError(f"model {model_id!r} declares unknown control {control.id!r}")
        if control.id not in BACKEND_CONTROL_IDS.get(backend, set()):
            raise ValueError(
                f"model {model_id!r} declares control {control.id!r} "
                f"unsupported by backend {backend!r}"
            )
        if control.id in seen:
            raise ValueError(f"model {model_id!r} repeats control {control.id!r}")
        seen.add(control.id)
        controls.append(control)
    return controls


def _registry_revision(
    raw_registry: bytes, specs: list[ModelSpec | AlignerSpec]
) -> str:
    """Hash startup registry bytes and local component metadata, never their paths."""
    metadata = []
    for spec in specs:
        components = []
        for path in spec.component_paths:
            if path.is_file():
                stat = path.stat()
                components.append({"size": stat.st_size, "mtimeNs": stat.st_mtime_ns})
            else:
                components.append({"missing": True})
        metadata.append({"id": spec.id, "components": components})
    digest = hashlib.sha256()
    digest.update(raw_registry)
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def load_config(models_path: Optional[Path] = None) -> ServerConfig:
    models_path = Path(
        models_path or os.getenv("SPEECH_SERVER_MODELS", REPO_ROOT / "models.toml")
    )
    raw_registry = models_path.read_bytes()
    data = tomllib.loads(raw_registry.decode("utf-8"))
    base = models_path.resolve().parent
    models: dict[str, ModelSpec] = {}
    raw_models = data.get("models", {})
    if not isinstance(raw_models, dict):
        raise ValueError("models must be a table")
    for model_id, model in raw_models.items():
        if not isinstance(model, dict):
            raise ValueError(f"model {model_id!r} must be a table")
        backend = str(model["backend"])
        task = str(model.get("task", "tts"))
        if task not in MODEL_TASKS:
            raise ValueError(f"model {model_id!r} has unsupported task {task!r}")
        codec = model.get("codec")
        sample_rate = _integer(model["sample_rate"], f"model {model_id!r} sample_rate")
        if sample_rate <= 0:
            raise ValueError(f"model {model_id!r} sample_rate must be positive")
        languages = _languages(
            model.get("languages", ["en"]), f"model {model_id!r} languages"
        )
        estimated_ram_mb = _integer(model["estimated_ram_mb"], f"model {model_id!r} estimated_ram_mb") if "estimated_ram_mb" in model else None
        estimated_vram_mb = _integer(model["estimated_vram_mb"], f"model {model_id!r} estimated_vram_mb") if "estimated_vram_mb" in model else None
        if estimated_ram_mb is not None and estimated_ram_mb <= 0:
            raise ValueError(f"model {model_id!r} estimated_ram_mb must be positive")
        if estimated_vram_mb is not None and estimated_vram_mb <= 0:
            raise ValueError(f"model {model_id!r} estimated_vram_mb must be positive")
        text_profile = str(model.get("text_profile", "plain"))
        if text_profile not in TEXT_PROFILES:
            raise ValueError(f"model {model_id!r} has unknown text_profile {text_profile!r}")
        cloning = model.get("cloning", task == "tts")
        if not isinstance(cloning, bool):
            raise ValueError(f"model {model_id!r} cloning must be a boolean")
        raw_controls = model.get("controls", [])
        if not isinstance(raw_controls, list) or any(
            not isinstance(control, dict) for control in raw_controls
        ):
            raise ValueError(f"model {model_id!r} controls must be an array of tables")
        segmentation = _parse_segmentation(model_id, model.get("segmentation"))
        voice_reference = _parse_voice_reference(
            model_id, model.get("voice_reference")
        )
        paralinguistic_tags = _parse_paralinguistic_tags(
            model_id, model.get("paralinguistic_tags")
        )
        asr = _parse_asr(model_id, model.get("transcription")) if task == "asr" else None
        if task == "asr":
            if cloning or raw_controls or segmentation is not None or paralinguistic_tags:
                raise ValueError(
                    f"ASR model {model_id!r} cannot declare cloning, TTS controls, "
                    "segmentation, or paralinguistic tags"
                )
            if model.get("voice_reference") is not None or text_profile != "plain":
                raise ValueError(
                    f"ASR model {model_id!r} cannot declare TTS text or voice-reference settings"
                )
            if sample_rate not in asr.sample_rates:
                raise ValueError(
                    f"ASR model {model_id!r} sample_rate must be one of its transcription sample_rates"
                )
        elif model.get("transcription") is not None:
            raise ValueError(f"TTS model {model_id!r} cannot declare transcription settings")
        models[model_id] = ModelSpec(
            id=model_id,
            backend=backend,
            model_path=_resolve_path(base, model["model"]),
            sample_rate=sample_rate,
            codec_path=_resolve_path(base, codec) if codec else None,
            task=task,
            cloning=cloning,
            languages=languages,
            controls=_parse_controls(model_id, backend, raw_controls),
            estimated_ram_mb=estimated_ram_mb,
            estimated_vram_mb=estimated_vram_mb,
            text_profile=text_profile,
            paralinguistic_tags=paralinguistic_tags,
            segmentation=segmentation,
            voice_reference=voice_reference,
            asr=asr,
            registry_bundle=_registry_bundle(
                model.get("registry_bundle"),
                f"model {model_id!r} registry_bundle",
            ),
        )
    upscaler = None
    upscale_data = data.get("upscaler")
    if upscale_data:
        if not isinstance(upscale_data, dict):
            raise ValueError("upscaler must be a table")
        upscaler_sample_rate = _integer(
            upscale_data.get("sample_rate", 48000), "upscaler sample_rate"
        )
        if upscaler_sample_rate <= 0:
            raise ValueError("upscaler sample_rate must be positive")
        upscaler_ram_mb = (
            _integer(upscale_data["estimated_ram_mb"], "upscaler estimated_ram_mb")
            if "estimated_ram_mb" in upscale_data
            else None
        )
        upscaler_vram_mb = (
            _integer(upscale_data["estimated_vram_mb"], "upscaler estimated_vram_mb")
            if "estimated_vram_mb" in upscale_data
            else None
        )
        if upscaler_ram_mb is not None and upscaler_ram_mb <= 0:
            raise ValueError("upscaler estimated_ram_mb must be positive")
        if upscaler_vram_mb is not None and upscaler_vram_mb <= 0:
            raise ValueError("upscaler estimated_vram_mb must be positive")
        upscaler = ModelSpec(
            id=str(upscale_data.get("id", "voxcpm2-vae")),
            backend=str(upscale_data.get("backend", "voxcpm2-vae")),
            model_path=_resolve_path(base, upscale_data["model"]),
            sample_rate=upscaler_sample_rate,
            task="audio-to-audio",
            cloning=False,
            languages=[],
            estimated_ram_mb=upscaler_ram_mb,
            estimated_vram_mb=upscaler_vram_mb,
            registry_bundle=_registry_bundle(
                upscale_data.get("registry_bundle"), "upscaler registry_bundle"
            ),
        )
    aligner = None
    aligner_data = data.get("aligner")
    if aligner_data:
        if not isinstance(aligner_data, dict):
            raise ValueError("aligner must be a table")
        languages = _languages(
            aligner_data.get("languages", ["en"]), "aligner languages"
        )
        n_threads = _integer(aligner_data.get("n_threads", 4), "aligner n_threads")
        if n_threads <= 0:
            raise ValueError("aligner n_threads must be positive")
        aligner_ram_mb = (
            _integer(aligner_data["estimated_ram_mb"], "aligner estimated_ram_mb")
            if "estimated_ram_mb" in aligner_data
            else None
        )
        aligner_vram_mb = (
            _integer(aligner_data["estimated_vram_mb"], "aligner estimated_vram_mb")
            if "estimated_vram_mb" in aligner_data
            else None
        )
        if aligner_ram_mb is not None and aligner_ram_mb <= 0:
            raise ValueError("aligner estimated_ram_mb must be positive")
        if aligner_vram_mb is not None and aligner_vram_mb <= 0:
            raise ValueError("aligner estimated_vram_mb must be positive")
        aligner = AlignerSpec(
            id=str(aligner_data.get("id", "canary-ctc-aligner")),
            backend=str(aligner_data.get("backend", "canary-ctc")),
            model_path=_resolve_path(base, aligner_data["model"]),
            languages=languages,
            n_threads=n_threads,
            estimated_ram_mb=aligner_ram_mb,
            estimated_vram_mb=aligner_vram_mb,
            registry_bundle=_registry_bundle(
                aligner_data.get("registry_bundle"), "aligner registry_bundle"
            ),
        )
    resident_limit = _integer(data.get("resident_limit", 1), "resident_limit")
    if resident_limit <= 0:
        raise ValueError("resident_limit must be positive")
    model_manifest_path = _model_manifest_path()
    installed_specs: dict[str, ModelSpec | AlignerSpec] = {
        **{f"model:{model_id}": spec for model_id, spec in models.items()},
        **({"upscaler": upscaler} if upscaler else {}),
        **({"aligner": aligner} if aligner else {}),
    }
    _apply_model_installations(model_manifest_path, installed_specs)
    cfg = ServerConfig(
        models=models,
        upscaler=upscaler,
        aligner=aligner,
        resident_limit=resident_limit,
        auth_token=os.getenv("SPEECH_SERVER_TOKEN", "").strip(),
        host=os.getenv("SPEECH_SERVER_HOST", "127.0.0.1"),
        port=int(os.getenv("SPEECH_SERVER_PORT", "8100")),
        lib_path=_default_lib_path(),
        voice_dir=Path(
            os.getenv("SPEECH_SERVER_VOICES", REPO_ROOT / "audio_prompts")
        ).expanduser().resolve(),
        model_manifest_path=model_manifest_path,
        debug=_environment_flag("SPEECH_SERVER_DEBUG"),
    )
    cfg.registry_revision = _registry_revision(
        raw_registry,
        [
            *cfg.models.values(),
            *([upscaler] if upscaler else []),
            *([aligner] if aligner else []),
        ],
    )
    return cfg

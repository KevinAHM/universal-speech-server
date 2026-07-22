import json
from pathlib import Path

import pytest

import speech_server.config as config_module
from speech_server.config import (
    ControlSpec,
    ModelSpec,
    ParalinguisticTagSpec,
    VoiceReferenceSpec,
    _default_lib_path,
    _library_from_runtime_manifest,
    load_config,
)

TOML = """
resident_limit = 2

[upscaler]
backend = "voxcpm2-vae"
model = "voxcpm2-vae.gguf"
sample_rate = 48000

[models.omnivoice]
backend = "omnivoice"
registry_bundle = "omnivoice"
model = "models/omnivoice-q8_0.gguf"
codec = "models/omnivoice-tokenizer-f16.gguf"
sample_rate = 24000
languages = ["en", "de", "fr", "es", "it", "pt"]
text_profile = "omnivoice"

[[models.omnivoice.paralinguistic_tags]]
token = "[laughter]"
aliases = ["[laugh]", "[laughs]", "[laughing]"]
description = "Inserts natural laughter."

[models.omnivoice.voice_reference]
transcript = "required"
preparation = "persistent"
preparation_inputs = ["audio", "transcript"]

[models.omnivoice.segmentation]
estimator = "reference-rate"
min_seconds = 8
target_seconds = 20
max_seconds = 28
fallback_characters_per_second = 14
fallback_words_per_second = 2.7
safety_factor = 1.15

[[models.omnivoice.controls]]
id = "numSteps"
type = "integer"
minimum = 8
maximum = 64
step = 4
default = 32

[models.chatterbox]
backend = "chatterbox"
model = "models/chatterbox-q8_0.gguf"
sample_rate = 24000

[models.parakeet]
task = "asr"
backend = "parakeet"
registry_bundle = "parakeet"
model = "models/parakeet.gguf"
sample_rate = 16000
cloning = false
languages = ["en", "de"]

[models.parakeet.transcription]
sample_rates = [16000]
max_seconds = 60
automatic_language_detection = true
timestamps = ["none", "segment", "word"]
bias_terms = true
"""


def test_explicit_runtime_manifest_must_exist(monkeypatch, tmp_path: Path):
    missing = tmp_path / "missing.json"
    monkeypatch.delenv("SPEECH_SERVER_LIB", raising=False)
    monkeypatch.setenv("SPEECH_SERVER_RUNTIME_MANIFEST", str(missing))
    with pytest.raises(ValueError, match="manifest does not exist"):
        _default_lib_path()


def test_default_lib_prefers_current_main_build(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SPEECH_SERVER_LIB", raising=False)
    monkeypatch.delenv("SPEECH_SERVER_RUNTIME_MANIFEST", raising=False)
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        config_module, "DEFAULT_RUNTIME_MANIFEST", tmp_path / "runtime" / "missing.json"
    )
    stale = tmp_path / "CrispASR" / "build-cuda12-runtime" / "bin" / "crispasr.dll"
    current = (
        tmp_path
        / "CrispASR"
        / "build-main-cuda"
        / "bin"
        / "Release"
        / "crispasr.dll"
    )
    stale.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    stale.touch()
    current.touch()

    assert _default_lib_path() == current


def test_load_config(tmp_path: Path, monkeypatch):
    path = tmp_path / "models.toml"
    path.write_text(TOML, encoding="utf-8")
    monkeypatch.setenv("SPEECH_SERVER_TOKEN", "secret")
    monkeypatch.setenv(
        "SPEECH_SERVER_MODEL_MANIFEST", str(tmp_path / "installed.json")
    )
    cfg = load_config(models_path=path)
    assert cfg.auth_token == "secret"
    assert cfg.resident_limit == 2
    omni = cfg.models["omnivoice"]
    assert omni.backend == "omnivoice"
    assert omni.registry_bundle == "omnivoice"
    assert omni.installable is True
    assert omni.sample_rate == 24000
    assert omni.text_profile == "omnivoice"
    assert omni.paralinguistic_tags == [
        ParalinguisticTagSpec(
            token="[laughter]",
            aliases=("[laugh]", "[laughs]", "[laughing]"),
            description="Inserts natural laughter.",
        )
    ]
    assert omni.paralinguistic_tag_map["[laugh]"] == "[laughter]"
    assert cfg.models["chatterbox"].paralinguistic_tags == []
    assert omni.segmentation is not None
    assert omni.segmentation.as_capability()["targetSeconds"] == 20
    assert omni.voice_reference.transcript == "required"
    assert omni.voice_reference.preparation_mode == "persistent"
    assert omni.voice_reference.preparation_inputs == ("audio", "transcript")
    assert len(omni.voice_preparation_revision) == 64
    assert omni.codec_path and omni.codec_path.name == "omnivoice-tokenizer-f16.gguf"
    assert cfg.models["chatterbox"].codec_path is None
    parakeet = cfg.models["parakeet"]
    assert parakeet.task == "asr"
    assert parakeet.cloning is False
    assert parakeet.asr is not None
    assert parakeet.asr.max_seconds == 60
    assert parakeet.asr.bias_terms is True
    assert parakeet.registry_bundle == "parakeet"
    assert omni.model_path.is_absolute()
    assert cfg.upscaler is not None
    assert cfg.upscaler.id == "voxcpm2-vae"
    assert cfg.upscaler.backend == "voxcpm2-vae"
    assert cfg.upscaler.registry_bundle is None
    assert cfg.upscaler.installable is False
    assert cfg.upscaler.sample_rate == 48000
    assert cfg.upscaler.model_path.name == "voxcpm2-vae.gguf"
    assert omni.controls[0].as_capability() == {
        "id": "numSteps",
        "type": "integer",
        "minimum": 8,
        "maximum": 64,
        "step": 4,
        "default": 32,
    }
    assert len(cfg.registry_revision) == 64


def test_shipped_model_catalog_loads():
    cfg = load_config(config_module.REPO_ROOT / "models.toml")
    canonical_bundle_bytes = {
        "omnivoice": 1_633_808_896,
        "chatterbox": 1_003_824_896,
        "parakeet-tdt-0.6b-v3": 417_894_912,
    }
    quantum = 256 * 1024 * 1024
    for model_id, bundle_bytes in canonical_bundle_bytes.items():
        expected_mb = ((bundle_bytes * 2 + quantum - 1) // quantum) * 256
        resources = cfg.models[model_id].resource_requirements()
        assert resources["ram"] == {
            "estimatedBytes": expected_mb * 1024 * 1024,
            "source": "registry",
            "confidence": "declared",
        }
        assert resources["vram"] == resources["ram"]
    parakeet = cfg.models["parakeet-tdt-0.6b-v3"]
    assert parakeet.task == "asr"
    assert parakeet.registry_bundle == "parakeet"
    assert cfg.models["omnivoice"].audio_tags == ["[laughter]"]
    assert cfg.models["chatterbox"].audio_tags == []


def test_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SPEECH_SERVER_TOKEN", raising=False)
    monkeypatch.delenv("SPEECH_SERVER_DEBUG", raising=False)
    path = tmp_path / "models.toml"
    path.write_text(
        "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\n",
        encoding="utf-8",
    )
    cfg = load_config(models_path=path)
    assert cfg.auth_token == ""
    assert cfg.resident_limit == 1
    assert cfg.upscaler is None
    assert cfg.models["m"].installable is False
    assert cfg.debug is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_debug_environment_flag_enables_chunk_diagnostics(
    tmp_path: Path, monkeypatch, value: str
):
    path = tmp_path / "models.toml"
    path.write_text(
        "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SPEECH_SERVER_DEBUG", value)
    assert load_config(path).debug is True


def test_invalid_debug_environment_flag_fails_closed(tmp_path: Path, monkeypatch):
    path = tmp_path / "models.toml"
    path.write_text(
        "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SPEECH_SERVER_DEBUG", "sometimes")
    with pytest.raises(ValueError, match="SPEECH_SERVER_DEBUG"):
        load_config(path)


def test_zero_byte_model_is_not_installed(tmp_path: Path):
    (tmp_path / "m.gguf").touch()
    path = tmp_path / "models.toml"
    path.write_text(
        "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\n",
        encoding="utf-8",
    )
    assert load_config(path).models["m"].installed is False


def test_languages_are_normalized_for_capability_matching(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text(
        "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\n"
        "sample_rate=24000\nlanguages=['EN_us', 'pt-BR']\n",
        encoding="utf-8",
    )
    assert load_config(path).models["m"].languages == ["en-us", "pt-br"]


def test_language_catalog_preserves_explicit_wildcard(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text(
        "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\n"
        "sample_rate=24000\nlanguages=['*']\n",
        encoding="utf-8",
    )
    assert load_config(path).models["m"].languages == ["*"]


def test_runtime_manifest_resolves_an_installed_library(tmp_path: Path):
    library = tmp_path / "lib" / "crispasr.dll"
    library.parent.mkdir()
    library.write_bytes(b"native-runtime")
    manifest = tmp_path / "installed.json"
    manifest.write_text(
        '{"schemaVersion":1,"library":"lib/crispasr.dll"}',
        encoding="utf-8",
    )
    assert _library_from_runtime_manifest(manifest) == library.resolve()


def test_runtime_manifest_rejects_path_traversal(tmp_path: Path):
    outside = tmp_path.parent / "crispasr.dll"
    outside.write_bytes(b"not-this-runtime")
    manifest = tmp_path / "installed.json"
    manifest.write_text(
        '{"schemaVersion":1,"library":"../crispasr.dll"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        _library_from_runtime_manifest(manifest)


def test_model_installation_manifest_rejects_path_traversal(
    monkeypatch, tmp_path: Path
):
    catalog = tmp_path / "models.toml"
    catalog.write_text(
        "[models.m]\nbackend='omnivoice'\nregistry_bundle='m'\n"
        "model='missing.gguf'\nsample_rate=24000\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "runtime" / "installed.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "installations": {
                    "model:m": {
                        "registryBundle": "m",
                        "artifacts": [
                            {
                                "kind": "primary",
                                "filename": "escaped.gguf",
                                "path": "../escaped.gguf",
                                "sha256": "a" * 64,
                                "size": 1,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SPEECH_SERVER_MODEL_MANIFEST", str(manifest))
    with pytest.raises(ValueError, match="escapes"):
        load_config(catalog)


def test_voice_preparation_revision_is_stable_until_new_registry_load(tmp_path: Path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"first")

    def spec():
        return ModelSpec(
            id="omnivoice",
            backend="omnivoice",
            model_path=model_path,
            sample_rate=24000,
            voice_reference=VoiceReferenceSpec(
                "required", "persistent", ("audio", "transcript")
            ),
        )

    running = spec()
    revision = running.voice_preparation_revision
    model_path.write_bytes(b"changed-size")
    assert running.voice_preparation_revision == revision
    assert spec().voice_preparation_revision != revision


def test_resource_estimates_and_revision_are_stable(tmp_path: Path):
    (tmp_path / "main.gguf").write_bytes(b"m" * 100)
    (tmp_path / "codec.gguf").write_bytes(b"c" * 200)
    path = tmp_path / "models.toml"
    path.write_text(
        """
[models.m]
backend = "omnivoice"
model = "main.gguf"
codec = "codec.gguf"
sample_rate = 24000
estimated_ram_mb = 1234
""",
        encoding="utf-8",
    )
    first = load_config(path)
    second = load_config(path)
    resources = first.models["m"].resource_requirements()
    assert resources["componentBytes"] == 300
    assert resources["ram"] == {
        "estimatedBytes": 1234 * 1024 * 1024,
        "source": "registry",
        "confidence": "declared",
    }
    assert resources["vram"]["estimatedBytes"] == 256 * 1024 * 1024
    assert resources["vram"]["source"] == "component-size-heuristic"
    assert first.registry_revision == second.registry_revision


def test_aligner_expands_home_and_participates_in_revision(tmp_path: Path, monkeypatch):
    fake_home = tmp_path / "home"
    cache = fake_home / ".cache" / "crispasr"
    cache.mkdir(parents=True)
    aligner = cache / "aligner.gguf"
    aligner.write_bytes(b"aligner")
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    registry = tmp_path / "models.toml"
    registry.write_text(
        """
[aligner]
id = "canary"
backend = "canary-ctc"
model = "~/.cache/crispasr/aligner.gguf"
languages = ["en"]
n_threads = 3
estimated_ram_mb = 768
estimated_vram_mb = 512
[models.m]
backend = "omnivoice"
model = "m.gguf"
sample_rate = 24000
""",
        encoding="utf-8",
    )
    first = load_config(registry)
    assert first.aligner is not None
    assert first.aligner.model_path == aligner.resolve()
    assert first.aligner.languages == ["en"]
    assert first.aligner.n_threads == 3
    assert first.aligner.resource_requirements()["ram"]["estimatedBytes"] == 768 * 1024 * 1024
    revision = first.registry_revision
    aligner.write_bytes(b"changed-aligner")
    assert load_config(registry).registry_revision != revision


def test_invalid_control_schema_is_rejected(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text(
        """
[models.m]
backend = "omnivoice"
model = "m.gguf"
sample_rate = 24000
[[models.m.controls]]
id = "numSteps"
type = "integer"
minimum = 64
maximum = 8
step = 4
default = 32
""",
        encoding="utf-8",
    )
    try:
        load_config(path)
    except ValueError as exc:
        assert "minimum exceeds maximum" in str(exc)
    else:
        raise AssertionError("invalid control schema was accepted")


def test_backend_specific_control_is_rejected(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text(
        """
[models.chatter]
backend = "chatterbox"
model = "m.gguf"
sample_rate = 24000
[[models.chatter.controls]]
id = "firstSegmentSteps"
type = "integer"
minimum = 1
maximum = 30
step = 1
default = 6
""",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert "unsupported by backend" in str(exc)
    else:
        raise AssertionError("backend-specific control mismatch was accepted")


def test_nonfinite_control_schema_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        ControlSpec("guidanceScale", "number", 0, float("nan"), 0.1, 2)


def test_control_default_must_land_on_advertised_step():
    with pytest.raises(ValueError, match="default must use increments"):
        ControlSpec("guidanceScale", "number", 0, 10, 0.25, 2.1)


@pytest.mark.parametrize(
    "registry, message",
    [
        (
            "[models.m]\ntask='chat'\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\n",
            "unsupported task",
        ),
        (
            "[models.m]\ntask='asr'\nbackend='parakeet'\nmodel='m.gguf'\nsample_rate=16000\ncloning=false\n",
            "requires a transcription table",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=0\n",
            "sample_rate must be positive",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\nlanguages='en'\n",
            "languages must be a list",
        ),
        (
            "resident_limit=0\n[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\n",
            "resident_limit must be positive",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\n[models.m.segmentation]\nmin_seconds=20\ntarget_seconds=8\nmax_seconds=28\n",
            "min_seconds <= target_seconds <= max_seconds",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=true\n",
            "sample_rate must be an integer",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\ncloning='false'\n",
            "cloning must be a boolean",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\ncontrols='bad'\n",
            "controls must be an array of tables",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\n"
            "[models.m.voice_reference]\ntranscript='sometimes'\n",
            "unsupported voice-reference transcript policy",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\nsample_rate=24000\n"
            "[models.m.voice_reference]\ntranscript='unused'\npreparation_inputs=['transcript']\n",
            "cannot use a transcript",
        ),
        (
            "[models.m]\ntask='asr'\nbackend='parakeet'\nmodel='m.gguf'\n"
            "sample_rate=16000\ncloning=false\nlanguages=[]\n"
            "[models.m.transcription]\nsample_rates=[16000]\n",
            "languages must be a list of strings",
        ),
        (
            "[models.m]\ntask='asr'\nbackend='parakeet'\nmodel='m.gguf'\n"
            "sample_rate=16000\ncloning=false\nlanguages=['en']\n"
            "[models.m.transcription]\nsample_rates=[8000]\n",
            "sample_rate must be one of",
        ),
        (
            "[models.m]\ntask='asr'\nbackend='parakeet'\nmodel='m.gguf'\n"
            "sample_rate=16000\ncloning=false\nlanguages=['en']\n"
            "[models.m.transcription]\nsample_rates=[16000]\nmax_bias_terms=1025\n",
            "max_bias_terms must be between",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\n"
            "sample_rate=24000\nlanguages=['en', 'EN']\n",
            "must not contain duplicates",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\n"
            "sample_rate=24000\nlanguages=['not a language']\n",
            "invalid language tags",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\n"
            "sample_rate=24000\nregistry_bundle=''\n",
            "registry_bundle must be a nonempty string",
        ),
        (
            "[models.m]\nbackend='omnivoice'\nmodel='m.gguf'\n"
            "sample_rate=24000\nregistry_bundle='not a backend'\n",
            "registry_bundle contains invalid characters",
        ),
    ],
)
def test_invalid_model_basics_are_rejected(tmp_path: Path, registry: str, message: str):
    path = tmp_path / "models.toml"
    path.write_text(registry, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)

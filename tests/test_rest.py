import base64
import hashlib
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from speech_server.app import create_app
from speech_server.config import (
    ASRSpec,
    AlignerSpec,
    ControlSpec,
    ModelSpec,
    ParalinguisticTagSpec,
    ServerConfig,
    VoiceReferenceSpec,
)
from speech_server.protocol import TranscribeRequest
from tests.fakes import FakeSession


def _wav_b64():
    data = b"\x00\x00\x00\x00"
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
    )
    return base64.b64encode(header + data).decode()


def test_transcription_language_accepts_normalized_bcp47_shape():
    request = TranscribeRequest(
        requestId="r",
        model="m",
        audioData="AAAA",
        language="zh_Hant_TW",
    )
    assert request.language == "zh-hant-tw"


def _client(tmp_path, token="", with_upscaler=False, with_aligner=False, with_asr=False):
    model_path = tmp_path / "x.gguf"
    model_path.write_bytes(b"model")
    spec = ModelSpec(
        id="omnivoice",
        backend="omnivoice",
        model_path=model_path,
        sample_rate=24000,
        languages=["en", "de"],
        controls=[
            ControlSpec("numSteps", "integer", 8, 64, 4, 32),
            ControlSpec("firstSegmentSteps", "integer", 8, 64, 4, 32),
            ControlSpec("guidanceScale", "number", 0, 10, 0.1, 2),
        ],
        paralinguistic_tags=[
            ParalinguisticTagSpec(
                "[laughter]",
                "Inserts natural laughter.",
                ("[laugh]", "[laughs]", "[laughing]"),
            )
        ],
        voice_reference=VoiceReferenceSpec(
            "required", "persistent", ("audio", "transcript")
        ),
    )
    upscaler_path = tmp_path / "voxcpm2-vae.gguf"
    if with_upscaler:
        upscaler_path.write_bytes(b"voxcpm2-vae")
    upscaler = (
        ModelSpec(
            id="voxcpm2-vae",
            backend="voxcpm2-vae",
            model_path=upscaler_path,
            sample_rate=48000,
            task="audio-to-audio",
            cloning=False,
        )
        if with_upscaler else None
    )
    aligner = None
    if with_aligner:
        aligner_path = tmp_path / "aligner.gguf"
        aligner_path.write_bytes(b"aligner")
        aligner = AlignerSpec("canary", "canary-ctc", aligner_path, ["en"])
    models = {"omnivoice": spec}
    if with_asr:
        model_path = tmp_path / "parakeet.gguf"
        model_path.write_bytes(b"asr-model")
        models["parakeet-tdt-0.6b-v3"] = ModelSpec(
            id="parakeet-tdt-0.6b-v3",
            backend="parakeet",
            model_path=model_path,
            sample_rate=16000,
            task="asr",
            cloning=False,
            languages=["en", "de"],
            asr=ASRSpec(
                automatic_language_detection=True,
                timestamps=("none", "segment", "word"),
                bias_terms=True,
            ),
        )
    cfg = ServerConfig(
        models=models,
        upscaler=upscaler,
        aligner=aligner,
        auth_token=token,
        voice_dir=tmp_path,
        resident_limit=2 if with_asr else 1,
    )
    return TestClient(create_app(cfg, session_factory=lambda cfg, model: FakeSession()))


def test_health_and_capabilities(tmp_path):
    client = _client(tmp_path)
    assert client.get("/v2/health").status_code == 200
    capabilities = client.get("/v2/capabilities").json()
    assert capabilities["protocolVersion"] == "2.0"
    assert capabilities["capabilitiesVersion"] == 8
    assert capabilities["loadPlanning"] is True
    model = capabilities["models"][0]
    assert {key: model[key] for key in (
        "id", "task", "backend", "sampleRate", "streaming", "cloning",
        "languages", "upscaling",
    )} == {
        "id": "omnivoice",
        "task": "tts",
        "backend": "omnivoice",
        "sampleRate": 24000,
        "streaming": False,
        "cloning": True,
        "languages": ["en", "de"],
        "upscaling": False,
    }
    assert [control["id"] for control in model["controls"]] == [
        "numSteps", "firstSegmentSteps", "guidanceScale"
    ]
    assert model["resources"]["componentBytes"] == 5
    assert model["available"] is True
    assert model["installed"] is True
    assert model["installable"] is False
    assert model["installation"] == {
        "installed": True,
        "installable": False,
        "registryBundle": None,
        "job": None,
    }
    assert model["voiceReference"]["transcript"] == "required"
    assert model["voiceReference"]["preparation"]["mode"] == "persistent"
    assert model["voiceReference"]["preparation"]["inputs"] == [
        "audio",
        "transcript",
    ]
    assert model["paralinguisticTags"] == [
        {
            "token": "[laughter]",
            "aliases": ["[laugh]", "[laughs]", "[laughing]"],
            "description": "Inserts natural laughter.",
        }
    ]
    assert model["audioTags"] == ["[laughter]"]
    assert model["tagAliases"]["[laugh]"] == "[laughter]"
    assert capabilities["upscaler"] is None


def test_missing_catalog_model_remains_visible_as_installable(tmp_path):
    spec = ModelSpec(
        id="omnivoice",
        backend="omnivoice",
        model_path=tmp_path / "missing.gguf",
        sample_rate=24000,
        languages=["en"],
        registry_bundle="omnivoice",
    )
    cfg = ServerConfig(models={"omnivoice": spec}, voice_dir=tmp_path)
    client = TestClient(
        create_app(cfg, session_factory=lambda cfg, model: FakeSession())
    )

    model = client.get("/v2/capabilities").json()["models"][0]
    assert model["available"] is False
    assert model["installed"] is False
    assert model["installable"] is True
    assert model["installation"]["registryBundle"] == "omnivoice"


def test_install_endpoint_rejects_a_catalog_model_without_a_bundle(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/v2/models/omnivoice:install", json={"acceptLicense": False}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "component_not_installable"


def test_installation_job_lookup_returns_404(tmp_path):
    client = _client(tmp_path)
    response = client.get("/v2/installations/not-a-job")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


def test_install_endpoint_starts_and_reports_a_job(tmp_path):
    spec = ModelSpec(
        id="omnivoice",
        backend="omnivoice",
        model_path=tmp_path / "missing.gguf",
        sample_rate=24000,
        registry_bundle="omnivoice",
    )
    cfg = ServerConfig(models={"omnivoice": spec}, voice_dir=tmp_path)

    class Installer:
        job = {
            "id": "job-1",
            "component": "model:omnivoice",
            "state": "downloading",
        }

        def component_state(self, component):
            return {
                "installed": False,
                "installable": True,
                "registryBundle": "omnivoice",
                "job": self.job,
            }

        def start(self, component, *, accept_license=False):
            assert component == "model:omnivoice"
            assert accept_license is True
            return self.job

        def plan(self, component):
            return {
                "component": component,
                "registryBundle": "omnivoice",
                "totalBytes": 123,
                "artifacts": [],
            }

        def get(self, job_id):
            assert job_id == "job-1"
            return self.job

        def list(self):
            return [self.job]

        def cancel(self, job_id):
            return {**self.job, "state": "cancelled"}

    client = TestClient(
        create_app(
            cfg,
            session_factory=lambda cfg, model: FakeSession(),
            installation_manager=Installer(),
        )
    )
    started = client.post(
        "/v2/models/omnivoice:install", json={"acceptLicense": True}
    )
    assert started.status_code == 202
    assert started.json()["id"] == "job-1"
    assert client.get("/v2/models/omnivoice:install-plan").json()["totalBytes"] == 123
    assert client.get("/v2/installations/job-1").json()["state"] == "downloading"
    assert client.delete("/v2/installations/job-1").json()["state"] == "cancelled"


def test_asr_capability_and_transcription(tmp_path):
    client = _client(tmp_path, with_asr=True)
    capabilities = client.get("/v2/capabilities").json()
    model = next(item for item in capabilities["models"] if item["task"] == "asr")
    assert model["available"] is True
    assert model["voiceReference"] is None
    assert model["transcription"]["audio"]["sampleRates"] == [16000]
    assert model["transcription"]["biasTerms"]["supported"] is True
    assert model["transcription"]["biasTerms"]["maxCount"] == 256
    assert model["transcription"]["biasTerms"]["maxLength"] == 128

    pcm = (np.zeros(1600, dtype="<i2")).tobytes()
    response = client.post(
        "/v2/transcribe",
        json={
            "requestId": "r1",
            "model": "parakeet-tdt-0.6b-v3",
            "audioData": base64.b64encode(pcm).decode(),
            "audio": {"encoding": "pcm_s16le", "sampleRate": 16000, "channels": 1},
            "language": "auto",
            "biasTerms": ["Hogwarts", "Hogwarts"],
            "timestamps": "word",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["text"] == "Hello Hogwarts."
    assert result["confidence"] is None
    assert result["detectedLanguage"] == "en"
    assert [word["text"] for word in result["words"]] == ["Hello", "Hogwarts."]
    assert result["throughputX"] > 0


def test_transcription_validation_and_task_guards(tmp_path):
    client = _client(tmp_path, with_asr=True)
    wrong_task = client.post(
        "/v2/transcribe",
        json={
            "requestId": "r1",
            "model": "omnivoice",
            "audioData": "AAAA",
        },
    )
    assert wrong_task.status_code == 400
    assert wrong_task.json()["error"]["code"] == "wrong_task"
    malformed = client.post(
        "/v2/transcribe",
        json={
            "requestId": "r2",
            "model": "parakeet-tdt-0.6b-v3",
            "audioData": "not-base64",
        },
    )
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "unsupported_audio"


def test_transcription_returns_specific_contract_errors(tmp_path):
    client = _client(tmp_path, with_asr=True)
    pcm = base64.b64encode(b"\0\0" * 16).decode()
    base = {
        "requestId": "r",
        "model": "parakeet-tdt-0.6b-v3",
        "audioData": pcm,
    }
    unsupported_audio = client.post(
        "/v2/transcribe",
        json={**base, "audio": {"encoding": "wav", "sampleRate": 16000, "channels": 1}},
    )
    assert unsupported_audio.status_code == 400
    assert unsupported_audio.json()["error"]["code"] == "unsupported_audio"

    unsupported_timestamps = client.post(
        "/v2/transcribe", json={**base, "timestamps": "phoneme"}
    )
    assert unsupported_timestamps.status_code == 400
    assert unsupported_timestamps.json()["error"]["code"] == "unsupported_timestamps"

    non_object = client.post("/v2/transcribe", json=["not", "an", "object"])
    assert non_object.status_code == 400
    assert non_object.json()["error"]["code"] == "bad_request"


def test_transcription_sanitizes_bias_terms(tmp_path):
    client = _client(tmp_path, with_asr=True)
    pcm = base64.b64encode(b"\0\0" * 16).decode()
    body = {
        "requestId": "r",
        "model": "parakeet-tdt-0.6b-v3",
        "audioData": pcm,
        "biasTerms": ["Hogwarts", " hogwarts "],
    }
    response = client.post("/v2/transcribe", json=body)
    assert response.status_code == 200
    session = client.app.state.pool.acquire("parakeet-tdt-0.6b-v3").raw
    assert ("hotwords", "Hogwarts", 2.0) in session.calls

    body["biasTerms"] = ["Hogwarts,Expelliarmus"]
    unsafe = client.post("/v2/transcribe", json=body)
    assert unsafe.status_code == 400
    assert unsafe.json()["error"]["code"] == "bad_request"


def test_combined_stack_plan_and_warmup(tmp_path):
    client = _client(tmp_path, with_asr=True)
    body = {
        "ttsModel": "omnivoice",
        "asrModel": "parakeet-tdt-0.6b-v3",
        "upscale": False,
        "alignment": False,
    }
    plan = client.post("/v2/stack:plan", json=body)
    assert plan.status_code == 200
    assert plan.json()["residentCapacitySatisfied"] is True
    assert {item["task"] for item in plan.json()["load"]} == {"tts", "asr"}
    warmup = client.post("/v2/stack:warmup", json=body)
    assert warmup.status_code == 200
    assert warmup.json()["success"] is True
    assert set(warmup.json()["resources"]["loadedModelIds"]) == {
        "omnivoice", "parakeet-tdt-0.6b-v3"
    }


def test_legacy_tts_load_routes_reject_asr_and_missing_aligner(tmp_path):
    client = _client(tmp_path, with_asr=True)
    plan = client.post("/v2/models/parakeet-tdt-0.6b-v3:plan", json={})
    warmup = client.post("/v2/models/parakeet-tdt-0.6b-v3:warmup")
    assert plan.status_code == warmup.status_code == 400
    assert plan.json()["error"]["code"] == "wrong_task"
    assert warmup.json()["error"]["code"] == "wrong_task"

    missing_aligner = client.post(
        "/v2/stack:plan",
        json={"ttsModel": "omnivoice", "alignment": True},
    )
    assert missing_aligner.status_code == 409
    assert missing_aligner.json()["error"]["code"] == "aligner_unavailable"


def test_voice_crud(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/v2/voices",
        json={
            "displayName": "Seb",
            "langCode": "EN_US",
            "audioData": _wav_b64(),
            "refText": "hi",
            "referenceHash": "ff00",
        },
    )
    assert response.status_code == 200
    assert response.json()["voiceId"] == "default__Seb"
    assert client.get("/v2/voices").json()["voices"][0]["referenceHash"] == "ff00"
    assert client.delete("/v2/voices/default__Seb").status_code == 200
    assert client.get("/v2/voices").json()["voices"] == []


def test_clone_rejects_garbage(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/v2/voices",
        json={
            "displayName": "X",
            "audioData": base64.b64encode(b"junkjunkjunkjunk").decode(),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


def test_auth_enforced(tmp_path):
    client = _client(tmp_path, token="tok")
    assert client.get("/v2/voices").status_code == 401
    assert (
        client.get("/v2/voices", headers={"Authorization": "Basic tok"}).status_code
        == 200
    )
    assert client.get("/v2/health").status_code == 200


def test_warmup(tmp_path):
    client = _client(tmp_path)
    assert client.post("/v2/models/omnivoice:warmup").status_code == 200
    assert client.post("/v2/models/nope:warmup").status_code == 404


def test_warmup_can_explicitly_load_upscaler(tmp_path):
    client = _client(tmp_path, with_upscaler=True)
    response = client.post("/v2/models/omnivoice:warmup?upscale=true")
    assert response.status_code == 200
    assert response.json() == {
        "loaded": "omnivoice", "upscalerLoaded": True, "alignerLoaded": False
    }


def test_warmup_can_load_aligner_from_transcribed_reference(tmp_path, monkeypatch):
    client = _client(tmp_path, with_aligner=True)
    missing = client.post("/v2/models/omnivoice:warmup?alignment=true")
    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "alignment_reference_required"

    client.post(
        "/v2/voices",
        json={"displayName": "Seb", "audioData": _wav_b64(), "refText": "[laugh] Hello."},
    )
    seen = {}
    monkeypatch.setattr(
        "speech_server.alignment.load_audio_mono",
        lambda *args, **kwargs: np.zeros(1600, dtype=np.float32),
    )

    def align(_model, transcript, _pcm, **_kwargs):
        seen["transcript"] = transcript
        return [SimpleNamespace(text="Hello", start=0.0, end=0.05)]

    client.app.state.aligner._align_func = align
    response = client.post("/v2/models/omnivoice:warmup?alignment=true")
    assert response.status_code == 200
    assert response.json()["alignerLoaded"] is True
    assert seen["transcript"] == "Hello."


def test_load_plan_and_component_residency(tmp_path):
    client = _client(tmp_path, with_upscaler=True)
    plan = client.post(
        "/v2/models/omnivoice:plan",
        json={"upscale": True, "adaptiveBatching": False},
    )
    assert plan.status_code == 200
    assert plan.json()["load"] == [
        {"kind": "model", "id": "omnivoice"},
        {"kind": "upscaler", "id": "voxcpm2-vae"},
    ]
    client.post("/v2/models/omnivoice:warmup?upscale=true")
    resources = client.get("/v2/resources").json()
    residency = [
        (item["kind"], item["id"], item["sticky"])
        for item in resources["components"]
    ]
    assert residency == [
        ("model", "omnivoice", False),
        ("upscaler", "voxcpm2-vae", True),
    ]
    loaded_plan = client.post(
        "/v2/models/omnivoice:plan",
        json={"upscale": True, "adaptiveBatching": False},
    ).json()
    assert loaded_plan["load"] == []
    assert loaded_plan["fit"]["status"] == "comfortable"


def test_load_plan_rejects_coerced_booleans_and_unknown_models(tmp_path):
    client = _client(tmp_path)
    coerced = client.post(
        "/v2/models/omnivoice:plan",
        json={"upscale": "false", "adaptiveBatching": False},
    )
    assert coerced.status_code == 422
    missing = client.post(
        "/v2/models/missing:plan",
        json={"upscale": False, "adaptiveBatching": False},
    )
    assert missing.status_code == 404


def test_prepare_voice_requires_transcript_and_records_model_marker(tmp_path):
    client = _client(tmp_path)
    clone = client.post(
        "/v2/voices",
        json={"displayName": "Seb", "audioData": _wav_b64()},
    ).json()
    missing = client.post(
        f"/v2/models/omnivoice/voices/{clone['voiceId']}:prepare"
    )
    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "voice_transcript_required"

    clone = client.post(
        "/v2/voices",
        json={"displayName": "Seb", "audioData": _wav_b64(), "refText": "Hello."},
    ).json()
    prepared = client.post(
        f"/v2/models/omnivoice/voices/{clone['voiceId']}:prepare"
    )
    assert prepared.status_code == 200
    marker = prepared.json()["preparation"]
    listed = client.get("/v2/voices").json()["voices"][0]
    assert listed["transcriptHash"]
    assert listed["preparedModels"]["omnivoice"] == marker

    (tmp_path / "default__Seb.txt").write_text("Changed.", encoding="utf-8")
    changed = client.get("/v2/voices").json()["voices"][0]
    assert changed["transcriptHash"] == hashlib.sha256(b"Changed.").hexdigest()
    assert (
        changed["preparedModels"]["omnivoice"]["inputHashes"]["transcript"]
        != changed["transcriptHash"]
    )


def test_resources_is_authenticated_and_tolerates_unavailable_telemetry(
    tmp_path, monkeypatch
):
    client = _client(tmp_path, token="tok")
    monkeypatch.setattr(
        "speech_server.app.sample_resources",
        lambda pool, aligner=None: {
            "ram": None,
            "processRamBytes": None,
            "gpus": [],
            "loadedModelIds": [],
            "upscalerLoaded": False,
        },
    )
    assert client.get("/v2/resources").status_code == 401
    response = client.get(
        "/v2/resources", headers={"Authorization": "Basic tok"}
    )
    assert response.status_code == 200
    assert response.json()["gpus"] == []

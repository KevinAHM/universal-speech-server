"""FastAPI application exposing REST management and WebSocket synthesis."""

import asyncio
import base64
import binascii
import hmac
import json
import logging
import queue
import time
from contextlib import suppress
from typing import Annotated, Callable, Optional

import numpy as np
from fastapi import Body, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import __version__
from .alignment import AlignmentEngine, AlignmentError
from .config import ServerConfig
from .engine import EngineBusyError, EngineError, SessionPool
from .model_install import ModelInstallError, ModelInstallationManager
from .protocol import StartRequest, TranscribeRequest, error_event
from .resources import build_load_plan, build_stack_load_plan, sample_resources
from .synthesis import run_request
from .textproc import normalize_unit
from .voices import MAX_SAMPLE_BYTES, VoiceError, VoiceStore

logger = logging.getLogger(__name__)


class CloneBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    displayName: str = Field(min_length=1, max_length=100)
    langCode: str = Field(default="EN_US", min_length=1, max_length=32)
    audioData: str = Field(
        min_length=16, max_length=(MAX_SAMPLE_BYTES * 4 // 3) + 1024
    )
    refText: Optional[str] = Field(default=None, max_length=32768)
    referenceHash: Optional[str] = Field(default=None, max_length=128)
    tags: Optional[
        list[Annotated[str, Field(min_length=1, max_length=64)]]
    ] = Field(default=None, max_length=32)


class LoadPlanBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upscale: bool = False
    adaptiveBatching: bool = False

    @field_validator("upscale", "adaptiveBatching", mode="before")
    @classmethod
    def validate_boolean_type(cls, value):
        if not isinstance(value, bool):
            raise ValueError("load-plan options must be booleans")
        return value


class StackLoadBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttsModel: Optional[str] = Field(default=None, min_length=1, max_length=128)
    asrModel: Optional[str] = Field(default=None, min_length=1, max_length=128)
    upscale: bool = False
    alignment: bool = False

    @field_validator("upscale", "alignment", mode="before")
    @classmethod
    def validate_boolean_type(cls, value):
        if not isinstance(value, bool):
            raise ValueError("stack options must be booleans")
        return value


class InstallBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acceptLicense: bool = False

    @field_validator("acceptLicense", mode="before")
    @classmethod
    def validate_boolean_type(cls, value):
        if not isinstance(value, bool):
            raise ValueError("acceptLicense must be a boolean")
        return value


def _err(status: int, code: str, message: str, details=None) -> JSONResponse:
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return JSONResponse(status_code=status, content={"error": error})


def create_app(
    cfg: ServerConfig,
    session_factory: Optional[Callable] = None,
    installation_manager: ModelInstallationManager | None = None,
) -> FastAPI:
    app = FastAPI(title="Universal Speech Server")
    pool = SessionPool(cfg, session_factory=session_factory)
    aligner = AlignmentEngine(cfg)
    voices = VoiceStore(cfg.voice_dir)
    installer = installation_manager or ModelInstallationManager(cfg)
    app.state.pool = pool
    app.state.voices = voices
    app.state.cfg = cfg
    app.state.aligner = aligner
    app.state.installer = installer

    def _component_available(spec) -> bool:
        return spec is not None and spec.installed

    def _authorized(auth_header: str) -> bool:
        if not cfg.auth_token:
            return True
        return hmac.compare_digest(auth_header or "", f"Basic {cfg.auth_token}")

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path != "/v2/health" and not _authorized(
            request.headers.get("authorization", "")
        ):
            return _err(401, "unauthorized", "missing or invalid token")
        return await call_next(request)

    @app.get("/v2/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "loadedModels": pool.loaded_ids(),
            "upscalerLoaded": pool.upscaler_loaded(),
            "alignerLoaded": aligner.loaded,
        }

    @app.get("/v2/capabilities")
    def capabilities():
        return {
            "protocolVersion": "2.0",
            "capabilitiesVersion": 8,
            "registryRevision": cfg.registry_revision,
            "residentLimit": cfg.resident_limit,
            "loadPlanning": True,
            "models": [
                {
                    "id": model.id,
                    "task": model.task,
                    "backend": model.backend,
                    "sampleRate": model.sample_rate,
                    "available": _component_available(model),
                    "installed": model.installed,
                    "installable": model.installable,
                    "installation": installer.component_state(f"model:{model.id}"),
                    "streaming": False,
                    "cloning": model.cloning,
                    "languages": model.languages,
                    "upscaling": model.task == "tts" and _component_available(cfg.upscaler),
                    "controls": [
                        control.as_capability() for control in model.controls
                    ],
                    "segmentation": (
                        model.segmentation.as_capability()
                        if model.segmentation
                        else None
                    ),
                    "textProfile": model.text_profile,
                    "paralinguisticTags": [
                        tag.as_capability() for tag in model.paralinguistic_tags
                    ],
                    # Compatibility fields for v7 clients. The structured field
                    # above is the authoritative contract from capabilities v8.
                    "audioTags": model.audio_tags,
                    "tagAliases": model.tag_aliases,
                    "resources": model.resource_requirements(),
                    "voiceReference": (
                        model.voice_reference_capability()
                        if model.task == "tts"
                        else None
                    ),
                    "transcription": model.asr.as_capability() if model.asr else None,
                }
                for model in cfg.models.values()
            ],
            "upscaler": (
                {
                    "id": cfg.upscaler.id,
                    "backend": cfg.upscaler.backend,
                    "sampleRate": cfg.upscaler.sample_rate,
                    "available": _component_available(cfg.upscaler),
                    "installed": cfg.upscaler.installed,
                    "installable": cfg.upscaler.installable,
                    "installation": installer.component_state("upscaler"),
                    "resources": cfg.upscaler.resource_requirements(),
                }
                if cfg.upscaler
                else None
            ),
            "alignment": (
                {
                    "id": cfg.aligner.id,
                    "backend": cfg.aligner.backend,
                    "available": aligner.available,
                    "installed": cfg.aligner.installed,
                    "installable": cfg.aligner.installable,
                    "installation": installer.component_state("aligner"),
                    "languages": cfg.aligner.languages,
                    "timingModes": ["auto", "word"],
                    "resources": cfg.aligner.resource_requirements(),
                }
                if cfg.aligner
                else None
            ),
        }

    def _install_response(component: str, body: InstallBody):
        try:
            return JSONResponse(
                status_code=202,
                content=installer.start(
                    component, accept_license=body.acceptLicense
                ),
            )
        except ModelInstallError as exc:
            status = 404 if exc.code == "component_not_found" else 409
            return _err(status, exc.code, str(exc))

    def _install_plan_response(component: str):
        try:
            return installer.plan(component)
        except ModelInstallError as exc:
            status = 404 if exc.code == "component_not_found" else 409
            return _err(status, exc.code, str(exc))

    @app.get("/v2/models/{model_id}:install-plan")
    def model_install_plan(model_id: str):
        return _install_plan_response(f"model:{model_id}")

    @app.get("/v2/upscaler:install-plan")
    def upscaler_install_plan():
        return _install_plan_response("upscaler")

    @app.get("/v2/alignment:install-plan")
    def aligner_install_plan():
        return _install_plan_response("aligner")

    @app.post("/v2/models/{model_id}:install")
    def install_model(model_id: str, body: InstallBody):
        return _install_response(f"model:{model_id}", body)

    @app.post("/v2/upscaler:install")
    def install_upscaler(body: InstallBody):
        return _install_response("upscaler", body)

    @app.post("/v2/alignment:install")
    def install_aligner(body: InstallBody):
        return _install_response("aligner", body)

    @app.get("/v2/installations")
    def installations():
        return {"jobs": installer.list()}

    @app.get("/v2/installations/{job_id}")
    def installation(job_id: str):
        try:
            return installer.get(job_id)
        except ModelInstallError as exc:
            return _err(404, exc.code, str(exc))

    @app.delete("/v2/installations/{job_id}")
    def cancel_installation(job_id: str):
        try:
            return installer.cancel(job_id)
        except ModelInstallError as exc:
            return _err(404, exc.code, str(exc))

    @app.get("/v2/resources")
    def resources():
        return sample_resources(pool, aligner)

    @app.post("/v2/models/{model_id}:plan")
    def load_plan(model_id: str, body: LoadPlanBody):
        model = cfg.models.get(model_id)
        if model is None:
            return _err(404, "model_not_found", model_id)
        if model.task != "tts":
            return _err(400, "wrong_task", f"model {model_id!r} is not a TTS model")
        if not _component_available(model):
            return _err(409, "model_unavailable", f"model {model_id!r} is not installed")
        if body.upscale and not _component_available(cfg.upscaler):
            return _err(409, "upscaler_unavailable", "the configured upscaler is not installed")
        if body.adaptiveBatching and not aligner.available:
            return _err(409, "aligner_unavailable", "the configured aligner is not installed")
        try:
            return build_load_plan(
                cfg,
                pool,
                aligner,
                sample_resources(pool, aligner),
                model_id=model_id,
                upscale=body.upscale,
                adaptive_batching=body.adaptiveBatching,
            )
        except ValueError as exc:
            return _err(400, "invalid_load_plan", str(exc))

    def _stack_models(
        body: StackLoadBody,
    ) -> tuple[list[str], tuple[int, str, str] | None]:
        requested = []
        for task, model_id in (("tts", body.ttsModel), ("asr", body.asrModel)):
            if not model_id:
                continue
            model = cfg.models.get(model_id)
            if model is None:
                return [], (404, "model_not_found", f"model {model_id!r} does not exist")
            if model.task != task:
                return [], (400, "wrong_task", f"model {model_id!r} is not a {task.upper()} model")
            if not _component_available(model):
                return [], (409, "model_unavailable", f"model {model_id!r} is not installed")
            requested.append(model_id)
        if not requested:
            return [], (400, "invalid_load_plan", "at least one TTS or ASR model is required")
        if body.upscale and not body.ttsModel:
            return [], (400, "invalid_load_plan", "upscaling requires a selected TTS model")
        if body.alignment and not body.ttsModel:
            return [], (400, "invalid_load_plan", "alignment requires a selected TTS model")
        if body.upscale and not _component_available(cfg.upscaler):
            return [], (409, "upscaler_unavailable", "the configured upscaler is not installed")
        if body.alignment and not aligner.available:
            return [], (409, "aligner_unavailable", "the configured aligner is not installed")
        return requested, None

    @app.post("/v2/stack:plan")
    def stack_plan(body: StackLoadBody):
        model_ids, error = _stack_models(body)
        if error:
            return _err(*error)
        try:
            return build_stack_load_plan(
                cfg,
                pool,
                aligner,
                sample_resources(pool, aligner),
                model_ids=model_ids,
                upscale=body.upscale,
                alignment=body.alignment,
            )
        except ValueError as exc:
            return _err(400, "invalid_load_plan", str(exc))

    @app.get("/v2/voices")
    def list_voices():
        return {"voices": voices.list()}

    @app.post("/v2/voices")
    def clone_voice(body: CloneBody):
        try:
            return voices.clone(
                display_name=body.displayName,
                lang_code=body.langCode,
                audio_b64=body.audioData,
                ref_text=body.refText,
                reference_hash=body.referenceHash,
                tags=body.tags,
            )
        except (VoiceError, ValueError) as exc:
            return _err(400, "bad_request", str(exc))

    @app.delete("/v2/voices/{voice_id}")
    def delete_voice(voice_id: str):
        try:
            deleted = voices.delete(voice_id)
        except VoiceError as exc:
            return _err(400, "bad_request", str(exc))
        if not deleted:
            return _err(404, "voice_not_found", voice_id)
        return {"deleted": voice_id}

    @app.post("/v2/models/{model_id}:warmup")
    def warmup(model_id: str, upscale: bool = False, alignment: bool = False):
        try:
            model = pool.spec(model_id)
        except KeyError:
            return _err(404, "model_not_found", model_id)
        if model.task != "tts":
            return _err(400, "wrong_task", f"model {model_id!r} is not a TTS model")
        if not _component_available(model):
            return _err(409, "model_unavailable", f"model {model_id!r} is not installed")
        if upscale and not _component_available(cfg.upscaler):
            return _err(409, "upscaler_unavailable", "the configured upscaler is not installed")
        if alignment and not aligner.available:
            return _err(409, "aligner_unavailable", "the configured aligner is not installed")

        alignment_reference = None
        if alignment and not aligner.loaded:
            if not aligner.available:
                return _err(
                    409,
                    "aligner_unavailable",
                    "forced alignment is not configured or its model is missing",
                )
            for voice in voices.list():
                if not voice.get("hasTranscript"):
                    continue
                try:
                    voice_wav, transcript = voices.resolve(voice["voiceId"])
                except (KeyError, VoiceError):
                    continue
                alignment_text = normalize_unit(
                    transcript or "",
                    model.text_profile or "plain",
                    model.paralinguistic_tag_map,
                ).alignment_text
                if alignment_text:
                    alignment_reference = (voice_wav, alignment_text)
                    break
            if alignment_reference is None:
                return _err(
                    409,
                    "alignment_reference_required",
                    "CTC warmup requires an uploaded voice reference with a transcript",
                )
        try:
            pool.acquire(model_id)
        except EngineBusyError as exc:
            return _err(
                409,
                "model_busy",
                str(exc),
                {"modelIds": exc.model_ids},
            )
        except EngineError as exc:
            return _err(500, "model_load_failed", str(exc))
        if upscale:
            try:
                pool.acquire_upscaler()
            except EngineError as exc:
                return _err(400, "upscaler_unavailable", str(exc))
        if alignment_reference is not None:
            try:
                aligner.warmup_reference(*alignment_reference)
            except AlignmentError as exc:
                return _err(500, "aligner_load_failed", str(exc))
        return {
            "loaded": model_id,
            "upscalerLoaded": pool.upscaler_loaded(),
            "alignerLoaded": aligner.loaded,
        }

    def _alignment_reference(model):
        if not aligner.available:
            return None
        for voice in voices.list():
            if not voice.get("hasTranscript"):
                continue
            try:
                voice_wav, transcript = voices.resolve(voice["voiceId"])
            except (KeyError, VoiceError):
                continue
            alignment_text = normalize_unit(
                transcript or "",
                model.text_profile or "plain",
                model.paralinguistic_tag_map,
            ).alignment_text
            if alignment_text:
                return voice_wav, alignment_text
        return None

    @app.post("/v2/stack:warmup")
    def stack_warmup(body: StackLoadBody):
        model_ids, error = _stack_models(body)
        if error:
            return _err(*error)
        try:
            plan = build_stack_load_plan(
                cfg,
                pool,
                aligner,
                sample_resources(pool, aligner),
                model_ids=model_ids,
                upscale=body.upscale,
                alignment=body.alignment,
            )
        except ValueError as exc:
            return _err(400, "invalid_load_plan", str(exc))
        if not plan["residentCapacitySatisfied"]:
            return _err(
                409,
                "resident_limit",
                f"the desired stack needs {len(model_ids)} resident model slots, "
                f"but the server allows {cfg.resident_limit}",
            )
        if plan["fit"]["status"] == "insufficient":
            return _err(
                409,
                "insufficient_resources",
                "the desired speech stack does not fit in the sampled server resources",
                plan["requirements"],
            )
        if plan["busy"]:
            return _err(409, "model_busy", "obsolete models are still active", plan["busy"])

        alignment_reference = None
        if body.alignment and not aligner.loaded:
            tts_model = cfg.models[body.ttsModel]
            alignment_reference = _alignment_reference(tts_model)
            if alignment_reference is None:
                return _err(
                    409,
                    "alignment_reference_required",
                    "CTC warmup requires an uploaded voice reference with a transcript",
                )

        results = []
        try:
            pool.evict_except(model_ids)
        except EngineBusyError as exc:
            return _err(409, "model_busy", str(exc), {"modelIds": exc.model_ids})
        for model_id in model_ids:
            try:
                pool.acquire(model_id)
                results.append({"kind": "model", "id": model_id, "loaded": True})
            except (EngineBusyError, EngineError) as exc:
                results.append(
                    {"kind": "model", "id": model_id, "loaded": False, "error": str(exc)}
                )
        if body.upscale:
            try:
                pool.acquire_upscaler()
                results.append({"kind": "upscaler", "id": cfg.upscaler.id, "loaded": True})
            except EngineError as exc:
                results.append(
                    {"kind": "upscaler", "id": cfg.upscaler.id, "loaded": False, "error": str(exc)}
                )
        if body.alignment:
            try:
                if alignment_reference is not None:
                    aligner.warmup_reference(*alignment_reference)
                results.append({"kind": "aligner", "id": cfg.aligner.id, "loaded": aligner.loaded})
            except AlignmentError as exc:
                results.append(
                    {"kind": "aligner", "id": cfg.aligner.id, "loaded": False, "error": str(exc)}
                )
        return {
            "success": all(result.get("loaded") for result in results),
            "results": results,
            "resources": sample_resources(pool, aligner),
        }

    @app.post("/v2/transcribe")
    def transcribe(raw_body: Annotated[object, Body()]):
        try:
            body = TranscribeRequest.model_validate(raw_body)
        except ValidationError as exc:
            details = [
                {"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
                for error in exc.errors()
            ]
            return _err(400, "bad_request", "invalid transcription request", details)
        model = cfg.models.get(body.model)
        if model is None:
            return _err(404, "model_not_found", body.model)
        if model.task != "asr" or model.asr is None:
            return _err(400, "wrong_task", f"model {body.model!r} is not an ASR model")
        if not _component_available(model):
            return _err(409, "model_unavailable", f"model {body.model!r} is not installed")
        spec = model.asr
        if body.audio.encoding != spec.encoding:
            return _err(400, "unsupported_audio", "unsupported audio encoding")
        if body.audio.channels != spec.channels:
            return _err(400, "unsupported_audio", "only mono input is accepted")
        if body.audio.sampleRate not in spec.sample_rates:
            return _err(400, "unsupported_audio", "unsupported input sample rate")
        if body.timestamps not in spec.timestamps:
            return _err(400, "unsupported_timestamps", "timestamp mode is unavailable")
        language = body.language.split("-", 1)[0]
        if language == "auto":
            if not spec.automatic_language_detection:
                return _err(400, "unsupported_language", "automatic language detection is unavailable")
            language_hint = None
        elif language not in {
            item.lower().replace("_", "-").split("-", 1)[0]
            for item in model.languages
        } and "*" not in model.languages:
            return _err(400, "unsupported_language", body.language)
        else:
            language_hint = language
        if any(
            "," in term or any(ord(character) < 32 for character in term)
            for term in body.biasTerms
        ):
            return _err(
                400,
                "bad_request",
                "bias terms cannot contain commas or control characters",
            )
        bias_terms = []
        seen_bias_terms = set()
        for raw_term in body.biasTerms:
            term = raw_term.strip()
            key = term.casefold()
            if key not in seen_bias_terms:
                seen_bias_terms.add(key)
                bias_terms.append(term)
        if bias_terms and not spec.bias_terms:
            return _err(400, "unsupported_bias_terms", "this model does not support bias terms")
        if len(bias_terms) > spec.max_bias_terms or any(
            len(term) > spec.max_bias_term_length for term in bias_terms
        ):
            return _err(400, "bad_request", "bias terms exceed the advertised limits")
        try:
            pcm_bytes = base64.b64decode(body.audioData, validate=True)
        except (binascii.Error, ValueError):
            return _err(400, "unsupported_audio", "audioData is not valid Base64")
        if not pcm_bytes or len(pcm_bytes) % 2:
            return _err(400, "unsupported_audio", "PCM16LE audio must contain complete samples")
        audio_seconds = len(pcm_bytes) / (2 * body.audio.sampleRate)
        if audio_seconds > spec.max_seconds:
            return _err(
                413,
                "audio_too_long",
                f"audio is {audio_seconds:.2f}s; maximum is {spec.max_seconds:g}s",
            )
        pcm = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        processing_started = time.perf_counter()
        pool.pin(body.model)
        try:
            # Pin before checking residency so another request cannot evict the
            # model between the check and acquire, which would under-report a
            # reload as a zero-millisecond warm acquisition.
            loaded_before = body.model in pool.loaded_ids()
            try:
                load_started = time.perf_counter()
                session = pool.acquire(body.model)
                model_load_ms = (
                    0.0
                    if loaded_before
                    else (time.perf_counter() - load_started) * 1000.0
                )
            except EngineBusyError as exc:
                return _err(409, "model_busy", str(exc), {"modelIds": exc.model_ids})
            except EngineError as exc:
                return _err(500, "model_load_failed", str(exc))
            try:
                result = session.transcribe(
                    pcm,
                    sample_rate=body.audio.sampleRate,
                    language=language_hint,
                    bias_terms=bias_terms,
                )
            except EngineError as exc:
                return _err(500, "transcription_failed", str(exc))
        finally:
            pool.unpin(body.model)
        processing_ms = (time.perf_counter() - processing_started) * 1000.0
        inference_ms = float(result.pop("inferenceMs", 0.0))
        response = {
            "requestId": body.requestId,
            "model": body.model,
            "text": result["text"],
            "confidence": None,
            "requestedLanguage": body.language,
            "detectedLanguage": result["detectedLanguage"],
            "audioSeconds": audio_seconds,
            "modelLoadMs": model_load_ms,
            "inferenceMs": inference_ms,
            "processingMs": processing_ms,
            "throughputX": (
                audio_seconds / (inference_ms / 1000.0) if inference_ms > 0 else None
            ),
        }
        if body.timestamps in {"segment", "word"}:
            response["segments"] = [
                {
                    key: value
                    for key, value in segment.items()
                    if body.timestamps == "word" or key != "words"
                }
                for segment in result["segments"]
            ]
        if body.timestamps == "word":
            response["words"] = result["words"]
        return response

    @app.post("/v2/models/{model_id}/voices/{voice_id}:prepare")
    def prepare_voice(model_id: str, voice_id: str):
        try:
            model = pool.spec(model_id)
        except KeyError:
            return _err(404, "model_not_found", model_id)
        if model.task != "tts":
            return _err(400, "wrong_task", f"model {model_id!r} is not a TTS model")
        if model.voice_reference.preparation_mode != "persistent":
            return _err(
                409,
                "voice_preparation_unsupported",
                f"model {model_id!r} does not expose persistent voice preparation",
            )
        try:
            voice_wav, stored_transcript = voices.resolve(voice_id)
        except (KeyError, VoiceError):
            return _err(404, "voice_not_found", voice_id)
        transcript_policy = model.voice_reference.transcript
        if transcript_policy == "required" and not stored_transcript:
            return _err(
                409,
                "voice_transcript_required",
                f"model {model_id!r} requires a reference transcript",
            )
        ref_text = None if transcript_policy == "unused" else stored_transcript
        pool.pin(model_id)
        try:
            session = pool.acquire(model_id)
            session.prepare_voice(voice_wav, ref_text)
            marker = voices.mark_prepared(
                voice_id,
                model_id=model_id,
                revision=model.voice_preparation_revision,
                inputs=model.voice_reference.preparation_inputs,
            )
        except EngineBusyError as exc:
            return _err(
                409,
                "model_busy",
                str(exc),
                {"modelIds": exc.model_ids},
            )
        except (EngineError, KeyError, OSError, ValueError, VoiceError) as exc:
            return _err(500, "voice_preparation_failed", str(exc))
        finally:
            pool.unpin(model_id)
        return {
            "modelId": model_id,
            "voiceId": voice_id,
            "prepared": True,
            "preparation": marker,
        }

    @app.websocket("/v2/synthesize")
    async def synthesize_ws(ws: WebSocket):
        if not _authorized(ws.headers.get("authorization", "")):
            await ws.close(code=4401)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    start_message = json.loads(raw)
                    if start_message.get("type") != "start":
                        raise ValueError("expected a start message")
                    start = StartRequest.model_validate(start_message)
                    model = cfg.models.get(start.model)
                    if model is None:
                        raise ValueError(f"unknown model {start.model!r}")
                    if model.task != "tts":
                        await ws.send_text(
                            json.dumps(
                                error_event(
                                    "wrong_task", f"model {start.model!r} is not a TTS model"
                                )
                            )
                        )
                        continue
                except (AttributeError, ValueError, ValidationError) as exc:
                    await ws.send_text(json.dumps(error_event("bad_request", str(exc))))
                    continue

                segment_queue: queue.Queue = queue.Queue()
                output_queue: asyncio.Queue = asyncio.Queue()

                def emit(item):
                    loop.call_soon_threadsafe(output_queue.put_nowait, item)

                def run_worker() -> None:
                    try:
                        run_request(
                            start,
                            segment_queue,
                            emit,
                            pool=pool,
                            voices=voices,
                            aligner=aligner,
                            debug=cfg.debug,
                        )
                    except Exception:
                        logger.exception("unexpected synthesis worker failure")
                        emit(
                            error_event(
                                "synthesis_failed",
                                "unexpected internal synthesis failure",
                            )
                        )

                worker = loop.run_in_executor(None, run_worker)

                async def receive_segments():
                    try:
                        while True:
                            try:
                                message = json.loads(await ws.receive_text())
                                if not isinstance(message, dict):
                                    raise ValueError("message must be an object")
                            except ValueError:
                                message = {"type": "invalid"}
                            segment_queue.put(message)
                            if message.get("type") in ("end", "abort"):
                                return
                    except WebSocketDisconnect:
                        segment_queue.put({"type": "abort"})
                        raise

                receive_task = asyncio.create_task(receive_segments())
                try:
                    while True:
                        item = await output_queue.get()
                        if isinstance(item, (bytes, bytearray)):
                            await ws.send_bytes(bytes(item))
                            continue
                        await ws.send_text(json.dumps(item))
                        if item.get("type") in ("done", "error"):
                            break
                finally:
                    if not receive_task.done():
                        segment_queue.put({"type": "abort"})
                        receive_task.cancel()
                    with suppress(asyncio.CancelledError, WebSocketDisconnect):
                        await receive_task
                    await worker
        except WebSocketDisconnect:
            pass

    return app

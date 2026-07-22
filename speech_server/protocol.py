"""Wire-protocol models for WebSocket protocol v2."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import LANGUAGE_TAG_RE


class SilenceOpts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minMs: float = 250.0
    maxMs: float = 1000.0

    @field_validator("minMs", "maxMs", mode="before")
    @classmethod
    def validate_numeric_type(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("silence values must be numbers")
        return value

    @model_validator(mode="after")
    def validate_range(self):
        if (
            not math.isfinite(self.minMs)
            or not math.isfinite(self.maxMs)
            or self.minMs < 0
            or self.maxMs < 0
            or self.minMs > self.maxMs
        ):
            raise ValueError("silence requires 0 <= minMs <= maxMs")
        return self


class SynthOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    numSteps: int | None = None
    firstSegmentSteps: int | None = None
    guidanceScale: float | None = None
    exaggeration: float | None = None
    seed: int | None = Field(default=None, ge=0, le=0xFFFFFFFFFFFFFFFF)
    upscale: bool = False
    timing: Literal["none", "auto", "word"] = "none"
    silence: SilenceOpts = Field(default_factory=SilenceOpts)

    @field_validator("numSteps", "firstSegmentSteps", "seed", mode="before")
    @classmethod
    def validate_integer_type(cls, value):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("value must be an integer")
        return value

    @field_validator("guidanceScale", "exaggeration", mode="before")
    @classmethod
    def validate_number_type(cls, value):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError("value must be a number")
        return value

    @field_validator("upscale", mode="before")
    @classmethod
    def validate_boolean_type(cls, value):
        if not isinstance(value, bool):
            raise ValueError("upscale must be a boolean")
        return value


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["start"] = "start"
    requestId: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    voiceId: str = Field(min_length=1, max_length=256)
    language: str = Field(default="English", min_length=1, max_length=64)
    options: SynthOptions = Field(default_factory=SynthOptions)


class SegmentUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=32768)

    @field_validator("id", "text")
    @classmethod
    def reject_blank(cls, value: str):
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class SegmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["segment"]
    idx: int = Field(ge=0)
    text: str | None = Field(default=None, max_length=32768)
    units: list[SegmentUnit] | None = Field(default=None, min_length=1, max_length=32)
    voiceId: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_payload(self):
        if (self.text is None) == (self.units is None):
            raise ValueError("segment requires exactly one of text or units")
        if self.text is not None and not self.text.strip():
            raise ValueError("segment text must not be blank")
        if self.units is not None:
            ids = [unit.id for unit in self.units]
            if len(ids) != len(set(ids)):
                raise ValueError("segment unit IDs must be unique")
            if sum(len(unit.text) for unit in self.units) > 32768:
                raise ValueError("segment unit text exceeds 32768 characters")
        return self


ERROR_CODES = {
    "bad_request",
    "model_not_found",
    "wrong_task",
    "model_load_failed",
    "model_busy",
    "voice_not_found",
    "voice_transcript_required",
    "synthesis_failed",
    "upscaler_unavailable",
    "upscaling_failed",
}


class ASRAudioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoding: str = Field(default="pcm_s16le", min_length=1, max_length=32)
    sampleRate: int = Field(default=16000, ge=1, le=384000)
    channels: int = Field(default=1, ge=1, le=8)

    @field_validator("sampleRate", "channels", mode="before")
    @classmethod
    def validate_integer_type(cls, value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("audio sample rate and channels must be integers")
        return value


class TranscribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requestId: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    audioData: str = Field(min_length=4, max_length=32 * 1024 * 1024)
    audio: ASRAudioInput = Field(default_factory=ASRAudioInput)
    language: str = Field(default="auto", min_length=2, max_length=32)
    biasTerms: list[Annotated[str, Field(min_length=1, max_length=1024)]] = Field(
        default_factory=list, max_length=1024
    )
    timestamps: str = Field(default="none", min_length=1, max_length=32)

    @field_validator("biasTerms")
    @classmethod
    def validate_bias_terms(cls, value: list[str]):
        if any(not isinstance(term, str) or not term.strip() for term in value):
            raise ValueError("bias terms must be nonempty strings")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str):
        normalized = value.strip().lower().replace("_", "-")
        if normalized != "auto" and not LANGUAGE_TAG_RE.fullmatch(normalized):
            raise ValueError("language must be 'auto' or a language code")
        return normalized


def error_event(code: str, message: str) -> dict:
    assert code in ERROR_CODES
    return {"type": "error", "code": code, "message": message}

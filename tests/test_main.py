from pathlib import Path

from speech_server.__main__ import _required_runtime_symbols
from speech_server.config import ASRSpec, AlignerSpec, ModelSpec, ServerConfig


def test_runtime_symbols_follow_installed_component_tasks(tmp_path: Path):
    tts_path = tmp_path / "tts.gguf"
    asr_path = tmp_path / "asr.gguf"
    aligner_path = tmp_path / "aligner.gguf"
    tts_path.write_bytes(b"tts")
    asr_path.write_bytes(b"asr")
    # Deliberately leave the aligner missing: startup must not require its ABI.
    cfg = ServerConfig(
        models={
            "tts": ModelSpec("tts", "omnivoice", tts_path, 24000),
            "asr": ModelSpec(
                "asr", "parakeet", asr_path, 16000,
                task="asr", cloning=False, languages=["en"], asr=ASRSpec(bias_terms=True),
            ),
        },
        aligner=AlignerSpec("canary", "canary-ctc", aligner_path, ["en"]),
    )
    assert _required_runtime_symbols(cfg) == [
        "crispasr_session_synthesize",
        "crispasr_session_set_tts_seed",
        "crispasr_session_transcribe",
        "crispasr_session_set_hotwords",
    ]


def test_missing_models_do_not_require_unavailable_backend_symbols(tmp_path: Path):
    cfg = ServerConfig(
        models={
            "asr": ModelSpec(
                "asr", "parakeet", tmp_path / "missing.gguf", 16000,
                task="asr", cloning=False, languages=["en"], asr=ASRSpec(),
            )
        }
    )
    assert _required_runtime_symbols(cfg) == []

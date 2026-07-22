"""Command-line entrypoint for the universal speech server."""

import logging

import uvicorn

from .app import create_app
from .config import load_config
from .crisp import CrispBindingError, load_library
from .engine import _register_windows_dll_dirs

logger = logging.getLogger(__name__)


def _installed(spec) -> bool:
    return spec is not None and spec.installed


def _required_runtime_symbols(cfg) -> list[str]:
    installed_models = [model for model in cfg.models.values() if _installed(model)]
    symbols = []
    if any(model.task == "tts" for model in installed_models):
        symbols.extend(("crispasr_session_synthesize", "crispasr_session_set_tts_seed"))
    installed_asr = [model for model in installed_models if model.task == "asr"]
    if installed_asr:
        symbols.append("crispasr_session_transcribe")
    if any(model.asr and model.asr.bias_terms for model in installed_asr):
        symbols.append("crispasr_session_set_hotwords")
    if _installed(cfg.upscaler):
        symbols.append("crispasr_session_speech_to_speech")
    if _installed(cfg.aligner):
        symbols.append("crispasr_align_words_abi")
    return symbols


def main():
    cfg = load_config()
    if cfg.lib_path is None:
        raise SystemExit(
            "libcrispasr not found — set SPEECH_SERVER_LIB or place "
            "libcrispasr-*/bin next to the repo root"
        )
    unavailable_models = [
        model.id for model in cfg.models.values() if not _installed(model)
    ]
    if unavailable_models:
        logger.warning(
            "configured models unavailable until their component files are installed: %s",
            ", ".join(unavailable_models),
        )
    if cfg.upscaler is not None and not _installed(cfg.upscaler):
        logger.warning("configured upscaler is unavailable until its model file is installed")
    if cfg.aligner is not None and not _installed(cfg.aligner):
        logger.warning(
            "configured aligner is unavailable; adaptive batching will remain disabled"
        )
    _register_windows_dll_dirs(cfg)
    try:
        load_library(cfg.lib_path, _required_runtime_symbols(cfg))
    except CrispBindingError as exc:
        raise SystemExit(str(exc)) from exc
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()

"""Self-contained Sonorus-compatible text preprocessing policy."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Match
_BRACKET_RE = re.compile(r"\[([^\]]*)\]")
_ALL_CAPS_RE = re.compile(r"\b([A-Z]{2,})(?:'S|'s)?\b")
_EXCLAMATION_RE = re.compile(r"!+")
_SENTENCE_START_RE = re.compile(r"(?:(?<=^)|(?<=[.!?]\s)|(?<=\]\s))([a-z])")
_ABBREVIATIONS_FILE = Path(__file__).resolve().parent.parent / "abbreviations.txt"


@lru_cache(maxsize=1)
def _load_uppercase_abbreviations() -> set[str]:
    try:
        return {
            line.strip().upper()
            for line in _ABBREVIATIONS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    except OSError:
        return set()


def _lowercase_caps_except_abbreviations(text: str) -> str:
    preserved = _load_uppercase_abbreviations()

    def repl(match: Match[str]) -> str:
        word = match.group(1)
        suffix = match.group(0)[len(word) :]
        return f"{word if word in preserved else word.lower()}{suffix}"

    return _ALL_CAPS_RE.sub(repl, text)


def _normalize_exclamation(text: str) -> str:
    return _EXCLAMATION_RE.sub(
        lambda match: "." if len(match.group(0)) == 1 else "!", text
    )


def _capitalize_sentence_starts(text: str) -> str:
    return _SENTENCE_START_RE.sub(lambda match: match.group(1).upper(), text)


def preprocess_text(text: str, normalize_exclamation: bool = False) -> str:
    """Normalize punctuation, quotes, whitespace, and shouting words."""
    replacements = [
        ("…", ","),
        ("—", ", "),
        ("–", ", "),
        (":", ","),
        (";", ","),
        ("\n", " "),
        ('"', ""),
        ("“", ""),
        ("”", ""),
        ("‘", "'"),
        ("’", "'"),
        ("‚", "'"),
        ("‛", "'"),
        ("ʼ", "'"),
        ("ʹ", "'"),
        ("ʻ", "'"),
        ("ʾ", "'"),
        ("ʿ", "'"),
        ("′", "'"),
        ("‵", "'"),
        ("＇", "'"),
        ("Ꞌ", "'"),
        ("*", ""),
    ]
    for source, replacement in replacements:
        text = text.replace(source, replacement)
    text = " ".join(text.split())
    text = _lowercase_caps_except_abbreviations(text)
    # Keep parity with the existing Sonorus helper. The option is retained for
    # API compatibility; the current local-parity path preserves exclamation.
    return text


@dataclass(frozen=True)
class AudioTagAnchor:
    tag: str
    before_word: int


@dataclass(frozen=True)
class NormalizedText:
    source_text: str
    tts_text: str
    alignment_text: str
    alignment_tokens: tuple[str, ...]
    audio_tags: tuple[AudioTagAnchor, ...]


def normalize_unit(
    text: str,
    profile: str = "plain",
    tag_map: Mapping[str, str] | None = None,
) -> NormalizedText:
    """Produce distinct TTS/alignment forms without losing tag anchors."""
    processed = preprocess_text(text, normalize_exclamation=True)
    accepted_tags = tag_map or {}
    parts: list[str] = []
    anchors: list[AudioTagAnchor] = []
    cursor = 0
    spoken_parts: list[str] = []
    for match in _BRACKET_RE.finditer(processed):
        prefix = processed[cursor : match.start()]
        parts.append(prefix)
        spoken_parts.append(prefix)
        raw = f"[{match.group(1)}]"
        lower = raw.lower()
        canonical = accepted_tags.get(lower)
        if canonical is not None:
            parts.extend((" ", canonical, " "))
            anchors.append(
                AudioTagAnchor(
                    canonical.strip("[]"),
                    len(" ".join(spoken_parts).split()),
                )
            )
        else:
            # A removed control is a boundary, not an empty string: joining
            # the surrounding text could silently turn "one[tag]two" into
            # the different lexical token "onetwo" for both TTS and CTC.
            parts.append(" ")
        cursor = match.end()
    suffix = processed[cursor:]
    parts.append(suffix)
    tts_text = " ".join("".join(parts).split())
    alignment_text = " ".join(_BRACKET_RE.sub(" ", tts_text).split())
    return NormalizedText(
        source_text=text,
        tts_text=tts_text,
        alignment_text=alignment_text,
        alignment_tokens=tuple(alignment_text.split()),
        audio_tags=tuple(anchors),
    )


def preprocess(
    text: str,
    profile: str = "plain",
    tag_map: Mapping[str, str] | None = None,
) -> str:
    """Backward-compatible TTS text helper."""
    return normalize_unit(text, profile, tag_map).tts_text

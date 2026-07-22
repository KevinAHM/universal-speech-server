from speech_server.textproc import normalize_unit, preprocess

LAUGHTER_TAGS = {
    "[laughter]": "[laughter]",
    "[laugh]": "[laughter]",
    "[laughs]": "[laughter]",
    "[laughing]": "[laughter]",
}


def test_keeps_allowed_audio_tags():
    assert (
        preprocess("Ha! [laughter] Good one.", "omnivoice", LAUGHTER_TAGS)
        == "Ha! [laughter] Good one."
    )


def test_strips_unknown_bracket_tags():
    assert preprocess("Hello [angry] there [whisper-x]") == "Hello there"


def test_collapses_whitespace_and_empty():
    assert preprocess("  a   b  ") == "a b"
    assert preprocess("[angry]") == ""


def test_omnivoice_aliases_and_alignment_text_are_separate():
    normalized = normalize_unit(
        "[laugh] I knew [angry] it.", "omnivoice", LAUGHTER_TAGS
    )
    assert normalized.source_text == "[laugh] I knew [angry] it."
    assert normalized.tts_text == "[laughter] I knew it."
    assert normalized.alignment_text == "I knew it."
    assert normalized.alignment_tokens == ("I", "knew", "it.")
    assert [(tag.tag, tag.before_word) for tag in normalized.audio_tags] == [
        ("laughter", 0)
    ]


def test_plain_profile_strips_every_bracket_control():
    normalized = normalize_unit("Hello [laughter] there.", "plain")
    assert normalized.tts_text == "Hello there."
    assert normalized.alignment_text == "Hello there."
    assert normalized.audio_tags == ()


def test_removed_and_supported_tags_never_join_neighboring_words():
    plain = normalize_unit("one[unknown]two", "plain")
    assert plain.tts_text == "one two"
    assert plain.alignment_text == "one two"
    omni = normalize_unit("one[LAUGH]two", "omnivoice", LAUGHTER_TAGS)
    assert omni.tts_text == "one [laughter] two"
    assert omni.alignment_text == "one two"
    assert [(tag.tag, tag.before_word) for tag in omni.audio_tags] == [
        ("laughter", 1)
    ]

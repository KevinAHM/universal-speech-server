import numpy as np

from speech_server.audio import (
    pcm16_bytes,
    silence_pcm16,
    trim_omnivoice_output_silence,
)


def test_pcm16_bytes_clips_and_converts():
    arr = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    out = np.frombuffer(pcm16_bytes(arr), dtype=np.int16)
    assert out.tolist() == [0, 32767, -32767, 32767]


def test_silence_length():
    data = silence_pcm16(500, 24000)
    assert len(data) == 24000
    assert set(data) == {0}


def test_omnivoice_silence_postprocess_matches_torch_edge_and_middle_keeps():
    sample_rate = 1000
    tone = np.full(200, 0.1, dtype=np.float32)
    audio = np.concatenate(
        [
            np.zeros(300, dtype=np.float32),
            tone,
            np.zeros(1500, dtype=np.float32),
            tone,
            np.zeros(400, dtype=np.float32),
        ]
    )

    trimmed = trim_omnivoice_output_silence(audio, sample_rate)

    # 100 ms outer padding + 200 ms speech + a middle gap capped at
    # 500 ms per side + 200 ms speech + 100 ms outer padding.
    assert len(trimmed) == 1600
    assert np.count_nonzero(trimmed) == 400
    np.testing.assert_array_equal(trimmed[:100], 0.0)
    np.testing.assert_array_equal(trimmed[-100:], 0.0)


def test_omnivoice_silence_postprocess_preserves_all_silent_fallback():
    audio = np.zeros(2400, dtype=np.float32)
    trimmed = trim_omnivoice_output_silence(audio, 24000)
    assert trimmed.shape == audio.shape


def test_omnivoice_silence_uses_long_window_across_short_noise_spikes():
    sample_rate = 1000
    tone = np.full(200, 0.1, dtype=np.float32)
    gap = np.zeros(1500, dtype=np.float32)
    for offset in range(100, len(gap), 250):
        gap[offset : offset + 10] = 0.01
    audio = np.concatenate([tone, gap, tone])

    trimmed = trim_omnivoice_output_silence(audio, sample_rate)

    # Individual 10 ms spike frames exceed -50 dBFS, but the same 500 ms RMS
    # windows used by pydub remain silent. The 1.5 s gap is therefore capped
    # to the 500 ms retained on each side instead of slipping through intact.
    assert len(trimmed) == 1400

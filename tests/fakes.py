import numpy as np
from types import SimpleNamespace


class FakeSession:
    """Mimics crispasr.Session's TTS surface."""

    def __init__(self, backend="omnivoice"):
        self.backend = backend
        self.calls = []
        self.voice = None
        self.closed = False
        self.samples_per_call = 2400
        self.pcm_sample_rate = 16000

    def set_codec_path(self, path):
        self.calls.append(("codec", path))

    def set_voice(self, path, ref_text=None):
        self.voice = (path, ref_text)
        self.calls.append(("voice", path, ref_text))

    def set_tts_steps(self, steps):
        self.calls.append(("steps", steps))

    def set_tts_seed(self, seed):
        self.calls.append(("seed", seed))

    def set_cfg_weight(self, weight):
        self.calls.append(("cfg", weight))

    def set_exaggeration(self, exaggeration):
        self.calls.append(("exaggeration", exaggeration))

    def set_tts_cfg_scale(self, scale):
        self.calls.append(("tts_cfg", scale))

    def synthesize(self, text):
        self.calls.append(("synth", text))
        return np.zeros(self.samples_per_call, dtype=np.float32)

    def synthesize_raw(self, text):
        return self.synthesize(text)

    def set_pcm_sample_rate(self, sample_rate):
        self.pcm_sample_rate = int(sample_rate)
        self.calls.append(("pcm_sample_rate", self.pcm_sample_rate))

    def speech_to_speech(self, audio, language=None):
        self.calls.append(("s2s", len(audio)))
        n_out = round(len(audio) * 48000 / self.pcm_sample_rate)
        return np.zeros(n_out, dtype=np.float32), ""

    def set_hotwords(self, hotwords, boost=2.0):
        self.calls.append(("hotwords", hotwords, boost))

    def transcribe(self, pcm, sample_rate=16000, language=None):
        self.calls.append(("transcribe", len(pcm), sample_rate, language))
        return [
            SimpleNamespace(
                text="Hello Hogwarts.",
                start=0.0,
                end=len(pcm) / sample_rate,
                words=[
                    SimpleNamespace(text="Hello", start=0.0, end=0.1),
                    SimpleNamespace(text="Hogwarts.", start=0.1, end=0.2),
                ],
            )
        ]

    def detected_language(self):
        return "en"

    def close(self):
        self.closed = True

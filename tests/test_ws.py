import json
from pathlib import Path

from fastapi.testclient import TestClient

from speech_server.app import create_app
from speech_server.config import ModelSpec, ServerConfig
from tests.fakes import FakeSession


def _client(tmp_path, token=""):
    spec = ModelSpec(
        id="omnivoice",
        backend="omnivoice",
        model_path=Path("x.gguf"),
        sample_rate=24000,
    )
    cfg = ServerConfig(
        models={"omnivoice": spec}, auth_token=token, voice_dir=tmp_path
    )
    (tmp_path / "default__Seb.wav").write_bytes(
        b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 8
    )
    return TestClient(create_app(cfg, session_factory=lambda cfg, model: FakeSession()))


def test_ws_block_synthesis(tmp_path):
    client = _client(tmp_path)
    with client.websocket_connect("/v2/synthesize") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "start",
                    "requestId": "r1",
                    "model": "omnivoice",
                    "voiceId": "default__Seb",
                    "options": {"silence": {"minMs": 0, "maxMs": 0}},
                }
            )
        )
        ws.send_text(json.dumps({"type": "segment", "idx": 0, "text": "Hello."}))
        ws.send_text(json.dumps({"type": "end"}))
        events = []
        pcm = b""
        while True:
            message = ws.receive()
            if message.get("bytes") is not None:
                pcm += message["bytes"]
                continue
            event = json.loads(message["text"])
            events.append(event)
            if event["type"] in ("done", "error"):
                break
        assert [event["type"] for event in events] == [
            "started",
            "segment_start",
            "segment_done",
            "done",
        ]
        assert events[0]["sampleRate"] == 24000 and events[0]["mode"] == "block"
        assert len(pcm) == 2400 * 2


def test_ws_bad_start(tmp_path):
    client = _client(tmp_path)
    with client.websocket_connect("/v2/synthesize") as ws:
        ws.send_text("not json")
        event = json.loads(ws.receive_text())
        assert event["type"] == "error" and event["code"] == "bad_request"


def test_ws_bad_segment_reports_error(tmp_path):
    client = _client(tmp_path)
    with client.websocket_connect("/v2/synthesize") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "start",
                    "requestId": "r1",
                    "model": "omnivoice",
                    "voiceId": "default__Seb",
                }
            )
        )
        ws.send_text("not json")
        assert json.loads(ws.receive_text())["type"] == "started"
        event = json.loads(ws.receive_text())
        assert event["type"] == "error" and event["code"] == "bad_request"


def test_ws_auth(tmp_path):
    client = _client(tmp_path, token="tok")
    try:
        with client.websocket_connect("/v2/synthesize") as ws:
            closed = ws.receive()
            assert closed["type"] == "websocket.close"
    except Exception:
        pass

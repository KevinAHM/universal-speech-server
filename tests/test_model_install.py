import hashlib
import io
import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from speech_server.config import ModelSpec, ServerConfig, load_config
from speech_server.crisp import CrispBindingError, RegistryArtifact, RegistryBundle
from speech_server.model_install import (
    InstallCancelled,
    LockedArtifact,
    ModelInstallError,
    ModelInstallationManager,
    _SafeRedirect,
    download_locked_artifact,
    lock_huggingface_artifact,
)


class FakeResponse(io.BytesIO):
    def __init__(self, body=b"", *, headers=None, status=200, url="https://example.invalid"):
        super().__init__(body)
        self.headers = headers or {}
        self.status = status
        self._url = url

    def geturl(self):
        return self._url

    def getcode(self):
        return self.status


def _locked(filename: str, body: bytes, kind: str = "primary") -> LockedArtifact:
    return LockedArtifact(
        kind=kind,
        filename=filename,
        source_url=f"https://huggingface.co/org/repo/resolve/main/{filename}",
        url=f"https://huggingface.co/org/repo/resolve/{'a' * 40}/{filename}",
        repository="org/repo",
        revision="a" * 40,
        sha256=hashlib.sha256(body).hexdigest(),
        size=len(body),
    )


def _wait(manager: ModelInstallationManager, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["state"] not in {"queued", "resolving", "downloading", "installing"}:
            return job
        time.sleep(0.01)
    raise AssertionError("installation job did not finish")


def test_huggingface_lock_pins_revision_digest_and_size(monkeypatch):
    headers = {
        "X-Repo-Commit": "a" * 40,
        "X-Linked-Etag": '"' + "b" * 64 + '"',
        "X-Linked-Size": "1234",
    }
    calls = []

    def head(url):
        calls.append(url)
        return FakeResponse(headers=headers, status=302, url=url)

    monkeypatch.setattr("speech_server.model_install._head_without_redirect", head)
    locked = lock_huggingface_artifact(
        RegistryArtifact(
            "primary",
            "model.gguf",
            "https://huggingface.co/org/repo/resolve/main/model.gguf",
            "~1 KB",
        )
    )
    assert locked.revision == "a" * 40
    assert locked.sha256 == "b" * 64
    assert locked.size == 1234
    assert locked.url == (
        "https://huggingface.co/org/repo/resolve/" + "a" * 40 + "/model.gguf"
    )
    assert calls == [locked.source_url, locked.url]


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/org/repo/resolve/main/model.gguf",
        "https://huggingface.co.evil.example/org/repo/resolve/main/model.gguf",
        "https://huggingface.co:444/org/repo/resolve/main/model.gguf",
        "https://example.com/model.gguf",
    ],
)
def test_huggingface_lock_rejects_unverifiable_hosts(url):
    with pytest.raises(ModelInstallError, match="verifiable huggingface.co"):
        lock_huggingface_artifact(
            RegistryArtifact("primary", "model.gguf", url, "~1 GB")
        )


def test_huggingface_lock_requires_lfs_sha256(monkeypatch):
    monkeypatch.setattr(
        "speech_server.model_install._head_without_redirect",
        lambda url: FakeResponse(
            headers={"X-Repo-Commit": "a" * 40, "X-Linked-Size": "12"}
        ),
    )
    with pytest.raises(ModelInstallError, match="LFS SHA-256"):
        lock_huggingface_artifact(
            RegistryArtifact(
                "primary",
                "model.gguf",
                "https://huggingface.co/org/repo/resolve/main/model.gguf",
                "~1 GB",
            )
        )


def test_cross_host_download_redirect_does_not_forward_hf_token():
    request = urllib.request.Request(
        "https://huggingface.co/org/repo/resolve/main/model.gguf",
        headers={"Authorization": "Bearer secret"},
    )
    redirected = _SafeRedirect().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://cdn.example/model.gguf?signature=valid",
    )
    assert redirected.get_header("Authorization") is None


def test_download_redirect_rejects_downgrade_to_http():
    request = urllib.request.Request(
        "https://huggingface.co/org/repo/resolve/main/model.gguf"
    )
    with pytest.raises(ModelInstallError, match="insecure redirect"):
        _SafeRedirect().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://cdn.example/model.gguf",
        )


def test_locked_downloader_verifies_and_promotes_atomically(monkeypatch, tmp_path: Path):
    body = b"model-data" * 1000
    artifact = _locked("model.gguf", body)
    monkeypatch.setattr(
        "speech_server.model_install._open_download",
        lambda request: FakeResponse(body, url=artifact.url),
    )
    progress = []
    destination = tmp_path / artifact.filename
    download_locked_artifact(
        artifact,
        destination,
        lambda current, total: progress.append((current, total)),
        threading.Event(),
    )
    assert destination.read_bytes() == body
    assert progress[-1] == (len(body), len(body))
    assert not (tmp_path / "model.gguf.part").exists()
    assert not (tmp_path / "model.gguf.part.json").exists()


def test_locked_downloader_resumes_a_matching_partial(monkeypatch, tmp_path: Path):
    body = b"0123456789" * 1000
    artifact = _locked("model.gguf", body)
    offset = 4321
    (tmp_path / "model.gguf.part").write_bytes(body[:offset])
    (tmp_path / "model.gguf.part.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "url": artifact.url,
                "sha256": artifact.sha256,
                "size": artifact.size,
            }
        ),
        encoding="utf-8",
    )

    def urlopen(request):
        assert request.headers["Range"] == f"bytes={offset}-"
        return FakeResponse(
            body[offset:],
            headers={"Content-Range": f"bytes {offset}-{len(body) - 1}/{len(body)}"},
            status=206,
            url=artifact.url,
        )

    monkeypatch.setattr("speech_server.model_install._open_download", urlopen)
    destination = tmp_path / artifact.filename
    download_locked_artifact(
        artifact, destination, lambda current, total: None, threading.Event()
    )
    assert destination.read_bytes() == body


def test_locked_downloader_keeps_partial_when_cancelled(monkeypatch, tmp_path: Path):
    body = b"x" * 100
    artifact = _locked("model.gguf", body)
    monkeypatch.setattr(
        "speech_server.model_install._open_download",
        lambda request: FakeResponse(body, url=artifact.url),
    )
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InstallCancelled):
        download_locked_artifact(
            artifact,
            tmp_path / artifact.filename,
            lambda current, total: None,
            cancelled,
        )
    assert (tmp_path / "model.gguf.part").exists()
    assert (tmp_path / "model.gguf.part.json").exists()


def test_large_locked_downloader_uses_verified_parallel_ranges(
    monkeypatch, tmp_path: Path
):
    body = b"0123456789abcdef"
    artifact = _locked("model.gguf", body)
    monkeypatch.setattr("speech_server.model_install.PARALLEL_MIN_BYTES", 1)
    monkeypatch.setattr("speech_server.model_install.PARALLEL_SEGMENT_BYTES", 4)
    monkeypatch.setattr("speech_server.model_install.PARALLEL_WORKERS", 3)
    requested = []

    def urlopen(request):
        byte_range = request.headers.get("Range")
        requested.append(byte_range)
        if byte_range == "bytes=0-0":
            return FakeResponse(
                body[:1],
                headers={"Content-Range": f"bytes 0-0/{len(body)}"},
                status=206,
                url=artifact.url,
            )
        start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
        return FakeResponse(
            body[start : end + 1],
            headers={"Content-Range": f"bytes {start}-{end}/{len(body)}"},
            status=206,
            url=artifact.url,
        )

    monkeypatch.setattr("speech_server.model_install._open_download", urlopen)
    destination = tmp_path / artifact.filename
    progress = []
    download_locked_artifact(
        artifact,
        destination,
        lambda current, total: progress.append((current, total)),
        threading.Event(),
    )

    assert destination.read_bytes() == body
    assert requested[0] == "bytes=0-0"
    assert set(requested[1:]) == {
        "bytes=0-3", "bytes=4-7", "bytes=8-11", "bytes=12-15"
    }
    assert progress[-1] == (len(body), len(body))
    assert not (tmp_path / "model.gguf.part").exists()
    assert not (tmp_path / "model.gguf.part.json").exists()


def test_parallel_downloader_resumes_only_unfinished_segments(
    monkeypatch, tmp_path: Path
):
    body = b"abcdefghijkl"
    artifact = _locked("model.gguf", body)
    monkeypatch.setattr("speech_server.model_install.PARALLEL_MIN_BYTES", 1)
    monkeypatch.setattr("speech_server.model_install.PARALLEL_SEGMENT_BYTES", 4)
    requested = []
    part = tmp_path / "model.gguf.part"
    part.write_bytes(body[:4] + b"\0" * (len(body) - 4))
    (tmp_path / "model.gguf.part.json").write_text(
        json.dumps({
            "schemaVersion": 2,
            "url": artifact.url,
            "sha256": artifact.sha256,
            "size": artifact.size,
            "segmentSize": 4,
            "completedSegments": [0],
        }),
        encoding="utf-8",
    )

    def urlopen(request):
        byte_range = request.headers.get("Range")
        requested.append(byte_range)
        if byte_range == "bytes=0-0":
            return FakeResponse(
                body[:1], headers={"Content-Range": "bytes 0-0/12"},
                status=206, url=artifact.url,
            )
        start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
        return FakeResponse(
            body[start : end + 1],
            headers={"Content-Range": f"bytes {start}-{end}/12"},
            status=206, url=artifact.url,
        )

    monkeypatch.setattr("speech_server.model_install._open_download", urlopen)
    destination = tmp_path / artifact.filename
    download_locked_artifact(
        artifact, destination, lambda current, total: None, threading.Event()
    )

    assert destination.read_bytes() == body
    assert "bytes=0-3" not in requested
    assert set(requested) == {"bytes=0-0", "bytes=4-7", "bytes=8-11"}


def test_parallel_downloader_falls_back_when_ranges_are_ignored(
    monkeypatch, tmp_path: Path
):
    body = b"range-fallback"
    artifact = _locked("model.gguf", body)
    monkeypatch.setattr("speech_server.model_install.PARALLEL_MIN_BYTES", 1)
    requests = []

    def urlopen(request):
        requests.append(request.headers.get("Range"))
        return FakeResponse(body, status=200, url=artifact.url)

    monkeypatch.setattr("speech_server.model_install._open_download", urlopen)
    destination = tmp_path / artifact.filename
    download_locked_artifact(
        artifact, destination, lambda current, total: None, threading.Event()
    )

    assert destination.read_bytes() == body
    assert requests == ["bytes=0-0", None]


def test_parallel_downloader_preserves_resumable_state_when_cancelled(
    monkeypatch, tmp_path: Path
):
    body = b"cancel-parallel"
    artifact = _locked("model.gguf", body)
    monkeypatch.setattr("speech_server.model_install.PARALLEL_MIN_BYTES", 1)
    monkeypatch.setattr("speech_server.model_install.PARALLEL_SEGMENT_BYTES", 4)

    def urlopen(request):
        assert request.headers.get("Range") == "bytes=0-0"
        return FakeResponse(
            body[:1],
            headers={"Content-Range": f"bytes 0-0/{len(body)}"},
            status=206,
            url=artifact.url,
        )

    monkeypatch.setattr("speech_server.model_install._open_download", urlopen)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InstallCancelled):
        download_locked_artifact(
            artifact,
            tmp_path / artifact.filename,
            lambda current, total: None,
            cancelled,
        )

    part = tmp_path / "model.gguf.part"
    metadata = tmp_path / "model.gguf.part.json"
    assert part.stat().st_size == len(body)
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    assert saved["schemaVersion"] == 2
    assert saved["completedSegments"] == []


def test_manager_installs_bundle_updates_config_and_survives_restart(
    monkeypatch, tmp_path: Path
):
    runtime = tmp_path / "crispasr.dll"
    runtime.write_bytes(b"runtime")
    primary_body = b"primary-model"
    codec_body = b"codec-model"
    locked = {
        "canonical.gguf": _locked("canonical.gguf", primary_body),
        "codec.gguf": _locked("codec.gguf", codec_body, "companion"),
    }
    bundle = RegistryBundle(
        backend="omnivoice",
        license="MIT",
        requires_acceptance=False,
        artifacts=(
            RegistryArtifact("primary", "canonical.gguf", locked["canonical.gguf"].source_url, "~1 GB"),
            RegistryArtifact("companion", "codec.gguf", locked["codec.gguf"].source_url, "~1 GB"),
        ),
    )
    spec = ModelSpec(
        id="omnivoice",
        backend="omnivoice",
        model_path=tmp_path / "missing-primary.gguf",
        codec_path=tmp_path / "missing-codec.gguf",
        sample_rate=24000,
        registry_bundle="omnivoice",
    )
    manifest = tmp_path / "runtime" / "models" / "installed.json"
    cfg = ServerConfig(
        models={"omnivoice": spec},
        lib_path=runtime,
        model_manifest_path=manifest,
    )
    missing_revision = spec.voice_preparation_revision

    def downloader(artifact, destination, progress, cancel_event):
        data = primary_body if artifact.kind == "primary" else codec_body
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        progress(len(data), len(data))

    manager = ModelInstallationManager(
        cfg,
        bundle_resolver=lambda backend, lib: bundle,
        artifact_locker=lambda artifact: locked[artifact.filename],
        downloader=downloader,
    )
    staging, _ = manager._paths("model:omnivoice")
    staging.mkdir(parents=True)
    (staging / "stale-old-quant.gguf").write_bytes(b"stale")
    plan = manager.plan("model:omnivoice")
    assert plan["canonicalBackend"] == "omnivoice"
    assert plan["totalBytes"] == len(primary_body) + len(codec_body)
    assert [artifact["kind"] for artifact in plan["artifacts"]] == [
        "primary",
        "companion",
    ]
    started = manager.start("model:omnivoice")
    completed = _wait(manager, started["id"])
    assert completed["state"] == "completed"
    assert completed["downloadedBytes"] == len(primary_body) + len(codec_body)
    assert spec.installed is True
    assert spec.model_path.name == "canonical.gguf"
    assert spec.codec_path.name == "codec.gguf"
    assert spec.voice_preparation_revision != missing_revision
    assert not (spec.model_path.parent / "stale-old-quant.gguf").exists()

    catalog = tmp_path / "models.toml"
    catalog.write_text(
        "[models.omnivoice]\n"
        "backend='omnivoice'\nregistry_bundle='omnivoice'\n"
        "model='old.gguf'\ncodec='old-codec.gguf'\nsample_rate=24000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SPEECH_SERVER_MODEL_MANIFEST", str(manifest))
    restarted = load_config(catalog).models["omnivoice"]
    assert restarted.installed is True
    assert restarted.model_path == spec.model_path
    assert restarted.codec_path == spec.codec_path


def test_activation_failure_rolls_back_manifest_config_and_verified_staging(
    tmp_path: Path,
):
    runtime = tmp_path / "crispasr.dll"
    runtime.write_bytes(b"runtime")
    body = b"verified-model"
    locked = _locked("model.gguf", body)
    bundle = RegistryBundle(
        backend="m",
        license="MIT",
        requires_acceptance=False,
        artifacts=(RegistryArtifact(
            "primary", "model.gguf", locked.source_url, "~1 KB"
        ),),
    )
    original_path = tmp_path / "original-missing.gguf"
    spec = ModelSpec(
        id="m", backend="m", model_path=original_path,
        sample_rate=24000, registry_bundle="m",
    )
    manifest = tmp_path / "runtime" / "models" / "installed.json"

    def downloader(artifact, destination, progress, cancel):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        progress(len(body), len(body))

    manager = ModelInstallationManager(
        ServerConfig(
            models={"m": spec}, lib_path=runtime, model_manifest_path=manifest
        ),
        bundle_resolver=lambda backend, lib: bundle,
        artifact_locker=lambda artifact: locked,
        downloader=downloader,
    )

    def fail_after_manifest(component, artifacts, final):
        spec.model_path = final / "incorrect.gguf"
        raise RuntimeError("simulated config activation failure")

    manager._apply_paths = fail_after_manifest
    failed = _wait(manager, manager.start("model:m")["id"])
    assert failed["state"] == "failed"
    assert spec.model_path == original_path
    assert not manifest.exists()
    staging, final = manager._paths("model:m")
    assert (staging / "model.gguf").read_bytes() == body
    assert not final.exists()


def test_manager_requires_explicit_license_acceptance(tmp_path: Path):
    runtime = tmp_path / "crispasr.dll"
    runtime.write_bytes(b"runtime")
    spec = ModelSpec(
        id="m",
        backend="m",
        model_path=tmp_path / "missing.gguf",
        sample_rate=24000,
        registry_bundle="m",
    )
    bundle = RegistryBundle(
        backend="m",
        license="restricted-license",
        requires_acceptance=True,
        artifacts=(
            RegistryArtifact(
                "primary",
                "m.gguf",
                "https://huggingface.co/org/repo/resolve/main/m.gguf",
                "~1 GB",
            ),
        ),
    )
    manager = ModelInstallationManager(
        ServerConfig(
            models={"m": spec},
            lib_path=runtime,
            model_manifest_path=tmp_path / "installed.json",
        ),
        bundle_resolver=lambda backend, lib: bundle,
        artifact_locker=lambda artifact: pytest.fail("locker must not run"),
    )
    plan = manager.plan("model:m")
    assert plan["locked"] is False
    assert plan["totalBytes"] is None
    assert plan["artifacts"][0]["approximateSize"] == "~1 GB"
    failed = _wait(manager, manager.start("model:m")["id"])
    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "license_acceptance_required"
    assert failed["requiresLicenseAcceptance"] is True


def test_upscaler_installs_from_the_canonical_voxcpm2_tts_bundle(tmp_path: Path):
    runtime = tmp_path / "crispasr.dll"
    runtime.write_bytes(b"runtime")
    body = b"full-voxcpm2-bundle-with-vae-tensors"
    locked = _locked("voxcpm2-q4_k.gguf", body)
    bundle = RegistryBundle(
        backend="voxcpm2-tts",
        license="",
        requires_acceptance=False,
        artifacts=(
            RegistryArtifact(
                "primary",
                locked.filename,
                locked.source_url,
                "~1.6 GB",
            ),
        ),
    )
    upscaler = ModelSpec(
        id="voxcpm2-vae",
        backend="voxcpm2-vae",
        model_path=tmp_path / "missing-vae.gguf",
        sample_rate=48000,
        registry_bundle="voxcpm2-tts",
    )

    def downloader(artifact, destination, progress, cancel):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        progress(len(body), len(body))

    manager = ModelInstallationManager(
        ServerConfig(
            models={},
            upscaler=upscaler,
            lib_path=runtime,
            model_manifest_path=tmp_path / "runtime" / "models" / "installed.json",
        ),
        bundle_resolver=lambda backend, lib: bundle,
        artifact_locker=lambda artifact: locked,
        downloader=downloader,
    )
    plan = manager.plan("upscaler")
    assert plan["registryBundle"] == "voxcpm2-tts"
    assert plan["canonicalBackend"] == "voxcpm2-tts"
    completed = _wait(manager, manager.start("upscaler")["id"])
    assert completed["state"] == "completed"
    assert upscaler.installed is True
    assert upscaler.backend == "voxcpm2-vae"
    assert upscaler.model_path.name == "voxcpm2-q4_k.gguf"


def test_manager_reports_a_runtime_without_the_bundle_api(tmp_path: Path):
    runtime = tmp_path / "crispasr.dll"
    runtime.write_bytes(b"old-runtime")
    spec = ModelSpec(
        id="m",
        backend="m",
        model_path=tmp_path / "missing.gguf",
        sample_rate=24000,
        registry_bundle="m",
    )

    def unavailable(backend, lib):
        raise CrispBindingError("default-bundle registry API is missing")

    manager = ModelInstallationManager(
        ServerConfig(
            models={"m": spec},
            lib_path=runtime,
            model_manifest_path=tmp_path / "installed.json",
        ),
        bundle_resolver=unavailable,
    )
    with pytest.raises(ModelInstallError) as exc:
        manager.plan("model:m")
    assert exc.value.code == "registry_unavailable"

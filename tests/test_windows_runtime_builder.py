import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_windows_runtime as builder


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_verified_materialization_preserves_existing_file_on_bad_input(
    tmp_path: Path,
):
    destination = tmp_path / "payload.zip"
    destination.write_bytes(b"known-good")
    bad_cache = tmp_path / "bad.zip"
    bad_cache.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        builder._materialize_verified(
            destination,
            url="https://example.test/payload.zip",
            expected_sha256=_digest(b"known-good"),
            cached=bad_cache,
            description="payload",
        )

    assert destination.read_bytes() == b"known-good"
    assert not list(tmp_path.glob(".*.candidate-*"))


def test_windows_overlay_builds_entirely_from_verified_cached_files(
    tmp_path: Path, monkeypatch
):
    python_data = b"embedded-python"
    pip_data = b"get-pip"
    sfw_data = b"sfw"
    python_cache = tmp_path / "python.zip"
    pip_cache = tmp_path / "get-pip.py"
    sfw_cache = tmp_path / "sfw.exe"
    python_cache.write_bytes(python_data)
    pip_cache.write_bytes(pip_data)
    sfw_cache.write_bytes(sfw_data)

    source_manifest = tmp_path / "python-runtime.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "platform": "windows-x86_64",
                "version": "3.13.14",
                "archive": "python.zip",
                "sha256": _digest(python_data),
                "upstreamUrl": "https://example.test/python.zip",
                "license": "PSF-2.0",
                "getPip": {
                    "archive": "get-pip.py",
                    "upstreamUrl": "https://example.test/get-pip.py",
                    "sha256": _digest(pip_data),
                },
            }
        ),
        encoding="utf-8",
    )
    security_manifest = tmp_path / "security-tools.json"
    security_manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "socketFirewall": {
                    "repository": "owner/repo",
                    "tag": "v1",
                    "platforms": {
                        "windows-x86_64": {
                            "asset": "sfw.exe",
                            "sha256": _digest(sfw_data),
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "SOURCE_MANIFEST", source_manifest)
    monkeypatch.setattr(builder, "SECURITY_MANIFEST", security_manifest)

    output = tmp_path / "output"
    builder.build(
        output,
        cached_python=python_cache,
        cached_get_pip=pip_cache,
        cached_sfw=sfw_cache,
    )

    bundle = json.loads(
        (output / "vendor/python/bundle-manifest.json").read_text(encoding="utf-8")
    )
    assert bundle["python"]["sha256"] == _digest(python_data)
    assert bundle["pipBootstrap"]["sha256"] == _digest(pip_data)
    assert (output / "bin/sfw.exe").read_bytes() == sfw_data
    assert (output / "THIRD_PARTY_NOTICES.md").is_file()

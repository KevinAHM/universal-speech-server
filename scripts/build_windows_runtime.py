"""Build the self-contained Windows Python payload for a release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parent.parent
SOURCE_MANIFEST = ROOT / "vendor" / "python" / "python-runtime.json"
SECURITY_MANIFEST = ROOT / "vendor" / "security-tools.json"
USER_AGENT = "omnivoice-windows-runtime-builder/1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.download-{uuid.uuid4().hex}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_value(value: object, description: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"invalid {description} SHA-256")
    return digest


def _filename(value: object, description: str) -> str:
    name = str(value)
    if (
        not name
        or PurePosixPath(name).name != name
        or PureWindowsPath(name).name != name
    ):
        raise RuntimeError(f"unsafe {description} filename: {name!r}")
    return name


def _materialize_verified(
    destination: Path,
    *,
    url: str,
    expected_sha256: str,
    cached: Path | None,
    description: str,
) -> str:
    if cached is None and destination.is_file():
        actual = sha256(destination)
        if actual == expected_sha256:
            return actual
    candidate = destination.with_name(
        f".{destination.name}.candidate-{uuid.uuid4().hex}"
    )
    try:
        if cached is not None:
            if not cached.is_file():
                raise RuntimeError(f"cached {description} is missing: {cached}")
            shutil.copy2(cached, candidate)
        else:
            fetch(url, candidate)
        actual = sha256(candidate)
        if actual != expected_sha256:
            raise RuntimeError(
                f"{description} SHA-256 mismatch: expected "
                f"{expected_sha256}, got {actual}"
            )
        os.replace(candidate, destination)
        return actual
    finally:
        candidate.unlink(missing_ok=True)


def _write_text_atomic(destination: Path, value: str) -> None:
    temporary = destination.with_name(
        f".{destination.name}.write-{uuid.uuid4().hex}"
    )
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build(
    output: Path,
    *,
    cached_python: Path | None = None,
    cached_get_pip: Path | None = None,
    cached_sfw: Path | None = None,
) -> None:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    security = json.loads(SECURITY_MANIFEST.read_text(encoding="utf-8"))
    if source.get("schemaVersion") != 1 or source.get("platform") != "windows-x86_64":
        raise RuntimeError("unsupported Python source manifest")
    if security.get("schemaVersion") != 1:
        raise RuntimeError("unsupported security-tools manifest")
    payload_output = output / "vendor" / "python"
    binary_output = output / "bin"
    payload_output.mkdir(parents=True, exist_ok=True)
    binary_output.mkdir(parents=True, exist_ok=True)

    sfw_config = security["socketFirewall"]
    sfw_platform = sfw_config["platforms"]["windows-x86_64"]
    sfw = binary_output / "sfw.exe"
    sfw_asset = _filename(sfw_platform["asset"], "Socket Firewall")
    sfw_hash = _sha256_value(sfw_platform["sha256"], "Socket Firewall")
    sfw_url = (
        f"https://github.com/{sfw_config['repository']}/releases/download/"
        f"{sfw_config['tag']}/{sfw_asset}"
    )
    _materialize_verified(
        sfw,
        url=sfw_url,
        expected_sha256=sfw_hash,
        cached=cached_sfw,
        description="Socket Firewall",
    )

    python_archive = payload_output / _filename(source["archive"], "Python archive")
    python_hash = _sha256_value(source["sha256"], "embedded Python")
    actual_python_hash = _materialize_verified(
        python_archive,
        url=str(source["upstreamUrl"]),
        expected_sha256=python_hash,
        cached=cached_python,
        description="embedded Python",
    )

    get_pip_config = source["getPip"]
    get_pip = payload_output / _filename(get_pip_config["archive"], "get-pip")
    get_pip_hash = _sha256_value(get_pip_config["sha256"], "get-pip.py")
    actual_get_pip_hash = _materialize_verified(
        get_pip,
        url=str(get_pip_config["upstreamUrl"]),
        expected_sha256=get_pip_hash,
        cached=cached_get_pip,
        description="get-pip.py",
    )

    bundle = {
        "schemaVersion": 1,
        "platform": source["platform"],
        "python": {
            "version": source["version"],
            "archive": python_archive.name,
            "sha256": actual_python_hash,
            "license": source["license"],
            "upstreamUrl": source["upstreamUrl"],
        },
        "pipBootstrap": {
            "archive": get_pip.name,
            "sha256": actual_get_pip_hash,
            "upstreamUrl": get_pip_config["upstreamUrl"],
        },
    }
    _write_text_atomic(
        payload_output / "bundle-manifest.json",
        json.dumps(bundle, indent=2) + "\n",
    )
    _write_text_atomic(
        output / "THIRD_PARTY_NOTICES.md",
        (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "dist" / "windows-runtime"
    )
    parser.add_argument("--python-archive", type=Path)
    parser.add_argument("--get-pip-archive", type=Path)
    parser.add_argument("--sfw-archive", type=Path)
    args = parser.parse_args()
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    bundled_python = ROOT / "vendor" / "python" / _filename(
        source["archive"], "Python archive"
    )
    bundled_get_pip = ROOT / "vendor" / "python" / _filename(
        source["getPip"]["archive"], "get-pip"
    )
    # The release asset keeps its upstream name, but the repository and
    # end-user bundle intentionally install it at the stable launcher path.
    bundled_sfw = ROOT / "bin" / "sfw.exe"
    if args.sfw_archive is not None:
        cached_sfw = args.sfw_archive.resolve()
    elif bundled_sfw.is_file():
        cached_sfw = bundled_sfw.resolve()
    else:
        cached_sfw = None
    build(
        args.output.resolve(),
        cached_python=(args.python_archive or bundled_python).resolve(),
        cached_get_pip=(args.get_pip_archive or bundled_get_pip).resolve(),
        cached_sfw=cached_sfw,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

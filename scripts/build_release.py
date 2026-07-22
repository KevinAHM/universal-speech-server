"""Build a clean public source tree and end-user platform archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

try:
    from scripts import build_windows_runtime
except ModuleNotFoundError:  # Direct execution as scripts/build_release.py.
    import build_windows_runtime  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIST = ROOT / "dist"
PRODUCT = "universal-speech-server"

COMMON_FILES = (
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "abbreviations.txt",
    "crispasr-compat.toml",
    "models.toml",
    "requirements-runtime.txt",
    "vendor/security-tools.json",
)
PUBLIC_SCRIPTS = (
    "scripts/bootstrap_posix.sh",
    "scripts/bootstrap_windows.ps1",
    "scripts/build_release.py",
    "scripts/build_windows_runtime.py",
    "scripts/install_sfw.sh",
)
FORBIDDEN_TOP_LEVEL = {
    ".venv-torch",
    "audio_prompts",
    "crispasr",
    "crispasr-main",
    "crispasr-worktrees",
    "logs",
    "models",
    "runtime",
    "samples",
    "sidon_gguf",
}
FORBIDDEN_SUFFIXES = {
    ".aac",
    ".bin",
    ".ckpt",
    ".flac",
    ".gguf",
    ".m4a",
    ".mp3",
    ".onnx",
    ".ogg",
    ".opus",
    ".pth",
    ".pt",
    ".safetensors",
    ".tflite",
    ".wav",
}
IGNORED_NAMES = {"__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(relative: str, destination: Path) -> None:
    source = ROOT / relative
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"required release file is missing or unsafe: {relative}")
    output = destination / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


def _copy_tree(relative: str, destination: Path) -> None:
    source = ROOT / relative
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"required release directory is missing or unsafe: {relative}")
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if any(part in IGNORED_NAMES for part in rel.parts):
            continue
        if path.is_symlink():
            raise RuntimeError(f"release source contains a symlink: {path}")
        if not path.is_file() or path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        output = destination / relative / rel
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)


def _replace_tree(output: Path, populate) -> Path:
    output = output.resolve()
    allowed_roots = ((ROOT / "dist").resolve(), Path(tempfile.gettempdir()).resolve())
    if not any(output != allowed and output.is_relative_to(allowed) for allowed in allowed_roots):
        raise RuntimeError(
            "release tree output must be below the repository dist directory "
            "or the system temporary directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        populate(staging)
        audit_tree(staging)
        if output.exists():
            if output.is_symlink() or not output.is_dir():
                raise RuntimeError(f"release output is not a directory: {output}")
            shutil.rmtree(output)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def audit_tree(root: Path, *, public_source: bool = False) -> None:
    """Fail closed if private, generated, model, or voice assets entered a tree."""

    offenders: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not relative.parts:
            continue
        top = relative.parts[0]
        top_path = root / top
        if top.casefold() in FORBIDDEN_TOP_LEVEL or (
            top_path.is_dir() and top.casefold().startswith("crispasr")
        ):
            offenders.append(relative.as_posix())
            continue
        if path.is_symlink():
            offenders.append(relative.as_posix())
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(relative.as_posix())
    if public_source and (root / "bin/sfw.exe").exists():
        offenders.append("bin/sfw.exe")
    if offenders:
        preview = ", ".join(sorted(offenders)[:10])
        raise RuntimeError(f"forbidden release content detected: {preview}")


def build_public_tree(output: Path) -> Path:
    def populate(staging: Path) -> None:
        for relative in COMMON_FILES:
            _copy_file(relative, staging)
        for relative in ("start_server.bat", "start_server.sh", *PUBLIC_SCRIPTS):
            _copy_file(relative, staging)
        for relative in (
            "speech_server",
            "tests",
            ".github",
            "release",
            "vendor/python",
        ):
            _copy_tree(relative, staging)
        legacy_test = staging / "tests/test_sonorus_api.py"
        legacy_test.unlink(missing_ok=True)
        shutil.copy2(staging / "release/public.gitignore", staging / ".gitignore")
        shutil.copy2(
            staging / "release/public.gitattributes", staging / ".gitattributes"
        )
        audit_tree(staging, public_source=True)

    return _replace_tree(output, populate)


def build_platform_tree(platform_name: str, output: Path) -> Path:
    if platform_name not in {"windows-x86_64", "linux-x86_64"}:
        raise ValueError(f"unsupported release platform: {platform_name}")

    def populate(staging: Path) -> None:
        for relative in COMMON_FILES:
            _copy_file(relative, staging)
        _copy_tree("speech_server", staging)
        if platform_name == "windows-x86_64":
            _copy_file("start_server.bat", staging)
            _copy_file("scripts/bootstrap_windows.ps1", staging)
            python_config = build_windows_runtime.SOURCE_MANIFEST
            python_archive = ROOT / "vendor/python" / json.loads(
                python_config.read_text(encoding="utf-8")
            )["archive"]
            get_pip = ROOT / "vendor/python/get-pip.py"
            sfw = ROOT / "bin/sfw.exe"
            build_windows_runtime.build(
                staging,
                cached_python=python_archive,
                cached_get_pip=get_pip,
                cached_sfw=sfw if sfw.is_file() else None,
            )
            _copy_file("vendor/python/python-runtime.json", staging)
        else:
            for relative in (
                "start_server.sh",
                "scripts/bootstrap_posix.sh",
                "scripts/install_sfw.sh",
            ):
                _copy_file(relative, staging)

    return _replace_tree(output, populate)


def _zip_tree(source: Path, archive: Path) -> None:
    prefix = f"{PRODUCT}/"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(prefix + relative, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes(), compresslevel=6)


def _tar_tree(source: Path, archive: Path) -> None:
    with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as bundle:
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(source).as_posix()
                info = bundle.gettarinfo(
                    str(path), arcname=str(PurePosixPath(PRODUCT) / relative)
                )
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o755 if path.suffix == ".sh" else 0o644
                with path.open("rb") as stream:
                    bundle.addfile(info, stream)


def build_archives(output: Path, version: str) -> list[Path]:
    safe_version_characters = (
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-_"
    )
    if not version or any(char not in safe_version_characters for char in version):
        raise ValueError("version contains unsafe characters")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archives: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="speech-release-") as temporary:
        staging_root = Path(temporary)
        windows_tree = build_platform_tree("windows-x86_64", staging_root / "windows")
        windows_archive = output / f"{PRODUCT}-{version}-windows-x86_64.zip"
        _zip_tree(windows_tree, windows_archive)
        archives.append(windows_archive)

        linux_tree = build_platform_tree("linux-x86_64", staging_root / "linux")
        linux_archive = output / f"{PRODUCT}-{version}-linux-x86_64.tar.gz"
        _tar_tree(linux_tree, linux_archive)
        archives.append(linux_archive)

    sums = output / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in archives),
        encoding="ascii",
    )
    archives.append(sums)
    return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    public = subparsers.add_parser("public-tree")
    public.add_argument("--output", type=Path, default=DEFAULT_DIST / "public-repo")
    archives = subparsers.add_parser("archives")
    archives.add_argument("--output", type=Path, default=DEFAULT_DIST / "release")
    archives.add_argument("--version", required=True)
    args = parser.parse_args()
    if args.command == "public-tree":
        result = build_public_tree(args.output)
        print(f"Built clean public source tree: {result}")
    else:
        for result in build_archives(args.output, args.version):
            print(f"Built release artifact: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

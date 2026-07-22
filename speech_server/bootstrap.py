"""Dependency-free CrispASR runtime selection and installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
COMPAT_PATH = ROOT / "crispasr-compat.toml"
RUNTIME_DIR = ROOT / "runtime" / "crispasr"
USER_AGENT = "omnivoice-speech-server-bootstrap/1"


class BootstrapError(RuntimeError):
    pass


class NoCompatibleReleaseError(BootstrapError):
    """The target has no eligible release asset, allowing auto-only fallback."""


@dataclass(frozen=True)
class Target:
    id: str
    os: str
    arch: str
    accelerator: str
    asset: str


@dataclass(frozen=True)
class Compatibility:
    repository: str
    minimum_release: tuple[int, ...]
    minimum_release_tag: str
    required_commits: tuple[str, ...]
    targets: tuple[Target, ...]
    binding_commit: str = ""
    binding_version: str = ""

    def target(self, target_id: str) -> Target:
        for target in self.targets:
            if target.id == target_id:
                return target
        raise BootstrapError(f"unknown CrispASR target {target_id!r}")


@dataclass(frozen=True)
class ResolvedAsset:
    tag: str
    target: Target
    name: str
    url: str
    sha256: str
    size: int | None = None


def _version(tag: str) -> tuple[int, ...]:
    value = tag.strip()
    if value.startswith("v"):
        value = value[1:]
    parts = value.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"unsupported release tag {tag!r}")
    return tuple(int(part) for part in parts)


def _is_basename(value: str) -> bool:
    return (
        bool(value)
        and PurePosixPath(value).name == value
        and PureWindowsPath(value).name == value
    )


def _compatibility_revision(compatibility: Compatibility) -> str:
    payload = {
        "repository": compatibility.repository,
        "minimumRelease": compatibility.minimum_release_tag,
        "requiredCommits": compatibility.required_commits,
        "bindingCommit": compatibility.binding_commit,
        "bindingVersion": compatibility.binding_version,
        "targets": [
            {
                "id": target.id,
                "os": target.os,
                "arch": target.arch,
                "accelerator": target.accelerator,
                "asset": target.asset,
            }
            for target in compatibility.targets
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_compatibility(path: Path = COMPAT_PATH) -> Compatibility:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise BootstrapError(f"cannot read compatibility manifest {path}: {exc}") from exc
    if data.get("schema_version") != 1:
        raise BootstrapError(f"unsupported compatibility manifest {path}")
    target_rows = data.get("targets")
    if not isinstance(target_rows, list):
        raise BootstrapError("compatibility manifest has no targets")
    try:
        targets = tuple(
            Target(
                id=str(row["id"]),
                os=str(row["os"]),
                arch=str(row["arch"]),
                accelerator=str(row["accelerator"]),
                asset=str(row["asset"]),
            )
            for row in target_rows
        )
        minimum_tag = str(data["minimum_release"])
        repository = str(data["repository"])
    except (KeyError, TypeError) as exc:
        raise BootstrapError(f"invalid compatibility manifest {path}: {exc}") from exc
    if len({target.id for target in targets}) != len(targets):
        raise BootstrapError("compatibility manifest repeats a target id")
    if any(not target.id or not _is_basename(target.asset) for target in targets):
        raise BootstrapError("compatibility manifest contains an unsafe target")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise BootstrapError("compatibility manifest has an invalid repository")
    try:
        minimum_release = _version(minimum_tag)
    except ValueError as exc:
        raise BootstrapError(str(exc)) from exc
    required_rows = data.get("required_ancestor_commits", [])
    if not isinstance(required_rows, list):
        raise BootstrapError("compatibility manifest has invalid required commits")
    required_commits = tuple(str(value) for value in required_rows)
    if any(
        not re.fullmatch(r"[0-9a-fA-F]{40}", value)
        for value in required_commits
    ):
        raise BootstrapError("compatibility manifest has an invalid required commit")
    if not targets:
        raise BootstrapError("compatibility manifest has no targets")
    binding_commit = str(data.get("binding_commit", ""))
    binding_version = str(data.get("binding_version", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{40}", binding_commit) or not binding_version:
        raise BootstrapError("compatibility manifest has invalid binding provenance")
    return Compatibility(
        repository=repository,
        minimum_release=minimum_release,
        minimum_release_tag=minimum_tag,
        required_commits=required_commits,
        targets=targets,
        binding_commit=binding_commit,
        binding_version=binding_version,
    )


def normalize_os(value: str | None = None) -> str:
    name = (value or platform.system()).lower()
    aliases = {"darwin": "macos", "win32": "windows", "windows": "windows"}
    return aliases.get(name, name)


def normalize_arch(value: str | None = None) -> str:
    machine = (value or platform.machine()).lower().replace("-", "_")
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    return aliases.get(machine, machine)


def _command_output(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return f"{result.stdout}\n{result.stderr}".strip() if result.returncode == 0 else ""


def detect_gpu_names(os_name: str | None = None) -> str:
    current_os = normalize_os(os_name)
    if current_os == "windows":
        output = _command_output(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ]
        )
        return output.lower()
    if current_os == "linux":
        return _command_output(["lspci"]).lower()
    if current_os == "macos":
        return _command_output(["system_profiler", "SPDisplaysDataType"]).lower()
    return ""


def recommend_accelerator(
    os_name: str,
    arch: str,
    *,
    command_exists: Callable[[str], str | None] = shutil.which,
    gpu_names: str | None = None,
) -> str:
    if os_name == "macos" and arch == "arm64":
        return "metal"
    names = (gpu_names if gpu_names is not None else detect_gpu_names(os_name)).lower()
    nvidia_smi = command_exists("nvidia-smi")
    nvidia_responds = bool(
        nvidia_smi
        and _command_output(
            [str(nvidia_smi), "--query-gpu=name", "--format=csv,noheader"]
        )
    )
    # A display-controller name alone does not prove the CUDA runtime is
    # usable; systems can retain an NVIDIA device with a missing/broken driver.
    if nvidia_responds:
        return "nvidia"
    amd_detected = any(token in names for token in ("amd", "radeon"))
    if os_name == "linux" and amd_detected and (
        command_exists("rocminfo") or Path("/dev/kfd").exists()
    ):
        return "amd"
    if any(token in names for token in ("amd", "radeon", "intel", "arc", "graphics")):
        return "vulkan"
    return "cpu"


def compatible_targets(
    compatibility: Compatibility,
    *,
    os_name: str | None = None,
    arch: str | None = None,
) -> list[Target]:
    current_os = normalize_os(os_name)
    current_arch = normalize_arch(arch)
    return [
        target
        for target in compatibility.targets
        if target.os == current_os and target.arch == current_arch
    ]


def choose_target(
    compatibility: Compatibility,
    choice: str = "auto",
    *,
    os_name: str | None = None,
    arch: str | None = None,
    command_exists: Callable[[str], str | None] = shutil.which,
    gpu_names: str | None = None,
) -> Target:
    choice = choice.strip().lower()
    current_os = normalize_os(os_name)
    current_arch = normalize_arch(arch)
    available = compatible_targets(
        compatibility, os_name=current_os, arch=current_arch
    )
    if not available:
        raise BootstrapError(
            f"no CrispASR runtime targets for {current_os}/{current_arch}"
        )
    for target in available:
        if target.id == choice:
            return target
    requested = choice
    if choice == "auto":
        requested = recommend_accelerator(
            current_os,
            current_arch,
            command_exists=command_exists,
            gpu_names=gpu_names,
        )
    aliases = {
        "cuda": "nvidia",
        "nvidia": "nvidia",
        "hip": "amd",
        "amd": "amd",
        "gpu": "vulkan",
        "other": "vulkan",
        "intel": "vulkan",
        "none": "cpu",
    }
    requested = aliases.get(requested, requested)
    for target in available:
        if target.accelerator == requested:
            return target
    if requested == "amd":
        for target in available:
            if target.accelerator == "vulkan":
                return target
    if requested == "cpu":
        for target in available:
            if target.accelerator in {"cpu", "metal"}:
                return target
    raise BootstrapError(
        f"target choice {choice!r} is unavailable for {current_os}/{current_arch}"
    )


class GitHubClient:
    def __init__(self, repository: str, token: str | None = None):
        self.repository = repository
        self.token = token or os.getenv("GITHUB_TOKEN", "").strip() or None
        self.api = f"https://api.github.com/repos/{repository}"

    def _json(self, url: str) -> Any:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BootstrapError(f"GitHub API request failed: {exc}") from exc

    def releases(self) -> list[dict[str, Any]]:
        releases: list[dict[str, Any]] = []
        for page in range(1, 11):
            result = self._json(
                f"{self.api}/releases?per_page=100&page={page}"
            )
            if not isinstance(result, list) or not all(
                isinstance(row, dict) for row in result
            ):
                raise BootstrapError(
                    "GitHub releases response has an unexpected shape"
                )
            releases.extend(result)
            if len(result) < 100:
                return releases
        raise BootstrapError("GitHub repository has too many releases to resolve safely")

    def tag_contains_commit(self, tag: str, commit: str) -> bool:
        comparison = urllib.parse.quote(f"{commit}...{tag}", safe=".")
        result = self._json(f"{self.api}/compare/{comparison}")
        return result.get("status") in {"ahead", "identical"}


def resolve_release(
    compatibility: Compatibility,
    target: Target,
    client: GitHubClient,
) -> ResolvedAsset:
    candidates: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    for release in client.releases():
        if not isinstance(release, dict):
            continue
        if release.get("draft") or release.get("prerelease"):
            continue
        try:
            release_version = _version(str(release.get("tag_name", "")))
        except ValueError:
            continue
        if release_version >= compatibility.minimum_release:
            candidates.append((release_version, release))
    candidates.sort(key=lambda item: item[0], reverse=True)

    for _, release in candidates:
        tag = str(release["tag_name"])
        assets = release.get("assets", [])
        if not isinstance(assets, list):
            continue
        asset = next(
            (
                row
                for row in assets
                if isinstance(row, dict) and row.get("name") == target.asset
            ),
            None,
        )
        if asset is None:
            continue
        if not all(
            client.tag_contains_commit(tag, commit)
            for commit in compatibility.required_commits
        ):
            continue
        digest = str(asset.get("digest") or "")
        sha256 = digest.removeprefix("sha256:").lower()
        if (
            not digest.startswith("sha256:")
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise BootstrapError(
                f"GitHub did not publish a SHA-256 for {tag}/{target.asset}"
            )
        url = asset.get("browser_download_url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise BootstrapError(
                f"GitHub published an invalid URL for {tag}/{target.asset}"
            )
        size_value = asset.get("size")
        try:
            size = int(size_value) if size_value is not None else None
        except (TypeError, ValueError) as exc:
            raise BootstrapError(
                f"GitHub published an invalid size for {tag}/{target.asset}"
            ) from exc
        if size is not None and size < 0:
            raise BootstrapError(
                f"GitHub published an invalid size for {tag}/{target.asset}"
            )
        return ResolvedAsset(
            tag=tag,
            target=target,
            name=target.asset,
            url=url,
            sha256=sha256,
            size=size,
        )
    raise NoCompatibleReleaseError(
        f"no compatible {target.id} release at or after "
        f"{compatibility.minimum_release_tag}; required commits: "
        f"{', '.join(compatibility.required_commits) or 'none'}"
    )


def resolve_with_auto_fallback(
    compatibility: Compatibility,
    selected: Target,
    client: GitHubClient,
    *,
    allow_cpu_fallback: bool,
) -> tuple[Target, ResolvedAsset]:
    """Resolve a target, falling back to CPU only for automatic selection."""

    try:
        return selected, resolve_release(compatibility, selected, client)
    except NoCompatibleReleaseError:
        if not allow_cpu_fallback or selected.accelerator in {"cpu", "cpu-legacy"}:
            raise
        fallback = next(
            (
                target
                for target in compatibility.targets
                if target.os == selected.os
                and target.arch == selected.arch
                and target.accelerator == "cpu"
            ),
            None,
        )
        if fallback is None:
            raise
        print(
            f"No compatible release publishes {selected.id}; "
            f"falling back to {fallback.id}.",
            flush=True,
        )
        return fallback, resolve_release(compatibility, fallback, client)


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise BootstrapError("invalid expected SHA-256")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open(
            "wb"
        ) as output:
            try:
                expected_size = int(response.headers.get("Content-Length") or 0)
            except ValueError:
                expected_size = 0
            downloaded = 0
            next_report = 16 * 1024 * 1024
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if expected_size:
                        print(
                            f"Downloaded {downloaded // (1024 * 1024)} / "
                            f"{expected_size // (1024 * 1024)} MiB...",
                            flush=True,
                        )
                    else:
                        print(
                            f"Downloaded {downloaded // (1024 * 1024)} MiB...",
                            flush=True,
                        )
                    next_report += 16 * 1024 * 1024
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise BootstrapError(f"download failed: {exc}") from exc
    actual = digest.hexdigest()
    if actual != expected_sha256:
        destination.unlink(missing_ok=True)
        raise BootstrapError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}"
        )


def _safe_relative(name: str) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not pure.parts
        or pure == PurePosixPath(".")
        or pure.is_absolute()
        or ".." in pure.parts
        or any(":" in part for part in pure.parts)
    ):
        raise BootstrapError(f"archive member escapes install directory: {name}")
    return Path(*pure.parts)


def _safe_symlink_target(member_name: str, link_name: str) -> str:
    normalized_link = link_name.replace("\\", "/")
    link = PurePosixPath(normalized_link)
    if (
        not link.parts
        or link == PurePosixPath(".")
        or link.is_absolute()
        or any(":" in part for part in link.parts)
    ):
        raise BootstrapError(f"unsafe TAR link: {member_name}")
    combined = PurePosixPath(member_name.replace("\\", "/")).parent / link
    depth = 0
    for part in combined.parts:
        depth += -1 if part == ".." else (0 if part == "." else 1)
        if depth < 0:
            raise BootstrapError(f"unsafe TAR link: {member_name}")
    return normalized_link


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                relative = _safe_relative(info.filename)
                mode = info.external_attr >> 16
                if (mode & 0o170000) == 0o120000:
                    raise BootstrapError(f"ZIP symlink is not allowed: {info.filename}")
                target = destination / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as bundle:
            links: list[tuple[tarfile.TarInfo, Path, str]] = []
            for member in bundle.getmembers():
                relative = _safe_relative(member.name)
                if member.isdev() or member.isfifo():
                    raise BootstrapError(f"unsafe TAR member: {member.name}")
                if member.issym() or member.islnk():
                    safe_link = _safe_symlink_target(member.name, member.linkname)
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(member)
                    if source is None:
                        raise BootstrapError(f"cannot read TAR member: {member.name}")
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    if os.name != "nt":
                        target.chmod(member.mode & 0o777)
                elif member.issym() or member.islnk():
                    links.append((member, target, safe_link))
                else:
                    raise BootstrapError(f"unsupported TAR member: {member.name}")
            for member, target, safe_link in links:
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.issym():
                    target.symlink_to(safe_link)
                else:
                    source = destination / _safe_relative(member.linkname)
                    if not source.is_file():
                        raise BootstrapError(
                            f"TAR hard-link target is unavailable: {member.linkname}"
                        )
                    os.link(source, target)
        return
    raise BootstrapError(f"unsupported runtime archive: {archive.name}")


def _find_library(root: Path, os_name: str) -> Path:
    names = {
        "windows": ("crispasr.dll",),
        "linux": ("libcrispasr.so",),
        "macos": ("libcrispasr.dylib",),
    }.get(os_name, ())
    candidates = [path for name in names for path in root.rglob(name)]
    if not candidates:
        raise BootstrapError(f"downloaded archive contains no libcrispasr for {os_name}")
    return min(candidates, key=lambda path: (len(path.parts), len(str(path))))


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def install_runtime(
    resolved: ResolvedAsset,
    *,
    destination: Path = RUNTIME_DIR,
    downloader: Callable[[str, Path, str], None] = _download,
    compatibility_revision: str | None = None,
) -> Path:
    if not _is_basename(resolved.name):
        raise BootstrapError(f"unsafe runtime asset name: {resolved.name}")
    destination = Path(os.path.abspath(destination.expanduser()))
    if destination.is_symlink():
        raise BootstrapError(f"runtime destination must not be a symlink: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crispasr-install-", dir=parent) as temp_name:
        temporary = Path(temp_name)
        archive = temporary / resolved.name
        staging = temporary / "staging"
        downloader(resolved.url, archive, resolved.sha256)
        print("Download verified; extracting CrispASR runtime...", flush=True)
        _extract_archive(archive, staging)
        library = _find_library(staging, resolved.target.os)
        manifest = {
            "schemaVersion": 1,
            "tag": resolved.tag,
            "target": resolved.target.id,
            "asset": resolved.name,
            "sha256": resolved.sha256,
            "library": library.relative_to(staging).as_posix(),
        }
        if compatibility_revision is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", compatibility_revision):
                raise BootstrapError("invalid compatibility revision")
            manifest["compatibilityRevision"] = compatibility_revision
        (staging / "installed.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        backup = parent / f".{destination.name}.previous"
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        if backup.exists():
            _remove_path(backup)
        if destination.exists():
            os.replace(destination, backup)
        try:
            print("Activating CrispASR runtime...", flush=True)
            os.replace(staging, destination)
        except Exception:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            _remove_path(backup)
    return destination / manifest["library"]


def _installed_target(
    destination: Path = RUNTIME_DIR,
    compatibility_revision: str | None = None,
) -> str | None:
    manifest = destination / "installed.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schemaVersion") != 1 or not isinstance(data.get("target"), str):
        return None
    if (
        compatibility_revision is not None
        and data.get("compatibilityRevision") != compatibility_revision
    ):
        return None
    relative = data.get("library")
    if not isinstance(relative, str) or not relative:
        return None
    root = destination.resolve()
    library = (root / relative).resolve()
    if not library.is_relative_to(root) or not library.is_file():
        return None
    return str(data["target"])


def _interactive_choice(targets: Iterable[Target], recommended: Target) -> str:
    options = list(targets)
    print(f"1. Auto (recommended: {recommended.id})")
    for index, target in enumerate(options, 2):
        print(f"{index}. {target.id}")
    print(f"{len(options) + 2}. Cancel")
    try:
        selected = int(input("Select a CrispASR runtime: ").strip())
    except (EOFError, ValueError) as exc:
        raise BootstrapError("invalid runtime selection") from exc
    if selected == 1:
        return "auto"
    if 2 <= selected < len(options) + 2:
        return options[selected - 2].id
    raise BootstrapError("setup cancelled")


def setup_native(args: argparse.Namespace) -> int:
    if os.getenv("SPEECH_SERVER_LIB", "").strip():
        print("SPEECH_SERVER_LIB is set; keeping the explicit native runtime override.")
        return 0
    compatibility = load_compatibility(Path(args.compat))
    compatibility_revision = _compatibility_revision(compatibility)
    current = _installed_target(Path(args.destination), compatibility_revision)
    requested_target = args.target.strip().lower()
    allow_cpu_fallback = requested_target == "auto"
    if requested_target == "auto" and not args.update:
        platform_target_ids = {
            target.id for target in compatible_targets(compatibility)
        }
        if current in platform_target_ids:
            print(f"CrispASR runtime already installed: {current}")
            return 0
    if requested_target == "prompt":
        recommended = choose_target(compatibility, "auto")
        interactive_choice = _interactive_choice(
            compatible_targets(compatibility), recommended
        )
        allow_cpu_fallback = interactive_choice == "auto"
        selected = (
            recommended
            if allow_cpu_fallback
            else compatibility.target(interactive_choice)
        )
    else:
        selected = choose_target(compatibility, requested_target)
    if current == selected.id and not args.update:
        print(f"CrispASR runtime already installed: {current}")
        return 0
    print(f"Selected CrispASR target: {selected.id}", flush=True)
    print("Checking GitHub for the newest compatible release...", flush=True)
    client = GitHubClient(compatibility.repository)
    selected, resolved = resolve_with_auto_fallback(
        compatibility,
        selected,
        client,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    print(f"Resolved {resolved.tag}: {resolved.name}", flush=True)
    if args.dry_run:
        return 0
    size_label = (
        f" ({resolved.size / (1024 * 1024):.1f} MiB)"
        if resolved.size is not None
        else ""
    )
    print(f"Downloading {resolved.name}{size_label}...", flush=True)
    library = install_runtime(
        resolved,
        destination=Path(args.destination),
        compatibility_revision=compatibility_revision,
    )
    print(f"Installed CrispASR runtime: {library}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup-native", help="install a compatible libcrispasr")
    setup.add_argument(
        "--target",
        default="auto",
        help="auto, prompt, CPU/GPU family, or an exact target id",
    )
    setup.add_argument("--update", action="store_true")
    setup.add_argument("--dry-run", action="store_true")
    setup.add_argument("--compat", default=str(COMPAT_PATH))
    setup.add_argument("--destination", default=str(RUNTIME_DIR))
    setup.set_defaults(func=setup_native)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BootstrapError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

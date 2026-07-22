"""Verified, resumable installation of CrispASR registry model bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from .config import AlignerSpec, ModelSpec, ServerConfig
from .crisp import (
    CrispBindingError,
    RegistryArtifact,
    RegistryBundle,
    registry_default_bundle,
)


USER_AGENT = "omnivoice-speech-server-model-installer/1"
CHUNK_BYTES = 1024 * 1024
PARALLEL_MIN_BYTES = 64 * 1024 * 1024
PARALLEL_SEGMENT_BYTES = 32 * 1024 * 1024
PARALLEL_WORKERS = 8
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
ACTIVE_STATES = {"queued", "resolving", "downloading", "installing"}


class ModelInstallError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class InstallCancelled(ModelInstallError):
    def __init__(self):
        super().__init__("cancelled", "model installation was cancelled")


@dataclass(frozen=True)
class LockedArtifact:
    kind: str
    filename: str
    source_url: str
    url: str
    repository: str
    revision: str
    sha256: str
    size: int

    def manifest(self, relative_path: str) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "filename": self.filename,
            "path": relative_path,
            "sourceUrl": self.source_url,
            "url": self.url,
            "repository": self.repository,
            "revision": self.revision,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass
class InstallJob:
    id: str
    component: str
    registry_bundle: str
    state: str = "queued"
    created_at: str = field(default_factory=lambda: _now())
    updated_at: str = field(default_factory=lambda: _now())
    canonical_backend: str | None = None
    license: str = ""
    requires_license_acceptance: bool = False
    current_artifact: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    error_code: str | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "component": self.component,
            "registryBundle": self.registry_bundle,
            "state": self.state,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "canonicalBackend": self.canonical_backend,
            "license": self.license or None,
            "requiresLicenseAcceptance": self.requires_license_acceptance,
            "currentArtifact": self.current_artifact,
            "artifacts": list(self.artifacts),
            "downloadedBytes": self.downloaded_bytes,
            "totalBytes": self.total_bytes,
            "error": (
                {"code": self.error_code, "message": self.error}
                if self.error_code and self.error
                else None
            ),
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_basename(value: str) -> bool:
    return (
        bool(value)
        and PurePosixPath(value).name == value
        and PureWindowsPath(value).name == value
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        if new.scheme != "https":
            raise ModelInstallError(
                "download_failed", "artifact download attempted an insecure redirect"
            )
        if old.netloc != new.netloc:
            redirected.remove_header("Authorization")
        return redirected


def _request_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _head_without_redirect(url: str) -> Any:
    request = urllib.request.Request(url, headers=_request_headers(), method="HEAD")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        return opener.open(request, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            return exc
        if exc.code in {401, 403}:
            raise ModelInstallError(
                "artifact_access_denied",
                "Hugging Face denied access; accept the model license and set HF_TOKEN",
            ) from exc
        if exc.code == 404:
            raise ModelInstallError("artifact_not_found", f"registry artifact not found: {url}") from exc
        raise ModelInstallError(
            "artifact_resolution_failed", f"artifact metadata request failed ({exc.code}): {url}"
        ) from exc
    except OSError as exc:
        raise ModelInstallError(
            "artifact_resolution_failed", f"artifact metadata request failed: {exc}"
        ) from exc


def _open_download(request: urllib.request.Request):
    return urllib.request.build_opener(_SafeRedirect).open(request, timeout=60)


def _header(headers: Any, name: str) -> str:
    return str(headers.get(name) or "").strip().strip('"').lower()


def _huggingface_url(url: str, filename: str) -> tuple[str, str, str, str]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise ModelInstallError(
            "unverifiable_artifact",
            "CrispASR registry artifacts must use verifiable huggingface.co HTTPS URLs",
        )
    segments = [urllib.parse.unquote(value) for value in parsed.path.split("/") if value]
    if len(segments) < 5 or segments[2] != "resolve":
        raise ModelInstallError("unverifiable_artifact", f"unsupported Hugging Face URL: {url}")
    owner, repo, _, revision, *file_parts = segments
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in (owner, repo)):
        raise ModelInstallError("unverifiable_artifact", f"unsafe Hugging Face repository URL: {url}")
    if not file_parts or file_parts[-1] != filename or not _is_basename(filename):
        raise ModelInstallError(
            "unverifiable_artifact", f"registry filename does not match artifact URL: {url}"
        )
    return owner, repo, revision, "/".join(file_parts)


def lock_huggingface_artifact(artifact: RegistryArtifact) -> LockedArtifact:
    """Pin a registry URL to an immutable HF revision and published LFS digest."""
    owner, repo, requested_revision, file_path = _huggingface_url(
        artifact.url, artifact.filename
    )
    response = _head_without_redirect(artifact.url)
    try:
        revision = _header(response.headers, "X-Repo-Commit")
        sha256 = _header(response.headers, "X-Linked-Etag")
        size_text = _header(response.headers, "X-Linked-Size")
    finally:
        response.close()
    if not REVISION_RE.fullmatch(revision):
        raise ModelInstallError(
            "unverifiable_artifact", f"Hugging Face did not publish a repository revision for {artifact.filename}"
        )
    if REVISION_RE.fullmatch(requested_revision.lower()) and requested_revision.lower() != revision:
        raise ModelInstallError(
            "artifact_revision_mismatch", f"Hugging Face resolved an unexpected revision for {artifact.filename}"
        )
    if not SHA256_RE.fullmatch(sha256):
        raise ModelInstallError(
            "unverifiable_artifact", f"Hugging Face did not publish an LFS SHA-256 for {artifact.filename}"
        )
    try:
        size = int(size_text)
    except ValueError as exc:
        raise ModelInstallError(
            "unverifiable_artifact", f"Hugging Face did not publish an exact size for {artifact.filename}"
        ) from exc
    if size <= 0:
        raise ModelInstallError("unverifiable_artifact", f"invalid size for {artifact.filename}")

    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in file_path.split("/"))
    pinned_url = f"https://huggingface.co/{owner}/{repo}/resolve/{revision}/{quoted_path}"
    pinned = _head_without_redirect(pinned_url)
    try:
        if (
            _header(pinned.headers, "X-Repo-Commit") != revision
            or _header(pinned.headers, "X-Linked-Etag") != sha256
            or _header(pinned.headers, "X-Linked-Size") != str(size)
        ):
            raise ModelInstallError(
                "artifact_revision_mismatch",
                f"immutable metadata changed while resolving {artifact.filename}",
            )
    finally:
        pinned.close()
    return LockedArtifact(
        kind=artifact.kind,
        filename=artifact.filename,
        source_url=artifact.url,
        url=pinned_url,
        repository=f"{owner}/{repo}",
        revision=revision,
        sha256=sha256,
        size=size,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _download_locked_artifact_sequential(
    artifact: LockedArtifact,
    destination: Path,
    progress: Callable[[int, int], None],
    cancel_event: threading.Event,
) -> None:
    """Download one locked artifact, retaining a validated partial for resume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    metadata = destination.with_name(destination.name + ".part.json")
    expected_metadata = {
        "schemaVersion": 1,
        "url": artifact.url,
        "sha256": artifact.sha256,
        "size": artifact.size,
    }
    if destination.is_file():
        if destination.stat().st_size == artifact.size and _hash_file(destination) == artifact.sha256:
            progress(artifact.size, artifact.size)
            return
        destination.unlink()

    resume = 0
    if part.is_file() and metadata.is_file():
        try:
            saved = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            saved = None
        if saved == expected_metadata and part.stat().st_size <= artifact.size:
            resume = part.stat().st_size
        else:
            part.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
    elif part.exists() or metadata.exists():
        part.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
    _atomic_json(metadata, expected_metadata)

    digest = hashlib.sha256()
    if resume:
        with part.open("rb") as existing:
            while chunk := existing.read(CHUNK_BYTES):
                if cancel_event.is_set():
                    raise InstallCancelled()
                digest.update(chunk)
        progress(resume, artifact.size)
    if resume == artifact.size:
        if digest.hexdigest() != artifact.sha256:
            part.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            raise ModelInstallError("checksum_mismatch", f"SHA-256 mismatch for {artifact.filename}")
        os.replace(part, destination)
        metadata.unlink(missing_ok=True)
        return

    headers = _request_headers()
    if resume:
        headers["Range"] = f"bytes={resume}-"
    request = urllib.request.Request(artifact.url, headers=headers)
    try:
        response = _open_download(request)
    except (OSError, urllib.error.HTTPError) as exc:
        raise ModelInstallError("download_failed", f"download failed for {artifact.filename}: {exc}") from exc
    try:
        final_url = urllib.parse.urlsplit(response.geturl())
        if final_url.scheme != "https":
            raise ModelInstallError("download_failed", "artifact download redirected away from HTTPS")
        status = getattr(response, "status", response.getcode())
        if resume and status != 206:
            resume = 0
            digest = hashlib.sha256()
        if resume:
            content_range = str(response.headers.get("Content-Range") or "")
            if not content_range.startswith(f"bytes {resume}-"):
                raise ModelInstallError(
                    "download_failed", f"server returned an invalid resume range for {artifact.filename}"
                )
        mode = "ab" if resume else "wb"
        downloaded = resume
        with part.open(mode) as output:
            while chunk := response.read(CHUNK_BYTES):
                if cancel_event.is_set():
                    raise InstallCancelled()
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded > artifact.size:
                    raise ModelInstallError(
                        "size_mismatch", f"download exceeded the locked size for {artifact.filename}"
                    )
                progress(downloaded, artifact.size)
    finally:
        response.close()
    if downloaded != artifact.size:
        raise ModelInstallError(
            "size_mismatch",
            f"downloaded {downloaded} bytes for {artifact.filename}; expected {artifact.size}",
        )
    actual = digest.hexdigest()
    if actual != artifact.sha256:
        part.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
        raise ModelInstallError(
            "checksum_mismatch",
            f"SHA-256 mismatch for {artifact.filename}: expected {artifact.sha256}, got {actual}",
        )
    os.replace(part, destination)
    metadata.unlink(missing_ok=True)


def _validate_download_response(
    response: Any,
    artifact: LockedArtifact,
    *,
    expected_start: int | None = None,
    expected_end: int | None = None,
) -> None:
    final_url = urllib.parse.urlsplit(response.geturl())
    if final_url.scheme != "https":
        raise ModelInstallError(
            "download_failed", "artifact download redirected away from HTTPS"
        )
    if expected_start is None or expected_end is None:
        return
    status = getattr(response, "status", response.getcode())
    expected_range = f"bytes {expected_start}-{expected_end}/{artifact.size}"
    if status != 206 or str(response.headers.get("Content-Range") or "") != expected_range:
        raise ModelInstallError(
            "download_failed",
            f"server returned an invalid range for {artifact.filename}",
        )


def _parallel_ranges_supported(artifact: LockedArtifact) -> bool:
    request = urllib.request.Request(
        artifact.url,
        headers={**_request_headers(), "Range": "bytes=0-0"},
    )
    try:
        response = _open_download(request)
    except (OSError, urllib.error.HTTPError) as exc:
        raise ModelInstallError(
            "download_failed",
            f"range probe failed for {artifact.filename}: {exc}",
        ) from exc
    try:
        final_url = urllib.parse.urlsplit(response.geturl())
        if final_url.scheme != "https":
            raise ModelInstallError(
                "download_failed", "artifact download redirected away from HTTPS"
            )
        status = getattr(response, "status", response.getcode())
        if status == 200:
            return False
        content_range = str(response.headers.get("Content-Range") or "")
        if status != 206 or content_range != f"bytes 0-0/{artifact.size}":
            raise ModelInstallError(
                "download_failed",
                f"server returned an invalid range probe for {artifact.filename}",
            )
        return True
    finally:
        response.close()


def _artifact_segments(artifact: LockedArtifact) -> list[tuple[int, int, int]]:
    return [
        (index, start, min(start + PARALLEL_SEGMENT_BYTES, artifact.size) - 1)
        for index, start in enumerate(
            range(0, artifact.size, PARALLEL_SEGMENT_BYTES)
        )
    ]


def _prepare_parallel_partial(
    artifact: LockedArtifact,
    part: Path,
    metadata: Path,
    segments: list[tuple[int, int, int]],
) -> tuple[dict[str, Any], set[int]]:
    identity = {
        "schemaVersion": 2,
        "url": artifact.url,
        "sha256": artifact.sha256,
        "size": artifact.size,
        "segmentSize": PARALLEL_SEGMENT_BYTES,
    }
    saved = None
    if metadata.is_file():
        try:
            saved = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass

    completed: set[int] = set()
    valid_indices = {index for index, _, _ in segments}
    core_matches = isinstance(saved, dict) and all(
        saved.get(key) == value for key, value in identity.items()
    )
    if core_matches and part.is_file() and part.stat().st_size == artifact.size:
        values = saved.get("completedSegments")
        if (
            isinstance(values, list)
            and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            and len(values) == len(set(values))
            and set(values).issubset(valid_indices)
        ):
            completed = set(values)
        else:
            part.unlink(missing_ok=True)
    elif (
        isinstance(saved, dict)
        and saved.get("schemaVersion") == 1
        and all(saved.get(key) == identity[key] for key in ("url", "sha256", "size"))
        and part.is_file()
        and part.stat().st_size <= artifact.size
    ):
        # Preserve every complete segment from the old sequential-prefix
        # format.  A partially written segment is deliberately downloaded
        # again from its boundary before it can be marked complete.
        sequential_bytes = part.stat().st_size
        completed = {
            index for index, _, end in segments if end + 1 <= sequential_bytes
        }
    else:
        part.unlink(missing_ok=True)

    if not part.is_file():
        part.parent.mkdir(parents=True, exist_ok=True)
        with part.open("wb") as output:
            output.truncate(artifact.size)
    elif part.stat().st_size != artifact.size:
        with part.open("r+b") as output:
            output.truncate(artifact.size)

    value = {**identity, "completedSegments": sorted(completed)}
    _atomic_json(metadata, value)
    return identity, completed


def _download_locked_artifact_parallel(
    artifact: LockedArtifact,
    destination: Path,
    progress: Callable[[int, int], None],
    cancel_event: threading.Event,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    metadata = destination.with_name(destination.name + ".part.json")
    if destination.is_file():
        if (
            destination.stat().st_size == artifact.size
            and _hash_file(destination) == artifact.sha256
        ):
            progress(artifact.size, artifact.size)
            return
        destination.unlink()

    if not _parallel_ranges_supported(artifact):
        part.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
        return _download_locked_artifact_sequential(
            artifact, destination, progress, cancel_event
        )

    segments = _artifact_segments(artifact)
    identity, completed = _prepare_parallel_partial(
        artifact, part, metadata, segments
    )
    segment_sizes = {index: end - start + 1 for index, start, end in segments}
    completed_bytes = sum(segment_sizes[index] for index in completed)
    progress(completed_bytes, artifact.size)
    pending = [segment for segment in segments if segment[0] not in completed]
    stop_event = threading.Event()
    state_lock = threading.Lock()
    in_flight: dict[int, int] = {}

    def fetch_segment(segment: tuple[int, int, int]) -> int:
        nonlocal completed_bytes
        index, start, end = segment
        if cancel_event.is_set():
            raise InstallCancelled()
        if stop_event.is_set():
            raise ModelInstallError(
                "download_failed",
                f"parallel download stopped for {artifact.filename}",
            )
        request = urllib.request.Request(
            artifact.url,
            headers={**_request_headers(), "Range": f"bytes={start}-{end}"},
        )
        try:
            response = _open_download(request)
        except (OSError, urllib.error.HTTPError) as exc:
            stop_event.set()
            raise ModelInstallError(
                "download_failed",
                f"range download failed for {artifact.filename}: {exc}",
            ) from exc
        received = 0
        try:
            _validate_download_response(
                response,
                artifact,
                expected_start=start,
                expected_end=end,
            )
            with part.open("r+b") as output:
                output.seek(start)
                while chunk := response.read(CHUNK_BYTES):
                    if cancel_event.is_set():
                        raise InstallCancelled()
                    if stop_event.is_set():
                        raise ModelInstallError(
                            "download_failed",
                            f"parallel download stopped for {artifact.filename}",
                        )
                    received += len(chunk)
                    if received > segment_sizes[index]:
                        raise ModelInstallError(
                            "size_mismatch",
                            f"range exceeded its locked size for {artifact.filename}",
                        )
                    output.write(chunk)
                    with state_lock:
                        in_flight[index] = received
                        progress(
                            completed_bytes + sum(in_flight.values()),
                            artifact.size,
                        )
        except Exception:
            stop_event.set()
            raise
        finally:
            response.close()
        if received != segment_sizes[index]:
            stop_event.set()
            raise ModelInstallError(
                "size_mismatch",
                f"range downloaded {received} bytes for {artifact.filename}; "
                f"expected {segment_sizes[index]}",
            )
        with state_lock:
            in_flight.pop(index, None)
            completed.add(index)
            completed_bytes += segment_sizes[index]
            _atomic_json(
                metadata,
                {**identity, "completedSegments": sorted(completed)},
            )
            progress(completed_bytes + sum(in_flight.values()), artifact.size)
        return index

    if pending:
        try:
            with ThreadPoolExecutor(
                max_workers=min(PARALLEL_WORKERS, len(pending)),
                thread_name_prefix="model-range",
            ) as executor:
                futures = [executor.submit(fetch_segment, segment) for segment in pending]
                for future in as_completed(futures):
                    future.result()
        except InstallCancelled:
            raise
        except ModelInstallError:
            if cancel_event.is_set():
                raise InstallCancelled()
            raise
        except Exception as exc:
            raise ModelInstallError(
                "download_failed",
                f"parallel download failed for {artifact.filename}: {exc}",
            ) from exc

    if cancel_event.is_set():
        raise InstallCancelled()
    if completed != {index for index, _, _ in segments}:
        raise ModelInstallError(
            "download_failed", f"parallel download is incomplete for {artifact.filename}"
        )
    actual = _hash_file(part)
    if actual != artifact.sha256:
        part.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
        raise ModelInstallError(
            "checksum_mismatch",
            f"SHA-256 mismatch for {artifact.filename}: expected "
            f"{artifact.sha256}, got {actual}",
        )
    os.replace(part, destination)
    metadata.unlink(missing_ok=True)
    progress(artifact.size, artifact.size)


def download_locked_artifact(
    artifact: LockedArtifact,
    destination: Path,
    progress: Callable[[int, int], None],
    cancel_event: threading.Event,
) -> None:
    """Download a locked artifact with verified resume and parallel ranges."""
    if artifact.size < PARALLEL_MIN_BYTES:
        return _download_locked_artifact_sequential(
            artifact, destination, progress, cancel_event
        )
    return _download_locked_artifact_parallel(
        artifact, destination, progress, cancel_event
    )


BundleResolver = Callable[[str, Path], RegistryBundle | None]
ArtifactLocker = Callable[[RegistryArtifact], LockedArtifact]
ArtifactDownloader = Callable[
    [LockedArtifact, Path, Callable[[int, int], None], threading.Event], None
]


class ModelInstallationManager:
    def __init__(
        self,
        cfg: ServerConfig,
        *,
        bundle_resolver: BundleResolver | None = None,
        artifact_locker: ArtifactLocker = lock_huggingface_artifact,
        downloader: ArtifactDownloader = download_locked_artifact,
    ):
        self.cfg = cfg
        self.manifest_path = cfg.model_manifest_path.resolve()
        self._bundle_resolver = bundle_resolver or (
            lambda backend, lib_path: registry_default_bundle(backend, lib_path=lib_path)
        )
        self._artifact_locker = artifact_locker
        self._downloader = downloader
        self._jobs: dict[str, InstallJob] = {}
        self._active: dict[str, str] = {}
        self._bundles: dict[str, RegistryBundle] = {}
        self._locked_plans: dict[str, tuple[LockedArtifact, ...]] = {}
        self._lock = threading.RLock()

    def _spec(self, component: str) -> ModelSpec | AlignerSpec:
        if component.startswith("model:"):
            model_id = component.removeprefix("model:")
            spec = self.cfg.models.get(model_id)
        elif component == "upscaler":
            spec = self.cfg.upscaler
        elif component == "aligner":
            spec = self.cfg.aligner
        else:
            spec = None
        if spec is None:
            raise ModelInstallError("component_not_found", f"unknown install component {component!r}")
        return spec

    def start(self, component: str, *, accept_license: bool = False) -> dict[str, Any]:
        spec = self._spec(component)
        if not spec.registry_bundle:
            raise ModelInstallError(
                "component_not_installable", f"{component!r} has no CrispASR registry bundle"
            )
        with self._lock:
            active_id = self._active.get(component)
            if active_id and self._jobs[active_id].state in ACTIVE_STATES:
                return self._jobs[active_id].response()
            job = InstallJob(
                id=uuid.uuid4().hex,
                component=component,
                registry_bundle=spec.registry_bundle,
            )
            self._jobs[job.id] = job
            if spec.installed:
                job.state = "completed"
                job.downloaded_bytes = spec.component_bytes
                job.total_bytes = spec.component_bytes
                job.updated_at = _now()
                return job.response()
            self._active[component] = job.id
            thread = threading.Thread(
                target=self._run,
                args=(job, accept_license),
                name=f"model-install-{job.id[:8]}",
                daemon=True,
            )
            thread.start()
            return job.response()

    def plan(self, component: str) -> dict[str, Any]:
        spec = self._spec(component)
        if not spec.registry_bundle:
            raise ModelInstallError(
                "component_not_installable", f"{component!r} has no CrispASR registry bundle"
            )
        bundle = self._resolve_bundle(component, spec.registry_bundle)
        artifacts = (
            () if bundle.requires_acceptance else self._lock_bundle(component, bundle)
        )
        return {
            "component": component,
            "registryBundle": spec.registry_bundle,
            "canonicalBackend": bundle.backend,
            "license": bundle.license or None,
            "requiresLicenseAcceptance": bundle.requires_acceptance,
            "locked": bool(artifacts),
            "totalBytes": (
                sum(artifact.size for artifact in artifacts) if artifacts else None
            ),
            "artifacts": (
                [
                    {
                        "kind": artifact.kind,
                        "filename": artifact.filename,
                        "repository": artifact.repository,
                        "revision": artifact.revision,
                        "sha256": artifact.sha256,
                        "size": artifact.size,
                        "url": artifact.url,
                    }
                    for artifact in artifacts
                ]
                if artifacts
                else [
                    {
                        "kind": artifact.kind,
                        "filename": artifact.filename,
                        "approximateSize": artifact.approx_size or None,
                    }
                    for artifact in bundle.artifacts
                ]
            ),
        }

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ModelInstallError("job_not_found", f"unknown installation job {job_id!r}")
            return job.response()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.response() for job in self._jobs.values()]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ModelInstallError("job_not_found", f"unknown installation job {job_id!r}")
            if job.state in ACTIVE_STATES:
                job.cancel_event.set()
                job.updated_at = _now()
            return job.response()

    def component_state(self, component: str) -> dict[str, Any]:
        spec = self._spec(component)
        with self._lock:
            job_id = self._active.get(component)
            job = self._jobs.get(job_id) if job_id else None
            return {
                "installed": spec.installed,
                "installable": bool(spec.registry_bundle),
                "registryBundle": spec.registry_bundle,
                "job": job.response() if job and job.state in ACTIVE_STATES else None,
            }

    def _set_job(self, job: InstallJob, **changes: Any) -> None:
        with self._lock:
            for name, value in changes.items():
                setattr(job, name, value)
            job.updated_at = _now()

    def _run(self, job: InstallJob, accept_license: bool) -> None:
        try:
            self._set_job(job, state="resolving")
            bundle = self._resolve_bundle(job.component, job.registry_bundle)
            self._set_job(
                job,
                canonical_backend=bundle.backend,
                license=bundle.license,
                requires_license_acceptance=bundle.requires_acceptance,
            )
            if bundle.requires_acceptance and not accept_license:
                raise ModelInstallError(
                    "license_acceptance_required",
                    f"installation requires acceptance of {bundle.license or 'the model license'}",
                )
            locked_tuple = self._lock_bundle(job.component, bundle)
            locked = list(locked_tuple)
            total = sum(artifact.size for artifact in locked)
            self._set_job(
                job,
                state="downloading",
                total_bytes=total,
                artifacts=[
                    {
                        "kind": artifact.kind,
                        "filename": artifact.filename,
                        "repository": artifact.repository,
                        "revision": artifact.revision,
                        "sha256": artifact.sha256,
                        "size": artifact.size,
                        "url": artifact.url,
                    }
                    for artifact in locked
                ],
            )
            staging, final = self._paths(job.component)
            self._prepare_staging(staging, locked)
            completed_before = 0
            for artifact in locked:
                if job.cancel_event.is_set():
                    raise InstallCancelled()
                self._set_job(job, current_artifact=artifact.filename)

                def progress(current: int, _total: int, base: int = completed_before) -> None:
                    self._set_job(job, downloaded_bytes=base + current)

                self._downloader(
                    artifact,
                    staging / artifact.filename,
                    progress,
                    job.cancel_event,
                )
                completed_before += artifact.size
                self._set_job(job, downloaded_bytes=completed_before)
            if job.cancel_event.is_set():
                raise InstallCancelled()
            self._set_job(job, state="installing", current_artifact=None)
            self._activate(job, bundle, locked, staging, final)
            self._set_job(job, state="completed", downloaded_bytes=total)
        except InstallCancelled as exc:
            self._set_job(job, state="cancelled", error_code=exc.code, error=str(exc))
        except ModelInstallError as exc:
            self._set_job(job, state="failed", error_code=exc.code, error=str(exc))
        except Exception as exc:
            self._set_job(job, state="failed", error_code="installation_failed", error=str(exc))
        finally:
            with self._lock:
                if self._active.get(job.component) == job.id and job.state not in ACTIVE_STATES:
                    self._active.pop(job.component, None)

    def _resolve_bundle(
        self, component: str, registry_bundle_name: str
    ) -> RegistryBundle:
        with self._lock:
            cached = self._bundles.get(component)
            if cached is not None:
                return cached
        if self.cfg.lib_path is None or not self.cfg.lib_path.is_file():
            raise ModelInstallError(
                "runtime_unavailable",
                "libcrispasr must be installed before resolving model bundles",
            )
        try:
            bundle = self._bundle_resolver(registry_bundle_name, self.cfg.lib_path)
        except CrispBindingError as exc:
            raise ModelInstallError("registry_unavailable", str(exc)) from exc
        if bundle is None:
            raise ModelInstallError(
                "registry_bundle_not_found",
                f"the loaded CrispASR runtime has no bundle {registry_bundle_name!r}",
            )
        self._validate_bundle(bundle)
        spec = self._spec(component)
        if (
            isinstance(spec, ModelSpec)
            and spec.codec_path is not None
            and not any(artifact.kind == "companion" for artifact in bundle.artifacts)
        ):
            raise ModelInstallError(
                "invalid_registry_bundle",
                "the canonical bundle is missing the catalog model's companion artifact",
            )
        with self._lock:
            self._bundles[component] = bundle
        return bundle

    def _lock_bundle(
        self, component: str, bundle: RegistryBundle
    ) -> tuple[LockedArtifact, ...]:
        with self._lock:
            cached = self._locked_plans.get(component)
            if cached is not None:
                return cached
        locked = tuple(self._artifact_locker(artifact) for artifact in bundle.artifacts)
        with self._lock:
            self._locked_plans[component] = locked
        return locked

    @staticmethod
    def _validate_bundle(bundle: RegistryBundle) -> None:
        filenames = [artifact.filename for artifact in bundle.artifacts]
        kinds = [artifact.kind for artifact in bundle.artifacts]
        if (
            not bundle.backend
            or not filenames
            or kinds.count("primary") != 1
            or kinds[0] != "primary"
            or kinds.count("companion") > 1
            or any(kind not in {"primary", "companion", "extra"} for kind in kinds)
            or any(not _is_basename(filename) for filename in filenames)
            or len(set(filenames)) != len(filenames)
            or any(not artifact.url.startswith("https://") for artifact in bundle.artifacts)
        ):
            raise ModelInstallError("invalid_registry_bundle", "CrispASR returned an invalid bundle")

    def _paths(self, component: str) -> tuple[Path, Path]:
        root = self.manifest_path.parent
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", component).strip(".-") or "component"
        slug = f"{slug}-{hashlib.sha256(component.encode()).hexdigest()[:10]}"
        return root / ".staging" / slug, root / "artifacts" / slug

    def _prepare_staging(
        self, staging: Path, artifacts: list[LockedArtifact]
    ) -> None:
        root = self.manifest_path.parent.resolve()
        if staging.is_symlink():
            raise ModelInstallError("unsafe_install_path", "model staging path is a symlink")
        staging.mkdir(parents=True, exist_ok=True)
        if not staging.resolve().is_relative_to(root):
            raise ModelInstallError("unsafe_install_path", "model staging path escapes its root")
        allowed = {
            name
            for artifact in artifacts
            for name in (
                artifact.filename,
                artifact.filename + ".part",
                artifact.filename + ".part.json",
            )
        }
        for child in staging.iterdir():
            if child.is_symlink() or child.name not in allowed:
                self._remove_path(child)

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"schemaVersion": 1, "installations": {}}
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelInstallError("manifest_invalid", f"cannot read model manifest: {exc}") from exc
        if (
            not isinstance(value, dict)
            or value.get("schemaVersion") != 1
            or not isinstance(value.get("installations"), dict)
        ):
            raise ModelInstallError("manifest_invalid", "unsupported model installation manifest")
        return value

    def _activate(
        self,
        job: InstallJob,
        bundle: RegistryBundle,
        artifacts: list[LockedArtifact],
        staging: Path,
        final: Path,
    ) -> None:
        for artifact in artifacts:
            candidate = staging / artifact.filename
            if (
                not candidate.is_file()
                or candidate.stat().st_size != artifact.size
                or _hash_file(candidate) != artifact.sha256
            ):
                raise ModelInstallError(
                    "verification_failed", f"staged artifact failed verification: {artifact.filename}"
                )
        final.parent.mkdir(parents=True, exist_ok=True)
        backup = final.with_name(final.name + ".previous")
        if backup.exists():
            if final.exists():
                self._remove_path(backup)
            else:
                os.replace(backup, final)
        if final.exists():
            os.replace(final, backup)
        spec = self._spec(job.component)
        previous_paths = (
            spec.model_path,
            spec.codec_path if isinstance(spec, ModelSpec) else None,
            list(spec.extra_paths),
        )
        missing_cache = object()
        previous_revision = spec.__dict__.get(
            "voice_preparation_revision", missing_cache
        )
        try:
            os.replace(staging, final)
            with self._lock:
                manifest_existed = self.manifest_path.is_file()
                previous_manifest = self._read_manifest()
                manifest = copy.deepcopy(previous_manifest)
                manifest_written = False
                try:
                    rows = []
                    for artifact in artifacts:
                        path = final / artifact.filename
                        rows.append(
                            artifact.manifest(
                                path.relative_to(self.manifest_path.parent).as_posix()
                            )
                        )
                    manifest["installations"][job.component] = {
                        "catalogId": job.component.removeprefix("model:"),
                        "registryBundle": job.registry_bundle,
                        "canonicalBackend": bundle.backend,
                        "license": bundle.license,
                        "installedAt": _now(),
                        "artifacts": rows,
                    }
                    _atomic_json(self.manifest_path, manifest)
                    manifest_written = True
                    self._apply_paths(job.component, artifacts, final)
                except Exception:
                    spec.model_path = previous_paths[0]
                    if isinstance(spec, ModelSpec):
                        spec.codec_path = previous_paths[1]
                    spec.extra_paths = previous_paths[2]
                    if previous_revision is missing_cache:
                        spec.__dict__.pop("voice_preparation_revision", None)
                    else:
                        spec.__dict__["voice_preparation_revision"] = previous_revision
                    if manifest_written:
                        if manifest_existed:
                            _atomic_json(self.manifest_path, previous_manifest)
                        else:
                            self.manifest_path.unlink(missing_ok=True)
                    raise
        except Exception:
            if final.exists():
                if staging.exists():
                    self._remove_path(staging)
                os.replace(final, staging)
            if backup.exists():
                os.replace(backup, final)
            raise
        if backup.exists():
            self._remove_path(backup)

    def _apply_paths(
        self, component: str, artifacts: list[LockedArtifact], final: Path
    ) -> None:
        spec = self._spec(component)
        primary = [artifact for artifact in artifacts if artifact.kind == "primary"]
        companion = [artifact for artifact in artifacts if artifact.kind == "companion"]
        extras = [artifact for artifact in artifacts if artifact.kind == "extra"]
        spec.model_path = final / primary[0].filename
        if isinstance(spec, ModelSpec):
            spec.codec_path = final / companion[0].filename if companion else None
        spec.extra_paths = [final / artifact.filename for artifact in extras]
        spec.__dict__.pop("voice_preparation_revision", None)

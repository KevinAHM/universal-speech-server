import argparse
import hashlib
import io
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from speech_server.bootstrap import (
    BootstrapError,
    Compatibility,
    GitHubClient,
    ResolvedAsset,
    Target,
    _extract_archive,
    _compatibility_revision,
    _installed_target,
    choose_target,
    install_runtime,
    load_compatibility,
    normalize_arch,
    normalize_os,
    resolve_release,
    resolve_with_auto_fallback,
    setup_native,
)


WINDOWS_TARGETS = (
    Target(
        "windows-x86_64-cuda",
        "windows",
        "x86_64",
        "nvidia",
        "libcrispasr-windows-x86_64-cuda.tar.gz",
    ),
    Target(
        "windows-x86_64-vulkan",
        "windows",
        "x86_64",
        "vulkan",
        "libcrispasr-windows-x86_64-vulkan.tar.gz",
    ),
    Target(
        "windows-x86_64-cpu",
        "windows",
        "x86_64",
        "cpu",
        "libcrispasr-windows-x86_64.tar.gz",
    ),
)


def compatibility(*targets: Target) -> Compatibility:
    return Compatibility(
        repository="owner/repo",
        minimum_release=(0, 8, 14),
        minimum_release_tag="v0.8.14",
        required_commits=("required",),
        targets=targets or WINDOWS_TARGETS,
    )


def test_checked_in_compatibility_manifest_loads():
    result = load_compatibility()
    assert result.repository == "CrispStrobe/CrispASR"
    assert result.minimum_release == (0, 8, 18)
    assert "104d85be1d78cdf9559b16ee93ab5aa93d35b480" in result.required_commits
    assert "de3409c67abf2c6cd620f0c2a291f472bc232622" in result.required_commits
    assert result.target("windows-x86_64-cuda").accelerator == "nvidia"
    upstream = tomllib.loads(
        (Path(__file__).parent.parent / "speech_server/_vendor/crispasr/UPSTREAM.toml")
        .read_text(encoding="utf-8")
    )
    assert result.binding_commit == upstream["commit"]
    assert result.binding_version == upstream["binding_version"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "schema_version = 1\nrepository = 'bad'\n"
            "minimum_release = 'v1'\ntargets = []\n",
            "invalid repository",
        ),
        (
            "schema_version = 1\nrepository = 'o/r'\n"
            "minimum_release = 'new'\ntargets = []\n",
            "unsupported release tag",
        ),
        (
            "schema_version = 1\nrepository = 'o/r'\nminimum_release = 'v1'\n"
            "required_ancestor_commits = ['short']\ntargets = []\n",
            "invalid required commit",
        ),
        (
            "schema_version = 1\nrepository = 'o/r'\nminimum_release = 'v1'\n"
            "[[targets]]\nid='same'\nos='linux'\narch='x86_64'\naccelerator='cpu'\nasset='a.tar.gz'\n"
            "[[targets]]\nid='same'\nos='linux'\narch='x86_64'\naccelerator='cpu'\nasset='b.tar.gz'\n",
            "repeats a target id",
        ),
        (
            "schema_version = 1\nrepository = 'o/r'\nminimum_release = 'v1'\n"
            "[[targets]]\nid='bad'\nos='linux'\narch='x86_64'\naccelerator='cpu'\nasset='../bad.tar.gz'\n",
            "unsafe target",
        ),
    ],
)
def test_compatibility_manifest_rejects_malformed_values(
    tmp_path: Path, body: str, message: str
):
    path = tmp_path / "compat.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(BootstrapError, match=message):
        load_compatibility(path)


def test_python_and_socket_payloads_are_pinned():
    root = Path(__file__).resolve().parent.parent
    python_payload = json.loads(
        (root / "vendor/python/python-runtime.json").read_text(encoding="utf-8")
    )
    bundle_payload = json.loads(
        (root / "vendor/python/bundle-manifest.json").read_text(encoding="utf-8")
    )
    security = json.loads(
        (root / "vendor/security-tools.json").read_text(encoding="utf-8")
    )
    assert python_payload["version"] == "3.13.14"
    assert len(python_payload["sha256"]) == 64
    assert len(python_payload["getPip"]["sha256"]) == 64
    python_archive = root / "vendor/python" / python_payload["archive"]
    get_pip = root / "vendor/python" / python_payload["getPip"]["archive"]
    assert hashlib.sha256(python_archive.read_bytes()).hexdigest() == python_payload["sha256"]
    assert hashlib.sha256(get_pip.read_bytes()).hexdigest() == python_payload["getPip"]["sha256"]
    assert bundle_payload["python"]["archive"] == python_archive.name
    assert bundle_payload["python"]["sha256"] == python_payload["sha256"]
    assert bundle_payload["pipBootstrap"]["archive"] == get_pip.name
    assert bundle_payload["pipBootstrap"]["sha256"] == python_payload["getPip"]["sha256"]
    sfw = security["socketFirewall"]
    assert sfw["tag"].startswith("v")
    assert len(sfw["platforms"]["windows-x86_64"]["sha256"]) == 64
    uv = security["uv"]
    assert uv["tag"]
    installer = (root / "scripts/install_sfw.sh").read_text(encoding="utf-8")
    assert f"tag={sfw['tag']}" in installer
    assert f"uv_tag={uv['tag']}" in installer
    for target in sfw["platforms"].values():
        if not target["asset"].startswith("sfw-free-windows"):
            assert target["asset"] in installer
            assert target["sha256"] in installer
    for target in uv["platforms"].values():
        assert target["asset"] in installer
        assert target["sha256"] in installer
    posix_bootstrap = (root / "scripts/bootstrap_posix.sh").read_text(
        encoding="utf-8"
    )
    assert 'managed_python_version=3.13.14' in posix_bootstrap
    assert '"$uv_command" python install "$managed_python_version"' in posix_bootstrap
    assert '"$sfw_command" "$uv_command" python install' not in posix_bootstrap
    assert '"$sfw_command" "$uv_command" venv --seed' in posix_bootstrap
    assert "--managed-python" in posix_bootstrap
    assert "--no-python-downloads" in posix_bootstrap
    assert "find_base_python" not in posix_bootstrap
    assert 'rm -rf "$root/runtime/python"' not in posix_bootstrap
    assert '"$uv_command" pip install --python "$runtime_python" pip' in posix_bootstrap
    assert "UV_SYSTEM_CERTS" not in posix_bootstrap
    assert "speech_server.native_check" in posix_bootstrap
    assert "speech_server.gpu_select" in posix_bootstrap
    posix_launcher = (root / "start_server.sh").read_text(encoding="utf-8")
    assert "--gpu" in posix_launcher
    windows_bootstrap = (root / "scripts/bootstrap_windows.ps1").read_text(
        encoding="utf-8"
    )
    assert windows_bootstrap.count("-FilePath $sfwExe") >= 2
    assert "$getPip," in windows_bootstrap
    assert '"-m", "pip", "install"' in windows_bootstrap
    assert '$ErrorActionPreference = "Continue"' in windows_bootstrap
    assert "speech_server.gpu_select" in windows_bootstrap
    windows_launcher = (root / "start_server.bat").read_text(encoding="utf-8")
    assert "--gpu" in windows_launcher
    assert "$exitCode = Invoke-NativeCommand" not in windows_bootstrap
    assert "Piping here" in windows_bootstrap
    native_bootstrap = (root / "speech_server/bootstrap.py").read_text(
        encoding="utf-8"
    )
    assert "Checking GitHub for the newest compatible release" in native_bootstrap
    assert "Download verified; extracting CrispASR runtime" in native_bootstrap


def test_start_wrappers_bootstrap_before_running_without_separate_setup_scripts():
    root = Path(__file__).resolve().parent.parent
    windows_start = (root / "start_server.bat").read_text(encoding="utf-8")
    posix_start = (root / "start_server.sh").read_text(encoding="utf-8")
    assert "bootstrap_windows.ps1" in windows_start
    assert "-RunServer" in windows_start
    assert 'bootstrap_posix.sh" run' in posix_start
    assert not (root / "setup_server.bat").exists()
    assert not (root / "setup_server.sh").exists()


@pytest.mark.parametrize(
    ("os_name", "arch", "expected"),
    [
        ("Windows", "AMD64", ("windows", "x86_64")),
        ("Darwin", "aarch64", ("macos", "arm64")),
        ("linux", "x86_64", ("linux", "x86_64")),
    ],
)
def test_platform_normalization(os_name, arch, expected):
    assert (normalize_os(os_name), normalize_arch(arch)) == expected


def test_auto_target_prefers_nvidia_when_nvidia_smi_responds(monkeypatch):
    monkeypatch.setattr(
        "speech_server.bootstrap._command_output", lambda command: "NVIDIA GeForce"
    )
    target = choose_target(
        compatibility(),
        os_name="windows",
        arch="amd64",
        command_exists=lambda name: "nvidia-smi" if name == "nvidia-smi" else None,
        gpu_names="NVIDIA GeForce",
    )
    assert target.id == "windows-x86_64-cuda"


def test_auto_target_does_not_assume_cuda_from_device_name_alone():
    target = choose_target(
        compatibility(),
        os_name="windows",
        arch="x86_64",
        command_exists=lambda name: None,
        gpu_names="NVIDIA GeForce",
    )
    assert target.id == "windows-x86_64-cpu"


def test_auto_target_uses_vulkan_for_detected_amd_gpu_on_windows():
    target = choose_target(
        compatibility(),
        os_name="windows",
        arch="x86_64",
        command_exists=lambda name: None,
        gpu_names="AMD Radeon RX 7900 XTX",
    )
    assert target.id == "windows-x86_64-vulkan"


def test_auto_target_falls_back_to_cpu_when_detection_is_uncertain():
    target = choose_target(
        compatibility(),
        os_name="windows",
        arch="x86_64",
        command_exists=lambda name: None,
        gpu_names="",
    )
    assert target.id == "windows-x86_64-cpu"


def test_target_choice_is_case_insensitive():
    target = choose_target(
        compatibility(), "CUDA", os_name="windows", arch="x86_64"
    )
    assert target.id == "windows-x86_64-cuda"


class FakeGitHubClient:
    def __init__(self, releases, ancestry):
        self._releases = releases
        self.ancestry = ancestry
        self.compared = []

    def releases(self):
        return self._releases

    def tag_contains_commit(self, tag, commit):
        self.compared.append((tag, commit))
        return self.ancestry.get((tag, commit), False)


def release(tag, target, *, digest="a" * 64):
    return {
        "tag_name": tag,
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": target.asset,
                "browser_download_url": f"https://example.test/{target.asset}",
                "digest": f"sha256:{digest}",
                "size": 123,
            }
        ],
    }


def test_release_resolution_skips_tag_without_required_commit():
    target = WINDOWS_TARGETS[0]
    client = FakeGitHubClient(
        [release("v0.8.16", target), release("v0.8.15", target, digest="b" * 64)],
        {("v0.8.15", "required"): True},
    )
    resolved = resolve_release(compatibility(), target, client)
    assert resolved.tag == "v0.8.15"
    assert resolved.sha256 == "b" * 64


def test_release_resolution_uses_older_eligible_release_when_newest_lacks_target():
    target = WINDOWS_TARGETS[0]
    newest_without_target = release("v0.8.16", WINDOWS_TARGETS[1])
    client = FakeGitHubClient(
        [newest_without_target, release("v0.8.15", target, digest="b" * 64)],
        {("v0.8.15", "required"): True},
    )
    resolved = resolve_release(compatibility(), target, client)
    assert resolved.tag == "v0.8.15"
    assert resolved.sha256 == "b" * 64


def test_auto_resolution_falls_back_to_cpu_when_gpu_asset_is_unpublished(capsys):
    vulkan = WINDOWS_TARGETS[1]
    cpu = WINDOWS_TARGETS[2]
    client = FakeGitHubClient(
        [release("v0.8.16", cpu)],
        {("v0.8.16", "required"): True},
    )
    selected, resolved = resolve_with_auto_fallback(
        compatibility(vulkan, cpu),
        vulkan,
        client,
        allow_cpu_fallback=True,
    )
    assert selected == cpu
    assert resolved.target == cpu
    assert "falling back to windows-x86_64-cpu" in capsys.readouterr().out


def test_explicit_gpu_resolution_does_not_fall_back_to_cpu():
    vulkan = WINDOWS_TARGETS[1]
    cpu = WINDOWS_TARGETS[2]
    client = FakeGitHubClient(
        [release("v0.8.16", cpu)],
        {("v0.8.16", "required"): True},
    )
    with pytest.raises(BootstrapError, match="no compatible"):
        resolve_with_auto_fallback(
            compatibility(vulkan, cpu),
            vulkan,
            client,
            allow_cpu_fallback=False,
        )


def test_github_release_listing_follows_pagination(monkeypatch):
    client = GitHubClient("owner/repo")
    pages = [[{"tag_name": "v1"}] * 100, [{"tag_name": "v0"}]]
    requested = []

    def fake_json(url):
        requested.append(url)
        return pages[len(requested) - 1]

    monkeypatch.setattr(client, "_json", fake_json)
    assert len(client.releases()) == 101
    assert requested[-1].endswith("page=2")


def test_release_resolution_rejects_missing_github_digest():
    target = WINDOWS_TARGETS[0]
    row = release("v0.8.16", target, digest="")
    row["assets"][0]["digest"] = None
    client = FakeGitHubClient([row], {("v0.8.16", "required"): True})
    with pytest.raises(BootstrapError, match="did not publish a SHA-256"):
        resolve_release(compatibility(), target, client)


def test_release_resolution_rejects_malformed_asset_metadata():
    target = WINDOWS_TARGETS[0]
    row = release("v0.8.16", target)
    row["assets"][0]["browser_download_url"] = "http://example.test/insecure"
    client = FakeGitHubClient([row], {("v0.8.16", "required"): True})
    with pytest.raises(BootstrapError, match="invalid URL"):
        resolve_release(compatibility(), target, client)


def test_zip_extraction_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", b"bad")
    with pytest.raises(BootstrapError, match="escapes"):
        _extract_archive(archive, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_extraction_rejects_escaping_symlink(tmp_path: Path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as bundle:
        link = tarfile.TarInfo("bundle/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../escaped.txt"
        bundle.addfile(link)
    with pytest.raises(BootstrapError, match="unsafe TAR link"):
        _extract_archive(archive, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_hard_link_can_precede_its_target(tmp_path: Path):
    archive = tmp_path / "links.tar"
    with tarfile.open(archive, "w") as bundle:
        link = tarfile.TarInfo("bundle/copy.dll")
        link.type = tarfile.LNKTYPE
        link.linkname = "bundle/crispasr.dll"
        bundle.addfile(link)
        payload = b"library"
        target = tarfile.TarInfo("bundle/crispasr.dll")
        target.size = len(payload)
        bundle.addfile(target, io.BytesIO(payload))
    output = tmp_path / "output"
    _extract_archive(archive, output)
    assert (output / "bundle/copy.dll").read_bytes() == b"library"


def test_runtime_install_writes_manifest_and_replaces_old_runtime(tmp_path: Path):
    target = WINDOWS_TARGETS[2]
    source_archive = tmp_path / "source.tar.gz"
    payload = tmp_path / "payload" / "bin"
    payload.mkdir(parents=True)
    (payload / "crispasr.dll").write_bytes(b"new-library")
    with tarfile.open(source_archive, "w:gz") as bundle:
        bundle.add(payload.parent, arcname="bundle")

    destination = tmp_path / "runtime" / "crispasr"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("old", encoding="utf-8")
    resolved = ResolvedAsset(
        tag="v0.8.16",
        target=target,
        name="runtime.tar.gz",
        url="https://example.test/runtime.tar.gz",
        sha256="f" * 64,
    )

    def downloader(url, output, expected):
        output.write_bytes(source_archive.read_bytes())

    library = install_runtime(
        resolved,
        destination=destination,
        downloader=downloader,
        compatibility_revision="c" * 64,
    )
    assert library.read_bytes() == b"new-library"
    manifest = json.loads((destination / "installed.json").read_text(encoding="utf-8"))
    assert manifest["tag"] == "v0.8.16"
    assert manifest["target"] == target.id
    assert manifest["compatibilityRevision"] == "c" * 64
    assert manifest["library"] == "bundle/bin/crispasr.dll"
    assert not (destination / "old.txt").exists()


def test_runtime_install_rejects_unsafe_asset_name_before_download(tmp_path: Path):
    resolved = ResolvedAsset(
        tag="v0.8.16",
        target=WINDOWS_TARGETS[2],
        name="../runtime.tar.gz",
        url="https://example.test/runtime.tar.gz",
        sha256="f" * 64,
    )
    with pytest.raises(BootstrapError, match="unsafe runtime asset name"):
        install_runtime(
            resolved,
            destination=tmp_path / "runtime",
            downloader=lambda *args: pytest.fail("downloader was called"),
        )


def test_installed_target_requires_library_inside_runtime(tmp_path: Path):
    destination = tmp_path / "runtime"
    destination.mkdir()
    (destination / "installed.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "target": "windows-x86_64-cpu",
                "library": "../outside.dll",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "outside.dll").write_bytes(b"outside")
    assert _installed_target(destination) is None


def test_installed_target_rejects_an_old_compatibility_revision(tmp_path: Path):
    destination = tmp_path / "runtime"
    destination.mkdir()
    (destination / "crispasr.dll").write_bytes(b"runtime")
    (destination / "installed.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "target": "windows-x86_64-cpu",
                "library": "crispasr.dll",
                "compatibilityRevision": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert _installed_target(destination, "b" * 64) is None


def test_auto_setup_keeps_an_existing_platform_runtime(
    tmp_path: Path, monkeypatch
):
    compatibility_data = load_compatibility()
    revision = _compatibility_revision(compatibility_data)
    destination = tmp_path / "runtime"
    destination.mkdir()
    library = destination / "crispasr.dll"
    library.write_bytes(b"runtime")
    (destination / "installed.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "target": "windows-x86_64-cuda",
                "library": "crispasr.dll",
                "compatibilityRevision": revision,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("SPEECH_SERVER_LIB", raising=False)
    monkeypatch.setattr("speech_server.bootstrap.normalize_os", lambda value=None: "windows")
    monkeypatch.setattr("speech_server.bootstrap.normalize_arch", lambda value=None: "x86_64")
    monkeypatch.setattr(
        "speech_server.bootstrap.choose_target",
        lambda *args, **kwargs: pytest.fail("hardware was redetected"),
    )
    args = argparse.Namespace(
        compat=str(Path(__file__).parent.parent / "crispasr-compat.toml"),
        destination=str(destination),
        target="auto",
        update=False,
        dry_run=False,
    )
    assert setup_native(args) == 0

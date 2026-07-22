import tarfile
from pathlib import Path

import pytest

from scripts import build_release


def test_public_tree_contains_only_publishable_sources(tmp_path: Path):
    output = build_release.build_public_tree(tmp_path / "public")

    assert (output / "speech_server/bootstrap.py").is_file()
    assert (output / "speech_server/_vendor/crispasr/LICENSE").is_file()
    assert (output / "vendor/python/python-3.13.14-embed-amd64.zip").is_file()
    assert (output / ".github/workflows/ci.yml").is_file()
    assert not (output / "audio_prompts").exists()
    assert not (output / "models").exists()
    assert not (output / "runtime").exists()
    assert not (output / "bin/sfw.exe").exists()
    assert not (output / "tests/test_sonorus_api.py").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "audio_prompts/voice.wav",
        "Audio_Prompts/voice.wav",
        "models/model.gguf",
        "weights/model.safetensors",
    ],
)
def test_release_audit_rejects_voice_and_model_assets(tmp_path: Path, relative: str):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"private")
    with pytest.raises(RuntimeError, match="forbidden release content"):
        build_release.audit_tree(tmp_path)


def test_release_tree_refuses_a_broad_filesystem_target(tmp_path: Path):
    with pytest.raises(RuntimeError, match="output must be below"):
        build_release.build_public_tree(Path(tmp_path.anchor))


def test_linux_archive_is_rooted_and_excludes_private_assets(tmp_path: Path):
    tree = build_release.build_platform_tree("linux-x86_64", tmp_path / "linux")
    archive = tmp_path / "linux.tar.gz"
    build_release._tar_tree(tree, archive)
    with tarfile.open(archive, "r:gz") as bundle:
        names = bundle.getnames()
        launcher = bundle.getmember("universal-speech-server/start_server.sh")
    assert names
    assert all(name.startswith("universal-speech-server/") for name in names)
    assert launcher.mode & 0o111
    assert not any("audio_prompts" in name or "/models/" in name for name in names)

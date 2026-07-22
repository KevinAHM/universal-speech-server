# Universal Speech Server

A Torch-free local speech server powered by
[CrispASR](https://github.com/CrispStrobe/CrispASR). It provides TTS and ASR
over HTTP and WebSocket APIs for Sonorus and other local clients.

The server downloads a compatible native CrispASR runtime automatically. Model
weights and voice samples are not bundled.

## Download and run

Download the archive for your platform from
[GitHub Releases](https://github.com/KevinAHM/universal-speech-server/releases),
extract it, and run the launcher from that directory.

### Windows

```powershell
.\start_server.bat
```

The Windows archive includes pinned copies of Python and Socket Firewall Free.
The launcher prepares everything else and then starts the server.

### Linux

```sh
sh ./scripts/install_sfw.sh
./start_server.sh
```

The first command installs pinned, hash-verified copies of Socket Firewall Free
and `uv` under `bin/`. The launcher then creates an isolated Python environment
and starts the server.

The default address is `http://127.0.0.1:8100`.

## Runtime selection

Automatic selection is the default. You can request a backend explicitly:

```powershell
.\start_server.bat nvidia
.\start_server.bat amd
.\start_server.bat intel
.\start_server.bat cpu
```

Use `./start_server.sh` instead on Linux. To choose a particular GPU:

```powershell
.\start_server.bat nvidia --gpu 0
```

Supported release targets include Windows x86-64 with CUDA, Vulkan, or CPU and
Linux x86-64 with CUDA, HIP, or CPU. When an automatically detected GPU backend
has no compatible published runtime, automatic setup falls back to CPU. An
explicit unavailable backend fails with an explanation.

Set `SPEECH_SERVER_UPDATE=1` before launching to re-detect hardware and resolve
the newest compatible CrispASR release.

## Models and voices

The release intentionally contains no models or voice recordings. Models can
be installed through the server's authenticated model-install endpoints, and
Sonorus can upload voice references. Downloads use immutable revisions, exact
sizes, and published SHA-256 hashes.

Until a model is installed, it remains visible in server capabilities but is
reported as unavailable.

## Configuration

Common environment variables:

- `SPEECH_SERVER_TOKEN` - authentication token.
- `SPEECH_SERVER_HOST` - bind address; defaults to `127.0.0.1`.
- `SPEECH_SERVER_PORT` - port; defaults to `8100`.
- `SPEECH_SERVER_GPU_DEVICE` - GPU index selected at startup.
- `SPEECH_SERVER_UPDATE=1` - re-resolve the native runtime.
- `HF_TOKEN` - access token for gated Hugging Face model files.

Advanced paths and catalog settings use the additional `SPEECH_SERVER_*`
variables defined in `speech_server/config.py`.

## Security and licenses

Python package operations are routed through Socket Firewall. Bundled and
downloaded runtime tools are pinned and hash-verified. Model files are fetched
directly only after their immutable metadata has been resolved and verified.

The project is licensed under Apache-2.0. Vendored and distributed components
are documented in `THIRD_PARTY_NOTICES.md`.

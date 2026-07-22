# Universal Speech Server

Torch-free FastAPI/WebSocket speech server backed by the vendored CrispASR
ctypes bindings. Sonorus connects with its Universal TTS and Speech/ASR
providers through one shared authenticated connection.

## Setup

For normal installation, download the archive for your platform from the
repository's GitHub Releases page rather than the automatically generated
source archive. Release archives are assembled from an explicit allowlist and
never contain model weights, voice prompts, development checkouts, logs, or
local runtime state.

Windows release bundles contain pinned copies of CPython 3.13.14 and Socket
Firewall Free. The start script verifies both before use, bootstraps pip from a
bundled verified `get-pip.py`, and runs both that bootstrap and every dependency
installation through Socket Firewall. It installs or updates any missing
runtime pieces before launching the server:

```powershell
.\start_server.bat
```

On Linux or macOS, run the preparation helper once. It downloads the exact
pinned Socket Firewall Free and `uv` release archives, verifies both SHA-256
values, and installs them locally under `bin/`. The bootstrap refuses to
download Python dependencies unless SFW is available:

```sh
sh ./scripts/install_sfw.sh
sh ./start_server.sh
```

The release intentionally starts without voices or model weights. Sonorus can
upload voice references and invoke the authenticated model-install endpoints;
model downloads are locked to immutable revisions, exact sizes, and published
SHA-256 values. Do not copy a development `audio_prompts/`, `models/`, or
`runtime/` directory into a public distribution.

### Release platforms

The tagged release workflow currently publishes and smoke-tests Windows
x86-64 and Linux x86-64 archives. Windows supports CUDA, Vulkan, and CPU
release assets. Linux supports CUDA, HIP, and CPU assets. CrispASR v0.8.18 does
not publish a Linux `libcrispasr` Vulkan asset: automatic selection therefore
falls back to CPU on a Linux Vulkan-only system, while an explicit Vulkan
request fails clearly instead of silently installing a different backend.
Linux CI exercises Ubuntu 24.04; other compatible glibc distributions use the
same archive but are not separately certified yet.

Pass `nvidia`, `amd`, `intel`, `cpu`, `prompt`, or an exact target ID to the
start script. The default is a conservative automatic choice; use `prompt` for
an interactive selection with an automatically detected recommendation. Once
installed, an automatic start retains that valid platform runtime so a transient
hardware probe failure cannot silently downgrade it. Set
`SPEECH_SERVER_UPDATE=1` to re-probe and resolve the newest compatible asset.
macOS release setup currently supports Apple silicon only. For a developer
environment, dependency installs must still be wrapped:

```powershell
sfw python -m pip install -r requirements-runtime.txt
```

After selecting the native runtime, the start scripts enumerate GPUs for that
runtime. A sole CUDA or Vulkan GPU is selected automatically; when several are
available, an interactive terminal asks which device to use. Non-interactive
launches select the first enumerated device. Pass `--gpu INDEX` to skip the
prompt and select a device explicitly:

```powershell
.\start_server.bat --gpu 0
```

```sh
./start_server.sh --gpu 0
```

The flag takes precedence over inherited visibility settings. For service
configuration, `SPEECH_SERVER_GPU_DEVICE` provides the same override. Without
either override, existing `CUDA_VISIBLE_DEVICES` or `GGML_VK_VISIBLE_DEVICES`
settings remain authoritative.

The default URL is `http://127.0.0.1:8100`. Configure authentication with
`SPEECH_SERVER_TOKEN`, or override the host, port, model registry, voice store,
model-installation manifest, and library path with the `SPEECH_SERVER_*`
variables defined in
`speech_server/config.py`.

The Python binding is pinned in `speech_server/_vendor/crispasr`; no CrispASR
checkout or separately installed `crispasr` package is required. Native runtime
selection prefers `SPEECH_SERVER_LIB`, then
`runtime/crispasr/installed.json` (or `SPEECH_SERVER_RUNTIME_MANIFEST`), followed
by the existing developer-build fallbacks. `crispasr-compat.toml` currently
requires CrispASR v0.8.18 or newer, verifies every server-required upstream
commit, and records the expected `libcrispasr-*` asset for each supported
OS/backend target. Resolution is per target, so it selects the newest eligible
release that actually publishes that target's asset.

Release builders create the Windows payload with
`python scripts/build_windows_runtime.py`. The generated overlay contains the
official pinned Python ZIP under `vendor/python`, pinned `get-pip.py`, pinned
`bin/sfw.exe`, and their generated checksum manifest. End-user setup does not
depend on python.org retaining historical embedded archives. If the shipped
SFW executable is unexpectedly absent, Windows setup downloads the exact
pinned release from `vendor/security-tools.json` and verifies its GitHub
SHA-256; it never trusts an unpinned `latest` binary.

Maintainers build the clean public repository export with:

```powershell
python scripts/build_release.py public-tree --output dist/public-repo
```

Tagged GitHub releases run Windows and Linux CPU bootstrap smoke tests before
publishing versioned platform archives and `SHA256SUMS`. To reproduce those
archives locally:

```powershell
python scripts/build_release.py archives --version v0.1.0 --output dist/release
```

## Verification

```powershell
python -m pytest tests/ -v
python scripts/integration_synth.py omnivoice "The quick brown fox." out.wav default__VoiceId
```

The WebSocket protocol is version 2.0 and emits JSON lifecycle events plus
PCM16-LE mono binary frames. See the approved design and implementation plan
under `docs/superpowers/` for the complete contract.

Set `SPEECH_SERVER_DEBUG=1` before starting the server to print one structured
JSON diagnostic immediately before each outbound synthesis PCM chunk. The
record includes the exact normalized text used for every internal generation,
unit IDs, model and voice, sample counts, stage timings, and synthesis and
end-to-end RTFx (audio duration divided by processing duration). This output
contains generated dialogue and voice identifiers, so leave it disabled unless
that diagnostic detail is appropriate for the environment.

### Capability and resource discovery

Authenticated `GET /v2/capabilities` responses retain
`protocolVersion: "2.0"` and advertise `capabilitiesVersion: 8`, a stable
startup `registryRevision`, the resident-model limit, ordered per-model
numeric control schemas, and resource requirements. Component byte counts are
exact sums of local GGUF files without exposing their paths. RAM/VRAM
estimates use `estimated_ram_mb` / `estimated_vram_mb` registry overrides when
present; otherwise they are twice the component bytes rounded up to 256 MiB
and explicitly marked low confidence.

The shipped catalog declares estimates for canonical model bundles so guidance
is available before installation. These values use the same conservative 2x,
256-MiB-rounded formula against the exact default-bundle byte total. When a
canonical CrispASR default bundle changes, update both catalog estimates from
the new locked artifact total.

Capability v8 adds each TTS model's structured `paralinguisticTags` contract.
The registry is the only source of accepted canonical tags and aliases; models
without that metadata advertise no supported non-verbal tags.

Authenticated `GET /v2/resources` samples system and process RAM plus NVIDIA
GPU total/used/free memory through `nvidia-smi`. Unsupported telemetry is
returned as unavailable rather than failing the request, and GPU values are
not presented as per-process attribution. Loaded model IDs and the AudioVAE's load
state are also returned. Capability v5 adds component-level residency with
busy, evictable, and sticky state. TTS models are LRU-evictable when idle;
the VoxCPM2 AudioVAE and the process-cached aligner are shared sticky components.

The local catalog owns Sonorus semantics: task, validated language support,
sample rate, voice-reference policy, controls, and resource guidance. Its
`registry_bundle` keys point to CrispASR's canonical default bundles; they do
not select or rewrite quantization. Capability entries retain legacy
`available` and also report `installed` plus catalog-declared `installable`.
The installer resolves and locks the named bundle from the loaded CrispASR
runtime before downloading it. Language tags are normalized to lowercase
hyphenated form so Sonorus can filter TTS and ASR selections against the game
language consistently.
Catalog `model`/`codec` paths remain valid manual-install overrides. A complete
installation-manifest record for the same `registry_bundle` takes precedence
on startup; stale, incomplete, or differently bundled records do not.

Capability v7 adds authenticated model installation. CrispASR remains the only
authority for the canonical primary, companion, and extra artifacts; the speech
server never guesses a quantization. Before downloading, the server resolves
each Hugging Face URL to an immutable repository commit and requires the
published LFS SHA-256 and exact byte size. Large files use a bounded set of
validated HTTPS byte ranges with schema-v2 completed-segment resume metadata;
small files and hosts that ignore ranges use the sequential path. Downloads go
to resumable `.part` files, receive a final full-file SHA-256 check, and are
atomically activated only after the complete bundle succeeds. Direct
model-file HTTPS downloads are not package-manager
operations and are not wrapped by Socket Firewall; every `pip`/`uv` dependency
operation remains SFW-wrapped.

Sonorus can preview the exact download and any license requirement with:

- `GET /v2/models/{model_id}:install-plan`
- `GET /v2/upscaler:install-plan`
- `GET /v2/alignment:install-plan`

It starts an asynchronous job with the corresponding authenticated `POST`
endpoint and `{"acceptLicense": true|false}`. `GET /v2/installations/{job_id}`
reports locked artifacts, byte progress, completion, cancellation, or a
structured failure; `DELETE` requests cancellation. `GET /v2/installations`
lists jobs from the current server process. Completed provenance is persisted
under `runtime/models/installed.json` (override with
`SPEECH_SERVER_MODEL_MANIFEST`) and is applied on restart. Interrupted partial
downloads remain resumable but are never presented as installed.
For a restricted bundle, the preview reports CrispASR's license and approximate
artifact sizes without touching gated files; immutable hashes and exact sizes
are resolved only after the install request explicitly accepts that license.
Set `HF_TOKEN` when Hugging Face requires authenticated access to an artifact;
the downloader strips that credential before any cross-host CDN redirect.

Capability v6 adds completed-utterance ASR discovery and whole-stack planning.
ASR entries declare accepted PCM encoding, channels and sample rates, maximum
duration, languages, automatic detection, timestamp modes, contextual-bias
support, residency, and resource requirements. Capability v1-v5 clients and
servers remain compatible for Universal TTS; Universal ASR requires v6.

Authenticated `POST /v2/models/{model_id}:plan` accepts the selected upscaler and
adaptive-batching options and returns the exact reuse/load/evict/wait actions,
additional and reclaimable resource estimates, and projected steady-state
fit. The session pool evicts idle LRU TTS victims before constructing a
replacement, preventing avoidable old-plus-new peak allocation. If every
required victim is pinned by active synthesis, warmup returns `model_busy`
instead of risking an out-of-memory load.

Authenticated `POST /v2/stack:plan` and `POST /v2/stack:warmup` accept the
desired TTS model, ASR model, AudioVAE state, and Canary alignment state. Planning
deduplicates resident shared components, identifies obsolete idle models to
evict, and reports projected steady-state fit. The default example registry
permits two resident task models so an ASR model and TTS model can alternate
without reload thrash. Servers configured with only one slot reject a combined
stack before loading it.

### Completed-utterance transcription

The example registry includes `parakeet-tdt-0.6b-v3`. When its configured file
is absent, Sonorus can use the capability entry and install-plan endpoint to
download CrispASR's canonical Parakeet bundle. Until installation completes,
the model remains visible but unavailable for transcription.

Authenticated `POST /v2/transcribe` accepts Base64-encoded mono PCM16LE. The
initial Parakeet profile accepts 16 kHz audio up to 60 seconds, supports
automatic language detection, optional word/segment timestamps, and bounded
contextual bias terms:

```json
{
  "requestId": "example-1",
  "model": "parakeet-tdt-0.6b-v3",
  "audioData": "<base64 PCM16LE>",
  "audio": {"encoding": "pcm_s16le", "sampleRate": 16000, "channels": 1},
  "language": "auto",
  "biasTerms": ["Hogwarts", "Expelliarmus"],
  "timestamps": "none"
}
```

Responses include transcript text, nullable confidence and detected language,
requested language, optional timestamp arrays, audio duration, cold model-load
time, inference and total processing time, and throughput. Unsupported or
unreported metadata is `null`; it is never fabricated. Malformed Base64,
unsupported audio/language, duration and bias limit violations, task mismatch,
busy residency, load failure, and transcription failure use structured errors.

Model options are optional. A request may send only controls advertised for
its selected model; omitted controls leave the backend-native setting alone.
OmniVoice advertises `numSteps`, `firstSegmentSteps`, and `guidanceScale`.
Chatterbox advertises shared `numSteps`, `guidanceScale`, and `exaggeration`.
Unknown or out-of-range controls are rejected before synthesis.

Capability v4 adds each model's `voiceReference` policy. It declares whether a
reference transcript is required, optional, or unused and whether reference
encoding has a useful persistent preparation path. OmniVoice requires a
transcript and uses CrispASR's content-addressed reference-code disk cache;
Chatterbox ignores transcripts and encodes lazily.

`POST /v2/models/{model_id}/voices/{voice_id}:prepare` loads the selected model
and applies an uploaded reference without synthesizing. Successful preparation
is recorded per voice with a model-local revision and input hashes. OmniVoice
declares both audio and transcript as preparation inputs, so editing either one
invalidates its cached reference conditioning. `GET /v2/voices` exposes those
markers plus transcript/audio hashes without exposing
the transcript or server paths. Synthesis remains lazy-safe and rejects a
missing required transcript instead of silently using degraded conditioning.

### Adaptive segmentation and word timing

Capability v3 adds model-owned `textProfile` and `segmentation` policies plus
an optional top-level forced aligner. The default OmniVoice registry profile
uses reference-audio speaking rate when possible, falls back to 14 characters
or 2.7 words per second, and advertises 8/20/28-second minimum/target/maximum
hints. The 28-second predicted maximum is enforced by the server before a
physical synthesis call. Chatterbox has no segmentation profile yet and is
therefore submitted one sentence at a time by Sonorus.

Segments may use legacy `text` or explicit logical units:

```json
{"type":"segment","idx":1,"units":[
  {"id":"s1","text":"[laugh] I knew it."},
  {"id":"s2","text":"You cannot hide anything from me."}
]}
```

Set `options.timing` to `auto` to align only multi-unit segments, or `word` to
also align single units. OmniVoice audio-tag aliases are canonicalized for
TTS, but every nonlexical tag is removed from the CTC transcript and returned
as an estimated zero-width event. The processing order is TTS, optional AudioVAE,
resample final PCM to 16 kHz, then CTC. A failed alignment never discards PCM:
the server emits low-confidence duration-weighted unit boundaries and marks
the status `failed`.

Timing events use absolute stream seconds and Inworld-shaped parallel word
arrays. `started` and `segment_done` also expose cold model load time separately
from synthesis, upscale and alignment durations; `throughputX` is audio seconds
divided by synthesis seconds. The aligner is lazy and appears loaded in
`/v2/resources` only after its first successful alignment.

## Optional VoxCPM2 AudioVAE super-resolution

The `[upscaler]` entry in `models.toml` configures the standalone VoxCPM2
AudioVAE V2 backend as a persistent 16 kHz to 48 kHz speech super-resolution
postprocessor. Send `"upscale": true` inside the start message's `options`
object to upscale each completed TTS segment before it is forwarded:

The standalone VAE currently has no canonical CrispASR registry bundle. Place
the converter's `--vae-only` output at `models/voxcpm2-vae-f32.gguf`, as named
by the shipped catalog. If that local file is absent, capabilities report the
upscaler as unavailable and non-installable instead of silently selecting a
different backend.

```json
{"type":"start","requestId":"r1","model":"omnivoice","voiceId":"default__VoiceId","options":{"upscale":true}}
```

The `started.sampleRate` is `48000` for these requests (and remains the TTS
model's native rate otherwise). AudioVAE is loaded lazily on the first upscale
request and is kept outside the normal TTS model LRU.

Use `POST /v2/models/{id}:warmup?upscale=true&alignment=true` to explicitly
load the selected TTS model, AudioVAE, and the configured CTC aligner. CTC has no
load-only API, so alignment warmup validates it against the first uploaded
voice that has a nonempty spoken transcript; the request reports a clear setup
error when no such reference exists. Omit either flag to skip that component.

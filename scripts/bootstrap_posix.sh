#!/usr/bin/env sh
set -eu

mode=${1:-run}
target=${2:-auto}
case "$mode" in
    run|setup) ;;
    *)
        echo "Unknown bootstrap mode: $mode" >&2
        exit 2
        ;;
esac
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
echo "Preparing Universal Speech Server..."

runtime_python="$root/runtime/python/bin/python"
managed_python=1
sfw_command=
uv_command=

if [ -x "$root/bin/sfw" ]; then
    sfw_command="$root/bin/sfw"
elif command -v sfw >/dev/null 2>&1; then
    sfw_command=sfw
fi
if [ -x "$root/bin/uv" ]; then
    uv_command="$root/bin/uv"
elif command -v uv >/dev/null 2>&1; then
    uv_command=uv
fi

if [ -n "${SPEECH_SERVER_PYTHON:-}" ]; then
    runtime_python=$SPEECH_SERVER_PYTHON
    managed_python=0
fi

require_protected_uv() {
    if [ -z "$sfw_command" ]; then
        echo "Socket Firewall Free is required before downloading Python dependencies." >&2
        echo "Run: sh ./scripts/install_sfw.sh" >&2
        exit 1
    fi
    if [ -z "$uv_command" ]; then
        echo "uv is required to create or update the isolated server environment." >&2
        echo "Run: sh ./scripts/install_sfw.sh" >&2
        echo "Or set SPEECH_SERVER_PYTHON to a prepared Python 3.11+ interpreter." >&2
        exit 1
    fi
}

if [ "$managed_python" -eq 1 ] && [ ! -x "$runtime_python" ]; then
    require_protected_uv
    mkdir -p "$root/runtime"
    "$sfw_command" "$uv_command" venv --seed --python 3.13.14 "$root/runtime/python"
fi

if [ ! -x "$runtime_python" ]; then
    echo "Python interpreter is unavailable or not executable: $runtime_python" >&2
    exit 1
fi

if [ "$managed_python" -eq 1 ]; then
    if ! "$runtime_python" -m pip --version >/dev/null 2>&1; then
        require_protected_uv
        "$sfw_command" "$uv_command" pip install --python "$runtime_python" pip
    fi
    requirements="$root/requirements-runtime.txt"
    marker="$root/runtime/python/.requirements.sha256"
    requirements_hash=$(
        "$runtime_python" -c \
            'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
            "$requirements"
    )
    installed_hash=
    if [ -f "$marker" ]; then
        installed_hash=$(tr -d '\r\n' < "$marker")
    fi
    dependencies_ok=0
    if [ "$installed_hash" = "$requirements_hash" ] && \
        "$runtime_python" -c "import dotenv, fastapi, httptools, numpy, pydantic, uvicorn, uvloop, watchfiles, websockets, yaml" >/dev/null 2>&1 && \
        "$runtime_python" -m pip check >/dev/null 2>&1; then
        dependencies_ok=1
    fi
    if [ "$dependencies_ok" -ne 1 ]; then
        require_protected_uv
        "$sfw_command" "$uv_command" pip install \
            --python "$runtime_python" \
            --requirements "$requirements"
        "$runtime_python" -c "import dotenv, fastapi, httptools, numpy, pydantic, uvicorn, uvloop, watchfiles, websockets, yaml"
        "$runtime_python" -m pip check
        printf '%s\n' "$requirements_hash" > "$marker"
    fi
fi

if [ "${SPEECH_SERVER_UPDATE:-0}" = 1 ]; then
    PYTHONUTF8=1 "$runtime_python" -m speech_server.bootstrap setup-native --target "$target" --update
else
    PYTHONUTF8=1 "$runtime_python" -m speech_server.bootstrap setup-native --target "$target"
fi
if [ "$mode" = run ]; then
    gpu_assignment=$(PYTHONUTF8=1 "$runtime_python" -m speech_server.gpu_select --target "$target")
    case "$gpu_assignment" in
        CUDA_VISIBLE_DEVICES=*)
            CUDA_VISIBLE_DEVICES=${gpu_assignment#*=}
            export CUDA_VISIBLE_DEVICES
            ;;
        GGML_VK_VISIBLE_DEVICES=*)
            GGML_VK_VISIBLE_DEVICES=${gpu_assignment#*=}
            export GGML_VK_VISIBLE_DEVICES
            ;;
        "") ;;
        *)
            echo "GPU selector returned an invalid assignment." >&2
            exit 1
            ;;
    esac
    echo "Starting Universal Speech Server..."
    exec env PYTHONUTF8=1 "$runtime_python" -m speech_server
fi
exit 0

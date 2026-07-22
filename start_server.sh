#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
target=auto
target_set=0
gpu_device=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpu)
            if [ "$#" -lt 2 ]; then
                echo "--gpu requires a device index." >&2
                exit 2
            fi
            gpu_device=$2
            shift 2
            ;;
        --help)
            echo "Usage: ./start_server.sh [target] [--gpu INDEX]"
            echo "Example: ./start_server.sh nvidia --gpu 0"
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            if [ "$target_set" -eq 1 ]; then
                echo "Unexpected argument: $1" >&2
                exit 2
            fi
            target=$1
            target_set=1
            shift
            ;;
    esac
done

exec sh "$root/scripts/bootstrap_posix.sh" run "$target" "$gpu_device"

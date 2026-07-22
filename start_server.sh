#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$root/scripts/bootstrap_posix.sh" run "${1:-auto}"

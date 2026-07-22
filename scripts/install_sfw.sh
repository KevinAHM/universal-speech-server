#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tag=v1.13.1
uv_tag=0.11.30
os=$(uname -s | tr '[:upper:]' '[:lower:]')
machine=$(uname -m)

case "$os:$machine" in
    linux:x86_64|linux:amd64)
        asset=sfw-free-linux-x86_64
        expected=4dc46b626a7c5b81c0b54e1984ee53be5a628dbfb2f55ab14e9b04c8a134db6a
        uv_asset=uv-x86_64-unknown-linux-gnu.tar.gz
        uv_expected=04bc7d180d6138bf6dc08387acf507a823f397a98fea55da36b0ccc7fbce3b68
        ;;
    linux:aarch64|linux:arm64)
        asset=sfw-free-linux-arm64
        expected=f87bbbca2192fca9740f9bdb115e7cfaa22e957a8f5234d5f97fce1383aa1d66
        uv_asset=uv-aarch64-unknown-linux-gnu.tar.gz
        uv_expected=8c11d90f5f66d232930cf8ae3a085c39877690d409e10878234802b028b20e2a
        ;;
    darwin:x86_64|darwin:amd64)
        asset=sfw-free-macos-x86_64
        expected=6c7d5fcf66bc5284b3320cf6e12e4654135eb64ef3a926ea77e3d0904782d862
        uv_asset=uv-x86_64-apple-darwin.tar.gz
        uv_expected=ce285fbbfbe294b1e1bc6c87c8b59d9622b85383b88b2b132a2df5c73e83d7c1
        ;;
    darwin:arm64|darwin:aarch64)
        asset=sfw-free-macos-arm64
        expected=30ab1981303fc18f41db9d1615d9a792015d9d9e52da658a387bc89fe344db8f
        uv_asset=uv-aarch64-apple-darwin.tar.gz
        uv_expected=9bed3567d496d8dab84ecf7a1247551ac94ef1baaebb7b65df008dd93e9dc357
        ;;
    *)
        echo "No pinned Socket Firewall binary for $os/$machine." >&2
        exit 1
        ;;
esac

mkdir -p "$root/bin"
destination="$root/bin/sfw"
temporary="$root/bin/.sfw-download-$$"
uv_archive="$root/bin/.uv-download-$$.tar.gz"
uv_staging="$root/bin/.uv-extract-$$"
trap 'rm -f "$temporary" "$uv_archive"; rm -rf "$uv_staging"' EXIT HUP INT TERM
url="https://github.com/SocketDev/sfw-free/releases/download/$tag/$asset"
echo "Downloading pinned Socket Firewall $tag..."
curl --fail --location --output "$temporary" "$url"

file_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "sha256sum or shasum is required to verify downloaded tools." >&2
        exit 1
    fi
}

actual=$(file_sha256 "$temporary")
if [ "$actual" != "$expected" ]; then
    echo "Socket Firewall SHA-256 mismatch: expected $expected, got $actual" >&2
    exit 1
fi

chmod 0755 "$temporary"
mv -f "$temporary" "$destination"
if [ "$os" = darwin ] && command -v xattr >/dev/null 2>&1; then
    xattr -dr com.apple.quarantine "$destination" 2>/dev/null || true
fi
echo "Installed verified Socket Firewall at $destination"

if ! command -v tar >/dev/null 2>&1; then
    echo "tar is required to install the pinned uv runtime manager." >&2
    exit 1
fi
uv_url="https://github.com/astral-sh/uv/releases/download/$uv_tag/$uv_asset"
echo "Downloading pinned uv $uv_tag..."
curl --fail --location --output "$uv_archive" "$uv_url"
uv_actual=$(file_sha256 "$uv_archive")
if [ "$uv_actual" != "$uv_expected" ]; then
    echo "uv SHA-256 mismatch: expected $uv_expected, got $uv_actual" >&2
    exit 1
fi

uv_directory=${uv_asset%.tar.gz}
uv_member="$uv_directory/uv"
mkdir -p "$uv_staging"
tar -xzf "$uv_archive" -C "$uv_staging" "$uv_member"
if [ ! -f "$uv_staging/$uv_member" ]; then
    echo "Pinned uv archive did not contain $uv_member." >&2
    exit 1
fi
chmod 0755 "$uv_staging/$uv_member"
mv -f "$uv_staging/$uv_member" "$root/bin/uv"
if [ "$os" = darwin ] && command -v xattr >/dev/null 2>&1; then
    xattr -dr com.apple.quarantine "$root/bin/uv" 2>/dev/null || true
fi
"$root/bin/uv" --version
echo "Installed verified uv at $root/bin/uv"
rm -f "$uv_archive"
rm -rf "$uv_staging"
trap - EXIT HUP INT TERM

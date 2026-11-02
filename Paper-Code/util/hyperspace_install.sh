#!/bin/sh
# Bootstrap installer — downloads hai and delegates to `hai update`.
# Usage: curl -fsSL https://.../install.sh | sh
set -eu

# Default channel; replaced by release pipeline before upload — do not rename
CHANNEL="${HAI_CHANNEL:-stable}"
case "$CHANNEL" in
  stable) BINARY_NAME="hai" ;;
  beta)   BINARY_NAME="hai-beta" ;;
  *) echo "Invalid HAI_CHANNEL: '$CHANNEL'. Must be one of: stable, beta" >&2; exit 1 ;;
esac

ARTIFACTORY="https://int.repositories.cloud.sap/artifactory/hyperspace-ai/bin"

# Detect platform
os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
case "$arch" in
  x86_64)        arch="amd64" ;;
  aarch64|arm64) arch="arm64" ;;
  *) echo "Unsupported arch: $arch" >&2; exit 1 ;;
esac
case "$os" in
  darwin|linux) ;;
  *) echo "Unsupported OS: $os" >&2; exit 1 ;;
esac

url="${ARTIFACTORY}/${CHANNEL}/latest/hai-${os}.${arch}.tar.gz"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

echo "Downloading hai (${os}/${arch})..."
if command -v curl >/dev/null 2>&1; then
  curl -fsS --proto '=https' --max-redirs 0 "$url"          -o "$tmp/hai.tar.gz"
  curl -fsS --proto '=https' --max-redirs 0 "${url}.sha256" -o "$tmp/hai.sha256"
else
  wget -qO "$tmp/hai.tar.gz"  --https-only --max-redirect=0 "$url"
  wget -qO "$tmp/hai.sha256"  --https-only --max-redirect=0 "${url}.sha256"
fi

echo "Verifying checksum..."
expected="$(awk '{print $1}' "$tmp/hai.sha256" | tr '[:upper:]' '[:lower:]')"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$tmp/hai.tar.gz" | awk '{print $1}')"
else
  actual="$(shasum -a 256 "$tmp/hai.tar.gz" | awk '{print $1}')"
fi
[ "$actual" = "$expected" ] || { echo "Checksum mismatch: got $actual, want $expected" >&2; exit 1; }

tar -xzf "$tmp/hai.tar.gz" -C "$tmp"
if [ ! -f "$tmp/hai" ]; then
  echo "Extraction error: expected $tmp/hai after extracting $tmp/hai.tar.gz" >&2; exit 1
fi
chmod +x "$tmp/hai"
"$tmp/hai" update --from "$tmp/hai" --name "$BINARY_NAME"
exit $?

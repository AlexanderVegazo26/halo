#!/usr/bin/env bash
# Install a pinned, checksum-verified k6 into a directory on PATH (CI and local use).
#
# Usage: scripts/install-k6.sh [dest-dir]      (default dest-dir: $HOME/.local/bin)
#   K6_VERSION / K6_SHA256 / K6_ASSET_ID override the pin (all together). GITHUB_TOKEN, if
#   set, authenticates the API fallback (unauthenticated: 60 requests/hour per IP).
#
# Why not grafana/setup-k6-action: it only tries the github.com download URL, twice, so a
# GitHub 504 on that URL failed build-and-test and tsan on bb41544 (2026-11-08). Here each
# source is retried, the REST API asset endpoint is the fallback, and the sha256 check
# refuses a corrupt or swapped tarball whichever source served it.
# On GitHub Actions the dest dir is added to $GITHUB_PATH for the following steps.
set -euo pipefail

version="${K6_VERSION:-2.3.0}"
# sha256 of k6-v2.3.0-linux-amd64.tar.gz from the release's k6-v2.3.0-checksums.txt
sha256="${K6_SHA256:-39c3117b6af817592dcd0ce4242105c0a7af10948c2a425306f0be8f7a8a8ab1}"
# GitHub release asset id of that tarball (fallback download through the REST API)
asset_id="${K6_ASSET_ID:-579287772}"
dest="${1:-$HOME/.local/bin}"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ;;
  *) echo "install-k6: only linux-amd64 is pinned here, got $(uname -s)-$(uname -m)" >&2; exit 2 ;;
esac

name="k6-v${version}-linux-amd64"
url="https://github.com/grafana/k6/releases/download/v${version}/${name}.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fetch() { # fetch <url> [curl args...]: retries every error, bounded in time
  local u="$1"; shift
  curl -fsSL --retry 4 --retry-all-errors --retry-delay 5 --retry-max-time 90 \
    --connect-timeout 20 "$@" -o "$tmp/k6.tar.gz" "$u"
}
src="$url"
if ! fetch "$url"; then
  # The github.com download URL and the REST asset endpoint are served separately; on
  # 2026-11-08 the first returned 504 for minutes while the second served the same file.
  api="https://api.github.com/repos/grafana/k6/releases/assets/${asset_id}"
  echo "install-k6: $url failed, trying $api" >&2
  src="$api"
  auth=()
  if [ -n "${GITHUB_TOKEN:-}" ]; then auth=(-H "Authorization: Bearer $GITHUB_TOKEN"); fi
  fetch "$api" -H 'Accept: application/octet-stream' "${auth[@]}"
fi

actual="$(sha256sum "$tmp/k6.tar.gz" | cut -d' ' -f1)"
if [ "$actual" != "$sha256" ]; then
  echo "install-k6: sha256 mismatch for $src: expected $sha256, got $actual" >&2
  exit 1
fi

tar -xzf "$tmp/k6.tar.gz" -C "$tmp"
mkdir -p "$dest"
install -m 0755 "$tmp/$name/k6" "$dest/k6"
if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$dest" >> "$GITHUB_PATH"
fi
"$dest/k6" version

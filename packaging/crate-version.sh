#!/usr/bin/env bash
# Prints the version of the gaudium crate (the `version` of `[package]` in gaudium/Cargo.toml).
#
#   packaging/crate-version.sh                     print the version, e.g. 0.0.1
#   packaging/crate-version.sh --check-tag v0.0.1  fail unless the tag is "v" + that version
#
# A release is cut by pushing the tag v<version>; the release workflow runs the second form so
# that a tag which does not name the crate version publishes nothing.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$root/gaudium/Cargo.toml"

version=$(awk '
  /^\[/ { in_package = ($0 == "[package]") }
  in_package && /^version[[:space:]]*=/ {
    if (match($0, /"[^"]+"/)) { print substr($0, RSTART + 1, RLENGTH - 2); exit }
  }' "$manifest")

if ! [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "crate-version: no X.Y.Z version under [package] in $manifest (read: '$version')" >&2
  exit 1
fi

case "${1:-}" in
  "")
    echo "$version"
    ;;
  --check-tag)
    tag=${2:?crate-version: --check-tag needs the tag name}
    if [[ $tag != "v$version" ]]; then
      echo "crate-version: tag '$tag' does not match gaudium/Cargo.toml, which says $version (expected tag 'v$version')" >&2
      exit 1
    fi
    echo "$version"
    ;;
  *)
    echo "usage: crate-version.sh [--check-tag <tag>]" >&2
    exit 2
    ;;
esac

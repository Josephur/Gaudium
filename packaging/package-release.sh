#!/usr/bin/env bash
# Builds the release archive of Gaudium from an already built tree.
#
#   packaging/package-release.sh --sdk-version <apt version of the Intel Gaudi SDK packages>
#                                [--bin <gaudium binary>]   default: gaudium/target/release/gaudium
#                                [--out <dir>]              default: dist
#
# Writes into the output directory:
#   gaudium-<version>-linux-x86_64.tar.gz           the archive
#   gaudium-<version>-linux-x86_64.tar.gz.sha256    its SHA-256 (`sha256sum -c`)
#   RELEASE_NOTES.md                                the text of the GitHub release
#
# The archive holds the engine binary, the Web UI, the documents, and the SOURCES of the
# kernels (`kernels/`). It holds no compiled kernel and no glue library: those are the output
# of the Intel Gaudi SDK's compiler, which puts code of its own into them, so every user
# builds them with their own SDK installation (INSTALL.md says how). `--sdk-version` only names
# the SDK release this build was tested with.
#
# Only files tracked by git go in, and the staged tree is checked before it is packed: the
# engine is its one binary, and no object file, library, compiled recipe or SDK file is in it.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sdk_version=""
bin="$root/gaudium/target/release/gaudium"
out="$root/dist"

die() { echo "package-release: $*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --sdk-version) sdk_version=${2:?--sdk-version needs a value}; shift 2 ;;
    --bin) bin=${2:?--bin needs a value}; shift 2 ;;
    --out) out=${2:?--out needs a value}; shift 2 ;;
    *) die "unknown argument '$1' (see the head of this script)" ;;
  esac
done

[[ $sdk_version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9]+)?$ ]] \
  || die "--sdk-version must look like 1.24.2-481 (got '$sdk_version')"
sdk_release=${sdk_version%%-*}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die "the archive is built on Linux x86_64 only"
[[ -x $bin ]] || die "no engine binary at $bin (cargo build --release in gaudium/)"
git -C "$root" rev-parse --git-dir >/dev/null 2>&1 || die "$root is not a git checkout"

version=$("$root/packaging/crate-version.sh")
reported=$("$bin" --version)
[[ $reported == "gaudium $version" || $reported == "gaudium $version ("*")" ]] \
  || die "$bin reports '$reported', and gaudium/Cargo.toml says $version: rebuild the binary"

name="gaudium-$version-linux-x86_64"
mkdir -p "$out"
out=$(cd "$out" && pwd)
stage_root=$(mktemp -d "$out/stage.XXXXXX")
trap 'rm -rf "$stage_root"' EXIT
stage="$stage_root/$name"
mkdir -p "$stage/bin"

# --- engine ---
install -m 0755 "$bin" "$stage/bin/gaudium"

# --- tracked files only: the kernel sources, the Web UI without its tests, the documents ---
git -C "$root" ls-files -z -- kernels ui ':(exclude)ui/test' docs LICENSE THIRD_PARTY_NOTICES.md README.md \
  | tar -C "$root" --null -T - -cf - | tar -C "$stage" -xf -
[[ -f $stage/ui/index.html ]] || die "the Web UI bundle is missing from the checkout"
[[ -f $stage/kernels/Makefile && -d $stage/kernels/tpc && -d $stage/kernels/glue ]] \
  || die "the kernel sources are missing from the checkout"

# --- facts about this build ---
commit=$(git -C "$root" rev-parse HEAD)
epoch=${SOURCE_DATE_EPOCH:-$(git -C "$root" log -1 --format=%ct)}
first_line() { { "$@" 2>/dev/null || true; } | head -n 1; }
or_unknown() { local v; v=$(cat); echo "${v:-unknown}"; }
# the newest glibc symbol version the binary asks for: it runs on that glibc or a later one
glibc_min=$({ objdump -T "$stage/bin/gaudium" 2>/dev/null || true; } | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' \
  | sort -Vu | tail -n 1 | sed 's/^GLIBC_//' | or_unknown)
os_name=$( (. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-}") | or_unknown)
cat > "$stage/BUILD-INFO.txt" <<INFO
Gaudium $version
commit            $commit
commit date       $(date -u -d "@$epoch" +%Y-%m-%dT%H:%M:%SZ)
target            linux x86_64 (needs glibc $glibc_min or later)
built on          $os_name
rustc             $(first_line rustc --version | or_unknown)
bend              $(first_line bend version | or_unknown)
lean              $(first_line lean --version | or_unknown)
Intel Gaudi SDK   $sdk_version: the release this build was tested with. Not included.
kernels           sources only (kernels/): build them with your own SDK installation, see INSTALL.md
                  (this build compiled them as a check with: $(first_line "${TPC_CC:-tpc-clang}" --version | or_unknown))
INFO

sed -e "s|@VERSION@|$version|g" -e "s|@SDK_VERSION@|$sdk_version|g" -e "s|@SDK_RELEASE@|$sdk_release|g" \
    -e "s|@GLIBC_MIN@|$glibc_min|g" -e "s|@OS_NAME@|$os_name|g" -e "s|@NAME@|$name|g" \
    "$root/packaging/INSTALL.md.in" > "$stage/INSTALL.md"
if grep -n '@[A-Z_]*@' "$stage/INSTALL.md"; then die "INSTALL.md.in has a placeholder this script does not fill"; fi

# --- what must never ship ---
# compiled kernels, glue libraries or any other object code, compiled recipes, SDK files
bad=$(find "$stage" \( -iname '*.o' -o -iname '*.elf' -o -iname '*.so' -o -iname '*.so.*' -o -iname '*.a' \
  -o -iname '*.recipe' -o -iname '*.habana_recipe' -o -iname '*.bc' -o -iname 'libSynapse*' \
  -o -iname 'libtpc_kernels*' -o -iname 'tpc-clang*' -o -iname 'tpc-llvm*' -o -iname '*habana*' \
  -o -iname 'libhl[-_]*' \) -print)
[[ -z $bad ]] || die "refusing to pack compiled kernels, libraries, recipes or SDK files:"$'\n'"$bad"
# whatever its name: the engine is the only ELF file of the archive
while IFS= read -r -d '' f; do
  [[ $f == "$stage/bin/gaudium" ]] && continue
  if [[ $(head -c 4 "$f" | od -An -c | tr -d ' ') == '177ELF' ]]; then
    die "refusing to pack an ELF file other than the engine: ${f#"$stage/"}"
  fi
done < <(find "$stage" -type f -print0)

# --- checksums of everything in the archive, then the archive itself (same bytes for the same inputs) ---
(cd "$stage" && find . -type f ! -name SHA256SUMS -printf '%P\0' | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS)
archive="$out/$name.tar.gz"
tar -C "$stage_root" --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$epoch" -cf - "$name" \
  | gzip -n -9 > "$archive"
(cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")

cat > "$out/RELEASE_NOTES.md" <<NOTES
Gaudium $version for the Intel Gaudi 2, Linux x86_64.

**\`$name.tar.gz\`** holds the \`gaudium\` binary, the Web UI and the sources of the TPC kernels.
\`INSTALL.md\` in the archive has the steps; \`$name.tar.gz.sha256\` is its SHA-256
(\`sha256sum -c $name.tar.gz.sha256\`).

**No Intel Gaudi software is included, and no compiled kernel.** Install the Intel Gaudi driver,
firmware and SDK (SynapseAI, \`tpc-clang\`) yourself from Intel, then build the kernels from the
sources in the archive with \`make -C kernels -j"\$(nproc)"\`. This release was tested with SDK
release **$sdk_release** (packages \`$sdk_version\`). See the
[Intel Gaudi installation guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

Built on $os_name; the binary needs glibc $glibc_min or later.

Gaudium is in the middle of a port to Rust: interfaces and flags still change, and this is not
a stable release.
NOTES

echo "package-release: wrote $archive"
echo "  $(cat "$out/$name.tar.gz.sha256")"
echo "  $(tar -tzf "$archive" | wc -l) entries, $(du -h "$archive" | cut -f1)"

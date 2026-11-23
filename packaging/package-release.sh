#!/usr/bin/env bash
# Builds the release archive of Gaudium from an already built tree.
#
#   packaging/package-release.sh --sdk-version <apt version of the Intel Gaudi SDK packages>
#                                [--bin <gaudium binary>]   default: gaudium/target/release/gaudium
#                                [--kernels <dir>]          default: build/kernels
#                                [--out <dir>]              default: dist
#
# Writes into the output directory:
#   gaudium-<version>-linux-x86_64-synapse<sdk>.tar.gz           the archive
#   gaudium-<version>-linux-x86_64-synapse<sdk>.tar.gz.sha256    its SHA-256 (`sha256sum -c`)
#   RELEASE_NOTES.md                                             the text of the GitHub release
#
# The archive holds what this repository builds and nothing else: the engine binary, the TPC
# kernel ELFs and graph-compiler glue libraries that `make -C kernels` wrote, the Web UI, the
# documents and licences. No file of the Intel Gaudi SDK and no compiled recipe goes in: the
# kernels directory is copied by an allow-list, and the staged tree is checked again before it
# is packed.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sdk_version=""
bin="$root/gaudium/target/release/gaudium"
kernels="$root/build/kernels"
out="$root/dist"

die() { echo "package-release: $*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --sdk-version) sdk_version=${2:?--sdk-version needs a value}; shift 2 ;;
    --bin) bin=${2:?--bin needs a value}; shift 2 ;;
    --kernels) kernels=${2:?--kernels needs a value}; shift 2 ;;
    --out) out=${2:?--out needs a value}; shift 2 ;;
    *) die "unknown argument '$1' (see the head of this script)" ;;
  esac
done

[[ $sdk_version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9]+)?$ ]] \
  || die "--sdk-version must look like 1.24.2-481 (got '$sdk_version')"
sdk_release=${sdk_version%%-*}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die "the archive is built on Linux x86_64 only"
[[ -x $bin ]] || die "no engine binary at $bin (cargo build --release in gaudium/)"
[[ -d $kernels ]] || die "no kernels directory at $kernels (make -C kernels)"
git -C "$root" rev-parse --git-dir >/dev/null 2>&1 || die "$root is not a git checkout"

version=$("$root/packaging/crate-version.sh")
reported=$("$bin" --version)
[[ $reported == "gaudium $version" || $reported == "gaudium $version ("*")" ]] \
  || die "$bin reports '$reported', and gaudium/Cargo.toml says $version: rebuild the binary"

name="gaudium-$version-linux-x86_64-synapse$sdk_release"
mkdir -p "$out"
out=$(cd "$out" && pwd)
stage_root=$(mktemp -d "$out/stage.XXXXXX")
trap 'rm -rf "$stage_root"' EXIT
stage="$stage_root/$name"
mkdir -p "$stage/bin" "$stage/kernels"

# --- engine ---
install -m 0755 "$bin" "$stage/bin/gaudium"

# --- kernels: our ELFs (*.o) and our glue libraries (libgaudium_*.so), nothing else ---
glue=(libgaudium_tpc.so libgaudium_ga.so libgaudium_st.so)
for lib in "${glue[@]}"; do
  [[ -f $kernels/$lib ]] || die "missing glue library $kernels/$lib"
  install -m 0644 "$kernels/$lib" "$stage/kernels/$lib"
done
elf_count=0
for elf in "$kernels"/*.o; do
  [[ -f $elf ]] || continue
  [[ $(head -c 4 "$elf" | od -An -c | tr -d ' ') == '177ELF' ]] || die "$elf is not an ELF file"
  install -m 0644 "$elf" "$stage/kernels/"
  elf_count=$((elf_count + 1))
done
((elf_count > 0)) || die "no kernel ELF (*.o) in $kernels"

# --- Web UI (tracked files, without its tests) and the documents ---
git -C "$root" ls-files -z -- ui ':(exclude)ui/test' docs LICENSE THIRD_PARTY_NOTICES.md README.md \
  | tar -C "$root" --null -T - -cf - | tar -C "$stage" -xf -
[[ -f $stage/ui/index.html ]] || die "the Web UI bundle is missing from the checkout"

# --- facts about this build ---
commit=$(git -C "$root" rev-parse HEAD)
epoch=${SOURCE_DATE_EPOCH:-$(git -C "$root" log -1 --format=%ct)}
first_line() { { "$@" 2>/dev/null || true; } | head -n 1; }
or_unknown() { local v; v=$(cat); echo "${v:-unknown}"; }
# the newest glibc symbol version a file asks for: it runs on that glibc or a later one
glibc_floor() {
  { objdump -T "$1" 2>/dev/null || true; } | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' | sort -Vu | tail -n 1 \
    | sed 's/^GLIBC_//' | or_unknown
}
glibc_min=$(for f in "$stage/bin/gaudium" "${glue[@]/#/$stage/kernels/}"; do glibc_floor "$f"; done \
  | sort -V | tail -n 1)
os_name=$( (. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-}") | or_unknown)
cat > "$stage/BUILD-INFO.txt" <<EOF
Gaudium $version
commit            $commit
commit date       $(date -u -d "@$epoch" +%Y-%m-%dT%H:%M:%SZ)
target            linux x86_64 (needs glibc $glibc_min or later)
built on          $os_name
Intel Gaudi SDK   $sdk_version (habanalabs-tools, habanalabs-graph headers; not included)
tpc-clang         $(first_line "${TPC_CC:-tpc-clang}" --version | or_unknown)
glue compiler     $(first_line "${CXX:-g++}" --version | or_unknown)
rustc             $(first_line rustc --version | or_unknown)
bend              $(first_line bend version | or_unknown)
lean              $(first_line lean --version | or_unknown)
kernel ELFs       $elf_count
glue libraries    ${glue[*]}
EOF

sed -e "s|@VERSION@|$version|g" -e "s|@SDK_VERSION@|$sdk_version|g" -e "s|@SDK_RELEASE@|$sdk_release|g" \
    -e "s|@GLIBC_MIN@|$glibc_min|g" -e "s|@OS_NAME@|$os_name|g" -e "s|@NAME@|$name|g" \
    "$root/packaging/INSTALL.md.in" > "$stage/INSTALL.md"
if grep -n '@[A-Z_]*@' "$stage/INSTALL.md"; then die "INSTALL.md.in has a placeholder this script does not fill"; fi

# --- what must never ship: SDK files, compiled recipes, anything unexpected beside the kernels ---
bad=$(find "$stage" \( -iname '*.recipe' -o -iname '*.habana_recipe' -o -iname 'libSynapse*' \
  -o -iname 'libtpc_kernels*' -o -iname 'tpc-clang*' -o -iname 'tpc-llvm*' -o -iname '*habana*' \
  -o -iname 'libhl[-_]*' -o -iname '*.bc' \) -print)
[[ -z $bad ]] || die "refusing to pack SDK files or recipes:"$'\n'"$bad"
bad=$(find "$stage/kernels" -mindepth 1 ! \( -type f \( -name '*.o' -o -name 'libgaudium_*.so' \) \) -print)
[[ -z $bad ]] || die "unexpected entries in the kernels directory:"$'\n'"$bad"
# the glue libraries link the C and C++ runtimes only (the graph compiler loads them; they load no SDK library)
for lib in "${glue[@]}"; do
  needed=$(readelf -d "$stage/kernels/$lib" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
  for n in $needed; do
    case "$n" in
      libstdc++.so.*|libm.so.*|libgcc_s.so.*|libc.so.*|ld-linux-x86-64.so.*) ;;
      *) die "$lib links $n, which is not a C or C++ runtime library" ;;
    esac
  done
done

# --- checksums of everything in the archive, then the archive itself (same bytes for the same inputs) ---
(cd "$stage" && find . -type f ! -name SHA256SUMS -printf '%P\0' | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS)
archive="$out/$name.tar.gz"
tar -C "$stage_root" --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$epoch" -cf - "$name" \
  | gzip -n -9 > "$archive"
(cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")

cat > "$out/RELEASE_NOTES.md" <<EOF
Gaudium $version for the Intel Gaudi 2, Linux x86_64.

**\`$name.tar.gz\`** holds the \`gaudium\` binary, the compiled TPC kernels and graph-compiler
glue libraries, and the Web UI. \`INSTALL.md\` in the archive has the steps;
\`$name.tar.gz.sha256\` is its SHA-256 (\`sha256sum -c $name.tar.gz.sha256\`).

**The Intel Gaudi software is not included.** Install the Intel Gaudi driver, firmware and SDK
(SynapseAI) yourself from Intel, in release **$sdk_release** (packages \`$sdk_version\`): the
kernels in this archive were compiled with that release's \`tpc-clang\`. See the
[Intel Gaudi installation guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).
With another SDK release, build Gaudium from source (see \`README.md\`).

Built on $os_name; the binaries need glibc $glibc_min or later.

Gaudium is in the middle of a port to Rust: interfaces and flags still change, and this is not
a stable release.
EOF

echo "package-release: wrote $archive"
echo "  $(cat "$out/$name.tar.gz.sha256")"
echo "  $(tar -tzf "$archive" | wc -l) entries, $elf_count kernel ELFs, $(du -h "$archive" | cut -f1)"

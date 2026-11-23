# Releasing Gaudium

A release is a git tag `vX.Y.Z` on a commit of `main`. Pushing the tag makes the CI workflow
(`.github/workflows/ci.yml`) build everything, pack it, and publish a GitHub Release.

## Cutting a release

1. **Bump the version in a pull request.** Set `version` in `gaudium/Cargo.toml`, run
   `cargo check` in `gaudium/` so that `Cargo.lock` follows, and merge the pull request like
   any other change. `gaudium --version` prints this version.
2. **Tag the merge commit and push the tag:**

   ```bash
   git fetch origin
   git tag -a v0.0.2 -m "Gaudium 0.0.2" origin/main
   git push origin v0.0.2
   ```

3. **The workflow does the rest.** It fails, and publishes nothing, when the tag is not `v`
   followed by the version in `gaudium/Cargo.toml`
   (`packaging/crate-version.sh --check-tag <tag>`), or when the tagged commit is not on
   `main`. Otherwise it runs the same checks as for a pull request, builds the release binary,
   and creates the release with two files:

   - `gaudium-<version>-linux-x86_64.tar.gz`: the archive;
   - `gaudium-<version>-linux-x86_64.tar.gz.sha256`: its SHA-256.

If the run fails, fix the cause on `main` through a pull request, delete the tag
(`git push origin :refs/tags/v0.0.2`, `git tag -d v0.0.2`) and tag the new commit.

## What the archive holds

`packaging/package-release.sh` builds it: `bin/gaudium`, `ui/` without its tests, the
**sources** of the kernels (`kernels/`: `Makefile`, `tpc/`, `glue/`), `docs/`, `README.md`,
`LICENSE`, `THIRD_PARTY_NOTICES.md`, `INSTALL.md` (from `packaging/INSTALL.md.in`),
`BUILD-INFO.txt` (commit and tool versions) and `SHA256SUMS`. Only files tracked by git go in,
plus the binary.

Every pull request and every push to `main` builds the same archive and keeps it for 3 days as
a workflow artifact of the run, so a release can be inspected before its tag exists.

## No compiled kernel and no Intel Gaudi SDK file is shipped

The kernels (`*.o`) and the glue libraries (`libgaudium_*.so`) are compiled by the Intel Gaudi
SDK: `tpc-clang` puts its own code for built-in functions into a kernel, and the glue is
compiled against an SDK header. The SDK's licence does not allow redistribution. So:

- a release has the kernel sources, and `INSTALL.md` tells users to install the driver, the
  firmware and the SDK from Intel and to run `make -C kernels` in the unpacked archive;
- the workflow installs the SDK on the runner and compiles the kernels twice, from the
  checkout and from the unpacked archive, only to check that they build. Nothing under
  `build/kernels/` goes into the archive, an artifact, a cache or this repository, and the SDK
  is not cached between runs;
- the packaging script refuses a staged tree that holds an object file, a library, a compiled
  recipe, an SDK file, or any ELF file other than `bin/gaudium`.

## Pinned tool versions

They are the `env` block at the top of `.github/workflows/ci.yml`:

| Variable | What |
|---|---|
| `SDK_VERSION` | The Intel Gaudi SDK packages the workflow compiles the kernels with as a check; `INSTALL.md` and `BUILD-INFO.txt` name it as the release the build was tested with. Change it in a pull request of its own, and test that SDK release on a card before tagging. |
| `RUST_VERSION` | The Rust toolchain. |
| `LEAN_VERSION` | The Lean release that `bend --verdict` asks for. It belongs to the Bend release in `gaudium/bend.pin`. |
| `BEND_SHA256` | The SHA-256 of the Bend archive for the release in `gaudium/bend.pin`; update both together. |
| `NODE_VERSION` | Node.js for the Web UI tests. |

The workflow builds on a GitHub-hosted Ubuntu 24.04 runner; `BUILD-INFO.txt` in the archive
states the oldest glibc the binary runs on. No job
opens a Gaudi card, and nothing runs on a schedule.

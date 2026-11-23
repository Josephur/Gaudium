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
   `main`. Otherwise it runs the same checks as for a pull request, builds the release binary
   and the kernels, and creates the release with two files:

   - `gaudium-<version>-linux-x86_64-synapse<sdk>.tar.gz`: the archive;
   - `gaudium-<version>-linux-x86_64-synapse<sdk>.tar.gz.sha256`: its SHA-256.

If the run fails, fix the cause on `main` through a pull request, delete the tag
(`git push origin :refs/tags/v0.0.2`, `git tag -d v0.0.2`) and tag the new commit.

## What the archive holds

`packaging/package-release.sh` builds it from the built tree: `bin/gaudium`, `kernels/` (the
TPC kernel ELFs and the glue libraries that `make -C kernels` wrote), `ui/` without its tests,
`docs/`, `README.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `INSTALL.md` (from
`packaging/INSTALL.md.in`), `BUILD-INFO.txt` (commit and tool versions) and `SHA256SUMS`.

Every pull request and every push to `main` builds the same archive and keeps it for 14 days
as a workflow artifact of the run, so a release can be inspected before its tag exists.

## The Intel Gaudi SDK is never shipped

The runner installs the Intel Gaudi SDK from Intel's package repository to compile the
kernels. Its licence does not allow redistribution:

- no SDK file (`libSynapse`, `tpc-clang`, Intel's kernel library, headers, firmware) and no
  compiled recipe goes into the archive, an artifact, a cache or this repository. The packaging
  script copies the kernels directory by an allow-list (`*.o`, `libgaudium_*.so`) and refuses a
  staged tree that holds anything else of that kind;
- the SDK is not cached between runs;
- `INSTALL.md` and the release text tell users to install the driver, the firmware and the SDK
  from Intel, in the release the archive is named after.

## Pinned tool versions

They are the `env` block at the top of `.github/workflows/ci.yml`:

| Variable | What |
|---|---|
| `SDK_VERSION` | The Intel Gaudi SDK packages the kernels are compiled with. The archive name carries its release (`synapse1.24.2`). Change it in a pull request of its own, and test the kernels of that SDK release on a card before tagging. |
| `RUST_VERSION` | The Rust toolchain. |
| `LEAN_VERSION` | The Lean release that `bend --verdict` asks for. It belongs to the Bend release in `gaudium/bend.pin`. |
| `BEND_SHA256` | The SHA-256 of the Bend archive for the release in `gaudium/bend.pin`; update both together. |
| `NODE_VERSION` | Node.js for the Web UI tests. |

The workflow builds on a GitHub-hosted Ubuntu 24.04 runner; the binaries need that glibc
release's symbol versions or later (`BUILD-INFO.txt` in the archive states the floor). No job
opens a Gaudi card, and nothing runs on a schedule.

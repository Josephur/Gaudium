# Gaudium

Gaudium is an inference engine for the **Intel Gaudi 2** (HL-225 HPU), built from scratch
for that chip: its MME matrix engines, TPC vector cores, HBM, DMA engines and SynapseAI's
compiled-recipe launch model. It has its own speculative decoding, whose output is
token-identical to plain one-token-at-a-time decoding, and an OpenAI-compatible HTTP/SSE
server with a bundled Web UI.

## Status: mid-port from C++ to Rust

> **Gaudium is in the middle of a port from C++ to Rust.** The project started as a C++
> engine. Since November 2026 it is being rewritten in **Rust with a formally verified
> Bend 2 core** (see [Architecture](#architecture)). The rewrite is not a line-by-line
> translation: each part is redesigned and has to match or beat the C++ engine's measured
> speed on the card before it replaces it.
>
> This repository holds only the Rust engine. Some features of the C++ engine are not
> ported yet, interfaces and flags still change, and nothing here is a stable release.

What the Rust engine does today:

- Runs Qwen3-family models (dense Qwen3, Qwen3.5 / Qwen3.8 hybrid DeltaNet + attention
  models, and Qwen3.6 MoE) and Gemma 4, in BF16 or FP8 weights.
- Speculative decoding with several drafter kinds (DSpine, a block drafter, a model's
  own MTP head or assistant drafter), always token-identical to the non-speculative run.
- Image input for vision-language checkpoints, also under the speculative methods that
  passed the vision regression suite ([`docs/vision-suite.md`](docs/vision-suite.md)).
- Long contexts (up to 256K on the models that support it), planned against HBM at load
  time: a context that does not fit is refused at start with the exact shortfall.
- An FP8 K/V cache for the Qwen3.5 / 3.8 hybrids (`--kv-cache fp8`, scale tables from
  `--kv-calibrate`): half the K/V bytes per position, so a longer context fits.
- Prefix reuse between consecutive requests, and from host memory when the card no longer
  holds a conversation (another conversation or model in between, the idle release of the
  card): `--cache-ram`.
- Several requests at once (`--parallel N`): each in a lane of its own (its own K/V rows,
  recurrent state and prefix reuse), the decoding lanes stepped together in one launch that
  reads the weights once, every stream token-identical to the same request run alone. Dense
  Qwen3 / Qwen3.5 / Qwen3.8 models, without a drafter or with DSpine, for now.
- An OpenAI-compatible (and Anthropic-style) chat API with streaming, tool calls, PDF input
  (its text layer, read by the server's own PDF reader), an optional MCP proxy for the Web UI,
  optional HTTPS on the same port, and A2UI interfaces.

## Architecture

Gaudium follows **"Verified Core, Rust Shell"**, built with
[RustyBends](https://github.com/Josephur/RustyBends):

| Part | Where | What |
|---|---|---|
| Verified core | `gaudium/verified/*.bend` | Decision logic written in [Bend 2](https://github.com/bendlang/bend) with proved `law`s: device and recipe lifecycles, KV-cache and HBM accounting, token / context budgets, tensor bounds, request routing. Proofs are checked at `cargo build` time and erased; the functions are emitted as plain Rust. |
| Rust shell | `gaudium/src/` | SynapseAI FFI, tensor buffers, Safetensors and GGUF loading (host dequantisation of the GGUF block types), tokenizer, chat templates, sampling, the HTTP server and CLI. No crates beyond the Rust standard library at run time. |
| Device kernels | `kernels/` | TPC-C kernels (compiled with `tpc-clang`) and the Synapse Graph Compiler glue plugins. |
| Web UI | `ui/` | A self-contained browser UI served by `gaudium --serve`. |

`rusty-bends-build/` is a copy of the RustyBends build crate (a build dependency only, so
nothing of Bend reaches the binary). Proved laws are locked in `gaudium/rusty-bends.lock`
so that a later change cannot quietly weaken one.

## Releases

The [Releases](https://github.com/Josephur/Gaudium/releases) page has archives with the
`gaudium` binary, the Web UI and the kernel sources. They hold no Intel Gaudi software and no
compiled kernel: install the Intel Gaudi driver and SDK from Intel, then build the kernels with
`make -C kernels` in the unpacked archive. `INSTALL.md` in the archive has the steps, and
`gaudium --version` prints the version. [`docs/releasing.md`](docs/releasing.md) describes how
a release is cut.

## Building

You need:

- A Linux host with an Intel Gaudi 2 card and the
  [Intel Gaudi SDK](https://docs.habana.ai/en/latest/Installation_Guide/index.html) (SynapseAI,
  `tpc-clang`, headers under `/usr/include/habanalabs`) to build the kernels and run on the card.
  Install it from [Intel's package repository](https://vault.habana.ai/) (`habanalabs-dkms`,
  `habanalabs-firmware`, `habanalabs-graph`, `habanalabs-tools`; tested with 1.24). Its license
  does not allow redistribution, so Gaudium does not include any of it.
- Rust (stable) and [Bend 2](https://github.com/bendlang/bend) at the version in
  `gaudium/bend.pin` (looked up at `~/.bend/bin/bend`, or set `BEND_BIN`).
- Lean (via `~/.elan/bin` or `PATH`), so the build also kernel-checks every proof. The
  commands below use `BEND_VERDICT=require`, which fails without Lean; on a host without
  it, leave `BEND_VERDICT` unset (proofs are then only type-checked, with a warning) or set
  it to `off`.

```bash
# Kernels and glue plugins (on the Gaudi host, needs the Intel Gaudi SDK)
make -C kernels -j"$(nproc)"

# Engine: checks every Bend law (needs Lean with BEND_VERDICT=require), then builds
cd gaudium
BEND_VERDICT=require cargo build --release
cargo test

# Web UI tests
cd ../ui && node --test test/
```

The verified core and the Rust tests build and run without a Gaudi card; only running a
model needs one.

## Running

```bash
gaudium --help                      # every flag and environment variable
gaudium --version                   # the version
gaudium                             # verified-core self-check
gaudium --hpu-smoke                 # small live test on the card
gaudium --serve --model <model_dir> --kernels <kernel_build_dir> --ui-dir ui \
        --max-ctx 32768             # API + Web UI on port 8090
gaudium --make-cert <dir> --cert-names <host>,<address>   # a certificate for HTTPS (below)
```

[`docs/reference.md`](docs/reference.md) has the full `--help` text and every environment
variable; [`docs/mcp-proxy-https.md`](docs/mcp-proxy-https.md) covers the MCP proxy and HTTPS
(with the certificate `--make-cert` makes and how to trust it on each device);
[`docs/vision-suite.md`](docs/vision-suite.md) the vision regression suite
(`gaudium --vision-suite`).

Only one process may open the card at a time.

## Contributing

Read [`AGENTS.md`](AGENTS.md) (the project rules: verified-first logic, exactness, memory
planning, repo hygiene) and [`BEND_GUIDE.md`](BEND_GUIDE.md) before changing code. Every
change lands through a pull request.

## License

Gaudium is released under the [MIT License](LICENSE). The copy of `rusty-bends-build` is
Apache-2.0, like RustyBends. Third-party libraries bundled with the Web UI are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

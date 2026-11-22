# AGENTS.md — Gaudium (Rust + RustyBends v1 for Intel Gaudi 2)

Gaudium is a from-scratch inference engine for the **Intel Gaudi 2 (HL-225 HPU)** written in Rust using **RustyBends v1 ("Verified Core, Rust Shell")**.

If a gitignored `AGENTS.local.md` file exists in the repository root, read it too for site-specific dev/HPU host topology, local model paths, and lab operating rules; never copy site-local hostnames, paths, or private operational details into tracked files.

---

## 1. First Principle: Built for Gaudi 2, Not Ported from a GPU Engine

Gaudium starts from the chip itself, never from GPU assumptions adapted afterwards:

- **Design from the real hardware.** Start every design from Gaudi 2's actual resources and costs: MME geometry, TPC VLIW/SIMD cores, HBM bandwidth, DMA engines, SRAM/VLM capacity, and SynapseAI's compiled-recipe launch model. Base decisions on hardware measurements and documented evidence (cite source and tier, or report what you measured).
- **Reference stacks are a yardstick, never a blueprint.** Never copy code, kernels, scheduling, graph structure, or architecture word for word from other inference engines or SDK samples (respect all third-party licenses). Never answer a design question with "how a GPU engine does it." When another stack is faster at something, measure *why* Gaudium is slower and design past it in Gaudium's own idiom.
- **No third-party comparator names in public text.** Do not name third-party stacks in source code, comments, or public docs outside explicitly published benchmark tables.
- **Originality earns its place by measured card results.** Keep a novel mechanism only when it proves a real win on Gaudi 2 (speed, exactness, or memory). If a plain technique is the fastest fit for the hardware, use it; if a clever idea loses on the card, drop it and document why.
- **One card first, without closing the door on multi-card.** Single-card Gaudi 2 performance is the primary target, while keeping per-device state encapsulated and avoiding mandatory defaults that depend on inter-card NIC engines.

---

## 2. Gaudi 2 Hardware Research Mandate

**Always use the Gaudi Grimoire MCP AND a web search combined whenever you need information about Intel Gaudi 2 hardware, and synthesize both sources before concluding.**

Never rely on memory or GPU analogies for Gaudi 2 MME geometry, TPC VLIW/SIMD instructions, FP8/BF16 numerics, HBM/SRAM/VLM limits, DMA engines, or SynapseAI API behavior. Cite sources and evidence tiers for hardware and model facts, and measure before claiming performance or runtime behavior.

---

## 3. Architecture: "Verified Core, Rust Shell" (RustyBends v1)

- **Verified Core (`gaudium/verified/*.bend`)**:
  All pure domain logic and safety-critical invariants live here, written in real **Bend 2 (2.0.40+)** and proved with checked `law` + proof `def` pairs:
  - HPU / SynapseAI device, `AllocGuard`, and recipe lifecycle state machines.
  - KV-cache & host-RAM block/page pool accounting (`allocated + free == capacity`, no double-free/underflow, slab & extent bounds, prefix tile-boundary rules).
  - HBM pre-allocation planning & 5-category ledger accounting.
  - MoE hybrid expert HBM slot cache lifecycle & epoch-quiescence rules.
  - Request / batch / lane / multi-model router lifecycle state machines (idempotent cancellation, exclusive vision isolation, single-resident card exclusivity).
  - Tensor shape, stride, padding, tile/memory alignment, channel/KV permutations, lane relocation, and draft-tree structural bounds.
  - Token / context budget arithmetic (`served_ctx`, prompt admission, remaining budget non-underflow, KV high-water mark, sampler family lattice).
- **Rust Shell (`gaudium/src/*.rs`)**:
  All Intel SynapseAI / HCL FFI, raw DMA/HBM pointers, BF16/FP16/FP8 tensor buffers, Safetensors/GGUF I/O, host CPU reference twins, tokenizers, Jinja2 templates, HTTP/SSE/Unix-socket serving, threads, and CLI glue live in standard Rust. Verified functions and ADTs are imported via:
  ```rust
  include!(concat!(env!("OUT_DIR"), "/verified.rs"));
  ```
- **Device Kernels (`tpc-clang` / Synapse GC plugin)**:
  Gaudi 2 TPC-C kernels (`*.c` compiled with `tpc-clang` into VLIW ELFs) and the Synapse Graph Compiler glue plugin remain device-side artifacts built with the Intel Gaudi SDK; the Rust engine binds, fingerprints, and validates their shapes via the Verified Core. Respect deinterleaved lane-order conventions on the hot path.
- **Zero Bend Runtime in Final Binary**:
  `rusty-bends-build` MUST appear ONLY under `[build-dependencies]` — NEVER under `[dependencies]`. All proofs and `q=0` terms are verified at `cargo build` time and completely erased before Rust emission.

---

## 3a. Verified-First Rule (applies to every change)

- **Start in `gaudium/verified/`.** Before writing Rust for a feature or fix, list its decisions: state machines, policies and permission checks, priority and selection rules, counters and accounting, index and bounds arithmetic. Each one is written as a Bend function with `law`s stating what must always hold, and the Rust shell calls that function. Re-implementing a verified rule in Rust (a condition that duplicates it) is a defect.
- **A change that adds or alters such logic adds or extends laws in the same change.** "Adding brand-new laws is allowed" (section 5) does not mean optional: a slice without laws is unfinished.
- **Only what Bend cannot state stays unproved:** I/O, FFI, byte and text handling, and formats defined by an outside implementation. That code gets Rust tests (differential tests against the reference where one exists).
- **Report it.** Every hand-off lists the laws added, the Rust call sites that use them, and what was left in the shell and why.

---

## 4. Bend 2.0.40 + RustyBends Emitter Conventions

Read `BEND_GUIDE.md` before writing `.bend` code. Key rules enforced by Bend 2.0.40 and `rusty-bends-build/src/emit.rs`:

1. **Always `import Base`** at the top of every `.bend` file.
2. **ADTs use `is Data:` and `{}` constructors:**
   ```bend
   type DevState is Data:
     DevUninit{}
     DevSealed{}
     DevClosed{}
   ```
3. **No `if/else` in Bend 2** — match on `Bool`'s `True{}` / `False{}` constructors, or use `Bool.pick(T, cond, then_val, else_val)`.
4. **Match scrutinees & def headers:** Bend 2 only matches on parameters or bound variables (never computed expressions like `match f(x):`); bind or pass intermediate values to a helper `def`. Both single-line and multi-line `def` headers, nested `match`, multi-scrutinee `match a b:`, and `case 1n+p:` successor patterns are supported.
5. **Exact integer arithmetic semantics (`Nat` vs fixed-width ints):**
   - Use **`Nat`** (emits saturating `u64` + total division) for all conservation, budget, alignment, and bounds math so Peano `Nat` proofs (`Nat.sub(0n, x) == 0n`, `Nat.div(a, 0n) == 0n`, `Nat.mod(a, 0n) == a`) match emitted Rust **exactly**.
   - Note: `Nat.is_zero` is not in Bend 2.0.40 `Base`; use `Nat.is_eq(x, 0n)`.
   - Supported `Base` stdlib calls lowered directly to zero-overhead Rust: `Nat.add`, `Nat.sub`, `Nat.mul`, `Nat.div`, `Nat.mod`, `Nat.divmod`, `Nat.min`, `Nat.max`, `Nat.succ`, `Nat.pred`, `Nat.is_le`, `Nat.is_lt`, `Nat.is_ge`, `Nat.is_gt`, `Nat.is_eq`, `Nat.is_ne`, `Bool.and`, `Bool.or`, `Bool.not`, `Bool.xor`, `Bool.is_eq`, `Bool.is_ne`, `Bool.pick`, `A & B` pairs, `Pair.fst`, `Pair.snd`.
   - Always type-annotate infix operators: `(a + b : Nat)` vs `(a + b : U32)` / `U64` / `I32` / `I64`.
6. **Real Bend 2 `law` + proof `def` syntax:**
   - Theorem: `law <name>: for x: T {lhs == rhs : T}`
   - Proof: `def <name>(x): ... {==}` (same name as the `law`), using `%lemma(args) : {goal_with_underscore}` for rewrites.
   - Standalone helper lemmas returning an equality type (`def lemma(...) -> {lhs == rhs : T}:`) are automatically erased before Rust codegen alongside `law`s and proof `def`s.
7. **Multiple `.bend` files in `gaudium/verified/`:**
   - `rusty-bends-build` checks every `.bend` file individually with `bend` and concatenates all `.bend` files into one `OUT_DIR/verified.rs`. Keep top-level type, constructor, function, and law names unique across all files in `gaudium/verified/`.

---

## 5. Build, Proof Gate & Anti-Cheat Law Lock Rules

- **Toolchain PATH:** `rusty-bends-build` automatically discovers `~/.bend/bin/bend` and `~/.elan/bin` (override with `BEND_BIN` if installed elsewhere); `gaudium/bend.pin` pins the Bend 2 release:
  ```bash
  cargo check
  cargo test
  ```
- **`BEND_VERDICT` modes:**
  - `auto` (default): always enforces `bend <file> --check-only`, plus Lean `bend <file> --verdict` whenever compatible Lean is installed.
  - `require`: fails the build if Lean `--verdict` is unavailable (use in CI and whenever Lean matches Bend's expected kernel version).
  - `off`: `--check-only` only (only when compiling on a host without Lean).
- **Anti-Cheat Law Lock (`gaudium/rusty-bends.lock`):**
  - Once a `law` is locked in `gaudium/rusty-bends.lock`, **NEVER weaken, edit, comment out, or delete it** to make `cargo check` pass.
  - If `E_BEND_CHECK_FAILED` or `E_LAW_LOCK_TAMPERED` fires, fix the implementation or proof — NOT the law.
  - If a locked law itself contradicts the real hardware or protocol specification, **STOP and ask the user** before touching any locked law.
  - Adding brand-new laws for new slices is allowed; always run `cargo check` *without* `RUSTY_BENDS_UPDATE_LAWS=1` first to confirm zero locked laws changed, and only seal new laws into `rusty-bends.lock` with user approval.
- **Incremental vertical slices:** Keep `cargo check` and `cargo test` green after every slice.

---

## 6. RustyBends Issues — Do Not Patch Locally, File via `gh`

- **NEVER silently modify `rusty-bends-build/`.**
- If you hit a bug, missing Bend 2 primitive, emitter limitation, or CLI mismatch in RustyBends:
  1. Document the minimal `.bend` reproducer and exact error output.
  2. File an issue directly on the RustyBends repository using `gh`:
     ```bash
     gh issue create -R Josephur/RustyBends --title "<concise title>" --body "<minimal reproducer + expected vs actual>"
     ```
  3. Ask the user before making any local modification to `rusty-bends-build/`.

---

## 7. Standing Orders: Exactness, Vision & Measurement

- **Token-identity exactness:**
  1. Every default execution mode MUST be token-identical to the engine's own non-speculative `K=0` run (same RNG seed for sampled requests), across any `K`, batch width, draft tree, lane count, text, or vision prompt.
  2. An opt-in speed mode may fork from `K=0` only at traced numerical near-ties, with each fork logged and margin-traced.
  3. Any mode that is exact only in distribution requires explicit user approval and never becomes a default.
- **Vision must keep working for every speculative method:**
  - Every speculative/prediction method must pass vision regression tests (both semantic correctness and token-identity to greedy `K=0`).
  - Keep RoPE positions separate from KV slot indices (they diverge after an image), feed vision embeddings (not image-pad embeddings) to drafters at image positions, and never let prompt-lookup match inside image-pad spans.
- **Measure before claiming:**
  - Report numbers with the exact configuration and method, and failures with full output.
  - Keep prefill (TTFT / tok/s) and decode (tok/s, tok/launch, ms/launch, acceptance) separate in benchmark records, always noting precision (weights, KV, recurrent state, verify, drafter) and exactness reference.

---

## 8. Gaudi 2 / SynapseAI Runtime & Memory Safety Discipline

- **One card, one process:** Only one process may own `/dev/accel/accel0` at a time; a second opener crashes both. Wait until any previous process has fully released the device and driver teardown completes before opening a new context.
- **Never exit with launches queued:** Exiting while Synapse launches remain queued can fault the device MMU against torn-down host DMA mappings and reset the card. Always drain compute, H2D, and D2H streams before fatal exit or teardown.
- **Checked launches & persistent section exclusivity:** Refuse unbound tensors, duplicate tensor addresses in one launch, and missing workspaces. Never bind a single HBM buffer to two persistent sections (`synLaunch` status 30).
- **Destroy graphs immediately after compiling:** Call `synGraphDestroy` as soon as the recipe is compiled and tensor IDs are extracted; retaining `synGraphHandle` leaks tens of GB of compiler IR in host RAM.
- **Device memory is planned, never discovered:** Sum every device allocation up front (`hbm_plan`), refuse configurations that do not fit with exact capacity/shortfall figures, allocate exactly the plan, and seal allocations (`AllocGuard`). Never allocate device memory dynamically after sealing; lazy recipes share the pre-planned workspace.
- **Compiled-recipe cache & placement guard:**
  - Route all graph construction (tensors, nodes, compiler attributes) through centralized builder entrypoints so recipe fingerprints cover 100% of what Synapse compiles.
  - Hash kernel ELFs excluding build-directory metadata so identical sources share cached recipes across worktrees.
  - Scan compiled recipes to ensure Synapse never places compiler intermediates inside persistent rings, KV caches, or recurrent state buffers; fix violations with `set_noreuse`, never by silencing checks.
  - Whenever an unexplained runtime failure, garbage output, or unexpected perf change occurs, suspect stale cached recipes first and verify with cache disabled.
- **Host RAM & CPU-tool hygiene:**
  - Budget pinned host memory (`synHostMap`), anonymous buffers, and mmap'd weights against host/cgroup limits before running large config sweeps.
  - For CPU-only PyTorch/HuggingFace scripts on Gaudi hosts, set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` and empty `HABANA_VISIBLE_DEVICES=` so Python never grabs the card accidentally.
  - Keep any C/C++ probe or FFI glue symbols namespaced so they cannot collide with unscoped symbols exported by `libSynapse` (e.g., `Graph`).

---

## 9. Public-Repo Hygiene (Enforced From Day 1)

- **Zero site-specific data in tracked files:** Never put private LAN hostnames, private IPv4 addresses, local mount paths, container IDs, usernames, SSH/sudo posture, or operations tooling in tracked source, comments, tests, or docs. Site configuration belongs in gitignored `AGENTS.local.md` and `config.env` (read via `GAUDIUM_*` environment variables).
- **Zero secrets, anywhere:** No API keys, bearer tokens, private keys, passwords, or local MCP/agent configs (`.agents/`, `.env*`, `config.env`) in any tracked file or test log.
- **Zero build outputs or large binaries:** Never commit `target/`, `build/`, `*.o`, `*.so`, `*.elf`, `*.recipe`, model weights (`*.safetensors`, `*.gguf`, `*.bin`, `*.pt`), pixel dumps, or runtime logs/locks. Keep every tracked file small and auditable.
- **Third-party code & licenses:** Any vendored third-party dependency goes under `third_party/` with its license file and an entry in `THIRD_PARTY_NOTICES.md`.
- **Keep docs in sync with code:** Whenever adding, renaming, or removing CLI flags or `GAUDIUM_*` environment variables, update the reference documentation and help text in the same commit.

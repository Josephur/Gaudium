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

- **Start in `gaudium/verified/`.** Before writing Rust for a feature or fix, list the decisions it makes. Each decision of the kinds below is written as a Bend function with `law`s stating what must always hold, and the Rust shell calls that function. Re-implementing a verified rule in Rust (a condition that duplicates it) is a defect.
- **Must be verified:**
  - memory and HBM plans;
  - cache, slot and pool accounting;
  - admission and lifecycle state machines (requests, batches, lanes, models, devices, recipes);
  - alignment, index and bounds arithmetic (shapes, strides, offsets, tiles, permutations);
  - any other policy, permission check or selection rule whose wrong answer can break token identity (section 7) or memory safety.
- **May stay in Rust with Rust tests.** Reviewers, human or automated, do not ask for these to move into Bend:
  - counters, timings and telemetry that are only reported (logged, shown, or returned by a status or metrics endpoint) and decide nothing;
  - parsing of command-line flags, environment variables, config files and request fields, including refusing an out-of-range input before it reaches the engine;
  - formats defined by an outside reference (tokenizers, chat templates, file and wire formats), with differential tests against that reference where one exists;
  - I/O, FFI, threads, and byte and text handling.
- **Where the line falls.** A value leaves the second list as soon as it decides something on the first: a count that admits or refuses work is accounting, and a limit that a memory plan relies on is a bound. The decision is then verified where it is made (in the planner, or in the admission check). The parser that feeds the decision stays in Rust.
- **A change that adds or alters such logic adds or extends laws in the same change.** "Adding brand-new laws is allowed" (section 5) does not mean optional: a slice without laws is unfinished.
- **A law states what must hold for every future version of the function, never what the function happens to return today.** A sealed law can never be weakened, so it must not fix a default, a tuning constant, a threshold or an exact formula that a later measurement may change. Seal bounds and safety properties (a result stays inside its range, a count is conserved, a state is never left, an unsafe combination is refused); keep the chosen default, the constant and the formula in the function body and pin today's values with Rust tests, which may change with the code. Before sealing a law, ask what legitimate future change it would forbid: if there is one, weaken the law to the bound behind it.
- **A law constrains the function the Rust shell calls.**
  - It names that function, or a function proved in Bend to equal it (an equivalence lemma).
  - A law stated only through a proof helper, a restatement written for the law, or a value the shell computes but never reads does not count.
  - Where a size or a count has two sides, bound both: it covers what it must, and it stays inside what it may.
  - Where the exact result is fixed by something outside the engine's tuning (a kernel's byte order, a file format), the law may compare the function with a small reference function written plainly in Bend. Anything a later measurement may change stays a bound.
- **Every new law names the wrong body it rules out.**
  - A `# catches:` line in the comment block above the law, beside `# promise:`, names one plausible wrong version of the code that the law refuses. Examples: `# catches: sum replaced by max`, `# catches: alignment of the part offset ignored`.
  - The author changes the code to that body in a scratch copy and shows `cargo check` failing with `E_LAW_FALSE` and a concrete input. A failed proof with "no counterexample" does not count, because the law may still hold for that body.
  - RustyBends ignores the line and does not store it in the lock. Adding one, also to a sealed law, never changes `rusty-bends.lock`.
  - `gaudium/tests/law_comments.rs` fails `cargo test` when a law lands without one.
- **Report it.** Every hand-off lists:
  - the laws added, each with its `# catches:` body and the Rust call sites of the function it constrains;
  - what was left in the shell, and why;
  - **law findings:** the proof failures that forced a code fix (the law, what was wrong in the code, the commit), and the weak laws found in review (the law, the wrong body that still passed, the fix). Write "none" when there were none. These lines are how the project measures what its laws catch.

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
   - Use **`Nat`** for all conservation, budget, alignment, and bounds math. It emits `u64` with checked growth (`Nat.add`, `Nat.mul`, `Nat.succ` panic with `rusty-bends: Nat overflow` past `u64::MAX`), saturating `Nat.sub` / `Nat.pred`, and total division, so the emitted Rust agrees with the Peano `Nat` proofs (`Nat.sub(0n, x) == 0n`, `Nat.div(a, 0n) == 0n`, `Nat.mod(a, 0n) == a`) wherever it returns.
   - A growth op on a value a client or a file controls can now panic. Write limit checks in a form that cannot overflow (compare `h` with `Nat.div(max, w)` rather than `w * h` with `max`). `RUSTY_BENDS_LIST_PANICS=1 cargo check` lists every growth op that can panic. A `def` with `# rusty-bends: total` in the comment block above it emits as `Option<T>` and returns `None` instead of panicking.
   - Note: `Nat.is_zero` is not in Bend 2.0.40 `Base`; use `Nat.is_eq(x, 0n)`.
   - Supported `Base` stdlib calls lowered directly to zero-overhead Rust: `Nat.add`, `Nat.sub`, `Nat.mul`, `Nat.div`, `Nat.mod`, `Nat.divmod`, `Nat.min`, `Nat.max`, `Nat.succ`, `Nat.pred`, `Nat.is_le`, `Nat.is_lt`, `Nat.is_ge`, `Nat.is_gt`, `Nat.is_eq`, `Nat.is_ne`, `Bool.and`, `Bool.or`, `Bool.not`, `Bool.xor`, `Bool.is_eq`, `Bool.is_ne`, `Bool.pick`, `A & B` pairs, `Pair.fst`, `Pair.snd`.
   - Always type-annotate infix operators: `(a + b : Nat)` vs `(a + b : U32)` / `U64` / `I32` / `I64`.
6. **Real Bend 2 `law` + proof `def` syntax:**
   - Theorem: `law <name>: for x: T {lhs == rhs : T}`
   - Proof: `def <name>(x): ... {==}` (same name as the `law`), using `%lemma(args) : {goal_with_underscore}` for rewrites.
   - Standalone helper lemmas returning an equality type (`def lemma(...) -> {lhs == rhs : T}:`) are automatically erased before Rust codegen alongside `law`s and proof `def`s.
   - Every new law has a one-sentence `# promise:` comment directly above it (`# harm:` optional), or the build fails with `E_LAW_NO_PROMISE`. It also has a `# catches:` line (section 3a). A new law must also be shown false for at least one small mutant of the code it depends on, or the build fails with `E_LAW_VACUOUS`. That check stops at the first mutant it catches and names only that one, so passing it does not show that the law rules out the wrong bodies that matter; the `# catches:` body does. `RUSTY_BENDS_MUTATE=report cargo check` builds every mutant of every law and lists each as `caught` or `survived` (also in `OUT_DIR/rusty-bends-mutants.txt`); a survivor is a wrong body the law still allows. Each law also runs as a generated Rust test on the emitted code under `cargo test`. When a proof fails, `E_LAW_FALSE` with a concrete input means the code breaks the law; "no counterexample" means only the proof is incomplete.
   - **Shared lemmas.** `gaudium/verified/nat_lemmas.bend` is the library RustyBends ships (`import nat_lemmas.bend as L`): `build.rs` rewrites it on every build, so never edit it. It covers Euclidean division: a split `q * b + r` with `r < b` determines `q` and `r` (`nat_split_unique_q`, `nat_split_unique_r`), its quotient is `q` and its remainder is `r` (`nat_div_split`, `nat_mod_split`), and it stays below `n * b` when `q < n` (`nat_split_lt`). It also covers multiples of `b` (`nat_mod_mul`, `nat_div_mul`, `nat_align_down_le`). Prove a bound or an injectivity over `Nat.div` / `Nat.mod` with these lemmas, not by evaluating every value: the `--verdict` kernel runs out of fuel on long evaluations. The `kv_wv_row` laws in `kv_cache.bend` are the worked example. A lemma that a second file needs and the library lacks goes in `gaudium/verified/lemmas.bend` (`import lemmas.bend as G`; create the file with its first lemma), rather than in a file-local copy under a new prefix.
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
  - **The build cache.** RustyBends skips every check when `OUT_DIR/rusty-bends.cache` matches the build. The cache records the content of the `.bend` files, a hash of the `rusty-bends-build/` sources, whether the verdict ran (`ran`, `skipped` or `off`), the Bend binary and version, and the check and verdict flags. So `BEND_VERDICT=require cargo check` after an `off` or skipped build runs `--verdict` again, a pass under `require` means the verdict ran on these sources, and the first build after updating `rusty-bends-build/` runs every check and writes a new `verified.rs`. `RUSTY_BENDS_NO_CACHE=1` forces a full run.
- **Anti-Cheat Law Lock (`gaudium/rusty-bends.lock`):**
  - Once a `law` is locked in `gaudium/rusty-bends.lock`, **NEVER weaken, edit, comment out, or delete it** to make `cargo check` pass.
  - If `E_BEND_CHECK_FAILED` or `E_LAW_LOCK_TAMPERED` fires, fix the implementation or proof — NOT the law.
  - If a locked law itself contradicts the real hardware or protocol specification, **STOP and ask the user** before touching any locked law.
  - Adding brand-new laws for new slices is allowed; always run `cargo check` *without* `RUSTY_BENDS_UPDATE_LAWS` first to confirm zero locked laws changed.
  - **What the update flag does.** `RUSTY_BENDS_UPDATE_LAWS` writes the lock only after the proofs and the mutation checks pass, and the build prints a `RustyBends lock:` line for every law it sealed, re-accepted, removed or whose comments it rewrote. Read those lines and the lock's diff before committing.
    - `RUSTY_BENDS_UPDATE_LAWS=<law>[,<law>...]` seals the named new laws and keeps every other check: other new laws stay unsealed, and a changed or deleted locked law that is not named still fails with `E_LAW_LOCK_TAMPERED`. This is the form used to seal reviewed laws. Naming a locked law re-accepts its changed statement, or removes it from the lock if it was deleted, so never name one without the user's OK (rule above).
    - `RUSTY_BENDS_UPDATE_LAWS=comments` rewrites the `# promise:` / `# harm:` / `# unchecked:` lines of locked laws whose statement is unchanged, and seals nothing.
    - `RUSTY_BENDS_UPDATE_LAWS=1` turns every lock check off and rewrites the whole lock: it seals every new law and re-accepts every changed locked law (printed as `RE-ACCEPTED changed law`). Do not seal with it.
  - **The comments of a sealed law.** A sealed law's `# promise:` and `# harm:` lines are stored in the lock, outside the law's hash. Rewording them is a change to a locked law: ask the user first. A reviewer then confirms that only the comment changed. `cargo check` without the update flag passes and prints `promise wording changed (lock hash unchanged)`, and the lock's diff changes only that law's `# promise` / `# harm` lines (as in #77). Refresh the lock with `RUSTY_BENDS_UPDATE_LAWS=comments`, which seals no other law. `# catches:` lines are not in the lock, so they may be added to a sealed law without asking.
- **New laws are sealed after one independent review, not by asking the user:**
  - **The author checks first.** Before asking for review, the author takes each new law through section 3a:
    - it belongs in Bend (the "must be verified" list);
    - it states a bound, not a pinned value;
    - it names the function the Rust shell calls (`RUSTY_BENDS_LIST_COVERAGE=1 cargo check` lists the laws that name only functions no Rust source mentions, and the functions no law reaches; a mention in a test counts, so a clean listing does not prove the call site);
    - it bounds both sides where there are two;
    - its `# catches:` body fails with `E_LAW_FALSE`.
    The author also answers "what wrong body still passes?". Try at least: a sum replaced by a max, a dropped term or condition, an off-by-one, a constant result, and an unaligned value where alignment matters. The law's survivors in `RUSTY_BENDS_MUTATE=report cargo check` are further candidates. A survivor that breaks the promise calls for a stronger law before the review.
  - **Who reviews.** A second agent, never the session that wrote the laws, running on the most capable model available to it at high reasoning effort or above. It works from the `.bend` sources and the Rust call sites, not from the author's summary.
  - **When.** One review, on the final code. It runs after the author's check, and after the other review findings on the change (automated ones included) are settled, not on every push.
  - **What it checks, law by law.** The law states a bound or a safety property as section 3a requires, and fixes no default, tuning constant, threshold or exact formula. It is not vacuous: it constrains the function it names, and that function is the one the Rust shell calls (no Rust condition restates it). Its proof is checked by `BEND_VERDICT=require cargo check` (the cache records whether the verdict ran, above). The reviewer reads the law's survivors in `RUSTY_BENDS_MUTATE=report cargo check`, tries the `# catches:` body and its own "what wrong body still passes?" candidates, and reports each survivor that breaks the promise. It names the legitimate future change the law would forbid, or says that there is none.
  - **What happens next.** The reviewer writes its verdict for each law in the pull request. Laws that pass are sealed on that pull request's branch with `RUSTY_BENDS_UPDATE_LAWS=<law>,<law>...`, naming exactly those laws. The build prints `RustyBends lock: sealed law '<law>'` for each and no other lock line, and the lock's diff must only add those laws' lines. A corrected law gets a re-review of that law only, not of the whole change. A law the reviewer doubts and that is not corrected is left out of the list and stays unsealed, with the reason stated; it is never sealed to get the change merged.
  - The user is still asked before any law that is already locked is touched (the rule above), and still approves the merge (section 11).
- **Incremental vertical slices:** Keep `cargo check` and `cargo test` green after every slice.

---

## 6. RustyBends Issues — Do Not Patch Locally, File via `gh`

- **NEVER silently modify `rusty-bends-build/`.**
- **RustyBends is a general tool, and Gaudium is one of its users.** Ask it only for what any project could use. Gaudium's own rules belong in Gaudium (this file, `gaudium/verified/`, Rust tests), never in RustyBends. An issue states the need in general Bend/Rust terms with a standalone reproducer whose names are neutral (`fits_area_limit`, `Ledger`), never Gaudium code or vocabulary pasted in: RustyBends declines requests that only one project needs. Before filing, ask: "Would a project that has never heard of Gaudium want this?" If not, or if the request only makes sense under Gaudium's own conventions (this file, its review process, its lock workflow), solve it in Gaudium instead. For example, the `# catches:` rule is enforced by a Gaudium test, not by RustyBends.
- If you hit a bug, missing Bend 2 primitive, emitter limitation, or CLI mismatch in RustyBends:
  1. Write a minimal standalone `.bend` reproducer, with neutral names, and the exact error output.
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
- **Never commit with a machine-made identity:** with no identity configured, git invents one from the login and the machine (`user@host.domain`), and that publishes a private host name in the history, where it cannot be taken back without rewriting it.
  - **Before the first commit** on any machine, clone or worktree, read `git config user.name` and `git config user.email`. If either is empty, or the address is one git made up, set both to the committer's real identity and do not commit until they are right. On the maintainer's machines the identity to use is given in `AGENTS.local.md`; a contributor uses their own.
  - **After committing**, check it: `git log -1 --format='%an <%ae> | %cn <%ce>'` shows the intended name and address twice, and no machine name.
  - **The same holds for everything else that is published:** no host name and no local address in a commit message, a `Co-authored-by` trailer, a pull request, an issue or a comment. A squash merge is given its subject and body explicitly, so that no trailer is generated from the branch's commits.
  - An agent's own attribution trailer with its vendor's `noreply` address is not affected.
- **Zero secrets, anywhere:** No API keys, bearer tokens, private keys, passwords, or local MCP/agent configs (`.agents/`, `.env*`, `config.env`) in any tracked file or test log.
- **Zero build outputs or large binaries:** Never commit `target/`, `build/`, `*.o`, `*.so`, `*.elf`, `*.recipe`, model weights (`*.safetensors`, `*.gguf`, `*.bin`, `*.pt`), pixel dumps, or runtime logs/locks. Keep every tracked file small and auditable.
- **Third-party code & licenses:** Any vendored third-party dependency goes under `third_party/` with its license file and an entry in `THIRD_PARTY_NOTICES.md`.
- **Every flag and environment variable is documented, in the same change:** A command-line flag the binary accepts has its line in `gaudium --help` (`print_usage` in `gaudium/src/main.rs`). An environment variable the engine reads (Rust or kernel glue) is named in `--help` too, and one that only tests read is listed in `docs/reference.md`. `docs/reference.md` holds a copy of the help text: refresh it with `GAUDIUM_UPDATE_REFERENCE=1 cargo test --test reference_docs` and commit it. Renaming or removing a flag or variable updates the same places. The test `gaudium/tests/reference_docs.rs` reads the sources and fails `cargo test` on a flag or variable that is missing, stale, or not in the copy; never get past it by building a name at run time, by moving a read into test code, or by adding to the test's exception lists (those are for other programs' flags and for the general process environment such as `HOME`).
- **Keep the other docs in sync with code:** a change that alters what a document describes updates that document in the same commit.

---

## 10. Leave the Card Serving When Testing Is Done

When card testing is complete, leave the server running for the user with this configuration (site-specific service names, model paths and run helpers are in `AGENTS.local.md`):

- **Model:** Qwen 3.8 27B (FP8 weights, the resident default).
- **Speculative decoding:** DSpine with adaptive draft widths (`--draft-model <id=dir>` with a DSpine drafter; keep the adaptive `--draft-k` default).
- **Vision:** on (the vision-language checkpoint, with its vision tower loaded).
- **Context:** 256K (`--max-ctx 262144`, or `--model-ctx <id>=262144`).
- **MCP proxy:** on.
- **A2UI:** on (`--a2ui`).

Before ending the session, confirm that the server is up and answers a request from the Web UI. If any of these features cannot be turned on yet, say which one in the hand-off rather than leaving a different configuration silently.

---

## 11. Every Change Lands Through a Pull Request

`main` moves only by merging pull requests. This holds for every change by every author, including one-line fixes and documentation.

- **One task, one branch, one PR.** Start each task on a new branch from the latest `main`, in a git worktree of its own under `.worktrees/` (`git worktree add .worktrees/<name> -b <type>/<name> origin/main`). Never edit files in the main checkout, never commit on `main`, and never put two unrelated tasks on one branch.
- **Name the issue.** The PR description says which issue the change belongs to: `Closes #N` when it meets everything the issue asks for, `Refs #N` when it is a part. Work without an issue gets one first, or one line in the PR on why none is needed.
- **Green before it is opened for review.** `BEND_VERDICT=require cargo check`, `cargo test` and the Web UI tests (`node --test test/` in `ui/`) pass on the branch. The description carries the hand-off of section 3a (laws added with their `# catches:` bodies, the Rust call sites that use them, what stayed in the shell and why, and the law findings), says whether `rusty-bends.lock` changed (with the review verdict for every law it seals, section 5), and gives card measurements with their configuration when the change claims any (section 7).
- **Squash merge, then clean up.** A PR merges as one commit once its checks pass and the user has approved it; the merge is given its message explicitly (section 9, commit identity). At merge its branch is deleted and its worktree removed (`git worktree remove`).
- **Nothing is left uncommitted.** At the end of a session everything it wrote is committed and pushed on its branch; unfinished work is a draft PR that says what is missing. A task that is given up has its PR closed with a note and its worktree removed.
- **Two tasks, one file.** The task that merges second rebases onto `main` and runs the checks again before it merges.

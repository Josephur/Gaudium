---
name: verified-first
description: Gaudium's verified-first workflow. Load it before you change a decision in gaudium/src/ or gaudium/verified/ (a state machine, a policy or permission check, a selection rule, a counter or accounting, or index, offset and bounds arithmetic), to list the decisions, write them in Bend with laws worth having, read the build's verdict, and write the hand-off.
---

# Verified first: laws worth having

`AGENTS.md` sections 3a and 5 are the rules; this skill is the order to apply them in while you
make the change. Read `BEND_GUIDE.md` before writing Bend. Run every command below from
`gaudium/`.

## 1. When it applies

The change adds or alters, in `gaudium/src/` or `gaudium/verified/`:

- a state machine (requests, batches, lanes, models, devices, recipes);
- a policy or a permission check;
- a selection rule;
- a counter or accounting that admits or refuses work, or a pool, cache or slot ledger;
- index, offset, stride, alignment or bounds arithmetic.

It does not apply to what section 3a lets stay in Rust with Rust tests: counters and timings that
are only reported, parsing of flags, variables, config files and request fields, formats defined
by an outside reference, and I/O, FFI, threads and byte or text handling. A value leaves that
list as soon as it decides something: a count that admits or refuses work is accounting.

## 2. List the decisions first

Before writing Rust, write down each decision the change makes, one line each ("a request whose
prompt exceeds the served context is refused"). Each one becomes a `def` in a `gaudium/verified/`
file, and the Rust shell calls the emitted function (`include!(concat!(env!("OUT_DIR"),
"/verified.rs"))`). A Rust condition that restates a verified rule is a defect: call the function
instead, and pass it what it needs.

## 3. Write laws in pairs

Each "never does the bad thing" law gets a weak "still does its job" companion, so the pair can't
pass because the code does nothing. `gaudium/verified/a2ui.bend` is the model: `a2ui_refusal_never_injects` (a
refused request never also gets the instructions) beside `a2ui_enabled_request_runs` (with A2UI on
and asked for, the request runs). Keep the companion weak, for example "some valid request is
accepted", so it doesn't pin a threshold.

## 4. Bounds, not pins

A sealed law can never be weakened. Before you seal one, answer: which legitimate future change
would this law forbid? If there is one, weaken the law to the bound behind it. Seal ranges,
conservation, absorbing states and refused combinations; keep defaults, tuning constants,
thresholds and exact formulas in the function body, and pin today's values with Rust tests.
Where a size or a count has two sides, bound both (it covers what it must, and stays inside what
it may).

## 5. Ordering questions as step functions

For bugs where steps land in a bad order (cancel twice, release twice, launch after drain), keep
the state in a Bend type, write one step function over every state and every event, and prove
that each pair keeps the invariant. That proof holds for runs of any length. The Rust side holds
the Bend state and changes it only through the step function. `gaudium/verified/lifecycle.bend` is the model:
`req_step` with `req_cancel_idempotent` and `req_terminal_absorbing`, held in Rust by
`RequestGuard` in `gaudium/src/core/lifecycle.rs`, which changes its state only through
`req_step` and `req_cancel`.

## 6. Promise, catches, and showing it

Every new law has, in the comment block directly above `law <name>:`:

- `# promise:` one sentence on what must always hold (the build fails with `E_LAW_NO_PROMISE`
  without it);
- `# catches:` one plausible wrong body the law refuses, such as `# catches: sum replaced by max`
  (`gaudium/tests/law_comments.rs` fails `cargo test` without it).

State the law through the function the shell calls, not a proof helper or a restatement written
for the law. Then show the `# catches:` body failing: change the code to it in a scratch copy,
run `cargo check`, and see `E_LAW_FALSE` with a concrete input. Undo the scratch change.

## 7. Read the build's verdict

| The build says | It means | Do |
| :--- | :--- | :--- |
| `[E_LAW_FALSE]` with inputs and `lhs` / `rhs` values | the code breaks the law | Fix the code. If the change is meant to break a locked law, stop and ask the user. Never edit, weaken, comment out or delete a locked law. |
| "no counterexample up to ...: the law may be true; the proof is what failed" | the law may hold; the proof is incomplete | Fix the proof (the `nat_lemmas.bend` lemmas, a helper lemma, a `match` on the right variable). |
| `[E_LAW_VACUOUS]` | the new law catches none of the build's mutants | Strengthen the law. |
| `[E_LAW_LOCK_TAMPERED]` | a sealed law changed | Undo that change. |
| `[E_LAW_UNCHECKED]` | the checker can't evaluate the new law | Rewrite it in a supported form, or add `# unchecked: <why>`. |
| `[E_MUTANT_BUILD]` | a mutant didn't compile | A RustyBends bug (step 11), not a weak law. |

`RUSTY_BENDS_MUTATE=report cargo check` lists every mutant of every law as `caught` or
`survived`. A survivor that breaks the law's promise is a wrong body the law still allows:
rewrite the law to refuse it.

What a rejection looks like. Removing the double-free guard in `gaudium/verified/kv_pool.bend` (passing `True{}`
instead of `Nat.is_le(rel, a)` to `pool_free_if` in `pool_free`) stops `cargo check` with:

```
[E_LAW_FALSE] law pool_free_conserved is FALSE for this code (evaluated with emitted Rust):
  a = 0, f = 0, rel = 1
  lhs pool_total(pool_free(PoolAcct{0, 0}, 1)) = 1
  rhs Nat.add(0, 0)                            = 0
fix the function (or, if the change is meant to break this law, stop: the law is locked)
```

The answer to that is to restore the guard, and to report it, not to touch the law.

Other listings worth running before review:

- `RUSTY_BENDS_LIST_LAWS=1 cargo check`: every law as `name: promise`.
- `RUSTY_BENDS_LIST_COVERAGE=1 cargo check`: laws that name only functions no Rust source calls
  (`rust=uncalled`) or only tests call (`rust=tests-only`), and functions no law reaches.
- `RUSTY_BENDS_LIST_PANICS=1 cargo check`: every checked-growth op that can panic. On values a
  client or a file controls, compare with a quotient instead of building a product, or mark the
  `def` `# rusty-bends: total`.

## 8. Be honest about scope

Name what stays unproved and why: FFI, Synapse stream ordering, threads, byte and text formats.
Give each of those Rust tests, and say so in the hand-off.

## 9. Review and seal

Per section 5: check each new law yourself first (it belongs in Bend, states a bound, names the
function the shell calls, bounds both sides, and its `# catches:` body fails). Then one
independent review, by a second agent that did not write the laws, on the most capable model
available at high reasoning effort or above, run on the final code. The reviewer writes a verdict
per law in the pull request. Seal the laws that pass with `RUSTY_BENDS_UPDATE_LAWS=<law>,<law>`,
naming exactly those, and check that the lock's diff only adds their lines. `BEND_VERDICT=require
cargo check` must pass on the branch.

## 10. The hand-off

The pull request description lists:

- each new law as `name: promise` (the `RUSTY_BENDS_LIST_LAWS=1` form), with its `# catches:`
  body and the Rust call sites of the function it constrains;
- what stayed in the shell, and why;
- law findings: the proof failures that forced a code fix (the law, what was wrong, the commit)
  and the weak laws found in review (the law, the wrong body that still passed, the fix), or
  "none";
- whether `rusty-bends.lock` changed, with the review verdict for every law it seals.

## 11. Problems in RustyBends

A bug, a missing primitive or an emitter limitation in RustyBends is filed with
`gh issue create -R Josephur/RustyBends`, with a standalone reproducer whose names are neutral,
and never patched in `rusty-bends-build/` (section 6). Ask the user before any local change there.

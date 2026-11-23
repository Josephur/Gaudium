fn main() {
    // 1. Checks bend.pin / BEND_PIN_VERSION (if configured)
    // 2. Enforces token-canonical rusty-bends.lock (anti-cheat LAWS lock)
    // 3. Runs real 'bend <file> --check-only' (+ '--verdict' via Lean when available/required)
    // 4. Erases q=0 types, laws, and proofs, emitting pure Rust into OUT_DIR/verified.rs
    // 0. Keeps verified/nat_lemmas.bend equal to the lemma library this RustyBends ships
    //    (proof helpers only, erased before emission; files under verified/ import it)
    rusty_bends_build::write_nat_lemmas("verified").expect("write verified/nat_lemmas.bend");
    rusty_bends_build::verify_and_emit("verified");
}

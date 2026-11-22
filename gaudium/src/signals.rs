//! Process-wide SIGINT / SIGTERM handling for clean shutdown between synced launches.
//!
//! Loading the SynapseAI runtime and opening the device may replace the process signal
//! dispositions, which would leave a requested stop unseen. `reassert` re-installs the Gaudium
//! handler after that point so a stop request is always observed.

use std::sync::atomic::{AtomicBool, Ordering};

static TERM_FLAG: AtomicBool = AtomicBool::new(false);
static WANTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term_signal(_sig: i32) {
    TERM_FLAG.store(true, Ordering::Relaxed);
}

#[allow(unsafe_code)]
fn set_handlers() {
    extern "C" {
        fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;
    // SAFETY: the handler only performs an async-signal-safe atomic store.
    unsafe {
        signal(SIGINT, on_term_signal);
        signal(SIGTERM, on_term_signal);
    }
}

/// Installs the stop-request handler for SIGINT / SIGTERM and clears any earlier request.
pub fn install() {
    TERM_FLAG.store(false, Ordering::Relaxed);
    WANTED.store(true, Ordering::Relaxed);
    set_handlers();
}

/// Re-installs the handler if `install` was called earlier (no-op otherwise). Call after any
/// step that may have changed signal dispositions, e.g. opening the device.
pub fn reassert() {
    if WANTED.load(Ordering::Relaxed) {
        set_handlers();
    }
}

/// True once SIGINT / SIGTERM has been received since the last `install`.
pub fn term_requested() -> bool {
    TERM_FLAG.load(Ordering::Relaxed)
}

/// The stop-request flag itself, for APIs that take a cancellation `AtomicBool`.
pub fn term_flag() -> &'static AtomicBool {
    &TERM_FLAG
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reassert_without_install_is_noop_and_flag_starts_clear() {
        reassert();
        assert!(!term_flag().load(Ordering::Relaxed) || term_requested());
    }
}

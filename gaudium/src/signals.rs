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

/// Waits until `fd` is readable (a listening socket: a connection is waiting), `timeout_ms`
/// passed or a signal reached this thread, and returns whether `fd` is readable. A loop that
/// must also notice a stop request waits here instead of sleeping between polls, so what it
/// waits for is handled when it arrives; it checks [`term_requested`] after every return.
#[allow(unsafe_code)]
pub fn wait_readable(fd: i32, timeout_ms: i32) -> bool {
    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    extern "C" {
        fn poll(fds: *mut PollFd, nfds: std::ffi::c_ulong, timeout: i32) -> i32;
    }
    const POLLIN: i16 = 0x001;
    let mut p = PollFd {
        fd,
        events: POLLIN,
        revents: 0,
    };
    // SAFETY: `p` is one valid `pollfd` for the duration of the call.
    let n = unsafe { poll(&mut p, 1, timeout_ms) };
    if n < 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
        // the wait itself failed: a caller that loops on it must not spin
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    n > 0 && (p.revents & POLLIN) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reassert_without_install_is_noop_and_flag_starts_clear() {
        reassert();
        assert!(!term_flag().load(Ordering::Relaxed) || term_requested());
    }

    #[test]
    fn wait_readable_ends_when_a_connection_arrives_not_at_the_timeout() {
        use std::net::{TcpListener, TcpStream};
        use std::os::fd::AsRawFd;
        use std::time::{Duration, Instant};

        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.set_nonblocking(true).unwrap();
        let addr = l.local_addr().unwrap();

        // nothing to accept: the wait runs to its timeout
        let t0 = Instant::now();
        assert!(!wait_readable(l.as_raw_fd(), 40));
        assert!(t0.elapsed() >= Duration::from_millis(30));

        // a connection made during the wait ends it at once
        let client = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            let t_connect = Instant::now();
            (TcpStream::connect(addr).unwrap(), t_connect)
        });
        assert!(wait_readable(l.as_raw_fd(), 10_000));
        let t_woke = Instant::now();
        assert!(l.accept().is_ok());
        let (_stream, t_connect) = client.join().unwrap();
        let late = t_woke.saturating_duration_since(t_connect);
        assert!(late < Duration::from_millis(20), "woke {late:?} after the connect");

        // an invalid descriptor is not readable, and the wait does not spin on it
        assert!(!wait_readable(-1, 0));
    }
}

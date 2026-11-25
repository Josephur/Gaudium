//! Rust Shell wrappers for the HTTPS listener decisions of `verified/tls_listener.bend`.
//!
//! Whether the server serves TLS, which protocol an accepted connection speaks, whether a
//! plain-HTTP request is answered or sent to HTTPS, and which responses carry
//! Strict-Transport-Security all delegate to the verified core; nothing here decides.

pub use crate::verified::LsHttpMode as PlainHttpMode;
pub use crate::verified::LsStart as TlsStart;
use crate::verified::{
    ls_is_hello, ls_plain_act, ls_plain_served, ls_proto, ls_sends_hsts, ls_start,
    ls_start_refuses, ls_start_serves_tls, ls_start_warns, LsProto,
};

/// What the certificate and key flags mean on this machine. `cert`, `key`: the files were given;
/// `lib_ok`: the TLS library was found; `files_ok`: it accepted the two files.
pub fn tls_start(cert: bool, key: bool, lib_ok: bool, files_ok: bool) -> TlsStart {
    ls_start(cert, key, lib_ok, files_ok)
}

/// The server answers TLS connections.
pub fn serves_tls(s: TlsStart) -> bool {
    ls_start_serves_tls(s)
}

/// The configuration is refused: the server does not start with it.
pub fn start_refused(s: TlsStart) -> bool {
    ls_start_refuses(s)
}

/// The server starts without HTTPS and says so.
pub fn start_warns(s: TlsStart) -> bool {
    ls_start_warns(s)
}

/// The protocol of one accepted connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnProtocol {
    Plain,
    Tls,
    /// A TLS client on a server that does not serve TLS: the connection is closed.
    Drop,
}

/// `first_byte`: the first byte the client sent; `tls_ready`: the server serves TLS.
pub fn conn_protocol(first_byte: u8, tls_ready: bool) -> ConnProtocol {
    match ls_proto(ls_is_hello(u64::from(first_byte)), tls_ready) {
        LsProto::LsPlain => ConnProtocol::Plain,
        LsProto::LsTls => ConnProtocol::Tls,
        LsProto::LsDrop => ConnProtocol::Drop,
    }
}

fn ls_of(p: ConnProtocol) -> LsProto {
    match p {
        ConnProtocol::Plain => LsProto::LsPlain,
        ConnProtocol::Tls => LsProto::LsTls,
        ConnProtocol::Drop => LsProto::LsDrop,
    }
}

/// A plain-HTTP request is answered (`false`: it is sent to HTTPS). `tls_ready`: the server
/// serves TLS; `mode`: `--tls-http`; `loopback`: the client connected over the loopback
/// interface.
pub fn plain_http_served(tls_ready: bool, mode: PlainHttpMode, loopback: bool) -> bool {
    ls_plain_served(ls_plain_act(tls_ready, mode, loopback))
}

/// Responses on a connection of this protocol carry Strict-Transport-Security.
pub fn sends_hsts(p: ConnProtocol) -> bool {
    ls_sends_hsts(ls_of(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_and_tls_clients_are_told_apart_by_their_first_byte() {
        // every method an HTTP request line can start with is plain, with or without TLS
        for b in *b"GPDHOCT" {
            assert_eq!(conn_protocol(b, true), ConnProtocol::Plain);
            assert_eq!(conn_protocol(b, false), ConnProtocol::Plain);
        }
        assert_eq!(conn_protocol(0x16, true), ConnProtocol::Tls);
        assert_eq!(conn_protocol(0x16, false), ConnProtocol::Drop);
    }

    #[test]
    fn start_states_of_today() {
        assert_eq!(tls_start(false, false, true, true), TlsStart::LsStOff);
        assert!(serves_tls(tls_start(true, true, true, true)));
        assert!(start_refused(tls_start(true, false, true, true)));
        assert!(start_refused(tls_start(true, true, true, false)));
        // no library: the server runs, plain HTTP only, and says so
        let s = tls_start(true, true, false, false);
        assert!(start_warns(s) && !start_refused(s) && !serves_tls(s));
    }

    #[test]
    fn plain_http_beside_tls_of_today() {
        use PlainHttpMode::*;
        for mode in [LsHttpRedirect, LsHttpAllow, LsHttpStrict] {
            for lo in [false, true] {
                // no TLS: plain HTTP is all there is
                assert!(plain_http_served(false, mode, lo));
            }
        }
        // the default: other machines go to HTTPS, this machine's loopback clients are answered
        assert!(!plain_http_served(true, LsHttpRedirect, false));
        assert!(plain_http_served(true, LsHttpRedirect, true));
        assert!(plain_http_served(true, LsHttpAllow, false));
        assert!(!plain_http_served(true, LsHttpStrict, true));
        assert!(sends_hsts(ConnProtocol::Tls));
        assert!(!sends_hsts(ConnProtocol::Plain) && !sends_hsts(ConnProtocol::Drop));
    }
}

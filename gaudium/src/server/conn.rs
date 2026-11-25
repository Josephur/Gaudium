//! One accepted client connection: plain HTTP or TLS on the same port.
//!
//! The first byte a client sends says which protocol it speaks (`verified/tls_listener.bend`,
//! `ls_proto`): a TLS handshake record, or the first letter of an HTTP request line. Handlers
//! read and write a [`Conn`] and do not know which of the two it is, except where the protocol
//! decides what is sent: a plain-HTTP request may be sent to HTTPS (`ls_plain_act`), and a TLS
//! connection's responses carry Strict-Transport-Security (`ls_sends_hsts`, [`Conn::hsts_line`]).

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::core::listener::{conn_protocol, sends_hsts, ConnProtocol};
use crate::tls::{TlsServer, TlsStream};

/// How long a client has to send its first byte and to finish the TLS handshake.
const OPENING_TIMEOUT: Duration = Duration::from_secs(30);

pub(super) struct Conn {
    io: Io,
    /// The `Strict-Transport-Security` header line responses on this connection carry, with its
    /// CRLF, or empty.
    hsts: Arc<str>,
}

enum Io {
    Plain(TcpStream),
    /// Shared so a streaming writer can hold a second handle; one thread uses it at a time.
    Tls(Arc<Mutex<TlsStream>>),
}

/// The header line for `max_age` seconds: no `includeSubDomains`, no `preload` (the server
/// speaks for its own name only).
pub(super) fn hsts_header(max_age: u64) -> String {
    format!("Strict-Transport-Security: max-age={max_age}\r\n")
}

impl From<TcpStream> for Conn {
    fn from(s: TcpStream) -> Self {
        Conn {
            io: Io::Plain(s),
            hsts: Arc::from(""),
        }
    }
}

impl Conn {
    /// Takes an accepted socket: looks at the client's first byte and, for a TLS client, runs
    /// the handshake. `None`: the client sent nothing, spoke TLS to a server without a
    /// certificate, or the handshake failed (the socket is closed). Responses on a TLS
    /// connection carry `Strict-Transport-Security: max-age=<hsts_max_age>`.
    pub fn open(tcp: TcpStream, tls: Option<&TlsServer>, hsts_max_age: u64) -> Option<Conn> {
        let _ = tcp.set_read_timeout(Some(OPENING_TIMEOUT));
        let mut first = [0u8; 1];
        match tcp.peek(&mut first) {
            Ok(1) => {}
            _ => return None,
        }
        let proto = conn_protocol(first[0], tls.is_some());
        let io = match proto {
            ConnProtocol::Plain => Io::Plain(tcp),
            ConnProtocol::Drop => return None,
            ConnProtocol::Tls => {
                let _ = tcp.set_write_timeout(Some(OPENING_TIMEOUT));
                let stream = tls?.accept(tcp).ok()?;
                let _ = stream.set_write_timeout(None);
                Io::Tls(Arc::new(Mutex::new(stream)))
            }
        };
        let hsts = if sends_hsts(proto) {
            Arc::from(hsts_header(hsts_max_age))
        } else {
            Arc::from("")
        };
        Some(Conn { io, hsts })
    }

    /// The connection speaks plain HTTP (not TLS).
    pub fn is_plain(&self) -> bool {
        matches!(self.io, Io::Plain(_))
    }

    /// The `Strict-Transport-Security` header line (with its CRLF) every response head on this
    /// connection carries, or `""`.
    pub fn hsts_line(&self) -> &str {
        &self.hsts
    }

    /// A second handle on the same connection.
    pub fn try_clone(&self) -> io::Result<Conn> {
        let io = match &self.io {
            Io::Plain(s) => Io::Plain(s.try_clone()?),
            Io::Tls(s) => Io::Tls(Arc::clone(s)),
        };
        Ok(Conn {
            io,
            hsts: Arc::clone(&self.hsts),
        })
    }

    pub fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        match &self.io {
            Io::Plain(s) => s.set_read_timeout(t),
            Io::Tls(s) => lock(s).set_read_timeout(t),
        }
    }

    /// How long one write may block before it returns `WouldBlock` (`None`: as long as it takes).
    pub fn set_write_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        match &self.io {
            Io::Plain(s) => s.set_write_timeout(t),
            Io::Tls(s) => lock(s).set_write_timeout(t),
        }
    }

    /// Ends the connection at once, without a goodbye that could wait on a client that stopped
    /// reading: every later write fails at once.
    pub fn abandon(&self) {
        match &self.io {
            Io::Plain(s) => {
                let _ = s.shutdown(std::net::Shutdown::Both);
            }
            Io::Tls(s) => lock(s).abandon(),
        }
    }
}

fn lock(s: &Mutex<TlsStream>) -> std::sync::MutexGuard<'_, TlsStream> {
    s.lock().unwrap_or_else(|e| e.into_inner())
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match &mut self.io {
            Io::Plain(s) => s.read(buf),
            Io::Tls(s) => lock(s).read(buf),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &mut self.io {
            Io::Plain(s) => s.write(buf),
            Io::Tls(s) => lock(s).write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.io {
            Io::Plain(s) => s.flush(),
            Io::Tls(s) => lock(s).flush(),
        }
    }
}

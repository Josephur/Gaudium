//! One accepted client connection: plain HTTP or TLS on the same port.
//!
//! The first byte a client sends says which protocol it speaks (`verified/tls_listener.bend`,
//! `ls_proto`): a TLS handshake record, or the first letter of an HTTP request line. Handlers
//! read and write a [`Conn`] and do not know which of the two it is.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::core::listener::{conn_protocol, ConnProtocol};
use crate::tls::{TlsServer, TlsStream};

/// How long a client has to send its first byte and to finish the TLS handshake.
const OPENING_TIMEOUT: Duration = Duration::from_secs(30);

pub(super) enum Conn {
    Plain(TcpStream),
    /// Shared so a streaming writer can hold a second handle; one thread uses it at a time.
    Tls(Arc<Mutex<TlsStream>>),
}

impl From<TcpStream> for Conn {
    fn from(s: TcpStream) -> Self {
        Conn::Plain(s)
    }
}

impl Conn {
    /// Takes an accepted socket: looks at the client's first byte and, for a TLS client, runs
    /// the handshake. `None`: the client sent nothing, spoke TLS to a server without a
    /// certificate, or the handshake failed (the socket is closed).
    pub fn open(tcp: TcpStream, tls: Option<&TlsServer>) -> Option<Conn> {
        let _ = tcp.set_read_timeout(Some(OPENING_TIMEOUT));
        let mut first = [0u8; 1];
        match tcp.peek(&mut first) {
            Ok(1) => {}
            _ => return None,
        }
        match conn_protocol(first[0], tls.is_some()) {
            ConnProtocol::Plain => Some(Conn::Plain(tcp)),
            ConnProtocol::Drop => None,
            ConnProtocol::Tls => {
                let _ = tcp.set_write_timeout(Some(OPENING_TIMEOUT));
                let stream = tls?.accept(tcp).ok()?;
                let _ = stream.set_write_timeout(None);
                Some(Conn::Tls(Arc::new(Mutex::new(stream))))
            }
        }
    }

    /// A second handle on the same connection.
    pub fn try_clone(&self) -> io::Result<Conn> {
        match self {
            Conn::Plain(s) => s.try_clone().map(Conn::Plain),
            Conn::Tls(s) => Ok(Conn::Tls(Arc::clone(s))),
        }
    }

    pub fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        match self {
            Conn::Plain(s) => s.set_read_timeout(t),
            Conn::Tls(s) => lock(s).set_read_timeout(t),
        }
    }

    /// How long one write may block before it returns `WouldBlock` (`None`: as long as it takes).
    pub fn set_write_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        match self {
            Conn::Plain(s) => s.set_write_timeout(t),
            Conn::Tls(s) => lock(s).set_write_timeout(t),
        }
    }

    /// Ends the connection at once, without a goodbye that could wait on a client that stopped
    /// reading: every later write fails at once.
    pub fn abandon(&self) {
        match self {
            Conn::Plain(s) => {
                let _ = s.shutdown(std::net::Shutdown::Both);
            }
            Conn::Tls(s) => lock(s).abandon(),
        }
    }
}

fn lock(s: &Mutex<TlsStream>) -> std::sync::MutexGuard<'_, TlsStream> {
    s.lock().unwrap_or_else(|e| e.into_inner())
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Conn::Plain(s) => s.read(buf),
            Conn::Tls(s) => lock(s).read(buf),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Conn::Plain(s) => s.write(buf),
            Conn::Tls(s) => lock(s).write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Conn::Plain(s) => s.flush(),
            Conn::Tls(s) => lock(s).flush(),
        }
    }
}

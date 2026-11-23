//! TLS through the system's OpenSSL, loaded at run time.
//!
//! The binary links no TLS library. `libssl` is opened with `dlopen` the first time TLS is asked
//! for, and every function is called through a pointer looked up by name, so a machine without
//! the library still runs the server: asking for TLS then returns an error that says what is
//! missing. Only functions present in every OpenSSL 3 release are used (3.0 and 3.5 are both in
//! use), and no symbol is defined here that another loaded library could collide with.
//!
//! Two uses:
//! - [`TlsServer`]: the server's own HTTPS listener (a certificate chain and a private key, read
//!   by the library from the files the operator names; the key never passes through this code);
//! - [`TlsClient`]: connections to HTTPS targets, verified against the system's trust store with
//!   the host name (or address) checked. The caller connects the socket itself, so the name sent
//!   and verified is independent of the address the socket went to.
//!
//! A [`TlsStream`] is a blocking stream over a connected socket. A read that runs into the
//! socket's read timeout returns `WouldBlock`, as a plain socket read does.

#![allow(unsafe_code)]

use std::ffi::{c_char, c_int, c_long, c_ulong, c_void, CStr, CString};
use std::io::{self, Read, Write};
use std::net::{IpAddr, TcpStream};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

/// Library names tried in order (the versioned name every OpenSSL 3 package installs first).
const LIBRARY_NAMES: &[&str] = &["libssl.so.3", "libssl.so"];

type SslCtx = c_void;
type Ssl = c_void;
type PasswdCb = extern "C" fn(*mut c_char, c_int, c_int, *mut c_void) -> c_int;

/// The functions of `libssl` / `libcrypto` this module calls.
struct Api {
    openssl_init_ssl: unsafe extern "C" fn(u64, *const c_void) -> c_int,
    openssl_version: unsafe extern "C" fn(c_int) -> *const c_char,
    tls_client_method: unsafe extern "C" fn() -> *const c_void,
    tls_server_method: unsafe extern "C" fn() -> *const c_void,
    ssl_ctx_new: unsafe extern "C" fn(*const c_void) -> *mut SslCtx,
    ssl_ctx_free: unsafe extern "C" fn(*mut SslCtx),
    ssl_ctx_ctrl: unsafe extern "C" fn(*mut SslCtx, c_int, c_long, *mut c_void) -> c_long,
    ssl_ctx_set_options: unsafe extern "C" fn(*mut SslCtx, u64) -> u64,
    ssl_ctx_set_verify: unsafe extern "C" fn(*mut SslCtx, c_int, *const c_void),
    ssl_ctx_set_default_verify_paths: unsafe extern "C" fn(*mut SslCtx) -> c_int,
    ssl_ctx_load_verify_locations:
        unsafe extern "C" fn(*mut SslCtx, *const c_char, *const c_char) -> c_int,
    ssl_ctx_use_certificate_chain_file: unsafe extern "C" fn(*mut SslCtx, *const c_char) -> c_int,
    ssl_ctx_use_private_key_file: unsafe extern "C" fn(*mut SslCtx, *const c_char, c_int) -> c_int,
    ssl_ctx_check_private_key: unsafe extern "C" fn(*const SslCtx) -> c_int,
    ssl_ctx_set_default_passwd_cb: unsafe extern "C" fn(*mut SslCtx, PasswdCb),
    ssl_new: unsafe extern "C" fn(*mut SslCtx) -> *mut Ssl,
    ssl_free: unsafe extern "C" fn(*mut Ssl),
    ssl_set_fd: unsafe extern "C" fn(*mut Ssl, c_int) -> c_int,
    ssl_ctrl: unsafe extern "C" fn(*mut Ssl, c_int, c_long, *mut c_void) -> c_long,
    ssl_set1_host: unsafe extern "C" fn(*mut Ssl, *const c_char) -> c_int,
    ssl_get0_param: unsafe extern "C" fn(*mut Ssl) -> *mut c_void,
    x509_verify_param_set1_ip_asc: unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int,
    ssl_connect: unsafe extern "C" fn(*mut Ssl) -> c_int,
    ssl_accept: unsafe extern "C" fn(*mut Ssl) -> c_int,
    ssl_read: unsafe extern "C" fn(*mut Ssl, *mut c_void, c_int) -> c_int,
    ssl_write: unsafe extern "C" fn(*mut Ssl, *const c_void, c_int) -> c_int,
    ssl_shutdown: unsafe extern "C" fn(*mut Ssl) -> c_int,
    ssl_get_error: unsafe extern "C" fn(*const Ssl, c_int) -> c_int,
    ssl_get_verify_result: unsafe extern "C" fn(*const Ssl) -> c_long,
    x509_verify_cert_error_string: unsafe extern "C" fn(c_long) -> *const c_char,
    err_get_error: unsafe extern "C" fn() -> c_ulong,
    err_error_string_n: unsafe extern "C" fn(c_ulong, *mut c_char, usize),
    err_clear_error: unsafe extern "C" fn(),
}

// Constants of the OpenSSL 3 headers (unchanged from 3.0 to 3.5).
const SSL_CTRL_MODE: c_int = 33;
const SSL_CTRL_SET_TLSEXT_HOSTNAME: c_int = 55;
const SSL_CTRL_SET_MIN_PROTO_VERSION: c_int = 123;
const TLSEXT_NAMETYPE_HOST_NAME: c_long = 0;
const TLS1_2_VERSION: c_long = 0x0303;
const SSL_MODE_AUTO_RETRY: c_long = 0x4;
const SSL_OP_IGNORE_UNEXPECTED_EOF: u64 = 1 << 7;
const SSL_VERIFY_PEER: c_int = 1;
const SSL_FILETYPE_PEM: c_int = 1;
const SSL_ERROR_WANT_READ: c_int = 2;
const SSL_ERROR_WANT_WRITE: c_int = 3;
const SSL_ERROR_SYSCALL: c_int = 5;
const SSL_ERROR_ZERO_RETURN: c_int = 6;
const X509_V_OK: c_long = 0;
const OPENSSL_VERSION_TEXT: c_int = 0;

mod dl {
    use std::ffi::{c_char, c_int, c_void};
    extern "C" {
        pub fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    pub const RTLD_NOW: c_int = 2;
    /// The library's symbols stay out of the process's global namespace.
    pub const RTLD_LOCAL: c_int = 0;
}

/// Looks one function up. `T` must be the function-pointer type of the C function `name`.
unsafe fn lookup<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, String> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    let c = CString::new(name).map_err(|_| format!("bad symbol name {name}"))?;
    // SAFETY: `handle` is a live `dlopen` handle and `c` a NUL-terminated name.
    let p = unsafe { dl::dlsym(handle, c.as_ptr()) };
    if p.is_null() {
        return Err(format!("it has no function {name}"));
    }
    // SAFETY: a non-null symbol address, read as the function pointer the caller named.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
}

impl Api {
    /// Opens the first of `names` that loads and looks every function up.
    fn load(names: &[&str]) -> Result<Api, String> {
        let mut handle = std::ptr::null_mut();
        let mut opened = "";
        for name in names {
            let Ok(c) = CString::new(*name) else { continue };
            // SAFETY: a NUL-terminated library name; the flags are valid.
            handle = unsafe { dl::dlopen(c.as_ptr(), dl::RTLD_NOW | dl::RTLD_LOCAL) };
            if !handle.is_null() {
                opened = name;
                break;
            }
        }
        if handle.is_null() {
            return Err(format!(
                "the system's OpenSSL library was not found (tried {}); install OpenSSL 3 (libssl3) to use HTTPS",
                names.join(", ")
            ));
        }
        let api = (|| {
            // SAFETY: each name is looked up as the type its C declaration has in OpenSSL 3.
            unsafe {
                Ok::<Api, String>(Api {
                    openssl_init_ssl: lookup(handle, "OPENSSL_init_ssl")?,
                    openssl_version: lookup(handle, "OpenSSL_version")?,
                    tls_client_method: lookup(handle, "TLS_client_method")?,
                    tls_server_method: lookup(handle, "TLS_server_method")?,
                    ssl_ctx_new: lookup(handle, "SSL_CTX_new")?,
                    ssl_ctx_free: lookup(handle, "SSL_CTX_free")?,
                    ssl_ctx_ctrl: lookup(handle, "SSL_CTX_ctrl")?,
                    ssl_ctx_set_options: lookup(handle, "SSL_CTX_set_options")?,
                    ssl_ctx_set_verify: lookup(handle, "SSL_CTX_set_verify")?,
                    ssl_ctx_set_default_verify_paths: lookup(
                        handle,
                        "SSL_CTX_set_default_verify_paths",
                    )?,
                    ssl_ctx_load_verify_locations: lookup(handle, "SSL_CTX_load_verify_locations")?,
                    ssl_ctx_use_certificate_chain_file: lookup(
                        handle,
                        "SSL_CTX_use_certificate_chain_file",
                    )?,
                    ssl_ctx_use_private_key_file: lookup(handle, "SSL_CTX_use_PrivateKey_file")?,
                    ssl_ctx_check_private_key: lookup(handle, "SSL_CTX_check_private_key")?,
                    ssl_ctx_set_default_passwd_cb: lookup(handle, "SSL_CTX_set_default_passwd_cb")?,
                    ssl_new: lookup(handle, "SSL_new")?,
                    ssl_free: lookup(handle, "SSL_free")?,
                    ssl_set_fd: lookup(handle, "SSL_set_fd")?,
                    ssl_ctrl: lookup(handle, "SSL_ctrl")?,
                    ssl_set1_host: lookup(handle, "SSL_set1_host")?,
                    ssl_get0_param: lookup(handle, "SSL_get0_param")?,
                    x509_verify_param_set1_ip_asc: lookup(handle, "X509_VERIFY_PARAM_set1_ip_asc")?,
                    ssl_connect: lookup(handle, "SSL_connect")?,
                    ssl_accept: lookup(handle, "SSL_accept")?,
                    ssl_read: lookup(handle, "SSL_read")?,
                    ssl_write: lookup(handle, "SSL_write")?,
                    ssl_shutdown: lookup(handle, "SSL_shutdown")?,
                    ssl_get_error: lookup(handle, "SSL_get_error")?,
                    ssl_get_verify_result: lookup(handle, "SSL_get_verify_result")?,
                    x509_verify_cert_error_string: lookup(handle, "X509_verify_cert_error_string")?,
                    err_get_error: lookup(handle, "ERR_get_error")?,
                    err_error_string_n: lookup(handle, "ERR_error_string_n")?,
                    err_clear_error: lookup(handle, "ERR_clear_error")?,
                })
            }
        })()
        .map_err(|e| {
            format!("the OpenSSL library {opened} cannot be used: {e} (OpenSSL 3 is needed)")
        })?;
        // SAFETY: the library initializes itself; no settings are passed.
        if unsafe { (api.openssl_init_ssl)(0, std::ptr::null()) } != 1 {
            return Err(format!("the OpenSSL library {opened} failed to initialize"));
        }
        Ok(api)
    }

    /// The oldest error of this thread's OpenSSL error queue as text, and the queue emptied.
    fn error_text(&self) -> String {
        // SAFETY: the buffer is valid for its length and the library NUL-terminates it.
        unsafe {
            let code = (self.err_get_error)();
            (self.err_clear_error)();
            if code == 0 {
                return "no further detail".into();
            }
            let mut buf = [0 as c_char; 256];
            (self.err_error_string_n)(code, buf.as_mut_ptr(), buf.len());
            CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
        }
    }

    fn version(&self) -> String {
        // SAFETY: the library returns a static NUL-terminated string.
        unsafe {
            let p = (self.openssl_version)(OPENSSL_VERSION_TEXT);
            if p.is_null() {
                return "OpenSSL".into();
            }
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

static LIBRARY: OnceLock<Result<Api, String>> = OnceLock::new();

/// The loaded library, or the message that says why there is none. Loaded once per process.
fn library() -> Result<&'static Api, String> {
    LIBRARY
        .get_or_init(|| Api::load(LIBRARY_NAMES))
        .as_ref()
        .map_err(String::clone)
}

/// Whether TLS can be used on this machine: the library's version text, or what is missing.
pub fn availability() -> Result<String, String> {
    library().map(Api::version)
}

/// What a machine without the library reports (the load of a name that does not exist).
#[cfg(test)]
pub(crate) fn missing_library_message() -> String {
    Api::load(&["libssl-not-installed.so.0"])
        .err()
        .expect("a library that does not exist loads")
}

/// An owned `SSL_CTX`.
struct Ctx {
    api: &'static Api,
    ptr: *mut SslCtx,
}

// SAFETY: an `SSL_CTX` may be shared between threads once configured (the library locks it), and
// it is only configured in `new` before it is shared.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

impl Ctx {
    fn new(api: &'static Api, server: bool) -> Result<Ctx, String> {
        // SAFETY: the method pointers are static library objects; the result is checked.
        let ptr = unsafe {
            let method = if server {
                (api.tls_server_method)()
            } else {
                (api.tls_client_method)()
            };
            (api.ssl_ctx_new)(method)
        };
        if ptr.is_null() {
            return Err(format!("OpenSSL context: {}", api.error_text()));
        }
        let ctx = Ctx { api, ptr };
        // SAFETY: `ptr` is a live context.
        unsafe {
            // TLS 1.2 at least; a blocking read or write finishes or fails, never asks to retry
            // for a renegotiation
            (api.ssl_ctx_ctrl)(
                ptr,
                SSL_CTRL_SET_MIN_PROTO_VERSION,
                TLS1_2_VERSION,
                std::ptr::null_mut(),
            );
            (api.ssl_ctx_ctrl)(
                ptr,
                SSL_CTRL_MODE,
                SSL_MODE_AUTO_RETRY,
                std::ptr::null_mut(),
            );
        }
        Ok(ctx)
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        // SAFETY: the context is owned and freed once.
        unsafe { (self.api.ssl_ctx_free)(self.ptr) }
    }
}

fn c_path(path: &Path) -> Result<CString, String> {
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| format!("{}: the path holds a NUL byte", path.display()))
}

/// An encrypted key file would make the library ask for its passphrase on the terminal; the
/// server has none, so the question is answered with "no passphrase" and the load fails.
extern "C" fn gaudium_tls_no_passphrase(
    _buf: *mut c_char,
    _size: c_int,
    _rwflag: c_int,
    _userdata: *mut c_void,
) -> c_int {
    -1
}

/// The server side: a certificate chain and its private key.
pub struct TlsServer {
    ctx: Ctx,
}

impl TlsServer {
    /// Reads the PEM certificate chain and the PEM private key (the library reads both files
    /// itself) and checks that they belong together. The error names the file and the library's
    /// reason; it never holds file contents.
    pub fn new(cert: &Path, key: &Path) -> Result<TlsServer, String> {
        let api = library()?;
        let ctx = Ctx::new(api, true)?;
        let c_cert = c_path(cert)?;
        let c_key = c_path(key)?;
        // SAFETY: `ctx.ptr` is live and the paths are NUL-terminated.
        unsafe {
            (api.err_clear_error)();
            (api.ssl_ctx_set_default_passwd_cb)(ctx.ptr, gaudium_tls_no_passphrase);
            if (api.ssl_ctx_use_certificate_chain_file)(ctx.ptr, c_cert.as_ptr()) != 1 {
                return Err(format!(
                    "certificate file {}: {}",
                    cert.display(),
                    api.error_text()
                ));
            }
            if (api.ssl_ctx_use_private_key_file)(ctx.ptr, c_key.as_ptr(), SSL_FILETYPE_PEM) != 1 {
                return Err(format!(
                    "private key file {}: {} (it must be an unencrypted PEM key that belongs to the certificate)",
                    key.display(),
                    api.error_text()
                ));
            }
            if (api.ssl_ctx_check_private_key)(ctx.ptr) != 1 {
                (api.err_clear_error)();
                return Err(format!(
                    "the private key in {} does not belong to the certificate in {}",
                    key.display(),
                    cert.display()
                ));
            }
        }
        Ok(TlsServer { ctx })
    }

    /// The server side of the handshake on an accepted socket. The socket's read and write
    /// timeouts bound it.
    pub fn accept(&self, tcp: TcpStream) -> io::Result<TlsStream> {
        let mut s = TlsStream::new(&self.ctx, tcp)?;
        // SAFETY: `s.ssl` is live and bound to the socket.
        let ret = unsafe {
            (s.api.err_clear_error)();
            (s.api.ssl_accept)(s.ssl)
        };
        if ret != 1 {
            let e = s.failure(ret, "TLS handshake");
            s.broken = true;
            return Err(e);
        }
        Ok(s)
    }
}

/// The client side: the trust store that server certificates are verified against.
pub struct TlsClient {
    ctx: Ctx,
}

impl TlsClient {
    /// A client that trusts the system's certificate authorities and, when `extra_ca` is given,
    /// those in that PEM file as well. Every connection verifies the server's certificate.
    pub fn new(extra_ca: Option<&Path>) -> Result<TlsClient, String> {
        let api = library()?;
        let ctx = Ctx::new(api, false)?;
        // SAFETY: `ctx.ptr` is live; the path is NUL-terminated.
        unsafe {
            (api.err_clear_error)();
            if (api.ssl_ctx_set_default_verify_paths)(ctx.ptr) != 1 {
                return Err(format!("the system trust store: {}", api.error_text()));
            }
            if let Some(ca) = extra_ca {
                let c_ca = c_path(ca)?;
                if (api.ssl_ctx_load_verify_locations)(ctx.ptr, c_ca.as_ptr(), std::ptr::null())
                    != 1
                {
                    return Err(format!(
                        "certificate authority file {}: {}",
                        ca.display(),
                        api.error_text()
                    ));
                }
            }
            (api.ssl_ctx_set_verify)(ctx.ptr, SSL_VERIFY_PEER, std::ptr::null());
            // a server that closes without a TLS goodbye ends the stream like a closed socket
            (api.ssl_ctx_set_options)(ctx.ptr, SSL_OP_IGNORE_UNEXPECTED_EOF);
        }
        Ok(TlsClient { ctx })
    }

    /// The client side of the handshake on a socket the caller connected. `host` is the name (or
    /// address literal) of the URL: it is sent as the server name and the certificate must be
    /// valid for it, whatever address the socket went to.
    pub fn connect(&self, tcp: TcpStream, host: &str) -> io::Result<TlsStream> {
        let mut s = TlsStream::new(&self.ctx, tcp)?;
        let c_host = CString::new(host)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "host holds a NUL byte"))?;
        let api = s.api;
        // SAFETY: `s.ssl` is live; `c_host` outlives the calls (the library copies it).
        let ret = unsafe {
            (api.err_clear_error)();
            let named = if host.parse::<IpAddr>().is_ok() {
                // an address: checked against the certificate's address entries, not sent as a name
                (api.x509_verify_param_set1_ip_asc)((api.ssl_get0_param)(s.ssl), c_host.as_ptr())
                    == 1
            } else {
                (api.ssl_ctrl)(
                    s.ssl,
                    SSL_CTRL_SET_TLSEXT_HOSTNAME,
                    TLSEXT_NAMETYPE_HOST_NAME,
                    c_host.as_ptr() as *mut c_void,
                ) == 1
                    && (api.ssl_set1_host)(s.ssl, c_host.as_ptr()) == 1
            };
            if !named {
                s.broken = true;
                return Err(io::Error::other(format!(
                    "TLS: the host name {host} was not accepted: {}",
                    api.error_text()
                )));
            }
            (api.ssl_connect)(s.ssl)
        };
        if ret != 1 {
            // SAFETY: `s.ssl` is live.
            let verify = unsafe { (api.ssl_get_verify_result)(s.ssl) };
            let e = if verify != X509_V_OK {
                // SAFETY: the library returns a static NUL-terminated string.
                let why = unsafe {
                    CStr::from_ptr((api.x509_verify_cert_error_string)(verify))
                        .to_string_lossy()
                        .into_owned()
                };
                (unsafe { (api.err_clear_error)() });
                io::Error::other(format!("the certificate of {host} was not accepted: {why}"))
            } else {
                s.failure(ret, "TLS handshake")
            };
            s.broken = true;
            return Err(e);
        }
        Ok(s)
    }
}

/// One TLS connection over a connected socket.
pub struct TlsStream {
    api: &'static Api,
    ssl: *mut Ssl,
    tcp: TcpStream,
    /// A fatal error happened: no goodbye is sent when the stream is dropped.
    broken: bool,
}

// SAFETY: an `SSL` object may move between threads; it is used by one at a time (`&mut self`).
unsafe impl Send for TlsStream {}

impl TlsStream {
    fn new(ctx: &Ctx, tcp: TcpStream) -> io::Result<TlsStream> {
        let api = ctx.api;
        // SAFETY: the context is live; the socket outlives the `SSL` (both owned by the stream).
        unsafe {
            (api.err_clear_error)();
            let ssl = (api.ssl_new)(ctx.ptr);
            if ssl.is_null() {
                return Err(io::Error::other(format!("TLS: {}", api.error_text())));
            }
            let s = TlsStream {
                api,
                ssl,
                tcp,
                broken: false,
            };
            if (api.ssl_set_fd)(ssl, s.tcp.as_raw_fd()) != 1 {
                return Err(io::Error::other(format!("TLS: {}", api.error_text())));
            }
            Ok(s)
        }
    }

    /// The error behind a failed call that returned `ret`.
    fn failure(&mut self, ret: c_int, what: &str) -> io::Error {
        // the operating system's error first: later calls may overwrite it
        let os = io::Error::last_os_error();
        // SAFETY: `self.ssl` is live.
        let code = unsafe { (self.api.ssl_get_error)(self.ssl, ret) };
        match code {
            SSL_ERROR_WANT_READ | SSL_ERROR_WANT_WRITE => {
                io::Error::new(io::ErrorKind::WouldBlock, format!("{what}: timed out"))
            }
            SSL_ERROR_SYSCALL => {
                self.broken = true;
                // SAFETY: clears this thread's error queue.
                unsafe { (self.api.err_clear_error)() };
                if os.raw_os_error().unwrap_or(0) != 0 {
                    os
                } else {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("{what}: connection closed"),
                    )
                }
            }
            _ => {
                self.broken = true;
                io::Error::other(format!("{what}: {}", self.api.error_text()))
            }
        }
    }

    /// The read timeout of the socket underneath (`None`: a read waits as long as it takes).
    pub fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.tcp.set_read_timeout(t)
    }

    pub fn set_write_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        self.tcp.set_write_timeout(t)
    }

    /// Ends the connection at once: the socket is shut down and no TLS goodbye is sent when the
    /// stream is dropped (a peer that stopped reading could hold that write).
    pub fn abandon(&mut self) {
        self.broken = true;
        let _ = self.tcp.shutdown(std::net::Shutdown::Both);
    }
}

impl Read for TlsStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let len = buf.len().min(c_int::MAX as usize) as c_int;
        // SAFETY: `buf` is valid for `len` bytes; `self.ssl` is live.
        let ret = unsafe {
            (self.api.err_clear_error)();
            (self.api.ssl_read)(self.ssl, buf.as_mut_ptr() as *mut c_void, len)
        };
        if ret > 0 {
            return Ok(ret as usize);
        }
        // SAFETY: `self.ssl` is live.
        if unsafe { (self.api.ssl_get_error)(self.ssl, ret) } == SSL_ERROR_ZERO_RETURN {
            return Ok(0);
        }
        Err(self.failure(ret, "TLS read"))
    }
}

impl Write for TlsStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let len = buf.len().min(c_int::MAX as usize) as c_int;
        // SAFETY: `buf` is valid for `len` bytes; `self.ssl` is live.
        let ret = unsafe {
            (self.api.err_clear_error)();
            (self.api.ssl_write)(self.ssl, buf.as_ptr() as *const c_void, len)
        };
        if ret > 0 {
            return Ok(ret as usize);
        }
        Err(self.failure(ret, "TLS write"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for TlsStream {
    fn drop(&mut self) {
        // SAFETY: the `SSL` is owned and freed once; the socket is still open here.
        unsafe {
            if !self.broken {
                // the TLS goodbye, so the peer can tell the end of the stream from a cut
                (self.api.ssl_shutdown)(self.ssl);
            }
            (self.api.err_clear_error)();
            (self.api.ssl_free)(self.ssl);
        }
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    //! A throwaway certificate for tests, made with the `openssl` command at run time (no key or
    //! certificate is kept in the repository).

    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A self-signed certificate and its key in a directory that is removed with this value.
    pub struct TestCert {
        dir: PathBuf,
        pub cert: PathBuf,
        pub key: PathBuf,
    }

    impl Drop for TestCert {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// A certificate valid for the DNS names and addresses in `names` (`DNS:localhost`,
    /// `IP:127.0.0.1`, ...). `None` when the test cannot run here (no TLS library or no
    /// `openssl` command), with the reason on stderr.
    pub fn test_cert(names: &[&str]) -> Option<TestCert> {
        if let Err(e) = super::availability() {
            eprintln!("skipped: {e}");
            return None;
        }
        let dir = std::env::temp_dir().join(format!(
            "gaudium-tls-test-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).ok()?;
        let cert = dir.join("cert.pem");
        let key = dir.join("key.pem");
        let made = Command::new("openssl")
            .args(["req", "-x509", "-newkey", "ec", "-pkeyopt"])
            .arg("ec_paramgen_curve:prime256v1")
            .args([
                "-nodes",
                "-days",
                "2",
                "-subj",
                "/CN=gaudium test",
                "-keyout",
            ])
            .arg(&key)
            .arg("-out")
            .arg(&cert)
            .arg("-addext")
            .arg(format!("subjectAltName={}", names.join(",")))
            .output();
        match made {
            Ok(out) if out.status.success() => Some(TestCert { dir, cert, key }),
            Ok(out) => {
                let _ = std::fs::remove_dir_all(&dir);
                eprintln!(
                    "skipped: the openssl command could not make a test certificate: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                None
            }
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dir);
                eprintln!("skipped: the openssl command is not available ({e})");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::test_cert;
    use super::*;
    use std::net::TcpListener;

    /// A TLS server on a loopback port that answers one connection: reads up to a newline and
    /// writes it back in upper case.
    fn echo_server(server: TlsServer) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = std::thread::spawn(move || {
            let (tcp, _) = l.accept().unwrap();
            let Ok(mut s) = server.accept(tcp) else {
                return;
            };
            let mut got = Vec::new();
            let mut b = [0u8; 64];
            while !got.contains(&b'\n') {
                match s.read(&mut b) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => got.extend_from_slice(&b[..n]),
                }
            }
            let _ = s.write_all(&got.to_ascii_uppercase());
        });
        (addr, t)
    }

    #[test]
    fn a_missing_library_is_reported_with_what_to_install() {
        let msg = missing_library_message();
        assert!(msg.contains("OpenSSL library was not found"), "{msg}");
        assert!(msg.contains("libssl-not-installed.so.0"), "{msg}");
    }

    #[test]
    fn client_and_server_talk_and_the_name_is_checked() {
        let Some(tc) = test_cert(&["DNS:mcp.example.com", "IP:127.0.0.1"]) else {
            return;
        };
        // by name: the socket goes to 127.0.0.1, the certificate is checked for the name
        let client = TlsClient::new(Some(&tc.cert)).unwrap();
        let (addr, t) = echo_server(TlsServer::new(&tc.cert, &tc.key).unwrap());
        let mut s = client
            .connect(TcpStream::connect(addr).unwrap(), "mcp.example.com")
            .unwrap();
        s.write_all(b"hello\n").unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        assert_eq!(out, "HELLO\n");
        t.join().unwrap();

        // by address: the certificate's address entry is checked
        let (addr, t) = echo_server(TlsServer::new(&tc.cert, &tc.key).unwrap());
        let mut s = client
            .connect(TcpStream::connect(addr).unwrap(), "127.0.0.1")
            .unwrap();
        s.write_all(b"ip\n").unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        assert_eq!(out, "IP\n");
        t.join().unwrap();

        // another name: refused, and the reason says so
        let (addr, t) = echo_server(TlsServer::new(&tc.cert, &tc.key).unwrap());
        let err = client
            .connect(TcpStream::connect(addr).unwrap(), "other.example.com")
            .err()
            .expect("a certificate for another name is refused");
        assert!(err.to_string().contains("was not accepted"), "{err}");
        t.join().unwrap();

        // a client that does not know the certificate's issuer: refused
        let strict = TlsClient::new(None).unwrap();
        let (addr, t) = echo_server(TlsServer::new(&tc.cert, &tc.key).unwrap());
        let err = strict
            .connect(TcpStream::connect(addr).unwrap(), "mcp.example.com")
            .err()
            .expect("an unknown issuer is refused");
        assert!(err.to_string().contains("was not accepted"), "{err}");
        t.join().unwrap();
    }

    #[test]
    fn server_files_are_checked_and_errors_hold_no_key_material() {
        let Some(a) = test_cert(&["DNS:localhost"]) else {
            return;
        };
        let Some(b) = test_cert(&["DNS:localhost"]) else {
            return;
        };
        let key_text = std::fs::read_to_string(&a.key).unwrap();
        let key_line = key_text.lines().nth(1).unwrap();

        // a key that belongs to another certificate
        let e = TlsServer::new(&a.cert, &b.key).err().unwrap();
        assert!(e.contains("belong"), "{e}");
        assert!(!e.contains(key_line), "{e}");
        // the two files swapped
        let e = TlsServer::new(&a.key, &a.cert).err().unwrap();
        assert!(e.contains("certificate file"), "{e}");
        assert!(!e.contains(key_line), "{e}");
        // a file that is not there
        let e = TlsServer::new(&a.cert, Path::new("/nonexistent/key.pem"))
            .err()
            .unwrap();
        assert!(e.contains("private key file /nonexistent/key.pem"), "{e}");
    }

    #[test]
    fn a_read_past_the_socket_timeout_is_would_block() {
        let Some(tc) = test_cert(&["DNS:localhost"]) else {
            return;
        };
        let server = TlsServer::new(&tc.cert, &tc.key).unwrap();
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = std::thread::spawn(move || {
            let (tcp, _) = l.accept().unwrap();
            let mut s = server.accept(tcp).unwrap();
            // say nothing until the client is done
            let mut b = [0u8; 8];
            let _ = s.read(&mut b);
        });
        let client = TlsClient::new(Some(&tc.cert)).unwrap();
        let mut s = client
            .connect(TcpStream::connect(addr).unwrap(), "localhost")
            .unwrap();
        s.set_read_timeout(Some(Duration::from_millis(60))).unwrap();
        let mut b = [0u8; 8];
        let e = s.read(&mut b).err().unwrap();
        assert_eq!(e.kind(), io::ErrorKind::WouldBlock, "{e}");
        // the stream is still usable after a timed-out read
        s.write_all(b"x").unwrap();
        drop(s);
        t.join().unwrap();
    }
}

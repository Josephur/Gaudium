//! The certificate of the server's HTTPS listener: `gaudium --make-cert <dir>`.
//!
//! A run writes four files into one directory:
//! - `ca.crt`, `ca.key`: a small local certificate authority. Its certificate is what a person
//!   installs once on each device that opens the Web UI; its key signs the server certificate.
//! - `server.crt`, `server.key`: the server's certificate (followed by the authority's, so the
//!   file is the chain `--tls-cert` takes) and its private key, for `--tls-key`.
//!
//! The authority is made on the first run and reused on every later one, so a new address, a new
//! name or a renewal needs a new server certificate only, and nothing on the devices. The
//! authority is limited by name constraints to the names given when it was made and to the
//! loopback and local-network address ranges: its key cannot make a certificate that the devices
//! trust for any other site.
//!
//! Which files a run writes, and how long the server certificate is valid, are decided by the
//! verified core (`verified/tls_cert.bend`, through [`crate::core::cert`]). Keys are EC P-256,
//! signatures SHA-256. Everything is made by the system's OpenSSL 3 (`libcrypto.so.3`), loaded at
//! run time like the TLS code of [`crate::tls`]; the binary links no TLS library. The library
//! writes the key files itself, into files this code created with mode 0600: no key material
//! passes through this code, and none is printed or logged.

#![allow(unsafe_code)]

use std::ffi::{c_char, c_int, c_long, c_uint, c_void, CStr, CString};
use std::fs::{self, File, OpenOptions};
use std::io;
use std::net::IpAddr;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::core::cert::{self as decide, CertFiles, CertPlan, Validity};
use crate::tls::{dl, gaudium_tls_no_passphrase, lookup, PasswdCb};

/// The authority's certificate (installed on the devices) and key.
pub const CA_CERT: &str = "ca.crt";
pub const CA_KEY: &str = "ca.key";
/// The server's certificate chain (`--tls-cert`) and key (`--tls-key`).
pub const SERVER_CERT: &str = "server.crt";
pub const SERVER_KEY: &str = "server.key";

/// Days the server certificate is valid by default, and at most: Apple's platforms refuse a TLS
/// server certificate valid for longer than 825 days, also one from an authority the person
/// installed themselves.
pub const DEFAULT_SERVER_DAYS: u64 = 825;
pub const MAX_SERVER_DAYS: u64 = 825;
/// Days a new authority is valid (no browser limits an authority the person installed).
pub const CA_DAYS: u64 = 3650;
/// Both certificates start this long before they are made, for a device whose clock is a little
/// behind; the end moves by the same amount, so the length of the validity stays the days asked.
const CLOCK_SKEW_SECS: c_long = 300;

/// Address ranges a new authority may make certificates for besides the addresses named: the
/// loopback, private, shared (carrier-grade NAT, also used by some overlay networks), link-local
/// and unique-local blocks. A device on the local network may change its address inside them
/// without a new authority.
const LOCAL_RANGES: &[&str] = &[
    "127.0.0.0/255.0.0.0",
    "10.0.0.0/255.0.0.0",
    "172.16.0.0/255.240.0.0",
    "192.168.0.0/255.255.0.0",
    "100.64.0.0/255.192.0.0",
    "169.254.0.0/255.255.0.0",
    "::1/ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
    "fc00::/fe00::",
    "fe80::/ffc0::",
];

/// Top-level labels of names that only a local network resolves. A name with another one may be
/// public (said in a note: a public name is better served by a public authority).
const LOCAL_SUFFIXES: &[&str] = &[
    "local",
    "lan",
    "home",
    "internal",
    "intranet",
    "corp",
    "private",
    "localdomain",
    "localhost",
    "test",
    "example",
    "invalid",
    "arpa",
];

// ------------------------------------------------------------------------------------------------
// Names
// ------------------------------------------------------------------------------------------------

/// One name the server certificate is valid for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertName {
    Dns(String),
    Ip(IpAddr),
}

impl std::fmt::Display for CertName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CertName::Dns(n) => f.write_str(n),
            CertName::Ip(a) => write!(f, "{a}"),
        }
    }
}

impl CertName {
    /// The form of a subject-alternative-name entry in OpenSSL's configuration syntax.
    fn san(&self) -> String {
        match self {
            CertName::Dns(n) => format!("DNS:{n}"),
            CertName::Ip(a) => format!("IP:{a}"),
        }
    }

    /// The name may be one that the internet resolves: worth a note, never a refusal (many
    /// networks use names of a registered domain inside).
    pub fn looks_public(&self) -> bool {
        match self {
            CertName::Dns(n) => match n.rsplit_once('.') {
                None => false,
                Some((_, tld)) => !LOCAL_SUFFIXES.contains(&tld),
            },
            CertName::Ip(IpAddr::V4(a)) => {
                !(a.is_loopback()
                    || a.is_private()
                    || a.is_link_local()
                    || (a.octets()[0] == 100 && (a.octets()[1] & 0xc0) == 64))
            }
            CertName::Ip(IpAddr::V6(a)) => {
                let s = a.segments()[0];
                !(a.is_loopback() || (s & 0xfe00) == 0xfc00 || (s & 0xffc0) == 0xfe80)
            }
        }
    }
}

/// One entry of `--cert-names`: an address (IPv6 with or without brackets) or a DNS name.
pub fn parse_name(raw: &str) -> Result<CertName, String> {
    let t = raw.trim();
    let bare = t
        .strip_prefix('[')
        .and_then(|r| r.strip_suffix(']'))
        .unwrap_or(t);
    if let Ok(a) = bare.parse::<IpAddr>() {
        if a.is_unspecified() || a.is_multicast() {
            return Err(format!(
                "'{t}': an unspecified or multicast address names no server"
            ));
        }
        return Ok(CertName::Ip(a));
    }
    let n = t.strip_suffix('.').unwrap_or(t).to_ascii_lowercase();
    if n.contains('*') {
        return Err(format!(
            "'{t}': wildcard names are not offered; list each name the server is opened by"
        ));
    }
    if n.is_empty() || n.len() > 253 {
        return Err(format!("'{t}' is not a host name"));
    }
    for label in n.split('.') {
        let ok = !label.is_empty()
            && label.len() <= 63
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
        if !ok {
            return Err(format!(
                "'{t}' is not a host name (letters, digits and '-' in dot-separated labels)"
            ));
        }
    }
    if n.rsplit('.')
        .next()
        .is_some_and(|tld| tld.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err(format!("'{t}' is neither an address nor a host name"));
    }
    Ok(CertName::Dns(n))
}

/// A comma-separated `--cert-names` list.
pub fn parse_names(list: &str) -> Result<Vec<CertName>, String> {
    list.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(parse_name)
        .collect()
}

/// The names given, with `localhost`, `127.0.0.1` and `::1` first, each name once.
pub fn with_local_names(names: &[CertName]) -> Vec<CertName> {
    let mut out = vec![
        CertName::Dns("localhost".into()),
        CertName::Ip(IpAddr::from([127, 0, 0, 1])),
        CertName::Ip(IpAddr::from([0u16, 0, 0, 0, 0, 0, 0, 1])),
    ];
    for n in names {
        if !out.contains(n) {
            out.push(n.clone());
        }
    }
    out
}

/// This machine's host name and the address it reaches other machines from (the one of its
/// default route), for a run without `--cert-names`. Nothing is sent on the network.
pub fn detect_names() -> Vec<CertName> {
    let mut out = Vec::new();
    for f in ["/proc/sys/kernel/hostname", "/etc/hostname"] {
        if let Ok(h) = fs::read_to_string(f) {
            if let Ok(n) = parse_name(h.trim()) {
                out.push(n);
                break;
            }
        }
    }
    // a UDP "connection" only picks the route and the local address; no packet leaves
    if let Ok(s) = std::net::UdpSocket::bind("0.0.0.0:0") {
        if s.connect("192.0.2.1:9").is_ok() {
            if let Ok(a) = s.local_addr() {
                let ip = a.ip();
                if !ip.is_loopback() && !ip.is_unspecified() {
                    out.push(CertName::Ip(ip));
                }
            }
        }
    }
    out
}

/// The name constraints of a new authority: the DNS names it is made for (each with the names
/// under it), the local address ranges, and every other address it is made for.
fn name_constraints(names: &[CertName]) -> String {
    let mut parts = vec!["critical".to_string()];
    for n in names {
        if let CertName::Dns(d) = n {
            parts.push(format!("permitted;DNS:{d}"));
        }
    }
    for r in LOCAL_RANGES {
        parts.push(format!("permitted;IP:{r}"));
    }
    for n in names {
        match n {
            CertName::Ip(IpAddr::V4(a)) if n.looks_public() => {
                parts.push(format!("permitted;IP:{a}/255.255.255.255"))
            }
            CertName::Ip(IpAddr::V6(a)) if n.looks_public() => parts.push(format!(
                "permitted;IP:{a}/ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"
            )),
            _ => {}
        }
    }
    parts.join(",")
}

// ------------------------------------------------------------------------------------------------
// The library
// ------------------------------------------------------------------------------------------------

const LIBRARY_NAMES: &[&str] = &["libcrypto.so.3", "libcrypto.so"];

type P = *mut c_void;
type CP = *const c_void;

/// The functions of `libcrypto` this module calls (all in OpenSSL 3.0 and later).
struct Crypto {
    pkey_ctx_new_from_name: unsafe extern "C" fn(P, *const c_char, *const c_char) -> P,
    pkey_keygen_init: unsafe extern "C" fn(P) -> c_int,
    pkey_ctx_set_group_name: unsafe extern "C" fn(P, *const c_char) -> c_int,
    pkey_generate: unsafe extern "C" fn(P, *mut P) -> c_int,
    pkey_ctx_free: unsafe extern "C" fn(P),
    pkey_free: unsafe extern "C" fn(P),
    x509_new: unsafe extern "C" fn() -> P,
    x509_free: unsafe extern "C" fn(P),
    x509_set_version: unsafe extern "C" fn(P, c_long) -> c_int,
    x509_get_serial_number: unsafe extern "C" fn(P) -> P,
    bn_bin2bn: unsafe extern "C" fn(*const u8, c_int, P) -> P,
    bn_to_asn1_integer: unsafe extern "C" fn(CP, P) -> P,
    bn_free: unsafe extern "C" fn(P),
    rand_bytes: unsafe extern "C" fn(*mut u8, c_int) -> c_int,
    x509_getm_not_before: unsafe extern "C" fn(CP) -> P,
    x509_getm_not_after: unsafe extern "C" fn(CP) -> P,
    x509_get0_not_after: unsafe extern "C" fn(CP) -> CP,
    x509_time_adj_ex: unsafe extern "C" fn(P, c_int, c_long, *mut c_long) -> P,
    x509_get_subject_name: unsafe extern "C" fn(CP) -> P,
    x509_set_issuer_name: unsafe extern "C" fn(P, CP) -> c_int,
    x509_name_add_entry_by_txt:
        unsafe extern "C" fn(P, *const c_char, c_int, *const u8, c_int, c_int, c_int) -> c_int,
    x509_name_oneline: unsafe extern "C" fn(CP, *mut c_char, c_int) -> *mut c_char,
    x509_set_pubkey: unsafe extern "C" fn(P, P) -> c_int,
    x509v3_set_ctx: unsafe extern "C" fn(P, P, P, P, P, c_int),
    x509v3_ext_nconf: unsafe extern "C" fn(P, P, *const c_char, *const c_char) -> P,
    x509_add_ext: unsafe extern "C" fn(P, P, c_int) -> c_int,
    x509_extension_free: unsafe extern "C" fn(P),
    x509_sign: unsafe extern "C" fn(P, P, CP) -> c_int,
    evp_sha256: unsafe extern "C" fn() -> CP,
    x509_digest: unsafe extern "C" fn(CP, CP, *mut u8, *mut c_uint) -> c_int,
    x509_check_private_key: unsafe extern "C" fn(CP, CP) -> c_int,
    x509_check_ca: unsafe extern "C" fn(P) -> c_int,
    x509_check_host:
        unsafe extern "C" fn(P, *const c_char, usize, c_uint, *mut *mut c_char) -> c_int,
    x509_check_ip_asc: unsafe extern "C" fn(P, *const c_char, c_uint) -> c_int,
    asn1_time_diff: unsafe extern "C" fn(*mut c_int, *mut c_int, CP, CP) -> c_int,
    asn1_time_print: unsafe extern "C" fn(P, CP) -> c_int,
    bio_s_mem: unsafe extern "C" fn() -> CP,
    bio_new: unsafe extern "C" fn(CP) -> P,
    bio_new_fd: unsafe extern "C" fn(c_int, c_int) -> P,
    bio_new_file: unsafe extern "C" fn(*const c_char, *const c_char) -> P,
    bio_free: unsafe extern "C" fn(P) -> c_int,
    bio_ctrl: unsafe extern "C" fn(P, c_int, c_long, P) -> c_long,
    pem_write_bio_x509: unsafe extern "C" fn(P, CP) -> c_int,
    pem_write_bio_private_key: unsafe extern "C" fn(P, CP, CP, *const u8, c_int, CP, P) -> c_int,
    pem_read_bio_x509: unsafe extern "C" fn(P, *mut P, CP, P) -> P,
    pem_read_bio_private_key: unsafe extern "C" fn(P, *mut P, Option<PasswdCb>, P) -> P,
    store_new: unsafe extern "C" fn() -> P,
    store_free: unsafe extern "C" fn(P),
    store_add_cert: unsafe extern "C" fn(P, P) -> c_int,
    store_ctx_new: unsafe extern "C" fn() -> P,
    store_ctx_free: unsafe extern "C" fn(P),
    store_ctx_init: unsafe extern "C" fn(P, P, P, P) -> c_int,
    store_ctx_set_purpose: unsafe extern "C" fn(P, c_int) -> c_int,
    store_ctx_get_error: unsafe extern "C" fn(P) -> c_int,
    verify_cert: unsafe extern "C" fn(P) -> c_int,
    verify_cert_error_string: unsafe extern "C" fn(c_long) -> *const c_char,
    err_get_error: unsafe extern "C" fn() -> std::ffi::c_ulong,
    err_error_string_n: unsafe extern "C" fn(std::ffi::c_ulong, *mut c_char, usize),
    err_clear_error: unsafe extern "C" fn(),
}

// Constants of the OpenSSL 3 headers (unchanged from 3.0 to 3.5).
const MBSTRING_UTF8: c_int = 0x1000 | 1;
const BIO_NOCLOSE: c_int = 0;
const BIO_CTRL_INFO: c_int = 3;
const BIO_CTRL_FLUSH: c_int = 11;
const X509_PURPOSE_SSL_SERVER: c_int = 2;
/// `X509_VERSION_3`: the version field holds the version minus one.
const X509_VERSION_3: c_long = 2;
/// Room for an `X509V3_CTX` (eight pointer-sized fields in OpenSSL 3), with a wide margin.
const V3_CTX_WORDS: usize = 32;

impl Crypto {
    fn load(names: &[&str]) -> Result<Crypto, String> {
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
                "the system's OpenSSL library was not found (tried {}); install OpenSSL 3 (libssl3) to make a certificate",
                names.join(", ")
            ));
        }
        let api = (|| {
            // SAFETY: each name is looked up as the type its C declaration has in OpenSSL 3.
            unsafe {
                Ok::<Crypto, String>(Crypto {
                    pkey_ctx_new_from_name: lookup(handle, "EVP_PKEY_CTX_new_from_name")?,
                    pkey_keygen_init: lookup(handle, "EVP_PKEY_keygen_init")?,
                    pkey_ctx_set_group_name: lookup(handle, "EVP_PKEY_CTX_set_group_name")?,
                    pkey_generate: lookup(handle, "EVP_PKEY_generate")?,
                    pkey_ctx_free: lookup(handle, "EVP_PKEY_CTX_free")?,
                    pkey_free: lookup(handle, "EVP_PKEY_free")?,
                    x509_new: lookup(handle, "X509_new")?,
                    x509_free: lookup(handle, "X509_free")?,
                    x509_set_version: lookup(handle, "X509_set_version")?,
                    x509_get_serial_number: lookup(handle, "X509_get_serialNumber")?,
                    bn_bin2bn: lookup(handle, "BN_bin2bn")?,
                    bn_to_asn1_integer: lookup(handle, "BN_to_ASN1_INTEGER")?,
                    bn_free: lookup(handle, "BN_free")?,
                    rand_bytes: lookup(handle, "RAND_bytes")?,
                    x509_getm_not_before: lookup(handle, "X509_getm_notBefore")?,
                    x509_getm_not_after: lookup(handle, "X509_getm_notAfter")?,
                    x509_get0_not_after: lookup(handle, "X509_get0_notAfter")?,
                    x509_time_adj_ex: lookup(handle, "X509_time_adj_ex")?,
                    x509_get_subject_name: lookup(handle, "X509_get_subject_name")?,
                    x509_set_issuer_name: lookup(handle, "X509_set_issuer_name")?,
                    x509_name_add_entry_by_txt: lookup(handle, "X509_NAME_add_entry_by_txt")?,
                    x509_name_oneline: lookup(handle, "X509_NAME_oneline")?,
                    x509_set_pubkey: lookup(handle, "X509_set_pubkey")?,
                    x509v3_set_ctx: lookup(handle, "X509V3_set_ctx")?,
                    x509v3_ext_nconf: lookup(handle, "X509V3_EXT_nconf")?,
                    x509_add_ext: lookup(handle, "X509_add_ext")?,
                    x509_extension_free: lookup(handle, "X509_EXTENSION_free")?,
                    x509_sign: lookup(handle, "X509_sign")?,
                    evp_sha256: lookup(handle, "EVP_sha256")?,
                    x509_digest: lookup(handle, "X509_digest")?,
                    x509_check_private_key: lookup(handle, "X509_check_private_key")?,
                    x509_check_ca: lookup(handle, "X509_check_ca")?,
                    x509_check_host: lookup(handle, "X509_check_host")?,
                    x509_check_ip_asc: lookup(handle, "X509_check_ip_asc")?,
                    asn1_time_diff: lookup(handle, "ASN1_TIME_diff")?,
                    asn1_time_print: lookup(handle, "ASN1_TIME_print")?,
                    bio_s_mem: lookup(handle, "BIO_s_mem")?,
                    bio_new: lookup(handle, "BIO_new")?,
                    bio_new_fd: lookup(handle, "BIO_new_fd")?,
                    bio_new_file: lookup(handle, "BIO_new_file")?,
                    bio_free: lookup(handle, "BIO_free")?,
                    bio_ctrl: lookup(handle, "BIO_ctrl")?,
                    pem_write_bio_x509: lookup(handle, "PEM_write_bio_X509")?,
                    pem_write_bio_private_key: lookup(handle, "PEM_write_bio_PrivateKey")?,
                    pem_read_bio_x509: lookup(handle, "PEM_read_bio_X509")?,
                    pem_read_bio_private_key: lookup(handle, "PEM_read_bio_PrivateKey")?,
                    store_new: lookup(handle, "X509_STORE_new")?,
                    store_free: lookup(handle, "X509_STORE_free")?,
                    store_add_cert: lookup(handle, "X509_STORE_add_cert")?,
                    store_ctx_new: lookup(handle, "X509_STORE_CTX_new")?,
                    store_ctx_free: lookup(handle, "X509_STORE_CTX_free")?,
                    store_ctx_init: lookup(handle, "X509_STORE_CTX_init")?,
                    store_ctx_set_purpose: lookup(handle, "X509_STORE_CTX_set_purpose")?,
                    store_ctx_get_error: lookup(handle, "X509_STORE_CTX_get_error")?,
                    verify_cert: lookup(handle, "X509_verify_cert")?,
                    verify_cert_error_string: lookup(handle, "X509_verify_cert_error_string")?,
                    err_get_error: lookup(handle, "ERR_get_error")?,
                    err_error_string_n: lookup(handle, "ERR_error_string_n")?,
                    err_clear_error: lookup(handle, "ERR_clear_error")?,
                })
            }
        })()
        .map_err(|e| {
            format!("the OpenSSL library {opened} cannot be used: {e} (OpenSSL 3 is needed)")
        })?;
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

    fn fail(&self, what: &str) -> String {
        format!("{what}: {}", self.error_text())
    }
}

static CRYPTO: OnceLock<Result<Crypto, String>> = OnceLock::new();

fn crypto() -> Result<&'static Crypto, String> {
    CRYPTO
        .get_or_init(|| Crypto::load(LIBRARY_NAMES))
        .as_ref()
        .map_err(String::clone)
}

// Owned library objects, each freed once.

struct Pkey(&'static Crypto, P);
impl Drop for Pkey {
    fn drop(&mut self) {
        // SAFETY: an owned EVP_PKEY.
        unsafe { (self.0.pkey_free)(self.1) }
    }
}

struct Cert(&'static Crypto, P);
impl Drop for Cert {
    fn drop(&mut self) {
        // SAFETY: an owned X509.
        unsafe { (self.0.x509_free)(self.1) }
    }
}

struct Bio(&'static Crypto, P);
impl Drop for Bio {
    fn drop(&mut self) {
        // SAFETY: an owned BIO.
        unsafe { (self.0.bio_free)(self.1) };
    }
}

fn cstr(s: &str) -> Result<CString, String> {
    CString::new(s).map_err(|_| format!("'{s}' holds a NUL byte"))
}

impl Crypto {
    /// A new EC P-256 key.
    fn new_key(&'static self) -> Result<Pkey, String> {
        // SAFETY: the context is checked, freed once; the key pointer is written by the library.
        unsafe {
            (self.err_clear_error)();
            let ctx = (self.pkey_ctx_new_from_name)(
                std::ptr::null_mut(),
                c"EC".as_ptr(),
                std::ptr::null(),
            );
            if ctx.is_null() {
                return Err(self.fail("key generation"));
            }
            let mut key: P = std::ptr::null_mut();
            let ok = (self.pkey_keygen_init)(ctx) == 1
                && (self.pkey_ctx_set_group_name)(ctx, c"P-256".as_ptr()) == 1
                && (self.pkey_generate)(ctx, &mut key) == 1;
            (self.pkey_ctx_free)(ctx);
            if !ok || key.is_null() {
                return Err(self.fail("key generation"));
            }
            Ok(Pkey(self, key))
        }
    }

    /// A certificate with a random serial, version 3, `subject`, `pubkey`, valid from now (less
    /// the clock allowance) for `days`.
    fn new_cert(&'static self, subject_cn: &str, key: &Pkey, days: u64) -> Result<Cert, String> {
        let days = c_int::try_from(days).map_err(|_| format!("{days} days is too long"))?;
        let cn = cstr(subject_cn)?;
        // SAFETY: every object is live; the library copies what it is given.
        unsafe {
            (self.err_clear_error)();
            let x = (self.x509_new)();
            if x.is_null() {
                return Err(self.fail("certificate"));
            }
            let cert = Cert(self, x);
            // a positive 16-byte serial: the top bit clear, the next one set
            let mut serial = [0u8; 16];
            if (self.rand_bytes)(serial.as_mut_ptr(), serial.len() as c_int) != 1 {
                return Err(self.fail("random serial number"));
            }
            serial[0] = (serial[0] & 0x7f) | 0x40;
            let bn = (self.bn_bin2bn)(serial.as_ptr(), serial.len() as c_int, std::ptr::null_mut());
            if bn.is_null() {
                return Err(self.fail("serial number"));
            }
            let set_serial =
                !(self.bn_to_asn1_integer)(bn, (self.x509_get_serial_number)(x)).is_null();
            (self.bn_free)(bn);
            let name = (self.x509_get_subject_name)(x);
            let ok = set_serial
                && (self.x509_set_version)(x, X509_VERSION_3) == 1
                && !(self.x509_time_adj_ex)(
                    (self.x509_getm_not_before)(x),
                    0,
                    -CLOCK_SKEW_SECS,
                    std::ptr::null_mut(),
                )
                .is_null()
                && !(self.x509_time_adj_ex)(
                    (self.x509_getm_not_after)(x),
                    days,
                    -CLOCK_SKEW_SECS,
                    std::ptr::null_mut(),
                )
                .is_null()
                && (self.x509_name_add_entry_by_txt)(
                    name,
                    c"O".as_ptr(),
                    MBSTRING_UTF8,
                    c"Gaudium".as_ptr().cast(),
                    -1,
                    -1,
                    0,
                ) == 1
                && (self.x509_name_add_entry_by_txt)(
                    name,
                    c"CN".as_ptr(),
                    MBSTRING_UTF8,
                    cn.as_ptr().cast(),
                    -1,
                    -1,
                    0,
                ) == 1
                && (self.x509_set_pubkey)(x, key.1) == 1;
            if !ok {
                return Err(self.fail("certificate"));
            }
            Ok(cert)
        }
    }

    /// Adds one extension in OpenSSL's configuration syntax (`basicConstraints`,
    /// `critical,CA:TRUE`). `issuer` is the certificate that signs `cert` (itself for the
    /// authority).
    fn add_ext(&self, cert: &Cert, issuer: P, name: &str, value: &str) -> Result<(), String> {
        let c_name = cstr(name)?;
        let c_value = cstr(value)?;
        let mut v3 = [0u64; V3_CTX_WORDS];
        // SAFETY: `v3` is zeroed room for the context, which the library fills; the extension
        // is freed once after the certificate copied it.
        unsafe {
            let ctx = v3.as_mut_ptr().cast::<c_void>();
            (self.x509v3_set_ctx)(
                ctx,
                issuer,
                cert.1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            let ext = (self.x509v3_ext_nconf)(
                std::ptr::null_mut(),
                ctx,
                c_name.as_ptr(),
                c_value.as_ptr(),
            );
            if ext.is_null() {
                return Err(self.fail(&format!("certificate extension {name}")));
            }
            let ok = (self.x509_add_ext)(cert.1, ext, -1) == 1;
            (self.x509_extension_free)(ext);
            if !ok {
                return Err(self.fail(&format!("certificate extension {name}")));
            }
        }
        Ok(())
    }

    /// The certificate's issuer name is `issuer`'s subject name (copied).
    fn set_issuer(&self, cert: &Cert, issuer: &Cert) -> Result<(), String> {
        // SAFETY: both certificates are live; the name is copied.
        if unsafe { (self.x509_set_issuer_name)(cert.1, (self.x509_get_subject_name)(issuer.1)) }
            != 1
        {
            return Err(self.fail("certificate issuer"));
        }
        Ok(())
    }

    fn sign(&self, cert: &Cert, key: &Pkey) -> Result<(), String> {
        // SAFETY: both objects are live; the digest is a static library object.
        if unsafe { (self.x509_sign)(cert.1, key.1, (self.evp_sha256)()) } <= 0 {
            return Err(self.fail("signature"));
        }
        Ok(())
    }

    /// The SHA-256 fingerprint of the certificate's DER form, as browsers show it.
    fn fingerprint(&self, cert: &Cert) -> Result<String, String> {
        let mut md = [0u8; 64];
        let mut len: c_uint = 0;
        // SAFETY: `md` holds the largest digest; the library writes `len` bytes.
        if unsafe { (self.x509_digest)(cert.1, (self.evp_sha256)(), md.as_mut_ptr(), &mut len) }
            != 1
        {
            return Err(self.fail("fingerprint"));
        }
        Ok(md[..len as usize]
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(":"))
    }

    fn subject(&self, cert: &Cert) -> String {
        let mut buf = [0 as c_char; 256];
        // SAFETY: the library writes at most `buf.len()` bytes, NUL-terminated.
        unsafe {
            let name = (self.x509_get_subject_name)(cert.1);
            if (self.x509_name_oneline)(name, buf.as_mut_ptr(), buf.len() as c_int).is_null() {
                return String::new();
            }
            CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
        }
    }

    /// The end of the certificate's validity as text, and the whole days left from now
    /// (negative once it has ended).
    fn not_after(&'static self, cert: &Cert) -> Result<(String, i64), String> {
        // SAFETY: the time belongs to the live certificate; the BIO is owned.
        unsafe {
            let t = (self.x509_get0_not_after)(cert.1);
            let (mut day, mut sec) = (0 as c_int, 0 as c_int);
            if (self.asn1_time_diff)(&mut day, &mut sec, std::ptr::null(), t) != 1 {
                return Err(self.fail("certificate validity"));
            }
            // whole days, rounded down (day and sec have the same sign)
            let days = i64::from(day) - i64::from(sec < 0);
            let bio = Bio(self, (self.bio_new)((self.bio_s_mem)()));
            if bio.1.is_null() || (self.asn1_time_print)(bio.1, t) != 1 {
                return Err(self.fail("certificate validity"));
            }
            let mut data: *mut c_char = std::ptr::null_mut();
            let n = (self.bio_ctrl)(
                bio.1,
                BIO_CTRL_INFO,
                0,
                (&mut data as *mut *mut c_char).cast(),
            );
            let text = if n > 0 && !data.is_null() {
                String::from_utf8_lossy(std::slice::from_raw_parts(data.cast::<u8>(), n as usize))
                    .into_owned()
            } else {
                String::new()
            };
            Ok((text, days))
        }
    }

    /// The first certificate of a PEM file.
    fn read_cert(&'static self, path: &Path) -> Result<Cert, String> {
        let c = crate::tls::c_path(path)?;
        // SAFETY: the BIO is owned; the library reads the file.
        unsafe {
            (self.err_clear_error)();
            let bio = Bio(self, (self.bio_new_file)(c.as_ptr(), c"r".as_ptr()));
            if bio.1.is_null() {
                return Err(self.fail(&path.display().to_string()));
            }
            let x = (self.pem_read_bio_x509)(
                bio.1,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            if x.is_null() {
                return Err(self.fail(&format!("{}: no PEM certificate", path.display())));
            }
            Ok(Cert(self, x))
        }
    }

    /// The private key of a PEM file, read by the library (an encrypted key is refused: there is
    /// no one to ask for its passphrase).
    fn read_key(&'static self, path: &Path) -> Result<Pkey, String> {
        let c = crate::tls::c_path(path)?;
        // SAFETY: the BIO is owned; the library reads the file.
        unsafe {
            (self.err_clear_error)();
            let bio = Bio(self, (self.bio_new_file)(c.as_ptr(), c"r".as_ptr()));
            if bio.1.is_null() {
                return Err(self.fail(&path.display().to_string()));
            }
            let k = (self.pem_read_bio_private_key)(
                bio.1,
                std::ptr::null_mut(),
                Some(gaudium_tls_no_passphrase),
                std::ptr::null_mut(),
            );
            if k.is_null() {
                (self.err_clear_error)();
                return Err(format!(
                    "{}: not an unencrypted PEM private key",
                    path.display()
                ));
            }
            Ok(Pkey(self, k))
        }
    }

    /// Checks the server certificate the way a client does, with `ca` as the only trusted
    /// authority: signature, validity, purpose (TLS server) and the authority's name
    /// constraints. Then checks that every name is in it.
    fn verify(&self, ca: &Cert, leaf: &Cert, names: &[CertName]) -> Result<(), String> {
        // SAFETY: the store and its context are owned and freed once, the context first.
        unsafe {
            (self.err_clear_error)();
            let store = (self.store_new)();
            let ctx = (self.store_ctx_new)();
            let result = if store.is_null() || ctx.is_null() {
                Err(self.fail("certificate check"))
            } else if (self.store_add_cert)(store, ca.1) != 1
                || (self.store_ctx_init)(ctx, store, leaf.1, std::ptr::null_mut()) != 1
                || (self.store_ctx_set_purpose)(ctx, X509_PURPOSE_SSL_SERVER) != 1
            {
                Err(self.fail("certificate check"))
            } else if (self.verify_cert)(ctx) != 1 {
                let code = (self.store_ctx_get_error)(ctx);
                let why = CStr::from_ptr((self.verify_cert_error_string)(c_long::from(code)))
                    .to_string_lossy()
                    .into_owned();
                (self.err_clear_error)();
                Err(why)
            } else {
                Ok(())
            };
            if !ctx.is_null() {
                (self.store_ctx_free)(ctx);
            }
            if !store.is_null() {
                (self.store_free)(store);
            }
            result?;
        }
        for n in names {
            if !self.covers(leaf, n)? {
                return Err(format!("the certificate does not hold the name {n}"));
            }
        }
        Ok(())
    }

    /// The certificate is valid for `name` (a host-name or address check as a TLS client makes).
    fn covers(&self, cert: &Cert, name: &CertName) -> Result<bool, String> {
        let text = cstr(&name.to_string())?;
        // SAFETY: the certificate is live and the name NUL-terminated.
        let r = unsafe {
            match name {
                CertName::Dns(d) => {
                    (self.x509_check_host)(cert.1, text.as_ptr(), d.len(), 0, std::ptr::null_mut())
                }
                CertName::Ip(_) => (self.x509_check_ip_asc)(cert.1, text.as_ptr(), 0),
            }
        };
        Ok(r == 1)
    }
}

// ------------------------------------------------------------------------------------------------
// Files
// ------------------------------------------------------------------------------------------------

/// A file written in this run under a temporary name, put in place by [`Staged::commit`].
struct Staged {
    tmp: PathBuf,
    path: PathBuf,
    replace: bool,
    committed: bool,
    /// The file did not exist before this run (removed again if a later step fails).
    created: bool,
}

impl Drop for Staged {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.tmp);
        }
    }
}

impl Staged {
    /// Puts the file in place: over the old one when replacing, otherwise only if the name is
    /// still free.
    fn commit(&mut self) -> Result<(), String> {
        let r = if self.replace {
            fs::rename(&self.tmp, &self.path)
        } else {
            match fs::hard_link(&self.tmp, &self.path) {
                Ok(()) => fs::remove_file(&self.tmp),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Err(e),
                // a file system without hard links: the name was free a moment ago
                Err(_) if !exists(&self.path) => fs::rename(&self.tmp, &self.path),
                Err(e) => Err(e),
            }
        };
        r.map_err(|e| format!("{}: {e}", self.path.display()))?;
        self.committed = true;
        self.created = !self.replace;
        Ok(())
    }
}

/// A name in the directory is taken, also by a dangling link.
fn exists(p: &Path) -> bool {
    fs::symlink_metadata(p).is_ok()
}

/// Writes `fill`'s output into a new temporary file beside `path`, with `mode`.
fn stage(
    c: &'static Crypto,
    path: &Path,
    mode: u32,
    replace: bool,
    fill: impl FnOnce(P) -> bool,
) -> Result<Staged, String> {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));
    let file: File = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .open(&tmp)
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    let staged = Staged {
        tmp,
        path: path.to_path_buf(),
        replace,
        committed: false,
        created: false,
    };
    // exactly `mode`, whatever the umask left of it
    fs::set_permissions(&staged.tmp, fs::Permissions::from_mode(mode))
        .map_err(|e| format!("{}: {e}", staged.tmp.display()))?;
    // SAFETY: the BIO writes to the open file descriptor and does not close it.
    let ok = unsafe {
        (c.err_clear_error)();
        let bio = Bio(c, (c.bio_new_fd)(file.as_raw_fd(), BIO_NOCLOSE));
        !bio.1.is_null()
            && fill(bio.1)
            && (c.bio_ctrl)(bio.1, BIO_CTRL_FLUSH, 0, std::ptr::null_mut()) == 1
    };
    if !ok {
        return Err(c.fail(&path.display().to_string()));
    }
    file.sync_all()
        .map_err(|e| format!("{}: {e}", staged.tmp.display()))?;
    Ok(staged)
}

// ------------------------------------------------------------------------------------------------
// The command
// ------------------------------------------------------------------------------------------------

/// What `gaudium --make-cert` was asked for.
#[derive(Debug, Clone)]
pub struct MakeCertOptions {
    pub dir: PathBuf,
    /// The names and addresses the server is opened by (`localhost` and the loopback addresses
    /// are added).
    pub names: Vec<CertName>,
    /// Days the server certificate is valid.
    pub days: u64,
    /// Replace an existing server certificate and key (never the authority).
    pub overwrite: bool,
    /// Days a new authority is valid ([`CA_DAYS`]; other values in tests only).
    pub ca_days: u64,
}

/// One certificate of the result.
#[derive(Debug, Clone)]
pub struct CertInfo {
    pub path: PathBuf,
    pub subject: String,
    pub not_after: String,
    pub days_left: i64,
    pub sha256: String,
}

/// What a run made.
#[derive(Debug, Clone)]
pub struct MakeCertReport {
    pub made_ca: bool,
    /// Days the server certificate is valid from its start (the days asked, or fewer when the
    /// authority ends sooner).
    pub server_days: u64,
    pub replaced_server: bool,
    pub ca: CertInfo,
    pub server: CertInfo,
    pub server_key: PathBuf,
    pub ca_key: PathBuf,
    pub names: Vec<CertName>,
}

fn info(c: &'static Crypto, cert: &Cert, path: PathBuf) -> Result<CertInfo, String> {
    let (not_after, days_left) = c.not_after(cert)?;
    Ok(CertInfo {
        path,
        subject: c.subject(cert),
        not_after,
        days_left,
        sha256: c.fingerprint(cert)?,
    })
}

/// The common name of a new authority: what a device lists after it is installed, named after
/// the first name given. At most 64 characters (the limit of the field).
fn ca_common_name(given: &[CertName]) -> String {
    let first = given
        .first()
        .map(|n| n.to_string())
        .unwrap_or_else(|| "localhost".into());
    let mut cn = format!("Gaudium local CA {first}");
    cn.truncate(64);
    cn
}

/// Makes (or renews) the server's certificate in `opts.dir`, with a new authority when the
/// directory has none. Nothing is written unless the whole result checks out.
pub fn make_cert(opts: &MakeCertOptions) -> Result<MakeCertReport, String> {
    let dir = &opts.dir;
    let names = with_local_names(&opts.names);
    let ca_cert = dir.join(CA_CERT);
    let ca_key = dir.join(CA_KEY);
    let srv_cert = dir.join(SERVER_CERT);
    let srv_key = dir.join(SERVER_KEY);
    let files = CertFiles {
        ca_cert: exists(&ca_cert),
        ca_key: exists(&ca_key),
        server_cert: exists(&srv_cert),
        server_key: exists(&srv_key),
        overwrite: opts.overwrite,
    };
    let plan = decide::plan(files);
    if !decide::writes_server(plan) {
        return Err(match plan {
            CertPlan::CgHalfCa => format!(
                "{} holds only one of {CA_CERT} and {CA_KEY}: the authority needs both. Put the missing file back, or move the other one away to make a new authority (which every device then has to trust again)",
                dir.display()
            ),
            _ => format!(
                "{} already holds a server certificate or key ({SERVER_CERT}, {SERVER_KEY}): add --cert-overwrite to replace them (the authority is kept)",
                dir.display()
            ),
        });
    }
    let make_ca = decide::writes_ca(plan);
    let c = crypto()?;

    // the authority: a new one, or the one in the directory
    let (ca, ca_pkey, ca_left) = if make_ca {
        let key = c.new_key()?;
        let cert = c.new_cert(&ca_common_name(&opts.names), &key, opts.ca_days)?;
        // self-signed: its own issuer
        c.set_issuer(&cert, &cert)?;
        c.add_ext(
            &cert,
            cert.1,
            "basicConstraints",
            "critical,CA:TRUE,pathlen:0",
        )?;
        c.add_ext(&cert, cert.1, "keyUsage", "critical,keyCertSign,cRLSign")?;
        c.add_ext(&cert, cert.1, "subjectKeyIdentifier", "hash")?;
        c.add_ext(&cert, cert.1, "nameConstraints", &name_constraints(&names))?;
        c.sign(&cert, &key)?;
        (cert, key, opts.ca_days)
    } else {
        let cert = c.read_cert(&ca_cert)?;
        let key = c.read_key(&ca_key)?;
        // SAFETY: both objects are live.
        if unsafe { (c.x509_check_private_key)(cert.1, key.1) } != 1 {
            (unsafe { (c.err_clear_error)() });
            return Err(format!(
                "the key in {} does not belong to the certificate in {}",
                ca_key.display(),
                ca_cert.display()
            ));
        }
        // SAFETY: a live certificate.
        if unsafe { (c.x509_check_ca)(cert.1) } <= 0 {
            return Err(format!(
                "{} is not a certificate authority",
                ca_cert.display()
            ));
        }
        let (_, left) = c.not_after(&cert)?;
        (cert, key, u64::try_from(left).unwrap_or(0))
    };

    let days = match decide::validity(opts.days, ca_left) {
        Validity::Days(d) => d,
        Validity::NoDaysAsked => return Err("--cert-days must be at least 1".into()),
        Validity::AuthoritySpent => {
            return Err(format!(
                "the authority in {} has expired (or ends within a day): move {CA_CERT} and {CA_KEY} away to make a new one, and trust the new {CA_CERT} on each device",
                ca_cert.display()
            ))
        }
    };

    // the server certificate, signed by the authority
    let key = c.new_key()?;
    let cert = c.new_cert("Gaudium server", &key, days)?;
    c.set_issuer(&cert, &ca)?;
    c.add_ext(&cert, ca.1, "basicConstraints", "critical,CA:FALSE")?;
    c.add_ext(&cert, ca.1, "keyUsage", "critical,digitalSignature")?;
    c.add_ext(&cert, ca.1, "extendedKeyUsage", "serverAuth")?;
    c.add_ext(&cert, ca.1, "subjectKeyIdentifier", "hash")?;
    c.add_ext(&cert, ca.1, "authorityKeyIdentifier", "keyid,issuer")?;
    let san = names
        .iter()
        .map(CertName::san)
        .collect::<Vec<_>>()
        .join(",");
    c.add_ext(&cert, ca.1, "subjectAltName", &san)?;
    c.sign(&cert, &ca_pkey)?;
    c.verify(&ca, &cert, &names).map_err(|e| {
        if make_ca {
            format!("the new certificate does not check out: {e}")
        } else {
            format!(
                "the authority in {} cannot sign for these names: {e}. It covers the names it was made for and local addresses; to add another name, move {CA_CERT} and {CA_KEY} away (a new authority is made, and each device has to trust the new {CA_CERT})",
                dir.display()
            )
        }
    })?;

    // the files: all staged first, then put in place
    if !dir.exists() {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let mut staged = Vec::new();
    if make_ca {
        // SAFETY (each closure): the BIO and the objects are live for the call.
        staged.push(stage(c, &ca_key, 0o600, false, |b| unsafe {
            (c.pem_write_bio_private_key)(
                b,
                ca_pkey.1,
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            ) == 1
        })?);
        staged.push(stage(c, &ca_cert, 0o644, false, |b| unsafe {
            (c.pem_write_bio_x509)(b, ca.1) == 1
        })?);
    }
    let replace = files.server_cert || files.server_key;
    staged.push(stage(c, &srv_key, 0o600, files.server_key, |b| unsafe {
        (c.pem_write_bio_private_key)(
            b,
            key.1,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
        ) == 1
    })?);
    staged.push(stage(c, &srv_cert, 0o644, files.server_cert, |b| unsafe {
        (c.pem_write_bio_x509)(b, cert.1) == 1 && (c.pem_write_bio_x509)(b, ca.1) == 1
    })?);
    for i in 0..staged.len() {
        if let Err(e) = staged[i].commit() {
            // files this run created are removed again; replaced ones cannot be brought back
            for s in &staged[..i] {
                if s.created {
                    let _ = fs::remove_file(&s.path);
                }
            }
            return Err(e);
        }
    }

    Ok(MakeCertReport {
        made_ca: make_ca,
        server_days: days,
        replaced_server: replace,
        ca: info(c, &ca, ca_cert)?,
        server: info(c, &cert, srv_cert)?,
        server_key: srv_key,
        ca_key,
        names,
    })
}

/// When the first certificate of a PEM file ends: the date as text and the whole days left
/// (negative once it has ended). For the server's start-up line.
pub fn cert_file_expiry(path: &Path) -> Result<(String, i64), String> {
    let c = crypto()?;
    let cert = c.read_cert(path)?;
    c.not_after(&cert)
}

/// How to trust the authority on each kind of device, for `ca` (the path of `ca.crt`).
pub fn trust_steps(ca: &Path) -> String {
    let ca = ca.display();
    format!(
        "  Firefox (any system)  Settings > Privacy & Security > Certificates > View Certificates >\n\
         \x20                       Authorities > Import {ca}, tick \"Trust this CA to identify websites\"\n\
         \x20 Chrome, Edge (Linux)  Settings > Privacy and security > Security > Manage certificates:\n\
         \x20                       import {ca} as a trusted authority, or from a shell (libnss3-tools):\n\
         \x20                       certutil -d sql:$HOME/.pki/nssdb -A -t C,, -n \"Gaudium local CA\" -i {ca}\n\
         \x20 macOS (Safari, Chrome)  sudo security add-trusted-cert -d -r trustRoot \\\n\
         \x20                       -k /Library/Keychains/System.keychain {ca}\n\
         \x20 Windows (Edge, Chrome)  certutil -addstore -user Root {ca}\n\
         \x20 iPhone, iPad          send {CA_CERT} to the device (AirDrop, mail, Files) and open it; install the\n\
         \x20                       profile in Settings > General > VPN & Device Management, then turn it on in\n\
         \x20                       Settings > General > About > Certificate Trust Settings\n\
         \x20 Android               Settings > Security > Encryption & credentials > Install a certificate >\n\
         \x20                       CA certificate, and pick {CA_CERT}"
    )
}

#[cfg(test)]
#[path = "tls_cert_tests.rs"]
mod tests;

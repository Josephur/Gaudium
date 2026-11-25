//! Tests of `gaudium --make-cert` (`tls_cert.rs`): the files, their modes and names, the refusals
//! to overwrite, the authority's limits, and TLS with the certificate it makes. Each test works in
//! its own directory under the system temp directory and needs the system's OpenSSL 3; without it
//! the test says so and passes.

use super::*;
use crate::tls::{TlsClient, TlsServer};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// A directory name that does not exist yet (the run creates it); removed with this value.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> TempDir {
    TempDir(std::env::temp_dir().join(format!(
        "gaudium-cert-test-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    )))
}

fn available() -> bool {
    match crypto() {
        Ok(_) => true,
        Err(e) => {
            eprintln!("skipped: {e}");
            false
        }
    }
}

fn opts(dir: &Path, names: &str) -> MakeCertOptions {
    MakeCertOptions {
        dir: dir.to_path_buf(),
        names: parse_names(names).unwrap(),
        days: DEFAULT_SERVER_DAYS,
        overwrite: false,
        ca_days: CA_DAYS,
    }
}

fn mode(p: &Path) -> u32 {
    fs::metadata(p).unwrap().permissions().mode() & 0o777
}

fn bytes(dir: &Path) -> Vec<Vec<u8>> {
    [CA_CERT, CA_KEY, SERVER_CERT, SERVER_KEY]
        .iter()
        .map(|f| fs::read(dir.join(f)).unwrap())
        .collect()
}

/// `openssl x509 -noout -text` of a certificate file, when the `openssl` command is here: a
/// second reading of the certificate, by another program.
fn openssl_text(path: &Path) -> Option<String> {
    let out = Command::new("openssl")
        .args(["x509", "-noout", "-text", "-in"])
        .arg(path)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

fn server_covers(dir: &Path, name: &str) -> bool {
    let c = crypto().unwrap();
    let cert = c.read_cert(&dir.join(SERVER_CERT)).unwrap();
    c.covers(&cert, &parse_name(name).unwrap()).unwrap()
}

#[test]
fn names_are_parsed_checked_and_completed() {
    let n = parse_names(" Gaudium.Example.LAN. ,192.168.7.20,[fd00::7], nas ,,").unwrap();
    assert_eq!(
        n,
        vec![
            CertName::Dns("gaudium.example.lan".into()),
            CertName::Ip("192.168.7.20".parse().unwrap()),
            CertName::Ip("fd00::7".parse().unwrap()),
            CertName::Dns("nas".into()),
        ]
    );
    for bad in [
        "*.example.lan",
        "a..b",
        "-a.lan",
        "a_b.lan",
        "192.168.7",
        "0.0.0.0",
        "::",
        "224.0.0.1",
        "é.lan",
    ] {
        assert!(parse_name(bad).is_err(), "{bad}");
    }
    assert!(parse_name(&"a".repeat(64)).is_err());
    // localhost and the loopback addresses come first, and each name only once
    let all =
        with_local_names(&parse_names("localhost,gaudium.lan,127.0.0.1,gaudium.lan").unwrap());
    let shown: Vec<String> = all.iter().map(ToString::to_string).collect();
    assert_eq!(shown, ["localhost", "127.0.0.1", "::1", "gaudium.lan"]);
}

#[test]
fn public_looking_names_are_told_apart() {
    for (name, public) in [
        ("gaudium.lan", false),
        ("nas", false),
        ("box.home.arpa", false),
        ("gaudium.example.com", true),
        ("10.9.8.7", false),
        ("100.100.1.1", false),
        ("fd12::1", false),
        ("fe80::1", false),
        ("203.0.113.9", true),
        ("2001:db8::1", true),
    ] {
        assert_eq!(parse_name(name).unwrap().looks_public(), public, "{name}");
    }
}

#[test]
fn name_constraints_cover_the_names_and_the_local_ranges() {
    let nc = name_constraints(&with_local_names(
        &parse_names("gaudium.lan,192.168.7.20,203.0.113.9").unwrap(),
    ));
    assert!(nc.starts_with("critical,"), "{nc}");
    assert!(nc.contains("permitted;DNS:localhost"), "{nc}");
    assert!(nc.contains("permitted;DNS:gaudium.lan"), "{nc}");
    assert!(nc.contains("permitted;IP:192.168.0.0/255.255.0.0"), "{nc}");
    // an address outside the local ranges is permitted on its own, a local one is not repeated
    assert!(
        nc.contains("permitted;IP:203.0.113.9/255.255.255.255"),
        "{nc}"
    );
    assert!(!nc.contains("192.168.7.20"), "{nc}");
    // no DNS name beyond those given: a stolen key cannot make a certificate for another site
    assert_eq!(nc.matches("DNS:").count(), 2, "{nc}");
}

#[test]
fn a_first_run_makes_the_authority_and_the_server_files() {
    if !available() {
        return;
    }
    let t = temp_dir();
    let r = make_cert(&opts(&t.0, "gaudium.example.lan,192.168.7.20,fd00::7")).unwrap();
    assert!(r.made_ca && !r.replaced_server);
    // the directory is the owner's only; the keys are 0600, the certificates readable
    assert_eq!(mode(&t.0), 0o700);
    assert_eq!(mode(&t.0.join(CA_KEY)), 0o600);
    assert_eq!(mode(&t.0.join(SERVER_KEY)), 0o600);
    assert_eq!(mode(&t.0.join(SERVER_CERT)) & 0o600, 0o600);
    // no temporary file is left
    let mut left: Vec<String> = fs::read_dir(&t.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(left, [CA_CERT, CA_KEY, SERVER_CERT, SERVER_KEY]);
    // the server file is the chain: the server's certificate, then the authority's
    let chain = fs::read_to_string(t.0.join(SERVER_CERT)).unwrap();
    assert_eq!(chain.matches("BEGIN CERTIFICATE").count(), 2);
    let key = fs::read_to_string(t.0.join(SERVER_KEY)).unwrap();
    assert!(key.contains("BEGIN PRIVATE KEY"));
    // the report holds no key material
    let report = format!("{r:?}");
    assert!(!report.contains(key.lines().nth(1).unwrap()));
    // every name, as a TLS client checks it
    for n in [
        "localhost",
        "127.0.0.1",
        "::1",
        "gaudium.example.lan",
        "192.168.7.20",
        "fd00::7",
    ] {
        assert!(server_covers(&t.0, n), "{n}");
    }
    assert!(!server_covers(&t.0, "other.example.lan"));
    // the validity: what was asked, less the clock allowance
    assert!(
        (824..=825).contains(&r.server.days_left),
        "{}",
        r.server.days_left
    );
    assert!(r.ca.days_left >= 3649, "{}", r.ca.days_left);
    assert_eq!(r.server.sha256.len(), 32 * 3 - 1);
    assert_ne!(r.server.sha256, r.ca.sha256);
    assert!(
        r.ca.subject
            .contains("Gaudium local CA gaudium.example.lan"),
        "{}",
        r.ca.subject
    );
    // read again by the openssl command, when there is one
    if let Some(text) = openssl_text(&t.0.join(SERVER_CERT)) {
        assert!(text.contains("DNS:localhost"), "{text}");
        assert!(text.contains("DNS:gaudium.example.lan"), "{text}");
        assert!(text.contains("IP Address:127.0.0.1"), "{text}");
        assert!(text.contains("IP Address:192.168.7.20"), "{text}");
        assert!(text.contains("TLS Web Server Authentication"), "{text}");
        assert!(text.contains("CA:FALSE"), "{text}");
    } else {
        eprintln!("the openssl command is not here: second reading skipped");
    }
    if let Some(text) = openssl_text(&t.0.join(CA_CERT)) {
        assert!(text.contains("CA:TRUE"), "{text}");
        assert!(text.contains("Name Constraints: critical"), "{text}");
        assert!(text.contains("DNS:gaudium.example.lan"), "{text}");
    }
}

#[test]
fn existing_files_are_never_overwritten_without_the_flag() {
    if !available() {
        return;
    }
    let t = temp_dir();
    make_cert(&opts(&t.0, "gaudium.example.lan")).unwrap();
    let before = bytes(&t.0);

    // the same run again: refused, nothing changed
    let e = make_cert(&opts(&t.0, "gaudium.example.lan")).unwrap_err();
    assert!(e.contains("--cert-overwrite"), "{e}");
    assert_eq!(bytes(&t.0), before);

    // a key alone is enough to refuse
    fs::remove_file(t.0.join(SERVER_CERT)).unwrap();
    let e = make_cert(&opts(&t.0, "gaudium.example.lan")).unwrap_err();
    assert!(e.contains("--cert-overwrite"), "{e}");
    assert!(!t.0.join(SERVER_CERT).exists());

    // with the flag: new server files for a new name, the authority untouched
    let mut o = opts(&t.0, "gaudium.example.lan,10.20.30.40");
    o.overwrite = true;
    let r = make_cert(&o).unwrap();
    assert!(!r.made_ca && r.replaced_server);
    let after = bytes(&t.0);
    assert_eq!(after[0], before[0], "{CA_CERT} kept");
    assert_eq!(after[1], before[1], "{CA_KEY} kept");
    assert_ne!(after[3], before[3], "{SERVER_KEY} replaced");
    assert_eq!(mode(&t.0.join(SERVER_KEY)), 0o600);
    assert!(server_covers(&t.0, "10.20.30.40"));
}

#[test]
fn half_an_authority_is_refused() {
    if !available() {
        return;
    }
    let t = temp_dir();
    make_cert(&opts(&t.0, "gaudium.example.lan")).unwrap();
    for f in [CA_KEY, SERVER_CERT, SERVER_KEY] {
        fs::remove_file(t.0.join(f)).unwrap();
    }
    let mut o = opts(&t.0, "gaudium.example.lan");
    o.overwrite = true;
    let e = make_cert(&o).unwrap_err();
    assert!(e.contains("only one of"), "{e}");
    assert!(!t.0.join(SERVER_CERT).exists() && !t.0.join(SERVER_KEY).exists());
    assert!(!t.0.join(CA_KEY).exists());
}

#[test]
fn a_reused_authority_signs_only_inside_its_constraints() {
    if !available() {
        return;
    }
    let t = temp_dir();
    make_cert(&opts(&t.0, "gaudium.example.lan")).unwrap();
    let before = bytes(&t.0);
    let renew = |names: &str| {
        let mut o = opts(&t.0, names);
        o.overwrite = true;
        make_cert(&o)
    };
    // another site's name: refused before anything is written
    let e = renew("www.example.com").unwrap_err();
    assert!(e.contains("cannot sign for these names"), "{e}");
    assert_eq!(bytes(&t.0), before);
    // a new local address, a name under the one given: no new authority needed
    renew("gaudium.example.lan,172.20.1.5,box.gaudium.example.lan").unwrap();
    assert!(server_covers(&t.0, "172.20.1.5"));
    assert!(server_covers(&t.0, "box.gaudium.example.lan"));
    assert_eq!(bytes(&t.0)[0], before[0]);
}

#[test]
fn the_server_certificate_never_outlives_its_authority() {
    if !available() {
        return;
    }
    let t = temp_dir();
    let mut o = opts(&t.0, "gaudium.example.lan");
    o.ca_days = 10;
    let r = make_cert(&o).unwrap();
    assert!(r.server.days_left <= r.ca.days_left, "{r:?}");
    assert!((9..=10).contains(&r.server.days_left), "{r:?}");
    // renewed with the same authority: bounded by what the authority has left
    o.overwrite = true;
    o.ca_days = CA_DAYS;
    let r = make_cert(&o).unwrap();
    assert!(!r.made_ca);
    assert!(
        r.server.days_left <= r.ca.days_left && r.server.days_left >= 8,
        "{r:?}"
    );
    // the start-up line's reading of the file agrees
    let (_, left) = cert_file_expiry(&t.0.join(SERVER_CERT)).unwrap();
    assert_eq!(left, r.server.days_left);
}

/// A TLS server on `addr` that answers one connection: reads up to a newline and writes it back
/// in upper case.
fn echo_server(server: TlsServer, l: TcpListener) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
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
    })
}

fn echo(
    client: &TlsClient,
    server: TlsServer,
    bind: &str,
    name: &str,
) -> Option<Result<String, String>> {
    let l = TcpListener::bind(bind).ok()?;
    let addr: SocketAddr = l.local_addr().unwrap();
    let t = echo_server(server, l);
    let r = client
        .connect(TcpStream::connect(addr).unwrap(), name)
        .map_err(|e| e.to_string())
        .map(|mut s| {
            s.write_all(b"hello\n").unwrap();
            let mut out = String::new();
            let _ = s.read_to_string(&mut out);
            out
        });
    t.join().unwrap();
    Some(r)
}

#[test]
fn clients_that_trust_the_authority_accept_the_server() {
    if !available() {
        return;
    }
    let t = temp_dir();
    make_cert(&opts(&t.0, "gaudium.example.lan")).unwrap();
    let server = || TlsServer::new(&t.0.join(SERVER_CERT), &t.0.join(SERVER_KEY)).unwrap();
    // a client that trusts only the authority (as a device that installed ca.crt)
    let client = TlsClient::new(Some(&t.0.join(CA_CERT))).unwrap();
    for name in ["localhost", "127.0.0.1", "gaudium.example.lan"] {
        let got = echo(&client, server(), "127.0.0.1:0", name).unwrap();
        assert_eq!(got.as_deref(), Ok("HELLO\n"), "{name}");
    }
    match echo(&client, server(), "[::1]:0", "::1") {
        Some(got) => assert_eq!(got.as_deref(), Ok("HELLO\n"), "::1"),
        None => eprintln!("no IPv6 loopback here: ::1 skipped"),
    }
    // a name the certificate was not made for, and a client without the authority: refused
    let e = echo(&client, server(), "127.0.0.1:0", "other.example.lan")
        .unwrap()
        .unwrap_err();
    assert!(e.contains("was not accepted"), "{e}");
    let strict = TlsClient::new(None).unwrap();
    let e = echo(&strict, server(), "127.0.0.1:0", "localhost")
        .unwrap()
        .unwrap_err();
    assert!(e.contains("was not accepted"), "{e}");
}

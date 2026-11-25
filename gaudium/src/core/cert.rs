//! Rust Shell wrappers for the certificate-setup decisions of `verified/tls_cert.bend`
//! (`gaudium --make-cert`).
//!
//! Which files a run writes and how long the server certificate is valid both delegate to the
//! verified core; nothing here decides.

pub use crate::verified::CgPlan as CertPlan;
use crate::verified::{cg_plan, cg_validity, cg_writes_ca, cg_writes_server, CgValid};

/// Which files exist in the certificate directory, and whether the operator asked for the
/// server's two files to be replaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CertFiles {
    pub ca_cert: bool,
    pub ca_key: bool,
    pub server_cert: bool,
    pub server_key: bool,
    pub overwrite: bool,
}

/// What one run writes.
pub fn plan(f: CertFiles) -> CertPlan {
    cg_plan(
        f.ca_cert,
        f.ca_key,
        f.server_cert,
        f.server_key,
        f.overwrite,
    )
}

/// The run writes the server's certificate and key (otherwise it is refused).
pub fn writes_server(p: CertPlan) -> bool {
    cg_writes_server(p)
}

/// The run makes the authority; otherwise it signs with the existing one.
pub fn writes_ca(p: CertPlan) -> bool {
    cg_writes_ca(p)
}

/// How long the server certificate is valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validity {
    /// No days were asked for: refused.
    NoDaysAsked,
    /// The authority has less than a whole day left: refused.
    AuthoritySpent,
    /// The certificate is made, valid for this many days from now.
    Days(u64),
}

/// `asked`: the days the operator asked for; `ca_left`: whole days the authority that signs the
/// certificate is still valid.
pub fn validity(asked: u64, ca_left: u64) -> Validity {
    match cg_validity(asked, ca_left) {
        CgValid::CgNoDaysAsked => Validity::NoDaysAsked,
        CgValid::CgCaSpent => Validity::AuthoritySpent,
        CgValid::CgValidFor(d) => Validity::Days(d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(ca_cert: bool, ca_key: bool, sc: bool, sk: bool, overwrite: bool) -> CertFiles {
        CertFiles {
            ca_cert,
            ca_key,
            server_cert: sc,
            server_key: sk,
            overwrite,
        }
    }

    #[test]
    fn plans_of_today() {
        // an empty directory: authority and server files
        let p = plan(files(false, false, false, false, false));
        assert!(writes_server(p) && writes_ca(p));
        // a second run for new names: the same authority, the server files replaced on request
        let p = plan(files(true, true, true, true, true));
        assert!(writes_server(p) && !writes_ca(p));
        assert_eq!(
            plan(files(true, true, true, true, false)),
            CertPlan::CgServerExists
        );
        assert_eq!(
            plan(files(true, false, false, false, true)),
            CertPlan::CgHalfCa
        );
        assert_eq!(
            plan(files(false, true, false, false, false)),
            CertPlan::CgHalfCa
        );
    }

    #[test]
    fn validity_of_today() {
        assert_eq!(validity(825, 3650), Validity::Days(825));
        assert_eq!(validity(825, 100), Validity::Days(100));
        assert_eq!(validity(825, 0), Validity::AuthoritySpent);
        assert_eq!(validity(0, 3650), Validity::NoDaysAsked);
    }
}

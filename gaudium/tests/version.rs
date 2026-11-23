//! `gaudium --version` names the crate version, which is the version a release tag must carry.

use std::process::Command;

fn version_output(flag: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_gaudium"))
        .arg(flag)
        .output()
        .expect("gaudium runs");
    assert!(out.status.success(), "gaudium {flag} failed: {:?}", out.status);
    String::from_utf8(out.stdout).expect("the version line is UTF-8")
}

#[test]
fn version_flag_prints_the_crate_version() {
    let want = format!("gaudium {}", env!("CARGO_PKG_VERSION"));
    for flag in ["--version", "-V"] {
        let line = version_output(flag);
        let line = line.trim_end();
        // a release build adds " (<commit>)" after the version
        assert!(
            line == want || line.starts_with(&format!("{want} (")),
            "`gaudium {flag}` printed {line:?}, expected {want:?}"
        );
        assert_eq!(line.lines().count(), 1, "one line only: {line:?}");
    }
}

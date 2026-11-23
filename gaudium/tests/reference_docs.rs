//! Keeps `gaudium --help` and `docs/reference.md` true to the code.
//!
//! The test reads the sources of the engine (`src/`, `build.rs`, the kernel glue under
//! `../kernels`) and of its tests, takes every command-line flag and every environment variable
//! they name, and fails when one of them is not documented:
//!
//! - a flag the binary compares its arguments with (a string literal that is exactly `--name`,
//!   outside test code) is in the text `gaudium --help` prints, and the help names no flag the
//!   code no longer has;
//! - an environment variable the engine reads is in that text too, and the help names no
//!   `GAUDIUM_*` / `GAUDI_*` variable the engine no longer reads;
//! - a variable only the tests read is in the section of `docs/reference.md` for them;
//! - `docs/reference.md` holds the help text as the binary prints it.
//!
//! What counts as an environment variable: a string literal that is a whole `GAUDIUM_*` or
//! `GAUDI_*` name wherever it stands (so a name kept in a constant or passed to a helper
//! counts), and any other whole name passed directly to `var(..)`, `var_os(..)` or
//! `getenv(..)`. A name computed at run time is not seen; the per-kernel `GAUDIUM_<KERNEL>_ELF`
//! variables are such a family and are documented under that one spelling.
//!
//! Test code is a `#[cfg(test)]` item, a file whose name ends in `tests.rs`, and everything
//! under a `tests` directory.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The variable to set to `1` to rewrite the help block of `docs/reference.md` from the binary.
const UPDATE_VAR: &str = "GAUDIUM_UPDATE_REFERENCE";
/// Variables of the process environment at large: reading one is not an engine setting.
const GENERAL_ENV: &[&str] = &[
    "HOME",
    "PATH",
    "OUT_DIR",
    "USER",
    "TMPDIR",
    "HOSTNAME",
    "LD_LIBRARY_PATH",
];
/// Flags of other programs that the engine passes on. A flag of this binary never goes here.
const FOREIGN_FLAGS: &[&str] = &[];
/// How the help spells the per-kernel ELF variables.
const ELF_FAMILY: &str = "GAUDIUM_<KERNEL>_ELF";
const HELP_BEGIN: &str =
    "<!-- help:begin (written by the reference_docs test; do not edit by hand) -->";
const HELP_END: &str = "<!-- help:end -->";
const TEST_VARS_BEGIN: &str = "<!-- test-variables:begin -->";
const TEST_VARS_END: &str = "<!-- test-variables:end -->";

/// One string literal of a source file.
#[derive(Debug, PartialEq)]
struct Lit {
    text: String,
    /// Inside test code.
    in_test: bool,
    /// The literal is the first argument of `var(`, `var_os(` or `getenv(`.
    env_read: bool,
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// True when the code just before a literal reads the environment with it.
fn reads_env(tail: &[u8]) -> bool {
    for call in [&b"var("[..], b"var_os(", b"getenv("] {
        if tail.ends_with(call) {
            let before = tail.len() - call.len();
            // `set_var(` and `remove_var(` write the environment
            if before == 0 || !is_ident(tail[before - 1]) {
                return true;
            }
        }
    }
    false
}

/// The string literals of Rust or C source text, each with whether it stands in test code.
/// Comments, character literals and lifetimes are skipped; a `#[cfg(test)]` attribute puts the
/// item after it in test code.
fn literals(src: &str, whole_file_is_test: bool) -> Vec<Lit> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut braces = 0usize;
    // ( and [ that are open
    let mut nest = 0usize;
    // a `#[cfg(test)]` was read and its item has not opened a brace yet: the `nest` it stood at
    let mut pending: Option<usize> = None;
    // inside the braces of a test item: the brace depth to come back to
    let mut test_until: Option<usize> = None;
    // the last code bytes without white space, to see what a literal is an argument of
    let mut tail: Vec<u8> = Vec::new();
    let push_lit = |out: &mut Vec<Lit>, text: &[u8], tail: &mut Vec<u8>, in_test: bool| {
        out.push(Lit {
            text: String::from_utf8_lossy(text).into_owned(),
            in_test,
            env_read: reads_env(tail),
        });
        tail.clear();
        tail.push(b'"');
    };
    while i < b.len() {
        let c = b[i];
        let in_test = whole_file_is_test || pending.is_some() || test_until.is_some();
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'*') {
            let mut depth = 1usize;
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        // raw string: r"..", r#".."#, also after a `b`
        if c == b'r'
            && (i == 0
                || !is_ident(b[i - 1])
                || (b[i - 1] == b'b' && (i < 2 || !is_ident(b[i - 2]))))
        {
            let mut j = i + 1;
            while b.get(j) == Some(&b'#') {
                j += 1;
            }
            if b.get(j) == Some(&b'"') {
                let hashes = j - (i + 1);
                let start = j + 1;
                let mut k = start;
                let end = loop {
                    if k >= b.len() {
                        break b.len();
                    }
                    if b[k] == b'"'
                        && b[k + 1..]
                            .iter()
                            .take(hashes)
                            .filter(|&&h| h == b'#')
                            .count()
                            == hashes
                    {
                        break k;
                    }
                    k += 1;
                };
                push_lit(&mut out, &b[start..end], &mut tail, in_test);
                i = (end + 1 + hashes).min(b.len());
                continue;
            }
        }
        if c == b'"' {
            let start = i + 1;
            let mut k = start;
            while k < b.len() && b[k] != b'"' {
                k += if b[k] == b'\\' { 2 } else { 1 };
            }
            let end = k.min(b.len());
            push_lit(&mut out, &b[start..end], &mut tail, in_test);
            i = end + 1;
            continue;
        }
        if c == b'\'' {
            // a character literal ('x', '\n', '\u{1F600}') or a lifetime ('a)
            if b.get(i + 1) == Some(&b'\\') {
                let mut k = i + 2;
                while k < b.len() && b[k] != b'\'' {
                    k += 1;
                }
                i = k + 1;
                continue;
            }
            let mut k = i + 2;
            while k < b.len() && (b[k] & 0xC0) == 0x80 {
                k += 1;
            }
            if b.get(k) == Some(&b'\'') {
                i = k + 1;
                continue;
            }
            i += 1;
            continue;
        }
        if c == b'#' {
            // `#[cfg(test)]`, white space allowed between its tokens
            let mut k = i;
            let mut seen: Vec<u8> = Vec::new();
            while k < b.len() && seen.len() < 12 {
                if !b[k].is_ascii_whitespace() {
                    seen.push(b[k]);
                }
                k += 1;
            }
            if seen == b"#[cfg(test)]" {
                if pending.is_none() && test_until.is_none() {
                    pending = Some(nest);
                }
                i = k;
                continue;
            }
        }
        match c {
            b'{' => {
                if pending.take().is_some() {
                    test_until = Some(braces);
                }
                braces += 1;
            }
            b'}' => {
                braces = braces.saturating_sub(1);
                if test_until == Some(braces) {
                    test_until = None;
                }
            }
            b'(' | b'[' => nest += 1,
            b')' | b']' => nest = nest.saturating_sub(1),
            // an item without a body: `#[cfg(test)] mod tests;`
            b';' if pending == Some(nest) => pending = None,
            _ => {}
        }
        if !c.is_ascii_whitespace() {
            tail.push(c);
            if tail.len() > 32 {
                tail.drain(..16);
            }
        }
        i += 1;
    }
    out
}

/// `--name`: two dashes, then lower-case letters, digits and single dashes.
fn is_flag(s: &str) -> bool {
    let Some(rest) = s.strip_prefix("--") else {
        return false;
    };
    let ok = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit();
    rest.starts_with(ok) && rest.ends_with(ok) && rest.chars().all(|c| ok(c) || c == '-')
}

/// A whole environment-variable name: capitals, digits and underscores, not ending in one.
fn is_env_name(s: &str) -> bool {
    s.len() >= 2
        && s.starts_with(|c: char| c.is_ascii_uppercase())
        && !s.ends_with('_')
        && s.chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn is_project_name(s: &str) -> bool {
    s.starts_with("GAUDIUM_") || s.starts_with("GAUDI_")
}

/// The spelling a variable is documented under.
fn documented_as(name: &str) -> String {
    if name.starts_with("GAUDIUM_") && name.ends_with("_ELF") {
        ELF_FAMILY.to_string()
    } else {
        name.to_string()
    }
}

/// Every `--flag` a text names.
fn flags_in(text: &str) -> BTreeSet<String> {
    let b = text.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0usize;
    while i + 2 < b.len() {
        let starts = b[i] == b'-'
            && b[i + 1] == b'-'
            && (i == 0 || !(is_ident(b[i - 1]) || b[i - 1] == b'-'));
        if !starts {
            i += 1;
            continue;
        }
        let mut k = i + 2;
        while k < b.len() && (b[k].is_ascii_lowercase() || b[k].is_ascii_digit() || b[k] == b'-') {
            k += 1;
        }
        let cand = text[i..k].trim_end_matches('-');
        if is_flag(cand) {
            out.insert(cand.to_string());
        }
        i = k.max(i + 2);
    }
    out
}

/// Every word of capitals, digits and underscores a text has (`GAUDIUM_<KERNEL>_ELF` counts as
/// one word).
fn names_in(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let text = text.replace(ELF_FAMILY, " \u{1}ELF_FAMILY\u{1} ");
    for word in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '\u{1}')) {
        if word == "\u{1}ELF_FAMILY\u{1}" {
            out.insert(ELF_FAMILY.to_string());
        } else if is_env_name(word) {
            out.insert(word.to_string());
        }
    }
    out
}

fn source_files(dir: &Path, exts: &[&str], out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n != "target" && n != "data") {
                source_files(&p, exts, out);
            }
        } else if p
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| exts.contains(&e))
        {
            out.push(p);
        }
    }
}

/// `p` is a path below the crate's directory.
fn is_test_file(p: &Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with("tests.rs"))
        || p.components().any(|c| c.as_os_str() == "tests")
}

/// What the sources name.
#[derive(Default)]
struct Found {
    flags: BTreeSet<String>,
    engine_env: BTreeSet<String>,
    test_env: BTreeSet<String>,
}

fn scan_sources(manifest: &Path) -> Found {
    let mut files = Vec::new();
    source_files(&manifest.join("src"), &["rs"], &mut files);
    source_files(&manifest.join("tests"), &["rs"], &mut files);
    files.push(manifest.join("build.rs"));
    source_files(
        &manifest.join("../kernels"),
        &["c", "cpp", "h", "hpp", "inc"],
        &mut files,
    );
    let mut found = Found::default();
    let mut test_only = BTreeSet::new();
    for path in files {
        let Ok(src) = fs::read_to_string(&path) else {
            continue;
        };
        let rel = path.strip_prefix(manifest).unwrap_or(&path);
        for lit in literals(&src, is_test_file(rel)) {
            if is_flag(&lit.text) && !lit.in_test && !FOREIGN_FLAGS.contains(&lit.text.as_str()) {
                found.flags.insert(lit.text.clone());
            }
            if !is_env_name(&lit.text) {
                continue;
            }
            let counts = is_project_name(&lit.text)
                || (lit.env_read && !GENERAL_ENV.contains(&lit.text.as_str()));
            if !counts {
                continue;
            }
            if lit.in_test {
                // other programs' variables a test reads are that test's own business
                if is_project_name(&lit.text) {
                    test_only.insert(documented_as(&lit.text));
                }
            } else {
                found.engine_env.insert(documented_as(&lit.text));
            }
        }
    }
    found.test_env = test_only.difference(&found.engine_env).cloned().collect();
    found
}

fn help_text() -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_gaudium"))
        .arg("--help")
        .output()
        .expect("gaudium --help runs");
    assert!(
        out.status.success(),
        "gaudium --help failed: {:?}",
        out.status
    );
    String::from_utf8(out.stdout).expect("the help text is UTF-8")
}

/// Where the text between two marker lines of the reference document starts and ends.
fn span_between(doc: &str, begin: &str, end: &str) -> Option<(usize, usize)> {
    let from = doc.find(begin)? + begin.len();
    let to = from + doc[from..].find(end)?;
    Some((from, to))
}

/// The text between two marker lines of the reference document.
fn between<'a>(doc: &'a str, begin: &str, end: &str) -> Option<&'a str> {
    span_between(doc, begin, end).map(|(from, to)| &doc[from..to])
}

fn help_block(help: &str) -> String {
    format!("\n```text\n{}\n```\n", help.trim_end())
}

#[test]
fn every_flag_and_environment_variable_is_documented() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let doc_path = manifest.join("../docs/reference.md");
    let found = scan_sources(manifest);
    let help = help_text();
    let mut doc =
        fs::read_to_string(&doc_path).unwrap_or_else(|e| panic!("{}: {e}", doc_path.display()));
    let mut problems: Vec<String> = Vec::new();

    let help_flags = flags_in(&help);
    for f in found.flags.difference(&help_flags) {
        problems.push(format!("flag {f} is read by the code and is not in `gaudium --help` (print_usage in src/main.rs)"));
    }
    for f in help_flags.difference(&found.flags) {
        problems.push(format!(
            "`gaudium --help` names {f}, and no code compares an argument with it"
        ));
    }

    let help_names = names_in(&help);
    for v in found.engine_env.difference(&help_names) {
        problems.push(format!(
            "environment variable {v} is read by the engine and is not in `gaudium --help`"
        ));
    }
    for v in help_names
        .iter()
        .filter(|v| is_project_name(v) && !found.engine_env.contains(*v))
    {
        let why = if found.test_env.contains(v) {
            "only test code reads it"
        } else {
            "no code reads it"
        };
        problems.push(format!("`gaudium --help` names {v}, and {why}"));
    }

    let want = help_block(&help);
    match span_between(&doc, HELP_BEGIN, HELP_END) {
        None => problems.push(format!(
            "docs/reference.md has no help block ({HELP_BEGIN} .. {HELP_END})"
        )),
        Some((from, to)) if doc[from..to] != want => {
            if std::env::var(UPDATE_VAR).as_deref() == Ok("1") {
                doc.replace_range(from..to, &want);
                fs::write(&doc_path, &doc)
                    .unwrap_or_else(|e| panic!("{}: {e}", doc_path.display()));
            } else {
                problems.push(format!(
                    "the help block of docs/reference.md is not what `gaudium --help` prints: run this test once with {UPDATE_VAR}=1 and commit the file"
                ));
            }
        }
        Some(_) => {}
    }

    match between(&doc, TEST_VARS_BEGIN, TEST_VARS_END) {
        None => problems.push(format!(
            "docs/reference.md has no section {TEST_VARS_BEGIN} .. {TEST_VARS_END}"
        )),
        Some(section) => {
            let listed = names_in(section);
            for v in found.test_env.difference(&listed) {
                problems.push(format!("environment variable {v} is read by test code and is not in the test-variables section of docs/reference.md"));
            }
            for v in listed
                .iter()
                .filter(|v| is_project_name(v) && !found.test_env.contains(*v))
            {
                problems.push(format!(
                    "docs/reference.md lists {v} as read only by tests, and that is no longer so"
                ));
            }
        }
    }

    assert!(
        problems.is_empty(),
        "undocumented or stale command-line flags and environment variables:\n  - {}\n",
        problems.join("\n  - ")
    );
}

#[test]
fn the_scan_finds_this_crates_own_flags_and_variables() {
    // a scan that silently found nothing would pass everything
    let found = scan_sources(Path::new(env!("CARGO_MANIFEST_DIR")));
    for f in ["--serve", "--generate", "--model", "--help"] {
        assert!(found.flags.contains(f), "{f} not found in the sources");
    }
    for v in [
        "GAUDIUM_KERNELS",
        "GAUDIUM_REQUEST_RESET",
        ELF_FAMILY,
        "GC_KERNEL_PATH",
    ] {
        assert!(found.engine_env.contains(v), "{v} not found in the sources");
    }
    assert!(found.test_env.contains(UPDATE_VAR));
    assert!(!found
        .engine_env
        .iter()
        .any(|v| GENERAL_ENV.contains(&v.as_str())));
}

#[test]
fn literals_are_read_from_code_only() {
    let lits = |src: &str| -> Vec<(String, bool, bool)> {
        literals(src, false)
            .into_iter()
            .map(|l| (l.text, l.in_test, l.env_read))
            .collect()
    };
    let l = |t: &str, in_test: bool, env_read: bool| (t.to_string(), in_test, env_read);
    // comments, character literals and lifetimes hold no literal
    assert_eq!(
        lits("// \"--no\"\n/* \"--no\" /* nested */ \"--no\" */ fn f<'a>(x: &'a str) { let q = '\"'; let n = '\\n'; g(\"--yes\") }"),
        vec![l("--yes", false, false)]
    );
    // escapes and raw strings
    assert_eq!(
        lits(r####"a("x\"y"); b(r"raw\"); c(r#"with "quotes""#); d(br##"bytes"##);"####),
        vec![
            l("x\\\"y", false, false),
            l("raw\\", false, false),
            l("with \"quotes\"", false, false),
            l("bytes", false, false)
        ]
    );
    // what a literal is an argument of
    assert_eq!(
        lits("std::env::var(\"A_B\"); env::var_os( \"C_D\" ); set_var(\"E_F\", v); getenv(\"G_H\"); other(\"I_J\");"),
        vec![l("A_B", false, true), l("C_D", false, true), l("E_F", false, false), l("G_H", false, true), l("I_J", false, false)]
    );
    // a test item ends where its braces end, and an array type's `;` does not end it early
    assert_eq!(
        lits("fn a() { p(\"one\") }\n#[cfg(test)]\nfn t(x: [u8; 4]) { p(\"two\"); { p(\"three\") } }\nfn b() { p(\"four\") }"),
        vec![l("one", false, false), l("two", true, false), l("three", true, false), l("four", false, false)]
    );
    assert_eq!(
        lits("# [ cfg ( test ) ]\nmod tests;\nfn b() { p(\"after\") }\n#[cfg(test)]\nconst X: &str = \"in\";\nconst Y: &str = \"out\";\n#[cfg(test)]\nmod t { #[test] #[ignore = \"why\"] fn f() {} }"),
        vec![l("after", false, false), l("in", true, false), l("out", false, false), l("why", true, false)]
    );
    assert!(literals("p(\"x\")", true)[0].in_test);
}

#[test]
fn flags_and_names_are_told_apart_from_prose() {
    assert!(
        is_flag("--max-ctx")
            && is_flag("--k0")
            && !is_flag("--")
            && !is_flag("-h")
            && !is_flag("--Bad")
    );
    assert!(!is_flag("--trailing-") && !is_flag("--with space"));
    assert_eq!(
        flags_in("  --port <N>  as --host, not a--b, x -- y; `--draft-k`=3 (--no-cache)."),
        ["--draft-k", "--host", "--no-cache", "--port"]
            .map(String::from)
            .into()
    );
    assert!(
        is_env_name("GAUDIUM_KERNELS")
            && is_env_name("HOME")
            && !is_env_name("GAUDIUM_")
            && !is_env_name("Mixed_Case")
    );
    assert_eq!(
        names_in("or GAUDIUM_KERNELS; GAUDI_LIVE=0, GAUDIUM_<KERNEL>_ELF and K=0 HBM"),
        ["GAUDIUM_KERNELS", "GAUDI_LIVE", ELF_FAMILY, "HBM"]
            .map(String::from)
            .into()
    );
    assert_eq!(documented_as("GAUDIUM_GDN_ELF"), ELF_FAMILY);
    assert_eq!(documented_as("GAUDIUM_KERNELS"), "GAUDIUM_KERNELS");
}

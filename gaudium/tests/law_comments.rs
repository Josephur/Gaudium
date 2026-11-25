//! Every new law names the wrong body it rules out (`AGENTS.md` section 3a).
//!
//! A law's comment block is the run of comment lines directly above `law <name>:`, as
//! RustyBends reads it for `# promise:`. A `# catches:` line in that block names one plausible
//! wrong version of the code that the law refuses, for example `# catches: sum replaced by max`.
//! RustyBends ignores the line and keeps it out of `rusty-bends.lock`.
//!
//! The test fails when:
//! - a law that is not sealed yet (its name is not in `rusty-bends.lock`) has no `# catches:`
//!   line; the message names it;
//! - the number of laws without one is not `LAWS_WITHOUT_CATCHES`. The laws sealed before the
//!   rule have none; the number only goes down, when one of them is given its line. A count above
//!   it means a law was sealed without one.
//! - a `# catches:` line is empty.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Laws sealed before `# catches:` lines were required, none of which has one yet. Lower it when
/// one of them gets its line; never raise it.
const LAWS_WITHOUT_CATCHES: usize = 894;

struct LawComments {
    name: String,
    file: PathBuf,
    /// The text of each `# catches:` line in the law's comment block.
    catches: Vec<String>,
}

fn verified_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("verified")
}

/// The comment text of a line of a law's comment block (`# x` or `-- x`), or `None` when the
/// line ends the block.
fn comment_text(line: &str) -> Option<&str> {
    let t = line.trim();
    if t.is_empty() {
        return None;
    }
    t.strip_prefix('#')
        .or_else(|| t.strip_prefix("--"))
        .map(str::trim)
}

fn catches_text(comment: &str) -> Option<&str> {
    let head = comment.get(..8)?;
    head.eq_ignore_ascii_case("catches:")
        .then(|| comment[8..].trim())
}

/// The name of the law a top-level `law <name>` line declares.
fn law_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("law")?;
    if !rest.starts_with(|c: char| c.is_ascii_whitespace()) {
        return None;
    }
    let name = rest
        .trim_start()
        .split(|c: char| c.is_ascii_whitespace() || c == ':' || c == '(')
        .next()?;
    (!name.is_empty()).then_some(name)
}

fn laws_in(file: &Path) -> Vec<LawComments> {
    let text = fs::read_to_string(file).expect("read a .bend file");
    laws_in_text(&text, file)
}

fn laws_in_text(text: &str, file: &Path) -> Vec<LawComments> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(name) = law_name(line) else {
            continue;
        };
        let mut catches = Vec::new();
        for above in lines[..i].iter().rev() {
            let Some(comment) = comment_text(above) else {
                break;
            };
            if let Some(c) = catches_text(comment) {
                catches.push(c.to_string());
            }
        }
        out.push(LawComments {
            name: name.to_string(),
            file: file.to_path_buf(),
            catches,
        });
    }
    out
}

fn all_laws() -> Vec<LawComments> {
    let mut files: Vec<PathBuf> = fs::read_dir(verified_dir())
        .expect("read verified/")
        .map(|e| e.expect("a directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "bend"))
        .collect();
    files.sort();
    files.iter().flat_map(|f| laws_in(f)).collect()
}

fn sealed_names() -> BTreeSet<String> {
    let lock = Path::new(env!("CARGO_MANIFEST_DIR")).join("rusty-bends.lock");
    let text = fs::read_to_string(lock).expect("read rusty-bends.lock");
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .filter_map(|l| l.trim().splitn(3, "  ").nth(1))
        .map(|n| n.trim().to_string())
        .collect()
}

#[test]
fn new_laws_name_the_wrong_body_they_catch() {
    let laws = all_laws();
    assert!(!laws.is_empty(), "no laws found under verified/");
    let sealed = sealed_names();

    let empty: Vec<String> = laws
        .iter()
        .filter(|l| l.catches.iter().any(|c| c.is_empty()))
        .map(|l| format!("{} ({})", l.name, l.file.display()))
        .collect();
    assert!(
        empty.is_empty(),
        "empty `# catches:` line above: {}",
        empty.join(", ")
    );

    let unsealed_without: Vec<String> = laws
        .iter()
        .filter(|l| !sealed.contains(&l.name) && l.catches.is_empty())
        .map(|l| format!("{} ({})", l.name, l.file.display()))
        .collect();
    assert!(
        unsealed_without.is_empty(),
        "new law without a `# catches:` line (AGENTS.md section 3a): {}.\n\
         Add `# catches: <a plausible wrong body the law refuses>` to its comment block, and \
         show that body failing with E_LAW_FALSE.",
        unsealed_without.join(", ")
    );

    let without = laws.iter().filter(|l| l.catches.is_empty()).count();
    assert!(
        without <= LAWS_WITHOUT_CATCHES,
        "{without} laws have no `# catches:` line, {LAWS_WITHOUT_CATCHES} expected: a law was \
         sealed without one. Give it its `# catches:` line (AGENTS.md section 3a); never raise \
         LAWS_WITHOUT_CATCHES."
    );
    assert!(
        without >= LAWS_WITHOUT_CATCHES,
        "{without} laws have no `# catches:` line: lower LAWS_WITHOUT_CATCHES in \
         tests/law_comments.rs from {LAWS_WITHOUT_CATCHES} to {without}."
    );
}

#[test]
fn comment_block_ends_at_a_blank_line() {
    let src = "# promise: p\n# catches: max for sum\nlaw a:\n  {x == x : Nat}\n\n\
               # catches: far away\n\n# promise: q\nlaw b:\n  {x == x : Nat}\n  law c:\n";
    let laws = laws_in_text(src, Path::new("t.bend"));
    let got: Vec<(&str, Vec<&str>)> = laws
        .iter()
        .map(|l| (l.name.as_str(), l.catches.iter().map(String::as_str).collect()))
        .collect();
    // a blank line ends the block, and an indented `law` is not a declaration
    assert_eq!(got, vec![("a", vec!["max for sum"]), ("b", vec![])]);
}

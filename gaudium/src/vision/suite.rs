//! The vision regression suite (`gaudium --vision-suite <suite.json>`): every speculative method
//! with fixed images, checked against the engine's own K=0 stream (AGENTS.md section 7).
//!
//! The suite file (`gaudium/tests/vision_suite/suite.json`) lists the fixed image cases, a text
//! request, the sampler settings and the rows: models, each with its methods (K=0 and every
//! drafter). A row is one child process `gaudium --generate`, run one after another; this
//! process never opens the card, it only waits for each child to exit before it starts the
//! next. One child serves, on one engine load:
//!
//! 1. the text request, on an engine that has never seen an image;
//! 2. every image case (its images given to that request only);
//! 3. the text request again (`--after-prompt-file`): an image request leaves no state behind
//!    when these tokens equal those of step 1;
//!
//! each request under every sampler setting (greedy and seeded sampled), each run preceded by
//! its K=0 reference on the same engine (`--compare-k0`). Per row the suite checks:
//! - **identity**: every run is token-identical to its K=0 reference; an opt-in method (a draft
//!   tree) may fork only at a traced tie of the BF16 logits (the run's margin trace);
//! - **semantics**: every answer (decoded with the model's tokenizer) holds one of the accepted
//!   words of every expected group, so a stream that is identical but wrong still fails;
//! - **after**: step 3 gives step 1's tokens under every sampler setting.
//!
//! The verdict is the verified `vs_identity_ok` / `vs_row_ok` (`verified/vision_suite.bend`);
//! this module parses, runs processes and reads text. A new model family or drafter (Gemma 4
//! and its `-assistant` drafter) is a further `models` entry or `methods` entry of the suite
//! file; nothing here is specific to one.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::json::{parse_json, quote_json, JsonValue};
use crate::verified::{vs_identity_ok, vs_row_ok, VsIdentity};

/// Pause between two rows: the next child opens the card only after the previous one exited
/// and the driver has finished releasing the device.
const SETTLE: Duration = Duration::from_secs(3);

/// One accepted-answer check: the answer must hold one word of every group (case-insensitive).
pub type Expect = Vec<Vec<String>>;

#[derive(Debug, Clone, PartialEq)]
pub struct ImageCase {
    pub name: String,
    /// Image files, in prompt order (relative paths are relative to the suite file).
    pub images: Vec<PathBuf>,
    pub prompt: String,
    pub expect: Expect,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Method {
    pub name: String,
    /// `gaudium --generate` options of the method (`{models}`: the models directory).
    pub args: Vec<String>,
    /// An opt-in mode that may fork from K=0 at a traced tie (AGENTS.md section 7.2).
    pub opt_in_forks: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelRow {
    pub name: String,
    /// Model directory or file (relative: inside the models directory).
    pub model: String,
    pub args: Vec<String>,
    pub methods: Vec<Method>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Suite {
    /// Directory of the suite file (image paths are relative to it).
    pub dir: PathBuf,
    /// Options of every row (prompt form, token count).
    pub args: Vec<String>,
    /// `--sampling` values, in order (the first one is also the after request's first).
    pub samplings: Vec<String>,
    pub text_prompt: String,
    pub text_expect: Expect,
    pub cases: Vec<ImageCase>,
    pub models: Vec<ModelRow>,
}

fn strs(v: Option<&JsonValue>, what: &str) -> Result<Vec<String>, String> {
    let Some(v) = v else {
        return Ok(Vec::new());
    };
    v.as_array()
        .ok_or_else(|| format!("{what}: expected a list of strings"))?
        .iter()
        .map(|s| {
            s.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{what}: expected a list of strings"))
        })
        .collect()
}

fn text(o: &JsonValue, key: &str, what: &str) -> Result<String, String> {
    o.get(key)
        .and_then(JsonValue::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{what}: missing string '{key}'"))
}

fn expect(o: &JsonValue, what: &str) -> Result<Expect, String> {
    let groups = o
        .get("expect")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{what}: missing 'expect' (a list of lists of accepted words)"))?;
    let e: Expect = groups
        .iter()
        .map(|g| strs(Some(g), &format!("{what}: expect")))
        .collect::<Result<_, _>>()?;
    if e.is_empty() || e.iter().any(Vec::is_empty) {
        return Err(format!("{what}: every expect group needs an accepted word"));
    }
    Ok(e)
}

impl Suite {
    pub fn parse(json: &str, dir: &Path) -> Result<Self, String> {
        let j = parse_json(json).map_err(|e| format!("suite file: {e}"))?;
        let text_req = j.get("text").ok_or("suite file: missing 'text'")?;
        let mut cases = Vec::new();
        for c in j.get("cases").and_then(JsonValue::as_array).unwrap_or(&[]) {
            let name = text(c, "name", "case")?;
            let what = format!("case '{name}'");
            let images: Vec<PathBuf> = strs(c.get("images"), &what)?
                .into_iter()
                .map(PathBuf::from)
                .collect();
            if images.is_empty() {
                return Err(format!("{what}: no images"));
            }
            cases.push(ImageCase {
                prompt: text(c, "prompt", &what)?,
                expect: expect(c, &what)?,
                images,
                name,
            });
        }
        if cases.is_empty() {
            return Err("suite file: no image cases".into());
        }
        let mut models = Vec::new();
        for m in j.get("models").and_then(JsonValue::as_array).unwrap_or(&[]) {
            let name = text(m, "name", "model")?;
            let what = format!("model '{name}'");
            let mut methods = Vec::new();
            for k in m.get("methods").and_then(JsonValue::as_array).unwrap_or(&[]) {
                let mname = text(k, "name", &what)?;
                methods.push(Method {
                    args: strs(k.get("args"), &format!("{what} method '{mname}'"))?,
                    opt_in_forks: k
                        .get("opt_in_forks")
                        .and_then(JsonValue::as_bool)
                        .unwrap_or(false),
                    name: mname,
                });
            }
            if methods.is_empty() {
                return Err(format!("{what}: no methods"));
            }
            let mut seen: Vec<&str> = methods.iter().map(|k| k.name.as_str()).collect();
            seen.sort_unstable();
            if seen.windows(2).any(|w| w[0] == w[1]) {
                return Err(format!("{what}: method names must be unique"));
            }
            models.push(ModelRow {
                model: text(m, "model", &what)?,
                args: strs(m.get("args"), &what)?,
                methods,
                name,
            });
        }
        if models.is_empty() {
            return Err("suite file: no models".into());
        }
        let samplings = strs(j.get("samplings"), "samplings")?;
        if samplings.is_empty() {
            return Err("suite file: no sampler settings".into());
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            args: strs(j.get("args"), "args")?,
            samplings,
            text_prompt: text(text_req, "prompt", "text request")?,
            text_expect: expect(text_req, "text request")?,
            cases,
            models,
        })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("read suite file {}: {e}", path.display()))?;
        Self::parse(&json, path.parent().unwrap_or(Path::new(".")))
    }

    /// Requests of one child: the text request, the image cases, in this order.
    pub fn n_requests(&self) -> usize {
        1 + self.cases.len()
    }

    fn image_path(&self, p: &Path) -> PathBuf {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.dir.join(p)
        }
    }
}

// ============================================================================
// What a child printed
// ============================================================================

/// A run's identity report against its K=0 reference.
#[derive(Debug, Clone, PartialEq)]
pub enum IdentityLine {
    Identical,
    LengthDiffers,
    /// First divergence at a generated index; with a greedy run's margin trace there:
    /// `(K=0 token's logit, the run's token's logit)`.
    Diverged {
        index: usize,
        trace: Option<(f64, f64)>,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunReport {
    pub tokens: Option<Vec<i32>>,
    pub identity: Option<IdentityLine>,
    pub tokens_per_launch: Option<f64>,
    /// `(accepted, drafted)` drafts.
    pub drafts: Option<(u64, u64)>,
}

/// The runs of one child keyed by `(request, sampler setting)`, and its after-request tokens in
/// setting order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChildReport {
    pub runs: BTreeMap<(usize, usize), RunReport>,
    pub after: Vec<Vec<i32>>,
}

fn parse_token_list(s: &str) -> Option<Vec<i32>> {
    let inner = s.trim().strip_prefix('[')?.strip_suffix(']')?;
    inner
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<i32>().ok())
        .collect()
}

/// The number after `key` in `line` (up to the next space, `|`, `)`, `;`, `,` or `:`).
fn number_after(line: &str, key: &str) -> Option<f64> {
    let rest = &line[line.find(key)? + key.len()..];
    let end = rest
        .find(|c: char| matches!(c, ' ' | '|' | ')' | ';' | ',' | ':'))
        .unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Parses the stderr of a `gaudium --generate --compare-k0 --dump-tokens` child: a run starts
/// at its `[gaudium] request <i>:` line (the n-th line of request i is its sampler setting n).
pub fn parse_child_stderr(text: &str) -> ChildReport {
    let mut rep = ChildReport::default();
    let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
    let mut cur: Option<(usize, usize)> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("[gaudium] request ") {
            let idx = rest
                .split(':')
                .next()
                .and_then(|n| n.trim().parse::<usize>().ok());
            if let Some(i) = idx {
                let n = seen.entry(i).or_insert(0);
                cur = Some((i, *n));
                *n += 1;
                rep.runs.entry((i, *n - 1)).or_default();
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("AFTER TOKENS: ") {
            if let Some(t) = parse_token_list(rest) {
                rep.after.push(t);
            }
            continue;
        }
        let Some(run) = cur.and_then(|k| rep.runs.get_mut(&k)) else {
            continue;
        };
        if let Some(rest) = line.strip_prefix("TOKENS: ") {
            run.tokens = parse_token_list(rest);
        } else if line.starts_with("[gaudium] IDENTITY vs K=0") {
            run.identity = if line.contains("): identical,") {
                Some(IdentityLine::Identical)
            } else if line.contains("LENGTH DIFFERS") {
                Some(IdentityLine::LengthDiffers)
            } else {
                number_after(line, "FIRST DIVERGENCE at generated index ").map(|i| {
                    IdentityLine::Diverged {
                        index: i as usize,
                        trace: None,
                    }
                })
            };
        } else if line.starts_with("[gaudium] FORK TRACE at generated index ") {
            let at = number_after(line, "FORK TRACE at generated index ").map(|i| i as usize);
            let k0 = number_after(line, " logit ");
            let ours = line
                .find(" vs the run's token ")
                .and_then(|p| number_after(&line[p..], " logit "));
            if let (Some(IdentityLine::Diverged { index, trace }), Some(a), Some(b), Some(at)) =
                (run.identity.as_mut(), k0, ours, at)
            {
                if *index == at {
                    *trace = Some((a, b));
                }
            }
        } else if line.starts_with("[gaudium] method: ") {
            run.tokens_per_launch = number_after(line, "tokens/launch: ");
            if let Some(p) = line.find("drafts accepted: ") {
                let rest = &line[p + "drafts accepted: ".len()..];
                let frac = rest.split_whitespace().next().unwrap_or("");
                if let Some((a, d)) = frac.split_once('/') {
                    if let (Ok(a), Ok(d)) = (a.parse(), d.parse()) {
                        run.drafts = Some((a, d));
                    }
                }
            }
        }
    }
    rep
}

/// One unit in the last place of a BF16 value of magnitude `x` (8 significant bits).
pub fn bf16_ulp(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() || a < f64::from(f32::MIN_POSITIVE) {
        return f64::from(f32::MIN_POSITIVE);
    }
    (2.0f64).powi(a.log2().floor() as i32 - 7)
}

/// The verified outcome of one run: a fork counts as a traced tie when its margin trace shows
/// the two tokens' BF16 logits at most one unit in the last place apart (both printed rounded
/// to 4 decimals, hence the tolerance of that rounding).
pub fn classify(identity: Option<&IdentityLine>) -> VsIdentity {
    match identity {
        None => VsIdentity::VsMissing,
        Some(IdentityLine::Identical) => VsIdentity::VsSame,
        Some(IdentityLine::LengthDiffers) => VsIdentity::VsFork,
        Some(IdentityLine::Diverged { trace: None, .. }) => VsIdentity::VsFork,
        Some(IdentityLine::Diverged {
            trace: Some((k0, ours)),
            ..
        }) => {
            let ulp = bf16_ulp(k0.abs().max(ours.abs()));
            if (k0 - ours).abs() <= ulp + 1e-4 {
                VsIdentity::VsTracedTie
            } else {
                VsIdentity::VsFork
            }
        }
    }
}

/// The answer holds one accepted word of every group (case-insensitive substring).
pub fn answer_ok(answer: &str, expect: &Expect) -> bool {
    let a = answer.to_lowercase();
    expect
        .iter()
        .all(|g| g.iter().any(|w| a.contains(&w.to_lowercase())))
}

// ============================================================================
// Verdict of one row
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct RowResult {
    pub model: String,
    pub method: String,
    /// Child process exit status (`None`: it was not started or did not exit normally).
    pub exit: Option<i32>,
    pub identity_ok: bool,
    /// Runs and how many passed identity, with a note per failure.
    pub identity: (usize, usize),
    pub forks: Vec<String>,
    pub semantic: (usize, usize),
    pub wrong_answers: Vec<String>,
    pub after_ok: bool,
    pub drafts: (u64, u64),
    pub tokens_per_launch: Vec<f64>,
    pub pass: bool,
    pub seconds: f64,
}

/// Judges one child's report. `decode` turns token ids into text (the model's tokenizer).
pub fn judge(
    suite: &Suite,
    method: &Method,
    rep: &ChildReport,
    decode: &dyn Fn(&[i32]) -> String,
) -> RowResult {
    let ns = suite.samplings.len();
    let mut ident = (0usize, 0usize);
    let mut identity_ok = true;
    let mut forks = Vec::new();
    let mut sem = (0usize, 0usize);
    let mut wrong = Vec::new();
    let mut drafts = (0u64, 0u64);
    let mut tpl = Vec::new();
    for r in 0..suite.n_requests() {
        let (label, exp) = match r {
            0 => ("text".to_string(), &suite.text_expect),
            _ => (suite.cases[r - 1].name.clone(), &suite.cases[r - 1].expect),
        };
        for s in 0..ns {
            let run = rep.runs.get(&(r, s));
            let outcome = classify(run.and_then(|x| x.identity.as_ref()));
            let ok = vs_identity_ok(outcome, method.opt_in_forks);
            ident.1 += 1;
            if ok {
                ident.0 += 1;
            }
            identity_ok &= ok;
            match (outcome, run.and_then(|x| x.identity.as_ref())) {
                (VsIdentity::VsSame, _) => {}
                (o, Some(IdentityLine::Diverged { index, trace })) => forks.push(format!(
                    "{label} / sampling {s}: fork at token {index}{} ({})",
                    trace.map_or(String::new(), |(a, b)| format!(
                        ", K=0 logit {a:.4} vs {b:.4}"
                    )),
                    match o {
                        VsIdentity::VsTracedTie => "traced tie",
                        _ => "not a traced tie",
                    }
                )),
                (_, Some(IdentityLine::LengthDiffers)) => {
                    forks.push(format!("{label} / sampling {s}: length differs from K=0"))
                }
                _ => forks.push(format!("{label} / sampling {s}: no identity report")),
            }
            sem.1 += 1;
            match run.and_then(|x| x.tokens.as_deref()) {
                Some(t) => {
                    let answer = decode(t);
                    if answer_ok(&answer, exp) {
                        sem.0 += 1;
                    } else {
                        wrong.push(format!(
                            "{label} / sampling {s}: {}",
                            answer.trim().replace('\n', " ")
                        ));
                    }
                }
                None => wrong.push(format!("{label} / sampling {s}: no tokens")),
            }
            if let Some(x) = run {
                if let Some((a, d)) = x.drafts {
                    drafts.0 += a;
                    drafts.1 += d;
                }
                if let Some(t) = x.tokens_per_launch {
                    tpl.push(t);
                }
            }
        }
    }
    // the text request after the images: under every setting, the fresh engine's tokens
    let after_ok = rep.after.len() == ns
        && (0..ns).all(|s| {
            rep.runs
                .get(&(0, s))
                .and_then(|x| x.tokens.as_ref())
                .is_some_and(|t| *t == rep.after[s])
        });
    let semantic_ok = sem.0 == sem.1;
    RowResult {
        model: String::new(),
        method: method.name.clone(),
        exit: None,
        identity_ok,
        identity: ident,
        forks,
        semantic: sem,
        wrong_answers: wrong,
        after_ok,
        drafts,
        tokens_per_launch: tpl,
        pass: vs_row_ok(identity_ok, semantic_ok, after_ok),
        seconds: 0.0,
    }
}

// ============================================================================
// Running the rows
// ============================================================================

/// Options of `gaudium --vision-suite`.
#[derive(Debug, Clone, Default)]
pub struct SuiteOptions {
    pub suite: PathBuf,
    /// Directory relative model paths and `{models}` are resolved in.
    pub models_dir: Option<PathBuf>,
    pub kernels: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    /// Where prompts, child logs and `report.json` go.
    pub out_dir: PathBuf,
    /// Rows to run: `<model>` or `<model>/<method>` (empty: all).
    pub only: Vec<String>,
    /// Options added to every child (after the suite's own).
    pub extra: Vec<String>,
    /// Compile every row's recipes into the recipe cache (`--build-cache` children, which never
    /// open the card) instead of running the rows.
    pub build_cache: bool,
}

fn selected(only: &[String], model: &str, method: &str) -> bool {
    only.is_empty()
        || only
            .iter()
            .any(|o| o == model || *o == format!("{model}/{method}"))
}

/// The child's command line for one row (prompt files already written to `row_dir`).
pub fn child_args(
    suite: &Suite,
    opts: &SuiteOptions,
    row: &ModelRow,
    method: &Method,
    row_dir: &Path,
) -> Result<Vec<String>, String> {
    let models = opts.models_dir.as_ref().map(|p| p.display().to_string());
    let subst = |a: &String| -> Result<String, String> {
        match (&models, a.contains("{models}")) {
            (_, false) => Ok(a.clone()),
            (Some(m), true) => Ok(a.replace("{models}", m)),
            (None, true) => Err(format!("'{a}' needs --models <dir>")),
        }
    };
    let model = if Path::new(&row.model).is_absolute() {
        row.model.clone()
    } else {
        match &models {
            Some(m) => Path::new(m).join(&row.model).display().to_string(),
            None => return Err(format!("model '{}' is relative: give --models <dir>", row.model)),
        }
    };
    let mut a = vec!["--generate".to_string(), "--model".into(), model];
    if let Some(k) = &opts.kernels {
        a.extend(["--kernels".into(), k.display().to_string()]);
    }
    if let Some(c) = &opts.cache_dir {
        a.extend(["--cache-dir".into(), c.display().to_string()]);
    }
    for x in suite.args.iter().chain(&row.args).chain(&method.args) {
        a.push(subst(x)?);
    }
    a.extend(opts.extra.iter().cloned());
    let text = row_dir.join("text.txt");
    a.extend(["--prompt-file".into(), text.display().to_string()]);
    for (i, c) in suite.cases.iter().enumerate() {
        let p = row_dir.join(format!("case{i}-{}.txt", c.name));
        a.extend(["--also-prompt-file".into(), p.display().to_string()]);
        for img in &c.images {
            a.extend(["--image".into(), suite.image_path(img).display().to_string()]);
        }
    }
    a.extend(["--after-prompt-file".into(), text.display().to_string()]);
    a.extend(["--compare-k0".into(), "--dump-tokens".into()]);
    for s in &suite.samplings {
        a.extend(["--sampling".into(), s.clone()]);
    }
    Ok(a)
}

/// Sends SIGTERM (a stop between synced launches; never SIGKILL, which could leave launches
/// queued on the card).
#[allow(unsafe_code)]
fn term_child(pid: u32) {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    const SIGTERM: i32 = 15;
    // SAFETY: plain syscall on our own child's pid.
    unsafe {
        kill(pid as i32, SIGTERM);
    }
}

/// Runs one child to its exit; returns its exit code (`None` when it ended by a signal).
fn run_child(exe: &Path, args: &[String], row_dir: &Path) -> Result<Option<i32>, String> {
    let out = std::fs::File::create(row_dir.join("stdout.txt"))
        .map_err(|e| format!("create {}: {e}", row_dir.display()))?;
    let err = std::fs::File::create(row_dir.join("stderr.txt"))
        .map_err(|e| format!("create {}: {e}", row_dir.display()))?;
    std::fs::write(row_dir.join("command.txt"), args.join("\n") + "\n")
        .map_err(|e| format!("write command: {e}"))?;
    let mut child = Command::new(exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()
        .map_err(|e| format!("start {}: {e}", exe.display()))?;
    let mut forwarded = false;
    loop {
        match child.try_wait() {
            Ok(Some(st)) => return Ok(st.code()),
            Ok(None) => {}
            Err(e) => return Err(format!("wait for the row's process: {e}")),
        }
        if crate::signals::term_requested() && !forwarded {
            term_child(child.id());
            forwarded = true;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn row_json(r: &RowResult) -> String {
    let list = |v: &[String]| -> String {
        let items: Vec<String> = v.iter().map(|s| quote_json(s)).collect();
        format!("[{}]", items.join(","))
    };
    format!(
        "{{\"model\":{},\"method\":{},\"pass\":{},\"exit\":{},\"identity_ok\":{},\"identity\":[{},{}],\"forks\":{},\"semantic\":[{},{}],\"wrong_answers\":{},\"after_ok\":{},\"drafts\":[{},{}],\"tokens_per_launch\":[{}],\"seconds\":{:.1}}}",
        quote_json(&r.model),
        quote_json(&r.method),
        r.pass,
        r.exit.map_or("null".to_string(), |c| c.to_string()),
        r.identity_ok,
        r.identity.0,
        r.identity.1,
        list(&r.forks),
        r.semantic.0,
        r.semantic.1,
        list(&r.wrong_answers),
        r.after_ok,
        r.drafts.0,
        r.drafts.1,
        r.tokens_per_launch
            .iter()
            .map(|t| format!("{t:.3}"))
            .collect::<Vec<_>>()
            .join(","),
        r.seconds
    )
}

/// One line of the result table.
pub fn table_line(r: &RowResult) -> String {
    let acc = if r.drafts.1 > 0 {
        format!("{:.1}%", 100.0 * r.drafts.0 as f64 / r.drafts.1 as f64)
    } else {
        "-".into()
    };
    let tpl = if r.tokens_per_launch.is_empty() {
        "-".into()
    } else {
        format!(
            "{:.2}",
            r.tokens_per_launch.iter().sum::<f64>() / r.tokens_per_launch.len() as f64
        )
    };
    format!(
        "{:<22} {:<10} {:>7} {:>7} {:>5} {:>7} {:>6} {:>6.0}s  {}",
        r.model,
        r.method,
        format!("{}/{}", r.identity.0, r.identity.1),
        format!("{}/{}", r.semantic.0, r.semantic.1),
        if r.after_ok { "ok" } else { "FAIL" },
        acc,
        tpl,
        r.seconds,
        if r.pass { "PASS" } else { "FAIL" }
    )
}

/// Runs the selected rows one after another and prints the result table. Returns whether
/// every row passed.
pub fn run(opts: &SuiteOptions) -> Result<bool, String> {
    let suite = Suite::load(&opts.suite)?;
    for c in &suite.cases {
        for img in &c.images {
            let p = suite.image_path(img);
            if !p.is_file() {
                return Err(format!("case '{}': no image file {}", c.name, p.display()));
            }
        }
    }
    let exe = std::env::current_exe().map_err(|e| format!("this program's path: {e}"))?;
    std::fs::create_dir_all(&opts.out_dir)
        .map_err(|e| format!("create {}: {e}", opts.out_dir.display()))?;
    crate::signals::install();
    let rows: Vec<(&ModelRow, &Method)> = suite
        .models
        .iter()
        .flat_map(|m| m.methods.iter().map(move |k| (m, k)))
        .filter(|(m, k)| selected(&opts.only, &m.name, &k.name))
        .collect();
    if rows.is_empty() {
        return Err("no row of the suite file matches --only".into());
    }
    eprintln!(
        "[vision-suite] {} row(s), {} image case(s) + a text request, {} sampler setting(s); logs in {}",
        rows.len(),
        suite.cases.len(),
        suite.samplings.len(),
        opts.out_dir.display()
    );
    let mut results: Vec<RowResult> = Vec::new();
    for (i, (row, method)) in rows.iter().enumerate() {
        if crate::signals::term_requested() {
            eprintln!("[vision-suite] stop requested: the remaining rows are not run");
            break;
        }
        if i > 0 {
            std::thread::sleep(SETTLE);
        }
        let row_dir = opts.out_dir.join(format!("{}--{}", row.name, method.name));
        std::fs::create_dir_all(&row_dir)
            .map_err(|e| format!("create {}: {e}", row_dir.display()))?;
        std::fs::write(row_dir.join("text.txt"), &suite.text_prompt)
            .map_err(|e| format!("write prompt: {e}"))?;
        for (ci, c) in suite.cases.iter().enumerate() {
            std::fs::write(row_dir.join(format!("case{ci}-{}.txt", c.name)), &c.prompt)
                .map_err(|e| format!("write prompt: {e}"))?;
        }
        let mut args = child_args(&suite, opts, row, method, &row_dir)?;
        if opts.build_cache {
            args[0] = "--build-cache".into();
        }
        eprintln!("[vision-suite] {}/{}: {} / {} ...", i + 1, rows.len(), row.name, method.name);
        let t0 = Instant::now();
        let exit = run_child(&exe, &args, &row_dir)?;
        if opts.build_cache {
            let ok = exit == Some(0);
            eprintln!(
                "[vision-suite]   build-cache {} in {:.0}s{}",
                if ok { "done" } else { "FAILED" },
                t0.elapsed().as_secs_f64(),
                if ok {
                    String::new()
                } else {
                    format!(" (see {})", row_dir.join("stderr.txt").display())
                }
            );
            if !ok {
                return Ok(false);
            }
            continue;
        }
        let stderr = std::fs::read_to_string(row_dir.join("stderr.txt")).unwrap_or_default();
        let rep = parse_child_stderr(&stderr);
        // the model's own tokenizer for the answers
        let model_path = args[2].clone();
        let tok = crate::model_text::ModelText::open(Path::new(&model_path))
            .and_then(|t| t.tokenizer());
        let decode = |ids: &[i32]| -> String {
            match &tok {
                Ok(t) => t.decode(ids, true),
                Err(_) => String::new(),
            }
        };
        let mut r = judge(&suite, method, &rep, &decode);
        r.model = row.name.clone();
        r.exit = exit;
        r.seconds = t0.elapsed().as_secs_f64();
        if exit != Some(0) {
            r.pass = false;
            let tail: Vec<&str> = stderr.lines().rev().take(6).collect();
            r.forks.push(format!(
                "the row's process ended with {}: {}",
                exit.map_or("a signal".to_string(), |c| format!("exit code {c}")),
                tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
            ));
        }
        if let Err(e) = &tok {
            r.wrong_answers.push(format!("no tokenizer to read the answers: {e}"));
        }
        eprintln!("[vision-suite]   {}", table_line(&r));
        results.push(r);
    }
    if opts.build_cache {
        return Ok(!crate::signals::term_requested());
    }
    let all = !results.is_empty() && results.len() == rows.len() && results.iter().all(|r| r.pass);
    let mut out = std::io::stdout().lock();
    let _ = writeln!(
        out,
        "{:<22} {:<10} {:>7} {:>7} {:>5} {:>7} {:>6} {:>7}  result",
        "model", "method", "ident", "answer", "after", "accept", "tok/l", "time"
    );
    for r in &results {
        let _ = writeln!(out, "{}", table_line(r));
        for f in &r.forks {
            let _ = writeln!(out, "    identity: {f}");
        }
        for w in &r.wrong_answers {
            let _ = writeln!(out, "    answer: {w}");
        }
    }
    let _ = writeln!(
        out,
        "vision suite: {}",
        if all { "PASS" } else { "FAIL" }
    );
    let report = format!(
        "{{\"suite\":{},\"pass\":{},\"rows\":[{}]}}\n",
        quote_json(&opts.suite.display().to_string()),
        all,
        results.iter().map(row_json).collect::<Vec<_>>().join(",")
    );
    std::fs::write(opts.out_dir.join("report.json"), report)
        .map_err(|e| format!("write report.json: {e}"))?;
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracked_suite() -> Suite {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vision_suite/suite.json");
        Suite::load(&p).unwrap()
    }

    #[test]
    fn tracked_suite_file_is_complete() {
        let s = tracked_suite();
        for c in &s.cases {
            for img in &c.images {
                assert!(s.image_path(img).is_file(), "{}", img.display());
            }
        }
        // greedy first (the after request's first setting), then a seeded sampled setting
        assert_eq!(s.samplings[0], "0");
        assert!(s.samplings.len() >= 2);
        // the methods every Qwen3.8-27B row carries
        for m in &s.models {
            let names: Vec<&str> = m.methods.iter().map(|k| k.name.as_str()).collect();
            for want in ["k0", "dspine", "dflash2", "mtp-chain", "mtp-tree"] {
                assert!(names.contains(&want), "{}: {want}", m.name);
            }
            // only the draft tree may fork (opt-in)
            for k in &m.methods {
                assert_eq!(k.opt_in_forks, k.name == "mtp-tree", "{}", k.name);
            }
        }
        // a case whose image span crosses the 512-row prefill tile: two images in one request
        assert!(s.cases.iter().any(|c| c.images.len() >= 2));
    }

    #[test]
    fn child_command_carries_every_request_and_check() {
        let s = tracked_suite();
        let opts = SuiteOptions {
            models_dir: Some(PathBuf::from("/models")),
            kernels: Some(PathBuf::from("/k")),
            ..Default::default()
        };
        let row = &s.models[0];
        let k = row.methods.iter().find(|k| k.name == "dspine").unwrap();
        let a = child_args(&s, &opts, row, k, Path::new("/out/r")).unwrap();
        assert_eq!(a[..3], ["--generate", "--model", "/models/Qwen3.8-27B"]);
        let pos = |f: &str| a.iter().position(|x| x == f).unwrap();
        assert!(a.iter().any(|x| x == "/models/Qwen3.8-27B-DSpine-b16"));
        assert_eq!(a.iter().filter(|x| *x == "--also-prompt-file").count(), s.cases.len());
        assert_eq!(a.iter().filter(|x| *x == "--sampling").count(), s.samplings.len());
        // images follow the request they belong to; the text request runs first and last
        assert!(pos("--prompt-file") < pos("--also-prompt-file"));
        assert!(pos("--also-prompt-file") < pos("--image"));
        assert_eq!(a[pos("--prompt-file") + 1], a[pos("--after-prompt-file") + 1]);
        assert!(a.contains(&"--compare-k0".to_string()) && a.contains(&"--dump-tokens".to_string()));
        // a relative model or {models} without a models directory is refused
        let none = SuiteOptions::default();
        assert!(child_args(&s, &none, row, k, Path::new("/out/r")).is_err());
    }

    const STDERR: &str = "\
[gaudium] request 0: prompt_tokens=20 sampling: greedy
[gaudium] K=0 reference: plain | prefill 10.0 ms | decode: 5 tok in 10.0 ms (1 tok/s, 1 ms/step)
TOKENS: [1, 2, 3]
[gaudium] method: dspine K=7,15 | launches: 2 | tokens/launch: 1.500 | ms/launch: 20.00 | drafts accepted: 4/14 (28.6%)
[gaudium] IDENTITY vs K=0 (greedy): identical, 3 tokens
[gaudium] request 0: prompt_tokens=20 sampling: T=0.7 seed 7
TOKENS: [1, 2, 4]
[gaudium] IDENTITY vs K=0 (T=0.7): identical, 3 tokens
[gaudium] request 1: prompt_tokens=300 sampling: greedy
TOKENS: [5, 6]
[gaudium] IDENTITY vs K=0 (greedy): FIRST DIVERGENCE at generated index 1: K=0 7 vs speculative 6 (of 2 / 2 tokens)
[gaudium] FORK TRACE at generated index 1: K=0 token 7 logit 21.3750 vs the run's token 6 logit 21.2500 (margin 0.1250 BF16 logits; 0 earlier disagreements on the forced prefix)
[gaudium] request 1: prompt_tokens=300 sampling: T=0.7 seed 7
TOKENS: [5, 8]
[gaudium] IDENTITY vs K=0 (T=0.7): LENGTH DIFFERS (K=0 3 tokens, speculative 2), common prefix identical
AFTER TOKENS: [1, 2, 3]
[gaudium] text after image: prefill 20 tok
AFTER TOKENS: [1, 2, 4]
";

    #[test]
    fn child_report_is_parsed_per_request_and_setting() {
        let r = parse_child_stderr(STDERR);
        assert_eq!(r.runs.len(), 4);
        let a = &r.runs[&(0, 0)];
        assert_eq!(a.tokens, Some(vec![1, 2, 3]));
        assert_eq!(a.identity, Some(IdentityLine::Identical));
        assert_eq!(a.drafts, Some((4, 14)));
        assert_eq!(a.tokens_per_launch, Some(1.5));
        assert_eq!(r.runs[&(0, 1)].tokens, Some(vec![1, 2, 4]));
        assert_eq!(
            r.runs[&(1, 0)].identity,
            Some(IdentityLine::Diverged {
                index: 1,
                trace: Some((21.375, 21.25))
            })
        );
        assert_eq!(r.runs[&(1, 1)].identity, Some(IdentityLine::LengthDiffers));
        assert_eq!(r.after, vec![vec![1, 2, 3], vec![1, 2, 4]]);
    }

    #[test]
    fn only_a_traced_tie_within_one_bf16_step_is_a_tie() {
        // BF16 has 8 significant bits: logits in [16, 32) step by 1/8
        assert_eq!(bf16_ulp(21.375), 0.125);
        assert_eq!(bf16_ulp(-3.0), 1.0 / 64.0);
        let tie = |a: f64, b: f64| IdentityLine::Diverged {
            index: 3,
            trace: Some((a, b)),
        };
        assert_eq!(classify(Some(&tie(21.375, 21.25))), VsIdentity::VsTracedTie);
        assert_eq!(classify(Some(&tie(21.375, 21.375))), VsIdentity::VsTracedTie);
        assert_eq!(classify(Some(&tie(21.5, 21.25))), VsIdentity::VsFork);
        let untraced = IdentityLine::Diverged {
            index: 0,
            trace: None,
        };
        assert_eq!(classify(Some(&untraced)), VsIdentity::VsFork);
        assert_eq!(classify(Some(&IdentityLine::LengthDiffers)), VsIdentity::VsFork);
        assert_eq!(classify(None), VsIdentity::VsMissing);
    }

    #[test]
    fn a_row_needs_identity_answers_and_a_clean_after_request() {
        let mut s = tracked_suite();
        s.cases.truncate(1);
        s.samplings = vec!["0".into(), "0.7:20:0.95:0:7".into()];
        s.text_expect = vec![vec!["paris".into()]];
        s.cases[0].expect = vec![vec!["red".into()], vec!["circle".into(), "disc".into()]];
        let exact = Method {
            name: "dspine".into(),
            args: Vec::new(),
            opt_in_forks: false,
        };
        let tree = Method {
            opt_in_forks: true,
            ..exact.clone()
        };
        let decode = |ids: &[i32]| -> String {
            match ids.first() {
                Some(1) => "Paris.".into(),
                Some(5) => "A red disc.".into(),
                _ => "Lyon".into(),
            }
        };
        let mut rep = parse_child_stderr(STDERR);
        // the traced tie of request 1 passes the tree, never the default method; the length
        // difference fails both
        let r = judge(&s, &exact, &rep, &decode);
        assert!(!r.pass && !r.identity_ok && r.after_ok);
        assert_eq!(r.identity, (2, 4));
        assert_eq!(r.semantic, (4, 4));
        rep.runs.get_mut(&(1, 1)).unwrap().identity = Some(IdentityLine::Identical);
        assert!(!judge(&s, &exact, &rep, &decode).pass);
        let t = judge(&s, &tree, &rep, &decode);
        assert!(t.pass, "{t:?}");
        // a wrong answer fails the row even with identical streams
        let lyon = |_: &[i32]| -> String { "Lyon".into() };
        assert!(!judge(&s, &tree, &rep, &lyon).pass);
        // state left behind: the after request differs from the fresh engine's
        rep.after[1] = vec![1, 2, 5];
        assert!(!judge(&s, &tree, &rep, &decode).pass);
        // a missing after request fails too
        rep.after.truncate(1);
        assert!(!judge(&s, &tree, &rep, &decode).pass);
        // a run that never reported (the process died) fails
        let mut cut = parse_child_stderr(STDERR);
        cut.runs.remove(&(1, 1));
        assert!(!judge(&s, &tree, &cut, &decode).pass);
    }

    #[test]
    fn answers_need_one_word_of_every_group() {
        let e: Expect = vec![
            vec!["red".into()],
            vec!["square".into(), "rectangle".into()],
        ];
        assert!(answer_ok("A RED Rectangle and more", &e));
        assert!(!answer_ok("A blue square", &e));
        assert!(!answer_ok("", &e));
    }

    #[test]
    fn rows_are_selected_by_model_or_model_and_method() {
        assert!(selected(&[], "a", "k0"));
        let only = vec!["a".to_string(), "b/mtp-tree".to_string()];
        assert!(selected(&only, "a", "dspine"));
        assert!(selected(&only, "b", "mtp-tree"));
        assert!(!selected(&only, "b", "k0"));
    }
}

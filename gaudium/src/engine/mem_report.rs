//! Itemised device-memory report of a loaded engine: every planned buffer under one of the five
//! ledger categories and a named part of it, for the live record's `hbm`, `hbm_detail` and
//! `sources` fields and for the plan lines in the log.
//!
//! Every sum here is a verified ledger (`core::hbm::HbmLedgerGuard`, law
//! `hbm_ledger_add_conserved`): one for the five categories and one per named part. What stays
//! in this file is presentation: which buffer name goes under which part, and the JSON and log
//! text. The names of the parts are the ones the live protocol lists, plus finer ones.

use std::collections::BTreeMap;

use crate::core::hbm::HbmLedgerGuard;
use crate::json::quote_json;
use crate::verified::{align_up, HbmCat};

const GB: f64 = 1e9;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The live protocol's key of a category, in ledger order.
const CAT_KEYS: [&str; 5] = ["weights_gb", "kv_gb", "state_gb", "scratch_gb", "other_gb"];
/// The category names of the log lines, in ledger order.
const CAT_NAMES: [&str; 5] = ["weights", "kv", "state", "scratch", "other"];

fn cat_index(cat: HbmCat) -> usize {
    match cat {
        HbmCat::CatWeights => 0,
        HbmCat::CatKv => 1,
        HbmCat::CatCheckpoints => 2,
        HbmCat::CatScratch => 3,
        HbmCat::CatOther => 4,
    }
}

fn align256(n: u64) -> u64 {
    align_up(n.max(256), 256)
}

/// One named part of a category: its verified byte sum, how many buffers it holds and the
/// largest of them.
#[derive(Debug, Clone)]
struct Part {
    sum: HbmLedgerGuard,
    buffers: u64,
    largest: (String, u64),
    /// The part's buffers by kind (the buffer name without its layer index): bytes, count.
    kinds: BTreeMap<String, (HbmLedgerGuard, u64)>,
}

/// How many kinds of one part the live record and the log name; the rest is one "other" line.
const KINDS_SHOWN: usize = 10;

/// The itemised plan. Buffers are added as they are planned; the totals are read at the end.
#[derive(Debug, Clone)]
pub struct MemReport {
    cats: HbmLedgerGuard,
    parts: [BTreeMap<String, Part>; 5],
    sources: BTreeMap<String, String>,
}

impl Default for MemReport {
    fn default() -> Self {
        Self::new()
    }
}

impl MemReport {
    pub fn new() -> Self {
        Self {
            cats: HbmLedgerGuard::new(),
            parts: Default::default(),
            sources: BTreeMap::new(),
        }
    }

    /// Adds one buffer of exactly `bytes` (already aligned by the caller) to `part` of `cat`.
    pub fn add(&mut self, cat: HbmCat, part: &str, name: &str, bytes: u64) {
        if bytes == 0 {
            return;
        }
        self.cats.record(cat, bytes);
        let p = self.parts[cat_index(cat)]
            .entry(part.to_string())
            .or_insert_with(|| Part {
                sum: HbmLedgerGuard::new(),
                buffers: 0,
                largest: (String::new(), 0),
                kinds: BTreeMap::new(),
            });
        p.sum.record(cat, bytes);
        p.buffers += 1;
        let k = p
            .kinds
            .entry(kind_of(name).to_string())
            .or_insert_with(|| (HbmLedgerGuard::new(), 0));
        k.0.record(cat, bytes);
        k.1 += 1;
        if bytes > p.largest.1 {
            p.largest = (name.to_string(), bytes);
        }
    }

    /// Adds every buffer of a plan map (`name -> bytes`, each rounded up to the 256-byte
    /// allocation unit as the allocator does), under the part `part_of(name)` names.
    pub fn add_map(
        &mut self,
        cat: HbmCat,
        specs: &BTreeMap<String, u64>,
        part_of: impl Fn(&str) -> &'static str,
    ) {
        for (name, &bytes) in specs {
            self.add(cat, part_of(name), name, align256(bytes));
        }
    }

    /// What a part was loaded from or sized for, as a short name (never a path).
    pub fn set_source(&mut self, part: &str, text: impl Into<String>) {
        self.sources.insert(part.to_string(), text.into());
    }

    /// Bytes of the five categories `[weights, kv, state, scratch, other]`.
    pub fn totals(&self) -> [u64; 5] {
        let (w, kv, st, sc, ot) = self.cats.breakdown();
        [w, kv, st, sc, ot]
    }

    /// Bytes of everything in the report.
    pub fn total_bytes(&self) -> u64 {
        self.cats.total_bytes()
    }

    /// Bytes of one named part across the categories it appears in.
    pub fn part_bytes(&self, part: &str) -> u64 {
        self.parts
            .iter()
            .filter_map(|m| m.get(part))
            .map(|p| p.sum.total_bytes())
            .sum()
    }

    /// The parts of one category, largest first.
    fn sorted(&self, ci: usize) -> Vec<(&String, &Part)> {
        let mut v: Vec<(&String, &Part)> = self.parts[ci].iter().collect();
        v.sort_by(|a, b| {
            b.1.sum
                .total_bytes()
                .cmp(&a.1.sum.total_bytes())
                .then(a.0.cmp(b.0))
        });
        v
    }

    /// The live record's `hbm_detail` object: each category by named part, in GB (1e9 bytes)
    /// with three decimals. The parts of a category add up to the category's own printed
    /// value exactly: the largest part takes what the rounding of the others leaves.
    pub fn detail_json(&self) -> String {
        let milli = |b: u64| (b as f64 / GB * 1000.0).round() as i64;
        let totals = self.totals();
        let mut out = String::from("{");
        let mut first_cat = true;
        for (ci, key) in CAT_KEYS.iter().enumerate() {
            let parts = self.sorted(ci);
            if parts.is_empty() {
                continue;
            }
            let rest: i64 = parts[1..]
                .iter()
                .map(|(_, p)| milli(p.sum.total_bytes()))
                .sum();
            let head = (milli(totals[ci]) - rest).max(0);
            if !first_cat {
                out.push(',');
            }
            first_cat = false;
            out.push_str(&format!("\"{key}\":{{"));
            for (i, (name, p)) in parts.iter().enumerate() {
                let m = if i == 0 {
                    head
                } else {
                    milli(p.sum.total_bytes())
                };
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("{}:{}.{:03}", quote_json(name), m / 1000, m % 1000));
            }
            out.push('}');
        }
        out.push('}');
        out
    }

    /// The kinds of one part, largest first: `(kind, bytes, buffers)`. At most `KINDS_SHOWN`
    /// are named; what is left is one last entry `other` (its count = the buffers in it).
    fn kinds_of(p: &Part) -> Vec<(String, u64, u64)> {
        let mut v: Vec<(String, u64, u64)> = p
            .kinds
            .iter()
            .map(|(k, (sum, n))| (k.clone(), sum.total_bytes(), *n))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        if v.len() > KINDS_SHOWN {
            let rest = v.split_off(KINDS_SHOWN - 1);
            v.push((
                "other".to_string(),
                rest.iter().map(|x| x.1).sum(),
                rest.iter().map(|x| x.2).sum(),
            ));
        }
        v
    }

    /// The live record's `hbm_kinds` object: under each category and part, the part's buffers
    /// by kind as `[[kind, GB, buffers], ...]`, largest first. A part made of one buffer kind
    /// that carries the part's own name is left out.
    pub fn kinds_json(&self) -> String {
        let mut cats: Vec<String> = Vec::new();
        for (ci, key) in CAT_KEYS.iter().enumerate() {
            let mut parts: Vec<String> = Vec::new();
            for (name, p) in self.sorted(ci) {
                let kinds = Self::kinds_of(p);
                if kinds.len() == 1 && kinds[0].0 == *name {
                    continue;
                }
                let body: Vec<String> = kinds
                    .iter()
                    .map(|(k, b, n)| format!("[{},{:.3},{n}]", quote_json(k), *b as f64 / GB))
                    .collect();
                parts.push(format!("{}:[{}]", quote_json(name), body.join(",")));
            }
            if !parts.is_empty() {
                cats.push(format!("\"{key}\":{{{}}}", parts.join(",")));
            }
        }
        format!("{{{}}}", cats.join(","))
    }

    /// The live record's `sources` object.
    pub fn sources_json(&self) -> String {
        let body: Vec<String> = self
            .sources
            .iter()
            .map(|(k, v)| format!("{}:{}", quote_json(k), quote_json(v)))
            .collect();
        format!("{{{}}}", body.join(","))
    }

    /// Log lines of the itemised plan, in GiB: one per category, then one per part with its
    /// buffer count and largest buffer.
    pub fn log_lines(&self) -> Vec<String> {
        let gib = |b: u64| b as f64 / GIB;
        let totals = self.totals();
        let mut out = vec![format!(
            "HBM plan by part: {:.2} GiB in {} buffers",
            gib(self.total_bytes()),
            self.parts
                .iter()
                .flat_map(|m| m.values())
                .map(|p| p.buffers)
                .sum::<u64>()
        )];
        for (ci, cat) in CAT_NAMES.iter().enumerate() {
            let parts = self.sorted(ci);
            if parts.is_empty() {
                continue;
            }
            out.push(format!("  {cat} {:.2} GiB", gib(totals[ci])));
            for (name, p) in parts {
                let b = p.sum.total_bytes();
                let size = if b >= (GIB as u64) / 100 {
                    format!("{:.2} GiB", gib(b))
                } else {
                    format!("{:.2} MiB", b as f64 / (1024.0 * 1024.0))
                };
                let src = self
                    .sources
                    .get(name.as_str())
                    .map(|s| format!(" [{s}]"))
                    .unwrap_or_default();
                out.push(format!(
                    "    {name:<16} {size:>12}  {} buffer(s), largest {} {:.2} MiB{src}",
                    p.buffers,
                    p.largest.0,
                    p.largest.1 as f64 / (1024.0 * 1024.0)
                ));
                let kinds = Self::kinds_of(p);
                if kinds.len() > 1 {
                    for (k, kb, n) in kinds {
                        out.push(format!(
                            "      {k:<28} {:>10.2} MiB  x{n}",
                            kb as f64 / (1024.0 * 1024.0)
                        ));
                    }
                }
            }
        }
        out
    }
}

/// The kind of a buffer: its name without a leading layer or block index (`L11_scores` and
/// `L3_scores` are both `scores`, `V26_fc1` is `fc1`). A name without such an index is its own
/// kind.
pub fn kind_of(name: &str) -> &str {
    // `L11_scores` -> `scores`; the tile recipes' generic `L_vnat` -> `vnat`
    let name = strip_index(name, 3).unwrap_or(name);
    let name = name.strip_prefix("L_").filter(|r| !r.is_empty()).unwrap_or(name);
    // a chunk group of a tile recipe: `g12_scores` -> `scores`
    match strip_index(name, 1) {
        Some(rest) if name.starts_with('g') => rest,
        _ => name,
    }
}

/// `rest` of `<1..=max_letters letters><digits>_<rest>`, when `name` has that form.
fn strip_index(name: &str, max_letters: usize) -> Option<&str> {
    let letters = name.chars().take_while(|c| c.is_ascii_alphabetic()).count();
    let digits = name[letters..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .count();
    let rest = &name[letters + digits..];
    if (1..=max_letters).contains(&letters) && digits > 0 && rest.len() > 1 && rest.starts_with('_')
    {
        Some(&rest[1..])
    } else {
        None
    }
}

/// The part of a buffer of the speculative step's plan: the drafter's own buffers, or the
/// verify rows' activations (trunk layer buffers at the verify width).
pub fn spec_act_part(name: &str) -> &'static str {
    let drafter = name.starts_with("ds")
        || name.starts_with("df")
        || name.starts_with("mtp")
        || name.starts_with("as_")
        || (name.starts_with('D') && name[1..].starts_with(|c: char| c.is_ascii_digit()));
    if drafter {
        "drafter"
    } else {
        "verify"
    }
}

/// The part of a trunk weight or table, from its device name.
pub fn trunk_weight_part(name: &str) -> &'static str {
    match name {
        "embed_table" => "embedding",
        "W_lm" => "lm-head",
        "rope_c" | "rope_s" => "rope",
        "w_final" | "norm_ones" | "zero_slot" | "fp8_one" => "tables",
        _ => match layer_suffix(name) {
            Some("W_qg" | "W_k" | "W_v" | "qn_w" | "kn_w") => "target-attn",
            Some("W_qkvz" | "W_ba" | "conv_w" | "conv_b" | "A_log" | "dt_bias" | "norm_w") => {
                "target-deltanet"
            }
            Some("W_o") => "target-out-proj",
            Some("W_gate" | "W_up" | "W_down") => "target-mlp",
            Some("w_in" | "w_post") => "target-norms",
            Some(_) => "target-scales",
            None => "target",
        },
    }
}

/// `"W_up"` of `"L12_W_up"`: the per-layer weight name without its layer index.
fn layer_suffix(name: &str) -> Option<&str> {
    let rest = name.strip_prefix('L')?;
    let sep = rest.find('_')?;
    let (idx, tail) = rest.split_at(sep);
    if idx.is_empty() || !idx.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(&tail[1..])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> MemReport {
        let mut r = MemReport::new();
        r.add(HbmCat::CatWeights, "target-mlp", "L0_W_up", 30_000_000_512);
        r.add(HbmCat::CatWeights, "target-mlp", "L1_W_up", 1_000_000_256);
        r.add(HbmCat::CatWeights, "drafter", "ds_w", 4_530_000_128);
        r.add(HbmCat::CatKv, "kv-target", "L3_kcache", 17_179_869_184);
        r.add(HbmCat::CatScratch, "arena", "prefill attention", 13_840_000_000);
        r.add(HbmCat::CatOther, "vision", "V_scores", 12_944_000_000);
        r.add(HbmCat::CatOther, "vision", "V0_fc1", 1_059_000_064);
        r.add(HbmCat::CatOther, "workspace", "shared workspace", 268_435_456);
        r.add(HbmCat::CatOther, "empty", "nothing", 0);
        r
    }

    #[test]
    fn parts_add_up_to_their_category_and_the_categories_to_the_total() {
        let r = sample();
        let totals = r.totals();
        for (ci, total) in totals.iter().enumerate() {
            let parts: u64 = r.parts[ci].values().map(|p| p.sum.total_bytes()).sum();
            assert_eq!(parts, *total, "category {ci}");
        }
        assert_eq!(totals.iter().sum::<u64>(), r.total_bytes());
        assert_eq!(r.part_bytes("vision"), 12_944_000_000 + 1_059_000_064);
        // a zero-byte buffer adds no part
        assert!(!r.parts[4].contains_key("empty"));
    }

    #[test]
    fn detail_json_parts_sum_to_the_printed_category_value() {
        let r = sample();
        let v = crate::json::parse_json(&r.detail_json()).unwrap();
        let totals = r.totals();
        for (ci, key) in CAT_KEYS.iter().enumerate() {
            let Some(obj) = v.get(key).and_then(|o| o.as_object()) else {
                assert_eq!(totals[ci], 0, "{key} is missing");
                continue;
            };
            let sum_milli: i64 = obj
                .iter()
                .map(|(_, x)| (x.as_f64().unwrap() * 1000.0).round() as i64)
                .sum();
            let cat_milli = (totals[ci] as f64 / 1e9 * 1000.0).round() as i64;
            assert_eq!(sum_milli, cat_milli, "{key}");
        }
        let vision = v.get("other_gb").and_then(|o| o.get("vision")).unwrap();
        assert_eq!(vision.as_f64().unwrap(), 14.003);
        assert!(v.get("state_gb").is_none());
    }

    #[test]
    fn sources_and_log_lines_name_the_parts() {
        let mut r = sample();
        r.set_source("vision", "vision tower, \"8192\" patches");
        let s = crate::json::parse_json(&r.sources_json()).unwrap();
        assert_eq!(
            s.get("vision").and_then(|x| x.as_str()),
            Some("vision tower, \"8192\" patches")
        );
        let lines = r.log_lines().join("\n");
        assert!(lines.contains("other 13.29 GiB"), "{lines}");
        assert!(lines.contains("vision") && lines.contains("largest V_scores"), "{lines}");
        assert!(lines.contains("2 buffer(s)"), "{lines}");
    }

    #[test]
    fn kinds_group_buffers_across_layers_and_keep_the_part_total() {
        let mut r = MemReport::new();
        for l in 0..12 {
            r.add(HbmCat::CatOther, "verify", &format!("L{l}_scores"), 1 << 20);
            r.add(HbmCat::CatOther, "verify", &format!("L{l}_p8"), 1 << 19);
        }
        for i in 0..10 {
            r.add(HbmCat::CatOther, "verify", &format!("misc{i}x"), 1000 + i);
        }
        r.add(HbmCat::CatScratch, "arena", "arena", 4096);
        let p = &r.parts[4]["verify"];
        let kinds = MemReport::kinds_of(p);
        assert_eq!(kinds.len(), KINDS_SHOWN);
        assert_eq!(kinds[0], ("scores".to_string(), 12 << 20, 12));
        assert_eq!(kinds[1], ("p8".to_string(), 12 << 19, 12));
        assert_eq!(kinds.last().unwrap().0, "other");
        assert_eq!(kinds.iter().map(|k| k.1).sum::<u64>(), p.sum.total_bytes());
        assert_eq!(kinds.iter().map(|k| k.2).sum::<u64>(), p.buffers);
        let v = crate::json::parse_json(&r.kinds_json()).unwrap();
        let first = v
            .get("other_gb")
            .and_then(|o| o.get("verify"))
            .and_then(|a| a.as_array())
            .unwrap()[0]
            .as_array()
            .unwrap()
            .to_vec();
        assert_eq!(first[0].as_str(), Some("scores"));
        assert_eq!(first[2].as_u64(), Some(12));
        // one buffer kind named like its part adds nothing to say
        assert!(v.get("scratch_gb").is_none());
    }

    #[test]
    fn buffer_names_map_to_kinds_and_spec_parts() {
        assert_eq!(kind_of("L11_scores"), "scores");
        assert_eq!(kind_of("V26_fc1"), "fc1");
        assert_eq!(kind_of("D0_kcache"), "kcache");
        assert_eq!(kind_of("embed_table"), "embed_table");
        assert_eq!(kind_of("ds_W_lmd_f8"), "ds_W_lmd_f8");
        assert_eq!(kind_of("L_w_in"), "w_in");
        assert_eq!(kind_of("L_g12_scores"), "scores");
        assert_eq!(kind_of("L_vnat"), "vnat");
        assert_eq!(kind_of("g_x"), "g_x");
        assert_eq!(kind_of("vb_scores"), "vb_scores");
        assert_eq!(kind_of("x2_"), "x2_");
        assert_eq!(spec_act_part("ds_ctx_ring"), "drafter");
        assert_eq!(spec_act_part("D3_scores"), "drafter");
        assert_eq!(spec_act_part("L11_scores"), "verify");
        assert_eq!(spec_act_part("Dense_x"), "verify");
    }

    #[test]
    fn trunk_weight_names_map_to_parts() {
        assert_eq!(trunk_weight_part("embed_table"), "embedding");
        assert_eq!(trunk_weight_part("W_lm"), "lm-head");
        assert_eq!(trunk_weight_part("rope_s"), "rope");
        assert_eq!(trunk_weight_part("L12_W_up"), "target-mlp");
        assert_eq!(trunk_weight_part("L3_W_qg"), "target-attn");
        assert_eq!(trunk_weight_part("L0_W_qkvz"), "target-deltanet");
        assert_eq!(trunk_weight_part("L63_W_o"), "target-out-proj");
        assert_eq!(trunk_weight_part("L5_w_in"), "target-norms");
        assert_eq!(trunk_weight_part("L5_some_scale"), "target-scales");
        assert_eq!(trunk_weight_part("Lx_W_up"), "target");
    }
}

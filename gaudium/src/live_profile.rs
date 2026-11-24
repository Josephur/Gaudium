//! Level-0 live profiling for the live-stats record's optional `profile` block.
//!
//! Two host-only parts, both free of device calls:
//!
//! - [`Traffic`]: the engine's own byte model of one launch. Every graph node is charged the HBM
//!   bytes of the tensors it reads and writes (from the graph description's tensor sizes), split
//!   by data kind (weights / KV / recurrent state / activations) and by phase (verify trunk /
//!   drafts / readout / other), plus the FLOPs of its GEMMs. It is a model, not a hardware
//!   counter, and it never touches the graph or its fingerprint.
//! - [`StepTimeline`]: a rolling ~1 s window of decode launches timed on the host clock, turned
//!   into the `profile` JSON object (step time, launch-outstanding time, idle gap between
//!   launches, host split, modelled HBM GB/s by kind, modelled step split by phase) and the
//!   current request's prefill entry.
//!
//! Timing basis: this engine synchronises every decode launch, so all times are host-clock
//! intervals around the launch. `dev_ms` is the time from entering the launch call until the
//! launch's tokens are on the host, which is an upper bound of the device time of the step (it
//! includes the token read-back); the gap is the host time from one launch's tokens to the next
//! launch call, a lower bound of the device's idle time between launches. The block says so in
//! its `timing_basis` field.
//!
//! A decode loop that keeps more than one launch in flight (enqueue-ahead speculative steps) sets
//! no launch marks. Its steps are timed by the interval between consecutive launches' tokens
//! reaching the host only ([`StepSample::marked`] is `false`): the block then carries `step_ms`,
//! the HBM flow and a split over `step_ms`, and leaves out `dev_ms`, `busy_pct`, the gaps and the
//! host split, which were not measured.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::synapse::graph::GraphBuilder;

/// Gaudi 2 HBM2E bandwidth reference, GB/s (vendor spec sheet figure recorded in the project's
/// hardware notes, evidence tier B). Used only as the scale of the dashboard's HBM flow chart.
pub const HBM_PEAK_GBPS: f64 = 2450.0;

/// Data kind of a tensor: what the bytes a launch moves are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Weight = 0,
    Kv = 1,
    State = 2,
    Act = 3,
}

/// Phase of a node inside a step: what the launch spends its bytes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Verify = 0,
    Draft = 1,
    Readout = 2,
    Other = 3,
}

/// A weight table above this size read by a custom TPC kernel is indexed, not streamed, and is
/// charged [`INDEXED_TABLE_CHARGE`] bytes per launch (a deliberate cap, not a measurement).
const INDEXED_TABLE_MIN: f64 = 64.0 * 1024.0 * 1024.0;
const INDEXED_TABLE_CHARGE: f64 = 16.0 * 1024.0 * 1024.0;

pub const KIND_NAMES: [&str; 4] = ["weights", "kv", "state", "act"];
pub const PHASE_NAMES: [&str; 4] = ["verify", "drafts", "readout", "other"];

/// What a graph's passes look like, for the per-node byte rules.
#[derive(Debug, Clone, Copy)]
pub struct TrafficRules {
    /// Rows (token positions) one pass of the graph processes.
    pub rows: u64,
    /// Token positions held by one attention cache tensor of the graph (`chunk x chunks`);
    /// `0` charges a cache write at the tensor's full size.
    pub kv_positions: u64,
    /// Routed mixture-of-experts graphs: experts one row selects. A stacked expert tensor
    /// (`..EW12` / `..EW3`, expert axis last) is then charged for at most `rows x experts_per_row`
    /// of its experts (an upper bound: rows may share experts). `0`: stacked tensors are charged
    /// in full.
    pub experts_per_row: u64,
}

/// Modelled HBM bytes and GEMM FLOPs of one launch of a graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Traffic {
    /// Bytes per launch (read + written), `[phase][kind]`.
    pub bytes: [[f64; 4]; 4],
    pub read: f64,
    pub written: f64,
    /// GEMM FLOPs per launch by phase.
    pub flops: [f64; 4],
    pub nodes: u64,
}

fn has_layer_prefix(name: &str, c: char) -> bool {
    let mut it = name.chars();
    it.next() == Some(c) && it.next().is_some_and(|d| d.is_ascii_digit())
}

/// Default node-name rules: which phase of a step a node belongs to.
pub fn phase_of(node_name: &str, guid: &str) -> Phase {
    if node_name.contains("logits") || guid.contains("argmax") || guid.contains("samp_") {
        return Phase::Readout;
    }
    if node_name.starts_with("mtp_")
        || node_name.starts_with("draft_")
        || node_name.starts_with("c_")
        || node_name.starts_with("ctx_")
        || node_name.starts_with("blk_")
        || node_name.starts_with("df_")
        || node_name.starts_with("ds_")
        || node_name.starts_with("mc_")
        || node_name.starts_with("M_")
        || node_name == "next_ids"
        || has_layer_prefix(node_name, 'd')
        || has_layer_prefix(node_name, 'D')
    {
        return Phase::Draft;
    }
    if has_layer_prefix(node_name, 'L')
        || node_name.starts_with("L_")
        || node_name.starts_with("embed")
        || node_name.contains("final")
    {
        return Phase::Verify;
    }
    Phase::Other
}

/// Default tensor-name rules for caches and recurrent state; `is_weight` decides the rest.
pub fn kind_of(tensor_name: &str, is_weight: bool) -> Kind {
    if ["kcache", "vcache", "kring", "vring"]
        .iter()
        .any(|s| tensor_name.ends_with(s))
    {
        return Kind::Kv;
    }
    // (`S_base`, `rp*`: the replay form's committed state and per-row update buffers)
    for s in [
        "S_in", "S_out", "S_ckpt", "cs_in", "cs_out", "cs_ckpt", "S_base", "_rpk_in", "_rpk_out",
        "_rpd_in", "_rpd_out", "_rpg_in", "_rpg_out",
    ] {
        if tensor_name.ends_with(s) {
            return Kind::State;
        }
    }
    if is_weight {
        Kind::Weight
    } else {
        Kind::Act
    }
}

impl Traffic {
    /// Charges every node of `gb` with the bytes it moves. `is_weight` says whether a tensor
    /// name is a weight or table of the model (engine-specific naming).
    ///
    /// Rules: a `reshape` moves nothing; a gather reads only the table rows it outputs; a RoPE
    /// table is read for the pass's rows; a slotted state ring is read one slot at a time and a
    /// checkpoint ring is written one slot per row; an attention cache update writes the pass's
    /// rows (a cache read is charged in full, because a decode step reads the whole bucket); a
    /// custom TPC kernel reading a weight table above 64 MiB is charged 16 MiB (it indexes rows);
    /// workspace intermediates are not counted.
    pub fn from_graph(
        gb: &GraphBuilder,
        rules: TrafficRules,
        is_weight: &dyn Fn(&str) -> bool,
    ) -> Self {
        Self::from_graph_kinds(gb, rules, &|name| kind_of(name, is_weight(name)))
    }

    /// [`Traffic::from_graph`] for an engine whose tensor names the default rules of
    /// [`kind_of`] do not cover: `kind_fn` gives the data kind of every persistent tensor (the byte
    /// rules then follow that kind: a [`Kind::Kv`] output is charged for the pass's rows).
    pub fn from_graph_kinds(
        gb: &GraphBuilder,
        rules: TrafficRules,
        kind_fn: &dyn Fn(&str) -> Kind,
    ) -> Self {
        Self::from_graph_positions(gb, rules, kind_fn, &|_, _| None)
    }

    /// [`Traffic::from_graph_kinds`] for a graph whose attention state tensors hold different
    /// numbers of positions (fixed rings beside chunked caches): `positions_fn(name, dims)` gives
    /// the positions one [`Kind::Kv`] output holds (`None`: `rules.kv_positions`), and its write
    /// is charged for the pass's rows out of those.
    pub fn from_graph_positions(
        gb: &GraphBuilder,
        rules: TrafficRules,
        kind_fn: &dyn Fn(&str) -> Kind,
        positions_fn: &dyn Fn(&str, &[u64]) -> Option<u64>,
    ) -> Self {
        let mut tr = Self::default();
        let rows = rules.rows.max(1) as f64;
        for node in gb.node_io_views() {
            if node.guid.starts_with("reshape") {
                continue;
            }
            tr.nodes += 1;
            let ph = phase_of(node.name, node.guid) as usize;
            let gather = node.guid.starts_with("gather");
            let custom_kernel = node.guid.starts_with("gaudium_");
            let out0 = node.outputs.first().map_or(0, |t| t.bytes) as f64;
            let mut seen: Vec<usize> = Vec::with_capacity(node.inputs.len() + node.outputs.len());
            for (i, t) in node.inputs.iter().enumerate() {
                if !t.persistent || seen.contains(&t.id) {
                    continue;
                }
                seen.push(t.id);
                let mut by = t.bytes as f64;
                let last = t.dims.last().copied().unwrap_or(1).max(1) as f64;
                if gather && i == 0 {
                    by = out0;
                } else if t.name.contains("rope") {
                    by = by * rows.min(last) / last;
                } else if rules.experts_per_row > 0
                    && (t.name.ends_with("EW12") || t.name.ends_with("EW3"))
                    && t.dims.len() == 3
                {
                    by = by * (rows * rules.experts_per_row as f64).min(last) / last;
                } else if (t.name.ends_with("S_in") || t.name.ends_with("cs_in"))
                    && t.dims.len() >= 3
                    && last > 1.0
                {
                    by /= last;
                }
                let kind = kind_fn(t.name);
                if kind == Kind::Weight && custom_kernel && by > INDEXED_TABLE_MIN {
                    // a TPC kernel indexes a large table (candidate rows, codes, scores): it
                    // does not stream it the way a GEMM streams a weight matrix
                    by = INDEXED_TABLE_CHARGE;
                }
                tr.bytes[ph][kind as usize] += by;
                tr.read += by;
            }
            for t in &node.outputs {
                if !t.persistent || seen.contains(&t.id) {
                    continue;
                }
                seen.push(t.id);
                let mut by = t.bytes as f64;
                let k = kind_fn(t.name);
                let kv_pos = if k == Kind::Kv {
                    positions_fn(t.name, t.dims).unwrap_or(rules.kv_positions)
                } else {
                    0
                };
                if kv_pos > 0 {
                    let pos = kv_pos as f64;
                    by = by * rows.min(pos) / pos;
                } else if (t.name.ends_with("S_ckpt") || t.name.ends_with("cs_ckpt"))
                    && t.dims.len() >= 3
                {
                    // a checkpoint ring takes one slot per verify row, not the whole ring
                    let slots = t.dims.last().copied().unwrap_or(1).max(1) as f64;
                    by = by * rows.min(slots) / slots;
                }
                tr.bytes[ph][k as usize] += by;
                tr.written += by;
            }
            if node.guid == "gemm" || node.guid == "batch_gemm" || node.guid.starts_with("fp8_gemm")
            {
                if let (Some(a), Some(o)) = (node.inputs.first(), node.outputs.first()) {
                    let out_elems: f64 = o.dims.iter().map(|&d| d as f64).product();
                    let shared = a.dims.first().copied().unwrap_or(1) as f64;
                    tr.flops[ph] += 2.0 * out_elems * shared;
                }
            }
        }
        tr
    }

    /// Adds `w` launches of `other`.
    pub fn add(&mut self, other: &Traffic, w: f64) {
        for p in 0..4 {
            for k in 0..4 {
                self.bytes[p][k] += w * other.bytes[p][k];
            }
            self.flops[p] += w * other.flops[p];
        }
        self.read += w * other.read;
        self.written += w * other.written;
        self.nodes += (w * other.nodes as f64) as u64;
    }

    pub fn total(&self) -> f64 {
        self.read + self.written
    }

    pub fn by_phase(&self, p: usize) -> f64 {
        self.bytes[p].iter().sum()
    }

    pub fn by_kind(&self, k: usize) -> f64 {
        self.bytes.iter().map(|row| row[k]).sum()
    }

    pub fn total_flops(&self) -> f64 {
        self.flops.iter().sum()
    }
}

/// One consumed decode launch, timed on the host clock.
#[derive(Debug, Clone)]
pub struct StepSample {
    /// Host time the launch's tokens reached the host, seconds on the reporter's clock.
    pub t_s: f64,
    /// Wall time since the previous launch's tokens (token inter-arrival time).
    pub wall_ms: f64,
    /// Launch call entered -> tokens on the host.
    pub dev_ms: f64,
    /// Previous launch's tokens on the host -> this launch call entered; `None` for the first.
    pub gap_ms: Option<f64>,
    /// Host work of the enqueue (here: the launch call).
    pub enqueue_us: f64,
    /// The launch call alone.
    pub launch_us: f64,
    /// Launch call returned -> tokens on the host.
    pub wait_ms: f64,
    /// Whether the launch call was marked (`dev_ms`, `gap_ms` and the host split are measured).
    /// `false`: only `wall_ms` is a measurement, the other timing fields are unused.
    pub marked: bool,
    pub traffic: Option<Arc<Traffic>>,
}

const GAP_BINS: usize = 5;

fn gap_bin(ms: f64) -> usize {
    if ms < 0.01 {
        0
    } else if ms < 0.1 {
        1
    } else if ms < 1.0 {
        2
    } else if ms < 10.0 {
        3
    } else {
        4
    }
}

fn jnum(v: f64, prec: usize) -> String {
    if v.is_finite() {
        format!("{v:.prec$}")
    } else {
        "null".to_string()
    }
}

/// Rolling window of decode launches plus the request's prefill totals.
#[derive(Debug, Default)]
pub struct StepTimeline {
    win: VecDeque<StepSample>,
    steps: u64,
    pf_ms: f64,
    pf_bytes: f64,
    pf_flops: f64,
    pf_launches: u64,
    last_emit_s: Option<f64>,
    block: String,
}

impl StepTimeline {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Adds `launches` prefill launches moving `traffic` in total.
    pub fn prefill_launches(&mut self, traffic: &Traffic, launches: u64) {
        self.pf_bytes += traffic.total();
        self.pf_flops += traffic.total_flops();
        self.pf_launches += launches;
    }

    /// Updates the running prefill entry: `ms` since the request began.
    pub fn prefill_running(&mut self, ms: f64) {
        self.pf_ms = ms;
    }

    /// Closes the prefill entry: `ms` is the request's time to first token.
    pub fn prefill_done(&mut self, ms: f64) {
        self.pf_ms = ms;
        self.last_emit_s = None;
    }

    pub fn add(&mut self, s: StepSample) {
        let t = s.t_s;
        self.steps += 1;
        self.win.push_back(s);
        while self.win.len() > 2 && self.win.front().is_some_and(|f| f.t_s < t - 1.0) {
            self.win.pop_front();
        }
    }

    /// The `profile` JSON object over the window; rebuilt at most every 0.5 s.
    pub fn block(&mut self, now_s: f64, phase: &str) -> &str {
        self.refresh(now_s, phase);
        &self.block
    }

    /// Makes the next `refresh` rebuild the block whatever its age.
    pub fn invalidate(&mut self) {
        self.last_emit_s = None;
    }

    /// The block as last built (empty before the first `refresh`).
    pub fn block_str(&self) -> &str {
        &self.block
    }

    /// Rebuilds the block when it is at least 0.5 s old; returns whether it was rebuilt.
    pub fn refresh(&mut self, now_s: f64, phase: &str) -> bool {
        if let Some(last) = self.last_emit_s {
            if now_s - last < 0.5 && !self.block.is_empty() {
                return false;
            }
        }
        self.last_emit_s = Some(now_s);
        let (mut dev, mut wall, mut by, mut fl) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut enq, mut lau, mut wai) = (0.0f64, 0.0f64, 0.0f64);
        let (mut gap, mut gmax) = (0.0f64, 0.0f64);
        let (mut n, mut gn, mut nm) = (0u64, 0u64, 0u64);
        let mut hist = [0u64; GAP_BINS];
        let mut ph = [0.0f64; 4];
        let mut kb = [0.0f64; 4];
        for s in &self.win {
            if s.wall_ms <= 0.0 {
                continue;
            }
            n += 1;
            wall += s.wall_ms;
            if s.marked {
                nm += 1;
                dev += s.dev_ms;
                enq += s.enqueue_us;
                lau += s.launch_us;
                wai += s.wait_ms;
                if let Some(g) = s.gap_ms {
                    gap += g;
                    gmax = gmax.max(g);
                    gn += 1;
                    hist[gap_bin(g)] += 1;
                }
            }
            if let Some(tr) = &s.traffic {
                let tb = tr.total().max(1.0);
                by += tr.total();
                fl += tr.total_flops();
                // a marked launch is split over its launch-outstanding time, an unmarked one
                // over the interval between token arrivals (the only time measured for it)
                let over = if s.marked { s.dev_ms } else { s.wall_ms };
                for (p, slot) in ph.iter_mut().enumerate() {
                    *slot += over * tr.by_phase(p) / tb;
                }
                for (k, slot) in kb.iter_mut().enumerate() {
                    *slot += tr.by_kind(k);
                }
            }
        }
        // device timing is reported only when every launch of the window was marked
        let timed = n > 0 && nm == n;
        let mut j = format!("{{\"level\":0,\"phase\":\"{phase}\"");
        if n > 0 && wall > 0.0 {
            let nf = n as f64;
            j.push_str(&format!(
                ",\"window_steps\":{n},\"step_ms\":{}",
                jnum(wall / nf, 3)
            ));
            if timed {
                let busy = (dev / wall).min(1.0);
                j.push_str(&format!(
                    ",\"dev_ms\":{},\"busy_pct\":{},\"idle_pct\":{}",
                    jnum(dev / nf, 3),
                    jnum(100.0 * busy, 2),
                    jnum(100.0 * (1.0 - busy), 2),
                ));
                j.push_str(&format!(
                    ",\"gap_ms_mean\":{},\"gap_ms_max\":{},\"gap_hist\":[{},{},{},{},{}]",
                    jnum(if gn > 0 { gap / gn as f64 } else { 0.0 }, 4),
                    jnum(gmax, 4),
                    hist[0],
                    hist[1],
                    hist[2],
                    hist[3],
                    hist[4],
                ));
                j.push_str(&format!(
                    ",\"host\":{{\"enqueue_us\":{},\"launch_us\":{},\"wait_ms\":{}}}",
                    jnum(enq / nf, 1),
                    jnum(lau / nf, 1),
                    jnum(wai / nf, 3),
                ));
            }
            if by > 0.0 {
                let wall_s = wall / 1e3;
                j.push_str(&format!(
                    ",\"hbm\":{{\"gbps\":{},\"gb_per_step\":{}",
                    jnum(by / wall_s / 1e9, 1),
                    jnum(by / nf / 1e9, 4),
                ));
                if timed && dev > 0.0 {
                    j.push_str(&format!(
                        ",\"gbps_busy\":{}",
                        jnum(by / (dev / 1e3) / 1e9, 1)
                    ));
                }
                for (k, name) in KIND_NAMES.iter().enumerate() {
                    j.push_str(&format!(
                        ",\"{name}_gbps\":{}",
                        jnum(kb[k] / wall_s / 1e9, 1)
                    ));
                }
                j.push_str(&format!(",\"peak_gbps\":{}}}", jnum(HBM_PEAK_GBPS, 0)));
                j.push_str(&format!(",\"tflops\":{}", jnum(fl / wall_s / 1e12, 2)));
                j.push_str(",\"split_ms\":{");
                for (p, name) in PHASE_NAMES.iter().enumerate() {
                    if p > 0 {
                        j.push(',');
                    }
                    j.push_str(&format!("\"{name}\":{}", jnum(ph[p] / nf, 3)));
                }
                j.push_str(if timed {
                    "},\"split_basis\":\"dev_ms x share of the step's modelled HBM bytes\""
                } else {
                    "},\"split_basis\":\"step_ms x share of the step's modelled HBM bytes \
                     (device time not measured for these launches)\""
                });
            }
            j.push_str(if timed {
                ",\"timing_basis\":\"host clock: dev_ms = launch call entered to tokens on the \
                 host (upper bound of device time), gap = tokens on the host to the next launch \
                 call\""
            } else {
                ",\"timing_basis\":\"host clock: step_ms = interval between consecutive \
                 launches' tokens reaching the host; the launch calls are not marked, so device \
                 time, idle gaps and the host split are not reported\""
            });
        }
        if self.pf_ms > 0.0 {
            let s = self.pf_ms / 1e3;
            j.push_str(&format!(
                ",\"prefill\":{{\"ms\":{},\"gb\":{},\"gbps\":{},\"tflops\":{},\"launches\":{}}}",
                jnum(self.pf_ms, 1),
                jnum(self.pf_bytes / 1e9, 3),
                jnum(self.pf_bytes / s / 1e9, 1),
                jnum(self.pf_flops / s / 1e12, 2),
                self.pf_launches,
            ));
        }
        j.push('}');
        self.block = j;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;
    use crate::synapse::ffi::SynDataType;

    fn tiny_graph(t: u64, positions: u64) -> GraphBuilder {
        let mut gb = GraphBuilder::new("g_test");
        let table = gb.add_persistent_tensor("embed_table", SynDataType::Bf16, &[8, 100]);
        let ids = gb.add_persistent_tensor("ids", SynDataType::Int32, &[t]);
        let x0 = gb.add_persistent_tensor("x0", SynDataType::Bf16, &[8, t]);
        gb.op_gather_bf16("embed", table, ids, x0, 1);
        let w = gb.add_persistent_tensor("L0_W_k", SynDataType::Bf16, &[8, 16]);
        let k = gb.add_persistent_tensor("L0_k", SynDataType::Bf16, &[16, t]);
        gb.op_gemm("L0_k_gemm", x0, w, k, false, true);
        let cache = gb.add_persistent_tensor("L0_kcache", SynDataType::Bf16, &[16, positions, 1]);
        gb.add_node("gaudium_attn_prep", "L0_attn_prep", &[k], &[cache], &[]);
        let scratch = gb.add_intermediate("L0_tmp", SynDataType::Bf16, &[16, t]);
        gb.op_reshape("L0_rs", k, scratch);
        let lm = gb.add_persistent_tensor("W_lm", SynDataType::Bf16, &[16, 100]);
        let logits = gb.add_persistent_tensor("logits", SynDataType::Bf16, &[100, t]);
        gb.op_gemm("logits_gemm", k, lm, logits, false, true);
        gb
    }

    fn is_w(n: &str) -> bool {
        matches!(n, "embed_table" | "L0_W_k" | "W_lm")
    }

    #[test]
    fn traffic_charges_nodes_by_kind_and_phase() {
        let t = 2u64;
        let gb = tiny_graph(t, 64);
        let tr = Traffic::from_graph(
            &gb,
            TrafficRules {
                rows: t,
                kv_positions: 64,
                experts_per_row: 0,
            },
            &is_w,
        );
        // reshape is not a node of the model
        assert_eq!(tr.nodes, 4);
        // gather reads only the rows it outputs (8 x T x 2 B), not the 1.6 kB table
        let gather_rd = (8 * t * 2) as f64;
        // weights: gathered rows + W_k (8 x 16 x 2) in verify, W_lm (16 x 100 x 2) in readout
        assert_eq!(
            tr.bytes[Phase::Verify as usize][Kind::Weight as usize],
            gather_rd + 256.0
        );
        assert_eq!(
            tr.bytes[Phase::Readout as usize][Kind::Weight as usize],
            3200.0
        );
        // a cache update writes the pass's rows: 16 x 64 x 2 B x T / 64
        assert_eq!(tr.by_kind(Kind::Kv as usize), (16 * 2 * t) as f64);
        assert_eq!(tr.by_kind(Kind::State as usize), 0.0);
        assert!((tr.total() - (tr.read + tr.written)).abs() < 1e-9);
        let sum: f64 = (0..4).map(|k| tr.by_kind(k)).sum();
        assert!((sum - tr.total()).abs() < 1e-9);
        // GEMM FLOPs = 2 x output elements x shared dimension
        assert_eq!(
            tr.flops[Phase::Verify as usize],
            2.0 * (16 * t) as f64 * 8.0
        );
        assert_eq!(
            tr.flops[Phase::Readout as usize],
            2.0 * (100 * t) as f64 * 16.0
        );
    }

    #[test]
    fn name_rules_classify_phases_and_kinds() {
        assert_eq!(phase_of("L12_qk", "batch_gemm"), Phase::Verify);
        assert_eq!(phase_of("L_attn_prep", "gaudium_attn_prep"), Phase::Verify);
        assert_eq!(phase_of("logits_gemm", "gemm"), Phase::Readout);
        assert_eq!(phase_of("argmax", "argmax_fwd_bf16"), Phase::Readout);
        assert_eq!(phase_of("mtp_L0_gate_gemm", "gemm"), Phase::Draft);
        assert_eq!(phase_of("d3_qk", "batch_gemm"), Phase::Draft);
        assert_eq!(phase_of("copy_ids", "memcpy"), Phase::Other);
        assert_eq!(kind_of("L3_vcache", false), Kind::Kv);
        assert_eq!(kind_of("L3_kring", false), Kind::Kv);
        assert_eq!(kind_of("L3_S_out", false), Kind::State);
        assert_eq!(kind_of("L3_cs_ckpt", false), Kind::State);
        assert_eq!(kind_of("L3_W_up", true), Kind::Weight);
        assert_eq!(kind_of("L3_act", false), Kind::Act);
    }

    #[test]
    fn timeline_block_reports_window_rates_and_prefill() {
        let gb = tiny_graph(1, 64);
        let tr = Arc::new(Traffic::from_graph(
            &gb,
            TrafficRules {
                rows: 1,
                kv_positions: 64,
                experts_per_row: 0,
            },
            &is_w,
        ));
        let mut tl = StepTimeline::default();
        let mut pf = Traffic::default();
        pf.add(&tr, 10.0);
        tl.prefill_launches(&pf, 10);
        tl.prefill_done(200.0);
        for i in 0..20u32 {
            tl.add(StepSample {
                t_s: 10.0 + f64::from(i) * 0.010,
                wall_ms: 10.0,
                dev_ms: 9.0,
                gap_ms: if i == 0 { None } else { Some(1.0) },
                enqueue_us: 50.0,
                launch_us: 40.0,
                wait_ms: 8.9,
                marked: true,
                traffic: Some(Arc::clone(&tr)),
            });
        }
        let v = json::parse_json(tl.block(10.2, "decode")).expect("valid profile JSON");
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap();
        assert_eq!(v.get("window_steps").and_then(|x| x.as_u64()), Some(20));
        assert!((f("step_ms") - 10.0).abs() < 1e-6);
        assert!((f("dev_ms") - 9.0).abs() < 1e-6);
        assert!((f("busy_pct") - 90.0).abs() < 1e-6);
        assert!((f("idle_pct") - 10.0).abs() < 1e-6);
        assert!((f("gap_ms_mean") - 1.0).abs() < 1e-6);
        let hbm = v.get("hbm").unwrap();
        let gbps = hbm.get("gbps").and_then(|x| x.as_f64()).unwrap();
        // bytes per step / 10 ms wall, in GB/s (1 decimal in the record)
        assert!((gbps - tr.total() / 0.010 / 1e9).abs() < 0.06);
        let split = v.get("split_ms").unwrap();
        let s: f64 = PHASE_NAMES
            .iter()
            .map(|p| split.get(p).and_then(|x| x.as_f64()).unwrap())
            .sum();
        assert!((s - 9.0).abs() < 0.01, "split sums to dev_ms, got {s}");
        let p = v.get("prefill").unwrap();
        assert_eq!(p.get("launches").and_then(|x| x.as_u64()), Some(10));
        assert!((p.get("ms").and_then(|x| x.as_f64()).unwrap() - 200.0).abs() < 1e-6);
        assert!(v.get("timing_basis").and_then(|x| x.as_str()).is_some());
    }

    #[test]
    fn unmarked_launches_report_step_time_and_flow_but_no_device_timing() {
        let gb = tiny_graph(1, 64);
        let tr = Arc::new(Traffic::from_graph(
            &gb,
            TrafficRules {
                rows: 1,
                kv_positions: 64,
                experts_per_row: 0,
            },
            &is_w,
        ));
        let mut tl = StepTimeline::default();
        for i in 0..10u32 {
            tl.add(StepSample {
                t_s: 5.0 + f64::from(i) * 0.030,
                wall_ms: 30.0,
                dev_ms: 0.0,
                gap_ms: None,
                enqueue_us: 0.0,
                launch_us: 0.0,
                wait_ms: 0.0,
                marked: false,
                traffic: Some(Arc::clone(&tr)),
            });
        }
        let v = json::parse_json(tl.block(5.3, "decode")).expect("valid profile JSON");
        assert!((v.get("step_ms").and_then(|x| x.as_f64()).unwrap() - 30.0).abs() < 1e-6);
        for absent in [
            "dev_ms",
            "busy_pct",
            "idle_pct",
            "gap_ms_mean",
            "gap_hist",
            "host",
        ] {
            assert!(v.get(absent).is_none(), "{absent} was not measured");
        }
        let hbm = v.get("hbm").unwrap();
        assert!(hbm.get("gbps_busy").is_none());
        let gbps = hbm.get("gbps").and_then(|x| x.as_f64()).unwrap();
        assert!((gbps - tr.total() / 0.030 / 1e9).abs() < 0.06);
        let split = v.get("split_ms").unwrap();
        let s: f64 = PHASE_NAMES
            .iter()
            .map(|p| split.get(p).and_then(|x| x.as_f64()).unwrap())
            .sum();
        assert!((s - 30.0).abs() < 0.01, "split sums to step_ms, got {s}");
        assert!(v
            .get("split_basis")
            .and_then(|x| x.as_str())
            .unwrap()
            .starts_with("step_ms"));
    }

    #[test]
    fn checkpoint_rings_are_charged_one_slot_per_row() {
        let mut gb = GraphBuilder::new("g_ckpt");
        let x = gb.add_persistent_tensor("L0_x", SynDataType::Bf16, &[8, 4]);
        let s_in = gb.add_persistent_tensor("L0_S_in", SynDataType::Float, &[8, 8, 16]);
        let ck = gb.add_persistent_tensor("L0_S_ckpt", SynDataType::Float, &[8, 8, 16]);
        gb.add_node("gaudium_gdn_f32", "L0_gdn", &[x, s_in], &[ck], &[]);
        let tr = Traffic::from_graph(
            &gb,
            TrafficRules {
                rows: 4,
                kv_positions: 64,
                experts_per_row: 0,
            },
            &|_| false,
        );
        let slot = (8 * 8 * 4) as f64;
        // one slot read, four of sixteen slots written
        assert_eq!(tr.by_kind(Kind::State as usize), slot + 4.0 * slot);
    }

    #[test]
    fn routed_expert_stacks_are_charged_for_the_selected_experts() {
        let mut gb = GraphBuilder::new("g_moe");
        let st = gb.add_persistent_tensor("L0_EW3", SynDataType::Bf16, &[4, 8, 16]);
        let part = gb.add_intermediate("L0_EW3_p0", SynDataType::Bf16, &[4, 8, 16]);
        gb.add_node("split", "L0_EW3_split", &[st], &[part], &[]);
        let rules = |rows, experts_per_row| TrafficRules {
            rows,
            kv_positions: 64,
            experts_per_row,
        };
        let full = (4 * 8 * 16 * 2) as f64;
        let w =
            |r: TrafficRules| Traffic::from_graph(&gb, r, &|_| true).by_kind(Kind::Weight as usize);
        assert_eq!(w(rules(1, 0)), full);
        assert_eq!(w(rules(1, 4)), full / 4.0, "4 of 16 experts for one row");
        assert_eq!(w(rules(8, 4)), full, "never more than the stack");
    }

    #[test]
    fn a_table_indexed_by_a_custom_kernel_is_capped_and_a_gemm_streams_it() {
        let mut gb = GraphBuilder::new("g_tab");
        // 5120 x 16384 BF16 = 160 MiB
        let w = gb.add_persistent_tensor("W_lm", SynDataType::Bf16, &[5120, 16384]);
        let x = gb.add_persistent_tensor("blk_x", SynDataType::Bf16, &[5120, 1]);
        let cand = gb.add_persistent_tensor("blk_cand", SynDataType::Int32, &[16, 1]);
        gb.add_node("gaudium_ds_base", "blk_base_k", &[x, w], &[cand], &[]);
        let r = TrafficRules {
            rows: 1,
            kv_positions: 64,
            experts_per_row: 0,
        };
        let tr = Traffic::from_graph(&gb, r, &|n| n == "W_lm");
        assert_eq!(tr.by_kind(Kind::Weight as usize), 16.0 * 1024.0 * 1024.0);
        let y = gb.add_persistent_tensor("blk_logits", SynDataType::Bf16, &[16384, 1]);
        gb.op_gemm("blk_logits_gemm", x, w, y, false, true);
        let tr2 = Traffic::from_graph(&gb, r, &|n| n == "W_lm");
        let full = (5120u64 * 16384 * 2) as f64;
        assert_eq!(
            tr2.by_kind(Kind::Weight as usize),
            16.0 * 1024.0 * 1024.0 + full
        );
    }

    #[test]
    fn drafter_nodes_count_as_drafts() {
        for n in [
            "ctx_f_gemm",
            "blk_walk",
            "D3_qk",
            "df_ids",
            "next_ids",
            "ds_cb",
            "M_qg_gemm",
        ] {
            assert_eq!(phase_of(n, "gemm"), Phase::Draft, "{n}");
        }
        assert_eq!(phase_of("blk_logits_gemm", "gemm"), Phase::Readout);
        assert_eq!(phase_of("accept", "gaudium_accept_t"), Phase::Other);
    }

    #[test]
    fn timeline_without_traffic_keeps_timing_and_omits_the_byte_model() {
        let mut tl = StepTimeline::default();
        for i in 0..3u32 {
            tl.add(StepSample {
                t_s: f64::from(i) * 0.02,
                wall_ms: 20.0,
                dev_ms: 19.0,
                gap_ms: Some(0.5),
                enqueue_us: 30.0,
                launch_us: 30.0,
                wait_ms: 18.9,
                marked: true,
                traffic: None,
            });
        }
        let v = json::parse_json(tl.block(1.0, "decode")).unwrap();
        assert!(v.get("busy_pct").is_some());
        assert!(v.get("hbm").is_none());
        assert!(v.get("split_ms").is_none());
        assert!(v.get("prefill").is_none());
    }
}

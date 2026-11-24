//! Inkling 2 geometry: the schedule of layer applications, block-state counts, ring sizes and the
//! shape guard. Every decision delegates to `verified/inkling2.bend`.

use crate::model_spec::{Inkling2Spec, ModelSpec};
use crate::verified::{
    ik_app_layer, ik_app_msgs, ik_app_pass, ik_apps, ik_attn_geom_valid, ik_buf_clobbers,
    ik_buf_reusable, ik_counts_valid, ik_dims_valid, ik_eff_loop_end, ik_eff_loop_start,
    ik_hist_rows, ik_layer_looped, ik_loop_index, ik_msg_states, ik_opens_block,
    ik_ring_slots_within, ik_schedule_valid, ik_states_attn, ik_states_final, ik_states_mlp,
    ik_vocab_valid,
};

/// Block states one residual mix of the family's kernels reads beside the partial state.
pub const IK_MAX_STATES: u64 = 4;

/// Slots of the largest K / V ring the softmax kernel stages as one chunk
/// (`kernels/tpc/ik_softmax_gaudi2.c`: a chunk of at most 4096 keys in its local memory).
pub const IK_MAX_RING_SLOTS: u64 = 4096;

/// Layer applications of one token a model may run (the schedule and the plans hold one entry each).
pub const IK_MAX_APPS: u64 = 4096;

/// Longest sliding window a model may declare: a window must fit beside a pass in the largest ring.
pub const IK_MAX_WINDOW: u64 = IK_MAX_RING_SLOTS;

/// Most query heads a model may declare (the projection widths are multiples of the head count).
pub const IK_MAX_HEADS: u64 = 1024;

/// One application of a layer's weights to the rows of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IkApp {
    /// Index of the application in execution order.
    pub index: usize,
    /// The layer whose weights it runs.
    pub layer: usize,
    /// Pass over the looped span (0: the first pass, or a layer outside the span).
    pub pass: usize,
    /// The layer attends a sliding window (a ring); otherwise the whole context (a cache).
    pub sliding: bool,
    /// The layer opens a block: its input becomes a block state.
    pub opens_block: bool,
    /// Block states its attention-input mix reads (0: the first layer, which has no mix).
    pub ns_attn: u64,
    /// Block states its MLP-input mix reads.
    pub ns_mlp: u64,
    /// The application receives messages of the previous pass.
    pub msgs: bool,
    /// Position of the layer inside the looped span (meaningful when the layer is looped).
    pub loop_index: usize,
    /// The layer is part of the looped span.
    pub looped: bool,
}

/// Checks an Inkling 2 spec against the shapes and schedules the kernels cover.
pub fn validate_inkling2(spec: &ModelSpec, ik: &Inkling2Spec) -> Result<(), String> {
    if !ik_dims_valid(
        spec.hidden,
        spec.n_heads,
        spec.n_kv_heads,
        spec.head_dim,
        ik.d_rel,
        spec.n_experts,
        spec.experts_per_tok,
        spec.moe_intermediate,
        ik.conv_taps,
    ) {
        return Err(format!(
            "Inkling 2 shape outside the kernels' range: hidden {} heads {}q/{}kv x {} d_rel {} experts {} (top {}) x {} taps {}",
            spec.hidden,
            spec.n_heads,
            spec.n_kv_heads,
            spec.head_dim,
            ik.d_rel,
            spec.n_experts,
            spec.experts_per_tok,
            spec.moe_intermediate,
            ik.conv_taps
        ));
    }
    let (ls, le) = span_of(ik);
    // before anything multiplies or adds them: the application count, the window, the heads
    if !ik_counts_valid(
        spec.n_layers,
        ls,
        le,
        ik.passes,
        ik.window,
        spec.n_heads,
        IK_MAX_APPS,
        IK_MAX_WINDOW,
        IK_MAX_HEADS,
    ) {
        return Err(format!(
            "Inkling 2 counts outside the engine's limits: {} layers, layers {}..{} x {} passes (at most {IK_MAX_APPS} layer applications), sliding window {} (at most {IK_MAX_WINDOW}), {} query heads (at most {IK_MAX_HEADS})",
            spec.n_layers, ik.loop_start, ik.loop_end, ik.passes, ik.window, spec.n_heads
        ));
    }
    if !ik_schedule_valid(spec.n_layers, ls, le, ik.passes, ik.block, IK_MAX_STATES) {
        return Err(format!(
            "Inkling 2 schedule outside the kernels' range: {} layers, blocks of {}, layers {}..{} x {} passes (at most {IK_MAX_STATES} block states per mix)",
            spec.n_layers, ik.block, ik.loop_start, ik.loop_end, ik.passes
        ));
    }
    if spec.layer_types.len() as u64 != spec.n_layers {
        return Err("Inkling 2 spec: one attention kind per layer is required".into());
    }
    if !ik_attn_geom_valid(ik.window, ik.rel_extent) {
        return Err(format!(
            "Inkling 2 spec: sliding window {} and bias extent {} (the extent must be within the window)",
            ik.window, ik.rel_extent
        ));
    }
    if !ik_vocab_valid(ik.unpadded_vocab, spec.vocab) {
        return Err(format!(
            "Inkling 2 spec: {} emitted tokens of a vocabulary of {}",
            ik.unpadded_vocab, spec.vocab
        ));
    }
    Ok(())
}

/// The loop span of a spec (an empty span behind the first block for a model without a loop).
fn span_of(ik: &Inkling2Spec) -> (u64, u64) {
    (
        ik_eff_loop_start(ik.passes, ik.loop_start, ik.block),
        ik_eff_loop_end(ik.passes, ik.loop_end, ik.block),
    )
}

/// The layer applications of one token, in execution order.
pub fn schedule(spec: &ModelSpec, ik: &Inkling2Spec) -> Vec<IkApp> {
    let (ls, le) = span_of(ik);
    let n = ik_apps(spec.n_layers, ls, le, ik.passes);
    (0..n)
        .map(|a| {
            let layer = ik_app_layer(a, ls, le, ik.passes);
            let pass = ik_app_pass(a, ls, le, ik.passes);
            let looped = ik_layer_looped(layer, ls, le);
            IkApp {
                index: a as usize,
                layer: layer as usize,
                pass: pass as usize,
                sliding: spec.layer_types[layer as usize] == "sliding_attention",
                opens_block: ik_opens_block(layer, ik.block),
                ns_attn: ik_states_attn(layer, ik.block),
                ns_mlp: ik_states_mlp(layer, ik.block),
                msgs: ik_app_msgs(a, ls, le, ik.passes),
                loop_index: if looped { ik_loop_index(layer, ls) as usize } else { 0 },
                looped,
            }
        })
        .collect()
}

/// Block states a message is mixed from (beside the previous pass's last output).
pub fn msg_states(ik: &Inkling2Spec) -> u64 {
    let (ls, le) = span_of(ik);
    // a model of one pass has an empty span and no messages (law `ik_one_pass_no_messages`)
    ik_msg_states(ls, le, ik.block)
}

/// Block states the final mix reads.
pub fn final_states(spec: &ModelSpec, ik: &Inkling2Spec) -> u64 {
    ik_states_final(spec.n_layers, ik.block)
}

/// Slots of a sliding layer's K / V ring for passes of at most `rows` rows.
pub fn ring_slots(ik: &Inkling2Spec, rows: u64) -> Result<u64, String> {
    match ik_ring_slots_within(ik.window, rows, IK_MAX_RING_SLOTS) {
        0 => Err(format!(
            "no K / V ring of at most {IK_MAX_RING_SLOTS} slots (one softmax chunk) holds a window of {} beside a pass of {rows} rows",
            ik.window
        )),
        n => Ok(n),
    }
}

/// Rows of a short convolution's history ring for passes of at most `rows` rows.
pub fn hist_rows(ik: &Inkling2Spec, rows: u64) -> Result<u64, String> {
    match ik_hist_rows(rows, ik.conv_taps) {
        0 => Err(format!(
            "no history ring holds a pass of {rows} rows and {} taps",
            ik.conv_taps
        )),
        n => Ok(n),
    }
}

/// Where an application's inputs come from. A *source* `s` is the output of application `s - 1`;
/// source 0 is the embedding of the pass's rows.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IkWire {
    /// The layer's input.
    pub x_in: usize,
    /// Block states of the attention-input mix (`ns_attn` of them).
    pub attn_states: Vec<usize>,
    /// Block states of the MLP-input mix (`ns_mlp`: a layer that opens a block has added its input).
    pub mlp_states: Vec<usize>,
    /// Block states of the previous pass a message is mixed from, and that pass's last output.
    pub msg_states: Vec<usize>,
    pub msg_prev: Option<usize>,
}

/// The wiring of every application and the block states of the final mix, by following the
/// passes: a later pass over the looped span restarts from the input and the block states the
/// span was entered with, and reads the previous pass's block states and output as messages.
pub fn wiring(apps: &[IkApp]) -> (Vec<IkWire>, Vec<usize>) {
    let mut wires = Vec::with_capacity(apps.len());
    let mut stack: Vec<usize> = Vec::new();
    let mut cur = 0usize;
    let mut entry: Option<(usize, Vec<usize>)> = None;
    let mut prev: Option<(Vec<usize>, usize)> = None;
    for app in apps {
        if app.looped && app.loop_index == 0 {
            match (&entry, app.pass) {
                (None, _) | (_, 0) => entry = Some((cur, stack.clone())),
                (Some((e_in, e_stack)), _) => {
                    // the span's first layer opened a block: the states behind it are the pass's own
                    prev = Some((stack[e_stack.len() + 1..].to_vec(), cur));
                    cur = *e_in;
                    stack = e_stack.clone();
                }
            }
        }
        let mut w = IkWire {
            x_in: cur,
            attn_states: stack.clone(),
            ..IkWire::default()
        };
        if app.opens_block {
            stack.push(cur);
        }
        w.mlp_states = stack.clone();
        if app.msgs {
            if let Some((st, out)) = &prev {
                w.msg_states = st.clone();
                w.msg_prev = Some(*out);
            }
        }
        wires.push(w);
        cur = app.index + 1;
    }
    (wires, stack)
}

/// Assigns every source a buffer so that no application's output overwrites a source that is
/// still read (by a later application or by the final mix). Returns the buffer of each source
/// (`apps.len() + 1` entries) and the number of buffers.
pub fn source_buffers(wires: &[IkWire], final_states: &[usize]) -> (Vec<usize>, usize) {
    source_buffers_keep(wires, final_states, &[])
}

/// [`source_buffers`] with `keep`: sources that must also stay in their buffers until the tile
/// ends (a drafter reads them after the last application).
pub fn source_buffers_keep(
    wires: &[IkWire],
    final_states: &[usize],
    keep: &[usize],
) -> (Vec<usize>, usize) {
    let n_src = wires.len() + 1;
    // the last application (index) that reads each source; the final mix reads after every layer
    let mut last_use = vec![0usize; n_src];
    for (a, w) in wires.iter().enumerate() {
        let mut touch = |s: usize| last_use[s] = last_use[s].max(a);
        touch(w.x_in);
        w.attn_states.iter().for_each(|&s| touch(s));
        w.mlp_states.iter().for_each(|&s| touch(s));
        w.msg_states.iter().for_each(|&s| touch(s));
        if let Some(s) = w.msg_prev {
            touch(s);
        }
    }
    for &s in final_states.iter().chain(keep.iter().filter(|&&s| s < n_src)) {
        last_use[s] = wires.len();
    }
    last_use[n_src - 1] = wires.len();
    let mut buf_of = vec![usize::MAX; n_src];
    let mut held: Vec<Option<usize>> = Vec::new(); // buffer -> the source it holds
    buf_of[0] = 0;
    held.push(Some(0));
    for a in 0..wires.len() {
        // application a writes source a + 1: a buffer whose source no application from a on reads
        let free = held
            .iter()
            .position(|h| h.is_none_or(|s| ik_buf_reusable(last_use[s] as u64, a as u64)));
        let b = match free {
            Some(b) => b,
            None => {
                held.push(None);
                held.len() - 1
            }
        };
        held[b] = Some(a + 1);
        buf_of[a + 1] = b;
    }
    // no application's output lands on a source that it or a later application still reads
    for a in 0..wires.len() {
        for (s, &lu) in last_use.iter().enumerate().take(a + 1) {
            let (bs, bo) = (buf_of[s] as u64, buf_of[a + 1] as u64);
            assert!(
                !ik_buf_clobbers(bs, bo, s as u64, a as u64, lu as u64),
                "Inkling 2 row buffers: application {a} would overwrite source {s} (last read by {lu})"
            );
        }
    }
    (buf_of, held.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn mini_json() -> String {
        let local: Vec<String> = (0..24u32)
            .filter(|l| (l + 1) % 6 != 0)
            .map(|l| l.to_string())
            .collect();
        format!(
            r#"{{"architectures":["Inkling2LoopForConditionalGeneration"],"model_type":"inkling2_loop","eos_token_id":200006,
            "text_config":{{"model_type":"inkling2_loop_text","hidden_size":2304,"num_hidden_layers":24,"vocab_size":201024,
            "num_attention_heads":24,"num_key_value_heads":8,"head_dim":128,"d_rel":16,"rel_extent":1024,"log_scaling_n_floor":8192,
            "log_scaling_alpha":0.1,"rms_norm_eps":1e-06,"qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,
            "local_layer_ids":[{}],"sconv_kernel_size":4,"unpadded_vocab_size":200058,"final_logit_softcapping":30.0,
            "sliding_window_size":1024,"n_layers_per_attn_res_block":6,"n_routed_experts":128,"num_experts_per_tok":8,
            "intermediate_size":1152,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":8.0,
            "loop_config":{{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64}}}},
            "vision_config":{{"hidden_size":1024}}}}"#,
            local.join(",")
        )
    }

    #[test]
    fn mini_schedule_runs_the_span_three_times() {
        let spec = ModelSpec::from_config_json(&mini_json()).unwrap();
        spec.validate().unwrap();
        // the checkpoint's quantisation note does not make it a block-FP8 model
        let noted = mini_json().replace(
            r#""vision_config""#,
            r#""quantization_config":{"quant_method":"modelopt"},"vision_config""#,
        );
        assert!(!ModelSpec::from_config_json(&noted).unwrap().fp8_quantized);
        let ik = spec.inkling2.as_ref().unwrap();
        let apps = schedule(&spec, ik);
        assert_eq!(apps.len(), 48);
        let layers: Vec<usize> = apps.iter().map(|a| a.layer).collect();
        let mut want: Vec<usize> = (0..6).collect();
        for _ in 0..3 {
            want.extend(6..18);
        }
        want.extend(18..24);
        assert_eq!(layers, want);
        // the first layer has no mix; a layer that opens a block adds its input before the MLP mix
        assert_eq!((apps[0].ns_attn, apps[0].ns_mlp, apps[0].opens_block), (0, 1, true));
        assert_eq!((apps[5].ns_attn, apps[5].ns_mlp, apps[5].sliding), (1, 1, false));
        assert_eq!((apps[6].ns_attn, apps[6].ns_mlp, apps[6].msgs), (1, 2, false));
        // the second pass restarts from the states before the span and receives messages
        assert_eq!((apps[18].layer, apps[18].pass, apps[18].ns_attn, apps[18].msgs), (6, 1, 1, true));
        assert_eq!((apps[41].layer, apps[41].pass, apps[41].loop_index), (17, 2, 11));
        assert_eq!((apps[42].layer, apps[42].ns_attn, apps[42].ns_mlp, apps[42].msgs), (18, 3, 4, false));
        assert_eq!(final_states(&spec, ik), 4);
        assert_eq!(msg_states(ik), 1);
        assert_eq!(apps.iter().filter(|a| !a.sliding).count(), 8);
    }

    #[test]
    fn rings_hold_the_window_beside_the_widest_pass() {
        let spec = ModelSpec::from_config_json(&mini_json()).unwrap();
        let ik = spec.inkling2.as_ref().unwrap();
        assert_eq!(ring_slots(ik, 1).unwrap(), 2048);
        assert_eq!(ring_slots(ik, 256).unwrap(), 2048);
        assert_eq!(ring_slots(ik, 1026).unwrap(), 4096);
        assert_eq!(hist_rows(ik, 1).unwrap(), 128);
        assert_eq!(hist_rows(ik, 256).unwrap(), 512);
    }

    #[test]
    fn shapes_outside_the_kernels_are_refused() {
        let bad = mini_json().replace("\"head_dim\":128", "\"head_dim\":64");
        let spec = ModelSpec::from_config_json(&bad).unwrap();
        assert!(spec.validate().is_err());
        let bad = mini_json().replace("\"core_start\":6", "\"core_start\":5");
        let spec = ModelSpec::from_config_json(&bad).unwrap();
        assert!(spec.validate().is_err());
    }
    #[test]
    fn kept_sources_survive_the_tile() {
        let spec = ModelSpec::from_config_json(&mini_json()).unwrap();
        let ik = spec.inkling2.clone().unwrap();
        let (w, fin) = wiring(&schedule(&spec, &ik));
        let keep = [5usize, 23, 41];
        let (buf, n) = source_buffers_keep(&w, &fin, &keep);
        assert!(n <= 12, "{n} buffers");
        for (a, _) in w.iter().enumerate() {
            for &s in &keep {
                if s <= a {
                    assert_ne!(buf[s], buf[a + 1], "application {a} clobbers kept source {s}");
                }
            }
        }
    }

    #[test]
    fn wiring_restarts_each_pass_and_carries_messages() {
        let spec = ModelSpec::from_config_json(&mini_json()).unwrap();
        let ik = spec.inkling2.as_ref().unwrap();
        let apps = schedule(&spec, ik);
        let (w, fin) = wiring(&apps);
        // first pass: layer 6 reads the output of layer 5 (source 6) and opens block 1 with it
        assert_eq!((w[6].x_in, w[6].attn_states.clone(), w[6].mlp_states.clone()), (6, vec![0], vec![0, 6]));
        assert_eq!(w[12].mlp_states, vec![0, 6, 12]);
        assert!(w[6].msg_prev.is_none());
        // second pass: restarts from source 6, messages from the first pass's block (source 12) and output (18)
        assert_eq!((w[18].x_in, w[18].attn_states.clone()), (6, vec![0]));
        assert_eq!((w[18].msg_states.clone(), w[18].msg_prev), (vec![12], Some(18)));
        assert_eq!(w[24].mlp_states, vec![0, 6, 24]);
        // third pass: messages from the second pass (its block state is source 24, its output 30)
        assert_eq!((w[30].x_in, w[30].msg_states.clone(), w[30].msg_prev), (6, vec![24], Some(30)));
        // behind the span: the third pass's states, then block 3 opened by layer 18
        assert_eq!((w[42].x_in, w[42].attn_states.clone(), w[42].mlp_states.clone()), (42, vec![0, 6, 36], vec![0, 6, 36, 42]));
        assert_eq!(fin, vec![0, 6, 36, 42]);
        for (a, wire) in w.iter().enumerate() {
            assert_eq!(wire.attn_states.len() as u64, apps[a].ns_attn, "application {a}");
            assert_eq!(wire.mlp_states.len() as u64, apps[a].ns_mlp, "application {a}");
        }
        // no application's output buffer holds a source that is still read
        let (buf, n) = source_buffers(&w, &fin);
        assert!(n <= 8, "{n} buffers");
        for (a, wire) in w.iter().enumerate() {
            let out = buf[a + 1];
            let mut reads = vec![wire.x_in];
            reads.extend(&wire.attn_states);
            reads.extend(&wire.mlp_states);
            reads.extend(&wire.msg_states);
            reads.extend(wire.msg_prev);
            for s in reads {
                assert_ne!(buf[s], out, "application {a} would overwrite source {s}");
            }
            // and every source a later application reads is still in its buffer
            for (b, later) in w.iter().enumerate().skip(a + 1) {
                let mut lr = vec![later.x_in];
                lr.extend(&later.attn_states);
                lr.extend(&later.mlp_states);
                lr.extend(&later.msg_states);
                lr.extend(later.msg_prev);
                for s in lr {
                    if s <= a {
                        assert_ne!(buf[s], out, "application {a} clobbers source {s} read by {b}");
                    }
                }
            }
            for &s in &fin {
                if s <= a {
                    assert_ne!(buf[s], out, "application {a} clobbers final state {s}");
                }
            }
        }
    }
}

/// Today's constants and formulas of `verified/inkling2.bend` that its laws leave open on purpose
/// (they bound these values but do not fix them).
#[cfg(test)]
mod pinned_values {
    use super::*;
    use crate::verified::{
        ik_apps, ik_bias_off, ik_bias_src, ik_bias_width, ik_f8_row_ok, ik_f8_row_scales,
        ik_gate_off, ik_heads_width, ik_is_pow2, ik_kx_width, ik_nib_row_bytes, ik_nib_row_scales,
        ik_pow2_rows, ik_qx_width, ik_rel_off, ik_ring_slots, ik_tile_last_row, ik_tile_rows,
        ik_tile_take,
    };

    #[test]
    fn one_pass_runs_an_empty_span_behind_the_first_block() {
        assert_eq!((ik_eff_loop_start(1, 6, 6), ik_eff_loop_end(1, 18, 6)), (6, 6));
        assert_eq!((ik_eff_loop_start(0, 6, 6), ik_eff_loop_end(0, 18, 6)), (6, 6));
        assert_eq!((ik_eff_loop_start(3, 6, 6), ik_eff_loop_end(3, 18, 6)), (6, 18));
        assert_eq!(ik_msg_states(6, 6, 6), 0);
        assert_eq!(ik_msg_states(6, 18, 6), 1);
        assert_eq!(ik_loop_index(17, 6), 11);
        assert!(ik_layer_looped(6, 6, 18) && !ik_layer_looped(18, 6, 18) && !ik_layer_looped(5, 6, 18));
    }

    #[test]
    fn rings_are_the_smallest_power_of_two_of_at_least_128() {
        assert_eq!(ik_ring_slots(1024, 1), 2048);
        assert_eq!(ik_ring_slots(1024, 1026), 4096);
        assert_eq!(ik_ring_slots(32, 1), 128);
        assert_eq!(ik_pow2_rows(1), 128);
        assert_eq!(ik_pow2_rows(129), 256);
        // the search ends at 2^47 rows: beyond it no ring exists (0)
        assert_eq!(ik_pow2_rows(1 << 47), 1 << 47);
        assert_eq!(ik_pow2_rows((1 << 47) + 1), 0);
        for x in [1u64, 2, 128, 4096, 1 << 40, 1 << 63] {
            assert!(ik_is_pow2(x), "{x}");
        }
        for x in [0u64, 3, 96, 4097, (1 << 40) + 1] {
            assert!(!ik_is_pow2(x), "{x}");
        }
        // one softmax chunk holds at most 4096 keys: a wider window has no ring
        assert_eq!(ik_ring_slots_within(1024, 256, IK_MAX_RING_SLOTS), 2048);
        assert_eq!(ik_ring_slots_within(4096, 1, IK_MAX_RING_SLOTS), 0);
        let spec = ModelSpec::from_config_json(&tests::mini_json()).unwrap();
        let ik = spec.inkling2.as_ref().unwrap();
        assert_eq!(ring_slots(ik, 256).unwrap(), 2048);
        assert!(ring_slots(ik, 3100).is_err());
    }

    #[test]
    fn bias_table_is_reversed_and_padded_by_one_block() {
        assert_eq!(ik_bias_off(1024), 1087);
        assert_eq!(ik_bias_width(1024), 1152);
        assert_eq!(ik_bias_src(1024, 1087), 0);
        assert_eq!(ik_bias_src(1024, 64), 1023);
        assert_eq!(ik_bias_src(1024, 63), 1024);
        assert_eq!(ik_bias_src(1024, 1100), 1024);
    }

    #[test]
    fn projection_layout_and_four_bit_rows() {
        assert_eq!((ik_gate_off(24), ik_rel_off(24)), (3072, 6144));
        assert_eq!(ik_qx_width(24, 16), 6528);
        assert_eq!(ik_kx_width(8, 4), 1152);
        assert_eq!(ik_kx_width(8, 0), 1024);
        assert_eq!((ik_nib_row_bytes(2304), ik_nib_row_scales(2304)), (1152, 144));
        assert_eq!(ik_nib_row_bytes(24), 0);
        assert_eq!(ik_nib_row_bytes(0), 0);
        assert!(ik_f8_row_ok(2304) && ik_f8_row_scales(2304) == 72);
        assert!(!ik_f8_row_ok(48) && !ik_f8_row_ok(0));
        assert_eq!(ik_heads_width(24), 3072);
    }

    #[test]
    fn prefill_tiles_pad_to_a_power_of_two_of_at_least_8() {
        assert_eq!(ik_tile_rows(5, 256, 1000), 8);
        assert_eq!(ik_tile_rows(200, 256, 1000), 256);
        assert_eq!(ik_tile_rows(200, 256, 210), 200);
        assert_eq!(ik_tile_rows(100, 64, 1000), 100);
        assert_eq!((ik_tile_take(10, 256), ik_tile_take(1000, 256)), (10, 256));
        assert_eq!(ik_tile_last_row(5), 4);
    }

    #[test]
    fn guard_limits_of_todays_kernels() {
        // hidden, heads q/kv, head dim, relative states, experts, top k, expert width, taps
        assert!(ik_dims_valid(2304, 24, 8, 128, 16, 128, 8, 1152, 4));
        assert!(!ik_dims_valid(2304, 24, 8, 64, 16, 128, 8, 1152, 4));
        assert!(!ik_dims_valid(2304, 24, 8, 128, 8, 128, 8, 1152, 4));
        assert!(!ik_dims_valid(2304, 20, 4, 128, 16, 128, 8, 1152, 4));
        assert!(!ik_dims_valid(2304, 24, 8, 128, 16, 640, 8, 1152, 4));
        assert!(!ik_dims_valid(2304, 24, 8, 128, 16, 128, 13, 1152, 4));
        assert!(!ik_dims_valid(2304, 24, 8, 128, 16, 128, 8, 1152, 17));
        assert!(!ik_dims_valid(2300, 24, 8, 128, 16, 128, 8, 1152, 4));
        assert!(ik_attn_geom_valid(1024, 1024) && !ik_attn_geom_valid(1024, 1025));
        assert!(!ik_attn_geom_valid(0, 0) && !ik_attn_geom_valid(1024, 0));
        assert!(ik_vocab_valid(200058, 201024) && !ik_vocab_valid(201025, 201024));
        assert!(ik_buf_reusable(3, 4) && !ik_buf_reusable(4, 4));
        assert!(ik_buf_clobbers(2, 2, 3, 4, 4) && !ik_buf_clobbers(2, 1, 3, 4, 4));
    }

    #[test]
    fn counts_guard_refuses_counts_that_would_overflow_the_sizing() {
        let ok = |n, ls, le, p, w, nh| {
            ik_counts_valid(n, ls, le, p, w, nh, IK_MAX_APPS, IK_MAX_WINDOW, IK_MAX_HEADS)
        };
        // the publisher's mini model: 24 layers, layers 6..18 run 3 times
        assert!(ok(24, 6, 18, 3, 1024, 24));
        assert_eq!(ik_apps(24, 6, 18, 3), 48);
        // exactly at the application limit, and one pass past it
        assert!(ok(16, 0, 16, 256, 1024, 24) && ik_apps(16, 0, 16, 256) == IK_MAX_APPS);
        assert!(!ok(16, 0, 16, 257, 1024, 24));
        // a pass count from a file that would overflow passes x span, and an empty span
        assert!(!ok(24, 6, 18, u64::MAX, 1024, 24));
        assert!(ok(24, 4, 4, u64::MAX, 1024, 24));
        assert!(!ok(IK_MAX_APPS + 1, 4, 4, 1, 1024, 24));
        assert!(!ok(24, 6, 18, 3, u64::MAX, 24) && !ok(24, 6, 18, 3, 1024, u64::MAX));
        assert!(ok(24, 6, 18, 3, IK_MAX_WINDOW, IK_MAX_HEADS));
    }
}

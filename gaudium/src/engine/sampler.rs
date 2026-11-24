//! On-card sampler subgraph: replaces the readout argmax of a recipe for sampled requests.
//!
//! Sampling is Gumbel-max over an exactly searched support (temperature, then top-k, top-p,
//! min-p), with noise that is a pure function of `(seed, n, token id)`, `n` = the sequence index
//! of the token being sampled. A row therefore draws the same token whenever it is evaluated on
//! the same prefix, whatever recipe, row or step it lands in: a speculative step whose verify
//! rows sample at their own positions emits exactly the tokens of the non-speculative run with
//! the same seed, and the acceptance test stays "draft equals the row's token". The math and the
//! bit-exact CPU twin are in `crate::sampler`.
//!
//! Why it is a chain of small TPC nodes (measured on Gaudi 2 with the project's legacy engine,
//! see the sampling note of the legacy tree): a vocabulary-wide sort costs milliseconds, and a
//! TPC cannot branch on a vector lane it has just computed. The thresholds are therefore found
//! by a 16-ary search in the 16-bit order-key space of BF16 (4 rounds per phase, exact), every
//! round as two nodes: `part` (all TPCs, per 1024-token block: counts or masses at 16 test
//! thresholds) and `red` (one TPC per row: sums the blocks and stores the next bracket, which
//! the following node reads with scalar loads). `bmax` stores every block's maximum so later
//! nodes skip the blocks below their lowest threshold.
//!
//! ```text
//! logits {V, T} -> bmax {NB, R} -> init {R} -> 8 x [part {NB, R} -> red {R}] -> gumbel {NB, R}
//!               -> final {R} -> tokens {1, T}
//! ```
//!
//! Request parameters are device tensors (`samp_f` f32 `{8}`, `samp_i` i32 `{8}`), so no sampler
//! setting compiles anything. Greedy requests never run these graphs: they keep the vendor
//! argmax recipes, unchanged.

use crate::synapse::ffi::SynDataType;
use crate::synapse::graph::{GraphBuilder, TensorId};
use crate::synapse::ops::u32_params;

/// Kernel ELFs of the sampler nodes.
pub const SAMPLER_ELFS: &[&str] = &[
    "GAUDIUM_SBMAX_ELF",
    "GAUDIUM_SINIT_ELF",
    "GAUDIUM_SPART_ELF",
    "GAUDIUM_SRED_ELF",
    "GAUDIUM_SGUMBEL_ELF",
    "GAUDIUM_SFINAL_ELF",
];

/// Kernel ELF of the coupled DSpine walk (shares the sampler's noise code).
pub const CWALK_ELF: &str = "GAUDIUM_DSCWALK_ELF";

/// Request sampler parameters, float half: `[1/T, 20.5 T, top_p, T ln(min_p), ..]`.
pub const SAMP_F: &str = "samp_f";
/// Request sampler parameters, integer half: `[top_k, seed_lo, seed_hi, flags, ..]`.
pub const SAMP_I: &str = "samp_i";
/// First logits row a sampler node reads (always 0 here: every row of a readout samples).
pub const SAMP_RSEL: &str = "samp_rsel";
/// Sequence index of the first token of a request (the prefill readout samples at it).
pub const SAMP_POS0: &str = "samp_pos0";

/// Vocabulary block of the sampler kernels (8 BF16 vectors).
pub const SAMP_BLOCK: u64 = 1024;
/// Search rounds per recipe (4 for top-k, 4 for top-p).
const SEARCH_ROUNDS: u32 = 8;

/// Device tensors of the sampler that the engine owns and writes per request.
pub fn is_sampler_param(name: &str) -> bool {
    matches!(name, SAMP_F | SAMP_I | SAMP_RSEL | SAMP_POS0)
}

/// Bytes of each engine-owned sampler tensor.
pub fn sampler_param_specs() -> [(&'static str, u64); 4] {
    [(SAMP_F, 32), (SAMP_I, 32), (SAMP_RSEL, 4), (SAMP_POS0, 4)]
}

/// The sampler kernels walk the vocabulary in 128-lane vectors.
pub fn vocab_supported(vocab: u64) -> bool {
    vocab > 0 && vocab.is_multiple_of(128)
}

/// Where the rows of a sampled readout sit in the sequence.
#[derive(Debug, Clone, Copy)]
pub struct SamplerAt {
    /// `{1}` i32: row `r` samples the token at sequence index `base[0] + off + r`.
    pub base: TensorId,
    pub off: u32,
    /// 0 = target row (verify), 1 = draft row.
    pub role: u32,
}

/// Appends the sampler over all `t` rows of `logits {vocab, t}` and returns `tokens {1, t}` named
/// `name` (a persistent tensor, like the argmax output it replaces).
pub fn append_sampler(
    gb: &mut GraphBuilder,
    logits: TensorId,
    vocab: u64,
    t: u64,
    at: &SamplerAt,
    name: &str,
) -> TensorId {
    append_sampler_as(gb, logits, vocab, t, at, name, SAMP_F, SAMP_I, None)
}

/// [`append_sampler`] reading the request parameters from the tensors named `samp_f` / `samp_i`
/// (one request's parameters per readout, e.g. one per lane of a batched step) and writing the
/// tokens into `out` when given (otherwise a new persistent tensor named `name`).
#[allow(clippy::too_many_arguments)]
pub fn append_sampler_as(
    gb: &mut GraphBuilder,
    logits: TensorId,
    vocab: u64,
    t: u64,
    at: &SamplerAt,
    name: &str,
    samp_f: &str,
    samp_i: &str,
    out: Option<TensorId>,
) -> TensorId {
    let f32s = SynDataType::Float;
    let nb = vocab.div_ceil(SAMP_BLOCK);
    let spf = gb.get_or_add_persistent_tensor(samp_f, f32s, &[8]);
    let spi = gb.get_or_add_persistent_tensor(samp_i, SynDataType::Int32, &[8]);
    let rsel = gb.get_or_add_persistent_tensor(SAMP_RSEL, SynDataType::Int32, &[1]);

    let bm = gb.add_intermediate(format!("{name}_sbm"), f32s, &[64, nb, t]);
    gb.add_node(
        "gaudium_samp_bmax",
        format!("{name}_sbmax"),
        &[logits, rsel],
        &[bm],
        &[],
    );
    let mut st = gb.add_intermediate(format!("{name}_sst"), f32s, &[64, 8, t]);
    gb.add_node(
        "gaudium_samp_init",
        format!("{name}_sinit"),
        &[bm],
        &[st],
        &[],
    );
    for it in 0..SEARCH_ROUNDS {
        let prm = u32_params(&[it, at.role]);
        let part = gb.add_intermediate(format!("{name}_spart{it}"), f32s, &[64, nb, t]);
        gb.add_node(
            "gaudium_samp_part",
            format!("{name}_spart_n{it}"),
            &[logits, rsel, spf, spi, st, bm],
            &[part],
            &prm,
        );
        let st2 = gb.add_intermediate(format!("{name}_sst{}", it + 1), f32s, &[64, 8, t]);
        gb.add_node(
            "gaudium_samp_red",
            format!("{name}_sred_n{it}"),
            &[part, spf, spi, st],
            &[st2],
            &prm,
        );
        st = st2;
    }
    let gp = gb.add_intermediate(format!("{name}_sgp"), f32s, &[64, 2, nb, t]);
    gb.add_node(
        "gaudium_samp_gumbel",
        format!("{name}_sgumbel"),
        &[logits, rsel, at.base, spf, spi, st, bm],
        &[gp],
        &u32_params(&[at.off, at.role, 1]),
    );
    let tok = out.unwrap_or_else(|| gb.add_persistent_tensor(name, SynDataType::Int32, &[1, t]));
    gb.add_node(
        "gaudium_samp_final",
        format!("{name}_sfinal"),
        &[gp, rsel],
        &[tok],
        &[],
    );
    tok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_subgraph_has_the_kernel_contract_shapes() {
        let mut gb = GraphBuilder::new("g_test_samp");
        let (v, t) = (248_320u64, 16u64);
        let logits = gb.add_persistent_tensor("logits", SynDataType::Bf16, &[v, t]);
        let cursor = gb.add_persistent_tensor("cursor", SynDataType::Int32, &[1]);
        let at = SamplerAt {
            base: cursor,
            off: 1,
            role: 0,
        };
        let tok = append_sampler(&mut gb, logits, v, t, &at, "tokens");
        assert_eq!(gb.tensor_name(tok), "tokens");
        let specs = gb.persistent_tensor_specs();
        let dims = |n: &str| specs.iter().find(|s| s.0 == n).map(|s| s.2.clone());
        assert_eq!(dims("tokens"), Some(vec![1, 16]));
        assert_eq!(dims(SAMP_F), Some(vec![8]));
        assert_eq!(dims(SAMP_I), Some(vec![8]));
        assert_eq!(dims(SAMP_RSEL), Some(vec![1]));
        // the search state and the per-block partials stay in the recipe workspace
        assert!(!specs.iter().any(|s| s.0.contains("_spart")));
        let nodes = gb.node_io_views();
        let count = |g: &str| nodes.iter().filter(|n| n.guid == g).count();
        assert_eq!(count("gaudium_samp_bmax"), 1);
        assert_eq!(count("gaudium_samp_init"), 1);
        assert_eq!(count("gaudium_samp_part"), 8);
        assert_eq!(count("gaudium_samp_red"), 8);
        assert_eq!(count("gaudium_samp_gumbel"), 1);
        assert_eq!(count("gaudium_samp_final"), 1);
        // 243 blocks of 1024 tokens (the last one half full), 64 lanes
        let bmax = nodes
            .iter()
            .find(|n| n.guid == "gaudium_samp_bmax")
            .unwrap();
        assert_eq!(bmax.outputs[0].dims, &[64, 243, 16]);
        let gum = nodes
            .iter()
            .find(|n| n.guid == "gaudium_samp_gumbel")
            .unwrap();
        assert_eq!(gum.outputs[0].dims, &[64, 2, 243, 16]);
        assert_eq!(gum.inputs.len(), 7);
        assert!(vocab_supported(v) && vocab_supported(151_936) && !vocab_supported(100_000));
        assert!(is_sampler_param("samp_f") && !is_sampler_param("tokens"));
    }
}

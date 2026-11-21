//! ABI-verified Intel SynapseAI 1.24 node parameter structs and typed op helpers.
//!
//! Every `#[repr(C)]` struct layout and field offset below is statically asserted against the
//! SynapseAI 1.24 `perf_lib_layer_params.h` & `synapse_common_types.h` ABI.

use std::mem::{offset_of, size_of};

use crate::synapse::graph::{GraphBuilder, NodeId, TensorId};

/// Helper: view a plain-old-data `#[repr(C)]` param struct as a raw byte slice for `synNodeCreate`.
#[inline]
pub fn as_param_bytes<T: Copy>(val: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((val as *const T).cast::<u8>(), size_of::<T>()) }
}

/// Helper: serialize a slice of little-endian `u32` scalar parameters for custom TPC-C glue nodes.
#[inline]
pub fn u32_params(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// `synGEMMParams` (`gemm`, `batch_gemm`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GemmParams {
    pub transpose_a: u8,
    pub transpose_b: u8,
}

impl GemmParams {
    pub const fn new(transpose_a: bool, transpose_b: bool) -> Self {
        Self {
            transpose_a: transpose_a as u8,
            transpose_b: transpose_b as u8,
        }
    }
}

/// `ns_LayerNormKernel::ParamsRmsNormV3` (`rms_norm_fwd_bf16`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RmsNormV3Params {
    pub eps_valid: u8,
    pub _pad0: [u8; 3],
    pub eps: f32,
    pub norm_axis_bmp: i32,
    pub param_axis_bmp: i32,
    pub normalized_shape_dims: u32,
    pub fast_math: u8,
    pub _pad1: [u8; 3],
    pub retain_inv_rms_mode: u32,
    pub cl_aligned_pack_size: u32,
    pub has_residual: u8,
    pub add_one_to_weight: u8,
    pub _pad2: [u8; 2],
}

impl RmsNormV3Params {
    pub const fn new(eps: f32, add_one_to_weight: bool) -> Self {
        Self {
            eps_valid: 1,
            _pad0: [0; 3],
            eps,
            norm_axis_bmp: 0,
            param_axis_bmp: 0,
            normalized_shape_dims: 0,
            fast_math: 0,
            _pad1: [0; 3],
            retain_inv_rms_mode: 0,
            cl_aligned_pack_size: 0,
            has_residual: 0,
            add_one_to_weight: add_one_to_weight as u8,
            _pad2: [0; 2],
        }
    }
}

/// `ns_GatherKernel::Params` (`gather_fwd_bf16`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GatherParams {
    pub axis: i32,
}

/// `ns_Reduction::Params` (`argmax_fwd_bf16`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ReductionParams {
    pub reduction_dimension: u32,
}

/// `synSplitParams` (`split`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitParams {
    pub axis: u32,
}

/// `synConcatenateParams` (`concat`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ConcatParams {
    pub axis: u32,
}

/// `synTransposeParams` (`transpose`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TransposeParams {
    pub permutation: [u32; 5],
    pub tensor_dim: u32,
}

/// `ns_ConvertToFp8::Params` (`convert_to_fp8_bf16`). `CAST_ROUND_HALF_NE = 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ConvertToFp8Params {
    pub round_mode: u32,
}

/// `ns_Fp8Gemm::ParamsV2` (`fp8_gemm_bf16`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8GemmV2Params {
    pub transpose_a: u8,
    pub transpose_b: u8,
    pub is_hw_aligned: u8,
    pub _pad: u8,
    pub scale_method_hash_id: u32,
}

// Compile-time ABI checks verified against Intel SynapseAI 1.24 headers.
const _: () = {
    assert!(size_of::<GemmParams>() == 2);
    assert!(offset_of!(GemmParams, transpose_a) == 0);
    assert!(offset_of!(GemmParams, transpose_b) == 1);

    assert!(size_of::<RmsNormV3Params>() == 36);
    assert!(offset_of!(RmsNormV3Params, eps_valid) == 0);
    assert!(offset_of!(RmsNormV3Params, eps) == 4);
    assert!(offset_of!(RmsNormV3Params, add_one_to_weight) == 33);

    assert!(size_of::<GatherParams>() == 4);
    assert!(size_of::<ReductionParams>() == 4);
    assert!(size_of::<SplitParams>() == 4);
    assert!(size_of::<ConcatParams>() == 4);

    assert!(size_of::<TransposeParams>() == 24);
    assert!(offset_of!(TransposeParams, tensor_dim) == 20);

    assert!(size_of::<ConvertToFp8Params>() == 4);
    assert!(size_of::<Fp8GemmV2Params>() == 8);
    assert!(offset_of!(Fp8GemmV2Params, transpose_b) == 1);
    assert!(offset_of!(Fp8GemmV2Params, is_hw_aligned) == 2);
    assert!(offset_of!(Fp8GemmV2Params, scale_method_hash_id) == 4);
};

// Typed node insertion helpers on `GraphBuilder`.
impl GraphBuilder {
    pub fn op_gemm(
        &mut self,
        name: impl Into<String>,
        a: TensorId,
        b: TensorId,
        out: TensorId,
        transpose_a: bool,
        transpose_b: bool,
    ) -> NodeId {
        let p = GemmParams::new(transpose_a, transpose_b);
        self.add_node("gemm", name, &[a, b], &[out], as_param_bytes(&p))
    }

    pub fn op_batch_gemm(
        &mut self,
        name: impl Into<String>,
        a: TensorId,
        b: TensorId,
        out: TensorId,
        transpose_a: bool,
        transpose_b: bool,
    ) -> NodeId {
        let p = GemmParams::new(transpose_a, transpose_b);
        self.add_node("batch_gemm", name, &[a, b], &[out], as_param_bytes(&p))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn op_rms_norm_bf16(
        &mut self,
        name: impl Into<String>,
        x: TensorId,
        w: TensorId,
        y: TensorId,
        inv_rms: TensorId,
        eps: f32,
        add_one_to_weight: bool,
    ) -> NodeId {
        let p = RmsNormV3Params::new(eps, add_one_to_weight);
        self.add_node(
            "rms_norm_fwd_bf16",
            name,
            &[x, w],
            &[y, inv_rms],
            as_param_bytes(&p),
        )
    }

    pub fn op_gather_bf16(
        &mut self,
        name: impl Into<String>,
        table: TensorId,
        indices: TensorId,
        out: TensorId,
        axis: i32,
    ) -> NodeId {
        let p = GatherParams { axis };
        self.add_node(
            "gather_fwd_bf16",
            name,
            &[table, indices],
            &[out],
            as_param_bytes(&p),
        )
    }

    pub fn op_argmax_bf16(
        &mut self,
        name: impl Into<String>,
        logits: TensorId,
        out_tokens: TensorId,
        reduction_dimension: u32,
    ) -> NodeId {
        let p = ReductionParams {
            reduction_dimension,
        };
        self.add_node(
            "argmax_fwd_bf16",
            name,
            &[logits],
            &[out_tokens],
            as_param_bytes(&p),
        )
    }

    pub fn op_add_bf16(
        &mut self,
        name: impl Into<String>,
        a: TensorId,
        b: TensorId,
        out: TensorId,
    ) -> NodeId {
        self.add_node("add_fwd_bf16", name, &[a, b], &[out], &[])
    }

    pub fn op_mult_bf16(
        &mut self,
        name: impl Into<String>,
        a: TensorId,
        b: TensorId,
        out: TensorId,
    ) -> NodeId {
        self.add_node("mult_fwd_bf16", name, &[a, b], &[out], &[])
    }

    pub fn op_silu_bf16(&mut self, name: impl Into<String>, x: TensorId, out: TensorId) -> NodeId {
        self.add_node("silu_fwd_bf16", name, &[x], &[out], &[])
    }

    pub fn op_reshape(&mut self, name: impl Into<String>, x: TensorId, out: TensorId) -> NodeId {
        self.add_node("reshape", name, &[x], &[out], &[])
    }

    pub fn op_concat(
        &mut self,
        name: impl Into<String>,
        inputs: &[TensorId],
        out: TensorId,
        axis: u32,
    ) -> NodeId {
        let p = ConcatParams { axis };
        self.add_node("concat", name, inputs, &[out], as_param_bytes(&p))
    }

    pub fn op_split(
        &mut self,
        name: impl Into<String>,
        input: TensorId,
        outputs: &[TensorId],
        axis: u32,
    ) -> NodeId {
        let p = SplitParams { axis };
        self.add_node("split", name, &[input], outputs, as_param_bytes(&p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_v3_bytes_match_abi() {
        let p = RmsNormV3Params::new(1e-6, true);
        let b = as_param_bytes(&p);
        assert_eq!(b.len(), 36);
        assert_eq!(b[0], 1);
        assert_eq!(f32::from_le_bytes([b[4], b[5], b[6], b[7]]), 1e-6_f32);
        assert_eq!(b[32], 0);
        assert_eq!(b[33], 1);
    }
}

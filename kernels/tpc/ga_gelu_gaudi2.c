// gaudium_ga_gelu: the Gemma 4 MLP activation and its product in one node (docs/notes/engine-gemma4.md).
//   m = bf16( bf16(gelu_tanh(g)) * u ),   gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))
// (transformers Gemma4TextMLP: act_fn(gate_proj(x)) * up_proj(x), both in bf16: two roundings). The vendor gelu_fwd_bf16 is a
// low-precision form (1.3e-2 relative against the formula, measured for the vision tower, src/vision/vit.hpp); this is the formula in f32,
// which is also how the reference computes it on bf16 tensors (every bf16 input checked against torch 2.15 CPU: f32 arithmetic, one
// rounding; tests/ref/gemma4/bf16_ops.py, docs/notes/engine-gemma4.md). Far below zero the reference's 1 + tanh cancels in f32 (|gelu| < 1e-6 there); this form keeps its relative precision.
//  in0 g bf16 {F, T}   in1 u bf16 {F, T}   out0 m bf16 {F, T}.   F a multiple of 256.   index space {F / 256, T}
#define RBF(a, b) ({ float128 _p; _p.v1 = (a); _p.v2 = (b); v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(_p)); })
// e^d for d <= 0: the Cody-Waite + degree-6 polynomial of gaudium_attn_softmax_chunk (max 1 ulp on [-87, 0])
#define EXPD(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f); \
    float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0); \
    float64 _r = v_f32_mac_b(_z, -0.693359375f, _d, 0, 1, 0); _r = v_f32_mac_b(_z, 2.12194440e-4f, _r, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.0013836704893037677f, (float64)0.008374780416488647f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.04166822507977486f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.1666642129421234f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.49999991059303284f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
// 1 / a with one Newton step on the table reciprocal
#define RECIP(a_) ({ float64 _a = (a_); float64 _q = v_reciprocal_f32(_a); _q + _q * (1.0f - _a * _q); })
// gelu_tanh(x) = 0.5 x (1 + tanh(y)) = x sigmoid(2 y), y = kBeta (x + kKappa x^3) in the reference's f32 operation order (torch
// GeluKernelImpl). sigmoid(2 y) from t = e^(-2 |y|): 1 / (1 + t) for y >= 0, t / (1 + t) below. The vendor v_tanh_f32 is a table
// approximation: with it 13 % of the bf16 activations were one step off the reference (tests/sim/sim_gemma4.cpp).
#define GELU(x_) ({ float64 _x = (x_); float64 _y = (_x + _x * _x * _x * 0.044715f) * 0.7978845608028654f; \
    float64 _t = EXPD(v_f32_abs_b(_y, 0, 0, 1, 0) * -2.0f); float64 _q = RECIP(_t + 1.0f); \
    _x * v_f32_sel_geq_f32_b(_y, 0.0f, _q, _t * _q, 0, 0, 1, 0); })

void main(tensor g, tensor u, tensor m)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        int5 c = {b * 256, t, 0, 0, 0};
        bfloat128 gb0 = v_bf16_ld_tnsr_b(c, g), ub0 = v_bf16_ld_tnsr_b(c, u); c[0] += 128;
        bfloat128 gb1 = v_bf16_ld_tnsr_b(c, g), ub1 = v_bf16_ld_tnsr_b(c, u);
        float128 g0 = v_convert_bf16_to_f32_all_b(gb0), g1 = v_convert_bf16_to_f32_all_b(gb1);
        float128 u0 = v_convert_bf16_to_f32_all_b(ub0), u1 = v_convert_bf16_to_f32_all_b(ub1);
        float128 a0 = RBF(GELU(g0.v1), GELU(g0.v2)), a1 = RBF(GELU(g1.v1), GELU(g1.v2));
        float128 o0; o0.v1 = a0.v1 * u0.v1; o0.v2 = a0.v2 * u0.v2;
        float128 o1; o1.v1 = a1.v1 * u1.v1; o1.v2 = a1.v2 * u1.v2;
        int5 oc = {b * 256, t, 0, 0, 0};
        v_bf16_st_tnsr(oc, m, v_convert_f32_to_bf16_all_b(o0)); oc[0] += 128;
        v_bf16_st_tnsr(oc, m, v_convert_f32_to_bf16_all_b(o1));
    }
}

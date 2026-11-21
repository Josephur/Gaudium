// gaudium_ga_softcap: the Gemma 4 final logit soft cap (docs/notes/engine-gemma4.md).
//   y = bf16( bf16(tanh( bf16(x / cap) )) * cap )      (transformers Gemma4ForCausalLM: logits / cap; tanh; * cap, each a bf16 op)
// tanh and the roundings are monotone, so the largest logit stays a largest one; but the cap MERGES neighbouring bf16 logits into exact
// ties, and the reference's argmax then takes the lowest id among them. The readout (argmax, top-k, the sampler) therefore reads the
// capped logits. x / cap is the f32 quotient (one Newton correction of x * (1 / cap), as s_div_fast_f32), so it rounds like the division.
//  in0 x bf16 {V, T}   out0 y bf16 {V, T}.   V a multiple of 256.   scalar: cap_bits (the f32 bits of cap).   index space {V / 256, T}
#define RBF(a, b) ({ float128 _p; _p.v1 = (a); _p.v2 = (b); v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(_p)); })
#define QUOT(x_) ({ float64 _x = (x_); float64 _q = _x * rc; float64 _e = v_f32_mac_b(_q, ncap, _x, 0, 1, 0); v_f32_mac_b(_e, rc, _q, 0, 1, 0); })
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
#define RECIP(a_) ({ float64 _a = (a_); float64 _q = v_reciprocal_f32(_a); _q + _q * (1.0f - _a * _q); })
// tanh(d) to f32 accuracy: (1 - t) / (1 + t) with t = e^(-2 |d|) for |d| >= 1/64, the odd series d (1 - d^2 / 3 + 2 d^4 / 15) below
// (there 1 - t would cancel); the sign of d. The vendor v_tanh_f32 is a table approximation: with it 4 % of the capped logits were one
// bf16 step off the reference, and the cap's ties decide the argmax (tests/sim/sim_gemma4.cpp).
#define TANH(d_) ({ float64 _d = (d_); float64 _a = v_f32_abs_b(_d, 0, 0, 1, 0); float64 _t = EXPD(_a * -2.0f); \
    float64 _big = (1.0f - _t) * RECIP(_t + 1.0f); float64 _d2 = _a * _a; \
    float64 _small = _a * ((_d2 * 0.13333333333f - 0.33333333333f) * _d2 + 1.0f); \
    float64 _m = v_f32_sel_less_f32_b(_a, 0.015625f, _small, _big, 0, 0, 1, 0); \
    v_f32_sel_less_f32_b(_d, 0.0f, _m * -1.0f, _m, 0, 0, 1, 0); })

void main(tensor x, tensor y, int cap_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const float cap = *(float*)&cap_bits;
    const float64 rc = 1.0f / cap, ncap = -cap;
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        int5 c = {b * 256, t, 0, 0, 0};
        bfloat128 xb0 = v_bf16_ld_tnsr_b(c, x); c[0] += 128;
        bfloat128 xb1 = v_bf16_ld_tnsr_b(c, x);
        float128 x0 = v_convert_bf16_to_f32_all_b(xb0), x1 = v_convert_bf16_to_f32_all_b(xb1);
        float128 d0 = RBF(QUOT(x0.v1), QUOT(x0.v2)), d1 = RBF(QUOT(x1.v1), QUOT(x1.v2));
        float128 h0 = RBF(TANH(d0.v1), TANH(d0.v2)), h1 = RBF(TANH(d1.v1), TANH(d1.v2));
        float128 o0; o0.v1 = h0.v1 * cap; o0.v2 = h0.v2 * cap;
        float128 o1; o1.v1 = h1.v1 * cap; o1.v2 = h1.v2 * cap;
        int5 oc = {b * 256, t, 0, 0, 0};
        v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o0)); oc[0] += 128;
        v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(o1));
    }
}

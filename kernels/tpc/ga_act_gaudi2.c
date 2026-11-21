// gaudium_ga_act: the two activations of a Gemma 4 audio encoder layer, the formula in f32 from bf16 inputs
// (docs/notes/engine-gemma4.md, section 14):
//   mode 0 (SiLU):  y = bf16( x sigmoid(x) )                 x bf16 {C, T}  -> y bf16 {C, T}       one rounding (torch silu on a bf16 tensor)
//   mode 1 (GLU):   y = bf16( a bf16(sigmoid(b)) )           x bf16 {2 C, T}: a = rows 0 .. C - 1, b = rows C .. 2 C - 1 -> y bf16 {C, T}
//                   two roundings (torch glu: a * sigmoid(b), two bf16 operations).   C a multiple of 128
// sigmoid(x) from t = e^(-|x|): 1 / (1 + t) for x >= 0, t / (1 + t) below (no overflow, full relative precision on both sides).
//  index space {ceil(C / 128), T} over the output: a last partial block loads padding beyond C and stores nothing there.   scalar: mode
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
#define RECIP(a_) ({ float64 _a = (a_); float64 _q = v_reciprocal_f32(_a); _q + _q * (1.0f - _a * _q); })
#define SIG(x_) ({ float64 _x = (x_); float64 _t = EXPD(v_f32_abs_b(_x, 0, 0, 1, 0) * -1.0f); float64 _q = RECIP(_t + 1.0f); \
    v_f32_sel_geq_f32_b(_x, 0.0f, _q, _t * _q, 0, 0, 1, 0); })

void main(tensor x, tensor y, int mode)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int C = get_dim_size(y, 0);
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t)
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
        int5 c = {b * 128, t, 0, 0, 0};
        const float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x));
        float128 oo;
        if (mode == 0) { oo.v1 = a.v1 * SIG(a.v1); oo.v2 = a.v2 * SIG(a.v2); }
        else {
            int5 gc = {C + b * 128, t, 0, 0, 0};
            const float128 g = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, x));
            const float128 s = RBF(SIG(g.v1), SIG(g.v2));
            oo.v1 = a.v1 * s.v1; oo.v2 = a.v2 * s.v2;
        }
        v_bf16_st_tnsr(c, y, v_convert_f32_to_bf16_all_b(oo));
    }
}

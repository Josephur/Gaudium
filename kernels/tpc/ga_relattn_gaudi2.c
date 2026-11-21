// gaudium_ga_relattn: the attention of a Gemma 4 audio encoder layer (docs/notes/engine-gemma4.md, section 14).
// A frame t sees itself and the W - 1 frames before it (W = attention_context_left - 1: 12 in the checkpoints), and every key carries a
// term of its distance d = t - j:
//   score(t, d) = (Q_t qs) . (ks K_{t-d}) + (Q_t qs) . R_d          d = 0 .. W - 1, t - d >= 0
//   a = softmax_d( cap tanh(score / cap) ),      out_t = sum_d a_d V_{t-d}
// everything in f32 from the bf16 rows, the output rounded to bf16 once (transformers Gemma4AudioAttention: q, k, v .float(), the
// scores, the soft cap and the softmax in f32; its blocks of 12 queries with a 24-frame context, the relative shift and the sliding
// mask are this band written for a matrix product). The band is 12 keys wide, so there is no T x T score matrix to make: one index-space member is one
// head of one frame, 12 dot products of 128 channels and 12 weighted value rows, which is TPC work, and the memory is linear in T.
// A frame's row does not depend on any later frame, so frames appended to a clip (padding) leave the earlier rows unchanged.
//  in0 q bf16 {NH 128, T}   in1 k bf16 {NH 128, T}   in2 v bf16 {NH 128, T}   (head h = channels 128 h .. 128 h + 127)
//  in3 r f32 {NH 128, W}    the projected distance embeddings, column d = distance d (made by the loader)
//  in4 qs f32 {128}         the per-channel query scale (head_dim^-0.5 / ln 2) softplus(per_dim_scale)
//      r and qs in the lane order of a converted bf16 block: inside each 128-block, f32 lane i < 64 is channel 2 i, lane 64 + i is 2 i + 1
//  out0 bf16 {NH 128, T}.   scalars: cap (f32 bits), ks (f32 bits: the key scale ln(1 + e) / ln 2).   index space {NH, T}.   W <= 16
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
// x / cap as the f32 quotient (one Newton correction of x * (1 / cap)), e^d for d <= 0 (max 1 ulp on [-87, 0]), 1 / a with one Newton
// step, tanh to f32 accuracy: the forms of gaudium_ga_softcap (the vendor v_tanh_f32 is a table approximation)
#define QUOT(x_) ({ float64 _x = (x_); float64 _q = _x * rcap; float64 _e = v_f32_mac_b(_q, ncap, _x, 0, 1, 0); v_f32_mac_b(_e, rcap, _q, 0, 1, 0); })
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
#define TANH(d_) ({ float64 _d = (d_); float64 _a = v_f32_abs_b(_d, 0, 0, 1, 0); float64 _t = EXPD(_a * -2.0f); \
    float64 _big = (1.0f - _t) * RECIP(_t + 1.0f); float64 _d2 = _a * _a; \
    float64 _small = _a * ((_d2 * 0.13333333333f - 0.33333333333f) * _d2 + 1.0f); \
    float64 _m = v_f32_sel_less_f32_b(_a, 0.015625f, _small, _big, 0, 0, 1, 0); \
    v_f32_sel_less_f32_b(_d, 0.0f, _m * -1.0f, _m, 0, 0, 1, 0); })

__local__ float64 XV[16];   // one row's capped scores, then its exponentials (every lane the same value)

void main(tensor q, tensor k, tensor v, tensor r, tensor qs, tensor o, int cap_bits, int ks_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int W = get_dim_size(r, 1);
    const float cap = *(float*)&cap_bits, ks = *(float*)&ks_bits;
    const float64 rcap = 1.0f / cap, ncap = -cap;
    int5 z = {0, 0, 0, 0, 0};
    const float64 qs1 = v_f32_ld_tnsr_b(z, qs); z[0] = 64;
    const float64 qs2 = v_f32_ld_tnsr_b(z, qs);
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        int5 c = {h * 128, t, 0, 0, 0};
        const float128 qq = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, q));
        const float64 q1 = qq.v1 * qs1, q2 = qq.v2 * qs2;
        const int nd = (t + 1 < W) ? t + 1 : W;
        float64 m = -1e30f;
        for (int d = 0; d < nd; ++d) {
            c[1] = t - d;
            const float128 kk = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, k));
            int5 rd = {h * 128, d, 0, 0, 0};
            const float64 r1 = v_f32_ld_tnsr_b(rd, r); rd[0] += 64;
            const float64 r2 = v_f32_ld_tnsr_b(rd, r);
            const float64 ac = REDUCE64(q1 * (kk.v1 * ks) + q2 * (kk.v2 * ks), P8, P4, P2, P1);
            const float64 bd = REDUCE64(q1 * r1 + q2 * r2, P8, P4, P2, P1);
            const float64 x = TANH(QUOT(ac + bd)) * cap;
            XV[d] = x; m = v_f32_max_b(m, x);
        }
        float64 sum = 0.f;
        for (int d = nd - 1; d >= 0; --d) { const float64 e = EXPD(XV[d] - m); XV[d] = e; sum = sum + e; }   // oldest key first, as the reference sums
        const float64 inv = RECIP(sum);
        float64 o1 = 0.f, o2 = 0.f;
        for (int d = nd - 1; d >= 0; --d) {
            c[1] = t - d;
            const float128 vv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, v));
            const float64 w = XV[d] * inv;
            o1 = o1 + w * vv.v1; o2 = o2 + w * vv.v2;
        }
        float128 oo; oo.v1 = o1; oo.v2 = o2;
        c[1] = t;
        v_bf16_st_tnsr(c, o, v_convert_f32_to_bf16_all_b(oo));
    }
}

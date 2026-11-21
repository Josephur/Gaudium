// gaudium_ds_read / _codes / _msg / _finish: the DSpine code bridge after each drafter layer (docs/notes/dspine-spec.md section 2).
// One index point = one block row t (index space T). Activations {features, T} bf16; per-channel f32 parameters in the
// deinterleaved lane order (lane_) per 128-channel block, matching v_convert_bf16_to_f32_all_b's {v1 = even, v2 = odd} split.
//  READ   in0 x bf16 {H, T} (the layer output = raw)  -> out0 n bf16 {H, T} = bf16(x / (rms + eps)), out1 rms f32 {64, T} (broadcast)
//         rms = sqrt(max(mean(x^2), FLT_MIN)).  scalar: H
//  CODES  in0 z bf16 {C, T} (n @ R_l^T), in1 codebook bf16 {C, V}, in2 ids_blk i32 {T}  -> out0 codes bf16 {C, T}
//         row 0 = bf16(sqrt(C) codebook[ids_blk[0]]); row t >= 1 = bf16(sqrt(C) z_t / (|z_t| + eps)).  scalars: C, sqrt(C) (f32)
//  MSG    in0 packed bf16 {3R, T} = [q | s | v], in1 gate bias f32 {R}  -> out0 msg bf16 {R, T}
//         row 0 = 0; row t >= 1 = bf16(sigmoid(q_t + s_{t-1} + b) * v_{t-1}).  scalar: R
//  FINISH in0 raw bf16 {H, T}, in1 delta bf16 {H, T} (msg @ W_l^T), in2 rms f32 {64, T}, in3 w f32 {H} (next norm, lane order)
//         -> out0 out bf16 {H, T} = bf16(raw + rms delta) (row 0: raw), out1 normed bf16 {H, T} = bf16(bf16(out rsqrt(mean(out^2) + 1e-6)) w)
//         scalar: H.  The second pass recomputes out from raw / delta: a kernel never reads back what it wrote (TPC coherency).
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
static inline float64 round_bf16(float64 x) { float128 t; t.v1 = x; t.v2 = x; return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(t)).v1; }
#define BEPS 1e-6f
#define F32_TINY 1.1754943508222875e-38f

#if defined(READ)
void main(tensor x, tensor nrm, tensor rms, int H)
{
    const int5 idx = get_index_space_offset(); const int5 isz = get_index_space_size();
    const uchar256 l16 = (read_lane_id_1b_b() >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float inv_h = 1.0f / (float)H;
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        float64 acc = 0.f;
        for (int h = 0; h < H; h += 128) { int5 c = {h, t, 0, 0, 0}; float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x)); acc = v_f32_mac_b(a.v1, a.v1, acc); acc = v_f32_mac_b(a.v2, a.v2, acc); }
        float64 ms = v_f32_max_b(REDUCE64(acc, P8, P4, P2, P1) * inv_h, F32_TINY);
        float64 r = ms * v_rsqrt_f32(ms);                       // sqrt(ms)
        float64 inv = v_reciprocal_f32(r + BEPS);
        for (int h = 0; h < H; h += 128) { int5 c = {h, t, 0, 0, 0}; float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x)); a.v1 = a.v1 * inv; a.v2 = a.v2 * inv; v_bf16_st_tnsr(c, nrm, v_convert_f32_to_bf16_all_b(a)); }
        int5 oc = {0, t, 0, 0, 0}; v_f32_st_tnsr(oc, rms, r);
    }
}
#elif defined(CODES)
void main(tensor z, tensor cb, tensor ids, tensor codes, int C, float sqrtC)
{
    const int5 idx = get_index_space_offset(); const int5 isz = get_index_space_size();
    const uchar256 l16 = (read_lane_id_1b_b() >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float sc = sqrtC;   // sqrt(code_size), from the glue (32 for C = 1024: exact)
    int5 zc = {0, 0, 0, 0, 0}; const int anchor = s_i32_ld_g(gen_addr(zc, ids));
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        if (t == 0) {
            for (int c = 0; c < C; c += 128) { int5 ic = {c, anchor, 0, 0, 0}; float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ic, cb)); a.v1 = a.v1 * sc; a.v2 = a.v2 * sc;
                int5 oc = {c, 0, 0, 0, 0}; v_bf16_st_tnsr(oc, codes, v_convert_f32_to_bf16_all_b(a)); }
            continue;
        }
        float64 acc = 0.f;
        for (int c = 0; c < C; c += 128) { int5 ic = {c, t, 0, 0, 0}; float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ic, z)); acc = v_f32_mac_b(a.v1, a.v1, acc); acc = v_f32_mac_b(a.v2, a.v2, acc); }
        float64 ss = v_f32_max_b(REDUCE64(acc, P8, P4, P2, P1), F32_TINY);
        float64 f = sc * v_reciprocal_f32(ss * v_rsqrt_f32(ss) + BEPS);
        for (int c = 0; c < C; c += 128) { int5 ic = {c, t, 0, 0, 0}; float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ic, z)); a.v1 = a.v1 * f; a.v2 = a.v2 * f; v_bf16_st_tnsr(ic, codes, v_convert_f32_to_bf16_all_b(a)); }
    }
}
#elif defined(MSG)
void main(tensor pk, tensor gb, tensor msg, int R)
{
    const int5 idx = get_index_space_offset(); const int5 isz = get_index_space_size();
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t)
        for (int r = 0; r < R; r += 128) {
            int5 oc = {r, t, 0, 0, 0}; float128 m;
            if (t == 0) { m.v1 = 0.f; m.v2 = 0.f; }
            else {
                int5 qc = {r, t, 0, 0, 0}, sc = {R + r, t - 1, 0, 0, 0}, vc = {2 * R + r, t - 1, 0, 0, 0}, bc = {r, 0, 0, 0, 0};
                float128 q = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(qc, pk)), s = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, pk)), v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(vc, pk));
                float64 b1 = v_f32_ld_tnsr_b(bc, gb); bc[0] = r + 64; float64 b2 = v_f32_ld_tnsr_b(bc, gb);
                m.v1 = v_sigmoid_f32(q.v1 + s.v1 + b1) * v.v1; m.v2 = v_sigmoid_f32(q.v2 + s.v2 + b2) * v.v2;
            }
            v_bf16_st_tnsr(oc, msg, v_convert_f32_to_bf16_all_b(m));
        }
}
#elif defined(FINISH)
void main(tensor raw, tensor delta, tensor rms, tensor w, tensor out, tensor normed, int H)
{
    const int5 idx = get_index_space_offset(); const int5 isz = get_index_space_size();
    const uchar256 l16 = (read_lane_id_1b_b() >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float inv_h = 1.0f / (float)H;
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        int5 rc = {0, t, 0, 0, 0}; float64 rv = v_f32_ld_tnsr_b(rc, rms); if (t == 0) rv = 0.f;   // row 0 (the anchor) gets no write
        float64 acc = 0.f;
        for (int h = 0; h < H; h += 128) { int5 c = {h, t, 0, 0, 0};
            float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, raw)), d = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, delta));
            a.v1 = v_f32_mac_b(rv, d.v1, a.v1); a.v2 = v_f32_mac_b(rv, d.v2, a.v2);
            bfloat128 o = v_convert_f32_to_bf16_all_b(a); v_bf16_st_tnsr(c, out, o);
            float128 ob = v_convert_bf16_to_f32_all_b(o); acc = v_f32_mac_b(ob.v1, ob.v1, acc); acc = v_f32_mac_b(ob.v2, ob.v2, acc); }
        float64 r = v_rsqrt_f32(REDUCE64(acc, P8, P4, P2, P1) * inv_h + 1e-6f);
        for (int h = 0; h < H; h += 128) { int5 c = {h, t, 0, 0, 0}, wc = {h, 0, 0, 0, 0};
            float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, raw)), d = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, delta));
            a.v1 = v_f32_mac_b(rv, d.v1, a.v1); a.v2 = v_f32_mac_b(rv, d.v2, a.v2);
            float128 ob = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(a));
            float64 w1 = v_f32_ld_tnsr_b(wc, w); wc[0] = h + 64; float64 w2 = v_f32_ld_tnsr_b(wc, w);
            float128 y; y.v1 = round_bf16(ob.v1 * r) * w1; y.v2 = round_bf16(ob.v2 * r) * w2;
            v_bf16_st_tnsr(c, normed, v_convert_f32_to_bf16_all_b(y)); }
    }
}
#endif

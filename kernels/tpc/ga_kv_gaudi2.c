// gaudium_ga_kv: the key / value half of gaudium_ga_prep for rows that bring no queries: the context rows of a block drafter whose
// layers have Gemma-shaped heads (docs/notes/engine-gemma4.md, the DFlash2 slice).
//   K = rope(norm(k) w_k)  -> K cache row;   V = v as it is (no norm) -> V cache row perm(.)
//   norm(x) = x rsqrt(mean(x^2) + 1e-6) over the head's 128 NV channels, in f32; one rounding to bf16 at the store.
//  in0 k bf16 {128 NV NKV, T}   in1 v bf16 {128 NV NKV, T}   kn_w f32 {128 NV}   rope_c f32 {128 RV, Lpos}   rope_s f32 {128 RV, Lpos}
//  cursor int32 {1}
//  out0 kcache bf16 {128 NV, C, NKV nch}   out1 vcache bf16 {128 NV, C, NKV nch}: gaudium_ga_prep's layouts (ring == 1: one chunk of
//  C slots, position pos -> slot pos & (C - 1); ring == 0: chunk-major, pos -> {pos % C, (pos / C) NKV + g}), its channel order and its
//  V row permutation.  Row t is position cursor[0] + coff + t.
//  scalars: T, coff, C, NKV, NV, RV, ring.   index space: NKV points (the kv heads)
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define VPERM(j) (((j) & ~127) + ((((j) & 127) < 64) ? 2 * ((j) & 127) : 2 * (((j) & 127) - 64) + 1))

__local__ float64 XB[8];   // one head row in f32: block b -> XB[2 b] (v1), XB[2 b + 1] (v2); NV <= 4
__local__ float64 WB[8];   // the k norm weight, same layout

void main(tensor k, tensor v, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor kcache, tensor vcache,
          int T, int coff, int C, int NKV, int NV, int RV, int ring)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int HV = NV / 2;
    const float eps = 1e-6f, inv_n = 1.0f / (128.0f * (float)NV);
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    for (int b = 0; b < NV; ++b) {
        int5 wc = {128 * b, 0, 0, 0, 0};
        WB[2 * b] = v_f32_ld_tnsr_b(wc, kn_w); wc[0] += 64; WB[2 * b + 1] = v_f32_ld_tnsr_b(wc, kn_w);
    }
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        for (int t = 0; t < T; ++t) {
            const int pos = cur + t;
            int pin, batch;
            if (ring) { pin = pos & (C - 1); batch = h; }
            else { const int chk = pos / C; pin = pos - chk * C; batch = chk * NKV + h; }
            const int vrow = VPERM(pin);
            float64 acc = 0.f;
            for (int b = 0; b < NV; ++b) {
                int5 xc = {128 * (NV * h + b), t, 0, 0, 0};
                float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, k));
                XB[2 * b] = a.v1; XB[2 * b + 1] = a.v2;
                acc = acc + a.v1 * a.v1 + a.v2 * a.v2;
                int5 vo = {128 * b, vrow, batch, 0, 0}; v_bf16_st_tnsr(vo, vcache, v_bf16_ld_tnsr_b(xc, v));
            }
            const float64 r = v_rsqrt_f32(REDUCE64(acc, P8, P4, P2, P1) * inv_n + eps);
            for (int b = 0; b < HV; ++b) {   // vector pair (b, b + HV): weight, rotation, store
                float64 a1 = XB[2 * b] * r * WB[2 * b], a2 = XB[2 * b + 1] * r * WB[2 * b + 1];
                float64 b1 = XB[2 * (b + HV)] * r * WB[2 * (b + HV)], b2 = XB[2 * (b + HV) + 1] * r * WB[2 * (b + HV) + 1];
                if (b < RV) {
                    int5 rc = {128 * b, pos, 0, 0, 0};
                    const float64 C1 = v_f32_ld_tnsr_b(rc, rope_c), S1 = v_f32_ld_tnsr_b(rc, rope_s);
                    rc[0] += 64;
                    const float64 C2 = v_f32_ld_tnsr_b(rc, rope_c), S2 = v_f32_ld_tnsr_b(rc, rope_s);
                    const float64 na1 = a1 * C1 - b1 * S1, nb1 = b1 * C1 + a1 * S1;
                    const float64 na2 = a2 * C2 - b2 * S2, nb2 = b2 * C2 + a2 * S2;
                    a1 = na1; b1 = nb1; a2 = na2; b2 = nb2;
                }
                float128 oa; oa.v1 = a1; oa.v2 = a2;
                float128 ob; ob.v1 = b1; ob.v2 = b2;
                int5 kc = {128 * b, pin, batch, 0, 0};
                v_bf16_st_tnsr(kc, kcache, v_convert_f32_to_bf16_all_b(oa)); kc[0] = 128 * (b + HV); v_bf16_st_tnsr(kc, kcache, v_convert_f32_to_bf16_all_b(ob));
            }
        }
    }
}

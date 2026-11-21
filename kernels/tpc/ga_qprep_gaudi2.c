// gaudium_ga_qprep: the QUERY half of gaudium_ga_prep, for a pass that owns no keys or values: the layers of the Gemma 4 drafter, which
// attend to the target's K / V ring and cache (docs/notes/engine-gemma4.md section 8).
//   q heads (NKV x GQ):  q = rope(norm(q) w_q)  -> bf16 Q rows for the QK batch_gemm (no 1 / sqrt(d): Gemma scales by 1)
//   norm(x) = x rsqrt(mean(x^2) + 1e-6) over the head's 128 NV channels, in f32; one rounding to bf16 at the store.
//  in0 q bf16 {128 NV NH, T}   qn_w f32 {128 NV}   rope_c f32 {128 RV, Lpos}   rope_s f32 {128 RV, Lpos}   cursor int32 {1}
//  out0 qb bf16 {128 NV, GQ T, NKV nch}: row hl T + t of batch ch NKV + g holds q head GQ g + hl (written once per chunk ch)
//  Channel order, the tables' packing and the RoPE pairing are gaudium_ga_prep's (lane order inside each 128-channel block; channel i
//  pairs with i + 64 NV; pairs beyond RV are not rotated), so the Q rows meet the target's K rows in the same order.
//  scalars: T, coff, nch, NKV, GQ, NV, RV, same.  Position of row t: cursor[0] + coff + (same ? 0 : t)  (same = 1: every row at one
//  position, the drafter's passes).   index space: NH points (the q heads)
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

__local__ float64 XB[8];   // one head row in f32: block b -> XB[2 b] (v1), XB[2 b + 1] (v2); NV <= 4
__local__ float64 WB[8];   // the norm weight, same layout

void main(tensor q, tensor qn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor qb,
          int T, int coff, int nch, int NKV, int GQ, int NV, int RV, int same)
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
        WB[2 * b] = v_f32_ld_tnsr_b(wc, qn_w); wc[0] += 64; WB[2 * b + 1] = v_f32_ld_tnsr_b(wc, qn_w);
    }
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        for (int t = 0; t < T; ++t) {
            const int pos = same ? cur : cur + t;
            float64 acc = 0.f;
            for (int b = 0; b < NV; ++b) {
                int5 xc = {128 * (NV * h + b), t, 0, 0, 0};
                float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, q));
                XB[2 * b] = a.v1; XB[2 * b + 1] = a.v2;
                acc = acc + a.v1 * a.v1 + a.v2 * a.v2;
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
                const bfloat128 A = v_convert_f32_to_bf16_all_b(oa), B = v_convert_f32_to_bf16_all_b(ob);
                for (int ch = 0; ch < nch; ++ch) {
                    int5 oc = {128 * b, (h % GQ) * T + t, ch * NKV + (h / GQ), 0, 0};
                    v_bf16_st_tnsr(oc, qb, A); oc[0] = 128 * (b + HV); v_bf16_st_tnsr(oc, qb, B);
                }
            }
        }
    }
}

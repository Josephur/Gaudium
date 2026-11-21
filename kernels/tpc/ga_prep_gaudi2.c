// gaudium_ga_prep: Gemma 4 attention front end for T new tokens at device cursor (+ coff). docs/notes/engine-gemma4.md.
// SHAPE-GENERIC: the head counts and widths are scalars the glue reads from the tensors, so one ELF serves every layer kind and
// every size of the family. A head is NV vectors of 128 channels (NV = 2: 256-channel heads; NV = 4: 512).
//   q heads (NKV x GQ):  q = rope(norm(q) w_q)                         -> bf16 Q rows for the QK batch_gemm (no 1 / sqrt(d): Gemma scales by 1)
//   kv heads (NKV):      K = rope(norm(k) w_k)                          -> K cache row
//                        V = norm(v)            (NO weight)             -> V cache row perm(.)
//                        -DKEQV build: no v input; V = norm(k) of the SAME k rows (attention_k_eq_v: global layers of the 31B)
//                        -DVRAW build: V = v, no norm (the layers of a block drafter that reads Gemma-shaped heads: its q scale is
//                        folded into w_q by the loader)
//   norm(x) = x rsqrt(mean(x^2) + 1e-6) over the head's 128 NV channels, in f32; one rounding to bf16 at the store.
//  in0 q bf16 {128 NV NH, T}   in1 k bf16 {128 NV NKV, T}   [in2 v bf16 {128 NV NKV, T}]
//  qn_w f32 {128 NV}   kn_w f32 {128 NV}   rope_c f32 {128 RV, Lpos}   rope_s f32 {128 RV, Lpos}   cursor int32 {1}
//  out0 qb bf16 {128 NV, GQ T, NKV nch}: row hl T + t of batch ch NKV + g holds q head GQ g + hl (written once per chunk ch)
//  out1 kcache bf16 {128 NV, C, NKV nch}   out2 vcache bf16 {128 NV, C, NKV nch}
//      ring == 0 (global layers): the bucket of nch chunks of C positions, CHUNK-MAJOR batch b = ch NKV + g, position pos ->
//                {pos % C, (pos / C) NKV + g}: the layout of gaudium_attn_prep, so gaudium_attn_softmax_chunk and the chunk ladder apply.
//      ring == 1 (sliding layers): ONE chunk of C = RING slots (a power of two); position pos -> slot pos & (C - 1). A row sees the
//                last W positions, so with RING >= W + T - 1 the slots it reads still hold them (gaudium_ga_softmax_ring masks the rest).
//  V rows are permuted inside each 128-row block (VPERM) so that the softmax kernels' interleaving f32 -> bf16 conversion of P keeps
//  P[m] paired with the right V row (as gaudium_attn_prep).
//  CHANNEL ORDER. The loader stores the rows of W_q, W_k and W_v in LANE ORDER inside each 128-channel block (position p of a block
//  holds channel c with p = c odd ? 64 + c / 2 : c / 2), so every GEMM output arrives in lane order and the V cache, the PV partials
//  and the combined output carry it (gaudium_ga_post's interleaving conversion then gives natural order for the o projection). q and
//  K only need the SAME order. A bf16 load deinterleaves (v1 = positions 0, 2, .., v2 = positions 1, 3, ..): the per-channel tables
//  qn_w, kn_w, rope_c, rope_s are packed for that view (qw::Packer, pack_gemma4: element 128 b + 64 h + j <-> block b, vector h, lane j).
//  RoPE (x cos + rotate_half(x) sin): channel i of the first half pairs with channel i + 64 NV of the second: vector b with vector
//  b + NV / 2, lane by lane. The tables hold RV <= NV / 2 vector pairs; pairs beyond RV are not rotated (proportional RoPE of the
//  global layers: 64 of 256 pairs; a table lane of an unrotated channel inside a rotated vector holds cos 1, sin 0).
//  scalars: T, coff (positions are cursor[0] + coff + t), nch, C, NKV, GQ, NV, RV, ring.   index space: NH + NKV points (q heads, then kv heads)
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define VPERM(j) (((j) & ~127) + ((((j) & 127) < 64) ? 2 * ((j) & 127) : 2 * (((j) & 127) - 64) + 1))

__local__ float64 XB[8];   // one head row in f32: block b -> XB[2 b] (v1), XB[2 b + 1] (v2); NV <= 4
__local__ float64 WB[8];   // the head kind's norm weight, same layout
#if !defined(KEQV) && !defined(VRAW)
__local__ float64 VB[8];   // the v row
#endif

#ifdef KEQV
void main(tensor q, tensor k, tensor qn_w, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor,
          tensor qb, tensor kcache, tensor vcache, int T, int coff, int nch, int C, int NKV, int GQ, int NV, int RV, int ring)
#else
void main(tensor q, tensor k, tensor v, tensor qn_w, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor,
          tensor qb, tensor kcache, tensor vcache, int T, int coff, int nch, int C, int NKV, int GQ, int NV, int RV, int ring)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int NH = NKV * GQ, HV = NV / 2;
    const float eps = 1e-6f, inv_n = 1.0f / (128.0f * (float)NV);
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        const int is_q = (i < NH);
        const int h = is_q ? i : i - NH;
        for (int b = 0; b < NV; ++b) {
            int5 wc = {128 * b, 0, 0, 0, 0};
            if (is_q) { WB[2 * b] = v_f32_ld_tnsr_b(wc, qn_w); wc[0] += 64; WB[2 * b + 1] = v_f32_ld_tnsr_b(wc, qn_w); }
            else      { WB[2 * b] = v_f32_ld_tnsr_b(wc, kn_w); wc[0] += 64; WB[2 * b + 1] = v_f32_ld_tnsr_b(wc, kn_w); }
        }
        for (int t = 0; t < T; ++t) {
            const int pos = cur + t;
            float64 acc = 0.f;
            for (int b = 0; b < NV; ++b) {
                int5 xc = {128 * (NV * h + b), t, 0, 0, 0};
                float128 a;
                if (is_q) a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, q));
                else      a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, k));
                XB[2 * b] = a.v1; XB[2 * b + 1] = a.v2;
                acc = acc + a.v1 * a.v1 + a.v2 * a.v2;
            }
            const float64 r = v_rsqrt_f32(REDUCE64(acc, P8, P4, P2, P1) * inv_n + eps);
            int pin = 0, batch = 0;
            if (!is_q) {
                if (ring) { pin = pos & (C - 1); batch = h; }
                else { const int chk = pos / C; pin = pos - chk * C; batch = chk * NKV + h; }
                const int vrow = VPERM(pin);
#ifdef KEQV
                for (int b = 0; b < NV; ++b) {   // V = the unweighted norm of the k row
                    float128 o; o.v1 = XB[2 * b] * r; o.v2 = XB[2 * b + 1] * r;
                    int5 vo = {128 * b, vrow, batch, 0, 0}; v_bf16_st_tnsr(vo, vcache, v_convert_f32_to_bf16_all_b(o));
                }
#elif defined(VRAW)
                for (int b = 0; b < NV; ++b) {   // V = the v rows as they are (a layer without a value norm)
                    int5 xc = {128 * (NV * h + b), t, 0, 0, 0};
                    int5 vo = {128 * b, vrow, batch, 0, 0}; v_bf16_st_tnsr(vo, vcache, v_bf16_ld_tnsr_b(xc, v));
                }
#else
                float64 accv = 0.f;
                for (int b = 0; b < NV; ++b) {
                    int5 xc = {128 * (NV * h + b), t, 0, 0, 0};
                    float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, v));
                    VB[2 * b] = a.v1; VB[2 * b + 1] = a.v2;
                    accv = accv + a.v1 * a.v1 + a.v2 * a.v2;
                }
                const float64 rv = v_rsqrt_f32(REDUCE64(accv, P8, P4, P2, P1) * inv_n + eps);
                for (int b = 0; b < NV; ++b) {
                    float128 o; o.v1 = VB[2 * b] * rv; o.v2 = VB[2 * b + 1] * rv;
                    int5 vo = {128 * b, vrow, batch, 0, 0}; v_bf16_st_tnsr(vo, vcache, v_convert_f32_to_bf16_all_b(o));
                }
#endif
            }
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
                if (is_q) {
                    for (int ch = 0; ch < nch; ++ch) {
                        int5 oc = {128 * b, (h % GQ) * T + t, ch * NKV + (h / GQ), 0, 0};
                        v_bf16_st_tnsr(oc, qb, A); oc[0] = 128 * (b + HV); v_bf16_st_tnsr(oc, qb, B);
                    }
                } else {
                    int5 kc = {128 * b, pin, batch, 0, 0};
                    v_bf16_st_tnsr(kc, kcache, A); kc[0] = 128 * (b + HV); v_bf16_st_tnsr(kc, kcache, B);
                }
            }
        }
    }
}

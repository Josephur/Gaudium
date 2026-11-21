// gaudium_gdn_chunk: the per-row (independent) coefficient work of the chunked gated delta rule (Idea A, prefill path),
// one index-space point per (value head h, chunk j), C <= 64 rows per chunk, fp32 throughout. The forward substitution
// (the only part with a dependency between rows) is gaudium_gdn_solve; the GEMMs are on the MME (gdn_chunk.hpp).
//
//  per head and chunk, rows t, u = 0..C-1, G[t] = chunk-local cumsum of the log decay, beta[t] = sigmoid(b):
//     Gamma[t] = exp(G[t])     tail[t] = exp(G[C-1] - G[t])     gL = exp(G[C-1])
//     E[t][u]  = exp(G[t] - G[u])  (u <= t, else 0; values below 1e-30 flushed to 0: fp32 denormal operands made the
//                MME 5x slower in the Idea A probe, and they multiply q, k and contribute nothing)
//     An[t][u] = -beta[t] E[t][u] KK[t][u]  (u < t)        KK[t][u] = k[t].k[u]  from the MME       (= -A, strictly lower)
//     Pn[t][u] = -E[t][u] QK[t][u]          (u <= t)       QK[t][u] = q[t].k[u]  from the MME
//  Every decay factor is exp of a non-positive number (no ratios of exponentials).
//
//  in0 gbT {Tp, 128} f32   column h = G, column 64+h = beta (gaudium_gdn_gate CHUNK variant, transposed by a vendor node)
//  in1 KK  {C, C, HV, NCH} f32   FCD = u          in2 QK {C, C, HV, NCH}
//  in3 q   {128, C, HV, NCH} f32 (lane order)     in4 k  {128, C, HV, NCH}
//  out0 M3  {C, 2C, HV, NCH}:  rows 0..C-1 = Pn,  rows C..2C-1 = identity      (the MME forms [Pn Mw ; Mw] and [Pn Un ; Un] from it)
//  out1 An  {C, C, HV, NCH}    -A, the input of gaudium_gdn_solve
//  out2 gb2 {C, 2, HV, NCH}    row 0 = Gamma[u], row 1 = beta[u]  (for gaudium_gdn_solve)
//  out3 GQ0 {128, 2C, HV, NCH}: rows t < C = Gamma[t] q[t], rows >= C = 0     (the C operand of the batched [Qeff ; W] GEMM)
//  out4 AK  {128, 2C, HV, NCH}: rows t < C = 0, rows C+t = -tail[t] k[t]       (A operand of the state update, transposed)
//  out5 gL  {1, 1, HV, NCH}
//  scalars: C (16, 32 or 64), HV.  Index space: HV*NCH points, point i -> (h = i % HV, j = i / HV).
// exp (docs/notes/deltanet-exp.md): every exp argument here is <= 0 (cumulative log decays and their
// differences), EXPD's validated domain; EXPD replaces the vendor v_exp_f32 (wrong by a factor ~2 at 42 inputs in [-104, 0],
// research/exp_bug/vendor_bad_f32.txt). Lanes with a positive argument (u > t, lanes >= C) are masked or unused. EXPD clamps at
// e^-87; the 1e-30 flush below turns that into 0 as before. -DVENDOR_EXP restores v_exp_f32 (A/B builds).
#ifdef VENDOR_EXP
#define DEXP(x) v_exp_f32(x)
#else
#define DEXP(x) ({ float64 _d = v_f32_max_b((x), -87.0f); \
    float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0); \
    float64 _r = v_f32_mac_b(_z, -0.693359375f, _d, 0, 1, 0); _r = v_f32_mac_b(_z, 2.12194440e-4f, _r, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.0013836704893037677f, (float64)0.008374780416488647f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.04166822507977486f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.1666642129421234f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.49999991059303284f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
#endif
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REP_GROUP(a, gs) MOVDG_ALL((a), gs, gs, gs, gs, (a))
#define REP_ANY(dst, a, gs) do { switch (gs) { case 0: dst = REP_GROUP(a, 0); break; case 1: dst = REP_GROUP(a, 1); break; case 2: dst = REP_GROUP(a, 2); break; default: dst = REP_GROUP(a, 3); } } while (0)

static inline uchar256 pat_of(int e) { unsigned char b = (unsigned char)((e & 7) | ((e >> 3) << 5) | 0x80); uchar256 r = b; return r; }
static inline float64 bcast(float64 a, int l) { float64 r; REP_ANY(r, a, (l >> 4) & 3); uchar256 p = pat_of(l & 15); return v_f32_shuffle_b(r, p, 0, r); }

// -DCHAIN (gdn_chunk.hpp chain mode): one more output, gLI {128, 128, HV, NCH} = gL I (the chunk's decay times the
// identity), the C operand of the batched transition matrices A_j = AK_j^T AX_j + gL_j I.
#ifdef CHAIN
void main(tensor gbT, tensor KK, tensor QK, tensor q, tensor k,
          tensor M3, tensor An, tensor gb2, tensor GQ0, tensor AK, tensor gL, tensor gLI, int C, int HV)
#else
void main(tensor gbT, tensor KK, tensor QK, tensor q, tensor k,
          tensor M3, tensor An, tensor gb2, tensor GQ0, tensor AK, tensor gL, int C, int HV)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const float tiny = 1e-30f;
    const float64 zero = 0.f;

    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        const int h = i % HV, j = i / HV;
        int5 gc = {j * C, h, 0, 0, 0};
        const float64 G = v_f32_ld_tnsr_b(gc, gbT);
        gc[1] = 64 + h;
        const float64 B = v_f32_ld_tnsr_b(gc, gbT);
        const float64 glv = bcast(G, C - 1);
        float64 Gam = DEXP(G), tail = DEXP(glv - G), gl = DEXP(glv);
        Gam = v_f32_sel_less_f32_b(Gam, tiny, zero, Gam);
        tail = v_f32_sel_less_f32_b(tail, tiny, zero, tail);
        gl = v_f32_sel_less_f32_b(gl, tiny, zero, gl);
        { int5 c = {0, 0, h, j, 0}; v_f32_st_tnsr(c, gL, gl); c[1] = 0; v_f32_st_tnsr(c, gb2, Gam); c[1] = 1; v_f32_st_tnsr(c, gb2, B); }
#ifdef CHAIN
        for (int r = 0; r < 128; ++r) {   // gl is the same in every lane (bcast of G[C-1]); row r has it at column r
            int5 c = {0, r, h, j, 0}; const float64 dg = v_f32_sel_eq_i32_b(lane, r & 63, gl, zero);
            v_f32_st_tnsr(c, gLI, r < 64 ? dg : zero); c[0] = 64; v_f32_st_tnsr(c, gLI, r < 64 ? zero : dg); }
#endif

        for (int tg = 0; tg < (C + 15) / 16; ++tg) {
            float64 Gr, Br, Gamr, tailr;
            REP_ANY(Gr, G, tg); REP_ANY(Br, B, tg); REP_ANY(Gamr, Gam, tg); REP_ANY(tailr, tail, tg);
#pragma unroll 4
            for (int te = 0; te < 16; ++te) {
                const int t = 16 * tg + te;
                const uchar256 pt = pat_of(te);
                const float64 Gt = v_f32_shuffle_b(Gr, pt, 0, Gr), bt = v_f32_shuffle_b(Br, pt, 0, Br);
                const float64 gamt = v_f32_shuffle_b(Gamr, pt, 0, Gamr), tailt = v_f32_shuffle_b(tailr, pt, 0, tailr);
                int5 cc = {0, t, h, j, 0};
                const float64 KKt = v_f32_ld_tnsr_b(cc, KK), QKt = v_f32_ld_tnsr_b(cc, QK);
                int5 qc = {0, t, h, j, 0};
                const float64 q0 = v_f32_ld_tnsr_b(qc, q), k0 = v_f32_ld_tnsr_b(qc, k);
                qc[0] = 64;
                const float64 q1 = v_f32_ld_tnsr_b(qc, q), k1 = v_f32_ld_tnsr_b(qc, k);
                float64 E = DEXP(Gt - G);
                E = v_f32_sel_grt_i32_b(lane, t, zero, E);
                E = v_f32_sel_less_f32_b(E, tiny, zero, E);
                v_f32_st_tnsr(cc, M3, -(E * QKt));
                v_f32_st_tnsr(cc, An, v_f32_sel_geq_i32_b(lane, t, zero, -(bt * E * KKt)));
                { int5 ic = {0, C + t, h, j, 0}; v_f32_st_tnsr(ic, M3, v_f32_sel_eq_i32_b(lane, t, 1.f, zero)); }
                int5 oc = {0, t, h, j, 0};
                v_f32_st_tnsr(oc, GQ0, gamt * q0); v_f32_st_tnsr(oc, AK, zero);
                oc[0] = 64;
                v_f32_st_tnsr(oc, GQ0, gamt * q1); v_f32_st_tnsr(oc, AK, zero);
                oc[0] = 0; oc[1] = C + t;
                v_f32_st_tnsr(oc, GQ0, zero); v_f32_st_tnsr(oc, AK, -(tailt * k0));
                oc[0] = 64;
                v_f32_st_tnsr(oc, GQ0, zero); v_f32_st_tnsr(oc, AK, -(tailt * k1));
            }
        }
    }
}

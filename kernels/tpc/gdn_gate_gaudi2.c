// gaudium_gdn_gate: per-token, per-head gating of the gated delta rule.
//   beta = sigmoid(b)            eg = exp(g),  g = -exp(A_log) * softplus(a + dt_bias)
//  in0 ba bf16 {128, T}: channel 2j = b_j, 2j+1 = a_j (pack_qwen35 W_ba); the "all" bf16->f32 conversion gives v1 = b, v2 = a
//  in1 A_log f32 {64} (48 used)   in2 dt_bias f32 {64}   out0 eg f32 {64, T}   out1 beta f32 {64, T}   scalar: T
//
// exp (docs/notes/deltanet-exp.md): EXPD, the Cody-Waite + degree-6 formula of the attention kernels, not the
// vendor v_exp_f32 (wrong by a factor ~2 at isolated inputs just below odd multiples of ln 2, on both signs:
// research/exp_bug/vendor_bad_f32.txt). Every exp argument here is brought into EXPD's validated domains:
//   softplus(x) = max(x, 0) + log1p(exp(-|x|))        exp argument <= 0, log argument in (1, 2] (+ its exact rounding error)
//   exp(A_log)  per-head constant, A_log in [-5.6, 4.9] on Qwen3.8-27B: EXPD with the upper clamp at 88 (validated exhaustively on [-87, 88])
//   exp(g), g <= 0: EXPD; g < -87 gives 0 (the vendor exp and f32 underflow there too; EXPD alone would clamp at e^-87)
// -DVENDOR_EXP restores the old code (v_exp_f32 and softplus = log(1 + exp(x))) for A/B builds.
// -DPROBE (research/deltanet_exp, decode variant only): eg <- g (the decay exp's argument), beta <- x = a + dt_bias.
#define EXPD(d_) ({ float64 _d = v_f32_min_b(v_f32_max_b((d_), -87.0f), 88.0f); \
    float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0); \
    float64 _r = v_f32_mac_b(_z, -0.693359375f, _d, 0, 1, 0); _r = v_f32_mac_b(_z, 2.12194440e-4f, _r, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.0013836704893037677f, (float64)0.008374780416488647f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.04166822507977486f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.1666642129421234f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.49999991059303284f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
#ifdef VENDOR_EXP
static inline float64 softplus(float64 x) {
    float64 e = v_exp_f32(x);
    float64 sp = v_log_f32(e + 1.f);
    return v_f32_sel_grt_f32_b(x, 20.f, x, sp);     // x > 20 ? x : log(1 + exp(x))
}
#define EXPA(a) v_exp_f32(a)
#define EXPG(g) v_exp_f32(g)
#else
// log1p(e) for e = exp(-|x|) in (0, 1]: u = 1 + e rounds; c = e - (u - 1) is that rounding error EXACTLY (u in [1, 2], Sterbenz), and
// log1p(e) = log(u) + c/u + O(c^2) ~ log(u) + c (|c (1 - 1/u)| <= |c| e: below 1 ulp of log1p(e)). Without c, log(1 + e) is 0 for
// e < 2^-25 (x < -17.3) and loses relative precision below e ~ 2^-12, where HF's softplus (log1p) does not: with A = exp(A_log) up to
// 139 that left exp(g) = 1 instead of 1 - A e (research/deltanet_exp, sim_gdnexp gate). -DNO_LOG1P drops c (A/B).
static inline float64 softplus(float64 x) {
    float64 e = EXPD(-v_f32_abs_b(x));
    float64 u = e + 1.f;
#ifdef NO_LOG1P
    float64 l = v_log_f32(u);
#else
    float64 l = v_log_f32(u) + (e - (u - 1.f));
#endif
    float64 sp = v_f32_max_b(x, 0.f) + l;
    return v_f32_sel_grt_f32_b(x, 20.f, x, sp);     // x > 20 ? x : max(x, 0) + log1p(exp(-|x|))  (= log(1 + exp(x)))
}
#define EXPA(a) EXPD(a)
#define EXPG(g) ({ float64 _g = (g); v_f32_sel_less_f32_b(_g, -87.f, 0.f, EXPD(_g)); })
#endif
#ifdef CHUNK
// CHUNK variant (prefill on the MME, gdn_chunk.hpp): out0 gb f32 {128, Tp}, Tp = NCH*C >= T: lanes 0..63 = G[t] = sum of the
// log decays g over the rows of t's chunk up to t (restarted every C rows), lanes 64..127 = beta[t]; pad rows t >= T get
// G = G of the last real row of that chunk and beta = 0 (identity rows: they contribute nothing to the chunk).
// index space = NCH chunks (was 1 point for all Tp rows): the cumsum restarts every C rows, so one point per
// chunk computes exactly the same values (bit-identical) on NCH TPCs instead of one.
void main(tensor ba, tensor A_log, tensor dt_bias, tensor gb, int T, int C)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 c = {0, 0, 0, 0, 0};
    float64 A = v_f32_ld_tnsr_b(c, A_log);
    float64 dtb = v_f32_ld_tnsr_b(c, dt_bias);
    float64 negA = -EXPA(A);
    for (int j = idx[0]; j < idx[0] + isz[0]; ++j) {
    float64 G = 0.f;
    for (int t = j * C; t < (j + 1) * C; ++t) {
        int5 bc = {0, t, 0, 0, 0};
        float64 be = 0.f;
        if (t % C == 0) G = 0.f;
        if (t < T) {
            float128 f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(bc, ba));
            float64 b = f.v1, a = f.v2;
            be = v_sigmoid_f32(b);
            G = G + negA * softplus(a + dtb);
        }
        v_f32_st_tnsr(bc, gb, G);
        bc[0] = 64;
        v_f32_st_tnsr(bc, gb, be);
    }
    }
}
#else
void main(tensor ba, tensor A_log, tensor dt_bias, tensor eg, tensor beta, int T)
{
    int5 c = {0, 0, 0, 0, 0};
    float64 A = v_f32_ld_tnsr_b(c, A_log);
    float64 dtb = v_f32_ld_tnsr_b(c, dt_bias);
    float64 negA = -EXPA(A);
#ifdef PROBE
    for (int t = 0; t < T; ++t) {
        int5 bc = {0, t, 0, 0, 0};
        float128 f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(bc, ba));
        float64 b = f.v1, a = f.v2; (void)b;
        float64 x = a + dtb;
        float64 g = negA * softplus(x);
        v_f32_st_tnsr(bc, eg, g);
        v_f32_st_tnsr(bc, beta, x);
    }
#else
    // ONE special-function table at a time (2026-11-11, level-1 traces): sigmoid and log are LUT functions and a TPC holds one table; a
    // loop that alternates them per row reloads a table every row (this kernel 0.51 -> 0.14 ms per 16-row step with one table) and leaves
    // ~9 us of idle time after the node. All rows' sigmoids first, then all rows' softplus: same operations per element, bit-identical.
    for (int t = 0; t < T; ++t) {
        int5 bc = {0, t, 0, 0, 0};
        float128 f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(bc, ba));
        v_f32_st_tnsr(bc, beta, v_sigmoid_f32(f.v1));
    }
    for (int t = 0; t < T; ++t) {
        int5 bc = {0, t, 0, 0, 0};
        float128 f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(bc, ba));
        float64 g = negA * softplus(f.v2 + dtb);
        v_f32_st_tnsr(bc, eg, EXPG(g));
    }
#endif
}
#endif

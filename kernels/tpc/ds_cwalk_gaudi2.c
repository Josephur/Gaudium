// gaudium_ds_cwalk: DSpine's serial walk for sampled requests, COUPLED to the verify's sampler (study and design:
// docs/notes/engine-dspine.md section 8). One TPC. The verify row that checks draft t of this block samples the token at sequence
// index n = base + t with argmax_{v in S} ((l_v - m) invT + g(seed, n, v)); the coupled draft takes the same argmax over the row's 16
// candidates, with the drafter's calibrated scores in place of the target logits and the SAME noise g (samp_noise.h):
//   y_k  = score[t-1, j, k] * cs_t,  cs_t = alpha / rms_t   (score = refined logit x RMS of the refined row; rms_t = the pre-write RMS)
//   S    = the sampler's rules applied to the 16 values: C (y >= m - DELTA/invT), top-k (largest value with count(y >= .) >= k),
//          top-p (largest value >= tau_k with mass(y >= .) >= p mass(y >= tau_k), mass = sum exp((y - m) invT) over C, summed in
//          candidate order), min-p; with --draft-trunc 0 only C (temperature + noise)
//   k*   = argmax_{k in S} ((y_k - m) invT + g(rk(seed, base + t), cand[k, t])), ties -> lowest lane; j <- k*  (the serial walk)
// spi flags without F_COUPLED (--coupled 0): the greedy walk on the raw scores, identical to gaudium_df_walk (same ties).
// drafts[s] = cand[k*, s+1]; alts / alts2 = rank 2 / 3 of the same row (tree layouts; sampled requests run chains).
//  in0 scores f32 {16, 16, S}  in1 cand i32 {16, T}  in2 rms f32 {64, T}  in3 spf f32 {8}  in4 spi i32 {8}  in5 base i32 {1}
//  out0 drafts i32 {S}  out1 alts i32 {S}  out2 alts2 i32 {S}  out3 cs f32 {64, S} (the row scales, for the CPU twin in samp_ref.hpp)
//  scalar: alpha (f32 bits)
#include "samp_noise.h"
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_F(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define MOVDG_I(a, s0, s1, s2, s3, inc) v_i32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE_MAX_F(a) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_F(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_F(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE_MIN_I(a) ({ int64 _a = (a); int64 _r; \
    _r = MOVDG_I(_a, 1, 2, 3, 0, _a); _a = v_i32_min_b(_a, _r); _r = MOVDG_I(_a, 2, 3, 0, 1, _a); _a = v_i32_min_b(_a, _r); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P8, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P2, 0, _a)); _a = v_i32_min_b(_a, v_i32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE_MAX_I(a) ({ int64 _a = (a); int64 _r; \
    _r = MOVDG_I(_a, 1, 2, 3, 0, _a); _a = v_i32_max_b(_a, _r); _r = MOVDG_I(_a, 2, 3, 0, 1, _a); _a = v_i32_max_b(_a, _r); \
    _a = v_i32_max_b(_a, v_i32_shuffle_b(_a, P8, 0, _a)); _a = v_i32_max_b(_a, v_i32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_i32_max_b(_a, v_i32_shuffle_b(_a, P2, 0, _a)); _a = v_i32_max_b(_a, v_i32_shuffle_b(_a, P1, 0, _a)); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define NEG (-3.0e38f)
#define F_TOPP 1
#define F_MINP 2
#define F_COUPLED 4
#define F_DTRUNC 8
typedef union { float f; int i; unsigned u; } fu_t;
static inline float ldf(tensor t, int a) { int5 x = {a, 0, 0, 0, 0}; return s_f32_ld_g(gen_addr(x, t)); }
static inline int ldi(tensor t, int a) { int5 x = {a, 0, 0, 0, 0}; return s_i32_ld_g(gen_addr(x, t)); }

void main(tensor scores, tensor cand, tensor rms, tensor spf, tensor spi, tensor base, tensor drafts, tensor alts, tensor alts2, tensor csout, float alpha)
{
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    const int S = get_dim_size(scores, 2);
    const int flags = ldi(spi, 3); const int coupled = (flags & F_COUPLED) != 0, trunc = coupled && (flags & F_DTRUNC);
    const float invT = ldf(spf, 0), dT = ldf(spf, 1), topp = ldf(spf, 2), mpc = ldf(spf, 3); const int topk = ldi(spi, 0);
    const unsigned slo = (unsigned)ldi(spi, 1), shi = (unsigned)ldi(spi, 2); const int pos0 = ldi(base, 0);
    int64 prev = 0;
    for (int s = 0; s < S; ++s) {
        float64 R = NEG;
        for (int p = 0; p < 16; ++p) { int5 c = {0, p, s, 0, 0}; float64 row = v_f32_ld_tnsr_b(c, scores); R = v_f32_sel_eq_i32_b(prev, p, row, R); }
        R = v_f32_sel_less_i32_b(lane, 16, R, NEG);
        int5 cc = {0, s + 1, 0, 0, 0};
        int64 cv = v_i32_ld_tnsr_b(cc, cand);
        float64 Z = R;   // greedy: the raw scores
        if (coupled) {
            int5 rc = {0, s + 1, 0, 0, 0}; float64 cs = alpha * v_reciprocal_f32(v_f32_ld_tnsr_b(rc, rms));
            int5 oc = {0, s, 0, 0, 0}; v_f32_st_tnsr(oc, csout, cs);
            float64 y = R * cs; y = v_f32_sel_less_i32_b(lane, 16, y, NEG);
            float64 m = REDUCE_MAX_F(y);
            float64 inC = v_f32_sel_geq_f32_b(y, m - dT, 1.0f, 0.0f);
            float64 x = y - m; x = x * invT;
            float64 w = VEXP(v_f32_max_b(x, -30.0f)); w = v_f32_sel_eq_f32_b(inC, 1.0f, w, 0.0f);
            // per lane i: cnt_i = #{p : y_p >= y_i}, mass_i = sum_{p in C, y_p >= y_i} w_p, summed over p = 0..15 in order
            float64 cnt = 0.0f, mass = 0.0f;
            for (int p = 0; p < 16; ++p) {
                float64 yp = REDUCE_MAX_F(v_f32_sel_eq_i32_b(lane, p, y, NEG)), wp = REDUCE_MAX_F(v_f32_sel_eq_i32_b(lane, p, w, NEG));
                cnt = cnt + v_f32_sel_geq_f32_b(yp, y, 1.0f, 0.0f); mass = mass + v_f32_sel_geq_f32_b(yp, y, wp, 0.0f);
            }
            float64 thr = m - dT;   // C
            if (trunc) {
                float64 tk = thr;
                if (topk > 0) { tk = REDUCE_MAX_F(v_f32_sel_geq_f32_b(cnt, (float)topk, y, NEG)); tk = v_f32_max_b(tk, thr); }
                thr = tk;
                if (flags & F_TOPP) {
                    float64 inK = v_f32_sel_geq_f32_b(y, tk, inC, 0.0f);
                    float64 tot = REDUCE_MAX_F(v_f32_sel_eq_f32_b(inK, 1.0f, mass, 0.0f));
                    float64 tgt = tot * topp;
                    float64 tp = REDUCE_MAX_F(v_f32_sel_eq_f32_b(inK, 1.0f, v_f32_sel_geq_f32_b(mass, tgt, y, NEG), NEG));
                    thr = v_f32_max_b(thr, tp);
                }
                if (flags & F_MINP) thr = v_f32_max_b(thr, m + mpc);
            }
            float64 g = VGUMBEL(row_key(slo, shi, (unsigned)(pos0 + s + 1)), cv);
            Z = x + g;
            Z = v_f32_sel_geq_f32_b(y, thr, Z, NEG);
            Z = v_f32_sel_less_i32_b(lane, 16, Z, NEG);
        }
        int64 w = 0;
        for (int rank = 0; rank < 3; ++rank) {
            float64 mz = REDUCE_MAX_F(Z);
            int64 ix = v_i32_sel_eq_f32_b(Z, mz, lane, 0x7fffffff);
            w = REDUCE_MIN_I(ix);
            int64 tok = REDUCE_MAX_I(v_i32_sel_eq_i32_b(lane, w, cv, -1));
            int5 oc = {s, 0, 0, 0, 0};
            if (rank == 0) { v_i32_st_tnsr(oc, drafts, tok); prev = w; }
            else if (rank == 1) v_i32_st_tnsr(oc, alts, tok);
            else v_i32_st_tnsr(oc, alts2, tok);
            Z = v_f32_sel_eq_i32_b(lane, w, NEG, Z);
        }
    }
}

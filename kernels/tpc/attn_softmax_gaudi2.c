#include "attn_shape.h"   // ATT_NKV, ATT_GQ, ATT_NH, ATT_NPT, ATT_GATE: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_attn_softmax: cursor-aware masked softmax over the score rows of the attention batch_gemm.
//  in0 scores f32 {Lmax, R, 4} (R = 6T rows per kv group: row = hl*T + t)   in1 cursor int32 {1}
//  out0 P bf16 {Lmax, R, 4}: softmax over key positions [0, cursor + t], zero elsewhere.
//  The f32->bf16 "all" conversion interleaves its two inputs, so bf16 position m of each 128-block holds the probability
//  of GEMM column j(m) = m even ? m/2 : 64 + (m-1)/2; the V cache rows are stored in that same permutation (attn_prep).
//  index space: R*4 rows, point p -> row p % R, group p / R.   scalars: T, Lmax, coff (cursor offset, see attn_prep)
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE_MAX64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

void main(tensor scores, tensor cursor, tensor P, int T, int Lmax, int coff)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const int R = ATT_GQ * T;
    const float NEG = -1e30f;

    for (int p = idx[0]; p < idx[0] + isz[0]; ++p) {
        const int row = p % R, g = p / R, t = row % T;
        const int nvalid = cur + t + 1;
        int5 c = {0, row, g, 0, 0};
        // pass 1: max over valid columns
        float64 m = NEG;
        for (int j = 0; j < Lmax; j += 64) {
            c[0] = j;
            float64 x = v_f32_ld_tnsr_b(c, scores);
            int64 posv = lane + j;
            x = v_f32_sel_less_i32_b(posv, nvalid, x, NEG);
            m = v_f32_max_b(m, x);
        }
        m = REDUCE_MAX64(m, P8, P4, P2, P1);
        // pass 2: sum of exp
        float64 s = 0.f;
        for (int j = 0; j < Lmax; j += 64) {
            c[0] = j;
            float64 x = v_f32_ld_tnsr_b(c, scores);
            int64 posv = lane + j;
            float64 e = v_exp_f32(x - m);
            e = v_f32_sel_less_i32_b(posv, nvalid, e, 0.f);
            s = s + e;
        }
        s = REDUCE64(s, P8, P4, P2, P1);
        float64 inv = v_reciprocal_f32(s);
        // pass 3: normalised probabilities, two f32 vectors -> one bf16 vector (interleaved order, matched by the V cache)
        for (int j = 0; j < Lmax; j += 128) {
            float128 pr;
            c[0] = j;      float64 x0 = v_f32_ld_tnsr_b(c, scores); int64 p0 = lane + j;
            c[0] = j + 64; float64 x1 = v_f32_ld_tnsr_b(c, scores); int64 p1 = lane + (j + 64);
            pr.v1 = v_f32_sel_less_i32_b(p0, nvalid, v_exp_f32(x0 - m) * inv, 0.f);
            pr.v2 = v_f32_sel_less_i32_b(p1, nvalid, v_exp_f32(x1 - m) * inv, 0.f);
            int5 oc = {j, row, g, 0, 0};
            v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pr));
        }
    }
}

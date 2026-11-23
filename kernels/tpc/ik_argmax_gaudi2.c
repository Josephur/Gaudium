// gaudium_ik_argmax: the index of the largest of the first NV values of each column, the LOWEST index among equal maxima: the greedy
// readout of the Inkling 2 family. The unembedding is stored padded to a multiple of 128 rows; the rows beyond the vocabulary the model
// can emit are not candidates. The rule holds for any sign of the logits.
//  in0 x bf16 {V, T}   out0 idx int32 {1, T}.   index space {T}: one member scans one column.   scalar: NV (0: every row of x)
// Per f32 lane: the largest value so far and the block where it first appeared (replaced only by a strictly larger one); then the
// maximum over the lanes and the smallest index among the lanes that hold it. Indices are kept as f32 (exact below 2^24). A value
// that is not a number never wins; a column with nothing finite gives 0.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE_MAX64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE_MIN64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = v_f32_min_b(_a, _r); _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = v_f32_min_b(_a, _r); \
    _a = v_f32_min_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_min_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_min_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_min_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

void main(tensor x, tensor idx, int NV)
{
    const int5 ix = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int64 lane = (int64)read_lane_id_4b_b();
    const float64 lane2 = v_convert_i32_to_f32_b(lane, SW_RHNE) * 2.0f;   // the channel of a lane inside a converted block: 2 lane (first half), 2 lane + 1 (second)
    const int V = (NV > 0 && NV < get_dim_size(x, 0)) ? NV : get_dim_size(x, 0), NB = (V + 127) / 128, rem = V - (NB - 1) * 128;
    const float64 NEG = -3.4e38f, BIG = 1.0e9f;   // below every finite bf16 value; above every index
    for (int t = ix[0]; t < ix[0] + isz[0]; ++t) {
        float64 b1 = NEG, b2 = NEG, j1 = 0.f, j2 = 0.f;
        for (int j = 0; j < NB; ++j) {
            int5 c = {j * 128, t, 0, 0, 0};
            float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x));
            if (j == NB - 1 && rem < 128) { a.v1 = v_f32_sel_less_i32_b(lane, (rem + 1) / 2, a.v1, NEG); a.v2 = v_f32_sel_less_i32_b(lane, rem / 2, a.v2, NEG); }   // the lanes beyond V
            const float64 jf = (float)j;
            j1 = v_f32_sel_grt_f32_b(a.v1, b1, jf, j1); b1 = v_f32_sel_grt_f32_b(a.v1, b1, a.v1, b1);
            j2 = v_f32_sel_grt_f32_b(a.v2, b2, jf, j2); b2 = v_f32_sel_grt_f32_b(a.v2, b2, a.v2, b2);
        }
        const float64 m = REDUCE_MAX64(v_f32_max_b(b1, b2), P8, P4, P2, P1);
        const float64 i1 = v_f32_sel_eq_f32_b(b1, m, j1 * 128.0f + lane2, BIG);
        const float64 i2 = v_f32_sel_eq_f32_b(b2, m, j2 * 128.0f + lane2 + 1.0f, BIG);
        const float64 best = REDUCE_MIN64(v_f32_min_b(i1, i2), P8, P4, P2, P1);
        int5 oc = {0, t, 0, 0, 0};
        v_i32_st_tnsr(oc, idx, v_convert_f32_to_i32_b(best, SW_RHNE, 0, 1, 0));   // every lane holds the index; the tensor takes lane 0
    }
}

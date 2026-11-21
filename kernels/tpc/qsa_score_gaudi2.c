// gaudium_qsa_score: QSA block scores of each query row (docs/notes/engine-qsa.md).
//  in0 isc f32 {NB, 4T}: the MME's per-head dot products, column 4t + h = q_h(row t) . blockkey(b)
//  in1 cursor int32 {1}, in2 cursor_big int32 {1} (mode 1: the big cache's limit; mode 0: any, unread)
//  out0 iss f32 {NB, T}: s(b, t) = ((relu(x0) + relu(x1)) + relu(x2)) + relu(x3), for b < nb(t) of every row that needs a selection
//  (rows whose blocks all fit the budget are skipped: gaudium_qsa_select does not read them)
//  Row t: p = cursor + coff + t; mode 0: nb = (p + 1) / 4, budget = nblk; mode 1: nb = cursor_big / 4, budget = nblk - ((p + 1) / 4 - nb).
//  scalars: coff, mode, nblk.   Index space {ceil(NB / 1024), T}: 1024 blocks of one row per point.
//  The relu-sum order is fixed (the host twin in qsa_ref.hpp uses the same); a zero's sign may differ from the host's, which never
//  matters: the selection compares |s| bit patterns.
void main(tensor isc, tensor cursor, tensor cursor_big, tensor iss, int coff, int mode, int nblk)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const int cb = s_i32_ld_g(gen_addr(zc, cursor_big));
    const float64 z = 0.f;
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        const int p = cur + t;
        const int nb = mode == 0 ? (p + 1) >> 2 : cb >> 2;
        const int K = mode == 0 ? nblk : nblk - (((p + 1) >> 2) - nb);
        if (nb <= K || K <= 0) continue;   // dense row, or nothing to select: select does not read the scores
        for (int ck = idx[0]; ck < idx[0] + isz[0]; ++ck) {
            const int b1 = (ck + 1) * 1024 < nb ? (ck + 1) * 1024 : nb;
            // two 64-block vectors per iteration: 8 loads in flight (blocks past b1 in the last pair are past nb: stored, never read)
            for (int b0 = ck * 1024; b0 < b1; b0 += 128) {
                int5 c = {b0, 4 * t, 0, 0, 0}, d = {b0 + 64, 4 * t, 0, 0, 0};
                float64 x0 = v_f32_ld_tnsr_b(c, isc); c[1] = 4 * t + 1; float64 y0 = v_f32_ld_tnsr_b(d, isc); d[1] = 4 * t + 1;
                float64 x1 = v_f32_ld_tnsr_b(c, isc); c[1] = 4 * t + 2; float64 y1 = v_f32_ld_tnsr_b(d, isc); d[1] = 4 * t + 2;
                float64 x2 = v_f32_ld_tnsr_b(c, isc); c[1] = 4 * t + 3; float64 y2 = v_f32_ld_tnsr_b(d, isc); d[1] = 4 * t + 3;
                float64 x3 = v_f32_ld_tnsr_b(c, isc);                   float64 y3 = v_f32_ld_tnsr_b(d, isc);
                float64 s = ((v_f32_max_b(x0, z) + v_f32_max_b(x1, z)) + v_f32_max_b(x2, z)) + v_f32_max_b(x3, z);
                float64 r = ((v_f32_max_b(y0, z) + v_f32_max_b(y1, z)) + v_f32_max_b(y2, z)) + v_f32_max_b(y3, z);
                int5 o = {b0, t, 0, 0, 0}; v_f32_st_tnsr(o, iss, s); o[0] = b0 + 64; v_f32_st_tnsr(o, iss, r);
            }
        }
    }
}

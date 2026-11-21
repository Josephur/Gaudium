// gaudium_qsa_sp_out: gathered sparse attention core of QSA (docs/notes/flash-next-qsa-sparse.md), step 4 of 4: normalise the PV
// partials and lay them out as the chunk core's combine output (what gaudium_attn_post_f32 reads).
//  in0 ao_part f32 {256, GQ, BT}: P . vg per batch bt = g*T + t (unnormalised)
//  in1 stats f32 {64, 2, GQ, BT} (gaudium_qsa_sp_softmax)   in2 cursor int32 {1}
//  [in3 ao_d f32 {256, GQ*T, NKV}: the dense core's combined output over the first chunk (rows that see <= dlim positions)]
//  out0 ao f32 {256, GQ*T, NKV}: row hl*T + t of group g = ao_part[:, hl, g*T + t] * (1 / s)  (gaudium_attn_combine with one chunk:
//       weight exp(m - m) = 1), or, with in3 and p + 1 <= dlim, a copy of ao_d's row (bit for bit the dense path's result)
//  scalars: T, coff, dlim.   Index space {GQ * T, NKV}.
void main(tensor ao_part, tensor stats, tensor cursor,
#ifdef DENSE0
          tensor ao_d,
#endif
          tensor ao, int T, int coff, int dlim)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    for (int g = idx[1]; g < idx[1] + isz[1]; ++g) {
        for (int row = idx[0]; row < idx[0] + isz[0]; ++row) {
            const int h = row / T, t = row - h * T, bt = g * T + t;
            int5 oc = {0, row, g, 0, 0};
#ifdef DENSE0
            if (cur + t + 1 <= dlim) {
                int5 dc = {0, row, g, 0, 0};
                float64 d0 = v_f32_ld_tnsr_b(dc, ao_d); dc[0] = 64; float64 d1 = v_f32_ld_tnsr_b(dc, ao_d);
                dc[0] = 128; float64 d2 = v_f32_ld_tnsr_b(dc, ao_d); dc[0] = 192; float64 d3 = v_f32_ld_tnsr_b(dc, ao_d);
                v_f32_st_tnsr(oc, ao, d0); oc[0] = 64; v_f32_st_tnsr(oc, ao, d1); oc[0] = 128; v_f32_st_tnsr(oc, ao, d2); oc[0] = 192; v_f32_st_tnsr(oc, ao, d3);
                continue;
            }
#endif
            int5 sc = {0, 1, h, bt, 0};
            const float64 inv = v_reciprocal_f32(v_f32_ld_tnsr_b(sc, stats));
            int5 ac = {0, h, bt, 0, 0};
            float64 o0 = v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 64; float64 o1 = v_f32_ld_tnsr_b(ac, ao_part);
            ac[0] = 128; float64 o2 = v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 192; float64 o3 = v_f32_ld_tnsr_b(ac, ao_part);
            v_f32_st_tnsr(oc, ao, o0 * inv); oc[0] = 64; v_f32_st_tnsr(oc, ao, o1 * inv);
            oc[0] = 128; v_f32_st_tnsr(oc, ao, o2 * inv); oc[0] = 192; v_f32_st_tnsr(oc, ao, o3 * inv);
        }
    }
}

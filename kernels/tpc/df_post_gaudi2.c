// gaudium_df_post: re-layout of the DFlash2 attention output rows {128, 4T, 8} (row (h%4)*T + t of group h/4 = head h) into the
// bf16 row {4096, T} the o_proj GEMM reads (deinterleaving load, interleaving store: natural channel order). index space: 32 heads.
void main(tensor ao, tensor core, int T)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h)
        for (int t = 0; t < T; ++t) {
            int5 ac = {0, (h % 4) * T + t, h / 4, 0, 0};
            float128 o = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ac, ao));
            int5 cc = {128 * h, t, 0, 0, 0};
            v_bf16_st_tnsr(cc, core, v_convert_f32_to_bf16_all_b(o));
        }
}

// gaudium_ga_combine: merge the per-chunk partial outputs of a Gemma 4 attention core (docs/notes/engine-gemma4.md). The arithmetic of
// gaudium_attn_combine for heads of any width the family has; built per head width (-DGA_NV2 = f32 vectors per head row: 4 = 256
// channels, 8 = 512), because the accumulators live in registers.
//  in0 ao_part f32 {64 GA_NV2, R, B}  unnormalised partial outputs, B = G nch, CHUNK-MAJOR b = ch G + g
//  in1 stats   f32 {64, 2, R, B}      per (row, chunk) max m and sum s, 64 equal lanes each
//  out0 ao     f32 {64 GA_NV2, R, G}: ao[:, r, g] = sum_ch w_ch ao_part[:, r, ch G + g] / sum_ch w_ch s_ch,  w_ch = exp(m_ch - max_ch m_ch)
//  nch = 1 (a sliding layer's ring) is the plain 1 / s normalisation. Channel order as stored by the PV GEMM (the V rows' lane order).
//  index space 2-D: dim0 = R, dim1 = G.   scalars: nch
#ifndef GA_NV2
#define GA_NV2 4
#endif
#define EXPD(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f); \
    float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0); \
    float64 _r = v_f32_mac_b(_z, -0.693359375f, _d, 0, 1, 0); _r = v_f32_mac_b(_z, 2.12194440e-4f, _r, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.0013836704893037677f, (float64)0.008374780416488647f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.04166822507977486f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.1666642129421234f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.49999991059303284f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })

void main(tensor ao_part, tensor stats, tensor ao, int nch)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int G = get_dim_size(ao, 2);
    const float NEG = -1e30f;
    for (int g = idx[1]; g < idx[1] + isz[1]; ++g) {
        for (int row = idx[0]; row < idx[0] + isz[0]; ++row) {
            float64 M = NEG;
            for (int ch = 0; ch < nch; ++ch) { int5 sc = {0, 0, row, ch * G + g, 0}; M = v_f32_max_b(M, v_f32_ld_tnsr_b(sc, stats)); }
            float64 S = 0.f, o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#if GA_NV2 == 8
            float64 o4 = 0.f, o5 = 0.f, o6 = 0.f, o7 = 0.f;
#endif
            for (int ch = 0; ch < nch; ++ch) {
                const int b = ch * G + g;
                int5 sc = {0, 0, row, b, 0};
                float64 m = v_f32_ld_tnsr_b(sc, stats); sc[1] = 1; float64 s = v_f32_ld_tnsr_b(sc, stats);
                float64 w = EXPD(m - M);
                S = S + w * s;
                int5 ac = {0, row, b, 0, 0};
                o0 = o0 + w * v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 64;
                o1 = o1 + w * v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 128;
                o2 = o2 + w * v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 192;
                o3 = o3 + w * v_f32_ld_tnsr_b(ac, ao_part);
#if GA_NV2 == 8
                ac[0] = 256; o4 = o4 + w * v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 320;
                o5 = o5 + w * v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 384;
                o6 = o6 + w * v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 448;
                o7 = o7 + w * v_f32_ld_tnsr_b(ac, ao_part);
#endif
            }
            float64 inv = v_reciprocal_f32(S);
            int5 oc = {0, row, g, 0, 0};
            v_f32_st_tnsr(oc, ao, o0 * inv); oc[0] = 64;
            v_f32_st_tnsr(oc, ao, o1 * inv); oc[0] = 128;
            v_f32_st_tnsr(oc, ao, o2 * inv); oc[0] = 192;
            v_f32_st_tnsr(oc, ao, o3 * inv);
#if GA_NV2 == 8
            oc[0] = 256; v_f32_st_tnsr(oc, ao, o4 * inv); oc[0] = 320;
            v_f32_st_tnsr(oc, ao, o5 * inv); oc[0] = 384;
            v_f32_st_tnsr(oc, ao, o6 * inv); oc[0] = 448;
            v_f32_st_tnsr(oc, ao, o7 * inv);
#endif
        }
    }
}

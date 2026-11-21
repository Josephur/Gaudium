// gaudium_ga_dwconv: the causal depthwise convolution over time of a Gemma 4 audio encoder layer's light-convolution block
// (docs/notes/engine-gemma4.md, section 14):
//   y[c, t] = bf16( sum_{i < KW} w[c, i] x[c, t - (KW - 1) + i] )        frames before the first are zero
// f32 accumulation in the reference's tap order, one rounding. A frame's row reads only frames at or before it.
//  in0 x bf16 {C, T}   in1 w bf16 {C, KW} (column i = tap i)   out0 y bf16 {C, T}.   C a multiple of 128.   index space {C / 128, T}
void main(tensor x, tensor w, tensor y)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int KW = get_dim_size(w, 1);
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        float64 a1 = 0.f, a2 = 0.f;
        for (int i = 0; i < KW; ++i) {
            const int j = t - (KW - 1) + i;
            if (j < 0) continue;
            int5 c = {b * 128, j, 0, 0, 0}, wc = {b * 128, i, 0, 0, 0};
            const float128 xv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x));
            const float128 wv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
            a1 = a1 + xv.v1 * wv.v1; a2 = a2 + xv.v2 * wv.v2;
        }
        float128 oo; oo.v1 = a1; oo.v2 = a2;
        int5 oc = {b * 128, t, 0, 0, 0};
        v_bf16_st_tnsr(oc, y, v_convert_f32_to_bf16_all_b(oo));
    }
}

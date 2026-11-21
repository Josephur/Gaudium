// gaudium_ga_post: the combined f32 attention output of a Gemma 4 layer -> the bf16 row the o projection reads (no output gate).
//  in0 ao f32 {128 NV, GQ T, NKV} (row hl T + t of group g = head GQ g + hl), channels in LANE ORDER inside each 128-block (the order
//      of the V rows, see gaudium_ga_prep): a plain f32 load of lanes 0..63 / 64..127 is {even, odd} channels.
//  out0 core bf16 {128 NV NH, T}, natural channel order (the interleaving conversion).
//  index space: NH heads.   scalars: T, GQ, NV
void main(tensor ao, tensor core, int T, int GQ, int NV)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        for (int t = 0; t < T; ++t) {
            for (int b = 0; b < NV; ++b) {
                int5 ac = {128 * b, (h % GQ) * T + t, h / GQ, 0, 0};
                float128 o; o.v1 = v_f32_ld_tnsr_b(ac, ao); ac[0] += 64; o.v2 = v_f32_ld_tnsr_b(ac, ao);
                int5 cc = {128 * (NV * h + b), t, 0, 0, 0};
                v_bf16_st_tnsr(cc, core, v_convert_f32_to_bf16_all_b(o));
            }
        }
    }
}

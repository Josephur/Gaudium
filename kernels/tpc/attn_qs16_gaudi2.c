// gaudium_attn_qs16: the queries of a prefill attention layer scaled by 512 log2 e, for the half-width core (GAUDIUM_PREFILL_S16):
// the QK GEMM then writes its bf16 scores in 1/512 octaves, the unit gaudium_attn_softmax16 splits into exponent and mantissa.
// The multiply is done in f32 (a bf16 multiply would round the constant to 370).
//  in0 qb bf16 {DH, R, B} -> out0 qs bf16 {DH, R, B}.  index space 2-D: dim0 = R, dim1 = B.
void main(tensor qb, tensor qs)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int DHq = get_dim_size(qb, 0);
    for (int b = idx[1]; b < idx[1] + isz[1]; ++b)
        for (int r = idx[0]; r < idx[0] + isz[0]; ++r)
            for (int d = 0; d < DHq; d += 128) {
                int5 c = {d, r, b, 0, 0};
                float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, qb));
                x.v1 = x.v1 * 738.6598609351492f; x.v2 = x.v2 * 738.6598609351492f;
                v_bf16_st_tnsr(c, qs, v_convert_f32_to_bf16_all_b(x));
            }
}

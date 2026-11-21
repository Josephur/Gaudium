// gaudium_ga_clamp: y = min(max(x, lo), hi), bf16 {C, cols} (docs/notes/engine-gemma4.md, section 12).
// The towers of the Gemma 4 E sizes bound the input and the output of every projection of an encoder layer by four stored scalars
// (Gemma4ClippableLinear). One recipe serves every layer of a tower, so the bounds are tensors bound per layer, not node parameters:
// lo and hi are bf16 {128}, the bound repeated in every lane (the loader makes them), read once per index-space member.
// min and max of bf16 values are exact: the result is one of the operands.
//  in0 x bf16 {C, cols}   in1 lo bf16 {128}   in2 hi bf16 {128}   out0 y bf16 {C, cols}.
//  index space {ceil(C / 128), cols}: a last partial block loads padding beyond C, and its lanes beyond C are not stored.
void main(tensor x, tensor lo, tensor hi, tensor y)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 z = {0, 0, 0, 0, 0};
    const bfloat128 vlo = v_bf16_ld_tnsr_b(z, lo), vhi = v_bf16_ld_tnsr_b(z, hi);
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t)
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
        int5 c = {b * 128, t, 0, 0, 0};
        v_bf16_st_tnsr(c, y, v_bf16_min_b(v_bf16_max_b(v_bf16_ld_tnsr_b(c, x), vlo), vhi));
    }
}

// gaudium_ga_copy: y = x, bf16 {C, cols}: a node the graph compiler cannot see through (docs/notes/engine-gemma4.md, section 10).
// It stands after a cast to bf16 at a point where the model rounds. Measured on the Gemma 4 vision tower: when such a cast fed a
// bf16 add or product directly, the compiler let that consumer read the unrounded f32 value (one rounding where the reference has
// two): 23 % of a layer's residual sums and 27 % of its activation products came out one bf16 step away, and 27 layers turned that
// into several percent at the tower's end. A cast followed by this copy is rounded where the graph says.
//  in0 x bf16 {C, cols}   out0 y bf16 {C, cols}.   index space {ceil(C / 128), cols}: a last partial block loads padding beyond C,
//  and its lanes beyond C are not stored (the store stays inside the tensor).
void main(tensor x, tensor y)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t)
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
        int5 c = {b * 128, t, 0, 0, 0};
        v_bf16_st_tnsr(c, y, v_bf16_ld_tnsr_b(c, x));
    }
}

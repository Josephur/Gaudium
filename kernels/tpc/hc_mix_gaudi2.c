// gaudium_hc_mix: the hyper-connection read-out (qwen4_exp; HF Qwen4ExpTextGatedResidual.forward, mixed_input).
//   in0 xn bf16 {S*HID, T}  the per-stream normed residual (gaudium_hc_norm)
//   in1 g  bf16 {S*HID, T}  input_mix_weight_up(silu(input_mix_weight_down(xn) / S)), the up GEMM's output (before the sigmoid)
//   out0 x bf16 {HID, T}    the block input
//   scalar: HID (multiple of 256), S
//   x[c, t] = bf16( (sum over s of p_s) * (1 / S) ),  p_s = bf16( bf16(sigmoid(g[s*HID + c])) * xn[s*HID + c] )
//   (HF: input_mix_weight = sigmoid(...) in bf16; (input_mix_weight * hyper_input_normed) in bf16; .mean(dim=-2): f32 sum, one rounding)
//   index space {HID / 256, T}
#define RBF(a, b) ({ float128 _p; _p.v1 = (a); _p.v2 = (b); v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(_p)); })

void main(tensor xn, tensor g, tensor x, int HID, int S)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const float invS = 1.f / (float)S;
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        float64 a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int s = 0; s < S; ++s) {
            int5 c = {s * HID + b * 256, t, 0, 0, 0};
            bfloat128 xb0 = v_bf16_ld_tnsr_b(c, xn), gb0 = v_bf16_ld_tnsr_b(c, g); c[0] += 128;
            bfloat128 xb1 = v_bf16_ld_tnsr_b(c, xn), gb1 = v_bf16_ld_tnsr_b(c, g);
            float128 x0 = v_convert_bf16_to_f32_all_b(xb0), x1 = v_convert_bf16_to_f32_all_b(xb1);
            float128 g0 = v_convert_bf16_to_f32_all_b(gb0), g1 = v_convert_bf16_to_f32_all_b(gb1);
            float128 s0 = RBF(v_sigmoid_f32(g0.v1), v_sigmoid_f32(g0.v2)), s1 = RBF(v_sigmoid_f32(g1.v1), v_sigmoid_f32(g1.v2));
            float128 p0 = RBF(s0.v1 * x0.v1, s0.v2 * x0.v2), p1 = RBF(s1.v1 * x1.v1, s1.v2 * x1.v2);
            a0 = a0 + p0.v1; a1 = a1 + p0.v2; a2 = a2 + p1.v1; a3 = a3 + p1.v2;
        }
        float128 o0; o0.v1 = a0 * invS; o0.v2 = a1 * invS;
        float128 o1; o1.v1 = a2 * invS; o1.v2 = a3 * invS;
        int5 oc = {b * 256, t, 0, 0, 0};
        v_bf16_st_tnsr(oc, x, v_convert_f32_to_bf16_all_b(o0)); oc[0] += 128;
        v_bf16_st_tnsr(oc, x, v_convert_f32_to_bf16_all_b(o1));
    }
}

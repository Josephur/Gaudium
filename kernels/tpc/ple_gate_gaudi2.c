// gaudium_ple_gate: the n-gram embedding's stream gate (qwen4_exp; HF Qwen4ExpTextPLELayer.forward, gate and gated_value).
//   in0 kn  bf16 {S*HID, T}  norm_key(key_proj(emb))            (gaudium_hc_norm)
//   in1 qn  bf16 {S*HID, T}  norm_query(hidden_states)           (gaudium_hc_norm of the residual)
//   in2 val bf16 {HID, T}    value_proj(emb)
//   out0 gv bf16 {S*HID, T}  gated_value
//   scalar: HID (multiple of 256), rs = 1 / sqrt(HID) as float bits
//   g = bf16( bf16(sum_c bf16(kn * qn)) * rs );  g' = bf16( sqrt(bf16(max(|g|, 1e-6))) ) * sign(g);  gv = bf16( bf16(sigmoid(g')) * val )
//   (HF: (key_normed * query_normed).sum(-1) / sqrt(H), .abs().clamp_min(1e-6).sqrt() * .sign(), sigmoid(gate) * value: bf16 tensors)
//   index space {S, T}: one point = one stream of one row (a full HID reduction, then the scaled copy of the value row)
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define RBF(a, b) ({ float128 _p; _p.v1 = (a); _p.v2 = (b); v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(_p)); })
#define RB1(a) ({ float128 _q = RBF((a), (a)); _q.v1; })

void main(tensor kn, tensor qn, tensor val, tensor gv, int HID, float rs)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const int NB = HID / 256;
    for (int s = idx[0]; s < idx[0] + isz[0]; ++s)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        const int base = s * HID;
        float64 a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < NB; ++b) {
            int5 c = {base + b * 256, t, 0, 0, 0};
            bfloat128 k0b = v_bf16_ld_tnsr_b(c, kn), q0b = v_bf16_ld_tnsr_b(c, qn); c[0] += 128;
            bfloat128 k1b = v_bf16_ld_tnsr_b(c, kn), q1b = v_bf16_ld_tnsr_b(c, qn);
            float128 k0 = v_convert_bf16_to_f32_all_b(k0b), k1 = v_convert_bf16_to_f32_all_b(k1b);
            float128 q0 = v_convert_bf16_to_f32_all_b(q0b), q1 = v_convert_bf16_to_f32_all_b(q1b);
            float128 p0 = RBF(k0.v1 * q0.v1, k0.v2 * q0.v2), p1 = RBF(k1.v1 * q1.v1, k1.v2 * q1.v2);
            a0 = a0 + p0.v1; a1 = a1 + p0.v2; a2 = a2 + p1.v1; a3 = a3 + p1.v2;
        }
        const float64 sum = REDUCE64((a0 + a1) + (a2 + a3), P8, P4, P2, P1);   // every lane
        const float64 g = RB1(RB1(sum) * rs);
        const float64 zero = 0.f, tiny = 1e-6f;
        const float64 mag = RB1(v_sqrt_f32(v_f32_max_b(v_f32_abs_b(g), tiny)));
        const float64 gs = v_f32_sel_less_f32_b(g, zero, -mag, v_f32_sel_grt_f32_b(g, zero, mag, zero));   // sign(g) * mag (sign(0) = 0)
        const float64 sg = RB1(v_sigmoid_f32(gs));
        for (int b = 0; b < NB; ++b) {
            int5 vc = {b * 256, t, 0, 0, 0};
            bfloat128 vb0 = v_bf16_ld_tnsr_b(vc, val); vc[0] += 128; bfloat128 vb1 = v_bf16_ld_tnsr_b(vc, val);
            float128 v0 = v_convert_bf16_to_f32_all_b(vb0), v1 = v_convert_bf16_to_f32_all_b(vb1);
            float128 o0; o0.v1 = sg * v0.v1; o0.v2 = sg * v0.v2;
            float128 o1; o1.v1 = sg * v1.v1; o1.v2 = sg * v1.v2;
            int5 oc = {base + b * 256, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, gv, v_convert_f32_to_bf16_all_b(o0)); oc[0] += 128;
            v_bf16_st_tnsr(oc, gv, v_convert_f32_to_bf16_all_b(o1));
        }
    }
}

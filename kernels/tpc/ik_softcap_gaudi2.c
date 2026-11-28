// gaudium_ik_softcap: logit soft cap + padding-tail masking (-MAX bf16 = 0xFF7F for v >= NV) on a 128-lane grid.
//   y = bf16( bf16(tanh( bf16(x / cap) )) * cap ) when cap > 0, else x; if NV > 0 and v >= NV: 0xFF7F
//   in0 x bf16 {V, T}   out0 y bf16 {V, T}.   V divisible by 128.   index space {V / 128, T}.   scalars: cap_bits (f32), NV (i32)
#include "ik_common.h"
#define QUOT(x_) ({ float64 _x = (x_); float64 _q = _x * rc; float64 _e = v_f32_mac_b(_q, ncap, _x, 0, 1, 0); v_f32_mac_b(_e, rc, _q, 0, 1, 0); })

void main(tensor x, tensor y, int cap_bits, int NV)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const float cap = *(float*)&cap_bits;
    const unsigned ninf_u = 0xFF7F0000u;
    const float ninf = *(const float*)&ninf_u;
    const bool use_cap = cap > 0.0f;
    const float64 rc = use_cap ? (1.0f / cap) : 0.0f, ncap = -cap;
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        int5 c = {b * 128, t, 0, 0, 0};
        const float128 xf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x));
        float128 o;
        if (use_cap) {
            const float128 d = RBF(QUOT(xf.v1), QUOT(xf.v2));
            const float128 h = RBF(TANH(d.v1), TANH(d.v2));
            o.v1 = h.v1 * cap; o.v2 = h.v2 * cap;
        } else {
            o = xf;
        }
        if (NV > 0) {
            const int64 ie = lane * 2 + (int64)(b * 128);
            o.v1 = v_f32_sel_less_i32_b(ie, NV, o.v1, ninf);
            o.v2 = v_f32_sel_less_i32_b(ie + 1, NV, o.v2, ninf);
        }
        v_bf16_st_tnsr(c, y, v_convert_f32_to_bf16_all_b(o));
    }
}

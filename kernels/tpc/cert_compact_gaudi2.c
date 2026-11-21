// gaudium_cert_compact: candidate index lists of the certified screening readout (scalar code).
//  in0 f   f32 {V}        from gaudium_cert_mark (>= 0: candidate)      in1 vf f32 {64, NV} per-vector maxima (lane 0 read)
//  out0 cidx i32 {BS*NM}  member g's candidates in increasing j order in slots [g BS, (g+1) BS) (unused slots = 0, a valid filler)
//  out1 ccnt i32 {NM}     member g's TRUE candidate count (may exceed BS: overflow, only the first BS are kept)
//  member g scans vectors v = g, g + NM, ... (same interleave as gaudium_cert_mark). index space: NM members; scalar NM.
void main(tensor f, tensor vf, tensor cidx, tensor ccnt, int NM)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int V = get_dim_size(f, 0), NV = V / 64, BS = get_dim_size(cidx, 0) / NM;
    for (int g = idx[0]; g < idx[0] + isz[0]; ++g) {
        int cnt = 0;
        for (int v = g; v < NV; v += NM) {
            int5 vc = {0, v, 0, 0, 0};
            float mx = s_f32_ld_g(gen_addr(vc, vf));
            if (mx < 0.0f) continue;
            for (int l = 0; l < 64; ++l) {
                int5 jc = {64 * v + l, 0, 0, 0, 0};
                float x = s_f32_ld_g(gen_addr(jc, f));
                if (x >= 0.0f) { if (cnt < BS) { int5 oc = {g * BS + cnt, 0, 0, 0, 0}; s_i32_st_g(gen_addr(oc, cidx), 64 * v + l); } ++cnt; }
            }
        }
        for (int k = cnt; k < BS; ++k) { int5 oc = {g * BS + k, 0, 0, 0, 0}; s_i32_st_g(gen_addr(oc, cidx), 0); }
        int5 nc = {g, 0, 0, 0, 0}; s_i32_st_g(gen_addr(nc, ccnt), cnt);
    }
}

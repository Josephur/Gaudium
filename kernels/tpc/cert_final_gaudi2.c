// gaudium_cert_final: exact argmax over the certified candidate set (scalar code).
//  in0 Lc   bf16 {BS*NM, M}  exact logits of the gathered candidate rows (same bf16 MME GEMM as the full readout)
//  in1 cidx i32 {BS*NM}     in2 ccnt i32 {NM}
//  out0 tok  i32 {1, M}      argmax over the candidates: largest logit, ties -> smallest vocabulary index (= argmax_fwd_bf16)
//  out1 diag i32 {4, M}      [0] |C| (sum of true counts), [1] overflow (some ccnt > BS: the token is NOT certified),
//                            [2] NaN seen among the candidate logits, [3] number of candidates scanned
//  index space: M points (one row each).
void main(tensor Lc, tensor cidx, tensor ccnt, tensor tok, tensor diag)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int NM = get_dim_size(ccnt, 0), BS = get_dim_size(cidx, 0) / NM;
    for (int m = idx[0]; m < idx[0] + isz[0]; ++m) {
        float best = -3.0e38f; int bi = 0x7FFFFFFF, total = 0, over = 0, nan = 0, scanned = 0, any = 0;
        for (int g = 0; g < NM; ++g) {
            int5 nc = {g, 0, 0, 0, 0}; int cnt = s_i32_ld_g(gen_addr(nc, ccnt));
            total += cnt; if (cnt > BS) { over = 1; cnt = BS; }
            for (int e = 0; e < cnt; ++e) {
                int5 ic = {g * BS + e, 0, 0, 0, 0}; int j = s_i32_ld_g(gen_addr(ic, cidx));
                int5 lc = {g * BS + e, m, 0, 0, 0}; float x = s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(lc, Lc)));
                ++scanned;
                if (x != x) { nan = 1; continue; }
                if (!any || x > best || (x == best && j < bi)) { best = x; bi = j; any = 1; }
            }
        }
        int5 tc = {0, m, 0, 0, 0}; s_i32_st_g(gen_addr(tc, tok), any ? bi : 0);
        int5 dc = {0, m, 0, 0, 0}; s_i32_st_g(gen_addr(dc, diag), total); dc[0] = 1; s_i32_st_g(gen_addr(dc, diag), over);
        dc[0] = 2; s_i32_st_g(gen_addr(dc, diag), nan); dc[0] = 3; s_i32_st_g(gen_addr(dc, diag), scanned);
    }
}

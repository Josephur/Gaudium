// gaudium_accept_tree: speculative acceptance of a draft TREE on the device (model-agnostic tree manager part).
// One TPC, scalar code. General trees: node 0 = root (last emitted token), parent[t] < t, depth[t] = depth[parent[t]] + 1.
// Walk from the root: at accepted node u the target's token is tokens[u]; the first child c of u (lowest row) with ids[c] ==
// tokens[u] is accepted, else the walk stops. The emitted tokens are the accepted drafts and the bonus tokens[u_last], which is
// exactly what plain greedy emits (every accepted draft equals the target's argmax at its parent).
//  in0 ids {N}  in1 tokens {1, N} (verify argmax per row)  in2 cursor {1}  in3 parent {N}  in4 depth {N} (unused by the walk; kept
//      for the interface: shapes whose node order is not depth-sorted)
//  out0 a {1}          row of the last accepted node = checkpoint slot the next step resumes the recurrent state from
//  out1 cursor_out {1} cursor + d + 1 (d = accepted depth)
//  out2 emitted {EM}   accepted draft ids in path order, tokens[a], then -1 ...; emitted[EM-2] = d, emitted[EM-1] = a (host stats)
//  out3 ncommit {1}    d + 1: rows of this step's MTP catch-up window the next step commits (catch-up runs on the path)
//  out4 fix {2F}       (src, dst) KV slot pairs for the next step's attention prep: path node p_i at slot cursor + p_i moves to
//                      cursor + i when p_i != i (ascending i), then (-1, -1)
//  out5 dep {1}        d
//  out6 c_ids {Tc}     tokens[pidx[i]]: the MTP catch-up's input ids (the target's next token after each path row)
//  out7 pidx {Tc}      catch-up row i -> verify row: p_i for i <= d, else i (the spine's continuation, clamped to N-1)
// Tables are loaded once into scalar local memory (3N global loads); the walk itself is local (N <= 32).
// -DRANK build (6 inputs: + rank {N}, for device-planned trees): emitted[EM-3] = rank[a], emitted[EM-4] = the spine length (rows
// 1..S with parent t-1 and rank 0) so the host can account for the plan it did not choose.
__local__ int LI[32];
__local__ int LK[32];
__local__ int LP[32];
__local__ int PATH[33];
#ifdef RANK
void main(tensor ids, tensor tokens, tensor cursor, tensor parent, tensor depth, tensor rank,
          tensor a_out, tensor cursor_out, tensor emitted, tensor ncommit, tensor fix, tensor dep_out, tensor c_ids, tensor pidx_out)
#else
void main(tensor ids, tensor tokens, tensor cursor, tensor parent, tensor depth,
          tensor a_out, tensor cursor_out, tensor emitted, tensor ncommit, tensor fix, tensor dep_out, tensor c_ids, tensor pidx_out)
#endif
{
    int5 c = {0, 0, 0, 0, 0}, ct = {0, 0, 0, 0, 0};
    const int N = get_dim_size(ids, 0);
    const int EM = get_dim_size(emitted, 0);
    const int F = get_dim_size(fix, 0) / 2;
    const int Tc = get_dim_size(c_ids, 0);
    const int cur = s_i32_ld_g(gen_addr(c, cursor));
    for (int t = 0; t < N; ++t) {
        c[0] = t; ct[1] = t;
        LI[t] = s_i32_ld_g(gen_addr(c, ids));
        LK[t] = s_i32_ld_g(gen_addr(ct, tokens));
        LP[t] = s_i32_ld_g(gen_addr(c, parent));
    }
    int u = 0, d = 0; PATH[0] = 0;
    for (int step = 0; step < N; ++step) {
        const int want = LK[u];
        int found = -1;
        for (int t = u + 1; t < N; ++t) { if (found < 0 && LP[t] == u && LI[t] == want) found = t; }
        if (found < 0) break;
        u = found; ++d; PATH[d] = u;
    }
    for (int i = 0; i < EM; ++i) {
        int v = -1;
        if (i < d) v = LI[PATH[i + 1]];
        else if (i == d) v = LK[u];
        if (i == EM - 2) v = d;
        if (i == EM - 1) v = u;
#ifdef RANK
        if (i == EM - 3) { int5 rc = {u, 0, 0, 0, 0}; v = s_i32_ld_g(gen_addr(rc, rank)); }
        if (i == EM - 4) { int S = 0; for (int t = 1; t < N; ++t) { int5 rc = {t, 0, 0, 0, 0}; if (S == t - 1 && LP[t] == t - 1 && s_i32_ld_g(gen_addr(rc, rank)) == 0) S = t; } v = S; }
#endif
        c[0] = i; s_i32_st_g(gen_addr(c, emitted), v);
    }
    int nf = 0;
    for (int i = 1; i <= d; ++i) {
        if (PATH[i] != i && nf < F) {
            c[0] = 2 * nf; s_i32_st_g(gen_addr(c, fix), cur + PATH[i]);
            c[0] = 2 * nf + 1; s_i32_st_g(gen_addr(c, fix), cur + i);
            ++nf;
        }
    }
    for (int f = nf; f < F; ++f) { c[0] = 2 * f; s_i32_st_g(gen_addr(c, fix), -1); c[0] = 2 * f + 1; s_i32_st_g(gen_addr(c, fix), -1); }
    for (int i = 0; i < Tc; ++i) {
        int r = (i <= d) ? PATH[i] : i;
        if (r > N - 1) r = N - 1;
        c[0] = i; s_i32_st_g(gen_addr(c, pidx_out), r); s_i32_st_g(gen_addr(c, c_ids), LK[r]);
    }
    c[0] = 0;
    s_i32_st_g(gen_addr(c, a_out), u);
    s_i32_st_g(gen_addr(c, cursor_out), cur + d + 1);
    s_i32_st_g(gen_addr(c, ncommit), d + 1);
    s_i32_st_g(gen_addr(c, dep_out), d);
}

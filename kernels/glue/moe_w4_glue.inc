// moe_w4_glue.inc: glue for the W4L / W8 routed-expert kernels (feature/flash-next-moe, docs/notes/moe-w4l.md 1). Included by
// gdn_glue.cpp after its helpers (pattern, all_dtype, node_u32, put_elf). Every tensor is mapped whole (allRequired): the kernels
// address experts and rows by data-dependent coordinates.
#define MOE_W4_GUIDS "gaudium_moe_isa", "gaudium_moe_gate", "gaudium_moe_plan", "gaudium_moe_q8", "gaudium_moe_gu", "gaudium_moe_dn", "gaudium_moe_combine", \
    "gaudium_moe_mdq13", "gaudium_moe_mdq2", "gaudium_moe_mswiglu", "gaudium_moe_mgather", "gaudium_moe_mcomb"   /* MME prefill path (docs/notes/flash-next-prefill.md) */

static GlueCodeReturn w4_whole(HabanaKernelParams* p, HabanaKernelInstantiation* out, uint32_t nin, uint32_t nout) {
    if (p->inputTensorNr != nin) { p->inputTensorNr = nin; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (p->outputTensorNr != nout) { p->outputTensorNr = nout; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    for (uint32_t i = 0; i < nin; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
    for (uint32_t i = 0; i < nout; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
    return GLUE_SUCCESS;
}
static uint64_t w4_dim(const Tensor& t, int d) { return d < (int)t.geometry.dims ? t.geometry.maxSizes[d] : 1; }

static GlueCodeReturn moe_w4_glue_(const std::string& g, HabanaKernelParams* p, HabanaKernelInstantiation* out, bool& handled);
static GlueCodeReturn moe_w4_glue(const std::string& g, HabanaKernelParams* p, HabanaKernelInstantiation* out, bool& handled) {
    const GlueCodeReturn r = moe_w4_glue_(g, p, out, handled);
    if (handled && r != GLUE_SUCCESS && getenv("GAUDIUM_W4_GLUE_DEBUG")) {   // which shape did the graph compiler hand us
        fprintf(stderr, "moe_w4 glue %s -> %d:", g.c_str(), (int)r);
        for (uint32_t i = 0; i < p->inputTensorNr; ++i) { fprintf(stderr, " in%u(dt %d){", i, (int)p->inputTensors[i].geometry.dataType); for (uint32_t d = 0; d < p->inputTensors[i].geometry.dims; ++d) fprintf(stderr, "%s%llu", d ? "," : "", (unsigned long long)p->inputTensors[i].geometry.maxSizes[d]); fprintf(stderr, "}"); }
        for (uint32_t i = 0; i < p->outputTensorNr; ++i) { fprintf(stderr, " out%u(dt %d){", i, (int)p->outputTensors[i].geometry.dataType); for (uint32_t d = 0; d < p->outputTensors[i].geometry.dims; ++d) fprintf(stderr, "%s%llu", d ? "," : "", (unsigned long long)p->outputTensors[i].geometry.maxSizes[d]); fprintf(stderr, "}"); }
        fprintf(stderr, "\n"); }
    return r;
}
static GlueCodeReturn moe_w4_glue_(const std::string& g, HabanaKernelParams* p, HabanaKernelInstantiation* out, bool& handled) {
    handled = true;
    if (g == "gaudium_moe_isa") {   // ISA probe (tests/sim/sim_moe_isa.cpp): 3 inputs, 3 outputs, one index point
        GlueCodeReturn r = w4_whole(p, out, 3, 3); if (r != GLUE_SUCCESS) return r;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1; out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_MOEISA_ELF");
    }
    if (g == "gaudium_moe_plan") {   // ids i32 {K, T}, slot_of i32 {E} -> plan i32 {1 + 5 K T}, dev i32 {K, T}
        const TensorDataType in[2] = {DATA_I32, DATA_I32}, ot[2] = {DATA_I32, DATA_I32};
        GlueCodeReturn r = w4_whole(p, out, 2, 2); if (r != GLUE_SUCCESS) return r;
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t K = w4_dim(p->inputTensors[0], 0), T = w4_dim(p->inputTensors[0], 1), E = w4_dim(p->inputTensors[1], 0);
        if (K == 0 || K > 16 || E == 0 || E > 1024) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (w4_dim(p->outputTensors[0], 0) != 1 + 5 * K * T || w4_dim(p->outputTensors[1], 0) != K || w4_dim(p->outputTensors[1], 1) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)K; out->kernel.scalarParams[1] = (uint32_t)T; out->kernel.scalarParams[2] = (uint32_t)E;
        return put_elf(out, "GAUDIUM_W4PLAN_ELF");
    }
    if (g == "gaudium_moe_q8") {   // x bf16 {H, T} -> xq i8 {256, H/32, T}, xs f32 {64, H/32, T}
        const TensorDataType in[1] = {DATA_BF16}, ot[2] = {DATA_I8, DATA_F32};
        GlueCodeReturn r = w4_whole(p, out, 1, 2); if (r != GLUE_SUCCESS) return r;
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t H = w4_dim(p->inputTensors[0], 0), T = w4_dim(p->inputTensors[0], 1);
        if (H == 0 || H % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (w4_dim(p->outputTensors[0], 0) != 256 || w4_dim(p->outputTensors[0], 1) != H / 32 || w4_dim(p->outputTensors[0], 2) != T
            || w4_dim(p->outputTensors[1], 0) != 64 || w4_dim(p->outputTensors[1], 1) != H / 32 || w4_dim(p->outputTensors[1], 2) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = H / 128; out->indexSpaceGeometry[1] = T; out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_W4Q8_ELF");
    }
    if (g == "gaudium_moe_gu" || g == "gaudium_moe_dn") {
        // gu: xq i8 {256, H/32, T}, xs, plan, codes, scales [, lut] -> hq i8 {256, EF/16, K, T}, hs f32 {64, EF/32, K, T}
        // dn: hq i8 {256, EF/16, K, T}, hs ...
        // dn: hq, hs, plan, codes, scales [, lut] -> y f32 {H, K, T}
        // W4L: codes u8 {256, in/2, NT, S} + lut i8 {256} (6 inputs); W8: codes i8 {256, in, NT, S} (5 inputs). node param 0: M cap (default 240)
        const bool gu = g == "gaudium_moe_gu";
        const bool w8 = p->inputTensorNr == 5;
        if (p->inputTensorNr != 5 && p->inputTensorNr != 6) { p->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        const uint32_t nout = gu ? 2 : 1;
        GlueCodeReturn r = w4_whole(p, out, p->inputTensorNr, nout); if (r != GLUE_SUCCESS) return r;
        const TensorDataType in[6] = {DATA_I8, DATA_F32, DATA_I32, w8 ? DATA_I8 : DATA_U8, DATA_F16, DATA_I8};
        const TensorDataType ot_gu[2] = {DATA_I8, DATA_F32}, ot_dn[1] = {DATA_F32};
        if (!all_dtype(p, in, gu ? ot_gu : ot_dn)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& xq = p->inputTensors[0]; const Tensor& codes = p->inputTensors[3]; const Tensor& sc = p->inputTensors[4];
        const uint64_t in_cols = (gu ? 32 : 16) * w4_dim(xq, 1), NT = w4_dim(codes, 2), S = w4_dim(codes, 3);
        uint64_t K, T;
        if (gu) { T = w4_dim(xq, 2); K = w4_dim(p->outputTensors[0], 2);
            if (w4_dim(p->outputTensors[0], 1) != 8 * NT || w4_dim(p->outputTensors[0], 3) != T || w4_dim(p->outputTensors[1], 1) != 4 * NT) return GLUE_INCOMPATIBLE_OUTPUT_SIZE; }
        else { K = w4_dim(xq, 2); T = w4_dim(xq, 3);
            if (w4_dim(p->outputTensors[0], 0) != 256 * NT || w4_dim(p->outputTensors[0], 1) != K || w4_dim(p->outputTensors[0], 2) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE; }
        if (w4_dim(codes, 0) != 256 || w4_dim(codes, 1) != (w8 ? in_cols : in_cols / 2) || w4_dim(sc, 0) != 256 || w4_dim(sc, 1) != in_cols / 32
            || w4_dim(sc, 2) != NT || w4_dim(sc, 3) != S || NT == 0 || K == 0 || K > 16) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!w8 && w4_dim(p->inputTensors[5], 0) != 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint32_t cap = node_u32(p, 0) ? node_u32(p, 0) : 240;
        const uint64_t KT = K * T, M = KT < cap ? KT : cap;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NT * M;
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)NT; out->kernel.scalarParams[1] = (uint32_t)M; out->kernel.scalarParams[2] = (uint32_t)KT;
        return put_elf(out, gu ? (w8 ? "GAUDIUM_W4GU8_ELF" : "GAUDIUM_W4GU_ELF") : (w8 ? "GAUDIUM_W4DN8_ELF" : "GAUDIUM_W4DN_ELF"));
    }
    if (g == "gaudium_moe_gate") {   // dev i32 {K, T}, plan i32 {1 + 5 K T}, cursor i32 {1}, ack i32 {64} [, acc_in i32 {64}] -> devo, plano, st i32 {64}, acc i32 {64}; node params: gate index, max polls
        const bool first = p->inputTensorNr == 4;
        if (p->inputTensorNr != 4 && p->inputTensorNr != 5) { p->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        GlueCodeReturn r = w4_whole(p, out, p->inputTensorNr, 4); if (r != GLUE_SUCCESS) return r;
        const TensorDataType in[5] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32}, ot[4] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t K = w4_dim(p->inputTensors[0], 0), T = w4_dim(p->inputTensors[0], 1); const uint32_t G = node_u32(p, 0);
        if (K == 0 || K > 16 || T == 0 || w4_dim(p->inputTensors[1], 0) != 1 + 5 * K * T || w4_dim(p->inputTensors[2], 0) != 1 || w4_dim(p->inputTensors[3], 0) != 64 || G >= 64 ||
            (!first && w4_dim(p->inputTensors[4], 0) != 64)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (w4_dim(p->outputTensors[0], 0) != K || w4_dim(p->outputTensors[0], 1) != T || w4_dim(p->outputTensors[1], 0) != 1 + 5 * K * T || w4_dim(p->outputTensors[2], 0) != 64 || w4_dim(p->outputTensors[3], 0) != 64) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        out->kernel.paramsNr = 4; out->kernel.scalarParams[0] = (uint32_t)K; out->kernel.scalarParams[1] = (uint32_t)T; out->kernel.scalarParams[2] = G; out->kernel.scalarParams[3] = node_u32(p, 1);
        return put_elf(out, first ? "GAUDIUM_W4GATE0_ELF" : "GAUDIUM_W4GATE_ELF");
    }
    if (g == "gaudium_moe_combine") {   // yd, yh f32 {H, K, T}, w f32 {K, T}, dev i32 {K, T} [, add bf16 {H, T}] -> out bf16 {H, T}
        const bool add = p->inputTensorNr == 5;
        if (p->inputTensorNr != 4 && p->inputTensorNr != 5) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        GlueCodeReturn r = w4_whole(p, out, p->inputTensorNr, 1); if (r != GLUE_SUCCESS) return r;
        const TensorDataType in[5] = {DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t H = w4_dim(p->inputTensors[0], 0), K = w4_dim(p->inputTensors[0], 1), T = w4_dim(p->inputTensors[0], 2);
        if (H % 128 || K == 0 || K > 16 || w4_dim(p->inputTensors[2], 0) != K || w4_dim(p->inputTensors[3], 0) != K) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (w4_dim(p->outputTensors[0], 0) != H || w4_dim(p->outputTensors[0], 1) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = H / 128; out->indexSpaceGeometry[1] = T;
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = (uint32_t)K;
        return put_elf(out, add ? "GAUDIUM_W4COMBA_ELF" : "GAUDIUM_W4COMB_ELF");
    }
    // ---- the MME prefill path (feature/flash-next-prefill, docs/notes/flash-next-prefill.md)
    if (g == "gaudium_moe_mdq13" || g == "gaudium_moe_mdq2") {   // codes, scales [, lut], src i32 {B} -> W bf16 {256 NT, in, B}
        const bool gu = g == "gaudium_moe_mdq13", w8 = p->inputTensorNr == 3;
        if (p->inputTensorNr != 3 && p->inputTensorNr != 4) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        GlueCodeReturn r = w4_whole(p, out, p->inputTensorNr, 1); if (r != GLUE_SUCCESS) return r;
        const TensorDataType in8[3] = {DATA_I8, DATA_F16, DATA_I32}, in4[4] = {DATA_U8, DATA_F16, DATA_I8, DATA_I32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, w8 ? in8 : in4, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& codes = p->inputTensors[0]; const Tensor& sc = p->inputTensors[1]; const Tensor& src = p->inputTensors[w8 ? 2 : 3]; const Tensor& W = p->outputTensors[0];
        const uint64_t in_cols = 32 * w4_dim(sc, 1), NT = w4_dim(codes, 2), B = w4_dim(src, 0);
        if (w4_dim(codes, 0) != 256 || w4_dim(codes, 1) != (w8 ? in_cols : in_cols / 2) || w4_dim(sc, 0) != 256 || w4_dim(sc, 2) != NT || w4_dim(sc, 3) != w4_dim(codes, 3) || NT == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!w8 && w4_dim(p->inputTensors[2], 0) != 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (w4_dim(W, 0) != 256 * NT || w4_dim(W, 1) != in_cols || w4_dim(W, 2) != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = NT; out->indexSpaceGeometry[1] = B; out->kernel.paramsNr = 0;
        return put_elf(out, gu ? (w8 ? "GAUDIUM_MDQ13W8_ELF" : "GAUDIUM_MDQ13_ELF") : (w8 ? "GAUDIUM_MDQ2W8_ELF" : "GAUDIUM_MDQ2_ELF"));
    }
    if (g == "gaudium_moe_mswiglu") {   // G f32 {2 EF, C, B}, src i32 {B} -> M bf16 {EF, C, B}
        const TensorDataType in[2] = {DATA_F32, DATA_I32}, ot[1] = {DATA_BF16};
        GlueCodeReturn r = w4_whole(p, out, 2, 1); if (r != GLUE_SUCCESS) return r;
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& G = p->inputTensors[0]; const Tensor& M = p->outputTensors[0]; const uint64_t B = w4_dim(p->inputTensors[1], 0);
        if (w4_dim(G, 0) % 256 || w4_dim(G, 2) != B) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (w4_dim(M, 0) != w4_dim(G, 0) / 2 || w4_dim(M, 1) != w4_dim(G, 1) || w4_dim(M, 2) != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = B; out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_MSWIGLU_ELF");
    }
    if (g == "gaudium_moe_mgather" || g == "gaudium_moe_mcomb") {   // gather: x {H, T}, pos {K, T} -> xg {H, R}; comb: yb {H, R}, pos, w {K, T}, add {H, T} -> out {H, T}
        const bool ga = g == "gaudium_moe_mgather"; const uint32_t nin = ga ? 2 : 4;
        GlueCodeReturn r = w4_whole(p, out, nin, 1); if (r != GLUE_SUCCESS) return r;
        const TensorDataType ig[2] = {DATA_BF16, DATA_I32}, ic[4] = {DATA_BF16, DATA_I32, DATA_F32, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, ga ? ig : ic, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& pos = p->inputTensors[1]; const uint64_t K = w4_dim(pos, 0), T = w4_dim(pos, 1), H = w4_dim(p->inputTensors[0], 0);
        if (K == 0 || K > 16 || H % 128 || w4_dim(p->outputTensors[0], 0) != H) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (ga ? w4_dim(p->inputTensors[0], 1) != T : (w4_dim(p->inputTensors[2], 0) != K || w4_dim(p->inputTensors[2], 1) != T || w4_dim(p->inputTensors[3], 0) != H || w4_dim(p->inputTensors[3], 1) != T || w4_dim(p->outputTensors[0], 1) != T)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = T; out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = (uint32_t)K;
        return put_elf(out, ga ? "GAUDIUM_MGATHER_ELF" : "GAUDIUM_MCOMB_ELF");
    }
    handled = false;
    return GLUE_NODE_NOT_FOUND;
}

// attn_shape.h: compile-time shape of the full-attention kernels (issue #7).
//   ATT_NKV  key / value heads (kv groups)
//   ATT_GQ   query heads per kv group (GQA): NH = ATT_NKV * ATT_GQ query heads of 256 channels (two 128-lane halves)
// Derived: ATT_NH, ATT_NPT = index points of attn_prep (one per q head, then one per kv head), ATT_GATE = channel of the output gate
// inside qg (q is channels 0 .. 256 NH - 1, the gate follows).
// Defaults: Qwen3.8-27B (24 q heads in 4 kv groups of 6). Other shapes are built as separate ELFs (src/kernels/tpc/kernels.mk), named by
// the suffix _g<NKV>q<GQ>; the engine picks the GUID from the model's dims (qw::Model::attn_variant).
// The FP8 KV scale table L<l>_kv8 {32} keeps 4 slots per kind (q / K / V / P), so ATT_NKV <= 4 in the KV8 builds.
//   ATT_DH   channels per head: 256 (Qwen3.5: two 128-lane halves, partial RoPE on 64 dims, gated output) or 128 (plain Qwen3:
//            one 128-lane row, full RoPE on all 128 dims, no output gate: q_proj carries q only). The 128 builds are the separate
//            GUIDs gaudium_attn128_* (glue), so a shape is never guessed from a width.
//   ATT_GATED 1: qg = [q | output gate] and attn_post applies sigmoid(gate); 0: qg = q (the post is a plain relayout to bf16).
#ifndef ATT_DH
#define ATT_DH 256
#endif
#ifndef ATT_GATED
#define ATT_GATED (ATT_DH == 256)
#endif
#ifndef ATT_NKV
#define ATT_NKV 4
#endif
#ifndef ATT_GQ
#define ATT_GQ 6
#endif
#define ATT_NH (ATT_NKV * ATT_GQ)
#define ATT_NPT (ATT_NH + ATT_NKV)
#define ATT_GATE (ATT_DH * ATT_NH)
#if ATT_NKV > 4 && defined(KV8)
#error "ATT_NKV > 4: the L<l>_kv8 scale table has 4 slots per kind"
#endif
#if ATT_DH != 256 && ATT_DH != 128
#error "ATT_DH must be 256 or 128"
#endif
#if ATT_DH == 128 && defined(KV8)
#error "the FP8 KV builds are 256-channel only (byte layout of one 256-channel row = one fp8 vector)"
#endif

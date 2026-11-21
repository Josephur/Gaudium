// gdn_shape.h: compile-time shape of the gated-DeltaNet (linear attention) kernels (issue #7).
//   GDN_HK   key heads: the q and the k projections have GDN_HK heads of 128 channels each
//   GDN_REP  value heads per key head (GQA repeat): HV = GDN_HK * GDN_REP value heads of 128 channels
// Derived: GDN_HV, GDN_QKV = channels of the q | k | v projections (the gate z follows at channel GDN_QKV of qkvz), GDN_NBLK = 128-channel
// blocks of q | k | v (the prep kernels' index space).
// Defaults: Qwen3.8-27B (16 key heads, 48 value heads). Other shapes are built as separate ELFs (src/kernels/tpc/kernels.mk), named by
// the suffix _k<HK>r<REP>; the engine picks the GUID from the model's dims (qw::Model::gdn_variant).
#ifndef GDN_HK
#define GDN_HK 16
#endif
#ifndef GDN_REP
#define GDN_REP 3
#endif
#define GDN_HV (GDN_HK * GDN_REP)
#define GDN_NBLK (2 * GDN_HK + GDN_HV)
#define GDN_QKV (128 * GDN_NBLK)

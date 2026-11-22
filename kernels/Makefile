# Gaudium Gaudi 2 TPC-C kernels & Synapse GC glue build.
# Run on a host/container with the Intel Gaudi SDK (tpc-clang + /usr/include/habanalabs).
#
#   make -j16              Build all TPC-C kernels + libgaudium_tpc.so + libgaudium_ga.so + libgaudium_st.so
#   make -j16 qwen3        Build the minimal Qwen3-8B kernel subset + libgaudium_tpc.so
#   make glue              Build only the Synapse Graph Compiler glue plugins
#   make clean             Remove built kernel artifacts in $(KDIR)

SHELL      := /bin/bash
ROOT       := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD      ?= $(abspath $(ROOT)/../build)
KDIR       := $(abspath $(BUILD)/kernels)
TPC_SRC    := $(ROOT)/tpc
GLUE_SRC   := $(ROOT)/glue
TPC_CC     ?= tpc-clang
CXX        ?= g++
HABANA_INC ?= /usr/include/habanalabs

.PHONY: all qwen3 glue clean
# kernels.mk (included next) starts with per-ELF rules: without this a bare `make` would build only its first object
.DEFAULT_GOAL := all

include $(TPC_SRC)/kernels.mk

all: $(KERNEL_OBJS) $(KDIR)/libgaudium_tpc.so $(KDIR)/libgaudium_ga.so $(KDIR)/libgaudium_st.so

glue: $(KDIR)/libgaudium_tpc.so $(KDIR)/libgaudium_ga.so $(KDIR)/libgaudium_st.so

QWEN3_OBJS := \
  $(KDIR)/attn128_prep_g8q4_gaudi2.o \
  $(KDIR)/attn128_post_g8q4_gaudi2.o \
  $(KDIR)/attn128_combine_gaudi2.o \
  $(KDIR)/attn128_combinex_gaudi2.o \
  $(KDIR)/attn_softmax_chunk_gaudi2.o \
  $(KDIR)/attn_softmax_chunkx_gaudi2.o \
  $(KDIR)/add_norm_gaudi2.o \
  $(KDIR)/add_normp_gaudi2.o \
  $(KDIR)/zero_gaudi2.o \
  $(KDIR)/samp_bmax_gaudi2.o \
  $(KDIR)/samp_init_gaudi2.o \
  $(KDIR)/samp_part_gaudi2.o \
  $(KDIR)/samp_red_gaudi2.o \
  $(KDIR)/samp_gumbel_gaudi2.o \
  $(KDIR)/samp_final_gaudi2.o

qwen3: $(QWEN3_OBJS) $(KDIR)/libgaudium_tpc.so

$(KDIR):
	@mkdir -p $@

clean:
	rm -rf $(KDIR)

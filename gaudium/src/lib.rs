//! Gaudium — Intel Gaudi 2 Inference Engine (RustyBends v1: Verified Core, Rust Shell).
//!
//! All pure domain state machines, accounting, budgets, and bounds are formally verified
//! in `verified/*.bend` at build time (Bend 2 `--check-only` + Lean `--verdict`) and
//! emitted as zero-overhead pure Rust into `OUT_DIR/verified.rs`.

/// The process allocator: the system allocator, with a per-thread heap count that only a
/// metered task (the PDF reader) turns on ([`heap_meter`]).
#[global_allocator]
static ALLOCATOR: heap_meter::Meter = heap_meter::Meter;

pub mod verified {
    #![allow(dead_code, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/verified.rs"));
}

pub mod cache_ram;
pub mod core;
pub mod engine;
pub mod fp8;
pub mod gguf;
pub mod grammar;
pub mod heap_meter;
pub mod hostmem;
pub mod json;
pub mod kernels;
pub mod live_profile;
pub mod live_report;
pub mod model_spec;
pub mod model_text;
pub mod moe_cache;
pub mod pdf;
pub mod proto;
pub mod safetensors;
pub mod sampler;
pub mod sampler_penalty;
pub mod server;
pub mod signals;
pub mod synapse;
pub mod tls;
pub mod tls_cert;
pub mod tokenizer;
pub mod tree;
pub mod vision;

//! Gaudium — Intel Gaudi 2 Inference Engine (RustyBends v1: Verified Core, Rust Shell).
//!
//! All pure domain state machines, accounting, budgets, and bounds are formally verified
//! in `verified/*.bend` at build time (Bend 2 `--check-only` + Lean `--verdict`) and
//! emitted as zero-overhead pure Rust into `OUT_DIR/verified.rs`.

pub mod verified {
    #![allow(dead_code, unused_parens, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/verified.rs"));
}

pub mod cache_ram;
pub mod core;
pub mod fp8;
pub mod json;
pub mod model_spec;
pub mod moe_cache;
pub mod proto;
pub mod safetensors;
pub mod sampler;
pub mod tree;

//! Intel Gaudi 2 SynapseAI runtime, HBM allocator (`AllocGuard`), centralized graph builder,
//! compiled-recipe cache, and checked launcher.

pub mod device;
pub mod ffi;
pub mod graph;
pub mod ops;
pub mod recipe_cache;

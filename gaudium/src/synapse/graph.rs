//! Centralized SynapseAI graph builder, deterministic recipe fingerprinter, immediate
//! `synGraphDestroy` compiler, and checked launcher (`common.hpp` + `recipe_cache.hpp`).
//!
//! Routing ALL graph construction (tensors, sections, nodes, compiler attributes) through
//! `GraphBuilder` guarantees 100% fingerprint coverage and enforces `synGraphDestroy`
//! immediately after `synGraphCompile` so compiler IR never leaks host RAM (`AGENTS.md` §8).

use std::collections::BTreeMap;
use std::ffi::CString;
use std::sync::Arc;

use crate::core::lifecycle::RecipeGuard;
use crate::synapse::device::HpuDevice;
use crate::synapse::ffi::{
    syn_check, SynDataType, SynDeviceType, SynGeometryType, SynGraphAttribute, SynGraphHandle,
    SynLaunchTensorInfoExt, SynNodeId, SynRecipeHandle, SynRetrievedLaunchTensorInfoExt,
    SynSectionHandle, SynTensor, SynTensorGeometry, SynTensorType, SynapseApi, SynapseError,
    HABANA_DIM_MAX,
    SynUserExecOrder, SynUserProgrammability,
};
use crate::synapse::recipe_cache::{
    verify_launch_bindings, RecipeCache, RetrievedTensorDesc, Sha256,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SectionId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
struct SectionSpec {
    persistent: bool,
    rmw: bool,
}

#[derive(Debug, Clone)]
struct TensorSpec {
    name: String,
    dtype: SynDataType,
    dims: Vec<u64>, // Synapse FCD-first dimension order
    section: Option<(SectionId, u64)>,
    tensor_type: SynTensorType,
    allow_permutation: bool,
}

#[derive(Debug, Clone)]
struct NodeSpec {
    guid: String,
    name: String,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
    params: Vec<u8>,
}

/// One tensor as seen by a graph node (`GraphBuilder::node_io_views`).
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub id: usize,
    pub name: &'a str,
    pub dims: &'a [u64],
    pub bytes: u64,
    pub persistent: bool,
}

/// One graph node with its input / output tensors (`GraphBuilder::node_io_views`).
#[derive(Debug, Clone)]
pub struct NodeIoView<'a> {
    pub guid: &'a str,
    pub name: &'a str,
    pub inputs: Vec<TensorView<'a>>,
    pub outputs: Vec<TensorView<'a>>,
}

/// Pure, host-testable graph description & fingerprinter that compiles into a verified `CompiledRecipe`.
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    recipe_name: String,
    device_type: SynDeviceType,
    sections: Vec<SectionSpec>,
    tensors: Vec<TensorSpec>,
    nodes: Vec<NodeSpec>,
    deps: Vec<(NodeId, NodeId)>,
    graph_attrs: Vec<(SynGraphAttribute, u64)>,
    no_pipeline_mgmt: bool,
    no_persistent_reuse: bool,
    /// Graph-compiler settings applied while this graph compiles (`synConfigurationSet` names and
    /// values, set with the experimental flags on and restored afterwards). Part of the
    /// fingerprint: a setting changes what is compiled.
    gc_settings: Vec<(String, String)>,
    kernel_elf_hashes: Vec<(String, String)>,
    /// User execution order: every node `i` (creation order) is given the compiler's
    /// `synUserExecOrder {group, i}`. `None` leaves the order to the compiler. Part of the
    /// fingerprint.
    user_exec_group: Option<u32>,
    /// Per-node user execution order `(node, group, index)`: the compiler runs the nodes of one
    /// group in index order and leaves nodes of different groups free of each other. Part of the
    /// fingerprint.
    node_exec_orders: Vec<(NodeId, u32, u32)>,
    /// [fp8] The quantisable weight matrices of this graph are `fp8_143` (`engine/fp8_graph.rs`).
    /// A build-time switch only: the tensors and nodes it leads to are what the fingerprint covers.
    fp8_weights: bool,
}

/// Measurement switch: graph-compiler settings for the graphs of this process.
/// `GAUDIUM_GC_SET` holds `NAME=value[,NAME=value...]`; `GAUDIUM_GC_ONLY` limits them to recipes
/// whose name starts with one of its `|`-separated prefixes. They are part of each graph's
/// settings, so its fingerprint.
fn env_gc_settings(recipe_name: &str) -> Vec<(String, String)> {
    static SETS: std::sync::OnceLock<(Vec<(String, String)>, Vec<String>)> =
        std::sync::OnceLock::new();
    let (sets, only) = SETS.get_or_init(|| {
        let sets = std::env::var("GAUDIUM_GC_SET")
            .unwrap_or_default()
            .split(',')
            .filter_map(|kv| kv.trim().split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        let only = std::env::var("GAUDIUM_GC_ONLY")
            .unwrap_or_default()
            .split('|')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(str::to_string)
            .collect();
        (sets, only)
    });
    if !only.is_empty() && !only.iter().any(|p| recipe_name.starts_with(p.as_str())) {
        return Vec::new();
    }
    sets.clone()
}

impl GraphBuilder {
    pub fn new(recipe_name: impl Into<String>) -> Self {
        let recipe_name = recipe_name.into();
        let gc_settings = env_gc_settings(&recipe_name);
        Self {
            recipe_name,
            device_type: SynDeviceType::Gaudi2,
            sections: Vec::new(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            deps: Vec::new(),
            graph_attrs: Vec::new(),
            no_pipeline_mgmt: false,
            no_persistent_reuse: false,
            gc_settings,
            kernel_elf_hashes: Vec::new(),
            user_exec_group: None,
            node_exec_orders: Vec::new(),
            fp8_weights: false,
        }
    }

    /// Asks the graph compiler to execute the nodes in their creation order (execution group
    /// `group`, index = creation index) instead of the order it would pick. The compiler may
    /// still refuse an order that contradicts the data flow; `None` returns the choice to it.
    pub fn set_user_exec_order(&mut self, group: Option<u32>) {
        self.user_exec_group = group;
    }

    /// Puts node `node` at position `index` of execution group `group`: the compiler executes the
    /// nodes of a group in index order (it refuses an order that contradicts the data flow) and
    /// adds no ordering between nodes of different groups. One group per engine class fixes the
    /// order of each engine's queue without making one engine wait for the other.
    pub fn set_node_exec_order(&mut self, node: NodeId, group: u32, index: u32) {
        self.node_exec_orders.push((node, group, index));
    }

    /// [fp8] Selects FP8 (`true`) or BF16 (`false`) for the quantisable weight matrices built next.
    pub fn set_fp8_weights(&mut self, enabled: bool) {
        self.fp8_weights = enabled;
    }

    /// [fp8] Whether the quantisable weight matrices of this graph are built as `fp8_143`.
    pub fn fp8_weights(&self) -> bool {
        self.fp8_weights
    }

    /// Name of a tensor declared in this graph.
    pub fn tensor_name(&self, id: TensorId) -> &str {
        &self.tensors[id.0].name
    }

    /// Finds a persistent tensor of this graph by name.
    pub fn find_persistent_tensor(&self, name: &str) -> Option<TensorId> {
        self.tensors
            .iter()
            .position(|t| t.section.is_some() && t.name == name)
            .map(TensorId)
    }

    /// Finds any tensor of this graph (persistent or intermediate) by name.
    pub fn find_tensor(&self, name: &str) -> Option<TensorId> {
        self.tensors
            .iter()
            .position(|t| t.name == name)
            .map(TensorId)
    }

    /// Nodes created so far.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Puts the nodes in a new creation order: node `order[i]` becomes node `i`. The graph
    /// compiler keeps the creation order of nodes that do not depend on each other, so this is
    /// how a graph with several independent bodies chooses the order its engines' queues get.
    /// `order` must be a permutation of `0..node_count()` (anything else leaves the graph as it is
    /// and returns `false`); the fingerprint follows the new order.
    pub fn reorder_nodes(&mut self, order: &[usize]) -> bool {
        let n = self.nodes.len();
        let mut seen = vec![false; n];
        if order.len() != n
            || order
                .iter()
                .any(|&i| i >= n || std::mem::replace(&mut seen[i], true))
        {
            return false;
        }
        let mut new_index = vec![0usize; n];
        for (new, &old) in order.iter().enumerate() {
            new_index[old] = new;
        }
        let mut old_nodes: Vec<Option<NodeSpec>> = self.nodes.drain(..).map(Some).collect();
        self.nodes = order
            .iter()
            .map(|&old| old_nodes[old].take().expect("permutation"))
            .collect();
        for d in &mut self.deps {
            *d = (NodeId(new_index[d.0 .0]), NodeId(new_index[d.1 .0]));
        }
        for o in &mut self.node_exec_orders {
            o.0 = NodeId(new_index[o.0 .0]);
        }
        true
    }

    /// Name of the recipe this graph compiles to.
    pub fn recipe_name(&self) -> &str {
        &self.recipe_name
    }

    /// Renames the recipe of a graph that a wrapper extends after another builder created it.
    pub fn set_recipe_name(&mut self, recipe_name: impl Into<String>) {
        self.recipe_name = recipe_name.into();
    }

    pub fn set_nopm(&mut self, enabled: bool) {
        self.no_pipeline_mgmt = enabled;
    }

    /// Disables persistent-output buffer reuse in the graph compiler (fixes placement-guard hazards).
    pub fn set_noreuse(&mut self, enabled: bool) {
        self.no_persistent_reuse = enabled;
    }

    /// Compiles this graph with the graph-compiler setting `key` = `value` (see `gc_settings`).
    pub fn set_gc_setting(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.gc_settings.push((key.into(), value.into()));
    }

    pub fn set_graph_attr(&mut self, attr: SynGraphAttribute, value: u64) {
        self.graph_attrs.push((attr, value));
    }

    /// Records a normalized TPC-C kernel ELF hash into this graph's recipe fingerprint.
    pub fn record_kernel_elf_hash(
        &mut self,
        kernel_name: impl Into<String>,
        norm_sha256: impl Into<String>,
    ) {
        self.kernel_elf_hashes
            .push((kernel_name.into(), norm_sha256.into()));
    }

    pub fn add_persistent_section(&mut self, rmw: bool) -> SectionId {
        let id = SectionId(self.sections.len());
        self.sections.push(SectionSpec {
            persistent: true,
            rmw,
        });
        id
    }

    /// Creates a persistent tensor in its own dedicated persistent section (offset 0).
    pub fn add_persistent_tensor(
        &mut self,
        name: impl Into<String>,
        dtype: SynDataType,
        dims: &[u64],
    ) -> TensorId {
        let sec = self.add_persistent_section(false);
        self.add_tensor_in_section(name, dtype, dims, sec, 0)
    }

    /// Returns the existing `TensorId` if a persistent tensor with `name` already exists (verifying
    /// matching dtype & dims), otherwise creates a new dedicated persistent tensor. Prevents
    /// duplicate persistent sections bound to the same HBM address (`synLaunch` status 30).
    pub fn get_or_add_persistent_tensor(
        &mut self,
        name: impl Into<String>,
        dtype: SynDataType,
        dims: &[u64],
    ) -> TensorId {
        let name_str = name.into();
        for (i, t) in self.tensors.iter().enumerate() {
            if t.section.is_some() && t.name == name_str {
                debug_assert_eq!(t.dtype, dtype, "tensor {name_str} dtype mismatch on reuse");
                debug_assert_eq!(
                    t.dims.as_slice(),
                    dims,
                    "tensor {name_str} dims mismatch on reuse"
                );
                return TensorId(i);
            }
        }
        self.add_persistent_tensor(name_str, dtype, dims)
    }

    /// Lists all persistent tensors declared in this graph as `(name, dtype, dims, byte_size)`.
    pub fn persistent_tensor_specs(&self) -> Vec<(String, SynDataType, Vec<u64>, u64)> {
        self.tensors
            .iter()
            .filter(|t| t.section.is_some())
            .map(|t| {
                let elems: u64 = t.dims.iter().copied().product();
                (
                    t.name.clone(),
                    t.dtype,
                    t.dims.clone(),
                    elems.saturating_mul(t.dtype.byte_size()),
                )
            })
            .collect()
    }

    /// Creates a persistent tensor at `byte_offset` inside a shared persistent arena section (V2-ready).
    pub fn add_tensor_in_section(
        &mut self,
        name: impl Into<String>,
        dtype: SynDataType,
        dims: &[u64],
        section: SectionId,
        byte_offset: u64,
    ) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.push(TensorSpec {
            name: name.into(),
            dtype,
            dims: dims.to_vec(),
            section: Some((section, byte_offset)),
            tensor_type: SynTensorType::DataTensor,
            allow_permutation: false,
        });
        id
    }

    /// Creates an intermediate (workspace-allocated) tensor.
    pub fn add_intermediate(
        &mut self,
        name: impl Into<String>,
        dtype: SynDataType,
        dims: &[u64],
    ) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.push(TensorSpec {
            name: name.into(),
            dtype,
            dims: dims.to_vec(),
            section: None,
            tensor_type: SynTensorType::DataTensor,
            allow_permutation: false,
        });
        id
    }

    pub fn add_node(
        &mut self,
        guid: impl Into<String>,
        name: impl Into<String>,
        inputs: &[TensorId],
        outputs: &[TensorId],
        params: &[u8],
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(NodeSpec {
            guid: guid.into(),
            name: name.into(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            params: params.to_vec(),
        });
        id
    }

    pub fn add_dependency(&mut self, blocking: NodeId, blocked: NodeId) {
        self.deps.push((blocking, blocked));
    }

    /// Upper bound on workspace bytes needed by all declared non-persistent intermediate tensors.
    pub fn declared_intermediate_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|t| t.section.is_none())
            .map(|t| {
                let elems: u64 = t.dims.iter().copied().product();
                elems.saturating_mul(t.dtype.byte_size())
            })
            .sum()
    }

    /// Read-only view of every node with the names, dims and byte sizes of its tensors, for
    /// host-side byte / FLOP models (live profiling). Never changes the graph or its fingerprint.
    pub fn node_io_views(&self) -> Vec<NodeIoView<'_>> {
        let view = |id: &TensorId| {
            let t = &self.tensors[id.0];
            let elems: u64 = t.dims.iter().copied().product();
            TensorView {
                id: id.0,
                name: t.name.as_str(),
                dims: t.dims.as_slice(),
                bytes: elems.saturating_mul(t.dtype.byte_size()),
                persistent: t.section.is_some(),
            }
        };
        self.nodes
            .iter()
            .map(|n| NodeIoView {
                guid: n.guid.as_str(),
                name: n.name.as_str(),
                inputs: n.inputs.iter().map(view).collect(),
                outputs: n.outputs.iter().map(view).collect(),
            })
            .collect()
    }

    /// Computes a deterministic SHA-256 fingerprint covering 100% of sections, tensors, nodes,
    /// params, dependencies, compiler attributes, and normalized kernel ELF hashes.
    pub fn fingerprint(&self, driver_version: &str) -> String {
        let mut h = Sha256::new();
        h.update_str("gaudium-rust-graph-fp-v1");
        h.update_str(driver_version);
        h.update_u64(self.device_type as u64);
        h.update_u64(u64::from(self.no_pipeline_mgmt));
        h.update_u64(u64::from(self.no_persistent_reuse));

        h.update_u64(self.graph_attrs.len() as u64);
        for &(a, v) in &self.graph_attrs {
            h.update_u64(a as u64);
            h.update_u64(v);
        }

        h.update_u64(self.sections.len() as u64);
        for s in &self.sections {
            h.update_u64(u64::from(s.persistent));
            h.update_u64(u64::from(s.rmw));
        }

        h.update_u64(self.tensors.len() as u64);
        for t in &self.tensors {
            h.update_str(&t.name);
            h.update_u64(t.dtype as u64);
            h.update_u64(t.tensor_type as u64);
            h.update_u64(u64::from(t.allow_permutation));
            h.update_u64(t.dims.len() as u64);
            for &d in &t.dims {
                h.update_u64(d);
            }
            match t.section {
                None => h.update_u64(0),
                Some((sid, off)) => {
                    h.update_u64(1);
                    h.update_u64(sid.0 as u64);
                    h.update_u64(off);
                }
            }
        }

        h.update_u64(self.nodes.len() as u64);
        for n in &self.nodes {
            h.update_str(&n.guid);
            h.update_str(&n.name);
            h.update_u64(n.inputs.len() as u64);
            for &i in &n.inputs {
                h.update_u64(i.0 as u64);
            }
            h.update_u64(n.outputs.len() as u64);
            for &o in &n.outputs {
                h.update_u64(o.0 as u64);
            }
            h.update_u64(n.params.len() as u64);
            h.update(&n.params);
        }

        h.update_u64(self.deps.len() as u64);
        for &(a, b) in &self.deps {
            h.update_u64(a.0 as u64);
            h.update_u64(b.0 as u64);
        }

        h.update_u64(self.kernel_elf_hashes.len() as u64);
        for (k, d) in &self.kernel_elf_hashes {
            h.update_str(k);
            h.update_str(d);
        }
        if let Some(g) = self.user_exec_group {
            h.update_str("user-exec-order");
            h.update_u64(u64::from(g));
        }
        if !self.node_exec_orders.is_empty() {
            h.update_str("node-exec-orders");
            h.update_u64(self.node_exec_orders.len() as u64);
            for &(n, g, i) in &self.node_exec_orders {
                h.update_u64(n.0 as u64);
                h.update_u64(u64::from(g));
                h.update_u64(u64::from(i));
            }
        }
        // graphs without extra compiler settings keep the keys they had before the field existed
        if !self.gc_settings.is_empty() {
            h.update_str("gc-settings");
            h.update_u64(self.gc_settings.len() as u64);
            for (k, v) in &self.gc_settings {
                h.update_str(k);
                h.update_str(v);
            }
        }
        h.finalize_hex()
    }

    /// Compiles (or loads from `cache`) this graph into a `CompiledRecipe`, immediately calling
    /// `synGraphDestroy` so compiler IR is never retained in host RAM (`AGENTS.md` §8).
    pub fn compile(
        &self,
        api: Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
    ) -> Result<CompiledRecipe, SynapseError> {
        let fp = self.fingerprint(driver_version);

        // 1. Try verified cache hit first
        if let Some(r) = self.load_cached_fp(&api, cache, &fp)? {
            return Ok(r);
        }
        let mut guard = RecipeGuard::new();

        // 2. Cache miss: check GC_KERNEL_PATH before invoking graph compiler so failures are actionable
        if std::env::var("GC_KERNEL_PATH")
            .map(|v| v.trim().is_empty())
            .unwrap_or(true)
        {
            return Err(SynapseError::InvalidArg(
                "GC_KERNEL_PATH is unset; set GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so (plus any custom TPC glue .so)".into(),
            ));
        }

        // Apply compiler settings, construct graph, compile, and DESTROY graph immediately
        let use_exp =
            self.no_pipeline_mgmt || self.no_persistent_reuse || !self.gc_settings.is_empty();
        if use_exp {
            let k = CString::new("ENABLE_EXPERIMENTAL_FLAGS").unwrap();
            let v = CString::new("true").unwrap();
            let _ = unsafe { (api.syn_configuration_set)(k.as_ptr(), v.as_ptr()) };
        }
        if self.no_pipeline_mgmt {
            let k = CString::new("ENABLE_PIPELINE_MANAGEMENT").unwrap();
            let v = CString::new("false").unwrap();
            let _ = unsafe { (api.syn_configuration_set)(k.as_ptr(), v.as_ptr()) };
        }
        if self.no_persistent_reuse {
            let k = CString::new("ENABLE_PERSISTENT_OUTPUT_REUSE").unwrap();
            let v = CString::new("false").unwrap();
            let _ = unsafe { (api.syn_configuration_set)(k.as_ptr(), v.as_ptr()) };
        }

        // extra settings: remember what each one was, so the next compile sees the defaults again
        let mut gc_saved: Vec<(CString, Option<CString>)> = Vec::new();
        for (key, value) in &self.gc_settings {
            let (Ok(k), Ok(v)) = (CString::new(key.as_str()), CString::new(value.as_str())) else {
                continue;
            };
            let mut buf = vec![0u8; 256];
            let st = unsafe {
                (api.syn_configuration_get)(k.as_ptr(), buf.as_mut_ptr().cast(), buf.len() as u64)
            };
            let old = (st == 0)
                .then(|| buf.iter().position(|&b| b == 0).map(|n| buf[..n].to_vec()))
                .flatten()
                .and_then(|b| CString::new(b).ok());
            let st = unsafe { (api.syn_configuration_set)(k.as_ptr(), v.as_ptr()) };
            if st != 0 {
                eprintln!("[gaudium] graph compiler setting {key}={value} was refused (status {st})");
            } else if std::env::var_os("GAUDIUM_GC_TRACE").is_some() {
                eprintln!(
                    "[gaudium] graph compiler setting {key}: {} -> {value} ({})",
                    old.as_ref().map(|c| c.to_string_lossy().into_owned()).unwrap_or_else(|| "?".into()),
                    self.recipe_name
                );
            }
            gc_saved.push((k, old));
        }

        let mut graph: SynGraphHandle = std::ptr::null_mut();
        syn_check("synGraphCreate", unsafe {
            (api.syn_graph_create)(&mut graph, self.device_type as u32)
        })?;

        let compile_result = self.build_and_compile_graph(&api, graph);

        // CRITICAL INVARIANT: ALWAYS destroy the graph immediately after compile (prevents 60+ GB host IR leak!)
        unsafe {
            (api.syn_graph_destroy)(graph);
        }

        // Restore compiler toggles if modified
        for (k, old) in gc_saved.iter().rev() {
            if let Some(v) = old {
                let _ = unsafe { (api.syn_configuration_set)(k.as_ptr(), v.as_ptr()) };
            }
        }
        if self.no_pipeline_mgmt {
            let k = CString::new("ENABLE_PIPELINE_MANAGEMENT").unwrap();
            let v = CString::new("true").unwrap();
            let _ = unsafe { (api.syn_configuration_set)(k.as_ptr(), v.as_ptr()) };
        }
        if self.no_persistent_reuse {
            let k = CString::new("ENABLE_PERSISTENT_OUTPUT_REUSE").unwrap();
            let v = CString::new("true").unwrap();
            let _ = unsafe { (api.syn_configuration_set)(k.as_ptr(), v.as_ptr()) };
        }
        if use_exp {
            let k = CString::new("ENABLE_EXPERIMENTAL_FLAGS").unwrap();
            let v = CString::new("false").unwrap();
            let _ = unsafe { (api.syn_configuration_set)(k.as_ptr(), v.as_ptr()) };
        }

        let handle = compile_result?;
        guard.compile();

        // Save to on-disk recipe cache if enabled
        if cache.is_enabled() {
            let _ = cache.ensure_dir();
            if let Some(rpath) = cache.recipe_path(&self.recipe_name, &fp) {
                cache.mark_in_use(&self.recipe_name, &fp);
                if let Ok(cpath) = CString::new(rpath.to_string_lossy().as_bytes()) {
                    let st = unsafe { (api.syn_recipe_serialize)(handle, cpath.as_ptr()) };
                    if st == 0 {
                        let _ = cache.record_sha256(&self.recipe_name, &fp);
                    }
                }
            }
        }

        CompiledRecipe::from_handle(api, handle, guard, self.recipe_name.clone(), fp, false)
    }

    /// Loads this graph's recipe from `cache` when it is there (verified file), without ever
    /// invoking the graph compiler: `Ok(None)` on a miss. For recipes that are worth having ready
    /// but not worth a compile now.
    pub fn load_cached(
        &self,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        driver_version: &str,
    ) -> Result<Option<CompiledRecipe>, SynapseError> {
        self.load_cached_fp(api, cache, &self.fingerprint(driver_version))
    }

    fn load_cached_fp(
        &self,
        api: &Arc<SynapseApi>,
        cache: &RecipeCache,
        fp: &str,
    ) -> Result<Option<CompiledRecipe>, SynapseError> {
        let Some(cached_path) = cache.lookup_verified(&self.recipe_name, fp) else {
            return Ok(None);
        };
        let cpath = CString::new(cached_path.to_string_lossy().as_bytes())
            .map_err(|e| SynapseError::InvalidArg(e.to_string()))?;
        let mut handle: SynRecipeHandle = std::ptr::null_mut();
        // A trim in another process may remove the file after `lookup_verified` checked it: the
        // load then fails and this is a cache miss (the caller compiles), never a partial read.
        let st = unsafe { (api.syn_recipe_deserialize)(&mut handle, cpath.as_ptr()) };
        if st != 0 || handle.is_null() {
            return Ok(None);
        }
        let mut guard = RecipeGuard::new();
        guard.compile();
        CompiledRecipe::from_handle(
            Arc::clone(api),
            handle,
            guard,
            self.recipe_name.clone(),
            fp.to_owned(),
            true,
        )
        .map(Some)
    }

    fn build_and_compile_graph(
        &self,
        api: &SynapseApi,
        graph: SynGraphHandle,
    ) -> Result<SynRecipeHandle, SynapseError> {
        if !self.graph_attrs.is_empty() {
            let attrs: Vec<u32> = self.graph_attrs.iter().map(|&(a, _)| a as u32).collect();
            let vals: Vec<u64> = self.graph_attrs.iter().map(|&(_, v)| v).collect();
            syn_check("synGraphSetAttribute", unsafe {
                (api.syn_graph_set_attribute)(
                    graph,
                    attrs.as_ptr(),
                    vals.as_ptr(),
                    attrs.len() as u32,
                )
            })?;
        }

        let mut sec_handles: Vec<SynSectionHandle> = Vec::with_capacity(self.sections.len());
        for s in &self.sections {
            let mut sh: SynSectionHandle = std::ptr::null_mut();
            syn_check("synSectionCreate", unsafe {
                (api.syn_section_create)(&mut sh, 0, graph)
            })?;
            syn_check("synSectionSetPersistent", unsafe {
                (api.syn_section_set_persistent)(sh, s.persistent)
            })?;
            if s.rmw {
                syn_check("synSectionSetRMW", unsafe {
                    (api.syn_section_set_rmw)(sh, true)
                })?;
            }
            sec_handles.push(sh);
        }

        let mut tensor_handles: Vec<SynTensor> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            if t.dims.is_empty() || t.dims.len() > HABANA_DIM_MAX {
                return Err(SynapseError::InvalidArg(format!(
                    "tensor {} invalid rank {}",
                    t.name,
                    t.dims.len()
                )));
            }
            let cname = CString::new(t.name.as_str())
                .map_err(|e| SynapseError::InvalidArg(e.to_string()))?;
            let mut th: SynTensor = std::ptr::null_mut();
            syn_check("synTensorHandleCreate", unsafe {
                (api.syn_tensor_handle_create)(&mut th, graph, t.tensor_type as u32, cname.as_ptr())
            })?;

            let mut geom = SynTensorGeometry {
                sizes: [0; HABANA_DIM_MAX],
                dims: t.dims.len() as u32,
            };
            for (i, &d) in t.dims.iter().enumerate() {
                geom.sizes[i] = d;
            }
            syn_check("synTensorSetGeometry", unsafe {
                (api.syn_tensor_set_geometry)(th, &geom, SynGeometryType::MaxSizes as u32)
            })?;

            syn_check("synTensorSetDeviceDataType", unsafe {
                (api.syn_tensor_set_device_data_type)(th, t.dtype as u32)
            })?;

            syn_check("synTensorSetAllowPermutation", unsafe {
                (api.syn_tensor_set_allow_permutation)(th, i8::from(t.allow_permutation))
            })?;

            if let Some((sid, off)) = t.section {
                syn_check("synTensorAssignToSection", unsafe {
                    (api.syn_tensor_assign_to_section)(th, sec_handles[sid.0], off)
                })?;
            }
            tensor_handles.push(th);
        }

        let mut node_ids: Vec<SynNodeId> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let cguid = CString::new(n.guid.as_str())
                .map_err(|e| SynapseError::InvalidArg(e.to_string()))?;
            let cname = CString::new(n.name.as_str())
                .map_err(|e| SynapseError::InvalidArg(e.to_string()))?;
            let ins: Vec<SynTensor> = n.inputs.iter().map(|i| tensor_handles[i.0]).collect();
            let outs: Vec<SynTensor> = n.outputs.iter().map(|o| tensor_handles[o.0]).collect();
            let mut nid: SynNodeId = 0;
            let p_ptr = if n.params.is_empty() {
                std::ptr::null()
            } else {
                n.params.as_ptr() as *const std::ffi::c_void
            };
            syn_check("synNodeCreateWithId", unsafe {
                (api.syn_node_create_with_id)(
                    graph,
                    ins.as_ptr(),
                    outs.as_ptr(),
                    ins.len() as u32,
                    outs.len() as u32,
                    p_ptr,
                    n.params.len() as std::ffi::c_uint,
                    cguid.as_ptr(),
                    cname.as_ptr(),
                    &mut nid,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            })?;
            node_ids.push(nid);
        }

        if self.user_exec_group.is_some() || !self.node_exec_orders.is_empty() {
            let set = api.syn_node_set_user_programmability.ok_or_else(|| {
                SynapseError::InvalidArg(
                    "this SDK has no synNodeSetUserProgrammability (user execution order)".into(),
                )
            })?;
            let whole = self
                .user_exec_group
                .into_iter()
                .flat_map(|g| (0..node_ids.len()).map(move |i| (NodeId(i), g, i as u32)));
            for (node, group, index) in whole.chain(self.node_exec_orders.iter().copied()) {
                let order = SynUserExecOrder {
                    group_id: group,
                    execution_ordered_index: index,
                };
                let up = SynUserProgrammability {
                    user_exec_order: &order,
                    user_bundling: std::ptr::null(),
                };
                syn_check("synNodeSetUserProgrammability", unsafe {
                    set(graph, node_ids[node.0], &up)
                })?;
            }
        }

        for &(blocking, blocked) in &self.deps {
            let b0 = node_ids[blocking.0];
            let b1 = node_ids[blocked.0];
            syn_check("synNodeDependencySet", unsafe {
                (api.syn_node_dependency_set)(graph, &b0, &b1, 1, 1)
            })?;
        }

        let cname = CString::new(self.recipe_name.as_str())
            .map_err(|e| SynapseError::InvalidArg(e.to_string()))?;
        let mut recipe: SynRecipeHandle = std::ptr::null_mut();
        let compile_st = unsafe {
            (api.syn_graph_compile)(&mut recipe, graph, cname.as_ptr(), std::ptr::null())
        };

        // Clean up temporary tensor and section handles before returning (graph itself is destroyed by caller)
        for th in tensor_handles {
            unsafe {
                (api.syn_tensor_destroy)(th);
            }
        }
        for sh in sec_handles {
            unsafe {
                (api.syn_section_destroy)(sh);
            }
        }

        syn_check("synGraphCompile", compile_st)?;
        Ok(recipe)
    }
}

/// Compiled SynapseAI recipe with verified Bend 2 `RecipeGuard` and retrieved launch metadata.
pub struct CompiledRecipe {
    api: Arc<SynapseApi>,
    handle: SynRecipeHandle,
    guard: RecipeGuard,
    pub name: String,
    pub fingerprint: String,
    pub from_cache: bool,
    pub ws_size: u64,
    ws_addr: u64,
    tensors: Vec<RetrievedTensorDesc>,
    raw_infos: Vec<SynRetrievedLaunchTensorInfoExt>,
}

impl CompiledRecipe {
    fn from_handle(
        api: Arc<SynapseApi>,
        handle: SynRecipeHandle,
        guard: RecipeGuard,
        name: String,
        fingerprint: String,
        from_cache: bool,
    ) -> Result<Self, SynapseError> {
        let mut ws_size = 0u64;
        syn_check("synWorkspaceGetSize", unsafe {
            (api.syn_workspace_get_size)(&mut ws_size, handle)
        })?;

        let mut count = 0u32;
        syn_check("synTensorRetrieveLaunchAmount", unsafe {
            (api.syn_tensor_retrieve_launch_amount)(handle, &mut count)
        })?;

        let mut ids = vec![0u64; count as usize];
        if count > 0 {
            syn_check("synTensorRetrieveLaunchIds", unsafe {
                (api.syn_tensor_retrieve_launch_ids)(handle, ids.as_mut_ptr(), count)
            })?;
        }

        let mut raw_infos = vec![SynRetrievedLaunchTensorInfoExt::default(); count as usize];
        for (info, &id) in raw_infos.iter_mut().zip(&ids) {
            info.tensor_id = id;
        }
        if count > 0 {
            syn_check("synTensorRetrieveLaunchInfoByIdExt", unsafe {
                (api.syn_tensor_retrieve_launch_info_by_id_ext)(
                    handle,
                    count,
                    raw_infos.as_mut_ptr(),
                )
            })?;
        }

        let tensors: Vec<RetrievedTensorDesc> = raw_infos
            .iter()
            .map(|r| RetrievedTensorDesc {
                name: r.name_str(),
                tensor_id: r.tensor_id,
                section_id: r.tensor_section_id,
                offset_in_section: r.tensor_offset_in_section,
                is_input: r.is_input != 0,
                dtype: r.tensor_data_type,
                dims: r.tensor_max_size[..(r.tensor_dims as usize).min(HABANA_DIM_MAX)].to_vec(),
            })
            .collect();

        Ok(Self {
            api,
            handle,
            guard,
            name,
            fingerprint,
            from_cache,
            ws_size,
            ws_addr: 0,
            tensors,
            raw_infos,
        })
    }

    pub fn tensors(&self) -> &[RetrievedTensorDesc] {
        &self.tensors
    }

    /// Binds this recipe to the device's pre-planned shared workspace pool (verified via `RecipeGuard`).
    pub fn bind_shared_workspace(&mut self, dev: &HpuDevice) -> Result<(), SynapseError> {
        let (pool_addr, pool_size) = dev.shared_workspace();
        self.guard
            .bind_shared_workspace(self.ws_size, pool_size)
            .map_err(|e| {
                SynapseError::Guard(format!("recipe {} workspace bind failed: {e:?}", self.name))
            })?;
        self.ws_addr = if self.ws_size > 0 { pool_addr } else { 0 };
        Ok(())
    }

    /// Pre-validates a fixed binding map once (`verify_launch_bindings`) and materializes the
    /// `SynLaunchTensorInfoExt` array so hot decode loops launch with zero per-step map lookups
    /// or heap allocations.
    pub fn prepare_launch(
        &self,
        bindings: &BTreeMap<String, u64>,
    ) -> Result<PreparedLaunch, SynapseError> {
        verify_launch_bindings(&self.tensors, bindings, self.ws_size, self.ws_addr).map_err(
            |e| SynapseError::Guard(format!("recipe {} binding check failed: {e:?}", self.name)),
        )?;
        let mut launch_infos: Vec<SynLaunchTensorInfoExt> =
            Vec::with_capacity(self.raw_infos.len());
        for (raw, desc) in self.raw_infos.iter().zip(&self.tensors) {
            let addr = *bindings.get(&desc.name).expect("verified bound");
            launch_infos.push(SynLaunchTensorInfoExt {
                tensor_name: std::ptr::null(),
                tensor_address: addr,
                tensor_type: raw.tensor_type,
                tensor_size: raw.tensor_max_size,
                tensor_id: raw.tensor_id,
            });
        }
        Ok(PreparedLaunch {
            recipe_name: self.name.clone(),
            launch_infos,
        })
    }

    /// Launches a pre-validated `PreparedLaunch` (zero map lookups, zero allocations), still
    /// enforcing live `DeviceGuard` and `RecipeGuard` checks.
    pub fn launch_prepared(
        &self,
        dev: &HpuDevice,
        prepared: &PreparedLaunch,
    ) -> Result<(), SynapseError> {
        dev.guard()
            .check_launch()
            .map_err(|e| SynapseError::Guard(format!("device launch refused: {e:?}")))?;
        self.guard.check_launch().map_err(|e| {
            SynapseError::Guard(format!("recipe {} launch refused: {e:?}", self.name))
        })?;
        debug_assert_eq!(prepared.recipe_name, self.name);
        syn_check("synLaunchWithExternalEventsExt", unsafe {
            (self.api.syn_launch_with_external_events_ext)(
                dev.compute_stream(),
                prepared.launch_infos.as_ptr(),
                prepared.launch_infos.len() as u32,
                self.ws_addr,
                self.handle,
                std::ptr::null_mut(),
                0,
                0,
            )
        })
    }

    /// Checked launch enforcing `DeviceGuard`, `RecipeGuard`, complete bindings, and section exclusivity.
    pub fn launch_checked(
        &self,
        dev: &HpuDevice,
        bindings: &BTreeMap<String, u64>,
    ) -> Result<(), SynapseError> {
        let prepared = self.prepare_launch(bindings)?;
        self.launch_prepared(dev, &prepared)
    }
}

// SAFETY: a `CompiledRecipe` exclusively owns its recipe handle and launch metadata; the
// SynapseAI calls made through it are thread-safe, so the owner may move to another thread.
unsafe impl Send for CompiledRecipe {}

/// One cache lookup: the recipe, `None` on a miss, or the error of a failed load.
pub type CachedRecipe = Result<Option<CompiledRecipe>, SynapseError>;

/// Loads the recipes of `n` graphs from `cache` on up to `threads` threads; `build(i)` describes
/// graph `i`. Entry `i` is `Ok(None)` when that recipe is not in the cache. Nothing is compiled.
///
/// Graph description, fingerprint, file verification and deserialization of one recipe are
/// independent of every other recipe's, so a start-up that takes hundreds of cached recipes does
/// not have to do them one after another.
pub fn load_cached_parallel<F>(
    n: usize,
    threads: usize,
    api: &Arc<SynapseApi>,
    cache: &RecipeCache,
    driver_version: &str,
    build: F,
) -> Vec<CachedRecipe>
where
    F: Fn(usize) -> GraphBuilder + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let slots: Vec<Mutex<Option<CachedRecipe>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let work = || loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        let res = build(i).load_cached(api, cache, driver_version);
        *slots[i].lock().unwrap_or_else(|p| p.into_inner()) = Some(res);
    };
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(work);
            }
        });
    }
    slots
        .into_iter()
        .map(|m| {
            m.into_inner()
                .unwrap_or_else(|p| p.into_inner())
                .unwrap_or(Ok(None))
        })
        .collect()
}

/// Threads for [`load_cached_parallel`] on this host.
pub fn recipe_load_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
}

/// Pre-validated launch descriptor for zero-allocation hot-path recipe launches.
pub struct PreparedLaunch {
    recipe_name: String,
    launch_infos: Vec<SynLaunchTensorInfoExt>,
}

impl Drop for CompiledRecipe {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.api.syn_recipe_destroy)(self.handle);
            }
            self.guard.destroy();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_fingerprint_captures_every_knob_and_intermediate_bytes() {
        let mut g1 = GraphBuilder::new("add_bf16");
        let a = g1.add_persistent_tensor("a", SynDataType::Bf16, &[128]);
        let b = g1.add_persistent_tensor("b", SynDataType::Bf16, &[128]);
        let tmp = g1.add_intermediate("tmp", SynDataType::Bf16, &[128]);
        let out = g1.add_persistent_tensor("out", SynDataType::Bf16, &[128]);
        g1.add_node("add_fwd_bf16", "n0", &[a, b], &[tmp], &[]);
        g1.add_node("add_fwd_bf16", "n1", &[tmp, b], &[out], &[]);
        assert_eq!(g1.declared_intermediate_bytes(), 256);

        let fp1 = g1.fingerprint("1.24.2");
        let fp1_same = g1.fingerprint("1.24.2");
        assert_eq!(fp1, fp1_same);

        // Changing set_noreuse MUST change the fingerprint!
        let mut g2 = g1.clone();
        g2.set_noreuse(true);
        assert_ne!(fp1, g2.fingerprint("1.24.2"));

        // Changing a kernel ELF hash MUST change the fingerprint!
        let mut g3 = g1.clone();
        g3.record_kernel_elf_hash("gdn_f32", "deadbeef");
        assert_ne!(fp1, g3.fingerprint("1.24.2"));
    }

    #[test]
    fn compiler_settings_are_part_of_the_fingerprint() {
        let build = || {
            let mut g = GraphBuilder::new("add_bf16");
            let a = g.add_persistent_tensor("a", SynDataType::Bf16, &[128]);
            let b = g.add_persistent_tensor("b", SynDataType::Bf16, &[128]);
            let out = g.add_persistent_tensor("out", SynDataType::Bf16, &[128]);
            g.add_node("add_fwd_bf16", "n0", &[a, b], &[out], &[]);
            g
        };
        let plain = build().fingerprint("1.24.2");
        let mut g = build();
        g.set_gc_setting("DEFAULT_PIPELINE_DEPTH", "1");
        let one = g.fingerprint("1.24.2");
        assert_ne!(plain, one, "a compiler setting changes what is compiled");
        let mut g = build();
        g.set_gc_setting("DEFAULT_PIPELINE_DEPTH", "2");
        assert_ne!(one, g.fingerprint("1.24.2"), "so does its value");
        let mut g = build();
        g.set_gc_setting("DEFAULT_PIPELINE_DEPTH", "1");
        assert_eq!(one, g.fingerprint("1.24.2"));
    }

    #[test]
    fn reordering_nodes_keeps_dependencies_on_their_nodes() {
        let mut g = GraphBuilder::new("three");
        let a = g.add_persistent_tensor("a", SynDataType::Bf16, &[128]);
        let b = g.add_persistent_tensor("b", SynDataType::Bf16, &[128]);
        let c = g.add_persistent_tensor("c", SynDataType::Bf16, &[128]);
        let d = g.add_persistent_tensor("d", SynDataType::Bf16, &[128]);
        let n0 = g.add_node("add_fwd_bf16", "n0", &[a, a], &[b], &[]);
        let _n1 = g.add_node("add_fwd_bf16", "n1", &[a, a], &[c], &[]);
        let n2 = g.add_node("add_fwd_bf16", "n2", &[a, a], &[d], &[]);
        g.add_dependency(n0, n2);
        assert_eq!(g.node_count(), 3);
        assert!(!g.reorder_nodes(&[0, 1]), "not a permutation of the nodes");
        assert!(!g.reorder_nodes(&[0, 0, 1]));
        assert!(g.reorder_nodes(&[2, 0, 1]));
        let names: Vec<String> = g.node_io_views().iter().map(|n| n.name.to_string()).collect();
        assert_eq!(names, ["n2", "n0", "n1"]);
        // the dependency still runs from n0 to n2, at their new places
        assert_eq!(g.deps, vec![(NodeId(1), NodeId(0))]);
    }
}

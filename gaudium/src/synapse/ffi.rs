//! Intel Gaudi 2 SynapseAI 1.24 C ABI definitions & dynamic `dlopen` loader (`libSynapse.so`).
//!
//! All `#[repr(C)]` struct sizes and field offsets are verified at compile time against the
//! Gaudi 2 SDK 1.24 headers (`/usr/include/habanalabs/synapse_api*.h`). Loading `libSynapse.so`
//! via `dlopen(..., RTLD_NOW | RTLD_LOCAL)` keeps the binary runnable on non-HPU dev hosts for
//! unit tests and prevents C++ symbol interposition with `libSynapse`.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::fmt;
use std::path::Path;

pub const HABANA_DIM_MAX: usize = 25;
pub const ENQUEUE_TENSOR_NAME_MAX_SIZE: usize = 1024;
pub const SYN_INVALID_DEVICE_ID: u32 = 0xFFFF_FFFF;

pub type SynStatus = c_int;
pub type SynDeviceId = u32;
pub type SynNodeId = u64;
pub type SynSectionId = u32;
pub type SynTSize = u64;

pub type SynStreamHandle = *mut c_void;
pub type SynEventHandle = *mut c_void;
pub type SynGraphHandle = *mut c_void;
pub type SynRecipeHandle = *mut c_void;
pub type SynSectionHandle = *mut c_void;
pub type SynTensor = *mut c_void;

// synStatus constants (verified against SynapseAI 1.24 headers)
pub const SYN_SUCCESS: SynStatus = 0;
pub const SYN_INVALID_ARGUMENT: SynStatus = 1;
pub const SYN_OUT_OF_DEVICE_MEMORY: SynStatus = 4;
pub const SYN_BUSY: SynStatus = 22;
pub const SYN_FAIL: SynStatus = 26;

// synDeviceType
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SynDeviceType {
    Gaudi = 2,
    Gaudi2 = 4,
    Gaudi3 = 5,
}

// synTensorType
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SynTensorType {
    DataTensor = 0,
    ShapeTensor = 1,
    HostToDeviceTensor = 6,
}

// synDmaDir
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SynDmaDir {
    HostToDram = 0,
    DramToHost = 1,
    DramToDram = 2,
}

// synDataType (verified exact discriminants from SynapseAI 1.24)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SynDataType {
    Int8 = 1,
    Bf16 = 2,
    Float = 4,
    Int32 = 16,
    Uint8 = 32,
    Fp16 = 256,
    Uint32 = 1024,
    Fp8_143 = 8192,
    Fp8_152 = 16384,
    Int64 = 32768,
    Uint64 = 65536,
}

impl SynDataType {
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::Int8 | Self::Uint8 | Self::Fp8_143 | Self::Fp8_152 => 1,
            Self::Bf16 | Self::Fp16 => 2,
            Self::Float | Self::Int32 | Self::Uint32 => 4,
            Self::Int64 | Self::Uint64 => 8,
        }
    }
}

// synGeometryType
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SynGeometryType {
    MinSizes = 0,
    MaxSizes = 1,
    Dims = 2,
}

// synGraphAttribute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SynGraphAttribute {
    Inference = 0,
    Quantization = 1,
    BackoffFactor = 2,
    IsHwAlignedScale = 3,
    ScaleMethodHashId = 4,
    UseDynamicQuantization = 5,
}

// synRecipeAttribute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SynRecipeAttribute {
    WorkspaceSize = 0,
    NumPersistentTensors = 1,
    HostMemSize = 2,
    NumExternalTensors = 3,
    PersistentTensorsSize = 4,
    ConstSectionsSize = 5,
    DeviceMemorySize = 6,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SynTensorGeometry {
    pub sizes: [SynTSize; HABANA_DIM_MAX],
    pub dims: u32,
}

impl Default for SynTensorGeometry {
    fn default() -> Self {
        Self {
            sizes: [0; HABANA_DIM_MAX],
            dims: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SynTensorDeviceFullLayout {
    pub strides: [u32; HABANA_DIM_MAX],
    pub device_data_type: u32,
}

impl Default for SynTensorDeviceFullLayout {
    fn default() -> Self {
        Self {
            strides: [0; HABANA_DIM_MAX],
            device_data_type: SynDataType::Bf16 as u32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SynLaunchTensorInfoExt {
    pub tensor_name: *const c_char,
    pub tensor_address: u64,
    pub tensor_type: u32,
    pub tensor_size: [SynTSize; HABANA_DIM_MAX],
    pub tensor_id: u64,
}

impl Default for SynLaunchTensorInfoExt {
    fn default() -> Self {
        Self {
            tensor_name: std::ptr::null(),
            tensor_address: 0,
            tensor_type: SynTensorType::DataTensor as u32,
            tensor_size: [0; HABANA_DIM_MAX],
            tensor_id: 0,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SynRetrievedLaunchTensorInfoExt {
    pub tensor_name: [c_char; ENQUEUE_TENSOR_NAME_MAX_SIZE],
    pub tensor_id: u64,
    pub tensor_type: u32,
    pub tensor_data_type: u32,
    pub tensor_dims: u32,
    pub tensor_max_size: [SynTSize; HABANA_DIM_MAX],
    pub tensor_min_size: [SynTSize; HABANA_DIM_MAX],
    pub tensor_permutation: [u8; HABANA_DIM_MAX],
    pub tensor_section_id: SynSectionId,
    pub tensor_offset_in_section: u64,
    pub is_input: u8,
    pub reserved: [u32; 10],
}

impl Default for SynRetrievedLaunchTensorInfoExt {
    fn default() -> Self {
        Self {
            tensor_name: [0; ENQUEUE_TENSOR_NAME_MAX_SIZE],
            tensor_id: 0,
            tensor_type: 0,
            tensor_data_type: 0,
            tensor_dims: 0,
            tensor_max_size: [0; HABANA_DIM_MAX],
            tensor_min_size: [0; HABANA_DIM_MAX],
            tensor_permutation: [0; HABANA_DIM_MAX],
            tensor_section_id: 0,
            tensor_offset_in_section: 0,
            is_input: 0,
            reserved: [0; 10],
        }
    }
}

impl SynRetrievedLaunchTensorInfoExt {
    pub fn name_str(&self) -> String {
        // SAFETY: `tensor_name` is a fixed NUL-terminated buffer from Synapse (clamp to buffer len).
        let bytes: Vec<u8> = self
            .tensor_name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// Compile-time ABI size & offset assertions verified directly against SynapseAI 1.24 headers!
const _: () = {
    assert!(std::mem::size_of::<SynTensorGeometry>() == 208);
    assert!(std::mem::offset_of!(SynTensorGeometry, sizes) == 0);
    assert!(std::mem::offset_of!(SynTensorGeometry, dims) == 200);

    assert!(std::mem::size_of::<SynTensorDeviceFullLayout>() == 104);
    assert!(std::mem::offset_of!(SynTensorDeviceFullLayout, strides) == 0);
    assert!(std::mem::offset_of!(SynTensorDeviceFullLayout, device_data_type) == 100);

    assert!(std::mem::size_of::<SynLaunchTensorInfoExt>() == 232);
    assert!(std::mem::offset_of!(SynLaunchTensorInfoExt, tensor_name) == 0);
    assert!(std::mem::offset_of!(SynLaunchTensorInfoExt, tensor_address) == 8);
    assert!(std::mem::offset_of!(SynLaunchTensorInfoExt, tensor_type) == 16);
    assert!(std::mem::offset_of!(SynLaunchTensorInfoExt, tensor_size) == 24);
    assert!(std::mem::offset_of!(SynLaunchTensorInfoExt, tensor_id) == 224);

    assert!(std::mem::size_of::<SynRetrievedLaunchTensorInfoExt>() == 1536);
    assert!(std::mem::offset_of!(SynRetrievedLaunchTensorInfoExt, tensor_id) == 1024);
    assert!(std::mem::offset_of!(SynRetrievedLaunchTensorInfoExt, tensor_type) == 1032);
    assert!(std::mem::offset_of!(SynRetrievedLaunchTensorInfoExt, tensor_data_type) == 1036);
    assert!(std::mem::offset_of!(SynRetrievedLaunchTensorInfoExt, tensor_dims) == 1040);
    assert!(std::mem::offset_of!(SynRetrievedLaunchTensorInfoExt, tensor_section_id) == 1476);
    assert!(
        std::mem::offset_of!(SynRetrievedLaunchTensorInfoExt, tensor_offset_in_section) == 1480
    );
    assert!(std::mem::offset_of!(SynRetrievedLaunchTensorInfoExt, is_input) == 1488);
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynapseError {
    LibraryLoad(String),
    MissingSymbol(String),
    Status { op: &'static str, code: SynStatus },
    Guard(String),
    InvalidArg(String),
}

impl fmt::Display for SynapseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryLoad(m) => write!(f, "libSynapse load error: {m}"),
            Self::MissingSymbol(s) => write!(f, "libSynapse missing symbol: {s}"),
            Self::Status { op, code } => write!(f, "Synapse {op} failed with synStatus {code}"),
            Self::Guard(m) => write!(f, "Synapse safety guard violation: {m}"),
            Self::InvalidArg(m) => write!(f, "Synapse invalid argument: {m}"),
        }
    }
}

impl std::error::Error for SynapseError {}

#[inline]
pub fn syn_check(op: &'static str, st: SynStatus) -> Result<(), SynapseError> {
    if st == SYN_SUCCESS {
        Ok(())
    } else {
        Err(SynapseError::Status { op, code: st })
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

/// Dynamically loaded Intel SynapseAI C API table.
pub struct SynapseApi {
    handle: *mut c_void,
    pub syn_initialize: unsafe extern "C" fn() -> SynStatus,
    pub syn_destroy: unsafe extern "C" fn() -> SynStatus,
    pub syn_driver_get_version: unsafe extern "C" fn(*mut c_char, c_int) -> SynStatus,
    pub syn_device_acquire_by_type: unsafe extern "C" fn(*mut SynDeviceId, u32) -> SynStatus,
    pub syn_device_release: unsafe extern "C" fn(SynDeviceId) -> SynStatus,
    pub syn_device_get_memory_info:
        unsafe extern "C" fn(SynDeviceId, *mut u64, *mut u64) -> SynStatus,
    pub syn_device_malloc: unsafe extern "C" fn(SynDeviceId, u64, u64, u32, *mut u64) -> SynStatus,
    pub syn_device_free: unsafe extern "C" fn(SynDeviceId, u64, u32) -> SynStatus,
    pub syn_host_map: unsafe extern "C" fn(SynDeviceId, u64, *const c_void) -> SynStatus,
    pub syn_host_unmap: unsafe extern "C" fn(SynDeviceId, *const c_void) -> SynStatus,
    pub syn_host_malloc: unsafe extern "C" fn(SynDeviceId, u64, u32, *mut *mut c_void) -> SynStatus,
    pub syn_host_free: unsafe extern "C" fn(SynDeviceId, *const c_void, u32) -> SynStatus,
    pub syn_stream_create_generic:
        unsafe extern "C" fn(*mut SynStreamHandle, SynDeviceId, u32) -> SynStatus,
    pub syn_stream_destroy: unsafe extern "C" fn(SynStreamHandle) -> SynStatus,
    pub syn_stream_synchronize: unsafe extern "C" fn(SynStreamHandle) -> SynStatus,
    pub syn_stream_wait_event:
        unsafe extern "C" fn(SynStreamHandle, SynEventHandle, u32) -> SynStatus,
    pub syn_event_create: unsafe extern "C" fn(*mut SynEventHandle, SynDeviceId, u32) -> SynStatus,
    pub syn_event_destroy: unsafe extern "C" fn(SynEventHandle) -> SynStatus,
    pub syn_event_record: unsafe extern "C" fn(SynEventHandle, SynStreamHandle) -> SynStatus,
    pub syn_event_synchronize: unsafe extern "C" fn(SynEventHandle) -> SynStatus,
    pub syn_event_elapsed_time:
        unsafe extern "C" fn(*mut u64, SynEventHandle, SynEventHandle) -> SynStatus,
    pub syn_mem_copy_async: unsafe extern "C" fn(SynStreamHandle, u64, u64, u64, u32) -> SynStatus,
    /// `synMemsetD32Async(device address, value, number of 32-bit elements, stream)`.
    pub syn_memset_d32_async: unsafe extern "C" fn(u64, u32, usize, SynStreamHandle) -> SynStatus,
    pub syn_graph_create: unsafe extern "C" fn(*mut SynGraphHandle, u32) -> SynStatus,
    pub syn_graph_destroy: unsafe extern "C" fn(SynGraphHandle) -> SynStatus,
    pub syn_graph_compile: unsafe extern "C" fn(
        *mut SynRecipeHandle,
        SynGraphHandle,
        *const c_char,
        *const c_char,
    ) -> SynStatus,
    pub syn_graph_set_attribute:
        unsafe extern "C" fn(SynGraphHandle, *const u32, *const u64, u32) -> SynStatus,
    pub syn_section_create:
        unsafe extern "C" fn(*mut SynSectionHandle, u64, SynGraphHandle) -> SynStatus,
    pub syn_section_set_persistent: unsafe extern "C" fn(SynSectionHandle, bool) -> SynStatus,
    pub syn_section_set_rmw: unsafe extern "C" fn(SynSectionHandle, bool) -> SynStatus,
    pub syn_section_destroy: unsafe extern "C" fn(SynSectionHandle) -> SynStatus,
    pub syn_tensor_handle_create:
        unsafe extern "C" fn(*mut SynTensor, SynGraphHandle, u32, *const c_char) -> SynStatus,
    pub syn_tensor_assign_to_section:
        unsafe extern "C" fn(SynTensor, SynSectionHandle, u64) -> SynStatus,
    pub syn_tensor_set_geometry:
        unsafe extern "C" fn(SynTensor, *const SynTensorGeometry, u32) -> SynStatus,
    pub syn_tensor_set_device_data_type: unsafe extern "C" fn(SynTensor, u32) -> SynStatus,
    pub syn_tensor_set_device_full_layout:
        unsafe extern "C" fn(SynTensor, *const SynTensorDeviceFullLayout) -> SynStatus,
    pub syn_tensor_set_allow_permutation: unsafe extern "C" fn(SynTensor, i8) -> SynStatus,
    pub syn_tensor_destroy: unsafe extern "C" fn(SynTensor) -> SynStatus,
    pub syn_node_create_with_id: unsafe extern "C" fn(
        SynGraphHandle,
        *const SynTensor,
        *const SynTensor,
        u32,
        u32,
        *const c_void,
        c_uint,
        *const c_char,
        *const c_char,
        *mut SynNodeId,
        *const *const c_char,
        *const *const c_char,
    ) -> SynStatus,
    pub syn_node_dependency_set: unsafe extern "C" fn(
        SynGraphHandle,
        *const SynNodeId,
        *const SynNodeId,
        u32,
        u32,
    ) -> SynStatus,
    /// `synNodeSetUserProgrammability` (absent from older SDKs: `None` there).
    pub syn_node_set_user_programmability: Option<
        unsafe extern "C" fn(SynGraphHandle, SynNodeId, *const SynUserProgrammability) -> SynStatus,
    >,
    pub syn_recipe_serialize: unsafe extern "C" fn(SynRecipeHandle, *const c_char) -> SynStatus,
    pub syn_recipe_deserialize:
        unsafe extern "C" fn(*mut SynRecipeHandle, *const c_char) -> SynStatus,
    pub syn_recipe_destroy: unsafe extern "C" fn(SynRecipeHandle) -> SynStatus,
    pub syn_recipe_get_attribute:
        unsafe extern "C" fn(*mut u64, *const u32, c_uint, SynRecipeHandle) -> SynStatus,
    pub syn_workspace_get_size: unsafe extern "C" fn(*mut u64, SynRecipeHandle) -> SynStatus,
    pub syn_tensor_retrieve_launch_amount:
        unsafe extern "C" fn(SynRecipeHandle, *mut u32) -> SynStatus,
    pub syn_tensor_retrieve_launch_ids:
        unsafe extern "C" fn(SynRecipeHandle, *mut u64, u32) -> SynStatus,
    pub syn_tensor_retrieve_launch_info_by_id_ext: unsafe extern "C" fn(
        SynRecipeHandle,
        u32,
        *mut SynRetrievedLaunchTensorInfoExt,
    ) -> SynStatus,
    pub syn_launch_with_external_events_ext: unsafe extern "C" fn(
        SynStreamHandle,
        *const SynLaunchTensorInfoExt,
        u32,
        u64,
        SynRecipeHandle,
        *mut SynEventHandle,
        u32,
        u32,
    ) -> SynStatus,
    pub syn_configuration_set: unsafe extern "C" fn(*const c_char, *const c_char) -> SynStatus,
    pub syn_configuration_get: unsafe extern "C" fn(*const c_char, *mut c_char, u64) -> SynStatus,
}

// SAFETY: Synapse C API function pointers are thread-safe to share across threads (device ops serialized per stream).
unsafe impl Send for SynapseApi {}
unsafe impl Sync for SynapseApi {}

/// `synUserExecOrder`: position `executionOrderedIndex` of a node inside execution group `groupId`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SynUserExecOrder {
    pub group_id: c_uint,
    pub execution_ordered_index: c_uint,
}

/// `synUserBundling`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SynUserBundling {
    pub bundle_user_index: c_uint,
}

/// `synUserProgrammability`: either pointer may be null.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SynUserProgrammability {
    pub user_exec_order: *const SynUserExecOrder,
    pub user_bundling: *const SynUserBundling,
}

unsafe fn load_sym<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, SynapseError> {
    let cname = CString::new(name).map_err(|_| SynapseError::MissingSymbol(name.to_owned()))?;
    // SAFETY: Caller supplies valid `dlopen` handle and NUL-terminated symbol name.
    let ptr = unsafe { dlsym(handle, cname.as_ptr()) };
    if ptr.is_null() {
        return Err(SynapseError::MissingSymbol(name.to_owned()));
    }
    // SAFETY: `T` is a C function pointer whose signature matches `synapse_api.h`.
    Ok(unsafe { std::mem::transmute_copy(&ptr) })
}

impl SynapseApi {
    /// Loads `libSynapse.so` from `GAUDIUM_SYNAPSE_LIB`, `/usr/lib/habanalabs/libSynapse.so`, or `LD_LIBRARY_PATH`.
    pub fn load_default() -> Result<Self, SynapseError> {
        let mut candidates = Vec::new();
        if let Ok(env_path) = std::env::var("GAUDIUM_SYNAPSE_LIB") {
            if !env_path.trim().is_empty() {
                candidates.push(env_path);
            }
        }
        candidates.push("/usr/lib/habanalabs/libSynapse.so".to_owned());
        candidates.push("libSynapse.so".to_owned());

        let mut last_err = String::new();
        for cand in &candidates {
            if cand.starts_with('/') && !Path::new(cand).exists() {
                last_err = format!("{cand} does not exist");
                continue;
            }
            match Self::load_from(cand) {
                Ok(api) => return Ok(api),
                Err(SynapseError::LibraryLoad(e)) => last_err = e,
                Err(e) => return Err(e),
            }
        }
        Err(SynapseError::LibraryLoad(last_err))
    }

    pub fn load_from(path: &str) -> Result<Self, SynapseError> {
        let cpath = CString::new(path).map_err(|e| SynapseError::LibraryLoad(e.to_string()))?;
        // SAFETY: Valid C string passed to POSIX `dlopen` with `RTLD_NOW | RTLD_LOCAL`.
        let handle = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            // SAFETY: `dlerror()` returns a valid C string or NULL after failed `dlopen`.
            let msg = unsafe {
                let err = dlerror();
                if err.is_null() {
                    format!("dlopen({path}) failed")
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned()
                }
            };
            return Err(SynapseError::LibraryLoad(msg));
        }

        let res = unsafe {
            Ok(Self {
                handle,
                syn_initialize: load_sym(handle, "synInitialize")?,
                syn_destroy: load_sym(handle, "synDestroy")?,
                syn_driver_get_version: load_sym(handle, "synDriverGetVersion")?,
                syn_device_acquire_by_type: load_sym(handle, "synDeviceAcquireByDeviceType")?,
                syn_device_release: load_sym(handle, "synDeviceRelease")?,
                syn_device_get_memory_info: load_sym(handle, "synDeviceGetMemoryInfo")?,
                syn_device_malloc: load_sym(handle, "synDeviceMalloc")?,
                syn_device_free: load_sym(handle, "synDeviceFree")?,
                syn_host_map: load_sym(handle, "synHostMap")?,
                syn_host_unmap: load_sym(handle, "synHostUnmap")?,
                syn_host_malloc: load_sym(handle, "synHostMalloc")?,
                syn_host_free: load_sym(handle, "synHostFree")?,
                syn_stream_create_generic: load_sym(handle, "synStreamCreateGeneric")?,
                syn_stream_destroy: load_sym(handle, "synStreamDestroy")?,
                syn_stream_synchronize: load_sym(handle, "synStreamSynchronize")?,
                syn_stream_wait_event: load_sym(handle, "synStreamWaitEvent")?,
                syn_event_create: load_sym(handle, "synEventCreate")?,
                syn_event_destroy: load_sym(handle, "synEventDestroy")?,
                syn_event_record: load_sym(handle, "synEventRecord")?,
                syn_event_synchronize: load_sym(handle, "synEventSynchronize")?,
                syn_event_elapsed_time: load_sym(handle, "synEventElapsedTime")?,
                syn_mem_copy_async: load_sym(handle, "synMemCopyAsync")?,
                syn_memset_d32_async: load_sym(handle, "synMemsetD32Async")?,
                syn_graph_create: load_sym(handle, "synGraphCreate")?,
                syn_graph_destroy: load_sym(handle, "synGraphDestroy")?,
                syn_graph_compile: load_sym(handle, "synGraphCompile")?,
                syn_graph_set_attribute: load_sym(handle, "synGraphSetAttribute")?,
                syn_section_create: load_sym(handle, "synSectionCreate")?,
                syn_section_set_persistent: load_sym(handle, "synSectionSetPersistent")?,
                syn_section_set_rmw: load_sym(handle, "synSectionSetRMW")?,
                syn_section_destroy: load_sym(handle, "synSectionDestroy")?,
                syn_tensor_handle_create: load_sym(handle, "synTensorHandleCreate")?,
                syn_tensor_assign_to_section: load_sym(handle, "synTensorAssignToSection")?,
                syn_tensor_set_geometry: load_sym(handle, "synTensorSetGeometry")?,
                syn_tensor_set_device_data_type: load_sym(handle, "synTensorSetDeviceDataType")?,
                syn_tensor_set_device_full_layout: load_sym(
                    handle,
                    "synTensorSetDeviceFullLayout",
                )?,
                syn_tensor_set_allow_permutation: load_sym(handle, "synTensorSetAllowPermutation")?,
                syn_tensor_destroy: load_sym(handle, "synTensorDestroy")?,
                syn_node_create_with_id: load_sym(handle, "synNodeCreateWithId")?,
                syn_node_dependency_set: load_sym(handle, "synNodeDependencySet")?,
                syn_node_set_user_programmability: load_sym(
                    handle,
                    "synNodeSetUserProgrammability",
                )
                .ok(),
                syn_recipe_serialize: load_sym(handle, "synRecipeSerialize")?,
                syn_recipe_deserialize: load_sym(handle, "synRecipeDeSerialize")?,
                syn_recipe_destroy: load_sym(handle, "synRecipeDestroy")?,
                syn_recipe_get_attribute: load_sym(handle, "synRecipeGetAttribute")?,
                syn_workspace_get_size: load_sym(handle, "synWorkspaceGetSize")?,
                syn_tensor_retrieve_launch_amount: load_sym(
                    handle,
                    "synTensorRetrieveLaunchAmount",
                )?,
                syn_tensor_retrieve_launch_ids: load_sym(handle, "synTensorRetrieveLaunchIds")?,
                syn_tensor_retrieve_launch_info_by_id_ext: load_sym(
                    handle,
                    "synTensorRetrieveLaunchInfoByIdExt",
                )?,
                syn_launch_with_external_events_ext: load_sym(
                    handle,
                    "synLaunchWithExternalEventsExt",
                )?,
                syn_configuration_set: load_sym(handle, "synConfigurationSet")?,
                syn_configuration_get: load_sym(handle, "synConfigurationGet")?,
            })
        };
        if res.is_err() {
            // SAFETY: Cleanup handle on partial symbol-load failure.
            unsafe {
                dlclose(handle);
            }
        }
        res
    }

    /// Queries the HabanaLabs driver version string (callable before `synInitialize`).
    pub fn driver_version(&self) -> Result<String, SynapseError> {
        let mut buf = [0u8; 256];
        // SAFETY: Valid writable buffer + exact length passed to `synDriverGetVersion`.
        let st = unsafe {
            (self.syn_driver_get_version)(buf.as_mut_ptr() as *mut c_char, buf.len() as c_int)
        };
        syn_check("synDriverGetVersion", st)?;
        let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..len]).into_owned())
    }
}

impl Drop for SynapseApi {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: Matches `dlopen` handle acquired in `load_from`.
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

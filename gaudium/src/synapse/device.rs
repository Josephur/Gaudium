//! Verified RAII Intel Gaudi 2 device, stream, HBM allocator (`AllocGuard`), arena, and DMA shell.
//!
//! Every lifecycle transition delegates to `core::lifecycle::DeviceGuard` (proven in Bend 2),
//! every HBM allocation is accounted in `core::hbm::HbmLedgerGuard`, and `Drop` ALWAYS drains
//! all streams before releasing the device (`AGENTS.md` §8).

use std::ffi::c_void;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::core::gemma4::HostPinGuard;
use crate::core::hbm::{check_hbm_fit, usable_hbm_bytes, HbmLedgerGuard};
use crate::core::lifecycle::DeviceGuard;
use crate::core::shapes::align_up_safe;
use crate::synapse::ffi::{
    syn_check, SynDeviceId, SynDeviceType, SynDmaDir, SynEventHandle, SynStreamHandle, SynapseApi,
    SynapseError, SYN_SUCCESS,
};
use crate::verified::{DevEvent, HbmCat};

#[derive(Debug, Clone)]
pub struct HbmAllocRecord {
    pub addr: u64,
    pub bytes: u64,
    pub category: HbmCat,
    pub label: String,
}

/// Contiguous pre-allocated HBM arena for low-binding-count weight/state layouts (V2-ready).
#[derive(Debug, Clone)]
pub struct HbmArena {
    base: u64,
    capacity: u64,
    cursor: u64,
}

impl HbmArena {
    pub fn new(base: u64, capacity: u64) -> Self {
        Self {
            base,
            capacity,
            cursor: 0,
        }
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn used(&self) -> u64 {
        self.cursor
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Sub-allocates `bytes` aligned up to `align` (div-by-zero safe via verified `align_up_safe`).
    /// Returns `(device_addr, offset_in_arena)`.
    pub fn suballoc(&mut self, bytes: u64, align: u64) -> Result<(u64, u64), SynapseError> {
        let aligned_off = align_up_safe(self.cursor, align);
        let end = aligned_off
            .checked_add(bytes)
            .ok_or_else(|| SynapseError::Guard("HbmArena offset overflow".into()))?;
        if end > self.capacity {
            return Err(SynapseError::Guard(format!(
                "HbmArena exhausted: need {end} > capacity {}",
                self.capacity
            )));
        }
        self.cursor = end;
        Ok((self.base + aligned_off, aligned_off))
    }
}

/// Checked Gaudi 2 device handle enforcing Bend 2 lifecycle & `AllocGuard` invariants.
pub struct HpuDevice {
    api: Arc<SynapseApi>,
    guard: DeviceGuard,
    dev_id: SynDeviceId,
    compute: SynStreamHandle,
    compute2: SynStreamHandle,
    h2d: SynStreamHandle,
    d2h: SynStreamHandle,
    ledger: HbmLedgerGuard,
    allocs: Vec<HbmAllocRecord>,
    shared_ws_addr: u64,
    shared_ws_size: u64,
}

impl HpuDevice {
    /// Initializes SynapseAI, acquires one Intel Gaudi 2 card (with retry loop under transient
    /// driver ENOMEM), and creates the 4 generic streams (`compute`, `compute2`, `h2d`, `d2h`).
    pub fn open_gaudi2(api: Arc<SynapseApi>) -> Result<Self, SynapseError> {
        let mut guard = DeviceGuard::new();
        // SAFETY: First Synapse lifecycle call.
        syn_check("synInitialize", unsafe { (api.syn_initialize)() })?;
        guard
            .transition(DevEvent::EvSynInit)
            .map_err(|e| SynapseError::Guard(format!("{e:?}")))?;

        let retries: usize = std::env::var("GAUDIUM_ACQUIRE_RETRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10)
            .max(1);

        let mut dev_id: SynDeviceId = 0;
        let mut last_st = 0;
        for attempt in 0..retries {
            // SAFETY: Valid output pointer and `SynDeviceType::Gaudi2`.
            last_st = unsafe {
                (api.syn_device_acquire_by_type)(&mut dev_id, SynDeviceType::Gaudi2 as u32)
            };
            if last_st == SYN_SUCCESS {
                break;
            }
            if attempt + 1 < retries {
                thread::sleep(Duration::from_secs(2));
            }
        }
        if last_st != SYN_SUCCESS {
            // SAFETY: Clean up Synapse on acquire failure.
            unsafe {
                (api.syn_destroy)();
            }
            return Err(SynapseError::Status {
                op: "synDeviceAcquireByDeviceType(Gaudi2)",
                code: last_st,
            });
        }
        guard
            .transition(DevEvent::EvDevAcquire)
            .map_err(|e| SynapseError::Guard(format!("{e:?}")))?;

        let create_stream = |label: &'static str| -> Result<SynStreamHandle, SynapseError> {
            let mut h: SynStreamHandle = std::ptr::null_mut();
            // SAFETY: Device is acquired; valid output pointer.
            syn_check(label, unsafe {
                (api.syn_stream_create_generic)(&mut h, dev_id, 0)
            })?;
            Ok(h)
        };

        let compute = create_stream("synStreamCreateGeneric(compute)")?;
        let compute2 = create_stream("synStreamCreateGeneric(compute2)")?;
        let h2d = create_stream("synStreamCreateGeneric(h2d)")?;
        let d2h = create_stream("synStreamCreateGeneric(d2h)")?;

        guard
            .transition(DevEvent::EvStreamsCreate)
            .map_err(|e| SynapseError::Guard(format!("{e:?}")))?;

        // The runtime replaces SIGINT/SIGTERM dispositions while initializing; a signal it
        // handles terminates in-flight API calls (`synSynapseTerminated`) instead of letting the
        // engine stop between synced launches. Put the engine's stop-request handler back.
        crate::signals::reassert();

        Ok(Self {
            api,
            guard,
            dev_id,
            compute,
            compute2,
            h2d,
            d2h,
            ledger: HbmLedgerGuard::new(),
            allocs: Vec::new(),
            shared_ws_addr: 0,
            shared_ws_size: 0,
        })
    }

    pub fn api(&self) -> &Arc<SynapseApi> {
        &self.api
    }

    pub fn device_id(&self) -> SynDeviceId {
        self.dev_id
    }

    pub fn guard(&self) -> &DeviceGuard {
        &self.guard
    }

    pub fn ledger(&self) -> &HbmLedgerGuard {
        &self.ledger
    }

    pub fn compute_stream(&self) -> SynStreamHandle {
        self.compute
    }

    pub fn compute2_stream(&self) -> SynStreamHandle {
        self.compute2
    }

    pub fn h2d_stream(&self) -> SynStreamHandle {
        self.h2d
    }

    pub fn d2h_stream(&self) -> SynStreamHandle {
        self.d2h
    }

    pub fn shared_workspace(&self) -> (u64, u64) {
        (self.shared_ws_addr, self.shared_ws_size)
    }

    /// Queries `(free_bytes, total_bytes)` of HBM on this Gaudi 2 card.
    pub fn memory_info(&self) -> Result<(u64, u64), SynapseError> {
        let (mut free, mut total) = (0u64, 0u64);
        // SAFETY: Device is open; valid output pointers.
        syn_check("synDeviceGetMemoryInfo", unsafe {
            (self.api.syn_device_get_memory_info)(self.dev_id, &mut free, &mut total)
        })?;
        Ok((free, total))
    }

    /// Verifies a planned byte total against actual card HBM using verified `core::hbm`.
    pub fn check_plan_fits(
        &self,
        planned_bytes: u64,
        spare_bytes: u64,
    ) -> Result<u64, SynapseError> {
        let (free, total) = self.memory_info()?;
        let driver_used = total.saturating_sub(free);
        let usable = usable_hbm_bytes(total, driver_used, spare_bytes);
        check_hbm_fit(planned_bytes, usable).map_err(|e| SynapseError::Guard(format!("{e:?}")))
    }

    /// Allocates device HBM memory ONLY while `DeviceGuard` permits allocation (before `seal`).
    pub fn dmalloc(
        &mut self,
        bytes: u64,
        category: HbmCat,
        label: impl Into<String>,
    ) -> Result<u64, SynapseError> {
        self.guard
            .check_alloc()
            .map_err(|e| SynapseError::Guard(format!("dmalloc refused: {e:?}")))?;
        let label = label.into();
        let mut addr = 0u64;
        // SAFETY: Device is open & unsealed; valid output pointer.
        syn_check("synDeviceMalloc", unsafe {
            (self.api.syn_device_malloc)(self.dev_id, bytes, 0, 0, &mut addr)
        })?;
        self.ledger.record(category, bytes);
        self.allocs.push(HbmAllocRecord {
            addr,
            bytes,
            category,
            label,
        });
        Ok(addr)
    }

    /// Allocates a contiguous HBM arena before sealing.
    pub fn alloc_arena(
        &mut self,
        bytes: u64,
        category: HbmCat,
        label: impl Into<String>,
    ) -> Result<HbmArena, SynapseError> {
        let base = self.dmalloc(bytes, category, label)?;
        Ok(HbmArena::new(base, bytes))
    }

    /// Allocates the one shared recipe workspace (if `ws_bytes > 0`) and seals `AllocGuard`.
    /// After this returns, `dmalloc` is permanently refused and recipe launches are enabled!
    pub fn seal_with_shared_workspace(&mut self, ws_bytes: u64) -> Result<(), SynapseError> {
        if ws_bytes > 0 {
            let addr = self.dmalloc(ws_bytes, HbmCat::CatScratch, "shared_recipe_workspace")?;
            self.shared_ws_addr = addr;
            self.shared_ws_size = ws_bytes;
        }
        self.guard
            .transition(DevEvent::EvSealAlloc)
            .map_err(|e| SynapseError::Guard(format!("seal failed: {e:?}")))?;
        Ok(())
    }

    pub fn is_sealed(&self) -> bool {
        self.guard.check_alloc().is_err() && self.guard.check_launch().is_ok()
    }

    /// Synchronizes a single stream.
    ///
    /// # Safety
    /// `stream` must be a live stream handle belonging to this `HpuDevice`.
    pub unsafe fn sync_stream(&self, stream: SynStreamHandle) -> Result<(), SynapseError> {
        // SAFETY: Caller supplies a valid stream owned by this device.
        syn_check("synStreamSynchronize", unsafe {
            (self.api.syn_stream_synchronize)(stream)
        })
    }

    /// Synchronizes all 4 device streams (compute, compute2, h2d, d2h).
    pub fn sync_all(&self) -> Result<(), SynapseError> {
        for s in [self.compute, self.compute2, self.h2d, self.d2h] {
            if !s.is_null() {
                // SAFETY: Streams are owned by `self` and alive until `Drop`.
                unsafe { self.sync_stream(s)? };
            }
        }
        Ok(())
    }

    /// Maps host memory extents atomically for DMA using verified `HostPinGuard` (rolls back ALL
    /// already-mapped extents if any single `synHostMap` fails!).
    ///
    /// # Safety
    /// `[base, base + sum(extents))` must point to valid, allocated host memory that remains
    /// live until unmapped via [`Self::host_unmap_extents`].
    pub unsafe fn host_map_extents(
        &self,
        base: *const u8,
        extents: &[u64],
    ) -> Result<Vec<(*const u8, u64)>, SynapseError> {
        let mut pin = HostPinGuard::new();
        let mut mapped: Vec<(*const u8, u64)> = Vec::with_capacity(extents.len());
        let mut offset = 0usize;
        for (i, &sz) in extents.iter().enumerate() {
            if sz == 0 {
                continue;
            }
            // SAFETY: Caller guarantees `[base, base + sum(extents))` is valid host memory.
            let ptr = unsafe { base.add(offset) };
            let st = unsafe { (self.api.syn_host_map)(self.dev_id, sz, ptr as *const c_void) };
            if st != SYN_SUCCESS {
                pin.rollback();
                for (p, _) in mapped {
                    unsafe {
                        (self.api.syn_host_unmap)(self.dev_id, p as *const c_void);
                    }
                }
                return Err(SynapseError::Status {
                    op: "synHostMap",
                    code: st,
                });
            }
            pin.record_extent(i + 1 == extents.len())
                .map_err(|e| SynapseError::Guard(format!("{e:?}")))?;
            mapped.push((ptr, sz));
            offset += sz as usize;
        }
        Ok(mapped)
    }

    pub fn host_unmap_extents(&self, mapped: &[(*const u8, u64)]) {
        for &(ptr, _) in mapped {
            // SAFETY: Matches pointers previously mapped via `host_map_extents`.
            unsafe {
                (self.api.syn_host_unmap)(self.dev_id, ptr as *const c_void);
            }
        }
    }

    /// Copies bytes from host memory to device HBM via a Synapse-pinned host staging buffer (blocking).
    pub fn copy_h2d_sync(&self, src: &[u8], dst_hbm: u64) -> Result<(), SynapseError> {
        self.guard
            .check_dma()
            .map_err(|e| SynapseError::Guard(format!("h2d refused: {e:?}")))?;
        if src.is_empty() {
            return Ok(());
        }
        let mut stage: *mut c_void = std::ptr::null_mut();
        // SAFETY: Valid device & byte size.
        syn_check("synHostMalloc", unsafe {
            (self.api.syn_host_malloc)(self.dev_id, src.len() as u64, 0, &mut stage)
        })?;
        // SAFETY: `stage` is valid writable pinned memory of `src.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), stage as *mut u8, src.len());
        }
        let copy_res = syn_check("synMemCopyAsync(H2D)", unsafe {
            (self.api.syn_mem_copy_async)(
                self.h2d,
                stage as u64,
                src.len() as u64,
                dst_hbm,
                SynDmaDir::HostToDram as u32,
            )
        })
        .and_then(|()| unsafe { self.sync_stream(self.h2d) });
        // SAFETY: Always free staging buffer after sync (or failed enqueue).
        unsafe {
            (self.api.syn_host_free)(self.dev_id, stage, 0);
        }
        copy_res
    }

    /// Copies bytes from device HBM to host memory via a Synapse-pinned host staging buffer (blocking).
    pub fn copy_d2h_sync(&self, src_hbm: u64, dst: &mut [u8]) -> Result<(), SynapseError> {
        self.guard
            .check_dma()
            .map_err(|e| SynapseError::Guard(format!("d2h refused: {e:?}")))?;
        if dst.is_empty() {
            return Ok(());
        }
        let mut stage: *mut c_void = std::ptr::null_mut();
        syn_check("synHostMalloc", unsafe {
            (self.api.syn_host_malloc)(self.dev_id, dst.len() as u64, 0, &mut stage)
        })?;
        let copy_res = syn_check("synMemCopyAsync(D2H)", unsafe {
            (self.api.syn_mem_copy_async)(
                self.d2h,
                src_hbm,
                dst.len() as u64,
                stage as u64,
                SynDmaDir::DramToHost as u32,
            )
        })
        .and_then(|()| unsafe { self.sync_stream(self.d2h) });
        if copy_res.is_ok() {
            // SAFETY: Sync completed; `stage` holds `dst.len()` valid bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(stage as *const u8, dst.as_mut_ptr(), dst.len());
            }
        }
        unsafe {
            (self.api.syn_host_free)(self.dev_id, stage, 0);
        }
        copy_res
    }

    /// Allocates a reusable Synapse-pinned host staging buffer (eliminates per-step `synHostMalloc`).
    pub fn alloc_pinned(&self, bytes: usize) -> Result<PinnedBuf, SynapseError> {
        let alloc_bytes = bytes.max(64);
        let mut ptr: *mut c_void = std::ptr::null_mut();
        syn_check("synHostMalloc", unsafe {
            (self.api.syn_host_malloc)(self.dev_id, alloc_bytes as u64, 0, &mut ptr)
        })?;
        Ok(PinnedBuf {
            api: Arc::clone(&self.api),
            dev_id: self.dev_id,
            ptr: ptr as *mut u8,
            len: alloc_bytes,
        })
    }

    /// Copies `bytes` from the start of a pinned host buffer to `dst_hbm` (blocking on H2D stream).
    pub fn copy_h2d_pinned_sync(
        &self,
        src: &PinnedBuf,
        bytes: usize,
        dst_hbm: u64,
    ) -> Result<(), SynapseError> {
        self.guard
            .check_dma()
            .map_err(|e| SynapseError::Guard(format!("h2d refused: {e:?}")))?;
        if bytes == 0 {
            return Ok(());
        }
        if bytes > src.len {
            return Err(SynapseError::InvalidArg(format!(
                "copy_h2d_pinned_sync: {bytes} > pinned capacity {}",
                src.len
            )));
        }
        syn_check("synMemCopyAsync(H2D)", unsafe {
            (self.api.syn_mem_copy_async)(
                self.h2d,
                src.ptr as u64,
                bytes as u64,
                dst_hbm,
                SynDmaDir::HostToDram as u32,
            )
        })?;
        unsafe { self.sync_stream(self.h2d) }
    }

    /// Enqueues multiple `(src_offset, bytes, dst_hbm)` H2D transfers from one `PinnedBuf`
    /// back-to-back and synchronizes the H2D stream once.
    pub fn copy_h2d_pinned_batch_sync(
        &self,
        src: &PinnedBuf,
        transfers: &[(usize, usize, u64)],
    ) -> Result<(), SynapseError> {
        self.guard
            .check_dma()
            .map_err(|e| SynapseError::Guard(format!("h2d refused: {e:?}")))?;
        if transfers.is_empty() {
            return Ok(());
        }
        for &(off, bytes, dst_hbm) in transfers {
            if bytes == 0 {
                continue;
            }
            if off.saturating_add(bytes) > src.len {
                return Err(SynapseError::InvalidArg(format!(
                    "copy_h2d_pinned_batch_sync: {}..{} > pinned capacity {}",
                    off,
                    off.saturating_add(bytes),
                    src.len
                )));
            }
            syn_check("synMemCopyAsync(H2D)", unsafe {
                (self.api.syn_mem_copy_async)(
                    self.h2d,
                    (src.ptr as u64) + (off as u64),
                    bytes as u64,
                    dst_hbm,
                    SynDmaDir::HostToDram as u32,
                )
            })?;
        }
        unsafe { self.sync_stream(self.h2d) }
    }

    /// Enqueues one H2D transfer of `bytes` at `src_off` of a pinned buffer, NOT synchronized:
    /// the buffer range must stay untouched until the H2D stream (or an event recorded on it
    /// after this call) is synchronized.
    pub fn enqueue_h2d_pinned(
        &self,
        src: &PinnedBuf,
        src_off: usize,
        bytes: usize,
        dst_hbm: u64,
    ) -> Result<(), SynapseError> {
        self.guard
            .check_dma()
            .map_err(|e| SynapseError::Guard(format!("h2d refused: {e:?}")))?;
        if bytes == 0 {
            return Ok(());
        }
        if src_off.saturating_add(bytes) > src.len {
            return Err(SynapseError::InvalidArg(format!(
                "enqueue_h2d_pinned: {}..{} > pinned capacity {}",
                src_off,
                src_off.saturating_add(bytes),
                src.len
            )));
        }
        syn_check("synMemCopyAsync(H2D)", unsafe {
            (self.api.syn_mem_copy_async)(
                self.h2d,
                (src.ptr as u64) + (src_off as u64),
                bytes as u64,
                dst_hbm,
                SynDmaDir::HostToDram as u32,
            )
        })
    }

    /// Copies `bytes` from `src_hbm` into the start of a pinned host buffer (blocking on D2H stream).
    pub fn copy_d2h_pinned_sync(
        &self,
        src_hbm: u64,
        dst: &mut PinnedBuf,
        bytes: usize,
    ) -> Result<(), SynapseError> {
        self.guard
            .check_dma()
            .map_err(|e| SynapseError::Guard(format!("d2h refused: {e:?}")))?;
        if bytes == 0 {
            return Ok(());
        }
        if bytes > dst.len {
            return Err(SynapseError::InvalidArg(format!(
                "copy_d2h_pinned_sync: {bytes} > pinned capacity {}",
                dst.len
            )));
        }
        syn_check("synMemCopyAsync(D2H)", unsafe {
            (self.api.syn_mem_copy_async)(
                self.d2h,
                src_hbm,
                bytes as u64,
                dst.ptr as u64,
                SynDmaDir::DramToHost as u32,
            )
        })?;
        unsafe { self.sync_stream(self.d2h) }
    }

    /// Enqueues an HBM -> HBM copy on the compute stream, ordered with the recipe launches of
    /// that stream. NOT synchronized: the caller syncs the compute stream like after a launch.
    pub fn copy_d2d_compute(
        &self,
        src_hbm: u64,
        bytes: u64,
        dst_hbm: u64,
    ) -> Result<(), SynapseError> {
        self.guard
            .check_dma()
            .map_err(|e| SynapseError::Guard(format!("d2d refused: {e:?}")))?;
        if bytes == 0 {
            return Ok(());
        }
        syn_check("synMemCopyAsync(D2D)", unsafe {
            (self.api.syn_mem_copy_async)(
                self.compute,
                src_hbm,
                bytes,
                dst_hbm,
                SynDmaDir::DramToDram as u32,
            )
        })
    }

    /// Uploads an arbitrarily large host slice to HBM in chunks using one reusable `PinnedBuf`.
    pub fn upload_h2d_chunked(
        &self,
        src: &[u8],
        dst_hbm: u64,
        stage: &mut PinnedBuf,
    ) -> Result<(), SynapseError> {
        let cap = stage.len();
        let mut off = 0usize;
        while off < src.len() {
            let n = (src.len() - off).min(cap);
            stage.as_mut_slice()[..n].copy_from_slice(&src[off..off + n]);
            self.copy_h2d_pinned_sync(stage, n, dst_hbm + off as u64)?;
            off += n;
        }
        Ok(())
    }

    /// Enqueues a device-side zero fill of `[dst_hbm, dst_hbm + bytes)` on the compute stream
    /// (`synMemsetD32Async`: a DMA engine writes the zeros, nothing crosses the host link), in
    /// pieces of at most 1 GiB. `bytes` must be a multiple of 4. NOT synchronized: the fill is
    /// ordered with the recipe launches of the compute stream, and the caller syncs that stream
    /// when it needs the completion (or an error) before going on.
    pub fn zero_hbm_compute(&self, dst_hbm: u64, bytes: u64) -> Result<(), SynapseError> {
        self.guard
            .check_dma()
            .map_err(|e| SynapseError::Guard(format!("device zero fill refused: {e:?}")))?;
        if !bytes.is_multiple_of(4) {
            return Err(SynapseError::InvalidArg(format!(
                "zero_hbm_compute: {bytes} bytes is not a multiple of 4"
            )));
        }
        const PIECE: u64 = 1 << 30;
        let mut off = 0u64;
        while off < bytes {
            let n = (bytes - off).min(PIECE);
            // SAFETY: the range lies inside a live device allocation of this `HpuDevice`.
            syn_check("synMemsetD32Async", unsafe {
                (self.api.syn_memset_d32_async)(dst_hbm + off, 0, (n / 4) as usize, self.compute)
            })?;
            off += n;
        }
        Ok(())
    }

    /// Zero-fills `bytes` of device HBM on the device ([`Self::zero_hbm_compute`]) and waits for
    /// it. Only a tail of fewer than 4 bytes is written from the pre-zeroed `zero_stage`.
    pub fn zero_hbm_chunked(
        &self,
        dst_hbm: u64,
        bytes: u64,
        zero_stage: &PinnedBuf,
    ) -> Result<(), SynapseError> {
        let body = bytes & !3;
        self.zero_hbm_compute(dst_hbm, body)?;
        // SAFETY: the compute stream is owned by `self`.
        unsafe { self.sync_stream(self.compute)? };
        if body < bytes {
            self.copy_h2d_pinned_sync(zero_stage, (bytes - body) as usize, dst_hbm + body)?;
        }
        Ok(())
    }

    /// Drains all streams immediately and moves `DeviceGuard` to `DevDraining` (`AGENTS.md` §8).
    pub fn drain_all(&mut self) {
        self.guard.emergency_drain();
        for s in [self.compute, self.compute2, self.h2d, self.d2h] {
            if !s.is_null() {
                unsafe {
                    (self.api.syn_stream_synchronize)(s);
                }
            }
        }
    }

    pub fn create_event(&self, collect_time: bool) -> Result<DeviceEvent, SynapseError> {
        let mut h: SynEventHandle = std::ptr::null_mut();
        let flags = if collect_time { 1 } else { 0 };
        syn_check("synEventCreate", unsafe {
            (self.api.syn_event_create)(&mut h, self.dev_id, flags)
        })?;
        Ok(DeviceEvent {
            api: Arc::clone(&self.api),
            handle: h,
        })
    }
}

impl Drop for HpuDevice {
    fn drop(&mut self) {
        // CRITICAL: Drain EVERY stream before freeing HBM or destroying streams/device!
        self.drain_all();
        for alloc in self.allocs.drain(..) {
            unsafe {
                (self.api.syn_device_free)(self.dev_id, alloc.addr, 0);
            }
        }
        for s in [self.compute, self.compute2, self.h2d, self.d2h] {
            if !s.is_null() {
                unsafe {
                    (self.api.syn_stream_destroy)(s);
                }
            }
        }
        unsafe {
            (self.api.syn_device_release)(self.dev_id);
            (self.api.syn_destroy)();
        }
        let _ = self.guard.transition(DevEvent::EvDevClose);
    }
}

/// SynapseAI initialized without acquiring a device: graph compilation and recipe
/// (de)serialization only. Holds no card, so it can run beside a process that does.
///
/// Must outlive every `CompiledRecipe` created under it (declare it first).
pub struct CompileSession {
    api: Arc<SynapseApi>,
}

impl CompileSession {
    pub fn open(api: Arc<SynapseApi>) -> Result<Self, SynapseError> {
        // SAFETY: first Synapse lifecycle call of this process.
        syn_check("synInitialize", unsafe { (api.syn_initialize)() })?;
        Ok(Self { api })
    }
}

impl Drop for CompileSession {
    fn drop(&mut self) {
        // SAFETY: matches the `synInitialize` of `open`; no device was acquired.
        unsafe {
            (self.api.syn_destroy)();
        }
    }
}

/// Reusable Synapse-pinned host staging buffer freed automatically on drop.
///
/// Must be dropped before the owning `HpuDevice` (declare `PinnedBuf` fields before `HpuDevice`
/// in structs that hold both).
pub struct PinnedBuf {
    api: Arc<SynapseApi>,
    dev_id: SynDeviceId,
    ptr: *mut u8,
    len: usize,
}

impl PinnedBuf {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` bytes for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is uniquely owned & valid for `len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.api.syn_host_free)(self.dev_id, self.ptr as *mut c_void, 0);
            }
        }
    }
}

/// Safe RAII Synapse event handle.
pub struct DeviceEvent {
    api: Arc<SynapseApi>,
    handle: SynEventHandle,
}

impl DeviceEvent {
    /// Records this event onto `dev`'s primary compute stream.
    pub fn record_compute(&self, dev: &HpuDevice) -> Result<(), SynapseError> {
        // SAFETY: `dev.compute_stream()` is a valid live stream owned by `dev`.
        syn_check("synEventRecord", unsafe {
            (self.api.syn_event_record)(self.handle, dev.compute_stream())
        })
    }

    /// Records this event onto an explicit raw stream handle.
    ///
    /// # Safety
    /// `stream` must be a valid live `SynStreamHandle` on the same device.
    pub unsafe fn record(&self, stream: SynStreamHandle) -> Result<(), SynapseError> {
        syn_check("synEventRecord", unsafe {
            (self.api.syn_event_record)(self.handle, stream)
        })
    }

    pub fn synchronize(&self) -> Result<(), SynapseError> {
        syn_check("synEventSynchronize", unsafe {
            (self.api.syn_event_synchronize)(self.handle)
        })
    }

    pub fn elapsed_ns(start: &Self, end: &Self) -> Result<u64, SynapseError> {
        let mut ns = 0u64;
        syn_check("synEventElapsedTime", unsafe {
            (start.api.syn_event_elapsed_time)(&mut ns, start.handle, end.handle)
        })?;
        Ok(ns)
    }
}

impl Drop for DeviceEvent {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.api.syn_event_destroy)(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hbm_arena_suballocates_aligned_and_checks_bounds() {
        let mut arena = HbmArena::new(0x1000_0000, 8192);
        let (a0, o0) = arena.suballoc(100, 128).unwrap();
        assert_eq!((a0, o0), (0x1000_0000, 0));
        let (a1, o1) = arena.suballoc(4096, 4096).unwrap();
        assert_eq!((a1, o1), (0x1000_1000, 4096));
        assert!(arena.suballoc(1, 128).is_err()); // 4096 + 4096 == 8192 full!
    }
}

//! Bulk copies between device memory and ordinary host memory, in both directions, through a
//! small ring of pinned buffers.
//!
//! The device can only reach host memory that is pinned and mapped for it, and pinning is the
//! one expensive and risky thing to do with a large host buffer: every page is taken long-term,
//! pages that were already touched may first have to be moved, and a mapping dies with the
//! device handle. So the memory a copy starts from or ends in (the host-RAM prefix cache) stays
//! ordinary memory, and only this ring is pinned: a few buffers, mapped once per device open
//! before anything touched them, one mapping per buffer.
//!
//! A copy is a list of [`Piece`]s (a device range and where its bytes live on the host). The
//! ring packs consecutive pieces into its buffers, hands each full buffer to the DMA at once
//! and moves on to the next one, so the link and the host copy threads work at the same time:
//! - device to host: the DMA fills buffers while the buffers it finished are copied out;
//! - host to device: buffers are filled from host memory while the DMA drains the ones before.
//!
//! Pieces whose device ranges touch become one DMA command, so the number of commands follows
//! the layout of the device buffers, not the layout of the host copy.
//!
//! The ring is used between requests only (the engine takes part in no launch then). A copy to
//! the host is synchronised before its bytes are read; a copy to the device is synchronised
//! before this module returns, so a launch enqueued afterwards reads the copied rows.
//!
//! Nothing is ever in flight against a ring buffer that is being unmapped: every function here
//! drains what it queued before it returns, on an error too ([`then_drained`]), and the ring
//! drains both transfer streams once more when it is dropped, whenever that is.
//!
//! How a copy is packed is the verified core's (`verified/host_cache.bend`, section 17): the
//! bytes placed next (`xf_take`), when a buffer is handed over (`xf_buf_full`), which buffer
//! follows (`xf_next_buf`), which bytes join the command before them (`xf_cmd_joins`), and
//! how one buffer's host copy is cut among the threads (`xf_part_take`, `xf_part_full`). This
//! file adds the offsets it is given to addresses and calls the DMA.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::hostmem::{Admitted, AnonMap};
use crate::synapse::device::{DeviceEvent, HpuDevice, PinnedBuf};
use crate::synapse::ffi::{SynDeviceId, SynStreamHandle, SynapseApi, SynapseError};
use crate::verified::{
    hc_budget_fits, xf_buf_full, xf_cmd_joins, xf_next_buf, xf_part_full, xf_part_take,
    xf_probe_dev_off, xf_probe_host_need, xf_probe_span, xf_probe_span_fits, xf_slice_fits,
    xf_slice_off, xf_take,
};

/// Buffers and bytes per buffer of the ring an engine keeps.
pub const RING_BUFS: usize = 4;
pub const RING_BUF_BYTES: usize = 64 * 1024 * 1024;
/// Host copy threads of an engine's ring. Measured on the Gaudi 2 host (`--probe-copy`): the
/// copy is bound by memory traffic, two threads carry what the machine gives and more only
/// take from each other (14.9 GiB/s to host with two, 10.4 with six).
pub const COPY_THREADS: usize = 2;
/// A copy to the host asks whether a request waits after each this many bytes it has copied out
/// of the ring (and whenever a ring buffer is full).
const STOP_CHECK_BYTES: usize = 8 * 1024 * 1024;
/// Host copies below this stay on the calling thread.
const PAR_MIN_BYTES: usize = 2 * 1024 * 1024;

/// How the ring's memory is obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingKind {
    /// Anonymous memory backed by huge pages where the host has them, mapped for the device
    /// before it is touched (`synHostMap`).
    MappedHuge,
    /// The SDK's own pinned allocation (`synHostMalloc`).
    Malloc,
}

enum RingMem {
    Mapped {
        map: AnonMap,
        api: Arc<SynapseApi>,
        dev_id: SynDeviceId,
        dev_open: Arc<AtomicBool>,
    },
    Pinned(PinnedBuf),
    /// Host memory that is mapped for no device (a ring driven by a made-up link in tests).
    #[cfg(test)]
    Plain(AnonMap),
}

impl RingMem {
    fn ptr(&self) -> *mut u8 {
        match self {
            Self::Mapped { map, .. } => map.as_ptr() as *mut u8,
            Self::Pinned(p) => p.as_slice().as_ptr() as *mut u8,
            #[cfg(test)]
            Self::Plain(map) => map.as_ptr() as *mut u8,
        }
    }
}

impl Drop for RingMem {
    fn drop(&mut self) {
        if let Self::Mapped {
            map,
            api,
            dev_id,
            dev_open,
        } = self
        {
            // A device that is closed took its mappings with it.
            if dev_open.load(Ordering::Acquire) {
                // SAFETY: the pointer was mapped by `synHostMap` in `XferRing::new`; no
                // transfer reads or writes it now (`XferRing::drop` drained both transfer
                // streams before its buffers are dropped).
                unsafe {
                    (api.syn_host_unmap)(*dev_id, map.as_ptr() as *const std::ffi::c_void);
                }
            }
        }
    }
}

/// One device range and where its bytes live on the host.
#[derive(Debug, Clone, Copy)]
pub struct Piece {
    pub dev: u64,
    pub host: *mut u8,
    pub len: usize,
}

/// Counters of one copy.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct XferStats {
    pub bytes: u64,
    /// DMA commands issued.
    pub commands: u64,
    /// Whole copy, first byte to last synchronisation.
    pub wall_s: f64,
    /// Host time copying between the ring and host memory.
    pub copy_s: f64,
    /// Host time waiting for the link.
    pub wait_s: f64,
}

impl XferStats {
    /// Bytes per second of the whole copy (0 for an empty one).
    pub fn rate(&self) -> u64 {
        if self.bytes == 0 || self.wall_s <= 0.0 {
            return 0;
        }
        (self.bytes as f64 / self.wall_s) as u64
    }

    pub fn add(&mut self, o: &XferStats) {
        self.bytes += o.bytes;
        self.commands += o.commands;
        self.wall_s += o.wall_s;
        self.copy_s += o.copy_s;
        self.wait_s += o.wait_s;
    }
}

/// A host copy between a ring buffer and host memory.
#[derive(Clone, Copy)]
struct Job {
    ring: *mut u8,
    host: *mut u8,
    len: usize,
}

// SAFETY: a job names two disjoint byte ranges that stay valid for the copy; the threads of one
// copy get disjoint jobs.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

/// The pinned ring of one open device. It may be dropped before the device or after it: while
/// the device is open the ring drains the two transfer streams before its buffers are
/// unmapped, and a device that closes first drains every stream itself with the buffers still
/// mapped. (Declare it before the `HpuDevice` in a struct that holds both all the same: its
/// events and pinned allocations are the device's.)
pub struct XferRing {
    bufs: Vec<RingMem>,
    events: Vec<DeviceEvent>,
    buf_bytes: usize,
    threads: usize,
    kind: RingKind,
    /// The device's two transfer streams, drained before the buffers are unmapped (`None`: a
    /// ring no device reads).
    drain: Option<RingDrain>,
}

struct RingDrain {
    api: Arc<SynapseApi>,
    d2h: SynStreamHandle,
    h2d: SynStreamHandle,
    dev_open: Arc<AtomicBool>,
}

// SAFETY: the ring owns its buffers and events; they are used by one thread at a time.
unsafe impl Send for XferRing {}

impl Drop for XferRing {
    fn drop(&mut self) {
        // Runs before the buffers (fields) are dropped and unmapped. Every copy drains what it
        // queued before it returns; this is for the path that did not.
        if let Some(d) = self.drain.as_ref() {
            if d.dev_open.load(Ordering::Acquire) {
                for stream in [d.d2h, d.h2d] {
                    // SAFETY: the device is open, so its streams are live.
                    unsafe {
                        (d.api.syn_stream_synchronize)(stream);
                    }
                }
            }
        }
    }
}

/// What the ring asks of a device: the two directions of the DMA, a mark behind what was
/// queued for one ring buffer, and the two ways to wait. The engine's ring talks to the card
/// ([`DevLink`]); host tests drive the same copy loops with a link made of host memory.
pub(crate) trait Link {
    /// Queues a copy of device memory into a ring buffer.
    ///
    /// # Safety
    /// `host` must be valid for writes of `bytes` until the copy has been waited for.
    unsafe fn enqueue_d2h(
        &self,
        dev_addr: u64,
        bytes: u64,
        host: *mut u8,
    ) -> Result<(), SynapseError>;
    /// Queues a copy of a ring buffer into device memory.
    ///
    /// # Safety
    /// `host` must be valid for reads of `bytes` until the copy has been waited for.
    unsafe fn enqueue_h2d(
        &self,
        host: *const u8,
        bytes: u64,
        dev_addr: u64,
    ) -> Result<(), SynapseError>;
    /// Marks the end of what was queued for ring buffer `buf` in that direction.
    fn mark(&self, buf: usize, to_host: bool) -> Result<(), SynapseError>;
    /// Waits for everything queued before the mark of ring buffer `buf`.
    fn wait(&self, buf: usize) -> Result<(), SynapseError>;
    /// Waits for everything queued in that direction.
    fn drain(&self, to_host: bool) -> Result<(), SynapseError>;
}

/// The card as a [`Link`]: its two transfer streams and the ring's events on them.
struct DevLink<'a> {
    dev: &'a HpuDevice,
    events: &'a [DeviceEvent],
}

impl DevLink<'_> {
    fn stream(&self, to_host: bool) -> SynStreamHandle {
        if to_host {
            self.dev.d2h_stream()
        } else {
            self.dev.h2d_stream()
        }
    }
}

impl Link for DevLink<'_> {
    unsafe fn enqueue_d2h(
        &self,
        dev_addr: u64,
        bytes: u64,
        host: *mut u8,
    ) -> Result<(), SynapseError> {
        // SAFETY: the caller's contract; the ring buffer is mapped for `dev`.
        unsafe { self.dev.enqueue_d2h_raw(dev_addr, bytes, host) }
    }

    unsafe fn enqueue_h2d(
        &self,
        host: *const u8,
        bytes: u64,
        dev_addr: u64,
    ) -> Result<(), SynapseError> {
        // SAFETY: the caller's contract; the ring buffer is mapped for `dev`.
        unsafe { self.dev.enqueue_h2d_raw(host, bytes, dev_addr) }
    }

    fn mark(&self, buf: usize, to_host: bool) -> Result<(), SynapseError> {
        // SAFETY: the stream belongs to `dev`.
        unsafe { self.events[buf].record(self.stream(to_host)) }
    }

    fn wait(&self, buf: usize) -> Result<(), SynapseError> {
        self.events[buf].synchronize()
    }

    fn drain(&self, to_host: bool) -> Result<(), SynapseError> {
        // SAFETY: the stream belongs to `dev`.
        unsafe { self.dev.sync_stream(self.stream(to_host)) }
    }
}

/// Runs `enqueue`, then `sync` whatever `enqueue` returned, and gives back the first error. A
/// command that was queued before a later one failed is complete on the device, or has failed,
/// before the caller sees the error and lets go of the host memory it names.
fn then_drained<T, E>(
    enqueue: impl FnOnce() -> Result<T, E>,
    sync: impl FnOnce() -> Result<(), E>,
) -> Result<T, E> {
    let queued = enqueue();
    let synced = sync();
    let value = queued?;
    synced.map(|()| value)
}

impl XferRing {
    /// Allocates and maps the ring: `bufs` buffers of `buf_bytes`, each its own mapping, none
    /// touched before it is pinned. Every buffer, of either kind, is taken out of `leave`
    /// ([`Self::planned_bytes`] of host memory, asked for by the caller just before): pinning
    /// makes the pages at once, so a ring the host cannot hold must be refused before it.
    pub fn new(
        dev: &HpuDevice,
        bufs: usize,
        buf_bytes: usize,
        kind: RingKind,
        threads: usize,
        mut leave: Admitted,
    ) -> Result<Self, SynapseError> {
        let bufs = bufs.max(2);
        let buf_bytes = AnonMap::mapped_len(buf_bytes as u64) as usize;
        let mut ring = Vec::with_capacity(bufs);
        let mut events = Vec::with_capacity(bufs);
        for _ in 0..bufs {
            let mem = match kind {
                RingKind::Malloc => {
                    leave.draw(buf_bytes as u64).map_err(SynapseError::Guard)?;
                    RingMem::Pinned(dev.alloc_pinned(buf_bytes)?)
                }
                RingKind::MappedHuge => {
                    let map =
                        AnonMap::new(buf_bytes as u64, &mut leave).map_err(SynapseError::Guard)?;
                    // SAFETY: `map` is live host memory of exactly this length and stays live
                    // until the unmap in `RingMem::drop`.
                    unsafe { dev.host_map_extents(map.as_ptr(), &[map.len() as u64])? };
                    RingMem::Mapped {
                        map,
                        api: Arc::clone(dev.api()),
                        dev_id: dev.device_id(),
                        dev_open: dev.open_flag(),
                    }
                }
            };
            ring.push(mem);
            events.push(dev.create_event(false)?);
        }
        Ok(Self {
            bufs: ring,
            events,
            buf_bytes,
            threads: threads.clamp(1, 16),
            kind,
            drain: Some(RingDrain {
                api: Arc::clone(dev.api()),
                d2h: dev.d2h_stream(),
                h2d: dev.h2d_stream(),
                dev_open: dev.open_flag(),
            }),
        })
    }

    /// A ring of plain host memory that no device reads, for a made-up [`Link`].
    #[cfg(test)]
    fn on_host_memory(bufs: usize, buf_bytes: usize, threads: usize) -> Self {
        let buf_bytes = AnonMap::mapped_len(buf_bytes as u64) as usize;
        let mut leave = Admitted::ask(Self::planned_bytes(bufs, buf_bytes), u64::MAX).unwrap();
        Self {
            bufs: (0..bufs.max(2))
                .map(|_| RingMem::Plain(AnonMap::new(buf_bytes as u64, &mut leave).unwrap()))
                .collect(),
            events: Vec::new(),
            buf_bytes,
            threads: threads.clamp(1, 16),
            kind: RingKind::MappedHuge,
            drain: None,
        }
    }

    /// Host copy threads for a machine with the given parallelism: the copy is bound by memory
    /// traffic, a few threads carry it.
    pub fn default_threads() -> usize {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 6)
    }

    /// Pinned host bytes of the ring.
    pub fn pinned_bytes(&self) -> u64 {
        (self.bufs.len() * self.buf_bytes) as u64
    }

    /// Pinned host bytes of a ring of `bufs` buffers of `buf_bytes`.
    pub fn planned_bytes(bufs: usize, buf_bytes: usize) -> u64 {
        bufs.max(2) as u64 * AnonMap::mapped_len(buf_bytes as u64)
    }

    pub fn kind(&self) -> RingKind {
        self.kind
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.clamp(1, 16);
    }

    /// Copies device memory to host memory. `stop` is asked before the copy starts and whenever
    /// a ring buffer is full: once it says yes nothing more is enqueued, the transfers in flight
    /// are waited for, and what they brought is left in the ring. Returns how many pieces (from
    /// the start of `pieces`) are complete in host memory; every byte this call enqueued is
    /// synchronised when it returns, also on an error.
    ///
    /// # Safety
    /// Every piece's `host` range must be valid for writes of `len` bytes for the whole call,
    /// and no two pieces' host ranges may overlap. The device ranges must lie inside live
    /// allocations of `dev`, and nothing may be writing them.
    pub unsafe fn d2h(
        &mut self,
        dev: &HpuDevice,
        pieces: &[Piece],
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<(usize, XferStats), SynapseError> {
        let link = DevLink {
            dev,
            events: &self.events,
        };
        // SAFETY: the caller's contract.
        unsafe { self.d2h_on(&link, pieces, stop) }
    }

    /// [`Self::d2h`] over any [`Link`].
    ///
    /// # Safety
    /// As [`Self::d2h`].
    unsafe fn d2h_on<L: Link>(
        &self,
        link: &L,
        pieces: &[Piece],
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<(usize, XferStats), SynapseError> {
        let t0 = Instant::now();
        let n = self.bufs.len();
        let cap = self.buf_bytes;
        let mut stats = XferStats::default();
        let mut busy = vec![false; n];
        let mut jobs: Vec<Vec<Job>> = vec![Vec::new(); n];
        // pieces complete once this buffer and every one before it are copied out
        let mut ends = vec![0usize; n];
        let mut cmds: Vec<(u64, usize, usize)> = Vec::new(); // (device, ring offset, bytes)
        let (mut cur, mut fill) = (0usize, 0usize);
        // pieces whose last byte has a place in the ring / whose bytes are in host memory
        let (mut placed, mut done) = (0usize, 0usize);
        let mut res: Result<(), SynapseError> = Ok(());
        let mut stopped = stop();

        'pieces: for (i, piece) in pieces.iter().enumerate() {
            if stopped {
                break;
            }
            let mut off = 0usize;
            while off < piece.len {
                if xf_buf_full(fill as u64, cap as u64) {
                    // the buffer is full: hand it to the link and take the next one
                    ends[cur] = placed;
                    if let Err(e) = self.close_d2h(link, cur, &mut cmds, &mut busy, &mut stats) {
                        res = Err(e);
                        break 'pieces;
                    }
                    cur = xf_next_buf(cur as u64, n as u64) as usize;
                    fill = 0;
                    if busy[cur] {
                        match self.drain_d2h(link, cur, &mut jobs[cur], &mut stats, stop) {
                            Ok(true) => {
                                busy[cur] = false;
                                done = ends[cur];
                            }
                            Ok(false) => {
                                stopped = true;
                                break 'pieces;
                            }
                            Err(e) => {
                                res = Err(e);
                                break 'pieces;
                            }
                        }
                    }
                    if stop() {
                        stopped = true;
                        break 'pieces;
                    }
                }
                // what is left of the piece, and no more than the buffer has room for
                let take = xf_take(piece.len as u64, off as u64, cap as u64, fill as u64) as usize;
                let dev_addr = piece.dev + off as u64;
                match cmds.last_mut() {
                    Some((d, r, l))
                        if xf_cmd_joins(*d, *l as u64, dev_addr, *r as u64, fill as u64) =>
                    {
                        *l += take
                    }
                    _ => cmds.push((dev_addr, fill, take)),
                }
                jobs[cur].push(Job {
                    // SAFETY: `fill + take <= cap`, inside the ring buffer
                    // (`xf_take_within_buffer`).
                    ring: unsafe { self.bufs[cur].ptr().add(fill) },
                    // SAFETY: `off + take <= piece.len`, inside the piece's host range
                    // (`xf_take_within_piece`).
                    host: unsafe { piece.host.add(off) },
                    len: take,
                });
                fill += take;
                off += take;
                stats.bytes += take as u64;
            }
            placed = i + 1;
        }
        if res.is_ok() && !stopped {
            // hand over what is left, then copy out every buffer in the order they were queued
            ends[cur] = placed;
            res = self.close_d2h(link, cur, &mut cmds, &mut busy, &mut stats);
            let mut b = cur;
            for _ in 0..n {
                b = xf_next_buf(b as u64, n as u64) as usize;
                if busy[b] {
                    match self.drain_d2h(link, b, &mut jobs[b], &mut stats, stop) {
                        Ok(true) => {
                            busy[b] = false;
                            done = ends[b];
                        }
                        Ok(false) => {
                            stopped = true;
                            break;
                        }
                        Err(e) => {
                            res = Err(e);
                            break;
                        }
                    }
                }
            }
        }
        if res.is_err() || stopped {
            // whatever was enqueued must not be in flight when the caller frees host memory or
            // the ring is used again; its bytes stay in the ring and count for nothing
            let t1 = Instant::now();
            let synced = link.drain(true);
            stats.wait_s += t1.elapsed().as_secs_f64();
            if res.is_ok() {
                res = synced;
            }
        }
        stats.wall_s = t0.elapsed().as_secs_f64();
        res.map(|()| (done, stats))
    }

    /// Enqueues the commands of buffer `b` and marks its end on the stream.
    fn close_d2h<L: Link>(
        &self,
        link: &L,
        b: usize,
        cmds: &mut Vec<(u64, usize, usize)>,
        busy: &mut [bool],
        stats: &mut XferStats,
    ) -> Result<(), SynapseError> {
        if cmds.is_empty() {
            return Ok(());
        }
        let base = self.bufs[b].ptr();
        for &(dev_addr, ring_off, len) in cmds.iter() {
            // SAFETY: `ring_off + len` lies inside ring buffer `b`, which stays live until the
            // copy has been waited for.
            unsafe { link.enqueue_d2h(dev_addr, len as u64, base.add(ring_off))? };
            stats.commands += 1;
        }
        cmds.clear();
        link.mark(b, true)?;
        busy[b] = true;
        Ok(())
    }

    /// Waits for buffer `b`'s transfers and copies its bytes out to host memory, asking `stop`
    /// between slices of the copy. `Ok(false)`: it said yes and the rest was left in the ring.
    fn drain_d2h<L: Link>(
        &self,
        link: &L,
        b: usize,
        jobs: &mut Vec<Job>,
        stats: &mut XferStats,
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<bool, SynapseError> {
        let t0 = Instant::now();
        let res = link.wait(b);
        stats.wait_s += t0.elapsed().as_secs_f64();
        let mut whole = true;
        if res.is_ok() {
            let t1 = Instant::now();
            let mut from = 0usize;
            while from < jobs.len() {
                if stop() {
                    whole = false;
                    break;
                }
                let (mut to, mut bytes) = (from, 0usize);
                while to < jobs.len() && (bytes == 0 || bytes + jobs[to].len <= STOP_CHECK_BYTES) {
                    bytes += jobs[to].len;
                    to += 1;
                }
                // SAFETY: the transfers into the ring buffer are complete; the host ranges are
                // the caller's (`d2h`'s contract).
                unsafe { run_jobs(&jobs[from..to], self.threads, true) };
                from = to;
            }
            stats.copy_s += t1.elapsed().as_secs_f64();
        }
        jobs.clear();
        res.map(|()| whole)
    }

    /// Copies host memory to device memory. Every transfer is complete when this returns.
    ///
    /// # Safety
    /// Every piece's `host` range must be valid for reads of `len` bytes for the whole call.
    /// The device ranges must lie inside live allocations of `dev`, and nothing may be reading
    /// or writing them.
    pub unsafe fn h2d(
        &mut self,
        dev: &HpuDevice,
        pieces: &[Piece],
    ) -> Result<XferStats, SynapseError> {
        let link = DevLink {
            dev,
            events: &self.events,
        };
        // SAFETY: the caller's contract.
        unsafe { self.h2d_on(&link, pieces) }
    }

    /// [`Self::h2d`] over any [`Link`].
    ///
    /// # Safety
    /// As [`Self::h2d`].
    unsafe fn h2d_on<L: Link>(
        &self,
        link: &L,
        pieces: &[Piece],
    ) -> Result<XferStats, SynapseError> {
        let t0 = Instant::now();
        let n = self.bufs.len();
        let cap = self.buf_bytes;
        let mut stats = XferStats::default();
        let mut busy = vec![false; n];
        let mut jobs: Vec<Job> = Vec::new();
        let mut cmds: Vec<(u64, usize, usize)> = Vec::new();
        let (mut cur, mut fill) = (0usize, 0usize);
        let mut res: Result<(), SynapseError> = Ok(());

        'pieces: for piece in pieces {
            let mut off = 0usize;
            while off < piece.len {
                if xf_buf_full(fill as u64, cap as u64) {
                    if let Err(e) =
                        self.close_h2d(link, cur, &mut jobs, &mut cmds, &mut busy, &mut stats)
                    {
                        res = Err(e);
                        break 'pieces;
                    }
                    cur = xf_next_buf(cur as u64, n as u64) as usize;
                    fill = 0;
                    if busy[cur] {
                        let t1 = Instant::now();
                        let r = link.wait(cur);
                        stats.wait_s += t1.elapsed().as_secs_f64();
                        busy[cur] = false;
                        if let Err(e) = r {
                            res = Err(e);
                            break 'pieces;
                        }
                    }
                }
                let take = xf_take(piece.len as u64, off as u64, cap as u64, fill as u64) as usize;
                let dev_addr = piece.dev + off as u64;
                match cmds.last_mut() {
                    Some((d, r, l))
                        if xf_cmd_joins(*d, *l as u64, dev_addr, *r as u64, fill as u64) =>
                    {
                        *l += take
                    }
                    _ => cmds.push((dev_addr, fill, take)),
                }
                jobs.push(Job {
                    // SAFETY: as in `d2h`.
                    ring: unsafe { self.bufs[cur].ptr().add(fill) },
                    host: unsafe { piece.host.add(off) },
                    len: take,
                });
                fill += take;
                off += take;
                stats.bytes += take as u64;
            }
        }
        if res.is_ok() {
            res = self.close_h2d(link, cur, &mut jobs, &mut cmds, &mut busy, &mut stats);
        }
        // Everything queued must be on the device before a launch reads it (and before the
        // ring is reused or freed).
        let t1 = Instant::now();
        let synced = link.drain(false);
        stats.wait_s += t1.elapsed().as_secs_f64();
        stats.wall_s = t0.elapsed().as_secs_f64();
        res.and(synced).map(|()| stats)
    }

    /// Fills buffer `b` from host memory, enqueues its commands and marks its end.
    fn close_h2d<L: Link>(
        &self,
        link: &L,
        b: usize,
        jobs: &mut Vec<Job>,
        cmds: &mut Vec<(u64, usize, usize)>,
        busy: &mut [bool],
        stats: &mut XferStats,
    ) -> Result<(), SynapseError> {
        if cmds.is_empty() {
            return Ok(());
        }
        let t0 = Instant::now();
        // SAFETY: no transfer reads buffer `b` now (it was waited for before it was refilled);
        // the host ranges are the caller's (`h2d`'s contract).
        unsafe { run_jobs(jobs, self.threads, false) };
        stats.copy_s += t0.elapsed().as_secs_f64();
        jobs.clear();
        let base = self.bufs[b].ptr();
        for &(dev_addr, ring_off, len) in cmds.iter() {
            // SAFETY: `ring_off + len` lies inside ring buffer `b`, which is not refilled before
            // the copy has been waited for.
            unsafe { link.enqueue_h2d(base.add(ring_off), len as u64, dev_addr)? };
            stats.commands += 1;
        }
        cmds.clear();
        link.mark(b, false)?;
        busy[b] = true;
        Ok(())
    }

    /// Raw link rates without a host copy: `(device to host, host to device)` bytes per second
    /// for `bytes` moved in commands of `cmd_bytes` between the ring and the first bytes of
    /// the device range `[dev_addr, dev_addr + dev_bytes)`, whose content is read and written
    /// back unchanged. A round that would reach past that range is refused. Every command
    /// this queues is complete, or has failed, when it returns, also on an error.
    pub fn link_rates(
        &mut self,
        dev: &HpuDevice,
        dev_addr: u64,
        dev_bytes: u64,
        bytes: u64,
        cmd_bytes: usize,
    ) -> Result<(u64, u64), SynapseError> {
        let link = DevLink {
            dev,
            events: &self.events,
        };
        self.link_rates_on(&link, dev_addr, dev_bytes, bytes, cmd_bytes)
    }

    /// [`Self::link_rates`] over any [`Link`].
    fn link_rates_on<L: Link>(
        &self,
        link: &L,
        dev_addr: u64,
        dev_bytes: u64,
        bytes: u64,
        cmd_bytes: usize,
    ) -> Result<(u64, u64), SynapseError> {
        let cmd = cmd_bytes.clamp(4096, self.buf_bytes) as u64;
        let buf = self.buf_bytes as u64;
        // the commands of this size that lie inside one ring buffer (`xf_slice_fits`)
        let per_buf = (0u64..).take_while(|&k| xf_slice_fits(k, cmd, buf)).count() as u64;
        let n = self.bufs.len() as u64;
        let span = xf_probe_span(n, per_buf, cmd);
        if !xf_probe_span_fits(n, per_buf, cmd, dev_bytes) {
            return Err(SynapseError::Guard(format!(
                "link probe: a round of {span} bytes reaches past the {dev_bytes} device bytes it was given"
            )));
        }
        let rounds = bytes.div_ceil(span.max(1)).max(1);
        let mut rates = [0u64; 2];
        for (dir, rate) in rates.iter_mut().enumerate() {
            let to_host = dir == 0;
            let t0 = Instant::now();
            for _ in 0..rounds {
                then_drained(
                    || {
                        for b in 0..n {
                            let base = self.bufs[b as usize].ptr();
                            for k in 0..per_buf {
                                let at = dev_addr + xf_probe_dev_off(b, per_buf, k, cmd);
                                // SAFETY: slice `k` lies inside ring buffer `b`
                                // (`xf_slice_fits`), and the command inside the round's span
                                // on the device (`xf_probe_cmd_within_span`), which lies
                                // inside the caller's range (checked above).
                                unsafe {
                                    let host = base.add(xf_slice_off(k, cmd) as usize);
                                    if to_host {
                                        link.enqueue_d2h(at, cmd, host)?;
                                    } else {
                                        link.enqueue_h2d(host, cmd, at)?;
                                    }
                                }
                            }
                        }
                        Ok(())
                    },
                    || link.drain(to_host),
                )?;
            }
            let s = t0.elapsed().as_secs_f64().max(1e-9);
            *rate = ((rounds * span) as f64 / s) as u64;
        }
        Ok((rates[0], rates[1]))
    }
}

/// Runs the host copies of one ring buffer on up to `threads` threads. `to_host`: ring to host
/// memory, else host memory to ring.
///
/// # Safety
/// Every job's two ranges must be valid (the destination for writes) and disjoint from every
/// other job's destination.
unsafe fn run_jobs(jobs: &[Job], threads: usize, to_host: bool) {
    let total: usize = jobs.iter().map(|j| j.len).sum();
    let copy = |j: &Job| {
        // SAFETY: the caller's contract.
        unsafe {
            if to_host {
                stream_copy(j.ring as *const u8, j.host, j.len);
            } else {
                stream_copy(j.host as *const u8, j.ring, j.len);
            }
        }
    };
    if threads <= 1 || total < PAR_MIN_BYTES {
        jobs.iter().for_each(copy);
        return;
    }
    // Shares of whole pages (any share is correct: it only spreads the work); a job larger
    // than what is left of a share is cut, and the last thread takes whatever is left.
    let share = total.div_ceil(threads).next_multiple_of(4096);
    let mut parts: Vec<Vec<Job>> = vec![Vec::new(); threads];
    let (mut t, mut used) = (0usize, 0usize);
    for j in jobs {
        let mut off = 0usize;
        while off < j.len {
            if xf_part_full(used as u64, share as u64, t + 1 == threads) {
                t += 1;
                used = 0;
            }
            let take = xf_part_take(
                j.len as u64,
                off as u64,
                share as u64,
                used as u64,
                t + 1 == threads,
            ) as usize;
            parts[t].push(Job {
                // SAFETY: `off + take <= j.len` (`xf_part_take_within_job`).
                ring: unsafe { j.ring.add(off) },
                host: unsafe { j.host.add(off) },
                len: take,
            });
            used += take;
            off += take;
        }
    }
    thread::scope(|s| {
        for part in parts.iter().filter(|p| !p.is_empty()) {
            let copy = &copy;
            s.spawn(move || part.iter().for_each(copy));
        }
    });
}

/// Copies below this use the plain copy.
const STREAM_MIN_BYTES: usize = 16 * 1024;

/// `copy_nonoverlapping` for a destination that is not read again soon (a ring buffer the DMA
/// takes, a cache segment that waits for a later request): the bytes are written with
/// non-temporal stores, so the destination's lines are not first read into the cache. The C
/// library does the same above a size of several megabytes; the pieces of a K/V tile are
/// smaller than that, and measured on the Gaudi 2 host a plain copy of 256 KiB pieces ran at
/// two thirds of the rate of one large copy.
///
/// # Safety
/// As `std::ptr::copy_nonoverlapping`.
pub unsafe fn stream_copy(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if len >= STREAM_MIN_BYTES {
            use std::arch::x86_64::{
                __m128i, _mm_loadu_si128, _mm_prefetch, _mm_sfence, _mm_stream_si128, _MM_HINT_NTA,
            };
            // SAFETY: the caller's contract; every access below stays inside `len` bytes.
            unsafe {
                // bring the destination to a 64-byte line
                let head = (64 - (dst as usize & 63)) & 63;
                std::ptr::copy_nonoverlapping(src, dst, head);
                let (mut s, mut d) = (src.add(head), dst.add(head));
                let lines = (len - head) / 64;
                for _ in 0..lines {
                    _mm_prefetch(s.add(512) as *const i8, _MM_HINT_NTA);
                    let a = _mm_loadu_si128(s as *const __m128i);
                    let b = _mm_loadu_si128(s.add(16) as *const __m128i);
                    let c = _mm_loadu_si128(s.add(32) as *const __m128i);
                    let e = _mm_loadu_si128(s.add(48) as *const __m128i);
                    _mm_stream_si128(d as *mut __m128i, a);
                    _mm_stream_si128(d.add(16) as *mut __m128i, b);
                    _mm_stream_si128(d.add(32) as *mut __m128i, c);
                    _mm_stream_si128(d.add(48) as *mut __m128i, e);
                    s = s.add(64);
                    d = d.add(64);
                }
                _mm_sfence();
                let tail = len - head - lines * 64;
                std::ptr::copy_nonoverlapping(s, d, tail);
            }
            return;
        }
    }
    // SAFETY: the caller's contract.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len) }
}

/// Whether the probe may take its host memory: the ordinary memory it holds at its peak
/// (`xf_probe_host_need`: a buffer of `device_bytes`, or the two buffers of `copy_bytes` of the
/// host-copy test) and the `ring_bytes` pinned beside it must fit in what the host and the
/// control groups of this process have left (`hc_budget_fits`). The pages of a mapping exist
/// only once a copy writes them, so a probe that does not fit would be stopped by the kernel
/// in the middle of transfers, not by a failed allocation; an error here says what is missing
/// before anything is allocated.
fn probe_admission(
    device_bytes: u64,
    copy_bytes: u64,
    ring_bytes: u64,
    mem: &crate::hostmem::HostMemory,
) -> Result<(), String> {
    let need = xf_probe_host_need(device_bytes, copy_bytes);
    let available = mem.available();
    if hc_budget_fits(need, ring_bytes, available) {
        return Ok(());
    }
    let g = |b: u64| b as f64 / (1u64 << 30) as f64;
    Err(format!(
        "the copy probe needs {:.2} GiB of host memory ({:.2} GiB of buffers and {:.2} GiB of pinned transfer buffers), and {:.2} GiB are available ({}). Give it fewer GiB, or free host memory",
        g(need.saturating_add(ring_bytes)),
        g(need),
        g(ring_bytes),
        g(available),
        mem.describe()
    ))
}

/// Measures the copies of this machine and card and prints them: the raw link in both
/// directions for two kinds of pinned memory and several command sizes, the host copy alone,
/// and whole copies through the ring. `gib`: device bytes used, in GiB.
pub fn probe(gib: u64) -> Result<(), String> {
    use crate::verified::HbmCat;
    let gib = gib.clamp(1, 16);
    let bytes = gib << 30;
    // the host-copy test at the end: two buffers of this size
    let copy_bytes = (1u64 << 30).min(bytes);
    // host memory first: nothing is allocated, and no card is opened, for a probe that would
    // not fit
    let mem = crate::hostmem::HostMemory::read();
    probe_admission(
        bytes,
        copy_bytes,
        XferRing::planned_bytes(RING_BUFS, RING_BUF_BYTES),
        &mem,
    )?;
    let api = Arc::new(SynapseApi::load_default().map_err(|e| format!("load libSynapse.so: {e}"))?);
    let mut dev =
        HpuDevice::open_gaudi2(Arc::clone(&api)).map_err(|e| format!("open Gaudi 2: {e}"))?;
    let addr = dev
        .dmalloc(bytes, HbmCat::CatScratch, "copy_probe")
        .map_err(|e| format!("allocate {gib} GiB of device memory: {e}"))?;
    dev.seal_with_shared_workspace(0)
        .map_err(|e| format!("seal: {e}"))?;
    dev.zero_hbm_compute(addr, bytes)
        .map_err(|e| format!("zero device memory: {e}"))?;
    // SAFETY: the stream belongs to `dev`.
    unsafe { dev.sync_stream(dev.compute_stream()) }.map_err(|e| format!("sync: {e}"))?;
    let g = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!(
        "copy probe: {gib} GiB of device memory; host: {}; {} logical CPUs",
        mem.describe(),
        thread::available_parallelism().map_or(0, |n| n.get())
    );

    for kind in [RingKind::MappedHuge, RingKind::Malloc] {
        let t0 = Instant::now();
        let leave = crate::hostmem::HostMemory::read().admit(
            "the probe's pinned transfer ring",
            XferRing::planned_bytes(RING_BUFS, RING_BUF_BYTES),
        )?;
        let mut ring = XferRing::new(&dev, RING_BUFS, RING_BUF_BYTES, kind, 4, leave)
            .map_err(|e| format!("ring {kind:?}: {e}"))?;
        println!(
            "ring {kind:?}: {} x {} MiB pinned in {:.1} ms",
            RING_BUFS,
            RING_BUF_BYTES >> 20,
            t0.elapsed().as_secs_f64() * 1e3
        );
        for cmd in [256usize << 10, 2 << 20, 8 << 20, 64 << 20] {
            let (d2h, h2d) = ring
                .link_rates(&dev, addr, bytes, bytes.min(2 << 30), cmd)
                .map_err(|e| format!("link rates: {e}"))?;
            println!(
                "  link, commands of {:>6} KiB: device to host {:.2} GiB/s, host to device {:.2} GiB/s",
                cmd >> 10,
                g(d2h),
                g(h2d)
            );
        }
        // whole copies through the ring: into fresh host memory, out of it again, and into
        // memory that was already written
        for threads in [1usize, 2, 4, 6, 8] {
            ring.set_threads(threads);
            let mut leave = crate::hostmem::HostMemory::read()
                .admit("the probe's host buffer", AnonMap::mapped_len(bytes))?;
            let mut host = AnonMap::new(bytes, &mut leave)?;
            for piece_bytes in [256u64 << 10, 32 << 20] {
                // the pieces of this size that lie inside the device bytes and inside `host`
                let pieces: Vec<Piece> = (0u64..)
                    .take_while(|&i| xf_slice_fits(i, piece_bytes, bytes))
                    .map(|i| {
                        let off = xf_slice_off(i, piece_bytes);
                        Piece {
                            dev: addr + off,
                            // SAFETY: inside `host`, a mapping of `bytes` (`xf_slice_fits`).
                            host: unsafe { host.as_mut_ptr().add(off as usize) },
                            len: piece_bytes as usize,
                        }
                    })
                    .collect();
                // SAFETY: the pieces tile `host` and the probe's device allocation.
                let (_, save) = unsafe { ring.d2h(&dev, &pieces, &mut || false) }
                    .map_err(|e| format!("d2h: {e}"))?;
                let restore =
                    unsafe { ring.h2d(&dev, &pieces) }.map_err(|e| format!("h2d: {e}"))?;
                println!(
                    "  {threads} copy thread(s), pieces of {:>6} KiB: to host {:.2} GiB/s ({} commands, copy {:.0} ms, link wait {:.0} ms of {:.0} ms); to device {:.2} GiB/s (copy {:.0} ms, link wait {:.0} ms of {:.0} ms)",
                    piece_bytes >> 10,
                    g(save.rate()),
                    save.commands,
                    save.copy_s * 1e3,
                    save.wait_s * 1e3,
                    save.wall_s * 1e3,
                    g(restore.rate()),
                    restore.copy_s * 1e3,
                    restore.wait_s * 1e3,
                    restore.wall_s * 1e3,
                );
            }
        }
    }
    // the host copy alone, between two anonymous mappings
    let n = copy_bytes as usize;
    let admit = || {
        crate::hostmem::HostMemory::read().admit(
            "a buffer of the probe's host-copy test",
            AnonMap::mapped_len(n as u64),
        )
    };
    let mut a = AnonMap::new(n as u64, &mut admit()?)?;
    // written before the next buffer is asked for: that asking counts it
    a.as_mut_slice().fill(1);
    for threads in [1usize, 2, 4, 6, 8] {
        let mut b = AnonMap::new(n as u64, &mut admit()?)?;
        let job = [Job {
            ring: a.as_mut_ptr(),
            host: b.as_mut_ptr(),
            len: n,
        }];
        let t0 = Instant::now();
        // SAFETY: two live mappings of `n` bytes.
        unsafe { run_jobs(&job, threads, true) };
        let fresh = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        unsafe { run_jobs(&job, threads, true) };
        let warm = t1.elapsed().as_secs_f64();
        println!(
            "host copy of 1 GiB on {threads} thread(s): {:.2} GiB/s into fresh memory, {:.2} GiB/s into written memory",
            g((n as f64 / fresh) as u64),
            g((n as f64 / warm) as u64)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    const MIB: usize = 1 << 20;
    const BASE: u64 = 0x4000_0000;

    /// `bytes` bytes that differ from place to place (built once, copied for each test).
    fn pattern(bytes: usize) -> Vec<u8> {
        static PATTERN: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        let p = PATTERN.get_or_init(|| {
            (0..16 * MIB)
                .map(|i| ((i * 2654435761) >> 7) as u8)
                .collect()
        });
        p[..bytes].to_vec()
    }

    /// One queued command of the made-up link.
    struct Cmd {
        serial: usize,
        to_host: bool,
        dev: u64,
        host: *mut u8,
        len: usize,
    }

    /// A device made of host memory behind a [`Link`]. As on the card, a command only moves
    /// its bytes once it is waited for (by the mark of its ring buffer, or by a drain): a copy
    /// loop that read a ring buffer before waiting for it, or filled one again before its
    /// transfers ran, moves the wrong bytes here too. `fail_at`: the enqueue with this serial
    /// number fails.
    struct FakeLink {
        mem: RefCell<Vec<u8>>,
        queue: RefCell<Vec<Cmd>>,
        marks: RefCell<Vec<(usize, bool)>>,
        serial: Cell<usize>,
        fail_at: Cell<Option<usize>>,
        ran: Cell<usize>,
    }

    impl FakeLink {
        fn new(bytes: usize, bufs: usize) -> Self {
            Self {
                mem: RefCell::new(pattern(bytes)),
                queue: RefCell::new(Vec::new()),
                marks: RefCell::new(vec![(0, true); bufs]),
                serial: Cell::new(0),
                fail_at: Cell::new(None),
                ran: Cell::new(0),
            }
        }

        fn enqueue(
            &self,
            to_host: bool,
            dev: u64,
            host: *mut u8,
            len: usize,
        ) -> Result<(), SynapseError> {
            let serial = self.serial.get();
            self.serial.set(serial + 1);
            if self.fail_at.get() == Some(serial) {
                return Err(SynapseError::Guard(format!(
                    "made-up failure of command {serial}"
                )));
            }
            self.queue.borrow_mut().push(Cmd {
                serial,
                to_host,
                dev,
                host,
                len,
            });
            Ok(())
        }

        /// Runs the queued commands of one direction that `keep` does not hold back.
        fn run(&self, to_host: bool, before: usize) {
            let mut queue = self.queue.borrow_mut();
            let mut mem = self.mem.borrow_mut();
            queue.retain(|c| {
                if c.to_host != to_host || c.serial >= before {
                    return true;
                }
                let at = (c.dev - BASE) as usize;
                // SAFETY: the ring and the tests keep `host` valid for `len` bytes.
                unsafe {
                    if c.to_host {
                        std::ptr::copy_nonoverlapping(mem[at..at + c.len].as_ptr(), c.host, c.len);
                    } else {
                        std::ptr::copy_nonoverlapping(
                            c.host,
                            mem[at..at + c.len].as_mut_ptr(),
                            c.len,
                        );
                    }
                }
                self.ran.set(self.ran.get() + 1);
                false
            });
        }

        fn queued(&self) -> usize {
            self.queue.borrow().len()
        }
    }

    impl Link for FakeLink {
        unsafe fn enqueue_d2h(
            &self,
            dev_addr: u64,
            bytes: u64,
            host: *mut u8,
        ) -> Result<(), SynapseError> {
            self.enqueue(true, dev_addr, host, bytes as usize)
        }

        unsafe fn enqueue_h2d(
            &self,
            host: *const u8,
            bytes: u64,
            dev_addr: u64,
        ) -> Result<(), SynapseError> {
            self.enqueue(false, dev_addr, host as *mut u8, bytes as usize)
        }

        fn mark(&self, buf: usize, to_host: bool) -> Result<(), SynapseError> {
            self.marks.borrow_mut()[buf] = (self.serial.get(), to_host);
            Ok(())
        }

        fn wait(&self, buf: usize) -> Result<(), SynapseError> {
            let (before, to_host) = self.marks.borrow()[buf];
            self.run(to_host, before);
            Ok(())
        }

        fn drain(&self, to_host: bool) -> Result<(), SynapseError> {
            self.run(to_host, usize::MAX);
            Ok(())
        }
    }

    /// Pieces of uneven sizes over 15 MiB of device memory, some following each other on the
    /// device (they become one command), some not, one larger than two ring buffers; their host
    /// places are in another order, with gaps.
    fn pieces(host: &mut [u8]) -> Vec<Piece> {
        let lens = [
            3 * MIB + 17,
            4096,
            1,
            2 * MIB,
            5 * MIB + 3,
            123_456,
            2 * MIB - 4097,
            700_000,
        ];
        // device: the first four follow each other; then a gap before each of the others
        let mut dev = Vec::new();
        let mut at = 0usize;
        for (i, &len) in lens.iter().enumerate() {
            if i >= 4 {
                at += 8192 + i;
            }
            dev.push(at);
            at += len;
        }
        assert!(at <= 15 * MIB);
        // host: back to front, 64 bytes apart
        let mut out = Vec::new();
        let mut end = host.len();
        for (i, &len) in lens.iter().enumerate() {
            end -= len + 64;
            out.push(Piece {
                dev: BASE + dev[i] as u64,
                // SAFETY: inside `host`.
                host: unsafe { host.as_mut_ptr().add(end) },
                len,
            });
        }
        out
    }

    fn host_of(p: &Piece) -> &[u8] {
        // SAFETY: the piece's host range is live for the test.
        unsafe { std::slice::from_raw_parts(p.host, p.len) }
    }

    fn dev_of(link: &FakeLink, p: &Piece) -> Vec<u8> {
        let at = (p.dev - BASE) as usize;
        link.mem.borrow()[at..at + p.len].to_vec()
    }

    #[test]
    fn a_copy_to_the_host_brings_every_piece_through_a_ring_smaller_than_the_copy() {
        for threads in [1usize, 2, 3] {
            // three buffers of 2 MiB for 13 MiB of pieces: the ring goes round twice
            let ring = XferRing::on_host_memory(3, 2 * MIB, threads);
            let link = FakeLink::new(15 * MIB, 3);
            let mut host = vec![0u8; 16 * MIB];
            let ps = pieces(&mut host);
            let (done, st) = unsafe { ring.d2h_on(&link, &ps, &mut || false) }.unwrap();
            assert_eq!(done, ps.len());
            for (i, p) in ps.iter().enumerate() {
                assert!(
                    host_of(p) == dev_of(&link, p),
                    "piece {i}, {threads} thread(s)"
                );
            }
            let total: usize = ps.iter().map(|p| p.len).sum();
            assert_eq!(st.bytes, total as u64);
            // nothing is left queued, and pieces that follow each other on the device and in
            // the ring went as one command
            assert_eq!(link.queued(), 0);
            assert_eq!(st.commands as usize, link.ran.get());
            let slices = total.div_ceil(2 * MIB) + ps.len();
            assert!(
                (st.commands as usize) < slices - 3,
                "{} commands",
                st.commands
            );
        }
    }

    #[test]
    fn a_copy_to_the_device_brings_every_piece_through_a_ring_smaller_than_the_copy() {
        for threads in [1usize, 2] {
            let ring = XferRing::on_host_memory(3, 2 * MIB, threads);
            let link = FakeLink::new(15 * MIB, 3);
            let mut host: Vec<u8> = (0..16 * MIB).map(|i| ((i * 40503) >> 5) as u8).collect();
            let ps = pieces(&mut host);
            let before = link.mem.borrow().clone();
            let st = unsafe { ring.h2d_on(&link, &ps) }.unwrap();
            for (i, p) in ps.iter().enumerate() {
                assert!(
                    host_of(p) == dev_of(&link, p),
                    "piece {i}, {threads} thread(s)"
                );
            }
            assert_eq!(link.queued(), 0);
            assert_eq!(st.bytes, ps.iter().map(|p| p.len as u64).sum::<u64>());
            // device bytes outside the pieces are as they were
            let mut expect = before;
            for p in &ps {
                let at = (p.dev - BASE) as usize;
                expect[at..at + p.len].copy_from_slice(host_of(p));
            }
            assert!(*link.mem.borrow() == expect);
        }
    }

    #[test]
    fn a_copy_that_is_stopped_reports_only_whole_pieces_and_leaves_nothing_queued() {
        // how often an unstopped copy asks
        let asks = {
            let ring = XferRing::on_host_memory(3, 2 * MIB, 2);
            let link = FakeLink::new(15 * MIB, 3);
            let mut host = vec![0u8; 16 * MIB];
            let ps = pieces(&mut host);
            let mut n = 0usize;
            unsafe {
                ring.d2h_on(&link, &ps, &mut || {
                    n += 1;
                    false
                })
            }
            .unwrap();
            n
        };
        assert!(asks > 8);
        let mut seen = std::collections::BTreeSet::new();
        for stop_at in 0..=asks {
            let ring = XferRing::on_host_memory(3, 2 * MIB, 2);
            let link = FakeLink::new(15 * MIB, 3);
            let mut host = vec![0xEEu8; 16 * MIB];
            let ps = pieces(&mut host);
            let mut n = 0usize;
            let (done, _) = unsafe {
                ring.d2h_on(&link, &ps, &mut || {
                    n += 1;
                    n > stop_at
                })
            }
            .unwrap();
            // every piece counted as done is whole in host memory
            for (i, p) in ps.iter().take(done).enumerate() {
                assert!(
                    host_of(p) == dev_of(&link, p),
                    "stop at {stop_at}: piece {i}"
                );
            }
            // whatever was queued when it stopped was waited for before it returned
            assert_eq!(link.queued(), 0, "stop at {stop_at}");
            seen.insert(done);
        }
        // it stopped at the start, in the middle and not at all
        assert!(
            seen.contains(&0) && seen.contains(&8) && seen.len() > 3,
            "{seen:?}"
        );
    }

    #[test]
    fn a_failed_command_leaves_nothing_queued_in_either_direction_or_in_the_probe() {
        // which commands an unfailed run queues
        let count = |run: &dyn Fn(&XferRing, &FakeLink, &[Piece]) -> bool| {
            let ring = XferRing::on_host_memory(3, 2 * MIB, 2);
            let link = FakeLink::new(15 * MIB, 3);
            let mut host = vec![0u8; 16 * MIB];
            let ps = pieces(&mut host);
            assert!(run(&ring, &link, &ps));
            link.serial.get()
        };
        let to_host: &dyn Fn(&XferRing, &FakeLink, &[Piece]) -> bool =
            &|ring, link, ps| unsafe { ring.d2h_on(link, ps, &mut || false) }.is_ok();
        let to_dev: &dyn Fn(&XferRing, &FakeLink, &[Piece]) -> bool =
            &|ring, link, ps| unsafe { ring.h2d_on(link, ps) }.is_ok();
        let probe: &dyn Fn(&XferRing, &FakeLink, &[Piece]) -> bool = &|ring, link, _| {
            ring.link_rates_on(link, BASE, 15 * MIB as u64, 12 * MIB as u64, MIB)
                .is_ok()
        };
        for (name, run) in [
            ("to host", to_host),
            ("to device", to_dev),
            ("probe", probe),
        ] {
            let n = count(run);
            assert!(n >= 6, "{name}: {n} commands");
            for fail_at in 0..n {
                let ring = XferRing::on_host_memory(3, 2 * MIB, 2);
                let link = FakeLink::new(15 * MIB, 3);
                let mut host = vec![0u8; 16 * MIB];
                let ps = pieces(&mut host);
                link.fail_at.set(Some(fail_at));
                assert!(
                    !run(&ring, &link, &ps),
                    "{name}: command {fail_at} failed unnoticed"
                );
                // every command queued before the one that failed has run: none can touch
                // the ring or the caller's memory after the error is returned
                assert_eq!(link.queued(), 0, "{name}: failure of command {fail_at}");
                assert_eq!(
                    link.ran.get(),
                    fail_at,
                    "{name}: failure of command {fail_at}"
                );
            }
        }
    }

    #[test]
    fn the_link_probe_stays_inside_the_device_bytes_it_is_given() {
        let ring = XferRing::on_host_memory(3, 2 * MIB, 1);
        let link = FakeLink::new(15 * MIB, 3);
        let before = link.mem.borrow().clone();
        // a round of three buffers of 2 MiB in commands of 512 KiB: 12 commands a round
        let (down, up) = ring
            .link_rates_on(&link, BASE, 15 * MIB as u64, 12 * MIB as u64, 512 * 1024)
            .unwrap();
        assert!(down > 0 && up > 0);
        assert_eq!(link.ran.get(), 2 * 2 * 12);
        // what the device holds was read and written back unchanged
        assert!(*link.mem.borrow() == before);
        // a device range smaller than one round is refused before anything is queued
        let small = FakeLink::new(15 * MIB, 3);
        assert!(ring
            .link_rates_on(
                &small,
                BASE,
                6 * MIB as u64 - 1,
                12 * MIB as u64,
                512 * 1024
            )
            .is_err());
        assert_eq!(small.serial.get(), 0);
    }

    #[test]
    fn the_probe_is_refused_when_the_host_cannot_hold_its_buffers() {
        use crate::hostmem::{Group, HostMemory};
        let gib = 1u64 << 30;
        let ring = XferRing::planned_bytes(RING_BUFS, RING_BUF_BYTES);
        assert_eq!(ring, 256 << 20);
        // the default probe: 2 GiB of device bytes, a host-copy test of two 1 GiB buffers
        let host = |bytes: u64| HostMemory::of_host(bytes);
        assert!(probe_admission(2 * gib, gib, ring, &host(2 * gib + ring)).is_ok());
        let e = probe_admission(2 * gib, gib, ring, &host(2 * gib + ring - 1)).unwrap_err();
        assert!(
            e.contains("needs 2.25 GiB") && e.contains("2.25 GiB are available"),
            "{e}"
        );
        // a probe of 1 GiB holds more in its host-copy test (two buffers) than in its copies
        assert!(probe_admission(gib, gib, ring, &host(2 * gib + ring)).is_ok());
        assert!(probe_admission(gib, gib, ring, &host(2 * gib)).is_err());
        // a control group above the process bounds it, whatever the host has
        let limited = HostMemory {
            host_available: 64 * gib,
            groups: vec![
                Group {
                    limit: None,
                    high: None,
                    in_use: Some(gib),
                },
                Group {
                    limit: Some(3 * gib),
                    high: None,
                    in_use: Some(gib),
                },
            ],
            unread: None,
        };
        let e = probe_admission(2 * gib, gib, ring, &limited).unwrap_err();
        assert!(
            e.contains("2.00 GiB are available") && e.contains("the group above"),
            "{e}"
        );
        assert!(probe_admission(gib, 256 << 20, ring, &limited).is_ok());
    }

    #[test]
    fn the_drain_runs_whatever_the_enqueue_returned_and_the_first_error_wins() {
        let drained = Cell::new(0);
        let sync = |r: Result<(), &'static str>| {
            let d = &drained;
            move || {
                d.set(d.get() + 1);
                r
            }
        };
        assert_eq!(then_drained(|| Ok::<_, &str>(7), sync(Ok(()))), Ok(7));
        assert_eq!(
            then_drained(|| Err::<u8, _>("enqueue"), sync(Ok(()))),
            Err("enqueue")
        );
        assert_eq!(
            then_drained(|| Err::<u8, _>("enqueue"), sync(Err("sync"))),
            Err("enqueue")
        );
        assert_eq!(
            then_drained(|| Ok::<_, &str>(7), sync(Err("sync"))),
            Err("sync")
        );
        assert_eq!(drained.get(), 4);
    }

    #[test]
    fn host_copies_are_exact_for_every_thread_count_and_cut() {
        let n = 5 * 1024 * 1024 + 123;
        let src: Vec<u8> = (0..n).map(|i| (i * 131 % 251) as u8).collect();
        for threads in [1usize, 2, 3, 4, 7] {
            // several jobs of uneven sizes, one of them larger than a thread's share
            let cuts = [0usize, 17, 4096, 1_000_000, 1_000_001, 4_500_000, n];
            let mut ring = src.clone();
            let mut host = vec![0u8; n];
            let jobs: Vec<Job> = cuts
                .windows(2)
                .map(|w| Job {
                    ring: unsafe { ring.as_mut_ptr().add(w[0]) },
                    host: unsafe { host.as_mut_ptr().add(w[0]) },
                    len: w[1] - w[0],
                })
                .collect();
            unsafe { run_jobs(&jobs, threads, true) };
            assert!(host == src, "to host, threads {threads}");
            ring.iter_mut().for_each(|b| *b = 0);
            unsafe { run_jobs(&jobs, threads, false) };
            assert!(ring == src, "to ring, threads {threads}");
        }
    }

    #[test]
    fn stream_copy_equals_a_plain_copy_at_every_alignment_and_length() {
        let n = 300_000usize;
        let src: Vec<u8> = (0..n + 200).map(|i| (i * 7 % 253) as u8).collect();
        for (so, d_off, len) in [
            (0usize, 0usize, n),
            (1, 3, n - 5),
            (63, 1, STREAM_MIN_BYTES),
            (5, 64, STREAM_MIN_BYTES + 1),
            (0, 7, STREAM_MIN_BYTES - 1),
            (9, 9, 0),
            (2, 61, 70_001),
        ] {
            let mut dst = vec![0xEEu8; n + 200];
            unsafe { stream_copy(src.as_ptr().add(so), dst.as_mut_ptr().add(d_off), len) };
            assert!(
                dst[d_off..d_off + len] == src[so..so + len],
                "{so} {d_off} {len}"
            );
            // nothing outside the range is written
            assert!(dst[..d_off].iter().all(|&b| b == 0xEE));
            assert!(dst[d_off + len..].iter().all(|&b| b == 0xEE));
        }
    }

    /// Host copy rates of this machine (`cargo test --release -- --ignored host_copy_rates
    /// --nocapture`): pieces of a K/V tile's size and one large copy, plain and streamed.
    #[test]
    #[ignore]
    fn host_copy_rates() {
        let n = 1usize << 30;
        // each buffer is asked for against what this machine has when it is mapped
        let map = || {
            let mut leave = crate::hostmem::HostMemory::read()
                .admit("a buffer of the copy test", n as u64)
                .unwrap();
            AnonMap::new(n as u64, &mut leave).unwrap()
        };
        let mut a = map();
        a.as_mut_slice().fill(3);
        let gib = |s: f64| n as f64 / s / (1u64 << 30) as f64;
        for piece in [256usize << 10, 2 << 20, 32 << 20] {
            for fresh in [true, false] {
                let mut b = map();
                if !fresh {
                    b.as_mut_slice().fill(1);
                }
                let t0 = Instant::now();
                for i in 0..n / piece {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            a.as_ptr().add(i * piece),
                            b.as_mut_ptr().add(i * piece),
                            piece,
                        )
                    };
                }
                let plain = t0.elapsed().as_secs_f64();
                let mut c = map();
                if !fresh {
                    c.as_mut_slice().fill(1);
                }
                let t1 = Instant::now();
                for i in 0..n / piece {
                    unsafe {
                        stream_copy(
                            a.as_ptr().add(i * piece),
                            c.as_mut_ptr().add(i * piece),
                            piece,
                        )
                    };
                }
                let streamed = t1.elapsed().as_secs_f64();
                assert!(b.as_slice() == c.as_slice());
                println!(
                    "pieces of {:>6} KiB into {} memory: plain {:.2} GiB/s, streamed {:.2} GiB/s",
                    piece >> 10,
                    if fresh { "fresh  " } else { "written" },
                    gib(plain),
                    gib(streamed)
                );
            }
        }
    }

    #[test]
    fn planned_ring_bytes_are_whole_huge_pages() {
        assert_eq!(
            XferRing::planned_bytes(RING_BUFS, RING_BUF_BYTES),
            (RING_BUFS * RING_BUF_BYTES) as u64
        );
        assert_eq!(XferRing::planned_bytes(1, 1), 2 * crate::hostmem::HUGE_PAGE);
    }

    #[test]
    fn stats_give_the_rate_of_the_whole_copy() {
        let s = XferStats {
            bytes: 1 << 30,
            wall_s: 0.5,
            ..XferStats::default()
        };
        assert_eq!(s.rate(), 2 << 30);
        assert_eq!(XferStats::default().rate(), 0);
    }
}

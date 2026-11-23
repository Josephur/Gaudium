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

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::hostmem::AnonMap;
use crate::synapse::device::{DeviceEvent, HpuDevice, PinnedBuf};
use crate::synapse::ffi::{SynDeviceId, SynapseApi, SynapseError};

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
    },
    Pinned(PinnedBuf),
}

impl RingMem {
    fn ptr(&self) -> *mut u8 {
        match self {
            Self::Mapped { map, .. } => map.as_ptr() as *mut u8,
            Self::Pinned(p) => p.as_slice().as_ptr() as *mut u8,
        }
    }
}

impl Drop for RingMem {
    fn drop(&mut self) {
        if let Self::Mapped { map, api, dev_id } = self {
            // SAFETY: the pointer was mapped by `synHostMap` in `XferRing::new`; no transfer
            // reads or writes it now (every copy of this module is synchronised before it
            // returns, error paths included).
            unsafe {
                (api.syn_host_unmap)(*dev_id, map.as_ptr() as *const std::ffi::c_void);
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

/// The pinned ring of one open device. Must be dropped before the device (declare it before the
/// `HpuDevice` in a struct that holds both).
pub struct XferRing {
    bufs: Vec<RingMem>,
    events: Vec<DeviceEvent>,
    buf_bytes: usize,
    threads: usize,
    kind: RingKind,
}

// SAFETY: the ring owns its buffers and events; they are used by one thread at a time.
unsafe impl Send for XferRing {}

impl XferRing {
    /// Allocates and maps the ring: `bufs` buffers of `buf_bytes`, each its own mapping, none
    /// touched before it is pinned.
    pub fn new(
        dev: &HpuDevice,
        bufs: usize,
        buf_bytes: usize,
        kind: RingKind,
        threads: usize,
    ) -> Result<Self, SynapseError> {
        let bufs = bufs.max(2);
        let buf_bytes = AnonMap::mapped_len(buf_bytes as u64) as usize;
        let mut ring = Vec::with_capacity(bufs);
        let mut events = Vec::with_capacity(bufs);
        for _ in 0..bufs {
            let mem = match kind {
                RingKind::Malloc => RingMem::Pinned(dev.alloc_pinned(buf_bytes)?),
                RingKind::MappedHuge => {
                    let map = AnonMap::new(buf_bytes as u64).map_err(SynapseError::Guard)?;
                    // SAFETY: `map` is live host memory of exactly this length and stays live
                    // until the unmap in `RingMem::drop`.
                    unsafe { dev.host_map_extents(map.as_ptr(), &[map.len() as u64])? };
                    RingMem::Mapped {
                        map,
                        api: Arc::clone(dev.api()),
                        dev_id: dev.device_id(),
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
        })
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
                if fill == cap {
                    // the buffer is full: hand it to the link and take the next one
                    ends[cur] = placed;
                    if let Err(e) = self.close_d2h(dev, cur, &mut cmds, &mut busy, &mut stats) {
                        res = Err(e);
                        break 'pieces;
                    }
                    cur = (cur + 1) % n;
                    fill = 0;
                    if busy[cur] {
                        match self.drain_d2h(cur, &mut jobs[cur], &mut stats, stop) {
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
                let take = (piece.len - off).min(cap - fill);
                let dev_addr = piece.dev + off as u64;
                match cmds.last_mut() {
                    Some((d, r, l)) if *d + *l as u64 == dev_addr && *r + *l == fill => *l += take,
                    _ => cmds.push((dev_addr, fill, take)),
                }
                jobs[cur].push(Job {
                    // SAFETY: `fill + take <= cap`, inside the ring buffer.
                    ring: unsafe { self.bufs[cur].ptr().add(fill) },
                    // SAFETY: `off + take <= piece.len`, inside the piece's host range.
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
            res = self.close_d2h(dev, cur, &mut cmds, &mut busy, &mut stats);
            for k in 1..=n {
                let b = (cur + k) % n;
                if busy[b] {
                    match self.drain_d2h(b, &mut jobs[b], &mut stats, stop) {
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
            // SAFETY: the stream belongs to `dev`.
            let synced = unsafe { dev.sync_stream(dev.d2h_stream()) };
            stats.wait_s += t1.elapsed().as_secs_f64();
            if res.is_ok() {
                res = synced;
            }
        }
        stats.wall_s = t0.elapsed().as_secs_f64();
        res.map(|()| (done, stats))
    }

    /// Enqueues the commands of buffer `b` and marks its end on the stream.
    fn close_d2h(
        &self,
        dev: &HpuDevice,
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
            // SAFETY: `ring_off + len` lies inside ring buffer `b`, which is mapped for `dev`.
            unsafe { dev.enqueue_d2h_raw(dev_addr, len as u64, base.add(ring_off))? };
            stats.commands += 1;
        }
        cmds.clear();
        // SAFETY: the stream belongs to `dev`.
        unsafe { self.events[b].record(dev.d2h_stream())? };
        busy[b] = true;
        Ok(())
    }

    /// Waits for buffer `b`'s transfers and copies its bytes out to host memory, asking `stop`
    /// between slices of the copy. `Ok(false)`: it said yes and the rest was left in the ring.
    fn drain_d2h(
        &self,
        b: usize,
        jobs: &mut Vec<Job>,
        stats: &mut XferStats,
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<bool, SynapseError> {
        let t0 = Instant::now();
        let res = self.events[b].synchronize();
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
                if fill == cap {
                    if let Err(e) =
                        self.close_h2d(dev, cur, &mut jobs, &mut cmds, &mut busy, &mut stats)
                    {
                        res = Err(e);
                        break 'pieces;
                    }
                    cur = (cur + 1) % n;
                    fill = 0;
                    if busy[cur] {
                        let t1 = Instant::now();
                        let r = self.events[cur].synchronize();
                        stats.wait_s += t1.elapsed().as_secs_f64();
                        busy[cur] = false;
                        if let Err(e) = r {
                            res = Err(e);
                            break 'pieces;
                        }
                    }
                }
                let take = (piece.len - off).min(cap - fill);
                let dev_addr = piece.dev + off as u64;
                match cmds.last_mut() {
                    Some((d, r, l)) if *d + *l as u64 == dev_addr && *r + *l == fill => *l += take,
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
            res = self.close_h2d(dev, cur, &mut jobs, &mut cmds, &mut busy, &mut stats);
        }
        // Everything queued must be on the device before a launch reads it (and before the
        // ring is reused or freed).
        let t1 = Instant::now();
        // SAFETY: the stream belongs to `dev`.
        let synced = unsafe { dev.sync_stream(dev.h2d_stream()) };
        stats.wait_s += t1.elapsed().as_secs_f64();
        stats.wall_s = t0.elapsed().as_secs_f64();
        res.and(synced).map(|()| stats)
    }

    /// Fills buffer `b` from host memory, enqueues its commands and marks its end.
    fn close_h2d(
        &self,
        dev: &HpuDevice,
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
            // SAFETY: `ring_off + len` lies inside ring buffer `b`, which is mapped for `dev`.
            unsafe { dev.enqueue_h2d_raw(base.add(ring_off), len as u64, dev_addr)? };
            stats.commands += 1;
        }
        cmds.clear();
        // SAFETY: the stream belongs to `dev`.
        unsafe { self.events[b].record(dev.h2d_stream())? };
        busy[b] = true;
        Ok(())
    }

    /// Raw link rates without a host copy: `(device to host, host to device)` bytes per second
    /// for `bytes` moved in commands of `cmd_bytes` between `[dev_addr, dev_addr + ring size)`
    /// and the ring. What the device range holds is read and written back unchanged.
    pub fn link_rates(
        &mut self,
        dev: &HpuDevice,
        dev_addr: u64,
        bytes: u64,
        cmd_bytes: usize,
    ) -> Result<(u64, u64), SynapseError> {
        let cmd = cmd_bytes.clamp(4096, self.buf_bytes);
        let per_buf = (self.buf_bytes / cmd).max(1);
        let n = self.bufs.len();
        let span = (n * per_buf * cmd) as u64;
        let rounds = bytes.div_ceil(span).max(1);
        let mut rates = [0u64; 2];
        for (dir, rate) in rates.iter_mut().enumerate() {
            let t0 = Instant::now();
            for _ in 0..rounds {
                for b in 0..n {
                    let base = self.bufs[b].ptr();
                    for k in 0..per_buf {
                        let at = dev_addr + ((b * per_buf + k) * cmd) as u64;
                        // SAFETY: inside ring buffer `b` and inside the caller's device range.
                        unsafe {
                            let host = base.add(k * cmd);
                            if dir == 0 {
                                dev.enqueue_d2h_raw(at, cmd as u64, host)?;
                            } else {
                                dev.enqueue_h2d_raw(host, cmd as u64, at)?;
                            }
                        }
                    }
                }
                // SAFETY: the streams belong to `dev`.
                unsafe {
                    dev.sync_stream(if dir == 0 {
                        dev.d2h_stream()
                    } else {
                        dev.h2d_stream()
                    })?
                };
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
    // shares of whole pages; a job larger than a share is cut
    let share = total.div_ceil(threads).next_multiple_of(4096);
    let mut parts: Vec<Vec<Job>> = vec![Vec::new(); threads];
    let (mut t, mut used) = (0usize, 0usize);
    for j in jobs {
        let mut off = 0usize;
        while off < j.len {
            if used == share && t + 1 < threads {
                t += 1;
                used = 0;
            }
            let room = if t + 1 < threads {
                share - used
            } else {
                usize::MAX
            };
            let take = (j.len - off).min(room);
            parts[t].push(Job {
                // SAFETY: `off + take <= j.len`.
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

/// Measures the copies of this machine and card and prints them: the raw link in both
/// directions for two kinds of pinned memory and several command sizes, the host copy alone,
/// and whole copies through the ring. `gib`: device bytes used, in GiB.
pub fn probe(gib: u64) -> Result<(), String> {
    use crate::verified::HbmCat;
    let gib = gib.clamp(1, 16);
    let bytes = gib << 30;
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
    let mem = crate::hostmem::HostMemory::read();
    println!(
        "copy probe: {gib} GiB of device memory; host: {}; {} logical CPUs",
        mem.describe(),
        thread::available_parallelism().map_or(0, |n| n.get())
    );

    for kind in [RingKind::MappedHuge, RingKind::Malloc] {
        let t0 = Instant::now();
        let mut ring = XferRing::new(&dev, RING_BUFS, RING_BUF_BYTES, kind, 4)
            .map_err(|e| format!("ring {kind:?}: {e}"))?;
        println!(
            "ring {kind:?}: {} x {} MiB pinned in {:.1} ms",
            RING_BUFS,
            RING_BUF_BYTES >> 20,
            t0.elapsed().as_secs_f64() * 1e3
        );
        for cmd in [256usize << 10, 2 << 20, 8 << 20, 64 << 20] {
            let (d2h, h2d) = ring
                .link_rates(&dev, addr, bytes.min(2 << 30), cmd)
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
            let mut host = AnonMap::new(bytes)?;
            for piece_bytes in [256usize << 10, 32 << 20] {
                let pieces: Vec<Piece> = (0..bytes as usize / piece_bytes)
                    .map(|i| Piece {
                        dev: addr + (i * piece_bytes) as u64,
                        // SAFETY: inside `host`.
                        host: unsafe { host.as_mut_ptr().add(i * piece_bytes) },
                        len: piece_bytes,
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
    let n = (1usize << 30).min(bytes as usize);
    let mut a = AnonMap::new(n as u64)?;
    a.as_mut_slice().fill(1);
    for threads in [1usize, 2, 4, 6, 8] {
        let mut b = AnonMap::new(n as u64)?;
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
        let mut a = AnonMap::new(n as u64).unwrap();
        a.as_mut_slice().fill(3);
        let gib = |s: f64| n as f64 / s / (1u64 << 30) as f64;
        for piece in [256usize << 10, 2 << 20, 32 << 20] {
            for fresh in [true, false] {
                let mut b = AnonMap::new(n as u64).unwrap();
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
                let mut c = AnonMap::new(n as u64).unwrap();
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

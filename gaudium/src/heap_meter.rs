//! A count of the heap one thread holds while a metered task runs (the PDF reader).
//!
//! The process allocator is the system allocator wrapped by [`Meter`]: every allocation,
//! reallocation and release made on a thread whose meter is on adds or takes its size from
//! that thread's count. A thread whose meter is off (every thread, outside a metered task) pays
//! one thread-local read per allocation and counts nothing.
//!
//! The count is the heap the task itself allocated since it started, less what it released.
//! Each block allocated while the meter is on is recorded (its address, the bytes it was
//! charged, and the frame that owns it), and a release lowers a count only for a block that
//! count owns (`meter_owns` in `verified/heap_meter.bend`): releasing memory allocated before
//! the start (the request body a task rewrites, say) never cancels the task's own allocations.
//! A reallocation keeps the block's owner and changes its charge as `meter_regrow` decides:
//! every byte it adds is charged, no more than the bytes it removes is credited, and a block
//! allocated before the start is charged only for what it grows by. The count itself changes only
//! through the verified core too (`meter_charge`, `meter_credit`, `meter_rebill`, `meter_fold`):
//! a count past 64 bits stays at the largest one, which every cap refuses, rather than panic the
//! allocator. The task reads the count at
//! its own checkpoints and decides there what to do with it; the allocator never refuses or
//! fails an allocation.
//!
//! A metered task may run inside another (one PDF read inside the expansion of all the
//! documents of a request). The inner task's reading also gives the enclosing count as it is now
//! and the enclosing cap ([`Outer`]), so the inner task can check the two together; the inner
//! count is added to the enclosing one when the inner task ends, and the blocks the inner task
//! kept pass to the enclosing one. A block of the enclosing task released while the inner one
//! runs lowers the enclosing count, never the inner one.
//!
//! The record of the blocks is a table allocated from the system allocator directly; its own
//! memory is charged to the task on top when it grows, so a task of many small blocks pays for
//! the record of them too. It lives while any meter of the thread is on and is released with
//! the outermost one. A block allocated under the meter and released on another thread leaves
//! its record behind until the outermost meter ends; were the system allocator to hand that
//! address back to another thread and the block come back to be released here, its old charge
//! would be credited. Memory mapped outside the allocator (pinned host buffers, mapped weight
//! files, device memory) is never counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::UnsafeCell;

use crate::verified as v;

/// The system allocator, with the per-thread count.
pub struct Meter;

/// Most metered tasks one thread runs one inside another.
const MAX_FRAMES: usize = 8;

/// One metered task of the thread.
#[derive(Clone, Copy)]
struct Frame {
    /// Its serial: frames started later have larger ones.
    serial: u64,
    /// Bytes charged to blocks it owns, less those credited back.
    live: u64,
    /// The largest `live` seen since the start (with the frames started inside it).
    peak: u64,
    /// The heap the task allows itself (read back by the task, never enforced here).
    cap: u64,
    /// The enclosing frame's count when this one started (0: none).
    held_at_start: u64,
}

const NO_FRAME: Frame = Frame {
    serial: 0,
    live: 0,
    peak: 0,
    cap: 0,
    held_at_start: 0,
};

/// One recorded block (`ptr` 0: an empty slot).
#[derive(Clone, Copy)]
struct Slot {
    ptr: usize,
    charge: u64,
    serial: u64,
}

const SLOT: Layout = Layout::new::<Slot>();

/// The recorded blocks: open addressing with linear probing, at most three quarters full.
struct Table {
    slots: *mut Slot,
    /// A power of two, or 0 (no table).
    cap: usize,
    len: usize,
}

struct Local {
    depth: usize,
    frames: [Frame; MAX_FRAMES],
    next_serial: u64,
    table: Table,
}

thread_local! {
    // const-initialised and without a destructor: reading it never allocates, and it can be
    // read on any thread at any time (during its own teardown too)
    static LOCAL: UnsafeCell<Local> = const {
        UnsafeCell::new(Local {
            depth: 0,
            frames: [NO_FRAME; MAX_FRAMES],
            next_serial: 1,
            table: Table {
                slots: std::ptr::null_mut(),
                cap: 0,
                len: 0,
            },
        })
    };
}

#[inline]
fn hash(ptr: usize, cap: usize) -> usize {
    ((ptr >> 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize & (cap - 1)
}

impl Table {
    /// The slot of `ptr`, or the empty slot where it would go (`cap` must be nonzero).
    ///
    /// # Safety
    /// `slots` holds `cap` initialised slots.
    unsafe fn find(&self, ptr: usize) -> usize {
        let mut i = hash(ptr, self.cap);
        loop {
            let p = unsafe { (*self.slots.add(i)).ptr };
            if p == ptr || p == 0 {
                return i;
            }
            i = (i + 1) & (self.cap - 1);
        }
    }

    /// Takes the record of `ptr` out of the table.
    fn remove(&mut self, ptr: usize) -> Option<(u64, u64)> {
        if self.cap == 0 {
            return None;
        }
        // SAFETY: the table holds `cap` initialised slots while `cap` is nonzero
        unsafe {
            let mut i = self.find(ptr);
            let hit = *self.slots.add(i);
            if hit.ptr == 0 {
                return None;
            }
            self.len -= 1;
            // backward-shift deletion: no tombstones, every probe run stays unbroken
            let mask = self.cap - 1;
            let mut j = i;
            loop {
                j = (j + 1) & mask;
                let s = *self.slots.add(j);
                if s.ptr == 0 {
                    break;
                }
                let home = hash(s.ptr, self.cap);
                // `s` may move to `i` when `i` lies on its probe path from `home` to `j`
                if (j.wrapping_sub(home) & mask) >= (j.wrapping_sub(i) & mask) {
                    *self.slots.add(i) = s;
                    i = j;
                }
            }
            (*self.slots.add(i)).ptr = 0;
            Some((hit.charge, hit.serial))
        }
    }

    /// Records `ptr`; returns the bytes the table grew by, or `None` when it could not grow
    /// (the block is then not recorded: its release credits nothing).
    fn insert(&mut self, ptr: usize, charge: u64, serial: u64) -> Option<u64> {
        let mut grown = 0u64;
        if (self.len + 1) * 4 > self.cap * 3 {
            let cap = (self.cap * 2).max(64);
            let layout = Layout::array::<Slot>(cap).ok()?;
            // SAFETY: a nonzero layout; the slots are zeroed (every `ptr` 0: empty)
            let slots = unsafe { System.alloc_zeroed(layout) }.cast::<Slot>();
            if slots.is_null() {
                return None;
            }
            let old = std::mem::replace(self, Table { slots, cap, len: 0 });
            for k in 0..old.cap {
                // SAFETY: the old table holds `old.cap` initialised slots
                let s = unsafe { *old.slots.add(k) };
                if s.ptr != 0 {
                    // SAFETY: the new table holds `cap` slots and has room for every old one
                    unsafe {
                        let i = self.find(s.ptr);
                        *self.slots.add(i) = s;
                    }
                    self.len += 1;
                }
            }
            grown = ((cap - old.cap) * SLOT.size()) as u64;
            old.free();
        }
        // SAFETY: the table holds `cap` initialised slots and has room
        unsafe {
            let i = self.find(ptr);
            if (*self.slots.add(i)).ptr == 0 {
                self.len += 1;
            }
            *self.slots.add(i) = Slot {
                ptr,
                charge,
                serial,
            };
        }
        Some(grown)
    }

    fn free(self) {
        if self.cap != 0 {
            // SAFETY: allocated by `insert` with this layout
            unsafe {
                System.dealloc(
                    self.slots.cast(),
                    Layout::array::<Slot>(self.cap).expect("the table's layout"),
                )
            };
        }
    }
}

impl Local {
    fn top(&mut self) -> &mut Frame {
        &mut self.frames[self.depth - 1]
    }

    /// The frame that owns a block recorded with `serial` (`meter_owns`), if any.
    fn owner(&self, serial: u64) -> Option<usize> {
        (0..self.depth).rev().find(|&i| {
            let later = i + 1 < self.depth;
            let next = if later { self.frames[i + 1].serial } else { 0 };
            v::meter_owns(self.frames[i].serial, serial, later, next)
        })
    }

    /// Charges `n` bytes to `frame` (`meter_charge`).
    fn charge(&mut self, frame: usize, n: u64) {
        let f = &mut self.frames[frame];
        f.live = settle(v::meter_charge(f.live, n));
        f.peak = f.peak.max(f.live);
    }

    /// Credits `frame` with a released block's charge `n` (`meter_credit`).
    fn credit(&mut self, frame: usize, n: u64) {
        let f = &mut self.frames[frame];
        f.live = v::meter_credit(f.live, n);
    }

    /// Rebills one of `frame`'s blocks from its charge `was` to `now` (`meter_rebill`).
    fn rebill(&mut self, frame: usize, was: u64, now: u64) {
        let f = &mut self.frames[frame];
        f.live = settle(v::meter_rebill(f.live, was, now));
        f.peak = f.peak.max(f.live);
    }

    /// A block of `size` bytes allocated at `ptr`.
    fn allocated(&mut self, ptr: usize, size: usize) {
        let top = self.depth - 1;
        self.charge(top, size as u64);
        let serial = self.frames[top].serial;
        if let Some(grown) = self.table.insert(ptr, size as u64, serial) {
            self.charge(top, grown);
        }
    }

    /// The block at `ptr` released.
    fn released(&mut self, ptr: usize) {
        if let Some((charge, serial)) = self.table.remove(ptr) {
            if let Some(f) = self.owner(serial) {
                self.credit(f, charge);
            }
        }
    }

    /// The block at `p` of `old` bytes now at `q` with `new` bytes.
    fn reallocated(&mut self, p: usize, old: usize, q: usize, new: usize) {
        let (old, new) = (old as u64, new as u64);
        let (frame, charge, serial) = match self
            .table
            .remove(p)
            .and_then(|(c, s)| self.owner(s).map(|f| (f, c, s)))
        {
            Some(x) => x,
            // allocated before the meter started: owned by the top frame from here on, and
            // charged only what it grows by
            None => (self.depth - 1, 0, self.top().serial),
        };
        let charged = v::meter_regrow(charge, old, new);
        self.rebill(frame, charge, charged);
        if charged > 0 {
            if let Some(grown) = self.table.insert(q, charged, serial) {
                let top = self.depth - 1;
                self.charge(top, grown);
            }
        }
    }
}

/// A count the verified core grew: past 64 bits (`None`) the largest count, which every cap
/// refuses. The allocator never panics on a count.
#[inline]
fn settle(count: Option<u64>) -> u64 {
    count.unwrap_or(u64::MAX)
}

/// Runs `f` on the thread's meter when it is on.
#[inline]
fn on(f: impl FnOnce(&mut Local)) {
    let _ = LOCAL.try_with(|l| {
        // SAFETY: only this thread reaches its own `Local`, and nothing `f` does re-enters
        // the allocator's hooks (the table is allocated from `System` directly)
        let l = unsafe { &mut *l.get() };
        if l.depth > 0 {
            f(l);
        }
    });
}

// SAFETY: every call is passed to `System` unchanged; the count only reads the sizes.
unsafe impl GlobalAlloc for Meter {
    #[inline]
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            on(|m| m.allocated(p as usize, l.size()));
        }
        p
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            on(|m| m.allocated(p as usize, l.size()));
        }
        p
    }

    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        on(|m| m.released(p as usize));
    }

    #[inline]
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new_size) };
        if !q.is_null() {
            on(|m| m.reallocated(p as usize, l.size(), q as usize, new_size));
        }
        q
    }
}

/// The meter of the current thread, on while this lives: [`reading`] gives the heap the thread
/// allocated since [`Metered::start`], less what it released of it, and the cap the task set.
/// Dropping it turns the meter back to the state it had before: off, or, inside another
/// metered task, that task's count with this one's added to it. Meters end in the reverse
/// order of their starts (each is a guard on the stack of the thread that started it).
pub struct Metered {
    depth: usize,
    // a meter belongs to the thread that started it
    _thread: std::marker::PhantomData<*const ()>,
}

impl Metered {
    /// Starts the count at zero, with the cap `cap` (bytes) for the task to check against.
    /// Inside another metered task, the reading also gives that task's count and cap
    /// ([`Reading::outer`]).
    ///
    /// # Panics
    /// When the thread already runs [`MAX_FRAMES`] metered tasks one inside another.
    pub fn start(cap: u64) -> Metered {
        let depth = LOCAL.with(|l| {
            // SAFETY: as in `on`
            let l = unsafe { &mut *l.get() };
            if l.depth == MAX_FRAMES {
                return None;
            }
            let held = if l.depth > 0 { l.top().live } else { 0 };
            let serial = l.next_serial;
            l.next_serial += 1;
            l.frames[l.depth] = Frame {
                serial,
                live: 0,
                peak: 0,
                cap,
                held_at_start: held,
            };
            l.depth += 1;
            Some(l.depth)
        });
        let Some(depth) = depth else {
            panic!("more than {MAX_FRAMES} heap meters one inside another");
        };
        Metered {
            depth,
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for Metered {
    fn drop(&mut self) {
        LOCAL.with(|l| {
            // SAFETY: as in `on`
            let l = unsafe { &mut *l.get() };
            debug_assert_eq!(l.depth, self.depth, "heap meters end in reverse order");
            let inner = *l.top();
            l.depth -= 1;
            if l.depth > 0 {
                let p = l.top();
                p.peak = p
                    .peak
                    .max(p.live.max(inner.held_at_start).saturating_add(inner.peak));
                p.live = settle(v::meter_fold(p.live, inner.live));
                p.peak = p.peak.max(p.live);
            } else {
                let t = std::mem::replace(
                    &mut l.table,
                    Table {
                        slots: std::ptr::null_mut(),
                        cap: 0,
                        len: 0,
                    },
                );
                t.free();
            }
        });
    }
}

/// The current thread's count, when its meter is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reading {
    /// Bytes allocated since the start, less those released.
    pub live: u64,
    /// The largest `live` since the start.
    pub peak: u64,
    /// The cap the task set.
    pub cap: u64,
    /// The enclosing task's count and its cap (`None`: not nested).
    pub outer: Option<Outer>,
}

/// The enclosing metered task of a nested one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outer {
    /// Its count now: what it held when the nested task started, less what of it was released
    /// since, plus what its blocks grew by.
    pub held: u64,
    /// Its cap.
    pub cap: u64,
}

/// The current thread's count (`None` when its meter is off).
#[inline]
pub fn reading() -> Option<Reading> {
    LOCAL.with(|l| {
        // SAFETY: as in `on`
        let l = unsafe { &*l.get() };
        (l.depth > 0).then(|| {
            let f = l.frames[l.depth - 1];
            Reading {
                live: f.live,
                peak: f.peak,
                cap: f.cap,
                outer: (l.depth > 1).then(|| {
                    let o = l.frames[l.depth - 2];
                    Outer {
                        held: o.live,
                        cap: o.cap,
                    }
                }),
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    fn live() -> u64 {
        reading().unwrap().live
    }

    #[test]
    fn counts_only_while_on_and_only_this_thread() {
        assert_eq!(reading(), None);
        let v: Vec<u8> = black_box(Vec::with_capacity(1 << 20));
        {
            let _m = Metered::start(7);
            assert_eq!(
                reading(),
                Some(Reading {
                    live: 0,
                    peak: 0,
                    cap: 7,
                    outer: None,
                })
            );
            let a: Vec<u8> = black_box(Vec::with_capacity(4 << 20));
            let Reading { live, peak, .. } = reading().unwrap();
            assert!(live >= 4 << 20 && peak >= live, "{live} {peak}");
            // another thread's allocations are not counted here
            std::thread::spawn(|| drop(black_box(Vec::<u8>::with_capacity(64 << 20))))
                .join()
                .unwrap();
            assert!(self::live() < 5 << 20);
            drop(a);
            let Reading { live, peak, .. } = reading().unwrap();
            assert!(live < 1 << 20, "{live}");
            assert!(peak >= 4 << 20);
            // releasing memory allocated before the start never takes the count below zero
            drop(v);
            assert!(self::live() < 1 << 20);
        }
        assert_eq!(reading(), None);
    }

    #[test]
    fn growth_by_realloc_is_counted() {
        let _m = Metered::start(0);
        let mut v: Vec<u8> = Vec::with_capacity(16);
        v.extend(std::iter::repeat_n(0u8, 8 << 20));
        assert!(live() >= 8 << 20);
        v.truncate(0);
        v.shrink_to_fit();
        assert!(live() < 1 << 20);
    }

    #[test]
    fn a_nested_meter_adds_its_count_to_the_outer_one() {
        let _outer = Metered::start(1);
        let keep: Vec<u8> = black_box(Vec::with_capacity(2 << 20));
        let kept = {
            let _inner = Metered::start(2);
            assert!(live() < 1 << 20);
            assert_eq!(reading().unwrap().cap, 2);
            // the enclosing count as it was when this one started, and its cap
            let o = reading().unwrap().outer.unwrap();
            assert!(o.held >= 2 << 20 && o.held < 3 << 20, "{o:?}");
            assert_eq!(o.cap, 1);
            drop(black_box(Vec::<u8>::with_capacity(8 << 20)));
            black_box(Vec::<u8>::with_capacity(3 << 20))
        };
        let r = reading().unwrap();
        assert_eq!(r.cap, 1);
        assert_eq!(r.outer, None);
        assert!(r.live >= 5 << 20 && r.live < 6 << 20, "{r:?}");
        assert!(r.peak >= 10 << 20, "{r:?}");
        drop((keep, kept));
    }

    #[test]
    fn releasing_memory_from_before_the_start_credits_nothing() {
        // Codex repro (#136): a free of a block allocated before the meter started (a request
        // body the task rewrites) took its size from the count and cancelled live allocations
        let before: Vec<u8> = black_box(Vec::with_capacity(4 << 20));
        let _m = Metered::start(0);
        let mine: Vec<u8> = black_box(Vec::with_capacity(4 << 20));
        drop(before);
        assert!(live() >= 4 << 20, "{}", live());
        drop(mine);
        assert!(live() < 1 << 20, "{}", live());
    }

    #[test]
    fn a_block_from_before_the_start_is_charged_only_what_it_grows_by() {
        let mut before: Vec<u8> = black_box(Vec::with_capacity(2 << 20));
        let _m = Metered::start(0);
        let mine: Vec<u8> = black_box(Vec::with_capacity(2 << 20));
        let base = live();
        before.reserve_exact(6 << 20);
        let grown = live() - base;
        assert!((4 << 20..5 << 20).contains(&grown), "{grown}");
        // shrinking it credits no more than it was charged, and releasing it the rest of that
        before.truncate(0);
        before.shrink_to(1 << 20);
        assert!(live() >= base, "{} {base}", live());
        drop(before);
        assert!(
            live() >= base && live() < base + (1 << 20),
            "{} {base}",
            live()
        );
        drop(mine);
        assert!(live() < 1 << 20);
    }

    #[test]
    fn a_release_credits_the_frame_that_owns_the_block() {
        let _outer = Metered::start(1);
        let outer_block: Vec<u8> = black_box(Vec::with_capacity(3 << 20));
        let kept = {
            let _inner = Metered::start(2);
            let inner_block: Vec<u8> = black_box(Vec::with_capacity(2 << 20));
            let before = reading().unwrap();
            // the enclosing task's block lowers the enclosing count, not this one
            drop(outer_block);
            let after = reading().unwrap();
            assert_eq!(after.live, before.live);
            let (b, a) = (before.outer.unwrap().held, after.outer.unwrap().held);
            assert!(b >= 3 << 20 && a < 1 << 20, "{b} {a}");
            inner_block
        };
        // the block the inner task kept passes to the enclosing one, and its release credits it
        let r = reading().unwrap();
        assert!(r.live >= 2 << 20 && r.live < 3 << 20, "{r:?}");
        drop(kept);
        assert!(live() < 1 << 20, "{}", live());
        // a new inner task does not own the enclosing task's blocks, nor those of an inner
        // task that has ended
        let again: Vec<u8> = black_box(Vec::with_capacity(2 << 20));
        let _inner = Metered::start(3);
        drop(again);
        assert!(live() < 1 << 10, "{}", live());
        assert!(reading().unwrap().outer.unwrap().held < 1 << 20);
    }

    #[test]
    fn a_count_past_64_bits_stays_at_the_largest() {
        // a hostile document can never panic the allocator: the growth that overflows is `None`
        // from the verified core, and the count stays at the largest one, which every cap refuses
        assert_eq!(v::meter_charge(u64::MAX, 1), None);
        assert_eq!(v::meter_fold(u64::MAX - 1, 2), None);
        assert_eq!(v::meter_rebill(u64::MAX, 0, 1), None);
        assert_eq!(settle(v::meter_charge(u64::MAX, 1)), u64::MAX);
        assert!(!v::pdf_within(settle(None), u64::MAX - 1));
        // today's rebill: the growth charged, the shrinkage credited, never below zero
        assert_eq!(v::meter_rebill(100, 10, 30), Some(120));
        assert_eq!(v::meter_rebill(100, 30, 10), Some(80));
        assert_eq!(v::meter_rebill(5, 30, 10), Some(0));
        assert_eq!(v::meter_credit(5, 30), 0);
    }

    #[test]
    fn many_small_blocks_are_recorded_and_released() {
        let _m = Metered::start(0);
        let blocks: Vec<Box<[u8; 32]>> = (0..100_000).map(|_| Box::new([0u8; 32])).collect();
        let held = live();
        // the blocks, the vector, and the record of them
        assert!(held >= 100_000 * 32, "{held}");
        drop(blocks);
        // only the record of them is left
        assert!(live() <= held - 100_000 * 32, "{} {held}", live());
        assert!(live() < 100_000 * 64, "{}", live());
    }
}

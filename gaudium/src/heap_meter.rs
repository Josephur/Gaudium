//! A count of the heap one thread holds while a metered task runs (the PDF reader).
//!
//! The process allocator is the system allocator wrapped by [`Meter`]: every allocation,
//! reallocation and release made on a thread whose meter is on adds or takes its size from
//! that thread's count. A thread whose meter is off (every thread, outside a metered task) pays
//! one thread-local read per allocation and counts nothing.
//!
//! The count is the heap the task itself allocated since it started, less what it released
//! (never below zero: a release of memory allocated before the start does not lower it). The
//! task reads it at its own checkpoints and decides there what to do with it; the allocator
//! never refuses or fails an allocation. Memory mapped outside the allocator (pinned host
//! buffers, mapped weight files, device memory) is never counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// The system allocator, with the per-thread count.
pub struct Meter;

#[derive(Clone, Copy)]
struct State {
    on: bool,
    /// Bytes allocated since the start, less those released.
    live: u64,
    /// The largest `live` seen since the start.
    peak: u64,
    /// The heap the task allows itself (read back by the task, never enforced here).
    cap: u64,
}

const OFF: State = State {
    on: false,
    live: 0,
    peak: 0,
    cap: 0,
};

thread_local! {
    // const-initialised and without a destructor: reading it never allocates, and it can be
    // read on any thread at any time (during its own teardown too)
    static STATE: Cell<State> = const { Cell::new(OFF) };
}

#[inline]
fn grew(n: usize) {
    let _ = STATE.try_with(|s| {
        let mut st = s.get();
        if st.on {
            st.live = st.live.saturating_add(n as u64);
            st.peak = st.peak.max(st.live);
            s.set(st);
        }
    });
}

#[inline]
fn shrank(n: usize) {
    let _ = STATE.try_with(|s| {
        let mut st = s.get();
        if st.on {
            st.live = st.live.saturating_sub(n as u64);
            s.set(st);
        }
    });
}

// SAFETY: every call is passed to `System` unchanged; the count only reads the sizes.
unsafe impl GlobalAlloc for Meter {
    #[inline]
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            grew(l.size());
        }
        p
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            grew(l.size());
        }
        p
    }

    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        shrank(l.size());
    }

    #[inline]
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new_size) };
        if !q.is_null() {
            if new_size >= l.size() {
                grew(new_size - l.size());
            } else {
                shrank(l.size() - new_size);
            }
        }
        q
    }
}

/// The meter of the current thread, on while this lives: [`reading`] gives the heap the thread
/// allocated since [`Metered::start`], less what it released, and the cap the task set.
/// Dropping it turns the meter back to the state it had before: off, or, inside another
/// metered task, that task's count with this one's added to it.
pub struct Metered {
    before: State,
}

impl Metered {
    /// Starts the count at zero, with the cap `cap` (bytes) for the task to check against.
    pub fn start(cap: u64) -> Metered {
        let before = STATE.with(|s| {
            s.replace(State {
                on: true,
                live: 0,
                peak: 0,
                cap,
            })
        });
        Metered { before }
    }
}

impl Drop for Metered {
    fn drop(&mut self) {
        STATE.with(|s| {
            let inner = s.get();
            let mut st = self.before;
            if st.on {
                st.peak = st.peak.max(st.live.saturating_add(inner.peak));
                st.live = st.live.saturating_add(inner.live);
            }
            s.set(st);
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
}

/// The current thread's count (`None` when its meter is off).
#[inline]
pub fn reading() -> Option<Reading> {
    let st = STATE.with(Cell::get);
    st.on.then_some(Reading {
        live: st.live,
        peak: st.peak,
        cap: st.cap,
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
                    cap: 7
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
            drop(black_box(Vec::<u8>::with_capacity(8 << 20)));
            black_box(Vec::<u8>::with_capacity(3 << 20))
        };
        let r = reading().unwrap();
        assert_eq!(r.cap, 1);
        assert!(r.live >= 5 << 20 && r.live < 6 << 20, "{r:?}");
        assert!(r.peak >= 10 << 20, "{r:?}");
        drop((keep, kept));
    }
}

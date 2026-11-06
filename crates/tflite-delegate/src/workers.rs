// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Persistent worker threads for the kernels, sized from the interpreter's thread count.

use std::cell::UnsafeCell;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, TryLockError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use edgefirst_kernels::Pool;

/// How long an idle worker polls for the next job before it sleeps. The kernels of one invoke
/// follow each other within microseconds, so workers stay awake through an invoke and sleep
/// between invokes.
const SPIN: Duration = Duration::from_micros(200);

type Task = &'static (dyn Fn(usize) + Sync);

/// State shared by the caller running a job and the workers.
///
/// A job is published by writing `job`, opening it and bumping `epoch`. A worker joins by
/// incrementing `active` and only then checking that the job it saw is still open; the caller
/// closes the job and waits for `active` to drop to zero before it returns. Both sides use
/// sequentially consistent accesses, so either the caller sees the worker in `active` or the
/// worker sees the job closed: no worker touches a task after `run` returns.
struct Shared {
    epoch: AtomicU32,
    open: AtomicBool,
    active: AtomicUsize,
    next: AtomicUsize,
    /// Written only while the job is closed and no worker is active.
    job: UnsafeCell<Option<(Task, usize)>>,
    panicked: AtomicBool,
    stop: AtomicBool,
    /// Set when the partition's invoke ends: idle workers sleep at once instead of polling,
    /// leaving the CPUs to the operators that run between partitions.
    rest: AtomicBool,
    parked: AtomicUsize,
    sleep: Mutex<()>,
    wake: Condvar,
}

// SAFETY: `job` is written only between jobs, when no worker can read it (see `Shared`); every
// other field is `Sync`.
unsafe impl Sync for Shared {}

impl Shared {
    fn drain(&self, task: &(dyn Fn(usize) + Sync), n: usize) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            task(i);
        }
    }

    fn worker(&self) {
        // The epoch the pool started at, so a worker that starts late still joins the first job.
        let mut seen = 0;
        loop {
            let mut idle_since = None;
            let mut polls = 0u32;
            loop {
                if self.stop.load(Ordering::Acquire) {
                    return;
                }
                let e = self.epoch.load(Ordering::SeqCst);
                if e != seen {
                    seen = e;
                    break;
                }
                polls = polls.wrapping_add(1);
                if self.rest.load(Ordering::Relaxed)
                    || polls.is_multiple_of(64)
                        && idle_since.get_or_insert_with(Instant::now).elapsed() >= SPIN
                {
                    self.park(seen);
                    idle_since = None;
                }
                std::hint::spin_loop();
            }
            self.active.fetch_add(1, Ordering::SeqCst);
            if self.open.load(Ordering::SeqCst) && self.epoch.load(Ordering::SeqCst) == seen {
                // SAFETY: the job is open and this worker is counted in `active`, so the
                // caller neither rewrites `job` nor returns until this worker leaves.
                if let Some((task, n)) = unsafe { *self.job.get() } {
                    if catch_unwind(AssertUnwindSafe(|| self.drain(task, n))).is_err() {
                        self.panicked.store(true, Ordering::Relaxed);
                    }
                }
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Sleeps until the epoch moves past `seen` or the pool stops.
    fn park(&self, seen: u32) {
        let mut guard = lock(&self.sleep);
        self.parked.fetch_add(1, Ordering::SeqCst);
        while self.epoch.load(Ordering::SeqCst) == seen && !self.stop.load(Ordering::SeqCst) {
            guard = self
                .wake
                .wait(guard)
                .unwrap_or_else(PoisonError::into_inner);
        }
        self.parked.fetch_sub(1, Ordering::SeqCst);
    }

    fn wake_parked(&self) {
        if self.parked.load(Ordering::SeqCst) > 0 {
            // Taking the lock orders this wake after any worker that counted itself parked has
            // started waiting.
            drop(lock(&self.sleep));
            self.wake.notify_all();
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Closes the running job and waits for the workers in it, also when the caller's own tasks
/// panic.
struct Close<'a>(&'a Shared);

impl Drop for Close<'_> {
    fn drop(&mut self) {
        self.0.open.store(false, Ordering::SeqCst);
        // Workers finish their last task within microseconds; one that was preempted could take
        // a scheduler slice, so spinning gives way to yielding.
        let mut spins = 0u32;
        while self.0.active.load(Ordering::SeqCst) != 0 {
            if spins < 1024 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        // SAFETY: the job is closed and no worker is active.
        unsafe { *self.0.job.get() = None };
    }
}

/// A [`Pool`] of the calling thread plus persistent workers.
pub(crate) struct Workers {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    running: Mutex<()>,
}

impl Workers {
    /// A pool of `threads` threads, the caller included. Fewer workers start if the system
    /// refuses to spawn some.
    pub(crate) fn new(threads: NonZeroUsize) -> Self {
        let shared = Arc::new(Shared {
            epoch: AtomicU32::new(0),
            open: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            rest: AtomicBool::new(false),
            parked: AtomicUsize::new(0),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
        });
        let handles = (1..threads.get())
            .filter_map(|i| {
                let s = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("edgefirst-{i}"))
                    .spawn(move || s.worker())
                    .ok()
            })
            .collect();
        Self {
            shared,
            handles,
            running: Mutex::new(()),
        }
    }

    /// Threads that run tasks, the caller included.
    pub(crate) fn threads(&self) -> usize {
        self.handles.len() + 1
    }

    /// Sends idle workers to sleep until the next job, as at the end of an invoke.
    pub(crate) fn rest(&self) {
        self.shared.rest.store(true, Ordering::Relaxed);
    }
}

impl Pool for Workers {
    fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        if self.handles.is_empty() || n_tasks <= 1 {
            (0..n_tasks).for_each(task);
            return;
        }
        // A call from inside a task runs inline. (Partitions on other interpreter threads wait
        // for the `Threads` guard instead, which their invoke holds throughout.)
        let _running = match self.running.try_lock() {
            Ok(g) => g,
            Err(TryLockError::Poisoned(p)) => p.into_inner(),
            Err(TryLockError::WouldBlock) => {
                (0..n_tasks).for_each(task);
                return;
            }
        };
        let s = &*self.shared;
        // SAFETY: `Close` keeps the workers from using `task` past this call, so extending
        // its lifetime is never observed.
        let erased: Task = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Task>(task) };
        // SAFETY: the previous `Close` closed the job, and workers read `job` only after they
        // see it open, which the store to `open` below publishes after this write.
        unsafe { *s.job.get() = Some((erased, n_tasks)) };
        s.panicked.store(false, Ordering::Relaxed);
        s.rest.store(false, Ordering::Relaxed);
        s.next.store(0, Ordering::Relaxed);
        s.open.store(true, Ordering::SeqCst);
        let close = Close(s);
        s.epoch.fetch_add(1, Ordering::SeqCst);
        s.wake_parked();
        s.drain(task, n_tasks);
        drop(close);
        assert!(
            !s.panicked.swap(false, Ordering::Relaxed),
            "a kernel task panicked on a worker thread"
        );
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        drop(lock(&self.shared.sleep));
        self.shared.wake.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// The CPUs this process may use, read once: `available_parallelism` reads cgroup files.
fn cpus() -> usize {
    static CPUS: OnceLock<usize> = OnceLock::new();
    *CPUS.get_or_init(|| std::thread::available_parallelism().map_or(1, NonZeroUsize::get))
}

/// The delegate's workers, shared by its partitions and resized to follow the interpreter.
#[derive(Default)]
pub(crate) struct Threads(Mutex<Option<Workers>>);

impl Threads {
    /// Workers for `threads` threads, at most one per CPU this process may use (`None` when
    /// that is one), held for the length of an invoke.
    pub(crate) fn get(&self, threads: usize) -> MutexGuard<'_, Option<Workers>> {
        let threads = threads.min(cpus());
        let mut guard = lock(&self.0);
        if guard.as_ref().map_or(1, Workers::threads) != threads {
            // Drop the old workers before starting new ones.
            *guard = None;
            *guard = NonZeroUsize::new(threads)
                .filter(|n| n.get() > 1)
                .map(Workers::new);
        }
        guard
    }
}

impl std::fmt::Debug for Threads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let threads = lock(&self.0).as_ref().map_or(1, Workers::threads);
        f.debug_struct("Threads")
            .field("threads", &threads)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_once_across_many_jobs() {
        let w = Workers::new(NonZeroUsize::new(4).unwrap());
        for round in 0..2000usize {
            let n = round % 37;
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            w.run(n, &|i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                counts.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                "round {round}"
            );
        }
    }

    #[test]
    fn workers_wake_after_sleeping() {
        let w = Workers::new(NonZeroUsize::new(3).unwrap());
        for _ in 0..3 {
            std::thread::sleep(SPIN * 5);
            let seen = Mutex::new(std::collections::HashSet::new());
            let sum = AtomicUsize::new(0);
            w.run(64, &|i| {
                sum.fetch_add(i, Ordering::Relaxed);
                lock(&seen).insert(std::thread::current().id());
                std::thread::sleep(Duration::from_millis(1));
            });
            assert_eq!(sum.load(Ordering::Relaxed), 64 * 63 / 2);
            assert!(lock(&seen).len() > 1, "only the caller ran tasks");
        }
    }

    #[test]
    fn nested_runs_run_inline() {
        let w = Workers::new(NonZeroUsize::new(3).unwrap());
        let total = AtomicUsize::new(0);
        w.run(8, &|_| {
            w.run(4, &|_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(total.load(Ordering::Relaxed), 32);
    }

    #[test]
    fn a_worker_panic_reaches_the_caller() {
        let w = Workers::new(NonZeroUsize::new(2).unwrap());
        let main = std::thread::current().id();
        let r = catch_unwind(AssertUnwindSafe(|| {
            w.run(64, &|_| {
                assert_eq!(std::thread::current().id(), main, "boom");
                std::thread::sleep(Duration::from_millis(1));
            });
        }));
        assert!(r.is_err());
        let n = AtomicUsize::new(0);
        w.run(10, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn threads_follow_the_requested_count_up_to_the_cpus() {
        let t = Threads::default();
        let count = |n: usize| t.get(n).as_ref().map_or(1, Workers::threads);
        assert_eq!(count(1), 1);
        assert_eq!(count(4), 4.min(cpus()));
        assert_eq!(count(2), 2.min(cpus()));
        assert_eq!(count(cpus() + 8), cpus());
        assert_eq!(count(1), 1);
    }

    #[test]
    fn resting_workers_sleep_at_once_and_wake_for_the_next_job() {
        let w = Workers::new(NonZeroUsize::new(3).unwrap());
        let n = AtomicUsize::new(0);
        w.run(16, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        w.rest();
        let workers = w.handles.len();
        let start = Instant::now();
        while w.shared.parked.load(Ordering::SeqCst) != workers {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "workers never slept"
            );
            std::thread::yield_now();
        }
        w.run(16, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 32);
        assert!(!w.shared.rest.load(Ordering::Relaxed));
    }
}

//! Keeping an Earth world's tiles local ahead of the generator.
//!
//! The generator's blocking forms stall a worker thread while a tile downloads. Two things keep that rare:
//!
//! * [`preload`] runs once when a world opens, before the player is placed: it fetches the tiles around the
//!   origin that the spawn and the first cubes read.
//! * A [`Prefetcher`] follows the players. On its own thread it works out, from the generator's `missing_tiles*`
//!   helpers, which tiles around them are not local yet, and fetches them one at a time with a gap between
//!   requests. The base-zoom tiles within the load distance plus a margin come first, then the coarser tiles
//!   the far view reads at each level (it reads zoom `zoom_for_step(1 << level)` for nodes of that level),
//!   ring by ring.
//!
//! * Tiles the far view found missing (`EarthGenerator::request_tiles`: its workers answer "later" and ask, and
//!   never wait) are fetched first, ahead of the plan.
//!
//! A tile that fails is left alone for a while (doubling up to five minutes) so a dead server is not hammered;
//! the generator's own blocking reads keep retrying what they need.
use horizon_earth::{FetchError, LoadError, TileId};
use horizon_world::cubic::custom::CustomCubicGenerator;
use horizon_world::cubic::earth::EarthGenerator;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

/// Chunks past the load distance whose base-zoom tiles are kept local.
pub const MARGIN_CHUNKS: i32 = 6;
/// The base-zoom radius kept local whatever the settings, in blocks: the finest far nodes (level 1) are built
/// from cubes out to about this far.
pub const MIN_RADIUS: i32 = 512;
/// The farthest far view, in chunks, that tiles are kept for (about 65 km).
pub const MAX_FAR_CHUNKS: i32 = 4096;
/// Between two requests of the prefetcher.
pub const GAP: Duration = Duration::from_millis(150);
/// A tile that failed is left alone this long at first, doubling.
const RETRY_FIRST: Duration = Duration::from_secs(5);
const RETRY_CAP: Duration = Duration::from_secs(300);
/// A missing tile is not asked for again within this long.
const NOT_FOUND_RETRY: Duration = Duration::from_secs(600);
/// How long the prefetcher sleeps with nothing to do before it looks again (for tiles whose retry is due).
const IDLE: Duration = Duration::from_secs(30);

/// Where the players are and how far the world reads around them.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Target {
    /// The chunk column of each player.
    pub centres: Vec<(i32, i32)>,
    /// The load distance, in chunks.
    pub near_chunks: i32,
    /// How far the far view reaches, in chunks (0 when there is none).
    pub far_chunks: i32,
}

/// The tiles around `target` that are not local, nearest first: the base-zoom ones within the load distance and
/// the margin, then the coarser ones the far view's levels read.
pub fn plan(generator: &EarthGenerator, target: &Target) -> Vec<TileId> {
    let mut seen = HashSet::new();
    let mut tiles = Vec::new();
    let mut add = |ids: Vec<TileId>| {
        for id in ids {
            if seen.insert(id) {
                tiles.push(id);
            }
        }
    };
    let far = target.far_chunks.clamp(0, MAX_FAR_CHUNKS) * 16;
    let near = (target.near_chunks.max(0) + MARGIN_CHUNKS).saturating_mul(16);
    let base_radius = if far > 0 { near.max(MIN_RADIUS) } else { near };
    for &(cx, cz) in &target.centres {
        let (x, z) = (
            cx.saturating_mul(16).saturating_add(8),
            cz.saturating_mul(16).saturating_add(8),
        );
        add(generator.missing_tiles(
            x.saturating_sub(base_radius),
            z.saturating_sub(base_radius),
            x.saturating_add(base_radius),
            z.saturating_add(base_radius),
        ));
    }
    // Level 1 reads the base zoom (it is built from level-0 cubes); each level from 2 up reads its own zoom
    // over a ring that widens with the level, to where the far view ends.
    if far > 0 {
        for level in 2..=24u32 {
            if (32i64 << level) >= i64::from(far) {
                break;
            }
            let step = 1i32 << level;
            let radius = (256i64 << level).min(i64::from(far)) as i32;
            for &(cx, cz) in &target.centres {
                let (x, z) = (
                    cx.saturating_mul(16).saturating_add(8),
                    cz.saturating_mul(16).saturating_add(8),
                );
                add(generator.missing_tiles_for_step(
                    x.saturating_sub(radius),
                    z.saturating_sub(radius),
                    x.saturating_add(radius),
                    z.saturating_add(radius),
                    step,
                ));
            }
        }
    }
    tiles
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Default)]
struct Wanted {
    target: Option<Target>,
    /// Counts every change of `target`, so a pass in progress notices.
    epoch: u64,
}

struct Shared {
    wanted: Mutex<Wanted>,
    wake: Condvar,
    stop: AtomicBool,
}

/// A background thread that fetches the tiles around the players. It stops when told to or dropped (after the
/// request in flight, which a stop does not wait for).
pub struct Prefetcher {
    shared: Arc<Shared>,
}

impl Prefetcher {
    /// Starts the thread for an Earth `generator`, a request every `gap` at most. `None` when it is not an
    /// Earth generator, or the thread cannot start.
    pub fn start(generator: &Arc<CustomCubicGenerator>, gap: Duration) -> Option<Self> {
        generator.earth_generator()?;
        let shared = Arc::new(Shared {
            wanted: Mutex::default(),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let weak = Arc::downgrade(generator);
        let theirs = shared.clone();
        // A request wakes the thread (taking the lock first, so a thread about to wait does not miss it).
        let waking = Arc::downgrade(&shared);
        generator
            .earth_generator()?
            .set_request_waker(Arc::new(move || {
                if let Some(shared) = waking.upgrade() {
                    let _wanted = lock(&shared.wanted);
                    shared.wake.notify_all();
                }
            }));
        std::thread::Builder::new()
            .name("Earth tiles".into())
            .spawn(move || run(&theirs, &weak, gap))
            .ok()?;
        Some(Self { shared })
    }

    /// Aims the prefetcher at `target`; a target equal to the last changes nothing. Cheap: the work is on the
    /// thread.
    pub fn follow(&self, target: Target) {
        let mut wanted = lock(&self.shared.wanted);
        if wanted.target.as_ref() != Some(&target) {
            wanted.target = Some(target);
            wanted.epoch += 1;
            self.shared.wake.notify_all();
        }
    }

    /// Tells the thread to stop. It does not wait: a download in flight ends by itself.
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        let _wanted = lock(&self.shared.wanted);
        self.shared.wake.notify_all();
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        self.stop();
    }
}

/// When a failed tile may be asked for again, and the wait that follows if it fails again.
struct Failure {
    retry_at: Instant,
    wait: Duration,
}

fn run(shared: &Shared, generator: &Weak<CustomCubicGenerator>, gap: Duration) {
    let mut failures: HashMap<TileId, Failure> = HashMap::new();
    let mut failing = false;
    loop {
        // Wait for a target (or for the idle time to pass, to retry what failed).
        let (target, epoch) = {
            let mut wanted = lock(&shared.wanted);
            loop {
                if shared.stop.load(Ordering::SeqCst) {
                    return;
                }
                if let Some(target) = wanted.target.clone() {
                    break (target, wanted.epoch);
                }
                wanted = shared
                    .wake
                    .wait(wanted)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let Some(generator) = generator.upgrade() else {
            return;
        };
        let Some(earth) = generator.earth_generator() else {
            return;
        };
        if earth.is_cancelled() {
            return;
        }
        let now = Instant::now();
        let due = |id: &TileId| failures.get(id).is_none_or(|f| f.retry_at <= now);
        let mut todo: VecDeque<TileId> = plan(earth, &target)
            .into_iter()
            .filter(|id| due(id))
            .collect();
        let mut interrupted = false;
        loop {
            // What the far view asked for since goes first.
            let now = Instant::now();
            for id in earth.take_requests().into_iter().rev() {
                if earth.field().store().cached(id).is_none()
                    && failures.get(&id).is_none_or(|f| f.retry_at <= now)
                {
                    todo.push_front(id);
                }
            }
            let Some(id) = todo.pop_front() else {
                break;
            };
            if shared.stop.load(Ordering::SeqCst)
                || earth.is_cancelled()
                || lock(&shared.wanted).epoch != epoch
            {
                interrupted = true;
                break;
            }
            match earth.field().store().load(id) {
                Ok(_) => {
                    failures.remove(&id);
                    if std::mem::take(&mut failing) {
                        log::info!("Horizon: Earth tile prefetching works again");
                    }
                }
                Err(e) => {
                    if !failing {
                        failing = true;
                        log::warn!("Horizon: Earth tile {id:?} did not download: {e}");
                    }
                    let wait = match &e {
                        LoadError::Fetch(FetchError::NotFound) => NOT_FOUND_RETRY,
                        _ => failures
                            .get(&id)
                            .map_or(RETRY_FIRST, |f| (f.wait * 2).min(RETRY_CAP)),
                    };
                    failures.insert(
                        id,
                        Failure {
                            retry_at: Instant::now() + wait,
                            wait,
                        },
                    );
                }
            }
            if !sleep(shared, gap) {
                return;
            }
        }
        if interrupted {
            continue;
        }
        // Nothing left to do for this target: wait for another, or for a retry to come due.
        let mut wanted = lock(&shared.wanted);
        if wanted.epoch == epoch && !shared.stop.load(Ordering::SeqCst) {
            let (guard, _) = shared
                .wake
                .wait_timeout(wanted, IDLE)
                .unwrap_or_else(PoisonError::into_inner);
            wanted = guard;
        }
        drop(wanted);
    }
}

/// Sleeps `gap`, waking early when asked to stop; whether it was not.
fn sleep(shared: &Shared, gap: Duration) -> bool {
    let wanted = lock(&shared.wanted);
    if shared.stop.load(Ordering::SeqCst) {
        return false;
    }
    let (_wanted, _) = shared
        .wake
        .wait_timeout_while(wanted, gap, |_| !shared.stop.load(Ordering::SeqCst))
        .unwrap_or_else(PoisonError::into_inner);
    !shared.stop.load(Ordering::SeqCst)
}

/// What [`preload`] did.
#[derive(Debug, Default)]
pub struct Preloaded {
    /// Tiles fetched.
    pub loaded: usize,
    /// Tiles that could not be had, with why.
    pub failed: Vec<(TileId, String)>,
}

/// Loads the tiles around the block position `at` that a world opening there reads: those within `chunks` of it
/// at the base zoom, and the coarser ones of the far view out to `far_chunks`. Each is retried a few times when
/// the failure is temporary; it stops at once when the generator is cancelled. A failure is reported, not fatal:
/// the generator's blocking reads try again as the cubes need the tiles.
pub fn preload(
    generator: &EarthGenerator,
    at: (i32, i32),
    chunks: i32,
    far_chunks: i32,
) -> Preloaded {
    let target = Target {
        centres: vec![(at.0 >> 4, at.1 >> 4)],
        near_chunks: chunks,
        far_chunks,
    };
    let mut done = Preloaded::default();
    for id in plan(generator, &target) {
        let mut delay = Duration::from_millis(200);
        for attempt in 0..3 {
            if generator.is_cancelled() {
                return done;
            }
            match generator.field().store().load(id) {
                Ok(_) => {
                    done.loaded += 1;
                    break;
                }
                Err(LoadError::Fetch(FetchError::Temporary(why))) if attempt < 2 => {
                    log::debug!("Horizon: Earth tile {id:?}: {why}; trying again");
                    std::thread::sleep(delay);
                    delay *= 2;
                }
                Err(e) => {
                    done.failed.push((id, e.to_string()));
                    break;
                }
            }
        }
    }
    done
}

#[cfg(test)]
mod tests;

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
//! * The coarse set goes first when the far view reaches voxels that wide ([`wants_coarse`]): the climate's coarse
//!   pieces (`EarthGenerator::coarse_climate_needs`), then the whole-world low-zoom tiles
//!   (`EarthGenerator::coarse_tiles_missing`, the player's place first), then `EarthGenerator::prepare_coarse`,
//!   which builds the sea mask on this thread so the far workers never do. The far nodes of those levels read
//!   nothing else.
//!
//! * Tiles the far view found missing (`EarthGenerator::request_tiles`: its workers answer "later" and ask, and
//!   never wait) are fetched first, ahead of the plan.
//!
//! * Land cover (`EarthGenerator::land_source`), when the world has it, is fetched on the same thread by the
//!   same rules: the byte ranges the far view found missing (`EarthGenerator::take_land_requests`; its nodes
//!   answered "later" and are asked again by the far view's own retry once the bytes are in), and the rounds of
//!   `land_needs` (the index of each file, then its tiles) for the base-zoom area around the players. Each
//!   piece goes to `LandCoverSource::satisfy`; a stale request is skipped (`LandCoverStore::is_missing`).
//!
//! A tile that fails is left alone for a while (doubling up to five minutes) so a dead server is not hammered;
//! the generator's own blocking reads keep retrying what they need.
use horizon_earth::{ClimateFetch, Fetch, FetchError, LoadError, TileId};
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
/// The farthest far view, in chunks, that tiles are kept for: the whole planet (its girth is about 40,075 km, which
/// is 2.5 million chunks at one metre a block). `far * 16` blocks still fits an `i32`.
pub const MAX_FAR_CHUNKS: i32 = 1 << 22;
/// The coarsest far level whose tiles [`plan`] asks for (nodes 2^22 blocks across, thousands of km at one metre a
/// block). The ring of a level is capped by the far distance, so the planet-wide levels read the few coarse tiles.
pub const MAX_PLAN_LEVEL: u32 = 22;
/// Tries one pass makes at building the sea mask (each after the tiles it named were fetched).
const COARSE_ROUNDS: u32 = 3;
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
        for level in 2..=MAX_PLAN_LEVEL {
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

/// Whether the far view of `target` reaches nodes whose voxels are wide enough for the coarse path
/// (`EarthGenerator::is_coarse_step`): then the coarse set is fetched first.
pub fn wants_coarse(generator: &EarthGenerator, target: &Target) -> bool {
    let far = i64::from(target.far_chunks.clamp(0, MAX_FAR_CHUNKS)) * 16;
    // A level's nodes are 16 << level blocks wide; the far view needs the levels whose nodes it spans (or half).
    far > 0
        && (2..=MAX_PLAN_LEVEL)
            .any(|level| (16i64 << level) <= far * 2 && generator.is_coarse_step(1 << level))
}

/// The coarse set's tiles still missing for the players, the first player's place first (the crosshair), then the
/// others'; none when the far view does not reach coarse nodes ([`wants_coarse`]).
pub fn coarse_plan(generator: &EarthGenerator, target: &Target) -> Vec<TileId> {
    if !wants_coarse(generator, target) {
        return Vec::new();
    }
    let mut seen = HashSet::new();
    let mut tiles = Vec::new();
    for &(cx, cz) in &target.centres {
        let at = (
            cx.saturating_mul(16).saturating_add(8),
            cz.saturating_mul(16).saturating_add(8),
        );
        for id in generator.coarse_tiles_missing(at) {
            if seen.insert(id) {
                tiles.push(id);
            }
        }
    }
    tiles
}

/// Rounds of `land_needs` one pass asks for (an index first, then its tiles, and a second file's); each piece that
/// arrives gives a round back, so a pass goes on while it makes progress and ends when a round finds nothing.
const LAND_ROUNDS: u32 = 6;

/// The land cover pieces still missing for the base-zoom area around each player, nearest first.
pub fn land_plan(generator: &EarthGenerator, target: &Target) -> Vec<Fetch> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let near = (target.near_chunks.max(0) + MARGIN_CHUNKS).saturating_mul(16);
    let radius = if target.far_chunks > 0 {
        near.max(MIN_RADIUS)
    } else {
        near
    };
    for &(cx, cz) in &target.centres {
        let (x, z) = (
            cx.saturating_mul(16).saturating_add(8),
            cz.saturating_mul(16).saturating_add(8),
        );
        for fetch in generator.land_needs(
            x.saturating_sub(radius),
            z.saturating_sub(radius),
            x.saturating_add(radius),
            z.saturating_add(radius),
            1,
        ) {
            if seen.insert(fetch) {
                out.push(fetch);
            }
        }
    }
    out
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
    let mut land_failures: HashMap<Fetch, Failure> = HashMap::new();
    let mut land_failing = false;
    let mut climate_failing = false;
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
        // The coarse set goes ahead of everything else: its climate pieces, its tiles (the player's place first),
        // then the sea mask built from them.
        let coarse_wanted = wants_coarse(earth, &target);
        let mut coarse_todo: VecDeque<TileId> = coarse_plan(earth, &target)
            .into_iter()
            .filter(|id| due(id))
            .collect();
        let mut climate_todo: VecDeque<ClimateFetch> = if coarse_wanted {
            earth.coarse_climate_needs().into()
        } else {
            VecDeque::new()
        };
        let mut coarse_prepared = !coarse_wanted;
        let mut coarse_tries = 0;
        let mut interrupted = false;
        let mut land_todo: VecDeque<Fetch> = VecDeque::new();
        let mut land_rounds = 0;
        loop {
            // Climate: a handful of pieces for the whole world (two indexes, a tile or two each); what the far
            // view asked for is fetched first of all. A failure is logged once and asked again later.
            if let Some(climate) = earth.climate_source() {
                for fetch in earth.take_climate_requests() {
                    if !climate.field().store().is_missing(&fetch) {
                        continue;
                    }
                    match climate.satisfy(&fetch) {
                        Ok(()) => climate_failing = false,
                        Err(e) => {
                            if !std::mem::replace(&mut climate_failing, true) {
                                log::warn!(
                                    "Horizon: Earth climate {:?} did not download: {e}",
                                    fetch.var
                                );
                            }
                        }
                    }
                }
            }
            // The coarse climate pieces come before the tiles; one a turn, with the gap between.
            if let Some(climate) = earth.climate_source()
                && let Some(fetch) = climate_todo.pop_front()
            {
                if shared.stop.load(Ordering::SeqCst)
                    || earth.is_cancelled()
                    || lock(&shared.wanted).epoch != epoch
                {
                    interrupted = true;
                    break;
                }
                if !climate.field().store().is_missing(&fetch) {
                    continue;
                }
                match climate.satisfy(&fetch) {
                    Ok(()) => climate_failing = false,
                    Err(e) => {
                        if !std::mem::replace(&mut climate_failing, true) {
                            log::warn!(
                                "Horizon: Earth climate {:?} did not download: {e}",
                                fetch.var
                            );
                        }
                    }
                }
                if !sleep(shared, gap) {
                    return;
                }
                continue;
            }
            // Land cover: what the far view asked for goes first, then the next round of what the area needs.
            if let Some(land) = earth.land_source() {
                let now = Instant::now();
                let due = |f: &Fetch, failures: &HashMap<Fetch, Failure>| {
                    failures.get(f).is_none_or(|x| x.retry_at <= now)
                };
                for fetch in earth.take_land_requests().into_iter().rev() {
                    if due(&fetch, &land_failures)
                        && land.field().store().is_missing(&fetch)
                        && !land_todo.contains(&fetch)
                    {
                        land_todo.push_front(fetch);
                    }
                }
                if land_todo.is_empty()
                    && todo.is_empty()
                    && coarse_todo.is_empty()
                    && land_rounds < LAND_ROUNDS
                {
                    land_rounds += 1;
                    land_todo.extend(
                        land_plan(earth, &target)
                            .into_iter()
                            .filter(|f| due(f, &land_failures)),
                    );
                }
                if let Some(fetch) = land_todo.pop_front() {
                    if shared.stop.load(Ordering::SeqCst)
                        || earth.is_cancelled()
                        || lock(&shared.wanted).epoch != epoch
                    {
                        interrupted = true;
                        break;
                    }
                    if !land.field().store().is_missing(&fetch) {
                        continue;
                    }
                    match land.satisfy(&fetch) {
                        Ok(()) => {
                            land_failures.remove(&fetch);
                            if std::mem::take(&mut land_failing) {
                                log::info!("Horizon: Earth land cover fetching works again");
                            }
                            // A piece came: the next round (an index leads to its tiles) is worth asking for.
                            land_rounds = land_rounds.saturating_sub(1);
                        }
                        // Marked as no data for the run by the source: nothing to retry.
                        Err(FetchError::Refused(why)) => {
                            log::warn!("Horizon: Earth land cover is not downloaded: {why}");
                        }
                        Err(e) => {
                            if !land_failing {
                                land_failing = true;
                                log::warn!(
                                    "Horizon: Earth land cover {} bytes {}+{} did not download: {e}",
                                    fetch.file,
                                    fetch.range.offset,
                                    fetch.range.len
                                );
                            }
                            let wait = land_failures
                                .get(&fetch)
                                .map_or(RETRY_FIRST, |f| (f.wait * 2).min(RETRY_CAP));
                            land_failures.insert(
                                fetch,
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
                    continue;
                }
            }
            // What the far view asked for since goes first.
            let now = Instant::now();
            for id in earth.take_requests().into_iter().rev() {
                if earth.field().store().cached(id).is_none()
                    && failures.get(&id).is_none_or(|f| f.retry_at <= now)
                {
                    todo.push_front(id);
                }
            }
            // The coarse tiles first. Once they are in, the sea mask is built here (never by a far worker).
            if coarse_todo.is_empty() && !coarse_prepared {
                match earth.prepare_coarse() {
                    Ok(()) => coarse_prepared = true,
                    Err(missing) => {
                        coarse_tries += 1;
                        let now = Instant::now();
                        for id in missing.into_iter().rev() {
                            if earth.field().store().cached(id).is_none()
                                && failures.get(&id).is_none_or(|f| f.retry_at <= now)
                            {
                                coarse_todo.push_front(id);
                            }
                        }
                        // Nothing to fetch for it (a failing tile, or another thread is building it), or too many
                        // tries: this pass goes on, a later pass tries again.
                        if coarse_todo.is_empty() || coarse_tries >= COARSE_ROUNDS {
                            coarse_prepared = true;
                        }
                    }
                }
            }
            let Some(id) = coarse_todo.pop_front().or_else(|| todo.pop_front()) else {
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

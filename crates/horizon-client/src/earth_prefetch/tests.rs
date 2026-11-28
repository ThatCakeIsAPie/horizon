//! Tests of the prefetcher over synthetic tiles: nothing here touches the network.
use super::*;
use horizon_earth::projection::latlon_to_tile_pixel;
use horizon_earth::testing::{MemoryFetcher, TempDir};
use horizon_earth::{COARSE_PREFETCH_ZOOM, Projection, TileFetcher, TileStore};
use horizon_world::cubic::custom::testing::default_registry;
use horizon_world::cubic::earth::EarthSettings;

const LAT: f64 = 46.0;
const LON: f64 = 7.5;
/// Metres a block spans: with the base zoom that gives, one tile is well over a hundred blocks wide, so the
/// area around a player is a handful of tiles.
const METRES_PER_BLOCK: f64 = 40.0;

fn terrain(lat: f64, lon: f64) -> f64 {
    1200.0 + 300.0 * (lat * 20.0).sin() + 200.0 * (lon * 30.0).cos()
}

struct Rig {
    fetcher: Arc<MemoryFetcher>,
    store: Arc<TileStore>,
    generator: Arc<CustomCubicGenerator>,
    projection: Projection,
    _dir: TempDir,
}

fn rig() -> Rig {
    let dir = TempDir::new();
    let fetcher = Arc::new(MemoryFetcher::new(terrain));
    let store = Arc::new(TileStore::new(dir.path(), fetcher.clone(), 256));
    let mut settings = EarthSettings::at(LAT, LON);
    settings.metres_per_block = METRES_PER_BLOCK;
    let generator = Arc::new(
        CustomCubicGenerator::earth(&settings, store.clone(), default_registry()).unwrap(),
    );
    Rig {
        fetcher,
        store,
        generator,
        projection: Projection::local(LAT, LON, METRES_PER_BLOCK).unwrap(),
        _dir: dir,
    }
}

impl Rig {
    fn earth(&self) -> &EarthGenerator {
        self.generator.earth_generator().unwrap()
    }

    /// The tile at zoom `zoom` under block (`x`, `z`), from the projection alone.
    fn tile_at(&self, x: i32, z: i32, zoom: u8) -> TileId {
        let at = self.projection.to_lat_lon(f64::from(x), f64::from(z));
        latlon_to_tile_pixel(at.lat, at.lon, zoom).0
    }

    fn wait_for(&self, done: impl Fn(&Rig) -> bool) -> bool {
        for _ in 0..300 {
            if done(self) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        done(self)
    }
}

fn around(chunks: i32, far: i32) -> Target {
    Target {
        centres: vec![(0, 0)],
        near_chunks: chunks,
        far_chunks: far,
    }
}

#[test]
fn the_plan_names_the_base_tiles_around_a_player_and_skips_the_local_ones() {
    let rig = rig();
    let base = rig.earth().base_zoom();
    // The blocks the player's load distance and the margin reach.
    let reach = (4 + MARGIN_CHUNKS) * 16;
    let corners = [
        (-reach, -reach),
        (reach, -reach),
        (-reach, reach),
        (reach, reach),
        (8, 8),
    ];
    let wanted: HashSet<TileId> = corners
        .iter()
        .map(|&(x, z)| rig.tile_at(x, z, base))
        .collect();
    assert!(!wanted.is_empty());
    let planned = plan(rig.earth(), &around(4, 0));
    assert!(
        wanted.iter().all(|id| planned.contains(id)),
        "{planned:?} lacks some of {wanted:?}"
    );
    assert!(
        planned.iter().all(|id| id.z == base),
        "no far view: only the base zoom"
    );
    // A tile that is local is not planned again.
    let have = *wanted.iter().next().unwrap();
    rig.store.load(have).unwrap();
    let again = plan(rig.earth(), &around(4, 0));
    assert!(!again.contains(&have));
    assert_eq!(again.len(), planned.len() - 1);
}

#[test]
fn the_plan_adds_coarser_tiles_for_the_far_view_after_the_base_ones() {
    let rig = rig();
    let base = rig.earth().base_zoom();
    let planned = plan(rig.earth(), &around(4, 400));
    let first_coarse = planned
        .iter()
        .position(|id| id.z < base)
        .expect("coarser tiles");
    assert!(planned[..first_coarse].iter().all(|id| id.z == base));
    // Each far level reads the zoom of its step, so zooms go down as the levels go up.
    let mut zooms: Vec<u8> = planned.iter().map(|id| id.z).collect();
    zooms.dedup();
    assert!(zooms.len() > 2, "{zooms:?}");
    assert!(zooms.windows(2).all(|w| w[0] >= w[1]), "{zooms:?}");
    // Nothing is planned twice.
    let unique: HashSet<_> = planned.iter().collect();
    assert_eq!(unique.len(), planned.len());
}

#[test]
fn a_player_far_from_the_origin_plans_the_tiles_there() {
    let rig = rig();
    let base = rig.earth().base_zoom();
    let (cx, cz) = (900, -400);
    let planned = plan(
        rig.earth(),
        &Target {
            centres: vec![(cx, cz)],
            near_chunks: 2,
            far_chunks: 0,
        },
    );
    let here = rig.tile_at(cx * 16 + 8, cz * 16 + 8, base);
    assert!(planned.contains(&here));
    assert!(!planned.contains(&rig.tile_at(8, 8, base)));
}

#[test]
fn the_thread_fetches_what_is_missing_around_the_player_and_nothing_twice() {
    let rig = rig();
    let base = rig.earth().base_zoom();
    let centre = rig.tile_at(8, 8, base);
    rig.store.load(centre).unwrap();
    assert_eq!(rig.fetcher.fetch_count(), 1);
    let wanted = plan(rig.earth(), &around(4, 0));
    assert!(!wanted.contains(&centre));
    let prefetcher = Prefetcher::start(&rig.generator, Duration::from_millis(1)).unwrap();
    prefetcher.follow(around(4, 0));
    assert!(rig.wait_for(|r| wanted.iter().all(|&id| r.store.is_local(id))));
    for &id in &wanted {
        assert_eq!(rig.fetcher.fetches_of(id), 1, "{id:?}");
    }
    assert_eq!(
        rig.fetcher.fetches_of(centre),
        1,
        "a local tile is not fetched again"
    );
    // Following the same place again asks for nothing.
    let count = rig.fetcher.fetch_count();
    prefetcher.follow(around(4, 0));
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(rig.fetcher.fetch_count(), count);
}

#[test]
fn the_thread_follows_the_player_to_a_new_place() {
    let rig = rig();
    let base = rig.earth().base_zoom();
    let prefetcher = Prefetcher::start(&rig.generator, Duration::from_millis(1)).unwrap();
    prefetcher.follow(around(2, 0));
    let first = rig.tile_at(8, 8, base);
    assert!(rig.wait_for(|r| r.store.is_local(first)));
    let (cx, cz) = (700, 300);
    prefetcher.follow(Target {
        centres: vec![(cx, cz)],
        near_chunks: 2,
        far_chunks: 0,
    });
    let there = rig.tile_at(cx * 16 + 8, cz * 16 + 8, base);
    assert!(rig.wait_for(|r| r.store.is_local(there)));
}

#[test]
fn a_dead_server_is_not_asked_again_and_again() {
    let rig = rig();
    rig.fetcher
        .fail_with(Some(FetchError::Temporary("no route".into())));
    let prefetcher = Prefetcher::start(&rig.generator, Duration::from_millis(1)).unwrap();
    prefetcher.follow(around(2, 0));
    assert!(rig.wait_for(|r| r.fetcher.fetch_count() > 0));
    std::thread::sleep(Duration::from_millis(300));
    let wanted = plan(rig.earth(), &around(2, 0));
    // Each missing tile was tried once, and the failures are left alone for seconds.
    let tried = rig.fetcher.fetch_count();
    assert!(
        tried <= wanted.len(),
        "{tried} requests for {} tiles",
        wanted.len()
    );
    for &id in &wanted {
        assert!(rig.fetcher.fetches_of(id) <= 1, "{id:?}");
    }
}

#[test]
fn a_stopped_thread_fetches_no_more() {
    let rig = rig();
    let prefetcher = Prefetcher::start(&rig.generator, Duration::from_millis(1)).unwrap();
    prefetcher.stop();
    std::thread::sleep(Duration::from_millis(50));
    prefetcher.follow(around(4, 0));
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(rig.fetcher.fetch_count(), 0);
}

#[test]
fn a_cancelled_generator_ends_the_thread_without_a_fetch() {
    let rig = rig();
    rig.generator.set_earth_cancelled(true);
    let prefetcher = Prefetcher::start(&rig.generator, Duration::from_millis(1)).unwrap();
    prefetcher.follow(around(4, 0));
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(rig.fetcher.fetch_count(), 0);
}

#[test]
fn only_an_earth_generator_has_a_prefetcher() {
    let flat = horizon_world::cubic::custom::testing::default_registry();
    let generator = Arc::new(
        CustomCubicGenerator::flat(
            &horizon_world::cubic::custom::FlatGeneratorSettings::default(),
            flat,
        )
        .unwrap(),
    );
    assert!(Prefetcher::start(&generator, Duration::from_millis(1)).is_none());
}

#[test]
fn preloading_fetches_the_tiles_around_the_origin_and_reports_the_ones_it_cannot_have() {
    let rig = rig();
    let base = rig.earth().base_zoom();
    let origin = rig.tile_at(0, 0, base);
    let done = preload(rig.earth(), (0, 0), 4, 0);
    assert!(done.failed.is_empty(), "{:?}", done.failed);
    assert!(done.loaded >= 1);
    assert!(rig.store.is_local(origin));
    assert!(rig.earth().missing_tiles(-100, -100, 100, 100).is_empty());
    // With a server that is gone, the failures are reported (a missing tile at once, a busy one after a few tries).
    let rig = self::rig();
    let missing = rig.tile_at(0, 0, base);
    rig.fetcher.make_missing(missing);
    let done = preload(rig.earth(), (0, 0), 1, 0);
    assert!(
        done.failed.iter().any(|(id, _)| *id == missing),
        "{:?}",
        done.failed
    );
    assert_eq!(
        rig.fetcher.fetches_of(missing),
        1,
        "a missing tile is not retried"
    );
}

#[test]
fn preloading_stops_at_once_when_cancelled_with_a_server_that_always_fails() {
    let rig = rig();
    rig.fetcher
        .fail_with(Some(FetchError::Temporary("down".into())));
    let generator = rig.generator.clone();
    let worker = std::thread::spawn(move || {
        let started = Instant::now();
        let done = preload(generator.earth_generator().unwrap(), (0, 0), 8, 100);
        (started.elapsed(), done)
    });
    std::thread::sleep(Duration::from_millis(100));
    rig.generator.set_earth_cancelled(true);
    let (elapsed, _) = worker.join().unwrap();
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
}

/// A far node that answered "later" because a land class was not held is built once the prefetcher has fetched
/// the pieces the node's read recorded (the far view asks again by its own retry; here the test asks).
#[test]
fn a_far_node_that_waited_for_its_class_is_built_once_the_prefetcher_has_the_data() {
    use horizon_earth::testing::MemoryLandCover;
    use horizon_earth::{LandCoverStore, WorldCoverFile};

    let dir = TempDir::new();
    let land_dir = TempDir::new();
    let fetcher = Arc::new(MemoryFetcher::new(terrain));
    let store = Arc::new(TileStore::new(dir.path(), fetcher, 256));
    let mut settings = EarthSettings::at(LAT, LON);
    settings.metres_per_block = METRES_PER_BLOCK;
    settings.land_cover = true;
    let file = WorldCoverFile::containing(LAT, LON).unwrap();
    let land = Arc::new(MemoryLandCover::new().with_classes(file, 192, 96, 2, |_, _| 30));
    let source = settings
        .land_cover_source(
            Arc::new(LandCoverStore::new(land_dir.path(), 16)),
            land.clone(),
        )
        .unwrap();
    let generator = Arc::new(
        CustomCubicGenerator::earth_with_land_cover(
            &settings,
            store.clone(),
            Some(source.clone()),
            default_registry(),
        )
        .unwrap(),
    );
    let earth = generator.earth_generator().unwrap();
    // Elevation held, land cover not: the node is later and says so by recording what it lacks.
    let level = 3;
    let waits = loop {
        match earth.node_columns(level, 0, 0) {
            Err(e) if !e.tiles.is_empty() => {
                for t in e.tiles {
                    store.load(t).unwrap();
                }
            }
            Err(e) => break e,
            Ok(_) => panic!("built without its class"),
        }
    };
    assert!(waits.tiles.is_empty());
    assert_eq!(land.fetch_count(), 0, "a read only records what it lacks");
    // The prefetcher takes the record and fetches it (the elevation rig's thread, now with land cover).
    let prefetcher = Prefetcher::start(&generator, Duration::from_millis(1)).unwrap();
    prefetcher.follow(around(1, 0));
    let mut built = false;
    for _ in 0..500 {
        match earth.node_columns(level, 0, 0) {
            Ok(_) => {
                built = true;
                break;
            }
            Err(e) => {
                for t in e.tiles {
                    store.load(t).unwrap();
                }
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(built, "the node was never built");
    assert!(land.fetch_count() >= 2, "an index and a tile at least");
    // The pass ends once the area around the player is held (it may fetch a piece or two more for it): the
    // retries that follow do not ask for the same pieces again and again.
    std::thread::sleep(Duration::from_millis(400));
    let count = land.fetch_count();
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(land.fetch_count(), count);
    prefetcher.stop();
}

/// The tiles a plan names are few whatever the far distance: a whole-world far view plans the same handful of
/// tiles per level a ring of nodes touches, down to level 22, and never more than the world has.
#[test]
fn a_whole_world_far_distance_plans_a_bounded_list_with_the_coarse_zooms_last_and_small() {
    for metres in [METRES_PER_BLOCK, 1.0] {
        let dir = TempDir::new();
        let store = Arc::new(TileStore::new(
            dir.path(),
            Arc::new(MemoryFetcher::new(terrain)),
            256,
        ));
        let mut settings = EarthSettings::at(LAT, LON);
        settings.metres_per_block = metres;
        let generator = CustomCubicGenerator::earth(&settings, store, default_registry()).unwrap();
        let earth = generator.earth_generator().unwrap();
        let base = earth.base_zoom();
        let started = Instant::now();
        let planned = plan(earth, &around(4, MAX_FAR_CHUNKS));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{metres} m: {:?}",
            started.elapsed()
        );
        assert!(
            !planned.is_empty() && planned.len() < 1500,
            "{metres} m: {} tiles",
            planned.len()
        );
        let unique: HashSet<_> = planned.iter().collect();
        assert_eq!(unique.len(), planned.len(), "{metres} m: a tile twice");
        // The base zoom first, then the zooms only go coarser: the far levels read the coarse ones, a few tiles
        // each.
        let mut zooms: Vec<u8> = planned.iter().map(|id| id.z).collect();
        zooms.dedup();
        assert!(
            zooms.windows(2).all(|w| w[0] > w[1]),
            "{metres} m: {zooms:?}"
        );
        assert_eq!(zooms[0], base, "{metres} m");
        let coarsest = *zooms.last().unwrap();
        assert!(coarsest <= COARSE_PREFETCH_ZOOM, "{metres} m: {zooms:?}");
        let coarse = planned.iter().filter(|id| id.z <= COARSE_PREFETCH_ZOOM);
        assert!(coarse.count() <= 341, "{metres} m: more than the world has");
        // A far distance past the cap plans what the cap does.
        let capped = plan(earth, &around(4, i32::MAX));
        assert_eq!(capped, planned, "{metres} m");
    }
}

/// A world with land cover (the far view's coarse path is on) whose fetcher records the order of its requests.
struct Recording {
    inner: MemoryFetcher,
    order: Mutex<Vec<TileId>>,
}

impl TileFetcher for Recording {
    fn fetch(&self, id: TileId) -> Result<Vec<u8>, FetchError> {
        lock(&self.order).push(id);
        self.inner.fetch(id)
    }
}

struct LandRig {
    fetcher: Arc<Recording>,
    generator: Arc<CustomCubicGenerator>,
    _dirs: (TempDir, TempDir),
}

fn land_rig() -> LandRig {
    use horizon_earth::testing::MemoryLandCover;
    use horizon_earth::{LandCoverStore, WorldCoverFile};

    let dir = TempDir::new();
    let land_dir = TempDir::new();
    let fetcher = Arc::new(Recording {
        inner: MemoryFetcher::new(terrain),
        order: Mutex::default(),
    });
    let store = Arc::new(TileStore::new(dir.path(), fetcher.clone(), 512));
    let mut settings = EarthSettings::at(LAT, LON);
    settings.metres_per_block = METRES_PER_BLOCK;
    settings.land_cover = true;
    let file = WorldCoverFile::containing(LAT, LON).unwrap();
    let land = Arc::new(MemoryLandCover::new().with_classes(file, 192, 96, 2, |_, _| 30));
    let source = settings
        .land_cover_source(Arc::new(LandCoverStore::new(land_dir.path(), 16)), land)
        .unwrap();
    let generator = Arc::new(
        CustomCubicGenerator::earth_with_land_cover(
            &settings,
            store,
            Some(source),
            default_registry(),
        )
        .unwrap(),
    );
    LandRig {
        fetcher,
        generator,
        _dirs: (dir, land_dir),
    }
}

#[test]
fn the_coarse_set_is_wanted_only_when_the_far_view_reaches_coarse_nodes() {
    let rig = land_rig();
    let earth = rig.generator.earth_generator().unwrap();
    // 40 m a block: voxels of 4 km from level 7 up, nodes of 2,048 blocks (128 chunks) and more.
    assert!(wants_coarse(earth, &around(4, 400)));
    assert!(!wants_coarse(earth, &around(4, 0)), "no far view");
    assert!(
        !wants_coarse(earth, &around(4, 20)),
        "a far view of fine nodes"
    );
    assert!(coarse_plan(earth, &around(4, 0)).is_empty());
    let planned = coarse_plan(earth, &around(4, 400));
    assert_eq!(planned.len(), 341, "the whole coarse set, zooms 0 to 4");
    assert!(planned.iter().all(|id| id.z <= COARSE_PREFETCH_ZOOM));
    assert_eq!(planned[0].z, 0, "the coarsest zoom first");
    // Without land cover the world has no coarse path at all.
    let plain = self::rig();
    assert!(!wants_coarse(plain.earth(), &around(4, 400)));
}

#[test]
fn the_prefetcher_fetches_the_coarse_set_first_and_builds_the_mask_on_its_thread() {
    let rig = land_rig();
    let earth = rig.generator.earth_generator().unwrap();
    let target = around(2, 400);
    let expected = earth.coarse_tiles_missing((8, 8));
    assert_eq!(expected.len(), 341);
    assert!(!earth.coarse_ready());
    let prefetcher = Prefetcher::start(&rig.generator, Duration::from_millis(1)).unwrap();
    prefetcher.follow(target);
    let mut ready = false;
    for _ in 0..1200 {
        if earth.coarse_ready() {
            ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(ready, "the mask was never built");
    // Before any other tile the thread fetched exactly the coarse set, in the order it names (zoom ascending,
    // the player's place first).
    let order = lock(&rig.fetcher.order).clone();
    assert!(order.len() >= expected.len());
    assert_eq!(order[..expected.len()], expected[..]);
    assert!(earth.coarse_tiles_missing((8, 8)).is_empty());
    prefetcher.stop();
}

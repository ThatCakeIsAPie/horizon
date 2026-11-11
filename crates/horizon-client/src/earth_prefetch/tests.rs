//! Tests of the prefetcher over synthetic tiles: nothing here touches the network.
use super::*;
use horizon_earth::projection::latlon_to_tile_pixel;
use horizon_earth::testing::{MemoryFetcher, TempDir};
use horizon_earth::{Projection, TileStore};
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
        .land_cover_source(Arc::new(LandCoverStore::new(land_dir.path(), 16)), land.clone())
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

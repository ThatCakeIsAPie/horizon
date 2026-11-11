//! Tests of the Earth generator with land cover, over synthetic elevation and synthetic WorldCover files served
//! from memory: no network, no real data.

use std::collections::BTreeSet;
use std::sync::Arc;

use horizon_core::position::SectionPos;
use horizon_earth::testing::{MemoryFetcher, MemoryLandCover, TempDir};
use horizon_earth::{
    LandClass, LandCoverFetcher, LandCoverSource, LandCoverStore, Lookup, TileStore, WorldCoverFile,
};

use super::surface::{LAKE_DEPTH, LAND_SNOW_FLOOR_M, MARSH_DEPTH, SOIL_DEPTH, SurfaceKind};
use super::tests::{LAT0, LON0, terrain};
use super::*;
use crate::cubic::custom::CustomCubicGenerator;
use crate::cubic::custom::biomes::BiomeColumns;
use crate::cubic::custom::testing::default_registry;

/// The codes of the dataset's classes, in the order of [`LandClass::ALL`] after no data.
const CODES: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 0];

/// A mosaic of every class in cells about 870 m wide (one pixel of the miniature files).
fn mosaic(lat: f64, lon: f64) -> u8 {
    let (a, b) = ((lat * 128.0).floor() as i64, (lon * 128.0).floor() as i64);
    CODES[(a * 7 + b * 13).rem_euclid(12) as usize]
}

/// An Earth generator with land cover over synthetic elevation and a miniature WorldCover, and the same world
/// without land cover over the same tiles.
struct LandRig {
    generator: EarthGenerator,
    plain: EarthGenerator,
    store: Arc<TileStore>,
    land: Arc<MemoryLandCover>,
    source: Arc<LandCoverSource>,
    blocks: EarthBlocks,
    settings: EarthSettings,
    _dirs: (TempDir, TempDir),
}

/// The files a test near `(lat, lon)` can touch (about 0.7 degrees around it).
fn files_near(lat: f64, lon: f64) -> BTreeSet<WorldCoverFile> {
    [-0.7, 0.7]
        .into_iter()
        .flat_map(|dlat| [-0.7, 0.7].into_iter().map(move |dlon| (dlat, dlon)))
        .filter_map(|(dlat, dlon)| WorldCoverFile::containing(lat + dlat, lon + dlon))
        .collect()
}

impl LandRig {
    /// Land cover `class(lat, lon)` (a code) over `elevation(lat, lon)` metres, with `settings` (land cover is
    /// turned on).
    fn new(
        mut settings: EarthSettings,
        elevation: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
        class: impl Fn(f64, f64) -> u8 + Clone,
    ) -> Self {
        let mut land = MemoryLandCover::new();
        for file in files_near(settings.origin_latitude, settings.origin_longitude) {
            land = land.with_classes(file, 384, 96, 2, class.clone());
        }
        Self::over(&mut settings, elevation, Arc::new(land))
    }

    /// The rig over a land-cover fetcher of the caller's.
    fn over(
        settings: &mut EarthSettings,
        elevation: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
        land: Arc<MemoryLandCover>,
    ) -> Self {
        settings.land_cover = true;
        let elevation_dir = TempDir::new();
        let land_dir = TempDir::new();
        let store = Arc::new(TileStore::new(
            elevation_dir.path(),
            Arc::new(MemoryFetcher::new(elevation)),
            4096,
        ));
        let blocks = EarthBlocks::from_registry(default_registry()).unwrap();
        let fetcher: Arc<dyn LandCoverFetcher> = land.clone();
        let source = settings
            .land_cover_source(Arc::new(LandCoverStore::new(land_dir.path(), 64)), fetcher)
            .unwrap();
        let generator = EarthGenerator::new(settings, store.clone(), blocks)
            .unwrap()
            .with_land_cover(source.clone());
        let mut off = settings.clone();
        off.land_cover = false;
        let plain = EarthGenerator::new(&off, store.clone(), blocks).unwrap();
        Self {
            generator,
            plain,
            store,
            land,
            source,
            blocks,
            settings: settings.clone(),
            _dirs: (elevation_dir, land_dir),
        }
    }

    /// A rig of one class over level ground of `metres` at `(lat0, LON0)`.
    fn level(lat0: f64, metres: f64, class: u8) -> Self {
        Self::new(
            EarthSettings::at(lat0, LON0),
            move |_, _| metres,
            move |_, _| class,
        )
    }

    /// The blocks of the column at `(x, z)` from `s - 8` to `s + 3`, and its surface `s`.
    fn column(&self, x: i32, z: i32) -> (i32, Vec<u32>) {
        let s = self.generator.surface_y_blocking(x, z).unwrap();
        (
            s,
            (s - 8..=s + 3)
                .map(|y| self.generator.block_at_blocking(x, y, z, 1))
                .collect(),
        )
    }

    /// Checks the column at `(x, z)` against `want(y, s)`; its surface.
    fn check(&self, x: i32, z: i32, want: impl Fn(i32, i32) -> u32) -> i32 {
        let (s, got) = self.column(x, z);
        for (i, &block) in got.iter().enumerate() {
            let y = s - 8 + i as i32;
            assert_eq!(block, want(y, s), "column {x},{z} (surface {s}) at y {y}");
        }
        s
    }

    /// Loads every elevation tile and land-cover piece a local read of `read` lacks, until it answers.
    fn loaded<T>(&self, read: impl Fn(&EarthGenerator) -> Result<T, NotReady>) -> T {
        for _ in 0..200 {
            match read(&self.generator) {
                Ok(value) => return value,
                Err(missing) => {
                    for &id in &missing.tiles {
                        self.store.load(id).unwrap();
                    }
                    for fetch in self.generator.take_land_requests() {
                        self.source.satisfy(&fetch).unwrap();
                    }
                }
            }
        }
        panic!("a read never became ready");
    }
}

fn cube(x: i32, y: i32, z: i32) -> SectionPos {
    SectionPos::new(x, y, z)
}

/// The cubes of a test area: the coast, the slope up to the spire, the open sea, and the heights around each.
fn area(rig: &LandRig) -> Vec<SectionPos> {
    let mut out = Vec::new();
    for cx in [-40, -20, -3, -1, 0, 1, 2, 50, 100, 140, 150, 156] {
        for cz in [-1, 0, 1] {
            let s = rig
                .plain
                .surface_y_blocking(cx * 16 + 8, cz * 16 + 8)
                .unwrap();
            for cy in (s >> 4) - 2..=(s >> 4) + 2 {
                out.push(cube(cx, cy, cz));
            }
            out.push(cube(cx, -3, cz));
        }
    }
    out
}

fn blocks_of(g: &EarthGenerator, c: SectionPos) -> (Box<[u32; 4096]>, Option<u32>) {
    let mut out = Box::new([0; 4096]);
    let uniform = g.generate_blocking(c, &mut out);
    (out, uniform)
}

fn terrain_rig(class: impl Fn(f64, f64) -> u8 + Clone) -> LandRig {
    LandRig::new(EarthSettings::at(LAT0, LON0), terrain, class)
}

// --- Off means off ---------------------------------------------------------------------------------------------

#[test]
fn with_land_cover_off_or_no_source_the_terrain_is_bit_for_bit_the_elevation_only_terrain() {
    // A map that would change every column (all water) is handed in, but the setting is off.
    let mut settings = EarthSettings::at(LAT0, LON0);
    let mut land = MemoryLandCover::new();
    for file in files_near(LAT0, LON0) {
        land = land.with_classes(file, 384, 96, 2, |_, _| 80);
    }
    let land = Arc::new(land);
    let rig = LandRig::over(&mut settings, terrain, land.clone());
    let mut off = rig.settings.clone();
    off.land_cover = false;
    let with_source_but_off = EarthGenerator::new(&off, rig.store.clone(), rig.blocks)
        .unwrap()
        .with_land_cover(rig.source.clone());
    assert!(!with_source_but_off.land_cover_active());
    // On, but no source was handed in.
    let on_without_source =
        EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks).unwrap();
    assert!(rig.settings.land_cover && !on_without_source.land_cover_active());

    let mut checked = 0;
    for c in area(&rig) {
        let (want, want_uniform) = blocks_of(&rig.plain, c);
        for g in [&with_source_but_off, &on_without_source] {
            let (got, uniform) = blocks_of(g, c);
            assert_eq!(&got[..], &want[..], "cube {c:?}");
            assert_eq!(uniform, want_uniform, "cube {c:?}");
        }
        checked += 1;
    }
    assert!(checked > 100);
    // Point queries and far nodes at every step.
    for step in [1, 2, 8, 64, 512] {
        for (x, z) in [(-300, 40), (5, 5), (700, -90), (2400, 10)] {
            let y = rig.plain.surface_y_blocking(x, z).unwrap();
            for dy in [-6, -1, 0, 1, 40] {
                let want = rig.plain.block_at_blocking(x, y + dy, z, step);
                assert_eq!(
                    with_source_but_off.block_at_blocking(x, y + dy, z, step),
                    want
                );
            }
        }
    }
    for level in [2u32, 4, 6] {
        let want = rig.loaded_plain(|g| g.node_columns(level, 0, -1));
        let got = rig.loaded_plain_on(&with_source_but_off, |g| g.node_columns(level, 0, -1));
        assert_eq!(want.surfaces, got.surfaces);
        for i in 0..256 {
            assert_eq!(want.kind(i), got.kind(i));
        }
    }
    assert_eq!(land.fetch_count(), 0, "the map was never touched");
    assert!(with_source_but_off.take_land_requests().is_empty());
    assert!(on_without_source.take_land_requests().is_empty());
    // The world kind built through the cubic generator behaves the same.
    let custom = CustomCubicGenerator::earth_with_land_cover(
        &off,
        rig.store.clone(),
        Some(rig.source.clone()),
        default_registry(),
    )
    .unwrap();
    assert!(custom.earth_biomes().is_none());
}

impl LandRig {
    fn loaded_plain<T>(&self, read: impl Fn(&EarthGenerator) -> Result<T, NotReady>) -> T {
        self.loaded_plain_on(&self.plain, read)
    }

    fn loaded_plain_on<T>(
        &self,
        g: &EarthGenerator,
        read: impl Fn(&EarthGenerator) -> Result<T, NotReady>,
    ) -> T {
        for _ in 0..50 {
            match read(g) {
                Ok(v) => return v,
                Err(missing) => {
                    for &id in &missing.tiles {
                        self.store.load(id).unwrap();
                    }
                }
            }
        }
        panic!("never ready");
    }
}

// --- The rule, class by class ----------------------------------------------------------------------------------

#[test]
fn tree_shrub_grass_and_crop_are_grass_over_three_dirt() {
    for code in [10, 20, 30, 40] {
        let rig = LandRig::level(45.0, 120.0, code);
        let b = rig.blocks;
        rig.check(3, 4, |y, s| match y.cmp(&s) {
            _ if y < s - SOIL_DEPTH as i32 => b.stone,
            std::cmp::Ordering::Less => b.dirt,
            std::cmp::Ordering::Equal => b.grass_block,
            std::cmp::Ordering::Greater => b.air,
        });
    }
}

#[test]
fn bare_flat_ground_is_sand_and_gravel_from_the_cold_latitudes() {
    let rig = LandRig::level(45.0, 120.0, 60);
    let b = rig.blocks;
    rig.check(0, 0, |y, s| {
        if y < s - 3 {
            b.stone
        } else if y <= s {
            b.sand
        } else {
            b.air
        }
    });
    let cold = LandRig::level(55.0, 120.0, 60);
    cold.check(0, 0, |y, s| match y.cmp(&s) {
        std::cmp::Ordering::Less => b.stone,
        std::cmp::Ordering::Equal => b.gravel,
        std::cmp::Ordering::Greater => b.air,
    });
}

#[test]
fn bare_steep_ground_and_bare_ground_in_the_alpine_band_is_stone() {
    let b = LandRig::level(45.0, 0.0, 60).blocks;
    // A slope of one block per block to the north, with a bare map.
    let steep = LandRig::new(
        EarthSettings::at(45.0, LON0),
        |lat, _| 500.0 + (lat - 45.0) * 111_194.9,
        |_, _| 60,
    );
    steep.check(0, 0, |y, s| if y <= s { b.stone } else { b.air });
    // Level ground at 3,000 m at 45 degrees is inside the band under the 3,850 m snow line.
    let band = LandRig::level(45.0, 3000.0, 60);
    band.check(0, 0, |y, s| if y <= s { b.stone } else { b.air });
    // A gentle slope (0.2 block per block) stays sand.
    let gentle = LandRig::new(
        EarthSettings::at(45.0, LON0),
        |lat, _| 300.0 + (lat - 45.0) * 111_194.9 * 0.2,
        |_, _| 60,
    );
    gentle.check(0, 0, |y, s| {
        if y < s - 3 {
            b.stone
        } else if y <= s {
            b.sand
        } else {
            b.air
        }
    });
}

#[test]
fn snow_and_ice_is_snow_whatever_the_latitude_and_height() {
    for (lat, metres) in [(45.0, 120.0), (5.0, 400.0), (60.0, 50.0)] {
        let rig = LandRig::level(lat, metres, 70);
        let b = rig.blocks;
        rig.check(0, 0, |y, s| {
            if y < s - 1 {
                b.stone
            } else if y <= s {
                b.snow_block
            } else {
                b.air
            }
        });
    }
}

#[test]
fn the_snow_line_still_whitens_peaks_the_map_calls_something_else() {
    // 4,000 m at 45 degrees (snow line 3,850 m): the map says grass, bare or no data, the ground is snow.
    for code in [30, 60, 10, 0] {
        let rig = LandRig::level(45.0, 4000.0, code);
        let b = rig.blocks;
        rig.check(0, 0, |y, s| {
            if y < s - 1 {
                b.stone
            } else if y <= s {
                b.snow_block
            } else {
                b.air
            }
        });
    }
    // The rule itself, on the places that matter.
    let steep = || Ok::<_, ()>(false);
    let on =
        |class, metres, lat| SurfaceKind::on_classified(class, metres, lat, false, steep).unwrap();
    for class in LandClass::ALL {
        if class == LandClass::NoData {
            continue;
        }
        // Everest: white whatever the map calls the summit.
        assert_eq!(on(class, 8849.0, 27.99), SurfaceKind::Snow, "{class:?}");
    }
    // Svalbard: 600 m at 78 degrees is above the latitude snow line (220 m) but below the floor, so the map
    // decides; without a map the old rule does.
    assert_eq!(on(LandClass::Grass, 600.0, 78.0), SurfaceKind::Grass);
    assert_eq!(on(LandClass::Bare, 600.0, 78.0), SurfaceKind::Alpine);
    assert_eq!(on(LandClass::NoData, 600.0, 78.0), SurfaceKind::Snow);
    assert_eq!(
        on(LandClass::Grass, LAND_SNOW_FLOOR_M, 78.0),
        SurfaceKind::Snow
    );
    // A frozen lake high up.
    assert_eq!(on(LandClass::Water, 4000.0, 45.0), SurfaceKind::Snow);
}

#[test]
fn a_lake_above_the_sea_is_water_to_its_surface_over_a_sand_bed() {
    let rig = LandRig::level(45.0, 120.0, 80);
    let b = rig.blocks;
    let bed = LAKE_DEPTH as i32;
    let s = rig.check(0, 0, |y, s| {
        if y < s - bed - 3 {
            b.stone
        } else if y <= s - bed {
            b.sand
        } else if y <= s {
            b.water
        } else {
            b.air
        }
    });
    assert!(s > 0);
}

#[test]
fn water_under_the_sea_level_is_the_sea_rule_and_wetland_is_one_block_of_water_over_mud() {
    let sea = LandRig::level(45.0, -30.0, 80);
    let b = sea.blocks;
    let s = sea.check(0, 0, |y, s| {
        if y < s - 3 {
            b.stone
        } else if y <= s {
            b.sand
        } else if y < 0 {
            b.water
        } else {
            b.air
        }
    });
    assert!(s < 0);
    let marsh = MARSH_DEPTH as i32;
    for (code, elevation) in [(90, 120.0), (95, 120.0), (90, -2.0)] {
        let rig = LandRig::level(45.0, elevation, code);
        let b = rig.blocks;
        rig.check(0, 0, |y, s| {
            if y < s - marsh - 3 {
                b.stone
            } else if y <= s - marsh {
                b.mud
            } else if y <= s {
                b.water
            } else {
                b.air
            }
        });
    }
}

#[test]
fn moss_is_moss_over_dirt_and_built_up_ground_is_gravel_over_stone() {
    let moss = LandRig::level(45.0, 120.0, 100);
    let b = moss.blocks;
    moss.check(0, 0, |y, s| match y.cmp(&s) {
        _ if y < s - 3 => b.stone,
        std::cmp::Ordering::Less => b.dirt,
        std::cmp::Ordering::Equal => b.moss_block,
        std::cmp::Ordering::Greater => b.air,
    });
    let built = LandRig::level(45.0, 120.0, 50);
    built.check(0, 0, |y, s| match y.cmp(&s) {
        std::cmp::Ordering::Less => b.stone,
        std::cmp::Ordering::Equal => b.gravel,
        std::cmp::Ordering::Greater => b.air,
    });
}

#[test]
fn land_under_the_sea_level_stays_dry_where_the_map_says_it_is_not_water() {
    // 86 m below the sea: Death Valley, the Dead Sea shore, a polder.
    for (code, top) in [(60, "sand"), (30, "grass"), (50, "gravel"), (70, "snow")] {
        let rig = LandRig::level(45.0, -86.0, code);
        let b = rig.blocks;
        let top = match top {
            "sand" => b.sand,
            "grass" => b.grass_block,
            "gravel" => b.gravel,
            _ => b.snow_block,
        };
        let at = |y: i32| rig.generator.block_at_blocking(0, y, 0, 1);
        let s = rig.generator.surface_y_blocking(0, 0).unwrap();
        assert!(s < 0);
        assert_eq!(at(s), top, "class {code}");
        // Air all the way up, under the sea level too, and no water in the ground.
        for y in s + 1..=s + 100 {
            assert_eq!(at(y), b.air, "class {code} at {y}");
        }
        for y in s - 20..s {
            assert_ne!(at(y), b.water, "class {code} at {y}");
        }
    }
    // Water or no data below the sea is flooded as before: sand and water to the sea level.
    for code in [80, 0] {
        let rig = LandRig::level(45.0, -86.0, code);
        let s = rig.generator.surface_y_blocking(0, 0).unwrap();
        assert_eq!(rig.generator.block_at_blocking(0, s, 0, 1), rig.blocks.sand);
        assert_eq!(
            rig.generator.block_at_blocking(0, s + 1, 0, 1),
            rig.blocks.water
        );
        assert_eq!(
            rig.generator.block_at_blocking(0, -1, 0, 1),
            rig.blocks.water
        );
        assert_eq!(rig.generator.block_at_blocking(0, 0, 0, 1), rig.blocks.air);
    }
}

#[test]
fn a_column_without_data_follows_the_elevation_rule() {
    let rig = LandRig::level(45.0, 120.0, 0);
    for (x, z) in [(0, 0), (30, -20), (-70, 90)] {
        let s = rig.generator.surface_y_blocking(x, z).unwrap();
        for y in s - 6..=s + 3 {
            assert_eq!(
                rig.generator.block_at_blocking(x, y, z, 1),
                rig.plain.block_at_blocking(x, y, z, 1)
            );
        }
    }
}

// --- Missing files, files not yet fetched ----------------------------------------------------------------------

#[test]
fn a_file_the_source_does_not_have_is_no_data_asked_for_once_and_never_waited_for() {
    // Open ocean: the source has no file anywhere here.
    let mut settings = EarthSettings::at(LAT0, LON0);
    let land = Arc::new(MemoryLandCover::new());
    let rig = LandRig::over(&mut settings, terrain, land.clone());
    let start = std::time::Instant::now();
    for c in area(&rig).into_iter().take(60) {
        let (got, uniform) = blocks_of(&rig.generator, c);
        let (want, want_uniform) = blocks_of(&rig.plain, c);
        assert_eq!(&got[..], &want[..], "cube {c:?}");
        assert_eq!(uniform, want_uniform);
    }
    assert!(start.elapsed() < std::time::Duration::from_secs(20));
    for file in files_near(LAT0, LON0) {
        assert!(land.fetches_of(file) <= 1, "{file} was asked for again");
    }
    // The plain forms answer now, with nothing to wait for.
    let c = cube(0, 0, 0);
    blocks_of(&rig.generator, c);
    let asked = land.fetch_count();
    let mut out = Box::new([0; 4096]);
    assert!(rig.generator.generate(c, &mut out).is_ok());
    assert_eq!(land.fetch_count(), asked);
    assert_eq!(
        rig.source.field().class_at_block(0, 0),
        Lookup::NoFile,
        "the file is known to be absent"
    );
}

#[test]
fn a_class_not_fetched_yet_is_not_ready_and_is_never_waited_for() {
    let rig = terrain_rig(mosaic);
    let g = &rig.generator;
    // Elevation first, so that only the land cover is missing.
    let c = cube(1, 0, 0);
    for _ in 0..10 {
        let mut out = Box::new([0; 4096]);
        let Err(missing) = g.generate(c, &mut out) else {
            panic!("the cube is ready with nothing fetched");
        };
        if missing.tiles.is_empty() {
            break;
        }
        for &id in &missing.tiles {
            rig.store.load(id).unwrap();
        }
    }
    let mut out = Box::new([0; 4096]);
    let err = g.generate(c, &mut out).unwrap_err();
    assert!(err.tiles.is_empty(), "only land cover is missing: {err}");
    assert_eq!(err.to_string(), "land cover is not ready");
    assert_eq!(rig.land.fetch_count(), 0, "a local read fetches nothing");
    // The far view's reads say the same and record what to fetch, and the first thing is an index.
    let columns = g.node_columns(4, 0, -1);
    assert!(columns.is_err());
    let requests = g.take_land_requests();
    assert!(!requests.is_empty());
    assert!(
        matches!(requests[0].wanted, horizon_earth::Wanted::Index),
        "{requests:?}"
    );
    assert!(g.take_land_requests().is_empty());
    assert_eq!(rig.land.fetch_count(), 0);
    // Handing the fetches over, round after round, makes the reads answer, with what the blocking form makes.
    let ready = rig.loaded(|g| {
        let mut out = Box::new([0; 4096]);
        g.generate(c, &mut out).map(|u| (out, u))
    });
    let (want, want_uniform) = blocks_of(&rig.generator, c);
    assert_eq!(&ready.0[..], &want[..]);
    assert_eq!(ready.1, want_uniform);
    assert!(rig.loaded(|g| g.node_columns(4, 0, -1)).surfaces.len() == 256);
}

#[test]
fn land_cover_requests_wake_the_taker_once_each() {
    let rig = terrain_rig(mosaic);
    let woken = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let w = woken.clone();
    rig.generator.set_request_waker(Arc::new(move || {
        w.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }));
    let _ = rig.generator.biomes_ready(0, 0, 1);
    let first = woken.load(std::sync::atomic::Ordering::SeqCst);
    assert!(first >= 1);
    let _ = rig.generator.biomes_ready(0, 0, 1);
    assert_eq!(woken.load(std::sync::atomic::Ordering::SeqCst), first);
}

#[test]
fn a_failing_land_fetcher_is_retried_by_the_blocking_form_until_it_recovers() {
    let rig = terrain_rig(mosaic);
    rig.land
        .fail_with(Some(horizon_earth::FetchError::Temporary("offline".into())));
    let flag = rig.generator.cancel_flag();
    let g = &rig.generator;
    std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            let mut out = Box::new([0; 4096]);
            g.generate_blocking(cube(1, 0, 0), &mut out);
            out
        });
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert!(!handle.is_finished(), "it waits while the fetcher fails");
        rig.land.fail_with(None);
        let out = handle.join().unwrap();
        assert!(!g.is_cancelled());
        let want = rig.loaded(|g| {
            let mut o = Box::new([0; 4096]);
            g.generate(cube(1, 0, 0), &mut o).map(|_| o)
        });
        assert_eq!(&out[..], &want[..]);
    });
    // And a cancelled wait returns at once.
    rig.land
        .fail_with(Some(horizon_earth::FetchError::Temporary("offline".into())));
    flag.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut out = Box::new([0; 4096]);
    let far = cube(300, 0, 300);
    let start = std::time::Instant::now();
    rig.generator.generate_blocking(far, &mut out);
    assert!(start.elapsed() < std::time::Duration::from_secs(5));
}

// --- Near and far agree ----------------------------------------------------------------------------------------

#[test]
fn near_and_far_give_the_same_surface_material_at_every_step() {
    let rig = terrain_rig(mosaic);
    let g = &rig.generator;
    let mut kinds = BTreeSet::new();
    let mut compared = 0;
    for stride in [1, 2, 4, 8, 16, 64, 256, 1024] {
        let side = 16 * stride;
        let align = |v: i32| v.div_euclid(side) * side;
        for x0 in [align(-1200), align(0), align(1800), align(2400)] {
            for z0 in [align(-300), align(100)] {
                let cols = rig.loaded(|g| g.columns_at(x0, z0, stride, stride));
                let (sx, sz) = (x0 + stride / 2, z0 + stride / 2);
                for i in (0..256).step_by(if stride < 8 { 5 } else { 1 }) {
                    let (wx, wz) = (sx + (i & 15) as i32 * stride, sz + (i >> 4) as i32 * stride);
                    kinds.insert(format!("{:?}", cols.kind(i)));
                    let s = cols.surfaces[i];
                    let floor = s.div_euclid(stride) * stride;
                    for lo in [floor - stride, floor, floor + stride] {
                        assert_eq!(
                            g.far_voxel(&cols, i, lo),
                            g.block_at_blocking(wx, lo + stride / 2, wz, stride),
                            "stride {stride}, column {wx},{wz}, voxel {lo}"
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(compared > 2000);
    // The mosaic put every kind of top there to compare.
    for kind in [
        SurfaceKind::Grass,
        SurfaceKind::Snow,
        SurfaceKind::Lake,
        SurfaceKind::Marsh,
        SurfaceKind::Moss,
        SurfaceKind::Gravel,
        SurfaceKind::Sand,
    ] {
        assert!(
            kinds.contains(&format!("{kind:?}")),
            "no {kind:?} in {kinds:?}"
        );
    }
}

#[test]
fn a_cube_and_the_level_zero_columns_agree_and_a_coarse_level_reads_a_coarse_class() {
    let rig = terrain_rig(mosaic);
    let g = &rig.generator;
    for (cx, cz) in [(-2, 0), (3, -1), (100, 0), (150, 1)] {
        let cols = rig.loaded(|g| g.columns_at(cx * 16, cz * 16, 1, 1));
        let (blocks, _) = blocks_of(g, cube(cx, (cols.surfaces[0] >> 4).max(-30), cz));
        for i in 0..256 {
            let s = cols.surfaces[i];
            let y0 = ((cols.surfaces[0] >> 4).max(-30)) * 16;
            for dy in 0..16 {
                let y = y0 + dy;
                if (y - s).abs() < 8 {
                    assert_eq!(
                        blocks[(dy as usize) << 8 | i],
                        g.far_voxel(&cols, i, y),
                        "{cx},{cz} column {i} y {y}"
                    );
                }
            }
        }
    }
    // The class read at a coarse step is the coarse overview's, which is not the fine one's everywhere in the
    // mosaic.
    let field = rig.source.field();
    let class = |x: i32, metres: f64| {
        rig.source
            .class_at_blocking(f64::from(x) + 0.5, 40.5, metres)
            .unwrap();
        field.class_at(f64::from(x) + 0.5, 40.5, metres)
    };
    let differ = (-3000..3000)
        .step_by(97)
        .filter(|&x| class(x, 1.0) != class(x, 4000.0))
        .count();
    assert!(differ > 0);
}

#[test]
fn a_lake_shows_in_the_far_view_and_dry_land_under_the_sea_level_is_not_flooded_there() {
    // A lake at 120 m and, in the other file of the same area, dry grass at -50 m.
    let lake = LandRig::level(45.0, 120.0, 80);
    let cols = lake.loaded(|g| g.node_columns(5, 0, 0));
    for i in 0..256 {
        let s = cols.surfaces[i];
        assert_eq!(cols.kind(i), SurfaceKind::Lake);
        let lo = s.div_euclid(32) * 32;
        assert_eq!(lake.generator.far_voxel(&cols, i, lo), lake.blocks.water);
        assert_eq!(lake.generator.far_voxel(&cols, i, lo + 32), lake.blocks.air);
    }
    let dry = LandRig::level(45.0, -50.0, 30);
    let cols = dry.loaded(|g| g.node_columns(5, 0, 0));
    for i in 0..256 {
        let s = cols.surfaces[i];
        let lo = s.div_euclid(32) * 32;
        assert_eq!(
            dry.generator.far_voxel(&cols, i, lo),
            dry.blocks.grass_block
        );
        assert_eq!(dry.generator.far_voxel(&cols, i, lo + 32), dry.blocks.air);
    }
}

#[test]
fn a_cube_above_a_lake_or_a_dry_basin_is_uniform_air_and_the_sea_is_still_water() {
    let dry = LandRig::level(45.0, -50.0, 30);
    let (blocks, uniform) = blocks_of(&dry.generator, cube(0, -2, 0));
    assert_eq!(
        uniform,
        Some(dry.blocks.air),
        "above the grass, under the sea level: air, not water"
    );
    assert!(blocks.iter().all(|&b| b == dry.blocks.air));
    let sea = LandRig::level(45.0, -50.0, 80);
    let (_, uniform) = blocks_of(&sea.generator, cube(0, -2, 0));
    assert_eq!(uniform, Some(sea.blocks.water));
    let lake = LandRig::level(45.0, 120.0, 80);
    let (_, uniform) = blocks_of(&lake.generator, cube(0, 20, 0));
    assert_eq!(uniform, Some(lake.blocks.air));
    // Deep rock under a lake's bed is still proven stone.
    let (_, uniform) = blocks_of(&lake.generator, cube(0, 0, 0));
    assert_eq!(uniform, Some(lake.blocks.stone));
}

// --- Threads ---------------------------------------------------------------------------------------------------

#[test]
fn generation_with_land_cover_is_the_same_on_every_thread_in_every_order() {
    let rig = terrain_rig(mosaic);
    let cubes: Vec<SectionPos> = area(&rig).into_iter().step_by(3).take(60).collect();
    let want: Vec<_> = cubes
        .iter()
        .map(|&c| blocks_of(&rig.generator, c))
        .collect();
    // A fresh world over the same data, hit from four threads at once, each in its own order.
    let fresh = terrain_rig(mosaic);
    let g = &fresh.generator;
    let results: Vec<Vec<_>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let cubes = &cubes;
                scope.spawn(move || {
                    let mut order: Vec<usize> = (0..cubes.len()).collect();
                    if t % 2 == 1 {
                        order.reverse();
                    }
                    order.rotate_left(t * 7 % cubes.len());
                    let mut out = vec![None; cubes.len()];
                    for i in order {
                        out[i] = Some(blocks_of(g, cubes[i]));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for got in results {
        for (i, got) in got.into_iter().enumerate() {
            let got = got.unwrap();
            assert_eq!(&got.0[..], &want[i].0[..], "cube {:?}", cubes[i]);
            assert_eq!(got.1, want[i].1);
        }
    }
}

// --- The cache key ---------------------------------------------------------------------------------------------

#[test]
fn the_terrain_key_of_a_land_cover_world_never_matches_one_without() {
    let mut off = EarthSettings::at(LAT0, LON0);
    let mut on = off.clone();
    on.land_cover = true;
    assert_ne!(off.terrain_key(), on.terrain_key());
    assert!(on.terrain_key().contains(&format!(
        "landcover{}.{}",
        horizon_earth::LAND_COVER_RULES,
        super::surface::LAND_RULES
    )));
    // Everything else equal, the key follows the switch both ways.
    off.land_cover = true;
    assert_eq!(off.terrain_key(), on.terrain_key());
}

// --- Biomes ----------------------------------------------------------------------------------------------------

#[test]
fn the_biome_table_says_what_it_says() {
    use super::biome::Biome::*;
    let of = Biome::of;
    assert_eq!(of(LandClass::Water, -10.0, 40.0), Ocean);
    assert_eq!(of(LandClass::Water, 300.0, 40.0), River);
    assert_eq!(of(LandClass::NoData, -10.0, 40.0), Ocean);
    assert_eq!(of(LandClass::NoData, 300.0, 40.0), Plains);
    assert_eq!(of(LandClass::Tree, 100.0, 3.0), Jungle);
    assert_eq!(of(LandClass::Tree, 100.0, -45.0), Forest);
    assert_eq!(of(LandClass::Tree, 100.0, 60.0), Taiga);
    assert_eq!(of(LandClass::Tree, 100.0, 68.0), SnowyTaiga);
    assert_eq!(of(LandClass::Shrub, 100.0, 20.0), Savanna);
    assert_eq!(of(LandClass::Grass, 2000.0, 46.0), Meadow);
    assert_eq!(of(LandClass::Grass, 100.0, 46.0), Plains);
    assert_eq!(of(LandClass::Crop, 100.0, 46.0), Plains);
    assert_eq!(of(LandClass::Bare, 400.0, 25.0), Desert);
    assert_eq!(of(LandClass::Bare, 2500.0, 25.0), StonyPeaks);
    assert_eq!(of(LandClass::Wetland, 5.0, 25.0), Swamp);
    assert_eq!(of(LandClass::Mangrove, 0.5, 5.0), MangroveSwamp);
    assert_eq!(of(LandClass::SnowIce, 100.0, 70.0), SnowyPlains);
    assert_eq!(of(LandClass::SnowIce, 2000.0, 46.0), SnowySlopes);
    assert_eq!(of(LandClass::Grass, 8849.0, 28.0), FrozenPeaks);
    // Every name is a vanilla biome, once, and plains is the first.
    assert_eq!(Biome::ALL[0], Plains);
    let names: BTreeSet<_> = Biome::ALL.iter().map(|b| b.name()).collect();
    assert_eq!(names.len(), Biome::ALL.len());
    assert!(names.iter().all(|n| n.starts_with("minecraft:")));
}

#[test]
fn a_column_has_one_biome_whatever_asks_and_it_follows_the_map() {
    let rig = terrain_rig(mosaic);
    let custom = CustomCubicGenerator::earth_with_land_cover(
        &rig.settings,
        rig.store.clone(),
        Some(rig.source.clone()),
        default_registry(),
    )
    .unwrap();
    let biomes = custom.earth_biomes().expect("land cover is on");
    let again = terrain_rig(mosaic);
    let other = EarthBiomes::new(Arc::new(again.generator));
    let mut seen = BTreeSet::new();
    for qx in (-4000..4000).step_by(401) {
        for qz in (-2000..2000).step_by(397) {
            let id = biomes.biome(qx, qz);
            // The same from another world over the same data, and from the quart's other blocks' view.
            assert_eq!(other.biome(qx, qz), id, "quart {qx},{qz}");
            assert_eq!(biomes.biome(qx, qz), id);
            seen.insert(biomes.biome_name(id).to_string());
        }
    }
    assert!(seen.len() >= 5, "{seen:?}");
    assert_eq!(biomes.possible_biomes().len(), Biome::ALL.len());
    // Open water below the sea is an ocean wherever the water is.
    let sea = LandRig::level(45.0, -50.0, 80);
    let b = EarthBiomes::new(Arc::new(sea.generator));
    assert_eq!(b.biome_name(b.biome(3, 3)), "minecraft:ocean");
}

#[test]
fn the_biomes_of_a_far_node_are_ready_before_they_are_asked_for_and_never_load_then() {
    let rig = terrain_rig(mosaic);
    let g = &rig.generator;
    assert!(g.biomes_ready(0, 0, 1).is_err());
    assert_eq!(rig.land.fetch_count(), 0);
    rig.loaded(|g| g.biomes_ready(0, 0, 1));
    rig.loaded(|g| g.biomes_ready(512, 512, 64));
    let asked = rig.land.fetch_count();
    let biomes = EarthBiomes::new(Arc::new(
        EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks)
            .unwrap()
            .with_land_cover(rig.source.clone()),
    ));
    for i in 0..16 {
        biomes.biome(i, i);
    }
    assert_eq!(rig.land.fetch_count(), asked, "nothing was left to fetch");
    // Without land cover there is nothing to wait for.
    assert!(rig.plain.biomes_ready(0, 0, 1).is_ok());
}

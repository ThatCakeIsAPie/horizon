//! Tests of the Earth generator over synthetic terrain: no tiles from the network, no game data.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use horizon_core::position::SectionPos;
use horizon_earth::projection::{EARTH_RADIUS_M, latlon_to_tile_pixel};
use horizon_earth::testing::{MemoryFetcher, TempDir};
use horizon_earth::{FetchError, TileStore};

use super::*;
use crate::cubic::custom::testing::default_registry;
use crate::cubic::custom::{BoxClass, CustomCubicGenerator};

const LAT0: f64 = 45.0;
const LON0: f64 = 7.0;

/// A coast along `east = 0` (sea to the west, a long slope up to the east), ripples, and a 7 km spire 2.5 km
/// east of the origin whose flanks are far steeper than one block per block.
fn terrain(lat: f64, lon: f64) -> f64 {
    let east = (lon - LON0).to_radians() * EARTH_RADIUS_M * LAT0.to_radians().cos();
    let north = (lat - LAT0).to_radians() * EARTH_RADIUS_M;
    let spire =
        7000.0 * (-(((east - 2500.0).powi(2) + north.powi(2)) / (2.0 * 500.0 * 500.0))).exp();
    0.4 * east + spire + 12.0 * (east / 37.0).sin() * (north / 53.0).cos()
}

struct Rig {
    fetcher: Arc<MemoryFetcher>,
    store: Arc<TileStore>,
    generator: EarthGenerator,
    settings: EarthSettings,
    blocks: EarthBlocks,
    _dir: TempDir,
}

fn rig_with(settings: EarthSettings) -> Rig {
    let dir = TempDir::new();
    let fetcher = Arc::new(MemoryFetcher::new(terrain));
    let store = Arc::new(TileStore::new(dir.path(), fetcher.clone(), 256));
    let blocks = EarthBlocks::from_registry(default_registry()).unwrap();
    let generator = EarthGenerator::new(&settings, store.clone(), blocks)
        .unwrap()
        .with_retry(Duration::from_millis(5), Duration::from_millis(20));
    Rig {
        fetcher,
        store,
        generator,
        settings,
        blocks,
        _dir: dir,
    }
}

fn rig() -> Rig {
    rig_with(EarthSettings::at(LAT0, LON0))
}

impl Rig {
    /// The surface of a column as the elevation field says, at the zoom of a sample spacing of `step`.
    fn surface(&self, x: i32, z: i32, step: i32) -> i32 {
        let field = self.generator.field();
        let zoom = field.zoom_for(f64::from(step) * self.settings.metres_per_block);
        field.surface_y_blocking(x, z, zoom).unwrap().unwrap()
    }

    /// The terrain rule, restated.
    fn expected(&self, y: i32, s: i32) -> u32 {
        let sea = self.settings.sea_level_y;
        let b = &self.blocks;
        if y <= s - 4 {
            b.stone
        } else if s >= sea {
            if y < s {
                b.dirt
            } else if y == s {
                b.grass_block
            } else {
                b.air
            }
        } else if y <= s {
            b.sand
        } else if y < sea {
            b.water
        } else {
            b.air
        }
    }

    fn expected_cube(&self, cube: SectionPos) -> Box<[u32; 4096]> {
        let mut out = Box::new([0; 4096]);
        for z in 0..16 {
            for x in 0..16 {
                let s = self.surface(cube.x * 16 + x, cube.z * 16 + z, 1);
                for y in 0..16 {
                    out[(y << 8 | z << 4 | x) as usize] = self.expected(cube.y * 16 + y, s);
                }
            }
        }
        out
    }

    /// Generates the cube and checks it block by block, and that it is reported uniform exactly when it is.
    fn check_cube(&self, cube: SectionPos) -> Option<u32> {
        let mut got = Box::new([0; 4096]);
        let uniform = self.generator.generate_blocking(cube, &mut got);
        let want = self.expected_cube(cube);
        let at = |i: usize| (i & 15, i >> 8, i >> 4 & 15);
        if let Some(i) = (0..4096).find(|&i| got[i] != want[i]) {
            panic!(
                "cube {cube:?}: block {:?} is {} but should be {}",
                at(i),
                got[i],
                want[i]
            );
        }
        let all_same = want.iter().all(|&b| b == want[0]);
        assert_eq!(uniform, all_same.then_some(want[0]), "cube {cube:?}");
        uniform
    }
}

fn cube(x: i32, y: i32, z: i32) -> SectionPos {
    SectionPos::new(x, y, z)
}

#[test]
fn a_cube_follows_the_terrain_rule_at_the_coast_on_a_slope_and_deep_underground() {
    let rig = rig();
    let mut seen = std::collections::BTreeSet::new();
    let mut note = |c: SectionPos| {
        let mut out = Box::new([0; 4096]);
        rig.generator.generate_blocking(c, &mut out);
        seen.extend(out.iter().copied());
    };
    // The coast: columns either side of x = 0, from rock to sky.
    for cx in -2..=1 {
        for cy in -3..=2 {
            rig.check_cube(cube(cx, cy, 0));
            note(cube(cx, cy, 0));
        }
    }
    // Out to sea, where the ground is under water, and up the slope and the spire's flank.
    for cx in [-40, -20] {
        for cy in -40..=0 {
            rig.check_cube(cube(cx, cy, 1));
        }
    }
    for cx in [50, 100, 140, 150, 156] {
        let s = rig.surface(cx * 16 + 8, 8, 1);
        for cy in (s >> 4) - 2..=(s >> 4) + 2 {
            rig.check_cube(cube(cx, cy, 0));
            note(cube(cx, cy, 0));
        }
    }
    // Deep underground and high in the sky are uniform.
    assert_eq!(rig.check_cube(cube(3, -400, -7)), Some(rig.blocks.stone));
    assert_eq!(rig.check_cube(cube(3, 700, -7)), Some(rig.blocks.air));
    let b = rig.blocks;
    for id in [b.air, b.stone, b.dirt, b.grass_block, b.sand, b.water] {
        assert!(seen.contains(&id), "no block {id} among the cubes tested");
    }
}

#[test]
fn a_column_of_known_height_is_made_of_the_blocks_the_rule_says() {
    // Elevation 20 m everywhere, sea level at 30: the ground is under water.
    let mut settings = EarthSettings::at(LAT0, LON0);
    settings.sea_level_y = 30;
    let dir = TempDir::new();
    let fetcher = Arc::new(MemoryFetcher::new(|_, _| 20.0));
    let store = Arc::new(TileStore::new(dir.path(), fetcher, 16));
    let blocks = EarthBlocks::from_registry(default_registry()).unwrap();
    let g = EarthGenerator::new(&settings, store, blocks).unwrap();
    let mut out = Box::new([0; 4096]);
    // 20 m of elevation at one metre a block over sea level 30: the surface is block 50, well above the sea.
    assert_eq!(g.surface_y_blocking(3, 3), Some(50));
    let at = |g: &EarthGenerator, y: i32| g.block_at_blocking(3, y, 3, 1);
    assert_eq!(at(&g, 46), blocks.stone);
    assert_eq!(at(&g, 47), blocks.dirt);
    assert_eq!(at(&g, 49), blocks.dirt);
    assert_eq!(at(&g, 50), blocks.grass_block);
    assert_eq!(at(&g, 51), blocks.air);
    g.generate_blocking(cube(0, 3, 0), &mut out);
    assert_eq!(out[(2 << 8) | (3 << 4) | 3], blocks.grass_block);

    let dir = TempDir::new();
    let fetcher = Arc::new(MemoryFetcher::new(|_, _| -20.0));
    let store = Arc::new(TileStore::new(dir.path(), fetcher, 16));
    let g = EarthGenerator::new(&settings, store, blocks).unwrap();
    // Surface at 30 - 20 = 10, sea at 30: sand 7 to 10, water 11 to 29, air from 30.
    assert_eq!(g.surface_y_blocking(3, 3), Some(10));
    assert_eq!(at(&g, 6), blocks.stone);
    assert_eq!(at(&g, 7), blocks.sand);
    assert_eq!(at(&g, 10), blocks.sand);
    assert_eq!(at(&g, 11), blocks.water);
    assert_eq!(at(&g, 29), blocks.water);
    assert_eq!(at(&g, 30), blocks.air);
}

#[test]
fn every_cube_reported_uniform_is_the_same_as_filling_it_block_by_block() {
    let rig = rig();
    let (mut air, mut stone, mut water, mut mixed) = (0, 0, 0, 0);
    let mut sweep = |cx: i32, cz: i32, cys: std::ops::RangeInclusive<i32>| {
        for cy in cys {
            match rig.check_cube(cube(cx, cy, cz)) {
                Some(b) if b == rig.blocks.air => air += 1,
                Some(b) if b == rig.blocks.stone => stone += 1,
                Some(b) if b == rig.blocks.water => water += 1,
                Some(b) => panic!("a cube of block {b}"),
                None => mixed += 1,
            }
        }
    };
    // The coast, with the soil, the sea level and the ripples all inside the range.
    for cx in -8..=6 {
        for cz in -2..=2 {
            sweep(cx, cz, -6..=4);
        }
    }
    // The sea floor, which slopes down to the west, and the open sea above it.
    for cx in [-60, -45, -30] {
        sweep(cx, 0, -60..=3);
    }
    // The land that climbs east and the spire.
    for cx in (20..=170).step_by(10) {
        let s = rig.surface(cx * 16 + 8, 8, 1);
        sweep(cx, 0, (s >> 4) - 12..=(s >> 4) + 12);
    }
    assert!(
        air > 100 && stone > 100 && water > 20 && mixed > 100,
        "{air} {stone} {water} {mixed}"
    );
}

/// Every block of the box `min..=max` as the terrain rule has it, at one-block samples.
fn blocks_in(
    rig: &Rig,
    min: [i32; 3],
    max: [i32; 3],
    step: i32,
) -> std::collections::BTreeSet<u32> {
    let mut set = std::collections::BTreeSet::new();
    for z in min[2]..=max[2] {
        for x in min[0]..=max[0] {
            let s = rig.surface(x, z, step);
            for y in min[1]..=max[1] {
                set.insert(rig.expected(y, s));
            }
        }
    }
    set
}

#[test]
fn a_classified_box_is_never_called_uniform_when_it_is_not() {
    let rig = rig();
    let g = &rig.generator;
    let (mut proven, mut total) = (0, 0);
    let mut seed = 5u64;
    let mut rand = |n: i32| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as i32).rem_euclid(n)
    };
    for _ in 0..400 {
        let size = [1, 2, 5, 16, 33][rand(5) as usize];
        let height = [1, 3, 16, 40][rand(4) as usize];
        // Mostly around the coast, the soil and the slope; some on the spire's flank.
        let x = if rand(4) == 0 {
            2000 + rand(700)
        } else {
            rand(120) - 60
        };
        let z = rand(80) - 40;
        let base = if x > 1000 { rig.surface(x, z, 1) } else { 0 };
        let y = base + rand(120) - 90;
        let (min, max) = ([x, y, z], [x + size - 1, y + height - 1, z + size - 1]);
        let fill = g.classify_box_blocking(min, max, 1);
        total += 1;
        if let Some(block) = g.fill_block(fill) {
            proven += 1;
            let blocks = blocks_in(&rig, min, max, 1);
            assert_eq!(
                blocks.len(),
                1,
                "{min:?}..{max:?} is {fill:?} but holds {blocks:?}"
            );
            assert!(
                blocks.contains(&block),
                "{min:?}..{max:?} is {fill:?} but holds {blocks:?}"
            );
        }
    }
    assert!(proven > total / 4, "only {proven} of {total} boxes proven");
    // Rock, sky and open sea are each proven.
    let class = |min, max| g.classify_box_blocking(min, max, 1);
    assert_eq!(class([0, -900, 0], [15, -500, 15]), Fill::Stone);
    assert_eq!(class([0, 900, 0], [15, 950, 15]), Fill::Air);
    assert_eq!(class([-900, -100, 0], [-890, -60, 10]), Fill::Water);
}

#[test]
fn a_far_node_is_never_called_uniform_when_its_voxels_are_not() {
    let rig = rig();
    let g = &rig.generator;
    for level in 0..=6u32 {
        let step = 1i32 << level;
        let side = 16 * step;
        let (mut proven, mut total) = (0, 0);
        // The sea, the coast, the land, and the spire's foot, flank and summit.
        for (x, z) in [
            (-3000 / side, 0),
            (-1, 0),
            (0, -1),
            (1000 / side, 0),
            (2300 / side, 0),
            (2500 / side, -1),
        ] {
            let (x0, z0) = (x * side, z * side);
            let r = g.field().range(
                x0,
                z0,
                x0 + side,
                z0 + side,
                rig.generator.zoom_for_step(step),
            );
            let (lo, hi) = match r {
                Some(r) => r,
                None => {
                    g.missing_tiles_for_node(level, x, z)
                        .into_iter()
                        .for_each(|t| {
                            rig.store.load(t).unwrap();
                        });
                    g.field()
                        .range(
                            x0,
                            z0,
                            x0 + side,
                            z0 + side,
                            rig.generator.zoom_for_step(step),
                        )
                        .unwrap()
                }
            };
            let (ylo, yhi) = (
                (rig.settings.sea_level_y as f32 + lo - 20.0) as i32 / side - 2,
                (rig.settings.sea_level_y as f32 + hi + 20.0) as i32 / side + 2,
            );
            for y in ylo..=yhi {
                let fill = g.classify_node_blocking(level, x, y, z);
                total += 1;
                // The voxels, sampled where the far terrain samples them.
                let mut blocks = std::collections::BTreeSet::new();
                for vz in 0..16 {
                    for vx in 0..16 {
                        let (wx, wz) = (x0 + vx * step + step / 2, z0 + vz * step + step / 2);
                        let s = rig.surface(wx, wz, step);
                        for vy in 0..16 {
                            let wy = y * side + vy * step + step / 2;
                            let want = rig.expected(wy, s);
                            blocks.insert(want);
                            if (vx + vy + vz) % 7 == 0 {
                                assert_eq!(
                                    g.block_at_blocking(wx, wy, wz, step),
                                    want,
                                    "level {level} step {step} at {wx},{wy},{wz}"
                                );
                            }
                        }
                    }
                }
                if let Some(block) = g.fill_block(fill) {
                    proven += 1;
                    assert_eq!(
                        blocks.iter().copied().collect::<Vec<_>>(),
                        vec![block],
                        "level {level} node {x},{y},{z} is {fill:?}"
                    );
                }
            }
        }
        assert!(
            proven >= 3 && proven < total,
            "level {level}: {proven} of {total} nodes proven"
        );
    }
}

#[test]
fn a_coarse_level_reads_the_zoom_of_its_step() {
    let rig = rig();
    let g = &rig.generator;
    assert_eq!(g.base_zoom(), rig.generator.field().zoom_for(1.0));
    assert_eq!(g.zoom_for_step(1), g.base_zoom());
    let zooms: Vec<u8> = (0..=6).map(|level| g.zoom_for_step(1 << level)).collect();
    assert!(
        zooms.windows(2).all(|w| w[0] >= w[1]) && zooms[6] < zooms[0],
        "{zooms:?}"
    );
    // The tiles a level-5 node needs are of that level's zoom, not the base zoom.
    let tiles = g.missing_tiles_for_node(5, 0, 0);
    assert!(
        !tiles.is_empty() && tiles.iter().all(|t| t.z == zooms[5]),
        "{tiles:?}"
    );
}

#[test]
fn the_tiles_missing_for_a_step_are_those_of_that_steps_zoom() {
    let rig = rig();
    let g = &rig.generator;
    // Step 1 is the plain call; coarser steps list tiles of their own zoom, and the same area needs no more of them.
    assert_eq!(
        g.missing_tiles_for_step(-500, -300, 700, 900, 1),
        g.missing_tiles(-500, -300, 700, 900)
    );
    let base = g.missing_tiles(-500, -300, 700, 900);
    for level in 2..=6 {
        let step = 1 << level;
        let tiles = g.missing_tiles_for_step(700, 900, -500, -300, step);
        assert!(!tiles.is_empty());
        assert!(
            tiles.iter().all(|t| t.z == g.zoom_for_step(step)),
            "{tiles:?}"
        );
        assert!(tiles.len() <= base.len());
    }
    // A local tile is left out.
    let coarse = g.missing_tiles_for_step(0, 0, 15, 15, 8);
    rig.store.load(coarse[0]).unwrap();
    assert!(
        !g.missing_tiles_for_step(0, 0, 15, 15, 8)
            .contains(&coarse[0])
    );
}

#[test]
fn the_surface_is_exact_however_far_above_the_hint_it_is() {
    let rig = rig();
    // The spire's top: about 8,000 blocks above a column at sea level, and 4,096 is as far as a search goes.
    let (x, z) = (2500, 0);
    let s = rig.surface(x, z, 1);
    assert!(s > 4096 + 3000, "the spire is only {s} high");
    assert_eq!(rig.generator.surface_y_blocking(x, z), Some(s));
    let custom =
        CustomCubicGenerator::earth(&rig.settings, rig.store.clone(), default_registry()).unwrap();
    for hint in [0, -9000, 100, s, 20_000] {
        assert_eq!(custom.surface_y(x, z, hint), Some(s), "hint {hint}");
    }
    // The generated column agrees: grass at the top, air above, soil below.
    assert_eq!(custom.block_at(x, s, z, 1), rig.blocks.grass_block);
    assert_eq!(custom.block_at(x, s + 1, z, 1), rig.blocks.air);
    assert_eq!(custom.block_at(x, s - 1, z, 1), rig.blocks.dirt);
}

#[test]
fn generation_is_deterministic_and_does_not_depend_on_what_came_before() {
    let a = rig();
    let b = rig();
    let cubes: Vec<SectionPos> = (-3..=3)
        .flat_map(|x| (-3..=2).flat_map(move |y| (-1..=1).map(move |z| cube(x, y, z))))
        .collect();
    let make = |g: &EarthGenerator, c: SectionPos| {
        let mut out = Box::new([0; 4096]);
        let uniform = g.generate_blocking(c, &mut out);
        (out, uniform)
    };
    let forwards: Vec<_> = cubes.iter().map(|&c| make(&a.generator, c)).collect();
    // A second generator over an empty store, in the opposite order, interleaved with distant cubes.
    let mut backwards: Vec<_> = Vec::new();
    for &c in cubes.iter().rev() {
        make(&b.generator, cube(c.x * 7 + 90, 5, c.z * 3 - 40));
        backwards.push(make(&b.generator, c));
    }
    backwards.reverse();
    let again: Vec<_> = cubes.iter().map(|&c| make(&a.generator, c)).collect();
    for (i, c) in cubes.iter().enumerate() {
        assert_eq!(forwards[i], backwards[i], "{c:?}");
        assert_eq!(forwards[i], again[i], "{c:?}");
    }
}

#[test]
fn a_missing_tile_is_not_ready_and_is_listed_then_loaded_by_the_blocking_form() {
    let rig = rig();
    let g = &rig.generator;
    let origin = latlon_to_tile_pixel(LAT0, LON0, 15).0;
    let mut out = Box::new([0; 4096]);
    let err = g.generate(cube(0, 0, 0), &mut out).unwrap_err();
    assert!(
        err.tiles.contains(&origin),
        "{err} does not list {origin:?}"
    );
    assert!(
        err.to_string()
            .contains(&format!("{}/{}/{}", origin.z, origin.x, origin.y))
    );
    assert!(g.block_at(3, 3, 3, 1).unwrap_err().tiles.contains(&origin));
    assert!(g.surface_y(3, 3).is_err());
    assert!(g.classify_box([0, 0, 0], [15, 15, 15]).is_err());
    assert!(g.classify_node(0, 0, 0, 0).is_err());
    assert!(g.chunk_tops(0, 0).is_err());
    assert!(g.first_at_column(3, 3, &[1, 2], 1, |_| true).is_err());
    assert!(g.missing_tiles(0, 0, 15, 15).contains(&origin));
    assert!(g.missing_tiles_for_cube(cube(0, 0, 0)).contains(&origin));
    // Asking invented nothing and fetched nothing.
    assert_eq!(rig.fetcher.fetch_count(), 0);
    assert_eq!(out.iter().filter(|&&b| b != 0).count(), 0);

    // The blocking form fetches through the store, and then the plain form answers too.
    let uniform = g.generate_blocking(cube(0, 0, 0), &mut out);
    assert!(rig.fetcher.fetches_of(origin) >= 1);
    let mut again = Box::new([0; 4096]);
    assert_eq!(g.generate(cube(0, 0, 0), &mut again), Ok(uniform));
    assert_eq!(out[..], again[..]);
    assert!(g.missing_tiles(0, 0, 15, 15).is_empty());
    // Once loaded, tiles are not fetched again.
    let fetched = rig.fetcher.fetch_count();
    g.generate_blocking(cube(0, 0, 0), &mut out);
    assert_eq!(rig.fetcher.fetch_count(), fetched);
}

#[test]
fn a_tile_missing_for_a_cube_far_from_the_loaded_ones_does_not_poison_it() {
    let rig = rig();
    let g = &rig.generator;
    let mut out = Box::new([0; 4096]);
    g.generate_blocking(cube(0, 0, 0), &mut out);
    // A cube a few tiles away is not ready, and says which tile: the one holding its blocks.
    let far = cube(500, 0, 0);
    let err = g.generate(far, &mut out).unwrap_err();
    assert!(!err.tiles.is_empty());
    let want = latlon_to_tile_pixel(
        rig.generator
            .field()
            .projection()
            .to_lat_lon(8000.5, 8.5)
            .lat,
        rig.generator
            .field()
            .projection()
            .to_lat_lon(8000.5, 8.5)
            .lon,
        15,
    )
    .0;
    assert!(err.tiles.contains(&want), "{err} lacks {want:?}");
}

#[test]
fn a_failing_fetcher_with_the_cancel_flag_set_returns_promptly() {
    let rig = rig();
    let g = &rig.generator;
    rig.fetcher
        .fail_with(Some(FetchError::Temporary("offline".into())));
    let air = rig.blocks.air;
    let mut out = Box::new([1; 4096]);

    // Cancelled before the call: nothing is even tried.
    g.set_cancelled(true);
    let start = Instant::now();
    assert_eq!(g.generate_blocking(cube(0, 0, 0), &mut out), Some(air));
    assert!(out.iter().all(|&b| b == air));
    assert_eq!(g.block_at_blocking(1, 1, 1, 1), air);
    assert_eq!(g.surface_y_blocking(1, 1), None);
    assert_eq!(g.first_at_column_blocking(1, 1, &[0, 1], 1, |_| true), None);
    assert_eq!(g.chunk_tops_blocking(0, 0), None);
    assert_eq!(g.classify_box_blocking([0; 3], [15; 3], 1), Fill::Mixed);
    assert_eq!(g.classify_node_blocking(2, 0, 0, 0), Fill::Mixed);
    assert!(start.elapsed() < Duration::from_secs(1));
    assert_eq!(rig.fetcher.fetch_count(), 0);

    // Cancelled while it retries: it keeps trying until the flag goes up, then returns.
    g.set_cancelled(false);
    let flag = g.cancel_flag();
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(120));
        flag.store(true, Ordering::SeqCst);
    });
    let start = Instant::now();
    assert_eq!(g.block_at_blocking(1, 1, 1, 1), air);
    canceller.join().unwrap();
    assert!(start.elapsed() < Duration::from_secs(5));
    assert!(
        rig.fetcher.fetch_count() >= 2,
        "{} fetches",
        rig.fetcher.fetch_count()
    );
}

#[test]
fn a_fetcher_that_recovers_is_retried_until_the_terrain_comes() {
    let rig = rig();
    let g = &rig.generator;
    rig.fetcher
        .fail_with(Some(FetchError::Temporary("offline".into())));
    let fetcher = rig.fetcher.clone();
    let healer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        fetcher.fail_with(None);
    });
    let mut out = Box::new([0; 4096]);
    g.generate_blocking(cube(0, -1, 0), &mut out);
    healer.join().unwrap();
    assert!(rig.fetcher.fetch_count() >= 2);
    assert_eq!(out[..], rig.expected_cube(cube(0, -1, 0))[..]);
}

#[test]
fn the_earth_kind_of_the_custom_generator_delegates_to_the_earth_generator() {
    let rig = rig();
    let custom =
        CustomCubicGenerator::earth(&rig.settings, rig.store.clone(), default_registry()).unwrap();
    let b = rig.blocks;
    // Cubes, whole and in blocks, with and without the carvers' stage.
    let want = rig.expected_cube(cube(-1, -1, 0));
    let mut out = Box::new([0; 4096]);
    custom.generate(cube(-1, -1, 0), &mut out);
    assert_eq!(out[..], want[..]);
    custom.base_cube(cube(-1, -1, 0), &mut out);
    assert_eq!(out[..], want[..]);
    assert_eq!(
        custom.block_at(-5, -20, 3, 1),
        rig.expected(-20, rig.surface(-5, 3, 1))
    );
    // Coarse samples read their own zoom.
    assert_eq!(
        custom.block_at(500, 100, -64, 8),
        rig.expected(100, rig.surface(500, -64, 8))
    );
    // The ground at x = 500 is about 200 up: the first of these heights that is not rock is the sky.
    let found = custom.first_at_column(500, 3, &[-9000, 0, 100, 9000], 1, |blk| blk != b.stone);
    assert_eq!(found, Some((3, b.air)));
    assert_eq!(
        custom.blocks_at_column(0, 0, &[-5000, 5000], 1),
        vec![b.stone, b.air]
    );
    let tops = custom.chunk_tops(0, 0);
    for z in 0..16 {
        for x in 0..16 {
            assert_eq!(tops[z << 4 | x], rig.surface(x as i32, z as i32, 1));
        }
    }
    assert_eq!(custom.sea_level(), 0);
    assert_eq!(custom.biome_scale(), 1.0);
    assert!(!custom.has_areas());
    // Classification, with water counting as "air" as it does for the density terrain.
    assert_eq!(
        custom.classify_box([0, -900, 0], [15, -500, 15]),
        BoxClass::Solid
    );
    assert_eq!(
        custom.uniform_block([0, -900, 0], [15, -500, 15]),
        Some(b.stone)
    );
    assert_eq!(
        custom.classify_box([0, 900, 0], [15, 950, 15]),
        BoxClass::Air
    );
    assert_eq!(
        custom.uniform_block([0, 900, 0], [15, 950, 15]),
        Some(b.air)
    );
    assert_eq!(
        custom.classify_box([-900, -100, 0], [-890, -60, 10]),
        BoxClass::Air
    );
    assert_eq!(
        custom.uniform_block([-900, -100, 0], [-890, -60, 10]),
        Some(b.water)
    );
    assert_eq!(
        custom.classify_box([-5, -3, 0], [5, 3, 10]),
        BoxClass::Mixed
    );
    assert_eq!(custom.uniform_block([-5, -3, 0], [5, 3, 10]), None);
    let sky = custom.classify_node(3, 0, 40, 0, 8.0);
    assert_eq!((sky.class, sky.block), (BoxClass::Air, Some(b.air)));
    let rock = custom.classify_node(3, 0, -40, 0, 8.0);
    assert_eq!((rock.class, rock.block), (BoxClass::Solid, Some(b.stone)));
    // Nothing for the GPU.
    assert!(custom.gpu_terms().is_none() && custom.density_bound().is_none());
    assert!(custom.noise_tables().is_none() && custom.density_shape().is_none());
    assert!(custom.column_terrain(0, 0).is_none());
    // The Earth generator is reachable, and cancelling reaches it.
    let earth = custom.earth_generator().unwrap();
    assert!(!earth.is_cancelled());
    custom.set_earth_cancelled(true);
    assert!(earth.is_cancelled());
    assert_eq!(custom.block_at(1, 1, 1, 1), b.air);
}

#[test]
fn settings_shape_the_world() {
    // Half a block per metre of height and double the relief: elevation 100 m at 2 m a block is 50 up, 100 with
    // the exaggeration; sea level at 10.
    let mut settings = EarthSettings::at(LAT0, LON0);
    settings.metres_per_block = 2.0;
    settings.vertical_exaggeration = 2.0;
    settings.sea_level_y = 10;
    let dir = TempDir::new();
    let fetcher = Arc::new(MemoryFetcher::new(|_, _| 100.0));
    let store = Arc::new(TileStore::new(dir.path(), fetcher, 16));
    let blocks = EarthBlocks::from_registry(default_registry()).unwrap();
    let g = EarthGenerator::new(&settings, store, blocks).unwrap();
    assert_eq!(g.surface_y_blocking(0, 0), Some(110));
    assert_eq!(g.sea_level(), 10);
    assert_eq!(g.biome(), "plains");
    assert_eq!(g.base_zoom(), g.field().zoom_for(2.0));
    // A setting the generator cannot work with is refused.
    settings.metres_per_block = 0.0;
    let dir = TempDir::new();
    let store = Arc::new(TileStore::new(
        dir.path(),
        Arc::new(MemoryFetcher::new(|_, _| 0.0)),
        1,
    ));
    assert!(EarthGenerator::new(&settings, store, blocks).is_err());
}

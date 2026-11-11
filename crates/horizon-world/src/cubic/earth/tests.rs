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

pub(super) const LAT0: f64 = 45.0;
pub(super) const LON0: f64 = 7.0;

/// The permanent snow line in metres, restated from the rule: 5,500 m to 30 degrees, then straight down to 0 m at
/// 80 degrees.
fn snow_line(latitude: f64) -> f64 {
    let lat = latitude.abs();
    if lat <= 30.0 {
        5500.0
    } else if lat >= 80.0 {
        0.0
    } else {
        5500.0 - (lat - 30.0) * 110.0
    }
}

/// A coast along `east = 0` (sea to the west, a long slope up to the east), ripples, and a 7 km spire 2.5 km
/// east of the origin whose flanks are far steeper than one block per block.
pub(super) fn terrain(lat: f64, lon: f64) -> f64 {
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

    /// The latitude of the column at block row `z`, from the projection's own definition.
    fn latitude(&self, z: i32) -> f64 {
        LAT0 - ((f64::from(z) + 0.5) * self.settings.metres_per_block / EARTH_RADIUS_M).to_degrees()
    }

    /// The terrain rule, restated: what the column at row `z` whose surface is `s` has at height `y` (one-block
    /// samples).
    fn expected(&self, y: i32, s: i32, z: i32) -> u32 {
        let sea = self.settings.sea_level_y;
        let b = &self.blocks;
        let elevation = f64::from(s - sea) * self.settings.metres_per_block
            / self.settings.vertical_exaggeration;
        let line = snow_line(self.latitude(z));
        if y <= s - 4 {
            b.stone
        } else if y > s {
            if s < sea && y < sea { b.water } else { b.air }
        } else if s < sea {
            b.sand
        } else if elevation >= line {
            if y >= s - 1 { b.snow_block } else { b.stone }
        } else if elevation >= line - 1500.0 {
            b.stone
        } else if y == s {
            b.grass_block
        } else {
            b.dirt
        }
    }

    /// The far voxel from `lo` to `lo + step - 1` of the column at row `z`: the surface block when the voxel
    /// holds the surface, else the block at its top (below the ground) or bottom (above it).
    fn expected_voxel(&self, lo: i32, step: i32, s: i32, z: i32) -> u32 {
        let top = lo + step - 1;
        let at = if lo <= s && s <= top {
            s
        } else if top < s {
            top
        } else {
            lo
        };
        self.expected(at, s, z)
    }

    fn expected_cube(&self, cube: SectionPos) -> Box<[u32; 4096]> {
        let mut out = Box::new([0; 4096]);
        for z in 0..16 {
            for x in 0..16 {
                let s = self.surface(cube.x * 16 + x, cube.z * 16 + z, 1);
                for y in 0..16 {
                    out[(y << 8 | z << 4 | x) as usize] =
                        self.expected(cube.y * 16 + y, s, cube.z * 16 + z);
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
    for id in [
        b.air,
        b.stone,
        b.dirt,
        b.grass_block,
        b.sand,
        b.water,
        b.snow_block,
    ] {
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
                set.insert(rig.expected(y, s, z));
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
                            let lo = y * side + vy * step;
                            let wy = lo + step / 2;
                            let want = rig.expected_voxel(lo, step, s, wz);
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
    // The generated column agrees: snow at the top (7 km is above the snow line), air above, snow then stone below.
    assert_eq!(custom.block_at(x, s, z, 1), rig.blocks.snow_block);
    assert_eq!(custom.block_at(x, s + 1, z, 1), rig.blocks.air);
    assert_eq!(custom.block_at(x, s - 1, z, 1), rig.blocks.snow_block);
    assert_eq!(custom.block_at(x, s - 2, z, 1), rig.blocks.stone);
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
        rig.expected(-20, rig.surface(-5, 3, 1), 3)
    );
    // Coarse samples read their own zoom.
    assert_eq!(
        custom.block_at(500, 100, -64, 8),
        rig.expected_voxel(96, 8, rig.surface(500, -64, 8), -64)
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

// --- The surface: snow line, alpine band, far voxels -----------------------------------------------------------

/// A generator over terrain whose elevation is `elevation(lat)` metres, with the origin at `lat0` degrees north,
/// `metres_per_block` metres to a block and the sea at block 0.
fn flat(
    lat0: f64,
    metres_per_block: f64,
    elevation: impl Fn(f64) -> f64 + Send + Sync + 'static,
) -> (EarthGenerator, EarthBlocks, TempDir) {
    let mut settings = EarthSettings::at(lat0, LON0);
    settings.metres_per_block = metres_per_block;
    let dir = TempDir::new();
    let store = Arc::new(TileStore::new(
        dir.path(),
        Arc::new(MemoryFetcher::new(move |lat, _| elevation(lat))),
        4096,
    ));
    let blocks = EarthBlocks::from_registry(default_registry()).unwrap();
    let g = EarthGenerator::new(&settings, store, blocks).unwrap();
    (g, blocks, dir)
}

/// The blocks of the column at `(0, z)` from `s - 5` to `s + 2`, at one-block samples.
fn column(g: &EarthGenerator, z: i32) -> (i32, Vec<u32>) {
    let s = g.surface_y_blocking(0, z).unwrap();
    let blocks = (s - 5..=s + 2)
        .map(|y| g.block_at_blocking(0, y, z, 1))
        .collect();
    (s, blocks)
}

#[test]
fn the_snow_line_falls_with_latitude_from_5500_m_at_30_to_nothing_at_80() {
    use super::surface::{SurfaceKind, snow_line_m};
    for (lat, line) in [
        (0.0, 5500.0),
        (30.0, 5500.0),
        (-30.0, 5500.0),
        (55.0, 2750.0),
        (-55.0, 2750.0),
        (45.0, 3850.0),
        (80.0, 0.0),
        (-85.0, 0.0),
    ] {
        assert!(
            (snow_line_m(lat) - line).abs() < 1e-9,
            "{lat}: {}",
            snow_line_m(lat)
        );
        assert_eq!(snow_line(lat), line);
    }
    // At 45 degrees: grass to 2,350 m, bare stone to 3,850 m, snow from there.
    let kind = |m| SurfaceKind::on_land(m, 45.0);
    assert_eq!(kind(0.0), SurfaceKind::Grass);
    assert_eq!(kind(2349.0), SurfaceKind::Grass);
    assert_eq!(kind(2350.0), SurfaceKind::Alpine);
    assert_eq!(kind(3849.0), SurfaceKind::Alpine);
    assert_eq!(kind(3850.0), SurfaceKind::Snow);
    assert_eq!(kind(8800.0), SurfaceKind::Snow);
}

#[test]
fn a_column_is_grass_bare_stone_or_snow_by_elevation_and_latitude_and_sand_under_the_sea() {
    // (origin latitude, elevation in metres, the top two blocks of the column from the surface down, and the one
    // under them)
    let cases = [
        // Tropics: the line is 5,500 m, the band 4,000 to 5,500.
        (10.0, 3999.0, "grass"),
        (10.0, 4000.0, "alpine"),
        (-10.0, 5499.0, "alpine"),
        (-10.0, 5500.0, "snow"),
        // 45 degrees: the line is 3,850 m.
        (45.0, 100.0, "grass"),
        (45.0, 2349.0, "grass"),
        (45.0, 3000.0, "alpine"),
        (45.0, 4000.0, "snow"),
        // 70 degrees: the line is 1,100 m, the band from -400 m: all land above sea level is bare or snowy.
        (70.0, 50.0, "alpine"),
        (70.0, 1200.0, "snow"),
        // Past 80 degrees there is snow down to the sea.
        (82.0, 5.0, "snow"),
        // Under the sea it is sand, whatever the latitude.
        (45.0, -30.0, "sand"),
        (82.0, -30.0, "sand"),
    ];
    for (lat, metres, want) in cases {
        let (g, b, _dir) = flat(lat, 1.0, move |_| metres);
        let (s, col) = column(&g, 0);
        assert_eq!(s, metres.floor() as i32);
        // From s - 5 to s + 2.
        let at = |dy: i32| col[(dy + 5) as usize];
        let (top, below) = match want {
            "grass" => (b.grass_block, b.dirt),
            "alpine" => (b.stone, b.stone),
            "snow" => (b.snow_block, b.snow_block),
            "sand" => (b.sand, b.sand),
            _ => unreachable!(),
        };
        assert_eq!(at(0), top, "{lat} {metres}: top");
        assert_eq!(at(-1), below, "{lat} {metres}: one under");
        match want {
            "grass" | "sand" => {
                assert_eq!((at(-2), at(-3)), (below, below), "{lat} {metres}");
                assert_eq!(at(-4), b.stone, "{lat} {metres}");
            }
            // Two blocks of snow, then stone.
            "snow" => {
                assert_eq!((at(-2), at(-3), at(-5)), (b.stone, b.stone, b.stone));
            }
            _ => assert!((-5..=0).all(|d| at(d) == b.stone)),
        }
        // Above the ground: air, or water up to the sea level over a sunken column.
        assert_eq!(at(1), if s < 0 { b.water } else { b.air }, "{lat} {metres}");
        assert_eq!(at(2), if s < -1 { b.water } else { b.air });
    }
}

#[test]
fn a_column_uses_its_own_latitude_not_the_origins() {
    // 1,000 m to a block: a degree of latitude is about 111 blocks. Land 3,000 m high is block 3. At the origin
    // (40 degrees north: line 4,400 m, band from 2,900 m) it is bare stone; 1,200 blocks south of it (about 29.2
    // degrees: line 5,500 m, band from 4,000 m) it is grass; 400 blocks north (43.6 degrees) it is still bare.
    let (g, b, _dir) = flat(40.0, 1000.0, |_| 3000.0);
    assert_eq!(g.surface_y_blocking(0, 0), Some(3));
    assert_eq!(g.block_at_blocking(0, 3, 0, 1), b.stone);
    assert_eq!(g.block_at_blocking(0, 3, -400, 1), b.stone);
    assert_eq!(g.block_at_blocking(0, 3, 1200, 1), b.grass_block);
    // The far view reads the same: a voxel of 8 holding the surface.
    assert_eq!(g.block_at_blocking(0, 0, 0, 8), b.stone);
    assert_eq!(g.block_at_blocking(0, 0, 1200, 8), b.grass_block);
    // And a cube agrees with the point queries across the change.
    let mut out = Box::new([0; 4096]);
    g.generate_blocking(cube(0, 0, 75), &mut out);
    assert_eq!(out[3 << 8], b.grass_block);
    g.generate_blocking(cube(0, 0, 0), &mut out);
    assert_eq!(out[3 << 8], b.stone);
}

#[test]
fn a_far_voxel_holding_the_surface_is_the_surface_block_of_each_kind_at_every_step() {
    // Grass, sand, snow and bare stone, each from its own latitude and elevation.
    let cases: [(f64, f64, fn(&EarthBlocks) -> u32); 4] = [
        (45.0, 120.5, |b| b.grass_block),
        (45.0, -37.5, |b| b.sand),
        (20.0, 6100.5, |b| b.snow_block),
        (45.0, 3000.5, |b| b.stone),
    ];
    for (lat, metres, surface_block) in cases {
        let (g, b, _dir) = flat(lat, 1.0, move |_| metres);
        let s = g.surface_y_blocking(5, 9).unwrap();
        let want = surface_block(&b);
        for step in [2, 8, 64] {
            let lo = s.div_euclid(step) * step;
            // Every height of the voxel that holds the surface asks for the same voxel.
            for y in lo..lo + step {
                assert_eq!(
                    g.block_at_blocking(5, y, 9, step),
                    want,
                    "lat {lat}, {metres} m, step {step}, y {y} (surface {s})"
                );
            }
            // Above it is air or water, below it ground, never the surface block's neighbour by accident.
            let above = g.block_at_blocking(5, lo + step, 9, step);
            assert_eq!(
                above,
                if s < 0 && lo + step < 0 {
                    b.water
                } else {
                    b.air
                }
            );
            let below = g.block_at_blocking(5, lo - step, 9, step);
            assert_ne!(below, b.air);
        }
    }
}

#[test]
fn the_far_blocks_along_a_smooth_slope_are_all_the_surface_material() {
    // The old rule took the block at a voxel's own height, so grass, dirt and stone alternated along a slope.
    // A gentle slope through the grass zone (up to about 700 m at 45 degrees, short of the spire: all below the alpine band at 2,350 m), seen at three steps.
    let rig = rig();
    let g = &rig.generator;
    let mut seen_heights = std::collections::BTreeSet::new();
    for step in [2, 8, 64] {
        let zoom_step = step;
        // The voxel columns of a row across the slope, from the coast eastwards, each at its own middle block.
        for column in 0..(1000 / step) {
            let x = 40 + column * step + step / 2;
            let z = 16 + step / 2;
            let s = rig.surface(x, z, zoom_step);
            assert!(s > 0, "the slope starts above the sea");
            // From well above the ground down to well below, a voxel at a time: the first that is not air.
            let top = (s.div_euclid(step) + 3) * step;
            let found = (0..8).find_map(|i| {
                let lo = top - i * step;
                let block = g.block_at_blocking(x, lo, z, step);
                (block != rig.blocks.air).then_some((lo, block))
            });
            let (lo, block) = found.expect("the ground is within reach");
            assert_eq!(
                block, rig.blocks.grass_block,
                "step {step}, column {x},{z}: the top voxel (from {lo}) over surface {s}"
            );
            assert!(lo <= s && s < lo + step, "the top voxel holds the surface");
            seen_heights.insert(s.rem_euclid(step));
        }
    }
    // The surface sat at many heights within its voxel, so the grid's phase was not what made it come out right.
    assert!(seen_heights.len() > 8, "{seen_heights:?}");
}

#[test]
fn step_one_is_the_old_rule_for_columns_below_the_alpine_band() {
    // The rule before the surface depended on elevation: stone to s - 4, three dirt, grass, air; sand and water
    // under the sea. Every column here is below 2,350 m at 45 degrees.
    let rig = rig();
    let g = &rig.generator;
    let b = rig.blocks;
    let old = |y: i32, s: i32| {
        if y <= s - 4 {
            b.stone
        } else if s >= 0 {
            match y.cmp(&s) {
                std::cmp::Ordering::Less => b.dirt,
                std::cmp::Ordering::Equal => b.grass_block,
                std::cmp::Ordering::Greater => b.air,
            }
        } else if y <= s {
            b.sand
        } else if y < 0 {
            b.water
        } else {
            b.air
        }
    };
    let mut checked = 0;
    for x in (-3000..=5000).step_by(97) {
        for z in [-60, 5, 70] {
            let s = rig.surface(x, z, 1);
            if s as f64 >= 2349.0 {
                continue;
            }
            for y in s - 6..=s + 3 {
                assert_eq!(g.block_at_blocking(x, y, z, 1), old(y, s), "{x},{y},{z}");
            }
            checked += 1;
        }
    }
    assert!(checked > 150, "{checked} columns");
}

#[test]
fn the_spire_has_every_kind_of_top_and_a_far_node_over_it_agrees_with_the_blocks() {
    use super::surface::SurfaceKind;
    let rig = rig();
    let g = &rig.generator;
    let mut kinds = std::collections::BTreeSet::new();
    for x in (-2000..2600).step_by(20) {
        let s = rig.surface(x, 0, 1);
        kinds.insert(format!("{:?}", g.surface_kind(x, 0, s)));
    }
    for k in [
        SurfaceKind::Grass,
        SurfaceKind::Alpine,
        SurfaceKind::Snow,
        SurfaceKind::Sand,
    ] {
        assert!(
            kinds.contains(&format!("{k:?}")),
            "no {k:?} among {kinds:?}"
        );
    }
    // A far node's columns, read at once, give each voxel the block a point query gives.
    for level in [2u32, 4, 6] {
        let step = 1i32 << level;
        let side = 16 * step;
        for x in [-3000 / side, 2200 / side, 2500 / side] {
            let cols = loop {
                match g.node_columns(level, x, -1) {
                    Ok(c) => break c,
                    Err(n) => n.tiles.iter().for_each(|&t| {
                        rig.store.load(t).unwrap();
                    }),
                }
            };
            for i in (0..256).step_by(7) {
                let (wx, wz) = (
                    x * side + (i & 15) as i32 * step + step / 2,
                    -side + (i >> 4) as i32 * step + step / 2,
                );
                let s = cols.surfaces[i];
                assert_eq!(s, rig.surface(wx, wz, step));
                for lo in [
                    (s.div_euclid(step) - 1) * step,
                    s.div_euclid(step) * step,
                    (s.div_euclid(step) + 1) * step,
                ] {
                    assert_eq!(
                        g.far_voxel(&cols, i, lo),
                        g.block_at_blocking(wx, lo + step / 2, wz, step),
                        "level {level} column {wx},{wz} voxel {lo}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_far_node_that_lacks_tiles_says_which_and_they_can_be_requested_once() {
    let rig = rig();
    let g = &rig.generator;
    let err = g.node_columns(3, 0, 0).unwrap_err();
    assert!(!err.tiles.is_empty());
    assert_eq!(err.tiles, g.missing_tiles_for_node(3, 0, 0));
    assert_eq!(rig.fetcher.fetch_count(), 0);
    // Requests are listed once, in order, and taken once; only the first wakes the taker.
    let woken = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let w = woken.clone();
    g.set_request_waker(Arc::new(move || {
        w.fetch_add(1, Ordering::SeqCst);
    }));
    g.request_tiles(&err.tiles);
    g.request_tiles(&err.tiles);
    g.request_tiles(&[]);
    assert_eq!(woken.load(Ordering::SeqCst), 1);
    assert_eq!(g.take_requests(), err.tiles);
    assert!(g.take_requests().is_empty());
    // With the tiles in, the same read answers.
    for &t in &err.tiles {
        rig.store.load(t).unwrap();
    }
    assert!(g.node_columns(3, 0, 0).is_ok());
}

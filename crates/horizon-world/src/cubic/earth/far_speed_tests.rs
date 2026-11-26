//! Pins the far view's speed-ups to the plain reads they stand in for: the biome memo against a biome read from
//! the maps, the river cells kept per point against walking the ring of cells, and whole far nodes (columns,
//! trees, snow, biomes, voxels) built cold and warm over patchy synthetic land cover, hills, sea and climate.

use std::sync::Arc;

use horizon_earth::testing::{MemoryClimate, SMALL_CLIMATE_GRID, TempDir};
use horizon_earth::{ClimateFetcher, ClimateStore, ClimateVar};

use super::biome::Biome;
use super::far::NodeColumns;
use super::far_trees::FarDecor;
use super::generator::Access;
use super::land_tests::LandRig;
use super::tests::LON0;
use super::water_gen_tests::river_rig;
use super::*;

const LAT0: f64 = 46.0;
const SNOWY: FarDecor = FarDecor {
    trees: true,
    snow: true,
};

/// Hills and a sea basin: from under the sea level to a few hundred metres, varying over a few hundred blocks.
fn elevation(lat: f64, lon: f64) -> f64 {
    let (a, b) = ((lat - LAT0) * 1400.0, (lon - LON0) * 1000.0);
    60.0 + 90.0 * a.sin() * b.cos() + 70.0 * (a * 2.3 + b * 1.7).sin()
}

/// Patchy classes (trees, shrub, grass, crops, built, bare, snow, water, wetland, mangrove) in cells of a few
/// dozen blocks: many class edges.
fn class(lat: f64, lon: f64) -> u8 {
    const CODES: [u8; 10] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95];
    let (a, b) = (((lat - LAT0) * 3000.0).floor() as i64, ((lon - LON0) * 3000.0).floor() as i64);
    let h = (a.wrapping_mul(73_856_093) ^ b.wrapping_mul(19_349_663)).rem_euclid(1000) as usize;
    CODES[h * h % 7 * 10 / 7 % 10]
}

/// The mixed world with a climate of mean `t` C and `rain` mm (varying a little over the map), and what it takes
/// to load what a read lacks.
fn world(t: f32, rain: u16) -> (LandRig, Arc<EarthGenerator>, TempDir) {
    let mut rig = LandRig::new(EarthSettings::at(LAT0, LON0), elevation, class);
    rig.settings.climate = true;
    let climate = Arc::new(
        MemoryClimate::new()
            .with_raw(ClimateVar::Temperature, 64, 1, move |_, x, y| {
                (((t + (x % 9) as f32 * 0.7 + (y % 5) as f32 * 0.4) + 273.15) * 10.0) as u16
            })
            .with_raw(ClimateVar::Precipitation, 64, 1, move |_, x, y| {
                rain + (x % 7) as u16 * 20 + (y % 3) as u16 * 10
            }),
    );
    let dir = TempDir::new();
    let store = Arc::new(ClimateStore::with_grid(dir.path(), 8, SMALL_CLIMATE_GRID));
    let fetcher: Arc<dyn ClimateFetcher> = climate;
    let source = rig.settings.climate_source(store, fetcher).unwrap();
    let g = Arc::new(
        EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks)
            .unwrap()
            .with_land_cover(rig.source.clone())
            .with_climate(source),
    );
    (rig, g, dir)
}

/// `read` once the tiles, land cover and climate it lacks are loaded, as the downloader does.
fn ready<T>(
    rig: &LandRig,
    g: &EarthGenerator,
    read: impl Fn(&EarthGenerator) -> Result<T, NotReady>,
) -> T {
    for _ in 0..300 {
        match read(g) {
            Ok(value) => return value,
            Err(missing) => {
                for &id in &missing.tiles {
                    rig.store.load(id).unwrap();
                }
                for fetch in g.take_land_requests() {
                    rig.source.satisfy(&fetch).unwrap();
                }
                for fetch in g.take_climate_requests() {
                    g.climate_source().unwrap().satisfy(&fetch).unwrap();
                }
            }
        }
    }
    panic!("a read never became ready");
}

/// The biome of each column of `cols` read from the maps, never the memo (what `column_biomes` was).
fn reference_biomes(g: &EarthGenerator, cols: &NodeColumns) -> Vec<Biome> {
    (0..cols.width * cols.width)
        .map(|i| {
            let (x, z) = cols.block_of(i);
            g.biome_uncached(Access::Local, x >> 2, z >> 2).ok().unwrap()
        })
        .collect()
}

/// Everything a far node holds, as numbers: surfaces, kinds, classes, each voxel and each biome.
fn node_contents(g: &EarthGenerator, cols: &NodeColumns, ny: i32, level: u32) -> Vec<u64> {
    let step = 1i32 << level;
    let mut out: Vec<u64> = cols.surfaces.iter().map(|&s| s as u64).collect();
    for i in 0..cols.width * cols.width {
        out.push(
            format!("{:?}|{:?}", cols.kind(i), cols.class(i))
                .bytes()
                .fold(0u64, |h, b| h.wrapping_mul(131).wrapping_add(u64::from(b))),
        );
        for y in 0..16 {
            out.push(u64::from(g.far_voxel(cols, i, ny * 16 * step + y * step)));
        }
    }
    out
}

#[test]
fn the_biome_memo_gives_what_the_maps_give() {
    let mut seen = std::collections::BTreeSet::new();
    for (t, rain) in [(9.0, 900), (24.0, 120), (-6.0, 400)] {
        let (rig, g, _dir) = world(t, rain);
        // Load the maps under the area first: the memo only keeps answers that were made.
        for stride in [1, 4, 16] {
            ready(&rig, &g, |g| g.biomes_ready(-256, -256, stride));
            ready(&rig, &g, |g| g.biomes_ready(0, 0, stride));
        }
        g.clear_far_memos();
        for qz in (-64..64).step_by(3) {
            for qx in (-64..64).step_by(3) {
                let plain = g.biome_uncached(Access::Local, qx, qz);
                let first = g.biome_core(Access::Local, qx, qz);
                let again = g.biome_core(Access::Local, qx, qz);
                match plain {
                    Ok(biome) => {
                        assert_eq!(first.ok(), Some(biome), "memo miss at {qx},{qz}");
                        assert_eq!(again.ok(), Some(biome), "memo hit at {qx},{qz}");
                        seen.insert(biome as u32);
                    }
                    Err(_) => assert!(first.is_err() && again.is_err(), "answered with no data"),
                }
            }
        }
    }
    assert!(seen.len() >= 4, "the worlds hold too few biomes: {seen:?}");
}

#[test]
fn an_unloaded_column_is_not_remembered() {
    let (rig, g, _dir) = world(9.0, 900);
    // Nothing loaded: every read is a gap, and stays one until the maps are there.
    assert!(g.biome_core(Access::Local, 5, 5).is_err());
    assert!(g.biome_core(Access::Local, 5, 5).is_err());
    ready(&rig, &g, |g| g.biomes_ready(0, 0, 1));
    assert!(g.biome_core(Access::Local, 5, 5).is_ok());
}

#[test]
fn the_river_cells_kept_per_point_give_what_walking_the_ring_gives() {
    let rig = river_rig(40.0);
    let g = &rig.generator;
    ready(&rig, g, |g| g.biomes_ready(-512, -64, 1));
    let mut hits = 0;
    for z in -40..40 {
        for x in (-500..500).step_by(3) {
            let rect = g.point_rect(x, z, 1);
            let walked = EarthGenerator::river_hit(
                &g.rivers_near(rect),
                f64::from(x) + 0.5,
                f64::from(z) + 0.5,
                1.0,
            )
            .filter(|h| h.distance <= h.half_width);
            let kept = g.river_channel_at(x, z);
            assert_eq!(kept, walked, "at {x},{z}");
            hits += usize::from(kept.is_some());
        }
    }
    assert!(hits > 20, "the river was never met ({hits})");
}

#[test]
fn far_nodes_are_the_same_whether_the_memo_is_cold_or_warm() {
    for (t, rain) in [(9.0, 900), (24.0, 120), (-6.0, 400)] {
        let (rig, g, _dir) = world(t, rain);
        for level in [2u32, 3, 4, 6] {
            let step = 1i32 << level;
            for (nx, nz) in [(0, 0), (-1, 0), (0, -1), (1, 1)] {
                // The node at the ground's height (and its neighbours' above and below).
                let ground = ready(&rig, &g, |g| {
                    g.surface_y_blocking(nx * 16 * step + 8 * step, nz * 16 * step + 8 * step)
                        .ok_or(NotReady { tiles: Vec::new() })
                });
                let ny = ground.div_euclid(16 * step);
                for ny in [ny - 1, ny, ny + 1] {
                    let build = |cold: bool| {
                        if cold {
                            g.clear_far_memos();
                        }
                        ready(&rig, &g, |g| {
                            g.biomes_ready(nx * 16 * step + step / 2, nz * 16 * step + step / 2, step)?;
                            g.biomes_ready(nx * 16 * step, nz * 16 * step, step)?;
                            let mut cols = g.node_columns(level, nx, nz)?;
                            g.decorate_columns(&mut cols, SNOWY)?;
                            let biomes = g.column_biomes(&cols)?;
                            Ok((node_contents(g, &cols, ny, level), biomes, reference_biomes(g, &cols)))
                        })
                    };
                    let (cold, cold_biomes, reference) = build(true);
                    let (warm, warm_biomes, _) = build(false);
                    assert_eq!(cold, warm, "level {level} node {nx},{ny},{nz}");
                    assert_eq!(cold_biomes, reference, "level {level} node {nx},{ny},{nz}");
                    assert_eq!(warm_biomes, reference, "level {level} node {nx},{ny},{nz}");
                }
            }
        }
    }
}

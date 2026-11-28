//! Tests of the far view's coarse levels ([`super::coarse`]) over synthetic whole-world elevation and climate:
//! no network, no real data.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use horizon_earth::projection::EARTH_RADIUS_M;
use horizon_earth::testing::{MemoryClimate, SMALL_CLIMATE_GRID, TempDir};
use horizon_earth::{ClimateFetcher, ClimateStore, ClimateVar, LandClass, TileId};

use super::coarse::COARSE_FROM_M;
use super::land_tests::LandRig;
use super::surface::SurfaceKind;
use super::*;

const LAT: f64 = 45.0;
const LON: f64 = 0.0;

/// A whole-world synthetic Earth, continuous everywhere (also across the antimeridian): one continent around
/// longitude 0 rising to 2 km, an ocean 4 km deep around longitude 180, ripples by latitude.
fn world(lat: f64, lon: f64) -> f64 {
    3000.0 * lon.to_radians().cos() - 1000.0 + 200.0 * (lat * 3.0).to_radians().sin()
}

/// A land-cover rig over [`world`] centred on `(lat, lon)` with the coarse path from [`COARSE_FROM_M`].
fn rig_at(lat: f64, lon: f64) -> LandRig {
    let rig = LandRig::new(EarthSettings::at(lat, lon), world, |_, _| 10);
    rig.generator.set_coarse_from_m(COARSE_FROM_M);
    rig
}

/// The block column index of the node at `level` holding the block `(x, z)`.
fn node_at(level: u32, x: i64, z: i64) -> (i32, i32) {
    let size = 16_i64 << level;
    (x.div_euclid(size) as i32, z.div_euclid(size) as i32)
}

/// Every tile id a read named before it answered, loading them as it goes.
fn asked(
    rig: &LandRig,
    read: impl Fn(&EarthGenerator) -> Result<far::NodeColumns, NotReady>,
) -> (far::NodeColumns, BTreeSet<TileId>) {
    let mut ids = BTreeSet::new();
    for _ in 0..50 {
        match read(&rig.generator) {
            Ok(cols) => return (cols, ids),
            Err(missing) => {
                assert!(
                    !missing.tiles.is_empty(),
                    "a coarse read lacking no tiles waits for nothing"
                );
                for &id in &missing.tiles {
                    ids.insert(id);
                    rig.store.load(id).unwrap();
                }
            }
        }
    }
    panic!("a coarse read never became ready");
}

/// The coarse sea mask built from the loaded set.
fn prepared(rig: &LandRig) {
    let missing = rig.generator.coarse_tiles_missing((0, 0));
    for id in missing {
        rig.store.load(id).unwrap();
    }
    rig.generator.prepare_coarse().unwrap();
    assert!(rig.generator.coarse_ready());
}

#[test]
fn the_data_level_follows_the_voxel_size() {
    let rig = rig_at(LAT, LON);
    let g = &rig.generator;
    assert!(!g.is_coarse_step(1));
    assert!(!g.is_coarse_step(2048));
    assert!(g.is_coarse_step(4096));
    assert!(g.is_coarse_step(1 << 22));
    // The zoom falls with the step: a level-20 voxel (1000 km) reads zoom 0.
    let zooms: Vec<u8> = (12..=22).map(|l| g.zoom_for_step(1 << l)).collect();
    assert!(zooms.windows(2).all(|w| w[1] <= w[0]), "{zooms:?}");
    assert!(zooms[0] <= 6, "{zooms:?}");
    assert_eq!(g.zoom_for_step(1 << 20), 0);
    // Without land cover the elevation-only far view stays as it is.
    let mut off = rig.settings.clone();
    off.land_cover = false;
    let plain = EarthGenerator::new(&off, rig.store.clone(), rig.blocks).unwrap();
    assert!(!plain.is_coarse_step(1 << 20));
}

#[test]
fn a_level_20_node_needs_only_a_handful_of_coarse_tiles_and_no_land_cover() {
    let rig = rig_at(LAT, LON);
    prepared(&rig);
    let before = rig.land.fetch_count();
    // A node far out (5,000 km east, 3,000 km south) at level 20, and nodes at every coarse level.
    for level in 12..=22_u32 {
        let (nx, nz) = node_at(level, 5_000_000, 3_000_000);
        let (cols, ids) = asked(&rig, |g| g.node_columns(level, nx, nz));
        let zoom = rig.generator.zoom_for_step(1 << level);
        assert!(ids.iter().all(|id| id.z == zoom), "level {level}: {ids:?}");
        if level >= 20 {
            assert!(ids.len() <= 4, "level {level} asked {} tiles", ids.len());
        }
        assert!(ids.len() <= 16, "level {level} asked {} tiles", ids.len());
        assert!(
            cols.coarse_biome(0).is_some(),
            "level {level} took the coarse path"
        );
        assert!(
            rig.generator
                .biomes_ready(nx << (level + 4), nz << (level + 4), 1 << level)
                .is_ok()
        );
    }
    assert_eq!(
        rig.land.fetch_count(),
        before,
        "the coarse levels read no land cover"
    );
    assert!(rig.generator.take_land_requests().is_empty());
}

#[test]
fn a_coarse_node_without_its_data_is_later_and_never_waits() {
    let rig = rig_at(LAT, LON);
    let (nx, nz) = node_at(16, 2_000_000, 0);
    // The elevation first, then the sea mask's tiles: each time "later", naming what to fetch, at once.
    let started = Instant::now();
    let first = rig.generator.node_columns(16, nx, nz).unwrap_err();
    assert!(!first.tiles.is_empty());
    for &id in &first.tiles {
        rig.store.load(id).unwrap();
    }
    let second = rig.generator.node_columns(16, nx, nz).unwrap_err();
    assert!(
        second.tiles.len() > 100,
        "the mask's tiles: {}",
        second.tiles.len()
    );
    assert!(
        second
            .tiles
            .iter()
            .all(|id| id.z <= horizon_earth::COARSE_PREFETCH_ZOOM)
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(!rig.generator.coarse_ready());
}

/// [`world`] with a climate everywhere of `t` degrees and `p` millimetres.
fn climate_rig(t: f32, p: f32) -> (LandRig, Arc<MemoryClimate>, EarthGenerator, TempDir) {
    let mut rig = rig_at(LAT, LON);
    rig.settings.climate = true;
    let raw_t = ((t + 273.15) * 10.0).round() as u16;
    let raw_p = p.round() as u16;
    let climate = Arc::new(
        MemoryClimate::new()
            .with_raw(ClimateVar::Temperature, 64, 1, move |_, _, _| raw_t)
            .with_raw(ClimateVar::Precipitation, 64, 1, move |_, _, _| raw_p),
    );
    let dir = TempDir::new();
    let store = Arc::new(ClimateStore::with_grid(dir.path(), 8, SMALL_CLIMATE_GRID));
    let fetcher: Arc<dyn ClimateFetcher> = climate.clone();
    let source = rig.settings.climate_source(store, fetcher).unwrap();
    let generator = EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks)
        .unwrap()
        .with_land_cover(rig.source.clone())
        .with_climate(source);
    generator.set_coarse_from_m(COARSE_FROM_M);
    (rig, climate, generator, dir)
}

#[test]
fn a_coarse_node_lacking_climate_is_later_asks_for_it_and_never_fetches() {
    let (rig, climate, g, _dir) = climate_rig(5.0, 900.0);
    prepared(&rig);
    g.prepare_coarse().unwrap();
    let (nx, nz) = node_at(14, 1_000_000, 0);
    let mut answered = None;
    for _ in 0..50 {
        let fetched = climate.fetch_count();
        let read = g.node_columns(14, nx, nz);
        assert_eq!(climate.fetch_count(), fetched, "a far read never fetches");
        match read {
            Ok(cols) => {
                answered = Some(cols);
                break;
            }
            Err(missing) => {
                for &id in &missing.tiles {
                    rig.store.load(id).unwrap();
                }
                let wanted = g.take_climate_requests();
                if missing.tiles.is_empty() {
                    assert!(!wanted.is_empty(), "a node waiting for climate names it");
                }
                let source = g.climate_source().unwrap();
                for fetch in wanted {
                    source.satisfy(&fetch).unwrap();
                }
            }
        }
    }
    let cols = answered.expect("ready once the climate came");
    // 5 C and 900 mm on land is humid temperate: trees.
    let land: Vec<usize> = (0..256)
        .filter(|&i| cols.classes.as_ref().unwrap()[i] != LandClass::Water)
        .collect();
    assert!(!land.is_empty());
    assert!(
        land.iter()
            .all(|&i| cols.classes.as_ref().unwrap()[i] == LandClass::Tree)
    );
    // The whole-world climate set the prefetcher asks for at world load.
    assert!(!g.coarse_climate_needs().is_empty() || climate.fetch_count() > 0);
}

#[test]
fn the_sea_at_coarse_levels_is_water_topped_at_the_sea_level() {
    let rig = rig_at(LAT, LON);
    prepared(&rig);
    let sea = rig.generator.sea_level();
    // Nodes in the ocean around longitude 180 (about 14,000 km west at 45 N) and on the coast at +-90.
    let lap = (std::f64::consts::PI * EARTH_RADIUS_M * LAT.to_radians().cos()) as i64;
    for (level, x) in [
        (14_u32, -lap / 2),
        (16, -lap + 100_000),
        (18, lap / 2),
        (12, -lap + 1_000_000),
    ] {
        let (nx, nz) = node_at(level, x, 0);
        let (cols, _) = asked(&rig, |g| g.node_columns(level, nx, nz));
        let mut water = 0;
        for i in 0..cols.surfaces.len() {
            let (bx, bz) = cols.block_of(i);
            let p = rig
                .generator
                .field()
                .projection()
                .to_lat_lon(f64::from(bx) + 0.5, f64::from(bz) + 0.5);
            if world(p.lat, p.lon) < -300.0 {
                water += 1;
                assert!(
                    matches!(cols.kind(i), SurfaceKind::Water { .. }),
                    "level {level} column {i} ({:.1}, {:.1}) is {:?}",
                    p.lat,
                    p.lon,
                    cols.kind(i)
                );
                assert_eq!(cols.surfaces[i], sea - 1, "level {level} column {i}");
                assert_eq!(cols.classes.as_ref().unwrap()[i], LandClass::Water);
            }
        }
        assert!(water > 0, "level {level} at {x} has sea");
    }
}

#[test]
fn coarse_and_fine_heights_agree_on_average_at_a_mid_level() {
    let rig = rig_at(LAT, LON);
    prepared(&rig);
    // A level-12 node (65 km, 4 km voxels) on the continent's slope, against the fine field read at 256 m.
    let (nx, nz) = node_at(12, 600_000, 200_000);
    let (cols, _) = asked(&rig, |g| g.node_columns(12, nx, nz));
    let coarse: f64 = cols.surfaces.iter().map(|&s| f64::from(s)).sum::<f64>() / 256.0;
    let field = rig.generator.field();
    let zoom = field.zoom_for(256.0);
    let (x0, z0) = (nx << 16, nz << 16);
    let mut sum = 0.0;
    let mut n = 0.0;
    for j in 0..64 {
        for i in 0..64 {
            let (x, z) = (x0 + i * 1024 + 512, z0 + j * 1024 + 512);
            sum += f64::from(field.surface_y_blocking(x, z, zoom).unwrap().unwrap());
            n += 1.0;
        }
    }
    let fine = sum / n;
    let relief = cols.surfaces.iter().max().unwrap() - cols.surfaces.iter().min().unwrap();
    assert!(
        (coarse - fine).abs() <= 10.0 + 0.05 * f64::from(relief),
        "coarse {coarse:.1} fine {fine:.1} relief {relief}"
    );
}

#[test]
fn coarse_columns_over_the_poles_and_the_antimeridian_are_finite_and_continuous() {
    for (lat0, lon0) in [(45.0, 170.0), (80.0, 0.0), (-80.0, 179.0)] {
        let rig = rig_at(lat0, lon0);
        prepared(&rig);
        let r = EARTH_RADIUS_M;
        // North pole, south pole, the antimeridian, and a node a full lap east.
        let to_pole = ((90.0 - lat0).to_radians() * r) as i64;
        let to_south = ((90.0 + lat0).to_radians() * r) as i64;
        let to_180 = ((180.0 - lon0).to_radians() * r * lat0.to_radians().cos()) as i64;
        let lap = (2.0 * std::f64::consts::PI * r * lat0.to_radians().cos()) as i64;
        for (level, x, z) in [
            (14_u32, 0, -to_pole),
            (16, 0, -to_pole - 500_000),
            (14, 0, to_south),
            (14, to_180, 0),
            (18, to_180, 0),
            (15, lap + to_180, 0),
            (21, 0, 0),
            (22, 30_000_000, -20_000_000),
        ] {
            let (nx, nz) = node_at(level, x, z);
            let (cols, _) = asked(&rig, |g| g.node_columns(level, nx, nz));
            let sea = rig.generator.sea_level();
            for (i, &s) in cols.surfaces.iter().enumerate() {
                assert!(
                    (s - sea).abs() < 20_000,
                    "({lat0}, {lon0}) level {level}: column {i} at {s}"
                );
            }
            // Neighbouring columns differ by no more than the synthetic world's own relief between them.
            let step = f64::from(cols.step);
            let most = (4000.0_f64).min(1.0 + step * 3000.0 / (r * 0.3)) + 2.0 * 200.0 + 50.0;
            for i in 0..256 {
                for j in [i + 1, i + 16] {
                    if j < 256 && (j != i + 1 || i % 16 != 15) {
                        let d = f64::from((cols.surfaces[i] - cols.surfaces[j]).abs());
                        let water = matches!(cols.kind(i), SurfaceKind::Water { .. })
                            != matches!(cols.kind(j), SurfaceKind::Water { .. });
                        assert!(
                            water || d <= most.max(4500.0),
                            "({lat0}, {lon0}) level {level}: columns {i}, {j} differ by {d}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_coarse_node_costs_well_under_a_millisecond_once_its_data_is_local() {
    let rig = rig_at(LAT, LON);
    prepared(&rig);
    let mut nodes = Vec::new();
    for level in [12_u32, 14, 16, 18, 20] {
        for k in 0..8_i64 {
            let (nx, nz) = node_at(level, 1_000_000 + k * (16 << level), -500_000);
            asked(&rig, |g| g.node_columns(level, nx, nz));
            nodes.push((level, nx, nz));
        }
    }
    let started = Instant::now();
    for &(level, nx, nz) in &nodes {
        rig.generator.node_columns(level, nx, nz).unwrap();
    }
    let per = started.elapsed().as_secs_f64() * 1000.0 / nodes.len() as f64;
    eprintln!("coarse node: {per:.3} ms (debug build)");
    // A debug build is ten or more times slower than the game's; the release figure is in the notes.
    assert!(per < 20.0, "{per} ms a node");
}

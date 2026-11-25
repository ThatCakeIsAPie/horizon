//! Tests of the generator's water ([`super::water_gen`]) over synthetic elevation, a miniature land cover and a
//! synthetic water atlas: rivers carved from the atlas, a flat lake, the sea's shelf and shores, nothing hanging,
//! single reads agreeing with cubes, and the far view agreeing with the near one.

use std::collections::HashMap;

use horizon_core::position::SectionPos;
use horizon_earth::projection::EARTH_RADIUS_M;
use horizon_earth::{Lake, LatLon, RiverReach, WaterAtlas};

use super::land_tests::LandRig;
use super::surface::SurfaceKind;
use super::tests::LON0;
use super::water::{Bed, SHELF_MAX_BLOCKS, river_geometry};
use super::*;

/// Metres east and north of `(lat0, LON0)` of the point `(lat, lon)`.
fn east_north(lat0: f64, lat: f64, lon: f64) -> (f64, f64) {
    (
        (lon - LON0).to_radians() * EARTH_RADIUS_M * lat0.to_radians().cos(),
        (lat - lat0).to_radians() * EARTH_RADIUS_M,
    )
}

/// A rig at `(lat0, LON0)` (1 m a block, sea level 0) whose generator has the atlas `atlas` makes (it is handed
/// the generator, to place features in block coordinates).
fn water_rig(
    lat0: f64,
    elevation: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
    class: impl Fn(f64, f64) -> u8 + Clone,
    atlas: impl FnOnce(&EarthGenerator) -> WaterAtlas,
) -> LandRig {
    let mut rig = LandRig::new(EarthSettings::at(lat0, LON0), elevation, class);
    let atlas: &'static WaterAtlas = Box::leak(Box::new(atlas(&rig.generator)));
    rig.generator = EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks)
        .unwrap()
        .with_land_cover(rig.source.clone())
        .with_water(atlas);
    rig
}

/// The latitude and longitude of the block point `(x, z)`.
fn at(g: &EarthGenerator, x: f64, z: f64) -> LatLon {
    g.field().projection().to_lat_lon(x, z)
}

/// A reach `id` along `z = 0.5` from block `x = x_up` down to `x_down`, through `n + 1` points, its level
/// falling evenly from `up_m` to `down_m`.
fn east_west_reach(
    g: &EarthGenerator,
    id: u32,
    (x_up, x_down): (f64, f64),
    (up_m, down_m): (f32, f32),
    n: usize,
    q: f32,
) -> RiverReach {
    let points = (0..=n)
        .map(|k| {
            let t = k as f64 / n as f64;
            at(g, x_up + (x_down - x_up) * t, 0.5)
        })
        .collect();
    let levels_m = (0..=n)
        .map(|k| up_m + (down_m - up_m) * (k as f32 / n as f32))
        .collect();
    RiverReach {
        id,
        points,
        levels_m,
        discharge_m3s: q,
        next_down: None,
    }
}

/// Gentle ground rising 1 m every 500 m to the east from 30 m, all grass, with one river of discharge `q`
/// whose level is 1 m under the ground at its ends.
fn river_rig(q: f32) -> LandRig {
    water_rig(
        45.0,
        |lat, lon| {
            let (east, _) = east_north(45.0, lat, lon);
            30.0 + east / 500.0
        },
        |_, _| 30,
        move |g| {
            WaterAtlas::from_parts(
                vec![east_west_reach(g, 7, (400.0, -400.0), (29.8, 28.2), 8, q)],
                Vec::new(),
            )
        },
    )
}

/// The cubes of an area, generated once each, read block by block.
struct Blocks<'a> {
    g: &'a EarthGenerator,
    cubes: HashMap<SectionPos, Box<[u32; 4096]>>,
}

impl<'a> Blocks<'a> {
    fn new(g: &'a EarthGenerator) -> Self {
        Self {
            g,
            cubes: HashMap::new(),
        }
    }

    fn block(&mut self, x: i32, y: i32, z: i32) -> u32 {
        let c = SectionPos::new(x >> 4, y >> 4, z >> 4);
        let g = self.g;
        let cube = self.cubes.entry(c).or_insert_with(|| {
            let mut out = Box::new([0; 4096]);
            g.generate_blocking(c, &mut out);
            out
        });
        cube[((y & 15) as usize) << 8 | ((z & 15) as usize) << 4 | (x & 15) as usize]
    }

    /// The highest block of column `(x, z)` from `y_hi` down that is not air: its height and block.
    fn top(&mut self, x: i32, z: i32, y_hi: i32) -> (i32, u32) {
        let air = self.g.blocks().air;
        let mut y = y_hi;
        loop {
            let b = self.block(x, y, z);
            if b != air {
                return (y, b);
            }
            y -= 1;
            assert!(y > y_hi - 200, "column {x},{z} is air all the way down");
        }
    }
}

/// Checks that no water block in the columns `x0..=x1` by `z0..=z1`, heights `y0..=y1`, hangs: the block under
/// it is not air, and a horizontal neighbour that is air at its height belongs to a water column with a lower
/// top (a still-water step). Gives the number of water blocks seen.
fn assert_nothing_hangs(
    b: &mut Blocks,
    [x0, x1]: [i32; 2],
    [z0, z1]: [i32; 2],
    [y0, y1]: [i32; 2],
) -> usize {
    let (air, water) = (b.g.blocks().air, b.g.blocks().water);
    let mut seen = 0;
    for z in z0..=z1 {
        for x in x0..=x1 {
            for y in y0..=y1 {
                if b.block(x, y, z) != water {
                    continue;
                }
                seen += 1;
                assert_ne!(b.block(x, y - 1, z), air, "water at {x},{y},{z} over air");
                for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, nz) = (x + dx, z + dz);
                    if b.block(nx, y, nz) != air {
                        continue;
                    }
                    let (_, top) = b.top(nx, nz, y);
                    assert_eq!(
                        top, water,
                        "water at {x},{y},{z} beside air over a dry column {nx},{nz}"
                    );
                }
            }
        }
    }
    seen
}

#[test]
fn a_river_from_the_atlas_runs_in_every_column_of_its_line_and_never_rises_downstream() {
    let rig = river_rig(20.0);
    let g = &rig.generator;
    let mut b = Blocks::new(g);
    let water = rig.blocks.water;
    let mut last = i32::MAX;
    // From upstream (east) to downstream (west), every column on the line is water, its top never rising.
    for x in (-380..=380).rev().step_by(5) {
        let (top, block) = b.top(x, 0, 60);
        assert_eq!(block, water, "column {x},0 on the river's line");
        assert!(
            top <= last,
            "the top rises downstream at x {x}: {top} after {last}"
        );
        // The water is under the ground beside it: the river is carved, not perched.
        let (bank, _) = b.top(x, 40, 60);
        assert!(
            top < bank,
            "the river at {x} ({top}) is not under its bank ({bank})"
        );
        last = top;
    }
    // The banks blend: from the channel's edge outwards the ground never steps more than 2 blocks at a time.
    let hw = river_geometry(20.0, 1.0, 1.0).half_width.ceil() as i32;
    for x in [-200, 0, 200] {
        let mut prev = b.top(x, hw + 1, 60).0;
        for z in hw + 2..hw + 24 {
            let (s, _) = b.top(x, z, 60);
            assert!(
                (s - prev).abs() <= 2,
                "a bank step at {x},{z}: {prev} to {s}"
            );
            prev = s;
        }
    }
}

#[test]
fn a_river_with_more_discharge_is_wider() {
    let width = |q: f32| {
        let rig = river_rig(q);
        let cols = rig.loaded(|g| g.columns_wide(0, -64, 128, 1, 1));
        (0..128)
            .filter(|&row| cols.kind(row * 128).is_water())
            .count()
    };
    let (small, big) = (width(5.0), width(200.0));
    assert!(
        small >= 2,
        "a small river is at least 2 columns wide: {small}"
    );
    assert!(
        big > small + 10,
        "a big river is wider: {big} against {small}"
    );
}

#[test]
fn nothing_hangs_along_a_river_and_a_single_block_read_is_the_cubes_block() {
    let rig = river_rig(50.0);
    let g = &rig.generator;
    let mut b = Blocks::new(g);
    let seen = assert_nothing_hangs(&mut b, [-40, 40], [-36, 36], [10, 40]);
    assert!(seen > 500, "the area has a river: {seen} water blocks");
    for (x, z) in [(-30, 0), (0, 3), (17, -10), (33, 20), (5, 30)] {
        let (top, _) = b.top(x, z, 60);
        for y in top - 4..=top + 1 {
            assert_eq!(
                g.block_at_blocking(x, y, z, 1),
                b.block(x, y, z),
                "block_at and the cube differ at {x},{y},{z}"
            );
        }
    }
}

#[test]
fn a_river_shows_in_the_far_view_where_the_near_cubes_have_it() {
    let rig = river_rig(200.0);
    let g = &rig.generator;
    let mut b = Blocks::new(g);
    let hw = river_geometry(200.0, 1.0, 1.0).half_width;
    for level in [2u32, 4] {
        let cols = rig.loaded(|g| g.node_columns(level, 0, -1));
        let cols_south = rig.loaded(|g| g.node_columns(level, 0, 0));
        let mut found = 0;
        for c in [&cols, &cols_south] {
            for i in 0..256 {
                let (x, z) = c.block_of(i);
                if f64::from(z).abs() <= hw - 1.0 {
                    found += 1;
                    assert!(c.kind(i).is_water(), "far column {x},{z} at level {level}");
                    let (top, block) = b.top(x, z, 60);
                    assert_eq!(block, rig.blocks.water, "near column {x},{z}");
                    assert_eq!(c.surfaces[i], top, "near and far tops at {x},{z}");
                }
            }
        }
        assert!(found > 0, "the far nodes at level {level} cross the river");
    }
}

/// Ground at 100 m with ripples, one land cover pixel of water (about 870 m) north east of the origin, and the
/// atlas lake over it at 99.5 m.
fn lake_rig(with_record: bool) -> LandRig {
    let pixel = 1.0 / 128.0;
    water_rig(
        45.0,
        |lat, lon| {
            let (east, north) = east_north(45.0, lat, lon);
            100.0 + 2.0 * (east / 41.0).sin() * (north / 29.0).cos()
        },
        move |lat, lon| {
            if (45.0..45.0 + pixel).contains(&lat) && (7.0..7.0 + pixel).contains(&lon) {
                80
            } else {
                30
            }
        },
        move |_| {
            let lakes = if with_record {
                vec![Lake {
                    id: 3,
                    south: 45.0,
                    west: 7.0,
                    north: 45.0 + pixel,
                    east: 7.0 + pixel,
                    level_m: 99.5,
                    area_km2: 0.5,
                    depth_avg_m: 6.0,
                    reservoir: false,
                }]
            } else {
                Vec::new()
            };
            WaterAtlas::from_parts(Vec::new(), lakes)
        },
    )
}

/// The block `(x, z)` of a point a fraction `(fx, fn)` across the lake's pixel from its south west corner.
fn in_lake(g: &EarthGenerator, fx: f64, fnorth: f64) -> (i32, i32) {
    let pixel = 1.0 / 128.0;
    let (x, z) = g
        .field()
        .projection()
        .from_lat_lon(45.0 + pixel * fnorth, 7.0 + pixel * fx);
    (x.floor() as i32, z.floor() as i32)
}

#[test]
fn a_lake_is_flat_at_its_recorded_level_near_and_far() {
    let rig = lake_rig(true);
    let g = &rig.generator;
    let want = g.field().vertical().block_y(99.5) - 1;
    let mut tops = Vec::new();
    for (fx, fnorth) in [(0.5, 0.5), (0.2, 0.3), (0.8, 0.7), (0.5, 0.9)] {
        let (x, z) = in_lake(g, fx, fnorth);
        let cols = rig.loaded(|g| g.columns_wide(x - 8, z - 8, 16, 1, 1));
        for i in 0..256 {
            if let SurfaceKind::Water { depth, .. } = cols.kind(i) {
                assert!(depth >= 1);
                tops.push(cols.surfaces[i]);
            }
        }
        // Far nodes at steps 4 and 16.
        for level in [2u32, 4] {
            let (nx, nz) = (x >> (4 + level), z >> (4 + level));
            let far = rig.loaded(|g| g.node_columns(level, nx, nz));
            for i in 0..256 {
                if far.kind(i).is_water() {
                    tops.push(far.surfaces[i]);
                }
            }
        }
    }
    assert!(tops.len() > 500, "the lake has water: {}", tops.len());
    assert!(
        tops.iter().all(|&t| t == want),
        "every water column of the lake is at {want}"
    );
}

#[test]
fn a_lake_without_a_record_is_still_flat_where_its_window_sees_one_level() {
    let rig = lake_rig(false);
    let g = &rig.generator;
    let (x, z) = in_lake(g, 0.5, 0.5);
    let cols = rig.loaded(|g| g.columns_wide(x - 8, z - 8, 16, 1, 1));
    let tops: Vec<i32> = (0..256)
        .filter(|&i| cols.kind(i).is_water())
        .map(|i| cols.surfaces[i])
        .collect();
    assert_eq!(tops.len(), 256, "the middle of the lake is all water");
    let lowest = *tops.iter().min().unwrap();
    // Without the record the level is the lowest surface the column sees, so it varies only by the ripples.
    assert!(tops.iter().all(|&t| t - lowest <= 4));
}

#[test]
fn nothing_hangs_at_a_lakes_shore() {
    let rig = lake_rig(true);
    let g = &rig.generator;
    // The lake's south west corner, with land on two sides.
    let (x, z) = in_lake(g, 0.0, 0.0);
    let mut b = Blocks::new(g);
    let seen = assert_nothing_hangs(&mut b, [x - 20, x + 20], [z - 20, z + 20], [85, 105]);
    assert!(seen > 100, "the corner has water: {seen}");
}

/// A coast along `lon = 7` (the origin): water class to the west over 1 m of elevation (as real coastal water
/// often reads), grass to the east: a beach of `beach` metres at 1.5 m, then rising by `slope` metres a metre.
fn coast_rig(lat0: f64, beach: f64, slope: f64) -> LandRig {
    water_rig(
        lat0,
        move |lat, lon| {
            let (east, _) = east_north(lat0, lat, lon);
            if east < 0.0 {
                1.0
            } else {
                1.5 + (east - beach).max(0.0) * slope
            }
        },
        |_, lon| if lon < 7.0 { 80 } else { 30 },
        |_| WaterAtlas::from_parts(Vec::new(), Vec::new()),
    )
}

#[test]
fn the_sea_has_its_top_under_the_sea_level_a_shelf_and_a_sand_or_gravel_shore() {
    for (lat0, shore) in [(45.0, "sand"), (55.0, "gravel")] {
        let rig = coast_rig(lat0, 0.0, 0.02);
        let g = &rig.generator;
        let cols = rig.loaded(|g| g.columns_wide(-32, 0, 48, 1, 1));
        let row = 20 * 48;
        // The first dry column from the west.
        let coast = (0..48)
            .find(|&i| !cols.kind(row + i).is_water())
            .expect("the row reaches land");
        assert!(coast > 20, "the sea comes up to the coast: {coast}");
        let mut last_depth = u32::MAX;
        for i in 0..coast {
            let SurfaceKind::Water { depth, .. } = cols.kind(row + i) else {
                panic!("column {i} west of the coast is water");
            };
            assert_eq!(cols.surfaces[row + i], g.sea_level() - 1, "the sea's top");
            assert!(depth <= last_depth, "the shelf deepens away from land");
            last_depth = depth;
        }
        let deep = match cols.kind(row) {
            SurfaceKind::Water { depth, .. } => depth,
            _ => unreachable!(),
        };
        assert_eq!(
            deep, SHELF_MAX_BLOCKS as u32,
            "away from land the shelf is at its deepest"
        );
        let next = match cols.kind(row + coast - 1) {
            SurfaceKind::Water { depth, .. } => depth,
            _ => unreachable!(),
        };
        assert!(next < deep, "shallow by the shore");
        let want = if shore == "sand" {
            SurfaceKind::Dune
        } else {
            SurfaceKind::Gravel
        };
        assert_eq!(cols.kind(row + coast), want, "a {shore} shore at {lat0}");
        // The bed of a cold sea is gravel too.
        let bed = if shore == "sand" {
            Bed::Sand
        } else {
            Bed::Gravel
        };
        assert!(matches!(cols.kind(row), SurfaceKind::Water { bed: b, .. } if b == bed));
        let mut b = Blocks::new(g);
        let seen = assert_nothing_hangs(&mut b, [-24, 8], [4, 28], [-12, 6]);
        assert!(seen > 100);
    }
}

#[test]
fn a_steep_shore_is_stone() {
    // A narrow beach under a cliff: low enough to be shore, steep by its slope.
    let rig = coast_rig(45.0, 10.0, 1.5);
    let cols = rig.loaded(|g| g.columns_wide(-32, 0, 48, 1, 1));
    let row = 20 * 48;
    let coast = (0..48)
        .find(|&i| !cols.kind(row + i).is_water())
        .expect("the row reaches land");
    assert_eq!(cols.kind(row + coast), SurfaceKind::Alpine);
}

#[test]
fn a_river_channel_is_a_river_biome_and_grows_water_plants_only() {
    use crate::cubic::custom::vegetation::VegetationMask;
    let rig = river_rig(50.0);
    let g = &rig.generator;
    assert_eq!(g.biome_at_quart_blocking(0, 0), Biome::River);
    assert_ne!(g.biome_at_quart_blocking(0, 30), Biome::River);
    let shared = std::sync::Arc::new(
        EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks)
            .unwrap()
            .with_land_cover(rig.source.clone())
            .with_water(g.water_atlas().unwrap()),
    );
    let mask = EarthVegetationMask::new(shared);
    assert!(!mask.allows(3, 2, "oak"));
    assert!(mask.allows(3, 2, "seagrass_simple"));
    assert!(!mask.allows(3, 120, "seagrass_simple"));
}

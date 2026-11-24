//! Tests of the pure water rules over synthetic data: no generator, no tiles, plain function calls.

use std::sync::Arc;

use horizon_earth::{Lake, LandClass, LatLon, RiverReach, VerticalMap};

use super::water::*;

/// The sea level block of the tests' world (1 m a block both ways).
const SEA: i32 = 64;

fn vertical() -> VerticalMap {
    VerticalMap::new(SEA, 1.0)
}

/// A reach through the block points `(x, z)` (stored as longitude and latitude, so the test projection is the
/// identity), with the levels in metres.
fn reach(id: u32, points: &[(f64, f64)], levels: &[f32], q: f32) -> Arc<RiverReach> {
    Arc::new(RiverReach {
        id,
        points: points
            .iter()
            .map(|&(x, z)| LatLon { lat: z, lon: x })
            .collect(),
        levels_m: levels.to_vec(),
        discharge_m3s: q,
        next_down: None,
    })
}

fn project(p: LatLon) -> (f64, f64) {
    (p.lon, p.lat)
}

fn index(reaches: &[Arc<RiverReach>]) -> RiverIndex {
    RiverIndex::build(reaches, project, 1.0, 1.0)
}

fn lake(id: u32, south: f64, north: f64, level_m: f32) -> Arc<Lake> {
    Arc::new(Lake {
        id,
        south,
        west: 20.0,
        north,
        east: 20.1,
        level_m,
        area_km2: 1.0,
        depth_avg_m: 5.0,
        reservoir: false,
    })
}

// ---- hydraulic geometry, tops ----

#[test]
fn rivers_get_wider_and_deeper_with_discharge_and_never_under_the_minimums() {
    let mut last = river_geometry(0.0, 1.0, 1.0);
    // The minimum river: 2 m wide is a half width of 1 block at 1 m a block, and one block deep.
    assert_eq!(last.half_width, 1.0);
    assert_eq!(last.depth, 1.0);
    for q in [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0] {
        let g = river_geometry(q, 1.0, 1.0);
        assert!(g.half_width > last.half_width, "q {q}");
        assert!(g.depth >= last.depth, "q {q}");
        last = g;
    }
    // Andreadis: 7.2 * sqrt(100) = 72 m wide, 0.27 * 100^0.39 = 1.6 m deep.
    let g = river_geometry(100.0, 1.0, 1.0);
    assert!((g.half_width - 36.0).abs() < 1e-9);
    assert!((g.depth - 0.27 * 100f64.powf(0.39)).abs() < 1e-9);
    // Coarser blocks make a narrower river in blocks, but never under a one block line; taller blocks per
    // metre make it deeper.
    assert_eq!(river_geometry(1.0, 10.0, 1.0).half_width, 0.5);
    assert!(
        (river_geometry(1000.0, 10.0, 1.0).half_width - 0.5 * 7.2 * 1000f64.sqrt() / 10.0).abs()
            < 1e-9
    );
    assert!(river_geometry(1000.0, 1.0, 2.0).depth > 1.9 * river_geometry(1000.0, 1.0, 1.0).depth);
    // Nonsense discharge is the minimum river.
    assert_eq!(
        river_geometry(f64::NAN, 1.0, 1.0),
        river_geometry(0.0, 1.0, 1.0)
    );
}

#[test]
fn the_top_of_water_is_the_block_of_the_level_less_one_and_rivers_stop_at_the_sea() {
    let v = vertical();
    assert_eq!(water_top(0.0, &v), SEA - 1);
    assert_eq!(water_top(100.5, &v), SEA + 100 - 1);
    assert_eq!(water_top(-430.0, &v), SEA - 431);
    assert_eq!(river_top(-5.0, &v), SEA - 1);
    assert_eq!(river_top(12.0, &v), SEA + 11);
    // A scale of 2 m a block with exaggeration 1: 100 m is 50 blocks.
    let coarse = VerticalMap::new(SEA, 2.0);
    assert_eq!(water_top(100.0, &coarse), SEA + 49);
    assert!((vertical_blocks_per_metre(&coarse) - 0.5).abs() < 1e-12);
}

#[test]
fn cold_shores_follow_the_temperature_or_the_latitude() {
    assert!(shore_is_cold(Some(1.0), 0.0));
    assert!(!shore_is_cold(Some(12.0), 70.0));
    assert!(shore_is_cold(None, -60.0));
    assert!(!shore_is_cold(None, 35.0));
}

#[test]
fn the_index_and_its_answers_can_cross_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<RiverIndex>();
    send_sync::<RiverHit>();
    send_sync::<LakeLevel>();
}

// ---- the river index ----

#[test]
fn nearest_on_a_straight_reach_gives_distance_and_interpolated_level() {
    let r = reach(7, &[(0.0, 0.0), (1000.0, 0.0)], &[100.0, 90.0], 100.0);
    let idx = index(&[r]);
    let hit = idx.nearest(500.0, 10.0, 1.0).unwrap();
    assert_eq!(hit.reach_id, 7);
    assert!((hit.distance - 10.0).abs() < 1e-9);
    assert!((hit.level_m - 95.0).abs() < 1e-9);
    assert!((hit.half_width - 36.0).abs() < 1e-9);
    assert!((hit.corridor - 36.0).abs() < 1e-9);
    // Either side, past the ends (the level of the end point).
    assert!((idx.nearest(500.0, -10.0, 1.0).unwrap().distance - 10.0).abs() < 1e-9);
    assert!((idx.nearest(-5.0, 0.0, 1.0).unwrap().level_m - 100.0).abs() < 1e-9);
    assert!((idx.nearest(1005.0, 0.0, 1.0).unwrap().level_m - 90.0).abs() < 1e-9);
    // The level never rises downstream along the line.
    let mut last = f64::MAX;
    for x in (0..=1000).step_by(25) {
        let level = idx.nearest(f64::from(x), 3.0, 1.0).unwrap().level_m;
        assert!(level <= last + 1e-9, "x {x}");
        last = level;
    }
    // Influence: half width 36, corridor 36, bank 16.
    assert!(idx.nearest(500.0, 87.0, 1.0).is_some());
    assert!(idx.nearest(500.0, 89.0, 1.0).is_none());
    assert!(idx.max_influence() >= 88.0);
}

#[test]
fn nearest_on_a_bent_reach_follows_the_nearest_segment() {
    let r = reach(
        1,
        &[(0.0, 0.0), (500.0, 0.0), (500.0, 500.0)],
        &[100.0, 95.0, 90.0],
        100.0,
    );
    let idx = index(&[r]);
    // Beside the second segment, half way down it.
    let hit = idx.nearest(530.0, 250.0, 1.0).unwrap();
    assert!((hit.distance - 30.0).abs() < 1e-9);
    assert!((hit.level_m - 92.5).abs() < 1e-9);
    // Outside the bend: the corner is nearest.
    let hit = idx.nearest(520.0, -20.0, 1.0).unwrap();
    assert!((hit.distance - 800f64.sqrt()).abs() < 1e-9);
    assert!((hit.level_m - 95.0).abs() < 1e-9);
    // Inside the bend: 20 from both legs.
    assert!((idx.nearest(480.0, 20.0, 1.0).unwrap().distance - 20.0).abs() < 1e-9);
    // Levels along the whole line are monotone downstream.
    let mut last = f64::MAX;
    for i in 0..=100 {
        let (x, z) = if i <= 50 {
            (f64::from(i) * 10.0, 0.0)
        } else {
            (500.0, f64::from(i - 50) * 10.0)
        };
        let level = idx.nearest(x + 1.0, z + 1.0, 1.0).unwrap().level_m;
        assert!(level <= last + 1e-9, "i {i}: {level} after {last}");
        last = level;
    }
}

#[test]
fn the_larger_river_wins_at_a_confluence() {
    let main = reach(2, &[(0.0, 0.0), (1000.0, 0.0)], &[50.0, 40.0], 1000.0);
    let trib = reach(1, &[(500.0, 400.0), (500.0, 5.0)], &[60.0, 45.0], 10.0);
    let idx = index(&[trib, main]);
    // Near the join: closer to the tributary's line, but inside the main river's wide channel.
    assert_eq!(idx.nearest(500.0, 20.0, 1.0).unwrap().reach_id, 2);
    // Far up the tributary the main river is out of reach.
    assert_eq!(idx.nearest(500.0, 350.0, 1.0).unwrap().reach_id, 1);
    // Out of the main river's channel, beside the tributary: the tributary.
    assert_eq!(idx.nearest(505.0, 250.0, 1.0).unwrap().reach_id, 1);
    // Ties go to the lower id: two reaches with the same line.
    let a = reach(5, &[(0.0, 0.0), (100.0, 0.0)], &[10.0, 9.0], 10.0);
    let b = reach(4, &[(0.0, 0.0), (100.0, 0.0)], &[10.0, 9.0], 10.0);
    assert_eq!(index(&[a, b]).nearest(50.0, 1.0, 1.0).unwrap().reach_id, 4);
}

#[test]
fn a_far_view_widens_a_river_to_one_voxel_column() {
    let stream = reach(1, &[(0.0, 0.0), (1000.0, 0.0)], &[100.0, 90.0], 1.0);
    let idx = index(&[stream]);
    // Q = 1: half width 3.6 blocks. Near, it stays as the data says.
    assert!((idx.nearest(500.0, 6.0, 1.0).unwrap().half_width - 3.6).abs() < 1e-9);
    // At a step of 16, 2 hw = 7.2 >= 4: half a step either side.
    let far = idx.nearest(500.0, 6.0, 16.0).unwrap();
    assert!((far.half_width - 8.0).abs() < 1e-9);
    assert!(far.distance <= far.half_width);
    // A creek under a quarter of a sample wide is left out of the far view's widening.
    let creek = reach(2, &[(0.0, 0.0), (1000.0, 0.0)], &[100.0, 90.0], 0.0);
    let idx = index(&[creek]);
    assert!((idx.nearest(500.0, 6.0, 16.0).unwrap().half_width - 1.0).abs() < 1e-9);
    assert!((idx.nearest(500.0, 6.0, 4.0).unwrap().half_width - 2.0).abs() < 1e-9);
    // The widening is by the step asked, so near and far queries on one index differ only in width.
    assert!(effective_half_width(3.6, 1.0) == 3.6);
}

#[test]
fn a_big_river_is_found_far_from_its_line_and_the_index_says_how_far() {
    // Q = 100000: 2277 m wide, a half width of 1138 blocks.
    let big = reach(1, &[(0.0, 0.0), (5000.0, 0.0)], &[10.0, 5.0], 100000.0);
    let idx = index(&[big]);
    assert!(idx.max_influence() > 2.0 * 1138.0);
    let hit = idx.nearest(2500.0, 1500.0, 1.0).unwrap();
    assert!(hit.distance > hit.half_width);
    assert!(idx.nearest(2500.0, 1138.0 * 2.0 + 10.0, 1.0).is_some());
    assert!(
        idx.nearest(2500.0, idx.max_influence() + 100.0, 1.0)
            .is_none()
    );
    assert!(RiverIndex::build(&[], project, 1.0, 1.0).is_empty());
}

// ---- lakes ----

#[test]
fn pick_lake_takes_the_closest_level_inside_the_grown_box_within_the_tolerance() {
    let lakes = vec![lake(1, 10.0, 10.1, 100.0), lake(2, 10.0, 10.1, 120.0)];
    // Inside, the closer level wins.
    assert_eq!(pick_lake(&lakes, 10.05, 20.05, 105.0).unwrap().id, 1);
    assert_eq!(pick_lake(&lakes, 10.05, 20.05, 115.0).unwrap().id, 2);
    // A margin of 0.002 degrees around the box.
    assert!(pick_lake(&lakes, 10.1015, 20.05, 100.0).is_some());
    assert!(pick_lake(&lakes, 10.103, 20.05, 100.0).is_none());
    assert!(pick_lake(&lakes, 10.05, 19.9985, 100.0).is_some());
    assert!(pick_lake(&lakes, 10.05, 19.997, 100.0).is_none());
    // The tolerance: 20 m.
    assert!(pick_lake(&lakes, 10.05, 20.05, 99.0 - 25.0).is_none());
    assert_eq!(pick_lake(&lakes, 10.05, 20.05, 80.5).unwrap().id, 1);
    assert!(pick_lake(&lakes, 10.05, 20.05, 150.0).is_none());
    // A tie in level goes to the lower id.
    let tied = vec![lake(9, 10.0, 10.1, 100.0), lake(3, 10.0, 10.1, 100.0)];
    assert_eq!(pick_lake(&tied, 10.05, 20.05, 101.0).unwrap().id, 3);
    assert!(pick_lake(&[], 10.05, 20.05, 100.0).is_none());
}

#[test]
fn a_lake_level_has_the_top_under_its_level_and_a_depth_from_its_mean_depth() {
    let v = vertical();
    let mut l = (*lake(1, 0.0, 1.0, 100.0)).clone();
    assert_eq!(
        LakeLevel::from_lake(&l, &v, 1.0),
        LakeLevel {
            top: SEA + 99,
            max_depth: 5
        }
    );
    l.depth_avg_m = 0.5;
    assert_eq!(
        LakeLevel::from_lake(&l, &v, 1.0).max_depth,
        LAKE_MIN_DEPTH_BLOCKS
    );
    l.depth_avg_m = 300.0;
    assert_eq!(
        LakeLevel::from_lake(&l, &v, 1.0).max_depth,
        LAKE_MAX_DEPTH_BLOCKS
    );
    l.depth_avg_m = 5.0;
    assert_eq!(LakeLevel::from_lake(&l, &v, 0.5).max_depth, 3);
    assert_eq!(LakeLevel::fallback(70).max_depth, FALLBACK_LAKE_DEPTH);
}

// ---- the window pass ----

/// One sample of a synthetic world.
#[derive(Clone, Copy)]
struct Sample {
    dem: i32,
    class: LandClass,
    lake: Option<LakeLevel>,
}

fn land(dem: i32) -> Sample {
    Sample {
        dem,
        class: LandClass::Grass,
        lake: None,
    }
}

fn water(dem: i32) -> Sample {
    Sample {
        dem,
        class: LandClass::Water,
        lake: None,
    }
}

/// The water of the `size` columns (`x`, `z`) whose first is at the block `origin`, `stride` apart, in a world
/// given by `sample(x, z)` and `rivers`.
fn run(
    rivers: Option<&RiverIndex>,
    sample: &dyn Fn(i64, i64) -> Sample,
    origin: (i64, i64),
    size: (usize, usize),
    stride: i32,
) -> Vec<ColumnWater> {
    let (w, h) = (size.0 + 2 * WINDOW_PAD, size.1 + 2 * WINDOW_PAD);
    let (mut dem, mut class, mut river, mut lake) = (vec![], vec![], vec![], vec![]);
    for j in 0..h {
        for i in 0..w {
            let x = origin.0 + (i as i64 - WINDOW_PAD as i64) * i64::from(stride);
            let z = origin.1 + (j as i64 - WINDOW_PAD as i64) * i64::from(stride);
            let s = sample(x, z);
            dem.push(s.dem);
            class.push(s.class);
            lake.push(s.lake);
            river.push(rivers.and_then(|r| r.nearest(x as f64, z as f64, f64::from(stride))));
        }
    }
    water_window(&WindowInput {
        width: w,
        rows: h,
        stride,
        vertical: vertical(),
        dem: &dem,
        class: &class,
        river: &river,
        lake: &lake,
    })
}

#[test]
fn a_window_without_inner_columns_is_empty() {
    let s = |_: i64, _: i64| land(70);
    assert!(run(None, &s, (0, 0), (0, 5), 1).is_empty());
    assert_eq!(run(None, &s, (0, 0), (3, 2), 1).len(), 6);
}

#[test]
fn dry_land_is_left_as_it_is() {
    let s = |x: i64, z: i64| land(70 + ((x * 3 + z) % 5) as i32);
    for (i, c) in run(None, &s, (0, 0), (10, 10), 1).iter().enumerate() {
        let (x, z) = ((i % 10) as i64, (i / 10) as i64);
        assert_eq!(c.s, s(x, z).dem);
        assert!(c.water.is_none() && !c.shore && !c.channel);
    }
}

#[test]
fn a_sea_strip_next_to_land_has_the_sea_top_a_shelf_and_a_shore() {
    // Sea at z < 0 with the elevation of a real coast (0 to 1 m); land at z >= 0 one metre up.
    let s = |_: i64, z: i64| if z < 0 { water(SEA) } else { land(SEA + 1) };
    let origin = (0, -14);
    let out = run(None, &s, origin, (4, 20), 1);
    let at = |z: i64| out[((z - origin.1) as usize) * 4 + 1];
    let mut last = 0;
    for z in (-14..0).rev() {
        let c = at(z);
        let (depth, bed) = c.water.expect("sea");
        assert_eq!(c.s, SEA - 1, "z {z}");
        assert_eq!(c.of, Some(WaterOf::Sea));
        assert_eq!(bed, Bed::Sand);
        assert!(depth >= last, "the shelf only deepens: z {z}");
        assert!(depth <= SHELF_MAX_BLOCKS as u32);
        last = depth;
    }
    assert_eq!(at(-1).water.unwrap().0, 1);
    assert_eq!(at(-3).water.unwrap().0, 3);
    assert_eq!(at(-8).water.unwrap().0, SHELF_MAX_BLOCKS as u32);
    assert_eq!(at(-14).water.unwrap().0, SHELF_MAX_BLOCKS as u32);
    // The shore: dry columns within 3 blocks of the sea and at most 2 over its top.
    for z in 0..=2 {
        assert!(at(z).shore && at(z).water.is_none(), "z {z}");
        assert_eq!(at(z).s, SEA + 1);
    }
    assert!(!at(3).shore);
    // Land a little higher than the shore height is no shore.
    let high = |_: i64, z: i64| if z < 0 { water(SEA) } else { land(SEA + 4) };
    let out = run(None, &high, origin, (4, 20), 1);
    assert!(!out[(14 + 1) * 4 + 1].shore);
}

#[test]
fn a_deep_sea_keeps_its_deep_bed() {
    let s = |_: i64, z: i64| {
        if z < 0 {
            water(SEA - 300)
        } else {
            land(SEA + 1)
        }
    };
    let out = run(None, &s, (0, -12), (3, 16), 1);
    let c = out[3 + 1];
    assert_eq!(c.s, SEA - 1);
    assert_eq!(c.water.unwrap().0, 299);
    // Open ocean with no data at all.
    let open = |_: i64, _: i64| Sample {
        dem: SEA - 4000,
        class: LandClass::NoData,
        lake: None,
    };
    let out = run(None, &open, (0, 0), (3, 3), 1);
    assert!(
        out.iter()
            .all(|c| c.s == SEA - 1 && c.of == Some(WaterOf::Sea))
    );
    // No data on high ground is dry.
    let dry = |_: i64, _: i64| Sample {
        dem: SEA + 50,
        class: LandClass::NoData,
        lake: None,
    };
    assert!(
        run(None, &dry, (0, 0), (3, 3), 1)
            .iter()
            .all(|c| c.water.is_none())
    );
}

#[test]
fn a_lake_with_a_record_is_flat_at_its_level_and_deeper_in_the_middle() {
    let level = LakeLevel {
        top: SEA + 99,
        max_depth: 6,
    };
    let in_lake = |x: i64, z: i64| x * x + z * z <= 100;
    let s = move |x: i64, z: i64| {
        if in_lake(x, z) {
            Sample {
                lake: Some(level),
                ..water(SEA + 100)
            }
        } else {
            land(SEA + 101)
        }
    };
    let origin = (-14, -14);
    let out = run(None, &s, origin, (29, 29), 1);
    let at = |x: i64, z: i64| out[((z - origin.1) * 29 + (x - origin.0)) as usize];
    for z in -14..=14 {
        for x in -14..=14 {
            let c = at(x, z);
            if in_lake(x, z) {
                assert_eq!(c.s, level.top, "({x},{z}) is flat at the lake's level");
                assert_eq!(c.of, Some(WaterOf::Lake));
                assert!(c.water.unwrap().0 >= 1);
            } else {
                assert!(c.water.is_none());
            }
        }
    }
    assert_eq!(at(0, 0).water.unwrap().0, 6);
    assert_eq!(at(10, 0).water.unwrap().0, 1);
    assert!(at(5, 0).water.unwrap().0 > at(9, 0).water.unwrap().0);
    // The shore of a lake counts too (sand beach candidates): ground 101 is 2 over the top 99.
    assert!(at(11, 0).shore);
    assert!(!at(14, 0).shore);
}

#[test]
fn a_small_lake_without_a_record_is_flat_at_the_lowest_surface() {
    let in_lake = |x: i64, z: i64| x * x + z * z <= 16;
    let s = move |x: i64, z: i64| {
        if in_lake(x, z) {
            // A bumpy DEM over the water: the lowest is 200.
            water(200 + ((x + z).rem_euclid(3)) as i32)
        } else {
            land(203)
        }
    };
    let origin = (-8, -8);
    let out = run(None, &s, origin, (17, 17), 1);
    for z in -8..=8 {
        for x in -8..=8 {
            let c = out[((z - origin.1) * 17 + (x - origin.0)) as usize];
            if in_lake(x, z) {
                assert_eq!(c.s, 199, "({x},{z})");
                let depth = c.water.unwrap().0;
                assert!(depth <= FALLBACK_LAKE_DEPTH as u32);
            }
        }
    }
    let centre = out[(8 * 17 + 8) as usize];
    assert_eq!(centre.water.unwrap().0, FALLBACK_LAKE_DEPTH as u32);
}

#[test]
fn a_lake_under_the_sea_level_is_a_lake_not_sea_when_the_atlas_has_it() {
    let level = LakeLevel {
        top: SEA - 30,
        max_depth: 4,
    };
    let s = move |x: i64, z: i64| {
        if x * x + z * z <= 36 {
            Sample {
                lake: Some(level),
                ..water(SEA - 29)
            }
        } else {
            land(SEA - 20)
        }
    };
    let out = run(None, &s, (-4, -4), (9, 9), 1);
    assert!(
        out.iter()
            .all(|c| c.s == SEA - 30 && c.of == Some(WaterOf::Lake))
    );
}

/// A river along z = 0 from x = -200 to 200 at 20 m falling to 0 m... over a valley DEM that rises away from it.
fn river_world(q: f32) -> (RiverIndex, impl Fn(i64, i64) -> Sample) {
    // Level 125 m at x = 0, falling 1 m every 8 blocks.
    let r = reach(1, &[(-200.0, 0.0), (200.0, 0.0)], &[150.0, 100.0], q);
    let sample = |x: i64, z: i64| land(SEA + 125 - (x / 8) as i32 + 2 + (z.abs() / 3) as i32);
    (index(&[r]), sample)
}

#[test]
fn a_river_over_a_sloping_valley_runs_in_every_column_and_never_rises() {
    let (idx, sample) = river_world(4.0); // half width 7.2
    let origin = (-60, -40);
    let (w, h) = (120usize, 80usize);
    let out = run(Some(&idx), &sample, origin, (w, h), 1);
    let at = |x: i64, z: i64| out[((z - origin.1) as usize) * w + (x - origin.0) as usize];
    let mut last_top = i32::MAX;
    for x in -60..60 {
        let c = at(x, 0);
        assert_eq!(c.of, Some(WaterOf::River), "x {x}: water on the line");
        assert!(c.channel);
        assert!(c.water.unwrap().0 >= 1);
        assert_eq!(c.water.unwrap().1, Bed::Gravel);
        assert!(c.s <= last_top, "x {x}: the top never rises downstream");
        last_top = c.s;
    }
    // The channel is as wide as the hydraulic geometry says (half width 7.2 -> |z| <= 7).
    let wide: Vec<i64> = (-40..40).filter(|&z| at(0, z).channel).collect();
    assert_eq!((wide[0], *wide.last().unwrap()), (-7, 7));
    // The banks blend: at the edge the ground is one block over the water, then it rises to the valley.
    let top = at(0, 0).s;
    let edge = at(0, 9);
    assert!(edge.water.is_none());
    assert!(
        edge.s <= top + 3,
        "bank at the water: {} vs {}",
        edge.s,
        top
    );
    assert!(edge.s < sample(0, 9).dem, "carved under the valley floor");
    let mut prev = edge.s;
    for z in 9..40 {
        let s = at(0, z).s;
        assert!(s >= prev, "the bank rises away from the water at z {z}");
        prev = s;
    }
    assert_eq!(
        at(0, 35).s,
        sample(0, 35).dem,
        "the valley beyond the bank is untouched"
    );
    // Beside the channel the dry ground is shore.
    assert!(at(0, 8).shore);
}

#[test]
fn a_far_view_keeps_a_small_river_continuous() {
    // A stream (Q = 1, half width 3.6) between the lattice rows of a step 16 view: still a column of water.
    let (idx, sample) = river_world(1.0);
    let out = run(Some(&idx), &sample, (-80, -37), (10, 4), 16);
    for i in 0..10 {
        let c = out[2 * 10 + i];
        assert!(
            c.channel && c.water.is_some(),
            "column {i} of the row 5 blocks from the line"
        );
        assert_eq!(out[i].water, None, "37 blocks away is dry");
    }
    // A creek under a quarter of a sample wide (Q = 0, half width 1) is not widened: it is between the rows.
    let (idx, sample) = river_world(0.0);
    let out = run(Some(&idx), &sample, (-80, -37), (10, 4), 16);
    assert!(out.iter().all(|c| !c.channel));
}

#[test]
fn a_river_with_more_discharge_has_a_wider_channel() {
    let width = |q: f32| {
        let (idx, sample) = river_world(q);
        let out = run(Some(&idx), &sample, (0, -40), (1, 80), 1);
        out.iter().filter(|c| c.channel).count()
    };
    let (small, big) = (width(1.0), width(16.0));
    assert!(small >= 5 && small < big, "{small} vs {big}");
}

#[test]
fn the_channel_is_deepest_in_the_middle() {
    // A big river on low ground: 161 m wide at 1 m a block (half width 80), 3 blocks deep in the middle.
    let r = reach(1, &[(-200.0, 0.0), (200.0, 0.0)], &[30.0, 30.0], 500.0);
    let idx = index(&[r]);
    let s = |_: i64, z: i64| land(SEA + 30 + 3 + (z.abs() / 6) as i32);
    let out = run(Some(&idx), &s, (0, -100), (1, 200), 1);
    let depth = |z: i64| out[(z + 100) as usize].water.map(|w| w.0);
    assert_eq!(depth(0), Some(3));
    assert_eq!(depth(40), Some(2));
    assert_eq!(depth(-40), Some(2));
    assert_eq!(depth(70), Some(1));
    assert!(depth(95).is_none());
}

#[test]
fn a_river_far_under_the_ground_leaves_the_ground_alone() {
    // 164 m of ground over a river level of 0 m: more than MAX_CARVE, no pit and no ring of carved banks.
    let r = reach(1, &[(-100.0, 0.0), (100.0, 0.0)], &[0.0, 0.0], 25.0);
    let idx = index(&[r]);
    let s = |_: i64, _: i64| land(SEA + 164);
    let out = run(Some(&idx), &s, (0, -20), (3, 40), 1);
    assert!(
        out.iter()
            .all(|c| c.water.is_none() && c.s == SEA + 164 && !c.channel)
    );
}

#[test]
fn a_river_in_a_canyon_sits_on_the_floor() {
    // The surface is the canyon floor (70) under the channel and rises 40 blocks in 40 blocks: walls beyond
    // the bank stay, the channel is on the floor.
    let r = reach(1, &[(-100.0, 0.0), (100.0, 0.0)], &[6.0, 6.0], 4.0);
    let idx = index(&[r]);
    let s = |_: i64, z: i64| {
        land(if z.abs() <= 8 {
            SEA + 7
        } else {
            SEA + 7 + (z.abs() - 8) as i32
        })
    };
    let out = run(Some(&idx), &s, (0, -30), (1, 60), 1);
    let c = out[30];
    assert_eq!(c.s, SEA + 5);
    assert!(c.channel);
    assert_eq!(out[0].s, s(0, -30).dem, "the wall is untouched");
}

#[test]
fn the_sea_keeps_its_top_where_a_river_mouth_meets_it() {
    // A river whose levels reach 0 m: its top is the sea's top, and it is never lower.
    let r = reach(1, &[(0.0, 0.0), (60.0, 0.0)], &[5.0, -3.0], 9.0);
    let idx = index(&[r]);
    let s = |x: i64, _: i64| if x < 40 { land(SEA + 8) } else { water(SEA) };
    let out = run(Some(&idx), &s, (0, -6), (60, 12), 1);
    for x in 0..60 {
        let c = out[6 * 60 + x];
        assert!(c.water.is_some(), "x {x}");
        assert!(c.s >= SEA - 1, "x {x}: never under the sea's top");
    }
}

/// A bumpy world with a sea in the west, a lake, a gap next to it and a river to the sea.
struct Mixed {
    rivers: RiverIndex,
}

fn hash(x: i64, z: i64) -> i32 {
    let mut h = (x.wrapping_mul(73856093) ^ z.wrapping_mul(19349663)) as u64;
    h ^= h >> 13;
    h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    ((h >> 33) % 4) as i32
}

impl Mixed {
    fn new() -> Self {
        // The river flows west to the sea, so its levels fall that way: upstream first.
        let r = reach(1, &[(60.0, -40.0), (-30.0, -40.0)], &[20.0, 0.0], 9.0);
        Self {
            rivers: index(&[r]),
        }
    }

    fn sample(&self, x: i64, z: i64) -> Sample {
        let ground = SEA + 3 + ((x + 30) / 4).max(0) as i32 + hash(x, z) / 2;
        if x < -30 {
            water(SEA + hash(x, z) / 2)
        } else if (x - 20).pow(2) + (z - 30).pow(2) <= 64 {
            Sample {
                lake: Some(LakeLevel {
                    top: SEA + 14 - 1,
                    max_depth: 5,
                }),
                ..water(SEA + 14)
            }
        } else if (x, z) == (29, 30) {
            land(SEA + 6)
        } else {
            land(ground)
        }
    }
}

fn mixed_window(
    m: &Mixed,
    origin: (i64, i64),
    size: (usize, usize),
    stride: i32,
) -> Vec<ColumnWater> {
    run(
        Some(&m.rivers),
        &|x, z| m.sample(x, z),
        origin,
        size,
        stride,
    )
}

#[test]
fn water_never_hangs_and_stays_inside_the_bounds() {
    let m = Mixed::new();
    for stride in [1, 4] {
        let size = (40usize, 40usize);
        let origin = (-60 + 3, -60 + 5);
        let out = mixed_window(&m, (origin.0, origin.1), size, stride);
        let mut wet = 0;
        for j in 0..size.1 {
            for i in 0..size.0 {
                let c = out[j * size.0 + i];
                let x = origin.0 + i as i64 * i64::from(stride);
                let z = origin.1 + j as i64 * i64::from(stride);
                let Some((depth, _)) = c.water else { continue };
                wet += 1;
                let dem = i64::from(m.sample_dem(x, z, stride));
                assert!(depth >= 1, "({x},{z})");
                let ground = i64::from(c.s) - i64::from(depth);
                assert!(
                    ground >= dem - MAX_CARVE,
                    "({x},{z}) ground {ground} dem {dem}"
                );
                if c.of != Some(WaterOf::Sea) {
                    assert!(
                        i64::from(c.s) <= dem + MAX_RAISE,
                        "({x},{z}) top {} dem {dem}",
                        c.s
                    );
                }
                // No dry 8-neighbour lower than the water's top.
                for dj in -1i64..=1 {
                    for di in -1i64..=1 {
                        let (ni, nj) = (i as i64 + di, j as i64 + dj);
                        if ni < 0 || nj < 0 || ni >= size.0 as i64 || nj >= size.1 as i64 {
                            continue;
                        }
                        let n = out[nj as usize * size.0 + ni as usize];
                        if n.water.is_none() {
                            assert!(
                                n.s >= c.s,
                                "stride {stride} ({x},{z}): water top {} beside dry {} at ({di},{dj})",
                                c.s,
                                n.s
                            );
                        }
                    }
                }
            }
        }
        assert!(wet > 200, "the world has water: {wet}");
    }
}

impl Mixed {
    fn sample_dem(&self, x: i64, z: i64, _stride: i32) -> i32 {
        self.sample(x, z).dem
    }
}

#[test]
fn the_mixed_world_has_all_three_kinds_of_water_and_a_channel() {
    let m = Mixed::new();
    let out = mixed_window(&m, (-50, -50), (100, 100), 1);
    for of in [WaterOf::Sea, WaterOf::River, WaterOf::Lake] {
        assert!(out.iter().any(|c| c.of == Some(of)), "{of:?}");
    }
    assert!(out.iter().any(|c| c.channel));
    assert!(out.iter().any(|c| c.shore));
}

#[test]
fn every_column_is_the_same_through_two_windows_offset_by_a_few_samples() {
    let m = Mixed::new();
    for stride in [1, 2, 8] {
        let size = (36usize, 36usize);
        let step = i64::from(stride);
        let a_origin = (-50 * step.min(2), -30);
        let (da, db) = (5i64, 3i64);
        let b_origin = (a_origin.0 + da * step, a_origin.1 + db * step);
        let a = mixed_window(&m, a_origin, size, stride);
        let b = mixed_window(&m, b_origin, size, stride);
        let mut compared = 0;
        for j in 0..size.1 as i64 {
            for i in 0..size.0 as i64 {
                let (bi, bj) = (i - da, j - db);
                if bi < 0 || bj < 0 || bi >= size.0 as i64 || bj >= size.1 as i64 {
                    continue;
                }
                assert_eq!(
                    a[(j as usize) * size.0 + i as usize],
                    b[(bj as usize) * size.0 + bi as usize],
                    "stride {stride} column ({i},{j}) of window a"
                );
                compared += 1;
            }
        }
        assert!(compared > 800);
    }
}

#[test]
fn a_window_does_not_depend_on_what_lies_beyond_the_window_pad() {
    // Change the world beyond WINDOW_PAD of a 4 by 4 patch and its answer stays.
    let m = Mixed::new();
    let origin = (-33, -3);
    let base = mixed_window(&m, origin, (4, 4), 1);
    let far_x = origin.0 + 4 + WINDOW_PAD as i64 + 1;
    let changed = run(
        Some(&m.rivers),
        &|x, z| {
            if x >= far_x {
                water(SEA - 500)
            } else {
                m.sample(x, z)
            }
        },
        origin,
        (4, 4),
        1,
    );
    assert_eq!(base, changed);
}

#[test]
fn a_water_column_that_cannot_hold_water_is_dry_and_no_water_hangs_next_to_it() {
    // A lake on a plateau whose neighbour is a sheer drop: the lake's edge cannot keep water over it.
    let level = LakeLevel {
        top: SEA + 99,
        max_depth: 4,
    };
    let s = move |x: i64, z: i64| {
        if x < 0 && z.abs() < 8 {
            Sample {
                lake: Some(level),
                ..water(SEA + 100)
            }
        } else if x >= 0 && z.abs() < 8 {
            land(SEA + 100 - 80)
        } else {
            land(SEA + 101)
        }
    };
    let out = run(None, &s, (-10, -6), (20, 12), 1);
    for j in 0..12usize {
        for i in 0..20usize {
            let c = out[j * 20 + i];
            if c.water.is_none() {
                continue;
            }
            for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let (ni, nj) = (i as i64 + di, j as i64 + dj);
                if ni < 0 || nj < 0 || ni >= 20 || nj >= 12 {
                    continue;
                }
                let n = out[nj as usize * 20 + ni as usize];
                assert!(
                    n.water.is_some() || n.s >= c.s,
                    "({i},{j}) top {} beside dry {}",
                    c.s,
                    n.s
                );
            }
        }
    }
}

//! The atlas shipped in the game: its rules hold everywhere, and a few real places look right.
//!
//! The whole-atlas walks decode every cell once without keeping it; they are slow in a debug build (tens of
//! seconds) but read nothing from the network.

use std::collections::HashMap;

use super::*;

fn atlas() -> &'static WaterAtlas {
    WaterAtlas::embedded().expect("the game ships a water atlas")
}

/// The distance in km from `(lat, lon)` to the segment `a`-`b`, and where along it (0..=1) the nearest point is.
fn segment_km(lat: f64, lon: f64, a: LatLon, b: LatLon) -> (f64, f64) {
    const KM_PER_DEG: f64 = 111.32;
    let c = lat.to_radians().cos();
    let (ax, ay) = ((a.lon - lon) * c, a.lat - lat);
    let (bx, by) = ((b.lon - lon) * c, b.lat - lat);
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    let f = if len2 == 0.0 {
        0.0
    } else {
        (-(ax * dx + ay * dy) / len2).clamp(0.0, 1.0)
    };
    ((ax + f * dx).hypot(ay + f * dy) * KM_PER_DEG, f)
}

/// Every reach within `km` of a place: (distance km, discharge, level at the nearest point, id).
fn reaches_near(lat: f64, lon: f64, km: f64) -> Vec<(f64, f32, f32, u32)> {
    let d = km / 100.0 + 0.01;
    let mut out = Vec::new();
    for r in atlas().reaches_in(lat - d, lon - d, lat + d, lon + d) {
        let best = r
            .points
            .windows(2)
            .zip(r.levels_m.windows(2))
            .map(|(p, l)| {
                let (dist, f) = segment_km(lat, lon, p[0], p[1]);
                (dist, l[0] + (l[1] - l[0]) * f as f32)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((dist, level)) = best.filter(|b| b.0 <= km) {
            out.push((dist, r.discharge_m3s, level, r.id));
        }
    }
    out
}

#[test]
fn the_embedded_atlas_decodes_and_is_not_small() {
    let a = atlas();
    assert!(a.reach_count() > 100_000, "{} reaches", a.reach_count());
    assert!(a.lake_count() > 10_000, "{} lakes", a.lake_count());
    assert!(EMBEDDED.len() <= 16_000_000, "{} bytes", EMBEDDED.len());
}

#[test]
fn every_reach_runs_downhill_and_joins_the_next_at_its_level() {
    let a = WaterAtlas::decode(EMBEDDED).unwrap();
    let mut first_level: HashMap<u32, f32> = HashMap::new();
    let mut last_level: Vec<(u32, u32, f32)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for key in a.cell_keys() {
        let (reaches, _) = a.read_cell(key).unwrap().expect("every cell decodes");
        for r in reaches {
            if !seen.insert(r.id) {
                continue;
            }
            assert_eq!(r.points.len(), r.levels_m.len(), "reach {}", r.id);
            assert!(
                r.levels_m.windows(2).all(|w| w[1] <= w[0]),
                "reach {} rises: {:?}",
                r.id,
                r.levels_m
            );
            assert!(
                r.discharge_m3s.is_finite() && r.discharge_m3s > 0.0,
                "reach {}",
                r.id
            );
            first_level.insert(r.id, r.levels_m[0]);
            if let Some(next) = r.next_down {
                last_level.push((r.id, next, *r.levels_m.last().unwrap()));
            }
        }
    }
    assert_eq!(seen.len(), a.reach_count());
    let mut joined = 0;
    for (id, next, level) in last_level {
        if let Some(&start) = first_level.get(&next) {
            assert_eq!(
                level, start,
                "reach {id} ends at {level} m, {next} starts at {start} m"
            );
            joined += 1;
        }
    }
    assert!(
        joined > seen.len() / 2,
        "{joined} junctions of {} reaches",
        seen.len()
    );
}

/// Memphis's riverfront. (The first spot tried, (35.0, -90.1), is 9.7 km from the river's line.)
#[test]
fn the_mississippi_at_memphis_is_a_big_river_near_its_level() {
    let near = reaches_near(35.15, -90.07, 2.0);
    let big = near
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("a reach near Memphis");
    assert!(big.1 > 5_000.0, "{near:?}");
    assert!(
        (50.0..=75.0).contains(&big.2),
        "level {} m: {near:?}",
        big.2
    );
}

/// Phantom Ranch. The data's line (15 arc-seconds) runs about 2.1 km north of it, so the search is 2.5 km.
#[test]
fn the_colorado_runs_on_the_grand_canyon_floor() {
    let near = reaches_near(36.10, -112.09, 2.5);
    let big = near
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("a reach near Phantom Ranch");
    assert!(big.1 > 100.0, "{near:?}");
    assert!(
        (700.0..=800.0).contains(&big.2),
        "level {} m: {near:?}",
        big.2
    );
}

#[test]
fn the_amazon_near_manaus_carries_the_most_water() {
    let near = reaches_near(-3.1, -59.9, 10.0);
    assert!(near.iter().any(|r| r.1 > 50_000.0), "{near:?}");
}

#[test]
fn lake_tahoe_is_at_its_level() {
    let lakes = atlas().lakes_in(39.09, -120.03, 39.09, -120.03);
    let tahoe = lakes
        .iter()
        .filter(|l| l.contains(39.09, -120.03))
        .max_by(|a, b| a.area_km2.total_cmp(&b.area_km2))
        .expect("a lake over Lake Tahoe");
    assert!((1_890.0..=1_905.0).contains(&tahoe.level_m), "{tahoe:?}");
    assert!(tahoe.area_km2 > 400.0 && !tahoe.reservoir, "{tahoe:?}");
}

#[test]
fn lake_mead_is_a_reservoir() {
    let lakes = atlas().lakes_in(36.0, -114.9, 36.3, -114.0);
    let mead = lakes
        .iter()
        .filter(|l| l.contains(36.15, -114.45))
        .max_by(|a, b| a.area_km2.total_cmp(&b.area_km2))
        .expect("a lake over Lake Mead");
    assert!(mead.reservoir && mead.area_km2 > 300.0, "{mead:?}");
}

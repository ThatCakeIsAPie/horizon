//! The atlas file: round trips, the Python writer's fixture, and bad files.

use super::*;
use crate::waterfile::{UNITS, log_code, log_value};

/// Whether `a` and `b` agree within the log code's half step (0.034 %).
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 4e-4 * a.abs().max(b.abs())
}

fn ll(lat: f64, lon: f64) -> LatLon {
    LatLon { lat, lon }
}

fn reach(
    id: u32,
    pts: &[(f64, f64)],
    levels: &[f32],
    q: f32,
    next_down: Option<u32>,
) -> RiverReach {
    RiverReach {
        id,
        points: pts.iter().map(|&(lat, lon)| ll(lat, lon)).collect(),
        levels_m: levels.to_vec(),
        discharge_m3s: q,
        next_down,
    }
}

fn lake(id: u32, box_: (f64, f64, f64, f64), level_m: f32, reservoir: bool) -> Lake {
    Lake {
        id,
        south: box_.0,
        west: box_.1,
        north: box_.2,
        east: box_.3,
        level_m,
        area_km2: 2.5,
        depth_avg_m: 7.25,
        reservoir,
    }
}

fn sample() -> (Vec<RiverReach>, Vec<Lake>) {
    (
        vec![
            reach(
                1,
                &[(10.5, 20.5), (10.6, 20.7), (11.2, 21.3)],
                &[120.5, 110.0, 110.0],
                12.5,
                Some(2),
            ),
            reach(
                2,
                &[(11.2, 21.3), (11.25, 21.4)],
                &[110.0, 3.2],
                30.25,
                None,
            ),
            reach(3, &[(0.5, -179.5), (0.6, -179.4)], &[5.0, -1.5], 1.0, None),
            reach(
                9,
                &[(-45.123_45, 170.0), (-44.0, 172.5), (-43.5, 172.6)],
                &[900.0, 450.3, 0.0],
                5000.0,
                None,
            ),
        ],
        vec![
            lake(7, (10.1, 20.1, 10.2, 20.2), 50.0, true),
            lake(8, (-33.5, -70.8, -33.4, -70.6), -12.3, false),
            lake(11, (36.0, 50.0, 38.5, 53.0), -28.0, false),
        ],
    )
}

fn same_reach(a: &RiverReach, b: &RiverReach) {
    let eps = 1.0 / f64::from(UNITS);
    assert_eq!(a.id, b.id);
    assert_eq!(a.next_down, b.next_down);
    assert!(
        close(a.discharge_m3s, b.discharge_m3s),
        "{} vs {}",
        a.discharge_m3s,
        b.discharge_m3s
    );
    assert_eq!(a.points.len(), b.points.len());
    for (p, q) in a.points.iter().zip(&b.points) {
        assert!(
            (p.lat - q.lat).abs() <= eps && (p.lon - q.lon).abs() <= eps,
            "{p:?} vs {q:?}"
        );
    }
    for (x, y) in a.levels_m.iter().zip(&b.levels_m) {
        assert!((x - y).abs() <= 0.051, "{x} vs {y}");
    }
}

/// Every query box a test asks, as `(south, west, north, east)`.
const BOXES: &[(f64, f64, f64, f64)] = &[
    (10.0, 20.0, 12.0, 22.0),
    (11.0, 21.0, 11.5, 21.5),
    (0.0, 179.0, 1.0, -179.0),
    (-46.0, 169.0, -43.0, 173.0),
    (-34.0, -71.0, -33.0, -70.0),
    (37.0, 51.0, 37.5, 51.5),
    (-90.0, -180.0, 90.0, 180.0),
    (60.0, 0.0, 61.0, 1.0),
];

#[test]
fn a_decoded_atlas_answers_like_the_one_it_was_made_from() {
    let (reaches, lakes) = sample();
    let memory = WaterAtlas::from_parts(reaches.clone(), lakes.clone());
    let decoded = WaterAtlas::decode(WaterAtlas::encode(&reaches, &lakes)).expect("decodes");
    assert_eq!(decoded.reach_count(), 4);
    assert_eq!(decoded.lake_count(), 3);
    assert_eq!(decoded.cell_keys(), memory.cell_keys());
    for &(s, w, n, e) in BOXES {
        let (a, b) = (
            memory.reaches_in(s, w, n, e),
            decoded.reaches_in(s, w, n, e),
        );
        assert_eq!(a.len(), b.len(), "reaches in {:?}", (s, w, n, e));
        for (x, y) in a.iter().zip(&b) {
            same_reach(x, y);
        }
        let (a, b) = (memory.lakes_in(s, w, n, e), decoded.lakes_in(s, w, n, e));
        assert_eq!(
            a.iter().map(|l| l.id).collect::<Vec<_>>(),
            b.iter().map(|l| l.id).collect::<Vec<_>>()
        );
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.level_m, y.level_m);
            assert!(close(x.area_km2, y.area_km2) && close(x.depth_avg_m, y.depth_avg_m));
            assert_eq!(x.reservoir, y.reservoir);
            assert!(
                y.south <= x.south && y.west <= x.west && y.north >= x.north && y.east >= x.east
            );
            let unit = 1.0 / f64::from(UNITS);
            assert!((y.north - x.north).abs() <= unit && (y.south - x.south).abs() <= unit);
        }
    }
}

#[test]
fn equal_levels_in_metres_stay_equal_after_a_round_trip() {
    let (reaches, lakes) = sample();
    let atlas = WaterAtlas::decode(WaterAtlas::encode(&reaches, &lakes)).unwrap();
    let r = atlas.reaches_in(10.0, 20.0, 12.0, 22.0);
    assert_eq!(r[0].levels_m.last(), r[1].levels_m.first());
    assert_eq!(r[1].levels_m, vec![110.0, 3.2]);
}

#[test]
fn a_rising_level_is_written_flat() {
    let r = reach(
        5,
        &[(1.0, 1.0), (1.1, 1.1), (1.2, 1.2)],
        &[10.0, 12.0, 9.0],
        1.0,
        None,
    );
    let atlas = WaterAtlas::decode(WaterAtlas::encode(&[r], &[])).unwrap();
    assert_eq!(
        atlas.reaches_in(1.0, 1.0, 2.0, 2.0)[0].levels_m,
        vec![10.0, 10.0, 9.0]
    );
}

#[test]
fn an_empty_atlas_round_trips() {
    let atlas = WaterAtlas::decode(WaterAtlas::encode(&[], &[])).unwrap();
    assert!(atlas.is_empty());
    assert!(atlas.reaches_in(-90.0, -180.0, 90.0, 180.0).is_empty());
}

#[test]
fn the_python_writers_fixture_decodes_to_its_listed_contents() {
    // Written by `python3 tools/earth_water.py fixture crates/horizon-earth/tests/data/water_fixture.bin`.
    let bytes: &'static [u8] = include_bytes!("../tests/data/water_fixture.bin");
    let atlas = WaterAtlas::decode(bytes).expect("the fixture decodes");
    assert_eq!((atlas.reach_count(), atlas.lake_count()), (3, 2));
    let (reaches, lakes) = sample();
    let all = atlas.reaches_in(-90.0, -180.0, 90.0, 180.0);
    assert_eq!(all.iter().map(|r| r.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    for (got, want) in all.iter().zip(&reaches) {
        same_reach(got, want);
        let want_levels: Vec<f32> = want
            .levels_m
            .iter()
            .map(|&l| (l * 10.0).round() / 10.0)
            .collect();
        assert_eq!(got.levels_m, want_levels);
    }
    let lakes_got = atlas.lakes_in(-90.0, -180.0, 90.0, 180.0);
    assert_eq!(lakes_got.len(), 2);
    for (got, want) in lakes_got.iter().zip(&lakes) {
        assert_eq!(got.id, want.id);
        assert_eq!(got.level_m, want.level_m);
        assert_eq!(got.reservoir, want.reservoir);
        let unit = 1.0 / f64::from(UNITS);
        assert!((got.south - want.south).abs() <= unit && (got.east - want.east).abs() <= unit);
    }
    for (got, (area, depth)) in lakes_got.iter().zip([(1.5, 4.25), (0.75, 2.0)]) {
        assert!(
            close(got.area_km2, area) && close(got.depth_avg_m, depth),
            "{got:?}"
        );
    }
    // The Rust encoder and the Python writer agree on what the bytes mean.
    let again = WaterAtlas::decode(WaterAtlas::encode(
        &all.iter().map(|r| (**r).clone()).collect::<Vec<_>>(),
        &lakes_got.iter().map(|l| (**l).clone()).collect::<Vec<_>>(),
    ))
    .unwrap();
    assert_eq!(again.cell_keys(), atlas.cell_keys());
}

fn valid() -> Vec<u8> {
    let (reaches, lakes) = sample();
    WaterAtlas::encode(&reaches, &lakes)
}

fn put_u32(bytes: &mut [u8], at: usize, v: u32) {
    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Decodes and queries everything; must never panic.
fn query_all(bytes: Vec<u8>) -> Option<usize> {
    let atlas = WaterAtlas::decode(bytes).ok()?;
    let mut n = 0;
    for key in atlas.cell_keys() {
        let (la, lo) = (f64::from(key.0), f64::from(key.1));
        n += atlas.reaches_in(la, lo, la + 0.5, lo + 0.5).len();
        n += atlas.lakes_in(la, lo, la + 0.5, lo + 0.5).len();
        let _ = atlas.read_cell(key);
    }
    Some(n)
}

#[test]
fn log_codes_keep_quantities_within_their_step() {
    assert_eq!(
        (log_code(0.0), log_code(-1.0), log_code(f32::NAN)),
        (0, 0, 0)
    );
    assert_eq!(log_value(0), 0.0);
    assert_eq!(log_value(log_code(1.0)), 1.0);
    for x in [
        1e-3f32, 0.1, 0.5, 2.5, 12.5, 299.9, 5_000.0, 209_000.0, 1.5e6,
    ] {
        assert!(close(log_value(log_code(x)), x), "{x}");
    }
    assert_eq!(log_code(f32::MAX), 65_535);
}

#[test]
fn a_bad_header_or_index_is_an_error() {
    let good = valid();
    assert!(WaterAtlas::decode(Vec::new()).is_err());
    assert!(WaterAtlas::decode(good[..10].to_vec()).is_err());
    let mut b = good.clone();
    b[0] = b'X';
    assert!(WaterAtlas::decode(b).unwrap_err().0.contains("magic"));
    let mut b = good.clone();
    put_u32(&mut b, 4, WATER_RULES + 1);
    assert!(WaterAtlas::decode(b).unwrap_err().0.contains("version"));
    let mut b = good.clone();
    put_u32(&mut b, 8, 0);
    assert!(WaterAtlas::decode(b).is_err());
    let mut b = good.clone();
    put_u32(&mut b, 20, u32::MAX);
    assert!(WaterAtlas::decode(b).is_err());
    // The first index entry: offset out of the file, then a length past the end.
    let mut b = good.clone();
    put_u32(&mut b, 24 + 4, u32::MAX - 3);
    assert!(WaterAtlas::decode(b).unwrap_err().0.contains("outside"));
    let mut b = good.clone();
    put_u32(&mut b, 24 + 8, good.len() as u32);
    assert!(WaterAtlas::decode(b).is_err());
    let mut b = good.clone();
    put_u32(&mut b, 24 + 12, u32::MAX);
    assert!(WaterAtlas::decode(b).unwrap_err().0.contains("too large"));
    // Two entries swapped: not sorted.
    let mut b = good.clone();
    let (first, second) = (b[24..40].to_vec(), b[40..56].to_vec());
    b[24..40].copy_from_slice(&second);
    b[40..56].copy_from_slice(&first);
    assert!(WaterAtlas::decode(b).unwrap_err().0.contains("sorted"));
    // A cell off the globe.
    let mut b = good;
    b[24..26].copy_from_slice(&120i16.to_le_bytes());
    assert!(WaterAtlas::decode(b).is_err());
}

#[test]
fn a_bad_block_reads_as_an_empty_cell() {
    let good = valid();
    let atlas = WaterAtlas::decode(good.clone()).unwrap();
    let key = atlas.cell_keys()[0];
    let mut b = good.clone();
    let offset = u32::from_le_bytes(b[28..32].try_into().unwrap()) as usize;
    let len = u32::from_le_bytes(b[32..36].try_into().unwrap()) as usize;
    for x in &mut b[offset..offset + len] {
        *x = 0xff;
    }
    let broken = WaterAtlas::decode(b).expect("the index is still fine");
    assert!(broken.read_cell(key).unwrap().is_err());
    let (la, lo) = (f64::from(key.0), f64::from(key.1));
    assert!(broken.reaches_in(la, lo, la + 0.9, lo + 0.9).is_empty());
    assert!(broken.lakes_in(la, lo, la + 0.9, lo + 0.9).is_empty());
    // A wrong inflated length is caught too.
    let mut b = good;
    let raw_len = u32::from_le_bytes(b[36..40].try_into().unwrap());
    put_u32(&mut b, 36, raw_len + 1);
    assert!(
        WaterAtlas::decode(b)
            .unwrap()
            .read_cell(key)
            .unwrap()
            .is_err()
    );
}

/// A block holding `raw` as cell (0, 0).
fn one_block(raw: &[u8]) -> Vec<u8> {
    let comp = miniz_oxide::deflate::compress_to_vec(raw, 6);
    let mut b = Vec::new();
    b.extend_from_slice(b"HZWA");
    for v in [WATER_RULES, UNITS, 1, 0, 1] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&0i16.to_le_bytes());
    b.extend_from_slice(&0i16.to_le_bytes());
    b.extend_from_slice(&40u32.to_le_bytes());
    b.extend_from_slice(&(comp.len() as u32).to_le_bytes());
    b.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    b.extend_from_slice(&comp);
    b
}

#[test]
fn bad_block_contents_are_errors_not_panics() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "huge counts",
            vec![0xff, 0xff, 0xff, 0xff, 0x0f, 0xff, 0xff, 0xff, 0xff, 0x0f],
        ),
        ("varint too long", vec![0xff; 12]),
        ("truncated reach", vec![1, 0, 5, 0]),
        // One reach (id 1, flowing nowhere, discharge 1, two points from the corner, levels 0 and drop 0), a byte.
        (
            "trailing",
            vec![1, 0, 1, 0, 0x00, 0x80, 2, 0, 0, 2, 2, 0, 0, 9],
        ),
        // A reach with one point.
        (
            "one point",
            vec![1, 0, 1, 0, 0x00, 0x80, 1, 0, 0, 0, 0, 0, 0, 0],
        ),
        // Two reaches with the same id (a zero id step).
        ("repeated id", {
            let one = [0, 0x00, 0x80, 2, 0, 0, 2, 2, 0, 0];
            let mut v = vec![2, 0, 1];
            v.extend_from_slice(&one);
            v.push(0);
            v.extend_from_slice(&one);
            v
        }),
        // A reach flowing into an id below 0.
        (
            "bad next id",
            vec![1, 0, 1, 5, 0x00, 0x80, 2, 0, 0, 2, 2, 0, 0],
        ),
        // A point at latitude 100 degrees (960,000 units, zigzag 1,920,000).
        ("off the globe", {
            let mut v = vec![1, 0, 1, 0, 0x00, 0x80, 2];
            let mut n: u64 = 1_920_000;
            loop {
                let b = (n & 0x7f) as u8;
                n >>= 7;
                if n == 0 {
                    v.push(b);
                    break;
                }
                v.push(b | 0x80);
            }
            v.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
            v
        }),
    ];
    for (what, raw) in cases {
        let atlas = WaterAtlas::decode(one_block(&raw)).expect(what);
        assert!(atlas.read_cell((0, 0)).unwrap().is_err(), "{what}");
        assert!(atlas.reaches_in(0.0, 0.0, 0.5, 0.5).is_empty(), "{what}");
    }
}

#[test]
fn flipped_and_cut_bytes_never_panic() {
    let good = valid();
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..2000 {
        let mut b = good.clone();
        let flips = 1 + next() % 4;
        for _ in 0..flips {
            let i = (next() as usize) % b.len();
            b[i] ^= 1 << (next() % 8);
        }
        if next() % 4 == 0 {
            let cut = (next() as usize) % b.len();
            b.truncate(cut);
        }
        let _ = query_all(b);
    }
    assert!(query_all(good).unwrap() > 0);
}

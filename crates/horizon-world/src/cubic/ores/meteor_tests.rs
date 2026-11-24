//! Tests of the ore meteors on a synthetic flat world: stone up to Y 60, dirt
//! 61 to 63, grass at 64, air above.

use std::cell::Cell;
use std::ops::Range;

use horizon_core::position::SectionPos;

use super::meteors::{MeteorPlan, Meteors, Placement};
use super::settings::{MeteorKind, MeteorSettings};
use crate::cubic::custom::populate::CubeWorld;
use crate::cubic::custom::rng::hash_pos;
use crate::cubic::custom::testing::{MapWorld, default_block_names, registry_of};
use crate::registry::BlockRegistry;

const EXTRA: &[&str] = &[
    "raw_iron_block",
    "raw_copper_block",
    "raw_gold_block",
    "diamond_block",
    "emerald_block",
    "lapis_block",
    "redstone_block",
    "coal_block",
    "oak_log",
    "oak_leaves",
    "short_grass",
];

thread_local! {
    static REG: &'static BlockRegistry = {
        let mut names = default_block_names();
        names.extend(EXTRA.iter().map(|n| format!("minecraft:{n}")));
        registry_of(&names)
    };
}

fn reg() -> &'static BlockRegistry {
    REG.with(|r| *r)
}

fn id(name: &str) -> u32 {
    reg()
        .default_state(&format!("minecraft:{name}"))
        .unwrap_or_else(|| panic!("no block {name}"))
        .0
}

fn flat(_x: i32, _z: i32) -> i32 {
    64
}

fn flat_block(y: i32) -> u32 {
    match y {
        ..=60 => id("stone"),
        61..=63 => id("dirt"),
        64 => id("grass_block"),
        _ => id("air"),
    }
}

/// A flat world over the given cube ranges (absent cubes read air).
fn flat_world(xs: Range<i32>, ys: Range<i32>, zs: Range<i32>) -> MapWorld {
    let mut w = MapWorld::new(id("air"));
    for cy in ys {
        let mut blocks = [0u32; 4096];
        for (i, b) in blocks.iter_mut().enumerate() {
            *b = flat_block(cy * 16 + (i >> 8) as i32);
        }
        for cx in xs.clone() {
            for cz in zs.clone() {
                w.insert([cx, cy, cz], blocks);
            }
        }
    }
    w
}

fn cubes_of(xs: Range<i32>, ys: Range<i32>, zs: Range<i32>) -> Vec<[i32; 3]> {
    let mut v = Vec::new();
    for x in xs {
        for y in ys.clone() {
            for z in zs.clone() {
                v.push([x, y, z]);
            }
        }
    }
    v
}

fn populate(
    world: &mut MapWorld,
    m: &Meteors,
    cubes: &[[i32; 3]],
    ground: &dyn Fn(i32, i32) -> i32,
) {
    for c in cubes {
        m.place(SectionPos::new(c[0], c[1], c[2]), world, ground);
    }
}

fn count(world: &MapWorld, state: u32) -> usize {
    world
        .cubes
        .values()
        .map(|c| c.iter().filter(|&&b| b == state).count())
        .sum()
}

fn kind(ore: &str, second: &str, weight: f64) -> MeteorKind {
    MeteorKind {
        ore: ore.into(),
        second: second.into(),
        weight,
    }
}

fn dense(per_square_km: f64) -> MeteorSettings {
    MeteorSettings {
        per_square_km,
        ..MeteorSettings::default()
    }
}

/// Every region has a surface meteor of diameter 25 of one kind.
fn giants(ore: &str, second: &str) -> MeteorSettings {
    MeteorSettings {
        per_square_km: 1000.0,
        min_diameter: 25,
        max_diameter: 25,
        surface_share: 1.0,
        sunk_share: 0.0,
        buried_share: 0.0,
        kinds: vec![kind(ore, second, 1.0)],
        ..MeteorSettings::default()
    }
}

/// The meteor of region (0, 0) of a world of giants, placed.
fn giant(seed: i64, ore: &str, second: &str) -> (MapWorld, MeteorPlan, [i32; 3]) {
    let m = Meteors::new(seed, &giants(ore, second), reg()).expect("meteors");
    let plan = m.plan(0, 0, &flat).expect("a meteor in region (0, 0)");
    let mut w = flat_world(0..2, -1..8, 0..2);
    populate(&mut w, &m, &cubes_of(0..2, -1..8, 0..2), &flat);
    let centre = [plan.x, plan.centre_y(64), plan.z];
    (w, plan, centre)
}

/// A world that records the positions written.
struct Recording<'a> {
    inner: &'a mut MapWorld,
    writes: Vec<[i32; 3]>,
}

impl CubeWorld for Recording<'_> {
    fn block(&self, p: [i32; 3]) -> u32 {
        self.inner.block(p)
    }
    fn set_block(&mut self, p: [i32; 3], state: u32) {
        self.writes.push(p);
        self.inner.set_block(p, state);
    }
}

fn in_window(p: [i32; 3], cube: [i32; 3]) -> bool {
    (0..3).all(|i| p[i] >= cube[i] * 16 && p[i] <= cube[i] * 16 + 31)
}

#[test]
fn whole_across_cube_boundaries() {
    let m = Meteors::new(11, &dense(600.0), reg()).expect("meteors");
    let (xs, ys, zs) = (0..16, -2..7, 0..16);
    let cubes = cubes_of(xs.clone(), ys.clone(), zs.clone());

    // The same world whatever the population order.
    let mut ascending = flat_world(xs.clone(), ys.clone(), zs.clone());
    populate(&mut ascending, &m, &cubes, &flat);
    let mut shuffled_cubes = cubes.clone();
    shuffled_cubes.sort_by_key(|c| hash_pos(99, 1, c[0], c[1], c[2]));
    let mut shuffled = flat_world(xs.clone(), ys.clone(), zs.clone());
    populate(&mut shuffled, &m, &shuffled_cubes, &flat);
    let mut reversed = flat_world(xs.clone(), ys.clone(), zs.clone());
    let reversed_cubes: Vec<_> = cubes.iter().rev().copied().collect();
    populate(&mut reversed, &m, &reversed_cubes, &flat);
    assert!(
        ascending.cubes == shuffled.cubes,
        "ascending and shuffled order differ"
    );
    assert!(
        ascending.cubes == reversed.cubes,
        "ascending and reversed order differ"
    );
    assert!(ascending.cubes != flat_world(xs.clone(), ys.clone(), zs.clone()).cubes);
    drop((ascending, shuffled, reversed));

    // Per region exactly the owner cube writes.
    let mut w = flat_world(xs, ys.clone(), zs);
    let mut meteors = 0;
    for rx in 0..8 {
        for rz in 0..8 {
            let plan = m.plan(rx, rz, &flat);
            let mut writers = Vec::new();
            for cy in ys.clone() {
                // Both cubes of the region column are checked, odd ones too.
                for (cx, cz) in [(2 * rx, 2 * rz), (2 * rx + 1, 2 * rz), (2 * rx, 2 * rz + 1)] {
                    let mut rec = Recording {
                        inner: &mut w,
                        writes: Vec::new(),
                    };
                    m.place(SectionPos::new(cx, cy, cz), &mut rec, &flat);
                    if !rec.writes.is_empty() {
                        writers.push([cx, cy, cz]);
                    }
                }
            }
            match plan {
                None => assert!(writers.is_empty(), "region {rx},{rz} has no meteor"),
                Some(p) => {
                    meteors += 1;
                    assert_eq!(
                        writers,
                        vec![[2 * rx, p.owner_cy, 2 * rz]],
                        "region {rx},{rz}: only the owner cube writes"
                    );
                }
            }
        }
    }
    assert!(meteors >= 20, "only {meteors} meteors in 64 regions");
}

#[test]
fn writes_stay_in_window() {
    let cubes = cubes_of(0..8, -3..8, 0..8);
    let mut total = 0;
    for seed in [1_i64, 2, 3, 4] {
        let m = Meteors::new(seed, &dense(1000.0), reg()).expect("meteors");
        for off in [-20, 0, 20] {
            let ground = move |_x: i32, _z: i32| 64 + off;
            let mut w = flat_world(0..8, -3..8, 0..8);
            for c in &cubes {
                let mut rec = Recording {
                    inner: &mut w,
                    writes: Vec::new(),
                };
                m.place(SectionPos::new(c[0], c[1], c[2]), &mut rec, &ground);
                for p in &rec.writes {
                    assert!(
                        in_window(*p, *c),
                        "seed {seed} off {off}: {p:?} outside {c:?}"
                    );
                }
                total += rec.writes.len();
            }
        }
    }
    assert!(total > 10_000, "only {total} writes");
}

#[test]
fn deterministic() {
    let cubes = cubes_of(0..8, -2..7, 0..8);
    let build = |seed: i64| {
        let m = Meteors::new(seed, &dense(1000.0), reg()).expect("meteors");
        let mut w = flat_world(0..8, -2..7, 0..8);
        populate(&mut w, &m, &cubes, &flat);
        w
    };
    let a = build(5);
    assert!(a.cubes == build(5).cubes, "same seed, different world");
    assert!(a.cubes != build(6).cubes, "different seed, same world");

    // A region without a meteor and the odd cubes never ask for the ground.
    let m = Meteors::new(5, &dense(5.0), reg()).expect("meteors");
    let calls = Cell::new(0);
    let ground = |_x: i32, _z: i32| {
        calls.set(calls.get() + 1);
        64
    };
    let mut w = flat_world(0..8, 0..8, 0..8);
    let empty = (0..16)
        .find(|&rx| m.plan(rx, 0, &flat).is_none())
        .expect("a region without a meteor");
    for cy in 0..8 {
        m.place(SectionPos::new(2 * empty, cy, 0), &mut w, &ground);
    }
    for c in cubes_of(1..8, 0..8, 1..8)
        .iter()
        .filter(|c| c[0] % 2 == 1 || c[2] % 2 == 1)
    {
        m.place(SectionPos::new(c[0], c[1], c[2]), &mut w, &ground);
    }
    assert_eq!(calls.get(), 0, "ground() called without a meteor to place");
}

#[test]
fn layers_in_order() {
    let (iron, copper) = (id("iron_ore"), id("copper_ore"));
    let (raw, deep, fleck) = (
        id("raw_iron_block"),
        id("deepslate"),
        id("deepslate_iron_ore"),
    );
    let stage = |b: u32| -> Option<u8> {
        if b == raw {
            Some(0)
        } else if b == iron || b == copper {
            Some(1)
        } else if b == deep || b == fleck {
            Some(2)
        } else {
            None
        }
    };

    let (w, plan, c) = giant(3, "iron", "copper");
    assert_eq!(plan.placement, Placement::Surface);
    assert_eq!(w.block(c), raw, "the centre block is the core");

    // Rays along the axes: core, ore layer, shell, then the outside.
    for dir in [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ] {
        let mut last = 0;
        let mut length = 0;
        for k in 0..20 {
            let p = [c[0] + k * dir[0], c[1] + k * dir[1], c[2] + k * dir[2]];
            let Some(s) = stage(w.block(p)) else { break };
            assert!(s >= last, "layers out of order along {dir:?} at {k}");
            last = s;
            length = k;
        }
        assert_eq!(last, 2, "the outermost block along {dir:?} is shell");
        assert!(length >= 10, "ray {dir:?} too short: {length}");
    }

    // Every body block with an outside neighbour is shell; flecks exist.
    let (mut shell, mut flecks, mut cores, mut ore_layer) = (0, 0, 0, 0);
    for x in c[0] - 16..=c[0] + 16 {
        for y in c[1] - 16..=c[1] + 16 {
            for z in c[2] - 16..=c[2] + 16 {
                let Some(s) = stage(w.block([x, y, z])) else {
                    continue;
                };
                match s {
                    0 => cores += 1,
                    1 => ore_layer += 1,
                    _ => shell += 1,
                }
                if w.block([x, y, z]) == fleck {
                    flecks += 1;
                }
                let outer = [
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                ]
                .iter()
                .any(|d| stage(w.block([x + d[0], y + d[1], z + d[2]])).is_none());
                if outer {
                    assert_eq!(s, 2, "outer block {x},{y},{z} is not shell");
                }
            }
        }
    }
    assert!(
        cores > 20 && ore_layer > 1000 && shell > 500,
        "{cores} {ore_layer} {shell}"
    );
    assert!(flecks > 0 && flecks < shell, "{flecks} flecks of {shell}");
    println!("giant: core {cores}, ore layer {ore_layer}, shell {shell} (flecks {flecks})");

    // Other kinds: the core is `<ore>_block` where there is no raw block.
    let (w, _, c) = giant(4, "diamond", "lapis");
    assert_eq!(w.block(c), id("diamond_block"));
    let (w, _, c) = giant(4, "coal", "iron");
    assert_eq!(w.block(c), id("coal_block"));
    let (w, _, c) = giant(4, "gold", "redstone");
    assert_eq!(w.block(c), id("raw_gold_block"));
}

#[test]
fn mix_share_near_a_fifth() {
    let m = Meteors::new(21, &giants("iron", "copper"), reg()).expect("meteors");
    let mut w = flat_world(0..8, -1..8, 0..8);
    populate(&mut w, &m, &cubes_of(0..8, -1..8, 0..8), &flat);
    let (iron, copper) = (count(&w, id("iron_ore")), count(&w, id("copper_ore")));
    let share = copper as f64 / (iron + copper) as f64;
    println!("mix share: {copper} copper / {iron} iron = {share:.4} (16 giants)");
    assert!(iron + copper > 50_000);
    assert!((share - 0.2).abs() < 0.03, "mix share {share}");
}

#[test]
fn sizes_follow_the_curve() {
    let s = dense(1000.0);
    let m = Meteors::new(8, &s, reg()).expect("meteors");
    let n = 200;
    let mut hist = [0usize; 26];
    for rx in 0..n {
        for rz in 0..n {
            let p = m.plan(rx, rz, &flat).expect("every region has one");
            assert!((5..=25).contains(&p.diameter), "diameter {}", p.diameter);
            hist[p.diameter as usize] += 1;
            // The whole footprint stays inside the region.
            let (x, z) = (f64::from(p.x) + 0.5, f64::from(p.z) + 0.5);
            let (x0, z0) = (f64::from(rx * 32), f64::from(rz * 32));
            assert!(x - p.footprint >= x0 && x + p.footprint <= x0 + 32.0);
            assert!(z - p.footprint >= z0 && z + p.footprint <= z0 + 32.0);
        }
    }
    let total = (n * n) as usize;
    let peak = (5..=25).max_by_key(|&d| hist[d]).expect("sizes");
    assert!((6..=8).contains(&peak), "histogram peaks at {peak}");
    let big: usize = hist[20..].iter().sum();
    assert!(big > 0, "no meteor of 20 or more");

    // Expected frequencies from the spline and the floor.
    let weights: Vec<f64> = (5..=25)
        .map(|d| s.size_curve.get(f64::from(d)).max(0.0))
        .collect();
    let top = weights.iter().copied().fold(0.0, f64::max);
    let weights: Vec<f64> = weights.iter().map(|w| w.max(s.size_floor * top)).collect();
    let sum: f64 = weights.iter().sum();
    let mut line = String::new();
    for d in 5..=25usize {
        let expected = weights[d - 5] / sum;
        let seen = hist[d] as f64 / total as f64;
        line += &format!("{d}:{seen:.4}/{expected:.4} ");
        if expected >= 0.03 {
            assert!(
                (seen / expected - 1.0).abs() < 0.15,
                "size {d}: seen {seen}, expected {expected}"
            );
        }
    }
    println!("sizes (seen/expected over {total} regions): {line}");
    println!("peak {peak}, {big} of {total} are 20 or more");
}

#[test]
fn placement_shares_and_depths() {
    let m = Meteors::new(9, &dense(1000.0), reg()).expect("meteors");
    let n = 200;
    let mut shares = [0usize; 3];
    for rx in 0..n {
        for rz in 0..n {
            let p = m.plan(rx, rz, &flat).expect("every region has one");
            let r = p.radius();
            let (centre, ground) = (f64::from(p.y_est) + 0.5, 64.0);
            match p.placement {
                Placement::Surface => {
                    shares[0] += 1;
                    assert!(centre >= ground - 0.1 * r - 1e-9, "surface centre {centre}");
                    assert!(centre - 0.9 * r <= ground + 1.0, "surface lowest {centre}");
                }
                Placement::Sunk => {
                    shares[1] += 1;
                    assert!(
                        centre <= ground - 0.4 * r + 1.0 + 1e-9,
                        "sunk centre {centre}"
                    );
                    assert!(
                        centre >= ground - 0.9 * r - 0.5 - 1e-9,
                        "sunk centre {centre}"
                    );
                }
                Placement::Buried => {
                    shares[2] += 1;
                    let depth = ground - f64::from(p.y_est);
                    assert!(depth >= r + 3.0, "buried depth {depth} for radius {r}");
                    assert!(depth <= 50.5, "buried depth {depth}");
                    assert!(centre + p.r_out() <= ground - 2.0, "buried top {centre}");
                }
            }
        }
    }
    let total = (n * n) as f64;
    let seen: Vec<f64> = shares.iter().map(|&c| c as f64 / total).collect();
    println!("placement shares surface/sunk/buried: {seen:.4?}");
    for (s, want) in seen.iter().zip([0.6, 0.25, 0.15]) {
        assert!((s - want).abs() < 0.02, "share {s}, wanted {want}");
    }

    // Placed buried meteors leave the surface alone.
    let s = MeteorSettings {
        per_square_km: 1000.0,
        surface_share: 0.0,
        sunk_share: 0.0,
        buried_share: 1.0,
        ..MeteorSettings::default()
    };
    let m = Meteors::new(9, &s, reg()).expect("meteors");
    let mut w = flat_world(0..8, -2..8, 0..8);
    populate(&mut w, &m, &cubes_of(0..8, -2..8, 0..8), &flat);
    for x in 0..128 {
        for z in 0..128 {
            assert_eq!(w.block([x, 64, z]), id("grass_block"), "surface at {x},{z}");
            assert_eq!(w.block([x, 65, z]), id("air"));
        }
    }
    let ore = count(&w, id("iron_ore")) + count(&w, id("copper_ore")) + count(&w, id("coal_ore"));
    assert!(ore > 0, "no buried ore at all");
}

#[test]
fn off_switches() {
    let cubes = cubes_of(0..4, -1..7, 0..4);
    let plain = flat_world(0..4, -1..7, 0..4);
    let off = [
        MeteorSettings {
            enabled: false,
            per_square_km: 1000.0,
            ..MeteorSettings::default()
        },
        dense(0.0),
        MeteorSettings {
            per_square_km: 1000.0,
            kinds: vec![kind("iron", "", 0.0), kind("coal", "", 0.0)],
            ..MeteorSettings::default()
        },
        MeteorSettings {
            per_square_km: 1000.0,
            kinds: Vec::new(),
            ..MeteorSettings::default()
        },
    ];
    for (i, s) in off.iter().enumerate() {
        let m = Meteors::new(1, s, reg()).expect("meteors");
        assert!(!m.is_active(), "settings {i} should be off");
        let mut w = flat_world(0..4, -1..7, 0..4);
        populate(&mut w, &m, &cubes, &flat);
        assert!(w.cubes == plain.cubes, "settings {i} wrote blocks");
    }
    // Disabled settings do not need their blocks.
    let bare = registry_of(&default_block_names());
    assert!(Meteors::new(1, &off[0], bare).is_ok());
    assert!(
        Meteors::new(1, &MeteorSettings::default(), reg())
            .expect("meteors")
            .is_active()
    );
}

#[test]
fn validate_settings() {
    assert!(MeteorSettings::default().validate().is_ok());
    let mut none = MeteorSettings::default();
    none.kinds.clear();
    assert!(none.validate().is_ok(), "no kinds is valid (inactive)");

    type Edit = fn(&mut MeteorSettings);
    let bad: Vec<(&str, Edit)> = vec![
        ("minDiameter", |s| s.min_diameter = 2),
        ("minDiameter", |s| s.min_diameter = 30),
        ("maxDiameter", |s| s.max_diameter = 26),
        ("maxDiameter", |s| s.max_diameter = 2),
        ("minDiameter", |s| {
            s.min_diameter = 20;
            s.max_diameter = 10;
        }),
        ("perSquareKm", |s| s.per_square_km = -1.0),
        ("perSquareKm", |s| s.per_square_km = 1001.0),
        ("perSquareKm", |s| s.per_square_km = f64::NAN),
        ("sizeFloor", |s| s.size_floor = -0.1),
        ("sizeFloor", |s| s.size_floor = 1.5),
        ("sizeCurve", |s| s.size_curve.0[0].v = f64::INFINITY),
        ("sizeCurve", |s| s.size_curve.0[1].y = f64::NAN),
        ("surfaceShare", |s| s.surface_share = -0.1),
        ("sunkShare", |s| s.sunk_share = f64::NAN),
        ("buriedShare", |s| s.buried_share = f64::INFINITY),
        ("are all 0", |s| {
            s.surface_share = 0.0;
            s.sunk_share = 0.0;
            s.buried_share = 0.0;
        }),
        ("buriedDepth", |s| s.buried_depth = -1.0),
        ("buriedDepth", |s| s.buried_depth = 1001.0),
        ("buriedDepth", |s| s.buried_depth = f64::NAN),
        ("mixShare", |s| s.mix_share = 1.1),
        ("mixShare", |s| s.mix_share = -0.1),
        ("weight", |s| s.kinds[0].weight = -1.0),
        ("weight", |s| s.kinds[1].weight = f64::NAN),
        ("ore is empty", |s| s.kinds[2].ore = String::new()),
    ];
    for (i, (name, edit)) in bad.iter().enumerate() {
        let mut s = MeteorSettings::default();
        edit(&mut s);
        let err = s.validate().expect_err(&format!("case {i} should fail"));
        assert!(err.contains(name), "case {i}: {err:?} does not name {name}");
        // `new` refuses them too (when enabled).
        assert!(Meteors::new(1, &s, reg()).is_err(), "case {i}");
    }
    // A registry without the raw blocks: the error names the block.
    let err = Meteors::new(
        1,
        &MeteorSettings::default(),
        registry_of(&default_block_names()),
    )
    .err()
    .expect("missing blocks");
    assert!(err.contains("raw_iron_block"), "{err}");
}

/// A surface meteor of the given diameter in region (0, 0), placed on the
/// flat world.
fn surface_meteor(seed: i64, diameter: i32) -> (MapWorld, MeteorPlan, [i32; 3]) {
    let settings = MeteorSettings {
        min_diameter: diameter,
        max_diameter: diameter,
        ..giants("iron", "copper")
    };
    let m = Meteors::new(seed, &settings, reg()).expect("meteors");
    let plan = m.plan(0, 0, &flat).expect("a meteor in region (0, 0)");
    let mut w = flat_world(0..2, -1..8, 0..2);
    populate(&mut w, &m, &cubes_of(0..2, -1..8, 0..2), &flat);
    let centre = [plan.x, plan.centre_y(64), plan.z];
    (w, plan, centre)
}

#[test]
fn no_air_under_the_body() {
    let body: Vec<u32> = [
        "raw_iron_block",
        "iron_ore",
        "copper_ore",
        "deepslate",
        "deepslate_iron_ore",
    ]
    .iter()
    .map(|n| id(n))
    .collect();
    let air = id("air");
    let mut checked = 0;
    for (seed, diameter) in [
        (7, 25),
        (3, 25),
        (11, 25),
        (1, 9),
        (2, 11),
        (5, 13),
        (8, 15),
    ] {
        let (w, plan, c) = surface_meteor(seed, diameter);
        assert_eq!(plan.placement, Placement::Surface);
        let reach = plan.r_out().ceil() as i32;
        for x in c[0] - reach..=c[0] + reach {
            for z in c[2] - reach..=c[2] + reach {
                // The lowest body block of the column.
                let lowest =
                    (c[1] - reach..=c[1] + reach).find(|&y| body.contains(&w.block([x, y, z])));
                let Some(y) = lowest else { continue };
                if y > 65 {
                    continue;
                }
                checked += 1;
                assert_ne!(
                    w.block([x, y - 1, z]),
                    air,
                    "air under the body at {x},{y},{z} (d {diameter}, seed {seed}, centre {c:?})"
                );
            }
        }
    }
    assert!(checked > 200, "only {checked} columns checked");
}

#[test]
fn cross_section() {
    let (w, plan, c) = giant(7, "iron", "copper");
    let (iron, copper) = (id("iron_ore"), id("copper_ore"));
    let glyph = |b: u32| -> char {
        match b {
            _ if b == id("deepslate") => '#',
            _ if b == id("deepslate_iron_ore") => 'f',
            _ if b == iron => 'i',
            _ if b == copper => 'c',
            _ if b == id("raw_iron_block") => 'R',
            _ if b == id("stone") => 'S',
            _ if b == id("dirt") => 'd',
            _ if b == id("grass_block") => 'g',
            _ if b == id("air") => '.',
            _ => '?',
        }
    };
    println!(
        "ore meteor: diameter {}, {:?}, centre {:?}, crater radius {:.1}, owner cube y {}",
        plan.diameter, plan.placement, c, plan.crater_radius, plan.owner_cy
    );
    println!(
        "x-y slice through z = {} (x {}..={}, y {} at the top)",
        c[2],
        c[0] - 18,
        c[0] + 18,
        c[1] + 17
    );
    println!("# deepslate  f deepslate_iron_ore  i iron_ore  c copper_ore  R raw_iron_block");
    println!("S stone  d dirt  g grass_block  . air");
    let mut text = String::new();
    for y in (c[1] - 17..=c[1] + 17).rev() {
        let row: String = (c[0] - 18..=c[0] + 18)
            .map(|x| glyph(w.block([x, y, c[2]])))
            .collect();
        println!("{y:4} {row}");
        text += &row;
    }
    assert!(text.contains('R') && text.contains('#') && text.contains('i'));
    assert!(text.contains('g') && text.contains('S'));
    assert!(!text.contains('?'));
}

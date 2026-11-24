//! Tests of the ore engine on synthetic worlds, plus the legacy-exactness
//! tests: the engine fed `OreGenSettings::from_legacy` places exactly the
//! blocks the old `standardOres` / `periodicGaussianOres` code did.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use horizon_core::position::SectionPos;

use super::*;
use crate::cubic::custom::populate::CubeWorld;
use crate::cubic::custom::rng::{JRandom, coords_seed_random};
use crate::cubic::custom::settings::{
    BiomeDesc, BlockStateDesc, CustomGeneratorSettings, PeriodicGaussianOreConfig,
    StandardOreConfig,
};
use crate::cubic::custom::testing::{MapWorld, default_registry};
use crate::cubic::custom::{CustomCubicGenerator, FixedBiome};
use crate::registry::BlockRegistry;

const SEED: i64 = 12345;
const PLAINS: &str = "minecraft:plains";

fn reg() -> &'static BlockRegistry {
    static REG: OnceLock<&'static BlockRegistry> = OnceLock::new();
    REG.get_or_init(default_registry)
}

fn id(name: &str) -> u32 {
    reg()
        .default_state(&format!("minecraft:{name}"))
        .unwrap_or_else(|| panic!("no block {name}"))
        .0
}

fn ground0(_: i32, _: i32) -> i32 {
    0
}
fn no_ground(_: i32, _: i32) -> i32 {
    panic!("the ground must not be asked for");
}

/// A tier-1 flat static entry over `min..=max` with `intensity` veins.
fn entry(block: &str, size: i32, min: f64, max: f64, intensity: f64) -> OreEntry {
    OreEntry {
        blockstate: BlockStateDesc::new(block),
        spawn_size: size,
        rich: RichZone {
            min,
            max,
            intensity,
            ..RichZone::default()
        },
        ..OreEntry::default()
    }
}

fn background(rate: f64) -> Background {
    Background {
        enabled: true,
        rate,
        ..Background::default()
    }
}

fn settings(entries: Vec<OreEntry>) -> OreGenSettings {
    OreGenSettings {
        ores: entries,
        ..OreGenSettings::default()
    }
}

fn engine(entries: Vec<OreEntry>) -> OreEngine {
    OreEngine::new(SEED, &settings(entries), reg()).unwrap()
}

/// A window (the 2 x 2 x 2 cubes from `cube` up) filled by `f(position)`.
fn world_of(cube: [i32; 3], f: impl Fn([i32; 3]) -> u32) -> MapWorld {
    let mut w = MapWorld::new(0);
    for d in 0..8 {
        let c = [
            cube[0] + (d & 1),
            cube[1] + ((d >> 1) & 1),
            cube[2] + (d >> 2),
        ];
        let mut blocks = [0_u32; 4096];
        for (i, b) in blocks.iter_mut().enumerate() {
            let i = i as i32;
            *b = f([
                c[0] * 16 + (i & 15),
                c[1] * 16 + (i >> 8),
                c[2] * 16 + ((i >> 4) & 15),
            ]);
        }
        w.insert(c, blocks);
    }
    w
}

fn filled(cube: [i32; 3], block: &str) -> MapWorld {
    let mut w = MapWorld::new(0);
    for d in 0..8 {
        w.insert(
            [
                cube[0] + (d & 1),
                cube[1] + ((d >> 1) & 1),
                cube[2] + (d >> 2),
            ],
            [id(block); 4096],
        );
    }
    w
}

/// Every position whose block satisfies `f`.
fn positions(w: &MapWorld, f: impl Fn(u32) -> bool) -> Vec<[i32; 3]> {
    let mut out = Vec::new();
    for (c, blocks) in &w.cubes {
        for (i, &b) in blocks.iter().enumerate() {
            if f(b) {
                let i = i as i32;
                out.push([
                    c[0] * 16 + (i & 15),
                    c[1] * 16 + (i >> 8),
                    c[2] * 16 + ((i >> 4) & 15),
                ]);
            }
        }
    }
    out.sort_unstable();
    out
}

fn is_ore(b: u32) -> bool {
    static ORES: OnceLock<Vec<u32>> = OnceLock::new();
    ORES.get_or_init(|| {
        crate::cubic::custom::testing::default_block_names()
            .iter()
            .filter(|n| n.ends_with("_ore"))
            .map(|n| reg().default_state(n).unwrap().0)
            .collect()
    })
    .contains(&b)
}

fn run(
    e: &OreEngine,
    cube: [i32; 3],
    w: &mut MapWorld,
    biome: &str,
    ground: &dyn Fn(i32, i32) -> i32,
    lat_lon: Option<&dyn Fn(i32, i32) -> (f64, f64)>,
) -> JRandom {
    let mut rand = coords_seed_random(SEED, cube[0], cube[1], cube[2]);
    let ctx = OreCtx {
        biome,
        ground,
        lat_lon,
    };
    e.place(
        SectionPos::new(cube[0], cube[1], cube[2]),
        w,
        &mut rand,
        &ctx,
    );
    rand
}

/// The ore positions after populating each of `cubes` in a fresh stone world.
fn scan_with(
    e: &OreEngine,
    cubes: &[[i32; 3]],
    biome: &str,
    ground: &dyn Fn(i32, i32) -> i32,
    lat_lon: Option<&dyn Fn(i32, i32) -> (f64, f64)>,
) -> Vec<[i32; 3]> {
    let mut all = Vec::new();
    for &c in cubes {
        let mut w = filled(c, "stone");
        run(e, c, &mut w, biome, ground, lat_lon);
        all.extend(positions(&w, is_ore));
    }
    all
}

fn scan(e: &OreEngine, cubes: &[[i32; 3]]) -> Vec<[i32; 3]> {
    scan_with(e, cubes, PLAINS, &ground0, None)
}

/// Cubes at the heights `cys` over a few columns.
fn column_cubes(cys: impl Iterator<Item = i32>, columns: &[(i32, i32)]) -> Vec<[i32; 3]> {
    cys.flat_map(|cy| columns.iter().map(move |&(x, z)| [x, cy, z]))
        .collect()
}

const COLS: [(i32, i32); 4] = [(0, 0), (3, -2), (-5, 7), (11, 11)];

fn ys(p: &[[i32; 3]]) -> (i32, i32) {
    (
        p.iter().map(|p| p[1]).min().unwrap(),
        p.iter().map(|p| p[1]).max().unwrap(),
    )
}

// --- tier 1 -----------------------------------------------------------------

#[test]
fn a_static_band_places_ores_only_inside_it() {
    let e = engine(vec![entry("coal_ore", 8, 0.0, 40.0, 6.0)]);
    // The ground is never asked for by a static entry.
    let near = column_cubes(-2..5, &COLS);
    let found = scan_with(&e, &near, PLAINS, &no_ground, None);
    assert!(found.len() > 100, "{}", found.len());
    // A vein reaches a few blocks past the position it starts at.
    let (lo, hi) = ys(&found);
    assert!(lo >= -4 && hi <= 44, "{lo}..{hi}");
    // Windows whose positions are all well outside get nothing.
    let far = column_cubes((-8..-4).chain(6..10), &COLS);
    assert!(scan(&e, &far).is_empty());
}

#[test]
fn a_triangle_is_densest_at_its_peak() {
    let mut z = entry("coal_ore", 8, 0.0, 64.0, 20.0);
    z.rich.shape = Shape::Triangle;
    let e = engine(vec![z]);
    let columns: Vec<(i32, i32)> = (0..8).map(|i| (i * 3, -i)).collect();
    let mid = scan(&e, &column_cubes(1..2, &columns)).len(); // positions 24..=39
    let edge = scan(&e, &column_cubes(3..4, &columns)).len(); // positions 56..=64 only
    assert!(mid > 60 && mid > 3 * edge, "{mid} vs {edge}");
}

#[test]
fn a_below_ground_band_follows_a_high_and_a_low_ground() {
    let mut z = entry("coal_ore", 8, 10.0, 30.0, 8.0);
    z.rich.mode = RichMode::BelowGround;
    let e = engine(vec![z]);
    for (ground, lo, hi) in [(500, 466, 494), (-300, -334, -306)] {
        let g = move |_: i32, _: i32| ground;
        let band = ground >> 4;
        let cubes = column_cubes(band - 6..band + 2, &COLS);
        let found = scan_with(&e, &cubes, PLAINS, &g, None);
        assert!(found.len() > 60, "ground {ground}: {}", found.len());
        let (a, b) = ys(&found);
        assert!(a >= lo && b <= hi, "ground {ground}: {a}..{b}");
        // Nothing near Y 0 under a ground this far away.
        let far = column_cubes(-3..3, &COLS);
        assert!(scan_with(&e, &far, PLAINS, &g, None).is_empty());
    }
    // A ground that varies per column: each column's ores sit under its own ground.
    let g = |x: i32, _: i32| if x < 0 { 100 } else { 300 };
    let found = scan_with(
        &e,
        &column_cubes(0..30, &[(-4, 0), (4, 0)]),
        PLAINS,
        &g,
        None,
    );
    assert!(!found.is_empty());
    for p in found {
        let ground = if p[0] < 0 { 100 } else { 300 };
        let depth = ground - p[1];
        assert!((10 - 5..=30 + 5).contains(&depth), "{p:?} depth {depth}");
    }
}

fn noise_entry(threshold: f64, seed: i64) -> OreEntry {
    let mut z = entry("coal_ore", 3, f64::NEG_INFINITY, f64::INFINITY, 40.0);
    z.rich.mode = RichMode::Noise;
    z.rich.noise_scale = 48.0;
    z.rich.noise_threshold = threshold;
    z.rich.noise_seed = seed;
    z
}

#[test]
fn noise_pockets_hold_every_ore_and_follow_the_noise_seed() {
    let cubes = column_cubes(-3..3, &COLS);
    let (a, b) = (
        engine(vec![noise_entry(0.25, 0)]),
        engine(vec![noise_entry(0.25, 7)]),
    );
    let (pa, pb) = (scan(&a, &cubes), scan(&b, &cubes));
    assert!(pa.len() > 40 && pb.len() > 40, "{} {}", pa.len(), pb.len());
    // A small vein sits within about 3 blocks of its position; the field is smooth.
    for (e, found) in [(&a, &pa), (&b, &pb)] {
        for &p in found {
            let v = e.noise_value(0, p).unwrap();
            assert!(v > 0.25 - 0.2, "{p:?} noise {v}");
        }
    }
    // The other seed's noise has different pockets: about half of one seed's
    // ores are where the other seed's field is low.
    let low = pa
        .iter()
        .filter(|&&p| b.noise_value(0, p).unwrap() < 0.0)
        .count();
    assert!(low * 4 > pa.len(), "{low} of {}", pa.len());
    assert_ne!(pa, pb);
    // A threshold above any noise value: no pockets at all.
    assert!(scan(&engine(vec![noise_entry(5.0, 0)]), &cubes).is_empty());
    // The noise zone's Y limits apply.
    let mut z = noise_entry(0.25, 0);
    z.rich.min = 0.0;
    z.rich.max = 31.0;
    let limited = scan(&engine(vec![z]), &column_cubes(-4..6, &COLS));
    let (lo, hi) = ys(&limited);
    assert!(lo >= -2 && hi <= 33, "{lo}..{hi}");
}

#[test]
fn fractional_intensity_is_a_chance_and_the_result_is_deterministic() {
    let e = engine(vec![entry(
        "coal_ore",
        8,
        f64::NEG_INFINITY,
        f64::INFINITY,
        0.25,
    )]);
    let cubes = column_cubes(0..400, &[(0, 0)]);
    let n = cubes
        .iter()
        .filter(|&&c| {
            let mut w = filled(c, "stone");
            run(&e, c, &mut w, PLAINS, &ground0, None);
            !positions(&w, is_ore).is_empty()
        })
        .count();
    assert!((60..140).contains(&n), "{n} of 400 cubes");
    let (mut a, mut b) = (filled([1, 2, 3], "stone"), filled([1, 2, 3], "stone"));
    let e2 = engine(vec![entry(
        "coal_ore",
        8,
        f64::NEG_INFINITY,
        f64::INFINITY,
        40.5,
    )]);
    run(&e2, [1, 2, 3], &mut a, PLAINS, &ground0, None);
    run(&e2, [1, 2, 3], &mut b, PLAINS, &ground0, None);
    assert_eq!(a.cubes, b.cubes);
}

#[test]
fn own_random_modes_leave_the_population_random_alone() {
    let fresh = |c: [i32; 3]| coords_seed_random(SEED, c[0], c[1], c[2]).long();
    let cube = [2, -4, 5];
    let mut below = entry("coal_ore", 8, 0.0, 100.0, 7.5);
    below.rich.mode = RichMode::BelowGround;
    let mut tier2 = entry("iron_ore", 8, 0.0, 0.0, 0.0);
    tier2.rich.enabled = false;
    tier2.background = background(6.5);
    let mut noisy = noise_entry(-0.5, 3);
    noisy.rich.intensity = 12.5;
    let g = |_: i32, _: i32| -10;
    for e in [below, tier2, noisy] {
        let e = engine(vec![e]);
        let mut w = filled(cube, "stone");
        let mut r = run(&e, cube, &mut w, PLAINS, &g, None);
        assert!(!positions(&w, is_ore).is_empty());
        assert_eq!(r.long(), fresh(cube));
    }
    // A flat static entry is the legacy standard ore: it draws on `rand`.
    let e = engine(vec![entry("coal_ore", 8, -100.0, 100.0, 5.0)]);
    let mut w = filled(cube, "stone");
    let mut r = run(&e, cube, &mut w, PLAINS, &g, None);
    assert_ne!(r.long(), fresh(cube));
}

// --- tier 2 -----------------------------------------------------------------

#[test]
fn the_background_reaches_far_outside_the_bands_and_can_be_off() {
    let mut z = entry("iron_ore", 6, 0.0, 16.0, 1.0);
    z.background = background(5.0);
    let e = engine(vec![z.clone()]);
    let deep = column_cubes(-313..-311, &COLS); // Y about -5000
    let high = column_cubes(5000..5002, &COLS);
    assert!(scan(&e, &deep).len() > 100);
    assert!(scan(&e, &high).len() > 100);
    z.background.enabled = false;
    assert!(scan(&engine(vec![z.clone()]), &deep).is_empty());
    z.background.enabled = true;
    z.background.rate = 0.0;
    assert!(scan(&engine(vec![z.clone()]), &deep).is_empty());
    // With tier 1 off, the background stands alone.
    z.background.rate = 5.0;
    z.rich.enabled = false;
    let found = scan(&engine(vec![z]), &column_cubes(0..1, &COLS));
    assert!(found.len() > 50);
}

#[test]
fn the_background_leaves_the_rich_zone_to_tier_1() {
    // Tier 1 is practically off, so every ore inside the zone would be background.
    let mut z = entry("iron_ore", 6, -1000.0, 1000.0, 1e-12);
    z.background = background(30.0);
    let e = engine(vec![z]);
    assert!(scan(&e, &column_cubes(0..4, &COLS)).is_empty());
    assert!(scan(&e, &column_cubes(100..102, &COLS)).len() > 100);
    // Below ground: the zone is depth 0..=50, the background everywhere else.
    let mut z = entry("iron_ore", 3, 0.0, 50.0, 1e-12);
    z.rich.mode = RichMode::BelowGround;
    z.background = background(30.0);
    let e = engine(vec![z]);
    let g = |_: i32, _: i32| 100;
    let found = scan_with(&e, &column_cubes(0..12, &COLS), PLAINS, &g, None);
    assert!(found.len() > 100);
    for p in found {
        assert!(p[1] <= 100 - 50 + 4 || p[1] >= 100 - 4, "{p:?}");
    }
}

#[test]
fn a_background_range_limits_it_by_height_or_depth() {
    let mut z = entry("iron_ore", 6, 0.0, 0.0, 0.0);
    z.rich.enabled = false;
    z.background = background(20.0);
    z.background.range = RangeMode::StaticHeight;
    z.background.min = 20.0;
    z.background.max = 60.0;
    let e = engine(vec![z.clone()]);
    let found = scan(&e, &column_cubes(-4..8, &COLS));
    assert!(found.len() > 60);
    let (lo, hi) = ys(&found);
    assert!(lo >= 17 && hi <= 63, "{lo}..{hi}");
    z.background.range = RangeMode::BelowGround;
    z.background.min = 5.0;
    z.background.max = 25.0;
    let e = engine(vec![z]);
    let g = |_: i32, _: i32| 200;
    let found = scan_with(&e, &column_cubes(5..20, &COLS), PLAINS, &g, None);
    assert!(found.len() > 60);
    let (lo, hi) = ys(&found);
    assert!(lo >= 200 - 25 - 3 && hi <= 200 - 5 + 3, "{lo}..{hi}");
}

// --- switches and the replace rule -----------------------------------------------

#[test]
fn the_world_switch_turns_every_ore_off() {
    let mut z = entry("coal_ore", 8, -1000.0, 1000.0, 20.0);
    z.background = background(20.0);
    let mut s = settings(vec![z]);
    assert!(OreEngine::new(SEED, &s, reg()).unwrap().is_active());
    s.enabled = false;
    let e = OreEngine::new(SEED, &s, reg()).unwrap();
    assert!(!e.is_active());
    let mut w = filled([0, 0, 0], "stone");
    let mut r = run(&e, [0, 0, 0], &mut w, PLAINS, &ground0, None);
    assert!(positions(&w, is_ore).is_empty());
    assert_eq!(r.long(), coords_seed_random(SEED, 0, 0, 0).long());
    // No entries, or entries that cannot place, are not active.
    assert!(!engine(vec![]).is_active());
    let mut off = entry("coal_ore", 8, 0.0, 10.0, 5.0);
    off.rich.enabled = false;
    assert!(!engine(vec![off]).is_active());
}

#[test]
fn ores_only_replace_stone_and_deepslate() {
    let mut z = entry("coal_ore", 12, f64::NEG_INFINITY, f64::INFINITY, 20.0);
    z.background = background(10.0);
    let e = engine(vec![z]);
    let cube = [0, 3, 0];
    for block in [
        "air",
        "water",
        "lava",
        "dirt",
        "gravel",
        "sand",
        "grass_block",
        "bedrock",
        "coal_ore",
    ] {
        let mut w = filled(cube, block);
        run(&e, cube, &mut w, PLAINS, &ground0, None);
        let before = id(block);
        assert!(positions(&w, |b| b != before).is_empty(), "{block}");
    }
    // Mixed: only the stone-like cells may change, and they become the ore
    // (or its deepslate variant in deepslate).
    let kinds = [
        "stone",
        "dirt",
        "air",
        "deepslate",
        "water",
        "gravel",
        "granite",
        "tuff",
    ]
    .map(id);
    let pick = move |p: [i32; 3]| kinds[(p[0] * 7 + p[1] * 3 + p[2] * 5).rem_euclid(8) as usize];
    let before = world_of(cube, pick);
    let mut w = world_of(cube, pick);
    run(&e, cube, &mut w, PLAINS, &ground0, None);
    let (coal, deep_coal) = (id("coal_ore"), id("deepslate_coal_ore"));
    let (stone, deepslate) = ([id("stone"), id("granite")], [id("deepslate"), id("tuff")]);
    let mut changed = 0;
    for (c, blocks) in &w.cubes {
        for (i, &b) in blocks.iter().enumerate() {
            let was = before.cubes[c][i];
            if b != was {
                changed += 1;
                let ok = (stone.contains(&was) && b == coal)
                    || (deepslate.contains(&was) && b == deep_coal);
                assert!(ok, "{was} -> {b}");
            }
        }
    }
    assert!(changed > 50, "{changed}");
}

#[test]
fn deepslate_gets_the_deepslate_variant() {
    let mut a = entry("iron_ore", 10, f64::NEG_INFINITY, f64::INFINITY, 10.0);
    a.background = background(5.0);
    let patch = entry("dirt", 10, f64::NEG_INFINITY, f64::INFINITY, 10.0);
    let e = engine(vec![a, patch]);
    let cube = [1, 1, 1];
    for fill in ["deepslate", "tuff"] {
        let mut w = filled(cube, fill);
        run(&e, cube, &mut w, PLAINS, &ground0, None);
        let was = id(fill);
        let changed = positions(&w, |b| b != was);
        assert!(changed.len() > 50, "{fill}");
        // Iron becomes its deepslate variant; dirt has none.
        let (deep, dirt) = (id("deepslate_iron_ore"), id("dirt"));
        assert!(positions(&w, |b| b == deep).len() > 20);
        assert!(positions(&w, |b| b == id("iron_ore")).is_empty());
        assert!(positions(&w, |b| b != was && b != deep && b != dirt).is_empty());
    }
    for fill in ["stone", "andesite"] {
        let mut w = filled(cube, fill);
        run(&e, cube, &mut w, PLAINS, &ground0, None);
        assert!(positions(&w, |b| b == id("deepslate_iron_ore")).is_empty());
        assert!(!positions(&w, |b| b == id("iron_ore")).is_empty());
    }
}

// --- biomes and richness ------------------------------------------------------------

#[test]
fn the_biome_filter_and_multipliers_apply() {
    let wide = f64::INFINITY;
    let mut z = entry("coal_ore", 4, -wide, wide, 2.0);
    z.biomes = Some(vec![
        BiomeDesc::new("plains"),
        BiomeDesc::new("minecraft:forest"),
    ]);
    let e = engine(vec![z]);
    let cubes = column_cubes(0..30, &[(0, 0), (4, 4)]);
    let at = |e: &OreEngine, biome: &str| scan_with(e, &cubes, biome, &ground0, None).len();
    assert_eq!(at(&e, "minecraft:desert"), 0);
    assert!(at(&e, "minecraft:plains") > 30);
    assert!(at(&e, "minecraft:forest") > 30);

    let mut z = entry("coal_ore", 4, -wide, wide, 2.0);
    z.biome_multipliers.insert("desert".into(), 4.0); // no namespace
    z.biome_multipliers.insert("minecraft:jungle".into(), 0.0);
    let e = engine(vec![z]);
    let (plains, desert, jungle) = (
        at(&e, PLAINS),
        at(&e, "minecraft:desert"),
        at(&e, "minecraft:jungle"),
    );
    assert_eq!(jungle, 0);
    assert!(plains > 30);
    assert!(
        desert * 10 > plains * 28 && desert * 10 < plains * 50,
        "{desert} vs {plains}"
    );
    // A biome left out is 1.
    assert_eq!(at(&e, "minecraft:swamp"), plains);
}

fn four(_: &str, _: f64, _: f64) -> f64 {
    4.0
}
fn five(_: &str, _: f64, _: f64) -> f64 {
    5.0
}
fn hundred(_: &str, _: f64, _: f64) -> f64 {
    100.0
}
fn point_three(_: &str, _: f64, _: f64) -> f64 {
    0.3
}
fn hundredth(_: &str, _: f64, _: f64) -> f64 {
    0.01
}
fn only_iron(ore: &str, _: f64, _: f64) -> f64 {
    if ore == "iron" { 4.0 } else { 1.0 }
}
fn boom(ore: &str, _: f64, _: f64) -> f64 {
    panic!("richness asked for {ore}");
}

#[test]
fn the_richness_factor_clamps() {
    let f = OreEngine::richness_factor;
    assert_eq!(f(4.0, 1.0), 4.0);
    assert_eq!(f(100.0, 1.0), 5.0);
    assert_eq!(f(0.01, 1.0), 0.3);
    assert_eq!(f(0.0, 1.0), 0.3);
    assert_eq!(f(2.0, 2.0), 4.0);
    assert_eq!(f(1.5, 0.0), 1.0);
    assert_eq!(f(f64::NAN, 1.0), 1.0);
}

#[test]
fn richness_scales_the_veins_of_the_eight_ores() {
    let wide = f64::INFINITY;
    let cubes = column_cubes(0..40, &[(0, 0), (6, -3)]);
    let ll = |_: i32, _: i32| (12.0, -34.0);
    let build = |entries: Vec<OreEntry>, strength: f64, map: bool, f: RichnessFn| {
        let mut s = settings(entries);
        s.richness_strength = strength;
        s.richness_map = map;
        OreEngine::new(SEED, &s, reg()).unwrap().with_richness(f)
    };
    let coal = || vec![entry("coal_ore", 4, -wide, wide, 2.0)];
    let count = |e: &OreEngine, lat_lon: bool| {
        scan_with(
            e,
            &cubes,
            PLAINS,
            &ground0,
            lat_lon.then_some(&ll as &dyn Fn(_, _) -> _),
        )
    };
    let base = count(&build(coal(), 1.0, true, boom), false); // no lat_lon: never asked
    assert!(base.len() > 40);
    // Four times the veins.
    let rich = count(&build(coal(), 1.0, true, four), true);
    let ratio = rich.len() as f64 / base.len() as f64;
    assert!((3.0..5.0).contains(&ratio), "{ratio}");
    // Without the map's coordinates, with the map off, or at strength 0: as base.
    assert_eq!(count(&build(coal(), 1.0, true, four), false), base);
    assert_eq!(count(&build(coal(), 1.0, false, four), true), base);
    assert_eq!(count(&build(coal(), 0.0, true, four), true), base);
    // Strength 2 squares it: 16 clamps to 5.
    assert_eq!(
        count(&build(coal(), 2.0, true, four), true),
        count(&build(coal(), 1.0, true, five), true)
    );
    // The factor clamps to 0.3 ..= 5.
    assert_eq!(
        count(&build(coal(), 1.0, true, hundred), true),
        count(&build(coal(), 1.0, true, five), true)
    );
    assert_eq!(
        count(&build(coal(), 1.0, true, hundredth), true),
        count(&build(coal(), 1.0, true, point_three), true)
    );
    // Only the eight ores ask: granite never calls the function.
    let granite = vec![entry("granite", 6, -wide, wide, 3.0)];
    let e = build(granite, 1.0, true, boom);
    assert!(
        !count(&e, true).is_empty()
            || !positions(
                &{
                    let mut w = filled([0, 0, 0], "stone");
                    run(&e, [0, 0, 0], &mut w, PLAINS, &ground0, Some(&ll));
                    w
                },
                |b| b == id("granite")
            )
            .is_empty()
    );
    // The short name is the ore's: iron and deepslate iron ask for "iron".
    let by_ore = |block: &str, f: RichnessFn, ll: bool| {
        let e = build(vec![entry(block, 4, -wide, wide, 2.0)], 1.0, true, f);
        count(&e, ll)
    };
    for block in ["iron_ore", "deepslate_iron_ore"] {
        let plain = by_ore(block, only_iron, false).len();
        let rich = by_ore(block, only_iron, true).len();
        let ratio = rich as f64 / plain as f64;
        assert!((3.0..5.0).contains(&ratio), "{block} {ratio}");
        // Another ore with the same function stays at 1.
        let other = block.replace("iron", "copper");
        assert_eq!(
            by_ore(&other, only_iron, true),
            by_ore(&other, only_iron, false)
        );
    }
}

// --- legacy exactness -----------------------------------------------------------------

/// The legacy ore lists of 26.3's vanilla ores (`defaults()` itself carries
/// them as `oreGen` entries now).
fn legacy_lists() -> (Vec<StandardOreConfig>, Vec<PeriodicGaussianOreConfig>) {
    crate::cubic::custom::settings::legacy_vanilla_ores()
}

/// `legacy` has the lists and no `oreGen`; `ported` the same lists as `oreGen`
/// entries (meteors off).
fn legacy_and_ported() -> (CustomGeneratorSettings, CustomGeneratorSettings) {
    let (standard, periodic) = legacy_lists();
    let legacy = CustomGeneratorSettings {
        standard_ores: standard.clone(),
        periodic_gaussian_ores: periodic.clone(),
        ore_gen: None,
        ..CustomGeneratorSettings::defaults()
    };
    let ported = CustomGeneratorSettings {
        standard_ores: Vec::new(),
        periodic_gaussian_ores: Vec::new(),
        ore_gen: Some(OreGenSettings::from_legacy(
            &standard,
            &periodic,
            legacy.expected_base_height,
            legacy.expected_height_variation,
        )),
        ..legacy.clone()
    };
    (legacy, ported)
}

/// Generated cubes, kept between populations.
struct Terrain {
    g: CustomCubicGenerator,
    cache: HashMap<[i32; 3], Box<[u32; 4096]>>,
}

impl Terrain {
    fn world(&mut self, lo: [i32; 3], size: i32) -> MapWorld {
        let mut w = MapWorld::new(id("stone"));
        for dy in 0..size {
            for dz in 0..size {
                for dx in 0..size {
                    let c = [lo[0] + dx, lo[1] + dy, lo[2] + dz];
                    let g = &self.g;
                    let blocks = self.cache.entry(c).or_insert_with(|| {
                        let mut out = Box::new([0_u32; 4096]);
                        g.generate(SectionPos::new(c[0], c[1], c[2]), &mut out);
                        out
                    });
                    w.insert(c, **blocks);
                }
            }
        }
        w
    }
}

fn generator(s: &CustomGeneratorSettings, biome: &str) -> CustomCubicGenerator {
    CustomCubicGenerator::new(SEED, s, reg(), Arc::new(FixedBiome::new(biome))).unwrap()
}

#[test]
fn legacy_lists_and_their_oregen_entries_generate_identical_blocks() {
    let (legacy, ported) = legacy_and_ported();
    let mut ore_blocks = 0;
    let mut changed_blocks = 0;
    for (biome, columns, cys) in [
        (
            "plains",
            vec![(0, 0), (5, -3)],
            vec![-10, -6, -3, -1, 0, 1, 2, 4, 8, 15, 18],
        ),
        ("badlands", vec![(-2, 4)], vec![-6, -2, 0, 3]),
    ] {
        let (a, b) = (generator(&legacy, biome), generator(&ported, biome));
        let mut terrain = Terrain {
            g: generator(&legacy, biome),
            cache: HashMap::new(),
        };
        for &(cx, cz) in &columns {
            for &cy in &cys {
                // One populated cube.
                let mut wa = terrain.world([cx, cy, cz], 2);
                let mut wb = terrain.world([cx, cy, cz], 2);
                let base = terrain.world([cx, cy, cz], 2);
                let cube = SectionPos::new(cx, cy, cz);
                a.populate(cube, &mut wa);
                b.populate(cube, &mut wb);
                assert_eq!(wa.cubes, wb.cubes, "{biome} cube {cx} {cy} {cz}");
                ore_blocks += positions(&wa, is_ore).len();
                changed_blocks += positions(&wa, |_| true)
                    .iter()
                    .filter(|&&p| wa.block(p) != base.block(p))
                    .count();
            }
        }
    }
    // The comparison is not between two empty populations.
    assert!(ore_blocks > 300, "{ore_blocks}");
    assert!(changed_blocks > 1000, "{changed_blocks}");
}

#[test]
fn a_populated_two_by_two_by_two_window_is_identical_too() {
    let (legacy, ported) = legacy_and_ported();
    let (a, b) = (generator(&legacy, "plains"), generator(&ported, "plains"));
    let mut terrain = Terrain {
        g: generator(&legacy, "plains"),
        cache: HashMap::new(),
    };
    for lo in [[0, -5, 0], [-3, 0, 2]] {
        // Populate the 2 x 2 x 2 cubes from `lo` in order; they write into the 3 x 3 x 3.
        let mut wa = terrain.world(lo, 3);
        let mut wb = terrain.world(lo, 3);
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let cube = SectionPos::new(lo[0] + dx, lo[1] + dy, lo[2] + dz);
                    a.populate(cube, &mut wa);
                    b.populate(cube, &mut wb);
                }
            }
        }
        assert_eq!(wa.cubes, wb.cubes, "window {lo:?}");
        assert!(positions(&wa, is_ore).len() > 100);
    }
}

#[test]
fn a_cwg_json_without_oregen_is_an_old_preset() {
    let (legacy, ported) = legacy_and_ported();
    let json = legacy.to_json_compact();
    assert!(!json.contains("oreGen"));
    let parsed = CustomGeneratorSettings::from_json(&json).unwrap();
    assert_eq!(parsed.ore_gen, None);
    assert_eq!(parsed.standard_ores.len(), legacy.standard_ores.len());
    assert_eq!(
        parsed.periodic_gaussian_ores.len(),
        legacy.periodic_gaussian_ores.len()
    );
    // An old preset has no engine and no meteors.
    let g = generator(&parsed, "plains");
    let mut w = world_of([0, -4, 0], |_| id("stone"));
    g.populate(SectionPos::new(0, -4, 0), &mut w);
    assert!(!positions(&w, is_ore).is_empty());
    // A settings object with oreGen survives the round trip.
    let back = CustomGeneratorSettings::from_json(&ported.to_json_compact()).unwrap();
    assert_eq!(back.ore_gen, ported.ore_gen);
    assert!(back.ore_gen.as_ref().unwrap().ores.len() > 10);
    assert!(back.standard_ores.is_empty());
    assert!(ported.to_json_compact().contains("\"oreGen\""));
}

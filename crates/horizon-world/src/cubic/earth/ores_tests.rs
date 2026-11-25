//! Tests of Earth's ores ([`super::ores`]): the deepslate gradient under the real ground, ores and stone
//! patches by depth, the richness map, old worlds unchanged and the vegetation's random untouched.

use std::sync::Arc;

use horizon_core::position::SectionPos;
use horizon_earth::TileStore;
use horizon_earth::testing::{MemoryFetcher, TempDir};

use super::ores::{DEEPSLATE_FULL, DEEPSLATE_TOP, EarthOres};
use super::tests::{LAT0, LON0};
use super::*;
use crate::cubic::custom::CustomCubicGenerator;
use crate::cubic::custom::populate::CubeWorld;
use crate::cubic::custom::rng::{JRandom, coords_seed_random};
use crate::cubic::custom::testing::{MapWorld, default_registry};
use crate::cubic::ores::OreGenSettings;
use crate::registry::BlockStateId;

const SEED: i64 = 7_331;

fn id(name: &str) -> u32 {
    default_registry()
        .default_state(&format!("minecraft:{name}"))
        .map(|BlockStateId(i)| i)
        .unwrap_or_else(|| panic!("no {name}"))
}

const ORES: [&str; 8] = [
    "coal", "iron", "copper", "gold", "redstone", "diamond", "lapis", "emerald",
];

fn ore_ids(ore: &str) -> [u32; 2] {
    [
        id(&format!("{ore}_ore")),
        id(&format!("deepslate_{ore}_ore")),
    ]
}

/// Every ore block id (looked up once: the registry's lookups by name are slow in a debug build).
fn all_ore_ids() -> &'static std::collections::HashSet<u32> {
    static IDS: std::sync::OnceLock<std::collections::HashSet<u32>> = std::sync::OnceLock::new();
    IDS.get_or_init(|| ORES.iter().flat_map(|o| ore_ids(o)).collect())
}

fn is_any_ore(b: u32) -> bool {
    all_ore_ids().contains(&b)
}

/// Earth defaults without meteors (they are another part's; the tests here are about the ores).
fn ore_settings() -> OreGenSettings {
    let mut s = OreGenSettings::earth_defaults();
    s.meteors.enabled = false;
    s
}

/// An Earth generator over level ground `elevation_m` metres high, with `ores` (None = an old world).
struct Flat {
    generator: CustomCubicGenerator,
    ground: i32,
    _dir: TempDir,
}

fn flat(elevation_m: f64, ores: Option<EarthOres>) -> Flat {
    let dir = TempDir::new();
    let store = Arc::new(TileStore::new(
        dir.path(),
        Arc::new(MemoryFetcher::new(move |_, _| elevation_m)),
        256,
    ));
    let settings = EarthSettings::at(LAT0, LON0);
    let mut generator = CustomCubicGenerator::earth(&settings, store, default_registry())
        .unwrap()
        .with_seed(SEED);
    if let Some(ores) = ores {
        generator = generator.with_earth_ores(ores);
    }
    let ground = generator.surface_y(8, 8, i32::MAX).unwrap();
    Flat {
        generator,
        ground,
        _dir: dir,
    }
}

fn ores_of(s: &OreGenSettings) -> EarthOres {
    EarthOres::new(SEED, s, default_registry()).unwrap()
}

/// Generates cubes x, z in 0..=2 and y in `lo..=hi + 1`, then populates x, z in 0..2 and y in `lo..=hi`: every
/// block of x, z in 16..32 and y in (lo + 1) * 16 .. (hi + 1) * 16 has had all the windows that reach it.
fn populated(f: &Flat, lo: i32, hi: i32, world: &mut MapWorld) {
    for cx in 0..=2 {
        for cz in 0..=2 {
            for cy in lo..=hi + 1 {
                let mut blocks = [0_u32; 4096];
                f.generator
                    .generate(SectionPos::new(cx, cy, cz), &mut blocks);
                world.insert([cx, cy, cz], blocks);
            }
        }
    }
    for cy in lo..=hi {
        for cx in 0..2 {
            for cz in 0..2 {
                f.generator.populate(SectionPos::new(cx, cy, cz), world);
            }
        }
    }
}

/// The cubes from `deep` blocks under the ground up to `up` blocks above it.
fn span(ground: i32, deep: i32, up: i32) -> (i32, i32) {
    ((ground - deep) >> 4, (ground + up) >> 4)
}

/// (depth under the ground, block) of the fully populated blocks.
fn blocks_by_depth(world: &MapWorld, ground: i32, lo: i32, hi: i32) -> Vec<(i32, u32)> {
    let mut out = Vec::new();
    for x in 16..32 {
        for z in 16..32 {
            for y in (lo + 1) * 16..(hi + 1) * 16 {
                out.push((ground - y, world.block([x, y, z])));
            }
        }
    }
    out
}

#[test]
fn deepslate_grows_from_64_to_72_blocks_under_a_high_and_a_low_ground() {
    let mut off = ore_settings();
    off.enabled = false; // the gradient only
    for elevation in [300.0, 2500.0] {
        let f = flat(elevation, Some(ores_of(&off)));
        let (lo, hi) = span(f.ground, 110, 0);
        let mut world = MapWorld::new(0);
        populated(&f, lo, hi, &mut world);
        let (deepslate, stone) = (id("deepslate"), id("stone"));
        let (mut mixed_deep, mut mixed_stone) = (0, 0);
        for (depth, b) in blocks_by_depth(&world, f.ground, lo, hi) {
            if depth <= DEEPSLATE_TOP {
                assert_ne!(
                    b, deepslate,
                    "deepslate {depth} under the ground at {elevation} m"
                );
            } else if depth >= DEEPSLATE_FULL {
                assert_ne!(b, stone, "stone {depth} under the ground at {elevation} m");
                assert_eq!(b, deepslate, "{depth} under the ground at {elevation} m");
            } else if b == deepslate {
                mixed_deep += 1;
            } else if b == stone {
                mixed_stone += 1;
            }
        }
        assert!(
            mixed_deep > 100 && mixed_stone > 100,
            "a mix between: {mixed_deep} deepslate, {mixed_stone} stone at {elevation} m"
        );
    }
}

#[test]
fn earth_defaults_place_patches_and_ores_by_depth_and_old_worlds_get_none() {
    let f = flat(300.0, Some(ores_of(&ore_settings())));
    let (lo, hi) = span(f.ground, 230, 20);
    let mut world = MapWorld::new(0);
    populated(&f, lo, hi, &mut world);
    let blocks = blocks_by_depth(&world, f.ground, lo, hi);
    let count = |names: &[u32], depths: std::ops::Range<i32>| {
        blocks
            .iter()
            .filter(|(d, b)| depths.contains(d) && names.contains(b))
            .count()
    };
    // No ore above the ground.
    assert_eq!(
        blocks
            .iter()
            .filter(|(d, b)| *d < 0 && is_any_ore(*b))
            .count(),
        0
    );
    // Coal near the top, diamonds deep (as deepslate diamonds under 72).
    let coal = ore_ids("coal");
    let (coal_top, coal_deep) = (count(&coal, 0..64), count(&coal, 130..194));
    assert!(
        coal_top > 4 * coal_deep.max(1),
        "coal {coal_top} near the top, {coal_deep} deep"
    );
    let diamond = ore_ids("diamond");
    let (diamond_shallow, diamond_band) = (count(&diamond, 0..48), count(&diamond, 48..208));
    assert!(
        diamond_band > 3 * diamond_shallow.max(1),
        "diamonds {diamond_band} in 48..208, {diamond_shallow} above"
    );
    assert!(count(&[id("deepslate_diamond_ore")], 72..208) > 0);
    // Stone patches.
    for patch in ["granite", "diorite", "andesite", "tuff", "gravel"] {
        assert!(count(&[id(patch)], 0..230) > 0, "{patch}");
    }
    // An old world (no oreGen): plain stone, no ores, no deepslate.
    let old = flat(300.0, None);
    let mut world = MapWorld::new(0);
    populated(&old, lo, hi, &mut world);
    let blocks = blocks_by_depth(&world, old.ground, lo, hi);
    let (deepslate, granite) = (id("deepslate"), id("granite"));
    assert!(
        blocks
            .iter()
            .all(|(_, b)| !is_any_ore(*b) && *b != deepslate && *b != granite)
    );
}

fn five(_: &str, _: f64, _: f64) -> f64 {
    5.0
}
fn tenth(_: &str, _: f64, _: f64) -> f64 {
    0.1
}

#[test]
fn the_richness_map_scales_earths_ores() {
    let mut counts = Vec::new();
    for richness in [five as fn(&str, f64, f64) -> f64, tenth] {
        let f = flat(
            300.0,
            Some(ores_of(&ore_settings()).with_richness(richness)),
        );
        let (lo, hi) = span(f.ground, 160, 0);
        let mut world = MapWorld::new(0);
        populated(&f, lo, hi, &mut world);
        let blocks = blocks_by_depth(&world, f.ground, lo, hi);
        counts.push(blocks.iter().filter(|(_, b)| is_any_ore(*b)).count());
    }
    // 5 against the clamp of 0.1 (0.3): about 16 times the veins.
    assert!(
        counts[0] > 5 * counts[1].max(1),
        "rich {} poor {}",
        counts[0],
        counts[1]
    );
}

/// A world whose vegetation hook records its random and plants a marker on the ground at random columns.
struct Planter {
    world: MapWorld,
    ground: i32,
    marker: u32,
    draws: Vec<(SectionPos, i64)>,
}

impl CubeWorld for Planter {
    fn block(&self, p: [i32; 3]) -> u32 {
        self.world.block(p)
    }
    fn set_block(&mut self, p: [i32; 3], state: u32) {
        self.world.set_block(p, state);
    }
    fn decorate_surface(&mut self, cube: SectionPos, random: &mut JRandom) {
        self.draws.push((cube, random.long()));
        for _ in 0..12 {
            let x = cube.x * 16 + 8 + random.int(16);
            let z = cube.z * 16 + 8 + random.int(16);
            let y = self.ground + 1;
            if y >> 4 == cube.y || y >> 4 == cube.y + 1 {
                self.world.set_block([x, y, z], self.marker);
            }
        }
    }
}

#[test]
fn ores_leave_the_vegetation_and_everything_above_the_ground_unchanged() {
    let run = |ores: Option<EarthOres>| {
        let f = flat(300.0, ores);
        let (lo, hi) = span(f.ground, 90, 30);
        let mut planter = Planter {
            world: MapWorld::new(0),
            ground: f.ground,
            marker: id("lily_pad"),
            draws: Vec::new(),
        };
        for cx in 0..=2 {
            for cz in 0..=2 {
                for cy in lo..=hi + 1 {
                    let mut blocks = [0_u32; 4096];
                    f.generator
                        .generate(SectionPos::new(cx, cy, cz), &mut blocks);
                    planter.world.insert([cx, cy, cz], blocks);
                }
            }
        }
        for cy in lo..=hi {
            for cx in 0..2 {
                for cz in 0..2 {
                    f.generator
                        .populate(SectionPos::new(cx, cy, cz), &mut planter);
                }
            }
        }
        let w = &planter.world;
        let mut above = Vec::new();
        for x in 16..32 {
            for z in 16..32 {
                for y in f.ground - 1..(hi + 1) * 16 {
                    above.push(w.block([x, y, z]));
                }
            }
        }
        (planter.draws, above, f.ground)
    };
    let (draws_old, above_old, g_old) = run(None);
    let (draws_new, above_new, g_new) = run(Some(ores_of(&ore_settings())));
    assert_eq!(g_old, g_new);
    assert_eq!(
        draws_old, draws_new,
        "the vegetation's random is the cube's seeded random"
    );
    assert!(!draws_new.is_empty());
    for (cube, long) in &draws_new {
        assert_eq!(
            *long,
            coords_seed_random(SEED, cube.x, cube.y, cube.z).long()
        );
    }
    assert!(above_old.contains(&id("lily_pad")));
    assert_eq!(
        above_old, above_new,
        "the ground and what grows on it are unchanged"
    );
}

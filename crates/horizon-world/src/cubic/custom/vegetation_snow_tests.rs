//! Tests of the top layer step of the vegetation ([`Vegetation::with_snow`]): snow on cold ground and on
//! leaves, ice on still water in cold biomes, nothing in warm ones, nothing at all with the step off. They
//! populate a synthetic flat world (stone, dirt, grass, an optional pool of water) with the real 26.3
//! features and are skipped when the bundle is not there.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use horizon_core::position::SectionPos;

use super::biomes::{BiomeColumns, FixedBiome};
use super::populate::CubeWorld;
use super::rng::{JRandom, coords_seed_random};
use super::vegetation::Vegetation;
use crate::registry::{BlockRegistry, BlockStateId};
use crate::worldgen::generator::GeneratorData;

/// The ground's top block (grass) is at this height unless a test says otherwise.
const GROUND: i32 = 70;
/// The pool of water: columns `POOL.0..POOL.1` on both axes, inside what the 3 x 3 populations cover.
const POOL: (i32, i32) = (36, 60);
/// The columns the populations decorate: the areas of cubes 1 to 3 on each axis.
const AREA: (i32, i32) = (24, 72);

struct Rig {
    registry: &'static BlockRegistry,
    data: Arc<GeneratorData>,
}

fn rig() -> Option<&'static Rig> {
    static RIG: OnceLock<Option<Rig>> = OnceLock::new();
    RIG.get_or_init(|| {
        let root = std::env::var_os("HORIZON_TEST_RESOURCES")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.local/horizon-client/26.3")
            });
        if !(root.join("worldgen/world_presets.json").exists()
            && root.join("block-behaviours.json").exists()
            && root.join("worldgen/blocks/block-properties.nbt").exists())
        {
            eprintln!("skipping: no worldgen bundle at {}", root.display());
            return None;
        }
        let registry: &'static BlockRegistry = Box::leak(Box::new(
            crate::level::data::load_registry(&root).expect("registry"),
        ));
        let data = GeneratorData::load(&root.join("worldgen"), registry).expect("worldgen data");
        Some(Rig { registry, data })
    })
    .as_ref()
}

/// Cubes of blocks that populate through the vegetation.
struct Ground {
    cubes: HashMap<[i32; 3], Box<[u32; 4096]>>,
    vegetation: Arc<Vegetation>,
    columns: Arc<dyn BiomeColumns>,
}

impl Ground {
    fn index(p: [i32; 3]) -> usize {
        (((p[1] & 15) << 8) | ((p[2] & 15) << 4) | (p[0] & 15)) as usize
    }
    fn at(&self, x: i32, y: i32, z: i32) -> u32 {
        self.block([x, y, z])
    }
}

impl CubeWorld for Ground {
    fn block(&self, p: [i32; 3]) -> u32 {
        self.cubes
            .get(&[p[0] >> 4, p[1] >> 4, p[2] >> 4])
            .map_or(0, |c| c[Self::index(p)])
    }
    fn set_block(&mut self, p: [i32; 3], state: u32) {
        if let Some(c) = self.cubes.get_mut(&[p[0] >> 4, p[1] >> 4, p[2] >> 4]) {
            c[Self::index(p)] = state;
        }
    }
    fn decorate_surface(&mut self, cube: SectionPos, random: &mut JRandom) {
        let (vegetation, columns) = (self.vegetation.clone(), self.columns.clone());
        vegetation.decorate(cube, random, self, &*columns);
    }
}

/// A flat world of one biome with its ground top at `ground`, and a pool of water level with it when `pool`,
/// after the population of the 3 x 3 cubes whose areas hold the ground.
fn populated(rig: &Rig, biome: &str, snow: bool, ground: i32, pool: bool) -> Ground {
    let id = |name: &str| rig.registry.default_state(name).expect(name).0;
    let (stone, dirt, grass, water) = (
        id("minecraft:stone"),
        id("minecraft:dirt"),
        id("minecraft:grass_block"),
        id("minecraft:water"),
    );
    let seed = 2026;
    let vegetation = Vegetation::new(&rig.data, seed).expect("features in the bundle");
    let vegetation = Arc::new(if snow {
        vegetation.with_snow()
    } else {
        vegetation
    });
    let mut world = Ground {
        cubes: HashMap::new(),
        vegetation,
        columns: Arc::new(FixedBiome::new(biome)),
    };
    let row = (ground - 8) >> 4;
    for cx in 0..5 {
        for cz in 0..5 {
            for cy in row - 1..=row + 2 {
                let mut blocks = Box::new([0_u32; 4096]);
                for (i, block) in blocks.iter_mut().enumerate() {
                    let (x, y, z) = (
                        cx * 16 + (i & 15) as i32,
                        cy * 16 + (i >> 8) as i32,
                        cz * 16 + ((i >> 4) & 15) as i32,
                    );
                    let wet =
                        pool && (POOL.0..POOL.1).contains(&x) && (POOL.0..POOL.1).contains(&z);
                    *block = match y - ground {
                        d if d > 0 => 0,
                        0 if wet => water,
                        0 => grass,
                        -1 | -2 if wet => water,
                        -3..=-1 => dirt,
                        _ => stone,
                    };
                }
                world.cubes.insert([cx, cy, cz], blocks);
            }
        }
    }
    // The cubes of the ground's row, then those of the row above (its trees' tops), as a world fills up.
    for cy in [row, row + 1] {
        for cx in 1..=3 {
            for cz in 1..=3 {
                let cube = SectionPos::new(cx, cy, cz);
                let mut random = coords_seed_random(seed, cx, cy, cz);
                world.decorate_surface(cube, &mut random);
            }
        }
    }
    world
}

/// The name and sorted properties of a block state.
fn state(rig: &Rig, id: u32) -> (String, Vec<(String, String)>) {
    let s = rig.registry.state(BlockStateId(id)).expect("a state");
    let mut properties: Vec<_> = s
        .properties
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    properties.sort();
    (s.name.clone(), properties)
}

fn name(rig: &Rig, id: u32) -> String {
    state(rig, id).0
}

/// How many blocks of the decorated columns, from the ground up, are named `wanted`.
fn count(rig: &Rig, world: &Ground, ground: i32, wanted: &str) -> usize {
    let mut n = 0;
    for x in AREA.0..AREA.1 {
        for z in AREA.0..AREA.1 {
            for y in ground - 3..ground + 40 {
                if name(rig, world.at(x, y, z)) == wanted {
                    n += 1;
                }
            }
        }
    }
    n
}

#[test]
fn snow_lies_on_the_ground_and_on_leaves_of_a_cold_forest_and_grass_under_it_is_snowy() {
    let Some(rig) = rig() else { return };
    let world = populated(rig, "snowy_taiga", true, GROUND, false);
    let (mut snow, mut on_leaves, mut on_grass, mut snowy) = (0, 0, 0, 0);
    for x in AREA.0..AREA.1 {
        for z in AREA.0..AREA.1 {
            for y in GROUND - 3..GROUND + 40 {
                let (n, props) = state(rig, world.at(x, y, z));
                if n != "minecraft:snow" {
                    continue;
                }
                snow += 1;
                assert!(props.contains(&("layers".into(), "1".into())), "{props:?}");
                let (below, below_props) = state(rig, world.at(x, y - 1, z));
                if below.ends_with("_leaves") {
                    on_leaves += 1;
                }
                if below == "minecraft:grass_block" {
                    on_grass += 1;
                    if below_props.contains(&("snowy".into(), "true".into())) {
                        snowy += 1;
                    }
                }
            }
        }
    }
    eprintln!("snow {snow}, on leaves {on_leaves}, on grass {on_grass} (snowy {snowy})");
    assert!(on_grass > 100, "snow on the ground: {on_grass}");
    assert_eq!(snowy, on_grass, "every grass block under snow is snowy");
    assert!(
        on_leaves > 10,
        "snow on the leaves of the trees: {on_leaves}"
    );
    assert!(
        count(rig, &world, GROUND, "minecraft:spruce_leaves") > 100,
        "the forest grew"
    );
}

#[test]
fn still_water_freezes_in_a_frozen_river_and_in_snowy_plains() {
    let Some(rig) = rig() else { return };
    for biome in ["frozen_river", "snowy_plains"] {
        let world = populated(rig, biome, true, GROUND, true);
        for x in POOL.0..POOL.1 {
            for z in POOL.0..POOL.1 {
                assert_eq!(
                    name(rig, world.at(x, GROUND, z)),
                    "minecraft:ice",
                    "{biome} at {x} {z}"
                );
                // The water under the ice stays.
                assert_eq!(name(rig, world.at(x, GROUND - 1, z)), "minecraft:water");
            }
        }
        // The shore has its snow, and the ice has none of its own.
        assert_eq!(
            name(rig, world.at(30, GROUND + 1, 30)),
            "minecraft:snow",
            "{biome}: shore"
        );
        // Without the step (an old world) the pool stays water.
        let old = populated(rig, biome, false, GROUND, true);
        assert_eq!(
            name(rig, old.at(48, GROUND, 48)),
            "minecraft:water",
            "{biome} off"
        );
    }
}

#[test]
fn a_warm_biome_gets_no_snow_and_no_ice_and_the_world_is_the_same_with_the_step_on_or_off() {
    let Some(rig) = rig() else { return };
    for biome in ["plains", "forest", "jungle", "savanna", "swamp"] {
        let on = populated(rig, biome, true, GROUND, true);
        let off = populated(rig, biome, false, GROUND, true);
        assert!(
            on.cubes == off.cubes,
            "{biome}: the cubes differ with snow on"
        );
        for wanted in ["minecraft:snow", "minecraft:ice"] {
            assert_eq!(count(rig, &on, GROUND, wanted), 0, "{biome}: {wanted}");
        }
        assert_eq!(
            name(rig, on.at(48, GROUND, 48)),
            "minecraft:water",
            "{biome}"
        );
    }
}

#[test]
fn there_is_no_height_drop_a_warm_biome_stays_bare_high_up_and_a_cold_one_snows() {
    let Some(rig) = rig() else { return };
    // Vanilla's temperature falls above sea level + 17: at 2,000 blocks plains would snow. Earth's biomes
    // already hold the real height's temperature, so the warm biome stays bare.
    let high = 2000;
    let plains = populated(rig, "plains", true, high, true);
    assert_eq!(count(rig, &plains, high, "minecraft:snow"), 0);
    assert_eq!(count(rig, &plains, high, "minecraft:ice"), 0);
    let cold = populated(rig, "snowy_slopes", true, high, false);
    assert!(count(rig, &cold, high, "minecraft:snow") > 1000);
}

#[test]
fn with_the_step_off_a_cold_biome_has_no_snow_or_ice_and_on_only_adds_to_the_vegetal_pass() {
    let Some(rig) = rig() else { return };
    let off = populated(rig, "snowy_taiga", false, GROUND, true);
    for wanted in ["minecraft:snow", "minecraft:ice"] {
        assert_eq!(count(rig, &off, GROUND, wanted), 0, "{wanted}");
    }
    // What the step changes is snow on tops, ice on water and snowy soil, and the plants of the populations
    // after it; the trees are the same.
    let on = populated(rig, "snowy_taiga", true, GROUND, true);
    let mut changed = 0;
    for (cube, before) in &off.cubes {
        let after = &on.cubes[cube];
        for i in 0..4096 {
            if before[i] == after[i] {
                continue;
            }
            changed += 1;
            let (was, now) = (name(rig, before[i]), name(rig, after[i]));
            // A plant a later population could not grow on a snow layer is gone, hence any `was`.
            let snowed = now == "minecraft:snow";
            let iced = now == "minecraft:ice" && was == "minecraft:water";
            let snowy = was == now
                && matches!(
                    now.as_str(),
                    "minecraft:grass_block" | "minecraft:podzol" | "minecraft:mycelium"
                );
            // Later populations grow their plants on the ground the layer raised or covered.
            let plant =
                |n: &str| n.ends_with("fern") || n.ends_with("_grass") || n == "minecraft:air";
            assert!(
                snowed || iced || snowy || (plant(&was) && plant(&now)),
                "{was} became {now}"
            );
        }
    }
    assert!(changed > 1000, "the step did something: {changed}");
    // The trees are the same ones.
    let trees = |world: &Ground, ending: &str| {
        let mut n = 0;
        for blocks in world.cubes.values() {
            n += blocks
                .iter()
                .filter(|&&b| name(rig, b).ends_with(ending))
                .count();
        }
        n
    };
    for ending in ["_log", "_leaves"] {
        assert_eq!(trees(&on, ending), trees(&off, ending), "{ending}");
    }
}

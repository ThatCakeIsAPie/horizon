//! Tests of Earth's vegetation rules: the kinds of feature, what each land class allows, the mask over a
//! synthetic land cover and the population hook. The last test grows real trees from the 26.3 bundle and is
//! skipped when it is not there.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use horizon_core::position::SectionPos;
use horizon_earth::testing::{MemoryFetcher, MemoryLandCover, TempDir};
use horizon_earth::{LandClass, LandCoverFetcher, LandCoverStore, TileStore, WorldCoverFile};

use super::land_tests::LandRig;
use super::tests::{LAT0, LON0};
use super::vegetation::{EarthVegetationMask, FeatureKind, allowed};
use super::*;
use crate::cubic::custom::CustomCubicGenerator;
use crate::cubic::custom::biomes::BiomeColumns;
use crate::cubic::custom::populate::CubeWorld;
use crate::cubic::custom::rng::{JRandom, coords_seed_random};
use crate::cubic::custom::testing::default_registry;
use crate::cubic::custom::vegetation::{Vegetation, VegetationMask};
use crate::registry::{BlockRegistry, BlockStateId};
use crate::worldgen::generator::GeneratorData;

use FeatureKind::{Dry, Plant, Tree, Water};

#[test]
fn feature_kinds_follow_the_names_of_the_26_3_features() {
    for name in [
        "trees_plains",
        "minecraft:trees_water",
        "trees_birch_and_oak_leaf_litter",
        "bamboo_light",
        "bamboo_vegetation",
        "birch_tall",
        "dark_forest_vegetation",
        "mushroom_island_vegetation",
        "pale_garden_vegetation",
        "rooted_azalea_tree",
        "fallen_oak_tree",
    ] {
        assert_eq!(FeatureKind::of(name), Tree, "{name}");
    }
    for name in [
        "seagrass_normal",
        "seagrass_swamp",
        "kelp_cold",
        "sea_pickle",
        "warm_ocean_vegetation",
    ] {
        assert_eq!(FeatureKind::of(name), Water, "{name}");
    }
    for name in [
        "patch_cactus_desert",
        "patch_dead_bush_2",
        "patch_dry_grass_desert",
        "patch_dead_bush_badlands",
    ] {
        assert_eq!(FeatureKind::of(name), Dry, "{name}");
    }
    for name in [
        "patch_grass_plain",
        "patch_grass_badlands",
        "patch_tall_grass_2",
        "flower_plains",
        "forest_flowers",
        "patch_bush",
        "patch_large_fern",
        "patch_berry_common",
        "patch_sugar_cane_desert",
        "patch_pumpkin",
        "patch_melon",
        "brown_mushroom_normal",
        "red_mushroom_taiga",
        "patch_waterlily",
        "vines",
        "glow_lichen",
        "patch_leaf_litter",
        "patch_firefly_bush_near_water",
    ] {
        assert_eq!(FeatureKind::of(name), Plant, "{name}");
    }
}

#[test]
fn each_class_allows_what_the_table_says() {
    let all = [Tree, Water, Dry, Plant];
    let allows = |class, kinds: &[FeatureKind]| {
        for kind in all {
            assert_eq!(
                allowed(class, kind, 5, 9),
                kinds.contains(&kind),
                "{class:?} {kind:?}"
            );
        }
    };
    allows(LandClass::Tree, &all);
    allows(LandClass::Crop, &[Plant]);
    allows(LandClass::MossLichen, &[Plant]);
    allows(LandClass::Bare, &[Dry]);
    allows(LandClass::BuiltUp, &[]);
    allows(LandClass::SnowIce, &[]);
    allows(LandClass::Water, &[Water]);
    allows(LandClass::Wetland, &all);
    allows(LandClass::Mangrove, &all);
    allows(LandClass::NoData, &all);
    // Shrub and grass allow trees on a share of the columns only (tested below); the rest is fixed.
    for (x, z) in [(0, 0), (-7, 31), (1000, -40)] {
        assert!(allowed(LandClass::Shrub, Plant, x, z));
        assert!(allowed(LandClass::Shrub, Dry, x, z));
        assert!(!allowed(LandClass::Shrub, Water, x, z));
        assert!(allowed(LandClass::Grass, Plant, x, z));
        assert!(!allowed(LandClass::Grass, Dry, x, z));
        assert!(!allowed(LandClass::Grass, Water, x, z));
    }
}

#[test]
fn shrub_allows_trees_on_a_fifth_of_the_columns_and_grass_on_two_in_a_hundred() {
    let share = |class| {
        let mut yes = 0;
        for x in -150..150 {
            for z in -100..100 {
                yes += i32::from(allowed(class, Tree, x, z));
            }
        }
        f64::from(yes) / 60_000.0
    };
    let (shrub, grass) = (share(LandClass::Shrub), share(LandClass::Grass));
    assert!((0.18..0.22).contains(&shrub), "{shrub}");
    assert!((0.012..0.028).contains(&grass), "{grass}");
    // The same columns every time.
    assert_eq!(
        allowed(LandClass::Shrub, Tree, 12, -34),
        allowed(LandClass::Shrub, Tree, 12, -34)
    );
}

#[test]
fn the_mask_reads_the_class_of_the_exact_column() {
    let mask = |code| {
        let rig = LandRig::level(LAT0, 100.0, code);
        let mask = EarthVegetationMask::new(Arc::new(rig.generator));
        (mask, rig.store, rig.land)
    };
    let (trees, _store, _land) = mask(10);
    assert!(trees.allows(3, 4, "trees_plains"));
    assert!(trees.allows(3, 4, "patch_grass_plain"));
    let (city, _store, _land) = mask(50);
    for feature in [
        "trees_plains",
        "patch_grass_plain",
        "seagrass_normal",
        "patch_cactus_desert",
    ] {
        assert!(!city.allows(3, 4, feature), "{feature}");
    }
    let (lake, _store, _land) = mask(80);
    assert!(lake.allows(3, 4, "seagrass_normal"));
    assert!(!lake.allows(3, 4, "trees_water"));
    assert!(!lake.allows(3, 4, "patch_grass_plain"));
    let (field, _store, _land) = mask(40);
    assert!(field.allows(3, 4, "flower_plains"));
    assert!(!field.allows(3, 4, "trees_plains"));
}

#[test]
fn the_mask_is_made_only_for_earth_with_land_cover() {
    let rig = LandRig::level(LAT0, 100.0, 10);
    let with = CustomCubicGenerator::earth_with_land_cover(
        &rig.settings,
        rig.store.clone(),
        Some(rig.source.clone()),
        default_registry(),
    )
    .unwrap();
    assert!(with.earth_vegetation_mask().is_some());
    let without =
        CustomCubicGenerator::earth(&rig.settings, rig.store.clone(), default_registry()).unwrap();
    assert!(without.earth_vegetation_mask().is_none());
}

/// A world that only records the calls to the vegetation hook.
#[derive(Default)]
struct Recorder {
    calls: Vec<(SectionPos, i64)>,
}

impl CubeWorld for Recorder {
    fn block(&self, _p: [i32; 3]) -> u32 {
        0
    }
    fn set_block(&mut self, _p: [i32; 3], _state: u32) {}
    fn decorate_surface(&mut self, cube: SectionPos, random: &mut JRandom) {
        self.calls.push((cube, random.long()));
    }
}

#[test]
fn earth_population_calls_the_vegetation_hook_once_with_the_cubes_seeded_random() {
    let rig = LandRig::level(LAT0, 100.0, 10);
    let make = |seed: Option<i64>| {
        let g = CustomCubicGenerator::earth_with_land_cover(
            &rig.settings,
            rig.store.clone(),
            Some(rig.source.clone()),
            default_registry(),
        )
        .unwrap();
        match seed {
            Some(seed) => g.with_seed(seed),
            None => g,
        }
    };
    let cube = SectionPos::new(3, -2, 5);
    for (seed, want) in [(Some(1234_i64), 1234), (None, 0)] {
        let mut world = Recorder::default();
        make(seed).populate(cube, &mut world);
        assert_eq!(world.calls.len(), 1);
        let expected = coords_seed_random(want, 3, -2, 5).long();
        assert_eq!(world.calls[0], (cube, expected));
        // Populating again gives the same random.
        let mut again = Recorder::default();
        make(seed).populate(cube, &mut again);
        assert_eq!(again.calls, world.calls);
    }
}

// --- Real trees ------------------------------------------------------------------------------------------------

fn resources() -> Option<PathBuf> {
    let root = std::env::var_os("HORIZON_TEST_RESOURCES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.local/horizon-client/26.3")
        });
    if root.join("worldgen/world_presets.json").exists()
        && root.join("block-behaviours.json").exists()
        && root.join("worldgen/blocks/block-properties.nbt").exists()
    {
        Some(root)
    } else {
        eprintln!("skipping: no worldgen bundle at {}", root.display());
        None
    }
}

/// A population world over generated cubes: the vegetation hook runs the real features.
struct Grown {
    cubes: HashMap<[i32; 3], Box<[u32; 4096]>>,
    vegetation: Arc<Vegetation>,
    columns: Arc<dyn BiomeColumns>,
}

impl Grown {
    fn index(p: [i32; 3]) -> usize {
        (((p[1] & 15) << 8) | ((p[2] & 15) << 4) | (p[0] & 15)) as usize
    }
}

impl CubeWorld for Grown {
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

/// Level ground of 300 m under one class (`code`) everywhere, with the real data: the number of log blocks, of
/// leaf blocks and of other blocks above the ground after the population of 5 by 5 cubes (so the features run
/// across the cubes' edges too).
fn grow(
    data: &Arc<GeneratorData>,
    registry: &'static BlockRegistry,
    code: u8,
    seed: i64,
) -> (usize, usize, usize) {
    let mut settings = EarthSettings::at(LAT0, LON0);
    settings.land_cover = true;
    settings.vegetation = true;
    let (elevation_dir, land_dir) = (TempDir::new(), TempDir::new());
    let store = Arc::new(TileStore::new(
        elevation_dir.path(),
        Arc::new(MemoryFetcher::new(|_, _| 300.0)),
        4096,
    ));
    let mut land = MemoryLandCover::new();
    for dlat in [-0.7, 0.7] {
        for dlon in [-0.7, 0.7] {
            if let Some(file) = WorldCoverFile::containing(LAT0 + dlat, LON0 + dlon) {
                land = land.with_classes(file, 384, 96, 2, move |_, _| code);
            }
        }
    }
    let fetcher: Arc<dyn LandCoverFetcher> = Arc::new(land);
    let source = settings
        .land_cover_source(Arc::new(LandCoverStore::new(land_dir.path(), 64)), fetcher)
        .unwrap();
    let custom =
        CustomCubicGenerator::earth_with_land_cover(&settings, store, Some(source), registry)
            .unwrap()
            .with_seed(seed);
    let surface = custom.surface_y(8, 8, i32::MAX).unwrap();
    // The cube whose population area (from 8 blocks above its floor, 16 high) holds the surface.
    let cy = (surface - 8) >> 4;
    let vegetation = Vegetation::new(data, seed)
        .expect("block behaviours and features in the bundle")
        .with_mask(custom.earth_vegetation_mask().expect("land cover is on"));
    let mut world = Grown {
        cubes: HashMap::new(),
        vegetation: Arc::new(vegetation),
        columns: custom.earth_biomes().expect("land cover is on"),
    };
    for cx in 0..6 {
        for cz in 0..6 {
            for y in cy..cy + 2 {
                let mut blocks = Box::new([0_u32; 4096]);
                custom.generate(SectionPos::new(cx, y, cz), &mut blocks);
                world.cubes.insert([cx, y, cz], blocks);
            }
        }
    }
    for cx in 0..5 {
        for cz in 0..5 {
            custom.populate(SectionPos::new(cx, cy, cz), &mut world);
        }
    }
    let (mut logs, mut leaves, mut other) = (0, 0, 0);
    for (cube, blocks) in &world.cubes {
        for (i, &b) in blocks.iter().enumerate() {
            if b == 0 || cube[1] * 16 + (i as i32 >> 8) <= surface {
                continue;
            }
            let name = &registry.state(BlockStateId(b)).unwrap().name;
            if name.ends_with("_log") {
                logs += 1;
            } else if name.ends_with("_leaves") {
                leaves += 1;
            } else {
                other += 1;
            }
        }
    }
    (logs, leaves, other)
}

#[test]
fn real_trees_grow_on_tree_cover_and_a_city_grows_nothing() {
    let Some(root) = resources() else { return };
    let registry: &'static BlockRegistry = Box::leak(Box::new(
        crate::level::data::load_registry(&root).expect("registry"),
    ));
    let data = GeneratorData::load(&root.join("worldgen"), registry).expect("worldgen data");
    let (logs, leaves, other) = grow(&data, registry, 10, 42);
    eprintln!("forest: {logs} logs, {leaves} leaves, {other} other");
    assert!(
        logs > 20 && leaves > 100,
        "forest: {logs} logs, {leaves} leaves"
    );
    let built_up = grow(&data, registry, 50, 42);
    assert_eq!(built_up, (0, 0, 0), "built-up ground");
    // Grassland keeps about 2 % of the trees.
    let (grass_logs, ..) = grow(&data, registry, 30, 42);
    assert!(
        grass_logs * 5 < logs,
        "grassland: {grass_logs} logs, forest {logs}"
    );
}

#[test]
fn every_vegetal_feature_of_an_earth_biome_lands_in_its_kind() {
    let Some(root) = resources() else { return };
    let registry: &'static BlockRegistry = Box::leak(Box::new(
        crate::level::data::load_registry(&root).expect("registry"),
    ));
    let data = GeneratorData::load(&root.join("worldgen"), registry).expect("worldgen data");
    let features = crate::worldgen::live::LiveFeatures::new(&data)
        .expect("features")
        .expect("feature bundle");
    let mut by_kind: HashMap<FeatureKind, std::collections::BTreeSet<String>> = HashMap::new();
    for biome in Biome::ALL {
        let id = data.biome_id(biome.name()).expect("a 26.3 biome");
        for name in features.biome_step_features(id, 9) {
            let bare = name.strip_prefix("minecraft:").unwrap_or(&name).to_string();
            by_kind
                .entry(FeatureKind::of(&name))
                .or_default()
                .insert(bare);
        }
    }
    let names = |kind| by_kind.get(&kind).cloned().unwrap_or_default();
    eprintln!("{by_kind:#?}");
    assert!(
        names(Tree)
            .iter()
            .all(|n| n.starts_with("trees_")
                || n.starts_with("bamboo")
                || n == "birch_tall"
                || n == "dark_forest_vegetation"),
        "{:?}",
        names(Tree)
    );
    assert!(names(Tree).contains("trees_plains") && names(Tree).contains("bamboo_light"));
    assert!(
        names(Water)
            .iter()
            .all(|n| n.starts_with("seagrass")
                || n.starts_with("kelp")
                || n == "sea_pickle"
                || n == "warm_ocean_vegetation"),
        "{:?}",
        names(Water)
    );
    assert!(names(Water).contains("seagrass_normal") && names(Water).contains("kelp_cold"));
    assert!(names(Dry).contains("patch_cactus_desert") && names(Dry).contains("patch_dead_bush"));
    assert!(
        names(Dry)
            .iter()
            .all(|n| n.contains("cactus") || n.contains("dead_bush") || n.contains("dry_grass"))
    );
    assert!(names(Plant).contains("patch_grass_plain") && names(Plant).contains("flower_plains"));
    assert!(
        names(Plant)
            .iter()
            .all(|n| !n.contains("tree") && !n.contains("seagrass") && !n.contains("kelp"))
    );
}

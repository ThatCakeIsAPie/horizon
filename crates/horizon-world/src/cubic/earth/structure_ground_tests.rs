//! Tests of the ground vanilla structures stand on ([`super::structure_ground`]).

use horizon_core::density::{Beardifier, RigidPiece, StructureBounds, TerrainAdjustment};
use horizon_core::position::SectionPos;

use super::land_tests::LandRig;
use super::structure_ground::{CARRIED, adapt_ground, moved_from};
use crate::cubic::custom::testing::default_registry;
use crate::cubic::custom::{CustomCubicGenerator, NO_TOP};

/// Land-cover codes: grassland, permanent water, herbaceous wetland.
const GRASS: u8 = 30;
const WATER: u8 = 80;
const WETLAND: u8 = 90;
const LAT: f64 = 45.0;

/// A cube of the generator, adapted and plain.
type Cubes = Vec<(SectionPos, Box<[u32; 4096]>, Box<[u32; 4096]>)>;

/// `[ground, water]` of chunk (0, 0)'s column (`x`, `z`) in the rig's generator.
fn column(rig: &LandRig, x: i32, z: i32) -> [i32; 2] {
    let ground = rig.generator.chunk_ground_blocking(0, 0).unwrap();
    assert_eq!(ground.len(), 256);
    ground[((z & 15) << 4 | (x & 15)) as usize]
}

/// A beard of one piece of `adjustment` over `min..=max`, as the planner makes one (the box inflated by 24).
fn beard_of(min: [i32; 3], max: [i32; 3], adjustment: TerrainAdjustment) -> Beardifier {
    let bounds = StructureBounds::new(min, max).unwrap();
    let affected = StructureBounds::new(min.map(|v| v - 24), max.map(|v| v + 24));
    Beardifier::new(
        vec![RigidPiece {
            bounds,
            adjustment,
            ground_level_delta: 0,
        }],
        Vec::new(),
        affected,
    )
}

// --- The ground -------------------------------------------------------------------------------------------------

#[test]
fn ground_is_the_real_top_at_any_height() {
    let mut seen = Vec::new();
    for metres in [3.0, 90.0, 1500.0] {
        let rig = LandRig::level(LAT, metres, GRASS);
        let b = rig.blocks;
        let s = rig.generator.surface_y_blocking(8, 8).unwrap();
        for (x, z) in [(0, 0), (8, 8), (15, 3)] {
            assert_eq!(column(&rig, x, z), [s, NO_TOP], "{metres} m at {x},{z}");
        }
        // The ground is the topmost solid block: grass, with air over it.
        let at = |y: i32| rig.generator.block_at_blocking(8, y, 8, 1);
        assert_eq!(at(s), b.grass_block);
        assert_eq!(at(s + 1), b.air);
        assert_eq!(at(s + 100), b.air);
        seen.push(s);
    }
    assert!(seen[0] < seen[1] && seen[1] < seen[2], "{seen:?}");
    assert!(seen[2] > 1000, "{seen:?}");
}

#[test]
fn custom_cubic_gives_the_earth_ground_and_adapts_the_same_tops_as_the_cubes() {
    let rig = LandRig::level(LAT, 90.0, GRASS);
    let custom = CustomCubicGenerator::earth_with_land_cover(
        &rig.settings,
        rig.store.clone(),
        Some(rig.source.clone()),
        default_registry(),
    )
    .unwrap();
    let own = custom.earth_generator().unwrap();
    let g = custom.chunk_ground(0, 0);
    let s = own.surface_y_blocking(0, 0).unwrap();
    assert!(g.iter().all(|&c| c == [s, NO_TOP]), "{:?}", g[0]);
    let beard = beard_of(
        [2, s + 6, 2],
        [12, s + 14, 12],
        TerrainAdjustment::BeardThin,
    );
    let mut tops: [i32; 256] = std::array::from_fn(|i| g[i][0]);
    let mut want = tops;
    adapt_ground(&beard, 0, 0, &mut want);
    custom.adapt_tops(0, 0, &mut tops, &beard);
    assert_eq!(tops, want);
    assert!(tops[8 << 4 | 8] > s);
    // The cube the structures' beard changes is the generator's adapted one.
    let c = SectionPos::new(0, (s + 3) >> 4, 0);
    let (mut a, mut b) = (Box::new([0; 4096]), Box::new([0; 4096]));
    custom.generate_adapted(c, &mut a, Some(&beard));
    own.generate_adapted_blocking(c, &mut b, &beard);
    assert!(a[..] == b[..]);
    custom.generate(c, &mut b);
    assert!(a[..] != b[..]);
}

#[test]
fn a_water_column_gives_the_sea_floor() {
    // A lake above the sea level: ground is the bed, water the top water block.
    let lake = LandRig::level(LAT, 120.0, WATER);
    let b = lake.blocks;
    let [ground, water] = column(&lake, 8, 8);
    assert!(
        water != NO_TOP && ground < water && water > 0,
        "{ground} {water}"
    );
    let at = |y: i32| lake.generator.block_at_blocking(8, y, 8, 1);
    assert_ne!(at(ground), b.water);
    assert_ne!(at(ground), b.air);
    assert_eq!(at(ground + 1), b.water);
    assert_eq!(at(water), b.water);
    assert_eq!(at(water + 1), b.air);

    // The sea: the bed is sand, the water fills to the sea level.
    let sea = LandRig::level(LAT, -30.0, WATER);
    let b = sea.blocks;
    let level = sea.generator.sea_level();
    let [ground, water] = column(&sea, 3, 5);
    assert_eq!(water, level - 1);
    assert!(ground < water);
    let at = |y: i32| sea.generator.block_at_blocking(3, y, 5, 1);
    assert_eq!(at(ground), b.sand);
    assert_eq!(at(ground + 1), b.water);
    assert_eq!(at(level - 1), b.water);
    assert_eq!(at(level), b.air);

    // Wetland: one block of water over mud.
    let marsh = LandRig::level(LAT, 120.0, WETLAND);
    let [ground, water] = column(&marsh, 8, 8);
    assert_eq!(water, ground + 1);
    assert_eq!(
        marsh.generator.block_at_blocking(8, ground, 8, 1),
        marsh.blocks.mud
    );
}

#[test]
fn dry_land_below_sea_level_is_ground() {
    // 86 m under the sea: land cover that is not water keeps it dry, so there is no sea over the ground.
    for code in [60, 30, 50, 70] {
        let rig = LandRig::level(LAT, -86.0, code);
        let [ground, water] = column(&rig, 8, 8);
        assert!(ground < rig.generator.sea_level(), "class {code}");
        assert_eq!(water, NO_TOP, "class {code}");
        let at = |y: i32| rig.generator.block_at_blocking(8, y, 8, 1);
        assert_ne!(at(ground), rig.blocks.air, "class {code}");
        assert_eq!(at(ground + 1), rig.blocks.air, "class {code}");
    }
    // Water under the sea level is the sea.
    let rig = LandRig::level(LAT, -86.0, WATER);
    let [ground, water] = column(&rig, 8, 8);
    assert!(ground < water);
    assert_eq!(water, rig.generator.sea_level() - 1);
}

// --- The terrain adapted -----------------------------------------------------------------------------------------

#[test]
fn moved_from_raises_and_lowers_the_top_layers() {
    // Raised by 4: over the new top, what was over the old one.
    let (s, t) = (10, 14);
    assert_eq!(moved_from(15, s, t), 11);
    assert_eq!(moved_from(40, s, t), 11);
    // The top layers move up by 4...
    for y in 10..=14 {
        assert_eq!(moved_from(y, s, t), y - 4, "y {y}");
    }
    // ...and under them is the block that was under the carried layers.
    for y in [9, 5, 0, -50] {
        assert_eq!(moved_from(y, s, t), s - CARRIED, "y {y}");
    }
    assert_eq!(moved_from(t, s, t), s);

    // Lowered by 4: the layers move down; over the new top is what was over the old top.
    let (s, t) = (10, 6);
    assert_eq!(moved_from(7, s, t), 11);
    assert_eq!(moved_from(t, s, t), s);
    for y in 2..=6 {
        assert_eq!(moved_from(y, s, t), y + 4, "y {y}");
    }
    for y in [1, 0, -30] {
        assert_eq!(moved_from(y, s, t), s - CARRIED, "y {y}");
    }

    // Always the old top's air or water over the new top, and a block of the old top layers or the one under them
    // under it.
    for lift in -30..30 {
        for y in -50..80 {
            let from = moved_from(y, 10, 10 + lift);
            if y > 10 + lift {
                assert_eq!(from, 11, "y {y} lift {lift}");
            } else {
                assert!(
                    (10 - CARRIED..=10).contains(&from),
                    "y {y} lift {lift}: {from}"
                );
            }
        }
    }
}

/// The cubes of chunk (0, 0) around height `s`, adapted to `beard` and plain.
fn cubes_around(rig: &LandRig, s: i32, beard: &Beardifier) -> Cubes {
    ((s >> 4) - 2..=(s >> 4) + 2)
        .map(|cy| {
            let c = SectionPos::new(0, cy, 0);
            let mut adapted = Box::new([0; 4096]);
            let mut plain = Box::new([0; 4096]);
            let uniform = rig
                .generator
                .generate_adapted_blocking(c, &mut adapted, beard);
            rig.generator.generate_blocking(c, &mut plain);
            if let Some(block) = uniform {
                assert!(adapted.iter().all(|&b| b == block), "cube {c:?}");
            }
            (c, adapted, plain)
        })
        .collect()
}

/// The block at (`x`, `y`, `z`) of chunk (0, 0)'s cubes `cubes`.
fn block_in(cubes: &Cubes, x: i32, y: i32, z: i32) -> u32 {
    let (_, adapted, _) = cubes.iter().find(|(c, ..)| c.y == y >> 4).unwrap();
    adapted[((y & 15) << 8 | (z & 15) << 4 | (x & 15)) as usize]
}

#[test]
fn away_from_the_beard_every_block_is_the_plain_generators() {
    let rig = LandRig::level(LAT, 90.0, GRASS);
    let s = rig.generator.surface_y_blocking(8, 8).unwrap();
    // Structures far from the chunk, over, in and under the ground; and one in the chunk's columns but far above
    // the ground.
    for (min, max) in [
        ([200, s + 4, 0], [230, s + 12, 30]),
        ([200, s - 30, 0], [230, s + 12, 30]),
        ([-300, s + 4, 0], [-280, s + 12, 30]),
        ([40, s + 6, 0], [60, s + 14, 10]),
        ([2, s + 3000, 2], [12, s + 3010, 12]),
    ] {
        for adjustment in [TerrainAdjustment::BeardBox, TerrainAdjustment::BeardThin] {
            let beard = beard_of(min, max, adjustment);
            for (c, adapted, plain) in cubes_around(&rig, s, &beard) {
                let diff: Vec<usize> = (0..4096).filter(|&i| adapted[i] != plain[i]).collect();
                assert!(
                    diff.is_empty(),
                    "cube {c:?} beard {min:?} {adjustment:?}: {} blocks differ, from {:?}",
                    diff.len(),
                    diff.first().map(|&i| [i & 15, i >> 8, (i >> 4) & 15])
                );
            }
        }
    }
}

/// Every block of `cubes` against the plain column of the same ground `s` whose top the beard moved to
/// `tops[column]`, read through [`moved_from`]; how many blocks moved.
fn check_columns(rig: &LandRig, s: i32, tops: &[i32; 256], cubes: &Cubes) -> usize {
    let mut moved = 0;
    for (c, adapted, plain) in cubes {
        for i in 0..4096usize {
            let x = c.x * 16 + (i & 15) as i32;
            let y = c.y * 16 + (i >> 8) as i32;
            let z = c.z * 16 + ((i >> 4) & 15) as i32;
            let t = tops[(z & 15) as usize * 16 + (x & 15) as usize];
            let changes = t != s && y > s.min(t) - CARRIED && y <= s.max(t);
            let want = if changes {
                moved += 1;
                rig.generator
                    .block_at_blocking(x, moved_from(y, s, t), z, 1)
            } else {
                plain[i]
            };
            assert_eq!(adapted[i], want, "block {x},{y},{z} (top {s} -> {t})");
        }
    }
    moved
}

#[test]
fn under_a_beard_raised_ground_is_filled_and_lowered_ground_cut() {
    let rig = LandRig::level(LAT, 90.0, GRASS);
    let b = rig.blocks;
    let s = rig.generator.surface_y_blocking(8, 8).unwrap();
    let ground = rig.generator.chunk_ground_blocking(0, 0).unwrap();
    assert!(ground.iter().all(|&c| c == [s, NO_TOP]));
    let plain_tops: [i32; 256] = std::array::from_fn(|i| ground[i][0]);

    // A piece 6 blocks over the ground: the ground under it is filled up to it.
    let beard = beard_of(
        [2, s + 6, 2],
        [12, s + 14, 12],
        TerrainAdjustment::BeardThin,
    );
    let mut tops = plain_tops;
    adapt_ground(&beard, 0, 0, &mut tops);
    let centre = tops[16 + 1];
    assert!(centre > s && centre <= s + 6, "raised to {centre} from {s}");
    assert!(tops.iter().all(|&t| t >= s), "ground only raised");
    // A column 19 blocks from the piece (chunk (1, 1)'s far corner) is out of the beard's reach.
    let mut far = plain_tops;
    adapt_ground(&beard, 1, 1, &mut far);
    assert_eq!(far[255], s, "a column 19 blocks from the piece");
    let cubes = cubes_around(&rig, s, &beard);
    assert!(check_columns(&rig, s, &tops, &cubes) > 20);
    // The middle column: grass over three dirt over the fill, with the old ground's soil under it; air over it.
    assert_eq!(block_in(&cubes, 1, centre, 1), b.grass_block);
    for y in centre - 3..centre {
        assert_eq!(block_in(&cubes, 1, y, 1), b.dirt, "y {y}");
    }
    assert_eq!(block_in(&cubes, 1, centre - 4, 1), b.stone);
    if centre - 4 > s {
        assert_eq!(block_in(&cubes, 1, s + 1, 1), b.stone);
    }
    assert_eq!(block_in(&cubes, 1, centre + 1, 1), b.air);
    assert_ne!(
        block_in(&cubes, 1, centre, 1),
        block_in(&cubes, 1, centre + 1, 1)
    );

    // A piece whose floor is 6 blocks under the ground and which stands out of it: the ground over it is cut.
    let beard = beard_of([0, s - 6, 0], [2, s + 8, 2], TerrainAdjustment::BeardThin);
    let mut tops = plain_tops;
    adapt_ground(&beard, 0, 0, &mut tops);
    let centre = tops[16 + 1];
    // Vanilla's thin beard cuts the piece's floor layer too (the piece fills it): the ground stops under it.
    assert!(
        centre < s && centre >= s - 7,
        "lowered to {centre} from {s}"
    );
    assert!(tops.iter().all(|&t| t <= s), "ground only lowered");
    assert_eq!(tops[255], s);
    let cubes = cubes_around(&rig, s, &beard);
    assert!(check_columns(&rig, s, &tops, &cubes) > 20);
    assert_eq!(block_in(&cubes, 1, centre, 1), b.grass_block);
    for y in centre + 1..=s + 3 {
        assert_eq!(block_in(&cubes, 1, y, 1), b.air, "y {y}");
    }
}

// --- The structures over it (with the worldgen bundle) --------------------------------------------------------------

mod with_data {
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, OnceLock};

    use horizon_core::position::ChunkPos;

    use super::{GRASS, LAT, LandRig, WATER, column};
    use crate::cubic::custom::biomes::FixedBiome;
    use crate::cubic::custom::settings::CustomGeneratorSettings;
    use crate::cubic::custom::structures::CubicStructures;
    use crate::cubic::custom::{CustomCubicGenerator, NO_TOP};
    use crate::heightmap::HeightmapType;
    use crate::registry::BlockRegistry;
    use crate::worldgen::generator::GeneratorData;

    struct Bundle {
        registry: &'static BlockRegistry,
        data: Arc<GeneratorData>,
    }

    /// The 26.3 worldgen bundle (`HORIZON_TEST_RESOURCES`), or `None`: the test is skipped.
    fn bundle() -> Option<&'static Bundle> {
        static BUNDLE: OnceLock<Option<Bundle>> = OnceLock::new();
        BUNDLE
            .get_or_init(|| {
                let root = std::env::var_os("HORIZON_TEST_RESOURCES")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| {
                        Path::new(env!("CARGO_MANIFEST_DIR"))
                            .join("../../.local/horizon-client/26.3")
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
                let data =
                    GeneratorData::load(&root.join("worldgen"), registry).expect("worldgen data");
                Some(Bundle { registry, data })
            })
            .as_ref()
    }

    /// The structures over the rig's world with every column of `biome`.
    fn structures(
        bundle: &Bundle,
        rig: &LandRig,
        biome: &str,
    ) -> (Arc<CustomCubicGenerator>, CubicStructures) {
        let generator = Arc::new(
            CustomCubicGenerator::earth_with_land_cover(
                &rig.settings,
                rig.store.clone(),
                Some(rig.source.clone()),
                bundle.registry,
            )
            .unwrap(),
        );
        let structures = CubicStructures::new(
            &bundle.data,
            2026,
            &CustomGeneratorSettings::default(),
            generator.clone(),
            Arc::new(FixedBiome::new(biome)),
        )
        .expect("structures in the bundle");
        (generator, structures)
    }

    #[test]
    fn the_heights_structures_read_are_the_real_ground_and_water() {
        let Some(bundle) = bundle() else { return };
        let surface = [HeightmapType::WorldSurface, HeightmapType::WorldSurfaceWg];
        let floor = [HeightmapType::OceanFloor, HeightmapType::OceanFloorWg];
        // Dry land at any height, and under the sea level.
        for metres in [3.0, 90.0, 1500.0, -86.0] {
            let rig = LandRig::level(LAT, metres, GRASS);
            let (generator, s) = structures(bundle, &rig, "minecraft:plains");
            let ground = column(&rig, 8, 8)[0];
            assert_eq!(column(&rig, 8, 8)[1], NO_TOP);
            assert_eq!(s.column_top(8, 8), ground);
            assert_eq!(generator.earth_generator().unwrap().sea_level(), 0);
            for kind in surface.into_iter().chain(floor) {
                assert_eq!(s.height(kind, 8, 8), ground + 1, "{metres} m {kind:?}");
            }
            // Where no structure adapts the terrain the pieces see the same ground.
            let cx = (0..64)
                .find(|&cx| s.beard(ChunkPos::new(cx, 0)).is_none())
                .expect("a chunk no structure adapts");
            for kind in surface.into_iter().chain(floor) {
                assert_eq!(
                    s.placed_height(kind, cx * 16 + 8, 8),
                    ground + 1,
                    "{metres} m {kind:?}"
                );
            }
        }
        // A lake over a bed.
        let rig = LandRig::level(LAT, 120.0, WATER);
        let (_, s) = structures(bundle, &rig, "minecraft:plains");
        let [ground, water] = column(&rig, 8, 8);
        assert!(ground < water && water > 0);
        assert_eq!(s.column_top(8, 8), ground);
        for kind in floor {
            assert_eq!(s.height(kind, 8, 8), ground + 1);
        }
        for kind in surface {
            assert_eq!(s.height(kind, 8, 8), water + 1);
        }
        // The sea: the floor under the water, the surface at the sea level.
        let rig = LandRig::level(LAT, -30.0, WATER);
        let (_, s) = structures(bundle, &rig, "minecraft:plains");
        let [ground, water] = column(&rig, 8, 8);
        assert_eq!(water, -1);
        for kind in floor {
            assert_eq!(s.height(kind, 8, 8), ground + 1);
        }
        for kind in surface {
            assert_eq!(s.height(kind, 8, 8), 0);
        }
    }

    /// The structure names of the starts within `r` blocks of the origin, with the pieces' lowest block and the
    /// ground under the first piece.
    fn starts(s: &CubicStructures, r: i32) -> Vec<(String, i32, i32)> {
        s.touching([-r, -r], [r, r])
            .iter()
            .map(|start| {
                let piece = start.pieces[0].bounding_box();
                let c = piece.center();
                (start.name.clone(), piece.min[1], s.column_top(c[0], c[2]))
            })
            .collect()
    }

    #[test]
    fn structures_follow_the_biome_of_the_ground_and_stand_on_it() {
        let Some(bundle) = bundle() else { return };
        let rig = LandRig::level(LAT, 90.0, GRASS);
        let ground = column(&rig, 8, 8)[0];
        let (_, desert) = structures(bundle, &rig, "minecraft:desert");
        let (_, plains) = structures(bundle, &rig, "minecraft:plains");
        let (in_desert, in_plains) = (starts(&desert, 3000), starts(&plains, 3000));
        let has =
            |all: &[(String, i32, i32)], name: &str| all.iter().any(|(n, ..)| n.contains(name));
        assert!(has(&in_desert, "desert_pyramid"), "{in_desert:?}");
        assert!(has(&in_desert, "village_desert"), "{in_desert:?}");
        assert!(!has(&in_plains, "desert_pyramid"), "{in_plains:?}");
        assert!(!has(&in_desert, "village_plains"), "{in_desert:?}");
        assert!(has(&in_plains, "village_plains"), "{in_plains:?}");
        // On this level ground every village's lowest piece sits at the ground or a few blocks under it (a
        // house's foundation), never over it or far below.
        for (name, low, top) in in_desert.iter().chain(&in_plains) {
            assert_eq!(*top, ground, "{name}");
            // (A temple's first piece is still at vanilla's placeholder height until its placement settles it.)
            if name.contains("village") {
                assert!(
                    (ground - 24..=ground + 2).contains(low),
                    "{name}: lowest piece {low}, ground {ground}"
                );
            }
        }
    }
}

//! Tests of the Earth far view over synthetic tiles: no network.
use super::*;
use horizon_earth::TileStore;
use horizon_earth::projection::EARTH_RADIUS_M;
use horizon_earth::testing::{MemoryFetcher, TempDir};
use horizon_world::cubic::custom::testing::default_registry;
use horizon_world::cubic::earth::{EarthBlocks, EarthSettings};

const LAT0: f64 = 45.0;
const LON0: f64 = 7.0;

/// A coast along `east = 0` (sea to the west, a slope up to the east) and a 7 km spire 2.5 km east of the origin.
fn terrain(lat: f64, lon: f64) -> f64 {
    let east = (lon - LON0).to_radians() * EARTH_RADIUS_M * LAT0.to_radians().cos();
    let north = (lat - LAT0).to_radians() * EARTH_RADIUS_M;
    let spire =
        7000.0 * (-(((east - 2500.0).powi(2) + north.powi(2)) / (2.0 * 500.0 * 500.0))).exp();
    0.4 * east + spire
}

struct Rig {
    fetcher: Arc<MemoryFetcher>,
    store: Arc<TileStore>,
    generator: EarthGenerator,
    blocks: EarthBlocks,
    _dir: TempDir,
}

fn rig() -> Rig {
    let dir = TempDir::new();
    let fetcher = Arc::new(MemoryFetcher::new(terrain));
    let store = Arc::new(TileStore::new(dir.path(), fetcher.clone(), 256));
    let blocks = EarthBlocks::from_registry(default_registry()).unwrap();
    let generator =
        EarthGenerator::new(&EarthSettings::at(LAT0, LON0), store.clone(), blocks).unwrap();
    Rig {
        fetcher,
        store,
        generator,
        blocks,
        _dir: dir,
    }
}

impl Rig {
    /// Loads the tiles of node column (`x`, `z`) at `level`.
    fn load(&self, level: u32, x: i32, z: i32) {
        for t in self.generator.missing_tiles_for_node(level, x, z) {
            self.store.load(t).unwrap();
        }
    }
}

const AIR: u32 = 1000;
fn state(blocks: &EarthBlocks, id: u32) -> u32 {
    if id == blocks.air { AIR } else { id }
}
fn kind_of(air: u32) -> impl Fn(BlockStateId) -> Kind {
    move |s| if s.0 == air { Kind::Air } else { Kind::Solid }
}

#[test]
fn the_classifier_prunes_sky_and_deep_rock_and_never_a_node_with_surface_in_it() {
    let rig = rig();
    let g = &rig.generator;
    let (mut air, mut rock, mut mixed) = (0, 0, 0);
    for level in 2..=6u32 {
        let step = 1i32 << level;
        let side = 16 * step;
        // The sea, the coast, the slope, the spire's foot, flank and summit.
        for x in [-3000 / side, -1, 0, 1000 / side, 2200 / side, 2500 / side] {
            for z in [-1, 0] {
                rig.load(level, x, z);
                for y in -8_000 / side - 2..=9_000 / side + 2 {
                    let key = CubeKey::new(level as u8, x, y, z);
                    let class = class(g, key, FarDecor::default()).unwrap();
                    // The column surfaces at the voxels' middles, and every voxel as the generator gives it.
                    let o = key.origin();
                    let mut surface_inside = false;
                    let mut all_stone = true;
                    let mut all_empty = true;
                    for vz in 0..16 {
                        for vx in 0..16 {
                            let (wx, wz) =
                                (o[0] + vx * step + step / 2, o[2] + vz * step + step / 2);
                            let zoom = g.zoom_for_step(step);
                            let s = g.field().surface_y(wx, wz, zoom).unwrap();
                            surface_inside |= s >= o[1] && s < o[1] + side;
                            for vy in 0..16 {
                                let b = g.block_at(wx, o[1] + vy * step, wz, step).unwrap();
                                all_stone &= b == rig.blocks.stone;
                                all_empty &= b == rig.blocks.air || b == rig.blocks.water;
                            }
                        }
                    }
                    match class {
                        NodeClass::Air => {
                            air += 1;
                            assert!(
                                all_empty && !surface_inside,
                                "{key:?} pruned but has ground"
                            );
                        }
                        NodeClass::Rock => {
                            rock += 1;
                            assert!(all_stone && !surface_inside, "{key:?} is rock but is not");
                        }
                        NodeClass::Mixed => mixed += 1,
                    }
                }
            }
        }
    }
    assert!(air > 50 && rock > 50 && mixed > 20, "{air} {rock} {mixed}");
    // The levels made of cubes are never classified.
    assert_eq!(
        class(g, CubeKey::new(1, 0, 200, 0), FarDecor::default()).unwrap(),
        NodeClass::Mixed
    );
}

#[test]
fn a_node_is_what_the_generator_gives_voxel_by_voxel_and_its_cover_is_ground_over_its_top() {
    let rig = rig();
    let g = &rig.generator;
    let biome = |_: i32, _: i32| 5u16;
    let kind = kind_of(AIR);
    for level in [0u8, 2, 3, 5] {
        let step = 1i32 << level;
        let side = 16 * step;
        // The foot of the spire's flank, where the nodes are mixed and the ground rises over the lower ones.
        let x = 1800 / side;
        rig.load(u32::from(level), x, 0);
        let s_mid = g
            .field()
            .surface_y(x * side + side / 2, side / 2, g.zoom_for_step(step))
            .unwrap();
        for y in s_mid.div_euclid(side) - 2..=s_mid.div_euclid(side) + 3 {
            let key = CubeKey::new(level, x, y, 0);
            let made = node(g, key, AIR, &biome, &kind, FarDecor::default()).unwrap();
            let o = key.origin();
            for i in (0..4096).step_by(5) {
                let (vx, vy, vz) = ((i & 15) as i32, (i >> 8) as i32, (i >> 4 & 15) as i32);
                let (wx, wz) = (o[0] + vx * step + step / 2, o[2] + vz * step + step / 2);
                let want = state(
                    &rig.blocks,
                    g.block_at(wx, o[1] + vy * step, wz, step).unwrap(),
                );
                let got = made.at(i);
                assert_eq!(
                    got.state.0, want,
                    "level {level} node {key:?} voxel {vx},{vy},{vz}"
                );
                assert_eq!(got.biome, 5);
            }
            // The cover: a column is covered when its ground reaches the first look over the node.
            let cover = cover(g, key).unwrap().unwrap();
            let top = o[1] + side;
            for vz in 0..16 {
                for vx in 0..16 {
                    let (wx, wz) = (o[0] + vx * step + step / 2, o[2] + vz * step + step / 2);
                    let s = g.field().surface_y(wx, wz, g.zoom_for_step(step)).unwrap();
                    let look = (top + COVER_LOOKS[0]).div_euclid(step) * step;
                    let bit = (vz * 16 + vx) as usize;
                    assert_eq!(
                        cover[bit >> 6] >> (bit & 63) & 1 == 1,
                        s >= look,
                        "level {level} node {key:?} column {vx},{vz} (surface {s}, top {top})"
                    );
                }
            }
        }
    }
    // The levels drawn lit have no cover.
    rig.load(9, 0, 0);
    assert!(cover(g, CubeKey::new(9, 0, 0, 0)).unwrap().is_none());
}

#[test]
fn the_sea_ends_on_the_top_of_its_node_at_every_level() {
    // Earth's sea tops at sea level - 1 (sea level 0 here), so its surface lies on the top of the node under it at
    // every level. Those nodes say their water ends there (the mesher draws its top, not the bed under it); the
    // nodes under them, whose water goes on above, say nothing.
    let rig = rig();
    let g = &rig.generator;
    assert_eq!(g.sea_level(), 0);
    let biome = |_: i32, _: i32| 0u16;
    let water = state(&rig.blocks, rig.blocks.water);
    let kind = move |s: BlockStateId| match s.0 {
        AIR => Kind::Air,
        s if s == water => Kind::Fluid,
        _ => Kind::Solid,
    };
    let fluid = |s: BlockStateId| kind(s) == Kind::Fluid;
    for level in [0u8, 2, 3, 5] {
        let side = 16 << level;
        // Out at sea, west of the coast.
        let x = -3000 / side - 1;
        rig.load(u32::from(level), x, 0);
        let key = CubeKey::new(level, x, -1, 0);
        let made = node(g, key, AIR, &biome, &kind, FarDecor::default()).unwrap();
        let tops = fluid_tops(&made, &water_surfaces(g, &surfaces(g, key).unwrap()), fluid);
        let made = made.with_fluid_tops(tops);
        let mut sea = 0;
        for z in 0..16 {
            for x in 0..16 {
                if made.get(x, 15, z).state.0 == water {
                    sea += 1;
                    assert_eq!(
                        made.fluid_top(x, z),
                        Some(side as f32),
                        "level {level} column {x},{z}"
                    );
                }
            }
        }
        assert_eq!(sea, 256, "level {level}: open sea");
        let under = CubeKey::new(level, x, -2, 0);
        let deep = node(g, under, AIR, &biome, &kind, FarDecor::default()).unwrap();
        let surfaces = water_surfaces(g, &surfaces(g, under).unwrap());
        assert!(fluid_tops(&deep, &surfaces, fluid).is_none(), "level {level}");
    }
}

#[test]
fn a_node_whose_tile_is_missing_is_not_ready_and_names_the_tile() {
    let rig = rig();
    let g = &rig.generator;
    let biome = |_: i32, _: i32| 0u16;
    let kind = kind_of(AIR);
    for key in [
        CubeKey::new(0, 0, 0, 0),
        CubeKey::new(1, 3, 0, 3),
        CubeKey::new(4, 2, 1, -1),
    ] {
        let missing = node(g, key, AIR, &biome, &kind, FarDecor::default()).unwrap_err();
        assert!(!missing.tiles.is_empty(), "{key:?}");
        assert!(class(g, key, FarDecor::default()).is_ok() || key.level >= 2);
        let missing = cover(g, key).err();
        assert!(missing.is_some() || key.level > COVER_LEVEL);
    }
    assert_eq!(rig.fetcher.fetch_count(), 0);
}

/// With land cover a node is "later" until the class of every column and the biomes it needs are held, never
/// built from the elevation rule alone; the sampler waits for nothing, and what it lacks is recorded.
#[test]
fn with_land_cover_a_node_is_later_until_its_class_is_held_and_then_shows_the_class() {
    use horizon_earth::testing::MemoryLandCover;
    use horizon_earth::{LandCoverStore, WorldCoverFile};

    let mut settings = EarthSettings::at(LAT0, LON0);
    settings.land_cover = true;
    let dir = TempDir::new();
    let land_dir = TempDir::new();
    let store = Arc::new(TileStore::new(
        dir.path(),
        Arc::new(MemoryFetcher::new(terrain)),
        256,
    ));
    // Every file of the area says water: the ground above the sea is a lake.
    let mut land = MemoryLandCover::new();
    for (lat, lon) in [(45.5, 7.5), (44.5, 7.5), (45.5, 6.5), (44.5, 6.5)] {
        let file = WorldCoverFile::containing(lat, lon).unwrap();
        land = land.with_classes(file, 192, 96, 2, |_, _| 80);
    }
    let land = Arc::new(land);
    let source = settings
        .land_cover_source(
            Arc::new(LandCoverStore::new(land_dir.path(), 16)),
            land.clone(),
        )
        .unwrap();
    let blocks = EarthBlocks::from_registry(default_registry()).unwrap();
    let g = EarthGenerator::new(&settings, store.clone(), blocks)
        .unwrap()
        .with_land_cover(source.clone());
    let biome = |_: i32, _: i32| 0u16;
    let kind = kind_of(AIR);
    // The slope east of the coast, a node that holds ground.
    let (level, x) = (4u8, 20);
    let step = 1i32 << level;
    let side = 16 * step;
    let s_mid = |g: &EarthGenerator| {
        g.field()
            .surface_y(x * side + side / 2, side / 2, g.zoom_for_step(step))
            .unwrap()
    };
    for t in g.missing_tiles_for_node(u32::from(level), x, 0) {
        store.load(t).unwrap();
    }
    let key = CubeKey::new(level, x, s_mid(&g).div_euclid(side), 0);
    // The biomes read a coarse elevation zoom of their own: have those tiles too, so that only land cover lacks.
    let err = loop {
        match node(&g, key, AIR, &biome, &kind, FarDecor::default()) {
            Err(e) if !e.tiles.is_empty() => {
                for t in e.tiles {
                    store.load(t).unwrap();
                }
            }
            Err(e) => break e,
            Ok(_) => panic!("built without land cover"),
        }
    };
    // Elevation is held, land cover is not: later, and nothing was fetched, only recorded.
    assert!(err.tiles.is_empty(), "{err}");
    assert_eq!(land.fetch_count(), 0);
    let mut asked = g.take_land_requests();
    assert!(!asked.is_empty());
    // The downloader's round trips: fetch what was recorded, hand it over, ask again.
    let mut made = None;
    for _ in 0..40 {
        for fetch in asked.drain(..) {
            source.satisfy(&fetch).unwrap();
        }
        match node(&g, key, AIR, &biome, &kind, FarDecor::default()) {
            Ok(n) => {
                made = Some(n);
                break;
            }
            Err(missing) => {
                for t in missing.tiles {
                    store.load(t).unwrap();
                }
                asked = g.take_land_requests();
            }
        }
    }
    let made = made.expect("the node was built once its data was held");
    // The class shows: no grass. Every file says water, but this is a slope, and the water rules of
    // `horizon_world`'s Earth generator keep a lake flat and never carve more than 32 blocks under the ground, so
    // the slope's columns far over the lowest water are dry, with a sand shore; those near it are water.
    let o = key.origin();
    let (mut water, mut sand, mut grass) = (0, 0, 0);
    for i in 0..4096 {
        let v = made.at(i);
        water += usize::from(v.state.0 == blocks.water);
        sand += usize::from(v.state.0 == blocks.sand);
        grass += usize::from(v.state.0 == blocks.grass_block);
    }
    assert!(
        water + sand > 0 && grass == 0,
        "water {water}, sand {sand}, grass {grass} in {o:?}"
    );
    // Without land cover the same node over the same tiles shows grass.
    let mut off = settings.clone();
    off.land_cover = false;
    let plain = EarthGenerator::new(&off, store.clone(), blocks).unwrap();
    let plain_node = node(&plain, key, AIR, &biome, &kind, FarDecor::default()).unwrap();
    assert!((0..4096).any(|i| plain_node.at(i).state.0 == blocks.grass_block));
    // The sky cover reads heights only: it is ready without any land cover.
    let fresh = EarthGenerator::new(&settings, store.clone(), blocks)
        .unwrap()
        .with_land_cover(source.clone());
    assert!(cover(&fresh, key).is_ok());
}

/// An Earth level (and its far sampler) over synthetic tiles, when the game data is there.
mod sampler {
    use super::*;
    use crate::biome::Biomes;
    use crate::chunk::LevelShape;
    use crate::cubic::CubicLevel;
    use crate::cubic_gen::{CubicGeneration, Generation};
    use crate::earth_net::EarthTiles;
    use crate::streaming::LodSampler;

    pub struct NoVanilla;
    impl LodSampler for NoVanilla {
        fn sample(&mut self, _x: i32, _z: i32) -> Option<crate::lod::Surface> {
            None
        }
    }

    pub struct Earth {
        pub sampler: CubicLodSampler,
        pub level: Arc<CubicLevel>,
        pub fetcher: Arc<MemoryFetcher>,
        _dir: std::path::PathBuf,
    }

    impl Drop for Earth {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self._dir);
        }
    }

    pub fn earth(name: &str) -> Option<Earth> {
        let root = std::path::PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES")?);
        let assets = crate::assets::Assets::load(&root).expect("assets");
        let biomes = Arc::new(
            Biomes::load(&horizon_assets::resource_source::ResourceRoot::directory(
                root.clone(),
            ))
            .expect("biomes"),
        );
        let dir = std::env::temp_dir().join(format!(
            "horizon-cubic-lod-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let fetcher = Arc::new(MemoryFetcher::new(terrain));
        let tiles_dir = dir.join("tiles");
        let tiles = EarthTiles {
            store: Arc::new(TileStore::new(&tiles_dir, fetcher.clone(), 64)),
            downloads: true,
            dir: tiles_dir,
            land: None,
            climate: None,
        };
        let generation = CubicGeneration {
            kind: crate::cubic_gen::CubicKind::Earth,
            settings_json: EarthSettings::at(LAT0, LON0).to_json_compact(),
        };
        let wg =
            Generation::with_tiles(&generation, 1, assets.registry, None, Some(Arc::new(tiles)))
                .expect("generation");
        let window = LevelShape::cubic_window(LevelShape::OVERWORLD, -192, 512);
        let level =
            CubicLevel::with_generation(&dir, 1, assets.registry, &biomes, window, &generation, wg)
                .expect("level");
        let sampler = level.lod_sampler(Box::new(NoVanilla), assets.registry);
        Some(Earth {
            sampler,
            level,
            fetcher,
            _dir: dir,
        })
    }
}

#[test]
fn far_sampling_with_a_missing_tile_is_later_asks_for_it_and_stores_nothing() {
    let Some(mut e) = sampler::earth("later") else {
        return;
    };
    let generator = e.level.generation().generator.clone();
    let earth = generator.earth_generator().unwrap();
    let keys = [
        CubeKey::new(1, 40, 3, 5),
        CubeKey::new(3, 20, 0, 4),
        CubeKey::new(5, 6, 1, 1),
    ];
    for key in keys {
        assert!(earth.take_requests().is_empty());
        let wanted = if key.level < 2 {
            earth.missing_tiles(key.x * 32, key.z * 32, key.x * 32 + 31, key.z * 32 + 31)
        } else {
            earth.missing_tiles_for_node(u32::from(key.level), key.x, key.z)
        };
        assert!(!wanted.is_empty());
        // Later, not a node built of air or of a guess; the tile it lacks is asked for, and nothing is fetched here.
        assert!(e.sampler.node(key).is_none(), "{key:?}");
        let asked = earth.take_requests();
        assert!(
            wanted.iter().any(|t| asked.contains(t)),
            "{key:?}: wanted {wanted:?}, asked {asked:?}"
        );
        // Nothing was stored, for this node or any it would complete.
        let cache = e.sampler.cache.clone().unwrap();
        let mut up = key;
        for _ in 0..4 {
            assert!(!cache.contains(up), "{up:?} was stored");
            up = up.parent();
        }
        // The classification is not given either, rather than filed as unknown.
        if key.level >= 2 {
            assert_eq!(e.sampler.classify_ready(key), None);
            assert!(!earth.take_requests().is_empty());
        }
    }
    assert_eq!(e.fetcher.fetch_count(), 0, "a far worker fetched");
    // With the tiles in the node comes, whole.
    for key in keys {
        let tiles = if key.level < 2 {
            earth.missing_tiles(key.x * 32, key.z * 32, key.x * 32 + 31, key.z * 32 + 31)
        } else {
            earth.missing_tiles_for_node(u32::from(key.level), key.x, key.z)
        };
        for t in tiles {
            earth.field().store().load(t).unwrap();
        }
        let node = e.sampler.node(key);
        assert!(node.is_some(), "{key:?} did not come with its tiles");
    }
}

#[test]
fn the_samplers_classification_prunes_sky_and_rock_through_the_trait() {
    let Some(mut e) = sampler::earth("classify") else {
        return;
    };
    let generator = e.level.generation().generator.clone();
    let earth = generator.earth_generator().unwrap();
    let (sky, rock, ground) = (
        CubeKey::new(4, 0, 40, 0),
        CubeKey::new(4, 0, -40, 0),
        CubeKey::new(4, 0, 0, 0),
    );
    for key in [sky, rock, ground] {
        earth
            .missing_tiles_for_node(4, key.x, key.z)
            .into_iter()
            .for_each(|t| {
                earth.field().store().load(t).unwrap();
            });
    }
    assert_eq!(e.sampler.classify_ready(sky), Some(NodeClass::Air));
    assert_eq!(e.sampler.classify(sky), NodeClass::Air);
    assert_eq!(e.sampler.classify_ready(rock), Some(NodeClass::Rock));
    // The coast is somewhere in the node at the origin's height.
    assert_eq!(e.sampler.classify_ready(ground), Some(NodeClass::Mixed));
}

/// The far view with trees over a slope of tree cover: leaf and log voxels at level 0 and coarser, none with
/// trees off or over farmland, and a node over the ground is not called sky.
#[test]
fn far_nodes_over_tree_cover_get_trees_and_a_node_just_over_the_ground_is_not_sky() {
    use horizon_earth::testing::MemoryLandCover;
    use horizon_earth::{LandCoverStore, WorldCoverFile};

    let rig_of = |class: u8| {
        let mut settings = EarthSettings::at(LAT0, LON0);
        settings.land_cover = true;
        let dir = TempDir::new();
        let land_dir = TempDir::new();
        let store = Arc::new(TileStore::new(
            dir.path(),
            Arc::new(MemoryFetcher::new(terrain)),
            256,
        ));
        let mut land = MemoryLandCover::new();
        for (lat, lon) in [(45.5, 7.5), (44.5, 7.5), (45.5, 6.5), (44.5, 6.5)] {
            let file = WorldCoverFile::containing(lat, lon).unwrap();
            land = land.with_classes(file, 192, 96, 2, move |_, _| class);
        }
        let source = settings
            .land_cover_source(Arc::new(LandCoverStore::new(land_dir.path(), 16)), Arc::new(land))
            .unwrap();
        let blocks = EarthBlocks::from_registry(default_registry()).unwrap();
        let g = EarthGenerator::new(&settings, store.clone(), blocks)
            .unwrap()
            .with_land_cover(source.clone());
        (g, store, source, blocks, (dir, land_dir))
    };
    let biome = |_: i32, _: i32| 0u16;
    let kind = kind_of(AIR);
    let trees = FarDecor {
        trees: true,
        snow: false,
    };
    // Builds `key` the way the downloader does: load what is named, hand over what was recorded, ask again.
    let build = |class: u8, key: CubeKey, decor: FarDecor| {
        let (g, store, source, blocks, _dirs) = rig_of(class);
        for _ in 0..80 {
            match node(&g, key, AIR, &biome, &kind, decor) {
                Ok(n) => return (n, blocks),
                Err(missing) => {
                    for t in missing.tiles {
                        store.load(t).unwrap();
                    }
                    for fetch in g.take_land_requests() {
                        source.satisfy(&fetch).unwrap();
                    }
                }
            }
        }
        panic!("never ready");
    };
    let count = |n: &CubeNode, blocks: &EarthBlocks, ids: &[u32]| {
        let _ = blocks;
        (0..4096).filter(|&i| ids.contains(&n.at(i).state.0)).count()
    };
    // The slope 500 m east of the coast (the spire is further out): ground at about 200 blocks.
    let (g, ..) = rig_of(10);
    let leaves: Vec<u32> = g.blocks().far.leaves.to_vec();
    let logs: Vec<u32> = g.blocks().far.logs.to_vec();
    let ground_y = |level: u32, x: i32| {
        let step = 1i32 << level;
        let side = 16 * step;
        // The tiles it needs are loaded by `build`, so read the height from a throwaway rig of the same.
        let (g, store, ..) = rig_of(10);
        for t in g.missing_tiles_for_node(level, x, 0) {
            store.load(t).unwrap();
        }
        g.field()
            .surface_y(x * side + side / 2, side / 2, g.zoom_for_step(step))
            .unwrap()
    };
    // Level 0: the cube holding the surface.
    let s0 = ground_y(0, 500 / 16);
    let key0 = CubeKey::new(0, 500 / 16, s0.div_euclid(16), 0);
    let (n, blocks) = build(10, key0, trees);
    let (l, t) = (count(&n, &blocks, &leaves), count(&n, &blocks, &logs));
    assert!(l + t > 0, "no tree in a cube of forest");
    let (n, blocks) = build(10, key0, FarDecor::default());
    assert_eq!(count(&n, &blocks, &leaves) + count(&n, &blocks, &logs), 0);
    let (n, blocks) = build(40, key0, trees);
    assert_eq!(count(&n, &blocks, &leaves) + count(&n, &blocks, &logs), 0, "trees on farmland");
    // Level 3: the voxels of the canopy.
    let s3 = ground_y(3, 500 / 128);
    let key3 = CubeKey::new(3, 500 / 128, s3.div_euclid(128), 0);
    let (n, blocks) = build(10, key3, trees);
    assert!(count(&n, &blocks, &leaves) > 0, "no canopy at level 3");
    let (n, blocks) = build(10, key3, FarDecor::default());
    assert_eq!(count(&n, &blocks, &leaves), 0);
    // A node of 64 blocks starting a little over the ground is not pruned as sky with trees on. The ground slopes
    // across a node, so look along the slope for the nodes where sky starts.
    let (g, store, ..) = rig_of(10);
    let mut probe = Vec::new();
    for x in 7..15 {
        for t in g.missing_tiles_for_node(2, x, 0) {
            store.load(t).unwrap();
        }
        let ground = g
            .field()
            .surface_y(x * 64 + 32, 32, g.zoom_for_step(4))
            .unwrap();
        for n in 0..5 {
            let key = CubeKey::new(2, x, ground.div_euclid(64) + n, 0);
            probe.push((
                class(&g, key, FarDecor::default()).unwrap(),
                class(&g, key, trees).unwrap(),
            ));
        }
    }
    assert!(
        probe.contains(&(NodeClass::Air, NodeClass::Mixed)),
        "trees did not keep a low node of sky: {probe:?}"
    );
    assert!(probe.contains(&(NodeClass::Air, NodeClass::Air)));
    // Trees only ever take away sky.
    assert!(probe.iter().all(|&(p, t)| p != NodeClass::Mixed || t == NodeClass::Mixed));
}

/// The far view's water tops (`CubeNode::fluid_tops`) over the sea of the test terrain, through the sampler, on
/// each way a far node is made. The sea tops at `sea_level - 1`, on the top of the node under the sea level, at
/// every level: a node without tops draws its bed (sand) in place of its water.
mod water_tops {
    use super::*;
    use crate::chunk::{BIOME_STRATEGY, Chunk, ColumnLayout, Section, block_strategy};
    use crate::cubic_lod::Saved;
    use horizon_core::paletted::PalettedContainer;

    /// Tiles are asked for, never fetched, by a far worker: this loads what was asked, as the downloader does.
    fn load_asked(level: &Arc<crate::cubic::CubicLevel>) {
        let generator = level.generation().generator.clone();
        let earth = generator.earth_generator().unwrap();
        for t in earth.take_requests() {
            earth.field().store().load(t).unwrap();
        }
    }

    /// `make` until the tiles it needed (asked for each time it said later) are in; the number of laters.
    fn when_ready<T>(
        level: &Arc<crate::cubic::CubicLevel>,
        mut make: impl FnMut() -> Option<T>,
    ) -> (T, usize) {
        for laters in 0..40 {
            if let Some(made) = make() {
                return (made, laters);
            }
            load_asked(level);
        }
        panic!("never ready");
    }

    /// Every column of `node` ends its water on the node's top.
    fn sea_on_top(node: &CubeNode, what: &str) {
        let size = node.key.size() as f32;
        for z in 0..16 {
            for x in 0..16 {
                assert_eq!(node.fluid_top(x, z), Some(size), "{what}: column {x},{z}");
            }
        }
    }

    /// Cube (`x`, `y`, 0) of the sea: stone for its lowest `floor` layers, water over them.
    fn save_sea_cube(e: &sampler::Earth, x: i32, y: i32, floor: i32) {
        let blocks = generator_blocks(e);
        let section = Section {
            blocks: PalettedContainer::new(block_strategy(), blocks.water),
            biomes: PalettedContainer::new(BIOME_STRATEGY, 0u32),
            ..Section::default()
        };
        let mut cube = Chunk::with_layout(x, 0, ColumnLayout::window(y * 16, 16));
        cube.sections[0] = Arc::new(section);
        for ly in 0..floor {
            for lz in 0..16 {
                for lx in 0..16 {
                    cube.set(lx, y * 16 + ly, lz, blocks.stone);
                }
            }
        }
        assert!(e.level.save_cube(&cube, y));
    }

    fn generator_blocks(e: &sampler::Earth) -> EarthBlocks {
        let generator = e.level.generation().generator.clone();
        *generator.earth_generator().unwrap().blocks()
    }

    /// 1. Saved cubes: a node of the cubes the player walked over, as the sampler hands it out.
    #[test]
    fn saved_cubes_over_the_sea_get_their_water_tops_at_levels_0_and_1() {
        let Some(mut e) = sampler::earth("saved-tops") else {
            return;
        };
        let level = e.level.clone();
        // Cube (-188, -1, 0) holds blocks y -16..-1 at x -3008: out at sea (the terrain 1,200 blocks under).
        save_sea_cube(&e, -188, -1, 3);
        let (key0, key1) = (CubeKey::new(0, -188, -1, 0), CubeKey::new(1, -94, -1, 0));
        let generator = e.level.generation().generator.clone();
        let earth = generator.earth_generator().unwrap();

        // Level 0: the saved cube itself. Its tiles missing, it is later (and they are asked for), not drawn
        // without tops.
        assert!(e.sampler.node(key0).is_none());
        assert!(!earth.take_requests().is_empty(), "the tiles were asked for");
        let (node, laters) = when_ready(&level, || e.sampler.node(key0));
        assert!(laters > 0);
        // Its voxels are the saved ones (stone floor, water over it), its tops the sea's.
        assert_eq!(node.get(3, 0, 3).state.0, earth.blocks().stone);
        assert_eq!(node.get(3, 15, 3).state.0, earth.blocks().water);
        sea_on_top(&node, "saved cube");

        // Level 1: one saved cube of its eight, the others the generator's.
        let (node, _) = when_ready(&level, || e.sampler.node(key1));
        sea_on_top(&node, "saved level 1");

        // A cube above the sea (nothing wet) gets no tops and its cover as before.
        let dry = CubeKey::new(0, -188, 400, 0);
        let (node, _) = when_ready(&level, || e.sampler.node(dry));
        assert!(node.fluid_tops().is_none());
    }

    /// 1b. The keep flag: a level 1 node of saved cubes is kept when exact (its surfaces are known), only then.
    #[test]
    fn a_saved_level_1_node_of_an_earth_world_is_kept_when_its_tiles_are_in_and_later_before() {
        let Some(mut e) = sampler::earth("saved-keep") else {
            return;
        };
        let level = e.level.clone();
        save_sea_cube(&e, -188, -1, 3);
        let key1 = CubeKey::new(1, -94, -1, 0);
        // No tiles yet: later, nothing to keep and no node of generated terrain in place of the saved cubes.
        assert!(matches!(e.sampler.of_saved_cubes(key1), Saved::Later));
        assert!(e.sampler.node(key1).is_none());
        let (saved, _) = when_ready(&level, || match e.sampler.of_saved_cubes(key1) {
            Saved::Node(node, keep) => Some((node, keep)),
            Saved::Later => None,
            Saved::No => panic!("a cube was saved"),
        });
        // One of eight saved, outside a Hybrid band: exact, kept.
        assert!(saved.1);
        sea_on_top(&saved.0, "level 1");
        // Nothing saved under it: not made from saved cubes at all.
        assert!(matches!(
            e.sampler.of_saved_cubes(CubeKey::new(1, -60, -1, 0)),
            Saved::No
        ));
        assert!(matches!(
            e.sampler.of_saved_cubes(CubeKey::new(2, -47, -1, 0)),
            Saved::No
        ));
    }

    /// 2. Generated nodes: levels 0 and 1 (cubes), 2 and up (surfaces), none of them saved.
    #[test]
    fn generated_sea_nodes_get_their_water_tops_at_every_level() {
        let Some(mut e) = sampler::earth("generated-tops") else {
            return;
        };
        let level = e.level.clone();
        for key in [
            CubeKey::new(0, -190, -1, 0),
            CubeKey::new(1, -100, -1, 0),
            CubeKey::new(2, -50, -1, 0),
            CubeKey::new(3, -26, -1, 0),
            CubeKey::new(5, -7, -1, 0),
        ] {
            let (node, _) = when_ready(&level, || e.sampler.node(key));
            sea_on_top(&node, &format!("{key:?}"));
            // The node under it: water going on above, no top in it.
            let under = CubeKey::new(key.level, key.x, -2, key.z);
            let (node, _) = when_ready(&level, || e.sampler.node(under));
            assert!(node.fluid_tops().is_none(), "{under:?}");
        }
    }

    /// 3. The coarse pass of the refining, levels 0 and 1 from the columns' surfaces (`from_surfaces`), then
    /// `sampled`'s own step (`earth_covered`).
    #[test]
    fn coarse_nodes_from_the_surfaces_get_their_water_tops_too() {
        let Some(e) = sampler::earth("coarse-tops") else {
            return;
        };
        let level = e.level.clone();
        let generator = e.level.generation().generator.clone();
        let earth = generator.earth_generator().unwrap();
        let biome = |x: i32, z: i32| level.biome_id_at(x, z);
        for key in [CubeKey::new(0, -190, -1, 0), CubeKey::new(1, -100, -1, 0)] {
            let (node, _) = when_ready(&level, || {
                let made = from_surfaces(earth, key, e.sampler.air, &biome, FarDecor::default());
                match made {
                    Ok(node) => e.sampler.earth_covered(key, node),
                    Err(missing) => {
                        e.sampler.or_request(Err::<(), _>(missing));
                        None
                    }
                }
            });
            assert!(
                (0..4096).any(|i| node.at(i).state.0 == earth.blocks().water),
                "{key:?} has water"
            );
            sea_on_top(&node, &format!("coarse {key:?}"));
        }
    }
}

#[test]
fn a_cover_proven_from_the_bounds_is_the_cover_its_columns_give_and_every_rock_node_has_one() {
    let rig = rig();
    let g = &rig.generator;
    let (mut proven, mut rock) = (0, 0);
    for level in 2..=6u32 {
        let side = 16i32 << level;
        // The sea, the coast, the slope, the spire's foot, flank and summit.
        for x in [-3000 / side, -1, 0, 1000 / side, 2200 / side, 2500 / side] {
            for z in [-1, 0] {
                rig.load(level, x, z);
                for y in -8_000 / side - 2..=9_000 / side + 2 {
                    let key = CubeKey::new(level as u8, x, y, z);
                    let is_rock = class(g, key, FarDecor::default()).unwrap() == NodeClass::Rock;
                    rock += usize::from(is_rock);
                    let Some(mask) = proven_cover(g, key) else {
                        assert!(!is_rock, "{key:?} is rock but its cover is not proven");
                        continue;
                    };
                    proven += 1;
                    let read = cover_of(key, &surfaces(g, key).unwrap());
                    assert_eq!(read.as_deref(), Some(&*mask), "{key:?}");
                }
            }
        }
    }
    assert!(rock > 50 && proven >= rock, "{proven} {rock}");
}

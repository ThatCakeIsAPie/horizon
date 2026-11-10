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
                    let class = class(g, key).unwrap();
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
        class(g, CubeKey::new(1, 0, 200, 0)).unwrap(),
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
            let made = node(g, key, AIR, &biome, &kind).unwrap();
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
        let missing = node(g, key, AIR, &biome, &kind).unwrap_err();
        assert!(!missing.tiles.is_empty(), "{key:?}");
        assert!(class(g, key).is_ok() || key.level >= 2);
        let missing = cover(g, key).err();
        assert!(missing.is_some() || key.level > COVER_LEVEL);
    }
    assert_eq!(rig.fetcher.fetch_count(), 0);
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
        let biomes = Arc::new(Biomes::load(&root).expect("biomes"));
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

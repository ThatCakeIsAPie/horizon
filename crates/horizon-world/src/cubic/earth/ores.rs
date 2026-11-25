//! Earth's ores ([`crate::cubic::ores`]): when an Earth world's settings have
//! `oreGen`, each populated cube gets, in order,
//!
//! 1. the deepslate gradient: stone at least [`DEEPSLATE_FULL`] blocks under
//!    the local ground is deepslate, none is shallower than [`DEEPSLATE_TOP`],
//!    and in between a positional random turns a growing share of it (as
//!    26.3's gradient between Y 8 and 0);
//! 2. stone patches and ores by depth (the ore engine) on a random of their
//!    own, so the vegetation's random is untouched;
//!
//! then the integrator's vegetation, then [`EarthOres::place_meteors`].
//!
//! The ground is the real surface (the top block of each column from the
//! elevation data), the biome is Earth's biome of the cube's column and the
//! richness map reads the latitude and longitude of the window's centre.

use std::cell::RefCell;
use std::collections::HashMap;

use horizon_core::position::SectionPos;

use super::generator::EarthGenerator;
use crate::cubic::custom::populate::CubeWorld;
use crate::cubic::custom::replacers::resolve_block;
use crate::cubic::custom::rng::{coords_seed_random, hash_unit};
use crate::cubic::custom::settings::BlockStateDesc;
use crate::cubic::ores::engine::{OreCtx, OreEngine, RichnessFn};
use crate::cubic::ores::meteors::Meteors;
use crate::cubic::ores::settings::OreGenSettings;
use crate::registry::BlockRegistry;

/// Depth under the ground (blocks) where deepslate starts to appear.
pub const DEEPSLATE_TOP: i32 = 64;
/// Depth under the ground from which all stone is deepslate.
pub const DEEPSLATE_FULL: i32 = 72;

/// Mixed into the world seed for the ores' population random.
const ORE_SALT: i64 = 0x4561_7274_684f_7265; // "EarthOre"
/// The deepslate gradient's positional random.
const DEEPSLATE_SALT: i32 = 0x6465_6570; // "deep"
/// The ground of a column the data could not give (cancelled): nothing is
/// under it.
const NO_GROUND: i32 = i32::MIN / 2;

/// The compiled ores of one Earth world.
pub struct EarthOres {
    engine: OreEngine,
    meteors: Meteors,
    seed: i64,
    stone: u32,
    deepslate: u32,
}

impl EarthOres {
    /// The ores of `settings` for the world seed `seed`. Fails naming a bad
    /// setting or a block the registry lacks.
    pub fn new(seed: i64, settings: &OreGenSettings, reg: &BlockRegistry) -> Result<Self, String> {
        settings.validate()?;
        let id = |n: &str| resolve_block(reg, &BlockStateDesc::new(n));
        Ok(Self {
            engine: OreEngine::new(seed, settings, reg)?,
            meteors: Meteors::new(seed, &settings.meteors, reg)?,
            seed,
            stone: id("stone")?,
            deepslate: id("deepslate")?,
        })
    }

    /// Replaces the richness map (tests).
    pub fn with_richness(mut self, f: RichnessFn) -> Self {
        self.engine = self.engine.with_richness(f);
        self
    }

    /// Steps 1 and 2 for `cube` (call before the vegetation).
    pub fn place_ores(
        &self,
        generator: &EarthGenerator,
        cube: SectionPos,
        world: &mut dyn CubeWorld,
        ground: &EarthGround,
    ) {
        self.deepslate(cube, world, ground);
        if !self.engine.is_active() {
            return;
        }
        let (x, z) = (cube.x * 16 + 8, cube.z * 16 + 8);
        let biome = if generator.land_cover_active() {
            generator.biome_at_quart_blocking(x >> 2, z >> 2).name()
        } else {
            super::Biome::Plains.name()
        };
        let mut rand = coords_seed_random(self.seed ^ ORE_SALT, cube.x, cube.y, cube.z);
        let g = |x: i32, z: i32| ground.at(x, z);
        let lat_lon = |x: i32, z: i32| generator.lat_lon_at(x, z);
        let ctx = OreCtx {
            biome,
            ground: &g,
            lat_lon: Some(&lat_lon),
        };
        self.engine.place(cube, world, &mut rand, &ctx);
    }

    /// The ore meteors `cube` owns (call after the vegetation).
    pub fn place_meteors(&self, cube: SectionPos, world: &mut dyn CubeWorld, ground: &EarthGround) {
        if self.meteors.is_active() {
            self.meteors.place(cube, world, &|x, z| ground.at(x, z));
        }
    }

    /// Step 1 over the whole 2 x 2 x 2 window of `cube`. The choice of each
    /// block depends only on its position and depth, so windows that overlap
    /// agree and a block is never turned twice.
    fn deepslate(&self, cube: SectionPos, world: &mut dyn CubeWorld, ground: &EarthGround) {
        let min = [cube.x * 16, cube.y * 16, cube.z * 16];
        let top = min[1] + 31;
        for x in min[0]..min[0] + 32 {
            for z in min[2]..min[2] + 32 {
                let g = ground.at(x, z);
                if g == NO_GROUND {
                    continue;
                }
                // Highest Y with any chance of deepslate: depth > DEEPSLATE_TOP.
                let high = (g - DEEPSLATE_TOP - 1).min(top);
                for y in min[1]..=high {
                    let depth = g - y;
                    let deep = depth >= DEEPSLATE_FULL
                        || hash_unit(self.seed, DEEPSLATE_SALT, x, y, z)
                            < f64::from(depth - DEEPSLATE_TOP)
                                / f64::from(DEEPSLATE_FULL - DEEPSLATE_TOP);
                    if deep && world.block([x, y, z]) == self.stone {
                        world.set_block([x, y, z], self.deepslate);
                    }
                }
            }
        }
    }
}

/// The ground of an Earth world's columns for one population: the real
/// surface (each column's top block from the data), read per chunk and kept
/// for the call.
pub struct EarthGround<'a> {
    generator: &'a EarthGenerator,
    chunks: RefCell<HashMap<(i32, i32), Option<Vec<i32>>>>,
}

impl<'a> EarthGround<'a> {
    pub fn new(generator: &'a EarthGenerator) -> Self {
        Self {
            generator,
            chunks: RefCell::new(HashMap::new()),
        }
    }

    /// The Y of the top block of column (`x`, `z`); very low when the data
    /// could not be read (cancelled).
    pub fn at(&self, x: i32, z: i32) -> i32 {
        let key = (x >> 4, z >> 4);
        let mut chunks = self.chunks.borrow_mut();
        let tops = chunks
            .entry(key)
            .or_insert_with(|| self.generator.chunk_tops_blocking(key.0, key.1));
        tops.as_ref()
            .and_then(|t| t.get(((z & 15) << 4 | (x & 15)) as usize).copied())
            .unwrap_or(NO_GROUND)
    }
}

//! Vegetation for a cube's population (the integrator's
//! [`CubeWorld::decorate_surface`] hook): 26.3's own biome features of the
//! vegetal decoration step (trees, grass, flowers, ...), run through the
//! live feature system (`LiveFeatures`, the block behaviours' `Sim`) on the
//! 2 x 2 x 2 cubes the population may write.
//!
//! # What runs
//!
//! For the population of cube `c` (area `c * 16 + 8 .. c * 16 + 24` on each
//! axis; the region is the 32 blocks from `c * 16`):
//!
//! 1. A quick look at 16 columns of the area: no surface (all air or all
//!    solid) means nothing to decorate.
//! 2. The biomes of the area's 16 quarts; the union of their vegetal
//!    features, the first biome's order first.
//! 3. Each feature, with its own random drawn from the population random, is
//!    placed at the area's corner. Its placement modifiers pick the columns;
//!    the `biome` modifier asks whether the column's biome has the feature;
//!    heightmaps look at the 32 blocks of the region and answer only for a
//!    surface block inside the area, so every surface belongs to exactly one
//!    population (no tree is placed twice).
//!
//! # Mask
//!
//! [`Vegetation::with_mask`] adds a [`VegetationMask`]: a feature's biome gate then also asks the mask
//! whether the named feature may grow at the column. Earth uses it to thin or forbid features by land
//! cover (`cubic::earth::vegetation`). The random is used exactly the same way with or without a mask, and
//! without one (the default) nothing changes.
//!
//! # Snow
//!
//! [`Vegetation::with_snow`] adds the top layer step after the vegetal features: the biomes' own
//! `freeze_top_layer` (snow on cold tops and leaves, ice on still water), run on the same region
//! with a random of its own. The mask does not gate it; only the biome list does. The tops it finds are
//! those of the whole region, so the population that sees a canopy's top snows it (the same layer from
//! two populations is one layer), as long as the trees grew before that population's pass. Vanilla drops the
//! temperature above sea level + 17; this pass hides that drop (the region's sea level is far above
//! any block), so only a biome's own temperature decides. Earth's biomes already come from the real
//! temperature at the real height. Without `with_snow` (the default) nothing changes.
//!
//! # Not done
//!
//! * The other decoration steps (ores and lakes are CubicWorldGen's own, caves are carved at
//!   generation).
//! * A feature that grows past the region (the tallest trees on a surface
//!   high in the area) is cut at the region's top.
//! * Order within the vanilla biome feature graph is approximated by the
//!   first biome's order.

use std::sync::{Arc, OnceLock};

use horizon_core::position::{BlockPos, SectionPos};
use horizon_core::random::LegacyRandom;

use super::biomes::BiomeColumns;
use super::populate::CubeWorld;
use super::rng::JRandom;
use crate::level::{Behaviours, LevelAccess, place_biome_feature};
use crate::registry::BlockStateId;
use crate::worldgen::biome::BiomeId;
use crate::worldgen::generator::GeneratorData;
use crate::worldgen::live::LiveFeatures;

/// The decoration step of vegetation (`GenerationStep.Decoration.VEGETAL_DECORATION`).
const VEGETAL_STEP: usize = 9;
/// The decoration step of the top layer (`GenerationStep.Decoration.TOP_LAYER_MODIFICATION`).
const TOP_LAYER_STEP: usize = 10;
/// The region is 2 cubes wide; the area a population decorates its middle.
const REGION: i32 = 32;
/// The sea level the snow pass shows the features: so high that vanilla's height drop of the
/// temperature (above sea level + 17) never applies.
const SNOW_SEA_LEVEL: i32 = i32::MAX / 4;

/// Whether a named feature may grow at a block column, asked on top of the biome. Called from the
/// threads that populate, possibly many times per cube, so it should be quick (it may block on data).
pub trait VegetationMask: Send + Sync {
    /// `feature` is the placed feature's name without the `minecraft:` namespace (`trees_plains`).
    fn allows(&self, x: i32, z: i32, feature: &str) -> bool;
}

/// The vegetation of one generator.
pub struct Vegetation {
    data: Arc<GeneratorData>,
    features: Arc<LiveFeatures>,
    behaviours: Arc<Behaviours>,
    seed: i64,
    air: u32,
    /// The vegetal features of each biome id, built when first asked.
    lists: Vec<OnceLock<Arc<[String]>>>,
    /// An extra say on where features grow; `None` for the plain Cubic generator.
    mask: Option<Arc<dyn VegetationMask>>,
    /// Whether the top layer step runs after the vegetal one (Earth's `snow`).
    snow: bool,
    /// The top layer features of each biome id, built when first asked.
    top_lists: Vec<OnceLock<Arc<[String]>>>,
}

impl Vegetation {
    /// `None` when the bundle has no feature resources or block behaviours.
    pub fn new(data: &Arc<GeneratorData>, seed: i64) -> Option<Self> {
        let features = Arc::new(LiveFeatures::new(data).ok()??);
        let behaviours = data.behaviours()?.clone();
        let air = data.registry().default_state("minecraft:air")?.0;
        let lists = (0..data.biome_names().len())
            .map(|_| OnceLock::new())
            .collect();
        let top_lists = (0..data.biome_names().len())
            .map(|_| OnceLock::new())
            .collect();
        Some(Self {
            data: data.clone(),
            features,
            behaviours,
            seed,
            air,
            lists,
            mask: None,
            snow: false,
            top_lists,
        })
    }

    /// The same vegetation that also runs the top layer step (snow and ice) after the vegetal one.
    pub fn with_snow(mut self) -> Self {
        self.snow = true;
        self
    }

    /// The same vegetation where a feature also needs `mask` to allow its column.
    pub fn with_mask(mut self, mask: Arc<dyn VegetationMask>) -> Self {
        self.mask = Some(mask);
        self
    }

    fn list(&self, biome: BiomeId) -> Arc<[String]> {
        self.step_list(&self.lists, biome, VEGETAL_STEP)
    }

    /// The features of decoration step `step` of `biome`, kept in `cells`.
    fn step_list(
        &self,
        cells: &[OnceLock<Arc<[String]>>],
        biome: BiomeId,
        step: usize,
    ) -> Arc<[String]> {
        match cells.get(biome.0 as usize) {
            Some(cell) => cell
                .get_or_init(|| self.features.biome_step_features(biome, step).into())
                .clone(),
            None => Arc::from(Vec::new()),
        }
    }

    /// Decorates the area of `cube`'s population. `columns` gives the biome
    /// of each quart (the generator's own source).
    pub fn decorate(
        &self,
        cube: SectionPos,
        random: &mut JRandom,
        world: &mut dyn CubeWorld,
        columns: &dyn BiomeColumns,
    ) {
        let min = [cube.x * 16, cube.y * 16, cube.z * 16];
        // The random is used the same way whether or not anything grows.
        let base = random.long();
        if !has_surface(world, min, self.air) {
            return;
        }
        // The biome of each quart of the region, by name and id.
        let ids: Vec<BiomeId> = (0..64)
            .map(|i| {
                let (qx, qz) = (i % 8, i / 8);
                let id = columns.biome((min[0] >> 2) + qx, (min[2] >> 2) + qz);
                self.data
                    .biome_id(columns.biome_name(id))
                    .unwrap_or(BiomeId(0))
            })
            .collect();
        let names: Vec<Arc<str>> = ids
            .iter()
            .map(|&id| Arc::from(self.data.biome_name(id)))
            .collect();
        // Union of the area's biomes' features, in first-seen order.
        let mut order: Vec<String> = Vec::new();
        let mut seen_biomes: Vec<BiomeId> = Vec::new();
        for qz in 2..6 {
            for qx in 2..6 {
                let id = ids[qz * 8 + qx];
                if seen_biomes.contains(&id) {
                    continue;
                }
                seen_biomes.push(id);
                for name in self.list(id).iter() {
                    if !order.contains(name) {
                        order.push(name.clone());
                    }
                }
            }
        }
        if order.is_empty() && !self.snow {
            return;
        }
        let lists: Vec<Arc<[String]>> = ids.iter().map(|&id| self.list(id)).collect();
        let mut level = Region {
            world,
            min,
            random: LegacyRandom::new(base),
            names,
            seed: self.seed,
            sea_level: None,
        };
        let origin = [min[0] + 8, min[1] + 8, min[2] + 8];
        let surface = (min[1] + 8, min[1] + 24);
        let mask = self.mask.as_deref();
        for (i, name) in order.iter().enumerate() {
            let bare = name.strip_prefix("minecraft:").unwrap_or(name);
            let mut feature_random = LegacyRandom::new(
                base ^ (i as i64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15_u64 as i64),
            );
            let gate = |p: [i32; 3]| -> bool {
                let (qx, qz) = ((p[0] - min[0]) >> 2, (p[2] - min[2]) >> 2);
                if !(0..8).contains(&qx) || !(0..8).contains(&qz) {
                    return false;
                }
                lists[(qz * 8 + qx) as usize].contains(name)
                    && mask.is_none_or(|m| m.allows(p[0], p[2], bare))
            };
            place_biome_feature(
                &self.behaviours,
                &self.features,
                &mut level,
                name,
                origin,
                &mut feature_random,
                &gate,
                surface,
            );
        }
        if self.snow {
            self.top_layer(&mut level, &ids, base, origin);
        }
    }

    /// The top layer step of the area: each biome's features of it, from the first biome's order,
    /// placed like the vegetal ones but without the mask (land cover has no say on snow).
    fn top_layer(&self, level: &mut Region<'_>, ids: &[BiomeId], base: i64, origin: [i32; 3]) {
        let min = level.min;
        // Every column of the area takes its top from the whole region, not from the area's own band of
        // heights: the layer is the same whichever population puts it, and a canopy above the band is
        // snowed by the population whose band holds its top.
        let surface = (min[1], min[1] + REGION);
        let mut order: Vec<String> = Vec::new();
        let mut seen: Vec<BiomeId> = Vec::new();
        for qz in 2..6 {
            for qx in 2..6 {
                let id = ids[qz * 8 + qx];
                if seen.contains(&id) {
                    continue;
                }
                seen.push(id);
                for name in self.step_list(&self.top_lists, id, TOP_LAYER_STEP).iter() {
                    if !order.contains(name) {
                        order.push(name.clone());
                    }
                }
            }
        }
        if order.is_empty() {
            return;
        }
        let lists: Vec<Arc<[String]>> = ids
            .iter()
            .map(|&id| self.step_list(&self.top_lists, id, TOP_LAYER_STEP))
            .collect();
        // No height drop of the temperature: see the module's Snow section.
        level.sea_level = Some(SNOW_SEA_LEVEL);
        for (i, name) in order.iter().enumerate() {
            let mut feature_random =
                LegacyRandom::new(base ^ (i as i64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D));
            let gate = |p: [i32; 3]| -> bool {
                let (qx, qz) = ((p[0] - min[0]) >> 2, (p[2] - min[2]) >> 2);
                if !(0..8).contains(&qx) || !(0..8).contains(&qz) {
                    return false;
                }
                lists[(qz * 8 + qx) as usize].contains(name)
            };
            place_biome_feature(
                &self.behaviours,
                &self.features,
                level,
                name,
                origin,
                &mut feature_random,
                &gate,
                surface,
            );
        }
        level.sea_level = None;
    }
}

/// Whether some sampled column of the area has air over solid in it.
fn has_surface(world: &dyn CubeWorld, min: [i32; 3], air: u32) -> bool {
    for dz in 0..4 {
        for dx in 0..4 {
            let (x, z) = (min[0] + 8 + dx * 4 + 1, min[2] + 8 + dz * 4 + 1);
            let mut above_air = world.block([x, min[1] + 24, z]) == air;
            for y in (min[1] + 7..min[1] + 24).rev() {
                let is_air = world.block([x, y, z]) == air;
                if above_air && !is_air {
                    return true;
                }
                above_air = is_air;
            }
        }
    }
    false
}

/// The 32 blocks of a population's region as a level for the features.
struct Region<'a> {
    world: &'a mut dyn CubeWorld,
    min: [i32; 3],
    random: LegacyRandom,
    /// The biome name of each quart of the region (8 x 8).
    names: Vec<Arc<str>>,
    seed: i64,
    /// A sea level of the region's own; `None` is the level's.
    sea_level: Option<i32>,
}

impl Region<'_> {
    fn inside(&self, p: BlockPos) -> bool {
        (0..REGION).contains(&(p.x.wrapping_sub(self.min[0])))
            && (0..REGION).contains(&(p.y.wrapping_sub(self.min[1])))
            && (0..REGION).contains(&(p.z.wrapping_sub(self.min[2])))
    }
}

impl crate::level::BlockEntityNeighbors for Region<'_> {}

impl LevelAccess for Region<'_> {
    fn block_state(&self, pos: BlockPos) -> BlockStateId {
        if self.inside(pos) {
            BlockStateId(self.world.block([pos.x, pos.y, pos.z]))
        } else {
            BlockStateId(0)
        }
    }
    fn set_raw(&mut self, pos: BlockPos, state: BlockStateId) -> Option<BlockStateId> {
        if !self.inside(pos) {
            return None;
        }
        let old = self.world.block([pos.x, pos.y, pos.z]);
        self.world.set_block([pos.x, pos.y, pos.z], state.0);
        Some(BlockStateId(old))
    }
    fn is_loaded(&self, pos: BlockPos) -> bool {
        self.inside(pos)
    }
    fn min_y(&self) -> i32 {
        self.min[1]
    }
    fn max_y(&self) -> i32 {
        self.min[1] + REGION
    }
    fn random(&mut self) -> &mut LegacyRandom {
        &mut self.random
    }
    fn game_time(&self) -> i64 {
        0
    }
    fn biome_name(&self, pos: BlockPos) -> Option<&str> {
        let (qx, qz) = ((pos.x - self.min[0]) >> 2, (pos.z - self.min[2]) >> 2);
        if !(0..8).contains(&qx) || !(0..8).contains(&qz) {
            return None;
        }
        self.names.get((qz * 8 + qx) as usize).map(|n| &**n)
    }
    fn world_seed(&self) -> i64 {
        self.seed
    }
    fn sea_level(&self) -> i32 {
        self.sea_level
            .unwrap_or_else(|| crate::level::sea_level_of(self.level_key()))
    }
}

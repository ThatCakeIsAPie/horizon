//! Which of 26.3's vegetation grows where on Earth: the density of the world's plants follows ESA land cover.
//!
//! Earth runs the same vanilla vegetal-step biome features as Cubic ([`crate::cubic::custom::vegetation`]).
//! The biome table ([`super::biome`]) is coarse (about 500 m), so on its own it would put a forest on a city
//! block or a field. [`EarthVegetationMask`] is a [`VegetationMask`]: it reads the land class of the exact
//! block column a feature is placed at and lets a feature grow there only if the class allows its kind.
//!
//! # Kinds of feature
//!
//! [`FeatureKind::of`] sorts a placed feature by its name:
//!
//! | Kind | Names | Examples |
//! |---|---|---|
//! | Tree | contains `tree` or `bamboo`; `birch_tall`, `dark_forest_vegetation`, `mushroom_island_vegetation`, `pale_garden_vegetation`; starts with `fallen_` | `trees_plains`, `trees_water`, `bamboo_light`, `birch_tall`, `rooted_azalea_tree` |
//! | Water | contains `seagrass`, `kelp`, `sea_pickle`, `coral`; `warm_ocean_vegetation` | `seagrass_normal`, `kelp_cold`, `sea_pickle` |
//! | Dry | contains `cactus`, `dead_bush`, `dry_grass` | `patch_cactus_desert`, `patch_dead_bush_2`, `patch_dry_grass_desert` |
//! | Plant | everything else | `patch_grass_plain`, `flower_plains`, `patch_bush`, `patch_large_fern`, `patch_berry_common`, `patch_sugar_cane`, `patch_pumpkin`, `patch_melon`, `brown_mushroom_normal`, `patch_waterlily`, `vines`, `glow_lichen`, `patch_firefly_bush_near_water` |
//!
//! The list was checked against the 26.3 bundle: the union of `LiveFeatures::biome_step_features(biome, 9)`
//! over all biomes is 118 names (`vegetation_tests` checks the 17 Earth biomes' features against the kinds).
//! Three names do not say `tree` and are Trees by hand: `birch_tall`, `pale_garden_vegetation`, and the two
//! mushroom features (`dark_forest_vegetation` also places trees). Nether and End features (`*_fungi`,
//! `chorus_plant`, `weeping_vines`) land in Plant; no Earth biome has them. Lily pads (`patch_waterlily`) are
//! Plant, so they grow only where a class allows plants: swamps and mangroves, not open lakes.
//!
//! # What each class allows
//!
//! | Land class | Tree | Water | Dry | Plant |
//! |---|---|---|---|---|
//! | Tree cover | yes | yes | yes | yes |
//! | Shrub | on about 20 % of the columns | no | yes | yes |
//! | Grass | on about 2 % of the columns | no | yes | yes |
//! | Crop | no | no | no | yes |
//! | Moss and lichen | no | no | no | yes |
//! | Bare | no | no | yes | no |
//! | Built-up, snow and ice | no | no | no | no |
//! | Water | no | yes | no | no |
//! | Wetland, mangrove | yes | yes | yes | yes |
//! | No data | yes | yes | yes | yes |
//!
//! In a dry biome ([`Biome::few_plants`]: desert, badlands, eroded badlands, savanna, savanna plateau; a world with
//! climate) Plant features on shrub and grass grow on only [`PLANT_SHARE`](super::dry::PLANT_SHARE) (30 %) of the columns, by the same kind
//! of fixed column hash, so dry ground has few plants. Dead bushes, cactus and dry grass (Dry) are the dry biomes'
//! own and are not thinned; only dry biomes have them, so wet places are unchanged.
//!
//! A tree allowed on a share of columns uses a fixed hash of the column, not the world seed: the same columns
//! whatever the world, and the vanilla feature still decides which of them get a tree, so the share thins
//! vanilla's density by that factor.
//!
//! The mask reads the class at one block (the world's scale), like the surface. It waits for land cover
//! tiles like the generator's other blocking reads and answers `false` if the world is shutting down.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use horizon_earth::{LandClass, LoadError};

use super::biome::Biome;
use super::dry::few_plants_allow;
use super::generator::{EarthGenerator, Gap, Rect};
use crate::cubic::custom::vegetation::VegetationMask;

/// The version of the vegetation rules above. It goes up when what grows where changes, and is in the terrain
/// key while land cover and vegetation are on.
pub const VEGETATION_RULES: u32 = 2;
/// The version of the far view's trees (`far_trees`). It is in the terrain key while vegetation is on, so far
/// caches made before trees were drawn far away are rebuilt.
pub const FAR_TREE_RULES: u32 = 1;
/// The version of the snow rules (snow on cold ground and trees, ice on cold still water), near and far. It is
/// in the terrain key while vegetation and `snow` are on.
pub const SNOW_RULES: u32 = 1;

/// Entries the mask's biome cache holds before it starts over.
const BIOME_CACHE: usize = 8192;
/// Columns out of 1,000 that allow a tree on shrub.
const SHRUB_TREES: u64 = 200;
/// Columns out of 1,000 that allow a tree on grass.
const GRASS_TREES: u64 = 20;

/// What a placed feature is, for the land class to allow or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FeatureKind {
    /// Trees, bamboo, huge mushrooms.
    Tree,
    /// Seagrass, kelp, sea pickles, coral.
    Water,
    /// Cactus, dead bushes, dry grass.
    Dry,
    /// Grass, ferns, flowers, bushes, berries, crops, small mushrooms, and the rest.
    Plant,
}

impl FeatureKind {
    /// The kind of the placed feature `name` (with or without the `minecraft:` namespace); see the table above.
    pub fn of(name: &str) -> Self {
        let name = name.strip_prefix("minecraft:").unwrap_or(name);
        let any = |parts: &[&str]| parts.iter().any(|p| name.contains(p));
        if any(&["tree", "bamboo"])
            || name.starts_with("fallen_")
            || matches!(
                name,
                "birch_tall"
                    | "dark_forest_vegetation"
                    | "mushroom_island_vegetation"
                    | "pale_garden_vegetation"
            )
        {
            Self::Tree
        } else if any(&["seagrass", "kelp", "sea_pickle", "coral"])
            || name == "warm_ocean_vegetation"
        {
            Self::Water
        } else if any(&["cactus", "dead_bush", "dry_grass"]) {
            Self::Dry
        } else {
            Self::Plant
        }
    }
}

/// A fixed hash of a column to 0..1000 (splitmix64 finalizer).
fn column_bucket(x: i32, z: i32) -> u64 {
    super::dry::column_bucket(x, z, 0)
}

/// The share (0 to 1) of columns [`allowed`] lets a tree grow on in a land class: the table above.
pub fn tree_share(class: LandClass) -> f64 {
    match class {
        LandClass::Tree | LandClass::Wetland | LandClass::Mangrove | LandClass::NoData => 1.0,
        LandClass::Shrub => SHRUB_TREES as f64 / 1000.0,
        LandClass::Grass => GRASS_TREES as f64 / 1000.0,
        _ => 0.0,
    }
}

/// Whether a feature of `kind` may grow at the column `(x, z)` whose land class is `class`; the table above.
/// The biome-blind form of [`allowed_in`].
pub fn allowed(class: LandClass, kind: FeatureKind, x: i32, z: i32) -> bool {
    allowed_in(class, kind, x, z, None)
}

/// [`allowed`] where the biome of the column is `biome` (when known): in a biome that carries few plants
/// ([`Biome::few_plants`]: deserts, badlands, savannas) plant features grow on only [`PLANT_SHARE`](super::dry::PLANT_SHARE) of the
/// shrub and grass columns.
pub fn allowed_in(
    class: LandClass,
    kind: FeatureKind,
    x: i32,
    z: i32,
    biome: Option<Biome>,
) -> bool {
    let sparse = matches!(class, LandClass::Shrub | LandClass::Grass)
        && kind == FeatureKind::Plant
        && biome.is_some_and(Biome::few_plants);
    if sparse && !few_plants_allow(x, z) {
        return false;
    }
    use FeatureKind::{Dry, Plant, Tree, Water};
    match class {
        LandClass::Tree => true,
        LandClass::Shrub => match kind {
            Tree => column_bucket(x, z) < SHRUB_TREES,
            Plant | Dry => true,
            Water => false,
        },
        LandClass::Grass => match kind {
            Tree => column_bucket(x, z) < GRASS_TREES,
            Plant | Dry => true,
            Water => false,
        },
        LandClass::Crop | LandClass::MossLichen => kind == Plant,
        LandClass::Bare => kind == Dry,
        LandClass::BuiltUp | LandClass::SnowIce => false,
        LandClass::Water => kind == Water,
        LandClass::Wetland | LandClass::Mangrove | LandClass::NoData => true,
    }
}

/// The vegetation mask of an Earth world with land cover: features grow where the class of the exact column
/// allows their kind ([`allowed`]).
pub struct EarthVegetationMask {
    generator: Arc<EarthGenerator>,
    /// The biomes asked for lately, by 4 by 4 column.
    cache: Mutex<HashMap<(i32, i32), Biome>>,
}

impl EarthVegetationMask {
    pub fn new(generator: Arc<EarthGenerator>) -> Self {
        Self {
            generator,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The biome of the block column `(x, z)` in a world with climate; `None` without (the biomes there do
    /// not say which places are dry). Remembered per 4 by 4 column.
    fn biome_at(&self, x: i32, z: i32) -> Option<Biome> {
        let g = &*self.generator;
        if !g.climate_wanted() || g.climate_source().is_none() {
            return None;
        }
        let key = (x >> 2, z >> 2);
        if let Some(&b) = self
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
        {
            return Some(b);
        }
        let biome = g.biome_at_quart_blocking(key.0, key.1);
        // A read cut short by the world shutting down is `plains` and not worth keeping.
        if !g.is_cancelled() {
            let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
            if cache.len() >= BIOME_CACHE {
                cache.clear();
            }
            cache.insert(key, biome);
        }
        Some(biome)
    }

    /// The class of the block column `(x, z)`, loading what it lacks; `None` without land cover or when the
    /// world shuts down first.
    fn class_at(&self, x: i32, z: i32) -> Option<LandClass> {
        let g = &*self.generator;
        let land = g.land_source()?;
        let metres = g.field().projection().metres_per_block();
        let rect = Rect {
            x0: x,
            z0: z,
            x1: x,
            z1: z,
            zoom: g.base_zoom(),
        };
        g.wait_for(rect, || {
            land.class_grid_blocking(x, z, 1, 1, 1, metres)
                .map_err(|e| Gap::Load(LoadError::Fetch(e)))
        })?
        .first()
        .copied()
    }
}

impl VegetationMask for EarthVegetationMask {
    fn allows(&self, x: i32, z: i32, feature: &str) -> bool {
        match self.class_at(x, z) {
            Some(class) => {
                let kind = FeatureKind::of(feature);
                // The biome is asked for only where it can matter.
                let biome = (kind == FeatureKind::Plant
                    && matches!(class, LandClass::Shrub | LandClass::Grass))
                .then(|| self.biome_at(x, z))
                .flatten();
                allowed_in(class, kind, x, z, biome)
            }
            None => false,
        }
    }
}

//! Dry places: what a dry climate does to the ground, the biome and the plants of an Earth world.
//!
//! Every threshold and share of the dry rules is a named constant here, so they can be tuned in one place; the
//! rules themselves are in [`super::surface`] (the top of a column), [`super::biome`] (the biome) and
//! [`super::vegetation`] (the plants), and `docs/earth.md` ("Dry places") describes them whole.
//!
//! The rules apply only where the world has the real climate ([`horizon_earth::Climate`]) at the column. The
//! measure of dryness is the Koppen-style aridity ratio `r = P / (20 (T + 7))` ([`aridity`]; `T` mean annual
//! temperature in degrees C, `P` annual rain in mm).

use horizon_core::random::XoroshiroRandom;
use horizon_earth::Climate;

use super::generator::EarthBlocks;
use crate::registry::BlockStateId;
use crate::worldgen::material_surface::{ClayBandBlocks, ClayBands};

/// An aridity ratio under this is a dry climate (steppe and desert). Koppen's dry boundary for rain spread over
/// the year is `r` = 1 and for summer rain about 1.3 to 1.4; CHELSA gives the Grand Canyon's South Rim 1.26
/// (10.0 C, 429 mm), Denver 1.28 and Lubbock 1.13 (real semi-arid steppe), while Rome (1.69), Paris (1.80) and
/// Flagstaff's pine forest (1.68) stay out. At 10 C this is about 440 mm of rain a year, at 20 C about 700 mm.
pub const DRY_ARIDITY: f64 = 1.3;
/// An aridity ratio under this is desert: sparser ground cover (Koppen's BW). About 170 mm a year at 10 C,
/// 270 mm at 20 C; Phoenix 0.36, Las Vegas 0.25, the Air 0.18; Phantom Ranch in the Grand Canyon is 0.57.
pub const DESERT_ARIDITY: f64 = 0.5;
/// A mean annual temperature (C) from which dry bare ground is sand dunes rather than barren red ground (the
/// start of vanilla's warm and hot rows).
pub const HOT_C: f64 = 18.0;
/// The mean annual temperature (C) from which a dry flat grassland or shrubland is savanna rather than the
/// table's own cold pick (vanilla's temperate row).
pub const SAVANNA_FROM_C: f64 = 8.0;
/// Blocks one terracotta band spans in height. Vanilla's bands are one block; real strata are tens of metres
/// thick, which on a 1,000 m wall would be pin-stripes.
pub const BAND_BLOCKS: i64 = 4;
/// Blocks of banded terracotta under the top of a dry column; stone below.
pub const DRY_ROCK_DEPTH: i64 = 32;
/// The relief (m of elevation range over the middle and four neighbours half a biome cell away) from which dry
/// open ground is badlands.
pub const RUGGED_M: f64 = 150.0;
/// The relief (m) from which dry rugged ground is eroded badlands.
pub const ERODED_M: f64 = 400.0;
/// Columns out of 1,000 where a dry biome ([`super::biome::Biome::few_plants`]) lets a plant feature grow on
/// shrub and grass.
pub const PLANT_SHARE: u64 = 300;
/// The seed of the terracotta bands: fixed, not the world's, so every world and every cache agrees.
pub const BAND_SEED: i64 = 0x00D2_1ED5_7A7A;
/// The bands a stripe pattern has: vanilla's table size.
pub const BAND_COUNT: usize = 192;

/// The shares (out of 1,000) of the tops of a dry column; the rest is red sand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mix {
    pub grass_block: u16,
    pub coarse_dirt: u16,
}

/// Desert (`r` under [`DESERT_ARIDITY`]) shrub and grass.
pub const DESERT_MIX: Mix = Mix {
    grass_block: 150,
    coarse_dirt: 450,
};
/// Dry scrub (`r` between [`DESERT_ARIDITY`] and [`DRY_ARIDITY`]) shrub and grass.
pub const SCRUB_MIX: Mix = Mix {
    grass_block: 400,
    coarse_dirt: 400,
};
/// Dry bare ground that is not hot.
pub const BARREN_MIX: Mix = Mix {
    grass_block: 0,
    coarse_dirt: 450,
};

/// The block on top of a sparse dry column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DryTop {
    GrassBlock,
    CoarseDirt,
    RedSand,
}

impl Mix {
    /// The top a column with `bucket` (0..1000, [`column_bucket`]) takes.
    pub fn pick(self, bucket: u16) -> DryTop {
        if bucket < self.grass_block {
            DryTop::GrassBlock
        } else if bucket < self.grass_block + self.coarse_dirt {
            DryTop::CoarseDirt
        } else {
            DryTop::RedSand
        }
    }
}

/// The aridity ratio of a climate: `P / (20 (T + 7))`, with the divisor at least 20 (very cold air holds no
/// rain, and the ratio would blow up).
pub fn aridity(temperature_c: f64, precipitation_mm: f64) -> f64 {
    precipitation_mm.max(0.0) / (20.0 * (temperature_c + 7.0).max(1.0))
}

/// Whether a climate of temperature `t` (C) and aridity `r` is dry: `r` under [`DRY_ARIDITY`] and not frozen
/// (below 0 C the ground is under snow and the biome is a frozen one).
pub fn is_dry(t: f64, r: f64) -> bool {
    r < DRY_ARIDITY && t >= 0.0
}

/// A fixed hash of the block column `(x, z)` and a `salt` to 0..1000 (splitmix64 finalizer); no world seed, so
/// the near and far view and every world agree. Salt 0 is the column hash the vegetation rules use.
pub fn column_bucket(x: i32, z: i32, salt: u64) -> u64 {
    let mut h = (u64::from(x as u32) << 32 | u64::from(z as u32))
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (h ^ (h >> 31)) % 1000
}

/// The salt of [`DryColumn::bucket`].
const TOP_SALT: u64 = 0xD27;
/// The salt of the few-plants hash.
const PLANT_SALT: u64 = 0x91A;

/// Whether a plant feature may grow at `(x, z)` in a biome that carries few plants: [`PLANT_SHARE`] of columns.
pub fn few_plants_allow(x: i32, z: i32) -> bool {
    column_bucket(x, z, PLANT_SALT) < PLANT_SHARE
}

/// What the dry rules need to know of a column: the climate at it and the column's own hash.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DryColumn {
    /// Mean annual temperature, C.
    pub temperature_c: f64,
    /// The aridity ratio ([`aridity`]).
    pub aridity: f64,
    /// 0..1000, a fixed hash of the block column, choosing its top out of a [`Mix`].
    pub bucket: u16,
}

impl DryColumn {
    /// The column at block `(x, z)` with `climate`.
    pub fn new(climate: Climate, x: i32, z: i32) -> Self {
        let t = f64::from(climate.temperature_c);
        Self {
            temperature_c: t,
            aridity: aridity(t, f64::from(climate.precipitation_mm)),
            bucket: column_bucket(x, z, TOP_SALT) as u16,
        }
    }

    /// A column of temperature `t`, aridity `r` and hash `bucket`, for tests and callers that have no climate
    /// value.
    pub fn of(t: f64, r: f64, bucket: u16) -> Self {
        Self {
            temperature_c: t,
            aridity: r,
            bucket,
        }
    }

    pub fn is_dry(&self) -> bool {
        is_dry(self.temperature_c, self.aridity)
    }

    pub fn is_hot(&self) -> bool {
        self.temperature_c >= HOT_C
    }

    pub fn is_desert(&self) -> bool {
        self.aridity < DESERT_ARIDITY
    }
}

/// The index into the band table of the terracotta at absolute height `y`: one table entry per [`BAND_BLOCKS`]
/// of height, repeating every `BAND_COUNT * BAND_BLOCKS` blocks. Level strata: it depends on `y` alone.
pub fn band_index(y: i64) -> usize {
    y.div_euclid(BAND_BLOCKS).rem_euclid(BAND_COUNT as i64) as usize
}

/// The terracotta band table: Horizon's port of vanilla's clay bands, made once from [`BAND_SEED`], as block
/// state ids.
pub fn band_table(blocks: &EarthBlocks) -> Box<[u32; BAND_COUNT]> {
    let id = BlockStateId;
    let bands = ClayBands::generate(
        &mut XoroshiroRandom::new(BAND_SEED),
        ClayBandBlocks {
            terracotta: id(blocks.terracotta),
            orange: id(blocks.orange_terracotta),
            yellow: id(blocks.yellow_terracotta),
            brown: id(blocks.brown_terracotta),
            red: id(blocks.red_terracotta),
            white: id(blocks.white_terracotta),
            light_gray: id(blocks.light_gray_terracotta),
        },
    );
    let mut table = Box::new([0_u32; BAND_COUNT]);
    for (out, state) in table.iter_mut().zip(bands.states()) {
        *out = state.0;
    }
    table
}

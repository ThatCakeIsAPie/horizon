//! The mixed tops of two land-cover classes: built-up ground and moss and lichen.
//!
//! A column of either class takes its top block from a fixed table of shares, picked by a fixed hash of the
//! column (no world seed), so the near and far view and every world agree. Every share is a constant here.
//!
//! * **Built-up** ([`CITY_MIX`]): grey roofs and paving (stone, smooth stone, andesite, grey and light grey
//!   concrete, stone bricks) with [`CityTop::Grass`] parks and yards. The hash is of a parcel of
//!   [`PARCEL_BLOCKS`] by [`PARCEL_BLOCKS`] columns, so near ground reads as lots, and far samples (one per
//!   parcel or more) still differ from their neighbours: the distance is a grey and green patchwork, not one
//!   flat colour. In a dry climate a yard is sand ([`CityTop::Sand`]). The ground under it is stone (soil under
//!   grass).
//! * **Moss and lichen** ([`MOSS_MIX`]): mostly moss block, some grass block, the rest bare (stone, gravel,
//!   coarse dirt): cold tundra and lava fields, not a lawn. The hash is of the column.

use super::dry::column_bucket;

/// The side of a built-up parcel, in blocks: the columns of one parcel share a top.
pub const PARCEL_BLOCKS: i32 = 8;
/// Columns out of 1,000 of a built-up grass parcel where a tree may grow (street and garden trees; vanilla's
/// feature and the far view's style still decide).
pub const STREET_TREES: u64 = 300;

/// The top of a built-up column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CityTop {
    Stone,
    SmoothStone,
    Andesite,
    GrayConcrete,
    LightGrayConcrete,
    StoneBricks,
    /// A park or yard: a grass block over dirt.
    Grass,
    /// A yard in a dry climate: sand.
    Sand,
}

/// The top of a moss and lichen column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MossTop {
    Moss,
    Grass,
    Stone,
    Gravel,
    CoarseDirt,
}

/// Shares out of 1,000 of the built-up tops; they add up to 1,000. Grass is a quarter (parks and yards).
pub const CITY_MIX: [(CityTop, u16); 7] = [
    (CityTop::Grass, 250),
    (CityTop::GrayConcrete, 180),
    (CityTop::LightGrayConcrete, 150),
    (CityTop::Stone, 120),
    (CityTop::SmoothStone, 100),
    (CityTop::Andesite, 100),
    (CityTop::StoneBricks, 100),
];

/// Shares out of 1,000 of the moss and lichen tops; they add up to 1,000.
pub const MOSS_MIX: [(MossTop, u16); 5] = [
    (MossTop::Moss, 550),
    (MossTop::Grass, 150),
    (MossTop::Stone, 120),
    (MossTop::Gravel, 100),
    (MossTop::CoarseDirt, 80),
];

/// The salt of the parcel hash.
const CITY_SALT: u64 = 0xC17E;
/// The salt of the moss column hash.
const MOSS_SALT: u64 = 0x4055;

/// The entry of `mix` that `bucket` (0..1000) falls in; the last entry past the end.
pub fn pick<T: Copy>(mix: &[(T, u16)], bucket: u16) -> T {
    let mut edge = 0_u16;
    for &(top, share) in mix {
        edge += share;
        if bucket < edge {
            return top;
        }
    }
    mix[mix.len() - 1].0
}

/// The top of the built-up column `(x, z)`; `dry` when the climate there is dry (a yard is sand then).
pub fn city_top(x: i32, z: i32, dry: bool) -> CityTop {
    let bucket = column_bucket(
        x.div_euclid(PARCEL_BLOCKS),
        z.div_euclid(PARCEL_BLOCKS),
        CITY_SALT,
    );
    match pick(&CITY_MIX, bucket as u16) {
        CityTop::Grass if dry => CityTop::Sand,
        top => top,
    }
}

/// The top of the moss and lichen column `(x, z)`.
pub fn moss_top(x: i32, z: i32) -> MossTop {
    pick(&MOSS_MIX, column_bucket(x, z, MOSS_SALT) as u16)
}

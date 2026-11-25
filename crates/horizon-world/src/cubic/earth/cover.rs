//! The mixed top of moss and lichen ground.
//!
//! A moss and lichen column takes its top block from a fixed table of shares ([`MOSS_MIX`]), picked by a fixed
//! hash of the column (no world seed), so the near and far view and every world agree. Every share is a
//! constant here: mostly moss block, some grass block, the rest bare (stone, gravel, coarse dirt): cold tundra
//! and lava fields, not a lawn. Built-up land has no mix: it is the natural land around it (see
//! `horizon_earth::unbuild`).

use super::dry::column_bucket;

/// The top of a moss and lichen column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MossTop {
    Moss,
    Grass,
    Stone,
    Gravel,
    CoarseDirt,
}

/// Shares out of 1,000 of the moss and lichen tops; they add up to 1,000.
pub const MOSS_MIX: [(MossTop, u16); 5] = [
    (MossTop::Moss, 550),
    (MossTop::Grass, 150),
    (MossTop::Stone, 120),
    (MossTop::Gravel, 100),
    (MossTop::CoarseDirt, 80),
];

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

/// The top of the moss and lichen column `(x, z)`.
pub fn moss_top(x: i32, z: i32) -> MossTop {
    pick(&MOSS_MIX, column_bucket(x, z, MOSS_SALT) as u16)
}

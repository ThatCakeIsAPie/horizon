//! Water in Earth terrain: the sea with its shelf and shores, rivers carved from the water atlas, flat lakes.
//!
//! (Scaffold: the shared types and bounds. The rules land in later commits of this branch; the module docs will
//! explain them in plain words.)

/// The version of the generator's water rules: it goes up when what the same column with the same data is made of
/// changes. It is in the terrain key (with `horizon_earth::WATER_RULES`) while land cover is on.
pub const WATER_GEN_RULES: u32 = 1;

/// The most blocks the water rules lower a column's ground under its elevation surface (a river channel, a lake
/// bed, a shelf, a carved bank). Box classification widens its stone margin by this.
pub const MAX_CARVE: i64 = 32;

/// The most blocks the water rules put a column's top (ground or water) over its elevation surface (a raised bank,
/// water held over low ground). Box classification widens its air margin by this.
pub const MAX_RAISE: i64 = 8;

/// What the bed under a water column is made of (its top ground block).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bed {
    Sand,
    Gravel,
    Mud,
    Clay,
    Stone,
}

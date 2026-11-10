//! Earth: terrain from real-world elevation tiles (`horizon-earth`) as a cubic generator.
//!
//! A column's surface height comes from the elevation under it; below the surface are soil and stone, and
//! the ground under the sea level is sand with water over it; on land the top is grass, bare stone or snow by
//! elevation and latitude ([`surface`]); [`generator`] has the exact rule. The settings
//! of a world are [`EarthSettings`]; [`CustomCubicGenerator::earth`](super::custom::CustomCubicGenerator::earth)
//! wraps the generator for the game's cubic levels.
//!
//! Missing tiles are never guessed: the generator's plain queries give [`NotReady`] naming the tiles to fetch,
//! and its `*_blocking` forms ([`blocking`]) fetch and wait. The far view reads whole nodes in one batch without
//! waiting ([`far`]).

pub mod blocking;
pub mod far;
pub mod generator;
pub mod settings;
pub mod surface;
#[cfg(test)]
mod tests;

pub use generator::{BIOME, EarthBlocks, EarthGenerator, Fill, NotReady};
pub use settings::EarthSettings;
pub use surface::SurfaceKind;

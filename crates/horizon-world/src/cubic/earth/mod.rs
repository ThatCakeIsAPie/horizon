//! Earth: terrain from real-world elevation tiles (`horizon-earth`) as a cubic generator.
//!
//! A column's surface height comes from the elevation under it; below the surface are soil and stone, and
//! the ground under the sea level is sand with water over it; on land the top is grass, bare stone or snow by
//! elevation and latitude ([`surface`]); [`generator`] has the exact rule. The settings
//! of a world are [`EarthSettings`]; [`CustomCubicGenerator::earth`](super::custom::CustomCubicGenerator::earth)
//! wraps the generator for the game's cubic levels.
//!
//! With the `landCover` setting and a source of land cover handed to the generator, the top of a column and its
//! biome follow the map ([`biome`], [`surface`]); without either the terrain is exactly the elevation-only one.
//!
//! With land cover and the `vegetation` setting, vanilla trees, grass and flowers grow, their density following
//! the class of each column ([`vegetation`]).
//!
//! Missing tiles are never guessed: the generator's plain queries give [`NotReady`] naming the tiles to fetch,
//! and its `*_blocking` forms ([`blocking`]) fetch and wait. The far view reads whole nodes in one batch without
//! waiting ([`far`]).

pub mod biome;
pub mod blocking;
#[cfg(test)]
mod climate_tests;
pub mod dry;
#[cfg(test)]
mod dry_tests;
pub mod far;
pub mod far_trees;
#[cfg(test)]
mod far_trees_tests;
pub mod generator;
#[cfg(test)]
mod land_tests;
pub mod settings;
pub mod surface;
#[cfg(test)]
mod tests;
pub mod vegetation;
#[cfg(test)]
mod vegetation_tests;
pub mod water;
#[cfg(test)]
mod water_tests;

pub use biome::{Biome, EarthBiomes};
pub use far_trees::{FarBlocks, FarDecor, MAX_TREE_UP};
pub use generator::{BIOME, EarthBlocks, EarthGenerator, Fill, NotReady};
pub use settings::EarthSettings;
pub use surface::SurfaceKind;
pub use vegetation::{
    EarthVegetationMask, FAR_TREE_RULES, FeatureKind, SNOW_RULES, VEGETATION_RULES,
};

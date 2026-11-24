//! Cubic's ore generation for every cubic world type where Cubic places ores
//! (Custom Cubic, Hybrid outside vanilla's band, Earth): the two-tier ore
//! entries ([`settings`]), their defaults ([`defaults`]) and the ore meteors
//! ([`meteors`]). See docs/cubic-ores.md.

pub mod defaults;
pub mod engine;
pub mod meteors;
pub mod settings;

pub use engine::{OreCtx, OreEngine, RichnessFn};
pub use settings::{
    Background, MeteorKind, MeteorSettings, ORE_GEN_VERSION, OreEntry, OreGenSettings, RangeMode,
    RichMode, RichZone, Shape,
};

#[cfg(test)]
mod defaults_tests;
#[cfg(test)]
mod engine_tests;
#[cfg(test)]
mod meteor_tests;

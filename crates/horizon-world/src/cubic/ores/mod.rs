//! Cubic's ore generation for every cubic world type where Cubic places ores
//! (Custom Cubic, Hybrid outside vanilla's band, Earth): the two-tier ore
//! entries ([`settings`]), their defaults ([`defaults`]) and the ore meteors
//! ([`meteors`]). See docs/cubic-ores.md.

pub mod defaults;
pub mod meteors;
pub mod settings;

pub use settings::{
    Background, MeteorKind, MeteorSettings, ORE_GEN_VERSION, OreEntry, OreGenSettings, RangeMode,
    RichMode, RichZone, Shape,
};

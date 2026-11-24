//! Ore richness from real data. (CONTRACT STUB: Part C replaces the body
//! with the shipped grid; the signature stays.)

/// The version of the richness data and rules (part of an Earth world's
/// cache key when its ores follow the map).
pub const ORE_RICHNESS_RULES: u32 = 0;

/// The richness multiplier of `ore` (short name: `iron`, `copper`, `coal`,
/// `gold`, `redstone`, `lapis`, `diamond`, `emerald`; anything else = 1) at a
/// place, smooth in latitude and longitude (no grid edges), clamped to
/// 0.3 ..= 5.0. 1 = an average place.
pub fn richness(_ore: &str, _latitude: f64, _longitude: f64) -> f64 {
    1.0
}

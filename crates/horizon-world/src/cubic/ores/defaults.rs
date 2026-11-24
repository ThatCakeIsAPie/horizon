//! The default ore entries of new worlds. (CONTRACT STUB: Part A replaces the
//! bodies with the real defaults; the signatures stay.)

use super::settings::OreGenSettings;
use crate::cubic::custom::settings::CustomGeneratorSettings;

impl OreGenSettings {
    /// A new Custom Cubic / Hybrid world's ores: vanilla's bands as tier 1
    /// (static height) plus a small background, meteors on.
    pub fn custom_defaults() -> Self {
        let s = CustomGeneratorSettings::defaults();
        let mut o = Self::from_legacy(
            &s.standard_ores,
            &s.periodic_gaussian_ores,
            s.expected_base_height,
            s.expected_height_variation,
        );
        o.meteors = super::settings::MeteorSettings::default();
        o
    }

    /// A new Earth world's ores: the same ores with tier 1 below the ground,
    /// stone patches by depth, the same background, meteors on, richness map
    /// on.
    pub fn earth_defaults() -> Self {
        Self::custom_defaults()
    }
}

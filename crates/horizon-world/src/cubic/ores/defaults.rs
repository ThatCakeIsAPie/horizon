//! The default ore entries of new worlds.
//!
//! * [`OreGenSettings::custom_defaults`]: 26.3's vanilla ores as tier 1 at the
//!   same Y bands (`from_legacy` of `legacy_vanilla_ores`, base 64 / variation
//!   64), emerald at a tenth outside the mountains, biome multipliers
//!   ([`BIOME_MULTIPLIERS`]) and a background of 8 % of each ore's main band
//!   peak everywhere else, meteors on.
//! * [`OreGenSettings::earth_defaults`]: the same with every band moved from
//!   "Y" to "depth under the ground" (`depth = 64 - Y`), richness map on.

use super::settings::{Background, MeteorSettings, OreEntry, OreGenSettings, RichMode, Shape};
use crate::cubic::custom::settings::{MOUNTAIN_BIOMES, legacy_vanilla_ores};

/// Vanilla's reference ground level: `depth = SEA - Y` for Earth's bands.
const SEA: f64 = 64.0;

/// The background's share of the main band's peak (veins per cube).
const BACKGROUND_SHARE: f64 = 0.08;

/// Emerald's intensity outside the mountains, as a share of vanilla's (which
/// the x10 multiplier in `MOUNTAIN_BIOMES` restores).
const EMERALD_BASE_SHARE: f64 = 0.1;

const OCEANS: &[&str] = &[
    "ocean",
    "deep_ocean",
    "cold_ocean",
    "deep_cold_ocean",
    "lukewarm_ocean",
    "deep_lukewarm_ocean",
    "warm_ocean",
    "frozen_ocean",
    "deep_frozen_ocean",
];
const DRY: &[&str] = &["desert", "badlands", "eroded_badlands", "wooded_badlands"];
const SNOWY: &[&str] = &["snowy_plains", "ice_spikes", "snowy_taiga", "snowy_beach"];
const WINDSWEPT: &[&str] = &[
    "windswept_hills",
    "windswept_gravelly_hills",
    "windswept_forest",
];
const WET: &[&str] = &["swamp", "mangrove_swamp"];
const JUNGLE_AND_SWAMP: &[&str] = &[
    "jungle",
    "bamboo_jungle",
    "sparse_jungle",
    "swamp",
    "mangrove_swamp",
];
const COLD_FOR_GOLD: &[&str] = &[
    "snowy_plains",
    "ice_spikes",
    "snowy_taiga",
    "frozen_river",
    "snowy_beach",
];
const OPEN_LAND: &[&str] = &["plains", "sunflower_plains", "savanna", "savanna_plateau"];

/// How much richer or poorer each ore is in some biomes: (ore short name,
/// multiplier, biomes). Applied to every entry of the ore, both tiers. A
/// biome left out is 1; none is below 0.5 so no ore ever vanishes.
pub const BIOME_MULTIPLIERS: &[(&str, f64, &[&str])] = &[
    ("coal", 0.5, DRY),
    ("iron", 0.6, OCEANS),
    ("iron", 1.5, WINDSWEPT),
    ("copper", 1.5, WET),
    ("copper", 0.6, SNOWY),
    ("gold", 0.5, COLD_FOR_GOLD),
    ("redstone", 1.5, DRY),
    ("redstone", 0.6, JUNGLE_AND_SWAMP),
    ("lapis", 0.5, OPEN_LAND),
    ("diamond", 0.6, OCEANS),
];

/// The main band of each ore, the one whose peak the background follows:
/// (ore short name, shape, spawn size, min Y, max Y) as `from_legacy` leaves
/// them.
const MAIN_BANDS: &[(&str, Shape, i32, f64, f64)] = &[
    ("coal", Shape::Bell, 17, 0.0, 192.0),
    ("iron", Shape::Bell, 9, -24.0, 56.0),
    ("copper", Shape::Bell, 10, -16.0, 112.0),
    ("gold", Shape::Bell, 9, -96.0, -32.0),
    ("redstone", Shape::Bell, 8, -96.0, -32.0),
    ("diamond", Shape::Bell, 8, -144.0, 16.0),
    ("lapis", Shape::Flat, 7, -64.0, 64.0),
    ("emerald", Shape::Bell, 3, -16.0, 480.0),
];

/// `minecraft:iron_ore` -> `iron` (deepslate variants too); `None` for
/// anything that is not an `_ore` block.
fn ore_short_name(e: &OreEntry) -> Option<&str> {
    let n = e.blockstate.name.strip_prefix("minecraft:")?;
    let n = n.strip_prefix("deepslate_").unwrap_or(n);
    n.strip_suffix("_ore")
}

fn is_main_band(e: &OreEntry, ore: &str) -> bool {
    MAIN_BANDS.iter().any(|&(o, shape, size, lo, hi)| {
        o == ore
            && e.rich.shape == shape
            && e.spawn_size == size
            && (e.rich.min - lo).abs() < 1e-6
            && (e.rich.max - hi).abs() < 1e-6
    })
}

impl OreGenSettings {
    /// A new Custom Cubic / Hybrid world's ores: vanilla's bands as tier 1
    /// (static height) with biome multipliers, plus a small background
    /// everywhere, meteors on.
    pub fn custom_defaults() -> Self {
        let (standard, periodic) = legacy_vanilla_ores();
        let mut o = Self::from_legacy(&standard, &periodic, SEA, 64.0);
        o.meteors = MeteorSettings::default();
        for e in &mut o.ores {
            let Some(ore) = ore_short_name(e).map(str::to_owned) else {
                continue;
            };
            if ore == "emerald" {
                // A tenth of vanilla's veins outside the mountains (tries =
                // ceil(peak), probability = the rest); the x10 multiplier
                // gives the mountains vanilla's number back.
                let peak = e.rich.intensity * e.rich.probability * EMERALD_BASE_SHARE;
                e.rich.intensity = peak.ceil().max(1.0);
                e.rich.probability = (peak / e.rich.intensity).min(1.0);
                e.biomes = None;
                for b in MOUNTAIN_BIOMES {
                    e.biome_multipliers.insert(format!("minecraft:{b}"), 10.0);
                }
            }
            for &(name, m, biomes) in BIOME_MULTIPLIERS {
                if name == ore {
                    for b in biomes {
                        e.biome_multipliers.insert(format!("minecraft:{b}"), m);
                    }
                }
            }
            if is_main_band(e, &ore) {
                e.background = Background {
                    enabled: true,
                    rate: BACKGROUND_SHARE * e.rich.intensity * e.rich.probability,
                    ..Background::default()
                };
            }
        }
        o
    }

    /// A new Earth world's ores: the same ores with tier 1 below the ground
    /// (a band of Y `lo..hi` becomes depth `64 - hi .. 64 - lo`, cut at the
    /// surface; bands wholly above ground are dropped), the same background,
    /// meteors on, richness map on.
    pub fn earth_defaults() -> Self {
        let mut o = Self::custom_defaults();
        o.ores.retain_mut(|e| {
            let r = &mut e.rich;
            let (min, max) = (SEA - r.max, SEA - r.min);
            if max < 0.0 {
                return false;
            }
            r.mode = RichMode::BelowGround;
            r.min = min.max(0.0);
            r.max = max;
            r.peak = r.peak.map(|p| (SEA - p).clamp(r.min, r.max));
            true
        });
        o.richness_map = true;
        o.richness_strength = 1.0;
        o
    }
}

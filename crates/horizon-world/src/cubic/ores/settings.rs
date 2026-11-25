//! The settings of Cubic's two-tier ore generation (`"oreGen"` in a Custom /
//! Hybrid Cubic world's CWG JSON and in an Earth world's settings).
//!
//! Per ore entry:
//! * tier 1, the rich zone ([`RichZone`]): a band in absolute Y
//!   ([`RichMode::StaticHeight`]), a band of depth under the local ground
//!   ([`RichMode::BelowGround`]) or pockets where a 3D noise field is above a
//!   threshold ([`RichMode::Noise`]), with an intensity (veins per cube at the
//!   peak) and a [`Shape`];
//! * tier 2, the background ([`Background`]): a constant rate of veins per
//!   cube everywhere else (optionally within a Y or depth range), or off.
//!
//! [`OreGenSettings::enabled`] switches every ore of the world off at once;
//! [`MeteorSettings`] tunes the ore meteors.
//!
//! A CWG JSON without `oreGen` is an old preset: its `standardOres` and
//! `periodicGaussianOres` generate exactly as before (no background, no
//! meteors). [`OreGenSettings::from_legacy`] turns those lists into entries
//! (tier 1 static height, background off), which generate the same veins.
//! An Earth settings file without `oreGen` has no ores (as before ores
//! existed on Earth).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::cubic::custom::settings::{
    BiomeDesc, BlockStateDesc, GenerationCondition, PeriodicGaussianOreConfig, StandardOreConfig,
    UserFunction,
};

/// The version of the `oreGen` object. 1 = the first two-tier format.
pub const ORE_GEN_VERSION: i32 = 1;

fn neg_inf() -> f64 {
    f64::NEG_INFINITY
}
fn pos_inf() -> f64 {
    f64::INFINITY
}
fn one() -> f64 {
    1.0
}
fn yes() -> bool {
    true
}

/// Every ore setting of one world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct OreGenSettings {
    pub version: i32,
    /// The world-wide ores switch: off = no ore entry generates (meteors have
    /// their own switch).
    pub enabled: bool,
    /// In order: earlier entries are placed first.
    pub ores: Vec<OreEntry>,
    pub meteors: MeteorSettings,
    /// Earth only (ignored elsewhere): whether each ore's intensities follow
    /// the real-data richness map.
    pub richness_map: bool,
    /// Earth only: how strongly the richness map applies, 0 (flat) to 2;
    /// 1 = as the data says. The factor is `richness ^ strength`, clamped.
    pub richness_strength: f64,
}

impl Default for OreGenSettings {
    fn default() -> Self {
        Self {
            version: ORE_GEN_VERSION,
            enabled: true,
            ores: Vec::new(),
            meteors: MeteorSettings::default(),
            richness_map: true,
            richness_strength: 1.0,
        }
    }
}

/// One ore (or stone patch) entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct OreEntry {
    pub blockstate: BlockStateDesc,
    /// Blocks per vein (CWG `spawnSize`), 1 to 64.
    pub spawn_size: i32,
    /// `None` = every biome.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biomes: Option<Vec<BiomeDesc>>,
    /// Which blocks a vein may replace; `None` = stone, deepslate and their
    /// varieties (granite, diorite, andesite, tuff).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub place_block_when: Option<GenerationCondition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_when: Option<GenerationCondition>,
    /// Multiplier on both tiers' intensities by biome name
    /// (`minecraft:badlands` -> 3.0); a biome left out is 1.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub biome_multipliers: BTreeMap<String, f64>,
    pub rich: RichZone,
    pub background: Background,
}

impl Default for OreEntry {
    fn default() -> Self {
        Self {
            blockstate: BlockStateDesc::new("stone"),
            spawn_size: 8,
            biomes: None,
            place_block_when: None,
            generate_when: None,
            biome_multipliers: BTreeMap::new(),
            rich: RichZone::default(),
            background: Background::default(),
        }
    }
}

/// Where tier 1 is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RichMode {
    /// `min`..`max` are block Ys.
    #[default]
    StaticHeight,
    /// `min`..`max` are depths under the local ground surface, in blocks
    /// (0 = the surface, positive = down).
    BelowGround,
    /// Pockets where a 3D noise field is above `noiseThreshold`.
    Noise,
}

/// How tier 1's intensity falls off inside its band.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Shape {
    /// The full intensity across the band.
    #[default]
    Flat,
    /// Full at `peak`, falling linearly to 0 at `min` and `max`.
    Triangle,
    /// A bell curve around `peak` with standard deviation `spread`
    /// (repeating every `repeat` blocks when `repeat > 0`).
    Bell,
}

/// Tier 1: the rich zone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RichZone {
    #[serde(default = "yes")]
    pub enabled: bool,
    pub mode: RichMode,
    /// The band: block Ys (static) or depths (below ground). Unused by noise
    /// mode except as an optional limit. May be infinite (`"-Infinity"`).
    #[serde(with = "crate::cubic::custom::settings::inf", default = "neg_inf")]
    pub min: f64,
    #[serde(with = "crate::cubic::custom::settings::inf", default = "pos_inf")]
    pub max: f64,
    /// Veins per cube at the peak (fractional = a chance of one more).
    pub intensity: f64,
    /// CWG's `spawnProbability`: each try's chance (legacy presets).
    #[serde(default = "one")]
    pub probability: f64,
    pub shape: Shape,
    /// Triangle / bell peak; `None` = the middle of the band.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak: Option<f64>,
    /// Bell standard deviation in blocks; `None` = a sixth of the band.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spread: Option<f64>,
    /// Bell repeat spacing in blocks (CWG periodic Gaussian ores); 0 = once.
    pub repeat: f64,
    /// Noise mode: the noise's feature size in blocks.
    pub noise_scale: f64,
    /// Noise mode: the field (about -1..1) must be above this.
    pub noise_threshold: f64,
    /// Noise mode: added to the world seed so ores get different pockets.
    pub noise_seed: i64,
}

impl Default for RichZone {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: RichMode::StaticHeight,
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            intensity: 1.0,
            probability: 1.0,
            shape: Shape::Flat,
            peak: None,
            spread: None,
            repeat: 0.0,
            noise_scale: 64.0,
            noise_threshold: 0.4,
            noise_seed: 0,
        }
    }
}

/// Where tier 2 applies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RangeMode {
    /// Everywhere outside the rich zone.
    #[default]
    Everywhere,
    /// Only between block Ys `min`..`max`.
    StaticHeight,
    /// Only between depths `min`..`max` under the local ground.
    BelowGround,
}

/// Tier 2: the background rate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Background {
    pub enabled: bool,
    /// Veins per cube.
    pub rate: f64,
    pub range: RangeMode,
    #[serde(with = "crate::cubic::custom::settings::inf", default = "neg_inf")]
    pub min: f64,
    #[serde(with = "crate::cubic::custom::settings::inf", default = "pos_inf")]
    pub max: f64,
}

impl Default for Background {
    fn default() -> Self {
        Self {
            enabled: false,
            rate: 0.0,
            range: RangeMode::Everywhere,
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
        }
    }
}

/// One kind of meteor: its main ore, the minority ore mixed in, and how
/// often it is picked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MeteorKind {
    /// Short ore name: `iron`, `copper`, `coal`, `gold`, `redstone`, `lapis`,
    /// `diamond`, `emerald` (blocks `<name>_ore`, `deepslate_<name>_ore`, and
    /// the core `raw_<name>_block` or `<name>_block`).
    pub ore: String,
    /// The minority ore mixed into the ore layer (same naming); empty = none.
    pub second: String,
    /// Relative weight among the kinds.
    pub weight: f64,
}

impl Default for MeteorKind {
    fn default() -> Self {
        Self {
            ore: "iron".into(),
            second: String::new(),
            weight: 1.0,
        }
    }
}

/// The ore meteors. `Default` is the default for new worlds
/// (`super::meteors`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MeteorSettings {
    pub enabled: bool,
    /// Average meteors per square kilometre (1,000 x 1,000 blocks); 0 = none.
    pub per_square_km: f64,
    /// Diameter range in blocks (5 to 25 by default; 3 to 25 allowed).
    pub min_diameter: i32,
    pub max_diameter: i32,
    /// The size spline: relative chance (v) by diameter (y), piecewise
    /// linear. Every diameter in range keeps at least `size_floor` of the
    /// curve's peak.
    pub size_curve: UserFunction,
    pub size_floor: f64,
    /// Relative weights of resting on the surface, sunk part-way, and buried.
    pub surface_share: f64,
    pub sunk_share: f64,
    pub buried_share: f64,
    /// Deepest a buried meteor's centre goes under the ground, in blocks.
    pub buried_depth: f64,
    /// Share of the ore layer that is the second ore (0.2 = 80 / 20).
    pub mix_share: f64,
    pub kinds: Vec<MeteorKind>,
}

impl OreGenSettings {
    /// Entries for a CWG preset's `standardOres` and `periodicGaussianOres`
    /// (heights normalised with `base` and `variation`, the preset's
    /// `expectedBaseHeight` and `expectedHeightVariation`): tier 1 static
    /// height, flat for standard ores and a repeating bell for periodic
    /// Gaussian ores, background off, in the order the old generator placed
    /// them (all standard ores, then all periodic ones). Meteors off.
    pub fn from_legacy(
        standard: &[StandardOreConfig],
        periodic: &[PeriodicGaussianOreConfig],
        base: f64,
        variation: f64,
    ) -> Self {
        let y = |h: f64| {
            if h.is_finite() {
                (h * variation + base + 0.5).floor()
            } else {
                h
            }
        };
        let mut ores = Vec::with_capacity(standard.len() + periodic.len());
        for o in standard {
            ores.push(OreEntry {
                blockstate: o.blockstate.clone(),
                spawn_size: o.spawn_size,
                biomes: o.biomes.clone(),
                place_block_when: o.place_block_when.clone(),
                generate_when: o.generate_when.clone(),
                biome_multipliers: BTreeMap::new(),
                rich: RichZone {
                    min: y(o.min_height),
                    max: y(o.max_height),
                    intensity: f64::from(o.spawn_tries),
                    probability: o.spawn_probability,
                    shape: Shape::Flat,
                    ..RichZone::default()
                },
                background: Background::default(),
            });
        }
        for o in periodic {
            let spacing = if o.height_spacing == 0.0 {
                0.5
            } else {
                o.height_spacing
            };
            ores.push(OreEntry {
                blockstate: o.blockstate.clone(),
                spawn_size: o.spawn_size,
                biomes: o.biomes.clone(),
                place_block_when: o.place_block_when.clone(),
                generate_when: o.generate_when.clone(),
                biome_multipliers: BTreeMap::new(),
                rich: RichZone {
                    min: y(o.min_height),
                    max: y(o.max_height),
                    intensity: f64::from(o.spawn_tries),
                    probability: o.spawn_probability,
                    shape: Shape::Bell,
                    peak: Some(y(o.height_mean)),
                    spread: Some(o.height_std_deviation * variation),
                    repeat: (spacing * variation + 0.5).floor().max(1.0),
                    ..RichZone::default()
                },
                background: Background::default(),
            });
        }
        let meteors = MeteorSettings {
            enabled: false,
            ..MeteorSettings::default()
        };
        Self {
            ores,
            meteors,
            ..Self::default()
        }
    }
}

impl OreGenSettings {
    /// Checks the numbers the engine cannot work with, naming the setting
    /// that is wrong (`oreGen.ores[3].rich.intensity: must be 0 to 4096`).
    pub fn validate(&self) -> Result<(), String> {
        const MAX_COUNT: f64 = 4096.0;
        fn count(path: &str, v: f64) -> Result<(), String> {
            if v.is_finite() && (0.0..=MAX_COUNT).contains(&v) {
                Ok(())
            } else {
                Err(format!("{path}: must be 0 to {MAX_COUNT}, not {v}"))
            }
        }
        fn band(path: &str, min: f64, max: f64) -> Result<(), String> {
            if min.is_nan() || max.is_nan() || min > max {
                Err(format!(
                    "{path}.min: must not be above {path}.max (min {min}, max {max})"
                ))
            } else {
                Ok(())
            }
        }
        fn positive(path: &str, v: f64) -> Result<(), String> {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(format!("{path}: must be a number above 0, not {v}"))
            }
        }

        if !(0..=ORE_GEN_VERSION).contains(&self.version) {
            return Err(format!(
                "oreGen.version: must be 0 to {ORE_GEN_VERSION}, not {}",
                self.version
            ));
        }
        let s = self.richness_strength;
        if !(s.is_finite() && (0.0..=2.0).contains(&s)) {
            return Err(format!("oreGen.richnessStrength: must be 0 to 2, not {s}"));
        }
        for (i, o) in self.ores.iter().enumerate() {
            let at = format!("oreGen.ores[{i}]");
            if !(1..=64).contains(&o.spawn_size) {
                return Err(format!(
                    "{at}.spawnSize: must be 1 to 64, not {}",
                    o.spawn_size
                ));
            }
            for (biome, m) in &o.biome_multipliers {
                if !(m.is_finite() && *m >= 0.0) {
                    return Err(format!(
                        "{at}.biomeMultipliers[{biome}]: must be 0 or more, not {m}"
                    ));
                }
            }
            let r = &o.rich;
            let rp = format!("{at}.rich");
            count(&format!("{rp}.intensity"), r.intensity)?;
            if !(r.probability.is_finite() && (0.0..=1.0).contains(&r.probability)) {
                return Err(format!(
                    "{rp}.probability: must be 0 to 1, not {}",
                    r.probability
                ));
            }
            band(&rp, r.min, r.max)?;
            if let Some(sp) = r.spread
                && !(sp.is_finite() && sp > 0.0)
            {
                return Err(format!("{rp}.spread: must be a number above 0, not {sp}"));
            }
            if let Some(p) = r.peak
                && p.is_nan()
            {
                return Err(format!("{rp}.peak: must be a number"));
            }
            if !(r.repeat.is_finite() && r.repeat >= 0.0) {
                return Err(format!("{rp}.repeat: must be 0 or more, not {}", r.repeat));
            }
            positive(&format!("{rp}.noiseScale"), r.noise_scale)?;
            if !r.noise_threshold.is_finite() {
                return Err(format!(
                    "{rp}.noiseThreshold: must be a number, not {}",
                    r.noise_threshold
                ));
            }
            let b = &o.background;
            let bp = format!("{at}.background");
            count(&format!("{bp}.rate"), b.rate)?;
            band(&bp, b.min, b.max)?;
        }
        self.meteors.validate().map_err(|e| format!("oreGen.{e}"))
    }
}

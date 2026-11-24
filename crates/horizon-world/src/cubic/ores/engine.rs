//! The two-tier ore engine: places an [`OreGenSettings`]' entries into one
//! populated cube (see [`OreEngine::place`]).
//!
//! # Per entry and cube
//!
//! 1. An entry whose `biomes` list does not contain the cube's biome is
//!    skipped. Otherwise its factor is `f = biomeMultiplier * richness`
//!    (richness only when the caller supplies [`OreCtx::lat_lon`] and the
//!    settings' `richnessMap` is on).
//! 2. Tier 1 (the rich zone) tries `floor(intensity * f)` veins, plus one
//!    more with the chance of the fraction.
//!    * Static height, flat: exactly the legacy `standardOres` draws on the
//!      cube's population random.
//!    * Static height, triangle or bell: exactly the legacy
//!      `periodicGaussianOres` draws on the population random.
//!    * Below ground and noise: the same per-try scheme on the entry's own
//!      random (deterministic per world seed, entry and cube).
//! 3. Tier 2 (the background) tries `rate * f` veins at random positions of
//!    the window that are outside the rich zone and inside the background's
//!    range, on the entry's own (second) random.
//!
//! The vein, window and replace rule are populate.rs's (shared code): a vein
//! is CWG's `CustomVeinGenerator`, positions are `cube origin + 8 + nextInt(16)`
//! and writes are clamped to the 2 x 2 x 2 cubes from the populated cube up.
//!
//! # Noise
//!
//! Noise mode uses one octave of Minecraft's improved Perlin noise
//! ([`horizon_core::noise::PerlinNoise`]) seeded by `worldSeed + noiseSeed`,
//! sampled at `pos / noiseScale`: values about -0.8..0.8, mean 0.

use std::collections::{HashMap, HashSet};

use horizon_core::noise::{PerlinNoise, wrap};
use horizon_core::position::SectionPos;
use horizon_core::random::XoroshiroRandom;

use super::settings::{OreEntry, OreGenSettings, RangeMode, RichMode, Shape};
use crate::cubic::custom::populate::{CubeWorld, VeinEnv, VeinSpec, Window, bell_cyclic, try_vein};
use crate::cubic::custom::rng::{JRandom, coords_seed_random};
use crate::registry::BlockRegistry;

/// `richness(ore short name, latitude, longitude)`.
pub type RichnessFn = fn(&str, f64, f64) -> f64;

/// The ores whose richness the real-data map knows.
const RICH_ORES: [&str; 8] = [
    "coal", "iron", "copper", "gold", "redstone", "diamond", "lapis", "emerald",
];

/// Bounds of the richness factor applied to an entry's intensity.
pub const RICHNESS_MIN: f64 = 0.3;
pub const RICHNESS_MAX: f64 = 5.0;

/// More tries than this per entry and cube are never made.
const MAX_TRIES: f64 = 100_000.0;

const SALT: i64 = 0x6f72_655f_656e_6769; // "ore_engi"
const PRIME: i64 = 0x2545_f491_4f6c_dd1d;
const TIER2_SALT: i64 = 0x1b87_3593_cc9e_2d51;

/// What the engine needs to know about the cube it populates.
pub struct OreCtx<'a> {
    /// The namespaced biome (`minecraft:plains`) at the cube's column
    /// `min + 8`.
    pub biome: &'a str,
    /// The local ground Y of a column; called lazily, only by below-ground
    /// modes.
    pub ground: &'a dyn Fn(i32, i32) -> i32,
    /// Earth only: latitude and longitude of a block column, for the
    /// richness map.
    pub lat_lon: Option<&'a dyn Fn(i32, i32) -> (f64, f64)>,
}

struct Rich {
    mode: RichMode,
    shape: Shape,
    min: f64,
    max: f64,
    intensity: f64,
    probability: f64,
    peak: f64,
    spread: f64,
    repeat: f64,
    noise: Option<Noise>,
    threshold: f64,
}

struct Noise {
    perlin: PerlinNoise,
    inv_scale: f64,
}

impl Noise {
    fn at(&self, p: [i32; 3]) -> f64 {
        f64::from(self.perlin.get3(
            wrap(f64::from(p[0]) * self.inv_scale),
            wrap(f64::from(p[1]) * self.inv_scale),
            wrap(f64::from(p[2]) * self.inv_scale),
        ))
    }
}

struct Background {
    rate: f64,
    range: RangeMode,
    min: f64,
    max: f64,
}

struct Entry {
    vein: VeinSpec,
    biomes: Option<HashSet<String>>,
    multipliers: HashMap<String, f64>,
    /// `Some("iron")` for iron / deepslate iron ore, `None` for other blocks.
    ore: Option<&'static str>,
    /// `Some` when tier 1 can place.
    rich: Option<Rich>,
    background: Option<Background>,
    seed: i64,
}

/// The compiled ore entries of one world.
pub struct OreEngine {
    env: VeinEnv,
    enabled: bool,
    entries: Vec<Entry>,
    richness: RichnessFn,
    richness_map: bool,
    richness_strength: f64,
}

impl OreEngine {
    pub fn new(world_seed: i64, s: &OreGenSettings, reg: &BlockRegistry) -> Result<Self, String> {
        let mut entries = Vec::with_capacity(s.ores.len());
        for (i, o) in s.ores.iter().enumerate() {
            entries.push(
                Entry::compile(world_seed, i, o, reg)
                    .map_err(|e| format!("oreGen.ores[{i}]: {e}"))?,
            );
        }
        Ok(Self {
            env: VeinEnv::new(world_seed, reg)?,
            enabled: s.enabled,
            entries,
            richness: horizon_earth::ore_richness::richness,
            richness_map: s.richness_map,
            richness_strength: s.richness_strength,
        })
    }

    /// Replaces the richness function (tests; Earth's default is the real-data map).
    pub fn with_richness(mut self, f: RichnessFn) -> Self {
        self.richness = f;
        self
    }

    /// Whether any entry can place something: the world switch is on and some
    /// entry has a tier with veins to try.
    pub fn is_active(&self) -> bool {
        self.enabled
            && self
                .entries
                .iter()
                .any(|e| e.rich.is_some() || e.background.is_some())
    }

    #[cfg(test)]
    /// The noise value of entry `index`'s noise zone at `p` (`None` when the
    /// entry has no noise zone). Tests use it to check where pockets are.
    pub(crate) fn noise_value(&self, index: usize, p: [i32; 3]) -> Option<f64> {
        let z = self.entries.get(index)?.rich.as_ref()?;
        z.noise.as_ref().map(|n| n.at(p))
    }

    /// The richness factor of `v` (the data's value) at strength `strength`.
    pub fn richness_factor(v: f64, strength: f64) -> f64 {
        let f = v.powf(strength);
        if f.is_finite() {
            f.clamp(RICHNESS_MIN, RICHNESS_MAX)
        } else {
            1.0
        }
    }

    /// Places every entry's veins for `cube`. `rand` is the cube's population
    /// random (legacy order: flat and periodic static-height entries draw on
    /// it exactly as the legacy ore lists did); everything new uses
    /// randoms of its own.
    pub fn place(
        &self,
        cube: SectionPos,
        world: &mut dyn CubeWorld,
        rand: &mut JRandom,
        ctx: &OreCtx,
    ) {
        if !self.enabled {
            return;
        }
        let win = Window {
            min: [cube.x * 16, cube.y * 16, cube.z * 16],
        };
        for e in &self.entries {
            if e.rich.is_none() && e.background.is_none() {
                continue;
            }
            if e.biomes.as_ref().is_some_and(|b| !b.contains(ctx.biome)) {
                continue;
            }
            let f = e.multipliers.get(ctx.biome).copied().unwrap_or(1.0)
                * self.richness_of(e, &win, ctx);
            if !(f > 0.0 && f.is_finite()) {
                continue;
            }
            if let Some(z) = &e.rich {
                self.tier1(e, z, f, cube, world, rand, ctx, &win);
            }
            if let Some(b) = &e.background {
                self.tier2(e, b, f, cube, world, ctx, &win);
            }
        }
    }

    /// 1, or the richness factor of this entry's ore at the window centre.
    fn richness_of(&self, e: &Entry, win: &Window, ctx: &OreCtx) -> f64 {
        let (Some(ore), Some(lat_lon), true) = (e.ore, ctx.lat_lon, self.richness_map) else {
            return 1.0;
        };
        let (lat, lon) = lat_lon(win.min[0] + 16, win.min[2] + 16);
        Self::richness_factor((self.richness)(ore, lat, lon), self.richness_strength)
    }

    #[allow(clippy::too_many_arguments)]
    fn tier1(
        &self,
        e: &Entry,
        z: &Rich,
        f: f64,
        cube: SectionPos,
        world: &mut dyn CubeWorld,
        rand: &mut JRandom,
        ctx: &OreCtx,
        win: &Window,
    ) {
        let n = z.intensity * f;
        let mut own = OwnRandom::new(e.seed, cube);
        match z.mode {
            RichMode::StaticHeight if z.shape == Shape::Flat => {
                // populate.rs `ore()`, the standard-ore branch.
                if f64::from(win.min[1] + 8) > z.max || f64::from(win.min[1] + 8 + 15) < z.min {
                    return;
                }
                for _ in 0..try_count(n, &mut own) {
                    if rand.double() > z.probability {
                        continue;
                    }
                    let y = win.min[1] + rand.int(16) + 8;
                    if f64::from(y) > z.max || f64::from(y) < z.min {
                        continue;
                    }
                    let x = win.min[0] + rand.int(16) + 8;
                    let zz = win.min[2] + rand.int(16) + 8;
                    try_vein(&self.env, &e.vein, rand, world, win, [x, y, zz]);
                }
            }
            RichMode::StaticHeight => {
                // populate.rs `ore()`, the periodic branch.
                for _ in 0..try_count(n, &mut own) {
                    let y = win.min[1] + rand.int(16) + 8;
                    if f64::from(y) > z.max || f64::from(y) < z.min {
                        continue;
                    }
                    let m = z.shape_at(f64::from(y));
                    if rand.double() > z.probability * m {
                        continue;
                    }
                    let x = win.min[0] + rand.int(16) + 8;
                    let zz = win.min[2] + rand.int(16) + 8;
                    try_vein(&self.env, &e.vein, rand, world, win, [x, y, zz]);
                }
            }
            RichMode::BelowGround | RichMode::Noise => {
                for _ in 0..try_count(n, &mut own) {
                    let r = own.get();
                    let y = win.min[1] + r.int(16) + 8;
                    let x = win.min[0] + r.int(16) + 8;
                    let zz = win.min[2] + r.int(16) + 8;
                    let p = [x, y, zz];
                    let m = z.weight_at(p, ctx.ground);
                    if m <= 0.0 || r.double() > z.probability * m {
                        continue;
                    }
                    try_vein(&self.env, &e.vein, r, world, win, p);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tier2(
        &self,
        e: &Entry,
        b: &Background,
        f: f64,
        cube: SectionPos,
        world: &mut dyn CubeWorld,
        ctx: &OreCtx,
        win: &Window,
    ) {
        let mut own = OwnRandom::new(e.seed ^ TIER2_SALT, cube);
        for _ in 0..try_count(b.rate * f, &mut own) {
            let r = own.get();
            let y = win.min[1] + r.int(16) + 8;
            let x = win.min[0] + r.int(16) + 8;
            let zz = win.min[2] + r.int(16) + 8;
            let p = [x, y, zz];
            if e.rich.as_ref().is_some_and(|z| z.inside(p, ctx.ground)) {
                continue;
            }
            let inside_range = match b.range {
                RangeMode::Everywhere => true,
                RangeMode::StaticHeight => (b.min..=b.max).contains(&f64::from(y)),
                RangeMode::BelowGround => {
                    (b.min..=b.max).contains(&(f64::from((ctx.ground)(x, zz)) - f64::from(y)))
                }
            };
            if inside_range {
                try_vein(&self.env, &e.vein, r, world, win, p);
            }
        }
    }
}

impl Rich {
    /// Whether tier 1 is rich at `p`: the band (a depth band under the local
    /// ground, or Y) for static and below-ground modes, the noise above its
    /// threshold (within the band) for noise mode.
    fn inside(&self, p: [i32; 3], ground: &dyn Fn(i32, i32) -> i32) -> bool {
        match self.mode {
            RichMode::StaticHeight => (self.min..=self.max).contains(&f64::from(p[1])),
            RichMode::BelowGround => {
                let depth = f64::from(ground(p[0], p[2])) - f64::from(p[1]);
                (self.min..=self.max).contains(&depth)
            }
            RichMode::Noise => self.noise_inside(p),
        }
    }

    fn noise_inside(&self, p: [i32; 3]) -> bool {
        (self.min..=self.max).contains(&f64::from(p[1]))
            && self
                .noise
                .as_ref()
                .is_some_and(|n| n.at(p) > self.threshold)
    }

    /// The tier-1 weight `m` of a try at `p` for the modes on the entry's own
    /// random: 0 outside the zone.
    fn weight_at(&self, p: [i32; 3], ground: &dyn Fn(i32, i32) -> i32) -> f64 {
        match self.mode {
            RichMode::StaticHeight => self.shape_at(f64::from(p[1])),
            RichMode::BelowGround => {
                let depth = f64::from(ground(p[0], p[2])) - f64::from(p[1]);
                if depth < self.min || depth > self.max {
                    0.0
                } else {
                    self.shape_at(depth)
                }
            }
            RichMode::Noise => {
                if self.noise_inside(p) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// The shape of the band at `v` (a Y or a depth), 0..=1; `v` must be
    /// inside the band.
    fn shape_at(&self, v: f64) -> f64 {
        match self.shape {
            Shape::Flat => 1.0,
            Shape::Triangle => {
                let rising = if v <= self.peak {
                    (v - self.min) / (self.peak - self.min)
                } else {
                    (self.max - v) / (self.max - self.peak)
                };
                // An infinite end of the band makes the slope flat there.
                if rising.is_nan() {
                    1.0
                } else {
                    rising.clamp(0.0, 1.0)
                }
            }
            Shape::Bell => {
                if self.repeat > 0.0 {
                    bell_cyclic(v, self.peak, self.spread, self.repeat)
                } else {
                    let d = v - self.peak;
                    (-(d * d) / (2.0 * self.spread * self.spread)).exp()
                }
            }
        }
    }
}

impl Entry {
    fn compile(
        world_seed: i64,
        index: usize,
        o: &OreEntry,
        reg: &BlockRegistry,
    ) -> Result<Self, String> {
        let vein = VeinSpec::compile(
            reg,
            &o.blockstate,
            &o.place_block_when,
            &o.generate_when,
            o.spawn_size,
        )?;
        let r = &o.rich;
        let can_rich = r.enabled && r.intensity > 0.0;
        let noise = (can_rich && r.mode == RichMode::Noise).then(|| Noise {
            perlin: PerlinNoise::new(&mut XoroshiroRandom::new(
                world_seed.wrapping_add(r.noise_seed),
            )),
            inv_scale: 1.0 / r.noise_scale,
        });
        let (peak, spread) = band_defaults(r.min, r.max, r.peak, r.spread);
        let rich = can_rich.then_some(Rich {
            mode: r.mode,
            shape: r.shape,
            min: r.min,
            max: r.max,
            intensity: r.intensity,
            probability: r.probability,
            peak,
            spread,
            repeat: r.repeat,
            noise,
            threshold: r.noise_threshold,
        });
        let b = &o.background;
        let background = (b.enabled && b.rate > 0.0).then_some(Background {
            rate: b.rate,
            range: b.range,
            min: b.min,
            max: b.max,
        });
        Ok(Self {
            vein,
            biomes: o
                .biomes
                .as_ref()
                .map(|v| v.iter().map(|b| b.0.clone()).collect()),
            multipliers: o
                .biome_multipliers
                .iter()
                .map(|(k, v)| {
                    let key = if k.contains(':') {
                        k.clone()
                    } else {
                        format!("minecraft:{k}")
                    };
                    (key, if v.is_nan() { 1.0 } else { v.max(0.0) })
                })
                .collect(),
            ore: ore_short(&o.blockstate.name),
            rich,
            background,
            seed: world_seed ^ SALT ^ (index as i64).wrapping_mul(PRIME),
        })
    }
}

/// `iron` for `minecraft:iron_ore` and `minecraft:deepslate_iron_ore`; `None`
/// for any block that is not one of the eight ores.
fn ore_short(name: &str) -> Option<&'static str> {
    let short = name.strip_prefix("minecraft:").unwrap_or(name);
    let short = short.strip_prefix("deepslate_").unwrap_or(short);
    let short = short.strip_suffix("_ore")?;
    RICH_ORES.iter().copied().find(|o| *o == short)
}

/// The peak (default: the middle of the band) and spread (default: a sixth of
/// the band) of a shape, with fallbacks for an unbounded band.
fn band_defaults(min: f64, max: f64, peak: Option<f64>, spread: Option<f64>) -> (f64, f64) {
    let both = min.is_finite() && max.is_finite();
    let mid = if both {
        (min + max) / 2.0
    } else if min.is_finite() {
        min
    } else if max.is_finite() {
        max
    } else {
        0.0
    };
    let sixth = if both && max > min {
        (max - min) / 6.0
    } else {
        32.0
    };
    (
        peak.filter(|p| !p.is_nan()).unwrap_or(mid),
        spread.filter(|s| *s > 0.0).unwrap_or(sixth),
    )
}

/// `floor(n)` tries, plus one with the chance of the fraction (a draw from
/// `own`, only when there is a fraction).
fn try_count(n: f64, own: &mut OwnRandom) -> i32 {
    if n.is_nan() || n <= 0.0 {
        return 0;
    }
    let n = n.min(MAX_TRIES);
    let whole = n.floor();
    let frac = n - whole;
    let mut tries = whole as i32;
    if frac > 0.0 && own.get().double() < frac {
        tries += 1;
    }
    tries
}

/// An entry's own random for one cube, created on first use.
struct OwnRandom {
    seed: i64,
    cube: SectionPos,
    rand: Option<JRandom>,
}

impl OwnRandom {
    fn new(seed: i64, cube: SectionPos) -> Self {
        Self {
            seed,
            cube,
            rand: None,
        }
    }

    fn get(&mut self) -> &mut JRandom {
        let (seed, c) = (self.seed, self.cube);
        self.rand
            .get_or_insert_with(|| coords_seed_random(seed, c.x, c.y, c.z))
    }
}

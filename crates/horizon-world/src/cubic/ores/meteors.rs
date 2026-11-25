//! Ore meteors: balls of ore with a deepslate shell and a raw-ore-block core,
//! placed rarely while a world is populated. See docs/cubic-meteors.md.
//!
//! The world is cut into regions of 32 x 32 block columns. Each region holds
//! at most one meteor, and every block that meteor writes (body, crater, rim,
//! cleared vegetation) lies inside the region's columns. The meteor is written
//! whole by exactly one cube's population (the owner: the cube with even X and
//! Z over the region whose population window is centred nearest the meteor),
//! so the result does not depend on the order cubes are populated in.

use horizon_core::position::SectionPos;

use super::settings::{MeteorKind, MeteorSettings};
use crate::cubic::custom::populate::CubeWorld;
use crate::cubic::custom::rng::{hash_pos, hash_unit};
use crate::cubic::custom::settings::UserFunction;
use crate::registry::{BlockRegistry, BlockStateId};

const SALT_PRESENCE: i32 = 0x4D45_5401;
const SALT_PLAN: i32 = 0x4D45_5402;
const SALT_FLECK: i32 = 0x4D45_5403;
const SALT_MIX: i32 = 0x4D45_5404;

/// Side of a region in blocks (two cubes).
const REGION: i32 = 32;
/// Amplitude of the shape noise: the radius varies by +-10 %.
const NOISE_AMP: f64 = 0.10;
/// Chance that a shell block is a deepslate ore fleck.
const FLECK: f64 = 0.15;
/// Largest footprint radius that still fits a region.
const MAX_FOOTPRINT: f64 = 15.5;
/// Smallest diameter that gets a crater.
const CRATER_MIN_DIAMETER: i32 = 9;

// Block classes, by registry state id.
const GROUND: u8 = 0;
const AIR: u8 = 1;
const LIQUID: u8 = 2;
const PLANT: u8 = 3;
const WATER_PLANT: u8 = 4;
const BEDROCK: u8 = 5;

impl Default for MeteorSettings {
    fn default() -> Self {
        let kind = |ore: &str, second: &str, weight: f64| MeteorKind {
            ore: ore.into(),
            second: second.into(),
            weight,
        };
        Self {
            enabled: true,
            per_square_km: 5.0,
            min_diameter: 5,
            max_diameter: 25,
            size_curve: UserFunction::new(&[
                (5.0, 0.6),
                (7.0, 1.0),
                (10.0, 0.5),
                (16.0, 0.12),
                (25.0, 0.02),
            ]),
            size_floor: 0.02,
            surface_share: 0.6,
            sunk_share: 0.25,
            buried_share: 0.15,
            buried_depth: 50.0,
            mix_share: 0.2,
            kinds: vec![
                kind("iron", "copper", 30.0),
                kind("copper", "iron", 25.0),
                kind("coal", "iron", 20.0),
                kind("gold", "redstone", 8.0),
                kind("redstone", "lapis", 7.0),
                kind("lapis", "redstone", 6.0),
                kind("diamond", "lapis", 2.0),
                kind("emerald", "diamond", 2.0),
            ],
        }
    }
}

impl MeteorSettings {
    /// Checks the numbers, naming the setting that is wrong.
    pub fn validate(&self) -> Result<(), String> {
        for (name, d) in [
            ("minDiameter", self.min_diameter),
            ("maxDiameter", self.max_diameter),
        ] {
            if !(3..=25).contains(&d) {
                return Err(format!("meteors.{name} must be 3 to 25, got {d}"));
            }
        }
        if self.min_diameter > self.max_diameter {
            return Err(format!(
                "meteors.minDiameter ({}) is above meteors.maxDiameter ({})",
                self.min_diameter, self.max_diameter
            ));
        }
        if !self.per_square_km.is_finite() || !(0.0..=1000.0).contains(&self.per_square_km) {
            return Err(format!(
                "meteors.perSquareKm must be 0 to 1000, got {}",
                self.per_square_km
            ));
        }
        if !(0.0..=1.0).contains(&self.size_floor) {
            return Err(format!(
                "meteors.sizeFloor must be 0 to 1, got {}",
                self.size_floor
            ));
        }
        if self
            .size_curve
            .0
            .iter()
            .any(|p| !p.y.is_finite() || !p.v.is_finite())
        {
            return Err("meteors.sizeCurve has a point that is not a finite number".into());
        }
        for (name, v) in [
            ("surfaceShare", self.surface_share),
            ("sunkShare", self.sunk_share),
            ("buriedShare", self.buried_share),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(format!("meteors.{name} must be 0 or more, got {v}"));
            }
        }
        if self.surface_share + self.sunk_share + self.buried_share <= 0.0 {
            return Err("meteors.surfaceShare, sunkShare and buriedShare are all 0".into());
        }
        if !self.buried_depth.is_finite() || !(0.0..=1000.0).contains(&self.buried_depth) {
            return Err(format!(
                "meteors.buriedDepth must be 0 to 1000, got {}",
                self.buried_depth
            ));
        }
        if !(0.0..=1.0).contains(&self.mix_share) {
            return Err(format!(
                "meteors.mixShare must be 0 to 1, got {}",
                self.mix_share
            ));
        }
        for (i, k) in self.kinds.iter().enumerate() {
            if !k.weight.is_finite() || k.weight < 0.0 {
                return Err(format!(
                    "meteors.kinds[{i}].weight must be 0 or more, got {}",
                    k.weight
                ));
            }
            if k.ore.trim().is_empty() {
                return Err(format!("meteors.kinds[{i}].ore is empty"));
            }
        }
        Ok(())
    }
}

/// Where a meteor sits relative to the ground.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Placement {
    /// Resting on the ground, partly sunk, with a crater (if big enough).
    Surface,
    /// Sunk part-way, its top poking out a little or flush.
    Sunk,
    /// Fully underground.
    Buried,
}

/// A kind with its blocks resolved to state ids.
struct Kind {
    weight: f64,
    /// `<ore>_ore`.
    ore: u32,
    /// `deepslate_<ore>_ore`.
    ore_deepslate: u32,
    /// `<second>_ore`; `None` = no second ore.
    second: Option<u32>,
    /// `raw_<ore>_block`, else `<ore>_block`.
    core: u32,
}

/// A meteor [`Meteors::find`] found: its centre, diameter, kind (index into
/// the settings' kinds) and placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeteorSpot {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub diameter: i32,
    pub kind: usize,
    pub placement: &'static str,
}

/// The compiled meteors of one world.
pub struct Meteors {
    seed: i64,
    active: bool,
    /// Chance that a region holds a meteor.
    presence: f64,
    kinds: Vec<Kind>,
    kind_total: f64,
    min_diameter: i32,
    /// Cumulative diameter weights, from `min_diameter`.
    size_cdf: Vec<f64>,
    /// Surface, sunk and buried weights.
    shares: [f64; 3],
    buried_depth: f64,
    mix_share: f64,
    /// Block class by state id.
    classes: Vec<u8>,
    air: u32,
    water: u32,
    deepslate: u32,
}

/// A small splitmix64 random, seeded per meteor.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A number in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// Smooth noise on the sphere: the mean of three sines of a direction's
/// projection on random axes. In `[-1, 1]`.
#[derive(Clone, Debug)]
struct Noise {
    axes: [[f64; 3]; 3],
    freq: [f64; 3],
    phase: [f64; 3],
}

impl Noise {
    fn new(rng: &mut Rng) -> Self {
        let mut n = Self {
            axes: [[0.0; 3]; 3],
            freq: [0.0; 3],
            phase: [0.0; 3],
        };
        for k in 0..3 {
            let z = 2.0 * rng.unit() - 1.0;
            let a = std::f64::consts::TAU * rng.unit();
            let s = (1.0 - z * z).max(0.0).sqrt();
            n.axes[k] = [s * libm::cos(a), s * libm::sin(a), z];
            n.freq[k] = 2.0 + rng.unit();
            n.phase[k] = std::f64::consts::TAU * rng.unit();
        }
        n
    }

    fn at(&self, d: [f64; 3]) -> f64 {
        let mut sum = 0.0;
        for k in 0..3 {
            let w = self.axes[k];
            let dot = d[0] * w[0] + d[1] * w[1] + d[2] * w[2];
            sum += libm::sin(self.freq[k] * dot + self.phase[k]);
        }
        sum / 3.0
    }
}

/// The plan of one meteor: a pure function of the seed, the region and the
/// ground estimate.
#[derive(Clone, Debug)]
pub(crate) struct MeteorPlan {
    /// Index into the settings' kinds.
    pub kind: usize,
    pub diameter: i32,
    pub placement: Placement,
    /// Centre block X and Z.
    pub x: i32,
    pub z: i32,
    /// The ground estimate the plan was made with.
    pub g_est: i32,
    /// Centre block Y from the ground estimate (before the window clamp).
    pub y_est: i32,
    /// The cube Y whose population places the meteor.
    pub owner_cy: i32,
    /// Largest horizontal radius of anything the meteor writes.
    pub footprint: f64,
    /// Radius of the crater's outer edge (0 = no crater).
    pub crater_radius: f64,
    u: f64,
    buried_depth: f64,
    noise: Noise,
}

impl MeteorPlan {
    /// The radius `d / 2`.
    pub(crate) fn radius(&self) -> f64 {
        f64::from(self.diameter) / 2.0
    }

    /// The worst-case extent of the body, `R * (1 + a)`.
    pub(crate) fn r_out(&self) -> f64 {
        self.radius() * (1.0 + NOISE_AMP)
    }

    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t
    }

    /// Centre block Y for the ground level `g` (Y of the top solid block).
    fn raw_centre_y(&self, g: i32) -> i32 {
        let r = self.radius();
        let g = f64::from(g);
        let yc = match self.placement {
            Placement::Surface => g + Self::lerp(-0.1, 0.4, self.u) * r,
            Placement::Sunk => g - Self::lerp(0.4, 0.9, self.u) * r,
            Placement::Buried => {
                let lo = self.r_out() + 3.5;
                g - Self::lerp(lo, lo.max(self.buried_depth), self.u)
            }
        };
        yc.round() as i32
    }

    /// Centre block Y for the actual ground level `g`, clamped so the body
    /// fits inside the owner cube's population window.
    pub(crate) fn centre_y(&self, g: i32) -> i32 {
        let base = f64::from(self.owner_cy * 16);
        let lo = (base + self.r_out() - 0.5).ceil() as i32;
        let hi = (base + 32.0 - self.r_out() - 0.5).floor() as i32;
        self.raw_centre_y(g).clamp(lo, hi.max(lo))
    }
}

/// Whether a block name (without namespace) is vegetation that clearing
/// removes. A name list, not exhaustive.
fn is_plant(short: &str) -> bool {
    const SUFFIXES: &[&str] = &[
        "_log",
        "_wood",
        "_leaves",
        "_stem",
        "_hyphae",
        "_sapling",
        "_mushroom",
        "_mushroom_block",
        "_bush",
        "_tulip",
        "_petals",
        "_roots",
    ];
    const EXACT: &[&str] = &[
        "bush",
        "moss_carpet",
        "pale_moss_carpet",
        "short_grass",
        "tall_grass",
        "fern",
        "large_fern",
        "short_dry_grass",
        "tall_dry_grass",
        "dandelion",
        "poppy",
        "blue_orchid",
        "allium",
        "azure_bluet",
        "oxeye_daisy",
        "cornflower",
        "lily_of_the_valley",
        "wither_rose",
        "torchflower",
        "sunflower",
        "lilac",
        "peony",
        "pitcher_plant",
        "wildflowers",
        "leaf_litter",
        "cactus",
        "cactus_flower",
        "sugar_cane",
        "bamboo",
        "vine",
        "glow_lichen",
        "snow",
        "lily_pad",
        "cocoa",
        "bee_nest",
        "azalea",
        "flowering_azalea",
        "pale_hanging_moss",
        "spore_blossom",
        "big_dripleaf",
        "big_dripleaf_stem",
        "small_dripleaf",
        "mangrove_propagule",
    ];
    SUFFIXES.iter().any(|s| short.ends_with(s)) || EXACT.contains(&short)
}

fn classify(name: &str) -> u8 {
    let short = name.rsplit(':').next().unwrap_or(name);
    match short {
        "air" | "cave_air" | "void_air" => AIR,
        "water" | "lava" | "bubble_column" => LIQUID,
        "bedrock" => BEDROCK,
        "seagrass" | "tall_seagrass" | "kelp" | "kelp_plant" | "sea_pickle" => WATER_PLANT,
        s if is_plant(s) => PLANT,
        _ => GROUND,
    }
}

fn resolve(reg: &BlockRegistry, name: &str) -> Result<u32, String> {
    reg.default_state(name)
        .map(|id| id.0)
        .ok_or_else(|| format!("meteors need the block {name}, which the registry does not have"))
}

fn resolve_kind(reg: &BlockRegistry, k: &MeteorKind) -> Result<Kind, String> {
    let ore = k.ore.trim();
    let second = k.second.trim();
    let raw = format!("minecraft:raw_{ore}_block");
    let core = match reg.default_state(&raw) {
        Some(id) => id.0,
        None => resolve(reg, &format!("minecraft:{ore}_block")).map_err(|_| {
            format!(
                "meteors need the block {raw} (or minecraft:{ore}_block), \
                 which the registry does not have"
            )
        })?,
    };
    Ok(Kind {
        weight: k.weight,
        ore: resolve(reg, &format!("minecraft:{ore}_ore"))?,
        ore_deepslate: resolve(reg, &format!("minecraft:deepslate_{ore}_ore"))?,
        second: if second.is_empty() {
            None
        } else {
            Some(resolve(reg, &format!("minecraft:{second}_ore"))?)
        },
        core,
    })
}

impl Meteors {
    fn inactive(seed: i64) -> Self {
        Self {
            seed,
            active: false,
            presence: 0.0,
            kinds: Vec::new(),
            kind_total: 0.0,
            min_diameter: 0,
            size_cdf: Vec::new(),
            shares: [1.0, 0.0, 0.0],
            buried_depth: 0.0,
            mix_share: 0.0,
            classes: Vec::new(),
            air: 0,
            water: 0,
            deepslate: 0,
        }
    }

    pub fn new(world_seed: i64, s: &MeteorSettings, reg: &BlockRegistry) -> Result<Self, String> {
        if !s.enabled {
            return Ok(Self::inactive(world_seed));
        }
        s.validate()?;
        // A kind whose blocks the registry lacks (a trimmed or modded
        // registry) is skipped rather than failing the whole world.
        let kinds: Vec<Kind> = s
            .kinds
            .iter()
            .filter_map(|k| resolve_kind(reg, k).ok())
            .collect();
        let kind_total: f64 = kinds.iter().map(|k| k.weight).sum();
        let air = resolve(reg, "minecraft:air")?;
        let deepslate = resolve(reg, "minecraft:deepslate")?;
        let water = reg.default_state("minecraft:water").map_or(air, |id| id.0);
        let classes = (0..reg.len())
            .map(|i| {
                reg.state(BlockStateId(i as u32))
                    .map_or(GROUND, |st| classify(st.name.as_str()))
            })
            .collect();

        // Diameter weights: the curve, but never below `size_floor` of its peak.
        let curve = |d: i32| s.size_curve.get(f64::from(d)).max(0.0);
        let peak = (s.min_diameter..=s.max_diameter)
            .map(curve)
            .fold(0.0, f64::max);
        let mut size_cdf = Vec::new();
        let mut acc = 0.0;
        for d in s.min_diameter..=s.max_diameter {
            acc += if peak > 0.0 {
                curve(d).max(s.size_floor * peak)
            } else {
                1.0
            };
            size_cdf.push(acc);
        }

        Ok(Self {
            seed: world_seed,
            active: s.per_square_km > 0.0 && kind_total > 0.0,
            presence: (s.per_square_km * f64::from(REGION * REGION) / 1_000_000.0).min(1.0),
            kinds,
            kind_total,
            min_diameter: s.min_diameter,
            size_cdf,
            shares: [s.surface_share, s.sunk_share, s.buried_share],
            buried_depth: s.buried_depth,
            mix_share: s.mix_share,
            classes,
            air,
            water,
            deepslate,
        })
    }

    /// Whether any meteor can generate (enabled, rate above 0, some kind).
    pub fn is_active(&self) -> bool {
        self.active
    }

    fn class_of(&self, state: u32) -> u8 {
        self.classes.get(state as usize).copied().unwrap_or(GROUND)
    }

    /// Whether the region holds a meteor (one hash).
    fn present(&self, rx: i32, rz: i32) -> bool {
        hash_unit(self.seed, SALT_PRESENCE, rx, 0, rz) < self.presence
    }

    /// The plan of the meteor of region `(rx, rz)`, if it has one. Calls
    /// `ground` once, at the meteor's centre column.
    pub(crate) fn plan(
        &self,
        rx: i32,
        rz: i32,
        ground: &dyn Fn(i32, i32) -> i32,
    ) -> Option<MeteorPlan> {
        if !self.active || !self.present(rx, rz) {
            return None;
        }
        let mut rng = Rng(hash_pos(self.seed, SALT_PLAN, rx, 0, rz));

        // 1. kind
        let mut pick = rng.unit() * self.kind_total;
        let mut kind = 0;
        for (i, k) in self.kinds.iter().enumerate() {
            if k.weight > 0.0 {
                kind = i;
                if pick < k.weight {
                    break;
                }
                pick -= k.weight;
            }
        }
        // 2. diameter
        let total = self.size_cdf.last().copied().unwrap_or(0.0);
        let pick = rng.unit() * total;
        let at = self.size_cdf.partition_point(|&c| c <= pick);
        let diameter = self.min_diameter + at.min(self.size_cdf.len() - 1) as i32;
        // 3. placement
        let share_total: f64 = self.shares.iter().sum();
        let pick = rng.unit() * share_total;
        let placement = if pick < self.shares[0] {
            Placement::Surface
        } else if pick < self.shares[0] + self.shares[1] {
            Placement::Sunk
        } else {
            Placement::Buried
        };
        // 4. fraction, jitter, shape noise
        let u = rng.unit();
        let (jx, jz) = (rng.unit(), rng.unit());
        let noise = Noise::new(&mut rng);

        let r = f64::from(diameter) / 2.0;
        let r_out = r * (1.0 + NOISE_AMP);
        let crater = placement != Placement::Buried && diameter >= CRATER_MIN_DIAMETER;
        let crater_radius = if crater {
            (r * 1.7 + 1.0).min(MAX_FOOTPRINT)
        } else {
            0.0
        };
        let footprint = (r_out + 0.5).max(crater_radius).min(MAX_FOOTPRINT);

        // The centre is a block centre with the whole footprint in the region.
        let centre = |region: i32, j: f64| {
            let lo = f64::from(region * REGION) + footprint;
            let hi = f64::from(region * REGION + REGION) - footprint;
            let first = (lo - 0.5).ceil() as i32;
            let last = ((hi - 0.5).floor() as i32).max(first);
            let n = last - first + 1;
            first + ((j * f64::from(n)) as i32).min(n - 1)
        };
        let (x, z) = (centre(rx, jx), centre(rz, jz));

        // 5. ground estimate, 6. centre Y, 7. owner cube
        let g_est = ground(x, z);
        let mut plan = MeteorPlan {
            kind,
            diameter,
            placement,
            x,
            z,
            g_est,
            y_est: 0,
            owner_cy: 0,
            footprint,
            crater_radius,
            u,
            buried_depth: self.buried_depth,
            noise,
        };
        plan.y_est = plan.raw_centre_y(g_est);
        plan.owner_cy = ((f64::from(plan.y_est) + 0.5 - 16.0) / 16.0 + 0.5).floor() as i32;
        Some(plan)
    }

    /// Every meteor whose region overlaps the block box `x0..=x1`, `z0..=z1`
    /// (for tools that look for one, e.g. to take its picture). `ground` as
    /// in [`Self::place`]; the Y is the centre from that estimate.
    pub fn find(
        &self,
        x0: i32,
        z0: i32,
        x1: i32,
        z1: i32,
        ground: &dyn Fn(i32, i32) -> i32,
    ) -> Vec<MeteorSpot> {
        let mut out = Vec::new();
        for rx in x0.div_euclid(REGION)..=x1.div_euclid(REGION) {
            for rz in z0.div_euclid(REGION)..=z1.div_euclid(REGION) {
                if let Some(p) = self.plan(rx, rz, ground) {
                    out.push(MeteorSpot {
                        x: p.x,
                        y: p.y_est,
                        z: p.z,
                        diameter: p.diameter,
                        kind: p.kind,
                        placement: match p.placement {
                            Placement::Surface => "surface",
                            Placement::Sunk => "sunk",
                            Placement::Buried => "buried",
                        },
                    });
                }
            }
        }
        out
    }

    /// Places, whole, every meteor `cube` owns. All writes stay inside the
    /// 2 x 2 x 2 cubes from `cube` up (the population window). `ground(x, z)`
    /// is the generator's estimate of the Y of the top solid block of a
    /// column (the real surface on Earth). Uses its own seeded random, never
    /// the cube's population random. Call after the surface decoration.
    pub fn place(
        &self,
        cube: SectionPos,
        world: &mut dyn CubeWorld,
        ground: &dyn Fn(i32, i32) -> i32,
    ) {
        if !self.active || cube.x & 1 != 0 || cube.z & 1 != 0 {
            return;
        }
        let Some(plan) = self.plan(cube.x >> 1, cube.z >> 1, ground) else {
            return;
        };
        if plan.owner_cy != cube.y {
            return;
        }
        let mut b = Builder {
            m: self,
            world,
            min: [cube.x * 16, cube.y * 16, cube.z * 16],
            max: [cube.x * 16 + 31, cube.y * 16 + 31, cube.z * 16 + 31],
        };
        b.build(&plan);
    }
}

/// Writes one meteor into a cube's population window.
struct Builder<'a> {
    m: &'a Meteors,
    world: &'a mut dyn CubeWorld,
    /// The window, inclusive.
    min: [i32; 3],
    max: [i32; 3],
}

impl Builder<'_> {
    fn inside(&self, p: [i32; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] <= self.max[i])
    }

    fn class(&self, p: [i32; 3]) -> u8 {
        self.m.class_of(self.world.block(p))
    }

    /// The only write: skips positions outside the window and bedrock.
    fn set(&mut self, p: [i32; 3], state: u32) {
        if !self.inside(p) {
            debug_assert!(false, "meteor write outside the population window: {p:?}");
            return;
        }
        if self.class(p) == BEDROCK {
            return;
        }
        self.world.set_block(p, state);
    }

    /// Y of the top ground block of a column in the window, or `None` if the
    /// window's top block is already ground or the column has no ground.
    fn top_ground(&self, x: i32, z: i32) -> Option<i32> {
        let solid = |c: u8| c == GROUND || c == BEDROCK;
        if solid(self.class([x, self.max[1], z])) {
            return None;
        }
        (self.min[1]..self.max[1])
            .rev()
            .find(|&y| solid(self.class([x, y, z])))
    }

    fn build(&mut self, plan: &MeteorPlan) {
        let r = plan.radius();
        let off = ((r / 2.0).round() as i32).max(1);
        let (x, z) = (plan.x, plan.z);
        let mut samples: Vec<i32> = [
            (x, z),
            (x + off, z),
            (x - off, z),
            (x, z + off),
            (x, z - off),
        ]
        .into_iter()
        .filter_map(|(sx, sz)| self.top_ground(sx, sz))
        .collect();
        samples.sort_unstable();
        let g_act = if samples.is_empty() {
            plan.g_est
        } else {
            samples[(samples.len() - 1) / 2]
        };
        let centre = [x, plan.centre_y(g_act), z];

        let surfaced = plan.placement != Placement::Buried;
        if surfaced && plan.crater_radius > 0.0 {
            self.crater(plan, centre);
        }
        self.body(plan, centre);
        if surfaced {
            self.clear(plan, centre);
        }
    }

    /// The columns within `radius` of the centre column and inside the
    /// window, with their horizontal distance.
    fn columns(&self, centre: [i32; 3], radius: f64) -> Vec<(i32, i32, f64)> {
        let reach = radius.ceil() as i32;
        let mut out = Vec::new();
        for dz in -reach..=reach {
            for dx in -reach..=reach {
                let (x, z) = (centre[0] + dx, centre[2] + dz);
                if x < self.min[0] || x > self.max[0] || z < self.min[2] || z > self.max[2] {
                    continue;
                }
                let h = f64::from(dx * dx + dz * dz).sqrt();
                if h <= radius {
                    out.push((x, z, h));
                }
            }
        }
        out
    }

    /// Lowers the columns around the meteor into a bowl and raises a rim.
    fn crater(&mut self, plan: &MeteorPlan, centre: [i32; 3]) {
        let r = plan.radius();
        let fc = plan.crater_radius;
        let depth = ((r / 4.0).round() as i32).clamp(1, 3);
        for (x, z, h) in self.columns(centre, fc) {
            if h <= r || self.body_in_column(plan, centre, x, z) {
                continue;
            }
            let Some(gy) = self.top_ground(x, z) else {
                continue;
            };
            let top = self.world.block([x, gy, z]);
            let above = self.world.block([x, gy + 1, z]);
            let liquid = self.class([x, gy + 1, z]) == LIQUID;
            if h > fc - 1.5 && !liquid {
                // Rim: raised by one block.
                if gy > self.min[1] && self.class([x, gy - 1, z]) == GROUND {
                    let below = self.world.block([x, gy - 1, z]);
                    self.set([x, gy + 1, z], top);
                    self.set([x, gy, z], below);
                }
                continue;
            }
            let k = (f64::from(depth) * (1.0 - (h - r) / (fc - r))).round() as i32;
            if k <= 0 || gy - k < self.min[1] {
                continue;
            }
            let fill = if liquid { above } else { self.m.air };
            for y in gy - k + 1..=gy {
                self.set([x, y, z], fill);
            }
            self.set([x, gy - k, z], top);
        }
    }

    /// The body's inside test for the offset `d` from its centre: the
    /// distance and the noisy radius in that direction, or `None` if the
    /// block is outside. The centre is always inside.
    fn body_at(plan: &MeteorPlan, d: [i32; 3]) -> Option<(f64, f64)> {
        let r = plan.radius();
        let dist = f64::from(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let r_dir = if dist == 0.0 {
            r
        } else {
            let dir = [
                f64::from(d[0]) / dist,
                f64::from(d[1]) / dist,
                f64::from(d[2]) / dist,
            ];
            r * (1.0 + NOISE_AMP * plan.noise.at(dir))
        };
        (dist <= r_dir).then_some((dist, r_dir))
    }

    /// True when the body has at least one block in the column (x, z).
    fn body_in_column(&self, plan: &MeteorPlan, centre: [i32; 3], x: i32, z: i32) -> bool {
        let reach = plan.r_out().ceil() as i32;
        (-reach..=reach).any(|dy| {
            let y = centre[1] + dy;
            y >= self.min[1]
                && y <= self.max[1]
                && Self::body_at(plan, [x - centre[0], dy, z - centre[2]]).is_some()
        })
    }

    /// The ball: core, ore layer, deepslate shell.
    fn body(&mut self, plan: &MeteorPlan, centre: [i32; 3]) {
        let kind = &self.m.kinds[plan.kind];
        let r = plan.radius();
        let r_core = (0.28 * r).max(0.5);
        let t_shell = 1.0 + 0.06 * r;
        let seed = self.m.seed;
        let reach = plan.r_out().ceil() as i32;
        for dy in -reach..=reach {
            let y = centre[1] + dy;
            if y < self.min[1] || y > self.max[1] {
                continue;
            }
            for dz in -reach..=reach {
                let z = centre[2] + dz;
                if z < self.min[2] || z > self.max[2] {
                    continue;
                }
                for dx in -reach..=reach {
                    let x = centre[0] + dx;
                    if x < self.min[0] || x > self.max[0] {
                        continue;
                    }
                    let Some((dist, r_dir)) = Self::body_at(plan, [dx, dy, dz]) else {
                        continue;
                    };
                    let state = if dist < r_core {
                        kind.core
                    } else if r_dir - dist < t_shell {
                        if hash_unit(seed, SALT_FLECK, x, y, z) < FLECK {
                            kind.ore_deepslate
                        } else {
                            self.m.deepslate
                        }
                    } else {
                        match kind.second {
                            Some(second)
                                if hash_unit(seed, SALT_MIX, x, y, z) < self.m.mix_share =>
                            {
                                second
                            }
                            _ => kind.ore,
                        }
                    };
                    self.set([x, y, z], state);
                }
            }
        }
    }

    /// Removes vegetation in the footprint, from the ground up to the top of
    /// the window (grass, flowers, logs and leaves left floating).
    fn clear(&mut self, plan: &MeteorPlan, centre: [i32; 3]) {
        for (x, z, _) in self.columns(centre, plan.footprint) {
            for y in (self.min[1]..=self.max[1]).rev() {
                match self.class([x, y, z]) {
                    PLANT => self.set([x, y, z], self.m.air),
                    WATER_PLANT => self.set([x, y, z], self.m.water),
                    AIR | LIQUID => {}
                    _ => break,
                }
            }
        }
    }
}

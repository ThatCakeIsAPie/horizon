//! Water in Earth terrain: the sea with its shelf and shores, rivers carved from the water atlas, flat lakes.
//!
//! This module is the pure part of the water rules: plain functions over plain data (slices, a
//! [`VerticalMap`], [`LandClass`], atlas records). It knows nothing of the generator, so every rule is tested on
//! synthetic data (`water_tests.rs`) and the generator only feeds it and turns its answers into columns.
//!
//! # Units
//!
//! `x`, `z` and heights are blocks. `mpb` is the metres a block is wide; `vpm` the blocks a metre of elevation
//! is tall ([`vertical_blocks_per_metre`]). A sample step (`stride`, `step`) is the blocks between two samples
//! of a lattice: 1 for near cubes, more in the far view.
//!
//! # The top of a water column
//!
//! A level of `L` metres has the water top block `block_y(L) - 1` ([`water_top`]): a level of 0 m is the
//! sea's top, `sea_level_y - 1`, and a lake whose elevation surface is at its level sits one block under the
//! shore. Rivers never go under `sea_level_y - 1` ([`river_top`]).
//!
//! # Rivers
//!
//! A reach's width and depth come from its discharge by hydraulic geometry ([`river_geometry`]). A
//! [`RiverIndex`] holds the reaches of an area projected into block coordinates and finds, for a column, the
//! reach segment whose channel edge is nearest ([`RiverIndex::nearest`]), with the water level there (the
//! levels in the data never rise downstream, so the water only steps down). In the far view a river at least a
//! quarter of a sample wide shows as at least one voxel column wide ([`effective_half_width`]).
//!
//! * Channel (distance to the line at most the half width): water at the level, deepest in the middle.
//! * Corridor (land cover says water within [`RIVER_CORRIDOR_M`] or one half width beyond the channel): the
//!   wide river's real banks from the 10 m class: water at the same level, half as deep.
//! * Banks (dry, up to [`BANK_MIN_BLOCKS`] to [`BANK_MAX_BLOCKS`] beyond the channel): the ground blends from
//!   one block over the water to the elevation surface, so a river in a canyon sits on the canyon floor and
//!   a river over low ground has raised banks. A river more than [`MAX_CARVE`] under the ground is out of
//!   reach and leaves the ground alone (no pit, no carved ring).
//!
//! # The window pass
//!
//! [`water_window`] turns the elevation surfaces, classes, river hits and lake matches of a padded window of
//! samples into the final top and water of its inner columns:
//!
//! 1. Provisional water per column, by priority: river channel, river corridor, lake, sea, dry (with the bank
//!    rule). A lake is land-cover water above the sea (or any water of an atlas lake below it, such as a dead
//!    sea): flat at the lake's level from the atlas record, or from the lowest elevation surface of the lake's
//!    columns nearby; deeper away from the shore. The sea is land-cover water or no data at or under the sea
//!    level plus [`SEA_TOLERANCE_M`] (real coastal water often has an elevation of 0 to a few metres), with a
//!    shelf that deepens away from the nearest land column.
//! 2. Hard bounds: the ground is never more than [`MAX_CARVE`] under the elevation surface and a river or lake
//!    top is never more than [`MAX_RAISE`] over it (the sea is exempt: box classification treats everything
//!    above the sea level as air). A column the bounds leave without water is dry at its elevation surface.
//! 3. Containment, so no water hangs: a dry column next to a water column whose top is over its ground is
//!    raised to that top (up to [`MAX_RAISE`] over its elevation surface); then a water column next to a dry
//!    column lower than its top is lowered to it, its bed carved to keep one block of water, or turned dry
//!    when that carve is out of bounds; the lowering is done a second time so the columns next to one that
//!    just turned dry settle too. Steps between two water columns stay (a river's still-water step).
//! 4. Flags: dry columns near sea or lake water, or beside a river channel, and not far over its top are
//!    shore ([`SHORE_REACH_BLOCKS`], [`SHORE_HEIGHT`]); river channel columns are flagged for the biome.
//!
//! Every output column is a pure function of the inputs within [`WINDOW_PAD`] samples of it, so two windows
//! that both hold a column agree on it (neighbouring cubes and far nodes meet without seams).
//!
//! # Changes from the written design (`gen-design.md`)
//!
//! * [`WINDOW_PAD`] is `PAD + 3`, not `PAD + 2`: containment is a raise and two lowerings (radius 3), and the
//!   shore flag looks up to 3 samples around a column.
//! * [`water_top`] takes no sea level (the [`VerticalMap`] has it); [`river_top`] is the clamped form.
//! * A water column the hard bounds leave without water is dry at its elevation surface (not at its carved
//!   ground, which would be a pit).
//! * Atlas lakes under the sea level (the Caspian, a dead sea) claim their water-class columns even where the
//!   elevation is below the sea tolerance, instead of becoming sea.
//! * [`WindowInput`] holds a [`VerticalMap`] instead of `mpb` and `vpm` (the rules need neither once rivers and
//!   lakes are in blocks).

use std::collections::HashMap;
use std::sync::Arc;

use horizon_earth::{Lake, LandClass, LatLon, RiverReach, VerticalMap};

use super::surface::GRAVEL_FROM_LATITUDE;

/// The version of the generator's water rules: it goes up when what the same column with the same data is made of
/// changes. It is in the terrain key (with `horizon_earth::WATER_RULES`) while land cover is on.
pub const WATER_GEN_RULES: u32 = 1;

/// The most blocks the water rules lower a column's ground under its elevation surface (a river channel, a lake
/// bed, a shelf, a carved bank). Box classification widens its stone margin by this.
pub const MAX_CARVE: i64 = 32;

/// The most blocks the water rules put a column's top (ground or water) over its elevation surface (a raised bank,
/// water held over low ground). Box classification widens its air margin by this.
pub const MAX_RAISE: i64 = 8;

/// What the bed under a water column is made of (its top ground block).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bed {
    Sand,
    Gravel,
    Mud,
    Clay,
    Stone,
}

/// The coefficient of a river's width in metres, `w = coef * Q^exp` (Andreadis et al. 2013, `Q` in m3/s).
pub const RIVER_WIDTH_COEF: f64 = 7.2;
/// The exponent of a river's width in its discharge.
pub const RIVER_WIDTH_EXP: f64 = 0.5;
/// The coefficient of a river's depth in metres, `d = coef * Q^exp` (Andreadis et al. 2013).
pub const RIVER_DEPTH_COEF: f64 = 0.27;
/// The exponent of a river's depth in its discharge.
pub const RIVER_DEPTH_EXP: f64 = 0.39;
/// The narrowest river, in metres wide.
pub const RIVER_MIN_WIDTH_M: f64 = 2.0;
/// The smallest half width of a river, in blocks: every river in the data is at least a one block line.
pub const RIVER_MIN_HALF_WIDTH_BLOCKS: f64 = 0.5;
/// Land-cover water within `hw + max(hw, RIVER_CORRIDOR_M / mpb)` blocks of a reach is part of the river.
pub const RIVER_CORRIDOR_M: f64 = 30.0;
/// The narrowest a river bank is, in blocks.
pub const BANK_MIN_BLOCKS: i32 = 2;
/// The widest a river bank is, in blocks.
pub const BANK_MAX_BLOCKS: i32 = 16;

/// An elevation surface at most this many metres over the sea level still counts as sea where the land cover
/// says water or no data (the elevation over real coastal water is often 0 to a few metres).
pub const SEA_TOLERANCE_M: f64 = 3.0;
/// Blocks of shelf depth per block of distance from the nearest land column.
pub const SHELF_SLOPE: f64 = 0.75;
/// The deepest the shelf gets, in blocks of water.
pub const SHELF_MAX_BLOCKS: i32 = 6;

/// A lake record applies to a column whose elevation surface is within this many metres of the lake's level.
pub const LAKE_LEVEL_TOLERANCE_M: f64 = 20.0;
/// A lake's bounding box is grown by this many degrees when matching a column.
pub const LAKE_BOX_MARGIN_DEG: f64 = 0.002;
/// Blocks of lake depth per block of distance from the shore.
pub const LAKE_SHORE_SLOPE: f64 = 0.75;
/// The deepest a lake gets, in blocks of water.
pub const LAKE_MAX_DEPTH_BLOCKS: i32 = 16;
/// The least depth a record's mean depth gives a lake, in blocks.
pub const LAKE_MIN_DEPTH_BLOCKS: i32 = 2;
/// The deepest a lake with no record gets, in blocks of water.
pub const FALLBACK_LAKE_DEPTH: i32 = 3;

/// The samples around a column its distance rules look at (to the nearest land, shore or lake column).
pub const PAD: usize = 8;
/// The radius, in samples, of the containment and shore rules on top of [`PAD`]: a raise and two lowerings.
pub const CONTAINMENT_RADIUS: usize = 3;
/// The samples a window holds around the columns it answers for: every answer depends on the inputs within
/// this many samples and on nothing else.
pub const WINDOW_PAD: usize = PAD + CONTAINMENT_RADIUS;

/// A dry column within this many blocks of sea or lake water (at least one sample) can be shore.
pub const SHORE_REACH_BLOCKS: i32 = 3;
/// A shore column's ground is at most this many blocks over the water's top.
pub const SHORE_HEIGHT: i32 = 2;
/// A mean annual temperature (C) under this makes a shore gravel rather than sand.
pub const SHORE_GRAVEL_BELOW_C: f64 = 2.0;

/// The coarsest sample step, in blocks, a [`RiverIndex`] built with [`RiverIndex::build`] answers exactly (the
/// far view's widening of big rivers is bounded by it).
pub const DEFAULT_MAX_STEP: f64 = 256.0;
/// The side, in blocks, of the finest bin of a [`RiverIndex`]; each coarser level is 8 times wider.
const BIN_BLOCKS: f64 = 64.0;
/// How many bin levels a [`RiverIndex`] has (so big rivers are not listed in thousands of fine bins).
const BIN_LEVELS: usize = 6;

/// A river's size in blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverGeometry {
    /// Half the width of the channel, in blocks (at least [`RIVER_MIN_HALF_WIDTH_BLOCKS`]).
    pub half_width: f64,
    /// The depth of the channel in its middle, in blocks (at least 1).
    pub depth: f64,
}

/// The size of a river of `discharge_m3s` cubic metres a second, in blocks: `mpb` metres a block wide and `vpm`
/// blocks a metre tall. Wider and deeper with discharge; never under the minimums.
pub fn river_geometry(discharge_m3s: f64, mpb: f64, vpm: f64) -> RiverGeometry {
    let q = if discharge_m3s.is_finite() {
        discharge_m3s.max(0.0)
    } else {
        0.0
    };
    let width_m = (RIVER_WIDTH_COEF * q.powf(RIVER_WIDTH_EXP)).max(RIVER_MIN_WIDTH_M);
    let depth_m = RIVER_DEPTH_COEF * q.powf(RIVER_DEPTH_EXP);
    RiverGeometry {
        half_width: (width_m / mpb / 2.0).max(RIVER_MIN_HALF_WIDTH_BLOCKS),
        depth: (depth_m * vpm).max(1.0),
    }
}

/// The blocks a metre of elevation is tall under `vertical`.
pub fn vertical_blocks_per_metre(vertical: &VerticalMap) -> f64 {
    vertical.y_for_elevation(1.0) - vertical.y_for_elevation(0.0)
}

/// The top water block of water at `level_m` metres: the block of the level, less one.
pub fn water_top(level_m: f64, vertical: &VerticalMap) -> i32 {
    vertical.block_y(level_m) - 1
}

/// The top water block of a river at `level_m` metres: [`water_top`], and never under the sea's top.
pub fn river_top(level_m: f64, vertical: &VerticalMap) -> i32 {
    water_top(level_m, vertical).max(vertical.sea_level_y - 1)
}

/// The half width a river of half width `half_width` has at a sample step of `step` blocks: in the far view a
/// river at least a quarter of a sample wide still shows as at least one voxel column (half a step either side
/// of its line), so big rivers stay continuous in the distance. At a step of 1 nothing changes.
pub fn effective_half_width(half_width: f64, step: f64) -> f64 {
    if 2.0 * half_width >= step / 4.0 {
        half_width.max(step / 2.0)
    } else {
        half_width
    }
}

/// How far from its line a reach with `geometry` still matters at `step`: its (effective) half width, then the
/// corridor or the bank, whichever is wider (their sum, to be safe).
pub fn river_influence(geometry: &RiverGeometry, mpb: f64, step: f64) -> f64 {
    let hw = effective_half_width(geometry.half_width, step);
    hw + hw.max(RIVER_CORRIDOR_M / mpb) + f64::from(BANK_MAX_BLOCKS)
}

/// What a column finds of the river nearest to it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverHit {
    /// The id of the reach.
    pub reach_id: u32,
    /// The distance from the column to the reach's line, in blocks.
    pub distance: f64,
    /// The half width at the step asked for (see [`effective_half_width`]), in blocks.
    pub half_width: f64,
    /// The water surface elevation at the nearest point of the line, in metres.
    pub level_m: f64,
    /// The depth of the channel in its middle, in blocks.
    pub depth: f64,
    /// How far beyond the channel's edge land-cover water still belongs to the river, in blocks.
    pub corridor: f64,
}

/// A segment of a reach in block coordinates.
#[derive(Clone, Copy, Debug)]
struct Segment {
    a: (f64, f64),
    b: (f64, f64),
    level_a: f64,
    level_b: f64,
    reach: u32,
    index: u32,
}

/// What an index keeps of a reach.
#[derive(Clone, Copy, Debug)]
struct ReachInfo {
    id: u32,
    geometry: RiverGeometry,
}

/// One level of the uniform grids: segments listed in every bin their box, grown by their influence, meets.
#[derive(Debug, Default)]
struct BinLevel {
    side: f64,
    bins: HashMap<(i32, i32), Vec<u32>>,
}

/// The reaches of an area, projected into blocks and binned for nearest queries. Built once per area and shared
/// (`Send + Sync`, read only).
#[derive(Debug)]
pub struct RiverIndex {
    segments: Vec<Segment>,
    reaches: Vec<ReachInfo>,
    levels: Vec<BinLevel>,
    mpb: f64,
    max_influence: f64,
}

impl RiverIndex {
    /// An index of `reaches`, projected with `project` (the block `(x, z)` of a point: the generator hands in
    /// its projection), at `mpb` metres a block wide and `vpm` blocks a metre tall, answering exactly up to
    /// [`DEFAULT_MAX_STEP`].
    pub fn build(
        reaches: &[Arc<RiverReach>],
        project: impl Fn(LatLon) -> (f64, f64),
        mpb: f64,
        vpm: f64,
    ) -> Self {
        Self::build_with_max_step(reaches, project, mpb, vpm, DEFAULT_MAX_STEP)
    }

    /// [`RiverIndex::build`] for queries up to a sample step of `max_step` blocks.
    pub fn build_with_max_step(
        reaches: &[Arc<RiverReach>],
        project: impl Fn(LatLon) -> (f64, f64),
        mpb: f64,
        vpm: f64,
        max_step: f64,
    ) -> Self {
        let mut index = Self {
            segments: Vec::new(),
            reaches: Vec::new(),
            levels: (0..BIN_LEVELS)
                .map(|k| BinLevel {
                    side: BIN_BLOCKS * 8f64.powi(k as i32),
                    bins: HashMap::new(),
                })
                .collect(),
            mpb,
            max_influence: 0.0,
        };
        for reach in reaches {
            if reach.points.len() < 2 || reach.levels_m.len() != reach.points.len() {
                continue;
            }
            let points: Vec<(f64, f64)> = reach.points.iter().map(|&p| project(p)).collect();
            if points.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
                continue;
            }
            let geometry = river_geometry(f64::from(reach.discharge_m3s), mpb, vpm);
            let radius = Self::binning_radius(&geometry, mpb, max_step);
            index.max_influence = index.max_influence.max(radius);
            let level = index
                .levels
                .iter()
                .position(|l| l.side >= radius)
                .unwrap_or(BIN_LEVELS - 1);
            let reach_slot = index.reaches.len() as u32;
            index.reaches.push(ReachInfo {
                id: reach.id,
                geometry,
            });
            for i in 0..points.len() - 1 {
                let segment = Segment {
                    a: points[i],
                    b: points[i + 1],
                    level_a: f64::from(reach.levels_m[i]),
                    level_b: f64::from(reach.levels_m[i + 1]),
                    reach: reach_slot,
                    index: i as u32,
                };
                let slot = index.segments.len() as u32;
                index.segments.push(segment);
                let side = index.levels[level].side;
                let bin = |v: f64| (v / side).floor() as i32;
                let (x0, x1) = (segment.a.0.min(segment.b.0), segment.a.0.max(segment.b.0));
                let (z0, z1) = (segment.a.1.min(segment.b.1), segment.a.1.max(segment.b.1));
                for bx in bin(x0 - radius)..=bin(x1 + radius) {
                    for bz in bin(z0 - radius)..=bin(z1 + radius) {
                        index.levels[level]
                            .bins
                            .entry((bx, bz))
                            .or_default()
                            .push(slot);
                    }
                }
            }
        }
        index
    }

    /// The largest influence a reach with `geometry` has at any step from 1 to `max_step` (steps are powers
    /// of two in the far view).
    fn binning_radius(geometry: &RiverGeometry, mpb: f64, max_step: f64) -> f64 {
        let max_step = max_step.max(1.0);
        let mut radius = river_influence(geometry, mpb, 1.0);
        let mut step = 2.0;
        while step <= max_step {
            radius = radius.max(river_influence(geometry, mpb, step));
            step *= 2.0;
        }
        radius.max(river_influence(geometry, mpb, max_step))
    }

    /// Whether the index holds no segment.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The farthest, in blocks, any reach of the index reaches from its line (at any step up to the index's
    /// maximum): a caller grows the box it asks the atlas for by this.
    pub fn max_influence(&self) -> f64 {
        self.max_influence
    }

    /// The river whose channel edge is nearest to the column at block `(x, z)`, for a lattice of `step`
    /// blocks, or `None` beyond every reach's influence. The choice minimises the signed distance to the edge
    /// (`distance - half_width`, the widened half width of the far view), so a larger river wins over a
    /// smaller one at the same distance; ties go to the lower reach id, then the lower segment.
    pub fn nearest(&self, x: f64, z: f64, step: f64) -> Option<RiverHit> {
        let step = step.max(1.0);
        let mut best: Option<(f64, u32, u32, RiverHit)> = None;
        for level in &self.levels {
            if level.bins.is_empty() {
                continue;
            }
            let key = (
                (x / level.side).floor() as i32,
                (z / level.side).floor() as i32,
            );
            let Some(list) = level.bins.get(&key) else {
                continue;
            };
            for &slot in list {
                let seg = &self.segments[slot as usize];
                let reach = &self.reaches[seg.reach as usize];
                let (d, t) = distance_to_segment(x, z, seg);
                if d > river_influence(&reach.geometry, self.mpb, step) {
                    continue;
                }
                let hw = effective_half_width(reach.geometry.half_width, step);
                let signed = d - hw;
                let better = match &best {
                    None => true,
                    Some((s, id, idx, _)) => (signed, reach.id, seg.index)
                        .partial_cmp(&(*s, *id, *idx))
                        .is_some_and(|o| o.is_lt()),
                };
                if better {
                    let level_m = if t <= 0.0 {
                        seg.level_a
                    } else if t >= 1.0 {
                        seg.level_b
                    } else {
                        seg.level_a + (seg.level_b - seg.level_a) * t
                    };
                    best = Some((
                        signed,
                        reach.id,
                        seg.index,
                        RiverHit {
                            reach_id: reach.id,
                            distance: d,
                            half_width: hw,
                            level_m,
                            depth: reach.geometry.depth,
                            corridor: hw.max(RIVER_CORRIDOR_M / self.mpb),
                        },
                    ));
                }
            }
        }
        best.map(|b| b.3)
    }
}

/// The distance from `(x, z)` to a segment and the parameter (0 to 1, clamped) of its nearest point.
fn distance_to_segment(x: f64, z: f64, seg: &Segment) -> (f64, f64) {
    let (dx, dz) = (seg.b.0 - seg.a.0, seg.b.1 - seg.a.1);
    let len2 = dx * dx + dz * dz;
    let t = if len2 > 0.0 {
        ((x - seg.a.0) * dx + (z - seg.a.1) * dz) / len2
    } else {
        0.0
    };
    let (px, pz) = if t <= 0.0 {
        seg.a
    } else if t >= 1.0 {
        seg.b
    } else {
        (seg.a.0 + dx * t, seg.a.1 + dz * t)
    };
    (
        ((x - px).powi(2) + (z - pz).powi(2)).sqrt(),
        t.clamp(0.0, 1.0),
    )
}

/// The level of the lake that holds a column: its water top and how deep it gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LakeLevel {
    /// The top water block (flat across the lake).
    pub top: i32,
    /// The most blocks of water over the bed, in the lake's middle.
    pub max_depth: i32,
}

impl LakeLevel {
    /// The level of `lake` under `vertical` (`vpm` blocks a metre tall): its top is [`water_top`] of its level,
    /// its depth the record's mean depth in blocks, between [`LAKE_MIN_DEPTH_BLOCKS`] and
    /// [`LAKE_MAX_DEPTH_BLOCKS`].
    pub fn from_lake(lake: &Lake, vertical: &VerticalMap, vpm: f64) -> Self {
        let depth = (f64::from(lake.depth_avg_m) * vpm).round();
        let depth = if depth.is_finite() { depth } else { 0.0 };
        Self {
            top: water_top(f64::from(lake.level_m), vertical),
            max_depth: (depth as i32).clamp(LAKE_MIN_DEPTH_BLOCKS, LAKE_MAX_DEPTH_BLOCKS),
        }
    }

    /// The level of a lake with no record (see [`FALLBACK_LAKE_DEPTH`]) whose top is `top`.
    pub fn fallback(top: i32) -> Self {
        Self {
            top,
            max_depth: FALLBACK_LAKE_DEPTH,
        }
    }
}

/// The lake record for the point `(lat, lon)` whose elevation surface is `dem_elevation_m` metres: among the
/// lakes whose box, grown by [`LAKE_BOX_MARGIN_DEG`], holds the point, the one whose level is closest to the
/// elevation, if within [`LAKE_LEVEL_TOLERANCE_M`]; ties go to the lower id.
pub fn pick_lake(lakes: &[Arc<Lake>], lat: f64, lon: f64, dem_elevation_m: f64) -> Option<&Lake> {
    let m = LAKE_BOX_MARGIN_DEG;
    let mut best: Option<(f64, u32, &Lake)> = None;
    for lake in lakes {
        let inside = lat >= lake.south - m
            && lat <= lake.north + m
            && lon >= lake.west - m
            && lon <= lake.east + m;
        if !inside {
            continue;
        }
        let diff = (f64::from(lake.level_m) - dem_elevation_m).abs();
        if diff.is_nan() || diff > LAKE_LEVEL_TOLERANCE_M {
            continue;
        }
        if best.is_none_or(|(d, id, _)| (diff, lake.id) < (d, id)) {
            best = Some((diff, lake.id, lake));
        }
    }
    best.map(|b| b.2)
}

/// Whether a shore is cold (gravel rather than sand): the mean annual temperature is under
/// [`SHORE_GRAVEL_BELOW_C`], or with no climate the latitude is at least [`GRAVEL_FROM_LATITUDE`] from the
/// equator.
pub fn shore_is_cold(mean_annual_c: Option<f64>, latitude: f64) -> bool {
    match mean_annual_c {
        Some(t) => t < SHORE_GRAVEL_BELOW_C,
        None => latitude.abs() >= GRAVEL_FROM_LATITUDE,
    }
}

/// What a water column is part of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaterOf {
    Sea,
    River,
    Lake,
}

/// The padded window of samples [`water_window`] works on: a lattice of `width` by `rows` samples, `stride`
/// blocks apart, row after row (`z` rows, `x` along a row), with [`WINDOW_PAD`] samples of margin on each side
/// of the columns the answer is for.
#[derive(Clone, Copy, Debug)]
pub struct WindowInput<'a> {
    pub width: usize,
    pub rows: usize,
    /// Blocks between two samples.
    pub stride: i32,
    pub vertical: VerticalMap,
    /// The elevation surface block of each sample.
    pub dem: &'a [i32],
    /// The land cover class of each sample.
    pub class: &'a [LandClass],
    /// The nearest river of each sample ([`RiverIndex::nearest`] at this stride).
    pub river: &'a [Option<RiverHit>],
    /// The atlas lake record that holds each sample ([`pick_lake`]).
    pub lake: &'a [Option<LakeLevel>],
}

/// What the water rules make of one column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnWater {
    /// The column's top block: the top water block of a water column, else the top of the ground (raised or
    /// carved from the elevation surface by banks and containment).
    pub s: i32,
    /// For a water column: the blocks of water over the bed (at least 1) and the bed. The ground is
    /// `s - depth`.
    pub water: Option<(u32, Bed)>,
    /// What the water is part of.
    pub of: Option<WaterOf>,
    /// A dry column at the water's edge: sand, gravel or stone by the caller's climate and slope rule.
    pub shore: bool,
    /// A river channel column (the biome is `river`).
    pub channel: bool,
}

/// The provisional and final state of a column in the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Dry(i32),
    Water {
        top: i32,
        ground: i32,
        of: WaterOf,
        channel: bool,
    },
}

/// `3u^2 - 2u^3` on 0 to 1.
fn smoothstep(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// The blocks of water at `distance_blocks` from the nearest land or shore: 1 next to it, growing by `slope` a
/// block, from 1 to `max`.
fn shore_depth(distance_blocks: f64, slope: f64, max: i32) -> i32 {
    let d = 1.0 + (distance_blocks - 1.0).max(0.0) * slope;
    (d.round() as i32).clamp(1, max.max(1))
}

/// The 8 neighbours of `(x, z)` inside a `w` by `h` grid, as indexes.
fn neighbours(x: usize, z: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let (xi, zi) = (x as isize, z as isize);
    (-1isize..=1)
        .flat_map(move |dz| (-1isize..=1).map(move |dx| (dx, dz)))
        .filter(|&(dx, dz)| dx != 0 || dz != 0)
        .filter_map(move |(dx, dz)| {
            let (nx, nz) = (xi + dx, zi + dz);
            (nx >= 0 && nz >= 0 && (nx as usize) < w && (nz as usize) < h)
                .then(|| nz as usize * w + nx as usize)
        })
}

/// The distance, in samples, from `(x, z)` to the nearest sample within [`PAD`] of it (in the grid) for which
/// `flag` holds, or `None`.
fn nearest_flag(flag: &[bool], w: usize, h: usize, x: usize, z: usize) -> Option<f64> {
    let (xi, zi) = (x as isize, z as isize);
    let mut best = f64::MAX;
    for r in 1..=PAD as isize {
        if (r as f64) > best {
            break;
        }
        for dz in -r..=r {
            let step = if dz.abs() == r { 1 } else { 2 * r };
            let mut dx = -r;
            while dx <= r {
                let (nx, nz) = (xi + dx, zi + dz);
                if nx >= 0 && nz >= 0 && (nx as usize) < w && (nz as usize) < h {
                    let i = nz as usize * w + nx as usize;
                    if flag[i] {
                        best = best.min(((dx * dx + dz * dz) as f64).sqrt());
                    }
                }
                dx += step;
            }
        }
    }
    (best < f64::MAX).then_some(best)
}

/// The final water of the inner columns of a window: the `width - 2 * WINDOW_PAD` by `rows - 2 * WINDOW_PAD`
/// columns, row after row. Empty when the window has no inner column.
///
/// Panics if the slices are not `width * rows` long.
pub fn water_window(input: &WindowInput<'_>) -> Vec<ColumnWater> {
    let (w, h) = (input.width, input.rows);
    let n = w * h;
    assert!(
        input.dem.len() == n
            && input.class.len() == n
            && input.river.len() == n
            && input.lake.len() == n,
        "water_window: slices must hold width * rows samples"
    );
    if w <= 2 * WINDOW_PAD || h <= 2 * WINDOW_PAD {
        return Vec::new();
    }
    let sea = input.vertical.sea_level_y;
    let vpm = vertical_blocks_per_metre(&input.vertical);
    let tol = (SEA_TOLERANCE_M * vpm).round().clamp(0.0, 1.0e6) as i32;
    let sea_cut = sea - 1 + tol;
    let below_sea_lake = sea - 1 - tol;
    let stride = f64::from(input.stride.max(1));

    // Which samples are water of which kind, before anything claims them.
    let is_class_water: Vec<bool> = input.class.iter().map(|&c| c == LandClass::Water).collect();
    let lake_cand: Vec<bool> = (0..n)
        .map(|i| {
            is_class_water[i]
                && (input.dem[i] > sea_cut || input.lake[i].is_some_and(|l| l.top < below_sea_lake))
        })
        .collect();
    let sea_cand: Vec<bool> = (0..n)
        .map(|i| {
            !lake_cand[i]
                && matches!(input.class[i], LandClass::Water | LandClass::NoData)
                && input.dem[i] <= sea_cut
        })
        .collect();
    let ctx = Context {
        input,
        lake_cand,
        sea_cand: sea_cand.clone(),
        not_water_class: is_class_water.iter().map(|&b| !b).collect(),
        land: sea_cand.iter().map(|&b| !b).collect(),
        stride,
    };

    // 1. Provisional water.
    let mut cells: Vec<Cell> = Vec::with_capacity(n);
    for z in 0..h {
        for x in 0..w {
            cells.push(ctx.provisional(x, z));
        }
    }

    // 2. Hard bounds.
    for (i, cell) in cells.iter_mut().enumerate() {
        *cell = bounded(*cell, input.dem[i]);
    }
    let bounded_cells = cells.clone();

    // 3. Containment: raise dry neighbours, then lower water twice.
    cells = raise_dry(&cells, input.dem, w, h);
    for _ in 0..2 {
        cells = lower_water(&cells, input.dem, w, h);
    }

    // 4. Flags and output.
    let reach = (SHORE_REACH_BLOCKS / input.stride.max(1)).max(1) as isize;
    let mut out = Vec::with_capacity((w - 2 * WINDOW_PAD) * (h - 2 * WINDOW_PAD));
    for z in WINDOW_PAD..h - WINDOW_PAD {
        for x in WINDOW_PAD..w - WINDOW_PAD {
            out.push(match cells[z * w + x] {
                Cell::Water {
                    top,
                    ground,
                    of,
                    channel,
                } => ColumnWater {
                    s: top,
                    water: Some((
                        (top - ground).max(1) as u32,
                        if of == WaterOf::River {
                            Bed::Gravel
                        } else {
                            Bed::Sand
                        },
                    )),
                    of: Some(of),
                    shore: false,
                    channel,
                },
                Cell::Dry(s) => ColumnWater {
                    s,
                    water: None,
                    of: None,
                    shore: is_shore(&bounded_cells, (w, h), (x, z), s, reach),
                    channel: false,
                },
            });
        }
    }
    out
}

/// The window and what the provisional rules read of it.
struct Context<'a> {
    input: &'a WindowInput<'a>,
    /// Land-cover water that is a lake (above the sea, or an atlas lake under it).
    lake_cand: Vec<bool>,
    /// Land-cover water or no data at the sea level that no lake claims.
    sea_cand: Vec<bool>,
    /// Not water class (the shore of a lake).
    not_water_class: Vec<bool>,
    /// Not a sea candidate (the land the shelf deepens away from).
    land: Vec<bool>,
    stride: f64,
}

impl Context<'_> {
    /// The provisional cell of the sample at `(x, z)`: river channel, river corridor, lake, sea, then dry.
    fn provisional(&self, x: usize, z: usize) -> Cell {
        let input = self.input;
        let (w, h) = (input.width, input.rows);
        let i = z * w + x;
        let sea = input.vertical.sea_level_y;
        let s0 = input.dem[i];
        let river = input.river[i];
        if let Some(hit) = river {
            let top = river_top(hit.level_m, &input.vertical);
            let hw = hit.half_width.max(f64::MIN_POSITIVE);
            if hit.distance <= hw {
                let f = (hit.distance / hw).clamp(0.0, 1.0);
                let depth = ((hit.depth * (1.0 - f * f)).round() as i32).max(1);
                return Cell::Water {
                    top,
                    ground: top - depth,
                    of: WaterOf::River,
                    channel: true,
                };
            }
            if input.class[i] == LandClass::Water && hit.distance <= hw + hit.corridor {
                let depth = ((hit.depth / 2.0).round() as i32).max(1);
                return Cell::Water {
                    top,
                    ground: top - depth,
                    of: WaterOf::River,
                    channel: false,
                };
            }
        }
        if self.lake_cand[i] {
            let level = match input.lake[i] {
                Some(level) => level,
                None => LakeLevel::fallback(self.fallback_lake_top(x, z)),
            };
            let depth = match nearest_flag(&self.not_water_class, w, h, x, z) {
                Some(d) => shore_depth(d * self.stride, LAKE_SHORE_SLOPE, level.max_depth),
                None => level.max_depth.max(1),
            };
            return Cell::Water {
                top: level.top,
                ground: (level.top - depth).min(s0),
                of: WaterOf::Lake,
                channel: false,
            };
        }
        if self.sea_cand[i] {
            let top = sea - 1;
            let shelf = match nearest_flag(&self.land, w, h, x, z) {
                Some(d) => shore_depth(d * self.stride, SHELF_SLOPE, SHELF_MAX_BLOCKS),
                None => SHELF_MAX_BLOCKS,
            };
            return Cell::Water {
                top,
                ground: s0.min(top - shelf),
                of: WaterOf::Sea,
                channel: false,
            };
        }
        // Dry; a bank beside a river blends the ground to the water.
        if let Some(hit) = river {
            let top = river_top(hit.level_m, &input.vertical);
            let b = f64::from(
                (s0 - (top + 1))
                    .abs()
                    .clamp(BANK_MIN_BLOCKS, BANK_MAX_BLOCKS),
            );
            let beyond = hit.distance - hit.half_width;
            // A river far under the ground (more than MAX_CARVE) is out of reach: leave the rim alone.
            let reachable = i64::from(top) + 1 >= i64::from(s0) - MAX_CARVE;
            if reachable && beyond > 0.0 && beyond <= b {
                let low = f64::from(top + 1);
                let target = low + (f64::from(s0) - low) * smoothstep(beyond / b);
                let s = (target.round() as i64)
                    .clamp(i64::from(s0) - MAX_CARVE, i64::from(s0) + MAX_RAISE);
                return Cell::Dry(s as i32);
            }
        }
        Cell::Dry(s0)
    }

    /// The top of a lake with no record at `(x, z)`: one block under the lowest elevation surface of the lake's
    /// columns within [`PAD`] samples.
    fn fallback_lake_top(&self, x: usize, z: usize) -> i32 {
        let input = self.input;
        let (w, h) = (input.width, input.rows);
        let mut lowest = input.dem[z * w + x];
        for nz in z.saturating_sub(PAD)..=(z + PAD).min(h - 1) {
            for nx in x.saturating_sub(PAD)..=(x + PAD).min(w - 1) {
                let j = nz * w + nx;
                if self.lake_cand[j] {
                    lowest = lowest.min(input.dem[j]);
                }
            }
        }
        lowest - 1
    }
}

/// A cell inside the hard bounds ([`MAX_CARVE`], [`MAX_RAISE`]) of its elevation surface `s0`.
fn bounded(cell: Cell, s0: i32) -> Cell {
    match cell {
        Cell::Dry(_) => cell,
        Cell::Water {
            top,
            ground,
            of,
            channel,
        } => {
            let top = if of == WaterOf::Sea {
                top
            } else {
                top.min(s0.saturating_add(MAX_RAISE as i32))
            };
            let ground = ground.max(s0.saturating_sub(MAX_CARVE as i32));
            if top <= ground {
                Cell::Dry(s0)
            } else {
                Cell::Water {
                    top,
                    ground,
                    of,
                    channel,
                }
            }
        }
    }
}

/// Containment, first half: a dry cell next to water whose top is over its ground is raised to the highest such
/// top, up to [`MAX_RAISE`] over its elevation surface.
fn raise_dry(cells: &[Cell], dem: &[i32], w: usize, h: usize) -> Vec<Cell> {
    let mut out = cells.to_vec();
    for z in 0..h {
        for x in 0..w {
            let i = z * w + x;
            let Cell::Dry(s) = cells[i] else { continue };
            let highest = neighbours(x, z, w, h)
                .filter_map(|j| match cells[j] {
                    Cell::Water { top, .. } => Some(top),
                    Cell::Dry(_) => None,
                })
                .max();
            if let Some(top) = highest
                && top > s
            {
                let cap = dem[i].saturating_add(MAX_RAISE as i32);
                out[i] = Cell::Dry(s.max(top.min(cap)));
            }
        }
    }
    out
}

/// Containment, second half: a water cell next to a dry cell lower than its top is lowered to that ground; if
/// no water is left its bed is carved to keep one block (within [`MAX_CARVE`]), else it turns dry.
fn lower_water(cells: &[Cell], dem: &[i32], w: usize, h: usize) -> Vec<Cell> {
    let mut out = cells.to_vec();
    for z in 0..h {
        for x in 0..w {
            let i = z * w + x;
            let Cell::Water {
                top,
                ground,
                of,
                channel,
            } = cells[i]
            else {
                continue;
            };
            let lowest = neighbours(x, z, w, h)
                .filter_map(|j| match cells[j] {
                    Cell::Dry(s) => Some(s),
                    Cell::Water { .. } => None,
                })
                .min();
            let Some(low) = lowest.filter(|&s| s < top) else {
                continue;
            };
            out[i] = if low > ground {
                Cell::Water {
                    top: low,
                    ground,
                    of,
                    channel,
                }
            } else if i64::from(low) > i64::from(dem[i]) - MAX_CARVE {
                Cell::Water {
                    top: low,
                    ground: low - 1,
                    of,
                    channel,
                }
            } else {
                Cell::Dry(dem[i])
            };
        }
    }
    out
}

/// Whether the dry column at `(x, z)` with top `s` is shore: sea or lake water within `reach` samples (at the
/// bounded stage, Euclidean) whose top is at most [`SHORE_HEIGHT`] under `s`, or a river channel beside it.
fn is_shore(
    bounded_cells: &[Cell],
    (w, h): (usize, usize),
    (x, z): (usize, usize),
    s: i32,
    reach: isize,
) -> bool {
    let (xi, zi) = (x as isize, z as isize);
    for dz in -reach..=reach {
        for dx in -reach..=reach {
            if (dx == 0 && dz == 0) || dx * dx + dz * dz > reach * reach {
                continue;
            }
            let (nx, nz) = (xi + dx, zi + dz);
            if nx < 0 || nz < 0 || nx as usize >= w || nz as usize >= h {
                continue;
            }
            if let Cell::Water { top, of, .. } = bounded_cells[nz as usize * w + nx as usize]
                && of != WaterOf::River
                && s <= top + SHORE_HEIGHT
            {
                return true;
            }
        }
    }
    neighbours(x, z, w, h).any(|j| {
        matches!(
            bounded_cells[j],
            Cell::Water { top, channel: true, .. } if s <= top + SHORE_HEIGHT
        )
    })
}

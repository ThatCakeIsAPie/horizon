//! Trees and snow in the far view of an Earth world.
//!
//! The far view never runs vanilla's features (they read blocks and wait). It draws simple trees from data it
//! already holds without waiting: the surface, the land class and the Earth [`Biome`] of each column. A node
//! whose class or biome is missing is [`NotReady`] like one whose height is missing.
//!
//! # Styles
//!
//! [`style`] gives each biome one tree: logs and leaves, a shape, a height range, a canopy radius and a spacing
//! (about 16 over the square root of vanilla's trees per chunk). No other biome has trees in the far view.
//!
//! | Biome | Tree | Shape | Height | Radius | Spacing |
//! |---|---|---|---|---|---|
//! | forest, flower forest | oak (one in five birch) | round | 5-7 | 2 | 5 |
//! | birch forest | birch | round | 5-7 | 2 | 5 |
//! | old growth birch forest | birch | round | 10-14 | 2 | 5 |
//! | dark forest | dark oak | round | 6-9 | 3 | 4 |
//! | taiga, snowy taiga | spruce | cone | 7-10 | 2 | 5 |
//! | old growth spruce and pine taiga | spruce | cone | 15-28 | 3 | 6 |
//! | jungle | jungle | round | 8-24 | 3 | 4 |
//! | bamboo jungle | jungle | round | 8-24 | 3 | 6 |
//! | sparse jungle | jungle | round | 6-12 | 3 | 9 |
//! | savanna, savanna plateau | acacia | flat | 5-8 | 3 | 11 |
//! | swamp | oak | round | 5-7 | 3 | 8 |
//! | mangrove swamp | mangrove | round | 6-10 | 3 | 5 |
//! | plains, sunflower plains, meadow | oak | round | 5-7 | 2 | 14 |
//! | snowy plains | spruce | cone | 7-10 | 2 | 20 |
//! | windswept gravelly hills | spruce | cone | 7-10 | 2 | 16 |
//! | wooded badlands | oak | round | 4-6 | 2 | 10 |
//!
//! Trees stand on dry ground only (grass, moss and marsh tops). The land class thins them as it does near
//! ([`super::vegetation::tree_share`]): tree cover, wetland, mangrove and no data all, shrub a fifth, grass a
//! fiftieth, the rest none.
//!
//! # Where they are drawn
//!
//! * Level 0 (exact cubes; level 1 is made of eight of them): [`EarthGenerator::decorate_cube`] puts the trees
//!   of a jittered grid, one candidate per cell of the style's spacing at a fixed hash of the cell (no world
//!   seed), kept where the class allows ([`super::vegetation::allowed`], the same rule as near). A tree whose
//!   trunk is in a neighbouring cube still draws its part: the columns are read [`MAX_RADIUS`] blocks beyond
//!   the cube. Leaves go only into air, logs into air or leaves.
//! * Levels 2 and up (voxels `step` blocks wide): no trees, a canopy per column. A column has one when a fixed
//!   hash of it is under the cover `share * (1 - exp(-pi (r + 1/2)^2 / spacing^2))`, from 16 blocks a step
//!   when that cover is 0.4 or more (a forest is then all leaves, not salt and pepper). The canopy covers the
//!   same heights as the level 0 tree of that height. Voxels whose middle it covers are leaves; when none
//!   does (the canopy is thinner than a voxel) the ground's top voxel is leaves. No logs are drawn.
//!
//! # Snow
//!
//! With snow on, a column is cold when its biome snows ([`Biome::snows`]). No height drop is applied: the
//! biomes already follow the real temperature at the real height. At level 0 a snow layer lies on the air
//! over the top block of a cold column (ground or a tree; not over snow blocks, ice or water), a grass block
//! under it is snowy, and the top block of cold water is ice. At levels 2 and up the top voxel of a cold column
//! is a snow block, the top water voxel ice, and the top voxel of a canopy is a snow block on one column in two
//! (speckled white and green, like snowy spruce from afar).
//!
//! Trees reach up to [`MAX_TREE_UP`] blocks over the ground, so a node is called air only from there on
//! ([`EarthGenerator::classify_node_above`]).

use std::collections::HashMap;

use horizon_core::position::SectionPos;
use super::biome::{BIOME_SAMPLE_M, Biome};
use super::far::NodeColumns;
use super::generator::{Access, EarthGenerator, Fill, NotReady, Rect};
use super::surface::SurfaceKind;
use super::vegetation::{FeatureKind, allowed, tree_share};
use crate::cubic::custom::replacers::resolve_block;
use crate::cubic::custom::settings::BlockStateDesc;
use crate::registry::BlockRegistry;

/// Blocks over the ground the highest tree block (and the snow on it) can be: the tallest tree is 28 blocks.
pub const MAX_TREE_UP: i32 = 32;
/// The widest canopy radius.
pub const MAX_RADIUS: i32 = 3;
/// A canopy cover at or over this makes a column leaves at the steps from [`SOLID_FROM_STEP`].
const SOLID_COVER: f64 = 0.4;
/// The voxel width from which a canopy is decided by its cover and not by a hash of the column.
const SOLID_FROM_STEP: i32 = 16;

/// What the far view trees and snow are made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FarBlocks {
    /// The log of each [`Species`], default state.
    pub logs: [u32; 7],
    /// The leaves of each [`Species`], default state.
    pub leaves: [u32; 7],
    /// A snow layer of one.
    pub snow: u32,
    pub ice: u32,
    /// A grass block with snow on it.
    pub snowy_grass: u32,
}

impl FarBlocks {
    /// The blocks in `registry`; fails naming the one it lacks.
    pub fn from_registry(registry: &BlockRegistry) -> Result<Self, String> {
        let get = |desc: BlockStateDesc| resolve_block(registry, &desc);
        let by = |suffix: &str| -> Result<[u32; 7], String> {
            let mut out = [0; 7];
            for (slot, species) in out.iter_mut().zip(Species::ALL) {
                *slot = get(BlockStateDesc::new(&format!("{}_{suffix}", species.name())))?;
            }
            Ok(out)
        };
        Ok(Self {
            logs: by("log")?,
            leaves: by("leaves")?,
            snow: get(BlockStateDesc::new("snow"))?,
            ice: get(BlockStateDesc::new("ice"))?,
            // A registry without the property (a synthetic one) has plain grass.
            snowy_grass: get(BlockStateDesc::with("grass_block", &[("snowy", "true")]))
                .or_else(|_| get(BlockStateDesc::new("grass_block")))?,
        })
    }

    fn is_leaves(&self, block: u32) -> bool {
        self.leaves.contains(&block)
    }
}

/// The kinds of tree the far view draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Oak,
    Birch,
    Spruce,
    DarkOak,
    Jungle,
    Acacia,
    Mangrove,
}

impl Species {
    pub const ALL: [Species; 7] = [
        Self::Oak,
        Self::Birch,
        Self::Spruce,
        Self::DarkOak,
        Self::Jungle,
        Self::Acacia,
        Self::Mangrove,
    ];

    /// The block name before `_log` and `_leaves`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Oak => "oak",
            Self::Birch => "birch",
            Self::Spruce => "spruce",
            Self::DarkOak => "dark_oak",
            Self::Jungle => "jungle",
            Self::Acacia => "acacia",
            Self::Mangrove => "mangrove",
        }
    }
}

/// The outline of a canopy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// A blob round the top of the trunk.
    Round,
    /// Rings widening from a tip at the top.
    Cone,
    /// Two flat discs on the top of the trunk.
    Flat,
}

/// The tree of a biome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Style {
    pub species: Species,
    /// Used on one tree in five, when there is one.
    pub mixed: Option<Species>,
    pub shape: Shape,
    /// The trunk's height in blocks, least and most.
    pub height: (i32, i32),
    /// How far the canopy reaches from the trunk.
    pub radius: i32,
    /// Blocks between trees (one candidate in each square of this side).
    pub spacing: i32,
}

/// The far-view tree of `biome`, if it has one (see the table in the module docs).
pub fn style(biome: Biome) -> Option<Style> {
    use Biome::*;
    use Shape::{Cone, Flat, Round};
    use Species::{Acacia, Birch, DarkOak, Mangrove, Oak, Spruce};
    let s = |species, shape, height, radius, spacing| Style {
        species,
        mixed: None,
        shape,
        height,
        radius,
        spacing,
    };
    Some(match biome {
        Forest | FlowerForest => Style {
            mixed: Some(Birch),
            ..s(Oak, Round, (5, 7), 2, 5)
        },
        BirchForest => s(Birch, Round, (5, 7), 2, 5),
        OldGrowthBirchForest => s(Birch, Round, (10, 14), 2, 5),
        DarkForest => s(DarkOak, Round, (6, 9), 3, 4),
        Taiga | SnowyTaiga => s(Spruce, Cone, (7, 10), 2, 5),
        OldGrowthSpruceTaiga | OldGrowthPineTaiga => s(Spruce, Cone, (15, 28), 3, 6),
        Jungle => s(Species::Jungle, Round, (8, 24), 3, 4),
        BambooJungle => s(Species::Jungle, Round, (8, 24), 3, 6),
        SparseJungle => s(Species::Jungle, Round, (6, 12), 3, 9),
        Savanna | SavannaPlateau => s(Acacia, Flat, (5, 8), 3, 11),
        Swamp => s(Oak, Round, (5, 7), 3, 8),
        MangroveSwamp => s(Mangrove, Round, (6, 10), 3, 5),
        Plains | SunflowerPlains | Meadow => s(Oak, Round, (5, 7), 2, 14),
        SnowyPlains => s(Spruce, Cone, (7, 10), 2, 20),
        WindsweptGravellyHills => s(Spruce, Cone, (7, 10), 2, 16),
        WoodedBadlands => s(Oak, Round, (4, 6), 2, 10),
        _ => return None,
    })
}

impl Style {
    /// The share of columns under canopy where every column may have a tree: trees of a spacing cover
    /// `pi (r + 1/2)^2` columns each, overlapping.
    pub fn cover(&self) -> f64 {
        let area = std::f64::consts::PI * (f64::from(self.radius) + 0.5).powi(2);
        1.0 - (-area / f64::from(self.spacing * self.spacing)).exp()
    }
}

/// What the far view draws over a world's terrain; both off is the terrain alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FarDecor {
    pub trees: bool,
    pub snow: bool,
}

impl FarDecor {
    /// Whether nothing is drawn.
    pub fn is_off(self) -> bool {
        !self.trees && !self.snow
    }
}

/// A fixed hash of two coordinates and a salt (splitmix64 finalizer).
fn mix(x: i32, z: i32, salt: u64) -> u64 {
    let mut h = (u64::from(x as u32) << 32 | u64::from(z as u32))
        .wrapping_add(salt.wrapping_mul(0xD6E8_FEB8_6659_FD93))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// A hash as 0..1.
fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// The tree a hash `h` gives in `style`: its species and trunk height.
fn pick(style: &Style, h: u64) -> (Species, i32) {
    let species = match style.mixed {
        Some(other) if (h >> 40) % 100 < 20 => other,
        _ => style.species,
    };
    let (lo, hi) = style.height;
    (species, lo + ((h >> 48) % (hi - lo + 1) as u64) as i32)
}

/// The heights a canopy of a trunk of `h` blocks over ground `s` covers, both ends included.
fn crown(style: &Style, s: i32, h: i32) -> (i32, i32) {
    let top = s + h + 1;
    let lo = match style.shape {
        Shape::Round => (s + h - 1 - style.radius).max(s + 1),
        Shape::Cone => s + 1 + (h / 4).max(2),
        Shape::Flat => s + h,
    };
    (lo, top)
}

/// One tree of the jittered grid.
#[derive(Clone, Copy, Debug)]
struct Tree {
    x: i32,
    z: i32,
    /// The surface block height of its column.
    s: i32,
    style: Style,
    species: Species,
    h: i32,
}

impl Tree {
    /// Calls `put(x, y, z)` for each leaf block of the canopy.
    fn leaves(&self, mut put: impl FnMut(i32, i32, i32)) {
        let (lo, top) = crown(&self.style, self.s, self.h);
        let r = self.style.radius;
        let ring = |put: &mut dyn FnMut(i32, i32, i32), y: i32, limit: i32| {
            let reach = (f64::from(limit).sqrt() as i32).max(0);
            for dz in -reach..=reach {
                for dx in -reach..=reach {
                    if dx * dx + dz * dz <= limit {
                        put(self.x + dx, y, self.z + dz);
                    }
                }
            }
        };
        match self.style.shape {
            Shape::Round => {
                let cy = self.s + self.h - 1;
                for y in lo..=top {
                    let dy = y - cy;
                    ring(&mut put, y, r * r + 1 - dy * dy);
                }
            }
            Shape::Cone => {
                for y in lo..=top {
                    let width = ((top - y + 1) / 2).min(r);
                    ring(&mut put, y, width * width + width.min(1));
                }
            }
            Shape::Flat => {
                ring(&mut put, top - 1, r * r + 1);
                ring(&mut put, top, (r - 1) * (r - 1) + 1);
            }
        }
    }
}

/// What a far node's column has over its terrain, made by [`EarthGenerator::decorate_columns`].
#[derive(Clone, Debug)]
pub(super) struct ColumnsDecor {
    cols: Vec<ColDecor>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ColDecor {
    /// The leaves of the canopy (0 for none) and the heights it covers.
    leaves: u32,
    bottom: i32,
    top: i32,
    /// No voxel above the ground is covered: the ground's top voxel is leaves.
    recolour: bool,
    /// The column is cold.
    snow: bool,
    /// The top of the column is water (not under a canopy): ice when cold.
    wet: bool,
    /// The lowest block of the column's top voxel.
    top_lo: i32,
    /// A cold canopy's top voxel is a snow block, not leaves.
    white: bool,
}

impl EarthGenerator {
    /// The biome of each column of `cols` (at the block it is sampled at), without loading anything.
    pub(super) fn column_biomes(&self, cols: &NodeColumns) -> Result<Vec<Biome>, NotReady> {
        let n = cols.width * cols.width;
        let mut seen: HashMap<(i32, i32), Biome> = HashMap::new();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (x, z) = cols.block_of(i);
            let q = (x >> 2, z >> 2);
            if let Some(&b) = seen.get(&q) {
                out.push(b);
                continue;
            }
            match self.biome_core(Access::Local, q.0, q.1) {
                Ok(b) => {
                    seen.insert(q, b);
                    out.push(b);
                }
                Err(_) => {
                    return Err(NotReady {
                        tiles: self.missing_in(Rect {
                            x0: q.0.saturating_mul(4),
                            z0: q.1.saturating_mul(4),
                            x1: q.0.saturating_mul(4).saturating_add(3),
                            z1: q.1.saturating_mul(4).saturating_add(3),
                            zoom: self.field.zoom_for(BIOME_SAMPLE_M),
                        }),
                    });
                }
            }
        }
        Ok(out)
    }

    /// Whether the top of a column of `kind` and surface `s` is water.
    fn water_top(&self, kind: SurfaceKind, s: i32) -> bool {
        match kind {
            SurfaceKind::Sand => s < self.sea_level(),
            SurfaceKind::Lake | SurfaceKind::Marsh => true,
            _ => false,
        }
    }

    /// The trees of the jittered grids in the block of columns `cols` (stride 1): each candidate is kept
    /// where its column's biome has a style of that spacing and the class allows a tree.
    fn tree_sites(&self, cols: &NodeColumns, biomes: &[Biome]) -> Vec<Tree> {
        let Some(classes) = &cols.classes else {
            return Vec::new();
        };
        let (ox, oz) = (cols.origin[0], cols.origin[1]);
        let w = cols.width as i32;
        let mut spacings: Vec<i32> = biomes
            .iter()
            .filter_map(|&b| style(b).map(|s| s.spacing))
            .collect();
        spacings.sort_unstable();
        spacings.dedup();
        let mut trees = Vec::new();
        for sp in spacings {
            let (c0x, c1x) = (ox.div_euclid(sp), (ox + w - 1).div_euclid(sp));
            let (c0z, c1z) = (oz.div_euclid(sp), (oz + w - 1).div_euclid(sp));
            for cz in c0z..=c1z {
                for cx in c0x..=c1x {
                    let h = mix(cx, cz, sp as u64);
                    let x = cx * sp + (h & 0xFFFF) as i32 % sp;
                    let z = cz * sp + (h >> 16 & 0xFFFF) as i32 % sp;
                    if x < ox || z < oz || x >= ox + w || z >= oz + w {
                        continue;
                    }
                    let i = ((z - oz) * w + (x - ox)) as usize;
                    let Some(style) = style(biomes[i]).filter(|s| s.spacing == sp) else {
                        continue;
                    };
                    if !matches!(
                        cols.kind(i),
                        SurfaceKind::Grass
                            | SurfaceKind::Scrub(_)
                            | SurfaceKind::Moss
                            | SurfaceKind::Marsh
                    ) || !allowed(classes[i], FeatureKind::Tree, x, z)
                    {
                        continue;
                    }
                    let (species, height) = pick(&style, h);
                    trees.push(Tree {
                        x,
                        z,
                        s: cols.surfaces[i],
                        style,
                        species,
                        h: height,
                    });
                }
            }
        }
        trees
    }

    /// Draws the far view's trees and snow of the cube `cube` into `out` (as [`EarthGenerator::generate`] made
    /// it); `true` when it changed anything. A cube proven to hold no ground, or only rock, within reach of a
    /// tree is not read. Gives [`NotReady`] when a height, class or biome around it is missing; `out` may then
    /// be half drawn and is to be thrown away.
    pub fn decorate_cube(
        &self,
        cube: SectionPos,
        out: &mut [u32; 4096],
        decor: FarDecor,
    ) -> Result<bool, NotReady> {
        if decor.is_off() || !self.land_cover_active() {
            return Ok(false);
        }
        let r = MAX_RADIUS;
        let (x0, y0, z0) = (cube.x * 16, cube.y * 16, cube.z * 16);
        let fill = self.classify_box_above(
            [x0 - r, y0, z0 - r],
            [x0 + 15 + r, y0 + 15, z0 + 15 + r],
            1,
        )?;
        if fill != Fill::Mixed {
            return Ok(false);
        }
        let w = (16 + 2 * r) as usize;
        let cols = self.columns_wide(x0 - r, z0 - r, w, 1, 1)?;
        let biomes = self.column_biomes(&cols)?;
        let b = *self.blocks();
        let sea = self.sea_level();
        let area = |c: usize| ((c >> 4) + r as usize) * w + (c & 15) + r as usize;
        // The highest block of each own column that is ground, water or tree, and whether a tree made it.
        let mut top = [0i32; 256];
        let mut tree = [false; 256];
        let mut wet = [false; 256];
        for c in 0..256 {
            let a = area(c);
            let (s, kind) = (cols.surfaces[a], cols.kind(a));
            wet[c] = self.water_top(kind, s);
            top[c] = if kind == SurfaceKind::Sand && s < sea {
                sea - 1
            } else {
                s
            };
        }
        let mut changed = false;
        if decor.trees {
            let trees = self.tree_sites(&cols, &biomes);
            for log_pass in [false, true] {
                for t in &trees {
                    let mut put = |x: i32, y: i32, z: i32, block: u32, log: bool| {
                        let (lx, lz) = (x - x0, z - z0);
                        if !(0..16).contains(&lx) || !(0..16).contains(&lz) {
                            return;
                        }
                        let c = (lz * 16 + lx) as usize;
                        if y > top[c] {
                            top[c] = y;
                            tree[c] = true;
                        }
                        if !(y0..y0 + 16).contains(&y) {
                            return;
                        }
                        let idx = ((y - y0) as usize) << 8 | c;
                        if out[idx] == b.air || (log && b.far.is_leaves(out[idx])) {
                            out[idx] = block;
                            changed = true;
                        }
                    };
                    if log_pass {
                        let log = b.far.logs[t.species as usize];
                        for y in t.s + 1..=t.s + t.h {
                            put(t.x, y, t.z, log, true);
                        }
                    } else {
                        let leaves = b.far.leaves[t.species as usize];
                        t.leaves(|x, y, z| put(x, y, z, leaves, false));
                    }
                }
            }
        }
        if decor.snow {
            let mut set = |c: usize, y: i32, block: u32, onto: Option<u32>| {
                if !(y0..y0 + 16).contains(&y) {
                    return;
                }
                let idx = ((y - y0) as usize) << 8 | c;
                if onto.is_none_or(|o| out[idx] == o) {
                    out[idx] = block;
                    changed = true;
                }
            };
            for c in 0..256 {
                let a = area(c);
                if !biomes[a].snows() {
                    continue;
                }
                if !tree[c] && wet[c] {
                    set(c, top[c], b.far.ice, Some(b.water));
                    continue;
                }
                let s = cols.surfaces[a];
                let ground = self.block_for(s, s, cols.kind(a));
                if !tree[c] && (ground == b.snow_block || ground == b.air) {
                    continue;
                }
                set(c, top[c] + 1, b.far.snow, Some(b.air));
                if !tree[c] && ground == b.grass_block {
                    set(c, top[c], b.far.snowy_grass, Some(b.grass_block));
                }
            }
        }
        Ok(changed)
    }

    /// Makes the trees and snow of the columns of a far node of levels 2 and up (see the module docs), to be
    /// read by [`EarthGenerator::far_voxel`]. Gives [`NotReady`] when a biome is missing.
    pub fn decorate_columns(
        &self,
        cols: &mut NodeColumns,
        decor: FarDecor,
    ) -> Result<(), NotReady> {
        let Some(classes) = cols.classes.clone() else {
            return Ok(());
        };
        if decor.is_off() {
            return Ok(());
        }
        let biomes = self.column_biomes(cols)?;
        let step = cols.step;
        let level = step.max(1).ilog2();
        let sea = self.sea_level();
        let floor = |y: i32| Self::voxel_floor(y, step);
        let b = *self.blocks();
        let mut out = Vec::with_capacity(biomes.len());
        for (i, &biome) in biomes.iter().enumerate() {
            let (x, z) = cols.block_of(i);
            let (s, kind) = (cols.surfaces[i], cols.kind(i));
            let mut c = ColDecor {
                snow: decor.snow && biome.snows(),
                wet: self.water_top(kind, s),
                ..ColDecor::default()
            };
            let ground_top = if kind == SurfaceKind::Sand && s < sea {
                sea - 1
            } else {
                s
            };
            c.top_lo = floor(ground_top);
            let styled = style(biome).filter(|_| {
                decor.trees
                    && matches!(
                        kind,
                        SurfaceKind::Grass
                            | SurfaceKind::Scrub(_)
                            | SurfaceKind::Moss
                            | SurfaceKind::Marsh
                    )
            });
            if let Some(style) = styled {
                let cover = tree_share(classes[i]) * style.cover();
                let h = mix(x, z, 0xFA57_0000 + u64::from(level));
                let covered = if step >= SOLID_FROM_STEP {
                    cover >= SOLID_COVER
                } else {
                    unit(h) < cover
                };
                if covered {
                    let (species, height) = pick(&style, h);
                    let (bottom, top) = crown(&style, s, height);
                    c.leaves = b.far.leaves[species as usize];
                    (c.bottom, c.top) = (bottom, top);
                    c.white = h >> 7 & 1 == 1;
                    // The highest voxel over the ground voxel whose middle the canopy covers.
                    let half = step / 2;
                    let lo = floor(top - half);
                    if lo > floor(s) && lo + half >= bottom {
                        c.top_lo = lo;
                    } else {
                        c.recolour = true;
                    }
                }
            }
            c.wet &= c.leaves == 0;
            out.push(c);
        }
        cols.decor = Some(Box::new(ColumnsDecor { cols: out }));
        Ok(())
    }

    /// The voxel of column `i` of `cols` whose lowest block is `lo`, `block` being the terrain's: with the
    /// canopy and the snow of its [`ColDecor`].
    pub(super) fn decorated_voxel(
        &self,
        cols: &NodeColumns,
        decor: &ColumnsDecor,
        i: usize,
        lo: i32,
        block: u32,
    ) -> u32 {
        let c = &decor.cols[i];
        let b = self.blocks();
        let step = cols.step;
        let mut block = block;
        if c.leaves != 0 {
            let s = cols.surfaces[i];
            if block == b.air {
                let mid = i64::from(lo) + i64::from(step / 2);
                if i64::from(c.bottom) <= mid && mid <= i64::from(c.top) {
                    block = c.leaves;
                }
            } else if c.recolour && lo <= s && i64::from(s) < i64::from(lo) + i64::from(step) {
                block = c.leaves;
            }
        }
        if c.snow && lo == c.top_lo && block != b.air {
            block = if c.wet {
                b.far.ice
            } else if block == c.leaves && c.white {
                b.snow_block
            } else if block == c.leaves {
                block
            } else {
                b.snow_block
            };
        }
        block
    }
}

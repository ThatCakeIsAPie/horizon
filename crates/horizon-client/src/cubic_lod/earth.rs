//! The far view of an Earth world.
//!
//! An Earth node is made from the elevation tiles alone, in three steps that never wait for a download:
//!
//! 1. the generator's classification of the whole node ([`EarthGenerator::classify_node`]): sky and sea, and rock
//!    deep under the ground, are uniform nodes that cost no sampling (and the planner prunes them without asking
//!    for them: [`class`]);
//! 2. otherwise the 16 by 16 columns' surfaces in one batched read ([`EarthGenerator::node_columns`]), and a voxel
//!    for each height from them ([`EarthGenerator::far_voxel`]);
//! 3. its sky cover from the same surfaces ([`cover`]).
//!
//! With trees on (the world's vegetation) the nodes also get trees and, with snow on, snow and ice
//! ([`horizon_world::cubic::earth::far_trees`]); they read the class and the biome of their columns, which are
//! "later" like a tile when missing. A node is air only from the height trees reach ([`class`]).
//!
//! When a tile is missing each of these says so instead of inventing a height. The sampler then asks the
//! generator to have the tile fetched ([`EarthGenerator::request_tiles`], taken by the prefetcher) and answers
//! "later" (`None`); a node is never built, drawn or kept without its data.

use super::refine::Pass;
use super::*;
use crate::lod::cube::{FluidTops, NO_FLUID_TOP, VOLUME};
use horizon_world::cubic::earth::far_trees::FarDecor;
use horizon_world::cubic::earth::{EarthGenerator, Fill, NotReady};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// What the Earth far view has done so far: classifications that pruned air, called rock or left a node mixed;
/// nodes built uniform from the classification, from their surfaces, or as cubes; and nodes (or classifications)
/// answered "later" for a missing tile. With `HORIZON_FAR_PROFILE=1` the totals are logged as they pass each
/// multiple of 4,096 nodes.
static CLASSED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static BUILT: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static LATER: AtomicU64 = AtomicU64::new(0);

fn log_stats() {
    if !crate::lod::cube::profile::on() {
        return;
    }
    let n = |c: &AtomicU64| c.load(Relaxed);
    let built = n(&BUILT[0]) + n(&BUILT[1]) + n(&BUILT[2]);
    if built == 0 || built % 4096 != 0 {
        return;
    }
    log::info!(
        "Horizon Earth far view so far: classified air {} rock {} mixed {}; built uniform {} from surfaces {} as cubes {}; later {}",
        n(&CLASSED[0]),
        n(&CLASSED[1]),
        n(&CLASSED[2]),
        n(&BUILT[0]),
        n(&BUILT[1]),
        n(&BUILT[2]),
        n(&LATER),
    );
}

/// Whether the far view of `level` is made in two passes, coarse first and whole later (an Earth world with
/// far trees; `HORIZON_FAR_TWO_PASS=0` turns it off to compare).
pub(super) fn two_pass(level: &CubicLevel) -> bool {
    let g = level.generation();
    g.kind == CubicKind::Earth
        && g.earth
            .as_ref()
            .is_some_and(|p| p.settings.land_cover && p.settings.vegetation)
        && crate::lod::cube::far_knob("HORIZON_FAR_TWO_PASS", 1) != 0
}

/// The class of node `key` from the elevation bounds alone (and, with trees, the height they reach): air and
/// water are pruned, stone is a closed shell, the rest is unknown. Levels below 2 are made from cubes and are always unknown.
pub(super) fn class(
    earth: &EarthGenerator,
    key: CubeKey,
    decor: FarDecor,
) -> Result<NodeClass, NotReady> {
    if key.level < 2 {
        return Ok(NodeClass::Mixed);
    }
    // With trees, air is only proven from the height they reach.
    let fill = if decor.trees {
        earth.classify_node_above(u32::from(key.level), key.x, key.y, key.z)
    } else {
        earth.classify_node(u32::from(key.level), key.x, key.y, key.z)
    };
    let class = match fill? {
        Fill::Air | Fill::Water => NodeClass::Air,
        Fill::Stone => NodeClass::Rock,
        Fill::Mixed => NodeClass::Mixed,
    };
    let slot = match class {
        NodeClass::Air => 0,
        NodeClass::Rock => 1,
        NodeClass::Mixed => 2,
    };
    CLASSED[slot].fetch_add(1, Relaxed);
    Ok(class)
}

/// The surfaces of the 16 by 16 columns of node `key` (the voxels' middles): level 0 and 1 at the base zoom (they
/// are made of cubes), the others at the zoom of their step. Only heights, so land cover is not waited for.
fn surfaces(earth: &EarthGenerator, key: CubeKey) -> Result<Vec<i32>, NotReady> {
    let origin = key.origin();
    match key.level {
        0 | 1 => earth.surfaces_at(origin[0], origin[2], 1 << key.level, 1),
        level => earth.surfaces_at(origin[0], origin[2], 1 << level, 1 << level),
    }
}

/// The sky cover of node `key`: a column is covered when ground reaches over the node's top, as
/// `cover_of` finds for the other worlds by probing `block_at` above it. `None` for the levels that are drawn
/// lit.
pub(super) fn cover(
    earth: &EarthGenerator,
    key: CubeKey,
) -> Result<Option<Box<CoverMask>>, NotReady> {
    if key.level > COVER_LEVEL {
        return Ok(None);
    }
    Ok(cover_of(key, &surfaces(earth, key)?))
}

/// [`cover`] from the node's column surfaces.
fn cover_of(key: CubeKey, surfaces: &[i32]) -> Option<Box<CoverMask>> {
    if key.level > COVER_LEVEL {
        return None;
    }
    let step = 1i32 << key.level;
    let top = key.origin()[1].saturating_add(key.size());
    // The lowest look over the node is the voxel at `top + COVER_LOOKS[0]`; ground reaching it covers the rest.
    let look = EarthGenerator::voxel_floor(top.saturating_add(COVER_LOOKS[0]), step);
    let mut mask = Box::new([0u64; 4]);
    for (i, &s) in surfaces.iter().enumerate() {
        if s >= look {
            mask[i >> 6] |= 1 << (i & 63);
        }
    }
    Some(mask)
}

/// `surfaces` with the sea's top where the sea is: without land cover they are the elevation's, under the sea
/// level in the sea, whose water reaches `sea_level - 1` ([`horizon_world::cubic::earth::SurfaceKind::sea_column`]).
/// With land cover they are already the final surfaces.
fn water_surfaces(earth: &EarthGenerator, surfaces: &[i32]) -> Vec<i32> {
    if earth.land_cover_active() {
        return surfaces.to_vec();
    }
    let top = earth.sea_level().saturating_sub(1);
    surfaces.iter().map(|&s| s.max(top)).collect()
}

/// Where the water of each column of `node` ends inside it, from the columns' final `surfaces` (the top water
/// block of a water column): its top fluid voxel's surface is drawn at the water's own level, also when that is the
/// node's top. Earth's sea tops at `sea_level - 1`, on the top of a node at every level, and a lake or river whose
/// top lies on a node's top is as common; without this the mesher took their water to go on in the node above,
/// drew no top, and showed the bed (sand, gravel) through it. A column whose water goes on over the node, or
/// with ice or anything over its top fluid voxel, has none. `fluid` says which states are fluids.
pub(super) fn fluid_tops(
    node: &CubeNode,
    surfaces: &[i32],
    fluid: impl Fn(BlockStateId) -> bool,
) -> Option<Box<FluidTops>> {
    let size = node.key.size();
    let step = i64::from(size / 16);
    let bottom = i64::from(node.key.origin()[1]);
    let mut tops = Box::new([NO_FLUID_TOP; 256]);
    let mut any = false;
    for (i, &s) in surfaces.iter().enumerate().take(256) {
        let (x, z) = (i & 15, i >> 4);
        let Some(y) = (0..16).rev().find(|&y| fluid(node.get(x, y, z).state)) else {
            continue;
        };
        let (lo, hi) = (y as i64 * step, (y as i64 + 1) * step);
        let h = i64::from(s) + 1 - bottom;
        if h > hi {
            continue;
        }
        // A surface under the fluid voxel (a coarse voxel that came out fluid all the same) is its top.
        let h = if h > lo { h } else { hi };
        tops[i] = CubeNode::fluid_top_units(h as f64, size);
        any = true;
    }
    any.then_some(tops)
}

/// Node `key` (without its sky cover) as the generator makes it, with the trees and snow `decor` asks for.
/// `biome` is the biome id of a block column; `kind` how downsampling counts a state.
pub(super) fn node(
    earth: &EarthGenerator,
    key: CubeKey,
    air: u32,
    biome: &dyn Fn(i32, i32) -> u16,
    kind: &dyn Fn(BlockStateId) -> Kind,
    decor: FarDecor,
) -> Result<CubeNode, NotReady> {
    let origin = key.origin();
    // With land cover the biomes read the map at a coarse level: have them before building, never wait for them.
    match key.level {
        0 => earth.biomes_ready(origin[0], origin[2], 1)?,
        1 => {}
        level => {
            // The columns' middles, and the node's own corner for a node that is one voxel.
            let step = 1i32 << level;
            earth.biomes_ready(origin[0] + step / 2, origin[2] + step / 2, step)?;
            earth.biomes_ready(origin[0], origin[2], step)?;
        }
    }
    let state = |id: u32| BlockStateId(if id == 0 { air } else { id });
    let uniform = |id: u32| {
        CubeNode::uniform(
            key,
            Voxel {
                state: state(id),
                biome: biome(origin[0], origin[2]),
            },
        )
    };
    match key.level {
        0 => {
            let mut blocks = Box::new([0u32; 4096]);
            let pos = SectionPos::new(key.x, key.y, key.z);
            let mut whole = earth.generate(pos, &mut blocks)?;
            // Trees (and snow) can stand in a cube of sky or of one block.
            if earth.decorate_cube(pos, &mut blocks, decor)? {
                whole = None;
            }
            if let Some(id) = whole {
                BUILT[0].fetch_add(1, Relaxed);
                return Ok(uniform(id));
            }
            BUILT[2].fetch_add(1, Relaxed);
            let biomes: Vec<u16> = (0..256)
                .map(|i| biome(origin[0] + (i & 15), origin[2] + (i >> 4)))
                .collect();
            Ok(CubeNode::from_fn(key, |x, y, z| Voxel {
                state: state(blocks[y << 8 | z << 4 | x]),
                biome: biomes[z << 4 | x],
            }))
        }
        1 => {
            // Eight exact cubes, downsampled: a node this near is made as the saved ones are.
            let mut nodes = Vec::with_capacity(8);
            for child in key.children() {
                nodes.push(node(earth, child, air, biome, kind, decor)?);
            }
            let refs: [&CubeNode; 8] = std::array::from_fn(|i| &nodes[i]);
            Ok(downsample(refs, BlockStateId(air), kind))
        }
        level => {
            let fill = if decor.trees {
                earth.classify_node_above(u32::from(level), key.x, key.y, key.z)?
            } else {
                earth.classify_node(u32::from(level), key.x, key.y, key.z)?
            };
            if let Some(id) = earth.fill_block(fill) {
                BUILT[0].fetch_add(1, Relaxed);
                return Ok(uniform(id));
            }
            BUILT[1].fetch_add(1, Relaxed);
            let mut cols = earth.node_columns(u32::from(level), key.x, key.z)?;
            earth.decorate_columns(&mut cols, decor)?;
            let step = 1i32 << level;
            let biomes: Vec<u16> = (0..256)
                .map(|i| {
                    biome(
                        origin[0] + (i & 15) * step + step / 2,
                        origin[2] + (i >> 4) * step + step / 2,
                    )
                })
                .collect();
            Ok(CubeNode::from_fn(key, |x, y, z| {
                let column = z << 4 | x;
                let lo = origin[1].saturating_add(y as i32 * step);
                Voxel {
                    state: state(earth.far_voxel(&cols, column, lo)),
                    biome: biomes[column],
                }
            }))
        }
    }
}

impl CubicLodSampler {
    /// Whether this is an Earth world's sampler.
    pub(super) fn is_earth(&self) -> bool {
        self.wg.kind == CubicKind::Earth
    }

    /// `got`, or `None` after asking for the tiles it lacks.
    fn or_request<T>(&self, got: Result<T, NotReady>) -> Option<T> {
        match got {
            Ok(value) => Some(value),
            Err(missing) => {
                LATER.fetch_add(1, Relaxed);
                if let Some(earth) = self.wg.generator.earth_generator() {
                    earth.request_tiles(&missing.tiles);
                }
                None
            }
        }
    }

    /// What the far view draws over the terrain: trees where the world has vegetation (it has land cover then),
    /// snow where it also has snow. The settings in force, not the ones asked for (see `Generation`).
    fn far_decor(&self) -> FarDecor {
        let Some(parts) = &self.wg.earth else {
            return FarDecor::default();
        };
        let trees = parts.settings.land_cover && parts.settings.vegetation;
        FarDecor {
            trees,
            snow: trees && parts.settings.snow,
        }
    }

    /// Node `key` of an Earth world without its sky cover; `None` (later) while a tile it reads is missing.
    /// With two passes ([`two_pass`]) a node is made coarse first, without its trees (as fast as the far view
    /// was before them), and whole when [`super::refine::Refine`] hands it out behind the rest.
    pub(super) fn earth_plain(&mut self, key: CubeKey) -> Option<CubeNode> {
        let generator = self.wg.generator.clone();
        let earth = generator.earth_generator()?;
        let level = self.level.clone();
        let biome = |x: i32, z: i32| level.biome_id_at(x, z);
        let decor = self.far_decor();
        let pass = match &self.refine {
            Some(refine) if decor.trees => refine.pass(key),
            _ => Pass::Whole,
        };
        let made_with = match pass {
            Pass::Coarse => FarDecor {
                trees: false,
                ..decor
            },
            Pass::Whole => decor,
        };
        let made = node(earth, key, self.air, &biome, &|s| self.kind(s), made_with);
        if decor.trees
            && let Some(refine) = &self.refine
        {
            refine.made(key, pass == Pass::Whole, made.is_ok());
        }
        log_stats();
        self.or_request(made)
    }

    /// Node `key` was read whole from the far cache.
    pub(super) fn made_whole(&self, key: CubeKey) {
        if let Some(refine) = &self.refine {
            refine.made(key, true, true);
        }
    }

    /// Whether node `key` was last made coarse (it is refined later, and not kept meanwhile).
    pub(super) fn is_coarse(&self, key: CubeKey) -> bool {
        self.is_earth()
            && self.far_decor().trees
            && self
                .refine
                .as_ref()
                .is_some_and(|r| r.pass(key) == Pass::Coarse)
    }

    /// `node` (Earth's, of `key`) with its sky cover, or `None` (later) when the tiles it reads are missing. A
    /// node that is all air draws nothing and has no use for one.
    pub(super) fn earth_covered(&self, key: CubeKey, node: CubeNode) -> Option<CubeNode> {
        if node.as_uniform().is_some_and(|v| v.state.0 == self.air) {
            return Some(node);
        }
        let generator = self.wg.generator.clone();
        let earth = generator.earth_generator()?;
        let fluid = |s: BlockStateId| self.kind(s) == Kind::Fluid;
        let wet = match node.as_uniform() {
            Some(v) => fluid(v.state),
            None => (0..VOLUME).any(|i| fluid(node.at(i).state)),
        };
        if key.level > COVER_LEVEL && !wet {
            return Some(node);
        }
        let surfaces = self.or_request(surfaces(earth, key))?;
        let tops = if wet {
            fluid_tops(&node, &water_surfaces(earth, &surfaces), fluid)
        } else {
            None
        };
        Some(
            node.with_covered(cover_of(key, &surfaces))
                .with_fluid_tops(tops),
        )
    }

    /// [`CubeSampler::classify`] of an Earth node: `None` (ask again) while a tile it reads is missing.
    pub(super) fn earth_class(&self, key: CubeKey) -> Option<NodeClass> {
        let generator = self.wg.generator.clone();
        let earth = generator.earth_generator()?;
        self.or_request(class(earth, key, self.far_decor()))
    }
}

#[cfg(test)]
mod tests;

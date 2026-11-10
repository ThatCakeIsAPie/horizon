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
//! When a tile is missing each of these says so instead of inventing a height. The sampler then asks the
//! generator to have the tile fetched ([`EarthGenerator::request_tiles`], taken by the prefetcher) and answers
//! "later" (`None`); a node is never built, drawn or kept without its data.

use super::*;
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

/// The class of node `key` from the elevation bounds alone: air and water are pruned, stone is a closed shell, the
/// rest is unknown. Levels below 2 are made from cubes and are always unknown.
pub(super) fn class(earth: &EarthGenerator, key: CubeKey) -> Result<NodeClass, NotReady> {
    if key.level < 2 {
        return Ok(NodeClass::Mixed);
    }
    let class = match earth.classify_node(u32::from(key.level), key.x, key.y, key.z)? {
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

/// The columns of node `key` (16 by 16, the voxels' middles): level 0 and 1 at the base zoom (they are made of
/// cubes), the others at the zoom of their step.
fn columns(
    earth: &EarthGenerator,
    key: CubeKey,
) -> Result<horizon_world::cubic::earth::far::NodeColumns, NotReady> {
    let origin = key.origin();
    match key.level {
        0 | 1 => earth.columns_at(origin[0], origin[2], 1 << key.level, 1),
        level => earth.node_columns(u32::from(level), key.x, key.z),
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
    let cols = columns(earth, key)?;
    let step = 1i32 << key.level;
    let top = key.origin()[1].saturating_add(key.size());
    // The lowest look over the node is the voxel at `top + COVER_LOOKS[0]`; ground reaching it covers the rest.
    let look = EarthGenerator::voxel_floor(top.saturating_add(COVER_LOOKS[0]), step);
    let mut mask = Box::new([0u64; 4]);
    for (i, &s) in cols.surfaces.iter().enumerate() {
        if s >= look {
            mask[i >> 6] |= 1 << (i & 63);
        }
    }
    Ok(Some(mask))
}

/// Node `key` (without its sky cover) as the generator makes it. `biome` is the biome id of a block column;
/// `kind` how downsampling counts a state.
pub(super) fn node(
    earth: &EarthGenerator,
    key: CubeKey,
    air: u32,
    biome: &dyn Fn(i32, i32) -> u16,
    kind: &dyn Fn(BlockStateId) -> Kind,
) -> Result<CubeNode, NotReady> {
    let origin = key.origin();
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
            if let Some(id) = earth.generate(pos, &mut blocks)? {
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
                nodes.push(node(earth, child, air, biome, kind)?);
            }
            let refs: [&CubeNode; 8] = std::array::from_fn(|i| &nodes[i]);
            Ok(downsample(refs, BlockStateId(air), kind))
        }
        level => {
            if let Some(id) =
                earth.fill_block(earth.classify_node(u32::from(level), key.x, key.y, key.z)?)
            {
                BUILT[0].fetch_add(1, Relaxed);
                return Ok(uniform(id));
            }
            BUILT[1].fetch_add(1, Relaxed);
            let cols = earth.node_columns(u32::from(level), key.x, key.z)?;
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

    /// Node `key` of an Earth world without its sky cover; `None` (later) while a tile it reads is missing.
    pub(super) fn earth_plain(&mut self, key: CubeKey) -> Option<CubeNode> {
        let generator = self.wg.generator.clone();
        let earth = generator.earth_generator()?;
        let level = self.level.clone();
        let biome = |x: i32, z: i32| level.biome_id_at(x, z);
        let made = node(earth, key, self.air, &biome, &|s| self.kind(s));
        log_stats();
        self.or_request(made)
    }

    /// `node` (Earth's, of `key`) with its sky cover, or `None` (later) when the tiles it reads are missing. A
    /// node that is all air draws nothing and has no use for one.
    pub(super) fn earth_covered(&self, key: CubeKey, node: CubeNode) -> Option<CubeNode> {
        if node.as_uniform().is_some_and(|v| v.state.0 == self.air) {
            return Some(node);
        }
        let generator = self.wg.generator.clone();
        let earth = generator.earth_generator()?;
        let cover = self.or_request(cover(earth, key))?;
        Some(node.with_covered(cover))
    }

    /// [`CubeSampler::classify`] of an Earth node: `None` (ask again) while a tile it reads is missing.
    pub(super) fn earth_class(&self, key: CubeKey) -> Option<NodeClass> {
        let generator = self.wg.generator.clone();
        let earth = generator.earth_generator()?;
        self.or_request(class(earth, key))
    }
}

#[cfg(test)]
mod tests;

//! Far terrain of a cubic level: an octree of cube nodes, "chunks of chunks".
//!
//! A node [`CubeKey`] `{ level, x, y, z }` covers `(16 << level)³` blocks at
//! `[x, y, z] * (16 << level)` and holds a 16³ grid of voxels, each
//! `1 << level` blocks wide. Level 0 is one cube at full resolution. A parent
//! is built from its eight children ([`downsample`]) or sampled at its own
//! resolution by a [`CubeSampler`]. Nodes are meshed as closed shells
//! (`mesh::mesh_cube`), so no crack opens between neighbours of different
//! levels, and drawn through the far-tile pipeline.
//!
//! [`plan_classified`] chooses the nodes (pure, no threads): a node splits
//! while its voxels would span more than the Cubic Far Detail setting's pixels
//! on screen ([`FarView`], [`Rule`]), within a node budget. [`CubeLod`] plans
//! on a thread of its own, asks for a coarse coverage layer first after a big
//! change, queues the rest by direction (the view's first) and distance, and
//! samples and meshes the nodes in the background. [`node_kind`] is where far
//! nodes will become 2D height-map tiles.
use super::mesh::{self, Mesher};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
    gfx::{
        cube_descent::{DescentRequest, GpuDescent},
        cube_gen::{GpuBackend, MIN_LEVEL, slot_words},
        surface_tile::{TileInfo, TileKey},
    },
    mesher::SectionMesh,
    streaming::WorkerContext,
};
use horizon_core::position::SectionPos;
use horizon_world::registry::BlockStateId;
use std::{
    cell::RefCell,
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

/// The coarsest level: a node 67,108,864 blocks wide, wider than the world
/// (vanilla's border, +-30,000,000 blocks) and than the cube range vertically.
pub const MAX_CUBE_LEVEL: u8 = 22;
/// Voxels along one side of a node.
pub const SIDE: usize = 16;
/// Voxels in a node.
pub const VOLUME: usize = SIDE * SIDE * SIDE;

/// One node of the octree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CubeKey {
    pub level: u8,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CubeKey {
    pub const fn new(level: u8, x: i32, y: i32, z: i32) -> Self {
        Self { level, x, y, z }
    }
    /// The node of `level` holding `cube`.
    pub const fn containing(level: u8, cube: SectionPos) -> Self {
        Self::new(level, cube.x >> level, cube.y >> level, cube.z >> level)
    }
    /// Blocks along one side.
    pub const fn size(self) -> i32 {
        16 << self.level
    }
    /// The node's lowest corner, in blocks.
    pub const fn origin(self) -> [i32; 3] {
        let size = self.size();
        [self.x * size, self.y * size, self.z * size]
    }
    /// The node whose extent is exactly the box `min..=max` of blocks, if one is.
    pub fn of_box(min: [i32; 3], max: [i32; 3]) -> Option<Self> {
        let size = i64::from(max[0]) - i64::from(min[0]) + 1;
        let level = (0..=MAX_CUBE_LEVEL).find(|&l| i64::from(16i32 << l) == size)?;
        let key = Self::new(
            level,
            min[0].div_euclid(16 << level),
            min[1].div_euclid(16 << level),
            min[2].div_euclid(16 << level),
        );
        let end = key.origin().map(|o| i64::from(o) + size - 1);
        (key.origin() == min && end == max.map(i64::from)).then_some(key)
    }
    /// The node one level up that holds this one (rounding toward negative
    /// infinity, so negative coordinates nest too).
    pub const fn parent(self) -> Self {
        Self::new(self.level + 1, self.x >> 1, self.y >> 1, self.z >> 1)
    }
    /// The eight nodes one level down. Index bit 0 is the X half, bit 1 the
    /// Z half, bit 2 the Y half.
    pub fn children(self) -> [Self; 8] {
        std::array::from_fn(|i| {
            Self::new(
                self.level - 1,
                self.x * 2 + (i & 1) as i32,
                self.y * 2 + (i >> 2) as i32,
                self.z * 2 + ((i >> 1) & 1) as i32,
            )
        })
    }
    /// Whether `cube` (a level-0 node) lies inside this node.
    pub const fn contains_cube(self, cube: SectionPos) -> bool {
        cube.x >> self.level == self.x
            && cube.y >> self.level == self.y
            && cube.z >> self.level == self.z
    }
    /// Whether `other` is this node or inside it.
    pub const fn contains(self, other: Self) -> bool {
        other.level <= self.level
            && other.x >> (self.level - other.level) == self.x
            && other.y >> (self.level - other.level) == self.y
            && other.z >> (self.level - other.level) == self.z
    }
    /// Distance in blocks from `point` to the nearest point of the node.
    pub fn distance(self, point: [f64; 3]) -> f64 {
        let size = f64::from(self.size());
        let mut sum = 0.0;
        for (c, o) in point.into_iter().zip(self.origin()) {
            let lo = f64::from(o);
            let gap = (lo - c).max(c - (lo + size)).max(0.0);
            sum += gap * gap;
        }
        sum.sqrt()
    }
}

/// One voxel: the block state standing in for `1 << level` blocks cubed, and
/// its biome (for tints).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Voxel {
    pub state: BlockStateId,
    pub biome: u16,
}

#[derive(Clone, Debug)]
enum Voxels {
    Uniform(Voxel),
    Dense(Box<[Voxel; VOLUME]>),
}

/// A node's voxels, indexed `y << 8 | z << 4 | x`. A node of one voxel
/// repeated (air, solid rock) stores just that voxel.
#[derive(Clone, Debug)]
pub struct CubeNode {
    pub key: CubeKey,
    voxels: Voxels,
    /// Columns (`z << 4 | x`) whose sky is blocked above the node: a set bit
    /// means something solid stands over the node there (an island, the
    /// vanilla terrain). `None`: all open. See [`CubeNode::column_covered`].
    covered: Option<Box<CoverMask>>,
    /// Where each column's fluid ends inside the node (`z << 4 | x`): the
    /// height of its surface, so a lake, river or sea whose top lies on the
    /// node's top is drawn there, and one inside a coarse voxel at its own
    /// level. `None`: not known, a fluid's top on the node's top is taken to
    /// go on in the node above. See [`CubeNode::fluid_top`].
    fluid_tops: Option<Box<FluidTops>>,
}

/// One bit per column of a node, `z << 4 | x`.
pub type CoverMask = [u64; 4];

/// Per column of a node (`z << 4 | x`), the height its fluid's surface lies at
/// in the node, in `1 / FLUID_TOP_SCALE` of the node's side (1 to
/// `FLUID_TOP_SCALE`, the node's top), or [`NO_FLUID_TOP`]: no fluid, or a
/// fluid going on over the node.
pub type FluidTops = [u16; 256];
/// A column of [`FluidTops`] without a surface in the node.
pub const NO_FLUID_TOP: u16 = u16::MAX;
/// The units of a [`FluidTops`] height per node side.
pub const FLUID_TOP_SCALE: u16 = 4096;

impl CubeNode {
    /// A hash of everything a node draws from: its voxels, sky cover and
    /// fluid tops (not its key).
    pub fn content_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::hash::DefaultHasher::new();
        match &self.voxels {
            Voxels::Uniform(v) => (0u8, v).hash(&mut h),
            Voxels::Dense(d) => (1u8, &d[..]).hash(&mut h),
        }
        self.covered.hash(&mut h);
        self.fluid_tops.as_deref().map(|t| &t[..]).hash(&mut h);
        h.finish()
    }
    pub fn uniform(key: CubeKey, voxel: Voxel) -> Self {
        Self {
            key,
            voxels: Voxels::Uniform(voxel),
            covered: None,
            fluid_tops: None,
        }
    }
    /// Voxels from `f(x, y, z)` (each 0..16); a node whose voxels all agree
    /// is stored as one.
    pub fn from_fn(key: CubeKey, mut f: impl FnMut(usize, usize, usize) -> Voxel) -> Self {
        let mut dense = Box::new([f(0, 0, 0); VOLUME]);
        for i in 1..VOLUME {
            dense[i] = f(i & 15, i >> 8, (i >> 4) & 15);
        }
        Self::dense(key, dense)
    }
    /// A node of exactly these voxels (`y << 8 | z << 4 | x`).
    pub fn from_dense(key: CubeKey, dense: Box<[Voxel; VOLUME]>) -> Self {
        Self::dense(key, dense)
    }
    fn dense(key: CubeKey, dense: Box<[Voxel; VOLUME]>) -> Self {
        let first = dense[0];
        if dense.iter().all(|&v| v == first) {
            Self::uniform(key, first)
        } else {
            Self {
                key,
                voxels: Voxels::Dense(dense),
                covered: None,
                fluid_tops: None,
            }
        }
    }
    /// The node with its covered columns set (an empty mask is none).
    pub fn with_covered(mut self, covered: Option<Box<CoverMask>>) -> Self {
        self.covered = covered.filter(|mask| mask.iter().any(|&w| w != 0));
        self
    }
    pub fn covered_mask(&self) -> Option<&CoverMask> {
        self.covered.as_deref()
    }
    /// Whether the sky over the node is blocked at column (`x`, `z`).
    pub fn column_covered(&self, x: usize, z: usize) -> bool {
        let bit = z << 4 | x;
        self.covered
            .as_ref()
            .is_some_and(|mask| mask[bit >> 6] >> (bit & 63) & 1 == 1)
    }
    /// The node with its columns' fluid surfaces set (none set is `None`).
    pub fn with_fluid_tops(mut self, tops: Option<Box<FluidTops>>) -> Self {
        self.fluid_tops = tops.filter(|t| t.iter().any(|&h| h != NO_FLUID_TOP));
        self
    }
    pub fn fluid_tops(&self) -> Option<&FluidTops> {
        self.fluid_tops.as_deref()
    }
    /// The height (blocks over the node's bottom) at which column (`x`, `z`)'s
    /// fluid ends inside the node: its surface, drawn there even on the node's
    /// top. `None` when unknown, or the fluid goes on over the node.
    pub fn fluid_top(&self, x: usize, z: usize) -> Option<f32> {
        let h = self.fluid_tops.as_ref()?[z << 4 | x];
        (h != NO_FLUID_TOP)
            .then(|| f32::from(h) * self.key.size() as f32 / f32::from(FLUID_TOP_SCALE))
    }
    /// The [`FluidTops`] height of a surface `blocks` over the bottom of a node
    /// of side `size`: at least one unit, at most the node's top.
    pub fn fluid_top_units(blocks: f64, size: i32) -> u16 {
        (blocks * f64::from(FLUID_TOP_SCALE) / f64::from(size))
            .round()
            .clamp(1.0, f64::from(FLUID_TOP_SCALE)) as u16
    }
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        y << 8 | z << 4 | x
    }
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        self.at(Self::index(x, y, z))
    }
    pub fn at(&self, index: usize) -> Voxel {
        match &self.voxels {
            Voxels::Uniform(v) => *v,
            Voxels::Dense(d) => d[index],
        }
    }
    /// The only voxel of the node, if it has just one kind.
    pub fn as_uniform(&self) -> Option<Voxel> {
        match &self.voxels {
            Voxels::Uniform(v) => Some(*v),
            Voxels::Dense(_) => None,
        }
    }
}

/// How a voxel's block state counts when voxels are merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Air,
    Fluid,
    Solid,
}

/// The node above eight children (in [`CubeKey::children`] order), each voxel
/// from its 2×2×2 child voxels: solid when at least 4 of 8 are solid (a tie
/// keeps solid), as the state seen most often among the solid ones, the
/// highest in Y winning a tie (grass over its dirt). Fluids do not count for
/// the majority; with no solid winner, at least 4 fluid voxels make a fluid
/// voxel. The rest is air.
pub fn downsample(
    children: [&CubeNode; 8],
    air: BlockStateId,
    kind: impl Fn(BlockStateId) -> Kind,
) -> CubeNode {
    let key = children[0].key.parent();
    // Eight equal uniform children: still uniform.
    if let Some(first) = children[0].as_uniform()
        && children.iter().all(|c| c.as_uniform() == Some(first))
    {
        return CubeNode::uniform(key, first)
            .with_covered(merged_cover(&children))
            .with_fluid_tops(merged_fluid_tops(&children));
    }
    let mut dense = Box::new(
        [Voxel {
            state: air,
            biome: 0,
        }; VOLUME],
    );
    for py in 0..SIDE {
        for pz in 0..SIDE {
            for px in 0..SIDE {
                let child = children[(px >> 3) | ((pz >> 3) << 1) | ((py >> 3) << 2)];
                let (cx, cy, cz) = ((px & 7) * 2, (py & 7) * 2, (pz & 7) * 2);
                let mut eight = [child.get(cx, cy, cz); 8];
                for (i, voxel) in eight.iter_mut().enumerate() {
                    *voxel = child.get(cx + (i & 1), cy + (i >> 2), cz + ((i >> 1) & 1));
                }
                dense[CubeNode::index(px, py, pz)] = merge_eight(&eight, air, &kind);
            }
        }
    }
    CubeNode::dense(key, dense)
        .with_covered(merged_cover(&children))
        .with_fluid_tops(merged_fluid_tops(&children))
}

/// The parent's fluid surfaces: over each parent column, the highest surface
/// of the four child columns under it in the upper children, or else in the
/// lower ones (a fluid going on through the upper children goes on over the
/// parent too).
fn merged_fluid_tops(children: &[&CubeNode; 8]) -> Option<Box<FluidTops>> {
    if children.iter().all(|c| c.fluid_tops.is_none()) {
        return None;
    }
    let mut tops = Box::new([NO_FLUID_TOP; 256]);
    for pz in 0..SIDE {
        for px in 0..SIDE {
            let quarter = (px >> 3) | ((pz >> 3) << 1);
            let (cx, cz) = ((px & 7) * 2, (pz & 7) * 2);
            for (half, lift) in [(4, FLUID_TOP_SCALE), (0, 0)] {
                let Some(child) = children[half | quarter].fluid_tops.as_deref() else {
                    continue;
                };
                let highest = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .into_iter()
                    .map(|(dx, dz)| child[(cz + dz) << 4 | (cx + dx)])
                    .filter(|&h| h != NO_FLUID_TOP)
                    .max();
                if let Some(h) = highest {
                    tops[pz << 4 | px] = ((u32::from(h) + u32::from(lift)) / 2).max(1) as u16;
                    break;
                }
            }
        }
    }
    Some(tops)
}

/// The parent's covered columns: what covers the upper four children, a
/// parent column being covered when two of the four child columns under it
/// are.
fn merged_cover(children: &[&CubeNode; 8]) -> Option<Box<CoverMask>> {
    if children[4..].iter().all(|c| c.covered.is_none()) {
        return None;
    }
    let mut mask = Box::new([0u64; 4]);
    for pz in 0..SIDE {
        for px in 0..SIDE {
            let child = children[4 | (px >> 3) | ((pz >> 3) << 1)];
            let (cx, cz) = ((px & 7) * 2, (pz & 7) * 2);
            let covered = [(0, 0), (1, 0), (0, 1), (1, 1)]
                .into_iter()
                .filter(|&(dx, dz)| child.column_covered(cx + dx, cz + dz))
                .count();
            if covered >= 2 {
                let bit = pz << 4 | px;
                mask[bit >> 6] |= 1 << (bit & 63);
            }
        }
    }
    Some(mask)
}

/// `eight` is in `x | z << 1 | y << 2` order: the top four are the upper layer.
fn merge_eight(
    eight: &[Voxel; 8],
    air: BlockStateId,
    kind: &impl Fn(BlockStateId) -> Kind,
) -> Voxel {
    let kinds = eight.map(|v| kind(v.state));
    let count = |of: Kind| kinds.iter().filter(|&&k| k == of).count();
    let wanted = if count(Kind::Solid) >= 4 {
        Kind::Solid
    } else if count(Kind::Fluid) >= 4 {
        Kind::Fluid
    } else {
        return Voxel {
            state: air,
            biome: eight[0].biome,
        };
    };
    // (state, count, highest layer seen, the voxel there)
    let mut tally: smallvec::SmallVec<[(BlockStateId, u8, u8, Voxel); 8]> = Default::default();
    for (i, &voxel) in eight.iter().enumerate() {
        if kinds[i] != wanted {
            continue;
        }
        let layer = (i >> 2) as u8;
        match tally.iter_mut().find(|t| t.0 == voxel.state) {
            Some(t) => {
                t.1 += 1;
                if layer >= t.2 {
                    t.2 = layer;
                    t.3 = voxel;
                }
            }
            None => tally.push((voxel.state, 1, layer, voxel)),
        }
    }
    tally
        .into_iter()
        .max_by_key(|t| (t.1, t.2))
        .map(|t| t.3)
        .expect("a kind that was counted")
}

/// Where nodes come from. The game provides one backed by its generator.
pub trait CubeSampler: Send {
    /// The node at `key`'s own resolution (a voxel is `1 << level` blocks),
    /// or `None` when it cannot be had now. Coarse levels should be sampled
    /// directly: when a level above 3 answers `None`, the node is not built
    /// from its children.
    fn node(&mut self, key: CubeKey) -> Option<CubeNode>;
    /// As [`CubeSampler::node`] for the GPU path: a sampler that generates on
    /// the GPU answers [`Resident::Slot`] for a node it left in the GPU pool,
    /// never read back. Every other answers the node.
    fn resident(&mut self, key: CubeKey) -> Option<Resident> {
        self.node(key).map(Resident::Node)
    }
    /// [`CubeSampler::resident`] for many nodes at once, in order. A sampler
    /// whose generator batches (the GPU's) answers them in one request, so a
    /// worker keeps the GPU's batches full instead of waiting on one node.
    fn resident_many(&mut self, keys: &[CubeKey]) -> Vec<Option<Resident>> {
        keys.iter().map(|&key| self.resident(key)).collect()
    }
    /// What the sampler knows of the whole node without sampling it (the GPU
    /// generator's 2D pre-pass): lets the planner leave out subtrees of air
    /// and stop at rock. Called on a background thread; the default knows
    /// nothing.
    fn classify(&mut self, _key: CubeKey) -> NodeClass {
        NodeClass::Mixed
    }
    /// Whether [`CubeSampler::classify`] knows something worth asking: a sampler on the CPU path that does
    /// gets the classifier thread the GPU path always has. The default does not (it always says `Mixed`).
    fn classifies(&self) -> bool {
        false
    }
    /// [`CubeSampler::classify`], or `None` when the sampler cannot tell yet (the data it reads has not
    /// arrived): the node is then asked about again at a later plan instead of being filed as unknown.
    fn classify_ready(&mut self, key: CubeKey) -> Option<NodeClass> {
        Some(self.classify(key))
    }
    /// The GPU descent behind this sampler (`gfx::cube_descent`), if it has
    /// one: it proves whole subtrees of the octree uniform from the density
    /// bound, in place of [`CubeSampler::classify`] node by node
    /// ([`CubeLod::set_gpu_descent`]).
    fn descent(&self) -> Option<Arc<GpuDescent>> {
        None
    }
    /// The GPU generator behind this sampler (its device, node pool and block
    /// states), if any: what lets [`CubeLod`] mesh and draw on the GPU.
    fn gpu(&self) -> Option<GpuBackend> {
        None
    }
    /// An independent sampler for another thread, if this one can have one.
    /// Without it the producer runs a single worker.
    fn fork(&self) -> Option<Box<dyn CubeSampler>>;
    /// Whether the far terrain can be drawn as surface tiles at the coarse
    /// levels (`gfx::surface_tile`): a world whose terrain the tiles' generator
    /// reproduces (a Custom world without `cubeAreas`). False by default.
    fn surface(&self) -> bool {
        false
    }
    /// The sea's surface (the sea level + 1) when the sampler fills a voxel
    /// that holds it with water where the sea is open there: the CPU mesher
    /// draws such water up to the surface only (the GPU's reads the
    /// generator's own). `None` by default.
    fn sea_surface(&self) -> Option<i32> {
        None
    }
    /// Cubes that changed since this was last asked (the producer drops the
    /// nodes it kept in memory over them). Nothing by default.
    fn changes(&mut self) -> Changes {
        Changes::None
    }
    /// Content the sampler adds to nodes after they were first drawn (a far
    /// view's structures, found in the background), if it has any: see
    /// [`LateContent`]. `None` by default.
    fn late(&self) -> Option<Arc<dyn LateContent>> {
        None
    }
}

/// Content that reaches the far view after its terrain (structures, found by
/// a background job): the producer tells it the view each update, and once
/// every planned node is drawn takes the boxes whose drawn nodes are out of
/// date and makes those nodes again, behind all other work. The old nodes
/// stay drawn until the new ones replace them.
pub trait LateContent: Send + Sync {
    /// The camera, where it faces, the far distance, and whether every
    /// planned node is drawn (late work should wait for that: the terrain
    /// comes first).
    fn view(&self, camera: [f64; 3], forward: [f64; 3], max_distance: f64, complete: bool);
    /// Boxes of blocks (minimum, maximum, inclusive) whose nodes are out of
    /// date, taken.
    fn take_refreshes(&self) -> Vec<([i32; 3], [i32; 3])>;
    /// The boxes just taken that are no drawn node's own extent (the plan
    /// dropped that node): nothing is made for them, so whatever waits on them
    /// can stop waiting. Nothing by default.
    fn not_drawn(&self, _boxes: &[([i32; 3], [i32; 3])]) {}
    /// Whether every box [`LateContent::take_refreshes`] hands out is exactly
    /// one node's extent ([`CubeKey::of_box`]) and means that node only: the
    /// producer then remakes just that node, not every drawn node the box
    /// touches. False by default (structures' boxes are arbitrary).
    fn exact(&self) -> bool {
        false
    }
    /// Which nodes must come coarse first because nothing is drawn over their
    /// place yet (the plan's holes and the coverage nodes over them); `all`:
    /// every node (a big fill: a join, a teleport, the coverage phase). Any
    /// other node replaces one still drawn, so it may be made whole at once
    /// (`HORIZON_FAR_WHOLE_WHEN_COVERED`). Told when it changes, and only
    /// when [`LateContent::wants_coarse_first`]; nothing by default.
    fn coarse_first(&self, _keys: &[CubeKey], _all: bool) {}
    /// Whether [`LateContent::coarse_first`] is wanted at all (the list costs
    /// a pass over the holes and the coverage layer on the game thread).
    /// False by default.
    fn wants_coarse_first(&self) -> bool {
        false
    }
    /// Boxes just taken whose nodes are not made now (they are about to split
    /// a level: their children replace them): handed out again later, if
    /// still drawn. Nothing by default.
    fn not_now(&self, _boxes: &[([i32; 3], [i32; 3])]) {}
}

/// A sampled node on the GPU path: its voxels, or a slot of the GPU node pool
/// (`gfx::cube_gen::NodePool`) they were generated into. The slot's owner
/// frees it.
#[derive(Debug)]
pub enum Resident {
    Node(CubeNode),
    Slot(u32),
}

/// What a [`CubeSampler`] reports as changed (see `CubeLodCache::invalidate`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Changes {
    None,
    /// These cubes were saved again.
    Cubes(Vec<SectionPos>),
    /// So many that the list was lost: forget every node.
    All,
}

// ---------------------------------------------------------------- the plan

/// The distance rule the plan used before detail followed the screen, kept
/// as the tests' reference ([`Rule::distance`]): a node splits while nearer
/// than this many times its size. A far voxel then spans about 1/40 of its
/// distance, 13 pixels on a 720-pixel, 70 degree view.
pub const DETAIL: f64 = 2.5;
/// A node the previous plan had split merges back only when its voxels are
/// this many times smaller on screen than the size that splits it (it is
/// this many times farther), so the camera moving a little back and forth
/// across a boundary does not flip nodes between levels.
pub const MERGE: f64 = 1.25;
/// Nodes this level or finer ask `is_loaded` about each cube they hold; coarser
/// ones that meet the loaded box simply subdivide.
const PROBE_LEVEL: u8 = 2;
/// A view whose detail in view is less than this many times the unzoomed
/// one's is planned as unzoomed: a bow's draw or a sprint's wider view does
/// not make the plan depend on where the camera looks.
const ZOOMED: f64 = 1.5;

/// What the player sees, for choosing detail by size on screen
/// ([`CubeLod::update`]).
///
/// A node splits while one of its voxels, `1 << level` blocks wide, would
/// span more than `target_px` pixels at the node's nearest point:
/// `voxel / distance * height_px / (2 * tan(fov_y / 2)) > target_px`. A
/// narrower field of view (the spyglass) makes every voxel bigger on screen,
/// so the nodes in view refine; nodes out of view keep the unzoomed rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarView {
    /// The direction the camera looks (a unit vector).
    pub forward: [f64; 3],
    /// The vertical field of view drawn now, in radians: the spyglass, zoom
    /// and dynamic field of view included.
    pub fov_y: f64,
    /// The vertical field of view without any zoom (the FOV option), in
    /// radians: what nodes out of view are planned for.
    pub base_fov_y: f64,
    /// The viewport's width over its height.
    pub aspect: f64,
    /// The viewport's height in pixels.
    pub height_px: f64,
    /// The most pixels a voxel may span, at its node's nearest point, before
    /// the node splits (the Cubic Far Detail setting).
    pub target_px: f64,
}

impl FarView {
    /// A 70 degree, 16:9 view 720 pixels tall looking along +Z, aiming for
    /// `target_px` (the benchmark's window).
    pub fn standard(target_px: f64) -> Self {
        Self {
            forward: [0.0, 0.0, 1.0],
            fov_y: 70f64.to_radians(),
            base_fov_y: 70f64.to_radians(),
            aspect: 16.0 / 9.0,
            height_px: 720.0,
            target_px,
        }
    }
    /// The same view looking along `forward` with a field of view of
    /// `fov_y` radians (a zoom: the base stays).
    pub fn looking(self, forward: [f64; 3], fov_y: f64) -> Self {
        Self {
            forward,
            fov_y,
            ..self
        }
    }
    fn fov(fov: f64) -> f64 {
        if fov.is_finite() {
            fov.clamp(0.1f64.to_radians(), 179f64.to_radians())
        } else {
            70f64.to_radians()
        }
    }
    /// Pixels a block spans per block of distance with this vertical field of
    /// view: a voxel `w` blocks wide `d` blocks away spans `w / d * focal`.
    pub fn focal_px(&self, fov_y: f64) -> f64 {
        let height = if self.height_px.is_finite() {
            self.height_px.max(1.0)
        } else {
            720.0
        };
        height / (2.0 * (Self::fov(fov_y) / 2.0).tan())
    }
    /// Half the angle across the screen's diagonal.
    pub fn half_diagonal(&self) -> f64 {
        let aspect = if self.aspect.is_finite() && self.aspect > 0.0 {
            self.aspect
        } else {
            16.0 / 9.0
        };
        ((Self::fov(self.fov_y) / 2.0).tan() * (1.0 + aspect * aspect).sqrt()).atan()
    }
    /// How far a zoomed view may turn before the plan is redone: the plan's
    /// cone of finer nodes is this much wider than the screen.
    pub fn turn_margin(&self) -> f64 {
        (Self::fov(self.fov_y) * 0.25).max(0.5f64.to_radians())
    }
    /// The plan's rule for this view, with at most `budget` leaves.
    pub fn rule(&self, budget: usize) -> Rule {
        let target = if self.target_px.is_finite() {
            self.target_px.max(0.05)
        } else {
            8.0
        };
        // Out of view, the wider of the two fields of view: a sprint's or a
        // speed effect's wider view never asks for more than in view.
        let base = Self::fov(self.base_fov_y).max(Self::fov(self.fov_y));
        let out_of_view = self.focal_px(base) / (16.0 * target);
        let in_view = self.focal_px(self.fov_y) / (16.0 * target);
        if in_view < out_of_view * ZOOMED {
            return Rule {
                in_view: out_of_view,
                out_of_view,
                cone: None,
                budget,
            };
        }
        Rule {
            in_view,
            out_of_view,
            cone: Some(Cone::new(
                self.forward,
                self.half_diagonal() + self.turn_margin(),
            )),
            budget,
        }
    }
}

/// How finely a plan goes ([`plan_classified`]): a node splits while it is
/// nearer than `factor` times its size, `factor` being `in_view` for nodes
/// the cone meets and `out_of_view` for the rest (with [`FarView::rule`],
/// the voxel's size on screen against the target), and while the plan has
/// fewer than `budget` leaves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    pub in_view: f64,
    pub out_of_view: f64,
    /// Where `in_view` applies; `None` when the two are the same.
    pub cone: Option<Cone>,
    /// The most leaves a plan has. Past it, the nodes whose voxels are
    /// smallest on screen (the farthest for their size) stay whole: the plan
    /// coarsens from the far end rather than failing. Splits around the
    /// loaded cubes are made whatever the budget.
    pub budget: usize,
}

impl Rule {
    /// The distance rule, the same everywhere and unbounded.
    pub const fn distance(detail: f64) -> Self {
        Self {
            in_view: detail,
            out_of_view: detail,
            cone: None,
            budget: usize::MAX,
        }
    }
    /// The rule with voxels `times` bigger on screen, and at most `budget`
    /// leaves (the coverage layer's).
    pub fn coarser(&self, times: f64, budget: usize) -> Self {
        Self {
            in_view: self.in_view / times,
            out_of_view: self.out_of_view / times,
            cone: self.cone,
            budget,
        }
    }
    /// The factor for `node` seen from `camera`.
    pub fn factor(&self, camera: [f64; 3], node: CubeKey) -> f64 {
        match &self.cone {
            Some(cone) if cone.meets(camera, node) => self.in_view,
            _ => self.out_of_view,
        }
    }
    /// The factor for a box of blocks `lo..hi` (a surface tile's terrain).
    pub fn factor_box(&self, camera: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> f64 {
        match &self.cone {
            Some(cone) if cone.meets_box(camera, lo, hi) => self.in_view,
            _ => self.out_of_view,
        }
    }
}

/// A cone of directions from the camera: the view, widened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cone {
    axis: [f64; 3],
    half: f64,
}

impl Cone {
    /// Directions within `half` radians of `axis`.
    pub fn new(axis: [f64; 3], half: f64) -> Self {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let axis = if len > 1e-9 && len.is_finite() {
            axis.map(|v| v / len)
        } else {
            [0.0, 0.0, 1.0]
        };
        Self {
            axis,
            half: if half.is_finite() {
                half.clamp(0.0, std::f64::consts::PI)
            } else {
                std::f64::consts::PI
            },
        }
    }
    /// Whether any of `node` may lie inside the cone seen from `camera`:
    /// the node is taken as the sphere around it, whose own angular radius
    /// widens the cone, so a node straddling the edge counts in.
    pub fn meets(&self, camera: [f64; 3], node: CubeKey) -> bool {
        let size = f64::from(node.size());
        let lo = node.origin().map(f64::from);
        self.meets_box(camera, lo, lo.map(|v| v + size))
    }
    /// [`Cone::meets`] for a box of blocks `lo..hi`.
    pub fn meets_box(&self, camera: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> bool {
        let d = [0, 1, 2].map(|a| (lo[a] + hi[a]) / 2.0 - camera[a]);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let e = [0, 1, 2].map(|a| (hi[a] - lo[a]) / 2.0);
        let radius = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        if len <= radius {
            return true;
        }
        let reach = self.half + (radius / len).asin();
        if reach >= std::f64::consts::PI {
            return true;
        }
        (d[0] * self.axis[0] + d[1] * self.axis[1] + d[2] * self.axis[2]) / len >= reach.cos()
    }
}

/// The box of cubes around the camera's cube that the game may have loaded:
/// `horizontal` cubes each way in X and Z, `vertical` in Y. `is_loaded` is
/// only asked inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadedBox {
    pub horizontal: i32,
    pub vertical: i32,
}

fn cube_of(camera: [f64; 3]) -> SectionPos {
    let c = camera.map(|v| (v / 16.0).floor() as i32);
    SectionPos::new(c[0], c[1], c[2])
}

/// The nodes that draw the world out to `max_distance` blocks from `camera`
/// except the cubes `is_loaded` reports (which the near renderer draws):
/// non-overlapping, nearest first. Starting from nodes about as wide as the
/// range, a node splits while the camera is nearer than [`DETAIL`] times its
/// size, or while it holds loaded cubes (down to single cubes, which are
/// skipped), and a node farther than `max_distance` is left out.
pub fn plan(
    camera: [f64; 3],
    max_distance: f64,
    loaded: LoadedBox,
    is_loaded: &dyn Fn(SectionPos) -> bool,
) -> Vec<CubeKey> {
    plan_from(camera, max_distance, loaded, is_loaded, &[])
}

/// [`plan`] with hysteresis: a node that `previous` (the last plan) split
/// stays split until the camera is [`MERGE`] times farther than the distance
/// that splits it. The invariants of `plan` hold for any `previous`.
pub fn plan_from(
    camera: [f64; 3],
    max_distance: f64,
    loaded: LoadedBox,
    is_loaded: &dyn Fn(SectionPos) -> bool,
    previous: &[CubeKey],
) -> Vec<CubeKey> {
    plan_classified(
        camera,
        max_distance,
        &Rule::distance(DETAIL),
        loaded,
        is_loaded,
        previous,
        &|_| NodeClass::Mixed,
    )
    .leaves
}

/// Vanilla's world border, in blocks each way from the origin.
pub const WORLD_BORDER: i64 = 30_000_000;

/// Whether any of the node lies inside the world: within the border
/// horizontally and the cube range vertically. Overflow-safe at any level.
pub fn in_world(node: CubeKey) -> bool {
    let size = i64::from(node.size());
    // Whether the node's blocks `[at, at + size)` meet blocks `[lo, end)`.
    let within = |c: i32, lo: i64, end: i64| {
        let at = i64::from(c) * size;
        at < end && at + size > lo
    };
    // X and Z: vanilla's bounds, -30,000,000 inclusive to 30,000,000
    // exclusive; Y: the cube range, both ends inclusive.
    within(node.x, -WORLD_BORDER, WORLD_BORDER)
        && within(node.z, -WORLD_BORDER, WORLD_BORDER)
        && within(
            node.y,
            i64::from(horizon_world::cubic::MIN_Y),
            i64::from(horizon_world::cubic::MAX_Y) + 1,
        )
}

/// What the 2D pre-pass knows of a whole node (`CubeSampler::classify`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeClass {
    /// Unknown, or the surface may cross it.
    Mixed,
    /// Every voxel is air or every voxel is fluid: it draws nothing, and nor
    /// does anything inside it.
    Air,
    /// Every voxel is rock: it is a leaf however near, a closed shell.
    Rock,
}

/// The nodes the plan starts from: those of the first level at least as wide
/// as `max_distance` that the range around `camera` meets (at most 27).
pub fn root_nodes(camera: [f64; 3], max_distance: f64) -> Vec<CubeKey> {
    let top = (0..=MAX_CUBE_LEVEL)
        .find(|&l| f64::from(16 << l) >= max_distance)
        .unwrap_or(MAX_CUBE_LEVEL);
    let size = f64::from(16 << top);
    let span = |c: f64| {
        (
            ((c - max_distance) / size).floor() as i32,
            ((c + max_distance) / size).floor() as i32,
        )
    };
    let mut roots = Vec::new();
    let (x0, x1) = span(camera[0]);
    let (y0, y1) = span(camera[1]);
    let (z0, z1) = span(camera[2]);
    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                roots.push(CubeKey::new(top, x, y, z));
            }
        }
    }
    roots
}

/// What [`plan_far`] chose.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// The cube leaves: non-overlapping, nearest first.
    pub leaves: Vec<CubeKey>,
    /// The surface tiles (with [`PlanInput::surface`]), nearest first. No cube
    /// leaf lies in a tile's footprint, at any height.
    pub tiles: Vec<TileKey>,
    /// Nodes left out as uniform air or fluid, with their whole subtrees.
    pub pruned: usize,
    /// The budget kept some nodes whole that the rule would have split.
    pub limited: bool,
    /// How many times its target the first node the budget kept whole was
    /// (its voxels' size on screen over the target; 1 when the budget did
    /// not bind): the plan reaches about `target * coarsened`.
    pub coarsened: f64,
    /// The tile budget kept some tiles whole.
    pub tiles_limited: bool,
    /// Nodes that would split but wait to be drawn first ([`PlanInput::refined`]).
    pub held: usize,
    /// Those nodes ([`Plan::held`] of them): each splits as soon as it is
    /// drawn, so making it whole meanwhile is wasted work
    /// (`HORIZON_FAR_LATE_SKIP_HELD`).
    pub held_keys: Vec<CubeKey>,
}

/// What the plan needs to draw the coarse far terrain as surface tiles
/// (`gfx::surface_tile`): from [`SURFACE_LEVEL`] up, a quadtree of tiles over
/// X and Z, each standing for the whole height of its footprint.
#[derive(Clone, Copy)]
pub struct Surface<'a> {
    /// The headers of the tiles generated so far: complex columns, heights.
    pub info: &'a dyn Fn(TileKey) -> Option<TileInfo>,
    /// Where a tile not generated yet is taken to have its terrain.
    pub sea_level: i32,
    /// The most tiles a plan holds.
    pub budget: usize,
    /// The last plan's tiles, for the hysteresis.
    pub previous: &'a [TileKey],
}

/// Everything [`plan_far`] plans from.
#[derive(Clone, Copy)]
pub struct PlanInput<'a> {
    pub camera: [f64; 3],
    pub max_distance: f64,
    pub rule: &'a Rule,
    pub loaded: LoadedBox,
    pub is_loaded: &'a dyn Fn(SectionPos) -> bool,
    /// The last plan's cube leaves, for the hysteresis.
    pub previous: &'a [CubeKey],
    pub class: &'a dyn Fn(CubeKey) -> NodeClass,
    /// Surface tiles at the coarse levels; `None`: cube nodes everywhere.
    pub surface: Option<Surface<'a>>,
    /// The nodes drawn, and every node holding one ([`refined_set`]). With
    /// it, the plan refines one level at a time: a node splits only once it,
    /// or something inside it, is drawn, so a drawn node is replaced by its 8
    /// children, which split again in a later plan once they are in. `None`:
    /// the plan goes straight to the target.
    pub refined: Option<&'a FxHashSet<CubeKey>>,
}

/// The nodes `drawn` lists and every node holding one: what
/// [`PlanInput::refined`] takes.
pub fn refined_set(drawn: &[CubeKey]) -> FxHashSet<CubeKey> {
    let mut out = FxHashSet::default();
    for &key in drawn {
        let mut up = key;
        while out.insert(up) && up.level < MAX_CUBE_LEVEL {
            up = up.parent();
        }
    }
    out
}

/// Where a cube node's footprint lies with respect to the tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    /// All of it is planned as cube nodes (or no tile plan reaches it).
    Cubes,
    /// All of it is under one tile: the tile draws it.
    Tiles,
    /// Both: the node splits until each part is one or the other.
    Split,
}

/// A node the plan may split, by how far over the target its voxels are.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    /// It meets the loaded cubes, or both tiles and cube nodes: it splits
    /// whatever the budget.
    forced: bool,
    ratio: f64,
    distance: f64,
    key: CubeKey,
    region: Region,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.forced
            .cmp(&other.forced)
            .then(self.ratio.total_cmp(&other.ratio))
            .then(other.key.cmp(&self.key))
    }
}

/// What becomes of a node the plan reaches.
enum Fate {
    /// Outside the world or the range, all loaded, or under a tile: nothing.
    Out,
    /// Uniform air or fluid: left out with its subtree.
    Pruned,
    Leaf(f64, CubeKey),
    Split(Candidate),
}

/// The tile quadtree a plan chose: its tiles, the footprints planned as cube
/// nodes instead, and the footprints split further.
#[derive(Default)]
struct Quadtree {
    tiles: FxHashSet<TileKey>,
    cubes: FxHashSet<TileKey>,
    split: FxHashSet<TileKey>,
}

impl Quadtree {
    /// The region of `node`'s footprint, from that footprint alone (its
    /// parent's was split, or it is a root).
    fn region(&self, node: CubeKey) -> Region {
        let t = TileKey::new(node.level, node.x, node.z);
        if self.tiles.contains(&t) {
            Region::Tiles
        } else if self.split.contains(&t) {
            Region::Split
        } else {
            // A cube footprint, or one no tile plan reaches (out of range).
            Region::Cubes
        }
    }
}

/// A tile the plan may split.
#[derive(Clone, Copy, Debug)]
struct TileCandidate {
    forced: bool,
    ratio: f64,
    distance: f64,
    key: TileKey,
}

impl PartialEq for TileCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for TileCandidate {}
impl PartialOrd for TileCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TileCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.forced
            .cmp(&other.forced)
            .then(self.ratio.total_cmp(&other.ratio))
            .then(other.key.cmp(&self.key))
    }
}

enum TileFate {
    Out,
    /// Planned as cube nodes of its level.
    Cubes,
    Leaf(f64),
    Split(TileCandidate),
}

/// Whether any of a tile's footprint lies within the world border.
fn tile_in_world(t: TileKey) -> bool {
    let size = i64::from(t.size());
    let within = |c: i32| {
        let at = i64::from(c) * size;
        at < WORLD_BORDER && at + size > -WORLD_BORDER
    };
    within(t.x) && within(t.z)
}

/// The blocks a tile's terrain is taken to fill: its footprint, from the
/// lowest its ground can reach to its highest top (or water), from its header;
/// before it is generated, a band around the sea.
pub fn tile_box(t: TileKey, info: Option<TileInfo>, sea_level: i32) -> ([f64; 3], [f64; 3]) {
    let [ox, oz] = t.origin();
    let size = f64::from(t.size());
    let (lo, hi) = match info {
        Some(i) => (
            i.skirt.max(horizon_world::cubic::MIN_Y),
            i.highest.min(horizon_world::cubic::MAX_Y) + 1,
        ),
        None => (sea_level - UNKNOWN_BELOW, sea_level + UNKNOWN_ABOVE),
    };
    (
        [f64::from(ox), f64::from(lo), f64::from(oz)],
        [
            f64::from(ox) + size,
            f64::from(hi.max(lo)),
            f64::from(oz) + size,
        ],
    )
}

/// How far below and above the sea an unknown tile's terrain is taken to lie.
const UNKNOWN_BELOW: i32 = 64;
const UNKNOWN_ABOVE: i32 = 192;

/// Distance in blocks from `point` to the nearest point of the box.
fn box_distance(point: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let mut sum = 0.0;
    for a in 0..3 {
        let gap = (lo[a] - point[a]).max(point[a] - hi[a]).max(0.0);
        sum += gap * gap;
    }
    sum.sqrt()
}

/// The tile quadtree: from the roots' columns down, a tile splits into its
/// four children by `rule` (on the box of its terrain, so a tall mountain
/// or a sky island refines as near as it reaches), the ones furthest over
/// their target first, up to `surface.budget` tiles. A footprint becomes cube
/// nodes where [`node_kind`] says so (below [`SURFACE_LEVEL`]; a complex tile
/// sooner) and where it meets the loaded cubes or holds the camera: the
/// camera's own footprint is never a tile.
fn plan_tiles(
    input: &PlanInput,
    surface: &Surface,
    roots: &[CubeKey],
) -> (Quadtree, Vec<TileKey>, bool) {
    let camera = input.camera;
    let center = cube_of(camera);
    let h = input.loaded.horizontal.max(0);
    // The loaded box's footprint, in blocks: it always holds the camera.
    let near = (
        [i64::from(center.x - h) * 16, i64::from(center.z - h) * 16],
        [
            i64::from(center.x + h + 1) * 16,
            i64::from(center.z + h + 1) * 16,
        ],
    );
    let meets_near = |t: TileKey| {
        let [ox, oz] = t.origin().map(i64::from);
        let size = i64::from(t.size());
        ox < near.1[0] && ox + size > near.0[0] && oz < near.1[1] && oz + size > near.0[1]
    };
    // The footprints the last plan split: those holding its tiles, or its
    // cube leaves at least one level finer.
    let mut was_split = FxHashSet::<TileKey>::default();
    for &t in surface.previous {
        let mut level = t.level;
        while level < MAX_CUBE_LEVEL {
            level += 1;
            let up = level - t.level;
            if !was_split.insert(TileKey::new(level, t.x >> up, t.z >> up)) {
                break;
            }
        }
    }
    for &c in input.previous {
        let mut level = c.level.max(SURFACE_LEVEL - 1);
        while level < MAX_CUBE_LEVEL {
            level += 1;
            let up = level - c.level;
            if !was_split.insert(TileKey::new(level, c.x >> up, c.z >> up)) {
                break;
            }
        }
    }
    let fate = |t: TileKey| -> TileFate {
        if !tile_in_world(t) {
            return TileFate::Out;
        }
        let info = (surface.info)(t);
        let (lo, hi) = tile_box(t, info, surface.sea_level);
        let distance = box_distance(camera, lo, hi);
        if distance > input.max_distance {
            return TileFate::Out;
        }
        let as_node = CubeKey::new(t.level, t.x, 0, t.z);
        if meets_near(t) {
            return if t.level <= SURFACE_LEVEL {
                TileFate::Cubes
            } else {
                TileFate::Split(TileCandidate {
                    forced: true,
                    ratio: f64::INFINITY,
                    distance,
                    key: t,
                })
            };
        }
        if node_kind(as_node, None, info) == NodeKind::Cube {
            return TileFate::Cubes;
        }
        let merge = if was_split.contains(&t) { MERGE } else { 1.0 };
        let ratio = input.rule.factor_box(camera, lo, hi) * f64::from(t.size()) * merge / distance;
        if ratio > 1.0 {
            TileFate::Split(TileCandidate {
                forced: false,
                ratio,
                distance,
                key: t,
            })
        } else {
            TileFate::Leaf(distance)
        }
    };
    let mut q = Quadtree::default();
    let mut leaves: Vec<(f64, TileKey)> = Vec::new();
    let mut heap = std::collections::BinaryHeap::new();
    let mut count = 0usize;
    let take = |t: TileKey,
                f: TileFate,
                q: &mut Quadtree,
                leaves: &mut Vec<(f64, TileKey)>,
                heap: &mut std::collections::BinaryHeap<TileCandidate>| match f {
        TileFate::Out => {}
        TileFate::Cubes => {
            q.cubes.insert(t);
        }
        TileFate::Leaf(d) => leaves.push((d, t)),
        TileFate::Split(c) => heap.push(c),
    };
    let mut columns: Vec<TileKey> = roots
        .iter()
        .map(|r| TileKey::new(r.level, r.x, r.z))
        .collect();
    columns.sort_unstable();
    columns.dedup();
    for t in columns {
        let f = fate(t);
        if matches!(f, TileFate::Leaf(_) | TileFate::Split(_)) {
            count += 1;
        }
        take(t, f, &mut q, &mut leaves, &mut heap);
    }
    let mut full = false;
    let mut limited = false;
    while let Some(c) = heap.pop() {
        if !c.forced && full {
            leaves.push((c.distance, c.key));
            continue;
        }
        let t = c.key;
        let children: [TileKey; 4] = std::array::from_fn(|i| {
            TileKey::new(
                t.level - 1,
                t.x * 2 + (i & 1) as i32,
                t.z * 2 + (i >> 1) as i32,
            )
        });
        let fates = children.map(&fate);
        let added = fates
            .iter()
            .filter(|f| matches!(f, TileFate::Leaf(_) | TileFate::Split(_)))
            .count();
        if !c.forced && count - 1 + added > surface.budget {
            full = true;
            limited = true;
            leaves.push((c.distance, t));
            continue;
        }
        count = count - 1 + added;
        q.split.insert(t);
        for (child, f) in children.into_iter().zip(fates) {
            take(child, f, &mut q, &mut leaves, &mut heap);
        }
    }
    leaves.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    q.tiles = leaves.iter().map(|&(_, t)| t).collect();
    (q, leaves.into_iter().map(|(_, t)| t).collect(), limited)
}

/// The plan with what is known of nodes' contents and how fine it goes, cube
/// nodes everywhere ([`plan_far`] without tiles).
pub fn plan_classified(
    camera: [f64; 3],
    max_distance: f64,
    rule: &Rule,
    loaded: LoadedBox,
    is_loaded: &dyn Fn(SectionPos) -> bool,
    previous: &[CubeKey],
    class: &dyn Fn(CubeKey) -> NodeClass,
) -> Plan {
    plan_far(&PlanInput {
        camera,
        max_distance,
        rule,
        loaded,
        is_loaded,
        previous,
        class,
        surface: None,
        refined: None,
    })
}

/// The plan: a node `class` calls [`NodeClass::Air`] is left out with its
/// whole subtree, one it calls [`NodeClass::Rock`] stays a leaf (nothing
/// inside shows), unless it holds loaded cubes. A node splits by `rule` (with
/// hysteresis: one `previous` split splits while its ratio is above
/// `1 / MERGE`), the nodes furthest over their target first, until
/// `rule.budget` leaves. With `surface`, the tile quadtree is planned first
/// ([`plan_tiles`]) and the cube nodes fill the footprints it leaves them:
/// a node under a tile is left out, a node that is partly under one splits
/// (or, all rock, is left out: the tiles' skirts close the ground).
pub fn plan_far(input: &PlanInput) -> Plan {
    let PlanInput {
        camera,
        max_distance,
        rule,
        loaded,
        is_loaded,
        previous,
        class,
        surface,
        refined,
    } = *input;
    // One level a plan: a node not drawn (nor holding a drawn node) stays a
    // leaf, to be drawn before it splits.
    let may_split = |node: CubeKey| refined.is_none_or(|r| r.contains(&node));
    let held = std::cell::Cell::new(0usize);
    let held_keys = std::cell::RefCell::new(Vec::<CubeKey>::new());
    // The nodes holding a leaf of the last plan: the ones it split.
    let mut was_split = FxHashSet::<CubeKey>::default();
    for &leaf in previous {
        let mut up = leaf;
        while up.level < MAX_CUBE_LEVEL {
            up = up.parent();
            if !was_split.insert(up) {
                break;
            }
        }
    }
    let roots = root_nodes(camera, max_distance);
    let mut out = Plan {
        coarsened: 1.0,
        ..Plan::default()
    };
    let quadtree = match surface {
        Some(s) if roots.first().is_some_and(|r| r.level >= SURFACE_LEVEL) => {
            let (q, tiles, limited) = plan_tiles(input, &s, &roots);
            out.tiles = tiles;
            out.tiles_limited = limited;
            Some(q)
        }
        _ => None,
    };
    let center = cube_of(camera);
    // Whether `node`, a child of a node of `region`, would be left out (`Fate::Out` or `Fate::Pruned`, below)
    // when its parent splits.
    let bare = |node: CubeKey, region: Region| -> bool {
        if !in_world(node) || node.distance(camera) > max_distance {
            return true;
        }
        let region = match (&quadtree, region) {
            (None, _) | (Some(_), Region::Cubes) => Region::Cubes,
            (Some(q), _) => q.region(node),
        };
        if region == Region::Tiles {
            return true;
        }
        match loaded_state(node, center, loaded, is_loaded) {
            Loaded::All => true,
            Loaded::Some => false,
            Loaded::None => match class(node) {
                NodeClass::Air => true,
                NodeClass::Rock => region == Region::Split,
                NodeClass::Mixed => false,
            },
        }
    };
    let fate = |node: CubeKey, parent: Option<Region>| -> Fate {
        if !in_world(node) {
            return Fate::Out;
        }
        let distance = node.distance(camera);
        if distance > max_distance {
            return Fate::Out;
        }
        let region = match (&quadtree, parent) {
            (None, _) => Region::Cubes,
            (Some(_), Some(Region::Cubes)) => Region::Cubes,
            (Some(q), _) => q.region(node),
        };
        if region == Region::Tiles {
            return Fate::Out;
        }
        let split = |forced| {
            Fate::Split(Candidate {
                forced,
                ratio: f64::INFINITY,
                distance,
                key: node,
                region,
            })
        };
        match loaded_state(node, center, loaded, is_loaded) {
            Loaded::All => Fate::Out,
            // Over near cubes: split at once, drawn or not, so no far node is
            // ever drawn over near ground (the one-level steps are for the
            // far land only).
            Loaded::Some => split(true),
            Loaded::None => match class(node) {
                NodeClass::Air => Fate::Pruned,
                // Rock reaching under a tile: the tile's skirts close it.
                NodeClass::Rock if region == Region::Split => Fate::Out,
                NodeClass::Rock => Fate::Leaf(distance, node),
                NodeClass::Mixed if region == Region::Split => split(true),
                NodeClass::Mixed if beside_loaded(node, center, loaded, is_loaded) => split(true),
                NodeClass::Mixed if node.level == 0 => Fate::Leaf(distance, node),
                NodeClass::Mixed => {
                    let merge = if was_split.contains(&node) {
                        MERGE
                    } else {
                        1.0
                    };
                    let ratio =
                        rule.factor(camera, node) * f64::from(node.size()) * merge / distance;
                    if ratio > 1.0
                        && !may_split(node)
                        && node.children().iter().all(|&c| bare(c, region))
                    {
                        // Nothing of it would be planned once it splits (its eight are air, or drawn by
                        // tiles or near cubes): left out now. Held, it was drawn, split into nothing, left
                        // the drawn nodes and was held again: the plan went back and forth every replan.
                        Fate::Out
                    } else if ratio > 1.0 && !may_split(node) {
                        held.set(held.get() + 1);
                        held_keys.borrow_mut().push(node);
                        Fate::Leaf(distance, node)
                    } else if ratio > 1.0 {
                        Fate::Split(Candidate {
                            forced: false,
                            ratio,
                            distance,
                            key: node,
                            region,
                        })
                    } else {
                        Fate::Leaf(distance, node)
                    }
                }
            },
        }
    };
    let mut leaves: Vec<(f64, CubeKey)> = Vec::new();
    let mut heap = std::collections::BinaryHeap::new();
    // Leaves and candidates so far: what the plan holds if nothing more splits.
    let mut count = 0usize;
    let take = |f: Fate,
                leaves: &mut Vec<(f64, CubeKey)>,
                heap: &mut std::collections::BinaryHeap<Candidate>,
                pruned: &mut usize| match f {
        Fate::Out => {}
        Fate::Pruned => *pruned += 1,
        Fate::Leaf(distance, key) => leaves.push((distance, key)),
        Fate::Split(c) => heap.push(c),
    };
    for root in roots {
        let f = fate(root, None);
        if matches!(f, Fate::Leaf(..) | Fate::Split(_)) {
            count += 1;
        }
        take(f, &mut leaves, &mut heap, &mut out.pruned);
    }
    let mut full = false;
    while let Some(c) = heap.pop() {
        if !c.forced && full {
            leaves.push((c.distance, c.key));
            continue;
        }
        let fates = c.key.children().map(|child| fate(child, Some(c.region)));
        let added = fates
            .iter()
            .filter(|f| matches!(f, Fate::Leaf(..) | Fate::Split(_)))
            .count();
        if !c.forced && count - 1 + added > rule.budget {
            // Every node left is at most this far over its target.
            full = true;
            out.limited = true;
            out.coarsened = c.ratio;
            leaves.push((c.distance, c.key));
            continue;
        }
        count = count - 1 + added;
        for f in fates {
            take(f, &mut leaves, &mut heap, &mut out.pruned);
        }
    }
    leaves.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    out.leaves = leaves.into_iter().map(|(_, node)| node).collect();
    out.held = held.get();
    out.held_keys = held_keys.into_inner();
    out
}

enum Loaded {
    None,
    Some,
    All,
}

fn loaded_state(
    node: CubeKey,
    center: SectionPos,
    loaded: LoadedBox,
    is_loaded: &dyn Fn(SectionPos) -> bool,
) -> Loaded {
    let side = 1_i64 << node.level;
    let meets = |c: i32, n: i32, radius: i32| {
        let lo = i64::from(n) * side;
        lo <= i64::from(c) + i64::from(radius) && lo + side > i64::from(c) - i64::from(radius)
    };
    if !(meets(center.x, node.x, loaded.horizontal)
        && meets(center.y, node.y, loaded.vertical)
        && meets(center.z, node.z, loaded.horizontal))
    {
        return Loaded::None;
    }
    if node.level > PROBE_LEVEL {
        return Loaded::Some;
    }
    let side = side as i32;
    let (mut yes, mut no) = (0, 0);
    for dy in 0..side {
        for dz in 0..side {
            for dx in 0..side {
                let cube = SectionPos::new(
                    (node.x << node.level) + dx,
                    (node.y << node.level) + dy,
                    (node.z << node.level) + dz,
                );
                if is_loaded(cube) {
                    yes += 1;
                } else {
                    no += 1;
                }
            }
        }
    }
    match (yes, no) {
        (0, _) => Loaded::None,
        (_, 0) => Loaded::All,
        _ => Loaded::Some,
    }
}

/// Whether a node of level 1 up to [`PROBE_LEVEL`] that holds no loaded cube has one right beside it: a cube
/// one step away in X or Z, in the node's own height range. Such a node splits down to single cubes, which are
/// exact: a far voxel with the surface in it is solid whole, so its top stands up to a voxel above the
/// ground, and where it met the near terrain that showed as a wall as tall as the voxel.
fn beside_loaded(
    node: CubeKey,
    center: SectionPos,
    loaded: LoadedBox,
    is_loaded: &dyn Fn(SectionPos) -> bool,
) -> bool {
    if node.level == 0 || node.level > PROBE_LEVEL {
        return false;
    }
    let side = 1_i32 << node.level;
    let (x0, y0, z0) = (
        node.x << node.level,
        node.y << node.level,
        node.z << node.level,
    );
    // `is_loaded` is only asked inside the loaded box.
    let inside = |p: SectionPos| {
        (p.x - center.x).abs() <= loaded.horizontal
            && (p.y - center.y).abs() <= loaded.vertical
            && (p.z - center.z).abs() <= loaded.horizontal
    };
    for dy in 0..side {
        for dz in -1..=side {
            for dx in -1..=side {
                // The cubes along the four sides, not the corners or the node's own.
                let (off_x, off_z) = (dx < 0 || dx >= side, dz < 0 || dz >= side);
                if off_x == off_z {
                    continue;
                }
                let cube = SectionPos::new(x0 + dx, y0 + dy, z0 + dz);
                if inside(cube) && is_loaded(cube) {
                    return true;
                }
            }
        }
    }
    false
}

// ------------------------------------------------------------- node kinds

/// From this level up, the far terrain is drawn as 2D height-map tiles
/// (`gfx::surface_tile`) rather than 16^3 cube nodes, where tiles run (a
/// Custom world on the GPU): a quadtree over X and Z, each tile standing for
/// the whole height of its footprint, with per column a few solid spans at
/// exact heights. A level-12 cube node is 65,536 blocks wide with voxels 4,096
/// blocks tall: nearly all terrain lies inside one or two of them, so it snaps
/// the surface to whole voxels, where a tile keeps every block of height.
/// A guess to measure: with exact heights tiles may look right lower down.
pub const SURFACE_LEVEL: u8 = 12;
/// A tile flagged complex (some columns have more solid spans than a tile
/// holds: floating islands, overhangs, a cavern's ceiling) turns into cube
/// nodes this many levels sooner, coming from the coarse end.
pub const COMPLEX_LIFT: u8 = 3;

/// How a far node is drawn ([`node_kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// A 2D height-map tile over the node's X and Z (all Y).
    Tile,
    /// A 16^3 cube node.
    Cube,
}

/// How the footprint of `key` at its level is drawn where tiles run: the one
/// place this is decided. A tile at [`SURFACE_LEVEL`] or up, cube nodes below;
/// a tile whose header says it is complex turns into cube nodes
/// [`COMPLEX_LIFT`] levels sooner. `tile` is that header, `None` while the
/// tile is not generated: the tile is drawn first (it is the fastest coverage)
/// and refined into cube nodes once known to be complex. `view` is for a
/// screen-size threshold later (a tile stays a tile while its height steps are
/// below a pixel).
pub fn node_kind(key: CubeKey, view: Option<&FarView>, tile: Option<TileInfo>) -> NodeKind {
    let _ = view;
    let from = match tile {
        Some(info) if info.complex() => SURFACE_LEVEL.saturating_add(COMPLEX_LIFT),
        _ => SURFACE_LEVEL,
    };
    if key.level >= from {
        NodeKind::Tile
    } else {
        NodeKind::Cube
    }
}

// ---------------------------------------------------------------- producer

/// Nodes kept after sampling, so a node meshed again (or the parent of
/// children already sampled) is not sampled twice. Only the legacy size
/// (`HORIZON_FAR_CACHE_MB=0`): the cache is bounded by bytes otherwise
/// (192 MiB, dense nodes packed: some 40k nodes; see [`Cache`]).
const CACHE_NODES: usize = 1024;
/// How often the plan is recomputed when the camera stays in one cube, for
/// cubes that loaded or unloaded meanwhile.
const REPLAN: Duration = Duration::from_millis(250);
/// How soon after a node comes in the plan is redone while nodes wait to
/// refine a level ([`Plan::held`]): the pace the far view subdivides at.
const REFINE_REPLAN: Duration = Duration::from_millis(100);

/// A far-view tunable: the environment variable `name` when set to a number,
/// else `default`, read once per process. For the overnight loop's sweeps
/// (research/far-fill): a value tried without a rebuild. Unset, the game runs
/// on the defaults as before.
pub fn far_knob(name: &'static str, default: u64) -> u64 {
    static KNOBS: std::sync::OnceLock<Mutex<FxHashMap<&'static str, u64>>> =
        std::sync::OnceLock::new();
    *KNOBS
        .get_or_init(Mutex::default)
        .lock()
        .unwrap()
        .entry(name)
        .or_insert_with(|| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default)
        })
}

/// Opt-in timing of the far view's producers (`HORIZON_FAR_PROFILE=1`), for
/// the overnight loop (research/far-fill): where a worker's time goes, and how
/// busy the GPU thread is. Logged every 5 s while on; nothing is timed off.
pub mod profile {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    pub static POOL_WAIT: AtomicU64 = AtomicU64::new(0);
    pub static QUEUE_WAIT: AtomicU64 = AtomicU64::new(0);
    pub static RESIDENT: AtomicU64 = AtomicU64::new(0);
    pub static CPU_NODES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_NODES: AtomicU64 = AtomicU64::new(0);
    pub static SEND: AtomicU64 = AtomicU64::new(0);
    pub static KEYS: AtomicU64 = AtomicU64::new(0);
    pub static CPU_KEYS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_BUSY: AtomicU64 = AtomicU64::new(0);
    pub static GPU_ROUNDS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_KEYS: AtomicU64 = AtomicU64::new(0);
    // Inside the game's sampler (cubic_lod.rs): its CPU nodes and the far request.
    pub static S_CACHE: AtomicU64 = AtomicU64::new(0);
    pub static S_SAMPLE: AtomicU64 = AtomicU64::new(0);
    pub static S_COVER: AtomicU64 = AtomicU64::new(0);
    pub static S_KEEP: AtomicU64 = AtomicU64::new(0);
    pub static S_FAR: AtomicU64 = AtomicU64::new(0);
    pub static S_COVER_BOX: AtomicU64 = AtomicU64::new(0);
    // Earth's two passes (cubic_lod/refine.rs, `Refine::made`): nodes made
    // coarse, made whole when handed out to be refined, and made whole
    // without being handed out (straight away, or read whole). Counted always.
    pub static R_COARSE: AtomicU64 = AtomicU64::new(0);
    pub static R_WHOLE_REFRESH: AtomicU64 = AtomicU64::new(0);
    pub static R_WHOLE_DIRECT: AtomicU64 = AtomicU64::new(0);
    static LAST: AtomicU64 = AtomicU64::new(0);

    pub fn on() -> bool {
        super::far_knob("HORIZON_FAR_PROFILE", 0) != 0
    }
    pub fn add(counter: &AtomicU64, since: Instant) {
        counter.fetch_add(since.elapsed().as_nanos() as u64, Relaxed);
    }
    /// Logs the totals since the last log, at most every 5 s. Whether it
    /// logged.
    pub fn maybe_log() -> bool {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let now = START.get_or_init(Instant::now).elapsed().as_millis() as u64;
        let last = LAST.load(Relaxed);
        if now < last + 5000 || LAST.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
            return false;
        }
        let ms = |c: &AtomicU64| c.swap(0, Relaxed) as f64 / 1e6;
        let n = |c: &AtomicU64| c.swap(0, Relaxed);
        // The near terrain's streaming jobs too (worker CPU time by kind).
        if !crate::streaming::job_stats::enabled_pub() {
            crate::streaming::job_stats::enable(true);
        } else {
            let jobs: Vec<String> = crate::streaming::job_stats::take()
                .into_iter()
                .filter(|(_, _, count)| *count > 0)
                .map(|(kind, nanos, count)| {
                    format!("{kind} {count} jobs {:.0} ms", nanos as f64 / 1e6)
                })
                .collect();
            log::info!("Horizon near jobs ({} ms): {}", now - last, jobs.join(", "));
        }
        log::info!(
            "Horizon far profile ({} ms): pool wait {:.0} ms, queue wait {:.0} ms, resident_many {:.0} ms, cpu nodes {:.0} ms, gpu nodes {:.0} ms, send {:.0} ms; keys {} (cpu {}); gpu thread busy {:.0} ms over {} rounds, {} keys; sampler: cache {:.0} ms, sample {:.0} ms, cover {:.0} ms (slab proofs {:.0} ms), keep {:.0} ms, far request {:.0} ms",
            now - last,
            ms(&POOL_WAIT),
            ms(&QUEUE_WAIT),
            ms(&RESIDENT),
            ms(&CPU_NODES),
            ms(&GPU_NODES),
            ms(&SEND),
            n(&KEYS),
            n(&CPU_KEYS),
            ms(&GPU_BUSY),
            n(&GPU_ROUNDS),
            n(&GPU_KEYS),
            ms(&S_CACHE),
            ms(&S_SAMPLE),
            ms(&S_COVER),
            ms(&S_COVER_BOX),
            ms(&S_KEEP),
            ms(&S_FAR),
        );
        true
    }

    /// What the last log saw of the waste counters, to log what came since.
    struct Seen {
        at: Instant,
        waste: super::WasteSnap,
        cache: [u64; 5],
        earth: [u64; 3],
    }

    /// Logs the far view's waste counters over the window since the last call
    /// (the cache's size as it is now): the line to read what the workers made
    /// that was not needed. Called with [`maybe_log`], in the same window.
    pub(super) fn maybe_log_waste(shared: &super::Shared) {
        static SEEN: std::sync::Mutex<Option<Seen>> = std::sync::Mutex::new(None);
        let now = Instant::now();
        let waste = shared.waste.snap();
        let (cache, nodes, bytes) = {
            let c = shared.cache.lock().unwrap();
            (
                [c.hits, c.misses, c.evicted, c.forgotten, c.cleared],
                c.len(),
                c.bytes(),
            )
        };
        let earth = [&R_COARSE, &R_WHOLE_REFRESH, &R_WHOLE_DIRECT].map(|c| c.load(Relaxed));
        let mut seen = SEEN.lock().unwrap();
        let first = Seen {
            at: shared.near_epoch,
            waste: Default::default(),
            cache: [0; 5],
            earth: [0; 3],
        };
        let before = seen.as_ref().unwrap_or(&first);
        let w = waste.since(&before.waste);
        let c = std::array::from_fn::<u64, 5, _>(|i| cache[i].saturating_sub(before.cache[i]));
        let e = std::array::from_fn::<u64, 3, _>(|i| earth[i].saturating_sub(before.earth[i]));
        let window = now.saturating_duration_since(before.at).as_millis();
        log::info!(
            "Horizon far waste ({window} ms): jobs {}, built {} (uniform {}, empty {}, failed {}), delivered {}, rebuilt {}; plan +{} -{} (installs {}, reused {}, readded {}, dropped unbuilt {}); released {}, short-lived {} {:?}; late refreshed {} (skipped held {}, dropped {}, wasted {}, other {}, same {}); retries {}; refills {} ({} keys); replans {} (moved {}, classes {}, refine {}, timer {}); cache hits {} misses {} evicted {} forgotten {} cleared {}, {} nodes, {:.1} MB; earth coarse {}, whole refresh {}, whole direct {}",
            w.jobs,
            w.built,
            w.built_uniform,
            w.built_empty,
            w.failed_jobs,
            w.delivered,
            w.rebuilt,
            w.plan_added,
            w.plan_removed,
            w.plan_installs,
            w.plan_reused,
            w.plan_readded,
            w.plan_dropped_unbuilt,
            w.released,
            w.short_lived,
            w.short_lived_levels,
            w.late_refreshed,
            w.late_skipped_held,
            w.late_dropped,
            w.late_wasted,
            w.late_other,
            w.late_same,
            w.retries,
            w.refills,
            w.refill_keys,
            w.replans,
            w.replans_moved,
            w.replans_classes,
            w.replans_refine,
            w.replans_timer,
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
            nodes,
            bytes as f64 / (1 << 20) as f64,
            e[0],
            e[1],
            e[2],
        );
        *seen = Some(Seen {
            at: now,
            waste,
            cache,
            earth,
        });
    }
}

fn far_ms(name: &'static str, default: Duration) -> Duration {
    Duration::from_millis(far_knob(name, default.as_millis() as u64))
}

/// The box of blocks (minimum, maximum, inclusive) of a node's extent: the
/// inverse of [`CubeKey::of_box`].
fn box_of(key: CubeKey) -> ([i32; 3], [i32; 3]) {
    let (o, size) = (key.origin(), key.size());
    (o, o.map(|v| v + size - 1))
}

/// How long a node whose sampling failed waits before it is asked again.
const RETRY: Duration = Duration::from_secs(1);
/// The plan is redone when the view's detail changes by more than this ratio
/// either way: the spyglass in and out.
const REFOCUS: f64 = 1.25;
/// The coverage layer is the plan for voxels this many times bigger on
/// screen...
pub const COVER_SCALE: f64 = 8.0;
/// ...with at most this many nodes.
pub const COVER_NODES: usize = 768;
/// A plan with more than one in this many of its nodes undrawn (nothing
/// delivered over their place: a teleport, joining) asks for the coverage
/// layer first.
const COVER_HOLES: usize = 4;
/// ...or with more than this many holes, whatever the plan's size: at the fill
/// rate (about 1,000 to 3,000 nodes a second) that is a second or more of
/// holes, where the coverage layer's few hundred nodes take a fraction of
/// one. Rough terrain (a height variation of 600) left holes 11 s after
/// climbing from y 150 to 1,500 without it.
const COVER_MIN_HOLES: usize = 2048;
/// The longest the coverage layer holds the rest of the plan back.
const COVER_WAIT: Duration = Duration::from_millis(1500);
/// The most nodes a plan holds unless [`CubeLod::set_budget`] says
/// otherwise; the GPU's node table and arena may lower it
/// ([`CubeLod::budget`]).
pub const NODE_BUDGET: usize = 120_000;
/// The environment variable that replaces [`NODE_BUDGET`] (a number of nodes),
/// for benchmarks (`tools/cubic_far_bench.py --budget`).
pub const BUDGET_VAR: &str = "HORIZON_FAR_NODES";
/// After a jump (a move farther than [`JUMP`]) the plan waits this long at
/// most for the GPU descent's classes of the new place: without them every
/// node there counts as mixed and the plan asks for the sky too.
const DESCENT_WAIT: Duration = Duration::from_millis(300);
const JUMP: f64 = 4096.0;
/// The queue takes its buckets in a new order when the view turned this far
/// (radians, 5 degrees) or zoomed ([`Look::moved`]).
const TURN: f64 = 0.087;
/// Above this many nodes, the passes over the whole plan (releasing nodes,
/// looking for holes) run at most every [`SCAN_EVERY`].
const SLOW_SCAN: usize = 8192;
const SCAN_EVERY: Duration = Duration::from_millis(200);

/// The least time between two passes of `releasable` over a short stale set:
/// a stale node goes at most this much later than it could (below a frame at
/// 60 fps). `HORIZON_FAR_RELEASE_MS` overrides it; 0 in unit tests.
fn release_every() -> Duration {
    if cfg!(test) {
        return Duration::ZERO;
    }
    far_ms("HORIZON_FAR_RELEASE_MS", Duration::from_millis(25))
}
/// The most nodes [`CubeLod::removed`] lets go at a time. The GPU renderer
/// fades each out and frees it with a buffer write of its own (about a
/// microsecond each), so a whole plan let go at once after a teleport cost
/// over 200 ms in one frame at 120,000 nodes.
pub const RELEASE_BATCH: usize = 4096;

/// Progress of the far terrain, for the overlay and benchmarks
/// (`CubeLod::stats`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CubeLodStats {
    /// Nodes in the current plan.
    pub planned: usize,
    /// Planned nodes handed out and drawn (or, with nothing to draw, known
    /// empty: see `uniform_skipped`).
    pub ready: usize,
    /// Planned nodes still to sample, mesh or upload.
    pub pending: usize,
    /// Planned nodes that are uniform and draw nothing (air, fluid), and
    /// nodes the planner left out because the 2D pre-pass proved their whole
    /// subtree uniform air or fluid (never requested).
    pub uniform_skipped: usize,
    /// Nodes generated on the GPU, ever.
    pub gpu_generated: u64,
    /// Nodes sampled on the CPU and uploaded into the GPU pool for the GPU
    /// mesher, ever (an Earth world's far nodes on the GPU path).
    pub gpu_uploaded: u64,
    /// Nodes meshed on the GPU, ever (the renderer's count).
    pub gpu_meshed: u64,
    /// Bytes of the quad arena and the node pool in use.
    pub arena_bytes: u64,
    pub pool_bytes: u64,
    /// Every planned node is drawn or known empty, and the GPU mesher has
    /// nothing waiting.
    pub complete: bool,
    /// Drawn nodes queued to be made again for late content (structures
    /// found after the terrain: [`LateContent`]), ever.
    pub late_refreshed: u64,
    /// GPU descents run (`CubeLod::set_gpu_descent`), ever.
    pub descent_runs: u64,
    /// Nodes the last descent proved uniform (air, fluid or rock).
    pub descent_uniform: usize,
    /// Right-sized nodes the last descent found (the leaves of the GPU's
    /// levels, before the loaded cubes and the classes are applied).
    pub descent_leaves: usize,
    /// Wall time of the last descent, with its readback, in milliseconds.
    pub descent_ms: f32,
    /// The last descent filled a list: its classes are partial (safe).
    pub descent_overflow: bool,
    /// When the plan last changed.
    pub plan_changed: Option<Instant>,
    /// When the current plan last became complete; none while it is not.
    pub plan_complete: Option<Instant>,
    /// When every part of the plan's footprint last had a delivered node
    /// drawn over it (no holes, the coverage layer counting); none while some
    /// part has nothing.
    pub plan_covered: Option<Instant>,
    /// Planned nodes with nothing drawn over their place yet.
    pub holes: usize,
    /// Coverage nodes the last big change asked for first.
    pub cover_nodes: usize,
    /// The coverage layer is holding the rest of the plan back.
    pub covering: bool,
    /// The most leaves the plan may have now.
    pub budget: usize,
    /// The budget kept the plan coarser than its target.
    pub limited: bool,
    /// The target in pixels per voxel the plan was made for, and the size it
    /// reaches (bigger when the budget binds).
    pub target_px: f32,
    pub voxel_px: f32,
    /// The vertical field of view of the last update, in degrees (the spyglass
    /// narrows it).
    pub fov_deg: f32,
    /// The planner's time for the last plan, in milliseconds.
    pub plan_ms: f32,
    /// Surface tiles in the plan (counted in `planned` too), and of them
    /// those drawn.
    pub tiles: usize,
    pub tiles_ready: usize,
    /// The renderer's tiles: in its pool, their bytes, and those complex.
    pub tiles_resident: usize,
    pub tile_bytes: u64,
    pub tiles_complex: usize,
    /// The tile budget kept some tiles whole.
    pub tiles_limited: bool,
    /// Planned nodes that would split but wait to be drawn first: the
    /// refining still to come, one level a plan.
    pub held: usize,
    /// Far nodes drawn over loaded near cubes now, and the longest any of
    /// them has been (milliseconds): the far view lingering over near ground.
    pub linger_nodes: usize,
    pub linger_ms: f32,
    /// Per-frame delivery budget ([`DeliverBudget`]): finished nodes held
    /// back for a later frame now, and the frames that reached it.
    pub deliver_deferred: usize,
    pub deliver_frames_capped: u64,
    // The far view's waste counters, ever (always counted: `Waste`).
    /// Keys the workers took, nodes they made, and keys that failed (no node).
    pub jobs: u64,
    pub built: u64,
    pub failed_jobs: u64,
    /// Nodes made that are uniform (air, fluid or rock), and of them air.
    pub built_uniform: u64,
    pub built_empty: u64,
    /// Nodes newly handed out, and of them those handed out before, released
    /// since (made again).
    pub delivered: u64,
    pub rebuilt: u64,
    /// Changed plans put in, the nodes they added and dropped, the added ones
    /// still drawn (free), the added ones drawn before and released since
    /// (made again unless cached), and the dropped ones never drawn.
    pub plan_installs: u64,
    pub plan_added: u64,
    pub plan_removed: u64,
    pub plan_reused: u64,
    pub plan_readded: u64,
    pub plan_dropped_unbuilt: u64,
    /// Nodes released, and of them those released within
    /// `HORIZON_FAR_SHORT_MS` of being handed out (by level, 7 and up in the
    /// last).
    pub released: u64,
    pub short_lived: u64,
    pub short_lived_levels: [u64; 8],
    /// Late refreshes not queued because the plan splits the node next
    /// (`HORIZON_FAR_LATE_SKIP_HELD`), dropped because the plan dropped the
    /// node while it waited (`HORIZON_FAR_LATE_DROP`), released within
    /// `HORIZON_FAR_SHORT_MS` of being queued, and queued for a box that is no
    /// node's own extent (the old, not exact path).
    pub late_skipped_held: u64,
    pub late_dropped: u64,
    pub late_wasted: u64,
    pub late_other: u64,
    /// Failed nodes asked for again.
    pub retries: u64,
    /// Times the whole plan was queued again, and the keys that pushed.
    pub refills: u64,
    pub refill_keys: u64,
    /// Plan jobs sent, by the reason found first: the camera cube, range,
    /// loaded box or detail moved; new classes; the refine rule (held nodes
    /// and a node came in); only the timer.
    pub replans: u64,
    pub replans_moved: u64,
    pub replans_classes: u64,
    pub replans_refine: u64,
    pub replans_timer: u64,
    /// The workers' node cache: lookups found and not, nodes pushed out for
    /// room, forgotten for changed cubes, whole clears (ever); the nodes it
    /// holds now and their bytes.
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_evicted: u64,
    pub cache_forgotten: u64,
    pub cache_cleared: u64,
    pub cache_nodes: usize,
    pub cache_bytes: u64,
}

/// What a worker made of a node: a CPU mesh, or (the GPU path) a node for the
/// renderer's GPU mesher.
enum Product {
    Mesh(SectionMesh),
    Gpu(GpuNode),
}

/// The most the main thread takes in one `CubeLod::update` of the nodes the far
/// workers finished: how many, their mesh vertex bytes, and how long the taking
/// runs. The rest waits for the next frame (`CubeLod::backlog`). A walk
/// finishes a few nodes a frame; a teleport or a far-detail change finishes
/// thousands, and handing them all out in one frame was a spike (the renderer
/// uploads what it is handed, `upload_cube_lods`, at 2 ms a frame).
/// `HORIZON_FAR_DELIVER_MAX` (nodes), `HORIZON_FAR_DELIVER_BYTES` and
/// `HORIZON_FAR_DELIVER_US`; a knob of 0 is no limit of that kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeliverBudget {
    nodes: usize,
    bytes: usize,
    time: Duration,
}

/// What `update` has taken so far this frame.
#[derive(Clone, Copy, Debug, Default)]
struct Spent {
    nodes: usize,
    bytes: usize,
}

impl DeliverBudget {
    /// Nodes a frame. A fill after a teleport finishes about a thousand a
    /// second: this is far above a walk's rate and keeps a fill's pace.
    const NODES: u64 = 256;
    const BYTES: u64 = 4 << 20;
    const US: u64 = 1000;

    fn get() -> Self {
        Self {
            nodes: far_knob("HORIZON_FAR_DELIVER_MAX", Self::NODES) as usize,
            bytes: far_knob("HORIZON_FAR_DELIVER_BYTES", Self::BYTES) as usize,
            time: Duration::from_micros(far_knob("HORIZON_FAR_DELIVER_US", Self::US)),
        }
    }

    /// Whether `spent` (and `elapsed` since the frame began taking) used it up.
    fn reached(&self, spent: &Spent, elapsed: Duration) -> bool {
        (self.nodes > 0 && spent.nodes >= self.nodes)
            || (self.bytes > 0 && spent.bytes >= self.bytes)
            || (!self.time.is_zero() && elapsed >= self.time)
    }
}

struct Done {
    key: CubeKey,
    product: Option<Product>,
    /// [`CubeNode::content_hash`] of the node made, when it was made from a
    /// CPU node (not a GPU-generated slot): a late refresh that comes out the
    /// same as the drawn node is not swapped in.
    hash: Option<u64>,
}

/// What a node holds on the GPU path.
#[derive(Clone, Copy, Debug)]
pub enum GpuContent {
    /// Every voxel the same, with no cover: it needs no slot and is meshed
    /// without the GPU (air and fluid draw nothing, rock its six faces).
    Uniform(Voxel),
    /// A slot of the GPU node pool (`PoolHandle`) the renderer owns from here:
    /// it frees the slot once the node is meshed, or when the node goes.
    Slot(u32),
}

/// A node handed to the renderer's GPU mesher (`CubeLod::take_gpu_nodes`).
#[derive(Debug)]
pub struct GpuNode {
    pub key: CubeKey,
    pub content: GpuContent,
    /// The columns the sky is blocked over (`CubeNode::covered_mask`).
    pub cover: Option<Box<CoverMask>>,
    /// Where each column's fluid ends inside the node (`CubeNode::fluid_tops`:
    /// `z << 4 | x`, [`FLUID_TOP_SCALE`] units per node side over the node's
    /// bottom, [`NO_FLUID_TOP`] for none). Filled from a CPU node; `None` for
    /// a generated slot. The GPU mesher draws them as the CPU mesher does
    /// (`gfx::cube_mesh::MeshJob::fluid_tops`; a [`GpuContent::Uniform`] node's
    /// by `gfx::cube_mesh::uniform_quads`): a column with a top draws its
    /// fluid's top on the node's top, and a top inside a fluid voxel is its
    /// surface, to 1/256 of a voxel (the units' own precision).
    pub fluid_tops: Option<Box<FluidTops>>,
    /// The distinct block states of a CPU node (empty for a generated one:
    /// the generator's states are known from the start).
    pub states: Vec<u32>,
    /// [`CubeNode::content_hash`] of the CPU node it was made from.
    pub hash: Option<u64>,
}

/// What `CubeLod::take_gpu_nodes` hands the renderer.
#[derive(Debug, Default)]
pub struct GpuBatch {
    pub nodes: Vec<GpuNode>,
    /// Material records for block states the renderer has not seen
    /// (`gfx::cube_mesh::material_record`).
    pub materials: Vec<(u32, [u32; crate::gfx::cube_mesh::MATERIAL_STRIDE])>,
    /// The biome colour table, the first time.
    pub biomes: Option<Vec<u32>>,
}

// ------------------------------------------------------------- work order

// The far view's work order (Lyle, 2026-11-25): "prioritize what is in your
// actual vision, and half the priority based on proximity to the player ...
// based on where the player's crosshair is". Every job scores
//
//     FAR_PRIORITY_VIEW x crosshair weight + FAR_PRIORITY_NEAR x nearness
//
// and the highest goes first; of equal scores, the bigger on-screen error.
// Two kinds of job stay ahead of the score: nodes replacing far nodes drawn
// over loaded near ground (those show through the near terrain), nearest
// first, then holes (nothing drawn there at all), by score.
//
// The knobs for Lyle to tune are these five constants.

/// The weight of where the crosshair is.
pub const FAR_PRIORITY_VIEW: f64 = 1.0;
/// The weight of how near the player a job is.
pub const FAR_PRIORITY_NEAR: f64 = 0.5;
/// The crosshair weight at the screen's corners (1 at the crosshair, falling
/// smoothly to this) ...
pub const FAR_PRIORITY_EDGE: f64 = 0.35;
/// ... and straight behind the player (falling evenly from the corners).
pub const FAR_PRIORITY_BEHIND: f64 = 0.05;
/// Nearness: 1 at the player, a half this many blocks away, a fifth at four
/// times it.
pub const FAR_PRIORITY_NEAR_BLOCKS: f64 = 512.0;

/// The crosshair weight of a job `angle` radians off the look direction, on
/// a screen whose corners are `edge` radians off it.
pub fn crosshair_weight(angle: f64, edge: f64) -> f64 {
    let edge = edge.clamp(1e-3, std::f64::consts::PI - 1e-3);
    let angle = angle.clamp(0.0, std::f64::consts::PI);
    if angle <= edge {
        let t = angle / edge;
        1.0 - (1.0 - FAR_PRIORITY_EDGE) * t * t
    } else {
        let t = (angle - edge) / (std::f64::consts::PI - edge);
        FAR_PRIORITY_EDGE + (FAR_PRIORITY_BEHIND - FAR_PRIORITY_EDGE) * t
    }
}

/// How near the player a job `distance` blocks away is: 1 at the player.
pub fn nearness(distance: f64) -> f64 {
    1.0 / (1.0 + distance.max(0.0) / FAR_PRIORITY_NEAR_BLOCKS)
}

/// A job's score from its crosshair angle and distance.
pub fn far_priority(angle: f64, edge: f64, distance: f64) -> f64 {
    FAR_PRIORITY_VIEW * crosshair_weight(angle, edge) + FAR_PRIORITY_NEAR * nearness(distance)
}

/// What the queue's order was made for: the look direction (a unit vector)
/// and the angle from it to the screen's corners ([`FarView::half_diagonal`],
/// smaller when zoomed in).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    pub forward: [f64; 3],
    pub edge: f64,
}

impl Look {
    pub fn of(view: &FarView) -> Self {
        Self {
            forward: view.forward,
            edge: view.half_diagonal(),
        }
    }
    /// Whether the order made for `self` is out of date for `now`: the view
    /// turned past [`TURN`], or zoomed by a tenth.
    fn moved(&self, now: &Look) -> bool {
        angle_between(self.forward, now.forward) > TURN
            || now.edge > self.edge * 1.1
            || self.edge > now.edge * 1.1
    }
}

impl Default for Look {
    fn default() -> Self {
        Self {
            forward: [0.0, 0.0, 1.0],
            edge: FarView::standard(1.0).half_diagonal(),
        }
    }
}

/// The angle from `look` to the nearest part of `node` seen from `camera`
/// (0 when the node takes in the look direction or the camera).
fn node_angle(camera: [f64; 3], forward: [f64; 3], node: CubeKey) -> f64 {
    let size = f64::from(node.size());
    let o = node.origin();
    let d = [0, 1, 2].map(|a| f64::from(o[a]) + size / 2.0 - camera[a]);
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let radius = size * 0.866;
    if len <= radius {
        return 0.0;
    }
    (angle_between(forward, d) - (radius / len).asin()).max(0.0)
}

/// The score ([`far_priority`]) of a job `key` seen from `camera` with
/// `look`: for other queues of far work (the Earth trees pass) to keep the
/// same order.
pub fn job_score(camera: [f64; 3], look: Look, key: CubeKey) -> f64 {
    far_priority(
        node_angle(camera, look.forward, key),
        look.edge,
        key.distance(camera),
    )
}

/// A job's on-screen error: its voxels' size over their distance.
fn screen_error(camera: [f64; 3], node: CubeKey) -> f64 {
    f64::from(node.size()) / node.distance(camera).max(1.0)
}

/// The queue keeps its jobs in buckets of a direction cell and a nearness
/// ring each, so a turn only reorders the buckets (no job is scored again):
/// 48 azimuths by 24 elevation bands, 7.5 degrees a cell, plus cell 0 for the
/// nodes round the camera, in view whichever way it looks; 16 rings.
const AZIMUTHS: usize = 48;
const BANDS: usize = 24;
const CELLS: usize = 1 + AZIMUTHS * BANDS;
const RINGS: usize = 16;
const BUCKETS: usize = CELLS * RINGS;

/// The direction cell of `node` seen from `camera`.
fn cell(camera: [f64; 3], node: CubeKey) -> usize {
    let size = f64::from(node.size());
    let o = node.origin();
    let d = [0, 1, 2].map(|a| f64::from(o[a]) + size / 2.0 - camera[a]);
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if len <= size {
        return 0;
    }
    let turn = (d[2].atan2(d[0]) + std::f64::consts::PI) / std::f64::consts::TAU;
    let azimuth = ((turn * AZIMUTHS as f64) as usize).min(AZIMUTHS - 1);
    let rise = ((d[1] / len).clamp(-1.0, 1.0).asin() / std::f64::consts::PI) + 0.5;
    let band = ((rise * BANDS as f64) as usize).min(BANDS - 1);
    1 + band * AZIMUTHS + azimuth
}

/// The middle direction of cell `c` (not 0).
fn cell_axis(c: usize) -> [f64; 3] {
    let (band, azimuth) = ((c - 1) / AZIMUTHS, (c - 1) % AZIMUTHS);
    let angle =
        (azimuth as f64 + 0.5) / AZIMUTHS as f64 * std::f64::consts::TAU - std::f64::consts::PI;
    let rise = ((band as f64 + 0.5) / BANDS as f64 - 0.5) * std::f64::consts::PI;
    let (y, h) = (rise.sin(), rise.cos());
    [h * angle.cos(), y, h * angle.sin()]
}

/// The nearness ring of a job `distance` blocks away.
fn ring(distance: f64) -> usize {
    ((nearness(distance) * RINGS as f64) as usize).min(RINGS - 1)
}

/// The bucket of `node` seen from `camera`.
fn bucket(camera: [f64; 3], node: CubeKey) -> u16 {
    (cell(camera, node) * RINGS + ring(node.distance(camera))) as u16
}

/// The buckets in the order the queue takes them for `look`: by the score of
/// the cell's middle and the ring's middle, highest first.
fn bucket_order(look: Look) -> Vec<u16> {
    let cell_weight: Vec<f64> = (0..CELLS)
        .map(|c| {
            if c == 0 {
                1.0
            } else {
                // A cell's nearest edge, not its middle: half a cell's width.
                let half = std::f64::consts::PI / BANDS as f64 / 2.0;
                let angle = (angle_between(look.forward, cell_axis(c)) - half).max(0.0);
                crosshair_weight(angle, look.edge)
            }
        })
        .collect();
    // Scores to 1/65536 of the highest, then the bucket: one u32 to sort.
    let top = (FAR_PRIORITY_VIEW + FAR_PRIORITY_NEAR).max(1e-9);
    let mut by_score: Vec<u32> = (0..BUCKETS)
        .map(|b| {
            let (c, r) = (b / RINGS, b % RINGS);
            let near = (r as f64 + 0.5) / RINGS as f64;
            let score = FAR_PRIORITY_VIEW * cell_weight[c] + FAR_PRIORITY_NEAR * near;
            let q = ((score / top).clamp(0.0, 1.0) * 65535.0) as u32;
            (65535 - q) << 15 | b as u32
        })
        .collect();
    by_score.sort_unstable();
    by_score.into_iter().map(|e| (e & 0x7fff) as u16).collect()
}

fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let la = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let lb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if la * lb <= 0.0 || !(la * lb).is_finite() {
        return 0.0;
    }
    (dot / (la * lb)).clamp(-1.0, 1.0).acos()
}

/// The jobs, in the order the workers take them.
struct Queue {
    /// Taken first, in order: nodes replacing far nodes over near ground,
    /// the coverage layer, holes, and (at the front) a worker's put-backs.
    first: VecDeque<CubeKey>,
    /// The rest by bucket ([`bucket`]), the bigger on-screen error first in
    /// each.
    buckets: Vec<VecDeque<CubeKey>>,
    /// The order the buckets are taken in ([`bucket_order`]), and each
    /// bucket's place in it.
    order: Vec<u16>,
    rank: Vec<u16>,
    /// Every bucket before this place in `order` is empty.
    cursor: usize,
    /// Queued keys the plan dropped since: skipped when taken.
    dropped: FxHashSet<CubeKey>,
    running: FxHashSet<CubeKey>,
    /// Drawn nodes to make again ([`LateContent`]): taken after the rest,
    /// and kept when the plan is redone.
    later: VecDeque<CubeKey>,
}

impl Default for Queue {
    fn default() -> Self {
        let mut queue = Self {
            first: VecDeque::new(),
            buckets: (0..BUCKETS).map(|_| VecDeque::new()).collect(),
            order: Vec::new(),
            rank: Vec::new(),
            cursor: 0,
            dropped: FxHashSet::default(),
            running: FxHashSet::default(),
            later: VecDeque::new(),
        };
        queue.reorder(bucket_order(Look::default()));
        queue
    }
}

impl Queue {
    fn clear(&mut self) {
        self.first.clear();
        for b in &mut self.buckets {
            b.clear();
        }
        self.cursor = 0;
        self.dropped.clear();
    }
    /// Takes the buckets in `order` from now on.
    fn reorder(&mut self, order: Vec<u16>) {
        let mut rank = vec![0u16; BUCKETS];
        for (i, &b) in order.iter().enumerate() {
            rank[usize::from(b)] = i as u16;
        }
        self.order = order;
        self.rank = rank;
        self.cursor = 0;
    }
    /// Queues `key` in bucket `b`, behind the others there.
    fn push(&mut self, b: u16, key: CubeKey) {
        self.buckets[usize::from(b)].push_back(key);
        self.cursor = self.cursor.min(usize::from(self.rank[usize::from(b)]));
    }
    /// The next job.
    fn pop(&mut self) -> Option<CubeKey> {
        loop {
            let mut next = self.first.pop_front();
            while next.is_none() && self.cursor < self.order.len() {
                let b = usize::from(self.order[self.cursor]);
                next = self.buckets[b].pop_front();
                if next.is_none() {
                    self.cursor += 1;
                }
            }
            if next.is_none() {
                next = self.later.pop_front();
            }
            let key = next?;
            if self.dropped.remove(&key) || self.running.contains(&key) {
                continue;
            }
            return Some(key);
        }
    }
    #[cfg(test)]
    fn drain(&mut self) -> Vec<CubeKey> {
        std::iter::from_fn(|| self.pop()).collect()
    }
}

/// A node kept in the cache: the node itself (uniform nodes, nodes of more
/// than 256 kinds of voxel, or any node with packing off), or a dense node
/// packed to one byte a voxel plus its palette (about 4 KiB, not 32).
#[derive(Clone)]
enum Entry {
    Node(Arc<CubeNode>),
    Packed(Arc<Packed>),
}

/// A dense node as a palette of its distinct voxels and, per voxel, the
/// palette index (same order as the node's voxels), with its cover and
/// fluid tops as they were.
struct Packed {
    key: CubeKey,
    palette: Vec<Voxel>,
    indices: Box<[u8; VOLUME]>,
    covered: Option<Box<CoverMask>>,
    fluid_tops: Option<Box<FluidTops>>,
}

impl Entry {
    /// The entry for `node`: packed when `pack` and it is dense with at most
    /// 256 distinct voxels. Takes the time of a pass over 4096 voxels, so
    /// call it outside the cache's lock.
    fn pack(node: Arc<CubeNode>, pack: bool) -> Self {
        let Voxels::Dense(dense) = &node.voxels else {
            return Self::Node(node);
        };
        if !pack {
            return Self::Node(node);
        }
        let mut palette: Vec<Voxel> = Vec::new();
        let mut index_of: FxHashMap<Voxel, u8> = FxHashMap::default();
        let mut indices = Box::new([0u8; VOLUME]);
        for (slot, voxel) in indices.iter_mut().zip(dense.iter()) {
            *slot = match index_of.get(voxel) {
                Some(&i) => i,
                None => {
                    if palette.len() == 256 {
                        return Self::Node(node);
                    }
                    let i = palette.len() as u8;
                    palette.push(*voxel);
                    index_of.insert(*voxel, i);
                    i
                }
            };
        }
        Self::Packed(Arc::new(Packed {
            key: node.key,
            palette,
            indices,
            covered: node.covered.clone(),
            fluid_tops: node.fluid_tops.clone(),
        }))
    }

    /// The node this entry holds (a packed one is built again: call it
    /// outside the cache's lock).
    fn unpack(&self) -> Arc<CubeNode> {
        match self {
            Self::Node(node) => node.clone(),
            Self::Packed(p) => {
                let mut dense = Box::new([p.palette[0]; VOLUME]);
                for (voxel, &i) in dense.iter_mut().zip(p.indices.iter()) {
                    *voxel = p.palette[usize::from(i)];
                }
                Arc::new(
                    CubeNode::from_dense(p.key, dense)
                        .with_covered(p.covered.clone())
                        .with_fluid_tops(p.fluid_tops.clone()),
                )
            }
        }
    }

    fn key(&self) -> CubeKey {
        match self {
            Self::Node(node) => node.key,
            Self::Packed(p) => p.key,
        }
    }

    /// An estimate of the bytes the entry holds.
    fn bytes(&self) -> u64 {
        let (voxels, covered, tops) = match self {
            Self::Node(node) => (
                match &node.voxels {
                    Voxels::Uniform(_) => 0,
                    Voxels::Dense(_) => VOLUME * std::mem::size_of::<Voxel>(),
                },
                node.covered.is_some(),
                node.fluid_tops.is_some(),
            ),
            Self::Packed(p) => (
                VOLUME + p.palette.len() * std::mem::size_of::<Voxel>(),
                p.covered.is_some(),
                p.fluid_tops.is_some(),
            ),
        };
        let covered = if covered { 32 } else { 0 };
        let tops = if tops { 512 } else { 0 };
        (std::mem::size_of::<CubeNode>() + 64 + voxels + covered + tops) as u64
    }
}

/// An entry with the tick of its last use and its estimated bytes.
struct Slot {
    entry: Entry,
    tick: u64,
    bytes: u64,
}

/// The nodes the workers keep after sampling (see [`CACHE_NODES`]): bounded
/// by bytes (`HORIZON_FAR_CACHE_MB`, dense nodes packed unless
/// `HORIZON_FAR_CACHE_PACK=0`), least recently used out first. Every call is
/// O(1) amortized but [`Cache::forget`] (O(levels) per changed cube); packing
/// and unpacking are left to the caller, outside the lock.
struct Cache {
    nodes: FxHashMap<CubeKey, Slot>,
    /// Uses, oldest first: (tick, key). An item whose tick is not its key's
    /// current tick is stale (the key was used since, or went) and skipped.
    log: VecDeque<(u64, CubeKey)>,
    tick: u64,
    bytes: u64,
    budget: u64,
    pack: bool,
    max_nodes: usize,
    /// Counters, ever (the far view's waste counters: `CubeLodStats`):
    /// lookups found and not found, nodes pushed out for room, nodes dropped
    /// because cubes under them changed (`Changes::Cubes`), and whole clears
    /// (`Changes::All`).
    hits: u64,
    misses: u64,
    evicted: u64,
    forgotten: u64,
    cleared: u64,
}

impl Default for Cache {
    /// `HORIZON_FAR_CACHE_MB` (192; 0: the old cache, 1024 nodes unpacked) and
    /// `HORIZON_FAR_CACHE_PACK` (1).
    fn default() -> Self {
        let mb = far_knob("HORIZON_FAR_CACHE_MB", 192);
        if mb == 0 {
            return Self::with_budget(u64::MAX, false, CACHE_NODES);
        }
        let pack = far_knob("HORIZON_FAR_CACHE_PACK", 1) != 0;
        Self::with_budget(mb.saturating_mul(1 << 20), pack, usize::MAX)
    }
}

impl Cache {
    /// A cache of at most `bytes` (estimated) and `max_nodes` nodes.
    fn with_budget(bytes: u64, pack: bool, max_nodes: usize) -> Self {
        Self {
            nodes: FxHashMap::default(),
            log: VecDeque::new(),
            tick: 0,
            bytes: 0,
            budget: bytes,
            pack,
            max_nodes,
            hits: 0,
            misses: 0,
            evicted: 0,
            forgotten: 0,
            cleared: 0,
        }
    }
    /// Nodes kept now, and the bytes they take (an estimate).
    fn len(&self) -> usize {
        self.nodes.len()
    }
    fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Drops the nodes holding any of `cubes`: for each cube the node of each
    /// level that holds it (not a scan of the cache).
    fn forget(&mut self, cubes: &[SectionPos]) {
        for &cube in cubes {
            for level in 0..=MAX_CUBE_LEVEL {
                if self.remove_key(CubeKey::containing(level, cube)) {
                    self.forgotten += 1;
                }
            }
        }
    }
    /// Drops node `key` (counts nothing).
    fn remove(&mut self, key: CubeKey) {
        self.remove_key(key);
    }
    fn remove_key(&mut self, key: CubeKey) -> bool {
        match self.nodes.remove(&key) {
            Some(slot) => {
                self.bytes -= slot.bytes;
                true
            }
            None => false,
        }
    }
    /// Drops everything (counts nothing).
    fn clear(&mut self) {
        self.nodes.clear();
        self.log.clear();
        self.bytes = 0;
    }
    /// The entry of `key` if kept (counts a hit or a miss, and the use).
    /// Cheap: unpack it outside the lock.
    fn lookup(&mut self, key: CubeKey) -> Option<Entry> {
        let Some(slot) = self.nodes.get_mut(&key) else {
            self.misses += 1;
            return None;
        };
        self.hits += 1;
        self.tick += 1;
        slot.tick = self.tick;
        let entry = slot.entry.clone();
        self.log.push_back((self.tick, key));
        self.compact();
        Some(entry)
    }
    /// Keeps `entry` (packed already), pushing out the least recently used
    /// nodes until the budget holds; never the one just kept.
    fn put(&mut self, entry: Entry) {
        let key = entry.key();
        let bytes = entry.bytes();
        self.tick += 1;
        let tick = self.tick;
        if let Some(old) = self.nodes.insert(key, Slot { entry, tick, bytes }) {
            self.bytes -= old.bytes;
        }
        self.bytes += bytes;
        self.log.push_back((tick, key));
        while (self.bytes > self.budget || self.nodes.len() > self.max_nodes)
            && self.nodes.len() > 1
        {
            let Some((t, oldest)) = self.log.pop_front() else {
                break;
            };
            if oldest == key || self.nodes.get(&oldest).is_none_or(|s| s.tick != t) {
                continue;
            }
            self.remove_key(oldest);
            self.evicted += 1;
        }
        self.compact();
    }
    /// Keeps the log near the size of the cache: stale items out.
    fn compact(&mut self) {
        if self.log.len() > self.nodes.len() * 4 + 1024 {
            let nodes = &self.nodes;
            self.log
                .retain(|(t, key)| nodes.get(key).is_some_and(|s| s.tick == *t));
        }
    }
    #[cfg(test)]
    fn get(&mut self, key: CubeKey) -> Option<Arc<CubeNode>> {
        self.lookup(key).map(|entry| entry.unpack())
    }
    #[cfg(test)]
    fn insert(&mut self, node: Arc<CubeNode>) {
        let entry = Entry::pack(node, self.pack);
        self.put(entry);
    }
}

/// The cached node at `key`, unpacked outside the cache's lock.
fn cache_get(shared: &Shared, key: CubeKey) -> Option<Arc<CubeNode>> {
    let entry = shared.cache.lock().unwrap().lookup(key);
    entry.map(|entry| entry.unpack())
}

/// Keeps `node` in the cache, packed before the cache's lock is taken.
fn cache_put(shared: &Shared, node: Arc<CubeNode>) {
    let pack = shared.cache.lock().unwrap().pack;
    let entry = Entry::pack(node, pack);
    shared.cache.lock().unwrap().put(entry);
}

/// What the classifier thread has found of nodes, and what it is to look at.
#[derive(Default)]
struct Classes {
    known: KnownClasses,
    queue: VecDeque<CubeKey>,
    queued: FxHashSet<CubeKey>,
}

mod known_classes;
use known_classes::KnownClasses;

/// Counters of the far view's work and of what of it was wasted (nodes made
/// again, made for nothing, plans redone), always counted, read by
/// [`CubeLod::stats`] and the profile log ([`profile::maybe_log_waste`]).
/// `$f` is a counter; `short_lived_levels` counts by level on top.
macro_rules! waste_counters {
    ($($f:ident),* $(,)?) => {
        #[derive(Default)]
        struct Waste {
            $($f: std::sync::atomic::AtomicU64,)*
            short_lived_levels: [std::sync::atomic::AtomicU64; 8],
        }

        /// [`Waste`] read at one moment.
        #[derive(Clone, Copy, Default)]
        struct WasteSnap {
            $($f: u64,)*
            short_lived_levels: [u64; 8],
        }

        impl Waste {
            fn snap(&self) -> WasteSnap {
                WasteSnap {
                    $($f: self.$f.load(Ordering::Relaxed),)*
                    short_lived_levels: std::array::from_fn(|i| {
                        self.short_lived_levels[i].load(Ordering::Relaxed)
                    }),
                }
            }
        }

        impl WasteSnap {
            /// What came since `before`.
            fn since(&self, before: &WasteSnap) -> WasteSnap {
                WasteSnap {
                    $($f: self.$f.saturating_sub(before.$f),)*
                    short_lived_levels: std::array::from_fn(|i| {
                        self.short_lived_levels[i].saturating_sub(before.short_lived_levels[i])
                    }),
                }
            }
        }
    };
}

waste_counters!(
    jobs,
    built,
    failed_jobs,
    built_uniform,
    built_empty,
    delivered,
    rebuilt,
    plan_installs,
    plan_added,
    plan_removed,
    plan_reused,
    plan_readded,
    plan_dropped_unbuilt,
    released,
    short_lived,
    late_refreshed,
    late_skipped_held,
    late_dropped,
    late_wasted,
    late_other,
    late_same,
    retries,
    refills,
    refill_keys,
    replans,
    replans_moved,
    replans_classes,
    replans_refine,
    replans_timer,
);

/// Adds `n` to a [`Waste`] counter.
fn bump(counter: &std::sync::atomic::AtomicU64, n: usize) {
    counter.fetch_add(n as u64, Ordering::Relaxed);
}

impl Shared {
    fn new() -> Self {
        Self {
            classes: Mutex::default(),
            classes_wake: Condvar::new(),
            queue: Mutex::default(),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            cache: Mutex::default(),
            gpu_path: AtomicBool::new(false),
            gpu_unfit: AtomicBool::new(false),
            descent_on: AtomicBool::new(false),
            descent: Mutex::default(),
            descent_wake: Condvar::new(),
            classes_dirty: AtomicBool::new(false),
            near_loading: AtomicBool::new(false),
            near_seen_ms: std::sync::atomic::AtomicU64::new(0),
            near_epoch: Instant::now(),
            planner: Mutex::default(),
            planner_wake: Condvar::new(),
            waste: Waste::default(),
        }
    }
}

/// The request the descent thread is to run next (the latest wins), and what
/// the last run found.
#[derive(Default)]
struct DescentSlot {
    request: Option<DescentRequest>,
    busy: bool,
    runs: u64,
    uniform: usize,
    leaves: usize,
    millis: f32,
    overflow: bool,
}

struct Shared {
    classes: Mutex<Classes>,
    classes_wake: Condvar,
    queue: Mutex<Queue>,
    wake: Condvar,
    stop: AtomicBool,
    cache: Mutex<Cache>,
    /// The workers make [`GpuNode`]s (`CubeLod::set_gpu_path`).
    gpu_path: AtomicBool,
    /// A worker of a backend without a generator met a CPU node the GPU
    /// mesher cannot draw ([`needs_cpu_mesh`]): the world goes back to the CPU
    /// mesher for good ([`CubeLod::gpu_path_allowed`]).
    gpu_unfit: AtomicBool,
    /// Classes come from the GPU descent (`CubeLod::set_gpu_descent`), not
    /// from the classifier thread.
    descent_on: AtomicBool,
    descent: Mutex<DescentSlot>,
    descent_wake: Condvar,
    /// The descent added classes the plan has not looked at yet.
    classes_dirty: AtomicBool,
    /// The planner thread's job and answer (one at a time).
    planner: Mutex<PlanSlot>,
    planner_wake: Condvar,
    /// The near terrain is loading: the workers step behind it
    /// (`CubeLod::set_near_loading`).
    near_loading: AtomicBool,
    /// When it last was (ms since `near_epoch`): `terrain_loading` flickers
    /// off between the batches of columns arriving.
    near_seen_ms: std::sync::atomic::AtomicU64,
    near_epoch: Instant,
    /// What the far view did and wasted ([`Waste`]).
    waste: Waste,
}

// ---------------------------------------------------------------- planner

/// Which cubes of the loaded box `is_loaded` reported, taken on the game
/// thread for the planner thread.
#[derive(Clone, Debug, Default)]
struct LoadedSnapshot {
    lo: [i32; 3],
    size: [i32; 3],
    bits: Vec<u64>,
}

impl LoadedSnapshot {
    fn take(center: SectionPos, loaded: LoadedBox, is_loaded: &dyn Fn(SectionPos) -> bool) -> Self {
        let (h, v) = (loaded.horizontal.max(0), loaded.vertical.max(0));
        let lo = [center.x - h, center.y - v, center.z - h];
        let size = [2 * h + 1, 2 * v + 1, 2 * h + 1];
        let n = (size[0] as usize) * (size[1] as usize) * (size[2] as usize);
        let mut bits = vec![0u64; n.div_ceil(64)];
        let mut i = 0usize;
        for y in 0..size[1] {
            for z in 0..size[2] {
                for x in 0..size[0] {
                    if is_loaded(SectionPos::new(lo[0] + x, lo[1] + y, lo[2] + z)) {
                        bits[i >> 6] |= 1 << (i & 63);
                    }
                    i += 1;
                }
            }
        }
        Self { lo, size, bits }
    }
    fn get(&self, cube: SectionPos) -> bool {
        let r = [
            i64::from(cube.x) - i64::from(self.lo[0]),
            i64::from(cube.y) - i64::from(self.lo[1]),
            i64::from(cube.z) - i64::from(self.lo[2]),
        ];
        if (0..3).any(|a| r[a] < 0 || r[a] >= i64::from(self.size[a])) {
            return false;
        }
        let i = ((r[1] * i64::from(self.size[2]) + r[2]) * i64::from(self.size[0]) + r[0]) as usize;
        self.bits[i >> 6] >> (i & 63) & 1 == 1
    }

    /// The snapshot as summed counts ([`LoadedCounts`]), one pass over the box.
    fn counts(&self) -> LoadedCounts {
        let [sx, sy, sz] = self.size.map(|v| v.max(0) as usize);
        let (px, pz) = (sx + 1, sz + 1);
        let mut sum = vec![0u32; px * (sy + 1) * pz];
        let at = |x: usize, y: usize, z: usize| (y * pz + z) * px + x;
        let mut i = 0usize;
        for y in 0..sy {
            for z in 0..sz {
                for x in 0..sx {
                    let bit = (self.bits[i >> 6] >> (i & 63) & 1) as u32;
                    i += 1;
                    sum[at(x + 1, y + 1, z + 1)] = bit
                        + sum[at(x, y + 1, z + 1)]
                        + sum[at(x + 1, y, z + 1)]
                        + sum[at(x + 1, y + 1, z)]
                        - sum[at(x, y, z + 1)]
                        - sum[at(x, y + 1, z)]
                        - sum[at(x + 1, y, z)]
                        + sum[at(x, y, z)];
                }
            }
        }
        LoadedCounts {
            lo: self.lo,
            size: self.size,
            sum,
        }
    }
}

/// A [`LoadedSnapshot`] as summed counts: how many loaded cubes any box holds.
struct LoadedCounts {
    lo: [i32; 3],
    size: [i32; 3],
    sum: Vec<u32>,
}

impl LoadedCounts {
    /// Whether node `key` holds a loaded cube.
    fn holds(&self, key: CubeKey) -> bool {
        let side = 1_i64 << key.level;
        let o = [i64::from(key.x), i64::from(key.y), i64::from(key.z)].map(|v| v * side);
        let mut a = [0usize; 3];
        let mut b = [0usize; 3];
        for i in 0..3 {
            let lo = (o[i] - i64::from(self.lo[i])).clamp(0, i64::from(self.size[i]));
            let hi = (o[i] + side - i64::from(self.lo[i])).clamp(0, i64::from(self.size[i]));
            if lo >= hi {
                return false;
            }
            (a[i], b[i]) = (lo as usize, hi as usize);
        }
        let (px, pz) = (self.size[0] as usize + 1, self.size[2] as usize + 1);
        let s = |x: usize, y: usize, z: usize| i64::from(self.sum[(y * pz + z) * px + x]);
        let n =
            s(b[0], b[1], b[2]) - s(a[0], b[1], b[2]) - s(b[0], a[1], b[2]) - s(b[0], b[1], a[2])
                + s(a[0], a[1], b[2])
                + s(a[0], b[1], a[2])
                + s(b[0], a[1], a[2])
                - s(a[0], a[1], a[2]);
        n > 0
    }
}

/// What a plan job needs for surface tiles.
struct SurfaceJob {
    /// The tiles' headers known so far.
    info: Arc<FxHashMap<TileKey, TileInfo>>,
    sea_level: i32,
    budget: usize,
    /// The tiles in place, for the hysteresis.
    previous: Arc<Vec<TileKey>>,
}

/// A plan for the planner thread to make.
struct PlanJob {
    /// [`CubeLod::clear`] counts up: a job of an older count starts afresh.
    epoch: u64,
    camera: [f64; 3],
    max_distance: f64,
    rule: Rule,
    loaded: LoadedBox,
    snapshot: LoadedSnapshot,
    /// The plan in place, for the hysteresis.
    previous: Arc<Vec<CubeKey>>,
    /// The nodes drawn (or failed: not to wait on), for the refining one
    /// level at a time ([`PlanInput::refined`]); listed on the planner
    /// thread ([`Drawn::keys`]), not the game's.
    drawn: Drawn,
    /// Refine one level at a time from the coverage layer down, so the far
    /// view sharpens step by step in front of the player (Lyle's design). On
    /// by default; `HORIZON_FAR_STEP=0` goes straight to the target instead.
    step: bool,
    /// Plan surface tiles at the coarse levels.
    surface: Option<SurfaceJob>,
}

/// What a stepping plan counts as drawn, as copies of [`CubeLod`]'s sets:
/// the filter ([`Drawn::keys`]) runs on the planner thread.
#[derive(Default)]
struct Drawn {
    delivered: FxHashSet<CubeKey>,
    failed: Vec<CubeKey>,
    stale: FxHashSet<CubeKey>,
}

impl Drawn {
    /// The nodes handed out (or failed), except those inside a stale node
    /// ([`CubeLod::drawn_for_steps`]).
    fn keys(&self) -> Vec<CubeKey> {
        let levels = self.stale.iter().fold(0u32, |m, k| m | 1 << k.level);
        let inside_stale = |key: CubeKey| {
            levels >> (key.level + 1) != 0
                && (key.level + 1..=MAX_CUBE_LEVEL)
                    .filter(|l| levels >> l & 1 == 1)
                    .any(|l| self.stale.contains(&ancestor(key, l)))
        };
        (self.delivered.iter())
            .chain(&self.failed)
            .copied()
            .filter(|&k| !inside_stale(k))
            .collect()
    }
}

/// A far item: a cube node or a surface tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FarKey {
    Cube(CubeKey),
    Tile(TileKey),
}

/// A plan made, as the change from the one before.
struct PlanDone {
    epoch: u64,
    /// The plan, nearest first.
    leaves: Arc<Vec<CubeKey>>,
    changed: bool,
    added: Vec<CubeKey>,
    removed: Vec<CubeKey>,
    /// The plan's tiles, nearest first, and the change.
    tiles: Arc<Vec<TileKey>>,
    tiles_added: Vec<TileKey>,
    tiles_removed: Vec<TileKey>,
    /// The plan with each node's bucket ([`bucket`]), the bigger on-screen
    /// error first (empty when unchanged).
    order: Vec<(u16, CubeKey)>,
    /// The coverage layer's leaves and tiles (empty when unchanged).
    cover: FxHashSet<CubeKey>,
    cover_tiles: FxHashSet<TileKey>,
    /// The plan in tree order ([`tree_order`], empty when unchanged): the
    /// nodes inside any node are a run of it.
    tree: Vec<(u128, CubeKey)>,
    /// Cube leaves and tiles by footprint in Z order ([`footprint_order`]):
    /// those inside a footprint are a run of it.
    footprints: Vec<(u64, u8, FarKey)>,
    pruned: usize,
    limited: bool,
    coarsened: f64,
    tiles_limited: bool,
    /// The plan's nodes waiting to be drawn before they split.
    held: usize,
    /// Those nodes ([`Plan::held_keys`]), also when the plan did not change.
    held_keys: Vec<CubeKey>,
    elapsed: Duration,
}

/// Where a node's corner falls on the Z-order curve of the world's cubes:
/// the nodes inside a node of level `l` are exactly those of a lower level
/// whose code lies in `[code, code + 8^l)`.
fn tree_order(key: CubeKey) -> u128 {
    // Cube coordinates offset to be non-negative: 24 bits each.
    let axis = |c: i32| ((i64::from(c) << key.level) + (1 << 23)) as u64;
    let (x, y, z) = (axis(key.x), axis(key.y), axis(key.z));
    let mut code = 0u128;
    for bit in 0..24 {
        code |= u128::from(x >> bit & 1) << (3 * bit)
            | u128::from(z >> bit & 1) << (3 * bit + 1)
            | u128::from(y >> bit & 1) << (3 * bit + 2);
    }
    code
}

/// [`tree_order`] for the footprint (X and Z) of a node or tile of `level` at
/// `(x, z)`: the footprints inside one of level `l` are exactly those of a
/// level up to `l` whose code lies in `[code, code + 4^l)`.
fn footprint_order(level: u8, x: i32, z: i32) -> u64 {
    let axis = |c: i32| ((i64::from(c) << level) + (1 << 23)) as u64;
    let (x, z) = (axis(x), axis(z));
    let mut code = 0u64;
    for bit in 0..24 {
        code |= (x >> bit & 1) << (2 * bit) | (z >> bit & 1) << (2 * bit + 1);
    }
    code
}

#[derive(Default)]
struct PlanSlot {
    job: Option<PlanJob>,
    busy: bool,
    done: Option<PlanDone>,
}

/// The plan a job asks for, with `rule` and no hysteresis when `fresh`.
fn plan_job(
    job: &PlanJob,
    rule: &Rule,
    fresh: bool,
    class: &dyn Fn(CubeKey) -> NodeClass,
    refined: Option<&FxHashSet<CubeKey>>,
) -> Plan {
    let is_loaded = |cube: SectionPos| job.snapshot.get(cube);
    let info = |key: TileKey| job.surface.as_ref().and_then(|s| s.info.get(&key).copied());
    plan_far(&PlanInput {
        camera: job.camera,
        max_distance: job.max_distance,
        rule,
        loaded: job.loaded,
        is_loaded: &is_loaded,
        previous: if fresh { &[] } else { job.previous.as_slice() },
        class,
        surface: job.surface.as_ref().map(|s| Surface {
            info: &info,
            sea_level: s.sea_level,
            budget: s.budget,
            previous: if fresh { &[] } else { s.previous.as_slice() },
        }),
        refined,
    })
}

/// What a stepping plan may split ([`PlanInput::refined`]): the nodes drawn
/// and those holding one, and every node over a coverage leaf `cover` (the
/// coarse layer comes at once, as before; under it, a level a plan).
fn step_set(drawn: &[CubeKey], cover: &[CubeKey]) -> FxHashSet<CubeKey> {
    let mut out = refined_set(drawn);
    for &leaf in cover {
        let mut up = leaf;
        while up.level < MAX_CUBE_LEVEL {
            up = up.parent();
            if !out.insert(up) {
                break;
            }
        }
    }
    out
}

/// The change from `last` to `next`: what `next` adds, and what it drops.
fn diff<K: Copy + Eq + std::hash::Hash>(
    next: &[K],
    last: &FxHashSet<K>,
) -> (Vec<K>, Vec<K>, FxHashSet<K>) {
    let set: FxHashSet<K> = next.iter().copied().collect();
    let added = next.iter().copied().filter(|k| !last.contains(k)).collect();
    let removed = last.iter().copied().filter(|k| !set.contains(k)).collect();
    (added, removed, set)
}

/// Makes the plan of `job`, and its coverage layer, as the change from
/// `last` and `last_tiles` (the plan before, which becomes this one).
fn make_plan(
    job: &PlanJob,
    last: &mut FxHashSet<CubeKey>,
    last_tiles: &mut FxHashSet<TileKey>,
    class: &dyn Fn(CubeKey) -> NodeClass,
) -> PlanDone {
    let started = Instant::now();
    let cover_rule = job.rule.coarser(COVER_SCALE, COVER_NODES);
    let cover = job
        .step
        .then(|| plan_job(job, &cover_rule, true, class, None));
    let refined = cover
        .as_ref()
        .map(|c| step_set(&job.drawn.keys(), &c.leaves));
    let plan = plan_job(job, &job.rule, false, class, refined.as_ref());
    let (added, removed, next) = diff(&plan.leaves, last);
    let (tiles_added, tiles_removed, next_tiles) = diff(&plan.tiles, last_tiles);
    let changed = !(added.is_empty()
        && removed.is_empty()
        && tiles_added.is_empty()
        && tiles_removed.is_empty());
    *last = next;
    *last_tiles = next_tiles;
    let mut done = PlanDone {
        epoch: job.epoch,
        leaves: job.previous.clone(),
        changed,
        added,
        removed,
        tiles: job
            .surface
            .as_ref()
            .map_or_else(Arc::default, |s| s.previous.clone()),
        tiles_added,
        tiles_removed,
        order: Vec::new(),
        cover: FxHashSet::default(),
        cover_tiles: FxHashSet::default(),
        tree: Vec::new(),
        footprints: Vec::new(),
        pruned: plan.pruned,
        limited: plan.limited,
        coarsened: plan.coarsened,
        tiles_limited: plan.tiles_limited,
        held: plan.held,
        held_keys: plan.held_keys,
        elapsed: Duration::ZERO,
    };
    if changed {
        let mut order: Vec<(f64, u16, CubeKey)> = plan
            .leaves
            .iter()
            .map(|&k| (-screen_error(job.camera, k), bucket(job.camera, k), k))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        done.order = order.into_iter().map(|(_, b, k)| (b, k)).collect();
        done.tree = plan.leaves.iter().map(|&k| (tree_order(k), k)).collect();
        done.tree.sort_unstable();
        done.footprints = plan
            .leaves
            .iter()
            .map(|&k| (footprint_order(k.level, k.x, k.z), k.level, FarKey::Cube(k)))
            .chain(
                plan.tiles
                    .iter()
                    .map(|&t| (footprint_order(t.level, t.x, t.z), t.level, FarKey::Tile(t))),
            )
            .collect();
        done.footprints.sort_unstable();
        let cover = cover.unwrap_or_else(|| plan_job(job, &cover_rule, true, class, None));
        done.cover = cover.leaves.into_iter().collect();
        done.cover_tiles = cover.tiles.into_iter().collect();
        done.leaves = Arc::new(plan.leaves);
        done.tiles = Arc::new(plan.tiles);
    }
    done.elapsed = started.elapsed();
    done
}

/// The planner thread: plans off the game thread, one job at a time.
fn plan_loop(shared: &Shared) {
    let mut last = FxHashSet::default();
    let mut last_tiles = FxHashSet::default();
    let mut epoch = None;
    loop {
        let job = {
            let mut slot = shared.planner.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(job) = slot.job.take() {
                    slot.busy = true;
                    break job;
                }
                slot = shared.planner_wake.wait(slot).unwrap();
            }
        };
        if epoch != Some(job.epoch) {
            epoch = Some(job.epoch);
            last.clear();
            last_tiles.clear();
        }
        let class = |key: CubeKey| classify_cached(shared, key);
        let done = make_plan(&job, &mut last, &mut last_tiles, &class);
        let mut slot = shared.planner.lock().unwrap();
        slot.busy = false;
        slot.done = Some(done);
    }
}

/// Whether the plan waits for the coverage layer.
#[derive(Debug)]
enum Phase {
    Refine,
    /// The coverage layer (`wave`: coverage nodes and the coarsest planned
    /// nodes, one over each hole) is all that is queued, since `since`.
    Cover {
        since: Instant,
        wave: Vec<FarKey>,
    },
}

/// A planned node with nothing drawn over its place, and the coverage leaf
/// (a node or a tile) over it when the coverage layer has one.
type Hole = (CubeKey, Option<FarKey>);

/// Why a plan job was sent (the waste counters): the first that applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Replan {
    /// The camera's cube, the range, the loaded box or the detail changed.
    Moved,
    /// New classes came.
    Classes,
    /// Nodes are held a level and one came in.
    Refine,
    /// Only the timer.
    Timer,
}

/// What the last plan job was asked with.
#[derive(Clone, Copy, Debug)]
struct Asked {
    cube: SectionPos,
    camera: [f64; 3],
    max_distance: f64,
    loaded: LoadedBox,
    rule: Rule,
    at: Instant,
}

/// Whether the plan for `now` differs enough from the one for `before` to be
/// made again: the detail changed by [`REFOCUS`], the zoom came or went, a
/// zoomed view turned past its margin, or the budget moved a tenth.
fn refocused(before: &Rule, now: &Rule, view: &FarView) -> bool {
    let off = |a: f64, b: f64| a > b * REFOCUS || b > a * REFOCUS;
    if off(before.in_view, now.in_view) || off(before.out_of_view, now.out_of_view) {
        return true;
    }
    if before.budget.abs_diff(now.budget) * 10 > before.budget.max(1) {
        return true;
    }
    match (&before.cone, &now.cone) {
        (None, None) => false,
        (Some(a), Some(b)) => angle_between(a.axis, b.axis) > view.turn_margin(),
        _ => true,
    }
}

/// How far the arena's room for nodes may move before the budget follows it,
/// as (lower, upper) fractions of the budget held: the bytes a node takes move
/// with the plan (finer nodes, more empty ones), and a budget that followed
/// every move asked for a plan of another size each time (`refocused`), which
/// moved the bytes again: at Ultra, where the arena binds, the far terrain
/// went back and forth between levels. The lower bound keeps the arena's
/// room for the nodes a plan replaces mostly whole.
const ARENA_CAP_BAND: (f64, f64) = (0.9, 1.25);

/// Whether the arena's room for `fits` nodes leaves a held cap of `cap` as
/// it is.
fn arena_cap_holds(cap: usize, fits: usize) -> bool {
    let (lo, hi) = ARENA_CAP_BAND;
    (fits as f64) >= cap as f64 * lo && (fits as f64) <= cap as f64 * hi
}

/// The most surface tiles a plan holds.
pub const TILE_BUDGET: usize = 3584;
/// Surface tiles the renderer's pool holds (`Renderer::use_surface_tiles`):
/// a plan's, the ones it replaces and the coverage layer's (about 110 MB).
pub const TILE_CAPACITY: u32 = 8192;
/// Tile headers kept for the plan past which those of tiles neither planned
/// nor drawn are forgotten.
const TILE_INFO_KEPT: usize = 4 * TILE_BUDGET;
/// Levels a coverage leaf is coarser than the plan, about: `log2(COVER_SCALE)`.
const COVER_LEVELS: u8 = 3;

/// What the renderer's surface tiles are to do ([`CubeLod::take_tile_work`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileWork {
    /// Tiles to generate (`SurfaceTiles::request`), coverage first, then
    /// nearest first.
    pub request: Vec<TileKey>,
    /// Tiles to fade out (`SurfaceTiles::retire`): what replaces them is in.
    pub retire: Vec<TileKey>,
    /// Requests to drop, never drawn (`SurfaceTiles::free`).
    pub free: Vec<TileKey>,
}

/// The plan's surface tiles on the game thread.
#[derive(Default)]
struct TileState {
    /// Tiles run ([`CubeLod::set_surface`]).
    on: bool,
    sea_level: i32,
    plan: Arc<Vec<TileKey>>,
    wanted: FxHashSet<TileKey>,
    /// Generated and drawn (their header read back), not retired.
    delivered: FxHashSet<TileKey>,
    /// Delivered and not planned: drawn until what replaces them is in.
    stale: FxHashSet<TileKey>,
    /// Requested and not delivered yet.
    asked: FxHashSet<TileKey>,
    /// Planned and not delivered.
    missing: usize,
    /// The headers known, for the planner.
    info: Arc<FxHashMap<TileKey, TileInfo>>,
    /// The coverage layer's tiles for the current plan.
    cover: FxHashSet<TileKey>,
    /// Planned tiles with nothing drawn over their footprint.
    holes: Vec<TileKey>,
    work: TileWork,
}

/// The far terrain of a cubic level, built in the background.
///
/// ```ignore
/// // Once, when the level loads:
/// let mut far = CubeLod::new(Box::new(generator_sampler), context.clone(), air);
/// // Every frame:
/// let meshes = far.update(camera, max_distance, &view, LoadedBox { horizontal: h, vertical: v },
///                         |cube| world.has_cube(cube));
/// renderer.queue_cube_lods(meshes, far.removed());
/// renderer.upload_cube_lods(budget);
/// // On leaving the level:
/// far.clear();
/// renderer.clear_cube_lods();
/// ```
///
/// Meshes are handed out once per node. A node leaves through [`removed`]
/// only when every node that now covers its place has been handed out (or
/// none is wanted: out of range), so the renderer never opens a hole.
///
/// The plan is made on a thread of its own ([`plan_far`] with the view's
/// [`Rule`]). After a big change (a teleport, joining: a quarter of the plan
/// has nothing drawn over it) only the coverage layer is queued: the plan for
/// voxels [`COVER_SCALE`] times coarser, a few hundred nodes, over the holes.
/// Once it is in and shown, the rest of the plan is queued, holes first, then
/// by direction (the view's first) and distance; the coverage nodes stay
/// drawn until the nodes under them are in.
///
/// With surface tiles on ([`CubeLod::set_surface`]), the coarse levels are
/// tiles the renderer generates itself: [`CubeLod::take_tile_work`] hands it
/// the requests, fades and drops, and [`CubeLod::surface_feedback`] notes the
/// tiles it generated (their headers: a complex tile is planned again as cube
/// nodes). A tile stays drawn until everything planned in its footprint is
/// in, and a node under a planned tile until the tile is.
///
/// [`removed`]: CubeLod::removed
pub struct CubeLod {
    shared: Arc<Shared>,
    results: mpsc::Receiver<Done>,
    threads: Vec<thread::JoinHandle<()>>,
    /// The current plan, nearest first.
    plan: Arc<Vec<CubeKey>>,
    wanted: FxHashSet<CubeKey>,
    /// The plan with each node's bucket, the bigger on-screen error first:
    /// the queue's source.
    order: Vec<(u16, CubeKey)>,
    /// The plan in tree order (`tree_order`).
    tree: Vec<(u128, CubeKey)>,
    /// The plan's leaves and tiles by footprint (`footprint_order`).
    footprints: Vec<(u64, u8, FarKey)>,
    tiles: TileState,
    /// The sampler's world can have surface tiles (`CubeSampler::surface`).
    surface_allowed: bool,
    surface_stats: Option<crate::gfx::surface_tile::SurfaceStats>,
    /// The coverage layer's leaves for the current plan.
    cover_plan: FxHashSet<CubeKey>,
    /// Coverage nodes asked for (not in the plan): their results are kept.
    cover: FxHashSet<CubeKey>,
    /// Planned nodes with nothing drawn over their place, and the coverage
    /// leaf over each when the plan had one.
    holes: Vec<Hole>,
    /// Planned nodes holding an older plan's (`find_holes`): queued first.
    holding: Vec<CubeKey>,
    holes_dirty: bool,
    holes_at: Instant,
    phase: Phase,
    /// Handed out and not reported removed.
    delivered: FxHashSet<CubeKey>,
    /// Delivered nodes not in the plan: coverage nodes, and an older plan's
    /// nodes still covering a place.
    stale: FxHashSet<CubeKey>,
    /// The sampler's late content ([`LateContent`]).
    late: Option<Arc<dyn LateContent>>,
    /// Drawn nodes queued to be made again for it.
    refreshing: FxHashSet<CubeKey>,
    /// [`CubeNode::content_hash`] of each delivered node made from a CPU node.
    drawn_hash: FxHashMap<CubeKey, u64>,
    /// Planned nodes not delivered.
    missing: usize,
    failed: FxHashMap<CubeKey, Instant>,
    retry_at: Instant,
    release_dirty: bool,
    released_at: Instant,
    /// Stale nodes found releasable, handed out [`RELEASE_BATCH`] at a time.
    releasing: Vec<CubeKey>,
    asked: Option<Asked>,
    /// A replan is owed (new classes came while the planner was busy).
    replan: bool,
    /// [`CubeLod::clear`]s so far.
    epoch: u64,
    camera: [f64; 3],
    /// The look the queue's bucket order was made for.
    look: Option<Look>,
    /// The far nodes over near ground ([`track_lingering`]) the queue was
    /// last filled for: a new one fills it again, so the nodes replacing it
    /// that were queued already go first.
    linger_queued: FxHashSet<CubeKey>,
    budget: usize,
    /// The sampler's GPU generator, when it has one.
    backend: Option<GpuBackend>,
    /// Nodes finished for the GPU mesher, not yet taken.
    gpu_out: Vec<GpuNode>,
    /// Finished nodes pulled from the workers' channel and held back by the
    /// per-frame delivery budget ([`DeliverBudget`]), oldest first.
    backlog: VecDeque<Done>,
    context: Option<Arc<WorkerContext>>,
    /// Block states the renderer has material records for.
    known_states: FxHashSet<u32>,
    biomes_sent: bool,
    air: BlockStateId,
    stats: CubeLodStats,
    /// Delivered nodes that draw nothing, and how many of them are planned.
    empty: FxHashSet<CubeKey>,
    empty_planned: usize,
    /// Nodes the planner left out as uniform, from the class cache.
    pruned: usize,
    coarsened: f64,
    render: Option<crate::gfx::cube_far::FarStats>,
    /// The most nodes the quad arena holds two plans of, held between
    /// [`ARENA_CAP_BAND`]'s bounds ([`CubeLod::set_render_stats`]).
    arena_cap: Option<usize>,
    /// The sampler has a GPU descent.
    has_descent: bool,
    /// What the descent was last asked for, and its run count then.
    descent_asked: Option<(SectionPos, f64, LoadedBox, u64)>,
    descent_runs_asked: u64,
    /// The plan holds for the descent's answer after a jump, since.
    jumped: Option<Instant>,
    /// Nodes the plan holds back a level ([`Plan::held`]), and whether a node
    /// came in since the last plan was asked for: the next level is due.
    held: usize,
    delivered_since: bool,
    /// The drawn nodes over loaded near cubes, since when.
    lingering: FxHashMap<CubeKey, Instant>,
    /// Nodes ever handed out (cleared past 1<<20, and by `clear`): one handed
    /// out again after its release was made again (`Waste::rebuilt`).
    ever: FxHashSet<CubeKey>,
    /// When each drawn node was handed out, and when it was queued to be made
    /// again for late content: a release soon after was work for nothing.
    delivered_at: FxHashMap<CubeKey, Instant>,
    refreshed_at: FxHashMap<CubeKey, Instant>,
    /// The leaves the last plan held back a level ([`Plan::held_keys`]).
    held_keys: FxHashSet<CubeKey>,
    /// Counts up when the holes, the coverage layer or the phase change, and
    /// the count the late content was last told at
    /// (`HORIZON_FAR_WHOLE_WHEN_COVERED`).
    holes_version: u64,
    holes_told: u64,
}

impl CubeLod {
    /// `air` is the registry's air state; voxels of any other state are drawn.
    pub fn new(
        sampler: Box<dyn CubeSampler>,
        context: Arc<WorkerContext>,
        air: BlockStateId,
    ) -> Self {
        // All but two of the machine's threads, from 1 to 8
        // (`horizon_threads::sizing::far_workers`): the main/render thread and
        // the near terrain keep a core each, and the far threads run below them
        // for good (`far_below_main`). Walking on 8 threads, 8 workers took 3.1
        // cores and crowded the frame out; 6 take at most 6. The fill after a
        // teleport waits on the workers: 4 filled in 17.3 s, 6 in 14.8 s, 8 in
        // about 13 s (research/far-fill), so this trades about 2 s of fill for
        // the frame. `HORIZON_FAR_WORKERS` sets the count outright.
        let wanted_threads = far_knob(
            "HORIZON_FAR_WORKERS",
            horizon_threads::sizing::far_workers_here() as u64,
        )
        .max(1) as usize;
        let backend = sampler.gpu();
        let surface_allowed = sampler.surface();
        let late = sampler.late();
        // One more fork classifies nodes for the planner, on a thread of its own.
        let classifier = (backend.is_some() || sampler.classifies())
            .then(|| sampler.fork())
            .flatten();
        let descent = sampler.descent();
        let mut samplers = Vec::new();
        while samplers.len() + 1 < wanted_threads
            && let Some(fork) = sampler.fork()
        {
            samplers.push(fork);
        }
        samplers.push(sampler);
        let shared = Arc::new(Shared::new());
        let (tx, results) = mpsc::channel();
        let mut threads: Vec<thread::JoinHandle<()>> = classifier
            .map(|mut sampler| {
                let shared = shared.clone();
                thread::Builder::new()
                    .name("Cube far classes".into())
                    .spawn(move || {
                        horizon_threads::far_below_main();
                        classify_loop(&shared, sampler.as_mut())
                    })
                    .expect("spawn the cube classifier thread")
            })
            .into_iter()
            .collect();
        let has_descent = descent.is_some();
        if let Some(descent) = descent {
            let shared = shared.clone();
            threads.push(
                thread::Builder::new()
                    .name("Cube far descent".into())
                    .spawn(move || {
                        horizon_threads::far_below_main();
                        descent_loop(&shared, &descent)
                    })
                    .expect("spawn the cube descent thread"),
            );
        }
        {
            let shared = shared.clone();
            threads.push(
                thread::Builder::new()
                    .name("Cube far planner".into())
                    .spawn(move || {
                        horizon_threads::far_below_main();
                        plan_loop(&shared)
                    })
                    .expect("spawn the cube planner thread"),
            );
        }
        threads.extend(samplers.into_iter().enumerate().map(|(index, sampler)| {
            let (shared, tx, context) = (shared.clone(), tx.clone(), context.clone());
            let backend = backend.clone();
            thread::Builder::new()
                .name("Cube far terrain".into())
                .spawn(move || work(index, shared, tx, context, sampler, air, backend))
                .expect("spawn a cube far terrain thread")
        }));
        let mut lod = Self::assemble(shared, results, threads, backend, Some(context), air);
        lod.has_descent = has_descent;
        lod.surface_allowed = surface_allowed;
        lod.late = late;
        // Benchmarks and autoresearch loops set the node budget from outside.
        if let Some(nodes) = std::env::var(BUDGET_VAR)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            lod.set_budget(nodes);
        }
        lod
    }

    fn assemble(
        shared: Arc<Shared>,
        results: mpsc::Receiver<Done>,
        threads: Vec<thread::JoinHandle<()>>,
        backend: Option<GpuBackend>,
        context: Option<Arc<WorkerContext>>,
        air: BlockStateId,
    ) -> Self {
        let now = Instant::now();
        Self {
            shared,
            results,
            threads,
            plan: Arc::default(),
            wanted: FxHashSet::default(),
            order: Vec::new(),
            tree: Vec::new(),
            footprints: Vec::new(),
            tiles: TileState {
                sea_level: backend.as_ref().map_or(63, |b| b.params.sea_level),
                ..TileState::default()
            },
            surface_allowed: false,
            surface_stats: None,
            cover_plan: FxHashSet::default(),
            cover: FxHashSet::default(),
            holes: Vec::new(),
            holding: Vec::new(),
            holes_dirty: false,
            holes_at: now,
            phase: Phase::Refine,
            delivered: FxHashSet::default(),
            stale: FxHashSet::default(),
            late: None,
            refreshing: FxHashSet::default(),
            drawn_hash: FxHashMap::default(),
            missing: 0,
            failed: FxHashMap::default(),
            retry_at: now,
            release_dirty: false,
            released_at: now,
            releasing: Vec::new(),
            asked: None,
            replan: false,
            epoch: 0,
            camera: [0.0; 3],
            look: None,
            linger_queued: FxHashSet::default(),
            budget: NODE_BUDGET,
            gpu_out: Vec::new(),
            backlog: VecDeque::new(),
            context,
            known_states: FxHashSet::default(),
            biomes_sent: false,
            air,
            stats: CubeLodStats::default(),
            empty: FxHashSet::default(),
            empty_planned: 0,
            pruned: 0,
            coarsened: 1.0,
            render: None,
            arena_cap: None,
            has_descent: false,
            descent_asked: None,
            descent_runs_asked: 0,
            jumped: None,
            held: 0,
            delivered_since: false,
            lingering: FxHashMap::default(),
            ever: FxHashSet::default(),
            delivered_at: FxHashMap::default(),
            refreshed_at: FxHashMap::default(),
            held_keys: FxHashSet::default(),
            holes_version: 0,
            holes_told: u64::MAX,
            backend,
        }
    }

    /// Whether the near terrain is loading: the far workers step behind it
    /// meanwhile and come back to normal priority once it is in.
    pub fn set_near_loading(&self, loading: bool) {
        self.shared.near_loading.store(loading, Ordering::Relaxed);
        if loading {
            let ms = self.shared.near_epoch.elapsed().as_millis() as u64;
            self.shared.near_seen_ms.store(ms, Ordering::Relaxed);
        }
    }

    /// The renderer's GPU far terrain counters (`Renderer::cube_far_stats`),
    /// for [`CubeLod::stats`].
    pub fn set_render_stats(&mut self, stats: Option<crate::gfx::cube_far::FarStats>) {
        self.render = stats;
        if let Some(r) = stats
            && r.nodes >= 1024
            && r.arena_bytes > 0
        {
            let per_node = (r.arena_bytes / r.nodes as u64).max(1);
            let fits = (r.arena_capacity_bytes / 2 / per_node) as usize;
            self.arena_cap = Some(match self.arena_cap {
                Some(cap) if arena_cap_holds(cap, fits) => cap,
                _ => fits,
            });
        }
    }

    /// The renderer's surface tile counters (`Renderer::surface_stats`).
    pub fn set_surface_stats(&mut self, stats: Option<crate::gfx::surface_tile::SurfaceStats>) {
        self.surface_stats = stats;
    }

    /// Whether this world's far terrain can have surface tiles: a Custom world
    /// without `cubeAreas` (`CubeSampler::surface`) whose generator runs on
    /// the GPU with its climate pass.
    pub fn surface_allowed(&self) -> bool {
        self.surface_allowed
            && self
                .backend
                .as_ref()
                .is_some_and(|b| b.biome_params.is_some())
    }

    /// Turns surface tiles on (the renderer runs them:
    /// `Renderer::use_surface_tiles`) or off. They need the GPU path and a
    /// world that allows them. A change forgets every node handed out, as
    /// [`CubeLod::clear`]: clear the renderer's too. Returns whether it changed.
    pub fn set_surface(&mut self, on: bool) -> bool {
        let on = on && self.surface_allowed() && self.gpu_path();
        if self.tiles.on == on {
            return false;
        }
        self.clear();
        self.tiles.on = on;
        true
    }

    /// Whether the plan has surface tiles.
    pub fn surface_on(&self) -> bool {
        self.tiles.on
    }

    /// Notes the tiles the renderer generated since the last call: `info`
    /// reads a tile's header once it is generated and read back
    /// (`Renderer::surface_tile_info`). Call each frame before `update`.
    pub fn surface_feedback(&mut self, info: impl Fn(TileKey) -> Option<TileInfo>) {
        if self.tiles.asked.is_empty() {
            return;
        }
        let arrived: Vec<(TileKey, TileInfo)> = self
            .tiles
            .asked
            .iter()
            .filter_map(|&k| info(k).map(|i| (k, i)))
            .collect();
        for (key, info) in arrived {
            self.deliver_tile(key, info);
        }
    }

    /// Notes a tile as drawn, and its header for the plan: a complex tile is
    /// planned again as cube nodes, a tall one refines as near as it reaches.
    fn deliver_tile(&mut self, key: TileKey, info: TileInfo) {
        self.tiles.asked.remove(&key);
        if !self.tiles.delivered.insert(key) {
            return;
        }
        self.release_dirty = true;
        self.holes_dirty = true;
        if self.tiles.wanted.contains(&key) {
            self.tiles.missing = self.tiles.missing.saturating_sub(1);
        } else {
            self.tiles.stale.insert(key);
        }
        if self.tiles.info.get(&key) == Some(&info) {
            return;
        }
        let (wanted, delivered) = (&self.tiles.wanted, &self.tiles.delivered);
        let map = Arc::make_mut(&mut self.tiles.info);
        if map.len() >= TILE_INFO_KEPT {
            map.retain(|k, _| wanted.contains(k) || delivered.contains(k));
        }
        map.insert(key, info);
        if (info.complex() && key.level < SURFACE_LEVEL.saturating_add(COMPLEX_LIFT))
            || info.highest > self.tiles.sea_level + UNKNOWN_ABOVE
        {
            self.replan = true;
        }
    }

    /// What the renderer's surface tiles are to do since the last call
    /// (`Renderer::queue_surface_tiles`). Call after [`CubeLod::removed`],
    /// which finds the tiles to fade out.
    pub fn take_tile_work(&mut self) -> TileWork {
        std::mem::take(&mut self.tiles.work)
    }

    /// The most nodes a plan may hold (a terrain memory target, in nodes).
    pub fn set_budget(&mut self, nodes: usize) {
        self.budget = nodes.max(COVER_NODES);
    }

    /// The budget for the next plan: [`CubeLod::set_budget`]'s, lowered on
    /// the GPU path so the plan and the nodes it replaces fit in the node
    /// table and the quad arena (each holds two plans' worth at the quads
    /// per node measured so far).
    pub fn budget(&self) -> usize {
        let mut budget = self.budget;
        if self.gpu_path() {
            budget = budget.min(crate::gfx::cube_far::MAX_NODES as usize / 2);
            if let Some(cap) = self.arena_cap {
                budget = budget.min(cap);
            }
        }
        budget.max(COVER_NODES)
    }

    /// Progress of the plan. `complete` is exact: every planned node was
    /// handed out (a mesh, or a GPU node) and, on the GPU path, the renderer
    /// holds no node still waiting for its mesh; nodes that draw nothing count
    /// as ready.
    pub fn stats(&self) -> CubeLodStats {
        let mut out = self.stats;
        let w = self.shared.waste.snap();
        out.jobs = w.jobs;
        out.built = w.built;
        out.failed_jobs = w.failed_jobs;
        out.built_uniform = w.built_uniform;
        out.built_empty = w.built_empty;
        out.delivered = w.delivered;
        out.rebuilt = w.rebuilt;
        out.plan_installs = w.plan_installs;
        out.plan_added = w.plan_added;
        out.plan_removed = w.plan_removed;
        out.plan_reused = w.plan_reused;
        out.plan_readded = w.plan_readded;
        out.plan_dropped_unbuilt = w.plan_dropped_unbuilt;
        out.released = w.released;
        out.short_lived = w.short_lived;
        out.short_lived_levels = w.short_lived_levels;
        out.late_skipped_held = w.late_skipped_held;
        out.late_dropped = w.late_dropped;
        out.late_wasted = w.late_wasted;
        out.late_other = w.late_other;
        out.retries = w.retries;
        out.refills = w.refills;
        out.refill_keys = w.refill_keys;
        out.replans = w.replans;
        out.replans_moved = w.replans_moved;
        out.replans_classes = w.replans_classes;
        out.replans_refine = w.replans_refine;
        out.replans_timer = w.replans_timer;
        {
            let cache = self.shared.cache.lock().unwrap();
            out.cache_hits = cache.hits;
            out.cache_misses = cache.misses;
            out.cache_evicted = cache.evicted;
            out.cache_forgotten = cache.forgotten;
            out.cache_cleared = cache.cleared;
            out.cache_nodes = cache.len();
            out.cache_bytes = cache.bytes();
        }
        out.deliver_deferred = self.backlog.len();
        out.planned = self.plan.len() + self.tiles.plan.len();
        out.pending = self.missing + self.tiles.missing;
        out.ready = out.planned - out.pending;
        out.tiles = self.tiles.plan.len();
        out.tiles_ready = self.tiles.plan.len() - self.tiles.missing;
        if let Some(s) = self.surface_stats {
            out.tiles_resident = s.resident;
            out.tile_bytes = s.bytes;
            out.tiles_complex = s.complex;
        }
        out.uniform_skipped = self.pruned + self.empty_planned;
        if let Some(render) = self.render {
            out.gpu_meshed = render.meshed + render.uniform;
            out.arena_bytes = render.arena_bytes;
            out.pool_bytes = render.pool_bytes;
        }
        out.complete = self.is_complete();
        out.holes = self.holes.len() + self.tiles.holes.len();
        out.covering = matches!(self.phase, Phase::Cover { .. });
        out.budget = self.asked.map_or(self.budget(), |a| a.rule.budget);
        if let Ok(slot) = self.shared.descent.lock() {
            out.descent_runs = slot.runs;
            out.descent_uniform = slot.uniform;
            out.descent_leaves = slot.leaves;
            out.descent_ms = slot.millis;
            out.descent_overflow = slot.overflow;
        }
        out
    }

    /// Everything handed out has been meshed and shown: the renderer holds
    /// nothing waiting (always, on the CPU path).
    fn shown(&self) -> bool {
        self.gpu_out.is_empty()
            && (!self.gpu_path() || self.render.is_some_and(|r| r.pending == 0 && !r.meshing))
    }

    fn is_complete(&self) -> bool {
        !(self.plan.is_empty() && self.tiles.plan.is_empty())
            && self.missing == 0
            && self.tiles.missing == 0
            && self.shown()
    }

    fn track_completion(&mut self, now: Instant) {
        match (self.is_complete(), self.stats.plan_complete) {
            (true, None) => self.stats.plan_complete = Some(now),
            (false, Some(_)) => self.stats.plan_complete = None,
            _ => {}
        }
        let holes = !(self.holes.is_empty() && self.tiles.holes.is_empty());
        let covered = !(self.plan.is_empty() && self.tiles.plan.is_empty())
            && !holes
            && matches!(self.phase, Phase::Refine)
            && self.shown();
        match (covered, self.stats.plan_covered) {
            (true, None) => self.stats.plan_covered = Some(now),
            (false, Some(_)) if holes => self.stats.plan_covered = None,
            _ => {}
        }
    }

    /// Whether nodes take the GPU path now (`set_gpu_path`).
    pub fn on_gpu_path(&self) -> bool {
        self.shared.gpu_path.load(Ordering::Relaxed)
    }

    /// The sampler's GPU generator, if it has one: the device the renderer's
    /// GPU mesher must run on (`gfx::cube_mesh`) to read its nodes.
    pub fn gpu_backend(&self) -> Option<&GpuBackend> {
        self.backend.as_ref()
    }

    /// Switches between the GPU path (`on`, and the sampler has a GPU
    /// generator: workers hand out [`GpuNode`]s, `take_gpu_nodes`) and the CPU
    /// path (meshes from `update`). A change forgets every node handed out, as
    /// [`CubeLod::clear`]: clear the renderer's too (`clear_cube_lods`).
    /// Returns whether it changed.
    pub fn set_gpu_path(&mut self, on: bool) -> bool {
        let on = on && self.gpu_path_allowed();
        if self.shared.gpu_path.load(Ordering::Relaxed) == on {
            return false;
        }
        self.shared.gpu_path.store(on, Ordering::Relaxed);
        self.clear();
        true
    }

    /// Whether the GPU path can be taken: the sampler has a GPU backend, and
    /// one without a generator (`GpuBackend::generates` false: every node
    /// sampled on the CPU, an Earth world's) only where every node's states
    /// and biomes fit the GPU mesher ([`cpu_nodes_fit_gpu`]); else the world
    /// stays on the CPU mesher, never drawn wrong.
    pub fn gpu_path_allowed(&self) -> bool {
        match &self.backend {
            None => false,
            Some(backend) if backend.generates => true,
            Some(_) => {
                !self.shared.gpu_unfit.load(Ordering::Relaxed)
                    && self.context.as_ref().is_some_and(|context| {
                        cpu_nodes_fit_gpu(context.assets.registry.len(), context.biomes.list.len())
                    })
            }
        }
    }

    /// Whether the GPU path's nodes are all sampled on the CPU and uploaded
    /// (a backend without a generator, `GpuBackend::pool_only`).
    pub fn gpu_pool_only(&self) -> bool {
        self.backend.as_ref().is_some_and(|b| !b.generates)
    }

    /// Whether workers make [`GpuNode`]s.
    pub fn gpu_path(&self) -> bool {
        self.shared.gpu_path.load(Ordering::Relaxed)
    }

    /// Switches the planner's classes between the GPU descent (`on`, and the
    /// sampler has one: `GpuCubeSampler::with_descent`) and the classifier
    /// thread's 2D pre-pass. With the descent, each replan's camera cube sends
    /// one request to the GPU, which walks the octree level by level, proves
    /// whole subtrees uniform from the density bound and returns their keys;
    /// the plan, hysteresis and loaded cubes stay the CPU's, and nodes the
    /// descent has not classified yet count as mixed. The nodes a plan keeps
    /// are those of the CPU plan minus the proven-uniform subtrees. Returns
    /// whether it changed.
    pub fn set_gpu_descent(&mut self, on: bool) -> bool {
        let on = on && self.has_descent;
        if self.shared.descent_on.swap(on, Ordering::Relaxed) == on {
            return false;
        }
        self.asked = None;
        self.descent_asked = None;
        true
    }

    /// Whether the planner's classes come from the GPU descent.
    pub fn gpu_descent(&self) -> bool {
        self.shared.descent_on.load(Ordering::Relaxed)
    }

    /// Sends the descent its request for this camera (once per cube, range,
    /// box and detail), splitting as finely as the plan's out-of-view rule
    /// does (a zoom's finer nodes in view are not worth classifying the
    /// whole sphere for). Returns whether the plan should hold for its
    /// answer: the first plan always, and for [`DESCENT_WAIT`] after a jump.
    fn ask_descent(
        &mut self,
        camera: [f64; 3],
        max_distance: f64,
        loaded: LoadedBox,
        rule: &Rule,
        now: Instant,
    ) -> bool {
        let cube = cube_of(camera);
        // Asked again when it moves a fifth or more: not for every small
        // change of the budget's reach.
        let detail = rule.out_of_view / self.coarsened.max(1.0) * MERGE;
        let bits = (detail.max(1e-3).log2() * 4.0).round() as i64 as u64;
        let asked = (cube, max_distance, loaded, bits);
        if self.descent_asked != Some(asked) {
            self.descent_asked = Some(asked);
            let mut slot = self.shared.descent.lock().unwrap();
            self.descent_runs_asked = slot.runs;
            slot.request = Some(DescentRequest {
                camera,
                max_distance,
                // Up to `MERGE` times the plain rule: the plan's hysteresis.
                detail,
                min_level: MIN_LEVEL,
                loaded: Some(([cube.x, cube.y, cube.z], loaded)),
            });
            drop(slot);
            self.shared.descent_wake.notify_one();
        }
        let slot = self.shared.descent.lock().unwrap();
        let answered = slot.runs > self.descent_runs_asked && slot.request.is_none() && !slot.busy;
        if slot.runs == 0 {
            return slot.request.is_some() || slot.busy;
        }
        drop(slot);
        if answered {
            self.jumped = None;
        }
        self.jumped.is_some_and(|at| {
            !answered
                && now.saturating_duration_since(at)
                    < far_ms("HORIZON_FAR_DESCENT_WAIT_MS", DESCENT_WAIT)
        })
    }

    /// The nodes finished for the GPU mesher since the last call (their slots
    /// are the caller's from here), for the nodes in the plan.
    pub fn take_gpu_nodes(&mut self) -> GpuBatch {
        let nodes = std::mem::take(&mut self.gpu_out);
        let mut wanted: Vec<u32> = Vec::new();
        if let Some(backend) = &self.backend {
            wanted.extend(&backend.states);
        }
        for node in &nodes {
            if let GpuContent::Uniform(voxel) = node.content {
                wanted.push(voxel.state.0);
            }
            wanted.extend(&node.states);
        }
        let mut materials = Vec::new();
        for state in wanted.into_iter().filter(|_| self.context.is_some()) {
            let context = self.context.as_ref().expect("a context");
            if state != self.air.0
                && state <= crate::gfx::cube_mesh::MAX_STATE
                && self.known_states.insert(state)
            {
                materials.push((
                    state,
                    crate::gfx::cube_mesh::material_record(context, state),
                ));
            }
        }
        let biomes = self
            .context
            .as_ref()
            .filter(|_| !self.biomes_sent)
            .map(|context| {
                self.biomes_sent = true;
                crate::gfx::cube_mesh::biome_table(context)
            });
        GpuBatch {
            nodes,
            materials,
            biomes,
        }
    }

    /// Returns a node's slot to the pool.
    fn release(&self, node: &GpuNode) {
        if let (GpuContent::Slot(slot), Some(backend)) = (node.content, &self.backend) {
            backend.pool.free(slot);
        }
    }

    /// Plans (again, when the camera crossed into another cube, the range,
    /// the loaded box or the view's detail changed, or a moment passed),
    /// queues what is missing, and returns the meshes finished since: only
    /// for nodes in the plan and the coverage layer.
    pub fn update(
        &mut self,
        camera: [f64; 3],
        max_distance: f64,
        view: &FarView,
        loaded: LoadedBox,
        is_loaded: impl Fn(SectionPos) -> bool,
    ) -> Vec<(CubeKey, SectionMesh)> {
        self.camera = camera;
        let now = Instant::now();
        let cube = cube_of(camera);
        self.face(Look::of(view));
        let rule = view.rule(self.budget());
        self.stats.target_px = view.target_px as f32;
        self.stats.fov_deg = view.fov_y.to_degrees() as f32;
        // A plan the planner finished goes in.
        let done = self.shared.planner.lock().unwrap().done.take();
        if let Some(done) = done {
            self.install(done, now);
        }
        if self.jumped.is_none()
            && self
                .asked
                .is_some_and(|a| distance(a.camera, camera) > JUMP)
        {
            self.jumped = Some(now);
        }
        // With the descent on, the first plan waits for its first answer, and
        // a plan after a jump for a moment: a plan of nothing but unclassified
        // (mixed) nodes would queue the whole octree.
        let hold = self.shared.descent_on.load(Ordering::Relaxed)
            && self.ask_descent(camera, max_distance, loaded, &rule, now);
        if self.shared.classes_dirty.swap(false, Ordering::Relaxed) {
            self.replan = true;
        }
        let stale = self.replan
            || match &self.asked {
                Some(a) => {
                    a.cube != cube
                        || a.max_distance != max_distance
                        || a.loaded != loaded
                        || now.saturating_duration_since(a.at) >= self.replan_every()
                        || refocused(&a.rule, &rule, view)
                        || (self.held > 0
                            && self.delivered_since
                            && now.saturating_duration_since(a.at)
                                >= far_ms("HORIZON_FAR_REFINE_MS", REFINE_REPLAN))
                }
                None => true,
            };
        if stale && !hold {
            let mut slot = self.shared.planner.lock().unwrap();
            if slot.job.is_none() && !slot.busy && slot.done.is_none() {
                let reason = self.replan_reason(cube, max_distance, loaded, &rule, view, now);
                let snapshot = LoadedSnapshot::take(cube, loaded, &is_loaded);
                track_lingering(
                    &self.delivered,
                    &self.empty,
                    &mut self.lingering,
                    &mut self.stats,
                    &snapshot,
                    now,
                );
                self.delivered_since = false;
                slot.job = Some(PlanJob {
                    epoch: self.epoch,
                    camera,
                    max_distance,
                    rule,
                    loaded,
                    snapshot,
                    previous: self.plan.clone(),
                    drawn: self.drawn_input(),
                    step: far_knob("HORIZON_FAR_STEP", 1) != 0,
                    surface: self.tiles.on.then(|| SurfaceJob {
                        info: self.tiles.info.clone(),
                        sea_level: self.tiles.sea_level,
                        budget: TILE_BUDGET,
                        previous: self.tiles.plan.clone(),
                    }),
                });
                drop(slot);
                self.shared.planner_wake.notify_one();
                let w = &self.shared.waste;
                bump(&w.replans, 1);
                bump(
                    match reason {
                        Replan::Moved => &w.replans_moved,
                        Replan::Classes => &w.replans_classes,
                        Replan::Refine => &w.replans_refine,
                        Replan::Timer => &w.replans_timer,
                    },
                    1,
                );
                self.replan = false;
                self.jumped = None;
                self.asked = Some(Asked {
                    cube,
                    camera,
                    max_distance,
                    loaded,
                    rule,
                    at: now,
                });
            }
        }
        let mut out = Vec::new();
        self.drain_results(now, &mut out);
        self.retry(now);
        self.recheck_holes(now, false);
        self.advance_phase(now);
        self.track_completion(now);
        self.late_content(camera, view.forward, max_distance);
        out
    }

    /// Why the plan is redone now, the first that applies (for the waste
    /// counters only: the decision is `update`'s).
    fn replan_reason(
        &self,
        cube: SectionPos,
        max_distance: f64,
        loaded: LoadedBox,
        rule: &Rule,
        view: &FarView,
        now: Instant,
    ) -> Replan {
        let Some(a) = &self.asked else {
            return Replan::Moved;
        };
        if a.cube != cube
            || a.max_distance != max_distance
            || a.loaded != loaded
            || refocused(&a.rule, rule, view)
        {
            Replan::Moved
        } else if self.replan {
            Replan::Classes
        } else if self.held > 0
            && self.delivered_since
            && now.saturating_duration_since(a.at) >= far_ms("HORIZON_FAR_REFINE_MS", REFINE_REPLAN)
        {
            Replan::Refine
        } else {
            Replan::Timer
        }
    }

    /// Takes the finished nodes the workers sent, within the per-frame
    /// delivery budget ([`DeliverBudget`]): what is over it waits in
    /// `backlog` for the next `update`, in the order it came. Once the budget
    /// is reached the sibling group of the last node taken is still finished
    /// (the up to seven others of its parent that already came), so a swap
    /// that waits for all eight children is not held to the next frame by the
    /// cap, and a parent is never released ([`CubeLod::removed`]) before every
    /// node over its place is handed out: a node still in `backlog` is not
    /// `delivered`. Results nobody wants any more cost nothing and use none of it.
    fn drain_results(&mut self, now: Instant, out: &mut Vec<(CubeKey, SectionMesh)>) {
        self.drain_results_within(now, out, DeliverBudget::get());
    }

    /// [`CubeLod::drain_results`] under `budget`.
    fn drain_results_within(
        &mut self,
        now: Instant,
        out: &mut Vec<(CubeKey, SectionMesh)>,
        budget: DeliverBudget,
    ) {
        self.backlog.extend(self.results.try_iter());
        if self.backlog.is_empty() {
            return;
        }
        let started = Instant::now();
        let mut spent = Spent::default();
        let mut group: Option<CubeKey> = None;
        let mut capped = false;
        while let Some(done) = self.backlog.pop_front() {
            if spent.nodes > 0
                && budget.reached(&spent, started.elapsed())
                && group != Some(done.key.parent())
            {
                self.backlog.push_front(done);
                capped = true;
                break;
            }
            self.accept(done, now, out, &mut spent, &mut group);
        }
        if capped && let Some(parent) = group {
            let mut i = 0;
            while i < self.backlog.len() {
                if self.backlog[i].key.parent() == parent {
                    let done = self.backlog.remove(i).expect("a backlog entry");
                    self.accept(done, now, out, &mut spent, &mut group);
                } else {
                    i += 1;
                }
            }
        }
        if capped {
            self.stats.deliver_frames_capped += 1;
        }
    }

    /// One finished node: handed out (counted in `spent`), or dropped.
    fn accept(
        &mut self,
        done: Done,
        now: Instant,
        out: &mut Vec<(CubeKey, SectionMesh)>,
        spent: &mut Spent,
        group: &mut Option<CubeKey>,
    ) {
        let key = done.key;
        // A drawn node made again for late content replaces the old one
        // (the renderer swaps a key's node); a failed remake keeps it.
        let again = self.refreshing.remove(&key) && self.delivered.contains(&key);
        if !(self.wanted.contains(&key) || self.cover.contains(&key))
            || (self.delivered.contains(&key) && !again)
            || (again && done.product.is_none())
        {
            if let Some(Product::Gpu(node)) = &done.product {
                self.release(node);
            }
            return;
        }
        // Made again the same as drawn: nothing to swap (the same picture).
        if again
            && done.hash.is_some()
            && self.drawn_hash.get(&key) == done.hash.as_ref()
            && far_knob("HORIZON_FAR_LATE_SAME", 1) != 0
        {
            if let Some(Product::Gpu(node)) = &done.product {
                self.release(node);
            }
            self.refreshed_at.remove(&key);
            bump(&self.shared.waste.late_same, 1);
            return;
        }
        if done.product.is_some() {
            match done.hash {
                Some(hash) => self.drawn_hash.insert(key, hash),
                None => self.drawn_hash.remove(&key),
            };
        }
        match done.product {
            Some(Product::Mesh(mesh)) => {
                let empty = mesh.solid.is_empty() && mesh.translucent.is_empty();
                if again && !empty {
                    self.empty.remove(&key);
                }
                spent.nodes += 1;
                spent.bytes += (mesh.solid.len() + mesh.translucent.len())
                    * std::mem::size_of::<crate::mesher::TerrainVertex>();
                *group = Some(key.parent());
                self.deliver(key, empty);
                out.push((key, mesh));
            }
            Some(Product::Gpu(node)) => {
                let empty = matches!(node.content,
                    GpuContent::Uniform(voxel) if voxel.state == self.air);
                if matches!(node.content, GpuContent::Slot(_)) {
                    if node.states.is_empty() {
                        self.stats.gpu_generated += 1;
                    } else {
                        self.stats.gpu_uploaded += 1;
                    }
                }
                spent.nodes += 1;
                *group = Some(key.parent());
                self.deliver(key, empty);
                self.gpu_out.push(node);
            }
            None => {
                self.failed.insert(key, now);
            }
        }
    }

    /// Tells the late content the view, and once every planned node is
    /// drawn queues the nodes it says are out of date ([`LateContent`]).
    fn late_content(&mut self, camera: [f64; 3], forward: [f64; 3], max_distance: f64) {
        let Some(late) = self.late.clone() else {
            return;
        };
        let complete = self.is_complete();
        late.view(camera, forward, max_distance, complete);
        self.tell_coarse_first(&late);
        if !complete {
            return;
        }
        let boxes = late.take_refreshes();
        if boxes.is_empty() {
            return;
        }
        let touches = |key: &CubeKey| {
            let (o, size) = (key.origin(), i64::from(key.size()));
            boxes.iter().any(|(min, max)| {
                (0..3).all(|i| {
                    i64::from(o[i]) <= i64::from(max[i])
                        && i64::from(o[i]) + size > i64::from(min[i])
                })
            })
        };
        // A box that is no drawn node's own extent: its node left the plan, and
        // nothing is made for it.
        let not_drawn: Vec<([i32; 3], [i32; 3])> = (boxes.iter())
            .filter(|(min, max)| {
                let drawn = CubeKey::of_box(*min, *max).is_some_and(|k| {
                    self.delivered.contains(&k)
                        && (self.wanted.contains(&k) || self.cover.contains(&k))
                });
                !drawn
            })
            .copied()
            .collect();
        if !not_drawn.is_empty() {
            late.not_drawn(&not_drawn);
        }
        let exact = late.exact() && far_knob("HORIZON_FAR_LATE_EXACT", 1) != 0;
        let mut again: Vec<CubeKey> = if exact {
            // Each box is one node's extent: remake that node only, not every
            // drawn node the box touches (a coarser or finer one overlapping
            // it is another box's, or none's).
            let mut seen = FxHashSet::default();
            (boxes.iter())
                .filter_map(|(min, max)| CubeKey::of_box(*min, *max))
                .filter(|k| {
                    self.delivered.contains(k)
                        && (self.wanted.contains(k) || self.cover.contains(k))
                        && !self.refreshing.contains(k)
                        && seen.insert(*k)
                })
                .collect()
        } else {
            (self.delivered.iter())
                .filter(|k| self.wanted.contains(k) || self.cover.contains(k))
                .filter(|k| !self.refreshing.contains(k) && touches(k))
                .copied()
                .collect()
        };
        if !exact && !again.is_empty() {
            // Keys that are no box's own node: made for a box that crosses them.
            let own: FxHashSet<CubeKey> = (boxes.iter())
                .filter_map(|(min, max)| CubeKey::of_box(*min, *max))
                .collect();
            bump(
                &self.shared.waste.late_other,
                again.iter().filter(|k| !own.contains(k)).count(),
            );
        }
        // A node the plan splits as soon as it is drawn is not worth making
        // whole now: its box is handed out again later, if still drawn. Only
        // with exact boxes (they say they can be handed out again).
        if exact && far_knob("HORIZON_FAR_LATE_SKIP_HELD", 1) != 0 && !self.held_keys.is_empty() {
            let (skipped, rest): (Vec<CubeKey>, Vec<CubeKey>) =
                again.into_iter().partition(|k| self.held_keys.contains(k));
            again = rest;
            if !skipped.is_empty() {
                bump(&self.shared.waste.late_skipped_held, skipped.len());
                let skipped: Vec<_> = skipped.into_iter().map(box_of).collect();
                late.not_now(&skipped);
            }
        }
        if again.is_empty() {
            return;
        }
        self.stats.late_refreshed += again.len() as u64;
        bump(&self.shared.waste.late_refreshed, again.len());
        // Made again means asked of the sampler again: the node the workers
        // keep in memory is the out-of-date one (taking it back left the far
        // view's coarse Earth nodes coarse, without their trees, for good).
        {
            let mut cache = self.shared.cache.lock().unwrap();
            for key in &again {
                cache.remove(*key);
            }
        }
        let now = Instant::now();
        self.refreshed_at.extend(again.iter().map(|&k| (k, now)));
        self.refreshing.extend(again.iter().copied());
        {
            let mut queue = self.shared.queue.lock().unwrap();
            // A key the plan dropped while it was queued (`dropped`) is wanted
            // again: not to be skipped.
            for key in &again {
                queue.dropped.remove(key);
            }
            queue.later.extend(again);
        }
        self.shared.wake.notify_all();
    }

    /// Tells the late content which nodes must come coarse first (nothing is
    /// drawn over their place: the holes and the coverage layer over them),
    /// when that changed since it was told (`HORIZON_FAR_WHOLE_WHEN_COVERED`).
    fn tell_coarse_first(&mut self, late: &Arc<dyn LateContent>) {
        if self.holes_told == self.holes_version || !late.wants_coarse_first() {
            return;
        }
        self.holes_told = self.holes_version;
        let mut seen = FxHashSet::default();
        let mut keys = Vec::new();
        for &(key, over) in &self.holes {
            keys.push(key);
            if let Some(FarKey::Cube(c)) = over {
                keys.push(c);
            }
        }
        keys.extend(self.cover.iter().copied());
        keys.retain(|k| seen.insert(*k));
        let all = matches!(self.phase, Phase::Cover { .. }) || self.holes.len() > COVER_MIN_HOLES;
        late.coarse_first(&keys, all);
    }

    /// How often the plan is redone with nothing else asking: [`REPLAN`], or
    /// four times what the last plan took, so a big plan keeps the planner
    /// thread busy a quarter of the time at most.
    fn replan_every(&self) -> Duration {
        far_ms("HORIZON_FAR_REPLAN_MS", REPLAN).max(Duration::from_secs_f32(
            self.stats.plan_ms.max(0.0) * far_knob("HORIZON_FAR_REPLAN_PLANS", 4) as f32 / 1000.0,
        ))
    }

    /// Notes `key` as handed out (`empty`: it draws nothing).
    fn deliver(&mut self, key: CubeKey, empty: bool) {
        if !self.delivered.insert(key) {
            return;
        }
        let w = &self.shared.waste;
        bump(&w.delivered, 1);
        if self.ever.len() > 1 << 20 {
            self.ever.clear();
        }
        if !self.ever.insert(key) {
            bump(&w.rebuilt, 1);
        }
        self.delivered_at.insert(key, Instant::now());
        self.delivered_since = true;
        self.release_dirty = true;
        self.holes_dirty = true;
        if empty {
            self.empty.insert(key);
        }
        if self.wanted.contains(&key) {
            self.missing = self.missing.saturating_sub(1);
            if empty {
                self.empty_planned += 1;
            }
        } else {
            self.stale.insert(key);
        }
    }

    /// Puts in a plan the planner made: the plan, its holes, and the queue.
    fn install(&mut self, done: PlanDone, now: Instant) {
        if done.epoch != self.epoch {
            return;
        }
        self.pruned = done.pruned;
        self.coarsened = done.coarsened;
        self.stats.limited = done.limited;
        self.stats.tiles_limited = done.tiles_limited;
        self.stats.voxel_px = self.stats.target_px * done.coarsened.max(1.0) as f32;
        self.stats.plan_ms = done.elapsed.as_secs_f32() * 1000.0;
        self.held = done.held;
        self.stats.held = done.held;
        // Before the early return: the held leaves change with an unchanged plan.
        self.held_keys = done.held_keys.iter().copied().collect();
        if !done.changed {
            return;
        }
        let w = &self.shared.waste;
        bump(&w.plan_installs, 1);
        bump(&w.plan_added, done.added.len());
        bump(&w.plan_removed, done.removed.len());
        self.holes_version += 1;
        self.stats.plan_changed = Some(now);
        self.release_dirty = true;
        self.releasing.clear();
        let mut dropped = Vec::new();
        // Late refreshes queued for nodes this plan drops: not worth making.
        let mut refresh_dropped: Vec<CubeKey> = Vec::new();
        for &key in &done.removed {
            self.wanted.remove(&key);
            if far_knob("HORIZON_FAR_LATE_DROP", 1) != 0
                && !self.cover.contains(&key)
                && self.refreshing.remove(&key)
            {
                self.refreshed_at.remove(&key);
                refresh_dropped.push(key);
            }
            if self.delivered.contains(&key) {
                self.stale.insert(key);
                if self.empty.contains(&key) {
                    self.empty_planned = self.empty_planned.saturating_sub(1);
                }
            } else {
                self.missing = self.missing.saturating_sub(1);
                dropped.push(key);
                bump(&self.shared.waste.plan_dropped_unbuilt, 1);
            }
        }
        if !refresh_dropped.is_empty() {
            bump(&self.shared.waste.late_dropped, refresh_dropped.len());
            self.shared
                .queue
                .lock()
                .unwrap()
                .dropped
                .extend(refresh_dropped.iter().copied());
            if let Some(late) = &self.late {
                let boxes: Vec<_> = refresh_dropped.iter().copied().map(box_of).collect();
                late.not_drawn(&boxes);
            }
        }
        for &key in &done.added {
            self.wanted.insert(key);
            if self.delivered.contains(&key) {
                bump(&self.shared.waste.plan_reused, 1);
                self.stale.remove(&key);
                if self.empty.contains(&key) {
                    self.empty_planned += 1;
                }
            } else {
                if self.ever.contains(&key) {
                    bump(&self.shared.waste.plan_readded, 1);
                }
                self.missing += 1;
            }
        }
        // Tiles: the new ones are asked for at once (the renderer makes them
        // on its own), dropped ones never drawn are let go.
        self.tiles.cover = done.cover_tiles;
        for &key in &done.tiles_removed {
            self.tiles.wanted.remove(&key);
            if self.tiles.delivered.contains(&key) {
                self.tiles.stale.insert(key);
            } else {
                self.tiles.missing = self.tiles.missing.saturating_sub(1);
                if !self.tiles.cover.contains(&key) && self.tiles.asked.remove(&key) {
                    self.tiles.work.free.push(key);
                }
            }
        }
        for &key in &done.tiles_added {
            self.tiles.wanted.insert(key);
            if self.tiles.delivered.contains(&key) {
                self.tiles.stale.remove(&key);
            } else {
                self.tiles.missing += 1;
                if self.tiles.asked.insert(key) {
                    self.tiles.work.request.push(key);
                }
            }
        }
        let planned = done.leaves.len() + done.tiles.len();
        let big = (done.added.len() + done.tiles_added.len()) * COVER_HOLES > planned;
        self.plan = done.leaves;
        self.tiles.plan = done.tiles;
        self.order = done.order;
        self.tree = done.tree;
        self.footprints = done.footprints;
        self.cover_plan = done.cover;
        // The holes: all of them after a big change, else the old ones still
        // planned and the new nodes with nothing over them.
        let fresh: Vec<CubeKey> = if big {
            self.plan.to_vec()
        } else {
            done.added
                .iter()
                .copied()
                .filter(|k| !self.delivered.contains(k))
                .collect()
        };
        let covering = matches!(self.phase, Phase::Cover { .. });
        let (new_holes, holding) = self.find_holes(&fresh, big || covering);
        let fresh_tiles: Vec<TileKey> = if big {
            self.tiles.plan.to_vec()
        } else {
            done.tiles_added
                .iter()
                .copied()
                .filter(|k| !self.tiles.delivered.contains(k))
                .collect()
        };
        let new_tile_holes = self.find_tile_holes(&fresh_tiles, big || covering);
        if big {
            self.holes = new_holes;
            self.holding = holding;
            self.tiles.holes = new_tile_holes;
        } else {
            let wanted = &self.wanted;
            self.holes.retain(|(k, _)| wanted.contains(k));
            self.holes.extend(new_holes);
            self.holding.retain(|k| wanted.contains(k));
            self.holding.extend(holding);
            let wanted = &self.tiles.wanted;
            self.tiles.holes.retain(|k| wanted.contains(k));
            self.tiles.holes.extend(new_tile_holes);
        }
        self.holes_dirty = false;
        self.holes_at = now;
        let holes = self.holes.len() + self.tiles.holes.len();
        if holes > 0 {
            self.stats.plan_covered = None;
        }
        if (big && holes * COVER_HOLES > planned) || holes > COVER_MIN_HOLES || covering {
            self.begin_cover(now);
        } else if big
            || self
                .lingering
                .keys()
                .any(|k| !self.linger_queued.contains(k))
        {
            self.fill_queue();
        } else {
            // A small change: the new nodes over near ground go first, then
            // the new holes, the rest by score; the dropped ones are skipped
            // if they are queued.
            let added: Vec<CubeKey> = (done.added.iter().copied())
                .filter(|k| !self.delivered.contains(k))
                .collect();
            self.queue_jobs(added, dropped);
        }
    }

    /// What a stepping plan counts as drawn ([`PlanJob::drawn`]): the nodes
    /// handed out (or failed), except those inside a stale node (one the plan
    /// dropped that stands until what replaces it is all in). Such a node is
    /// not on screen yet: were it to split, its children would hold the
    /// stale one up again, a level at a time, and a coarse node stood over
    /// the near ground for many seconds after a flight.
    #[cfg(test)]
    fn drawn_for_steps(&self) -> Vec<CubeKey> {
        self.drawn_input().keys()
    }

    /// Copies of the sets [`Self::drawn_for_steps`] reads, for a plan job:
    /// copying them is a fraction of filtering every drawn node, which the
    /// planner thread does ([`Drawn::keys`]).
    fn drawn_input(&self) -> Drawn {
        Drawn {
            delivered: self.delivered.clone(),
            failed: self.failed.keys().copied().collect(),
            stale: self.stale.clone(),
        }
    }

    /// The look the queue's order is for now.
    fn look_now(&self) -> Look {
        self.look.unwrap_or_default()
    }

    /// A job's score ([`far_priority`]) for the camera and the look now.
    fn score(&self, key: CubeKey) -> f64 {
        job_score(self.camera, self.look_now(), key)
    }

    /// `keys` by score, highest first, then the bigger on-screen error.
    fn by_score(&self, keys: impl IntoIterator<Item = CubeKey>) -> Vec<CubeKey> {
        let mut scored: Vec<(f64, f64, CubeKey)> = keys
            .into_iter()
            .map(|k| (-self.score(k), -screen_error(self.camera, k), k))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        scored.into_iter().map(|e| e.2).collect()
    }

    /// Whether `key` lies inside a far node drawn over loaded near ground
    /// ([`track_lingering`]): it is part of what replaces that node.
    fn near_ground(&self) -> impl Fn(CubeKey) -> bool + '_ {
        let levels = self.lingering.keys().fold(0u32, |m, k| m | 1 << k.level);
        move |key: CubeKey| {
            levels != 0
                && (key.level + 1..=MAX_CUBE_LEVEL)
                    .filter(|l| levels >> l & 1 == 1)
                    .any(|l| self.lingering.contains_key(&ancestor(key, l)))
        }
    }

    /// Queues `keys` (not handed out) where they belong: those over near
    /// ground at the very front, nearest first; holes at the back of the
    /// first ones, by score; the rest in their buckets. `dropped` are no
    /// longer planned: skipped if queued.
    fn queue_jobs(&self, keys: Vec<CubeKey>, dropped: Vec<CubeKey>) {
        let camera = self.camera;
        let near = self.near_ground();
        let (mut urgent, rest): (Vec<CubeKey>, Vec<CubeKey>) =
            keys.into_iter().partition(|&k| near(k));
        urgent.sort_by(|a, b| a.distance(camera).total_cmp(&b.distance(camera)));
        let hole: FxHashSet<CubeKey> = (self.holes.iter().map(|h| h.0))
            .chain(self.holding.iter().copied())
            .collect();
        let (holes, mut rest): (Vec<CubeKey>, Vec<CubeKey>) =
            rest.into_iter().partition(|k| hole.contains(k));
        let holes = self.by_score(holes);
        rest.sort_by_cached_key(|&k| std::cmp::Reverse((screen_error(camera, k) * 1e6) as u64));
        let mut queue = self.shared.queue.lock().unwrap();
        for key in urgent.iter().chain(&holes).chain(&rest) {
            queue.dropped.remove(key);
        }
        queue.dropped.extend(dropped);
        for key in urgent.into_iter().rev() {
            queue.first.push_front(key);
        }
        queue.first.extend(holes);
        for key in rest {
            queue.push(bucket(camera, key), key);
        }
        drop(queue);
        self.shared.wake.notify_all();
    }

    /// The planned nodes among `keys` with nothing drawn over their place:
    /// not delivered, under no delivered node (a coverage node or an older
    /// plan's, kept by `removed`), and over none (an older plan's finer
    /// nodes, kept until this one is in). With `cover`, each comes with the
    /// coverage leaf over it, if any, and only a node as fine as that leaf or
    /// finer counts as drawn over it. Also returns the planned nodes holding
    /// an older plan's nodes: once in, those are let go (after a teleport,
    /// most of the last plan, far away under a few coarse nodes).
    fn find_holes(&self, keys: &[CubeKey], cover: bool) -> (Vec<Hole>, Vec<CubeKey>) {
        // Only nodes outside the plan can hold or be held by a planned node.
        let mut levels = 0u32;
        let mut inner = FxHashSet::default();
        for &s in &self.stale {
            levels |= 1 << s.level;
            let mut up = s;
            while up.level < MAX_CUBE_LEVEL && inner.insert(up.parent()) {
                up = up.parent();
            }
        }
        let mut holes = Vec::new();
        let mut holding = Vec::new();
        if !cover {
            for &key in keys {
                if self.delivered.contains(&key) {
                    continue;
                }
                if inner.contains(&key) {
                    holding.push(key);
                    continue;
                }
                let held = (key.level + 1..=MAX_CUBE_LEVEL)
                    .filter(|l| levels >> l & 1 == 1)
                    .any(|l| self.stale.contains(&ancestor(key, l)));
                if !held && !self.tile_drawn_over(key, MAX_CUBE_LEVEL) {
                    holes.push((key, None));
                }
            }
            return (holes, holding);
        }
        // Per node above a hole: whether it or a node over it is drawn, and
        // the coverage leaf at or over it. Siblings share the walk.
        let mut memo: FxHashMap<CubeKey, (bool, Option<CubeKey>)> = FxHashMap::default();
        let mut path = Vec::new();
        for &key in keys {
            if self.delivered.contains(&key) {
                continue;
            }
            if inner.contains(&key) {
                holding.push(key);
                continue;
            }
            path.clear();
            let mut found = (false, None);
            let mut up = key;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                if let Some(&m) = memo.get(&up) {
                    found = m;
                    break;
                }
                path.push(up);
            }
            // A node drawn over the place counts only at the coverage leaf's
            // size or finer: after a teleport, the last plan's nodes millions
            // of blocks wide still stand over the new place, a voxel there
            // tens of thousands of blocks tall.
            for &n in path.iter().rev() {
                found = if self.cover_plan.contains(&n) {
                    (self.stale.contains(&n), Some(n))
                } else if found.1.is_some() {
                    (found.0 || self.stale.contains(&n), found.1)
                } else {
                    (false, None)
                };
                memo.insert(n, found);
            }
            if found.0 {
                continue;
            }
            // Where the coverage layer has a tile over the node, the tile.
            let over = match found.1 {
                Some(c) => Some(FarKey::Cube(c)),
                None => self.cover_tile_over(key).map(FarKey::Tile),
            };
            let limit = match over {
                Some(FarKey::Cube(c)) => c.level,
                Some(FarKey::Tile(t)) => t.level,
                None => key.level,
            };
            if !self.tile_drawn_over(key, limit) {
                holes.push((key, over));
            }
        }
        (holes, holding)
    }

    /// Whether a drawn tile of level `limit` or finer covers `key`'s footprint.
    fn tile_drawn_over(&self, key: CubeKey, limit: u8) -> bool {
        if self.tiles.delivered.is_empty() {
            return false;
        }
        (key.level.max(SURFACE_LEVEL)..=limit.min(MAX_CUBE_LEVEL)).any(|l| {
            let up = l - key.level;
            self.tiles
                .delivered
                .contains(&TileKey::new(l, key.x >> up, key.z >> up))
        })
    }

    /// The coverage layer's tile over `key`'s footprint, if it has one.
    fn cover_tile_over(&self, key: CubeKey) -> Option<TileKey> {
        if self.tiles.cover.is_empty() {
            return None;
        }
        (key.level.max(SURFACE_LEVEL)..=MAX_CUBE_LEVEL)
            .map(|l| {
                let up = l - key.level;
                TileKey::new(l, key.x >> up, key.z >> up)
            })
            .find(|t| self.tiles.cover.contains(t))
    }

    /// The planned tiles among `keys` with nothing drawn over their
    /// footprint: not drawn, under no drawn tile (with `cover`, none more than
    /// [`COVER_LEVELS`] coarser counts) and over no drawn node or tile of an
    /// older plan.
    fn find_tile_holes(&self, keys: &[TileKey], cover: bool) -> Vec<TileKey> {
        if keys.is_empty() {
            return Vec::new();
        }
        // The footprints holding an older plan's tiles or nodes.
        let mut inner = FxHashSet::<TileKey>::default();
        let mut add = |level: u8, x: i32, z: i32, own: bool| {
            if own {
                inner.insert(TileKey::new(level, x, z));
            }
            let mut l = level;
            while l < MAX_CUBE_LEVEL {
                l += 1;
                let up = l - level;
                if !inner.insert(TileKey::new(l, x >> up, z >> up)) {
                    break;
                }
            }
        };
        for &t in &self.tiles.stale {
            add(t.level, t.x, t.z, false);
        }
        for &c in &self.stale {
            add(c.level, c.x, c.z, true);
        }
        keys.iter()
            .copied()
            .filter(|t| {
                if self.tiles.delivered.contains(t) || inner.contains(t) {
                    return false;
                }
                let limit = if cover {
                    t.level.saturating_add(COVER_LEVELS).min(MAX_CUBE_LEVEL)
                } else {
                    MAX_CUBE_LEVEL
                };
                !(t.level + 1..=limit).any(|l| {
                    let up = l - t.level;
                    self.tiles
                        .delivered
                        .contains(&TileKey::new(l, t.x >> up, t.z >> up))
                })
            })
            .collect()
    }

    /// Drops the holes that something now covers: the node itself, or the
    /// coverage node over it. A pass over every hole runs at most every
    /// [`SCAN_EVERY`] while there are many.
    fn recheck_holes(&mut self, now: Instant, force: bool) {
        if !self.holes_dirty || (self.holes.is_empty() && self.tiles.holes.is_empty()) {
            return;
        }
        // As `releasable`: a delivery every frame would otherwise rescan every
        // hole each frame; a covered hole leaves the list at most this later.
        let every = if self.holes.len() > SLOW_SCAN {
            SCAN_EVERY
        } else {
            release_every()
        };
        if !force && now.saturating_duration_since(self.holes_at) < every {
            return;
        }
        self.holes_dirty = false;
        self.holes_at = now;
        let (delivered, tiles) = (&self.delivered, &self.tiles.delivered);
        let before = self.holes.len();
        self.holes.retain(|(key, over)| {
            !(delivered.contains(key)
                || match over {
                    Some(FarKey::Cube(c)) => delivered.contains(c),
                    Some(FarKey::Tile(t)) => tiles.contains(t),
                    None => false,
                })
        });
        if self.holes.len() != before {
            self.holes_version += 1;
        }
        self.tiles.holes.retain(|t| !tiles.contains(t));
    }

    /// Queues only the coverage layer over the holes: a coverage node over a
    /// hole where the layer has one, else the hole itself (the plan is as
    /// coarse there).
    fn begin_cover(&mut self, now: Instant) {
        self.holes_version += 1;
        let since = match self.phase {
            Phase::Cover { since, .. } => since,
            Phase::Refine => now,
        };
        let mut seen = FxHashSet::default();
        let wave: Vec<FarKey> = self
            .holes
            .iter()
            .map(|&(key, over)| over.unwrap_or(FarKey::Cube(key)))
            .chain(self.holding.iter().map(|&k| FarKey::Cube(k)))
            .chain(self.tiles.holes.iter().map(|&t| FarKey::Tile(t)))
            .filter(|k| seen.insert(*k) && !self.far_delivered(*k))
            .collect();
        if wave.is_empty() {
            self.phase = Phase::Refine;
            self.fill_queue();
            return;
        }
        // The coverage tiles go to the renderer first: they are the fastest.
        let tiles: Vec<TileKey> = wave
            .iter()
            .filter_map(|k| match *k {
                FarKey::Tile(t) => Some(t),
                FarKey::Cube(_) => None,
            })
            .filter(|t| self.tiles.asked.insert(*t))
            .collect();
        self.tiles.work.request.splice(0..0, tiles);
        let cubes = self.by_score(wave.iter().filter_map(|k| match *k {
            FarKey::Cube(c) => Some(c),
            FarKey::Tile(_) => None,
        }));
        self.cover = cubes
            .iter()
            .copied()
            .filter(|k| !self.wanted.contains(k))
            .collect();
        self.stats.cover_nodes = wave.len();
        let mut queue = self.shared.queue.lock().unwrap();
        queue.clear();
        queue.first.extend(cubes);
        drop(queue);
        self.shared.wake.notify_all();
        self.phase = Phase::Cover { since, wave };
    }

    /// Whether a node or tile is drawn.
    fn far_delivered(&self, key: FarKey) -> bool {
        match key {
            FarKey::Cube(c) => self.delivered.contains(&c),
            FarKey::Tile(t) => self.tiles.delivered.contains(&t),
        }
    }

    /// Ends the coverage phase once its nodes are in and shown (or failed,
    /// or after [`COVER_WAIT`]), and queues the rest of the plan.
    fn advance_phase(&mut self, now: Instant) {
        let Phase::Cover { since, wave } = &self.phase else {
            return;
        };
        let landed = wave.iter().all(|k| match *k {
            FarKey::Cube(c) => {
                self.delivered.contains(&c)
                    || self.failed.contains_key(&c)
                    || !(self.wanted.contains(&c) || self.cover.contains(&c))
            }
            FarKey::Tile(t) => {
                self.tiles.delivered.contains(&t)
                    || !(self.tiles.asked.contains(&t) || self.tiles.wanted.contains(&t))
            }
        });
        if !(landed && self.shown()) && now.saturating_duration_since(*since) < COVER_WAIT {
            return;
        }
        self.phase = Phase::Refine;
        self.holes_version += 1;
        self.holes_dirty = true;
        self.recheck_holes(now, true);
        self.fill_queue();
    }

    /// Queues every planned node not handed out: the ones over near ground
    /// first (nearest first), then holes (by score), then the rest by bucket
    /// ([`bucket_order`]: the crosshair's and the near ones first).
    fn fill_queue(&mut self) {
        let now = Instant::now();
        {
            let (wanted, cover) = (&self.wanted, &self.cover);
            self.refreshing
                .retain(|k| wanted.contains(k) || cover.contains(k));
        }
        let waiting = |k: &CubeKey| {
            !self.delivered.contains(k)
                && self.failed.get(k).is_none_or(|at| {
                    now.saturating_duration_since(*at) >= far_ms("HORIZON_FAR_RETRY_MS", RETRY)
                })
        };
        self.linger_queued = self.lingering.keys().copied().collect();
        let camera = self.camera;
        let near = self.near_ground();
        let mut urgent: Vec<CubeKey> = if self.lingering.is_empty() {
            Vec::new()
        } else {
            (self.order.iter().map(|e| e.1))
                .filter(|&k| waiting(&k) && near(k))
                .collect()
        };
        urgent.sort_by(|a, b| a.distance(camera).total_cmp(&b.distance(camera)));
        let mut firsts: FxHashSet<CubeKey> = urgent.iter().copied().collect();
        let holes: Vec<CubeKey> = (self.holes.iter().map(|h| h.0))
            .chain(self.holding.iter().copied())
            .filter(|k| waiting(k) && !firsts.contains(k))
            .collect();
        let holes = self.by_score(holes);
        firsts.extend(holes.iter().copied());
        let mut queue = self.shared.queue.lock().unwrap();
        queue.clear();
        // Nodes to make again that the new plan dropped: not worth making.
        let (wanted, cover) = (&self.wanted, &self.cover);
        queue
            .later
            .retain(|k| wanted.contains(k) || cover.contains(k));
        queue.first.extend(urgent);
        queue.first.extend(holes);
        let mut pushed = queue.first.len();
        for &(b, key) in &self.order {
            if waiting(&key) && !firsts.contains(&key) {
                queue.push(b, key);
                pushed += 1;
            }
        }
        drop(queue);
        bump(&self.shared.waste.refills, 1);
        bump(&self.shared.waste.refill_keys, pushed);
        self.shared.wake.notify_all();
    }

    /// Asks again for nodes whose sampling failed a while ago.
    fn retry(&mut self, now: Instant) {
        if now < self.retry_at || self.failed.is_empty() {
            return;
        }
        self.retry_at = now + far_ms("HORIZON_FAR_RETRY_MS", RETRY) / 2;
        let (wanted, cover) = (&self.wanted, &self.cover);
        self.failed
            .retain(|k, _| wanted.contains(k) || cover.contains(k));
        let due: Vec<CubeKey> = self
            .failed
            .iter()
            .filter(|(_, at)| {
                now.saturating_duration_since(**at) >= far_ms("HORIZON_FAR_RETRY_MS", RETRY)
            })
            .map(|(&k, _)| k)
            .collect();
        if due.is_empty() {
            return;
        }
        for key in &due {
            self.failed.remove(key);
        }
        bump(&self.shared.waste.retries, due.len());
        self.queue_jobs(due, Vec::new());
    }

    /// Reorders the queue's buckets when the view turned past [`TURN`] or
    /// zoomed ([`Look::moved`]): no job is scored again.
    fn face(&mut self, look: Look) {
        if self.look.is_some_and(|l| !l.moved(&look)) {
            return;
        }
        self.look = Some(look);
        let order = bucket_order(look);
        self.shared.queue.lock().unwrap().reorder(order);
    }

    /// The nodes the renderer can drop: handed out before, no longer in the
    /// plan, and with every node over their place handed out too.
    ///
    /// At most [`RELEASE_BATCH`] a call: the renderer retires each with
    /// writes of its own, and a teleport lets a whole plan go at once.
    pub fn removed(&mut self) -> Vec<CubeKey> {
        if self.releasing.is_empty() {
            self.releasing = self.releasable();
        }
        let take = self.releasing.len().min(RELEASE_BATCH);
        let mut released: Vec<CubeKey> = self.releasing.drain(..take).collect();
        // Planned again since it was found, or a plan put in since: kept.
        released.retain(|k| self.stale.contains(k) && !self.wanted.contains(k));
        let short = far_ms("HORIZON_FAR_SHORT_MS", Duration::from_millis(5000));
        let now = Instant::now();
        let w = &self.shared.waste;
        for key in &released {
            self.delivered.remove(key);
            self.drawn_hash.remove(key);
            self.stale.remove(key);
            self.empty.remove(key);
            bump(&w.released, 1);
            if let Some(at) = self.delivered_at.remove(key)
                && now.saturating_duration_since(at) < short
            {
                bump(&w.short_lived, 1);
                bump(&w.short_lived_levels[usize::from(key.level).min(7)], 1);
            }
            if let Some(at) = self.refreshed_at.remove(key)
                && now.saturating_duration_since(at) < short
            {
                bump(&w.late_wasted, 1);
            }
        }
        let camera = self.camera;
        released.sort_by_cached_key(|k| (k.distance(camera) * 16.0) as u64);
        released
    }

    /// The stale nodes nothing needs any more: every node over their place
    /// is handed out, or none is wanted. A pass over the stale nodes, at most
    /// every [`SCAN_EVERY`] while many planned nodes are missing.
    fn releasable(&mut self) -> Vec<CubeKey> {
        let now = Instant::now();
        if (self.stale.is_empty() && self.tiles.stale.is_empty()) || !self.release_dirty {
            return Vec::new();
        }
        // A pass costs about the stale nodes and the missing ones; even a short
        // one runs at most every `release_every` (a delivery every frame would
        // otherwise rescan every stale node each frame at hundreds of fps).
        let every = if self.missing + self.stale.len() > SLOW_SCAN {
            SCAN_EVERY
        } else {
            release_every()
        };
        if now.saturating_duration_since(self.released_at) < every {
            return Vec::new();
        }
        self.release_dirty = false;
        self.released_at = now;
        // A node holding a planned node still missing covers a place nothing
        // else draws yet: the planned nodes inside it are a run of the tree.
        let blocked = |key: CubeKey| {
            if self.missing == 0 {
                return false;
            }
            let lo = tree_order(key);
            let hi = lo + (1u128 << (3 * u32::from(key.level)));
            let start = self.tree.partition_point(|e| e.0 < lo);
            self.tree[start..]
                .iter()
                .take_while(|e| e.0 < hi)
                .any(|e| e.1.level < key.level && !self.delivered.contains(&e.1))
        };
        // Per node over a stale one: whether it or one over it is planned
        // and not out yet (it then covers the stale node's place).
        let mut memo: FxHashMap<CubeKey, bool> = FxHashMap::default();
        let mut path = Vec::new();
        let mut released = Vec::new();
        for &key in &self.stale {
            // Under a planned tile not in yet, or (a coarse node) over one.
            if blocked(key)
                || self.tile_pending_over(key)
                || (key.level > SURFACE_LEVEL
                    && self.pending_in(TileKey::new(key.level, key.x, key.z), true))
            {
                continue;
            }
            path.clear();
            let mut covered = false;
            let mut up = key;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                if let Some(&m) = memo.get(&up) {
                    covered = m;
                    break;
                }
                path.push(up);
            }
            for &n in path.iter().rev() {
                covered = covered || (self.wanted.contains(&n) && !self.delivered.contains(&n));
                memo.insert(n, covered);
            }
            if !covered {
                released.push(key);
            }
        }
        // Tiles: faded out once nothing planned in or over their footprint
        // waits (few, and the renderer's fade is one write each).
        let retire: Vec<TileKey> = self
            .tiles
            .stale
            .iter()
            .copied()
            .filter(|&t| !(self.pending_in(t, false) || self.pending_over(t)))
            .collect();
        for t in &retire {
            self.tiles.delivered.remove(t);
            self.tiles.stale.remove(t);
        }
        self.tiles.work.retire.extend(retire);
        released
    }

    /// Whether a planned tile not drawn yet covers `key`'s footprint.
    fn tile_pending_over(&self, key: CubeKey) -> bool {
        self.tiles.missing > 0
            && (key.level.max(SURFACE_LEVEL)..=MAX_CUBE_LEVEL).any(|l| {
                let up = l - key.level;
                let t = TileKey::new(l, key.x >> up, key.z >> up);
                self.tiles.wanted.contains(&t) && !self.tiles.delivered.contains(&t)
            })
    }

    /// Whether something planned and not drawn yet lies in the footprint
    /// `fp` (a tile's, or a node's at its level): a tile, or (unless
    /// `tiles_only`) a node.
    fn pending_in(&self, fp: TileKey, tiles_only: bool) -> bool {
        if self.missing + self.tiles.missing == 0 {
            return false;
        }
        let lo = footprint_order(fp.level, fp.x, fp.z);
        let hi = lo + (1u64 << (2 * u32::from(fp.level)));
        let start = self.footprints.partition_point(|e| e.0 < lo);
        self.footprints[start..]
            .iter()
            .take_while(|e| e.0 < hi)
            .any(|e| {
                e.1 <= fp.level
                    && match e.2 {
                        FarKey::Cube(c) => !tiles_only && !self.delivered.contains(&c),
                        FarKey::Tile(t) => !self.tiles.delivered.contains(&t),
                    }
            })
    }

    /// Whether a planned tile or node not drawn yet has a footprint holding
    /// the footprint `fp` (a coarser one).
    fn pending_over(&self, fp: TileKey) -> bool {
        if self.missing + self.tiles.missing == 0 {
            return false;
        }
        (fp.level + 1..=MAX_CUBE_LEVEL).any(|l| {
            let up = l - fp.level;
            let (x, z) = (fp.x >> up, fp.z >> up);
            let t = TileKey::new(l, x, z);
            if self.tiles.wanted.contains(&t) && !self.tiles.delivered.contains(&t) {
                return true;
            }
            let code = footprint_order(l, x, z);
            let start = self.footprints.partition_point(|e| (e.0, e.1) < (code, l));
            self.footprints[start..]
                .iter()
                .take_while(|e| e.0 == code && e.1 == l)
                .any(|e| matches!(e.2, FarKey::Cube(c) if !self.delivered.contains(&c)))
        })
    }

    /// Forgets everything: nodes handed out, queued and cached.
    pub fn clear(&mut self) {
        {
            let mut queue = self.shared.queue.lock().unwrap();
            queue.clear();
            queue.later.clear();
        }
        self.refreshing.clear();
        self.drawn_hash.clear();
        self.ever.clear();
        self.delivered_at.clear();
        self.refreshed_at.clear();
        self.held_keys.clear();
        self.holes_version += 1;
        self.shared.cache.lock().unwrap().clear();
        {
            // A plan in the works belongs to the old count: dropped.
            let mut slot = self.shared.planner.lock().unwrap();
            slot.job = None;
            slot.done = None;
        }
        self.epoch += 1;
        // Results of jobs running now are dropped as unwanted.
        self.plan = Arc::default();
        self.wanted.clear();
        self.order.clear();
        self.tree.clear();
        self.footprints.clear();
        self.tiles = TileState {
            on: self.tiles.on,
            sea_level: self.tiles.sea_level,
            ..TileState::default()
        };
        self.cover_plan.clear();
        self.cover.clear();
        self.holes.clear();
        self.holding.clear();
        self.phase = Phase::Refine;
        self.delivered.clear();
        self.stale.clear();
        self.releasing.clear();
        self.missing = 0;
        self.failed.clear();
        self.empty.clear();
        self.empty_planned = 0;
        self.stats.plan_changed = None;
        self.stats.plan_complete = None;
        self.stats.plan_covered = None;
        self.asked = None;
        self.replan = false;
        self.jumped = None;
        self.held = 0;
        self.delivered_since = false;
        self.lingering.clear();
        for done in std::mem::take(&mut self.backlog)
            .into_iter()
            .chain(self.results.try_iter().collect::<Vec<_>>())
        {
            if let Some(Product::Gpu(node)) = &done.product {
                self.release(node);
            }
        }
        for node in std::mem::take(&mut self.gpu_out) {
            self.release(&node);
        }
    }
}

/// Notes the drawn nodes (`delivered`, less the `empty` ones) that hold
/// loaded near cubes (`snapshot`) and since when (`lingering`), for
/// [`CubeLodStats::linger_nodes`] and `linger_ms`.
fn track_lingering(
    delivered: &FxHashSet<CubeKey>,
    empty: &FxHashSet<CubeKey>,
    lingering: &mut FxHashMap<CubeKey, Instant>,
    stats: &mut CubeLodStats,
    snapshot: &LoadedSnapshot,
    now: Instant,
) {
    let counts = snapshot.counts();
    let mut next = FxHashMap::default();
    for &key in delivered {
        // The box test first: it is arithmetic, and most drawn nodes are far
        // outside the loaded box.
        if !counts.holds(key) || empty.contains(&key) {
            continue;
        }
        next.insert(key, lingering.get(&key).copied().unwrap_or(now));
    }
    stats.linger_nodes = next.len();
    stats.linger_ms = next
        .values()
        .map(|t| now.saturating_duration_since(*t).as_secs_f32() * 1000.0)
        .fold(0.0, f32::max);
    *lingering = next;
}

/// The node of `level` holding `key` (`level` at least `key.level`).
fn ancestor(key: CubeKey, level: u8) -> CubeKey {
    let up = level - key.level;
    CubeKey::new(level, key.x >> up, key.y >> up, key.z >> up)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

impl Drop for CubeLod {
    fn drop(&mut self) {
        {
            // Under the lock: a worker between its stop check and its wait
            // must not miss the wake.
            let _queue = self.shared.queue.lock();
            self.shared.stop.store(true, Ordering::Relaxed);
        }
        self.shared.wake.notify_all();
        {
            let _classes = self.shared.classes.lock();
            self.shared.classes_wake.notify_all();
        }
        {
            let _descent = self.shared.descent.lock();
            self.shared.descent_wake.notify_all();
        }
        {
            let _planner = self.shared.planner.lock();
            self.shared.planner_wake.notify_all();
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
        self.clear();
    }
}

/// What is known of `key`; unknown nodes are queued for the classifier thread
/// and count as mixed until it has looked.
fn classify_cached(shared: &Shared, key: CubeKey) -> NodeClass {
    if key.level < 2 {
        return NodeClass::Mixed;
    }
    let mut classes = shared.classes.lock().unwrap();
    if let Some(class) = classes.known.get(&key) {
        return class;
    }
    if shared.descent_on.load(Ordering::Relaxed) {
        // The descent fills the classes; nothing is asked of the classifier.
        return NodeClass::Mixed;
    }
    if classes.queued.insert(key) {
        classes.queue.push_back(key);
        drop(classes);
        shared.classes_wake.notify_one();
    }
    NodeClass::Mixed
}

/// Runs the GPU descent for each request (the latest wins) and files the
/// nodes it proved uniform as classes.
fn descent_loop(shared: &Shared, descent: &GpuDescent) {
    loop {
        let request = {
            let mut slot = shared.descent.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(request) = slot.request.take() {
                    slot.busy = true;
                    break request;
                }
                slot = shared.descent_wake.wait(slot).unwrap();
            }
        };
        let found = descent.descend(&request);
        if let Some(found) = &found {
            let mut classes = shared.classes.lock().unwrap();
            let uniform: Vec<_> = found.uniform().collect();
            classes.known.extend(uniform.into_iter());
        }
        let mut slot = shared.descent.lock().unwrap();
        slot.busy = false;
        slot.runs += 1;
        if let Some(found) = found {
            slot.uniform = found.uniform_count();
            slot.leaves = found.leaves.len();
            slot.millis = found.elapsed.as_secs_f32() * 1000.0;
            slot.overflow = found.overflow;
        }
        drop(slot);
        shared.classes_dirty.store(true, Ordering::Relaxed);
    }
}

fn classify_loop(shared: &Shared, sampler: &mut dyn CubeSampler) {
    loop {
        let key = {
            let mut classes = shared.classes.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(key) = classes.queue.pop_front() {
                    break key;
                }
                classes = shared.classes_wake.wait(classes).unwrap();
            }
        };
        let class = sampler.classify_ready(key);
        let mut classes = shared.classes.lock().unwrap();
        classes.queued.remove(&key);
        if let Some(class) = class {
            classes.known.insert(key, class);
        }
    }
}

/// The kind of a state, by name, cached per thread.
fn classifier(context: &WorkerContext, air: BlockStateId) -> impl Fn(BlockStateId) -> Kind + '_ {
    let cache = RefCell::new(FxHashMap::<u32, Kind>::default());
    move |state| {
        *cache.borrow_mut().entry(state.0).or_insert_with(|| {
            if state == air {
                return Kind::Air;
            }
            match context
                .assets
                .registry
                .state(state)
                .map_or("", |s| s.name.as_str())
            {
                "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air" => Kind::Air,
                "minecraft:water" | "minecraft:lava" | "minecraft:bubble_column" => Kind::Fluid,
                _ => Kind::Solid,
            }
        })
    }
}

fn work(
    index: usize,
    shared: Arc<Shared>,
    tx: mpsc::Sender<Done>,
    context: Arc<WorkerContext>,
    mut sampler: Box<dyn CubeSampler>,
    air: BlockStateId,
    backend: Option<GpuBackend>,
) {
    // The far workers go below the main/render thread for good (10 nice
    // levels on Linux, below normal on Windows, utility QoS on macOS) and stay
    // there: they used to step back to normal once the near terrain was in,
    // which let 8 of them take the frame's cores while walking. The near-loading
    // gate below (only `WORKERS_NEAR` workers while near terrain loads) stays.
    // `HORIZON_FAR_PRIORITY`: 0 (default) below main, with the near-loading gate;
    // 1 below main without the gate; 2 left as spawned (no gate).
    let priority = far_knob("HORIZON_FAR_PRIORITY", 0);
    if priority != 2 {
        horizon_threads::far_below_main();
    }
    let mut mesher = Mesher::default();
    let kind = classifier(&context, air);
    loop {
        let gpu = backend
            .as_ref()
            .filter(|_| shared.gpu_path.load(Ordering::Relaxed));
        // A backend with no generator (an Earth world's, `GpuBackend::pool_only`):
        // every node of every level is sampled on the CPU (1-4 ms each) and
        // uploaded. A batch of them would hold 64 keys out of the other
        // workers' reach for a quarter of a second and send none until the
        // last is made: such a worker takes one key at a time, as on the CPU
        // path, and only the GPU's meshing differs.
        let cpu_nodes = gpu.is_some_and(|b| !b.generates);
        // On the GPU path a worker takes a batch of the first jobs and asks
        // the sampler for them at once (`resident_many`); on the CPU path one.
        let gpu_batch = far_knob("HORIZON_FAR_GPU_BATCH", GPU_BATCH as u64).max(1) as usize;
        let take = if gpu.is_some() && !cpu_nodes {
            gpu_batch
        } else {
            1
        };
        if priority == 0 {
            let loading = shared.near_loading.load(Ordering::Relaxed);
            // While the near terrain loads, and for `NEAR_HOLD` after (its
            // `terrain_loading` flickers off between the batches of columns
            // arriving), only the first `WORKERS_NEAR` workers go on; the others
            // wait for it to be in. The near terrain is what the player stands
            // in: after a teleport to the surface it was in 21% sooner (4.17 s
            // against 5.31, 12 hops each), the far view complete 0.9 s later
            // (research/near-fill). HORIZON_FAR_WORKERS_NEAR / _NEAR_HOLD_MS.
            let hold = far_knob("HORIZON_FAR_NEAR_HOLD_MS", NEAR_HOLD.as_millis() as u64);
            let recent = hold > 0
                && shared.near_epoch.elapsed().as_millis() as u64
                    <= shared.near_seen_ms.load(Ordering::Relaxed) + hold;
            if (loading || recent)
                && index as u64 >= far_knob("HORIZON_FAR_WORKERS_NEAR", WORKERS_NEAR)
                && !shared.stop.load(Ordering::Relaxed)
            {
                thread::sleep(Duration::from_millis(5));
                continue;
            }
        }
        let profiling = profile::on();
        if profiling && profile::maybe_log() {
            profile::maybe_log_waste(&shared);
        }
        let started = Instant::now();
        if let Some(backend) = gpu {
            // The pool is full while the renderer meshes what is in it: wait
            // for room rather than fail the batch (a failed node waits RETRY).
            let mut waited = 0;
            // Room for the batch; for CPU nodes, for a few workers' nodes.
            let room = if cpu_nodes {
                far_knob("HORIZON_FAR_POOL_ROOM_CPU", CPU_POOL_ROOM as u64) as usize
            } else {
                gpu_batch
            };
            while backend.pool.free_slots() < room * far_knob("HORIZON_FAR_POOL_NEED", 1) as usize
                && waited < far_knob("HORIZON_FAR_POOL_WAITS", u64::from(POOL_WAITS)) as u32
                && !shared.stop.load(Ordering::Relaxed)
            {
                thread::sleep(Duration::from_micros(far_knob(
                    "HORIZON_FAR_POOL_WAIT_US",
                    POOL_WAIT.as_micros() as u64,
                )));
                waited += 1;
            }
        }
        if profiling {
            profile::add(&profile::POOL_WAIT, started);
        }
        let started = Instant::now();
        let keys: Vec<CubeKey> = {
            let mut queue = shared.queue.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                let mut keys = Vec::new();
                // The finest levels are sampled on the CPU one by one, before
                // the batch's GPU request: a few per batch, so they never hold
                // the GPU's nodes back for long. The rest wait at the front,
                // and the GPU's nodes behind them come in meanwhile.
                let (mut cpu, mut later) = (0, Vec::new());
                while keys.len() < take
                    && later.len() < 8 * take
                    && let Some(key) = queue.pop()
                {
                    if gpu.is_some() && key.level < MIN_LEVEL {
                        if cpu
                            == far_knob("HORIZON_FAR_CPU_PER_BATCH", CPU_PER_BATCH as u64) as usize
                        {
                            later.push(key);
                            continue;
                        }
                        cpu += 1;
                    }
                    queue.running.insert(key);
                    keys.push(key);
                }
                for key in later.into_iter().rev() {
                    queue.first.push_front(key);
                }
                if !keys.is_empty() {
                    break keys;
                }
                queue = shared.wake.wait(queue).unwrap();
            }
        };
        if profiling {
            profile::add(&profile::QUEUE_WAIT, started);
            profile::KEYS.fetch_add(keys.len() as u64, Ordering::Relaxed);
        }
        let started = Instant::now();
        match sampler.changes() {
            Changes::None => {}
            Changes::Cubes(cubes) => shared.cache.lock().unwrap().forget(&cubes),
            Changes::All => {
                let mut cache = shared.cache.lock().unwrap();
                cache.clear();
                cache.cleared += 1;
            }
        }
        let mut fetched: FxHashMap<CubeKey, Option<Resident>> = FxHashMap::default();
        if gpu.is_some() && !cpu_nodes && !shared.stop.load(Ordering::Relaxed) {
            let uncached: Vec<CubeKey> = {
                let mut cache = shared.cache.lock().unwrap();
                keys.iter()
                    .copied()
                    .filter(|&k| cache.lookup(k).is_none())
                    .collect()
            };
            if !uncached.is_empty() {
                let answers = sampler.resident_many(&uncached);
                fetched.extend(uncached.into_iter().zip(answers));
            }
        }
        if profiling {
            profile::add(&profile::RESIDENT, started);
        }
        for key in keys {
            let started = Instant::now();
            let mut mesh_hash = None;
            let product = match gpu {
                // Sampled as on the CPU path (cache, sampler, downsampled
                // fine levels), then uploaded for the GPU mesher.
                Some(backend) if cpu_nodes => node(&shared, sampler.as_mut(), key, air, &kind)
                    .and_then(|node| {
                        // Never drawn wrong: a node past the GPU mesher's ids
                        // (the world rule, `cpu_nodes_fit_gpu`, should have
                        // kept it off) sends the world back to the CPU.
                        if needs_cpu_mesh(&node) {
                            if !shared.gpu_unfit.swap(true, Ordering::Relaxed) {
                                log::warn!(
                                    "Horizon far: node {key:?} holds a block state or biome the GPU mesher cannot draw; far terrain back on the CPU mesher"
                                );
                            }
                            return None;
                        }
                        gpu_from_node(&node, air, backend)
                    })
                    .map(Product::Gpu),
                Some(backend) => gpu_node(
                    &shared,
                    sampler.as_mut(),
                    key,
                    air,
                    &kind,
                    backend,
                    fetched.remove(&key),
                )
                .map(Product::Gpu),
                None => node(&shared, sampler.as_mut(), key, air, &kind).map(|node| {
                    mesh_hash = Some(node.content_hash());
                    if let Some(voxel) = node.as_uniform() {
                        bump(&shared.waste.built_uniform, 1);
                        if voxel.state == air {
                            bump(&shared.waste.built_empty, 1);
                        }
                    }
                    let mut mesh = SectionMesh::default();
                    let sea = sampler.sea_surface();
                    mesh::mesh_cube(&mut mesher, &node, air, sea, &context, &mut mesh);
                    mesh.sort_solid();
                    Product::Mesh(mesh)
                }),
            };
            // What the worker made: counted for the waste log and the stats.
            {
                let w = &shared.waste;
                bump(&w.jobs, 1);
                match &product {
                    None => bump(&w.failed_jobs, 1),
                    Some(made) => {
                        bump(&w.built, 1);
                        if let Product::Gpu(GpuNode {
                            content: GpuContent::Uniform(voxel),
                            ..
                        }) = made
                        {
                            bump(&w.built_uniform, 1);
                            if voxel.state == air {
                                bump(&w.built_empty, 1);
                            }
                        }
                    }
                }
            }
            if profiling {
                if key.level < MIN_LEVEL || cpu_nodes {
                    profile::add(&profile::CPU_NODES, started);
                    profile::CPU_KEYS.fetch_add(1, Ordering::Relaxed);
                } else {
                    profile::add(&profile::GPU_NODES, started);
                }
            }
            let started = Instant::now();
            shared.queue.lock().unwrap().running.remove(&key);
            let hash = match &product {
                Some(Product::Gpu(node)) => node.hash,
                _ => mesh_hash,
            };
            if tx.send(Done { key, product, hash }).is_err() {
                return;
            }
            if profiling {
                profile::add(&profile::SEND, started);
            }
        }
    }
}

/// Far workers that go on while the near terrain loads (see `work`).
const WORKERS_NEAR: u64 = 1;
/// How long after the near terrain last loaded the others still wait.
const NEAR_HOLD: Duration = Duration::from_millis(1500);
/// Jobs a worker on the GPU path takes and asks for at once.
const GPU_BATCH: usize = 64;
/// At most this many of them are of the levels sampled on the CPU.
const CPU_PER_BATCH: usize = 2;
/// Free pool slots a worker of a backend without a generator waits for before
/// it samples a node (its node takes one; the other workers' take theirs).
const CPU_POOL_ROOM: usize = 32;

/// Whether the CPU nodes of a world with `states` block states and `biomes`
/// biomes can all go through the GPU mesher: its quads hold a 17-bit block
/// state id and an 8-bit biome id, and its material and biome colour tables
/// stop there (`gfx::cube_mesh`). A world past either stays on the CPU mesher
/// ([`CubeLod::gpu_path_allowed`]) rather than draw a wrong block or colour.
pub fn cpu_nodes_fit_gpu(states: usize, biomes: usize) -> bool {
    states <= crate::gfx::cube_mesh::MAX_STATE as usize + 1 && biomes <= 256
}
/// A worker finding the GPU pool without room for a batch waits this long, at
/// most this many times, before asking anyway.
const POOL_WAIT: Duration = Duration::from_millis(2);
const POOL_WAITS: u32 = 250;

/// Whether a CPU node cannot go through the GPU mesher, and must be meshed
/// on the CPU (`mesh::mesh_cube`): a voxel whose block state is past what a
/// quad holds (`gfx::cube_mesh::MAX_STATE`, 17 bits) or whose biome id is
/// past 255 (a quad's 8 bits, the slot's biome table). Fluid tops are no
/// reason: the GPU mesher draws them (`GpuNode::fluid_tops`).
pub fn needs_cpu_mesh(node: &CubeNode) -> bool {
    let unfit = |v: Voxel| v.state.0 > crate::gfx::cube_mesh::MAX_STATE || v.biome > 255;
    match node.as_uniform() {
        Some(voxel) => unfit(voxel),
        None => (0..VOLUME).any(|i| unfit(node.at(i))),
    }
}

/// A CPU node as a [`GpuNode`]: uniform and uncovered ones need no slot (a
/// uniform fluid's tops are drawn from [`GpuNode::fluid_tops`] alone), the
/// rest are uploaded into one. `None` when the pool is full.
fn gpu_from_node(node: &CubeNode, air: BlockStateId, backend: &GpuBackend) -> Option<GpuNode> {
    let cover = node.covered_mask().map(|mask| Box::new(*mask));
    let mut states: Vec<u32> = Vec::new();
    if let Some(voxel) = node.as_uniform() {
        // Its material is asked for even when it takes a slot (covered).
        states.push(voxel.state.0);
    } else {
        // A node holds a handful of states: a short list beats a hash set.
        let mut last = u32::MAX;
        for i in 0..VOLUME {
            let state = node.at(i).state.0;
            if state != last && !states.contains(&state) {
                states.push(state);
            }
            last = state;
        }
    }
    let content = match node.as_uniform() {
        // Air draws nothing, covered or not: no slot, no mesher job.
        Some(voxel) if cover.is_none() || voxel.state == air => GpuContent::Uniform(voxel),
        _ => {
            let slot = backend.pool.alloc()?;
            backend
                .pool
                .write(&backend.queue, slot, &slot_words(node, air.0));
            GpuContent::Slot(slot)
        }
    };
    Some(GpuNode {
        key: node.key,
        content,
        cover,
        fluid_tops: node.fluid_tops().map(|tops| Box::new(*tops)),
        states,
        hash: Some(node.content_hash()),
    })
}

/// The node at `key` for the GPU path: a slot the GPU generated it into, or a
/// CPU node (saved cubes, downsampled fine levels, the pre-pass's uniform
/// ones) uploaded to one.
/// `fetched` is the sampler's answer when the worker already asked for it in
/// a batch (`Some`, which may be `None`); otherwise it is asked here.
fn gpu_node(
    shared: &Shared,
    sampler: &mut dyn CubeSampler,
    key: CubeKey,
    air: BlockStateId,
    kind: &impl Fn(BlockStateId) -> Kind,
    backend: &GpuBackend,
    fetched: Option<Option<Resident>>,
) -> Option<GpuNode> {
    let cached = cache_get(shared, key);
    if let Some(node) = cached {
        if let Some(Some(Resident::Slot(slot))) = fetched {
            backend.pool.free(slot);
        }
        return gpu_from_node(&node, air, backend);
    }
    if shared.stop.load(Ordering::Relaxed) {
        if let Some(Some(Resident::Slot(slot))) = fetched {
            backend.pool.free(slot);
        }
        return None;
    }
    let answer = match fetched {
        Some(answer) => answer,
        None => sampler.resident(key),
    };
    match answer {
        Some(Resident::Slot(slot)) => Some(GpuNode {
            key,
            content: GpuContent::Slot(slot),
            cover: None,
            fluid_tops: None,
            states: Vec::new(),
            hash: None,
        }),
        Some(Resident::Node(node)) => {
            debug_assert_eq!(node.key, key);
            let node = Arc::new(node);
            cache_put(shared, node.clone());
            gpu_from_node(&node, air, backend)
        }
        None if (1..=3).contains(&key.level) => {
            let node = node(shared, sampler, key, air, kind)?;
            gpu_from_node(&node, air, backend)
        }
        None => None,
    }
}

/// The node at `key`: cached, sampled, or (fine levels the sampler cannot
/// answer) downsampled from its children.
fn node(
    shared: &Shared,
    sampler: &mut dyn CubeSampler,
    key: CubeKey,
    air: BlockStateId,
    kind: &impl Fn(BlockStateId) -> Kind,
) -> Option<Arc<CubeNode>> {
    if let Some(node) = cache_get(shared, key) {
        return Some(node);
    }
    if shared.stop.load(Ordering::Relaxed) {
        return None;
    }
    let node = match sampler.node(key) {
        Some(node) => node,
        None if (1..=3).contains(&key.level) => {
            let children = key.children();
            let mut nodes = Vec::with_capacity(8);
            for child in children {
                nodes.push(node(shared, sampler, child, air, kind)?);
            }
            let refs: [&CubeNode; 8] = std::array::from_fn(|i| &*nodes[i]);
            downsample(refs, air, kind)
        }
        None => return None,
    };
    debug_assert_eq!(node.key, key);
    let node = Arc::new(node);
    cache_put(shared, node.clone());
    Some(node)
}

#[cfg(test)]
mod tests {
    /// A world whose CPU nodes take the GPU path (a backend without a
    /// generator, Earth's) must have every block state id within the quads'
    /// 17 bits and every biome id within their 8; past either it stays on the
    /// CPU mesher.
    #[test]
    fn cpu_nodes_take_the_gpu_mesher_only_within_its_state_and_biome_ids() {
        use super::cpu_nodes_fit_gpu;
        use crate::gfx::cube_mesh::MAX_STATE;
        let all_states = MAX_STATE as usize + 1;
        // Today's registries and biome lists (vanilla: ~29k states, ~65 biomes).
        assert!(cpu_nodes_fit_gpu(29_873, 65));
        assert!(cpu_nodes_fit_gpu(all_states, 256));
        assert!(cpu_nodes_fit_gpu(0, 0));
        // One state or one biome more than a quad holds: CPU.
        assert!(!cpu_nodes_fit_gpu(all_states + 1, 65));
        assert!(!cpu_nodes_fit_gpu(29_873, 257));
        assert!(!cpu_nodes_fit_gpu(usize::MAX, usize::MAX));
    }

    /// Eight stored children with water tops (the sea's, on the top of the node
    /// at every level) make a parent with them: downsampling merges them, the
    /// way `keep_into` / `of_cached_children` / the fallback downsample build it.
    #[test]
    fn a_parent_of_eight_children_with_water_tops_has_them_merged() {
        use super::{CubeKey, CubeNode, FLUID_TOP_SCALE, Kind, NO_FLUID_TOP, Voxel, downsample};
        use horizon_world::registry::BlockStateId;
        let (air, water) = (BlockStateId(0), BlockStateId(9));
        let kind = |s: BlockStateId| if s == air { Kind::Air } else { Kind::Fluid };
        let parent = CubeKey::new(3, -4, -1, 2);
        let tops_of = |h: u16| {
            let mut tops = Box::new([NO_FLUID_TOP; 256]);
            tops.iter_mut().for_each(|t| *t = h);
            tops
        };
        let sea = |key: CubeKey, tops: Option<u16>| {
            CubeNode::uniform(
                key,
                Voxel {
                    state: water,
                    biome: 1,
                },
            )
            .with_fluid_tops(tops.map(tops_of))
        };
        // The sea: water on the top of the upper children (tops at their tops),
        // water going on through the lower ones (none).
        let children = parent.children();
        let nodes: Vec<CubeNode> = children
            .iter()
            .map(|&c| sea(c, (c.y > 2 * parent.y).then_some(FLUID_TOP_SCALE)))
            .collect();
        let refs = std::array::from_fn(|i| &nodes[i]);
        let merged = downsample(refs, air, kind);
        assert_eq!(merged.key, parent);
        let tops = merged
            .fluid_tops()
            .expect("the parent keeps the sea's tops");
        assert!(tops.iter().all(|&h| h == FLUID_TOP_SCALE));
        assert_eq!(merged.fluid_top(3, 4), Some(parent.size() as f32));
        // A lake halfway up the upper children: halfway up the upper half.
        let nodes: Vec<CubeNode> = children
            .iter()
            .map(|&c| sea(c, (c.y > 2 * parent.y).then_some(FLUID_TOP_SCALE / 2)))
            .collect();
        let refs = std::array::from_fn(|i| &nodes[i]);
        let merged = downsample(refs, air, kind);
        let want = (FLUID_TOP_SCALE / 2 + FLUID_TOP_SCALE) / 2;
        assert!(merged.fluid_tops().unwrap().iter().all(|&h| h == want));
        // No child with tops: none (the old rule).
        let nodes: Vec<CubeNode> = children.iter().map(|&c| sea(c, None)).collect();
        let refs = std::array::from_fn(|i| &nodes[i]);
        assert!(downsample(refs, air, kind).fluid_tops().is_none());
    }

    /// 6. The GPU path: a CPU node's tops ride in [`super::GpuNode::fluid_tops`]
    /// (the same units) and the GPU mesher draws them; only a state or biome
    /// past what a quad holds keeps a node on the CPU.
    #[test]
    fn a_node_with_water_tops_is_for_the_gpu_mesher() {
        use super::{CubeKey, CubeNode, FLUID_TOP_SCALE, NO_FLUID_TOP, Voxel, needs_cpu_mesh};
        use horizon_world::registry::BlockStateId;
        let key = CubeKey::new(4, 0, -1, 0);
        let voxel = |s| Voxel {
            state: BlockStateId(s),
            biome: 0,
        };
        let dry = CubeNode::from_fn(key, |x, _, _| voxel(u32::from(x > 7)));
        assert!(!needs_cpu_mesh(&dry));
        let mut tops = Box::new([NO_FLUID_TOP; 256]);
        tops[16 * 5 + 3] = FLUID_TOP_SCALE;
        let wet = dry.clone().with_fluid_tops(Some(tops));
        assert!(!needs_cpu_mesh(&wet));
        // What `gpu_from_node` copies into the GpuNode.
        let copy = wet.fluid_tops().map(|t| Box::new(*t)).unwrap();
        assert_eq!(copy[16 * 5 + 3], FLUID_TOP_SCALE);
        // A state past 17 bits or a biome past 255 does not fit a quad.
        let max = crate::gfx::cube_mesh::MAX_STATE;
        let big = |state, biome| {
            CubeNode::from_fn(key, move |x, _, _| {
                if x == 3 {
                    Voxel {
                        state: BlockStateId(state),
                        biome,
                    }
                } else {
                    voxel(0)
                }
            })
        };
        assert!(!needs_cpu_mesh(&big(max, 255)));
        assert!(needs_cpu_mesh(&big(max + 1, 0)));
        assert!(needs_cpu_mesh(&big(1, 256)));
        let past = Voxel {
            state: BlockStateId(max + 1),
            biome: 0,
        };
        assert!(needs_cpu_mesh(&CubeNode::uniform(key, past)));
    }

    #[test]
    fn known_classes_keep_the_last_answers_whole() {
        use super::{CubeKey, KnownClasses, NodeClass};
        let key = |i: usize| CubeKey::new(2, i as i32, 0, 0);
        let answer = |from: usize| (from..from + 150_000).map(move |i| (key(i), NodeClass::Air));
        let mut known = KnownClasses::default();
        // Two answers of 150,000 (an Ultra descent's): both kept.
        known.extend(answer(0));
        known.extend(answer(150_000));
        assert_eq!(known.len(), 300_000);
        assert!(known.get(&key(0)).is_some() && known.get(&key(299_999)).is_some());
        // A third: the first goes, the last two stay whole.
        known.extend(answer(300_000));
        assert!(known.get(&key(0)).is_none());
        assert!(known.get(&key(150_000)).is_some() && known.get(&key(449_999)).is_some());
    }

    #[test]
    fn the_arena_cap_holds_through_small_moves() {
        // Within the band the held cap stays; past it the cap follows.
        assert!(super::arena_cap_holds(100_000, 100_000));
        assert!(super::arena_cap_holds(100_000, 91_000));
        assert!(super::arena_cap_holds(100_000, 124_000));
        assert!(!super::arena_cap_holds(100_000, 89_000));
        assert!(!super::arena_cap_holds(100_000, 126_000));
    }

    use super::*;

    const AIR: BlockStateId = BlockStateId(0);
    const STONE: BlockStateId = BlockStateId(1);
    const GRASS: BlockStateId = BlockStateId(2);
    const WATER: BlockStateId = BlockStateId(3);
    const DIRT: BlockStateId = BlockStateId(4);

    fn kind(state: BlockStateId) -> Kind {
        match state {
            AIR => Kind::Air,
            WATER => Kind::Fluid,
            _ => Kind::Solid,
        }
    }
    fn voxel(state: BlockStateId) -> Voxel {
        Voxel { state, biome: 0 }
    }
    fn key(level: u8, x: i32, y: i32, z: i32) -> CubeKey {
        CubeKey::new(level, x, y, z)
    }

    #[test]
    fn keys_nest_at_negative_coordinates_and_the_world_edges() {
        for k in [
            key(0, -1, -1, -1),
            key(0, -17, 3, -2),
            key(3, -1, 0, 5),
            key(5, -3, -4, 2),
            key(0, 524_287, -524_288, -1_875_000),
        ] {
            let parent = k.parent();
            assert_eq!(parent.level, k.level + 1);
            assert!(parent.children().contains(&k), "{k:?}");
            assert!(parent.contains(k) && !k.contains(parent));
            for child in k.children_if_any() {
                assert_eq!(child.parent(), k);
            }
        }
        // Origins are exact multiples, floor-nested.
        assert_eq!(key(0, -1, 0, 0).origin(), [-16, 0, 0]);
        assert_eq!(key(2, -1, -1, 1).origin(), [-64, -64, 64]);
        assert_eq!(key(0, -1, 0, 0).parent().origin(), [-32, 0, 0]);
        // The extremes: y -8,388,608 and +8M blocks, x 30M, fit.
        let top = key(12, 0, 127, 0);
        assert_eq!(top.origin(), [0, 127 * 65_536, 0]);
        assert_eq!(key(0, 0, -(1 << 19), 0).origin()[1], -8_388_608);
        assert_eq!(
            key(0, 0, -(1 << 19), 0).containing_root(),
            key(MAX_CUBE_LEVEL, 0, (-(1i32 << 19)) >> MAX_CUBE_LEVEL, 0)
        );
        assert_eq!(key(0, 1_875_000, 0, 0).origin()[0], 30_000_000);
    }

    impl CubeKey {
        fn children_if_any(self) -> Vec<CubeKey> {
            if self.level == 0 {
                vec![]
            } else {
                self.children().to_vec()
            }
        }
        fn containing_root(self) -> CubeKey {
            let mut k = self;
            while k.level < MAX_CUBE_LEVEL {
                k = k.parent();
            }
            k
        }
    }

    #[test]
    fn a_node_holds_exactly_its_cubes() {
        let node = key(2, -1, 0, 1);
        assert!(node.contains_cube(SectionPos::new(-4, 0, 4)));
        assert!(node.contains_cube(SectionPos::new(-1, 3, 7)));
        assert!(!node.contains_cube(SectionPos::new(0, 0, 4)));
        assert!(!node.contains_cube(SectionPos::new(-5, 0, 4)));
        assert!(!node.contains_cube(SectionPos::new(-4, 4, 4)));
        assert_eq!(CubeKey::containing(2, SectionPos::new(-1, 3, 7)), node);
        assert_eq!(node.distance([-32.0, 32.0, 96.0]), 0.0);
        assert_eq!(node.distance([-100.0, 0.0, 64.0]), 36.0);
    }

    fn children_of(f: impl Fn(usize, usize, usize, usize) -> BlockStateId) -> Vec<CubeNode> {
        // f(child, x, y, z)
        let parent = key(1, 0, 0, 0);
        parent
            .children()
            .into_iter()
            .enumerate()
            .map(|(i, k)| CubeNode::from_fn(k, |x, y, z| voxel(f(i, x, y, z))))
            .collect()
    }
    fn down(children: &[CubeNode]) -> CubeNode {
        downsample(std::array::from_fn(|i| &children[i]), AIR, kind)
    }

    #[test]
    fn equal_uniform_children_stay_uniform() {
        let children = children_of(|_, _, _, _| STONE);
        let parent = down(&children);
        assert_eq!(parent.key, key(1, 0, 0, 0));
        assert_eq!(parent.as_uniform(), Some(voxel(STONE)));
    }

    #[test]
    fn a_voxel_is_solid_from_four_of_eight() {
        // In child 0, the voxel at (0,0,0) covers child voxels 0..2 each way.
        // `solid` of its eight cells are stone, the rest air.
        let with = |solid: usize| {
            let children = children_of(|c, x, y, z| {
                let cell = (x & 1) | ((z & 1) << 1) | ((y & 1) << 2);
                if c == 0 && x < 2 && y < 2 && z < 2 && cell < solid {
                    STONE
                } else {
                    AIR
                }
            });
            down(&children).get(0, 0, 0).state
        };
        assert_eq!(with(3), AIR);
        assert_eq!(with(4), STONE, "a tie keeps solid");
        assert_eq!(with(8), STONE);
        // The next voxel over saw only air.
        let children = children_of(|c, x, y, z| {
            if c == 0 && x == 0 && y == 0 && z == 0 {
                STONE
            } else {
                AIR
            }
        });
        assert_eq!(down(&children).get(1, 0, 0).state, AIR);
    }

    #[test]
    fn the_most_common_solid_wins_and_ties_go_to_the_higher_layer() {
        let at = |top: BlockStateId, bottom: BlockStateId| {
            let children = children_of(|c, x, y, z| {
                if c == 0 && x < 2 && y < 2 && z < 2 {
                    if y == 1 { top } else { bottom }
                } else {
                    AIR
                }
            });
            down(&children).get(0, 0, 0).state
        };
        assert_eq!(at(GRASS, DIRT), GRASS, "4 against 4: the top layer's");
        assert_eq!(at(DIRT, GRASS), DIRT);
        // Five stone cells against three grass: stone, wherever it is.
        let children = children_of(|c, x, y, z| {
            if c == 0 && x < 2 && y < 2 && z < 2 {
                if y == 1 && ((x == 0 && z == 0) || z == 1) {
                    GRASS
                } else {
                    STONE
                }
            } else {
                AIR
            }
        });
        // y=1 cells with (x=0,z=0) or z=1: three grass; five stone.
        assert_eq!(down(&children).get(0, 0, 0).state, STONE);
    }

    #[test]
    fn fluid_never_beats_a_solid_majority_and_needs_four_to_stand() {
        let mix = |solid: usize, fluid: usize| {
            let children = children_of(|c, x, y, z| {
                if c != 0 || x >= 2 || y >= 2 || z >= 2 {
                    return AIR;
                }
                let cell = (x & 1) | ((z & 1) << 1) | ((y & 1) << 2);
                if cell < solid {
                    STONE
                } else if cell < solid + fluid {
                    WATER
                } else {
                    AIR
                }
            });
            down(&children).get(0, 0, 0).state
        };
        assert_eq!(mix(4, 4), STONE);
        assert_eq!(mix(3, 5), WATER, "no solid majority, fluid has 5");
        assert_eq!(mix(0, 4), WATER);
        assert_eq!(mix(3, 3), AIR);
        assert_eq!(mix(2, 3), AIR);
    }

    #[test]
    fn each_parent_voxel_reads_its_own_child_and_corner() {
        // Stone only in child 5 (x half 1, z half 0, y half 1), its last
        // cell: parent voxel (8 + 7, 8 + 7, 7)... check positions map there.
        let children = children_of(|c, _, _, _| if c == 5 { STONE } else { AIR });
        let parent = down(&children);
        assert_eq!(parent.get(8, 8, 0).state, STONE);
        assert_eq!(parent.get(15, 15, 7).state, STONE);
        assert_eq!(parent.get(7, 8, 0).state, AIR);
        assert_eq!(parent.get(8, 7, 0).state, AIR);
        assert_eq!(parent.get(8, 8, 8).state, AIR);
    }

    // ---- plan

    const BOX: LoadedBox = LoadedBox {
        horizontal: 2,
        vertical: 1,
    };
    fn near(camera: [f64; 3]) -> impl Fn(SectionPos) -> bool {
        let c = cube_of(camera);
        move |p| {
            (p.x - c.x).abs() <= BOX.horizontal
                && (p.z - c.z).abs() <= BOX.horizontal
                && (p.y - c.y).abs() <= BOX.vertical
        }
    }

    /// How many leaves hold `cube`.
    fn holders(leaves: &FxHashSet<CubeKey>, cube: SectionPos) -> usize {
        (0..=MAX_CUBE_LEVEL)
            .filter(|&l| leaves.contains(&CubeKey::containing(l, cube)))
            .count()
    }

    fn check_plan(camera: [f64; 3], max: f64) -> Vec<CubeKey> {
        let loaded = near(camera);
        let leaves = plan(camera, max, BOX, &loaded);
        let set: FxHashSet<CubeKey> = leaves.iter().copied().collect();
        assert_eq!(set.len(), leaves.len(), "no node twice");
        // Non-overlapping: no leaf has a leaf ancestor.
        for &leaf in &leaves {
            let mut up = leaf;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                assert!(!set.contains(&up), "{leaf:?} inside {up:?}");
            }
        }
        // Every unloaded cube in range is held once, loaded ones never.
        let c = cube_of(camera);
        let r = (max / 16.0).ceil() as i32 + 1;
        let mut covered = 0;
        for y in c.y - r..=c.y + r {
            for z in c.z - r..=c.z + r {
                for x in c.x - r..=c.x + r {
                    let cube = SectionPos::new(x, y, z);
                    let n = holders(&set, cube);
                    if loaded(cube) {
                        assert_eq!(n, 0, "loaded {cube:?} drawn");
                        continue;
                    }
                    let d = CubeKey::containing(0, cube).distance(camera);
                    if d <= max {
                        assert_eq!(n, 1, "{cube:?} at {d} held {n} times");
                        covered += 1;
                    }
                }
            }
        }
        assert!(covered > 1000);
        leaves
    }

    /// Non-overlap only, for ranges too big to walk cube by cube.
    fn check_disjoint(camera: [f64; 3], max: f64) -> Vec<CubeKey> {
        let leaves = plan(camera, max, BOX, &near(camera));
        let set: FxHashSet<CubeKey> = leaves.iter().copied().collect();
        assert_eq!(set.len(), leaves.len());
        for &leaf in &leaves {
            let mut up = leaf;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                assert!(!set.contains(&up), "{leaf:?} inside {up:?}");
            }
        }
        leaves
    }

    #[test]
    fn the_plan_covers_the_range_once_minus_the_loaded_cubes() {
        check_plan([5.0, 70.0, 5.0], 400.0);
        check_plan([-1234.5, -3.0, 777.7], 300.0);
        check_plan([-15.9, -0.1, 0.1], 250.0);
        check_plan([29_999_000.0, 8_000_000.0, -29_999_000.0], 300.0);
        check_plan([100.0, -8_388_000.0, 100.0], 200.0);
    }

    #[test]
    fn the_level_grows_with_distance() {
        let camera = [3.0, 50.0, 3.0];
        let leaves = check_disjoint(camera, 2000.0);
        let mut by_level = [0; MAX_CUBE_LEVEL as usize + 1];
        for leaf in &leaves {
            by_level[leaf.level as usize] += 1;
            // Never coarser than the rule allows (the loaded box refines more).
            if leaf.level > 0 {
                let d = leaf.distance(camera);
                assert!(d >= DETAIL * f64::from(leaf.size()), "{leaf:?} at {d}");
            }
        }
        assert!(by_level[0] > 0 && by_level[3] > 0, "{by_level:?}");
        // Leaves are nearest first, and the nearest is a single cube.
        assert_eq!(leaves[0].level, 0);
        let near_level = |d: f64| {
            leaves
                .iter()
                .filter(|l| l.distance(camera) <= d)
                .map(|l| l.level)
                .max()
                .unwrap()
        };
        assert!(near_level(60.0) < near_level(600.0));
        assert!(near_level(600.0) < near_level(2000.0));
        // Detail is bounded: a few thousand nodes for a 2 km radius.
        assert!(leaves.len() < 20_000, "{}", leaves.len());
    }

    #[test]
    fn nothing_is_drawn_where_everything_is_loaded() {
        let camera = [0.5, 0.5, 0.5];
        // Everything within 3 cubes loaded: the small range is all near.
        let all = |_: SectionPos| true;
        let leaves = plan(
            camera,
            40.0,
            LoadedBox {
                horizontal: 100,
                vertical: 100,
            },
            &all,
        );
        assert!(leaves.is_empty());
    }

    #[test]
    fn a_small_camera_move_changes_few_nodes() {
        let a = [1000.0, 64.0, 1000.0];
        let b = [1003.0, 64.0, 1000.0];
        let pa: FxHashSet<_> = plan(a, 1500.0, BOX, &near(a)).into_iter().collect();
        let pb: FxHashSet<_> = plan(b, 1500.0, BOX, &near(b)).into_iter().collect();
        let changed = pa.symmetric_difference(&pb).count();
        assert!(changed * 4 < pa.len(), "{changed} of {}", pa.len());
    }

    fn assert_disjoint(leaves: &[CubeKey]) {
        let set: FxHashSet<CubeKey> = leaves.iter().copied().collect();
        assert_eq!(set.len(), leaves.len(), "no node twice");
        for &leaf in leaves {
            let mut up = leaf;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                assert!(!set.contains(&up), "{leaf:?} inside {up:?}");
            }
        }
    }

    #[test]
    fn a_split_node_merges_back_only_well_past_the_distance_that_split_it() {
        let nowhere = |_: SectionPos| false;
        let node = key(3, 0, 0, 0);
        let children = node.children().to_vec();
        // 340 blocks away: past 2.5 x 128 = 320 (a fresh plan keeps the node
        // whole) and inside 1.25 times that (a plan that had it split keeps
        // it split).
        let camera = [-340.0, 64.0, 64.0];
        let fresh = plan_from(camera, 2000.0, BOX, &nowhere, &[]);
        assert!(fresh.contains(&node));
        let kept = plan_from(camera, 2000.0, BOX, &nowhere, &children);
        assert_disjoint(&kept);
        assert!(!kept.contains(&node));
        assert!(children.iter().all(|c| kept.contains(c)), "{kept:?}");
        // 500 blocks away it merges, and nothing of its children stays.
        let away = [-500.0, 64.0, 64.0];
        let merged = plan_from(away, 2000.0, BOX, &nowhere, &children);
        assert_disjoint(&merged);
        assert!(merged.contains(&node));
        assert!(!merged.iter().any(|l| node.contains(*l) && *l != node));
        // Closer than the split distance, a plan that never split it does.
        let close = [-300.0, 64.0, 64.0];
        let split = plan_from(close, 2000.0, BOX, &nowhere, &[node]);
        assert!(!split.contains(&node));
    }

    #[test]
    fn a_camera_moving_a_little_back_and_forth_changes_no_node() {
        // Both in the same cube, so the loaded box is the same.
        let a = [1000.5, 64.5, 1000.5];
        let b = [1003.5, 64.5, 1000.5];
        let max = 1500.0;
        // Leaves near enough to both cameras not to be cut by the range.
        let settled = |leaves: &[CubeKey]| -> FxHashSet<CubeKey> {
            leaves
                .iter()
                .copied()
                .filter(|l| l.distance(a) < max - 10.0 && l.distance(b) < max - 10.0)
                .collect()
        };
        let mut previous = plan_from(a, max, BOX, &near(a), &[]);
        let mut seen = Vec::new();
        for step in 0..6 {
            let camera = if step % 2 == 0 { b } else { a };
            previous = plan_from(camera, max, BOX, &near(camera), &previous);
            assert_disjoint(&previous);
            seen.push(settled(&previous));
        }
        assert!(!seen[0].is_empty());
        for later in &seen[1..] {
            assert_eq!(*later, seen[0], "the plan flipped on a 3 block move");
        }
    }

    #[test]
    fn a_parent_column_is_covered_when_two_of_the_four_columns_over_it_are() {
        let cover = |columns: &[(usize, usize)]| {
            let mut mask = Box::new([0u64; 4]);
            for &(x, z) in columns {
                let bit = z << 4 | x;
                mask[bit >> 6] |= 1 << (bit & 63);
            }
            Some(mask)
        };
        let mut children = children_of(|_, _, _, _| STONE);
        assert!(!down(&children).column_covered(0, 0));
        // Child 4 is the upper one at x = z = 0: parent column (0, 0) holds
        // its columns (0..2, 0..2), parent column (1, 0) the ones at x 2..4.
        children[4] = children[4]
            .clone()
            .with_covered(cover(&[(0, 0), (1, 0), (2, 0)]));
        // A lower child's cover is under the node: it counts for nothing.
        children[0] = children[0]
            .clone()
            .with_covered(cover(&[(4, 4), (5, 4), (4, 5), (5, 5)]));
        let parent = down(&children);
        assert!(parent.column_covered(0, 0), "two of four");
        assert!(!parent.column_covered(1, 0), "one of four");
        assert!(!parent.column_covered(2, 2), "the lower child's");
        assert_eq!(parent.as_uniform(), Some(voxel(STONE)));
    }

    #[test]
    fn forgetting_cubes_drops_the_kept_nodes_over_them() {
        let mut cache = Cache::default();
        let kept = [key(0, 0, 0, 0), key(1, 5, 5, 5), key(2, -1, 0, 0)];
        let dropped = [key(1, 0, 0, 0), key(3, 0, 0, 0), key(0, 1, 1, 1)];
        for k in kept.iter().chain(&dropped) {
            cache.insert(Arc::new(CubeNode::uniform(*k, voxel(STONE))));
        }
        cache.forget(&[SectionPos::new(1, 1, 1)]);
        assert!(kept.iter().all(|k| cache.nodes.contains_key(k)));
        assert!(dropped.iter().all(|k| !cache.nodes.contains_key(k)));
    }

    // ---- producer

    /// A world of stone below y = 20.
    struct Flat;
    impl CubeSampler for Flat {
        fn node(&mut self, key: CubeKey) -> Option<CubeNode> {
            let scale = 1 << key.level;
            let oy = key.origin()[1];
            Some(CubeNode::from_fn(key, |_, y, _| {
                let wy = oy + y as i32 * scale;
                voxel(if wy < 20 { STONE } else { AIR })
            }))
        }
        fn fork(&self) -> Option<Box<dyn CubeSampler>> {
            Some(Box::new(Flat))
        }
    }

    #[test]
    fn nodes_the_sampler_lacks_are_built_from_their_children() {
        struct OnlyCubes;
        impl CubeSampler for OnlyCubes {
            fn node(&mut self, key: CubeKey) -> Option<CubeNode> {
                (key.level == 0).then(|| Flat.node(key).unwrap())
            }
            fn fork(&self) -> Option<Box<dyn CubeSampler>> {
                None
            }
        }
        let shared = Shared::new();
        let k = key(2, 0, 0, 0);
        let built = node(&shared, &mut OnlyCubes, k, AIR, &kind).unwrap();
        let direct = Flat.node(k).unwrap();
        for i in 0..VOLUME {
            assert_eq!(built.at(i), direct.at(i), "voxel {i}");
        }
        // Too coarse to build from children.
        assert!(node(&shared, &mut OnlyCubes, key(6, 0, 0, 0), AIR, &kind).is_none());
        // The children are cached for the next parent.
        assert!(shared.cache.lock().unwrap().nodes.len() >= 8);
    }

    /// A sampler whose data has not arrived says so (`None`): the classifier thread does not file the node as
    /// unknown for good, it is asked about again at a later plan.
    #[test]
    fn a_classification_that_cannot_be_made_yet_is_asked_for_again_not_remembered() {
        struct Waits(u32);
        impl CubeSampler for Waits {
            fn node(&mut self, _: CubeKey) -> Option<CubeNode> {
                None
            }
            fn fork(&self) -> Option<Box<dyn CubeSampler>> {
                None
            }
            fn classifies(&self) -> bool {
                true
            }
            fn classify_ready(&mut self, _: CubeKey) -> Option<NodeClass> {
                self.0 += 1;
                (self.0 > 1).then_some(NodeClass::Air)
            }
        }
        assert!(Waits(0).classifies() && !Flat.classifies());
        let shared = Arc::new(Shared::new());
        let k = key(3, 0, 0, 0);
        assert_eq!(classify_cached(&shared, k), NodeClass::Mixed);
        let worker = {
            let shared = shared.clone();
            std::thread::spawn(move || classify_loop(&shared, &mut Waits(0)))
        };
        let settled = |want: Option<NodeClass>| {
            for _ in 0..500 {
                {
                    let classes = shared.classes.lock().unwrap();
                    if classes.queued.is_empty() && classes.known.get(&k) == want {
                        return true;
                    }
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            false
        };
        // The first answer was "not yet": nothing is known, and the node is not waiting either.
        assert!(settled(None));
        // The next plan asks again, and now the answer is kept.
        assert_eq!(classify_cached(&shared, k), NodeClass::Mixed);
        assert!(settled(Some(NodeClass::Air)));
        assert_eq!(classify_cached(&shared, k), NodeClass::Air);
        shared.stop.store(true, Ordering::Relaxed);
        shared.classes_wake.notify_all();
        worker.join().unwrap();
    }

    /// A node key from its box of blocks ([`LateContent::not_drawn`]).
    #[test]
    fn a_node_is_found_from_its_own_box_only() {
        for key in [
            CubeKey::new(0, 3, -2, 7),
            CubeKey::new(4, -1, 0, 5),
            CubeKey::new(9, 0, -3, -1),
        ] {
            let (o, size) = (key.origin(), key.size());
            let max = o.map(|v| v + size - 1);
            assert_eq!(CubeKey::of_box(o, max), Some(key));
        }
        // Not a node's extent: off its grid, or not a cube.
        assert_eq!(CubeKey::of_box([1, 0, 0], [16, 15, 15]), None);
        assert_eq!(CubeKey::of_box([0, 0, 0], [15, 31, 15]), None);
        assert_eq!(CubeKey::of_box([0, 0, 0], [20, 20, 20]), None);
    }

    /// Late content (structures found after the terrain) has the drawn
    /// nodes over its boxes made again once the plan is complete: only those
    /// it touches, after every other job, and not once a plan drops them.
    #[test]
    fn late_content_remakes_the_drawn_nodes_over_it_after_the_rest() {
        struct Late(std::sync::Mutex<Vec<([i32; 3], [i32; 3])>>);
        impl LateContent for Late {
            fn view(&self, _: [f64; 3], _: [f64; 3], _: f64, _: bool) {}
            fn take_refreshes(&self) -> Vec<([i32; 3], [i32; 3])> {
                std::mem::take(&mut *self.0.lock().unwrap())
            }
        }
        let mut lod = lod_for_test();
        let (near, far, other) = (key(1, 0, 0, 0), key(1, 4, 0, 0), key(1, 9, 0, 0));
        put(&mut lod, &[near, far, other], &[]);
        lod.deliver(near, false);
        lod.deliver(far, false);
        let late = Arc::new(Late(std::sync::Mutex::new(vec![([20, 5, 5], [140, 6, 6])])));
        lod.late = Some(late.clone());
        // Not complete (`other` is missing): nothing taken yet.
        lod.late_content([0.0; 3], [0.0, 0.0, 1.0], 1e4);
        assert!(lod.refreshing.is_empty());
        lod.deliver(other, false);
        // The workers took the plan's jobs.
        lod.shared.queue.lock().unwrap().clear();
        lod.late_content([0.0; 3], [0.0, 0.0, 1.0], 1e4);
        let mut remade: Vec<CubeKey> = lod.refreshing.iter().copied().collect();
        remade.sort_by_key(|k| k.x);
        assert_eq!(remade, [near, far], "the nodes the box crosses");
        // Behind the plan's own jobs.
        lod.shared
            .queue
            .lock()
            .unwrap()
            .first
            .push_back(key(0, 50, 0, 0));
        let order = lod.shared.queue.lock().unwrap().drain();
        assert_eq!(order[0], key(0, 50, 0, 0));
        assert_eq!(order.len(), 3);
        // A plan without them drops them.
        lod.late_content([0.0; 3], [0.0, 0.0, 1.0], 1e4);
        late.0.lock().unwrap().push(([0, 0, 0], [140, 6, 6]));
        lod.late_content([0.0; 3], [0.0, 0.0, 1.0], 1e4);
        put(&mut lod, &[other], &[]);
        assert!(lod.refreshing.is_empty());
        assert!(lod.shared.queue.lock().unwrap().later.is_empty());
    }

    /// Late content hears which of its boxes are no drawn node's own extent
    /// (the plan dropped the node), so it does not wait on them.
    #[test]
    fn late_content_hears_of_boxes_with_no_drawn_node() {
        type Boxes = Vec<([i32; 3], [i32; 3])>;
        struct Late(std::sync::Mutex<Boxes>, std::sync::Mutex<Boxes>);
        impl LateContent for Late {
            fn view(&self, _: [f64; 3], _: [f64; 3], _: f64, _: bool) {}
            fn take_refreshes(&self) -> Vec<([i32; 3], [i32; 3])> {
                std::mem::take(&mut *self.0.lock().unwrap())
            }
            fn not_drawn(&self, boxes: &[([i32; 3], [i32; 3])]) {
                self.1.lock().unwrap().extend_from_slice(boxes);
            }
        }
        let mut lod = lod_for_test();
        let drawn = key(1, 0, 0, 0);
        put(&mut lod, &[drawn], &[]);
        lod.deliver(drawn, false);
        let box_of = |k: CubeKey| {
            let (o, size) = (k.origin(), k.size());
            (o, o.map(|v| v + size - 1))
        };
        let dropped = key(1, 7, 0, 0);
        let late = Arc::new(Late(
            std::sync::Mutex::new(vec![box_of(drawn), box_of(dropped)]),
            std::sync::Mutex::new(Vec::new()),
        ));
        lod.late = Some(late.clone());
        lod.shared.queue.lock().unwrap().clear();
        lod.late_content([0.0; 3], [0.0, 0.0, 1.0], 1e4);
        assert_eq!(*late.1.lock().unwrap(), [box_of(dropped)]);
        assert_eq!(lod.refreshing.iter().copied().collect::<Vec<_>>(), [drawn]);
    }

    /// A node made again is asked of the sampler again, not taken back from
    /// the workers' memory (where the out-of-date one is).
    #[test]
    fn late_content_forgets_the_kept_node_it_makes_again() {
        struct Late(std::sync::Mutex<Vec<([i32; 3], [i32; 3])>>);
        impl LateContent for Late {
            fn view(&self, _: [f64; 3], _: [f64; 3], _: f64, _: bool) {}
            fn take_refreshes(&self) -> Vec<([i32; 3], [i32; 3])> {
                std::mem::take(&mut *self.0.lock().unwrap())
            }
        }
        let mut lod = lod_for_test();
        let drawn = key(0, 2, 0, 0);
        put(&mut lod, &[drawn], &[]);
        lod.deliver(drawn, false);
        let air = BlockStateId(0);
        let node = CubeNode::from_fn(drawn, |_, _, _| Voxel {
            state: air,
            biome: 0,
        });
        lod.shared.cache.lock().unwrap().insert(Arc::new(node));
        assert!(lod.shared.cache.lock().unwrap().get(drawn).is_some());
        let (o, size) = (drawn.origin(), drawn.size());
        lod.late = Some(Arc::new(Late(std::sync::Mutex::new(vec![(
            o,
            o.map(|v| v + size - 1),
        )]))));
        lod.shared.queue.lock().unwrap().clear();
        lod.late_content([0.0; 3], [0.0, 0.0, 1.0], 1e4);
        assert_eq!(lod.refreshing.iter().copied().collect::<Vec<_>>(), [drawn]);
        assert!(lod.shared.cache.lock().unwrap().get(drawn).is_none());
    }

    fn lod_for_test() -> CubeLod {
        let shared = Arc::new(Shared::new());
        let (_tx, results) = mpsc::channel();
        CubeLod::assemble(shared, results, Vec::new(), None, None, AIR)
    }

    /// Puts `keys` in as the plan, as the planner thread would hand it over,
    /// with `cover` as its coverage layer.
    fn put(lod: &mut CubeLod, keys: &[CubeKey], cover: &[CubeKey]) {
        put_far(lod, keys, &[], cover);
    }

    /// [`put`] with surface tiles.
    fn put_far(lod: &mut CubeLod, keys: &[CubeKey], tiles: &[TileKey], cover: &[CubeKey]) {
        let next_tiles: FxHashSet<TileKey> = tiles.iter().copied().collect();
        let tiles_added = tiles
            .iter()
            .copied()
            .filter(|k| !lod.tiles.wanted.contains(k))
            .collect();
        let tiles_removed = lod
            .tiles
            .wanted
            .iter()
            .copied()
            .filter(|k| !next_tiles.contains(k))
            .collect();
        let mut footprints: Vec<(u64, u8, FarKey)> = keys
            .iter()
            .map(|&k| (footprint_order(k.level, k.x, k.z), k.level, FarKey::Cube(k)))
            .chain(
                tiles
                    .iter()
                    .map(|&t| (footprint_order(t.level, t.x, t.z), t.level, FarKey::Tile(t))),
            )
            .collect();
        footprints.sort_unstable();
        let next: FxHashSet<CubeKey> = keys.iter().copied().collect();
        let added = keys
            .iter()
            .copied()
            .filter(|k| !lod.wanted.contains(k))
            .collect();
        let removed = lod
            .wanted
            .iter()
            .copied()
            .filter(|k| !next.contains(k))
            .collect();
        let camera = lod.camera;
        let mut leaves = keys.to_vec();
        leaves.sort_by(|a, b| a.distance(camera).total_cmp(&b.distance(camera)));
        let mut tree: Vec<(u128, CubeKey)> = keys.iter().map(|&k| (tree_order(k), k)).collect();
        tree.sort_unstable();
        let done = PlanDone {
            epoch: lod.epoch,
            tree,
            order: leaves.iter().map(|&k| (bucket(camera, k), k)).collect(),
            leaves: Arc::new(leaves),
            changed: true,
            added,
            removed,
            cover: cover.iter().copied().collect(),
            tiles: Arc::new(tiles.to_vec()),
            tiles_added,
            tiles_removed,
            cover_tiles: FxHashSet::default(),
            footprints,
            pruned: 0,
            limited: false,
            coarsened: 1.0,
            tiles_limited: false,
            held: 0,
            held_keys: Vec::new(),
            elapsed: Duration::ZERO,
        };
        lod.install(done, Instant::now());
    }

    /// The plan of `camera` with nothing loaded and every node mixed, refined
    /// from what `drawn` lists ([`PlanInput::refined`]; `None`: no stepping).
    fn stepped(
        camera: [f64; 3],
        drawn: Option<&[CubeKey]>,
        is_loaded: &dyn Fn(SectionPos) -> bool,
    ) -> Vec<CubeKey> {
        let refined = drawn.map(refined_set);
        plan_far(&PlanInput {
            camera,
            max_distance: 3000.0,
            rule: &Rule::distance(DETAIL),
            loaded: BOX,
            is_loaded,
            previous: &[],
            class: &|_| NodeClass::Mixed,
            surface: None,
            refined: refined.as_ref(),
        })
        .leaves
    }

    #[test]
    fn the_far_view_refines_one_level_at_a_time_down_to_the_target() {
        let camera = [100.5, 64.5, -40.5];
        let nowhere = |_: SectionPos| false;
        // Whether a node's box takes in the near box: those split at once,
        // the steps are for the land beyond.
        let c = cube_of(camera);
        let over_near = |k: CubeKey| {
            let side = 1_i32 << k.level;
            let o = [k.x, k.y, k.z].map(|v| v * side);
            let (h, v) = (BOX.horizontal, BOX.vertical);
            o[0] <= c.x + h
                && c.x - h < o[0] + side
                && o[1] <= c.y + v
                && c.y - v < o[1] + side
                && o[2] <= c.z + h
                && c.z - h < o[2] + side
        };
        let target: FxHashSet<CubeKey> = stepped(camera, None, &nowhere).into_iter().collect();
        // Nothing drawn: roots, and beside the near box the children of the
        // nodes split around it.
        let mut drawn = stepped(camera, Some(&[]), &nowhere);
        let roots = root_nodes(camera, 3000.0);
        for k in &drawn {
            assert!(roots.contains(k) || over_near(k.parent()), "{k:?}");
        }
        let mut steps = 0;
        loop {
            let next = stepped(camera, Some(&drawn), &nowhere);
            assert_disjoint(&next);
            let before: FxHashSet<CubeKey> = drawn.iter().copied().collect();
            // Each new node is a drawn node's child, never a grandchild.
            for k in next.iter().filter(|k| !before.contains(k)) {
                assert!(
                    before.contains(&k.parent()) || over_near(k.parent()),
                    "{k:?} skips a level"
                );
            }
            if next.iter().copied().collect::<FxHashSet<_>>() == before {
                break;
            }
            drawn = next;
            steps += 1;
            assert!(steps < 32);
        }
        // It ends at the plan without stepping, after several visible steps.
        assert_eq!(drawn.into_iter().collect::<FxHashSet<_>>(), target);
        assert!(steps >= 3, "{steps}");
    }

    #[test]
    fn a_node_whose_eight_are_air_is_left_out_not_held_back_and_forth() {
        let camera = [100.5, 64.5, -40.5];
        let nowhere = |_: SectionPos| false;
        // A mixed node the target splits, whose children are all air (its class is coarser than theirs).
        let lone = stepped(camera, None, &nowhere)
            .into_iter()
            .filter(|k| k.level >= 3)
            .max_by(|a, b| a.distance(camera).total_cmp(&b.distance(camera)))
            .unwrap()
            .parent();
        let class = |k: CubeKey| {
            if k.level < lone.level && ancestor(k, lone.level) == lone {
                NodeClass::Air
            } else {
                NodeClass::Mixed
            }
        };
        let plan = |drawn: &[CubeKey]| {
            let refined = refined_set(drawn);
            plan_far(&PlanInput {
                camera,
                max_distance: 3000.0,
                rule: &Rule::distance(DETAIL),
                loaded: BOX,
                is_loaded: &nowhere,
                previous: drawn,
                class: &class,
                surface: None,
                refined: Some(&refined),
            })
            .leaves
        };
        let mut drawn = plan(&[]);
        let mut seen = vec![drawn.clone()];
        for _ in 0..40 {
            let next = plan(&drawn);
            if next == drawn {
                break;
            }
            assert!(
                !seen.contains(&next),
                "the plan went back to an earlier one"
            );
            seen.push(next.clone());
            drawn = next;
        }
        assert_eq!(plan(&drawn), drawn, "the plan settles");
        assert!(
            !drawn
                .iter()
                .any(|k| ancestor(*k, lone.level.max(k.level)) == lone || *k == lone)
        );
    }

    #[test]
    fn no_far_node_is_drawn_over_near_cubes_even_while_stepping() {
        let camera = [8.5, 64.5, 8.5];
        let loaded = near(camera);
        // Nothing drawn yet: the far land waits a level, the ground by the
        // near cubes does not.
        for leaf in stepped(camera, Some(&[]), &loaded) {
            let o = leaf.origin();
            let side = 1_i32 << leaf.level;
            let holds = (0..side).any(|y| {
                (0..side).any(|z| {
                    (0..side).any(|x| {
                        loaded(SectionPos::new(
                            (o[0] >> 4) + x,
                            (o[1] >> 4) + y,
                            (o[2] >> 4) + z,
                        ))
                    })
                })
            });
            assert!(!holds, "{leaf:?} is over near cubes");
        }
    }

    #[test]
    fn a_parent_goes_as_soon_as_its_eight_are_in_while_other_work_waits() {
        let mut lod = lod_for_test();
        let parent = key(2, 0, 0, 0);
        let other = key(2, 5, 0, 5);
        put(&mut lod, &[parent, other], &[]);
        lod.deliver(parent, false);
        let mut next = parent.children().to_vec();
        next.push(other);
        put(&mut lod, &next, &[]);
        for child in parent.children() {
            lod.deliver(child, false);
        }
        // `other` still waits; the parent goes anyway.
        assert!(!lod.delivered.contains(&other));
        assert_eq!(lod.removed(), vec![parent]);
    }

    #[test]
    fn far_nodes_drawn_over_near_cubes_are_counted_with_how_long() {
        let camera = [8.5, 64.5, 8.5];
        let c = cube_of(camera);
        let snapshot = LoadedSnapshot::take(c, BOX, &near(camera));
        let over = CubeKey::containing(1, c);
        let away = CubeKey::containing(1, SectionPos::new(c.x + 400, c.y, c.z));
        let empty = FxHashSet::default();
        let mut lingering = FxHashMap::default();
        let mut stats = CubeLodStats::default();
        let delivered: FxHashSet<CubeKey> = [over, away].into_iter().collect();
        let t0 = Instant::now();
        track_lingering(
            &delivered,
            &empty,
            &mut lingering,
            &mut stats,
            &snapshot,
            t0,
        );
        assert_eq!(stats.linger_nodes, 1);
        let later = t0 + Duration::from_millis(700);
        track_lingering(
            &delivered,
            &empty,
            &mut lingering,
            &mut stats,
            &snapshot,
            later,
        );
        assert!((stats.linger_ms - 700.0).abs() < 1.0, "{}", stats.linger_ms);
        let gone: FxHashSet<CubeKey> = [away].into_iter().collect();
        track_lingering(&gone, &empty, &mut lingering, &mut stats, &snapshot, later);
        assert_eq!((stats.linger_nodes, stats.linger_ms), (0, 0.0));
    }

    #[test]
    fn a_node_is_removed_only_once_what_replaces_it_is_in() {
        let mut lod = lod_for_test();
        let coarse = key(1, 0, 0, 0);
        let kids = coarse.children();
        put(&mut lod, &[coarse], &[]);
        lod.deliver(coarse, false);
        // The plan now wants the eight children; none is in yet.
        put(&mut lod, &kids, &[]);
        assert!(lod.removed().is_empty(), "children missing: a hole");
        for &k in &kids[..7] {
            lod.deliver(k, false);
        }
        assert!(lod.removed().is_empty(), "one child missing");
        lod.deliver(kids[7], false);
        assert_eq!(lod.removed(), vec![coarse]);
        assert!(lod.removed().is_empty(), "reported once");
        // The other way: children out, parent wanted.
        put(&mut lod, &[coarse], &[]);
        assert!(lod.removed().is_empty(), "the parent is not in");
        lod.deliver(coarse, false);
        assert_eq!(lod.removed().len(), 8);
        // Out of range, nothing replaces it.
        put(&mut lod, &[], &[]);
        assert_eq!(lod.removed(), vec![coarse]);
    }

    // ---- per-frame delivery budget

    fn budget(nodes: usize) -> DeliverBudget {
        DeliverBudget {
            nodes,
            bytes: 0,
            time: Duration::ZERO,
        }
    }

    /// A test far view whose worker channel the test feeds.
    fn lod_fed() -> (CubeLod, mpsc::Sender<Done>) {
        let shared = Arc::new(Shared::new());
        let (tx, results) = mpsc::channel();
        (
            CubeLod::assemble(shared, results, Vec::new(), None, None, AIR),
            tx,
        )
    }

    /// A worker's finished node with no vertices (it draws nothing).
    fn made(key: CubeKey) -> Done {
        Done {
            key,
            product: Some(Product::Mesh(SectionMesh::default())),
            hash: None,
        }
    }

    fn taken(lod: &mut CubeLod, budget: DeliverBudget) -> Vec<CubeKey> {
        let mut out = Vec::new();
        lod.drain_results_within(Instant::now(), &mut out, budget);
        out.into_iter().map(|(k, _)| k).collect()
    }

    #[test]
    fn the_delivery_budget_is_reached_by_nodes_bytes_or_time() {
        let b = DeliverBudget {
            nodes: 4,
            bytes: 1000,
            time: Duration::from_micros(500),
        };
        let spent = |nodes, bytes| Spent { nodes, bytes };
        let quick = Duration::from_micros(10);
        assert!(!b.reached(&spent(3, 999), quick));
        assert!(b.reached(&spent(4, 0), quick));
        assert!(b.reached(&spent(1, 1000), quick));
        assert!(b.reached(&spent(1, 0), Duration::from_micros(500)));
        // A knob of 0 is no limit of its kind.
        let open = DeliverBudget {
            nodes: 0,
            bytes: 0,
            time: Duration::ZERO,
        };
        assert!(!open.reached(&spent(1 << 20, 1 << 30), Duration::from_secs(9)));
        let d = DeliverBudget {
            nodes: DeliverBudget::NODES as usize,
            bytes: DeliverBudget::BYTES as usize,
            time: Duration::from_micros(DeliverBudget::US),
        };
        assert_eq!((d.nodes, d.bytes, d.time.as_micros()), (256, 4 << 20, 1000));
    }

    #[test]
    fn a_frame_takes_no_more_than_the_node_budget_and_leaves_the_rest() {
        let (mut lod, tx) = lod_fed();
        // Ten nodes of ten different parents: no sibling group to finish.
        let keys: Vec<CubeKey> = (0..10).map(|i| key(0, i * 2, 0, 0)).collect();
        put(&mut lod, &keys, &[]);
        for &k in &keys {
            tx.send(made(k)).unwrap();
        }
        assert_eq!(taken(&mut lod, budget(3)), keys[..3]);
        assert_eq!(lod.backlog.len(), 7);
        assert_eq!(lod.stats().deliver_deferred, 7);
        assert_eq!(lod.stats().deliver_frames_capped, 1);
        assert_eq!(lod.stats().pending, 7, "what waits is still missing");
        assert_eq!(taken(&mut lod, budget(3)), keys[3..6]);
        // The last four fit in a frame of 5: that frame is not capped.
        assert_eq!(taken(&mut lod, budget(5)), keys[6..]);
        assert!(lod.backlog.is_empty());
        assert_eq!(lod.stats().deliver_frames_capped, 2);
        assert_eq!(lod.stats().pending, 0);
    }

    #[test]
    fn the_time_budget_stops_a_frame_too() {
        let (mut lod, tx) = lod_fed();
        let keys: Vec<CubeKey> = (0..6).map(|i| key(0, i * 2, 0, 0)).collect();
        put(&mut lod, &keys, &[]);
        for &k in &keys {
            tx.send(made(k)).unwrap();
        }
        // One nanosecond is spent by the first node: one a frame.
        let tiny = DeliverBudget {
            nodes: 0,
            bytes: 0,
            time: Duration::from_nanos(1),
        };
        assert_eq!(taken(&mut lod, tiny), keys[..1]);
        assert_eq!(lod.backlog.len(), 5);
    }

    #[test]
    fn a_sibling_group_is_taken_together_whatever_the_budget() {
        let (mut lod, tx) = lod_fed();
        let parent = key(1, 4, 0, 0);
        let kids = parent.children();
        let others = [key(0, 100, 0, 0), key(0, 102, 0, 0), key(0, 104, 0, 0)];
        let all: Vec<CubeKey> = kids.iter().chain(&others).copied().collect();
        put(&mut lod, &all, &[]);
        // The first child, other nodes in between, then the seven others.
        tx.send(made(kids[0])).unwrap();
        tx.send(made(others[0])).unwrap();
        tx.send(made(others[1])).unwrap();
        for &k in &kids[1..] {
            tx.send(made(k)).unwrap();
        }
        tx.send(made(others[2])).unwrap();
        let got = taken(&mut lod, budget(1));
        assert_eq!(got.len(), 8, "{got:?}");
        assert!(kids.iter().all(|k| got.contains(k)));
        // The nodes of other parents wait, in their order.
        assert_eq!(
            lod.backlog.iter().map(|d| d.key).collect::<Vec<_>>(),
            others
        );
    }

    #[test]
    fn a_parent_is_not_removed_while_its_children_wait_in_the_backlog() {
        let (mut lod, tx) = lod_fed();
        let coarse = key(1, 0, 0, 0);
        let kids = coarse.children();
        let other = key(0, 100, 0, 0);
        put(&mut lod, &[coarse, other], &[]);
        lod.deliver(coarse, false);
        let next: Vec<CubeKey> = kids.iter().copied().chain([other]).collect();
        put(&mut lod, &next, &[]);
        // Another node first uses the frame's budget; the eight children wait.
        tx.send(made(other)).unwrap();
        for &k in &kids {
            tx.send(made(k)).unwrap();
        }
        assert_eq!(taken(&mut lod, budget(1)), [other]);
        assert_eq!(lod.backlog.len(), 8);
        assert!(lod.removed().is_empty(), "the replacement is not out yet");
        // Next frame all eight come out together, and only then the parent goes.
        assert_eq!(taken(&mut lod, budget(1)).len(), 8);
        assert_eq!(lod.removed(), vec![coarse]);
    }

    #[test]
    fn a_swap_set_out_of_order_over_frames_releases_the_parent_once_complete() {
        let (mut lod, tx) = lod_fed();
        let coarse = key(1, 0, 0, 0);
        let kids = coarse.children();
        put(&mut lod, &[coarse], &[]);
        lod.deliver(coarse, false);
        put(&mut lod, &kids, &[]);
        // Children finish out of order, three frames apart.
        for &i in &[5usize, 2, 7] {
            tx.send(made(kids[i])).unwrap();
        }
        assert_eq!(taken(&mut lod, budget(1)).len(), 3);
        assert!(lod.removed().is_empty());
        for &i in &[0usize, 6, 1, 4] {
            tx.send(made(kids[i])).unwrap();
        }
        assert_eq!(taken(&mut lod, budget(2)).len(), 4);
        assert!(lod.removed().is_empty(), "one child missing");
        tx.send(made(kids[3])).unwrap();
        assert_eq!(taken(&mut lod, budget(2)).len(), 1);
        assert_eq!(lod.removed(), vec![coarse]);
    }

    #[test]
    fn results_nobody_wants_use_no_budget_and_clear_drops_the_backlog() {
        let (mut lod, tx) = lod_fed();
        let keys: Vec<CubeKey> = (0..4).map(|i| key(0, i * 2, 0, 0)).collect();
        put(&mut lod, &keys[..2], &[]);
        // Two unplanned nodes first, then the two planned.
        tx.send(made(keys[2])).unwrap();
        tx.send(made(keys[3])).unwrap();
        tx.send(made(keys[0])).unwrap();
        tx.send(made(keys[1])).unwrap();
        assert_eq!(taken(&mut lod, budget(2)), keys[..2]);
        assert!(lod.backlog.is_empty());
        put(&mut lod, &keys, &[]);
        tx.send(made(keys[2])).unwrap();
        tx.send(made(keys[3])).unwrap();
        assert_eq!(taken(&mut lod, budget(1)), keys[2..3]);
        assert_eq!(lod.backlog.len(), 1);
        lod.clear();
        assert!(lod.backlog.is_empty());
    }

    #[test]
    fn the_plan_is_dropped_by_clear() {
        let mut lod = lod_for_test();
        put(&mut lod, &[key(0, 0, 0, 0)], &[]);
        lod.deliver(key(0, 0, 0, 0), false);
        lod.clear();
        assert!(lod.plan.is_empty() && lod.delivered.is_empty());
        assert!(lod.removed().is_empty());
        assert_eq!(lod.stats().pending, 0);
    }

    // ---- detail by size on screen

    const NONE: LoadedBox = LoadedBox {
        horizontal: 0,
        vertical: 0,
    };

    fn mixed(_: CubeKey) -> NodeClass {
        NodeClass::Mixed
    }

    /// The leaf of `plan` holding the block at `p`.
    fn leaf_at(plan: &[CubeKey], p: [f64; 3]) -> CubeKey {
        let cube = cube_of(p);
        *plan
            .iter()
            .find(|k| k.contains_cube(cube))
            .unwrap_or_else(|| panic!("nothing holds {p:?}"))
    }

    /// Pixels a voxel of `node` spans at its nearest point from `camera`,
    /// with a vertical field of view of `fov` degrees, 720 pixels tall.
    fn voxel_px(node: CubeKey, camera: [f64; 3], fov: f64) -> f64 {
        let focal = 720.0 / (2.0 * (fov.to_radians() / 2.0).tan());
        f64::from(1 << node.level) / node.distance(camera) * focal
    }

    /// The angle in degrees between `forward` and the direction to `node`'s
    /// centre.
    fn off_axis(node: CubeKey, camera: [f64; 3], forward: [f64; 3]) -> f64 {
        let half = f64::from(node.size()) / 2.0;
        let o = node.origin();
        let d = [0, 1, 2].map(|a| f64::from(o[a]) + half - camera[a]);
        angle_between(d, forward).to_degrees()
    }

    #[test]
    fn a_narrower_view_refines_the_nodes_in_view_only() {
        let camera = [0.5, 100.5, 0.5];
        let ahead = [0.0, 0.0, 1.0];
        let view = FarView::standard(8.0);
        // The spyglass: a tenth of the field of view.
        let zoom = view.looking(ahead, 7f64.to_radians());
        let range = 30_000.0;
        let wide = plan_classified(
            camera,
            range,
            &view.rule(usize::MAX),
            NONE,
            &|_| false,
            &[],
            &mixed,
        );
        let narrow = plan_classified(
            camera,
            range,
            &zoom.rule(usize::MAX),
            NONE,
            &|_| false,
            &[],
            &mixed,
        );
        let far_ahead = [0.5, 100.5, 8000.5];
        let far_behind = [0.5, 100.5, -8000.5];
        // Ten times the magnification is a bit over three levels finer.
        let (w, n) = (
            leaf_at(&wide.leaves, far_ahead).level,
            leaf_at(&narrow.leaves, far_ahead).level,
        );
        assert!(n + 3 <= w, "ahead: level {n} zoomed, {w} not");
        assert_eq!(
            leaf_at(&narrow.leaves, far_behind),
            leaf_at(&wide.leaves, far_behind),
            "behind the camera the zoom changes nothing"
        );
        // On screen, no voxel is over the target at its node's nearest point;
        // and each leaf's parent would have been.
        for (plan, fov, within) in [(&wide.leaves, 70.0, 20.0), (&narrow.leaves, 7.0, 2.0)] {
            let mut seen = 0;
            for &leaf in plan.iter().filter(|k| k.level > 0) {
                if off_axis(leaf, camera, ahead) > within {
                    continue;
                }
                seen += 1;
                let px = voxel_px(leaf, camera, fov);
                assert!(px <= 8.0 + 1e-9, "{leaf:?}: {px} px at {fov} degrees");
                let parent = voxel_px(leaf.parent(), camera, fov);
                assert!(parent > 8.0, "{leaf:?} split needlessly: {parent} px");
            }
            assert!(seen > 20, "{seen} leaves in view at {fov} degrees");
        }
        // Behind the zoomed camera, the unzoomed rule holds.
        for &leaf in narrow.leaves.iter().filter(|k| k.level > 0) {
            if off_axis(leaf, camera, ahead) > 120.0 {
                assert!(voxel_px(leaf, camera, 70.0) <= 8.0 + 1e-9, "{leaf:?}");
            }
        }
    }

    #[test]
    fn a_finer_target_asks_for_more_nodes_and_the_budget_coarsens_evenly() {
        let camera = [0.5, 100.5, 0.5];
        let range = 8000.0;
        let plan_for = |target: f64, budget: usize| {
            plan_classified(
                camera,
                range,
                &FarView::standard(target).rule(budget),
                NONE,
                &|_| false,
                &[],
                &mixed,
            )
        };
        let coarse = plan_for(16.0, usize::MAX);
        let fine = plan_for(4.0, usize::MAX);
        assert!(!coarse.limited && !fine.limited);
        assert!(fine.leaves.len() > coarse.leaves.len() * 4);
        let budget = fine.leaves.len() / 3;
        let capped = plan_for(4.0, budget);
        assert!(capped.limited && capped.coarsened > 1.0);
        assert!(capped.leaves.len() <= budget, "{}", capped.leaves.len());
        assert_disjoint(&capped.leaves);
        // What is smallest on screen stays whole first: no leaf's voxels are
        // over `coarsened` times the target, and no node was split under it.
        let over = |k: CubeKey| voxel_px(k, camera, 70.0) / 4.0;
        for &leaf in capped.leaves.iter().filter(|k| k.level > 0) {
            assert!(over(leaf) <= capped.coarsened + 1e-9, "{leaf:?}");
            assert!(over(leaf.parent()) >= capped.coarsened - 1e-9, "{leaf:?}");
        }
        // Far out it is coarser than the full plan, and every block in range
        // is still drawn once.
        let far = [0.5, 100.5, 7000.5];
        assert!(leaf_at(&capped.leaves, far).level >= leaf_at(&fine.leaves, far).level);
        for p in [
            far,
            [40.5, 100.5, 20.5],
            [-3000.5, -300.5, 4000.5],
            [5000.5, 1500.5, -100.5],
        ] {
            let cube = cube_of(p);
            let holders = capped
                .leaves
                .iter()
                .filter(|k| k.contains_cube(cube))
                .count();
            assert_eq!(holders, 1, "{p:?}");
        }
    }

    #[test]
    fn zooming_out_merges_only_past_the_hysteresis() {
        let camera = [0.5, 100.5, 0.5];
        let ahead = [0.0, 0.0, 1.0];
        let view = FarView::standard(8.0);
        let zoomed = view.looking(ahead, 7f64.to_radians());
        let range = 30_000.0;
        let plan_with = |v: &FarView, previous: &[CubeKey]| {
            plan_classified(
                camera,
                range,
                &v.rule(usize::MAX),
                NONE,
                &|_| false,
                previous,
                &mixed,
            )
            .leaves
        };
        let narrow = plan_with(&zoomed, &[]);
        let back = plan_with(&view, &narrow);
        assert_disjoint(&back);
        assert!(back.len() < narrow.len());
        // A node stays split after the zoom only while its voxels are more
        // than 1 / MERGE of the target on screen; a leaf is never over it.
        let focal = view.focal_px(view.fov_y);
        let ratio = |k: CubeKey| f64::from(1 << k.level) / k.distance(camera) * focal / 8.0;
        for &leaf in back.iter().filter(|k| k.level > 0) {
            assert!(ratio(leaf) <= 1.0 + 1e-9, "{leaf:?}");
            assert!(ratio(leaf.parent()) * MERGE > 1.0, "{leaf:?} kept split");
        }
        // A sprint's wider view and back again changes nothing.
        let sprint = view.looking(ahead, 77f64.to_radians());
        let settled = plan_with(&view, &plan_with(&view, &[]));
        let after = plan_with(&view, &plan_with(&sprint, &settled));
        let set = |v: &[CubeKey]| v.iter().copied().collect::<FxHashSet<_>>();
        assert_eq!(set(&after), set(&settled));
    }

    #[test]
    fn tiles_from_the_surface_level_complex_ones_sooner() {
        let high = key(SURFACE_LEVEL, 3, 0, -2);
        // Not generated yet: a tile first.
        assert_eq!(node_kind(high, None, None), NodeKind::Tile);
        let plain = Some(TileInfo::default());
        let complex = Some(TileInfo {
            overflow_columns: 3,
            ..TileInfo::default()
        });
        assert_eq!(node_kind(high, None, plain), NodeKind::Tile);
        assert_eq!(node_kind(high.children()[0], None, plain), NodeKind::Cube);
        // A complex tile turns into cube nodes sooner.
        assert_eq!(node_kind(high, None, complex), NodeKind::Cube);
        let higher = key(SURFACE_LEVEL + COMPLEX_LIFT, 0, 0, 0);
        assert_eq!(node_kind(higher, None, complex), NodeKind::Tile);
    }

    // ---- surface tiles in the plan

    const WHOLE: f64 = 67_108_864.0;

    /// Air above y 1,000, rock below y -1,000: the terrain lies between.
    fn banded(k: CubeKey) -> NodeClass {
        let [_, y, _] = k.origin();
        if i64::from(y) > 1000 {
            NodeClass::Air
        } else if i64::from(y) + i64::from(k.size()) < -1000 {
            NodeClass::Rock
        } else {
            NodeClass::Mixed
        }
    }

    /// A Whole World plan with tiles, at 16 px, with the headers `info` gives.
    fn tiled(camera: [f64; 3], info: &dyn Fn(TileKey) -> Option<TileInfo>) -> Plan {
        let loaded = near(camera);
        plan_far(&PlanInput {
            camera,
            max_distance: WHOLE,
            rule: &FarView::standard(16.0).rule(usize::MAX),
            loaded: BOX,
            is_loaded: &loaded,
            previous: &[],
            class: &banded,
            surface: Some(Surface {
                info,
                sea_level: 63,
                budget: usize::MAX,
                previous: &[],
            }),
            refined: None,
        })
    }

    /// Whether the footprint of `t` holds the block column `(x, z)`.
    fn tile_holds(t: TileKey, x: i64, z: i64) -> bool {
        let [ox, oz] = t.origin().map(i64::from);
        let size = i64::from(t.size());
        (ox..ox + size).contains(&x) && (oz..oz + size).contains(&z)
    }

    #[test]
    fn no_cube_node_shares_a_footprint_with_a_tile() {
        let camera = [0.5, 100.5, 0.5];
        let plan = tiled(camera, &|_| None);
        assert!(plan.tiles.len() > 100, "{} tiles", plan.tiles.len());
        assert!(!plan.leaves.is_empty());
        let tiles: FxHashSet<TileKey> = plan.tiles.iter().copied().collect();
        assert_eq!(tiles.len(), plan.tiles.len());
        // Tiles overlap neither each other nor any node's footprint, at any height.
        let mut under = FxHashSet::default();
        for &t in &plan.tiles {
            for l in t.level + 1..=MAX_CUBE_LEVEL {
                let up = l - t.level;
                let a = TileKey::new(l, t.x >> up, t.z >> up);
                assert!(!tiles.contains(&a), "{t:?} inside {a:?}");
                under.insert(a);
            }
        }
        for &c in &plan.leaves {
            assert!(
                !under.contains(&TileKey::new(c.level, c.x, c.z)),
                "{c:?} over a tile"
            );
            for l in c.level..=MAX_CUBE_LEVEL {
                let up = l - c.level;
                let t = TileKey::new(l, c.x >> up, c.z >> up);
                assert!(!tiles.contains(&t), "{c:?} under {t:?}");
            }
        }
        // Far away the ground is a tile, and no tile is finer than the
        // surface level.
        assert!(
            plan.tiles
                .iter()
                .any(|&t| tile_holds(t, 1_000_000, -2_000_000))
        );
        assert!(plan.tiles.iter().all(|t| t.level >= SURFACE_LEVEL));
    }

    #[test]
    fn the_camera_footprint_is_never_a_tile() {
        for camera in [
            [0.5, 100.5, 0.5],
            // Deep underground, in a cavern: the surface far above stays
            // tiles elsewhere, the rock between hides it.
            [12_345.5, -2000.5, -6789.5],
            // On a corner of the surface level's tiles.
            [65_535.5, 80.5, 65_535.5],
        ] {
            let plan = tiled(camera, &|_| None);
            assert!(!plan.tiles.is_empty());
            let c = cube_of(camera);
            let h = i64::from(BOX.horizontal);
            for &t in &plan.tiles {
                for dz in -h..=h {
                    for dx in -h..=h {
                        let x = (i64::from(c.x) + dx) * 16;
                        let z = (i64::from(c.z) + dz) * 16;
                        assert!(
                            !tile_holds(t, x, z),
                            "{t:?} over the loaded cubes at {camera:?}"
                        );
                    }
                }
            }
            // Around the camera, the far terrain is cube nodes.
            assert!(
                plan.leaves.iter().any(|k| k.distance(camera) < 64.0),
                "{camera:?}"
            );
        }
    }

    #[test]
    fn a_complex_tile_turns_into_cube_nodes_sooner() {
        let camera = [0.5, 100.5, 0.5];
        let first = tiled(camera, &|_| None);
        let pick = |level: u8| {
            *first
                .tiles
                .iter()
                .find(|t| t.level == level)
                .unwrap_or_else(|| panic!("no tile of level {level}"))
        };
        let lifted = pick(SURFACE_LEVEL + 1);
        let kept = pick(SURFACE_LEVEL + COMPLEX_LIFT);
        // The same heights the plan assumes before a tile is generated.
        let complex = TileInfo {
            overflow_columns: 7,
            highest: 63 + UNKNOWN_ABOVE - 1,
            skirt: 63 - UNKNOWN_BELOW,
            ..TileInfo::default()
        };
        let info = move |t: TileKey| (t == lifted || t == kept).then_some(complex);
        let second = tiled(camera, &info);
        assert!(
            !second.tiles.contains(&lifted),
            "a complex tile below the lift stays"
        );
        let inside = |c: &CubeKey| {
            let [x, _, z] = c.origin().map(i64::from);
            tile_holds(lifted, x, z) && c.size() <= lifted.size()
        };
        assert!(second.leaves.iter().any(inside), "no node under {lifted:?}");
        // From the lift's level up, a complex tile is still a tile.
        assert!(second.tiles.contains(&kept));
    }

    #[test]
    fn a_tile_fades_out_only_once_the_nodes_under_it_are_in() {
        let mut lod = lod_for_test();
        lod.tiles.on = true;
        let tile = TileKey::new(SURFACE_LEVEL, 3, -2);
        put_far(&mut lod, &[], &[tile], &[]);
        assert_eq!(lod.take_tile_work().request, vec![tile]);
        lod.surface_feedback(|k| (k == tile).then_some(TileInfo::default()));
        assert!(lod.stats().complete);
        // The plan turns the tile into the cube nodes of its footprint.
        let node = CubeKey::new(SURFACE_LEVEL, tile.x, 0, tile.z);
        let kids = node.children();
        put_far(&mut lod, &kids, &[], &[]);
        assert!(lod.removed().is_empty());
        assert!(
            lod.take_tile_work().retire.is_empty(),
            "nothing under it is in"
        );
        for &k in &kids[..7] {
            lod.deliver(k, false);
        }
        assert!(lod.removed().is_empty());
        assert!(lod.take_tile_work().retire.is_empty(), "one node missing");
        lod.deliver(kids[7], false);
        assert!(lod.removed().is_empty());
        assert_eq!(lod.take_tile_work().retire, vec![tile]);
        // And back: the nodes stay until the tile is in again.
        put_far(&mut lod, &[], &[tile], &[]);
        assert_eq!(lod.take_tile_work().request, vec![tile]);
        assert!(lod.removed().is_empty(), "the tile is not in");
        lod.surface_feedback(|k| (k == tile).then_some(TileInfo::default()));
        assert_eq!(lod.removed().len(), 8);
    }

    // ---- coverage first, and the queue's order

    #[test]
    fn a_big_change_draws_the_coverage_layer_first() {
        let mut lod = lod_for_test();
        let camera = [5000.5, 120.5, -3000.5];
        lod.camera = camera;
        let job = PlanJob {
            epoch: lod.epoch,
            camera,
            max_distance: 50_000.0,
            rule: FarView::standard(8.0).rule(usize::MAX),
            loaded: NONE,
            snapshot: LoadedSnapshot::default(),
            previous: Arc::default(),
            drawn: Drawn::default(),
            step: false,
            surface: None,
        };
        let done = make_plan(
            &job,
            &mut FxHashSet::default(),
            &mut FxHashSet::default(),
            &mixed,
        );
        let plan: Vec<CubeKey> = done.leaves.to_vec();
        lod.install(done, Instant::now());
        assert!(matches!(lod.phase, Phase::Cover { .. }));
        assert!(lod.stats().plan_covered.is_none());
        // Only the layer is queued: few nodes, one over each planned node.
        let wave = lod.shared.queue.lock().unwrap().drain();
        assert!(wave.len() <= COVER_NODES && wave.len() * 10 < plan.len());
        for &leaf in &plan {
            let over = wave.iter().filter(|w| w.contains(leaf)).count();
            assert_eq!(over, 1, "{leaf:?}");
        }
        // Nearest first within the layer's direction around the camera.
        assert!(wave[0].distance(camera) < 1.0);
        // The layer comes in: the place is covered, and the rest is queued.
        for &w in &wave {
            lod.deliver(w, false);
        }
        let now = Instant::now();
        lod.recheck_holes(now, false);
        lod.advance_phase(now);
        lod.track_completion(now);
        assert!(matches!(lod.phase, Phase::Refine));
        assert!(lod.stats().plan_covered.is_some());
        let rest = lod.shared.queue.lock().unwrap().drain();
        let delivered_plan = plan.iter().filter(|k| wave.contains(k)).count();
        assert_eq!(rest.len(), plan.len() - delivered_plan);
        // The coverage nodes stay until the nodes under them are in.
        if let Some(earlier) = Instant::now().checked_sub(Duration::from_secs(1)) {
            lod.released_at = earlier;
        }
        assert!(lod.removed().is_empty());
        for &k in &rest {
            lod.deliver(k, false);
        }
        let gone: FxHashSet<CubeKey> = lod.removed().into_iter().collect();
        let layer: FxHashSet<CubeKey> =
            wave.iter().copied().filter(|k| !plan.contains(k)).collect();
        assert_eq!(gone, layer);
        assert!(lod.stats().complete || lod.gpu_path());
    }

    #[test]
    fn a_small_change_skips_the_coverage_layer() {
        let mut lod = lod_for_test();
        let coarse = key(6, 0, 0, 0);
        put(&mut lod, &[coarse], &[]);
        lod.deliver(coarse, false);
        lod.advance_phase(Instant::now());
        lod.shared.queue.lock().unwrap().clear();
        // Its children replace it: the parent still covers them.
        put(&mut lod, &coarse.children(), &[coarse]);
        assert!(matches!(lod.phase, Phase::Refine));
        assert!(lod.holes.is_empty());
        let queued = lod.shared.queue.lock().unwrap().drain();
        assert_eq!(queued.len(), 8);
    }

    /// The look along `forward` of the standard view.
    fn looking(forward: [f64; 3]) -> Look {
        Look {
            forward,
            edge: FarView::standard(1.0).half_diagonal(),
        }
    }

    #[test]
    fn the_crosshair_comes_first_then_half_as_much_the_near() {
        let edge = FarView::standard(1.0).half_diagonal();
        let deg = |d: f64| d.to_radians();
        // The weights as given: crosshair 1, nearness a half.
        assert!(
            (far_priority(0.0, edge, 0.0) - (FAR_PRIORITY_VIEW + FAR_PRIORITY_NEAR)).abs() < 1e-9
        );
        // A far spot under the crosshair beats near land off to the side:
        // nearest first is what kept the view from sharpening.
        assert!(far_priority(0.0, edge, 3000.0) > far_priority(deg(60.0), edge, 300.0));
        assert!(far_priority(0.0, edge, 3000.0) > far_priority(deg(35.0), edge, 300.0));
        // Of two at the crosshair, the nearer; of two as far, the one nearer
        // the crosshair; the weight falls smoothly to the screen's edge.
        assert!(far_priority(0.0, edge, 300.0) > far_priority(0.0, edge, 3000.0));
        let mut last = f64::INFINITY;
        for d in (0..=180).step_by(5) {
            let w = crosshair_weight(deg(f64::from(d)), edge);
            assert!(w <= last, "{d}");
            last = w;
        }
        assert!((crosshair_weight(edge, edge) - FAR_PRIORITY_EDGE).abs() < 1e-9);
        assert!((crosshair_weight(std::f64::consts::PI, edge) - FAR_PRIORITY_BEHIND).abs() < 1e-9);
        // Behind the player is last, unless it is near him.
        assert!(far_priority(deg(180.0), edge, 3000.0) < far_priority(edge, edge, 3000.0));
        assert!(far_priority(deg(180.0), edge, 0.0) > far_priority(edge, edge, 3000.0));
        // Zoomed in (a spyglass), what is 10 degrees off counts for less.
        let zoomed = FarView::standard(1.0)
            .looking([0.0, 0.0, 1.0], deg(10.0))
            .half_diagonal();
        assert!(crosshair_weight(deg(10.0), zoomed) < crosshair_weight(deg(10.0), edge));
    }

    #[test]
    fn the_queue_takes_the_crosshair_first_and_resorts_on_a_turn() {
        let mut lod = lod_for_test();
        let camera = [0.5, 70.5, 0.5];
        lod.camera = camera;
        // Level-4 nodes (256 blocks) in a ring 2,000 blocks out, every 30
        // degrees, the node round the camera, and one 300 blocks off to the
        // side (+X).
        let ring: Vec<CubeKey> = (0..12)
            .map(|i| {
                let a = f64::from(i) * 30f64.to_radians();
                CubeKey::containing(4, cube_of([2000.0 * a.sin(), 70.0, 2000.0 * a.cos()]))
            })
            .collect();
        let near = key(4, 0, 0, 0);
        let side = CubeKey::containing(2, cube_of([300.0, 70.0, 0.0]));
        let mut all = ring.clone();
        all.extend([near, side]);
        lod.order = all.iter().map(|&k| (bucket(camera, k), k)).collect();
        let first_of = |lod: &mut CubeLod, forward: [f64; 3]| {
            lod.face(looking(forward));
            lod.fill_queue();
            lod.shared.queue.lock().unwrap().drain()
        };
        let at = |order: &[CubeKey], k: CubeKey| order.iter().position(|&o| o == k).unwrap();
        // Looking along +Z: round the camera, then the crosshair's, the one
        // behind last; the near one off to the side after the far ones in view.
        let order = first_of(&mut lod, [0.0, 0.0, 1.0]);
        assert_eq!(order.len(), all.len());
        assert_eq!(order[..2], [near, ring[0]], "{order:?}");
        assert_eq!(*order.last().unwrap(), ring[6]);
        assert!(at(&order, side) > at(&order, ring[1]), "{order:?}");
        assert!(at(&order, side) > at(&order, ring[11]), "{order:?}");
        // Turned around: behind comes first. A small turn changes nothing.
        let turned = first_of(&mut lod, [0.0, 0.0, -1.0]);
        assert_eq!(turned[1], ring[6]);
        assert_eq!(*turned.last().unwrap(), ring[0]);
        let nudged = first_of(&mut lod, [0.03, 0.0, -1.0]);
        assert_eq!(nudged, turned);
        // Turned to the side: the near node there comes right after.
        let aside = first_of(&mut lod, [1.0, 0.0, 0.0]);
        assert_eq!(aside[1], side, "{aside:?}");
        // A hole comes before the score, whichever way the camera looks ...
        lod.holes = vec![(ring[6], None)];
        let holed = first_of(&mut lod, [0.0, 0.0, 1.0]);
        assert_eq!(holed[0], ring[6]);
        assert_eq!(holed.iter().filter(|&&k| k == ring[6]).count(), 1);
        // ... and a node replacing a far node drawn over near ground before
        // that.
        lod.lingering.insert(ancestor(ring[3], 5), Instant::now());
        let urgent = first_of(&mut lod, [0.0, 0.0, 1.0]);
        assert_eq!(urgent[..2], [ring[3], ring[6]], "{urgent:?}");
    }

    #[test]
    fn of_equal_scores_the_bigger_on_screen_error_goes_first() {
        let mut lod = lod_for_test();
        let camera = [0.5, 70.5, 0.5];
        lod.camera = camera;
        lod.face(looking([0.0, 0.0, 1.0]));
        let big = CubeKey::containing(5, cube_of([0.0, 70.0, 2100.0]));
        let small = CubeKey::containing(3, cube_of([130.0, 192.0, 2100.0]));
        assert_eq!(bucket(camera, big), bucket(camera, small));
        lod.queue_jobs(vec![small, big], Vec::new());
        assert_eq!(lod.shared.queue.lock().unwrap().drain(), [big, small]);
    }

    #[test]
    fn a_turn_reorders_the_buckets_taken_from() {
        let mut queue = Queue::default();
        let camera = [0.5, 0.5, 0.5];
        let east = key(3, 20, 0, 0);
        let west = key(3, -21, 0, 0);
        for k in [east, west] {
            queue.push(bucket(camera, k), k);
        }
        queue.reorder(bucket_order(looking([1.0, 0.0, 0.0])));
        assert_eq!(queue.pop(), Some(east));
        queue.push(bucket(camera, east), east);
        queue.reorder(bucket_order(looking([-1.0, 0.0, 0.0])));
        assert_eq!(queue.pop(), Some(west));
        // A key the plan dropped is skipped.
        queue.dropped.insert(east);
        assert_eq!(queue.pop(), None);
        // A key pushed into a bucket taken earlier than the cursor comes next.
        queue.push(bucket(camera, east), east);
        queue.push(bucket(camera, west), west);
        assert_eq!(queue.pop(), Some(west));
    }

    #[test]
    fn a_stepped_node_splits_only_once_the_node_it_replaced_is_gone() {
        let mut lod = lod_for_test();
        let parent = key(5, 3, 0, 3);
        put(&mut lod, &[parent], &[]);
        lod.deliver(parent, false);
        let children: Vec<CubeKey> = parent.children().to_vec();
        put(&mut lod, &children, &[]);
        for &c in &children[..7] {
            lod.deliver(c, false);
        }
        // The parent stands (stale) until all eight are in: its children
        // don't count as drawn, so none splits and holds it up again.
        let drawn = lod.drawn_for_steps();
        assert!(drawn.contains(&parent));
        assert!(children.iter().all(|c| !drawn.contains(c)), "{drawn:?}");
        lod.deliver(children[7], false);
        assert_eq!(lod.removed(), vec![parent]);
        let drawn = lod.drawn_for_steps();
        assert!(children.iter().all(|c| drawn.contains(c)), "{drawn:?}");
    }

    /// Whether `node` is drawn over: it, a node holding it, or nodes inside
    /// it filling it, are in `cover`.
    fn filled(node: CubeKey, cover: &FxHashSet<CubeKey>) -> bool {
        let mut up = node;
        loop {
            if cover.contains(&up) {
                return true;
            }
            if up.level >= MAX_CUBE_LEVEL {
                break;
            }
            up = up.parent();
        }
        let inside: f64 = (cover.iter())
            .filter(|c| c.level < node.level && ancestor(**c, node.level) == node)
            .map(|c| 8f64.powi(i32::from(c.level)))
            .sum();
        (inside - 8f64.powi(i32::from(node.level))).abs() < 0.5
    }

    #[test]
    fn refining_and_merging_draw_one_node_on_every_path_with_no_gap() {
        use crate::lod::swap::Swap;
        let nowhere = |_: SectionPos| false;
        let mut lod = lod_for_test();
        let mut swap = Swap::default();
        let mut queued = FxHashSet::default();
        let mut shown: FxHashSet<CubeKey> = FxHashSet::default();
        let mut mix = 0x9e37_79b9_7f4a_7c15u64;
        let mut hash = |k: CubeKey| {
            mix = mix.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let v =
                (k.x as u64) ^ (k.y as u64) << 21 ^ (k.z as u64) << 42 ^ u64::from(k.level) << 60;
            (v ^ mix >> 17).wrapping_mul(0xff51_afd7_ed55_8ccd)
        };
        // Refine at one place, then fly away so it merges back.
        for camera in [[100.5, 64.5, -40.5], [2100.5, 64.5, -40.5]] {
            lod.camera = camera;
            let mut rounds = 0;
            loop {
                let drawn = lod.drawn_for_steps();
                let plan = stepped(camera, Some(&drawn), &nowhere);
                put(&mut lod, &plan, &[]);
                for &k in &plan {
                    if !lod.delivered.contains(&k) && queued.insert(k) {
                        swap.queued(k);
                    }
                }
                // About half of what waits comes in, in a scrambled order; a
                // fifth of it empty.
                let mut waiting: Vec<(u64, CubeKey)> = (plan.iter().copied())
                    .filter(|k| !lod.delivered.contains(k))
                    .map(|k| (hash(k), k))
                    .collect();
                waiting.sort_unstable();
                let take = waiting.len().div_ceil(2);
                for (h, k) in waiting.into_iter().take(take) {
                    let empty = h % 5 == 0;
                    lod.deliver(k, empty);
                    queued.remove(&k);
                    if empty {
                        swap.unwait(k);
                        swap.forget(k);
                    } else {
                        swap.installed(k, false);
                    }
                }
                for k in lod.removed() {
                    swap.retire(k);
                }
                let step = swap.step();
                let mut cover: FxHashSet<CubeKey> = shown.clone();
                for k in step.fade.iter().chain(&step.drop) {
                    cover.remove(k);
                }
                cover.extend(step.show.iter().chain(&step.fade_in).copied());
                // Empty nodes handed out draw nothing and need nothing.
                cover.extend(lod.empty.iter().copied());
                // A node fades only with every planned node over its place
                // drawn over (or empty); what the plan left out of range
                // (far behind) needs nothing.
                let planned: FxHashSet<CubeKey> = plan.iter().copied().collect();
                for k in (step.fade.iter().chain(&step.drop)).filter(|k| shown.contains(k)) {
                    let over: Vec<CubeKey> = (planned.iter().copied())
                        .filter(|p| {
                            (p.level <= k.level && ancestor(*p, k.level) == *k)
                                || (p.level > k.level && ancestor(*k, p.level) == *p)
                        })
                        .collect();
                    for p in over {
                        assert!(
                            filled(p, &cover),
                            "{k:?} faded with {p:?} not drawn (camera {camera:?}, round {rounds})"
                        );
                    }
                }
                for k in step.fade.iter().chain(&step.drop) {
                    shown.remove(k);
                }
                shown.extend(step.show);
                shown.extend(step.fade_in);
                // One drawn node on every root-to-leaf path.
                for &k in &shown {
                    let mut up = k;
                    while up.level < MAX_CUBE_LEVEL {
                        up = up.parent();
                        assert!(!shown.contains(&up), "{k:?} drawn with {up:?}");
                    }
                }
                let done = plan.iter().all(|k| lod.delivered.contains(k)) && swap.is_idle();
                if done && stepped(camera, Some(&plan), &nowhere) == plan {
                    break;
                }
                rounds += 1;
                assert!(rounds < 400);
            }
            assert!(rounds >= 3, "{rounds}");
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    fn voxel(state: u32, biome: u16) -> Voxel {
        Voxel {
            state: BlockStateId(state),
            biome,
        }
    }

    /// A dense node of `states` kinds of voxel (a stripe pattern), with a
    /// cover mask and fluid tops when asked.
    fn dense(key: CubeKey, states: u32, extras: bool) -> CubeNode {
        let node = CubeNode::from_fn(key, |x, y, z| {
            let i = CubeNode::index(x, y, z) as u32;
            voxel(i % states, (i % 2) as u16)
        });
        if !extras {
            return node;
        }
        let mut cover = Box::new([0u64; 4]);
        cover[1] = 0xdead_beef;
        let mut tops = Box::new([NO_FLUID_TOP; 256]);
        tops[7] = 100;
        tops[200] = FLUID_TOP_SCALE;
        node.with_covered(Some(cover)).with_fluid_tops(Some(tops))
    }

    fn k(x: i32) -> CubeKey {
        CubeKey::new(2, x, 0, 0)
    }

    fn same(a: &CubeNode, b: &CubeNode) {
        assert_eq!(a.key, b.key);
        for i in 0..VOLUME {
            assert_eq!(a.at(i), b.at(i), "voxel {i}");
        }
        assert_eq!(a.covered_mask(), b.covered_mask());
        assert_eq!(a.fluid_tops(), b.fluid_tops());
        assert_eq!(a.as_uniform(), b.as_uniform());
    }

    #[test]
    fn a_packed_node_comes_back_voxel_for_voxel() {
        for extras in [false, true] {
            for states in [2, 5, 256] {
                let node = Arc::new(dense(k(3), states, extras));
                let entry = Entry::pack(node.clone(), true);
                let Entry::Packed(p) = &entry else {
                    panic!("{states} states should pack");
                };
                let distinct: FxHashSet<Voxel> = (0..VOLUME).map(|i| node.at(i)).collect();
                assert_eq!(p.palette.len(), distinct.len());
                same(&node, &entry.unpack());
            }
        }
    }

    #[test]
    fn nodes_of_more_than_256_voxels_or_one_stay_whole() {
        let many = Arc::new(dense(k(1), 300, true));
        assert!(matches!(
            Entry::pack(many.clone(), true),
            Entry::Node(n) if Arc::ptr_eq(&n, &many)
        ));
        let uniform = Arc::new(CubeNode::uniform(k(2), voxel(1, 0)));
        assert!(matches!(Entry::pack(uniform.clone(), true), Entry::Node(_)));
        // Unpacked on request, dense or not.
        let few = Arc::new(dense(k(3), 4, false));
        assert!(matches!(Entry::pack(few, false), Entry::Node(_)));
        // The estimate: a uniform node costs next to nothing, a dense one 32 KiB, a packed one about 4 KiB.
        let base = (std::mem::size_of::<CubeNode>() + 64) as u64;
        assert_eq!(Entry::pack(uniform, true).bytes(), base);
        assert_eq!(
            Entry::pack(many, true).bytes(),
            base + (VOLUME * std::mem::size_of::<Voxel>()) as u64 + 32 + 512
        );
        let packed = Entry::pack(Arc::new(dense(k(4), 4, true)), true).bytes();
        assert_eq!(
            packed,
            base + (VOLUME + 4 * std::mem::size_of::<Voxel>()) as u64 + 32 + 512
        );
        println!("bytes per entry: packed dense with extras {packed}, uniform {base}");
    }

    #[test]
    fn the_least_recently_used_node_goes_first_and_the_budget_holds() {
        let one = Entry::pack(Arc::new(dense(k(0), 3, false)), true).bytes();
        let budget = one * 3 + one / 2;
        let mut cache = Cache::with_budget(budget, true, usize::MAX);
        for x in 0..3 {
            cache.insert(Arc::new(dense(k(x), 3, false)));
        }
        assert_eq!((cache.len(), cache.bytes()), (3, one * 3));
        // Using 0 makes 1 the least recently used.
        same(&cache.get(k(0)).unwrap(), &dense(k(0), 3, false));
        cache.insert(Arc::new(dense(k(3), 3, false)));
        assert!(cache.nodes.contains_key(&k(0)));
        assert!(!cache.nodes.contains_key(&k(1)));
        assert!(cache.nodes.contains_key(&k(2)) && cache.nodes.contains_key(&k(3)));
        assert!(cache.bytes() <= budget);
        assert_eq!((cache.hits, cache.misses, cache.evicted), (1, 0, 1));
        // Evicted, then fetched again: a miss.
        assert!(cache.get(k(1)).is_none());
        assert_eq!((cache.hits, cache.misses), (1, 1));
        // Many more: the bytes always hold, and what is kept is the newest.
        for x in 4..100 {
            cache.insert(Arc::new(dense(k(x), 3, false)));
            assert!(cache.bytes() <= budget);
        }
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.evicted, 1 + 96);
        assert!((97..100).all(|x| cache.nodes.contains_key(&k(x))));
        // Keeping a node again replaces it without counting its bytes twice.
        cache.insert(Arc::new(dense(k(99), 3, false)));
        assert_eq!(cache.bytes(), one * 3);
        // remove and clear count nothing.
        let before = (
            cache.hits,
            cache.misses,
            cache.evicted,
            cache.forgotten,
            cache.cleared,
        );
        cache.remove(k(99));
        assert_eq!((cache.len(), cache.bytes()), (2, one * 2));
        cache.clear();
        assert_eq!((cache.len(), cache.bytes()), (0, 0));
        assert_eq!(
            before,
            (
                cache.hits,
                cache.misses,
                cache.evicted,
                cache.forgotten,
                cache.cleared
            )
        );
    }

    #[test]
    fn the_node_just_kept_stays_even_when_it_alone_is_over_the_budget() {
        let mut cache = Cache::with_budget(1, false, usize::MAX);
        cache.insert(Arc::new(dense(k(0), 3, false)));
        assert_eq!(cache.len(), 1);
        cache.insert(Arc::new(dense(k(1), 3, false)));
        assert_eq!(cache.len(), 1);
        assert!(cache.nodes.contains_key(&k(1)));
    }

    #[test]
    fn the_legacy_cache_keeps_at_most_its_count_unpacked() {
        let mut cache = Cache::with_budget(u64::MAX, false, CACHE_NODES);
        for x in 0..(CACHE_NODES as i32 + 100) {
            cache.insert(Arc::new(dense(k(x), 3, false)));
        }
        assert_eq!(cache.len(), CACHE_NODES);
        assert_eq!(cache.evicted, 100);
        assert!(
            cache
                .nodes
                .values()
                .all(|s| matches!(s.entry, Entry::Node(_)))
        );
        assert!(cache.nodes.contains_key(&k(CACHE_NODES as i32 + 99)));
        assert!(!cache.nodes.contains_key(&k(99)) && cache.nodes.contains_key(&k(100)));
        // A small count too.
        let mut small = Cache::with_budget(u64::MAX, false, 4);
        for x in 0..10 {
            small.insert(Arc::new(CubeNode::uniform(k(x), voxel(1, 0))));
        }
        assert_eq!(small.len(), 4);
    }

    #[test]
    fn the_log_stays_small_however_often_nodes_are_used() {
        let mut cache = Cache::with_budget(u64::MAX, false, usize::MAX);
        for x in 0..10 {
            cache.insert(Arc::new(CubeNode::uniform(k(x), voxel(1, 0))));
        }
        for i in 0..100_000 {
            assert!(cache.get(k(i % 10)).is_some());
            assert!(cache.log.len() <= cache.len() * 4 + 1024 + 1);
        }
        // After all that, the order is still right: 5 is the oldest used.
        let mut cache = Cache::with_budget(u64::MAX, false, 10);
        for x in 0..10 {
            cache.insert(Arc::new(CubeNode::uniform(k(x), voxel(1, 0))));
        }
        for _ in 0..5000 {
            for x in (0..10).filter(|&x| x != 5) {
                cache.get(k(x));
            }
        }
        cache.insert(Arc::new(CubeNode::uniform(k(10), voxel(1, 0))));
        assert!(!cache.nodes.contains_key(&k(5)));
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn forgetting_drops_exactly_the_nodes_holding_the_changed_cubes() {
        let mut cache = Cache::with_budget(u64::MAX, false, usize::MAX);
        let cube = SectionPos::new(-3, 5, 9);
        let mut held = Vec::new();
        let mut kept = Vec::new();
        for level in 0..=MAX_CUBE_LEVEL {
            held.push(CubeKey::containing(level, cube));
            // Its neighbour at the same level that does not hold the cube.
            let mut other = CubeKey::containing(level, cube);
            other.x += 1;
            kept.push(other);
        }
        for key in held.iter().chain(&kept) {
            cache.insert(Arc::new(CubeNode::uniform(*key, voxel(1, 0))));
        }
        assert_eq!(cache.len(), 2 * 23);
        // Two cubes under the same nodes count once.
        cache.forget(&[cube, cube]);
        assert_eq!(cache.forgotten, 23);
        assert_eq!(cache.len(), 23);
        assert!(held.iter().all(|key| !cache.nodes.contains_key(key)));
        assert!(kept.iter().all(|key| cache.nodes.contains_key(key)));
        assert_eq!(
            cache.bytes(),
            23 * Entry::pack(Arc::new(CubeNode::uniform(k(0), voxel(1, 0))), true).bytes()
        );
        // Forgotten nodes' log items are skipped: filling it up evicts only live ones.
        assert!(cache.get(held[0]).is_none());
    }

    #[test]
    fn forgetting_by_level_matches_the_old_scan_on_random_keys() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |n: i32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as i32 - n / 2
        };
        for _round in 0..20 {
            let mut cache = Cache::with_budget(u64::MAX, false, usize::MAX);
            for _ in 0..600 {
                let level = next(1 << 20).rem_euclid(9) as u8;
                let key = if next(10) < 3 {
                    CubeKey::new(level, next(8), next(8), next(8))
                } else {
                    let cube = SectionPos::new(next(40), next(40), next(40));
                    CubeKey::containing(level, cube)
                };
                cache.insert(Arc::new(CubeNode::uniform(key, voxel(1, 0))));
            }
            let cubes: Vec<SectionPos> = (0..6)
                .map(|_| SectionPos::new(next(40), next(40), next(40)))
                .collect();
            let mut old: FxHashMap<CubeKey, ()> = cache.nodes.keys().map(|&k| (k, ())).collect();
            old.retain(|key, _| !cubes.iter().any(|&c| key.contains_cube(c)));
            let before = cache.len();
            cache.forget(&cubes);
            let mut now: Vec<CubeKey> = cache.nodes.keys().copied().collect();
            let mut expected: Vec<CubeKey> = old.keys().copied().collect();
            now.sort_by_key(|k| (k.level, k.x, k.y, k.z));
            expected.sort_by_key(|k| (k.level, k.x, k.y, k.z));
            assert_eq!(now, expected);
            assert_eq!(cache.forgotten as usize, before - cache.len());
        }
    }

    #[test]
    fn a_default_budget_holds_forty_thousand_packed_nodes() {
        let mut cache = Cache::with_budget(192 << 20, true, usize::MAX);
        let template = dense(k(0), 4, true);
        for x in 0..41_000 {
            let mut node = template.clone();
            node.key = k(x);
            cache.insert(Arc::new(node));
        }
        assert_eq!(cache.len(), 41_000);
        assert_eq!(cache.evicted, 0);
        assert!(cache.bytes() <= 192 << 20);
        let per_entry = cache.bytes() / cache.len() as u64;
        println!(
            "packed dense node estimate: {per_entry} bytes; 192 MiB / that = {}",
            (192u64 << 20) / per_entry
        );
        assert!(per_entry <= 5 * 1024, "{per_entry}");
        let back = cache.get(k(40_999)).unwrap();
        let mut want = template.clone();
        want.key = k(40_999);
        same(&want, &back);
    }
}

#[cfg(test)]
mod whole_world_tests {
    use super::*;

    const NONE: LoadedBox = LoadedBox {
        horizontal: 0,
        vertical: 0,
    };
    const WORLD: f64 = 67_108_864.0;

    fn disjoint(plan: &[CubeKey]) {
        let set: FxHashSet<CubeKey> = plan.iter().copied().collect();
        for &leaf in plan {
            let mut up = leaf;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                assert!(!set.contains(&up), "{leaf:?} lies inside {up:?}");
            }
        }
    }

    #[test]
    fn keys_stay_exact_at_the_top_level() {
        let top = CubeKey::new(MAX_CUBE_LEVEL, -1, -1, 0);
        assert_eq!(top.size(), 1 << 26);
        assert_eq!(top.origin(), [-(1 << 26), -(1 << 26), 0]);
        assert_eq!(top.children()[7].level, MAX_CUBE_LEVEL - 1);
        assert!(top.contains(top.children()[3]));
        assert!(in_world(CubeKey::new(MAX_CUBE_LEVEL, 0, 0, 0)));
        // The border: the last node inside it, and one wholly past it.
        let edge = 30_000_000 / 16;
        assert!(in_world(CubeKey::new(0, edge - 1, 0, 0)));
        assert!(!in_world(CubeKey::new(0, edge, 0, 0)));
        assert!(!in_world(CubeKey::new(0, 0, 1 << 19, 0)));
        assert!(in_world(CubeKey::new(0, 0, -(1 << 19), 0)));
    }

    #[test]
    fn the_whole_world_plan_is_bounded_disjoint_and_inside_the_world() {
        let Plan {
            leaves: plan,
            pruned,
            ..
        } = plan_classified(
            [0.0, 100.0, 0.0],
            WORLD,
            &Rule::distance(DETAIL),
            NONE,
            &|_| false,
            &[],
            &|_| NodeClass::Mixed,
        );
        assert_eq!(pruned, 0);
        assert!(!plan.is_empty() && plan.len() < 100_000, "{}", plan.len());
        assert!(plan.iter().all(|k| in_world(*k)));
        assert!(plan.iter().any(|k| k.level >= 18), "coarse nodes far out");
        assert!(plan.iter().any(|k| k.level == 0), "fine ones near");
        disjoint(&plan);
    }

    #[test]
    fn nodes_known_uniform_are_left_out_or_stay_whole() {
        let camera = [0.0, 100.0, 0.0];
        let all = |_: SectionPos| false;
        let rule = Rule::distance(DETAIL);
        let mixed =
            plan_classified(camera, WORLD, &rule, NONE, &all, &[], &|_| NodeClass::Mixed).leaves;
        // Everything above y = 1000 is air, everything below y = -1000 rock.
        let class = |k: CubeKey| {
            let [_, y, _] = k.origin();
            if i64::from(y) > 1000 {
                NodeClass::Air
            } else if i64::from(y) + i64::from(k.size()) < -1000 {
                NodeClass::Rock
            } else {
                NodeClass::Mixed
            }
        };
        let Plan {
            leaves: plan,
            pruned,
            ..
        } = plan_classified(camera, WORLD, &rule, NONE, &all, &[], &class);
        assert!(pruned > 0);
        assert!(
            plan.len() < mixed.len() / 2,
            "{} of {}",
            plan.len(),
            mixed.len()
        );
        assert!(plan.iter().all(|k| class(*k) != NodeClass::Air));
        // A rock node is a leaf however near the camera: none of its parts is planned.
        for k in plan.iter().filter(|k| class(**k) == NodeClass::Rock) {
            assert!(!plan.iter().any(|o| o != k && k.contains(*o)));
        }
        disjoint(&plan);
    }

    #[test]
    fn a_node_with_loaded_cubes_still_splits_around_them() {
        let loaded = LoadedBox {
            horizontal: 2,
            vertical: 2,
        };
        let plan = plan_classified(
            [8.0, 8.0, 8.0],
            4096.0,
            &Rule::distance(DETAIL),
            loaded,
            &|_| true,
            &[],
            &|_| NodeClass::Rock,
        )
        .leaves;
        // Every cube around the camera is drawn by the near renderer: none planned inside.
        for k in &plan {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    for dx in -2..=2 {
                        assert!(!k.contains_cube(SectionPos::new(dx, dy, dz)), "{k:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn far_nodes_beside_loaded_cubes_are_single_cubes() {
        // A coarse rule: left to it, the nodes beside the loaded cubes would be whole voxels.
        let rule = Rule::distance(0.25);
        let loaded = LoadedBox {
            horizontal: 8,
            vertical: 3,
        };
        let is_loaded = |p: SectionPos| p.x.abs() <= 6 && p.z.abs() <= 6 && p.y.abs() <= 2;
        let plan = plan_classified(
            [8.0, 8.0, 8.0],
            4096.0,
            &rule,
            loaded,
            &is_loaded,
            &[],
            &|_| NodeClass::Mixed,
        )
        .leaves;
        let set: FxHashSet<CubeKey> = plan.iter().copied().collect();
        let mut beside = 0;
        for y in -2..=2 {
            for z in -7..=7 {
                for x in -7..=7 {
                    let cube = SectionPos::new(x, y, z);
                    let next = |dx: i32, dz: i32| is_loaded(SectionPos::new(x + dx, y, z + dz));
                    if is_loaded(cube) || !(next(1, 0) || next(-1, 0) || next(0, 1) || next(0, -1))
                    {
                        continue;
                    }
                    beside += 1;
                    assert!(
                        set.contains(&CubeKey::containing(0, cube)),
                        "{cube:?} beside the loaded cubes is not a single cube"
                    );
                }
            }
        }
        assert_eq!(beside, 4 * 13 * 5);
        // Past that ring the rule decides again: coarser nodes stand.
        assert!(plan.iter().any(|k| k.level >= 2));
        disjoint(&plan);
    }
}

/// The far view's waste counters, and the late-refresh fixes that stop work
/// being made for nothing (`HORIZON_FAR_LATE_EXACT`, `_SKIP_HELD`, `_DROP`,
/// on by default, and `HORIZON_FAR_WHOLE_WHEN_COVERED`, off by default).
#[cfg(test)]
mod waste_tests {
    use super::*;
    use std::sync::Mutex;

    const AIR: BlockStateId = BlockStateId(0);
    type Boxes = Vec<([i32; 3], [i32; 3])>;

    fn key(level: u8, x: i32, y: i32, z: i32) -> CubeKey {
        CubeKey::new(level, x, y, z)
    }

    fn lod_for_test() -> CubeLod {
        let shared = Arc::new(Shared::new());
        let (_tx, results) = mpsc::channel();
        CubeLod::assemble(shared, results, Vec::new(), None, None, AIR)
    }

    /// Puts `keys` in as the plan, as the planner thread would hand it over,
    /// with `held` as the leaves it held back.
    fn put_held(lod: &mut CubeLod, keys: &[CubeKey], held: &[CubeKey]) {
        let next: FxHashSet<CubeKey> = keys.iter().copied().collect();
        let added = keys
            .iter()
            .copied()
            .filter(|k| !lod.wanted.contains(k))
            .collect();
        let removed = (lod.wanted.iter().copied())
            .filter(|k| !next.contains(k))
            .collect();
        let camera = lod.camera;
        let mut leaves = keys.to_vec();
        leaves.sort_by(|a, b| a.distance(camera).total_cmp(&b.distance(camera)));
        let mut tree: Vec<(u128, CubeKey)> = keys.iter().map(|&k| (tree_order(k), k)).collect();
        tree.sort_unstable();
        let mut footprints: Vec<(u64, u8, FarKey)> = keys
            .iter()
            .map(|&k| (footprint_order(k.level, k.x, k.z), k.level, FarKey::Cube(k)))
            .collect();
        footprints.sort_unstable();
        let done = PlanDone {
            epoch: lod.epoch,
            tree,
            order: leaves.iter().map(|&k| (bucket(camera, k), k)).collect(),
            leaves: Arc::new(leaves),
            changed: true,
            added,
            removed,
            cover: FxHashSet::default(),
            tiles: Arc::new(Vec::new()),
            tiles_added: Vec::new(),
            tiles_removed: Vec::new(),
            cover_tiles: FxHashSet::default(),
            footprints,
            pruned: 0,
            limited: false,
            coarsened: 1.0,
            tiles_limited: false,
            held: held.len(),
            held_keys: held.to_vec(),
            elapsed: Duration::ZERO,
        };
        lod.install(done, Instant::now());
    }

    fn put(lod: &mut CubeLod, keys: &[CubeKey]) {
        put_held(lod, keys, &[]);
    }

    /// A late content that hands out `take` and records what it is told.
    #[derive(Default)]
    struct Late {
        exact: bool,
        take: Mutex<Boxes>,
        not_drawn: Mutex<Boxes>,
        not_now: Mutex<Boxes>,
        coarse: Mutex<Vec<(Vec<CubeKey>, bool)>>,
    }

    impl LateContent for Late {
        fn view(&self, _: [f64; 3], _: [f64; 3], _: f64, _: bool) {}
        fn take_refreshes(&self) -> Boxes {
            std::mem::take(&mut *self.take.lock().unwrap())
        }
        fn not_drawn(&self, boxes: &[([i32; 3], [i32; 3])]) {
            self.not_drawn.lock().unwrap().extend_from_slice(boxes);
        }
        fn exact(&self) -> bool {
            self.exact
        }
        fn wants_coarse_first(&self) -> bool {
            true
        }
        fn coarse_first(&self, keys: &[CubeKey], all: bool) {
            let mut keys = keys.to_vec();
            keys.sort();
            self.coarse.lock().unwrap().push((keys, all));
        }
        fn not_now(&self, boxes: &[([i32; 3], [i32; 3])]) {
            self.not_now.lock().unwrap().extend_from_slice(boxes);
        }
    }

    fn late_now(lod: &mut CubeLod) {
        lod.late_content([0.0; 3], [0.0, 0.0, 1.0], 1e4);
    }

    fn sorted(keys: impl IntoIterator<Item = CubeKey>) -> Vec<CubeKey> {
        let mut keys: Vec<CubeKey> = keys.into_iter().collect();
        keys.sort();
        keys
    }

    #[test]
    fn a_node_released_and_planned_again_counts_as_made_again() {
        let mut lod = lod_for_test();
        let (a, b) = (key(1, 0, 0, 0), key(2, 5, 0, 0));
        put(&mut lod, &[a, b]);
        lod.deliver(a, false);
        // `b` is dropped before it is ever drawn.
        put(&mut lod, &[a]);
        assert_eq!(lod.stats().plan_dropped_unbuilt, 1);
        // `a` is dropped while drawn, and let go at once (inside SHORT_MS).
        put(&mut lod, &[]);
        assert_eq!(lod.removed(), [a]);
        let s = lod.stats();
        assert_eq!((s.released, s.short_lived), (1, 1));
        let mut levels = [0u64; 8];
        levels[1] = 1;
        assert_eq!(s.short_lived_levels, levels);
        assert_eq!((s.delivered, s.rebuilt, s.plan_readded), (1, 0, 0));
        // Planned again: not drawn, and drawn before.
        put(&mut lod, &[a]);
        let s = lod.stats();
        assert_eq!((s.plan_readded, s.plan_reused), (1, 0));
        assert_eq!((s.plan_installs, s.plan_added, s.plan_removed), (4, 3, 2));
        lod.deliver(a, false);
        let s = lod.stats();
        assert_eq!((s.delivered, s.rebuilt), (2, 1));
        // A node planned again while still drawn is free.
        put(&mut lod, &[]);
        put(&mut lod, &[a]);
        assert_eq!(lod.stats().plan_reused, 1);
    }

    /// An exact late content (Earth's) remakes the node whose extent a box is,
    /// not every drawn node the box touches.
    #[test]
    fn exact_late_content_remakes_only_the_nodes_of_its_boxes() {
        let run = |exact: bool| {
            let mut lod = lod_for_test();
            let (k, parent, child, other) = (
                key(1, 1, 0, 0),
                key(2, 0, 0, 0),
                key(0, 2, 0, 0),
                key(1, 9, 0, 0),
            );
            put(&mut lod, &[k, parent, child, other]);
            for n in [k, parent, child, other] {
                lod.deliver(n, false);
            }
            let late = Arc::new(Late {
                exact,
                take: Mutex::new(vec![box_of(k)]),
                ..Late::default()
            });
            lod.late = Some(late);
            lod.shared.queue.lock().unwrap().clear();
            late_now(&mut lod);
            let remade = sorted(lod.refreshing.iter().copied());
            let queued = sorted(lod.shared.queue.lock().unwrap().drain());
            assert_eq!(remade, queued);
            (k, parent, child, remade, lod.stats())
        };
        let (k, _, _, remade, s) = run(true);
        assert_eq!(remade, [k]);
        assert_eq!((s.late_refreshed, s.late_other), (1, 0));
        // Not exact: the old scan over every node the box touches.
        let (k, parent, child, remade, s) = run(false);
        assert_eq!(remade, sorted([k, parent, child]));
        assert_eq!((s.late_refreshed, s.late_other), (3, 2));
    }

    /// A leaf the plan held back splits once it is drawn: not made whole
    /// meanwhile, its box put back for later.
    #[test]
    fn a_held_leaf_is_not_remade_and_its_box_goes_back() {
        let mut lod = lod_for_test();
        let (held, other) = (key(1, 1, 0, 0), key(1, 4, 0, 0));
        put_held(&mut lod, &[held, other], &[held]);
        assert!(lod.held_keys.contains(&held));
        lod.deliver(held, false);
        lod.deliver(other, false);
        let late = Arc::new(Late {
            exact: true,
            take: Mutex::new(vec![box_of(held), box_of(other)]),
            ..Late::default()
        });
        lod.late = Some(late.clone());
        lod.shared.queue.lock().unwrap().clear();
        late_now(&mut lod);
        assert_eq!(sorted(lod.refreshing.iter().copied()), [other]);
        assert_eq!(*late.not_now.lock().unwrap(), [box_of(held)]);
        let s = lod.stats();
        assert_eq!((s.late_skipped_held, s.late_refreshed), (1, 1));
        // The leaves held are kept up to date by a plan that did not change.
        let done = PlanDone {
            epoch: lod.epoch,
            leaves: lod.plan.clone(),
            changed: false,
            added: Vec::new(),
            removed: Vec::new(),
            tiles: Arc::default(),
            tiles_added: Vec::new(),
            tiles_removed: Vec::new(),
            order: Vec::new(),
            cover: FxHashSet::default(),
            cover_tiles: FxHashSet::default(),
            tree: Vec::new(),
            footprints: Vec::new(),
            pruned: 0,
            limited: false,
            coarsened: 1.0,
            tiles_limited: false,
            held: 0,
            held_keys: Vec::new(),
            elapsed: Duration::ZERO,
        };
        lod.install(done, Instant::now());
        assert!(lod.held_keys.is_empty());
    }

    /// A refresh queued for a node the plan drops is not made, and Earth's
    /// batch place for it is freed at once.
    #[test]
    fn a_plan_dropping_a_queued_refresh_cancels_it() {
        let mut lod = lod_for_test();
        let (a, b) = (key(1, 0, 0, 0), key(1, 4, 0, 0));
        put(&mut lod, &[a, b]);
        lod.deliver(a, false);
        lod.deliver(b, false);
        let late = Arc::new(Late {
            exact: true,
            take: Mutex::new(vec![box_of(a)]),
            ..Late::default()
        });
        lod.late = Some(late.clone());
        lod.shared.queue.lock().unwrap().clear();
        late_now(&mut lod);
        assert_eq!(sorted(lod.refreshing.iter().copied()), [a]);
        assert_eq!(lod.stats().late_refreshed, 1);
        // The plan drops `a`: it stays drawn until replaced, but is not remade.
        put(&mut lod, &[b]);
        assert!(lod.refreshing.is_empty());
        assert_eq!(*late.not_drawn.lock().unwrap(), [box_of(a)]);
        assert_eq!(lod.stats().late_dropped, 1);
        assert!(lod.shared.queue.lock().unwrap().drain().is_empty());
        // Queued again (the plan took it back, a box for it came): not skipped.
        put(&mut lod, &[a, b]);
        late.take.lock().unwrap().push(box_of(a));
        late_now(&mut lod);
        assert_eq!(sorted(lod.refreshing.iter().copied()), [a]);
        assert_eq!(lod.shared.queue.lock().unwrap().drain(), [a]);
    }

    /// A refresh released soon after it was queued counts as wasted.
    #[test]
    fn a_refreshed_node_released_soon_counts_as_wasted() {
        let mut lod = lod_for_test();
        let (a, b) = (key(1, 0, 0, 0), key(1, 4, 0, 0));
        put(&mut lod, &[a, b]);
        lod.deliver(a, false);
        lod.deliver(b, false);
        lod.late = Some(Arc::new(Late {
            exact: true,
            take: Mutex::new(vec![box_of(a)]),
            ..Late::default()
        }));
        lod.shared.queue.lock().unwrap().clear();
        late_now(&mut lod);
        // The refresh is on its way (not cancelled: `a` is a coverage node
        // for the plan's purposes here), the plan lets `a` go anyway.
        lod.cover.insert(a);
        put(&mut lod, &[b]);
        assert_eq!(lod.stats().late_dropped, 0);
        lod.cover.remove(&a);
        assert_eq!(lod.removed(), [a]);
        assert_eq!(lod.stats().late_wasted, 1);
    }

    /// The late content hears which nodes come coarse first, when that
    /// changes: the holes, and everything in the coverage phase.
    #[test]
    fn the_late_content_is_told_the_holes_when_they_change() {
        let mut lod = lod_for_test();
        let nodes = [
            key(1, 0, 0, 0),
            key(1, 1, 0, 0),
            key(1, 2, 0, 0),
            key(1, 3, 0, 0),
            key(1, 4, 0, 0),
        ];
        put(&mut lod, &nodes);
        // A fresh plan asks for the coverage layer first: let that phase end.
        for n in nodes {
            lod.deliver(n, false);
        }
        lod.advance_phase(Instant::now());
        assert!(matches!(lod.phase, Phase::Refine));
        let late = Arc::new(Late {
            exact: true,
            ..Late::default()
        });
        lod.late = Some(late.clone());
        late_now(&mut lod);
        assert_eq!(*late.coarse.lock().unwrap(), [(vec![], false)]);
        // Nothing changed: not told again.
        late_now(&mut lod);
        assert_eq!(late.coarse.lock().unwrap().len(), 1);
        // A new node with nothing drawn over it: a hole, to come coarse first.
        let hole = key(1, 5, 0, 0);
        let mut more = nodes.to_vec();
        more.push(hole);
        put(&mut lod, &more);
        assert!(matches!(lod.phase, Phase::Refine));
        late_now(&mut lod);
        assert_eq!(late.coarse.lock().unwrap()[1], (vec![hole], false));
        late_now(&mut lod);
        assert_eq!(late.coarse.lock().unwrap().len(), 2);
        // Drawn: no hole any more.
        lod.deliver(hole, false);
        lod.recheck_holes(Instant::now(), false);
        late_now(&mut lod);
        assert_eq!(late.coarse.lock().unwrap()[2], (vec![], false));
    }

    #[test]
    fn the_whole_plan_comes_coarse_first_in_the_coverage_phase() {
        let mut lod = lod_for_test();
        let late = Arc::new(Late {
            exact: true,
            ..Late::default()
        });
        lod.late = Some(late.clone());
        // A fresh plan with nothing drawn: the coverage layer comes first.
        let hole = key(1, 0, 0, 0);
        put(&mut lod, &[hole]);
        assert!(matches!(lod.phase, Phase::Cover { .. }));
        late_now(&mut lod);
        assert_eq!(*late.coarse.lock().unwrap(), [(vec![hole], true)]);
        late_now(&mut lod);
        assert_eq!(late.coarse.lock().unwrap().len(), 1);
    }

    /// The nodes the planner holds back a level are exported with the count.
    #[test]
    fn the_planner_exports_the_leaves_it_holds_back() {
        let camera = [0.0, 100.0, 0.0];
        let nowhere = |_: SectionPos| false;
        let plan_of = |drawn: &[CubeKey]| {
            let refined = refined_set(drawn);
            plan_far(&PlanInput {
                camera,
                max_distance: 3000.0,
                rule: &Rule::distance(DETAIL),
                loaded: LoadedBox {
                    horizontal: 2,
                    vertical: 1,
                },
                is_loaded: &nowhere,
                previous: &[],
                class: &|_| NodeClass::Mixed,
                surface: None,
                refined: Some(&refined),
            })
        };
        // Nothing drawn: the roots wait.
        let mut plan = plan_of(&[]);
        for _ in 0..3 {
            assert_eq!(plan.held_keys.len(), plan.held);
            assert!(plan.held > 0, "some node waits to be drawn");
            let held: FxHashSet<CubeKey> = plan.held_keys.iter().copied().collect();
            assert_eq!(held.len(), plan.held, "each once");
            // They are leaves of the plan (the budget did not bind).
            assert!(held.iter().all(|k| plan.leaves.contains(k)));
            plan = plan_of(&plan.leaves);
        }
    }
}

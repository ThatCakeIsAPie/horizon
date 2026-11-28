//! GPU-driven culling and indirect draws of Earth's CPU-meshed far tiles
//! (`Renderer::lod_cubes`), which are whole: the CPU keeps a small, incrementally
//! maintained set instead of walking every tile each frame.
//!
//! API (for the integration in gfx.rs):
//! - [`FarRecords`] (pure CPU): `touch(key)` at every `lod_cubes` mutation, `clear()`,
//!   `forget()` (a frame was given up: upload every record again), `sync(tiles, now,
//!   closing_page)` before the frame's visibility code, `cpu_keys()` (the tiles the CPU
//!   still walks: `(key, true)` draws everything of the tile, `(key, false)` only its
//!   water), `animated_layers(mask)`, `check(tiles, now)` (O(all), tests and debug
//!   builds), `records()`, `cpu_len()`, `water_len()`.
//! - [`FarCull`]: `supported(..)` / `new(device, hiz_layout)` (`None` with
//!   `HORIZON_FAR_GPU_CULL=0` or a device without indirect counts; `HORIZON_FAR_HIZ=0`
//!   leaves occlusion off), a pub field `records`, `prepare(device, uploads, encoder,
//!   &FarInputs, timing) -> bool` (apply + cull + phase 1's draw lists; `false`: draw
//!   nothing of the far tiles on the GPU, the CPU walks them), `occluding()`,
//!   `draw(pass, arena, page_group, phase, looks)` (phase 0, or 1 after `test`),
//!   `draw_calls()`, `test(encoder, hiz_group, timing)` (the Hi-Z test and phase 2's
//!   lists, between the two world passes; only when `occluding()`), `copy_stats(encoder,
//!   serial)`, `submitted()`, `collect(device) -> Option<FarGpuStats>`.
//!
//! A tile has a record while the GPU draws its solid quads: `Tile::far_class` Gpu (whole:
//! shown, not fading, not dying). Tiles fading in or out are drawn whole by the CPU
//! (`cpu`), hidden ones by nobody. A Gpu tile with water (`sorted_indices`) is also listed
//! by `cpu_keys` for its translucent quads (`water`).
//!
//! The record table holds one record per id (the lowest free id is reused); the GPU's
//! passes, `far_cull.wgsl`:
//! - `apply`: writes the records that changed (each id once, in id order) and marks each
//!   seen (`hidden[id] = 0`): a new or changed record is drawn in phase 1.
//! - `cull`: a thread per id: an active record whose box meets the frustum is kept (its
//!   camera-relative origin is `vec3(origin - cam_int) - cam_frac`, `instances[id]` holds it
//!   and the UV step), with the runs of its solid quads that can face the eye
//!   (`Sides::visible_from`'s, from the record's `ends` and `keys`). Phase 1 draws it
//!   unless the frame occludes and its last test did not see its box (`hidden`): then it
//!   is a candidate for phase 2.
//! - `count`, `scan`, `scatter` (`*2` for phase 2): a page's draws in record id order,
//!   then run order, with no atomic deciding an order: `count` a workgroup per 64 ids
//!   counts each page's draws in the block; `scan` a workgroup per page scans its blocks'
//!   counts and writes its draw count (clamped to its region); `scatter` writes each
//!   block's draws where its count says. A draw: `[quads * 6, 1, 0, base + first * 4,
//!   id]`.
//! - `test` (needs `Cull`'s pyramid, `hiz_read_layout`): the conservative Hi-Z test
//!   (`gpu_occlusion.wgsl`'s `test_boxes`) of every kept record the frustum listed; stores
//!   `hidden`, and a candidate seen now draws in phase 2.
//!
//! Buffers (grown by power of two when a frame needs more; a new one is bound anew): table
//! `ids * 112` bytes (kept through growth: written whole again), updates `n * 128`,
//! instances `ids * 16`, runs `ids * 36`, hidden `ids * 4`, regions `pages * 8`, draws
//! `RUNS * records * 20` (a region of `RUNS * records on the page` draws per page), counts
//! `(8 + 2 * pages * (1 + ceil(ids / 64))) * 4` (cleared each frame), params 224.
//!
//! Every draw lies in its page's region, its first instance is an id below the instance
//! buffer's length, and its base vertex and quads are within the record's own quads.
#![allow(dead_code)]

use super::{
    Frustum,
    arena::{Arena, try_create},
    gpu_culling::{Grown, RUNS, step, threshold_key},
    lod::{FarClass, Tile},
    pack_timing::{PassTiming, compute_parts},
    upload::Uploads,
};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
    lod::cube::CubeKey,
    mesher::Sides,
};
use glam::{DVec3, Mat4, Vec3, Vec4};
use std::{
    collections::BTreeSet,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::Instant,
};

/// Words per id in `runs` (the shader's `RUN_WORDS`): its page, flags and run count, then
/// each run's first quad and quads.
const RUN_WORDS: u64 = 1 + 2 * RUNS as u64;
/// Statistics words at the start of `counts` (the shader's `STATS`): records the frustum
/// kept, boxes the test did not see, draws per phase, triangles per phase, boxes tested.
const STATS: u32 = 8;
const OCCLUDED: usize = 1;
const DRAWS: usize = 2;
const TRIANGLES: usize = 4;
const TESTED: usize = 6;
const ALL_TRIANGLES: usize = 7;
/// Threads per workgroup of the passes over ids.
const WORKGROUP: u32 = 64;
/// Workgroups along x of a dispatch over blocks of ids (the shader's `ROW`).
const ROW: u32 = 65535;
/// `Params::flags`.
const EYE: u32 = 1;
const OCCLUDE: u32 = 2;
/// `Record::a[2]`.
const ACTIVE: u32 = 1;
/// More changed records than this are uploaded as the whole table.
const UPDATE_LIMIT: usize = 8192;
/// The frustum test's box padding, in blocks plus a share of the box's largest
/// coordinate (the shader's `FRUSTUM_PAD`, `FRUSTUM_REL`): far above what f32 rounding of
/// a camera-relative corner moves a plane's distance.
const FRUSTUM_PAD: f32 = 0.015625;
const FRUSTUM_REL: f32 = 1.0 / 262_144.0;
/// The occlusion test's box padding (the shader's `TEST_PAD`).
const TEST_PAD: f32 = 1.015625;
/// The instances' UV step, `1 / TILED_UV_STEPS` (`lod::instance` for fade 0).
const UV_STEP: f32 = 1.0 / crate::mesher::TILED_UV_STEPS;
const READBACKS: usize = 3;
const FREE: u8 = 0;
const PENDING: u8 = 1;
const MAPPED: u8 = 2;
const FAILED: u8 = 3;

/// One far tile (`far_cull.wgsl`'s `Rec`, std430).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Record {
    /// The tile's origin (blocks) and its arena page.
    origin: [i32; 4],
    /// The base vertex of its first solid quad, its solid quads (where its last bucket by
    /// facing side ends), its flags (`ACTIVE`), spare.
    a: [u32; 4],
    /// Its mesh's bounds around the origin (`lod::Tile::bounds_box`).
    min: [f32; 4],
    max: [f32; 4],
    /// Where its first six buckets end, in `mesher::Sides` order: +X, +Z, the other
    /// bucket, +Y, -Z, -X (`gpu_culling::Record::facing`).
    ends: [u32; 6],
    /// What its buckets facing +X, +Z, +Y, -Z, -X and -Y compare the eye's keys with.
    keys: [u32; 6],
}

impl Record {
    const BYTES: u64 = std::mem::size_of::<Self>() as u64;
    const FREE: Self = Self {
        origin: [0; 4],
        a: [0; 4],
        min: [0.0; 4],
        max: [0.0; 4],
        ends: [0; 6],
        keys: [0; 6],
    };
    /// An active record of `quads` solid quads from vertex `base` of arena page `page`, the
    /// tile at `origin`, its mesh within `bounds`; all in the other bucket (drawn from
    /// everywhere), as a mesh that was not sorted or is sorted for two passes.
    pub fn new(origin: [i32; 3], page: u32, base: u32, quads: u32, bounds: [Vec3; 2]) -> Self {
        let [min, max] = bounds;
        Self {
            origin: [origin[0], origin[1], origin[2], page as i32],
            a: [base, quads, ACTIVE, 0],
            min: [min.x, min.y, min.z, 0.0],
            max: [max.x, max.y, max.z, 0.0],
            ends: [0, 0, quads, quads, quads, quads],
            keys: [0; 6],
        }
    }
    /// The record with its solid quads in `sides`' buckets by the side they face
    /// (`gpu_culling::Record::facing`); a mesh sorted for two passes stays one run.
    pub fn facing(mut self, sides: &Sides) -> Self {
        let Some((ends, t)) = sides.thresholds() else {
            return self;
        };
        let quads = self.a[1];
        debug_assert_eq!(ends[6], quads, "its buckets hold its solid quads");
        self.ends = std::array::from_fn(|b| ends[b].min(quads));
        self.keys = [0, 1, 3, 4, 5, 6].map(|b| threshold_key(t[b], b < 4));
        self
    }
    /// The record of the Gpu-class tile `tile` at `key`.
    pub fn of(key: CubeKey, tile: &Tile) -> Self {
        let s = tile.drawn();
        Self::new(
            key.origin(),
            s.page,
            s.solid_base(),
            s.solid,
            tile.bounds_box(),
        )
        .facing(&s.sides)
    }
    pub fn page(&self) -> u32 {
        self.origin[3] as u32
    }
    pub fn origin(&self) -> [i32; 3] {
        [self.origin[0], self.origin[1], self.origin[2]]
    }
    pub fn quads(&self) -> u32 {
        self.a[1]
    }
    pub fn is_active(&self) -> bool {
        self.a[2] & ACTIVE != 0
    }
    /// Whether both hold the same bytes (what the GPU would see).
    fn same(&self, other: &Self) -> bool {
        bytemuck::bytes_of(self) == bytemuck::bytes_of(other)
    }
}

/// `far_cull.wgsl`'s `Update`: the record for an id.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Update {
    id: [u32; 4],
    record: Record,
}

impl Update {
    const BYTES: u64 = std::mem::size_of::<Self>() as u64;
}

/// `far_cull.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    ids: u32,
    updates: u32,
    pages: u32,
    flags: u32,
    cam_int: [i32; 4],
    cam_frac: [f32; 4],
    eye: [f32; 4],
    planes: [[f32; 4]; 6],
    view_proj: [[f32; 4]; 4],
}

/// What a frame's passes read.
pub(super) struct FarInputs {
    /// The camera, in world blocks.
    pub camera: DVec3,
    /// The frustum's planes (`Frustum::planes`), camera-relative, facing in.
    pub planes: [Vec4; 6],
    /// The eye less the camera (`view::projection_center`) for facing runs; `None`: every
    /// bucket is drawn.
    pub eye: Option<Vec3>,
    /// The world pass's view projection, camera-relative, jitter included.
    pub view_proj: Mat4,
    /// Whether the frame occludes (`cull_on`), if there is a Hi-Z pyramid.
    pub occlude: bool,
}

impl FarInputs {
    pub fn new(
        camera: DVec3,
        frustum: &Frustum,
        eye: Option<Vec3>,
        view_proj: Mat4,
        occlude: bool,
    ) -> Self {
        Self {
            camera,
            planes: frustum.planes,
            eye,
            view_proj,
            occlude,
        }
    }
    /// The camera's block (floor) and the f32 remainder: `origin - cam_int - cam_frac`.
    fn split(&self) -> ([i32; 3], Vec3) {
        let floor = self.camera.floor();
        (
            [floor.x as i32, floor.y as i32, floor.z as i32],
            (self.camera - floor).as_vec3(),
        )
    }
}

/// The newest statistics that came back, with their frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct FarGpuStats {
    pub frame: u64,
    /// Records the frame's table held.
    pub records: u32,
    /// Records the frustum kept.
    pub visible: u32,
    /// Boxes the Hi-Z test did not see, and boxes it tested.
    pub occluded: u32,
    pub tested: u32,
    /// Draws and triangles per phase.
    pub draws: [u32; 2],
    pub triangles: [u32; 2],
    /// The triangles of every solid quad of the records drawn, each face direction.
    pub all_triangles: u32,
}

// ---------------------------------------------------------------------------------------
// The CPU side
// ---------------------------------------------------------------------------------------

/// The far tiles' records and who draws what, kept as tiles change.
#[derive(Default)]
pub(super) struct FarRecords {
    /// Id of each key with a record, and key of each id (`None`: free).
    ids: FxHashMap<CubeKey, u32>,
    keys: Vec<Option<CubeKey>>,
    /// The GPU's table, by id; a free id holds `Record::FREE`. The last id is never free.
    records: Vec<Record>,
    /// Animated layers of each id's mesh.
    layers: Vec<Box<[u16]>>,
    free: BTreeSet<u32>,
    /// Ids whose record changed since `take_updates`; all of them when `all_dirty`.
    pending: FxHashSet<u32>,
    all_dirty: bool,
    /// Per arena page, the records on it and their keys.
    page_records: Vec<u32>,
    page_keys: Vec<FxHashSet<CubeKey>>,
    /// Tiles that changed since `sync`.
    dirty: FxHashSet<CubeKey>,
    /// Tiles the CPU draws whole (fading, dying), and Gpu tiles with water.
    cpu: FxHashSet<CubeKey>,
    water: FxHashSet<CubeKey>,
    /// Records per animated layer.
    layer_counts: FxHashMap<u16, u32>,
}

impl FarRecords {
    /// The tile at `key` changed (installed, removed, shown, dying, its mesh moved).
    pub fn touch(&mut self, key: CubeKey) {
        self.dirty.insert(key);
    }

    /// The meshes on arena page `page` move (compaction): the next `sync` reads each again.
    pub fn touch_page(&mut self, page: u32) {
        if let Some(keys) = self.page_keys.get(page as usize) {
            self.dirty.extend(keys.iter().copied());
        }
    }

    /// Forgets everything (the tiles went); the GPU's table is written again as it fills.
    pub fn clear(&mut self) {
        *self = Self {
            all_dirty: true,
            ..Self::default()
        };
    }

    /// A frame was given up: its updates are lost, so every record goes again.
    pub fn forget(&mut self) {
        self.all_dirty = true;
    }

    /// Makes the records match the tiles that changed (`touch`), the ones the CPU draws (their
    /// fades move with time) and, when the arena is compacting, those on its closing page.
    /// O(changes + |cpu| + the page's records).
    pub fn sync(
        &mut self,
        tiles: &FxHashMap<CubeKey, Tile>,
        now: Instant,
        closing_page: Option<u32>,
    ) {
        let mut work: Vec<CubeKey> = self.dirty.drain().collect();
        work.extend(self.cpu.iter().copied());
        if let Some(keys) = closing_page.and_then(|p| self.page_keys.get(p as usize)) {
            work.extend(keys.iter().copied());
        }
        // In key order, so that the ids tiles get do not depend on the hash sets' order.
        work.sort_unstable();
        work.dedup();
        for key in work {
            self.apply(key, tiles.get(&key), now);
        }
    }

    fn apply(&mut self, key: CubeKey, tile: Option<&Tile>, now: Instant) {
        let class = tile.map_or(FarClass::Hidden, |t| t.far_class(key.level, now));
        match (class, tile) {
            (FarClass::Gpu, Some(tile)) => {
                self.cpu.remove(&key);
                let record = Record::of(key, tile);
                let layers = &*tile.drawn().animated_layers;
                self.set(key, record, layers);
                if tile.drawn().sorted_indices.is_some() {
                    self.water.insert(key);
                } else {
                    self.water.remove(&key);
                }
            }
            (FarClass::Cpu, _) => {
                self.remove(key);
                self.water.remove(&key);
                self.cpu.insert(key);
            }
            _ => {
                self.remove(key);
                self.water.remove(&key);
                self.cpu.remove(&key);
            }
        }
    }

    /// `key`'s record is `record`, its mesh with `layers` animated.
    fn set(&mut self, key: CubeKey, record: Record, layers: &[u16]) {
        let Some(&id) = self.ids.get(&key) else {
            let id = match self.free.pop_first() {
                Some(id) => id,
                None => {
                    self.records.push(Record::FREE);
                    self.keys.push(None);
                    self.layers.push(Box::default());
                    self.records.len() as u32 - 1
                }
            };
            self.ids.insert(key, id);
            self.keys[id as usize] = Some(key);
            self.records[id as usize] = record;
            self.add_layers(id, layers);
            self.add_to_page(record.page(), key);
            self.pending.insert(id);
            return;
        };
        let at = id as usize;
        if *self.layers[at] != *layers {
            self.drop_layers(id);
            self.add_layers(id, layers);
        }
        if self.records[at].same(&record) {
            return;
        }
        let old = self.records[at];
        if old.page() != record.page() {
            self.remove_from_page(old.page(), key);
            self.add_to_page(record.page(), key);
        }
        self.records[at] = record;
        self.pending.insert(id);
    }

    /// `key` has no record any more.
    fn remove(&mut self, key: CubeKey) {
        let Some(id) = self.ids.remove(&key) else {
            return;
        };
        let at = id as usize;
        self.remove_from_page(self.records[at].page(), key);
        self.drop_layers(id);
        self.records[at] = Record::FREE;
        self.keys[at] = None;
        self.free.insert(id);
        self.pending.insert(id);
        // Free ids at the end leave the table, so the passes test fewer.
        while self.keys.last().is_some_and(Option::is_none) {
            let last = self.keys.len() as u32 - 1;
            self.keys.pop();
            self.records.pop();
            self.layers.pop();
            self.free.remove(&last);
        }
    }

    fn add_to_page(&mut self, page: u32, key: CubeKey) {
        let page = page as usize;
        if self.page_records.len() <= page {
            self.page_records.resize(page + 1, 0);
            self.page_keys.resize_with(page + 1, Default::default);
        }
        self.page_records[page] += 1;
        self.page_keys[page].insert(key);
    }

    fn remove_from_page(&mut self, page: u32, key: CubeKey) {
        let page = page as usize;
        self.page_records[page] -= 1;
        self.page_keys[page].remove(&key);
        // Pages past the last with records are not counted.
        while self.page_records.last() == Some(&0) {
            self.page_records.pop();
            self.page_keys.pop();
        }
    }

    fn add_layers(&mut self, id: u32, layers: &[u16]) {
        for &layer in layers {
            *self.layer_counts.entry(layer).or_default() += 1;
        }
        self.layers[id as usize] = layers.into();
    }

    fn drop_layers(&mut self, id: u32) {
        for layer in std::mem::take(&mut self.layers[id as usize]).iter() {
            if let Some(n) = self.layer_counts.get_mut(layer) {
                *n -= 1;
                if *n == 0 {
                    self.layer_counts.remove(layer);
                }
            }
        }
    }

    /// The tiles the CPU still lists: `(key, true)` fading or dying ones, which it draws
    /// whole, `(key, false)` those with water, of which it draws the translucent quads.
    pub fn cpu_keys(&self) -> impl Iterator<Item = (CubeKey, bool)> + '_ {
        let cpu = self.cpu.iter().map(|&key| (key, true));
        cpu.chain(self.water.iter().map(|&key| (key, false)))
    }

    /// Sets the bit of every animated layer of the records' meshes.
    pub fn animated_layers(&self, mask: &mut [u64]) {
        for &layer in self.layer_counts.keys() {
            if let Some(word) = mask.get_mut(layer as usize / u64::BITS as usize) {
                *word |= 1 << (layer as usize % u64::BITS as usize);
            }
        }
    }

    /// Tiles with a record.
    pub fn records(&self) -> usize {
        self.ids.len()
    }
    pub fn cpu_len(&self) -> usize {
        self.cpu.len()
    }
    pub fn water_len(&self) -> usize {
        self.water.len()
    }
    /// The ids the passes test: one past the highest in use.
    pub fn id_count(&self) -> u32 {
        self.records.len() as u32
    }
    /// Records per arena page, up to the last page with any.
    pub fn page_records(&self) -> &[u32] {
        &self.page_records
    }
    pub fn id_of(&self, key: CubeKey) -> Option<u32> {
        self.ids.get(&key).copied()
    }
    pub fn record(&self, id: u32) -> Option<&Record> {
        self.records.get(id as usize).filter(|r| r.is_active())
    }

    /// The changed records, each id once and in id order, into `out`; `true` when the
    /// whole table is to be written instead (`all_dirty`, or too many changed).
    fn take_updates(&mut self, out: &mut Vec<Update>) -> bool {
        out.clear();
        let whole = self.all_dirty || self.pending.len() > UPDATE_LIMIT;
        if !whole {
            let mut ids: Vec<u32> = self.pending.iter().copied().collect();
            ids.retain(|&id| (id as usize) < self.records.len());
            ids.sort_unstable();
            out.extend(ids.into_iter().map(|id| Update {
                id: [id, 0, 0, 0],
                record: self.records[id as usize],
            }));
        }
        self.pending.clear();
        self.all_dirty = false;
        whole
    }

    /// Debug invariant after `sync` (O(all)): every Gpu tile has exactly one active record,
    /// equal to `Record::of`, and no other tile has; the sets and counters agree.
    pub fn check(&self, tiles: &FxHashMap<CubeKey, Tile>, now: Instant) {
        let (mut gpu, mut cpu, mut water) = (0, 0, 0);
        for (&key, tile) in tiles {
            match tile.far_class(key.level, now) {
                FarClass::Gpu => {
                    gpu += 1;
                    let id = self
                        .ids
                        .get(&key)
                        .unwrap_or_else(|| panic!("{key:?} has no record"));
                    assert!(
                        self.records[*id as usize].same(&Record::of(key, tile)),
                        "{key:?}'s record is stale"
                    );
                    assert!(!self.cpu.contains(&key), "{key:?} is also the CPU's");
                    let wet = tile.drawn().sorted_indices.is_some();
                    assert_eq!(self.water.contains(&key), wet, "{key:?}'s water");
                    water += usize::from(wet);
                }
                FarClass::Cpu => {
                    cpu += 1;
                    assert!(self.cpu.contains(&key), "{key:?} is not the CPU's");
                    assert!(!self.ids.contains_key(&key), "{key:?} has a record");
                    assert!(!self.water.contains(&key));
                }
                FarClass::Hidden => {
                    assert!(!self.cpu.contains(&key) && !self.water.contains(&key));
                    assert!(!self.ids.contains_key(&key), "{key:?} has a record");
                }
            }
        }
        assert_eq!(self.ids.len(), gpu, "records of tiles that are not Gpu");
        assert_eq!(self.cpu.len(), cpu);
        assert_eq!(self.water.len(), water);
        self.check_tables();
    }

    /// The tables agree with one another.
    fn check_tables(&self) {
        assert_eq!(self.keys.len(), self.records.len());
        assert_eq!(self.layers.len(), self.records.len());
        assert_eq!(self.ids.len() + self.free.len(), self.records.len());
        assert!(
            self.keys.last().is_none_or(Option::is_some),
            "the last id is in use"
        );
        let mut pages = vec![0u32; self.page_records.len()];
        let mut layer_counts: FxHashMap<u16, u32> = FxHashMap::default();
        for (id, key) in self.keys.iter().enumerate() {
            let record = &self.records[id];
            match key {
                Some(key) => {
                    assert_eq!(self.ids.get(key), Some(&(id as u32)));
                    assert!(record.is_active());
                    assert!(self.page_keys[record.page() as usize].contains(key));
                    pages[record.page() as usize] += 1;
                    for &layer in &*self.layers[id] {
                        *layer_counts.entry(layer).or_default() += 1;
                    }
                }
                None => {
                    assert!(self.free.contains(&(id as u32)));
                    assert!(!record.is_active());
                    assert!(self.layers[id].is_empty());
                }
            }
        }
        assert_eq!(pages, self.page_records);
        assert!(self.page_records.last().is_none_or(|&n| n > 0));
        assert_eq!(self.page_keys.len(), self.page_records.len());
        for (keys, &n) in self.page_keys.iter().zip(&self.page_records) {
            assert_eq!(keys.len() as u32, n);
        }
        assert_eq!(layer_counts, self.layer_counts);
        assert!(self.pending.iter().all(|&id| {
            let id = id as usize;
            id >= self.records.len() || self.keys[id].is_some() || !self.records[id].is_active()
        }));
    }
}

/// Each page's region of the draws: `regions` gets `[first, size]` per page, `pages` the
/// pages with records as (page, first draw, size); returns the draws in all. A region
/// has room for `RUNS` draws of each of its page's records.
fn regions(
    page_records: &[u32],
    pass_pages: u32,
    words: &mut Vec<u32>,
    pages: &mut Vec<(u32, u32, u32)>,
) -> u64 {
    words.clear();
    pages.clear();
    let mut first = 0u32;
    for page in 0..pass_pages {
        let size = RUNS * page_records.get(page as usize).copied().unwrap_or(0);
        words.extend([first, size]);
        if size > 0 {
            pages.push((page, first, size));
        }
        first += size;
    }
    u64::from(first)
}

/// Words of `counts`: the statistics, per phase and page its draw count, then per phase,
/// page and block of ids the block's draws.
fn count_words(pages: u32, ids: u32) -> u64 {
    let blocks = u64::from(ids.div_ceil(WORKGROUP));
    u64::from(STATS) + 2 * u64::from(pages) * (1 + blocks)
}

// ---------------------------------------------------------------------------------------
// The GPU side
// ---------------------------------------------------------------------------------------

/// Whether the device can run the passes and the count draws: Vulkan, Metal and OpenGL
/// where `multi_draw` is (`translucent_batchable`), not DX12 (its count draws leave each
/// draw's base vertex out of `vertex_index`, `gpu_culling::loses_base_vertex`), not Intel
/// Gen11/12's Vulkan driver (`flat::is_broken_intel_vulkan`, whose multi-draws get the
/// first instance wrong).
pub(super) fn supported(
    info: &wgpu::AdapterInfo,
    features: wgpu::Features,
    downlevel: wgpu::DownlevelFlags,
    limits: &wgpu::Limits,
    multi_draw: bool,
) -> bool {
    super::translucent_batchable(info, multi_draw)
        && info.backend != wgpu::Backend::Dx12
        && features.contains(
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | wgpu::Features::INDIRECT_FIRST_INSTANCE,
        )
        && downlevel.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        // The passes bind 8.
        && limits.max_storage_buffers_per_shader_stage >= 8
        && !super::flat::is_broken_intel_vulkan(info)
}

/// `HORIZON_FAR_GPU_CULL=0` turns the far tiles' GPU culling off.
fn enabled() -> bool {
    std::env::var("HORIZON_FAR_GPU_CULL").as_deref() != Ok("0")
}

/// `HORIZON_FAR_HIZ=0` turns its occlusion test off.
fn hiz_enabled() -> bool {
    std::env::var("HORIZON_FAR_HIZ").as_deref() != Ok("0")
}

/// Phase 2's passes: the Hi-Z test and the draw lists of what it saw.
struct Phase2 {
    count: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    test: wgpu::ComputePipeline,
}

/// One frame's statistics on their way back.
struct Readback {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    frame: u64,
    records: u32,
}

pub(super) struct FarCull {
    /// apply, cull, count, scan, scatter.
    pipelines: [wgpu::ComputePipeline; 5],
    phase2: Option<Phase2>,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    table: Grown,
    updates: Grown,
    instances: Grown,
    runs: Grown,
    hidden: Grown,
    draws: Grown,
    counts: Grown,
    regions: Grown,
    /// Dropped as soon as one of the buffers is made anew.
    group: Option<wgpu::BindGroup>,
    /// What `table` holds once `prepare` writes it.
    pub records: FarRecords,
    /// This frame's pages with records: (page, first draw of its region, the region's
    /// size).
    pages: Vec<(u32, u32, u32)>,
    ids: u32,
    n_updates: u32,
    pass_pages: u32,
    occluding: bool,
    scratch: Vec<Update>,
    words: Vec<u32>,
    readbacks: Vec<Readback>,
    next: usize,
    current: Option<usize>,
}

impl FarCull {
    /// `None` when it is turned off (`HORIZON_FAR_GPU_CULL=0`), the device cannot count
    /// draws on the GPU, the pipelines fail, or the memory for the buffers is missing.
    /// `hiz_layout`: `Cull::hiz_read_layout`, for the occlusion test.
    pub fn new(device: &wgpu::Device, hiz_layout: Option<&wgpu::BindGroupLayout>) -> Option<Self> {
        if !enabled()
            || !device.features().contains(
                wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | wgpu::Features::INDIRECT_FIRST_INSTANCE,
            )
        {
            return None;
        }
        let hiz_layout = hiz_layout.filter(|_| hiz_enabled());
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage =
            |binding, read_only| entry(binding, wgpu::BufferBindingType::Storage { read_only });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Far culling"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                storage(1, false),
                storage(2, true),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                storage(6, false),
                storage(7, false),
                storage(8, true),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Far culling"),
            source: wgpu::ShaderSource::Wgsl(include_str!("far_cull.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Far culling"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = |layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Far culling"),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = ["apply", "cull", "count", "scan", "scatter"]
            .map(|entry| pipeline(&pipeline_layout, entry));
        let phase2 = hiz_layout.map(|hiz| {
            let test_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Far culling test"),
                bind_group_layouts: &[Some(&layout), Some(hiz)],
                ..Default::default()
            });
            Phase2 {
                count: pipeline(&pipeline_layout, "count2"),
                scan: pipeline(&pipeline_layout, "scan2"),
                scatter: pipeline(&pipeline_layout, "scatter2"),
                test: pipeline(&test_layout, "test"),
            }
        });
        let memory = pollster::block_on(memory.pop());
        let internal = pollster::block_on(internal.pop());
        let validation = pollster::block_on(validation.pop());
        if let Some(e) = memory.or(internal).or(validation) {
            log::warn!("Horizon renderer: far GPU culling unavailable ({e})");
            return None;
        }
        use wgpu::BufferUsages as U;
        let readbacks = (0..READBACKS)
            .map(|_| {
                Some(Readback {
                    buffer: try_create(
                        device,
                        "Far culling statistics",
                        u64::from(STATS) * 4,
                        U::MAP_READ | U::COPY_DST,
                    )?,
                    state: Arc::new(AtomicU8::new(FREE)),
                    frame: 0,
                    records: 0,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let storage_usage = U::STORAGE | U::COPY_DST;
        Some(Self {
            pipelines,
            phase2,
            layout,
            params: try_create(
                device,
                "Far culling params",
                std::mem::size_of::<Params>() as u64,
                U::UNIFORM | U::COPY_DST,
            )?,
            table: Grown::new(
                device,
                "Far culling records",
                256 * Record::BYTES,
                storage_usage,
            )?,
            updates: Grown::new(
                device,
                "Far culling updates",
                64 * Update::BYTES,
                storage_usage,
            )?,
            instances: Grown::new(
                device,
                "Far culling instances",
                256 * 16,
                U::VERTEX | U::STORAGE | test_read(),
            )?,
            runs: Grown::new(
                device,
                "Far culling runs",
                256 * RUN_WORDS * 4,
                U::STORAGE | test_read(),
            )?,
            hidden: Grown::new(
                device,
                "Far culling hidden",
                256 * 4,
                storage_usage | test_read(),
            )?,
            draws: Grown::new(
                device,
                "Far culling draws",
                1024 * 20,
                U::STORAGE | U::INDIRECT | test_read(),
            )?,
            counts: Grown::new(
                device,
                "Far culling counts",
                1024 * 4,
                // Copied out for the statistics (`copy_stats`).
                U::STORAGE | U::INDIRECT | U::COPY_DST | U::COPY_SRC,
            )?,
            regions: Grown::new(device, "Far culling regions", 64 * 8, storage_usage)?,
            group: None,
            records: FarRecords::default(),
            pages: Vec::new(),
            ids: 0,
            n_updates: 0,
            pass_pages: 1,
            occluding: false,
            scratch: Vec::new(),
            words: Vec::new(),
            readbacks,
            next: 0,
            current: None,
        })
    }

    /// Records the frame's uploads into `encoder` and runs `apply`, `cull` and phase 1's
    /// draw lists. `false` (the camera or eye is not finite, or memory is missing): nothing
    /// of the frame is drawn by the GPU, so the CPU must walk the tiles. A frame given up
    /// after this call must `records.forget()`.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        uploads: &mut Uploads,
        encoder: &mut wgpu::CommandEncoder,
        input: &FarInputs,
        timing: Option<&mut PassTiming>,
    ) -> bool {
        self.pages.clear();
        self.ids = 0;
        self.n_updates = 0;
        self.occluding = false;
        if !input.camera.is_finite() || input.eye.is_some_and(|e| !e.is_finite()) {
            return false;
        }
        match self.prepare_buffers(device, uploads, encoder, input) {
            Some(()) => {
                self.dispatch(encoder, timing);
                true
            }
            None => {
                self.records.forget();
                false
            }
        }
    }

    fn prepare_buffers(
        &mut self,
        device: &wgpu::Device,
        uploads: &mut Uploads,
        encoder: &mut wgpu::CommandEncoder,
        input: &FarInputs,
    ) -> Option<()> {
        let ids = self.records.id_count();
        let pass_pages = (self.records.page_records().len() as u32).max(1);
        let draws = regions(
            self.records.page_records(),
            pass_pages,
            &mut self.words,
            &mut self.pages,
        );
        let n = u64::from(ids);
        // Every buffer grows before anything is written; one made anew drops the bind group.
        if self.table.reserve(device, n * Record::BYTES)? {
            self.group = None;
            self.records.all_dirty = true;
        }
        for (buffer, bytes) in [
            (&mut self.instances, n * 16),
            (&mut self.runs, n * RUN_WORDS * 4),
            (&mut self.hidden, n * 4),
            (&mut self.counts, count_words(pass_pages, ids) * 4),
            (&mut self.regions, u64::from(pass_pages) * 8),
            (&mut self.draws, draws * 20),
        ] {
            if buffer.reserve(device, bytes)? {
                self.group = None;
            }
        }
        let whole = self.records.take_updates(&mut self.scratch);
        if !whole
            && self
                .updates
                .reserve(device, self.scratch.len() as u64 * Update::BYTES)?
        {
            self.group = None;
        }
        if whole {
            if ids > 0 {
                uploads.buffer(
                    encoder,
                    &self.table.buffer,
                    0,
                    bytemuck::cast_slice(&self.records.records),
                );
            }
            self.scratch.clear();
            // Every record is new: seen, drawn in phase 1.
            encoder.clear_buffer(&self.hidden.buffer, 0, Some(n * 4));
        } else {
            uploads.buffer(
                encoder,
                &self.updates.buffer,
                0,
                bytemuck::cast_slice(&self.scratch),
            );
        }
        uploads.buffer(
            encoder,
            &self.regions.buffer,
            0,
            bytemuck::cast_slice(&self.words),
        );
        let (cam_int, cam_frac) = input.split();
        let occluding = input.occlude && self.phase2.is_some() && ids > 0;
        let mut flags = 0;
        if input.eye.is_some() {
            flags |= EYE;
        }
        if occluding {
            flags |= OCCLUDE;
        }
        let eye = input.eye.unwrap_or(Vec3::ZERO);
        let params = Params {
            ids,
            updates: self.scratch.len() as u32,
            pages: pass_pages,
            flags,
            cam_int: [cam_int[0], cam_int[1], cam_int[2], 0],
            cam_frac: [cam_frac.x, cam_frac.y, cam_frac.z, 0.0],
            eye: [eye.x, eye.y, eye.z, UV_STEP],
            planes: input.planes.map(|p| p.to_array()),
            view_proj: input.view_proj.to_cols_array_2d(),
        };
        uploads.buffer(encoder, &self.params, 0, bytemuck::bytes_of(&params));
        encoder.clear_buffer(
            &self.counts.buffer,
            0,
            Some(count_words(pass_pages, ids) * 4),
        );
        if self.group.is_none() {
            self.group = Some(self.bind(device));
        }
        self.ids = ids;
        self.n_updates = params.updates;
        self.pass_pages = pass_pages;
        self.occluding = occluding;
        Some(())
    }

    fn bind(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Far culling"),
            layout: &self.layout,
            entries: &[
                entry(0, &self.params),
                entry(1, &self.table.buffer),
                entry(2, &self.updates.buffer),
                entry(3, &self.instances.buffer),
                entry(4, &self.runs.buffer),
                entry(5, &self.hidden.buffer),
                entry(6, &self.draws.buffer),
                entry(7, &self.counts.buffer),
                entry(8, &self.regions.buffer),
            ],
        })
    }

    fn rows(&self) -> [u32; 2] {
        let blocks = self.ids.div_ceil(WORKGROUP);
        [blocks.min(ROW), blocks.div_ceil(ROW)]
    }

    /// Phase 1: the changed records into the table, then the kept records' draws.
    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, timing: Option<&mut PassTiming>) {
        let Some(group) = self.group.as_ref().filter(|_| self.ids > 0) else {
            return;
        };
        let [apply, cull, count, scan, scatter] = &self.pipelines;
        let rows = self.rows();
        let pages = [self.pass_pages, 1];
        let mut apply = step(group, apply, [self.n_updates.div_ceil(WORKGROUP), 1]);
        let mut cull = step(group, cull, rows);
        let mut count = step(group, count, rows);
        let mut scan = step(group, scan, pages);
        let mut scatter = step(group, scatter, rows);
        compute_parts(
            encoder,
            timing,
            "Far culling",
            &mut [
                ("apply", &mut apply),
                ("cull", &mut cull),
                ("count", &mut count),
                ("scan", &mut scan),
                ("scatter", &mut scatter),
            ],
        );
    }

    /// The kept tiles' solid draws, a count draw per page, with the quad indices bound by
    /// the caller. Leaves the instances at vertex slot 1 (the caller restores its own).
    /// `group`: the page's group 1 for pipelines that read the page themselves, else none.
    /// `phase`: 0, or 1 after `test`.
    pub fn draw<'a>(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        arena: &Arena,
        group: impl Fn(u32) -> Option<&'a wgpu::BindGroup>,
        phase: u64,
        looks: Option<&super::looks::LookGroups>,
    ) {
        if self.pages.is_empty() || (phase == 1 && !self.occluding) {
            return;
        }
        pass.set_vertex_buffer(1, self.instances.buffer.slice(..));
        for &(page, first, size) in &self.pages {
            if let Some(group) = group(page) {
                pass.set_bind_group(1, group, &[]);
            } else {
                pass.set_vertex_buffer(0, arena.buffer(page).slice(..));
            }
            if let Some(looks) = looks {
                pass.set_bind_group(super::looks::GROUP, looks.page(page, false), &[]);
            }
            pass.multi_draw_indexed_indirect_count(
                &self.draws.buffer,
                u64::from(first) * 20,
                &self.counts.buffer,
                (u64::from(STATS) + phase * u64::from(self.pass_pages) + u64::from(page)) * 4,
                size,
            );
        }
    }

    /// Draw calls `draw` makes.
    pub fn draw_calls(&self) -> usize {
        self.pages.len()
    }

    /// Whether the frame occludes (`prepare`): phase 1's `draw`, then `test` between the
    /// world passes, then phase 2's `draw`.
    pub fn occluding(&self) -> bool {
        self.occluding
    }

    /// Between the world passes, after the pyramid is built from phase 1's depth: the Hi-Z
    /// test of every kept record, then phase 2's draw lists, in the regions phase 1 drew from.
    pub fn test(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hiz: &wgpu::BindGroup,
        timing: Option<&mut PassTiming>,
    ) {
        let (Some(phase2), Some(group)) = (&self.phase2, &self.group) else {
            return;
        };
        if !self.occluding {
            return;
        }
        let rows = self.rows();
        let mut test = |pass: &mut wgpu::ComputePass<'_>| {
            if rows[0] > 0 {
                pass.set_pipeline(&phase2.test);
                pass.set_bind_group(0, group, &[]);
                pass.set_bind_group(1, hiz, &[]);
                pass.dispatch_workgroups(rows[0], rows[1], 1);
            }
        };
        let mut count = step(group, &phase2.count, rows);
        let mut scan = step(group, &phase2.scan, [self.pass_pages, 1]);
        let mut scatter = step(group, &phase2.scatter, rows);
        compute_parts(
            encoder,
            timing,
            "Far culling phase 2",
            &mut [
                ("test", &mut test),
                ("count2", &mut count),
                ("scan2", &mut scan),
                ("scatter2", &mut scatter),
            ],
        );
    }

    /// After the frame's world passes, in the same encoder: the statistics copied out
    /// for frame `serial` into the next readback, if it is free.
    pub fn copy_stats(&mut self, encoder: &mut wgpu::CommandEncoder, serial: u64) {
        let r = &mut self.readbacks[self.next];
        if r.state.load(Ordering::Acquire) != FREE || self.ids == 0 {
            return;
        }
        encoder.copy_buffer_to_buffer(&self.counts.buffer, 0, &r.buffer, 0, u64::from(STATS) * 4);
        (r.frame, r.records) = (serial, self.ids);
        self.current = Some(self.next);
    }

    /// The readback to start right after the frame's submission (the after-submit hook).
    pub fn submitted(&mut self) -> Option<impl FnOnce() + Send + 'static> {
        let i = self.current.take()?;
        let r = &self.readbacks[i];
        let (buffer, state) = (r.buffer.clone(), r.state.clone());
        state.store(PENDING, Ordering::Release);
        self.next = (i + 1) % READBACKS;
        Some(move || {
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    state.store(
                        if result.is_ok() { MAPPED } else { FAILED },
                        Ordering::Release,
                    );
                });
        })
    }

    /// The statistics that came back, the newest frame's; never waits.
    pub fn collect(&mut self, device: &wgpu::Device) -> Option<FarGpuStats> {
        self.current = None;
        if device.poll(wgpu::PollType::Poll).is_err() {
            return None;
        }
        let mut newest: Option<FarGpuStats> = None;
        for r in &self.readbacks {
            if r.state.load(Ordering::Acquire) != MAPPED {
                if r.state.load(Ordering::Acquire) == FAILED {
                    r.state.store(FREE, Ordering::Release);
                }
                continue;
            }
            {
                let Ok(data) = r.buffer.slice(..).get_mapped_range() else {
                    r.state.store(FAILED, Ordering::Release);
                    continue;
                };
                let words: [u32; STATS as usize] =
                    bytemuck::pod_read_unaligned(&data[..STATS as usize * 4]);
                if newest.is_none_or(|s| r.frame > s.frame) {
                    newest = Some(FarGpuStats {
                        frame: r.frame,
                        records: r.records,
                        visible: words[0],
                        occluded: words[OCCLUDED],
                        tested: words[TESTED],
                        draws: [words[DRAWS], words[DRAWS + 1]],
                        triangles: [words[TRIANGLES], words[TRIANGLES + 1]],
                        all_triangles: words[ALL_TRIANGLES],
                    });
                }
            }
            r.buffer.unmap();
            r.state.store(FREE, Ordering::Release);
        }
        newest
    }
}

/// Buffers tests read back also take `COPY_SRC`.
const fn test_read() -> wgpu::BufferUsages {
    if cfg!(test) {
        wgpu::BufferUsages::COPY_SRC
    } else {
        wgpu::BufferUsages::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::cull::tests::{cpu_build_hiz, cpu_query_hiz};
    use crate::gfx::gpu_culling::order_key;
    use crate::gfx::world_projection;
    use crate::mesher::Part;
    use std::time::Duration;

    /// xorshift64*.
    struct Rng(u64);
    impl Rng {
        fn bits(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u32) -> u32 {
            (self.bits() >> 33) as u32 % n
        }
        fn range(&mut self, low: i32, high: i32) -> i32 {
            low + self.below((high - low + 1) as u32) as i32
        }
        fn unit(&mut self) -> f32 {
            (self.bits() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn between(&mut self, low: f32, high: f32) -> f32 {
            low + (high - low) * self.unit()
        }
    }

    // ---- The shader's math on the CPU, with the shader's constants -------------------

    /// `far_cull.wgsl`'s `origin_rel`.
    fn origin_rel(origin: [i32; 3], cam_int: [i32; 3], cam_frac: Vec3) -> Vec3 {
        let d = |i: usize| origin[i].wrapping_sub(cam_int[i]) as f32;
        Vec3::new(d(0), d(1), d(2)) - cam_frac
    }

    /// `in_frustum`, its padding scaled by `scale` (1: the shader's).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn in_frustum(planes: &[Vec4; 6], lo: Vec3, hi: Vec3, scale: f32) -> bool {
        let size = lo.abs().max(hi.abs()).max_element();
        let pad = (FRUSTUM_PAD + FRUSTUM_REL * size) * scale;
        let (mn, mx) = (lo - pad, hi + pad);
        planes.iter().all(|p| {
            let far = Vec3::select(p.truncate().cmpge(Vec3::ZERO), mx, mn);
            !(p.truncate().dot(far) + p.w < 0.0)
        })
    }

    /// `cull_one`'s runs: the record's runs (first quad, quads) for an eye `d` less its
    /// origin; without an eye, all its solid quads.
    fn runs(record: &Record, d: Option<Vec3>) -> Vec<(u32, u32)> {
        let Some(d) = d else {
            return if record.a[1] > 0 {
                vec![(0, record.a[1])]
            } else {
                vec![]
            };
        };
        let eye = [order_key(d.x), order_key(d.y), order_key(d.z)];
        let seen = |b: usize| {
            if b == 2 {
                return true;
            }
            let t = record.keys[b - usize::from(b > 2)];
            match b {
                0 => eye[0] > t,
                1 => eye[2] > t,
                3 => eye[1] > t,
                4 => eye[2] < t,
                5 => eye[0] < t,
                _ => eye[1] < t,
            }
        };
        let (mut runs, mut start, mut first, mut open) = (Vec::new(), 0, 0, false);
        for b in 0..7 {
            let end = if b == 6 { record.a[1] } else { record.ends[b] };
            if end == start {
                continue;
            }
            let shown = seen(b);
            if shown && !open {
                (open, first) = (true, start);
            } else if !shown && open {
                if runs.len() < RUNS as usize {
                    runs.push((first, start - first));
                }
                open = false;
            }
            start = end;
        }
        if open && runs.len() < RUNS as usize {
            runs.push((first, start - first));
        }
        runs
    }

    /// `box_seen`: whether the box of `record` at camera-relative `o`, grown by `pad`, may be
    /// seen over the reversed-Z min pyramid.
    fn box_seen(
        view_proj: &Mat4,
        o: Vec3,
        record: &Record,
        pad: f32,
        pyramid: &[(Vec<f32>, [usize; 2])],
    ) -> bool {
        let mn = o + Vec3::from_slice(&record.min[..3]) - pad;
        let mx = o + Vec3::from_slice(&record.max[..3]) + pad;
        box_seen_in(view_proj, mn, mx, pyramid)
    }

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn box_seen_in(
        view_proj: &Mat4,
        mn: Vec3,
        mx: Vec3,
        pyramid: &[(Vec<f32>, [usize; 2])],
    ) -> bool {
        let (mut lo, mut hi, mut max_z) =
            (glam::Vec2::splat(1e30), glam::Vec2::splat(-1e30), -1e30f32);
        for c in 0..8u32 {
            let corner = Vec3::new(
                if c & 1 != 0 { mx.x } else { mn.x },
                if c & 2 != 0 { mx.y } else { mn.y },
                if c & 4 != 0 { mx.z } else { mn.z },
            );
            let p = *view_proj * corner.extend(1.0);
            if !(p.w > 1e-6 && p.z < p.w) {
                return true;
            }
            let inv_w = 1.0 / p.w;
            let ndc = glam::Vec2::new(p.x, p.y) * inv_w;
            lo = lo.min(ndc);
            hi = hi.max(ndc);
            max_z = max_z.max(p.z * inv_w);
        }
        if hi.x < -1.0 || lo.x > 1.0 || hi.y < -1.0 || lo.y > 1.0 || max_z < 0.0 {
            return false;
        }
        let (ndc_lo, ndc_hi) = (
            lo.max(glam::Vec2::splat(-1.0)),
            hi.min(glam::Vec2::splat(1.0)),
        );
        let uv_min = [ndc_lo.x * 0.5 + 0.5, 0.5 - ndc_hi.y * 0.5];
        let uv_max = [ndc_hi.x * 0.5 + 0.5, 0.5 - ndc_lo.y * 0.5];
        let occluder_min_z = cpu_query_hiz(pyramid, uv_min, uv_max);
        !(max_z < occluder_min_z)
    }

    /// A frame's inputs, split as `prepare` does.
    struct Frame {
        cam_int: [i32; 3],
        cam_frac: Vec3,
        eye: Option<Vec3>,
        planes: [Vec4; 6],
        view_proj: Mat4,
        occlude: bool,
    }

    impl Frame {
        fn new(input: &FarInputs) -> Self {
            let (cam_int, cam_frac) = input.split();
            Self {
                cam_int,
                cam_frac,
                eye: input.eye,
                planes: input.planes,
                view_proj: input.view_proj,
                occlude: input.occlude,
            }
        }
    }

    /// What `cull` finds of an id the frustum keeps.
    #[derive(Clone, Debug, PartialEq)]
    struct Culled {
        id: u32,
        origin: Vec3,
        runs: Vec<(u32, u32)>,
        /// Whether the frustum decision could go either way on a GPU (its padding changed
        /// by 5% decides otherwise).
        unstable: bool,
    }

    /// `cull` for every id: the kept ones, in id order.
    fn cull(records: &[Record], pages: u32, f: &Frame) -> Vec<Culled> {
        let mut out = Vec::new();
        for (id, r) in records.iter().enumerate() {
            if !r.is_active() || r.page() >= pages {
                continue;
            }
            let o = origin_rel(r.origin(), f.cam_int, f.cam_frac);
            let (lo, hi) = (
                o + Vec3::from_slice(&r.min[..3]),
                o + Vec3::from_slice(&r.max[..3]),
            );
            let kept = in_frustum(&f.planes, lo, hi, 1.0);
            if kept != in_frustum(&f.planes, lo, hi, 0.95)
                || kept != in_frustum(&f.planes, lo, hi, 1.05)
            {
                out.push(Culled {
                    id: id as u32,
                    origin: o,
                    runs: runs(r, f.eye.map(|e| e - o)),
                    unstable: true,
                });
                continue;
            }
            if kept {
                out.push(Culled {
                    id: id as u32,
                    origin: o,
                    runs: runs(r, f.eye.map(|e| e - o)),
                    unstable: false,
                });
            }
        }
        out
    }

    /// `count`, `scan`, `scatter`: the draws of each page for the culled ids `drawn` picks,
    /// in id order, then run order, clamped to the page's region.
    fn compact(
        records: &[Record],
        culled: &[Culled],
        page_records: &[u32],
        pages: u32,
        drawn: impl Fn(&Culled) -> bool,
    ) -> Vec<Vec<[u32; 5]>> {
        let mut out = vec![Vec::new(); pages as usize];
        for c in culled.iter().filter(|c| drawn(c)) {
            let r = &records[c.id as usize];
            let page = r.page() as usize;
            for &(first, quads) in &c.runs {
                out[page].push([quads * 6, 1, 0, r.a[0] + first * 4, c.id]);
            }
        }
        for (page, draws) in out.iter_mut().enumerate() {
            let size = RUNS as usize * page_records.get(page).copied().unwrap_or(0) as usize;
            draws.truncate(size);
        }
        out
    }

    // ---- Random inputs --------------------------------------------------------------

    fn part_ends(rng: &mut Rng, quads: u32) -> [u32; 14] {
        let mut cuts: Vec<u32> = (0..13).map(|_| rng.below(quads + 1)).collect();
        cuts.sort_unstable();
        let (mut ends, mut end) = ([quads; 14], 0);
        for (part, cut) in ends.iter_mut().zip(cuts) {
            if rng.below(3) > 0 {
                end = cut;
            }
            *part = end;
        }
        ends[13] = quads;
        ends
    }

    /// A mesh sorted for one pass with `quads` solid quads, each bucket's plane on the
    /// 1/16 grid, in its tile or a block out; now and then sorted for two passes.
    fn sides_of(rng: &mut Rng, quads: u32) -> Sides {
        let ends = part_ends(rng, quads);
        let planes = std::array::from_fn(|_| rng.range(-16, 272) as f32 / 16.0);
        let grouped = rng.below(6) == 0;
        Sides::from_parts(ends, grouped, rng.below(4) > 0, planes)
    }

    fn view(rng: &mut Rng) -> (Frustum, Mat4) {
        let pose = Mat4::from_rotation_x(rng.between(-1.4, 1.4))
            * Mat4::from_rotation_y(rng.between(-3.2, 3.2));
        let proj = world_projection(
            rng.between(30.0, 110.0),
            rng.between(0.8, 2.2),
            rng.between(300.0, 4000.0),
        );
        let vp = proj * pose;
        (Frustum::from_view_projection(vp), vp)
    }

    fn bounds(rng: &mut Rng, size: f32) -> [Vec3; 2] {
        let lo = Vec3::new(
            rng.between(-8.0, 8.0),
            rng.between(-8.0, 8.0),
            rng.between(-8.0, 8.0),
        );
        [
            lo,
            lo + Vec3::new(
                rng.between(0.5, size),
                rng.between(0.5, size),
                rng.between(0.5, size),
            ),
        ]
    }

    // ---- (1) the frustum test -------------------------------------------------------

    /// The shader's frustum test keeps every box `Frustum::intersects` keeps, from cameras
    /// near the origin and 20 million blocks out, with a thin shell more.
    #[test]
    fn the_frustum_test_keeps_what_the_cpu_keeps() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let (mut total, mut cpu_kept, mut gpu_kept) = (0, 0, 0);
        for case in 0..3000 {
            let (frustum, _) = view(&mut rng);
            let camera = match case % 3 {
                0 => DVec3::new(0.3, 70.5, -0.7),
                1 => DVec3::new(
                    f64::from(rng.range(-30_000, 30_000)) + f64::from(rng.unit()),
                    f64::from(rng.range(-64, 300)) + f64::from(rng.unit()),
                    f64::from(rng.range(-30_000, 30_000)) + f64::from(rng.unit()),
                ),
                _ => DVec3::new(
                    20_000_000.0 + f64::from(rng.unit()),
                    f64::from(rng.range(-64, 300)) + f64::from(rng.unit()),
                    -19_999_999.0 + f64::from(rng.unit()),
                ),
            };
            let input = FarInputs {
                camera,
                planes: frustum.planes,
                eye: None,
                view_proj: Mat4::IDENTITY,
                occlude: false,
            };
            let f = Frame::new(&input);
            for _ in 0..8 {
                let reach = [50, 600, 5000][rng.below(3) as usize];
                let origin = [
                    camera.x.floor() as i32 + rng.range(-reach, reach),
                    camera.y.floor() as i32 + rng.range(-reach, reach),
                    camera.z.floor() as i32 + rng.range(-reach, reach),
                ];
                let size = [16.0, 400.0, 3000.0][rng.below(3) as usize];
                let [min, max] = bounds(&mut rng, size);
                // `lod::Bounds::visible`.
                let o = DVec3::from(origin.map(f64::from)) - camera;
                let cpu = frustum.intersects(
                    (o + min.as_dvec3()).as_vec3(),
                    (o + max.as_dvec3()).as_vec3(),
                );
                let rel = origin_rel(origin, f.cam_int, f.cam_frac);
                let gpu = in_frustum(&f.planes, rel + min, rel + max, 1.0);
                total += 1;
                cpu_kept += usize::from(cpu);
                gpu_kept += usize::from(gpu);
                assert!(
                    !cpu || gpu,
                    "case {case}: the CPU keeps {origin:?} {min} {max}"
                );
            }
        }
        assert!(
            cpu_kept > total / 20 && cpu_kept < total * 19 / 20,
            "{cpu_kept}/{total}"
        );
        assert!(
            gpu_kept - cpu_kept < total / 50,
            "a thin shell: {cpu_kept} -> {gpu_kept}"
        );
    }

    // ---- (2) the Hi-Z test ----------------------------------------------------------

    /// A pyramid of a `w` x `h` depth image of `depth(u, v)`.
    fn pyramid(w: usize, h: usize, depth: impl Fn(f32, f32) -> f32) -> Vec<(Vec<f32>, [usize; 2])> {
        let image: Vec<f32> = (0..w * h)
            .map(|i| {
                depth(
                    ((i % w) as f32 + 0.5) / w as f32,
                    ((i / w) as f32 + 0.5) / h as f32,
                )
            })
            .collect();
        cpu_build_hiz(&image, [w, h])
    }

    /// The reversed-Z depth of a point `distance` ahead of the camera.
    fn depth_at(view_proj: &Mat4, distance: f32) -> f32 {
        let p = *view_proj * Vec4::new(0.0, 0.0, -distance, 1.0);
        p.z / p.w
    }

    /// The world's lens without a far plane, 20 million blocks out (looking west, toward -x):
    /// the level-22 node (67,108,864 blocks wide) holding the camera, and the one ahead of it
    /// whose near side is 2e7 blocks away, pass the frustum and, over the sky (depth 0), the
    /// Hi-Z test; a ridge 1e7 blocks out hides the one ahead.
    #[test]
    fn a_level_22_node_stays_in_view_of_the_infinite_lens() {
        let vp = world_projection(70.0, 16.0 / 9.0, crate::gfx::WORLD_FAR)
            * Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2);
        let input = FarInputs {
            camera: DVec3::new(20_000_000.3, 2_000.5, -19_999_999.7),
            planes: Frustum::from_view_projection(vp).planes,
            eye: None,
            view_proj: vp,
            occlude: true,
        };
        let f = Frame::new(&input);
        assert!(f.planes.iter().all(|p| p.is_finite()));
        let size = 16 << crate::lod::cube::MAX_CUBE_LEVEL;
        assert_eq!(size, 67_108_864);
        let sky = pyramid(128, 72, |_, _| 0.0);
        let ridge = {
            let ahead = vp * Vec4::new(-1e7, 0.0, 0.0, 1.0);
            pyramid(128, 72, move |_, _| ahead.z / ahead.w)
        };
        assert!(ridge[0].0[0] > 0.0, "terrain at 1e7 blocks is not the sky");
        // Node-local mesh bounds: its whole width, the ground from 0 to 300.
        let (min, max) = (Vec3::ZERO, Vec3::new(size as f32, 300.0, size as f32));
        for (x, ahead) in [(0, false), (-1, true)] {
            let key = CubeKey::new(crate::lod::cube::MAX_CUBE_LEVEL, x, 0, -1);
            let o = origin_rel(key.origin(), f.cam_int, f.cam_frac);
            assert!(o.is_finite());
            let (lo, hi) = (o + min, o + max);
            assert!(
                in_frustum(&f.planes, lo, hi, 1.0),
                "{key:?}: in the frustum"
            );
            assert!(box_seen_in(&vp, lo, hi, &sky), "{key:?}: seen over the sky");
            assert_eq!(
                !box_seen_in(&vp, lo, hi, &ridge),
                ahead,
                "{key:?} behind the ridge"
            );
        }
    }

    #[test]
    fn the_hiz_test_is_conservative() {
        let vp = world_projection(70.0, 16.0 / 9.0, 1000.0);
        let b = |center: [f32; 3], half: f32| {
            let c = Vec3::from(center);
            (c - half, c + half)
        };
        // A wall 20 blocks ahead over the whole screen.
        let wall = pyramid(128, 72, |_, _| depth_at(&vp, 20.0));
        let seen =
            |p: &[(Vec<f32>, [usize; 2])], (lo, hi): (Vec3, Vec3)| box_seen_in(&vp, lo, hi, p);
        assert!(
            !seen(&wall, b([0.0, 0.0, -60.0], 3.0)),
            "behind the wall: culled"
        );
        assert!(
            !seen(&wall, b([5.0, 2.0, -200.0], 20.0)),
            "behind the wall, larger: culled"
        );
        assert!(
            seen(&wall, b([0.0, 0.0, -10.0], 2.0)),
            "in front of the wall: seen"
        );
        // A box through the near plane, or around the camera, is seen over any wall.
        let near = depth_at(&vp, 2.0);
        let close = pyramid(128, 72, |_, _| near);
        assert!(
            seen(
                &close,
                (Vec3::new(-1.0, -1.0, -5.0), Vec3::new(1.0, 1.0, 1.0))
            ),
            "crossing the near plane"
        );
        assert!(
            seen(
                &close,
                (Vec3::new(-1.0, -1.0, -5.0), Vec3::new(1.0, 1.0, 0.0))
            ),
            "touching it"
        );
        assert!(seen(&close, b([0.0, 0.0, 0.0], 5.0)), "around the camera");
        assert!(
            seen(
                &close,
                (Vec3::new(-1.0, -1.0, 0.5), Vec3::new(1.0, 1.0, 3.0))
            ),
            "behind the camera"
        );
        // Off screen: not seen.
        assert!(
            !seen(&wall, b([1000.0, 0.0, -10.0], 5.0)),
            "off screen right"
        );
        assert!(
            !seen(&wall, b([0.0, -900.0, -10.0], 5.0)),
            "off screen below"
        );
        // A wall over the left half of the screen only: a box across the middle is partly
        // visible, one in the left half is hidden, one in the right half is seen.
        let half = pyramid(
            128,
            72,
            |u, _| if u < 0.5 { depth_at(&vp, 20.0) } else { 0.0 },
        );
        assert!(seen(&half, b([0.0, 0.0, -60.0], 6.0)), "partly visible");
        assert!(
            !seen(&half, b([-30.0, 0.0, -60.0], 3.0)),
            "behind the wall's part"
        );
        assert!(seen(&half, b([30.0, 0.0, -60.0], 3.0)), "over the sky");
        // A one-texel hole is found by any box over it (a hole in the wall, a window).
        for (px, py) in [(10usize, 7usize), (64, 36), (127, 71), (0, 0)] {
            let hole = pyramid(128, 72, |u, v| {
                let (x, y) = ((u * 128.0) as usize, (v * 72.0) as usize);
                if (x, y) == (px, py) {
                    0.0
                } else {
                    depth_at(&vp, 20.0)
                }
            });
            let u = (px as f32 + 0.5) / 128.0 * 2.0 - 1.0;
            let v = 1.0 - (py as f32 + 0.5) / 72.0 * 2.0;
            // The pixel's ray at 100 blocks, in a box ten blocks wide.
            let t = (70.0f32.to_radians() / 2.0).tan();
            let at = Vec3::new(u * t * 16.0 / 9.0 * 100.0, v * t * 100.0, -100.0);
            assert!(seen(&hole, (at - 5.0, at + 5.0)), "the hole at {px},{py}");
        }
    }

    // ---- (3) FarRecords -------------------------------------------------------------

    fn key(i: u32) -> CubeKey {
        CubeKey {
            level: 0,
            x: i as i32,
            y: 1,
            z: -(i as i32),
        }
    }

    fn tile(
        solid: u32,
        page: u32,
        start: u64,
        water: bool,
        layers: &[u16],
        shown: Option<Instant>,
        dying: Option<Instant>,
    ) -> Tile {
        let mut s = crate::gfx::tests::section(solid, 0, None);
        s.page = page;
        s.start = start;
        s.sorted_indices = water.then_some((0, 0));
        s.animated_layers = layers.into();
        Tile::for_test(s, [Vec3::ZERO, Vec3::splat(16.0)], shown, dying)
    }

    fn whole(now: Instant) -> Option<Instant> {
        Some(now - Duration::from_secs(2))
    }

    fn ids_and_updates(far: &mut FarRecords) -> (Vec<u32>, bool) {
        let mut updates = Vec::new();
        let whole = far.take_updates(&mut updates);
        (updates.iter().map(|u| u.id[0]).collect(), whole)
    }

    #[test]
    fn records_follow_a_tile_through_its_life() {
        let now = Instant::now() + Duration::from_secs(10);
        let ms = Duration::from_millis;
        let mut tiles: FxHashMap<CubeKey, Tile> = FxHashMap::default();
        let mut far = FarRecords::default();
        let mut mask = [0u64; 2];
        // Hidden: no record, in no set.
        tiles.insert(key(1), tile(5, 0, 0, false, &[], None, None));
        far.touch(key(1));
        far.sync(&tiles, now, None);
        far.check(&tiles, now);
        assert_eq!((far.records(), far.cpu_len(), far.water_len()), (0, 0, 0));
        // Fading in: the CPU's, whole.
        let fading = Some(now - ms(100));
        tiles
            .get_mut(&key(1))
            .unwrap()
            .set_times_for_test(fading, None);
        far.touch(key(1));
        far.sync(&tiles, now, None);
        far.check(&tiles, now);
        assert_eq!((far.records(), far.cpu_len()), (0, 1));
        assert_eq!(far.cpu_keys().collect::<Vec<_>>(), [(key(1), true)]);
        // Whole after the fade, with no touch: the record appears, the CPU lets go.
        let later = now + ms(700);
        far.sync(&tiles, later, None);
        far.check(&tiles, later);
        assert_eq!((far.records(), far.cpu_len()), (1, 0));
        assert_eq!(far.id_of(key(1)), Some(0));
        assert_eq!(far.record(0).map(Record::quads), Some(5));
        // Dying: the CPU draws it whole again.
        tiles
            .get_mut(&key(1))
            .unwrap()
            .set_times_for_test(whole(later), Some(later));
        far.touch(key(1));
        far.sync(&tiles, later, None);
        far.check(&tiles, later);
        assert_eq!((far.records(), far.cpu_len()), (0, 1));
        // Gone.
        tiles.remove(&key(1));
        far.touch(key(1));
        far.sync(&tiles, later, None);
        far.check(&tiles, later);
        assert_eq!((far.records(), far.cpu_len(), far.id_count()), (0, 0, 0));
        // Water: GPU solid, CPU translucent, with animated layers.
        tiles.insert(key(2), tile(3, 1, 8, true, &[3, 70], whole(later), None));
        tiles.insert(key(3), tile(2, 1, 40, false, &[3], whole(later), None));
        far.touch(key(2));
        far.touch(key(3));
        far.sync(&tiles, later, None);
        far.check(&tiles, later);
        assert_eq!(far.cpu_keys().collect::<Vec<_>>(), [(key(2), false)]);
        far.animated_layers(&mut mask);
        assert_eq!(mask, [1 << 3, 1 << 6]);
        assert_eq!(far.page_records(), [0, 2]);
        // Its mesh loses the water and a layer.
        tiles.insert(key(2), tile(3, 1, 8, false, &[], whole(later), None));
        far.touch(key(2));
        far.sync(&tiles, later, None);
        far.check(&tiles, later);
        assert_eq!(far.water_len(), 0);
        let mut mask = [0u64; 2];
        far.animated_layers(&mut mask);
        assert_eq!(mask, [1 << 3, 0]);
    }

    #[test]
    fn ids_are_reused_lowest_first_and_the_table_shrinks() {
        let now = Instant::now() + Duration::from_secs(10);
        let mut tiles: FxHashMap<CubeKey, Tile> = FxHashMap::default();
        let mut far = FarRecords::default();
        for i in 0..6 {
            tiles.insert(
                key(i),
                tile(1 + i, 0, u64::from(i) * 16, false, &[], whole(now), None),
            );
            far.touch(key(i));
        }
        far.sync(&tiles, now, None);
        assert_eq!(
            (0..6)
                .map(|i| far.id_of(key(i)).unwrap())
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5]
        );
        for i in [1, 3, 5] {
            tiles.remove(&key(i));
            far.touch(key(i));
        }
        far.sync(&tiles, now, None);
        far.check(&tiles, now);
        // The end went: id 5 was last, then 4 stays.
        assert_eq!(far.id_count(), 5);
        tiles.insert(key(10), tile(2, 0, 0, false, &[], whole(now), None));
        tiles.insert(key(11), tile(2, 0, 0, false, &[], whole(now), None));
        far.touch(key(10));
        far.touch(key(11));
        far.sync(&tiles, now, None);
        far.check(&tiles, now);
        assert_eq!(far.id_of(key(10)), Some(1));
        assert_eq!(far.id_of(key(11)), Some(3));
    }

    #[test]
    fn updates_are_listed_once_and_only_when_a_record_changed() {
        let now = Instant::now() + Duration::from_secs(10);
        let mut tiles: FxHashMap<CubeKey, Tile> = FxHashMap::default();
        let mut far = FarRecords::default();
        for i in 0..4 {
            tiles.insert(
                key(i),
                tile(2, 0, u64::from(i) * 16, false, &[], whole(now), None),
            );
            far.touch(key(i));
            far.touch(key(i));
        }
        far.sync(&tiles, now, None);
        assert_eq!(ids_and_updates(&mut far), (vec![0, 1, 2, 3], false));
        // Nothing changed: touched twice, no update.
        far.touch(key(2));
        far.touch(key(2));
        far.sync(&tiles, now, None);
        assert_eq!(ids_and_updates(&mut far), (vec![], false));
        // Changed twice before the frame: once. Removed and then back: still once.
        tiles.insert(key(2), tile(3, 0, 16, false, &[], whole(now), None));
        far.touch(key(2));
        far.sync(&tiles, now, None);
        tiles.insert(key(2), tile(4, 0, 16, false, &[], whole(now), None));
        far.touch(key(2));
        far.sync(&tiles, now, None);
        tiles.remove(&key(0));
        far.touch(key(0));
        far.sync(&tiles, now, None);
        assert_eq!(ids_and_updates(&mut far), (vec![0, 2], false));
        // A freed id at the end leaves with its update.
        tiles.remove(&key(3));
        far.touch(key(3));
        far.sync(&tiles, now, None);
        assert_eq!(ids_and_updates(&mut far), (vec![], false));
        assert_eq!(far.id_count(), 3);
        far.forget();
        assert_eq!(ids_and_updates(&mut far), (vec![], true));
        far.clear();
        assert_eq!((far.records(), far.id_count()), (0, 0));
    }

    #[test]
    fn a_page_closing_moves_its_records_without_a_touch() {
        let now = Instant::now() + Duration::from_secs(10);
        let mut tiles: FxHashMap<CubeKey, Tile> = FxHashMap::default();
        let mut far = FarRecords::default();
        for i in 0..6 {
            tiles.insert(
                key(i),
                tile(
                    2,
                    1 + i % 2,
                    u64::from(i) * 16,
                    false,
                    &[],
                    whole(now),
                    None,
                ),
            );
            far.touch(key(i));
        }
        far.sync(&tiles, now, None);
        assert_eq!(far.page_records(), [0, 3, 3]);
        ids_and_updates(&mut far);
        // The arena compacts page 2: its meshes go to page 0 and a new start.
        for (&k, t) in tiles.iter_mut() {
            if t.mesh.page == 2 {
                t.mesh.page = 0;
                t.mesh.start += 1000;
                let _ = k;
            }
        }
        // Without the closing page the records are stale; with it they follow.
        far.sync(&tiles, now, None);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| far.check(&tiles, now)))
                .is_err()
        );
        far.sync(&tiles, now, Some(2));
        far.check(&tiles, now);
        assert_eq!(far.page_records(), [3, 3]);
        let (ids, whole) = ids_and_updates(&mut far);
        assert!(!whole && ids.len() == 3, "{ids:?}");
    }

    /// The renderer touches every key on the page it compacts (`touch_page`), so the next
    /// plain `sync` reads their moved meshes, whether or not the page is still closing.
    #[test]
    fn touching_a_page_makes_sync_read_its_moved_tiles() {
        let now = Instant::now() + Duration::from_secs(10);
        let mut tiles: FxHashMap<CubeKey, Tile> = FxHashMap::default();
        let mut far = FarRecords::default();
        for i in 0..6 {
            tiles.insert(
                key(i),
                tile(
                    2,
                    1 + i % 2,
                    u64::from(i) * 16,
                    false,
                    &[],
                    whole(now),
                    None,
                ),
            );
            far.touch(key(i));
        }
        far.sync(&tiles, now, None);
        ids_and_updates(&mut far);
        for t in tiles.values_mut() {
            if t.mesh.page == 2 {
                t.mesh.page = 0;
                t.mesh.start += 1000;
            }
        }
        far.touch_page(2);
        far.touch_page(7); // a page without records
        far.sync(&tiles, now, None);
        far.check(&tiles, now);
        assert_eq!(far.page_records(), [3, 3]);
        let (ids, whole) = ids_and_updates(&mut far);
        assert!(!whole && ids.len() == 3, "{ids:?}");
    }

    /// Records of tiles that are not whole are never drawn: the mirror of `compact` over
    /// records in every state lists draws of Gpu tiles alone, a page's in id order.
    #[test]
    fn a_pages_draws_come_in_id_order_and_only_for_gpu_tiles() {
        let now = Instant::now() + Duration::from_secs(10);
        let mut tiles: FxHashMap<CubeKey, Tile> = FxHashMap::default();
        let mut far = FarRecords::default();
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for i in 0..60 {
            let shown = [
                None,
                Some(now - Duration::from_millis(100)),
                whole(now),
                whole(now),
            ][rng.below(4) as usize];
            let dying = (rng.below(5) == 0).then_some(now);
            tiles.insert(
                key(i),
                tile(
                    rng.below(12),
                    rng.below(3),
                    u64::from(i) * 64,
                    rng.below(2) == 0,
                    &[],
                    shown,
                    dying,
                ),
            );
            far.touch(key(i));
        }
        far.sync(&tiles, now, None);
        far.check(&tiles, now);
        let all = Frame {
            cam_int: [0; 3],
            cam_frac: Vec3::ZERO,
            eye: None,
            planes: [Vec4::new(0.0, 0.0, 0.0, 1.0); 6],
            view_proj: Mat4::IDENTITY,
            occlude: false,
        };
        let pages = far.page_records().len() as u32;
        let culled = cull(&far.records, pages, &all);
        let draws = compact(&far.records, &culled, far.page_records(), pages, |_| true);
        let gpu = tiles
            .iter()
            .filter(|(k, t)| t.far_class(k.level, now) == FarClass::Gpu)
            .count();
        assert_eq!(culled.len(), gpu);
        for (page, list) in draws.iter().enumerate() {
            assert!(
                list.windows(2).all(|w| w[0][4] <= w[1][4]),
                "page {page}: id order"
            );
            for d in list {
                let k = far.keys[d[4] as usize].expect("a record");
                let t = &tiles[&k];
                assert_eq!(t.far_class(k.level, now), FarClass::Gpu, "{k:?}");
                assert_eq!(t.mesh.page as usize, page);
                // Within the record's own quads.
                let r = far.record(d[4]).unwrap();
                assert!(d[3] >= r.a[0] && d[3] + d[0] / 6 * 4 <= r.a[0] + r.a[1] * 4);
            }
        }
        let (mut words, mut list) = (Vec::new(), Vec::new());
        let total = regions(far.page_records(), pages, &mut words, &mut list);
        assert_eq!(total, u64::from(RUNS) * far.records() as u64);
        assert!(
            list.iter()
                .all(|&(p, first, size)| words[2 * p as usize] == first
                    && words[2 * p as usize + 1] == size)
        );
    }

    /// After random installs, removals, fades, dyings, moves and time, the invariant holds.
    #[test]
    fn the_invariant_holds_after_random_operations() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let ms = Duration::from_millis;
        let mut now = Instant::now() + Duration::from_secs(100);
        let mut tiles: FxHashMap<CubeKey, Tile> = FxHashMap::default();
        let mut far = FarRecords::default();
        let mut moves = 0;
        for step in 0..5000 {
            let mut closing = None;
            match rng.below(9) {
                0..=2 => {
                    let k = key(rng.below(40));
                    let kind = rng.below(5);
                    let shown = match kind {
                        0 => None,
                        1 => Some(now - ms(rng.below(700) as u64)),
                        _ => whole(now),
                    };
                    let dying = (kind == 4).then(|| now - ms(5));
                    let layers: Vec<u16> =
                        (0..rng.below(3)).map(|_| rng.below(80) as u16).collect();
                    let t = tile(
                        rng.below(20),
                        rng.below(4),
                        u64::from(rng.below(100)) * 4,
                        rng.below(3) == 0,
                        &layers,
                        shown,
                        dying,
                    );
                    tiles.insert(k, t);
                    far.touch(k);
                }
                3 => {
                    let k = key(rng.below(40));
                    tiles.remove(&k);
                    far.touch(k);
                }
                4 => now += ms(u64::from(rng.below(900))),
                5 => {
                    let k = key(rng.below(40));
                    if let Some(t) = tiles.get_mut(&k) {
                        let (shown, dying) = if rng.below(2) == 0 {
                            (whole(now), Some(now))
                        } else {
                            (None, None)
                        };
                        t.set_times_for_test(shown, dying);
                        far.touch(k);
                    }
                }
                6 => {
                    let p = rng.below(4);
                    let to = (p + 1 + rng.below(3)) % 4;
                    for t in tiles.values_mut().filter(|t| t.mesh.page == p) {
                        t.mesh.page = to;
                        t.mesh.start += 20;
                        moves += 1;
                    }
                    closing = Some(p);
                }
                7 => far.touch(key(rng.below(40))),
                _ => {}
            }
            far.sync(&tiles, now, closing);
            far.check(&tiles, now);
            if step % 500 == 0 {
                far.forget();
                let (_, whole) = ids_and_updates(&mut far);
                assert!(whole);
            }
        }
        assert!(moves > 50);
        far.clear();
        assert_eq!(
            (
                far.records(),
                far.id_count(),
                far.cpu_len(),
                far.water_len()
            ),
            (0, 0, 0, 0)
        );
        far.check(&FxHashMap::default(), now);
    }

    // ---- (4) facing runs ------------------------------------------------------------

    /// A record's runs, from its ends and keys and the eye, are `Sides::visible_from`'s.
    #[test]
    fn the_runs_are_the_facing_runs() {
        let mut rng = Rng(0x3C6E_F372_FE94_F82B);
        let (mut hidden, mut several, mut grouped) = (0, 0, 0);
        for case in 0..6000 {
            let quads = rng.below(40);
            let sides = sides_of(&mut rng, quads);
            let record =
                Record::new([0; 3], 0, 100, quads, [Vec3::ZERO, Vec3::splat(16.0)]).facing(&sides);
            grouped += usize::from(sides.thresholds().is_none());
            // An eye (less the origin) anywhere about the tile, on the 1/16 grid now and
            // then so that planes (margin included) are met exactly.
            let eye = Vec3::from(std::array::from_fn(|_| {
                if rng.below(3) == 0 {
                    rng.range(-32, 300) as f32 / 16.0
                } else {
                    rng.between(-40.0, 60.0)
                }
            }));
            let mut expected = Vec::new();
            sides.visible_from(eye.to_array(), eye.y, Part::All, |first, quads| {
                expected.push((first, quads))
            });
            let got = runs(&record, Some(eye));
            if sides.thresholds().is_none() {
                // Sorted for two passes: drawn whole, a superset of what is seen.
                assert_eq!(got, if quads > 0 { vec![(0, quads)] } else { vec![] });
                continue;
            }
            assert_eq!(got, expected, "case {case}: {sides:?} eye {eye}");
            hidden += usize::from(got.iter().map(|r| r.1).sum::<u32>() < quads);
            several += usize::from(got.len() >= 3);
            let mut all = Vec::new();
            sides.every(Part::All, |first, quads| all.push((first, quads)));
            assert_eq!(runs(&record, None), all, "case {case}: without an eye");
        }
        assert!(
            hidden > 500 && several > 50 && grouped > 500,
            "{hidden} {several} {grouped}"
        );
    }

    // ---- the shader's source ----------------------------------------------------------

    fn wgsl_const(name: &str) -> String {
        let source = include_str!("far_cull.wgsl");
        let line = source
            .lines()
            .find(|l| l.starts_with(&format!("const {name}:")))
            .unwrap_or_else(|| panic!("const {name}"));
        let value = line.split('=').nth(1).unwrap().trim().trim_end_matches(';');
        value.trim_end_matches('u').to_owned()
    }

    #[test]
    fn the_shader_has_the_constants_and_layouts_of_this_file() {
        let int = |name: &str| wgsl_const(name).parse::<u64>().unwrap();
        let float = |name: &str| wgsl_const(name).parse::<f32>().unwrap();
        assert_eq!(float("FRUSTUM_PAD"), FRUSTUM_PAD);
        assert_eq!(float("FRUSTUM_REL"), FRUSTUM_REL);
        assert_eq!(float("TEST_PAD"), TEST_PAD);
        assert_eq!(int("RUNS"), u64::from(RUNS));
        assert_eq!(int("RUN_WORDS"), RUN_WORDS);
        assert_eq!(int("STATS"), u64::from(STATS));
        assert_eq!(int("OCCLUDED") as usize, OCCLUDED);
        assert_eq!(int("DRAWS") as usize, DRAWS);
        assert_eq!(int("TRIANGLES") as usize, TRIANGLES);
        assert_eq!(int("TESTED") as usize, TESTED);
        assert_eq!(int("BLOCK"), u64::from(WORKGROUP));
        assert_eq!(int("ROW"), u64::from(ROW));
        assert_eq!(int("EYE"), u64::from(EYE));
        assert_eq!(int("OCCLUDE"), u64::from(OCCLUDE));
        assert_eq!(int("ACTIVE"), u64::from(ACTIVE));
        assert_eq!(UV_STEP, 1.0 / 32.0);
        assert_eq!(std::mem::size_of::<Record>(), 112);
        assert_eq!(std::mem::size_of::<Update>(), 128);
        assert_eq!(std::mem::size_of::<Params>(), 224);
    }

    #[test]
    fn the_shader_validates() {
        let _gpu = crate::gfx::test_gpu::lock();
        use naga::valid::Capabilities as C;
        let source = include_str!("far_cull.wgsl");
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            C::all() - C::EARLY_DEPTH_TEST,
        )
        .validate(&module)
        .unwrap();
        for entry in [
            "apply", "cull", "count", "scan", "scatter", "count2", "scan2", "scatter2", "test",
        ] {
            let (msl, _) = naga::back::msl::write_string(
                &module,
                &info,
                &naga::back::msl::Options {
                    fake_missing_bindings: true,
                    ..Default::default()
                },
                &naga::back::msl::PipelineOptions {
                    entry_point: Some((naga::ShaderStage::Compute, entry.to_owned())),
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("{entry} MSL: {e}"));
            assert!(!msl.is_empty());
        }
    }

    // ---- (5) the GPU ------------------------------------------------------------------

    fn device(adapter: &wgpu::Adapter, name: &str) -> Option<(wgpu::Device, wgpu::Queue)> {
        let needed =
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | wgpu::Features::INDIRECT_FIRST_INSTANCE;
        if !adapter.features().contains(needed)
            || crate::gfx::flat::is_broken_intel_vulkan(&adapter.get_info())
        {
            return None;
        }
        eprintln!("far culling on {} ({name})", adapter.get_info().name);
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: needed,
            ..Default::default()
        }))
        .ok()
    }

    /// `test` on each device here that draws counts with first instances.
    fn on_each_device(test: impl Fn(&wgpu::Device, &wgpu::Queue)) {
        let mut tested = false;
        for (adapter, name) in crate::gfx::test_gpu::each_adapter() {
            if let Some((device, queue)) = device(&adapter, name) {
                test(&device, &queue);
                tested = true;
            }
        }
        if !tested {
            eprintln!("no GPU with indirect count draws: skipped");
        }
    }

    fn read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        bytes: u64,
    ) -> Vec<u32> {
        let (tx, rx) = std::sync::mpsc::channel();
        wgpu::util::DownloadBuffer::read_buffer(
            device,
            queue,
            &buffer.slice(..bytes),
            move |result| tx.send(result.unwrap().to_vec()).unwrap(),
        );
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        bytemuck::pod_collect_to_vec(&rx.recv().unwrap())
    }

    /// What a frame left in the buffers.
    struct Back {
        counts: Vec<u32>,
        draws: Vec<u32>,
        instances: Vec<u32>,
        hidden: Vec<u32>,
    }

    fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, far: &FarCull) -> Back {
        let ids = u64::from(far.ids);
        Back {
            counts: read(
                device,
                queue,
                &far.counts.buffer,
                count_words(far.pass_pages, far.ids) * 4,
            ),
            draws: read(device, queue, &far.draws.buffer, far.draws.bytes),
            instances: read(device, queue, &far.instances.buffer, ids * 16),
            hidden: read(device, queue, &far.hidden.buffer, ids * 4),
        }
    }

    fn submit(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uploads: &mut Uploads,
        encoder: wgpu::CommandEncoder,
    ) {
        let _ = device;
        uploads.finish(&encoder);
        queue.submit([encoder.finish()]);
    }

    /// Records at random about `cam_int`, `n` of them at keys from `first`.
    fn populate(
        far: &mut FarRecords,
        rng: &mut Rng,
        shapes: &mut Rng,
        cam_int: [i32; 3],
        first: u32,
        n: u32,
    ) {
        for i in first..first + n {
            let quads = shapes.below(30);
            let sides = sides_of(shapes, quads);
            let origin = [
                cam_int[0] + rng.range(-300, 300),
                cam_int[1] + rng.range(-100, 100),
                cam_int[2] + rng.range(-300, 300),
            ];
            let record = Record::new(
                origin,
                rng.below(3),
                rng.below(50_000),
                quads,
                bounds(rng, 120.0),
            )
            .facing(&sides);
            far.set(key(i), record, &[]);
        }
    }

    fn input_for(rng: &mut Rng, camera: DVec3, eye: bool, occlude: bool) -> FarInputs {
        let (frustum, view_proj) = view(rng);
        FarInputs {
            camera,
            planes: frustum.planes,
            eye: eye.then(|| {
                Vec3::new(
                    rng.between(-2.0, 2.0),
                    rng.between(-2.0, 2.0),
                    rng.between(-2.0, 2.0),
                )
            }),
            view_proj,
            occlude,
        }
    }

    /// Compares what phase `phase` left against `expected`: each page's count and draws.
    fn check_draws(
        far: &FarCull,
        back: &Back,
        phase: usize,
        expected: &[Vec<[u32; 5]>],
        what: &str,
    ) {
        let pages = far.pass_pages as usize;
        let mut words = Vec::new();
        let mut list = Vec::new();
        regions(
            far.records.page_records(),
            far.pass_pages,
            &mut words,
            &mut list,
        );
        for page in 0..pages {
            let n = back.counts[STATS as usize + phase * pages + page] as usize;
            let first = words[2 * page] as usize;
            let got: Vec<[u32; 5]> = (first..first + n)
                .map(|d| std::array::from_fn(|k| back.draws[d * 5 + k]))
                .collect();
            assert_eq!(got, expected[page], "{what}: page {page}, phase {phase}");
            assert!(n <= words[2 * page + 1] as usize);
        }
    }

    /// apply + cull + compact on records of every shape, from cameras near the origin and 20
    /// million blocks out, with and without an eye, through updates, the table made anew and
    /// written whole: the GPU's draws, instances and statistics are the mirror's.
    #[test]
    fn the_gpu_draws_what_the_mirror_draws() {
        let _gpu = crate::gfx::test_gpu::lock();
        on_each_device(|device, queue| {
            let mut far = FarCull::new(device, None).expect("far culling");
            assert!(!far.occluding());
            let mut uploads = Uploads::new(device);
            let mut rng = Rng(0x2545_F491_4F6C_DD1D);
            let mut shapes = Rng(0x3C6E_F372_FE94_F82B);
            let (mut compared, mut drawn_runs, mut serial) = (0, 0, 0u64);
            for (n, camera) in [
                DVec3::new(10.3, 70.2, -4.9),
                DVec3::new(20_000_000.4, 64.5, -19_999_990.2),
            ]
            .into_iter()
            .enumerate()
            {
                far.records.clear();
                let cam_int = [
                    camera.x.floor() as i32,
                    camera.y.floor() as i32,
                    camera.z.floor() as i32,
                ];
                // 300 records: more than the table's first room.
                populate(&mut far.records, &mut rng, &mut shapes, cam_int, 0, 300);
                for frame in 0..14 {
                    match frame {
                        3 => {
                            // Records come, go and change: updates, ids reused.
                            for i in 0..40 {
                                far.records.remove(key(rng.below(300)));
                                let _ = i;
                            }
                            populate(&mut far.records, &mut rng, &mut shapes, cam_int, 1000, 40);
                            for i in 0..40 {
                                let k = key(rng.below(300));
                                if far.records.id_of(k).is_some() {
                                    populate(
                                        &mut far.records,
                                        &mut rng,
                                        &mut shapes,
                                        cam_int,
                                        k.x as u32 % 300,
                                        1,
                                    );
                                    let _ = i;
                                }
                            }
                        }
                        6 => far.records.forget(),
                        9 => populate(&mut far.records, &mut rng, &mut shapes, cam_int, 2000, 2500),
                        _ => {}
                    }
                    let input = input_for(&mut rng, camera, frame % 3 != 0, false);
                    let mut encoder = device.create_command_encoder(&Default::default());
                    assert!(far.prepare(device, &mut uploads, &mut encoder, &input, None));
                    serial += 1;
                    far.copy_stats(&mut encoder, serial);
                    submit(device, queue, &mut uploads, encoder);
                    far.submitted().expect("a readback free")();
                    let back = read_back(device, queue, &far);
                    let f = Frame::new(&input);
                    let culled = cull(&far.records.records, far.pass_pages, &f);
                    if culled.iter().any(|c| c.unstable) {
                        eprintln!("camera {n} frame {frame}: a box on the frustum's edge; skipped");
                        continue;
                    }
                    let expected = compact(
                        &far.records.records,
                        &culled,
                        far.records.page_records(),
                        far.pass_pages,
                        |_| true,
                    );
                    check_draws(
                        &far,
                        &back,
                        0,
                        &expected,
                        &format!("camera {n} frame {frame}"),
                    );
                    assert_eq!(back.counts[0] as usize, culled.len(), "kept records");
                    let draws: usize = expected.iter().map(Vec::len).sum();
                    assert_eq!(back.counts[DRAWS] as usize, draws);
                    let triangles: u32 = culled.iter().flat_map(|c| &c.runs).map(|r| r.1 * 2).sum();
                    assert_eq!(back.counts[TRIANGLES], triangles);
                    for c in &culled {
                        let at = c.id as usize * 4;
                        let got = [0, 1, 2, 3].map(|k| back.instances[at + k]);
                        let want = [c.origin.x, c.origin.y, c.origin.z, UV_STEP].map(f32::to_bits);
                        assert_eq!(got, want, "instance of {}", c.id);
                        // Every draw's first instance is below the instance buffer's length.
                        assert!((c.id as u64 + 1) * 16 <= far.instances.bytes);
                    }
                    compared += culled.len();
                    drawn_runs += draws;
                    // The statistics come back with their frame.
                    for _ in 0..3 {
                        if let Some(stats) = far.collect(device) {
                            assert_eq!(stats.frame, serial);
                            assert_eq!(stats.records, far.ids);
                            assert_eq!(stats.visible as usize, culled.len());
                            assert_eq!(stats.draws, [draws as u32, 0]);
                            assert_eq!(stats.triangles, [triangles, 0]);
                            break;
                        }
                        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    }
                }
            }
            assert!(
                compared > 500 && drawn_runs > 500,
                "{compared} {drawn_runs}"
            );
            eprintln!("far culling: {compared} kept records, {drawn_runs} draws compared");
        });
    }

    /// A Hi-Z pyramid as a bind group of `layout`.
    fn hiz_group(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        pyramid: &[(Vec<f32>, [usize; 2])],
    ) -> wgpu::BindGroup {
        let size = |[w, h]: [usize; 2]| wgpu::Extent3d {
            width: w as u32,
            height: h as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test Hi-Z"),
            size: size(pyramid[0].1),
            mip_level_count: pyramid.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (mip, (data, dims)) in pyramid.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * dims[0] as u32),
                    rows_per_image: Some(dims[1] as u32),
                },
                size(*dims),
            );
        }
        let view = texture.create_view(&Default::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test Hi-Z"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        })
    }

    /// Two-phase occlusion: new records are seen and draw in phase 1; the test stores which
    /// boxes a wall over half the screen hides; the next frame draws only those seen in
    /// phase 1 and, once the wall is gone, the rest in phase 2: draws, `hidden` and
    /// statistics are the mirror's, no record is drawn twice and none is lost.
    #[test]
    fn the_gpu_occludes_in_two_phases() {
        let _gpu = crate::gfx::test_gpu::lock();
        on_each_device(|device, queue| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test Hi-Z read"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
            let mut far = FarCull::new(device, Some(&layout)).expect("far culling");
            let mut uploads = Uploads::new(device);
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
            let mut shapes = Rng(0xA54F_F53A_5F1D_36F1);
            let camera = DVec3::new(0.5, 70.5, 0.5);
            let cam_int = [0, 70, 0];
            let (mut hid, mut compared_hidden, mut phase2) = (0, 0, 0);
            for case in 0..6 {
                far.records.clear();
                populate(&mut far.records, &mut rng, &mut shapes, cam_int, 0, 400);
                let mut input = input_for(&mut rng, camera, case % 2 == 0, true);
                // A pose and projection of its own; the wall sits 30 blocks ahead.
                let pose = Mat4::from_rotation_x(rng.between(-1.0, 1.0))
                    * Mat4::from_rotation_y(rng.between(-3.0, 3.0));
                let proj = world_projection(70.0, 16.0 / 9.0, 2000.0);
                let vp = proj * pose;
                input.planes = Frustum::from_view_projection(vp).planes;
                input.view_proj = vp;
                let wall_depth = depth_at(&proj, 30.0);
                let wall = pyramid(256, 128, |u, _| if u < 0.5 { wall_depth } else { 0.0 });
                let sky = pyramid(256, 128, |_, _| 0.0);
                let (wall_group, sky_group) = (
                    hiz_group(device, queue, &layout, &wall),
                    hiz_group(device, queue, &layout, &sky),
                );
                // Frame A: every record is new, so seen: phase 1 draws all the runs.
                let mut encoder = device.create_command_encoder(&Default::default());
                assert!(far.prepare(device, &mut uploads, &mut encoder, &input, None));
                assert!(far.occluding());
                submit(device, queue, &mut uploads, encoder);
                let back = read_back(device, queue, &far);
                let f = Frame::new(&input);
                let culled = cull(&far.records.records, far.pass_pages, &f);
                if culled.iter().any(|c| c.unstable) {
                    eprintln!("case {case}: a box on the frustum's edge; skipped");
                    continue;
                }
                let pages = far.pass_pages;
                let phase1 = compact(
                    &far.records.records,
                    &culled,
                    far.records.page_records(),
                    pages,
                    |_| true,
                );
                check_draws(&far, &back, 0, &phase1, &format!("case {case} A"));
                let mut encoder = device.create_command_encoder(&Default::default());
                far.test(&mut encoder, &wall_group, None);
                submit(device, queue, &mut uploads, encoder);
                let after = read_back(device, queue, &far);
                let none = vec![Vec::new(); pages as usize];
                check_draws(&far, &after, 1, &none, &format!("case {case} A"));
                let with_runs: Vec<&Culled> =
                    culled.iter().filter(|c| !c.runs.is_empty()).collect();
                assert_eq!(
                    after.counts[TESTED] as usize,
                    with_runs.len(),
                    "boxes tested"
                );
                for c in &with_runs {
                    let r = &far.records.records[c.id as usize];
                    let seen = |pad| box_seen(&vp, c.origin, r, pad, &wall);
                    let got = after.hidden[c.id as usize] == 0;
                    if seen(TEST_PAD - 0.05) == seen(TEST_PAD + 0.05) {
                        assert_eq!(got, seen(TEST_PAD), "case {case}: box of {}", c.id);
                        compared_hidden += 1;
                    }
                }
                let hidden: Vec<bool> = after.hidden.iter().map(|&h| h != 0).collect();
                assert!(
                    after
                        .hidden
                        .iter()
                        .enumerate()
                        .all(|(id, &h)| h == 0 || with_runs.iter().any(|c| c.id as usize == id)),
                    "only tested boxes are marked hidden"
                );
                assert_eq!(
                    after.counts[OCCLUDED] as usize,
                    hidden.iter().filter(|&&h| h).count()
                );
                hid += hidden.iter().filter(|&&h| h).count();
                // Frame B: the same records and view, the pyramid of frame A's wall decided
                // `hidden`; phase 1 draws only the seen ones.
                let mut encoder = device.create_command_encoder(&Default::default());
                assert!(far.prepare(device, &mut uploads, &mut encoder, &input, None));
                submit(device, queue, &mut uploads, encoder);
                let back = read_back(device, queue, &far);
                let phase1 = compact(
                    &far.records.records,
                    &culled,
                    far.records.page_records(),
                    pages,
                    |c| !hidden[c.id as usize],
                );
                check_draws(&far, &back, 0, &phase1, &format!("case {case} B"));
                // With the wall gone the candidates are seen, and draw in phase 2.
                let mut encoder = device.create_command_encoder(&Default::default());
                far.test(&mut encoder, &sky_group, None);
                submit(device, queue, &mut uploads, encoder);
                let after2 = read_back(device, queue, &far);
                let seen_now = |id: u32| after2.hidden[id as usize] == 0;
                let phase2_expected = compact(
                    &far.records.records,
                    &culled,
                    far.records.page_records(),
                    pages,
                    |c| hidden[c.id as usize] && !c.runs.is_empty() && seen_now(c.id),
                );
                check_draws(
                    &far,
                    &after2,
                    1,
                    &phase2_expected,
                    &format!("case {case} B"),
                );
                phase2 += phase2_expected.iter().map(Vec::len).sum::<usize>();
                // Phases 1 and 2 never draw the same record.
                for c in with_runs.iter() {
                    let in1 = !hidden[c.id as usize];
                    let in2 = hidden[c.id as usize] && seen_now(c.id);
                    assert!(!(in1 && in2));
                }
                // Not occluding: phase 1 is every kept run and `hidden` is left alone.
                let mut flat = input_for(&mut rng, camera, true, false);
                flat.planes = input.planes;
                flat.view_proj = input.view_proj;
                let mut encoder = device.create_command_encoder(&Default::default());
                assert!(far.prepare(device, &mut uploads, &mut encoder, &flat, None));
                assert!(!far.occluding());
                submit(device, queue, &mut uploads, encoder);
                let back = read_back(device, queue, &far);
                let f = Frame::new(&flat);
                let culled = cull(&far.records.records, far.pass_pages, &f);
                if culled.iter().all(|c| !c.unstable) {
                    let all = compact(
                        &far.records.records,
                        &culled,
                        far.records.page_records(),
                        pages,
                        |_| true,
                    );
                    check_draws(&far, &back, 0, &all, &format!("case {case} flat"));
                }
                assert_eq!(
                    back.hidden, after2.hidden,
                    "a frame that does not occlude leaves `hidden`"
                );
            }
            assert!(
                hid > 20 && compared_hidden > 100 && phase2 > 20,
                "{hid} {compared_hidden} {phase2}"
            );
            eprintln!(
                "far occlusion: {hid} hidden, {compared_hidden} boxes compared, {phase2} phase 2 draws"
            );
        });
    }
}

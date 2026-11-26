//! Far meshes share the terrain arenas and passes, with their actual bounds
//! rather than the 16³ boxes used by the near-section residency walk.
use super::{
    Frustum, Renderer, SectionGpu, cube_far,
    cube_gen::GpuBackend,
    surface_tile::{self, TileInfo as SurfaceInfo, TileKey as SurfaceKey},
};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
    lod::{
        MAX_LEVEL, TileKey,
        cube::{CubeKey, GpuBatch, TileWork},
        swap::Swap,
    },
    mesher::SectionMesh,
};
use glam::{DVec3, Vec3};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

/// Finished tile meshes not uploaded yet, the tiles they replace (removed
/// once every mesh is in) and the columns they leave out.
#[derive(Default)]
pub(super) struct Pending {
    meshes: FxHashMap<TileKey, SectionMesh>,
    /// The order they go up in: as handed over (`Lod::update`: finer tiles
    /// first, so no tile clipped around a finer one is drawn before it).
    order: VecDeque<TileKey>,
    removed: FxHashSet<TileKey>,
    cutout: Option<FxHashSet<(i32, i32)>>,
}

impl Pending {
    /// Takes in the next job, with the same result as applying both in
    /// turn: newer meshes replace waiting ones, tiles meshed again are not
    /// removed, and tiles removed now drop their waiting meshes.
    fn add(
        &mut self,
        meshes: Vec<(TileKey, SectionMesh)>,
        removed: Vec<TileKey>,
        cutout: Option<FxHashSet<(i32, i32)>>,
    ) {
        for (key, mesh) in meshes {
            self.removed.remove(&key);
            if self.meshes.insert(key, mesh).is_none() {
                self.order.push_back(key);
            }
        }
        for key in removed {
            self.meshes.remove(&key);
            self.removed.insert(key);
        }
        if cutout.is_some() {
            self.cutout = cutout;
        }
    }
}

/// A far tile of either kind: a column tile of the surface model, or a node of
/// a cubic level's octree (`lod::cube`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum FarKey {
    Tile(TileKey),
    Cube(CubeKey),
}

impl FarKey {
    pub fn level(self) -> u8 {
        match self {
            Self::Tile(key) => key.level,
            Self::Cube(key) => key.level,
        }
    }
    pub fn origin(self) -> [i32; 3] {
        match self {
            Self::Tile(key) => key.origin(),
            Self::Cube(key) => key.origin(),
        }
    }
}

pub(super) fn far_tile<'a>(
    tiles: &'a FxHashMap<TileKey, Tile>,
    cubes: &'a FxHashMap<CubeKey, Tile>,
    key: FarKey,
) -> Option<&'a Tile> {
    match key {
        FarKey::Tile(key) => tiles.get(&key),
        FarKey::Cube(key) => cubes.get(&key),
    }
}

pub(super) fn far_tile_mut<'a>(
    tiles: &'a mut FxHashMap<TileKey, Tile>,
    cubes: &'a mut FxHashMap<CubeKey, Tile>,
    key: FarKey,
) -> Option<&'a mut Tile> {
    match key {
        FarKey::Tile(key) => tiles.get_mut(&key),
        FarKey::Cube(key) => cubes.get_mut(&key),
    }
}

/// Finished node meshes not uploaded yet, and which nodes show and which go
/// as the sets they belong to come in (`Swap`). It lives as long as the tiles
/// it tracks (`clear_cube_lods`).
#[derive(Default)]
pub(super) struct CubePending {
    meshes: FxHashMap<CubeKey, SectionMesh>,
    order: VecDeque<CubeKey>,
    swap: Swap,
}

impl CubePending {
    /// As [`Pending::add`]: newer meshes replace waiting ones, nodes meshed
    /// again are not removed, nodes removed drop their waiting meshes.
    fn add(&mut self, meshes: Vec<(CubeKey, SectionMesh)>, removed: Vec<CubeKey>) {
        for (key, mesh) in meshes {
            self.swap.queued(key);
            if self.meshes.insert(key, mesh).is_none() {
                self.order.push_back(key);
            }
        }
        for key in removed {
            if self.meshes.remove(&key).is_some() {
                self.swap.unwait(key);
            }
            self.swap.retire(key);
        }
    }
}

/// Tracks resident tiles needing per-frame or end-of-batch work so calm frames
/// (`Renderer::upload_lods`) run in O(1) instead of scanning every resident tile.
#[derive(Default)]
pub(super) struct LodTracker {
    /// Resident tiles with `tile.dying.is_some()`.
    dying: FxHashSet<TileKey>,
    /// Resident tiles with `tile.previous.is_some()`.
    previous: FxHashSet<TileKey>,
    /// Resident tiles with `tile.shown.is_none()`.
    unshown: FxHashSet<TileKey>,
}

trait LodTileState {
    fn set_shown(&mut self, shown: Instant);
    fn dying(&self) -> Option<Instant>;
    fn set_dying(&mut self, since: Instant, alone: bool);
    fn start_previous(&mut self, since: Instant);
}

fn ancestors_of<'a>(keys: impl IntoIterator<Item = &'a TileKey>) -> FxHashSet<TileKey> {
    let mut above = FxHashSet::default();
    for &key in keys {
        let mut key = key;
        while key.level < MAX_LEVEL && above.insert(key.parent()) {
            key = key.parent();
        }
    }
    above
}

/// Whether `key` equals, is an ancestor of, or descends from any tile in `set`
/// (where `above` holds every strict ancestor of every tile in `set`).
fn overlaps_set(key: TileKey, set: &FxHashSet<TileKey>, above: &FxHashSet<TileKey>) -> bool {
    if set.contains(&key) || above.contains(&key) {
        return true;
    }
    let mut ancestor = key;
    (key.level..MAX_LEVEL).any(|_| {
        ancestor = ancestor.parent();
        set.contains(&ancestor)
    })
}

/// Chunk columns newly taken by near terrain in a pending batch, with a tight
/// block bounding box to reject non-overlapping far tiles in O(1).
#[derive(Default)]
struct TakenColumns {
    columns: Vec<(i32, i32)>,
    bounds: Option<[i32; 4]>,
}

impl TakenColumns {
    fn new(cutout: Option<&FxHashSet<(i32, i32)>>, current: &FxHashSet<(i32, i32)>) -> Self {
        let mut columns = Vec::new();
        let mut bounds: Option<[i32; 4]> = None;
        if let Some(cutout) = cutout {
            for &(cx, cz) in cutout {
                if !current.contains(&(cx, cz)) {
                    columns.push((cx, cz));
                    let x = cx * 16;
                    let z = cz * 16;
                    bounds = Some(match bounds {
                        None => [x, z, x, z],
                        Some([x0, z0, x1, z1]) => [x0.min(x), z0.min(z), x1.max(x), z1.max(z)],
                    });
                }
            }
        }
        Self { columns, bounds }
    }

    fn overlaps(
        &self,
        key: TileKey,
        cutout: Option<&FxHashSet<(i32, i32)>>,
        current: &FxHashSet<(i32, i32)>,
    ) -> bool {
        let Some([bx0, bz0, bx1, bz1]) = self.bounds else {
            return false;
        };
        let [x, _, z] = key.origin();
        let size = key.size();
        if bx1 < x || bx0 >= x + size || bz1 < z || bz0 >= z + size {
            return false;
        }
        if key.level == 0 {
            return cutout.is_some_and(|c| c.contains(&(key.x, key.z)))
                && !current.contains(&(key.x, key.z));
        }
        self.columns.iter().any(|&(cx, cz)| {
            (x..x + size).contains(&(cx * 16)) && (z..z + size).contains(&(cz * 16))
        })
    }
}

impl LodTracker {
    pub(super) fn clear(&mut self) {
        self.dying.clear();
        self.previous.clear();
        self.unshown.clear();
    }

    fn remove(&mut self, key: &TileKey) {
        self.dying.remove(key);
        self.previous.remove(key);
        self.unshown.remove(key);
    }

    fn record_uploaded(&mut self, key: TileKey, shown: bool, has_previous: bool) {
        self.dying.remove(&key);
        if shown {
            self.unshown.remove(&key);
        } else {
            self.unshown.insert(key);
        }
        if has_previous {
            self.previous.insert(key);
        } else {
            self.previous.remove(&key);
        }
    }

    fn mark_dying(&mut self, key: TileKey) {
        self.dying.insert(key);
    }

    fn collect_faded<T: LodTileState>(
        &self,
        lods: &FxHashMap<TileKey, T>,
        start: Instant,
    ) -> Vec<TileKey> {
        if self.dying.is_empty() {
            return Vec::new();
        }
        self.dying
            .iter()
            .copied()
            .filter(|key| {
                lods.get(key).is_some_and(|tile| {
                    tile.dying()
                        .is_some_and(|d| start.saturating_duration_since(d) >= FADE)
                })
            })
            .collect()
    }

    fn expire_previous<T, M>(
        &mut self,
        lods: &mut FxHashMap<TileKey, T>,
        start: Instant,
        mut take_expired: impl FnMut(&mut T, Instant) -> Option<Option<M>>,
    ) -> Vec<M> {
        if self.previous.is_empty() {
            return Vec::new();
        }
        let mut expired = Vec::new();
        self.previous.retain(|key| {
            let Some(tile) = lods.get_mut(key) else {
                return false;
            };
            match take_expired(tile, start) {
                Some(Some(mesh)) => {
                    expired.push(mesh);
                    false
                }
                Some(None) => true,
                None => false,
            }
        });
        expired
    }

    /// Completes a drained `Pending` batch using only the tracked sets and
    /// `removed`, skipping all full-map scans of `lods` and skipping
    /// `replaced`/`kept`/`above` construction whenever `removed` is empty.
    fn finish_batch<T: LodTileState>(
        &mut self,
        lods: &mut FxHashMap<TileKey, T>,
        removed: FxHashSet<TileKey>,
        pending_cutout: Option<&FxHashSet<(i32, i32)>>,
        current_cutout: &FxHashSet<(i32, i32)>,
        start: Instant,
    ) {
        let unshown = std::mem::take(&mut self.unshown);
        let whole = start.checked_sub(FADE).unwrap_or(start);

        if !unshown.is_empty() {
            // A tile fades in over the ones it replaces. One over nothing
            // shows whole: its hidden chunks would open holes to the sky
            // or the sea bed (dark patches, as far terrain first loads).
            let replaced: FxHashSet<TileKey> = if removed.is_empty() {
                FxHashSet::default()
            } else {
                removed
                    .iter()
                    .copied()
                    .filter(|key| lods.contains_key(key))
                    .collect()
            };
            if replaced.is_empty() {
                for &key in &unshown {
                    if let Some(tile) = lods.get_mut(&key) {
                        tile.set_shown(whole);
                    }
                }
            } else {
                let above_replaced = ancestors_of(&replaced);
                for &key in &unshown {
                    if let Some(tile) = lods.get_mut(&key) {
                        let shown = if overlaps_set(key, &replaced, &above_replaced) {
                            start
                        } else {
                            whole
                        };
                        tile.set_shown(shown);
                    }
                }
            }
        }

        for &key in &self.previous {
            if let Some(tile) = lods.get_mut(&key) {
                tile.start_previous(start);
            }
        }

        if !removed.is_empty() {
            let newly_dying: Vec<TileKey> = removed
                .into_iter()
                .filter(|key| lods.get(key).is_some_and(|tile| tile.dying().is_none()))
                .collect();
            if !newly_dying.is_empty() {
                // A removed tile no new tile overlaps, and no near chunk takes
                // a column of, leaves the plan (out of range): it fades out.
                // Over near chunks fading in it stays whole: two patterns
                // fading at once left holes between them.
                // Two tiles overlap when one is the other or its ancestor:
                // `kept` holds the new tiles and the near chunks' columns
                // (level 0), `above` their ancestors.
                let mut kept = unshown;
                let cut = pending_cutout.unwrap_or(current_cutout);
                kept.extend(cut.iter().map(|&(x, z)| TileKey { x, z, level: 0 }));
                let above = ancestors_of(&kept);
                for key in newly_dying {
                    let alone = !overlaps_set(key, &kept, &above);
                    if let Some(tile) = lods.get_mut(&key)
                        && tile.dying().is_none()
                    {
                        tile.set_dying(start, alone);
                        self.mark_dying(key);
                    }
                }
            }
        }
    }
}

/// Small, its meshes boxed: `draw_frame` walks every tile each frame (thousands at long far
/// distances), and that walk is bound by the memory it reads.
pub(super) struct Tile {
    pub mesh: Box<SectionGpu>,
    bounds: Bounds,
    /// When it began to fade in; none while the rest of its job uploads.
    shown: Option<Instant>,
    /// When its replacements began to fade in; it goes once they have.
    dying: Option<Instant>,
    /// Dying with nothing to replace it (out of range): it fades out.
    alone: bool,
    /// Its mesh before near chunks took some of its columns, drawn in its
    /// place until they have faded in (from the job's end, the instant), so
    /// no hole shows through them.
    previous: Option<Box<(SectionGpu, Option<Instant>)>>,
    /// Its slot in occlusion culling's results (`cull::TILE`), kept while
    /// meshes of the same tile replace one another.
    pub cull_slot: u32,
}

/// How long a tile takes to fade in over the ones it replaces.
pub(super) const FADE: Duration = Duration::from_millis(600);

/// A tile or section fading in: 0 drawn whole, else 1 + `level` * 129 +
/// `shown`, `shown` 1 to 63 sixty-fourths of `FADE` (`fade_out_code`: 65 to
/// 128, fading out). Chunk by chunk (columns of its own grid, wider for
/// coarse tiles: 2^`level` blocks), each starting in turn over the first half
/// and fading over the second, as a fine ordered pattern fixed to the terrain
/// (terrain.wgsl's `faded`). It is drawn a hair toward the camera, the newest
/// most, over the solid tiles it replaces, which stay until it is whole
/// (`upload_lods`): no pixel fought over, no hole. None: nothing to draw yet.
pub(super) fn fade_code(shown: Option<Instant>, level: u8, now: Instant) -> Option<u32> {
    let faded = now.saturating_duration_since(shown?).as_secs_f32() / FADE.as_secs_f32();
    let shown = (faded * 64.0) as u32 + 1;
    Some(if shown >= 64 {
        0
    } else {
        1 + u32::from(level) * 129 + shown
    })
}
/// A section past the render distance that far terrain is coming in under:
/// whole, a fading-out code at its start (terrain.wgsl's `fade_depth` draws
/// leaving sections in front of far terrain).
pub(super) const HELD: u32 = 1 + 64;
/// `fade_code` for one leaving since `since`, chunk by chunk as it came, over
/// what stays behind it; once `FADE` has passed it draws nothing.
pub(super) fn fade_out_code(since: Instant, level: u8, now: Instant) -> u32 {
    let faded = now.saturating_duration_since(since).as_secs_f32() / FADE.as_secs_f32();
    1 + u32::from(level) * 129 + 64 + ((faded * 64.0) as u32 + 1).min(64)
}

/// Who draws a far tile's solid quads (`Tile::far_class`, gfx/far_cull.rs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FarClass {
    /// Not drawn: its replacement is not whole yet (`shown` unset).
    Hidden,
    /// The CPU's walk: fading in or out, dying, or standing in.
    Cpu,
    /// GPU culling and indirect draws: whole.
    Gpu,
}

#[derive(Clone, Copy)]
struct Bounds {
    min: Vec3,
    max: Vec3,
}

impl Bounds {
    fn mesh(mesh: &SectionMesh) -> Option<Self> {
        let [min, max] = mesh.bounds?;
        Some(Self { min, max })
    }

    /// In view: the plan alone decides which tiles are in range (a circle),
    /// so a tile leaving it fades out (`upload_lods`). Cut here at the range,
    /// tiles popped out as the player moved, before the plan dropped them.
    fn visible(self, origin: DVec3, frustum: &Frustum) -> Option<u64> {
        let min = origin + self.min.as_dvec3();
        let max = origin + self.max.as_dvec3();
        if !frustum.intersects(min.as_vec3(), max.as_vec3()) {
            return None;
        }
        Some(((min + max) * 0.5).length_squared().to_bits())
    }
}

/// A tile's instance: like a section's, with the UV step of merged faces.
/// A fading tile's step is negated past its `fade_code`, which the shaders
/// read back as the whole part (they take the step as `fract(abs(w))`).
pub(super) fn instance(origin: [f32; 3], fade: u32) -> [f32; 4] {
    let step = 1.0 / crate::mesher::TILED_UV_STEPS;
    [
        origin[0],
        origin[1],
        origin[2],
        if fade == 0 {
            step
        } else {
            -(fade as f32 + step)
        },
    ]
}

pub(super) struct Visible {
    pub key: FarKey,
    pub distance: u64,
    pub origin: [f32; 3],
    pub fade: u32,
    /// Replaced, standing until its replacements are whole: its water goes
    /// after theirs (`Renderer::render`'s water tiers), so depth leaves it only
    /// where they have none yet.
    pub old: bool,
}

impl Tile {
    /// The mesh drawn this frame: the previous one while it stands in.
    pub fn drawn(&self) -> &SectionGpu {
        self.previous
            .as_deref()
            .map_or(&self.mesh, |(mesh, _)| mesh)
    }
    pub fn drawn_mut(&mut self) -> &mut SectionGpu {
        match self.previous.as_deref_mut() {
            Some((mesh, _)) => mesh,
            None => &mut self.mesh,
        }
    }
    /// The box occlusion culling tests (`cull::Cull`), around its origin: its
    /// mesh's bounds grown by a block, while that mesh is the one drawn
    /// whole (no stand-in, no fade). A fading tile's dither moves its depth
    /// toward the camera (terrain.wgsl's `fade_depth`), out of the box.
    pub fn cull_bounds(&self, fade: u32) -> Option<[[f32; 3]; 2]> {
        (fade == 0 && self.previous.is_none()).then(|| {
            [
                (self.bounds.min - 1.0).to_array(),
                (self.bounds.max + 1.0).to_array(),
            ]
        })
    }
    /// Who draws this far tile's solid quads at `now` (`far_cull`): nobody while it is
    /// hidden, the CPU while it fades in or out, is dying or has a stand-in, else
    /// GPU-driven culling, whole.
    pub fn far_class(&self, level: u8, now: Instant) -> FarClass {
        if self.dying.is_some() || self.previous.is_some() {
            return FarClass::Cpu;
        }
        match fade_code(self.shown, level, now) {
            None => FarClass::Hidden,
            Some(0) => FarClass::Gpu,
            Some(_) => FarClass::Cpu,
        }
    }
    /// The box of its quads (`SectionMesh::bounds`) around its origin: (min, max).
    pub fn bounds_box(&self) -> [Vec3; 2] {
        [self.bounds.min, self.bounds.max]
    }
    /// Whether either mesh it holds is in the compact layout: a stand-in drawn in its place can
    /// hide a compact replacement (`Renderer::follow_packed_layout`).
    pub fn holds_packed(&self) -> bool {
        self.mesh.packed || self.previous.as_ref().is_some_and(|p| p.0.packed)
    }
    /// Both meshes it holds in the arena: its own and the stand-in, if any.
    pub fn meshes_mut(&mut self) -> impl Iterator<Item = &mut SectionGpu> {
        std::iter::once(&mut *self.mesh).chain(self.previous.as_deref_mut().map(|(mesh, _)| mesh))
    }
    pub fn visible(
        &self,
        key: FarKey,
        camera: DVec3,
        frustum: &Frustum,
        now: Instant,
    ) -> Option<Visible> {
        let fade = match self.dying {
            Some(since) if self.alone => fade_out_code(since, key.level(), now),
            _ => fade_code(self.shown, key.level(), now)?,
        };
        let origin = DVec3::from(key.origin().map(f64::from)) - camera;
        Some(Visible {
            key,
            distance: self.bounds.visible(origin, frustum)?,
            origin: origin.as_vec3().to_array(),
            fade,
            old: self.dying.is_some() || self.previous.is_some(),
        })
    }
}

#[cfg(test)]
impl Tile {
    /// A tile for tests (`far_cull`): `mesh`, its box `bounds` around its origin,
    /// shown since `shown`, dying since `dying`.
    pub(super) fn for_test(
        mesh: SectionGpu,
        bounds: [Vec3; 2],
        shown: Option<Instant>,
        dying: Option<Instant>,
    ) -> Self {
        Self {
            mesh: Box::new(mesh),
            bounds: Bounds {
                min: bounds[0],
                max: bounds[1],
            },
            shown,
            dying,
            alone: false,
            previous: None,
            cull_slot: 0,
        }
    }
    pub(super) fn set_times_for_test(&mut self, shown: Option<Instant>, dying: Option<Instant>) {
        (self.shown, self.dying) = (shown, dying);
    }
}

impl LodTileState for Tile {
    fn set_shown(&mut self, shown: Instant) {
        self.shown = Some(shown);
    }
    fn dying(&self) -> Option<Instant> {
        self.dying
    }
    fn set_dying(&mut self, since: Instant, alone: bool) {
        self.dying = Some(since);
        self.alone = alone;
    }
    fn start_previous(&mut self, since: Instant) {
        if let Some((_, started)) = self.previous.as_deref_mut() {
            started.get_or_insert(since);
        }
    }
}

impl Renderer {
    /// A tile meshed again (finer detail) takes its place at once, but for
    /// one meshed around near chunks or finer tiles that are new (`keep`):
    /// its old mesh stands in while they fade in. A new one waits for the rest of its job
    /// (`upload_lods`), then fades in.
    pub fn upload_lod(&mut self, key: TileKey, mesh: &SectionMesh, keep: bool) {
        if let Some(copies) = &mut self.capture_copies {
            copies.tile(key, mesh);
        }
        let old_slot = self.lods.get(&key).map(|tile| tile.cull_slot);
        let (shown, previous) = match self.lods.remove(&key) {
            Some(old) if old.dying.is_none() => match old.previous {
                Some(previous) => {
                    self.release_mesh(*old.mesh);
                    (old.shown, Some(previous))
                }
                None if keep => (old.shown, Some(Box::new((*old.mesh, None)))),
                None => {
                    self.release_mesh(*old.mesh);
                    (old.shown, None)
                }
            },
            Some(old) => {
                self.release_tile(old);
                (None, None)
            }
            None => (None, None),
        };
        match Bounds::mesh(mesh).zip(self.upload_mesh(mesh)) {
            Some((bounds, mesh)) => {
                let cull_slot = old_slot.unwrap_or_else(|| {
                    self.lod_cull_free.pop().unwrap_or_else(|| {
                        self.lod_cull_next += 1;
                        self.lod_cull_next - 1
                    })
                });
                self.lod_tracker
                    .record_uploaded(key, shown.is_some(), previous.is_some());
                self.lods.insert(
                    key,
                    Tile {
                        mesh: Box::new(mesh),
                        bounds,
                        shown,
                        dying: None,
                        alone: false,
                        previous,
                        cull_slot,
                    },
                );
            }
            None => {
                self.lod_tracker.remove(&key);
                if let Some((previous, _)) = previous.map(|p| *p) {
                    self.release_mesh(previous);
                }
                self.lod_cull_free.extend(old_slot);
            }
        }
    }

    fn release_tile(&mut self, tile: Tile) {
        self.release_mesh(*tile.mesh);
        if let Some((previous, _)) = tile.previous.map(|p| *p) {
            self.release_mesh(previous);
        }
    }

    pub fn remove_lod(&mut self, key: TileKey) {
        if let Some(copies) = &mut self.capture_copies {
            copies.forget_tile(key);
        }
        self.lod_tracker.remove(&key);
        if let Some(tile) = self.lods.remove(&key) {
            self.lod_cull_free.push(tile.cull_slot);
            self.release_tile(tile);
        }
    }

    pub fn clear_lods(&mut self) {
        if let Some(copies) = &mut self.capture_copies {
            copies.clear_tiles();
        }
        self.lod_pending = None;
        self.lod_tracker.clear();
        // Cube tiles share the culling slots: they keep theirs.
        let slots: Vec<u32> = self.lods.values().map(|tile| tile.cull_slot).collect();
        for (_, tile) in std::mem::take(&mut self.lods) {
            self.release_tile(tile);
        }
        if self.lod_cubes.is_empty() {
            self.lod_cull_free.clear();
            self.lod_cull_next = 0;
        } else {
            self.lod_cull_free.extend(slots);
        }
        self.set_lod_cutout(Default::default());
        // No far terrain left to fade out over. Drained after `set_lod_cutout`, which
        // moves the sections far terrain still held (`lod_retiring`) onto this list:
        // `release_left` runs only from `upload_lods`, and the game stops calling that
        // once far terrain is off, so a section left here would draw its unloaded
        // column past the render distance for the rest of the renderer's life.
        for (key, since) in std::mem::take(&mut self.lod_leaving) {
            if self
                .sections
                .find(key)
                .is_some_and(|s| s.leaving == Some(since))
            {
                self.release_section(key);
            }
        }
    }

    /// A node meshed again keeps its slot and fade; a new one waits for its
    /// set (`upload_cube_lods`) before it fades in. Whether the node is in and
    /// already shown; none if it has no mesh.
    fn upload_cube_lod(&mut self, key: CubeKey, mesh: &SectionMesh) -> Option<bool> {
        let shown = self.install_cube_lod(key, mesh);
        self.track_cube(key);
        shown
    }

    fn install_cube_lod(&mut self, key: CubeKey, mesh: &SectionMesh) -> Option<bool> {
        let (old_slot, shown) = match self.lod_cubes.remove(&key) {
            Some(old) => {
                let kept = (old.cull_slot, old.shown.filter(|_| old.dying.is_none()));
                self.release_tile(old);
                (Some(kept.0), kept.1)
            }
            None => (None, None),
        };
        match Bounds::mesh(mesh).zip(self.upload_mesh(mesh)) {
            Some((bounds, mesh)) => {
                let cull_slot = old_slot.unwrap_or_else(|| {
                    self.lod_cull_free.pop().unwrap_or_else(|| {
                        self.lod_cull_next += 1;
                        self.lod_cull_next - 1
                    })
                });
                self.lod_cubes.insert(
                    key,
                    Tile {
                        mesh: Box::new(mesh),
                        bounds,
                        shown,
                        dying: None,
                        alone: false,
                        previous: None,
                        cull_slot,
                    },
                );
                Some(shown.is_some())
            }
            None => {
                self.lod_cull_free.extend(old_slot);
                None
            }
        }
    }

    /// Whether far cube terrain can be meshed and drawn on the GPU here: the
    /// device runs compute shaders and draws indirect with a first instance.
    pub fn gpu_far_available(&self) -> bool {
        self.multi_draw
            && self.adapter.get_downlevel_capabilities().flags.contains(
                wgpu::DownlevelFlags::COMPUTE_SHADERS | wgpu::DownlevelFlags::VERTEX_STORAGE,
            )
            && self.device.limits().max_storage_buffers_per_shader_stage >= 6
    }

    /// Makes the GPU far terrain ready for nodes of `backend`'s pool, which
    /// must be on this renderer's device. `states` the registry's block state
    /// count, `air` its air state. False if it cannot run (stay on the CPU).
    pub fn use_gpu_cube_lods(&mut self, backend: &GpuBackend, states: usize, air: u32) -> bool {
        if let Some(far) = &self.cube_far {
            if far.serves(backend) {
                return true;
            }
            if *far.device() != backend.device {
                return false;
            }
            // Another level's generator (a world opened after another in
            // this game): its own node pool and terrain. The far terrain
            // built for the last one would read this one's slots from the old
            // pool, drawing stale nodes (rock in the sky), and its surface
            // tiles the old terrain: both are made again.
            self.clear_cube_lods();
            self.cube_far = None;
            self.surface_tiles = None;
            self.surface_tiles_failed = false;
        }
        if !self.gpu_far_available() || backend.device != self.device {
            return false;
        }
        let setup = cube_far::DrawSetup {
            bindings: &self.terrain_layout,
            format: self.output.frame.format,
            sprite_constants: self.terrain_sprites,
            counts: self
                .device
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
                && self.info.backend != wgpu::Backend::Dx12,
        };
        self.cube_far = cube_far::CubeFar::new(backend, states, air, &setup);
        self.cube_far.is_some()
    }

    /// Makes the surface tiles ready beside the GPU far terrain
    /// (`use_gpu_cube_lods` first), with a pool of `capacity` tiles, from the
    /// generator's parameters and its GPU climate pass. False when they cannot
    /// run: no GPU far terrain, no GPU climate pass (`GpuBackend::biome_params`,
    /// a Custom world's), or a pipeline that does not build.
    pub fn use_surface_tiles(&mut self, backend: &GpuBackend, capacity: u32) -> bool {
        if self.surface_tiles.is_some() {
            return true;
        }
        let (Some(far), Some(biomes)) = (&self.cube_far, &backend.biome_params) else {
            return false;
        };
        if self.surface_tiles_failed || *far.device() != backend.device {
            return false;
        }
        // Built once: a pipeline that does not build will not the next frame.
        self.surface_tiles_failed = true;
        let setup = cube_far::DrawSetup {
            bindings: &self.terrain_layout,
            format: self.output.frame.format,
            sprite_constants: self.terrain_sprites,
            counts: self
                .device
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
                && self.info.backend != wgpu::Backend::Dx12,
        };
        let Some(mut tiles) = surface_tile::SurfaceTiles::new(
            &backend.device,
            &backend.queue,
            backend.params.clone(),
            surface_tile::ColumnSource::Gpu(biomes.clone()),
            capacity,
        ) else {
            return false;
        };
        if !tiles.enable_draw(&setup, far.material_buffer(), far.biome_buffer()) {
            return false;
        }
        self.surface_tiles = Some(tiles);
        self.surface_tiles_failed = false;
        true
    }

    /// Whether the surface tiles run (`use_surface_tiles`).
    pub fn surface_tiles_on(&self) -> bool {
        self.surface_tiles.is_some()
    }

    /// Hands the surface tiles what `CubeLod::take_tile_work` returned: drops,
    /// fades and requests, in that order.
    pub fn queue_surface_tiles(&mut self, work: TileWork) {
        let Some(tiles) = &mut self.surface_tiles else {
            return;
        };
        for key in work.free {
            tiles.free(key);
        }
        for key in work.retire {
            tiles.retire(key);
        }
        // A tile asked for again while it fades out is made again: `request`
        // skips resident tiles, and this one is about to be freed.
        for &key in &work.request {
            if tiles.ready(key) {
                tiles.free(key);
            }
        }
        tiles.request(&work.request);
    }

    /// A surface tile's header, once generated and read back.
    pub fn surface_tile_info(&self, key: SurfaceKey) -> Option<SurfaceInfo> {
        self.surface_tiles.as_ref()?.info(key)
    }

    /// The surface tiles' counters, if they run.
    pub fn surface_stats(&self) -> Option<surface_tile::SurfaceStats> {
        self.surface_tiles.as_ref().map(|t| t.stats())
    }

    /// Queues the nodes `CubeLod::take_gpu_nodes` returned for meshing.
    pub fn queue_gpu_cube_lods(&mut self, batch: GpuBatch) {
        if let Some(far) = &mut self.cube_far {
            far.set_materials(&batch.materials, batch.biomes.as_deref());
            far.queue(batch.nodes, Vec::new());
        }
    }

    /// The GPU far terrain's counters, if it runs.
    pub fn cube_far_stats(&self) -> Option<cube_far::FarStats> {
        self.cube_far.as_ref().map(|far| far.stats())
    }

    pub fn remove_cube_lod(&mut self, key: CubeKey) {
        if let Some(tile) = self.lod_cubes.remove(&key) {
            self.lod_cull_free.push(tile.cull_slot);
            self.release_tile(tile);
        }
        self.cube_overlap.set(key, false);
    }

    /// Drops every cube tile and what waits to upload.
    pub fn clear_cube_lods(&mut self) {
        self.cube_pending = None;
        if let Some(far) = &mut self.cube_far {
            far.clear();
        }
        if let Some(tiles) = &mut self.surface_tiles {
            let keys: Vec<SurfaceKey> = tiles.keys().collect();
            for key in keys {
                tiles.free(key);
            }
        }
        for (_, tile) in std::mem::take(&mut self.lod_cubes) {
            self.lod_cull_free.push(tile.cull_slot);
            self.release_tile(tile);
        }
        self.cube_overlap.clear();
        if self.lods.is_empty() {
            self.lod_cull_free.clear();
            self.lod_cull_next = 0;
        }
    }

    /// Queues what `CubeLod::update` and `CubeLod::removed` returned for
    /// `upload_cube_lods`, joining a job still uploading.
    pub fn queue_cube_lods(&mut self, meshes: Vec<(CubeKey, SectionMesh)>, removed: Vec<CubeKey>) {
        if meshes.is_empty() && removed.is_empty() {
            return;
        }
        if let Some(far) = &mut self.cube_far {
            far.queue(Vec::new(), removed.clone());
        }
        self.cube_pending
            .get_or_insert_default()
            .add(meshes, removed);
    }

    /// Uploads queued node meshes for about `budget` (at least one). A node
    /// removed stands whole until what replaces it is in; then, in one frame,
    /// it goes and its replacements show whole (`Swap::step`): no frame draws
    /// both (z-fighting) or neither (a hole), and a set does not wait for the
    /// rest of the queue. Only a node over nothing fades in, and only one
    /// leaving nothing behind fades out.
    pub fn upload_cube_lods(&mut self, budget: Duration) {
        let start = Instant::now();
        if let Some(far) = &mut self.cube_far {
            far.upload();
        }
        let faded: Vec<_> = self
            .lod_cubes
            .iter()
            .filter(|(_, tile)| tile.dying.is_some_and(|d| start - d >= FADE))
            .map(|(&key, _)| key)
            .collect();
        for key in faded {
            self.remove_cube_lod(key);
            if let Some(pending) = &mut self.cube_pending {
                pending.swap.forget(key);
            }
        }
        while let Some(pending) = &mut self.cube_pending {
            let next = std::iter::from_fn(|| pending.order.pop_front())
                .find(|key| pending.meshes.contains_key(key));
            let Some(key) = next else {
                break;
            };
            let mesh = pending.meshes.remove(&key).expect("pending node");
            let shown = self.upload_cube_lod(key, &mesh);
            let swap = &mut self.cube_pending.as_mut().expect("pending nodes").swap;
            match shown {
                Some(shown) => swap.installed(key, shown),
                None => {
                    swap.unwait(key);
                    swap.forget(key);
                }
            }
            if start.elapsed() >= budget {
                break;
            }
        }
        let Some(pending) = &mut self.cube_pending else {
            self.cube_overlap.frame();
            return;
        };
        let mut step = pending.swap.step();
        // `HORIZON_FAR_CROSSFADE=1`: the old cross-fade, to measure it (the overlap
        // counter): a replaced node fades out under what replaces it fading in.
        if cross_fade() {
            let drop = std::mem::take(&mut step.drop);
            for key in drop {
                if self.lod_cubes.get(&key).is_some_and(|t| t.shown.is_some()) {
                    step.fade.push(key);
                } else {
                    step.drop.push(key);
                }
            }
            step.fade_in.append(&mut step.show);
        }
        // Replaced, a fade cut short, or never shown: gone this frame.
        for key in step.drop {
            self.remove_cube_lod(key);
        }
        // Nothing drawn over their place: they fade out alone.
        for key in step.fade {
            let Some(tile) = self.lod_cubes.get_mut(&key) else {
                continue;
            };
            if tile.shown.is_none() {
                self.remove_cube_lod(key);
            } else if tile.dying.is_none() {
                tile.dying = Some(start);
                tile.alone = true;
            }
        }
        // In place of the nodes dropped above: whole at once, no cross-fade
        // over them (two surfaces over the same ground fight for its pixels).
        let whole = start.checked_sub(FADE).unwrap_or(start);
        for key in step.show {
            if let Some(tile) = self.lod_cubes.get_mut(&key) {
                tile.shown = Some(whole);
            }
            self.track_cube(key);
        }
        // Over nothing drawn: they fade in.
        for key in step.fade_in {
            if let Some(tile) = self.lod_cubes.get_mut(&key) {
                tile.shown = Some(start);
            }
            self.track_cube(key);
        }
        self.cube_overlap.frame();
    }

    /// Notes whether cube node `key` draws now, for the overlap counter.
    fn track_cube(&mut self, key: CubeKey) {
        let drawn = self
            .lod_cubes
            .get(&key)
            .is_some_and(|tile| tile.shown.is_some());
        self.cube_overlap.set(key, drawn);
    }

    /// The far nodes drawn over one another (`lod::swap::Overlaps`), on the
    /// CPU and GPU paths together: 0 every frame when the swaps hold.
    pub fn cube_overlap_stats(&self) -> crate::lod::swap::OverlapStats {
        let cpu = self.cube_overlap.stats();
        match &self.cube_far {
            Some(far) => cpu.plus(far.overlap_stats()),
            None => cpu,
        }
    }

    /// Queues a finished mesh job (`Lod::update`, `Lod::removed`,
    /// `Lod::take_cutout`) for `upload_lods`, joining one still uploading
    /// (`Pending::add`).
    pub fn queue_lods(
        &mut self,
        meshes: Vec<(TileKey, SectionMesh)>,
        removed: Vec<TileKey>,
        cutout: Option<FxHashSet<(i32, i32)>>,
    ) {
        if meshes.is_empty() && removed.is_empty() && cutout.is_none() {
            return;
        }
        self.lod_pending
            .get_or_insert_default()
            .add(meshes, removed, cutout);
    }

    /// Uploads queued tile meshes for about `budget` (at least one), then,
    /// once all are in, fades them in over the tiles they replace.
    pub fn upload_lods(&mut self, budget: Duration) {
        let start = Instant::now();
        if self.lod_pending.is_some() {
            // Far tiles fill the arenas too: large translucent pages wait for them.
            self.shrink_gate.wait(start, super::SHRINK_SETTLE);
        }
        self.release_left(start);
        for key in self.lod_tracker.collect_faded(&self.lods, start) {
            self.remove_lod(key);
        }
        let stood_in = self
            .lod_tracker
            .expire_previous(&mut self.lods, start, |tile, now| {
                match tile.previous.as_deref() {
                    Some((_, Some(since))) if now.saturating_duration_since(*since) >= FADE => {
                        Some(tile.previous.take().map(|p| *p))
                    }
                    Some(_) => Some(None),
                    None => None,
                }
            });
        for (previous, _) in stood_in {
            self.release_mesh(previous);
        }
        let Some(pending) = self.lod_pending.as_ref() else {
            return;
        };
        // The columns near chunks take in this job, and the tiles holding the
        // job's new finer ones: a tile meshed again around them keeps its
        // old mesh until they are in, so no hole shows through them.
        let (around_new, taken) = if pending.meshes.is_empty() {
            (FxHashSet::default(), TakenColumns::default())
        } else {
            let new = pending
                .meshes
                .keys()
                .filter(|key| !self.lods.contains_key(key))
                .chain(self.lod_tracker.unshown.iter());
            let around_new = ancestors_of(new);
            let taken = TakenColumns::new(pending.cutout.as_ref(), &self.lod_cutout);
            (around_new, taken)
        };
        while let Some(pending) = &mut self.lod_pending {
            // Keys meshed again keep their place; removed ones are skipped.
            let next = std::iter::from_fn(|| pending.order.pop_front())
                .find(|key| pending.meshes.contains_key(key));
            let Some(key) = next else {
                let pending = self.lod_pending.take().expect("pending tiles");
                self.lod_tracker.finish_batch(
                    &mut self.lods,
                    pending.removed,
                    pending.cutout.as_ref(),
                    &self.lod_cutout,
                    start,
                );
                if let Some(cutout) = pending.cutout {
                    self.set_lod_cutout(cutout);
                }
                return;
            };
            let mesh = pending.meshes.remove(&key).expect("pending tile");
            let keep = mesh.keep
                || around_new.contains(&key)
                || taken.overlaps(key, pending.cutout.as_ref(), &self.lod_cutout);
            self.upload_lod(key, &mesh, keep);
            if start.elapsed() >= budget {
                return;
            }
        }
    }

    /// The tiles drawn for good, not fading out (`render-capture`).
    pub(super) fn standing_lods(&self) -> FxHashSet<TileKey> {
        self.lods
            .iter()
            .filter(|(_, tile)| tile.dying.is_none())
            .map(|(&key, _)| key)
            .collect()
    }

    /// The chunk columns the resident tiles now leave out (`Lod::take_cutout`).
    /// Removed sections they no longer leave out are released.
    pub fn set_lod_cutout(&mut self, cutout: crate::fast_hash::FxHashSet<(i32, i32)>) {
        self.lod_cutout = cutout;
        let released: Vec<_> = self
            .lod_retiring
            .iter()
            .copied()
            .filter(|key| !self.lod_cutout.contains(&(key.0, key.2)))
            .collect();
        for key in released {
            self.lod_retiring.remove(&key);
            self.leave_section(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesher::TerrainVertex;

    #[test]
    fn tiles_fade_in_on_their_own_grid() {
        let start = Instant::now();
        let at = |ms| start + Duration::from_millis(ms);
        // terrain.wgsl's `faded` reads the level and sixty-fourths back.
        let decode = |code: u32| ((code - 1) / 129, (code - 1) % 129);
        assert_eq!(fade_code(None, 0, at(0)), None, "waits for its job");
        assert_eq!(fade_code(Some(start), 3, at(0)).map(decode), Some((3, 1)));
        let mut last = 0;
        for ms in [0, 100, 300, 500, 590] {
            let (level, shown) = decode(fade_code(Some(start), 10, at(ms)).unwrap());
            assert_eq!(level, 10);
            assert!(shown > last && shown < 64, "{shown} at {ms} ms");
            last = shown;
        }
        assert_eq!(fade_code(Some(start), 10, at(600)), Some(0), "whole");
        // Fading out: 65 to 128, then nothing drawn (128) until released.
        let mut last = 64;
        for ms in [0, 100, 300, 500, 590] {
            let (level, out) = decode(fade_out_code(start, 3, at(ms)));
            assert_eq!(level, 3);
            assert!(out > last && out < 128, "{out} at {ms} ms");
            last = out;
        }
        assert_eq!(decode(fade_out_code(start, 3, at(700))), (3, 128));
        // A section's largest code keeps its UV step exact (terrain_uv).
        let w = 129.0f32 + 1.0 / 32768.0;
        assert_eq!(w.fract(), 1.0 / 32768.0);
    }

    #[test]
    fn joined_jobs_apply_as_if_in_turn() {
        let key = |x| TileKey { x, z: 0, level: 0 };
        let mut pending = Pending::default();
        // The first job meshes 1 and 2 and retires 3 and 4.
        pending.add(
            vec![
                (key(1), SectionMesh::default()),
                (key(2), SectionMesh::default()),
            ],
            vec![key(3), key(4)],
            Some([(0, 0)].into_iter().collect()),
        );
        // The second meshes 3 again and retires 2.
        pending.add(vec![(key(3), SectionMesh::default())], vec![key(2)], None);
        let mut meshes: Vec<_> = pending.meshes.keys().map(|k| k.x).collect();
        meshes.sort_unstable();
        let mut removed: Vec<_> = pending.removed.iter().map(|k| k.x).collect();
        removed.sort_unstable();
        assert_eq!((meshes, removed), (vec![1, 3], vec![2, 4]));
        assert!(pending.cutout.as_ref().is_some_and(|c| c.contains(&(0, 0))));
        // They go up in the order handed over, each once.
        pending.add(vec![(key(1), SectionMesh::default())], vec![], None);
        let order: Vec<_> = pending
            .order
            .iter()
            .filter(|k| pending.meshes.contains_key(k))
            .map(|k| k.x)
            .collect();
        assert_eq!(order, [1, 3]);
    }

    #[test]
    fn large_tile_bounds_include_water_and_do_not_cull_by_origin() {
        let vertex = |pos| TerrainVertex::tiled(pos, [0.0; 2], 0, 0, 0);
        let mesh = SectionMesh {
            solid: vec![vertex([0.0, -64.0, 0.0]), vertex([1024.0, 100.0, 1024.0])],
            translucent: vec![vertex([512.0, 120.0, 512.0])],
            bounds: Some([Vec3::new(0.0, -64.0, 0.0), Vec3::new(1024.0, 120.0, 1024.0)]),
            ..Default::default()
        };
        let bounds = Bounds::mesh(&mesh).unwrap();
        assert_eq!(bounds.min, Vec3::new(0.0, -64.0, 0.0));
        assert_eq!(bounds.max, Vec3::new(1024.0, 120.0, 1024.0));
        let view = Frustum {
            planes: super::super::frustum(
                super::super::world_projection(70.0, 16.0 / 9.0, 8192.0)
                    * crate::camera::view_matrix(0.0, 0.0),
            ),
        };
        // The tile origin is far behind, but its far edge reaches the camera.
        // Negative world origins therefore remain visible.
        assert!(
            bounds
                .visible(DVec3::new(-1000.0, -80.0, -512.0), &view)
                .is_some()
        );
        // Wholly behind the camera: out of view.
        assert!(
            bounds
                .visible(DVec3::new(-512.0, -80.0, -2000.0), &view)
                .is_none()
        );
        assert!(Bounds::mesh(&SectionMesh::default()).is_none());
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct MockTile {
        shown: Option<Instant>,
        dying: Option<Instant>,
        alone: bool,
        previous: Option<(u32, Option<Instant>)>,
    }

    impl LodTileState for MockTile {
        fn set_shown(&mut self, shown: Instant) {
            self.shown = Some(shown);
        }
        fn dying(&self) -> Option<Instant> {
            self.dying
        }
        fn set_dying(&mut self, since: Instant, alone: bool) {
            self.dying = Some(since);
            self.alone = alone;
        }
        fn start_previous(&mut self, since: Instant) {
            if let Some((_, started)) = &mut self.previous {
                started.get_or_insert(since);
            }
        }
    }

    fn reference_finish_batch(
        lods: &mut FxHashMap<TileKey, MockTile>,
        removed: &FxHashSet<TileKey>,
        pending_cutout: Option<&FxHashSet<(i32, i32)>>,
        current_cutout: &FxHashSet<(i32, i32)>,
        start: Instant,
    ) {
        let replaced: FxHashSet<TileKey> = removed
            .iter()
            .copied()
            .filter(|key| lods.contains_key(key))
            .collect();
        let above_replaced = ancestors_of(&replaced);
        let mut kept: FxHashSet<TileKey> = lods
            .iter()
            .filter(|(_, tile)| tile.shown.is_none())
            .map(|(&key, _)| key)
            .collect();
        let cut = pending_cutout.unwrap_or(current_cutout);
        kept.extend(cut.iter().map(|&(x, z)| TileKey { x, z, level: 0 }));
        let above = ancestors_of(&kept);
        let whole = start.checked_sub(FADE).unwrap_or(start);
        for (&key, tile) in lods.iter_mut() {
            if tile.shown.is_none() {
                tile.shown = Some(if overlaps_set(key, &replaced, &above_replaced) {
                    start
                } else {
                    whole
                });
            }
            if let Some((_, since)) = &mut tile.previous {
                since.get_or_insert(start);
            }
        }
        for &key in removed {
            let alone = !overlaps_set(key, &kept, &above);
            if let Some(tile) = lods.get_mut(&key)
                && tile.dying.is_none()
            {
                tile.dying = Some(start);
                tile.alone = alone;
            }
        }
    }

    fn assert_tracker_matches(tracker: &LodTracker, lods: &FxHashMap<TileKey, MockTile>) {
        let dying: FxHashSet<_> = lods
            .iter()
            .filter(|(_, t)| t.dying.is_some())
            .map(|(&k, _)| k)
            .collect();
        let previous: FxHashSet<_> = lods
            .iter()
            .filter(|(_, t)| t.previous.is_some())
            .map(|(&k, _)| k)
            .collect();
        let unshown: FxHashSet<_> = lods
            .iter()
            .filter(|(_, t)| t.shown.is_none())
            .map(|(&k, _)| k)
            .collect();
        assert_eq!(tracker.dying, dying);
        assert_eq!(tracker.previous, previous);
        assert_eq!(tracker.unshown, unshown);
    }

    #[test]
    fn tracker_calm_frame_is_o1_and_expires_only_tracked_fades() {
        let base = Instant::now();
        let whole = base.checked_sub(FADE).unwrap_or(base);
        let mut lods = FxHashMap::default();
        let mut tracker = LodTracker::default();
        for x in 0..50 {
            for z in 0..50 {
                let key = TileKey { x, z, level: 0 };
                lods.insert(
                    key,
                    MockTile {
                        shown: Some(whole),
                        dying: None,
                        alone: false,
                        previous: None,
                    },
                );
                tracker.record_uploaded(key, true, false);
            }
        }
        assert!(tracker.dying.is_empty());
        assert!(tracker.previous.is_empty());
        assert!(tracker.unshown.is_empty());
        assert_tracker_matches(&tracker, &lods);

        // Calm frame: empty tracking sets return immediately with no allocations.
        let faded = tracker.collect_faded(&lods, base);
        assert!(faded.is_empty());
        assert_eq!(faded.capacity(), 0);
        let expired = tracker.expire_previous(&mut lods, base, |tile, now| match tile.previous {
            Some((id, Some(since))) if now.saturating_duration_since(since) >= FADE => {
                tile.previous = None;
                Some(Some(id))
            }
            Some(_) => Some(None),
            None => None,
        });
        assert!(expired.is_empty());
        assert_eq!(expired.capacity(), 0);

        // Mark one tile dying and another holding a stand-in previous mesh.
        let dying_key = TileKey {
            x: 3,
            z: 4,
            level: 0,
        };
        let standin_key = TileKey {
            x: 7,
            z: 8,
            level: 0,
        };
        lods.get_mut(&dying_key).unwrap().set_dying(base, true);
        tracker.mark_dying(dying_key);
        lods.get_mut(&standin_key).unwrap().previous = Some((42, Some(base)));
        tracker.record_uploaded(standin_key, true, true);
        assert_tracker_matches(&tracker, &lods);

        // Before FADE elapses, neither expires.
        let before = base + FADE - Duration::from_millis(1);
        assert!(tracker.collect_faded(&lods, before).is_empty());
        let still = tracker.expire_previous(&mut lods, before, |tile, now| match tile.previous {
            Some((id, Some(since))) if now.saturating_duration_since(since) >= FADE => {
                tile.previous = None;
                Some(Some(id))
            }
            Some(_) => Some(None),
            None => None,
        });
        assert!(still.is_empty());
        assert_tracker_matches(&tracker, &lods);

        // At FADE, both expire and tracker returns to empty calm state.
        let after = base + FADE;
        let faded = tracker.collect_faded(&lods, after);
        assert_eq!(faded, vec![dying_key]);
        for key in faded {
            lods.remove(&key);
            tracker.remove(&key);
        }
        let released = tracker.expire_previous(&mut lods, after, |tile, now| match tile.previous {
            Some((id, Some(since))) if now.saturating_duration_since(since) >= FADE => {
                tile.previous = None;
                Some(Some(id))
            }
            Some(_) => Some(None),
            None => None,
        });
        assert_eq!(released, vec![42]);
        assert!(tracker.dying.is_empty());
        assert!(tracker.previous.is_empty());
        assert!(tracker.unshown.is_empty());
        assert_tracker_matches(&tracker, &lods);
    }

    #[test]
    fn finish_batch_matches_full_scan_for_empty_and_non_empty_removals() {
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_millis(100);
        let t2 = t0 + Duration::from_millis(200);
        let parent = TileKey {
            x: 0,
            z: 0,
            level: 1,
        };
        let child0 = TileKey {
            x: 0,
            z: 0,
            level: 0,
        };
        let child1 = TileKey {
            x: 1,
            z: 0,
            level: 0,
        };
        let distant = TileKey {
            x: 10,
            z: 10,
            level: 1,
        };
        let cutout_tile = TileKey {
            x: -3,
            z: -4,
            level: 0,
        };
        let both_unshown_and_removed = TileKey {
            x: 5,
            z: 5,
            level: 0,
        };

        let mut fast = FxHashMap::default();
        let mut tracker = LodTracker::default();
        let upload = |lods: &mut FxHashMap<TileKey, MockTile>,
                      tracker: &mut LodTracker,
                      key: TileKey,
                      keep: bool,
                      mesh_id: u32| {
            let (shown, previous) = match lods.remove(&key) {
                Some(old) if old.dying.is_none() => match old.previous {
                    Some(prev) => (old.shown, Some(prev)),
                    None if keep => (old.shown, Some((mesh_id, None))),
                    None => (old.shown, None),
                },
                _ => (None, None),
            };
            tracker.record_uploaded(key, shown.is_some(), previous.is_some());
            lods.insert(
                key,
                MockTile {
                    shown,
                    dying: None,
                    alone: false,
                    previous,
                },
            );
        };

        // Batch 1: pending.removed is empty (fast path) with unshown tiles + stand-in + large cutout.
        upload(&mut fast, &mut tracker, parent, false, 1);
        upload(&mut fast, &mut tracker, distant, false, 2);
        upload(&mut fast, &mut tracker, cutout_tile, false, 3);
        upload(&mut fast, &mut tracker, parent, true, 10);
        let mut ref_lods = fast.clone();

        let empty_removed = FxHashSet::default();
        let cutout_a: FxHashSet<(i32, i32)> = [(-3, -4), (0, 1)].into_iter().collect();
        let current_cutout = FxHashSet::default();
        tracker.finish_batch(
            &mut fast,
            empty_removed.clone(),
            Some(&cutout_a),
            &current_cutout,
            t0,
        );
        reference_finish_batch(
            &mut ref_lods,
            &empty_removed,
            Some(&cutout_a),
            &current_cutout,
            t0,
        );
        assert_eq!(fast, ref_lods);
        assert_tracker_matches(&tracker, &fast);
        // Empty removal shows new tiles whole immediately and arms stand-ins.
        let whole_t0 = t0.checked_sub(FADE).unwrap_or(t0);
        assert_eq!(fast[&parent].shown, Some(whole_t0));
        assert_eq!(fast[&parent].previous, Some((10, Some(t0))));

        // Batch 2: non-empty removed covering:
        // - `parent` replaced by finer `child0` & `child1` (over_replaced=true, alone=false)
        // - `distant` removed with nothing replacing it (alone=true)
        // - `cutout_tile` removed under near-chunk cutout (alone=false)
        // - `both_unshown_and_removed` uploaded then removed in the same batch
        // - absent key not in `lods`
        upload(&mut fast, &mut tracker, child0, false, 20);
        upload(&mut fast, &mut tracker, child1, false, 21);
        upload(&mut fast, &mut tracker, both_unshown_and_removed, false, 22);
        let mut ref_lods = fast.clone();
        let removed: FxHashSet<TileKey> = [
            parent,
            distant,
            cutout_tile,
            both_unshown_and_removed,
            TileKey {
                x: 99,
                z: 99,
                level: 2,
            },
        ]
        .into_iter()
        .collect();
        tracker.finish_batch(&mut fast, removed.clone(), None, &cutout_a, t1);
        reference_finish_batch(&mut ref_lods, &removed, None, &cutout_a, t1);
        assert_eq!(fast, ref_lods);
        assert_tracker_matches(&tracker, &fast);
        assert_eq!(
            fast[&child0].shown,
            Some(t1),
            "fades in over replaced parent"
        );
        assert_eq!(fast[&parent].dying, Some(t1));
        assert!(!fast[&parent].alone, "replaced by child0/child1");
        assert_eq!(fast[&distant].dying, Some(t1));
        assert!(fast[&distant].alone, "out of range with no replacement");
        assert_eq!(fast[&cutout_tile].dying, Some(t1));
        assert!(!fast[&cutout_tile].alone, "covered by cutout column");
        assert_eq!(fast[&both_unshown_and_removed].shown, Some(t1));
        assert_eq!(fast[&both_unshown_and_removed].dying, Some(t1));
        assert!(!fast[&both_unshown_and_removed].alone);

        // Batch 3: re-removing already-dying keys must not overwrite dying timestamps.
        let mut ref_lods = fast.clone();
        tracker.finish_batch(&mut fast, removed.clone(), None, &cutout_a, t2);
        reference_finish_batch(&mut ref_lods, &removed, None, &cutout_a, t2);
        assert_eq!(fast, ref_lods);
        assert_tracker_matches(&tracker, &fast);
        assert_eq!(fast[&distant].dying, Some(t1));
    }

    #[test]
    fn taken_columns_matches_linear_scan_across_levels_and_signs() {
        let current: FxHashSet<(i32, i32)> = [(-2, -2), (0, 0), (1, 1)].into_iter().collect();
        let cutout: FxHashSet<(i32, i32)> = [(-5, -3), (-2, -2), (0, 0), (3, -1), (4, 5)]
            .into_iter()
            .collect();
        let taken = TakenColumns::new(Some(&cutout), &current);
        let naive_list: Vec<_> = cutout
            .iter()
            .filter(|c| !current.contains(c))
            .copied()
            .collect();

        for level in 0..=4u8 {
            for x in -8..=8 {
                for z in -8..=8 {
                    let key = TileKey { x, z, level };
                    let [ox, _, oz] = key.origin();
                    let size = key.size();
                    let expected = naive_list.iter().any(|&(cx, cz)| {
                        (ox..ox + size).contains(&(cx * 16)) && (oz..oz + size).contains(&(cz * 16))
                    });
                    assert_eq!(
                        taken.overlaps(key, Some(&cutout), &current),
                        expected,
                        "mismatch for {key:?}"
                    );
                }
            }
        }
        let empty = TakenColumns::new(None, &current);
        assert!(!empty.overlaps(
            TileKey {
                x: 0,
                z: 0,
                level: 0,
            },
            None,
            &current,
        ));
    }
}

/// Whether `HORIZON_FAR_CROSSFADE=1` asks for the old cross-fade of replaced far nodes (a debug switch: it
/// draws two far surfaces over the same ground while they fade, the z-fighting the single-cut swap removed).
fn cross_fade() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("HORIZON_FAR_CROSSFADE").is_ok_and(|v| v == "1"))
}

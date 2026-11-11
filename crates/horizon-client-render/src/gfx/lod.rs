//! Far meshes share the terrain arenas and passes, with their actual bounds
//! rather than the 16³ boxes used by the near-section residency walk.
use super::{
    Frustum, Renderer, SectionGpu, cube_far,
    cube_gen::GpuBackend,
    surface_tile::{self, TileInfo as SurfaceInfo, TileKey as SurfaceKey},
};
use super::reveal::{self, Reveal};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
    lod::{
        MAX_LEVEL, TileKey,
        cube::{CubeKey, GpuBatch, TileWork},
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

/// Finished node meshes not uploaded yet, each with the job (`CubeJobs`) it came in.
#[derive(Default)]
pub(super) struct CubePending {
    meshes: FxHashMap<CubeKey, (SectionMesh, u64)>,
    order: VecDeque<CubeKey>,
}

impl CubePending {
    /// Newer meshes replace waiting ones; nodes removed drop their waiting meshes. Returns the
    /// jobs of the meshes dropped, which are not coming.
    fn add(
        &mut self,
        meshes: Vec<(CubeKey, SectionMesh)>,
        job: u64,
        removed: &[CubeKey],
    ) -> Vec<u64> {
        let mut dropped = Vec::new();
        for (key, mesh) in meshes {
            match self.meshes.insert(key, (mesh, job)) {
                Some((_, old)) => dropped.push(old),
                None => self.order.push_back(key),
            }
        }
        for key in removed {
            dropped.extend(self.meshes.remove(key).map(|(_, job)| job));
        }
        dropped
    }
}

/// The jobs `queue_cube_lods` hands over, a job the nodes finished in one `CubeLod::update` and
/// the nodes it lets go, and when each part fades. A job fades in, and its replaced nodes go,
/// once every node of it and of the jobs before it is uploaded: not once nothing at all is waiting
/// (a camera that keeps moving keeps a job or two behind, and the far terrain would stand
/// as it was, then change at once, as `cube_far::Plans` fixed for the GPU nodes).
#[derive(Default)]
pub(super) struct CubeJobs {
    plans: cube_far::Plans,
    /// Nodes uploaded and not yet fading in, with their jobs.
    unshown: FxHashMap<CubeKey, u64>,
    /// Nodes to retire, with the job that replaces them.
    removed: FxHashMap<CubeKey, u64>,
}

impl CubeJobs {
    /// A job of `meshes` meshes and the nodes `removed` with it: its number.
    fn queue(&mut self, meshes: usize, removed: &[CubeKey]) -> u64 {
        let job = self.plans.for_batch(meshes);
        (0..meshes).for_each(|_| self.plans.add(job));
        for &key in removed {
            self.removed.insert(key, job);
        }
        job
    }
    /// Meshes of these jobs are not coming (replaced while waiting, or removed).
    fn dropped(&mut self, jobs: impl IntoIterator<Item = u64>) {
        jobs.into_iter().for_each(|job| self.plans.land(job));
    }
    /// A mesh of `job` is uploaded: a node new to the view waits to fade in.
    fn uploaded(&mut self, key: CubeKey, job: u64, new: bool) {
        self.plans.land(job);
        self.removed.remove(&key);
        if new {
            self.unshown.insert(key, job);
        }
    }
    fn forget(&mut self, key: CubeKey) {
        self.unshown.remove(&key);
        self.removed.remove(&key);
    }
    /// The nodes to begin fading in and those to retire now: the jobs that are whole.
    fn due(&mut self) -> (Vec<CubeKey>, Vec<CubeKey>) {
        let whole = self.plans.whole();
        let take = |map: &mut FxHashMap<CubeKey, u64>| {
            let keys: Vec<CubeKey> = map
                .iter()
                .filter(|&(_, &job)| job < whole)
                .map(|(&key, _)| key)
                .collect();
            keys.iter().for_each(|key| {
                map.remove(key);
            });
            keys
        };
        (take(&mut self.unshown), take(&mut self.removed))
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
    /// Its mesh before near chunks took some of its columns, or before finer
    /// tiles came in over it (`Stand`).
    previous: Option<Box<Stand>>,
    /// Near chunks came in over columns it draws, or left them
    /// (`Renderer::open_window`): the shell dithers in front of them for `FADE`.
    window: Option<Window>,
    /// Its slot in occlusion culling's results (`cull::TILE`), kept while
    /// meshes of the same tile replace one another.
    pub cull_slot: u32,
}

/// Whether a near cube that comes in or leaves opens a reveal window: a far node stands whole over
/// it (`cover`), fine enough to have a shell (`level`), the cube has quads (`visible`), and
/// `HORIZON_REVEAL_OFF` (`off`) is not set.
fn wants_window(cover: Option<bool>, level: Option<u8>, visible: bool, off: bool) -> bool {
    cover == Some(true)
        && level.is_some_and(|level| level <= reveal::MAX_SHELL_LEVEL)
        && visible
        && !off
}

/// What far terrain did with its fades, counted behind the environment
/// variable `HORIZON_FADE_STATS` and logged once a second (`FadeStats::frame`):
/// how many tiles and cube nodes were drawn with a fade code, and how each kind
/// of replacement went, with or without one.
#[derive(Default)]
pub(super) struct FadeStats {
    on: bool,
    began: Option<Instant>,
    frames: u32,
    /// Most drawn in a frame this second: tiles with a code (own fade, stand-in
    /// out or in front), the parts drawn in front of near sections, cube nodes.
    max: [u32; 3],
    /// Frames with any fading tile, with a part in front, with a fading node.
    any: [u32; 3],
    /// Frames' drawn tiles, and nodes (summed).
    drawn: [u64; 2],
    /// Tiles shown over the ones they replace, over nothing (whole).
    new: [u32; 2],
    /// Removed tiles that fade out, that stand whole under finer ones.
    removed: [u32; 2],
    /// Tiles meshed again: replaced at once, a stand-in kept whole (`Over`), a
    /// stand-in dithering out (`Out`).
    remeshed: [u32; 3],
    /// Windows in which tiles dither out over sections that came in, in over
    /// sections that leave.
    windows: [u32; 2],
    /// The most tiles with a window open at a frame.
    window_open: u32,
    /// Cube nodes (CPU) new, and removed.
    nodes: [u32; 2],
    /// GPU cube nodes: began to fade in, began to go, meshed again in place, new
    /// (`CubeFar::take_fade_events`).
    gpu_nodes: [u32; 4],
    /// Near cubes that came in over a node standing whole (a window), over none, and over one
    /// not whole yet (whole at once); then the same for cubes that left (`Renderer::open_cube_window`).
    pub cubes: [u32; 6],
    /// Cubes that came in or left with no quads at all (nothing to show), not in `cubes`.
    pub cubes_empty: [u32; 2],
    /// The most windows open at a frame, and the most shells drawn (CPU nodes).
    windows_max: [u32; 2],
}

impl FadeStats {
    pub(super) fn on(&self) -> bool {
        self.on
    }
    pub(super) fn from_env() -> Self {
        Self {
            on: std::env::var_os("HORIZON_FADE_STATS").is_some(),
            ..Self::default()
        }
    }
    /// A frame's drawn far parts, logging the second's totals when it is over.
    pub(super) fn frame(
        &mut self,
        visible: &[Visible],
        gpu_nodes: Option<([u32; 2], [u32; 4])>,
        open: u32,
        windows: u32,
    ) {
        if !self.on {
            return;
        }
        let now = Instant::now();
        let began = *self.began.get_or_insert(now);
        self.frames += 1;
        self.window_open = self.window_open.max(open);
        self.windows_max[0] = self.windows_max[0].max(windows);
        let mut count = [0u32; 3];
        let mut drawn = [0u64; 2];
        let mut shells = 0;
        for v in visible {
            shells += u32::from(v.fade & reveal::REVEAL != 0);
            let cube = matches!(v.key, FarKey::Cube(_));
            drawn[usize::from(cube)] += 1;
            if v.fade != 0 {
                count[if cube { 2 } else { 0 }] += 1;
                if v.fade >= FRONT {
                    count[1] += 1;
                }
            }
        }
        self.windows_max[1] = self.windows_max[1].max(shells);
        if let Some(([fading_in, fading_out], events)) = gpu_nodes {
            count[2] += fading_in + fading_out;
            for (sum, event) in self.gpu_nodes.iter_mut().zip(events) {
                *sum += event;
            }
        }
        for i in 0..3 {
            self.max[i] = self.max[i].max(count[i]);
            self.any[i] += u32::from(count[i] > 0);
        }
        self.drawn[0] += drawn[0];
        self.drawn[1] += drawn[1];
        if now - began >= Duration::from_secs(1) {
            log::info!(
                "far fade: {} frames; tiles drawn {} avg, fading max {} ({} frames), in front max {} ({} frames); \
                 cube nodes fading max {} ({} frames); tiles new {} fading, {} whole; removed {} fading out, {} \
                 standing; meshed again {} at once, {} stand-in whole, {} stand-in dithering out; \
                 windows in front of near sections: {} out, {} in, {} open at most; cpu cube nodes new {}, removed {}; gpu cube nodes {} began to fade in, {} began to go, {} meshed again in place, {} new; \
                 near cubes: {} came in over a shell, {} over no node, {} over a node not whole yet; {} left under one, {} with no node, {} under one not whole yet; \
                 at most {} cube windows open, {} cpu shells drawn; cubes with no quads (not counted above): {} came in, {} left",
                self.frames,
                self.drawn[0] / u64::from(self.frames),
                self.max[0],
                self.any[0],
                self.max[1],
                self.any[1],
                self.max[2],
                self.any[2],
                self.new[0],
                self.new[1],
                self.removed[0],
                self.removed[1],
                self.remeshed[0],
                self.remeshed[1],
                self.remeshed[2],
                self.windows[1],
                self.windows[0],
                self.window_open,
                self.nodes[0],
                self.nodes[1],
                self.gpu_nodes[0],
                self.gpu_nodes[1],
                self.gpu_nodes[2],
                self.gpu_nodes[3],
                self.cubes[0],
                self.cubes[1],
                self.cubes[2],
                self.cubes[3],
                self.cubes[4],
                self.cubes[5],
                self.windows_max[0],
                self.windows_max[1],
                self.cubes_empty[0],
                self.cubes_empty[1],
            );
            *self = Self {
                on: true,
                began: Some(now),
                ..Self::default()
            };
        }
    }
}

/// A tile's mesh from before its re-meshing, still drawn after it (`Tile::previous`).
pub(super) struct Stand {
    mesh: SectionGpu,
    /// When its fade began: the job's end (the instant) for `Over`, none while
    /// the rest of the job uploads; the window's start for `Out`. It goes
    /// `FADE` after.
    since: Option<Instant>,
    mode: StandMode,
}

/// The time in which a tile dithers in front of near sections that stay whole
/// (`Tile::window`).
#[derive(Clone, Copy)]
pub(super) struct Window {
    since: Instant,
    /// Near sections came in over it (the tile dithers out, revealing them),
    /// else they left and it dithers in over them.
    out: bool,
}

impl Window {
    fn active(self, now: Instant) -> bool {
        now.saturating_duration_since(self.since) < fade_time()
    }
}

/// What a stand-in is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum StandMode {
    /// Finer tiles come in over it (or it was clipped around chunks that stay
    /// as they were): it stands whole in the mesh's place until they are whole,
    /// and the mesh waits.
    Over,
    /// Near chunks came in over it (`Window::out`) and its job meshed it
    /// again with their columns cut out, within the window: the mesh is drawn
    /// whole behind, and the stand-in, which still has the columns, goes on
    /// dithering out in front of the chunks, the holes of its pattern showing
    /// them whole.
    Out,
}

/// How long a tile takes to fade in over the ones it replaces.
pub(super) const FADE: Duration = Duration::from_millis(600);
/// `FADE`, stretched by the number in the environment variable `HORIZON_FADE_SLOW`
/// (for looking at a fade: the GPU-driven far terrain keeps its own time).
pub(super) fn fade_time() -> Duration {
    static TIME: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *TIME.get_or_init(|| {
        let slow = std::env::var("HORIZON_FADE_SLOW")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v >= 1.0);
        slow.map_or(FADE, |slow| FADE.mul_f32(slow))
    })
}

/// Whether the environment variable `HORIZON_REVEAL_OFF` is set: near cubes then come in and
/// leave over far cube nodes without shells, as before (for measuring what the shells cost).
fn reveal_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("HORIZON_REVEAL_OFF").is_some())
}

/// A tile fading in: 0 drawn whole, else 1 + `level` * 129 +
/// `shown`, `shown` 1 to 63 sixty-fourths of `FADE` (`fade_out_code`: 65 to
/// 128, fading out). Chunk by chunk (columns of its own grid, wider for
/// coarse tiles: 2^`level` blocks), each starting in turn over the first half
/// and fading over the second, as a fine ordered pattern fixed to the terrain
/// (terrain.wgsl's `faded`). It is drawn a hair toward the camera, the newest
/// most, over the solid tiles it replaces, which stay until it is whole
/// (`upload_lods`): no pixel fought over, no hole. None: nothing to draw yet.
/// Near sections never carry one; `FRONT` adds where a tile dithers over them.
pub(super) fn fade_code(shown: Option<Instant>, level: u8, now: Instant) -> Option<u32> {
    let faded = now.saturating_duration_since(shown?).as_secs_f32() / fade_time().as_secs_f32();
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
/// `fade_code` for a tile leaving since `since`, chunk by chunk as it came,
/// over what stays behind it; once `FADE` has passed it draws nothing.
pub(super) fn fade_out_code(since: Instant, level: u8, now: Instant) -> u32 {
    let faded = now.saturating_duration_since(since).as_secs_f32() / fade_time().as_secs_f32();
    1 + u32::from(level) * 129 + 64 + ((faded * 64.0) as u32 + 1).min(64)
}

/// Added to a tile's fade code to draw it in front of near sections
/// (terrain.wgsl's `FRONT`): a far shell that dithers in or out over chunks
/// that stay whole, every hole of its pattern showing finished terrain.
pub(super) const FRONT: u32 = 1 << 12;
/// `fade_code` of a tile coming in in front of near sections: `since` is when
/// the sections began to be held, none once `FADE` has passed.
pub(super) fn front_in_code(since: Instant, level: u8, now: Instant) -> Option<u32> {
    let faded = now.saturating_duration_since(since).as_secs_f32() / fade_time().as_secs_f32();
    // Never 0 within `FADE`: whole is the plain mesh drawn behind, and the
    // sections held in front of that one go as this one ends.
    (faded < 1.0).then(|| FRONT + 1 + u32::from(level) * 129 + ((faded * 64.0) as u32 + 1).min(63))
}
/// `fade_out_code` of a stand-in dithering out in front of near sections.
pub(super) fn front_out_code(since: Instant, level: u8, now: Instant) -> u32 {
    FRONT + fade_out_code(since, level, now)
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
    /// It draws the tile's stand-in (`Tile::previous`), not its mesh.
    pub stand_in: bool,
    /// Its water is drawn. Not the part that dithers in front of near sections
    /// (`FRONT`): the mesh drawn with it has the same, and a second layer would
    /// blend into it, opaque.
    pub water: bool,
    /// It draws the mesh of the entry before it, the shell in front of it: flat draws
    /// (`flat.rs`) have one instance for a mesh, so only the whole part gets one.
    pub shares_mesh: bool,
    /// Occlusion culling may test it (`Tile::cull_bounds`): a tile drawn whole,
    /// alone.
    pub testable: bool,
}

impl Tile {
    /// The mesh drawn in the tile's place this frame: the previous one while it
    /// stands in whole (`StandMode::Over`, or the tile dies), else its own.
    pub fn drawn(&self) -> &SectionGpu {
        match self.previous.as_deref() {
            Some(stand) if stand.mode == StandMode::Over || self.dying.is_some() => &stand.mesh,
            _ => &self.mesh,
        }
    }
    /// Whether it stands drawn whole at `now` (a shell may dither in front of near cubes
    /// over it): faded in, and not going alone.
    pub fn whole(&self, now: Instant) -> bool {
        self.shown
            .is_some_and(|since| now.saturating_duration_since(since) >= fade_time())
            && !(self.dying.is_some() && self.alone)
    }
    /// Whether a window is open at `now` (`HORIZON_FADE_STATS`).
    pub fn window_open(&self, now: Instant) -> bool {
        self.window.is_some_and(|window| window.active(now))
    }
    /// The mesh a `Visible` draws.
    pub fn mesh_of(&self, v: &Visible) -> &SectionGpu {
        match self.previous.as_deref() {
            Some(stand) if v.stand_in => &stand.mesh,
            _ => &self.mesh,
        }
    }
    pub fn mesh_of_mut(&mut self, v: &Visible) -> &mut SectionGpu {
        match self.previous.as_deref_mut() {
            Some(stand) if v.stand_in => &mut stand.mesh,
            _ => &mut self.mesh,
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
    /// Both meshes it holds in the arena: its own and the stand-in, if any.
    pub fn meshes_mut(&mut self) -> impl Iterator<Item = &mut SectionGpu> {
        std::iter::once(&mut *self.mesh).chain(self.previous.as_deref_mut().map(|s| &mut s.mesh))
    }
    /// What it draws this frame, if it is in view: one entry, or two while it
    /// dithers in front of near sections (`window`): what it draws whole, and
    /// the shell in front, drawn again with a fade code (`Visible::water`
    /// false). For a window out over a stand-in (`StandMode::Out`) the shell is
    /// the stand-in and the whole one the mesh, cut around the near columns.
    pub fn visible(
        &self,
        key: FarKey,
        camera: DVec3,
        frustum: &Frustum,
        now: Instant,
        reveal: Option<&Reveal>,
        out: &mut Vec<Visible>,
    ) {
        let level = key.level();
        let fade = match self.dying {
            Some(since) if self.alone => fade_out_code(since, level, now),
            _ => match fade_code(self.shown, level, now) {
                Some(fade) => fade,
                None => return,
            },
        };
        let origin = DVec3::from(key.origin().map(f64::from)) - camera;
        let Some(distance) = self.bounds.visible(origin, frustum) else {
            return;
        };
        let origin = origin.as_vec3().to_array();
        let entry = |fade, stand_in, old, water, testable, shares_mesh| Visible {
            key,
            distance,
            origin,
            fade,
            old,
            stand_in,
            water,
            shares_mesh,
            testable,
        };
        // Standing in over finer tiles, or dying: one mesh in the tile's place.
        let over = self.dying.is_some()
            || self
                .previous
                .as_deref()
                .is_some_and(|stand| stand.mode == StandMode::Over);
        let replaced = self.dying.is_some() || self.previous.is_some();
        // A tile that goes (replaced by finer ones, which it stays whole under, or alone) is
        // still under its window, as long as it lasts: it goes `FADE` after the tile began.
        let window = self.window.filter(|window| window.active(now));
        let tested = window.is_none() && self.previous.is_none();
        if over {
            out.push(entry(fade, replaced, replaced, true, window.is_none(), false));
        } else {
            out.push(entry(fade, false, false, true, tested, false));
        }
        // A far cube node standing whole is drawn again, a shell, where near cubes come in
        // or leave in front of it (`Renderer::open_cube_window`): once for each kind, since
        // the two lie at different depths (terrain.wgsl's `fade_depth`).
        if let Some(reveal) = reveal
            && fade == 0
            && level <= reveal::MAX_SHELL_LEVEL
            && !reveal.is_empty()
        {
            let lo = DVec3::from(key.origin().map(f64::from));
            let [min, max] = [lo + self.bounds.min.as_dvec3(), lo + self.bounds.max.as_dvec3()];
            let touching = reveal.touching(min, max, reveal::margin(level));
            for out_kind in [true, false] {
                if touching[usize::from(!out_kind)] {
                    out.push(entry(reveal::shell_code(level, out_kind), false, false, false, false, true));
                }
            }
        }
        let Some(window) = window else {
            return;
        };
        let stand_in = over
            || (window.out
                && self
                    .previous
                    .as_deref()
                    .is_some_and(|stand| stand.mode == StandMode::Out));
        let code = if window.out {
            front_out_code(window.since, level, now)
        } else {
            match front_in_code(window.since, level, now) {
                Some(code) => code,
                None => return,
            }
        };
        let base_stand_in = over && replaced;
        out.push(entry(code, stand_in, false, false, false, stand_in == base_stand_in));
    }
}

impl Renderer {
    /// A tile meshed again (finer detail) takes its place at once, but for
    /// one meshed around finer tiles that are new (`over`), whose old mesh
    /// stands in whole until they are (`Stand`), and one meshed while near
    /// chunks come in over it (`Window::out`), whose old mesh goes on
    /// dithering out in front of them. A new one waits for the rest of its
    /// job (`upload_lods`), then fades in.
    fn upload_lod(&mut self, key: TileKey, mesh: &SectionMesh, over: bool) {
        if let Some(copies) = &mut self.capture_copies {
            copies.tile(key, mesh);
        }
        let now = Instant::now();
        let old_slot = self.lods.get(&key).map(|tile| tile.cull_slot);
        let mut window = None;
        let (shown, previous) = match self.lods.remove(&key) {
            Some(old) if old.dying.is_none() => {
                window = old.window.filter(|window| window.active(now));
                let out = window.filter(|window| window.out);
                let stand = if over {
                    Some((StandMode::Over, None))
                } else {
                    out.map(|window| (StandMode::Out, Some(window.since)))
                };
                self.fade_stats.remeshed[match (&old.previous, &stand) {
                    (None, None) => 0,
                    (Some(stand), _) if stand.mode == StandMode::Over => 1,
                    (None, Some((StandMode::Over, _))) => 1,
                    _ => 2,
                }] += 1;
                match old.previous {
                    // Kept as it was: its job's end starts it, once.
                    Some(previous) => {
                        self.release_mesh(*old.mesh);
                        (old.shown, Some(previous))
                    }
                    None => match stand {
                        Some((mode, since)) => (
                            old.shown,
                            Some(Box::new(Stand {
                                mesh: *old.mesh,
                                since,
                                mode,
                            })),
                        ),
                        None => {
                            self.release_mesh(*old.mesh);
                            (old.shown, None)
                        }
                    },
                }
            }
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
                self.lods.insert(
                    key,
                    Tile {
                        mesh: Box::new(mesh),
                        bounds,
                        shown,
                        dying: None,
                        alone: false,
                        previous,
                        window,
                        cull_slot,
                    },
                );
            }
            None => {
                if let Some(stand) = previous {
                    self.release_mesh(stand.mesh);
                }
                self.lod_cull_free.extend(old_slot);
            }
        }
    }

    fn release_tile(&mut self, tile: Tile) {
        self.release_mesh(*tile.mesh);
        if let Some(stand) = tile.previous {
            self.release_mesh(stand.mesh);
        }
    }

    pub fn remove_lod(&mut self, key: TileKey) {
        if let Some(copies) = &mut self.capture_copies {
            copies.forget_tile(key);
        }
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
        // No far terrain left to fade out over.
        for (key, since) in std::mem::take(&mut self.lod_leaving) {
            if self
                .sections
                .find(key)
                .is_some_and(|s| s.leaving == Some(since))
            {
                self.release_section(key);
            }
        }
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
    }

    /// A node meshed again keeps its slot and fade; a new one waits for the
    /// rest of its job (`upload_cube_lods`) before it fades in.
    fn upload_cube_lod(&mut self, key: CubeKey, mesh: &SectionMesh, job: u64) {
        let (old_slot, shown) = match self.lod_cubes.remove(&key) {
            Some(old) => {
                let kept = (old.cull_slot, old.shown.filter(|_| old.dying.is_none()));
                self.release_tile(old);
                (Some(kept.0), kept.1)
            }
            None => {
                self.fade_stats.nodes[0] += 1;
                (None, None)
            }
        };
        // (a node new to the view, or one meshed again before it faded in, waits for its job)
        self.cube_jobs.uploaded(key, job, shown.is_none());
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
                        window: None,
                        cull_slot,
                    },
                );
            }
            None => self.lod_cull_free.extend(old_slot),
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
        self.cube_jobs.forget(key);
        if let Some(tile) = self.lod_cubes.remove(&key) {
            self.lod_cull_free.push(tile.cull_slot);
            self.release_tile(tile);
        }
    }

    /// Drops every cube tile and what waits to upload.
    pub fn clear_cube_lods(&mut self) {
        self.cube_pending = None;
        self.cube_jobs = Default::default();
        self.reveal.clear();
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
        let job = self.cube_jobs.queue(meshes.len(), &removed);
        let dropped = self
            .cube_pending
            .get_or_insert_default()
            .add(meshes, job, &removed);
        self.cube_jobs.dropped(dropped);
    }

    /// Uploads queued node meshes for about `budget` (at least one). Once all
    /// are in they fade in together, and the nodes they replace stand whole
    /// for `FADE` and go: nothing opens a hole while the new ones come in.
    pub fn upload_cube_lods(&mut self, budget: Duration) {
        let start = Instant::now();
        if let Some(far) = &mut self.cube_far {
            far.upload();
        }
        self.reveal.expire(start, fade_time());
        self.release_held_cubes(start);
        self.release_left(start);
        let faded: Vec<_> = self
            .lod_cubes
            .iter()
            .filter(|(_, tile)| tile.dying.is_some_and(|d| start - d >= fade_time()))
            .map(|(&key, _)| key)
            .collect();
        for key in faded {
            self.remove_cube_lod(key);
        }
        while let Some(pending) = &mut self.cube_pending {
            let next = std::iter::from_fn(|| pending.order.pop_front())
                .find(|key| pending.meshes.contains_key(key));
            let Some(key) = next else {
                self.cube_pending = None;
                break;
            };
            let (mesh, job) = pending.meshes.remove(&key).expect("pending node");
            self.upload_cube_lod(key, &mesh, job);
            if start.elapsed() >= budget {
                break;
            }
        }
        // The jobs that are whole: their nodes fade in, and the nodes they replace go.
        let (shown, removed) = self.cube_jobs.due();
        for key in shown {
            if let Some(tile) = self.lod_cubes.get_mut(&key) {
                tile.shown.get_or_insert(start);
            }
        }
        for key in removed {
            if let Some(tile) = self.lod_cubes.get_mut(&key)
                && tile.dying.is_none()
            {
                self.fade_stats.nodes[1] += 1;
                tile.dying = Some(start);
            }
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
        let faded: Vec<_> = self
            .lods
            .iter()
            .filter(|(_, tile)| tile.dying.is_some_and(|d| start - d >= fade_time()))
            .map(|(&key, _)| key)
            .collect();
        for key in faded {
            self.remove_lod(key);
        }
        let stood_in: Vec<_> = self
            .lods
            .values_mut()
            .filter_map(|tile| match tile.previous.as_deref() {
                Some(Stand {
                    since: Some(since), ..
                }) if start - *since >= fade_time() => tile.previous.take(),
                _ => None,
            })
            .collect();
        for stand in stood_in {
            self.release_mesh(stand.mesh);
        }
        for tile in self.lods.values_mut() {
            if tile.window.is_some_and(|window| !window.active(start)) {
                tile.window = None;
            }
        }
        // The tiles holding the job's new finer ones: a tile meshed again around
        // them keeps its old mesh until they are in, so no hole shows through
        // them.
        let mut around_new = FxHashSet::default();
        let new = self
            .lod_pending
            .iter()
            .flat_map(|pending| pending.meshes.keys())
            .filter(|key| !self.lods.contains_key(key))
            .chain(
                self.lods
                    .iter()
                    .filter(|(_, tile)| tile.shown.is_none())
                    .map(|(key, _)| key),
            );
        for &key in new {
            let mut key = key;
            while key.level < MAX_LEVEL && around_new.insert(key.parent()) {
                key = key.parent();
            }
        }
        while let Some(pending) = &mut self.lod_pending {
            // Keys meshed again keep their place; removed ones are skipped.
            let next = std::iter::from_fn(|| pending.order.pop_front())
                .find(|key| pending.meshes.contains_key(key));
            let Some(key) = next else {
                let pending = self.lod_pending.take().expect("pending tiles");
                // A tile fades in over the ones it replaces. One over nothing
                // shows whole: its hidden chunks would open holes to the sky
                // or the sea bed (dark patches, as far terrain first loads).
                let replaced: FxHashSet<TileKey> = pending
                    .removed
                    .iter()
                    .copied()
                    .filter(|key| self.lods.contains_key(key))
                    .collect();
                let mut above_replaced = FxHashSet::default();
                for &key in &replaced {
                    let mut key = key;
                    while key.level < MAX_LEVEL && above_replaced.insert(key.parent()) {
                        key = key.parent();
                    }
                }
                let over_replaced = |key: TileKey| {
                    let mut ancestor = key;
                    replaced.contains(&key)
                        || above_replaced.contains(&key)
                        || (key.level..MAX_LEVEL).any(|_| {
                            ancestor = ancestor.parent();
                            replaced.contains(&ancestor)
                        })
                };
                // A removed tile no new tile overlaps, and no near chunk takes
                // a column of, leaves the plan (out of range): it fades out.
                // Over near chunks fading in it stays whole: two patterns
                // fading at once left holes between them.
                // Two tiles overlap when one is the other or its ancestor:
                // `kept` holds the new tiles and the near chunks' columns
                // (level 0), `above` their ancestors.
                let mut kept: FxHashSet<TileKey> = self
                    .lods
                    .iter()
                    .filter(|(_, tile)| tile.shown.is_none())
                    .map(|(&key, _)| key)
                    .collect();
                let cut = pending.cutout.as_ref().unwrap_or(&self.lod_cutout);
                kept.extend(cut.iter().map(|&(x, z)| TileKey { x, z, level: 0 }));
                let mut above = FxHashSet::default();
                for &key in &kept {
                    let mut key = key;
                    while key.level < MAX_LEVEL && above.insert(key.parent()) {
                        key = key.parent();
                    }
                }
                let alone = |key: TileKey| {
                    let mut ancestor = key;
                    !kept.contains(&key)
                        && !above.contains(&key)
                        && !(key.level..MAX_LEVEL).any(|_| {
                            ancestor = ancestor.parent();
                            kept.contains(&ancestor)
                        })
                };
                let whole = start.checked_sub(fade_time()).unwrap_or(start);
                for (&key, tile) in &mut self.lods {
                    if tile.shown.is_none() {
                        let over = over_replaced(key);
                        self.fade_stats.new[usize::from(!over)] += 1;
                        tile.shown = Some(if over { start } else { whole });
                    }
                    if let Some(stand) = tile.previous.as_deref_mut() {
                        stand.since.get_or_insert(start);
                    }
                }
                for key in pending.removed {
                    let alone = alone(key);
                    if let Some(tile) = self.lods.get_mut(&key)
                        && tile.dying.is_none()
                    {
                        self.fade_stats.removed[usize::from(!alone)] += 1;
                        tile.dying = Some(start);
                        tile.alone = alone;
                    }
                }
                if let Some(cutout) = pending.cutout {
                    self.set_lod_cutout(cutout);
                }
                return;
            };
            let mesh = pending.meshes.remove(&key).expect("pending tile");
            // Finer tiles come in over it: it stands whole until they are.
            let over = mesh.finer || around_new.contains(&key);
            self.upload_lod(key, &mesh, over);
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
        let now = Instant::now();
        let mut windows = FxHashMap::default();
        for key in released {
            self.lod_retiring.remove(&key);
            let column = (key.0, key.2);
            let since = *windows
                .entry(column)
                .or_insert_with(|| self.open_window(column, now, false));
            self.leave_section(key, since);
        }
    }

    /// Whether far cube nodes are drawn here, on the CPU or the GPU path.
    pub(super) fn cubic_far(&self) -> bool {
        !self.lod_cubes.is_empty() || self.cube_far.as_ref().is_some_and(|far| far.drawn() > 0)
    }

    /// Whether the far cube node over section `key`'s place stands whole at `now`: `Some`
    /// with that, or `None` where no node is drawn there.
    fn cube_cover(&self, key: super::SectionKey, now: Instant) -> Option<bool> {
        self.cube_cover_level(key, now).0
    }

    /// `cube_cover`, and the level of the node standing whole there, if one does.
    fn cube_cover_level(&self, key: super::SectionKey, now: Instant) -> (Option<bool>, Option<u8>) {
        let pos = horizon_core::position::SectionPos {
            x: key.0,
            y: (self.layout.min_y() >> 4) + key.1,
            z: key.2,
        };
        let mut cover = None;
        for level in 0..=crate::lod::cube::MAX_CUBE_LEVEL {
            let node = CubeKey::containing(level, pos);
            let whole = self
                .lod_cubes
                .get(&node)
                .filter(|tile| tile.shown.is_some())
                .map(|tile| tile.whole(now))
                .or_else(|| self.cube_far.as_ref()?.cover(node));
            match whole {
                // (a node standing whole over it, whatever finer ones came in meanwhile)
                Some(true) => return (Some(true), Some(level)),
                Some(false) => cover = Some(false),
                None => {}
            }
        }
        (cover, None)
    }

    /// A cube at section `key` comes in (`out`) or leaves, over a far cube node that
    /// stands whole there: a window for the node's shell to dither in front of it
    /// (`reveal`, `Tile::visible`, `CubeFar::prepare`). The cube itself never dithers.
    /// Returns when the window began, which a cube that leaves is held until; none (and no
    /// window) where no node stands whole to dither: the cube is whole at once, or goes.
    pub(super) fn open_cube_window(
        &mut self,
        key: super::SectionKey,
        now: Instant,
        out: bool,
        visible: bool,
    ) -> Option<Instant> {
        let (cover, level) = self.cube_cover_level(key, now);
        // (a node too coarse to look like the terrain has no shell: the cube is whole at once;
        // nor has a cube with no quads, all air or buried: the node stays drawn behind it
        // either way, and its window would only take a slot from the cubes that show)
        let shell = wants_window(cover, level, visible, reveal_off());
        // (`visible`: the cube has quads. One with none, all air or buried, pops nothing, and
        // is counted apart so the counts mean visible pops.)
        if visible {
            self.fade_stats.cubes[usize::from(!out) * 3 + cover.map_or(1, |whole| usize::from(!whole) * 2)] += 1;
        } else {
            self.fade_stats.cubes_empty[usize::from(!out)] += 1;
        }
        if !shell {
            return None;
        }
        let lo = DVec3::new(
            f64::from(key.0) * 16.0,
            f64::from(self.layout.min_y() + key.1 * 16),
            f64::from(key.2) * 16.0,
        );
        Some(self.reveal.open(lo, lo + 16.0, now, out))
    }

    /// Cubes of removed columns, held whole while far nodes come in under them
    /// (`remove_columns`), let go once a node stands whole under each: the nodes dither
    /// in in front of them and they are released with the windows (`release_left`).
    /// Held for `HOLD` at most: past it they go at once.
    fn release_held_cubes(&mut self, now: Instant) {
        if self.cube_held.is_empty() {
            return;
        }
        const HOLD: Duration = Duration::from_secs(5);
        let columns: Vec<((i32, i32), Instant)> =
            self.cube_held.iter().map(|(&c, &at)| (c, at)).collect();
        let mut released = Vec::new();
        for ((x, z), removed) in columns {
            let held: Vec<super::SectionKey> = (0..self.layout.sections() as i32)
                .map(|sy| (x, sy, z))
                .filter(|key| self.lod_retiring.contains(key))
                .collect();
            let timed_out = now.saturating_duration_since(removed) >= HOLD;
            if !timed_out && !held.iter().all(|&key| self.cube_cover(key, now) == Some(true)) {
                continue;
            }
            self.cube_held.remove(&(x, z));
            for key in held {
                self.lod_retiring.remove(&key);
                match self.open_cube_window(key, now, false, true) {
                    Some(since) => self.leave_section(key, since),
                    None => released.extend(self.sections.remove(key)),
                }
            }
        }
        self.release_meshes(released);
    }

    /// Near sections of `column` come in over far terrain (`out`), or leave it,
    /// held whole until it has come in over them (`leave_section`): the tile
    /// that draws the column draws its shell a second time in front of them,
    /// dithering out or in over `FADE`, which shows them or hides them one pixel
    /// at a time; sections never dither themselves, their pattern showed the
    /// caves inside. A window already open is joined, not restarted: what it
    /// has shown stays shown. Returns when the window began, which sections
    /// that leave hold until; now, without a tile to dither.
    pub(super) fn open_window(&mut self, (x, z): (i32, i32), now: Instant, out: bool) -> Instant {
        for level in 0..=MAX_LEVEL {
            let key = TileKey {
                x: x >> level,
                z: z >> level,
                level,
            };
            if let Some(tile) = self.lods.get_mut(&key) {
                let open = tile.window.filter(|window| window.active(now));
                self.fade_stats.windows[usize::from(out)] += u32::from(open.is_none());
                let window = *tile.window.insert(open.unwrap_or(Window { since: now, out }));
                return window.since;
            }
        }
        now
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesher::TerrainVertex;
    const REVEAL_BIT: u32 = reveal::REVEAL;

    #[test]
    fn only_cubes_with_quads_over_a_fine_whole_node_open_a_window() {
        let fine = reveal::MAX_SHELL_LEVEL;
        assert!(wants_window(Some(true), Some(0), true, false));
        assert!(wants_window(Some(true), Some(fine), true, false));
        // An empty cube (all air or buried) opens none, so it takes no slot of the table.
        assert!(!wants_window(Some(true), Some(0), false, false));
        // Too coarse, not whole yet, no node, or the shells off: none either.
        assert!(!wants_window(Some(true), Some(fine + 1), true, false));
        assert!(!wants_window(Some(false), None, true, false));
        assert!(!wants_window(None, None, true, false));
        assert!(!wants_window(Some(true), Some(0), true, true));
        // The table is not touched by cubes that open nothing: only visible ones fill it.
        let mut reveal = reveal::Reveal::default();
        let now = Instant::now();
        for i in 0..200 {
            let lo = DVec3::new(f64::from(i) * 200.0, 0.0, 0.0);
            if wants_window(Some(true), Some(0), i % 2 == 0, false) {
                reveal.open(lo, lo + 16.0, now, true);
            }
        }
        assert_eq!(reveal.len(), 64.min(100));
        assert_eq!(reveal.evicted, 100 - 64);
    }

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

    /// A tile at the origin, whole since long ago, as `Tile::visible` sees it.
    fn tile(previous: Option<Stand>, window: Option<Window>, now: Instant) -> Tile {
        let section = super::super::tests::section;
        Tile {
            mesh: Box::new(section(8, 2, None)),
            bounds: Bounds {
                min: Vec3::ZERO,
                max: Vec3::splat(16.0),
            },
            shown: Some(now - 10 * FADE),
            dying: None,
            alone: false,
            previous: previous.map(Box::new),
            window,
            cull_slot: 0,
        }
    }
    fn stand(mode: StandMode, since: Option<Instant>) -> Stand {
        Stand {
            mesh: super::super::tests::section(9, 3, None),
            since,
            mode,
        }
    }
    fn drawn(tile: &Tile, now: Instant) -> Vec<Visible> {
        let all = Frustum {
            planes: [glam::Vec4::W; 6],
        };
        let mut out = Vec::new();
        tile.visible(FarKey::Tile(TileKey { x: 0, z: 0, level: 0 }), DVec3::ZERO, &all, now, None, &mut out);
        out
    }
    /// What terrain.wgsl's `faded` and `fade_depth` read back from a code: whether it is drawn in
    /// front of near sections, its level, whether it fades out, and its sixty-fourths.
    fn decode_front(code: u32) -> (bool, u32, bool, u32) {
        let front = code & FRONT != 0;
        let code = code & (FRONT - 1);
        let step = (code - 1) % 129;
        (front, (code - 1) / 129, step >= 64, step - if step >= 64 { 64 } else { 0 })
    }

    #[test]
    fn a_tile_dithers_out_in_front_of_chunks_that_came_in_whole() {
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let window = Some(Window {
            since: t0,
            out: true,
        });
        // Not meshed again: its mesh is drawn twice, whole behind (water and all) and in
        // front, without water, going out over `FADE`; neither is tested by occlusion culling.
        let coming = tile(None, window, t0);
        let mut last = 0;
        for ms in [0, 100, 300, 500, 590] {
            let parts = drawn(&coming, at(ms));
            assert_eq!(parts.len(), 2, "{ms} ms");
            assert!(parts[0].fade == 0 && parts[0].water && !parts[0].stand_in);
            let (front, level, out, gone) = decode_front(parts[1].fade);
            assert!(front && out && level == 0 && !parts[1].water && !parts[1].stand_in);
            // One mesh twice: flat draws have an instance for it, the whole part's.
            assert!(parts[1].shares_mesh && !parts[0].shares_mesh);
            assert!(parts.iter().all(|v| !v.testable));
            assert!(gone > last && gone < 64, "{gone} at {ms} ms");
            last = gone;
        }
        // Over, it is the plain mesh again, tested by occlusion culling.
        let parts = drawn(&coming, at(600));
        assert_eq!(parts.len(), 1);
        assert!(parts[0].fade == 0 && parts[0].testable);
    }

    #[test]
    fn meshed_again_in_the_window_the_old_mesh_goes_on_dithering_out() {
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let window = Some(Window {
            since: t0,
            out: true,
        });
        // The new mesh, cut around the chunks, whole behind; the stand-in, which still has their
        // columns, in front.
        let swapped = tile(Some(stand(StandMode::Out, Some(t0))), window, t0);
        let parts = drawn(&swapped, at(300));
        assert_eq!(parts.len(), 2);
        assert!(!parts[0].stand_in && parts[0].fade == 0 && parts[0].water && !parts[0].old);
        let (front, _, out, _) = decode_front(parts[1].fade);
        assert!(parts[1].stand_in && front && out && !parts[1].water);
        assert!(!parts[1].shares_mesh, "the stand-in is a mesh of its own");
        // The same pattern as before the swap: the code depends on the window alone.
        assert_eq!(parts[1].fade, drawn(&tile(None, window, t0), at(300))[1].fade);
        // Once the window is over only the mesh is left (the stand-in is released by then).
        assert_eq!(drawn(&swapped, at(700)).len(), 1);
    }

    #[test]
    fn a_stand_in_for_finer_tiles_stands_whole_in_the_mesh_place() {
        let t0 = Instant::now();
        let over = tile(Some(stand(StandMode::Over, Some(t0))), None, t0);
        let parts = drawn(&over, t0 + FADE / 2);
        assert_eq!(parts.len(), 1);
        assert!(parts[0].stand_in && parts[0].old && parts[0].water && parts[0].fade == 0);
        // A tile that goes draws its stand-in whichever the mode. The window is still open
        // over it, its shell in front; one that fades out alone has its own dither as well.
        let window = Some(Window {
            since: t0,
            out: true,
        });
        let mut dying = tile(Some(stand(StandMode::Out, Some(t0))), window, t0);
        dying.dying = Some(t0);
        let parts = drawn(&dying, t0);
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().all(|v| v.stand_in));
        assert_eq!(parts[1].fade & FRONT, FRONT);
        dying.alone = true;
        let parts = drawn(&dying, t0 + FADE / 2);
        assert_eq!(parts.len(), 2);
        assert!(parts[0].fade != 0 && parts[0].fade & FRONT == 0, "its own fade-out");
        assert_eq!(parts[1].fade & FRONT, FRONT);
    }

    #[test]
    fn far_terrain_dithers_in_over_sections_held_whole() {
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let coming = tile(
            None,
            Some(Window {
                since: t0,
                out: false,
            }),
            t0,
        );
        let mut last = 0;
        for ms in [0, 100, 300, 500, 599] {
            let parts = drawn(&coming, at(ms));
            // The plain mesh stays behind, its water too; the part in front has none.
            assert_eq!(parts.len(), 2, "{ms} ms");
            assert!(parts[0].fade == 0 && parts[0].water && !parts[0].testable);
            let (front, level, out, shown) = decode_front(parts[1].fade);
            assert!(front && !out && level == 0 && !parts[1].water && !parts[1].stand_in);
            assert!(shown > last && shown < 64, "{shown} at {ms} ms");
            last = shown;
        }
        // The window over, it is the plain mesh again, tested by occlusion culling.
        let parts = drawn(&coming, at(600));
        assert_eq!(parts.len(), 1);
        assert!(parts[0].fade == 0 && parts[0].testable);
        // A tile whose level is coarser dithers on its own grid.
        assert_eq!(front_in_code(t0, 5, t0).map(decode_front).unwrap().1, 5);
    }

    #[test]
    fn the_front_flag_leaves_the_dither_code_alone() {
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        for ms in [0, 250, 590] {
            let plain = decode_front(fade_out_code(t0, 3, at(ms)));
            let front = decode_front(front_out_code(t0, 3, at(ms)));
            assert_eq!((plain.0, front.0), (false, true));
            assert_eq!((plain.1, plain.2, plain.3), (front.1, front.2, front.3));
        }
        // A tile's instance word keeps the whole code, for the largest level and the flag.
        let code = front_in_code(t0, MAX_LEVEL, t0).unwrap();
        let w = instance([0.0; 3], code)[3];
        assert_eq!((-w).floor() as u32, code);
        assert_eq!(front_in_code(t0, 0, at(600)), None);
    }

    /// The depth `fade_depth` gives a vertex `distance` away, per instance word `w`, run on the
    /// GPU with the terrain shader's own function. Reversed depth: nearer is larger.
    fn depths(cases: &[(f32, f32)]) -> Option<Vec<f32>> {
        use wgpu::util::DeviceExt;
        let instance = super::super::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).ok()?;
        let source = super::super::terrain_shader()
            + "@group(3) @binding(0) var<storage, read> probes: array<vec4<f32>>;\n\
               @group(3) @binding(1) var<storage, read_write> depths: array<f32>;\n\
               @compute @workgroup_size(1) fn probe_depth(@builtin(global_invocation_id) id: vec3<u32>) {\n\
                   let p = probes[id.x];\n\
                   let clip = fade_depth(vec4(0.0, 0.0, 0.05, p.x), p.y);\n\
                   depths[id.x] = clip.z / clip.w;\n\
               }\n";
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let probes: Vec<[f32; 4]> = cases.iter().map(|&(d, w)| [d, w, 0.0, 0.0]).collect();
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&probes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bytes = (cases.len() * 4) as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("probe_depth"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(3),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(3, &group, &[]);
            pass.dispatch_workgroups(cases.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
        queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let view = staging.slice(..).get_mapped_range().ok()?;
        Some(bytemuck::cast_slice::<u8, f32>(&view).to_vec())
    }

    /// terrain.wgsl's depth rule for far shells drawn in front of near sections: out over
    /// sections that came in whole (`front_out_code`) they are just ahead of them, more so the
    /// coarser the tile, so their seals and skirts, buried under the near surface, stay hidden;
    /// in over held sections (`HELD`, 2% ahead of far terrain) they are ahead of those. What
    /// stands clearly nearer in front of the terrain hides them either way.
    #[test]
    fn a_far_shell_in_front_wins_the_depth_test_over_near_sections_but_not_what_stands_nearer() {
        let t0 = Instant::now();
        let section = |fade: u32| super::super::section_instance([0.0; 3], fade, 1.0)[3];
        let tile = |fade: u32| instance([0.0; 3], fade)[3];
        let half = t0 + FADE / 2;
        let d = 100.0;
        let cases = [
            // The sections: whole, held, and 10% nearer.
            (d, section(0)),
            (d, section(HELD)),
            (d * 0.9, section(0)),
            // Out, level 0: at the same distance, and 0.5% farther (its seals are deeper).
            (d, tile(front_out_code(t0, 0, half))),
            (d * 1.005, tile(front_out_code(t0, 0, half))),
            // Out, level 6: 2% farther still wins, a coarse tile's surface being that far off.
            (d * 1.02, tile(front_out_code(t0, 6, half))),
            // In, level 0: over held sections at the same distance, 1.5% farther than whole ones.
            (d, tile(front_in_code(t0, 0, half).unwrap())),
            (d * 1.015, tile(front_in_code(t0, 0, half).unwrap())),
            (d * 1.04, tile(front_in_code(t0, 0, half).unwrap())),
            // Without the flag, a tile stays behind a section at the same distance.
            (d, tile(0)),
            (d, tile(fade_out_code(t0, 6, half))),
        ];
        let Some(depth) = depths(&cases) else {
            eprintln!("SKIP far shell depth: no GPU");
            return;
        };
        let [whole, held, nearer, out, out_deep, out_coarse, into_held, into_whole, into_far, plain, plain_out] =
            depth[..]
        else {
            unreachable!()
        };
        assert!(out > whole, "{out} vs {whole}");
        assert!(out_deep < whole, "a seal 0.5% deep stays hidden: {out_deep} vs {whole}");
        assert!(out_coarse > whole, "{out_coarse} vs {whole}");
        assert!(into_held > held, "{into_held} vs {held}");
        assert!(into_whole > whole && into_far < whole, "{into_whole} {into_far} vs {whole}");
        // Whatever stands 10% nearer hides the shell in every case.
        assert!([out, out_coarse, into_held].iter().all(|&shell| shell < nearer));
        assert!(plain < whole && plain_out < whole);
    }

    /// A far cube node standing whole is drawn again as a shell where a window of near cubes
    /// meets it, one for each kind, and not where none does, nor while it fades in.
    #[test]
    fn a_cube_node_draws_a_shell_where_a_window_meets_it() {
        use crate::lod::cube::CubeKey;
        let t0 = Instant::now();
        let all = Frustum {
            planes: [glam::Vec4::W; 6],
        };
        let key = FarKey::Cube(CubeKey::new(2, 0, 0, 0));
        let parts = |tile: &Tile, reveal: &Reveal| {
            let mut out = Vec::new();
            tile.visible(key, DVec3::ZERO, &all, t0, Some(reveal), &mut out);
            out
        };
        let node = tile(None, None, t0);
        let mut reveal = Reveal::default();
        assert_eq!(parts(&node, &reveal).len(), 1, "no window: the node alone");
        // A cube that came in at the node's corner: a shell dithering out, no water, over the
        // node's own mesh, at its level.
        reveal.open(DVec3::splat(4.0), DVec3::splat(20.0), t0, true);
        let drawn = parts(&node, &reveal);
        assert_eq!(drawn.len(), 2);
        assert_eq!(drawn[1].fade, reveal::shell_code(2, true));
        assert!(!drawn[1].water && drawn[1].shares_mesh && !drawn[1].testable);
        assert!(drawn[0].fade == 0 && drawn[0].water);
        // The kinds are separate shells.
        reveal.open(DVec3::splat(4.0), DVec3::splat(20.0), t0, false);
        let kinds: Vec<u32> = parts(&node, &reveal).iter().map(|v| v.fade).collect();
        assert_eq!(kinds[1..], [reveal::shell_code(2, true), reveal::shell_code(2, false)]);
        // Nodes coarser than `MAX_SHELL_LEVEL` get no shell: it would be a slab in front of the
        // terrain it hands over to.
        let coarse = FarKey::Cube(CubeKey::new(reveal::MAX_SHELL_LEVEL + 1, 0, 0, 0));
        let mut out = Vec::new();
        node.visible(coarse, DVec3::ZERO, &all, t0, Some(&reveal), &mut out);
        assert_eq!(out.len(), 1, "a coarse node draws alone");
        // Far from the node: none. While the node fades in, or goes alone: none.
        let mut far = Reveal::default();
        far.open(DVec3::splat(500.0), DVec3::splat(516.0), t0, true);
        assert_eq!(parts(&node, &far).len(), 1);
        let mut fading = tile(None, None, t0);
        fading.shown = Some(t0);
        assert_eq!(parts(&fading, &reveal).len(), 1);
        let mut leaving = tile(None, None, t0);
        leaving.dying = Some(t0);
        leaving.alone = true;
        assert_eq!(parts(&leaving, &reveal).len(), 1);
        assert!(!leaving.whole(t0) && node.whole(t0));
        // What terrain.wgsl reads back from the code.
        let (_, level, out, _) = decode_front(reveal::shell_code(2, true));
        assert!(reveal::shell_code(2, true) & REVEAL_BIT != 0 && level == 2 && out);
        assert_eq!(reveal::shell_code(22, false) & !(REVEAL_BIT - 1), REVEAL_BIT);
    }

    /// terrain.wgsl's depth rule for cube shells (`REVEAL`): ahead of the cubes they dither
    /// over by three eighths of a voxel of their own and a sliver of the distance, and no more:
    /// a shell is closed, and a face inside it lies a voxel behind its surface at the least,
    /// which must not win over a near surface (at most half a voxel behind the shell's, as
    /// far terrain's coarse stand-in is off the real one) in the holes of its pattern: the cave
    /// under the grass, seen through. Run on the GPU with the terrain shader's own function.
    #[test]
    fn a_cube_shell_does_not_show_what_lies_inside_it_through_its_holes() {
        let t0 = Instant::now();
        let node = |code: u32| instance([0.0; 3], code)[3];
        let section = |fade: u32| super::super::section_instance([0.0; 3], fade, 1.0)[3];
        let mut cases = Vec::new();
        let grid: Vec<(u8, f32)> = (0u8..=8)
            .flat_map(|level| [16.0f32, 64.0, 128.0, 200.0, 300.0].map(move |d| (level, d)))
            .collect();
        for &(level, d) in &grid {
            let v = (1u32 << level) as f32;
            // The shell's surface, a face a voxel inside it, and a section's at the same place.
            for out in [true, false] {
                cases.push((d, node(reveal::shell_code(level, out))));
                cases.push((d + v, node(reveal::shell_code(level, out))));
            }
            cases.push((d, section(0)));
            cases.push((d, section(HELD)));
            // A section's surface half a voxel behind the shell's, or at once with it (level 0).
            cases.push((d + 0.5 * v, section(0)));
            cases.push((d + 0.5 * v, section(HELD)));
            cases.push((d * 0.9, section(0)));
        }
        let Some(depth) = depths(&cases) else {
            eprintln!("SKIP cube shell depth: no GPU");
            return;
        };
        let _ = t0;
        for (i, &(level, d)) in grid.iter().enumerate() {
            let [out_top, out_in, in_top, in_inside, whole, held, whole_half, held_half, nearer] =
                depth[i * 9..i * 9 + 9]
            else {
                unreachable!()
            };
            let v = (1u32 << level) as f32;
            let what = format!("level {level} at {d}");
            // Ahead of what it dithers over where the two are level: out over sections that
            // came in whole, in over cubes held (2% ahead of far terrain).
            assert!(out_top > whole, "{what}: {out_top} vs {whole}");
            assert!(in_top > held, "{what}: {in_top} vs {held}");
            // What stands clearly (10%) nearer than the shell hides it.
            assert!(out_top < nearer && in_top < nearer, "{what}");
            // Where the coarse surface is at most half a voxel off (none at level 0) and
            // the shell's own distance is within what nodes of its level are drawn at (nodes
            // split with distance: a voxel is at most 1/100 of it at the far detail setting that
            // splits least), a face a voxel inside the
            // shell does not win over the section's surface behind its top.
            let off = if level == 0 { 0.0 } else { 0.5 * v };
            if level == 0 || d >= 100.0 * v {
                let (section_depth, held_depth) = if level == 0 { (whole, held) } else { (whole_half, held_half) };
                assert!(out_in < section_depth, "{what} (v {v}, off {off}): {out_in} vs {section_depth}");
                assert!(in_inside < held_depth, "{what} (v {v}, off {off}): {in_inside} vs {held_depth}");
            }
        }
    }

    /// A camera that keeps moving hands over jobs faster than they upload (a slow node source:
    /// `CubeSampler::node` may answer later, so the meshes trickle in): the rule that waited
    /// for nothing at all to be waiting never faded a node in or out until the backlog drained;
    /// each job fades once it and the jobs before it are uploaded.
    #[test]
    fn cube_jobs_fade_as_they_complete_under_a_camera_that_keeps_moving() {
        use crate::lod::cube::CubeKey;
        let mut jobs = CubeJobs::default();
        let mut queue = VecDeque::new();
        let (mut frames_idle, mut shown, mut retired) = (0, 0, 0);
        let mut next = 0;
        // The nodes uploaded and not yet let go.
        let mut standing: VecDeque<CubeKey> = VecDeque::new();
        for _ in 0..100 {
            // Five new nodes a frame, which replace the oldest standing five; four upload a frame.
            let fresh: Vec<CubeKey> = (0..5)
                .map(|_| {
                    next += 1;
                    CubeKey::new(0, next, 0, 0)
                })
                .collect();
            let gone: Vec<CubeKey> = if standing.len() > 20 {
                standing.drain(..5).collect()
            } else {
                Vec::new()
            };
            let job = jobs.queue(fresh.len(), &gone);
            queue.extend(fresh.iter().map(|&key| (key, job)));
            for _ in 0..4 {
                if let Some((key, job)) = queue.pop_front() {
                    jobs.uploaded(key, job, true);
                    standing.push_back(key);
                }
            }
            frames_idle += usize::from(queue.is_empty());
            let (show, retire) = jobs.due();
            shown += show.len();
            retired += retire.len();
        }
        assert_eq!(frames_idle, 0, "the backlog never drains, so the old rule never fades");
        assert!(shown >= 300, "{shown} nodes faded in while the camera moved");
        assert!(retired >= 200, "{retired} nodes went");
        // A job with a node still out holds its own fade and every later one: no hole.
        let mut jobs = CubeJobs::default();
        let a = CubeKey::new(0, 1, 0, 0);
        let b = CubeKey::new(0, 2, 0, 0);
        let old = CubeKey::new(1, 0, 0, 0);
        let first = jobs.queue(2, &[]);
        let second = jobs.queue(1, &[old]);
        let c = CubeKey::new(0, 3, 0, 0);
        jobs.uploaded(a, first, true);
        assert_eq!(jobs.due(), (vec![], vec![]), "the first job is not whole");
        jobs.uploaded(c, first, true);
        let (mut show, retire) = jobs.due();
        show.sort();
        assert_eq!((show, retire), (vec![a, c], vec![]), "the second job still has a node out");
        // The second, the last, has its node in: it fades in and what it replaces goes.
        jobs.uploaded(b, second, true);
        assert_eq!(jobs.due(), (vec![b], vec![old]));
    }
}

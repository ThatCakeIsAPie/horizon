//! The generator's side of the water rules ([`super::water`]): it reads a padded window of samples around the
//! columns it is asked for, hands the window to [`water_window`] and turns the answer into columns.
//!
//! # When the rules run
//!
//! Only with land cover on and a source handed in. Then the sea (with its shelf), the shores and the lakes run
//! always; the rivers and the lakes' recorded levels also need the water atlas
//! ([`EarthGenerator::with_water`]). Without the atlas a lake is flat at the lowest elevation of its nearby
//! water columns ([`LakeLevel::fallback`], decided inside [`water_window`]).
//!
//! # The window
//!
//! For columns `stride` blocks apart the generator reads [`WINDOW_PAD`] more samples on every side, at the same
//! spacing and zoom: elevation surfaces, land cover classes, the nearest river of each sample and the atlas lake
//! that holds it. A column's answer depends only on the samples within [`WINDOW_PAD`] of it, so a near cube, a
//! single block read and a far node that hold the same sample agree on it.
//!
//! # Rivers and lakes from the atlas
//!
//! The reaches are projected into blocks and indexed per 1 degree cell of the atlas, lazily, and kept (at most
//! [`RIVER_CELLS_KEPT`] cells, then the cache starts over). A window asks the cells its latitude and longitude
//! box meets, and the ring of cells around them, so a reach whose line is in the next cell still counts. The
//! nearest river of a sample is the one whose channel edge is nearest over all those cells, with the same choice
//! as [`RiverIndex::nearest`]. Lakes are those the atlas lists for the window's box; a water class sample takes
//! the one [`pick_lake`] chooses.
//!
//! # From the answer to a column
//!
//! * A water column is [`SurfaceKind::Water`] with the depth the rules give; a sand bed turns to gravel where
//!   the shore is cold ([`shore_is_cold`]).
//! * A shore column is sand ([`SurfaceKind::Dune`]), gravel where cold, stone ([`SurfaceKind::Alpine`]) where
//!   steep; snow and marsh keep their own top.
//! * Any other dry column takes its land kind on its final height. A water class column the rules left dry is
//!   shore.

use std::sync::{Arc, Mutex, PoisonError};

use horizon_earth::fastmap::FastMap;
use horizon_earth::{
    ClassWindowPlan, Lake, LandClass, LandCoverSource, WaterAtlas, WindowPlan, WindowPlanError,
};

use super::generator::{Access, EarthGenerator, Gap, Rect};
use super::gpu_window::{WindowJob, WindowTicket};
use super::surface::SurfaceKind;
use super::water::{
    Bed, ColumnWater, LAKE_BOX_MARGIN_DEG, LakeLevel, RiverHit, RiverIndex, WINDOW_PAD,
    WindowInput, pick_lake, shore_is_cold, vertical_blocks_per_metre, water_window,
};

/// How many 1 degree cells of river index the generator keeps before it starts over.
pub const RIVER_CELLS_KEPT: usize = 256;

/// The rings of atlas cells around a window's box whose reaches a window also reads.
const CELL_RING: i32 = 1;

/// The generator's water: the atlas, and the river indexes built from it so far.
#[derive(Default)]
pub(super) struct WaterState {
    pub atlas: Option<&'static WaterAtlas>,
    rivers: Mutex<FastMap<(i32, i32), Arc<RiverIndex>>>,
    /// The river indexes [`EarthGenerator::rivers_near`] gives for a point, by the degree cell of the point:
    /// what a column's channel test reads, without walking the cells of the ring each time.
    near: Mutex<FastMap<(i32, i32), Arc<Vec<Arc<RiverIndex>>>>>,
}

/// What the water rules made of a lattice of columns: one answer and one class per column.
pub(super) struct WaterColumns {
    pub columns: Vec<ColumnWater>,
    pub classes: Vec<LandClass>,
    /// The window's elevation surfaces (the padded `width + 2 * WINDOW_PAD` square, row by row), before the
    /// rules changed any. Its inner part is the columns' own elevation surfaces ([`WaterColumns::own_surfaces`]).
    pub dem: Vec<i32>,
    /// Samples per row of `dem`.
    pub dem_width: usize,
    /// Whether the window starts exactly [`WINDOW_PAD`] samples before the columns on both axes (it does not
    /// when the padding saturated at the edge of the integer range).
    pub aligned: bool,
}

impl WaterColumns {
    /// The `width` by `depth` elevation surfaces of the columns themselves: the window without its padding. They
    /// are the same samples a separate strided read of the columns takes (same points, same zoom). `None` when
    /// the window was not aligned with the columns.
    #[allow(dead_code)]
    pub(super) fn own_surfaces(&self, width: usize, depth: usize) -> Option<Vec<i32>> {
        if !self.aligned {
            return None;
        }
        let mut out = Vec::with_capacity(width * depth);
        for row in WINDOW_PAD..WINDOW_PAD + depth {
            let from = row * self.dem_width + WINDOW_PAD;
            out.extend_from_slice(self.dem.get(from..from + width)?);
        }
        Some(out)
    }
}

/// The window's elevation, either already read or being computed by the hook.
enum WindowDem {
    Ready(Result<Vec<i32>, Gap>),
    /// On the hook, with the class plan when the classes went with it.
    Pending(
        Arc<WindowPlan>,
        Option<Arc<ClassWindowPlan>>,
        Box<dyn WindowTicket>,
    ),
}

impl EarthGenerator {
    /// The same generator with the rivers and lakes of `atlas`. It has an effect only while land cover is
    /// active (the water rules need the classes).
    pub fn with_water(mut self, atlas: &'static WaterAtlas) -> Self {
        self.water.atlas = Some(atlas);
        self.column_memo.clear();
        self
    }

    /// The water atlas in use, if any.
    pub fn water_atlas(&self) -> Option<&'static WaterAtlas> {
        self.water.atlas
    }

    /// The river index of the atlas cell `(lat, lon)` (its south west corner, whole degrees), built on first use.
    fn river_cell(&self, atlas: &WaterAtlas, cell: (i32, i32)) -> Arc<RiverIndex> {
        if let Some(index) = self
            .water
            .rivers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&cell)
        {
            return Arc::clone(index);
        }
        let (lat, lon) = (f64::from(cell.0), f64::from(cell.1));
        let reaches = atlas.reaches_in(lat, lon, lat + 1.0, lon + 1.0);
        let projection = self.field.projection();
        let mpb = projection.metres_per_block();
        let vpm = vertical_blocks_per_metre(self.field.vertical());
        let index = Arc::new(RiverIndex::build(
            &reaches,
            |p| projection.from_lat_lon(p.lat, p.lon),
            mpb,
            vpm,
        ));
        let mut rivers = self
            .water
            .rivers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if rivers.len() >= RIVER_CELLS_KEPT {
            rivers.clear();
        }
        Arc::clone(rivers.entry(cell).or_insert(index))
    }

    /// The latitude and longitude box `(south, west, north, east)` of the block rectangle `r`.
    pub(super) fn lat_lon_box(&self, r: Rect) -> (f64, f64, f64, f64) {
        let projection = self.field.projection();
        let (mut s, mut w, mut n, mut e) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for (x, z) in [(r.x0, r.z0), (r.x1, r.z0), (r.x0, r.z1), (r.x1, r.z1)] {
            let p = projection.to_lat_lon(f64::from(x) + 0.5, f64::from(z) + 0.5);
            s = s.min(p.lat);
            n = n.max(p.lat);
            w = w.min(p.lon);
            e = e.max(p.lon);
        }
        (s, w, n, e)
    }

    /// The river indexes of the atlas cells the box of `r` meets, and the ring around them; empty without the
    /// atlas.
    pub(super) fn rivers_near(&self, r: Rect) -> Vec<Arc<RiverIndex>> {
        let Some(atlas) = self.water.atlas else {
            return Vec::new();
        };
        let (s, w, n, e) = self.lat_lon_box(r);
        let (s, w, n, e) = (
            s.floor() as i32 - CELL_RING,
            w.floor() as i32 - CELL_RING,
            n.floor() as i32 + CELL_RING,
            e.floor() as i32 + CELL_RING,
        );
        let mut out = Vec::new();
        for lat in s.max(-90)..=n.min(89) {
            for lon in w..=e {
                let index = self.river_cell(atlas, (lat, lon));
                if !index.is_empty() {
                    out.push(index);
                }
            }
        }
        out
    }

    /// The nearest river at block `(x, z)` for a lattice of `step` over `indexes`: the hit whose channel edge
    /// is nearest, ties to the lower reach id ([`RiverIndex::nearest`]'s choice).
    pub(super) fn river_hit(indexes: &[Arc<RiverIndex>], x: f64, z: f64, step: f64) -> Option<RiverHit> {
        let mut best: Option<RiverHit> = None;
        for index in indexes {
            if let Some(hit) = index.nearest(x, z, step) {
                let better = best.is_none_or(|b| {
                    (hit.distance - hit.half_width, hit.reach_id)
                        .partial_cmp(&(b.distance - b.half_width, b.reach_id))
                        .is_some_and(|o| o.is_lt())
                });
                if better {
                    best = Some(hit);
                }
            }
        }
        best
    }

    /// The river channel at block `(x, z)`, at one-block resolution: the nearest river when the block's middle
    /// is within its half width. `None` without land cover or the atlas.
    pub(super) fn river_channel_at(&self, x: i32, z: i32) -> Option<RiverHit> {
        if self.land_source().is_none() || self.water.atlas.is_none() {
            return None;
        }
        let r = self.point_rect(x, z, 1);
        let indexes = self.rivers_at_point(r);
        Self::river_hit(&indexes, f64::from(x) + 0.5, f64::from(z) + 0.5, 1.0)
            .filter(|h| h.distance <= h.half_width)
    }

    /// [`EarthGenerator::rivers_near`] of the one-block rectangle `r`, kept by the degree cell of its point:
    /// the answer depends on that cell alone.
    fn rivers_at_point(&self, r: Rect) -> Arc<Vec<Arc<RiverIndex>>> {
        let (s, w, _, _) = self.lat_lon_box(r);
        let cell = (s.floor() as i32, w.floor() as i32);
        let kept = |m: &Mutex<FastMap<(i32, i32), Arc<Vec<Arc<RiverIndex>>>>>| {
            m.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&cell)
                .cloned()
        };
        if let Some(found) = kept(&self.water.near) {
            return found;
        }
        let made = Arc::new(self.rivers_near(r));
        let mut near = self
            .water
            .near
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if near.len() >= RIVER_CELLS_KEPT * 4 {
            near.clear();
        }
        Arc::clone(near.entry(cell).or_insert(made))
    }

    /// The elevation surfaces of `width` by `depth` samples `stride` blocks apart from `(x0, z0)` at `zoom`.
    fn window_surfaces(
        &self,
        a: Access,
        [x0, z0]: [i32; 2],
        [width, depth]: [usize; 2],
        stride: i32,
        zoom: u8,
    ) -> Result<Vec<i32>, Gap> {
        let got = if stride <= 1 {
            match a {
                Access::Local => self.field.surface_ys(x0, z0, width, depth, zoom),
                Access::Fetch => self
                    .field
                    .surface_ys_blocking(x0, z0, width, depth, zoom)
                    .map_err(Gap::Load)?,
            }
        } else {
            if a == Access::Fetch {
                // Loads every tile the rectangle touches, so the strided read below finds them.
                let span = |n: usize| (n.saturating_sub(1) as i64 * i64::from(stride)) as i32;
                self.field
                    .range_blocking(
                        x0,
                        z0,
                        x0.saturating_add(span(width)),
                        z0.saturating_add(span(depth)),
                        zoom,
                    )
                    .map_err(Gap::Load)?;
            }
            self.field
                .surface_ys_strided(x0, z0, width, depth, stride, zoom)
        };
        got.ok_or(Gap::Missing)
    }

    /// Starts reading the window's elevation. A local read goes through the separable plan: offered to the hook
    /// when one is on (the answer is fetched by [`EarthGenerator::finish_window_dem`]), else read on the CPU at
    /// once. A fetching read, and a window too wide for a plan, use [`EarthGenerator::window_surfaces`].
    fn start_window_dem(
        &self,
        a: Access,
        origin: [i32; 2],
        size: [usize; 2],
        stride: i32,
        zoom: u8,
        class: impl FnOnce() -> Option<Arc<ClassWindowPlan>>,
    ) -> WindowDem {
        if a == Access::Fetch {
            return WindowDem::Ready(self.window_surfaces(a, origin, size, stride, zoom));
        }
        let [x0, z0] = origin;
        let [width, depth] = size;
        match self.field.window_plan(x0, z0, width, depth, stride, zoom) {
            Ok(plan) => {
                let plan = Arc::new(plan);
                let pending = self.window_gpu.active().and_then(|gpu| {
                    let class = class();
                    let job = WindowJob {
                        plan: Arc::clone(&plan),
                        class: class.clone(),
                    };
                    gpu.submit(job).map(|ticket| (class, ticket))
                });
                match pending {
                    Some((class, ticket)) => WindowDem::Pending(plan, class, ticket),
                    None => WindowDem::Ready(Ok(self.plan_surfaces(&plan.heights_m()))),
                }
            }
            Err(WindowPlanError::TooManyTiles) => {
                WindowDem::Ready(self.window_surfaces(a, origin, size, stride, zoom))
            }
            Err(WindowPlanError::NotFinite | WindowPlanError::Missing(_)) => {
                WindowDem::Ready(Err(Gap::Missing))
            }
        }
    }

    /// Whether the window's classes went to the hook with its elevation.
    fn classes_on_hook(dem: &WindowDem) -> bool {
        matches!(dem, WindowDem::Pending(_, Some(_), _))
    }

    /// The window's elevation surfaces, and its land cover classes when they went to the hook and it answered
    /// them all: waits for the hook's answer, and reads the window on the CPU from the same plan when the hook has
    /// none (or one of the wrong size).
    fn finish_window_dem(&self, dem: WindowDem) -> Result<(Vec<i32>, Option<Vec<LandClass>>), Gap> {
        match dem {
            WindowDem::Ready(dem) => dem.map(|d| (d, None)),
            WindowDem::Pending(plan, class, ticket) => {
                let expected = plan.cols.len() * plan.rows.len();
                match ticket.wait() {
                    Some(result) if result.heights_m.len() == expected => {
                        let classes = class.zip(result.class_codes).and_then(|(c, codes)| {
                            (codes.len() == c.width * c.depth && codes.len() == expected)
                                .then(|| codes.into_iter().map(LandClass::from_code).collect::<Option<Vec<_>>>())
                                .flatten()
                        });
                        Ok((self.plan_surfaces(&result.heights_m), classes))
                    }
                    _ => Ok((self.plan_surfaces(&plan.heights_m()), None)),
                }
            }
        }
    }

    /// The block heights of elevations in metres.
    fn plan_surfaces(&self, heights_m: &[f32]) -> Vec<i32> {
        heights_m
            .iter()
            .map(|&h| self.field.block_y_of(h))
            .collect()
    }

    /// The water rules over the `width` by `depth` columns `stride` blocks apart from `(x0, z0)`, read at the
    /// zoom and land cover level of `zoom_step`: each column's answer and class.
    pub(super) fn water_columns(
        &self,
        a: Access,
        land: &LandCoverSource,
        [x0, z0]: [i32; 2],
        [width, depth]: [usize; 2],
        stride: i32,
        zoom_step: i32,
    ) -> Result<WaterColumns, Gap> {
        let stride = stride.max(1);
        let pad = WINDOW_PAD as i32 * stride;
        let (px, pz) = (x0.saturating_sub(pad), z0.saturating_sub(pad));
        let (w, h) = (width + 2 * WINDOW_PAD, depth + 2 * WINDOW_PAD);
        let zoom = self.zoom_for_step(zoom_step);
        // The classes go to the hook with the elevation when their plan can be made (every tile it may read is
        // held); otherwise, or without a hook, they are read here, on the CPU.
        let class_plan = || {
            let metres = f64::from(zoom_step.max(1)) * self.field.projection().metres_per_block();
            land.field()
                .class_window_plan(px, pz, w, h, stride, metres)
                .ok()
                .map(Arc::new)
        };
        // Both are asked for before either failure is reported, so one pass names everything the window lacks.
        let dem = self.start_window_dem(a, [px, pz], [w, h], stride, zoom, class_plan);
        let class = (!Self::classes_on_hook(&dem))
            .then(|| self.classes_core(a, land, [px, pz], [w, h], stride, zoom_step));
        // A failed elevation read is reported before a failed class read, as it always was.
        let dem = match dem {
            WindowDem::Ready(Err(gap)) => return Err(gap),
            dem => dem,
        };
        let class = class.transpose()?;
        let span = |n: usize| (n.saturating_sub(1) as i64 * i64::from(stride)) as i32;
        let r = Rect {
            x0: px,
            z0: pz,
            x1: px.saturating_add(span(w)),
            z1: pz.saturating_add(span(h)),
            zoom,
        };
        let at = |i: usize| {
            (
                px.saturating_add((i % w) as i32 * stride),
                pz.saturating_add((i / w) as i32 * stride),
            )
        };
        // The river search needs coordinates only, so it runs while the hook computes the elevation.
        let indexes = self.rivers_near(r);
        let river: Vec<Option<RiverHit>> = if indexes.is_empty() {
            vec![None; w * h]
        } else {
            (0..w * h)
                .map(|i| {
                    let (x, z) = at(i);
                    Self::river_hit(
                        &indexes,
                        f64::from(x) + 0.5,
                        f64::from(z) + 0.5,
                        f64::from(stride),
                    )
                })
                .collect()
        };
        let (dem, hook_class) = self.finish_window_dem(dem)?;
        // The hook's classes, else the CPU's (also when the hook had the classes but did not answer them all).
        let class = match (class, hook_class) {
            (Some(class), _) => class,
            (None, Some(class)) => class,
            (None, None) => self.classes_core(a, land, [px, pz], [w, h], stride, zoom_step)?,
        };
        let vertical = *self.field.vertical();
        let lakes: Vec<Arc<Lake>> = match self.water.atlas {
            Some(atlas) if class.contains(&LandClass::Water) => {
                let (s, w_, n, e) = self.lat_lon_box(r);
                let m = LAKE_BOX_MARGIN_DEG;
                atlas.lakes_in(s - m, w_ - m, n + m, e + m)
            }
            _ => Vec::new(),
        };
        let vpm = vertical_blocks_per_metre(&vertical);
        let projection = self.field.projection();
        let lake: Vec<Option<LakeLevel>> = (0..w * h)
            .map(|i| {
                if lakes.is_empty() || class[i] != LandClass::Water {
                    return None;
                }
                let (x, z) = at(i);
                let p = projection.to_lat_lon(f64::from(x) + 0.5, f64::from(z) + 0.5);
                let elevation = vertical.elevation_for_y(f64::from(dem[i]));
                pick_lake(&lakes, p.lat, p.lon, elevation)
                    .map(|l| LakeLevel::from_lake(l, &vertical, vpm))
            })
            .collect();
        let columns = water_window(&WindowInput {
            width: w,
            rows: h,
            stride,
            vertical,
            dem: &dem,
            class: &class,
            river: &river,
            lake: &lake,
        });
        let mut classes = Vec::with_capacity(width * depth);
        for row in WINDOW_PAD..WINDOW_PAD + depth {
            classes.extend_from_slice(&class[row * w + WINDOW_PAD..row * w + WINDOW_PAD + width]);
        }
        let aligned = x0.checked_sub(pad) == Some(px) && z0.checked_sub(pad) == Some(pz);
        Ok(WaterColumns {
            columns,
            classes,
            dem,
            dem_width: w,
            aligned,
        })
    }

    /// The column `(s, kind)` the water rules' answer `cw` makes at block `(x, z)` of land cover `class`.
    pub(super) fn water_column(
        &self,
        a: Access,
        [x, z]: [i32; 2],
        cw: ColumnWater,
        class: LandClass,
        stride: i32,
        zoom_step: i32,
    ) -> Result<(i32, SurfaceKind), Gap> {
        let cold = || -> Result<bool, Gap> {
            let t = self.temperature_c(a, x, z)?;
            Ok(shore_is_cold(t, self.latitude_at(x, z)))
        };
        if let Some((depth, bed)) = cw.water {
            let bed = if bed == Bed::Sand && cold()? {
                Bed::Gravel
            } else {
                bed
            };
            return Ok((cw.s, SurfaceKind::Water { depth, bed }));
        }
        let shore = |me: &Self| -> Result<SurfaceKind, Gap> {
            Ok(if me.is_steep(a, x, z, cw.s, stride, zoom_step)? {
                SurfaceKind::Alpine
            } else if cold()? {
                SurfaceKind::Gravel
            } else {
                SurfaceKind::Dune
            })
        };
        if class == LandClass::Water {
            return Ok((cw.s, shore(self)?));
        }
        let (s, kind) = self.land_column(a, [x, z], cw.s, stride, zoom_step, class)?;
        // A column the rules leave dry stays dry whatever its class says (no data under the sea level).
        let keeps_own_top = matches!(
            kind,
            SurfaceKind::Snow | SurfaceKind::Glacier | SurfaceKind::Icefall | SurfaceKind::Marsh
        ) || class == LandClass::BuiltUp;
        let kind = if kind.is_water() || s != cw.s || (cw.shore && !keeps_own_top) {
            shore(self)?
        } else {
            kind
        };
        Ok((cw.s, kind))
    }

    /// The mean annual temperature at block `(x, z)` (C), as the dry places read it; `None` without climate.
    fn temperature_c(&self, a: Access, x: i32, z: i32) -> Result<Option<f64>, Gap> {
        Ok(self.dry_column(a, x, z)?.map(|d| d.temperature_c))
    }
}

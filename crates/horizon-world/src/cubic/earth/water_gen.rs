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
//!   steep; snow, marsh and built-up ground keep their own top.
//! * Any other dry column takes its land kind on its final height. A water class column the rules left dry is
//!   shore.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use horizon_earth::{Lake, LandClass, LandCoverSource, WaterAtlas};

use super::generator::{Access, EarthGenerator, Gap, Rect};
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
    rivers: Mutex<HashMap<(i32, i32), Arc<RiverIndex>>>,
}

/// What the water rules made of a lattice of columns: one answer and one class per column.
pub(super) struct WaterColumns {
    pub columns: Vec<ColumnWater>,
    pub classes: Vec<LandClass>,
}

impl EarthGenerator {
    /// The same generator with the rivers and lakes of `atlas`. It has an effect only while land cover is
    /// active (the water rules need the classes).
    pub fn with_water(mut self, atlas: &'static WaterAtlas) -> Self {
        self.water.atlas = Some(atlas);
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
    fn lat_lon_box(&self, r: Rect) -> (f64, f64, f64, f64) {
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
    fn rivers_near(&self, r: Rect) -> Vec<Arc<RiverIndex>> {
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
    fn river_hit(indexes: &[Arc<RiverIndex>], x: f64, z: f64, step: f64) -> Option<RiverHit> {
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
        let indexes = self.rivers_near(r);
        Self::river_hit(&indexes, f64::from(x) + 0.5, f64::from(z) + 0.5, 1.0)
            .filter(|h| h.distance <= h.half_width)
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
        // Both are asked for before either failure is reported, so one pass names everything the window lacks.
        let dem = self.window_surfaces(a, [px, pz], [w, h], stride, zoom);
        let class = self.classes_core(a, land, [px, pz], [w, h], stride, zoom_step);
        let (dem, class) = (dem?, class?);
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
        Ok(WaterColumns { columns, classes })
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
        let keeps_own_top =
            matches!(kind, SurfaceKind::Snow | SurfaceKind::Marsh) || class == LandClass::BuiltUp;
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

//! The separable sampling plan of a window of elevation samples.
//!
//! For the tangent projection a sample's latitude depends only on its `z`, its longitude only on its `x`, and the
//! Mercator pixel `px` only on the longitude and `py` only on the latitude. So a `width` by `depth` lattice of
//! samples needs `width` column axes and `depth` row axes, each computed once with exactly the expressions
//! [`ElevationField::height_m`] uses per point (the same floating point operations in the same order), and every
//! sample is then four gathers and the bilinear mix. A [`WindowPlan`] holds those axes and the tiles they touch,
//! resolved once, so the CPU ([`WindowPlan::heights_m`]) and a GPU (which gets the plan and the tiles' heights) both
//! produce the heights the per-point path gives, bit for bit on the CPU.

use std::sync::Arc;

use crate::height::ElevationField;
use crate::projection::{MAX_LATITUDE, pixel_x, pixel_y, world_pixels};
use crate::tile::{TILE_SIZE, Tile, TileId};

/// A pixel position along one axis: which of the plan's distinct tiles along that axis holds it, and the pixel
/// inside the tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisIndex {
    /// Ordinal into [`WindowPlan::tile_xs`] (for a column axis) or [`WindowPlan::tile_ys`] (for a row axis).
    pub tile: u16,
    /// The pixel inside that tile, `0..256`.
    pub local: u16,
}

/// One column's or row's share of a sample: the bilinear fraction and the two pixels it blends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisSample {
    /// Fraction between pixel `a` and pixel `b` (`fx - floor(fx)` with `fx = pixel position - 0.5`), in `[0, 1)`.
    pub t: f64,
    /// The floor pixel (x wraps round the world, y clamps at the poles' rows).
    pub a: AxisIndex,
    /// The pixel after it (same wrap or clamp).
    pub b: AxisIndex,
}

/// Why a window plan could not be made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WindowPlanError {
    /// Some sample's pixel position is not a finite number ([`ElevationField::height_m`] says `None` there too).
    NotFinite,
    /// These tiles (distinct, in the plan's tile order) are not in the store; nothing was fetched.
    Missing(Vec<TileId>),
    /// More distinct tiles along one axis than a [`AxisIndex`] can number, or a window that reaches the polar cap
    /// (a row past the Mercator limit, where the field folds over the pole and blends to the pole's height, which
    /// the separable axes cannot express): read the samples one by one instead
    /// ([`ElevationField::surface_ys_strided`] does).
    TooManyTiles,
}

/// A window of `width` by `depth` elevation samples, ready to read from tiles that are all held.
///
/// Sample `(i, j)` is at the world point `x = x0 + i * stride + 0.5`, `z = z0 + j * stride + 0.5` and its height is
/// `rows[j]` against `cols[i]`: with `a`/`b` the floor pixel and the next on each axis, the four corner heights
/// `h00 = (cols a, rows a)`, `h10 = (cols b, rows a)`, `h01 = (cols a, rows b)`, `h11 = (cols b, rows b)`,
/// `top = h00 + (h10 - h00) * tx`, `bottom = h01 + (h11 - h01) * tx` and
/// `height = (top + (bottom - top) * ty) as f32` (all in `f64`).
///
/// **Extending later:** a land cover part would add its own axes (or a per-sample warp) next to these fields, not
/// change them.
#[derive(Clone, Debug)]
pub struct WindowPlan {
    /// The zoom the samples are read at (the requested one limited to the field's maximum).
    pub zoom: u8,
    /// Samples along x (the number of column axes).
    pub width: usize,
    /// Samples along z (the number of row axes).
    pub depth: usize,
    /// One axis per column `i`, `width` of them (empty when the window is empty).
    pub cols: Vec<AxisSample>,
    /// One axis per row `j`, `depth` of them (empty when the window is empty).
    pub rows: Vec<AxisSample>,
    /// The distinct tile columns (`TileId::x`) the plan reads, in first-use order along `cols`.
    pub tile_xs: Vec<u32>,
    /// The distinct tile rows (`TileId::y`) the plan reads, in first-use order along `rows`.
    pub tile_ys: Vec<u32>,
    /// Every tile of the product `tile_ys` by `tile_xs`, row-major: index `ty_ordinal * tile_xs.len() +
    /// tx_ordinal`. All of them are read by some sample.
    pub tiles: Vec<(TileId, Arc<Tile>)>,
}

impl WindowPlan {
    /// The tile at the ordinals `(tx, ty)` into [`WindowPlan::tile_xs`] and [`WindowPlan::tile_ys`].
    pub fn tile(&self, tx: u16, ty: u16) -> &(TileId, Arc<Tile>) {
        &self.tiles[usize::from(ty) * self.tile_xs.len() + usize::from(tx)]
    }

    /// The elevations in metres of the whole window, row by row (index `j * width + i`): each bit for bit what
    /// [`ElevationField::height_m`] gives at the sample's point.
    pub fn heights_m(&self) -> Vec<f32> {
        const SIDE: usize = TILE_SIZE as usize;
        let nx = self.tile_xs.len();
        let slices: Vec<&[f32]> = self.tiles.iter().map(|(_, t)| t.heights()).collect();
        let mut out = Vec::with_capacity(self.cols.len() * self.rows.len());
        for row in &self.rows {
            let ty = row.t;
            let (base_a, base_b) = (usize::from(row.a.tile) * nx, usize::from(row.b.tile) * nx);
            let (off_a, off_b) = (
                usize::from(row.a.local) * SIDE,
                usize::from(row.b.local) * SIDE,
            );
            for col in &self.cols {
                let tx = col.t;
                let (ca, cb) = (usize::from(col.a.tile), usize::from(col.b.tile));
                let (la, lb) = (usize::from(col.a.local), usize::from(col.b.local));
                let h00 = f64::from(slices[base_a + ca][off_a + la]);
                let h10 = f64::from(slices[base_a + cb][off_a + lb]);
                let h01 = f64::from(slices[base_b + ca][off_b + la]);
                let h11 = f64::from(slices[base_b + cb][off_b + lb]);
                let top = h00 + (h10 - h00) * tx;
                let bottom = h01 + (h11 - h01) * tx;
                out.push((top + (bottom - top) * ty) as f32);
            }
        }
        out
    }
}

/// Pixel indices on one axis from the fractional world pixel `p`, as [`ElevationField::height_m`] makes them: the
/// fraction, the floor pixel and the next, before they are wrapped or clamped.
fn split(p: f64) -> (f64, i64) {
    // Pixel `i` is centred at `i + 0.5`, so the point sits between pixels `floor(p - 0.5)` and the next.
    let f = p - 0.5;
    let floor = f.floor();
    (f - floor, floor as i64)
}

/// The ordinal of `v` in `list`, which it joins at the end when new; `None` when the ordinal does not fit a u16.
fn ordinal(list: &mut Vec<u32>, v: u32) -> Option<u16> {
    let at = match list.iter().position(|&x| x == v) {
        Some(at) => at,
        None => {
            list.push(v);
            list.len() - 1
        }
    };
    u16::try_from(at).ok()
}

impl ElevationField {
    /// The plan of the `width` by `depth` samples at the points `(x0 + i * stride + 0.5, z0 + j * stride + 0.5)`
    /// read at `zoom` (limited to the field's maximum), over tiles the store already holds. The same samples
    /// [`ElevationField::surface_ys_strided`] reads; a `stride` of 1 is [`ElevationField::heights_m`]'s window.
    /// Nothing is fetched: tiles the store lacks are listed in [`WindowPlanError::Missing`].
    pub fn window_plan(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        stride: i32,
        zoom: u8,
    ) -> Result<WindowPlan, WindowPlanError> {
        let zoom = zoom.min(self.max_zoom());
        let mut plan = WindowPlan {
            zoom,
            width,
            depth,
            cols: Vec::new(),
            rows: Vec::new(),
            tile_xs: Vec::new(),
            tile_ys: Vec::new(),
            tiles: Vec::new(),
        };
        if width == 0 || depth == 0 {
            return Ok(plan);
        }
        let stride = f64::from(stride);
        let size = world_pixels(zoom) as i64;
        let side = i64::from(TILE_SIZE);
        let projection = self.projection();
        let mut finite = true;

        // Longitude depends on x alone and latitude on z alone; the fixed other coordinate is never read.
        let mut cols = Vec::with_capacity(width);
        for i in 0..width {
            let x = f64::from(x0) + (i as f64) * stride + 0.5;
            let lon = projection.to_lat_lon_unwrapped(x, 0.0).lon;
            let px = pixel_x(lon, zoom);
            finite &= px.is_finite();
            let (t, floor) = split(px);
            cols.push((
                t,
                floor.rem_euclid(size),
                floor.saturating_add(1).rem_euclid(size),
            ));
        }
        let mut rows = Vec::with_capacity(depth);
        let mut polar = false;
        for j in 0..depth {
            let z = f64::from(z0) + (j as f64) * stride + 0.5;
            let lat = projection.to_lat_lon_unwrapped(0.0, z).lat;
            // Past the Mercator limit the field folds over the pole and blends to the pole's height, which also
            // moves the longitude by half a lap: not separable into column and row axes.
            polar |= lat.abs() > MAX_LATITUDE;
            let py = pixel_y(lat, zoom);
            finite &= py.is_finite();
            let (t, floor) = split(py);
            rows.push((
                t,
                floor.clamp(0, size - 1),
                floor.saturating_add(1).clamp(0, size - 1),
            ));
        }
        if !finite {
            return Err(WindowPlanError::NotFinite);
        }
        if polar {
            return Err(WindowPlanError::TooManyTiles);
        }

        let mut too_many = false;
        let mut axis = |list: &mut Vec<u32>, (t, a, b): (f64, i64, i64)| {
            let mut index = |g: i64| {
                let tile = ordinal(list, (g / side) as u32);
                too_many |= tile.is_none();
                AxisIndex {
                    tile: tile.unwrap_or(0),
                    local: (g % side) as u16,
                }
            };
            let a = index(a);
            let b = index(b);
            AxisSample { t, a, b }
        };
        plan.cols = cols
            .into_iter()
            .map(|c| axis(&mut plan.tile_xs, c))
            .collect();
        plan.rows = rows
            .into_iter()
            .map(|r| axis(&mut plan.tile_ys, r))
            .collect();
        if too_many {
            return Err(WindowPlanError::TooManyTiles);
        }

        let mut missing = Vec::new();
        let mut tiles = Vec::with_capacity(plan.tile_xs.len() * plan.tile_ys.len());
        for &ty in &plan.tile_ys {
            for &tx in &plan.tile_xs {
                let id = TileId {
                    z: zoom,
                    x: tx,
                    y: ty,
                };
                match self.store().cached(id) {
                    Some(tile) => tiles.push((id, tile)),
                    None => missing.push(id),
                }
            }
        }
        if !missing.is_empty() {
            return Err(WindowPlanError::Missing(missing));
        }
        plan.tiles = tiles;
        Ok(plan)
    }
}

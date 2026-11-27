//! Stage timings of [`EarthGenerator::columns_at`] for the far view's profiling tool
//! (`examples/earth_far_profile.rs`): the same steps as `columns_wide` and `water_columns`, each timed. Not used
//! by the game.

use std::sync::Arc;
use std::time::{Duration, Instant};

use horizon_earth::{Lake, LandClass};

use super::generator::{Access, EarthGenerator, Rect};
use super::water::{
    LAKE_BOX_MARGIN_DEG, LakeLevel, RiverHit, WINDOW_PAD, WindowInput, pick_lake,
    vertical_blocks_per_metre, water_window,
};

/// Time spent in each step of one columns read.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stages {
    /// The node's own elevation surfaces.
    pub surfaces: Duration,
    /// The padded window's elevation surfaces through the separable plan (plan, heights and block heights), the
    /// way `columns_at` reads them.
    pub window_dem: Duration,
    /// Just building that plan (axes and tile lookups).
    pub window_plan: Duration,
    /// The same window through the old point-by-point read
    /// ([`horizon_earth::ElevationField::surface_ys_strided_reference`]).
    pub window_dem_reference: Duration,
    /// The padded window's land cover classes (warp and vote).
    pub window_class: Duration,
    /// The river index search of every window sample.
    pub rivers: Duration,
    /// The lakes of the window.
    pub lakes: Duration,
    /// The water rules over the window.
    pub water_rules: Duration,
    /// Each column's kind (climate, slope, shore).
    pub kinds: Duration,
    /// Samples in the window.
    pub window_samples: usize,
}

impl EarthGenerator {
    /// [`EarthGenerator::columns_at`]'s steps timed; `None` when something is not loaded.
    #[doc(hidden)]
    pub fn columns_at_stages(
        &self,
        x0: i32,
        z0: i32,
        stride: i32,
        zoom_step: i32,
    ) -> Option<Stages> {
        let mut st = Stages::default();
        let land = self.land_source()?.clone();
        let stride = stride.max(1);
        let (sx, sz) = (x0 + stride / 2, z0 + stride / 2);
        let zoom = self.zoom_for_step(zoom_step);
        let t = Instant::now();
        self.field()
            .surface_ys_strided(sx, sz, 16, 16, stride, zoom)?;
        st.surfaces = t.elapsed();
        let pad = WINDOW_PAD as i32 * stride;
        let (px, pz) = (sx - pad, sz - pad);
        let (w, h) = (16 + 2 * WINDOW_PAD, 16 + 2 * WINDOW_PAD);
        st.window_samples = w * h;
        let t = Instant::now();
        let reference = self
            .field()
            .surface_ys_strided_reference(px, pz, w, h, stride, zoom)?;
        st.window_dem_reference = t.elapsed();
        let t = Instant::now();
        let _plan = self.field().window_plan(px, pz, w, h, stride, zoom).ok()?;
        st.window_plan = t.elapsed();
        drop(_plan);
        let t = Instant::now();
        let dem = self
            .field()
            .surface_ys_strided(px, pz, w, h, stride, zoom)?;
        st.window_dem = t.elapsed();
        debug_assert_eq!(dem, reference);
        let t = Instant::now();
        let class = self
            .classes_core(Access::Local, &land, [px, pz], [w, h], stride, zoom_step)
            .ok()?;
        st.window_class = t.elapsed();
        let r = Rect {
            x0: px,
            z0: pz,
            x1: px + (w as i32 - 1) * stride,
            z1: pz + (h as i32 - 1) * stride,
            zoom,
        };
        let at = |i: usize| (px + (i % w) as i32 * stride, pz + (i / w) as i32 * stride);
        let t = Instant::now();
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
        st.rivers = t.elapsed();
        let t = Instant::now();
        let vertical = *self.field().vertical();
        let lakes: Vec<Arc<Lake>> = match self.water_atlas() {
            Some(atlas) if class.contains(&LandClass::Water) => {
                let (s, w_, n, e) = self.lat_lon_box(r);
                let m = LAKE_BOX_MARGIN_DEG;
                atlas.lakes_in(s - m, w_ - m, n + m, e + m)
            }
            _ => Vec::new(),
        };
        let vpm = vertical_blocks_per_metre(&vertical);
        let projection = self.field().projection();
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
        st.lakes = t.elapsed();
        let t = Instant::now();
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
        st.water_rules = t.elapsed();
        let mut classes = Vec::with_capacity(256);
        for row in WINDOW_PAD..WINDOW_PAD + 16 {
            classes.extend_from_slice(&class[row * w + WINDOW_PAD..row * w + WINDOW_PAD + 16]);
        }
        let t = Instant::now();
        for (i, (&cw, &c)) in columns.iter().zip(&classes).enumerate() {
            let (x, z) = (sx + (i % 16) as i32 * stride, sz + (i / 16) as i32 * stride);
            let _ = self.water_column(Access::Local, [x, z], cw, c, stride, zoom_step);
        }
        st.kinds = t.elapsed();
        Some(st)
    }
}

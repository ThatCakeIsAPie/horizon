//! Elevation over world blocks: bilinear heights, conservative height ranges and the tiles a region needs.
//!
//! Every query comes in a form that only reads tiles the store already holds and says `None` when one is missing,
//! so a caller never gets a made-up height; `*_blocking` forms load missing tiles through the store's fetcher.

use std::collections::HashMap;
use std::sync::Arc;

use crate::projection::{Projection, VerticalMap, latlon_to_pixel, world_pixels};
use crate::source::{LoadError, TileStore};
use crate::tile::{MAX_ZOOM, TILE_SIZE, Tile, TileId};

/// Metres per pixel at zoom 0 on the equator: Web Mercator's circumference over 256 pixels.
pub const EQUATOR_METRES_PER_PIXEL: f64 = 156_543.033_92;

/// Pixel rectangle of a block rectangle, split where it crosses the antimeridian: each x interval and the y
/// interval are inclusive pixel indices over the whole world.
struct Span {
    xs: Vec<(i64, i64)>,
    y: (i64, i64),
}

/// The terrain's height at any world block, read from a [`TileStore`] through a [`Projection`].
#[derive(Clone)]
pub struct ElevationField {
    store: Arc<TileStore>,
    projection: Projection,
    vertical: VerticalMap,
    max_zoom: u8,
}

impl ElevationField {
    /// A field over `store`, placed by `projection` and scaled in height by `vertical`. The finest zoom is 15.
    pub fn new(store: Arc<TileStore>, projection: Projection, vertical: VerticalMap) -> Self {
        Self {
            store,
            projection,
            vertical,
            max_zoom: MAX_ZOOM,
        }
    }

    /// The same field that never uses a zoom above `max_zoom` (at most 15).
    pub fn with_max_zoom(mut self, max_zoom: u8) -> Self {
        self.max_zoom = max_zoom.min(MAX_ZOOM);
        self
    }

    pub fn store(&self) -> &Arc<TileStore> {
        &self.store
    }

    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    pub fn vertical(&self) -> &VerticalMap {
        &self.vertical
    }

    pub fn max_zoom(&self) -> u8 {
        self.max_zoom
    }

    /// The coarsest zoom whose pixels, at the origin's latitude, are no bigger than `metres_per_sample`, from 0 to
    /// the maximum zoom. A spacing that is not a positive number gets the finest zoom.
    pub fn zoom_for(&self, metres_per_sample: f64) -> u8 {
        if metres_per_sample.is_nan() || metres_per_sample <= 0.0 {
            return self.max_zoom;
        }
        let at_zoom_0 =
            EQUATOR_METRES_PER_PIXEL * self.projection.reference_latitude().to_radians().cos();
        let zoom = (at_zoom_0 / metres_per_sample).log2().ceil();
        // `as` saturates, and the clamp does the rest.
        (zoom.max(0.0) as u32).min(u32::from(self.max_zoom)) as u8
    }

    /// The elevation in metres at the world point `(x, z)` (fractional blocks; add 0.5 for a block's centre),
    /// interpolated bilinearly between the four nearest pixel centres at `zoom`, across tile seams and the
    /// antimeridian alike. `None` when any tile it needs is not in the store or the point is not finite.
    pub fn height_m(&self, x: f64, z: f64, zoom: u8) -> Option<f32> {
        self.sample(x, z, zoom, &mut |id| {
            Ok::<_, std::convert::Infallible>(self.store.cached(id))
        })
        .unwrap_or(None)
    }

    /// [`ElevationField::height_m`] that fetches the tiles it lacks, blocking.
    pub fn height_m_blocking(&self, x: f64, z: f64, zoom: u8) -> Result<Option<f32>, LoadError> {
        self.sample(x, z, zoom, &mut |id| self.store.load(id).map(Some))
    }

    /// The block height of the surface at block `(x, z)`: the elevation at the block's centre, mapped by the
    /// vertical scale and rounded down. `None` as for [`ElevationField::height_m`].
    pub fn surface_y(&self, x: i32, z: i32, zoom: u8) -> Option<i32> {
        let h = self.height_m(f64::from(x) + 0.5, f64::from(z) + 0.5, zoom)?;
        Some(self.vertical.block_y(f64::from(h)))
    }

    /// [`ElevationField::surface_y`] that fetches the tiles it lacks, blocking.
    pub fn surface_y_blocking(&self, x: i32, z: i32, zoom: u8) -> Result<Option<i32>, LoadError> {
        let h = self.height_m_blocking(f64::from(x) + 0.5, f64::from(z) + 0.5, zoom)?;
        Ok(h.map(|h| self.vertical.block_y(f64::from(h))))
    }

    /// The elevations in metres at the centres of the blocks `x0..x0 + width` by `z0..z0 + depth`, row by row
    /// (index `dz * width + dx`), each exactly what [`ElevationField::height_m`] gives at `(x + 0.5, z + 0.5)`.
    /// It resolves every tile once for the whole rectangle, so it takes the store's lock once per tile rather
    /// than once per block. `None` when any tile the rectangle needs is not in the store (nothing is fetched),
    /// or a point is not finite. An empty rectangle gives an empty list.
    pub fn heights_m(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        zoom: u8,
    ) -> Option<Vec<f32>> {
        self.grid_with(x0, z0, width, depth, 1.0, zoom, &mut |id| {
            Ok::<_, std::convert::Infallible>(self.store.cached(id))
        })
        .unwrap_or(None)
    }

    /// [`ElevationField::heights_m`] that fetches the tiles it lacks, blocking.
    pub fn heights_m_blocking(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        zoom: u8,
    ) -> Result<Option<Vec<f32>>, LoadError> {
        self.grid_with(x0, z0, width, depth, 1.0, zoom, &mut |id| {
            self.store.load(id).map(Some)
        })
    }

    /// The surface block heights of the same rectangle as [`ElevationField::heights_m`], each exactly what
    /// [`ElevationField::surface_y`] gives.
    pub fn surface_ys(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        zoom: u8,
    ) -> Option<Vec<i32>> {
        let heights = self.heights_m(x0, z0, width, depth, zoom)?;
        Some(heights.into_iter().map(|h| self.block_y_of(h)).collect())
    }

    /// [`ElevationField::surface_ys`] that fetches the tiles it lacks, blocking.
    pub fn surface_ys_blocking(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        zoom: u8,
    ) -> Result<Option<Vec<i32>>, LoadError> {
        let heights = self.heights_m_blocking(x0, z0, width, depth, zoom)?;
        Ok(heights.map(|hs| hs.into_iter().map(|h| self.block_y_of(h)).collect()))
    }

    /// [`ElevationField::surface_ys`] for samples `stride` blocks apart: `width` by `depth` of them, the one at
    /// `(i, j)` exactly what [`ElevationField::surface_y`] gives at block `(x0 + i * stride, z0 + j * stride)`.
    /// This is how a far node of the octree reads its 16 by 16 columns in one go. `None` when a tile is not in the
    /// store (nothing is fetched).
    pub fn surface_ys_strided(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        stride: i32,
        zoom: u8,
    ) -> Option<Vec<i32>> {
        let heights = self
            .grid_with(x0, z0, width, depth, f64::from(stride), zoom, &mut |id| {
                Ok::<_, std::convert::Infallible>(self.store.cached(id))
            })
            .unwrap_or(None)?;
        Some(heights.into_iter().map(|h| self.block_y_of(h)).collect())
    }

    fn block_y_of(&self, height_m: f32) -> i32 {
        self.vertical.block_y(f64::from(height_m))
    }

    /// A lowest and highest elevation, in metres, that bound [`ElevationField::height_m`] at every point of the
    /// block rectangle `x0..=x1` by `z0..=z1` (corners in either order, so the area is `[x0, x1 + 1]` by
    /// `[z0, z1 + 1]` in world coordinates). It covers every pixel the rectangle touches and one more around it, and
    /// reads whole grid cells where it can, so a large rectangle is cheap. `None` when any tile it covers is
    /// not in the store.
    pub fn range(&self, x0: i32, z0: i32, x1: i32, z1: i32, zoom: u8) -> Option<(f32, f32)> {
        self.range_with(x0, z0, x1, z1, zoom, &mut |id| {
            Ok::<_, std::convert::Infallible>(self.store.cached(id))
        })
        .unwrap_or(None)
    }

    /// [`ElevationField::range`] that fetches the tiles it lacks, blocking.
    pub fn range_blocking(
        &self,
        x0: i32,
        z0: i32,
        x1: i32,
        z1: i32,
        zoom: u8,
    ) -> Result<Option<(f32, f32)>, LoadError> {
        self.range_with(x0, z0, x1, z1, zoom, &mut |id| {
            self.store.load(id).map(Some)
        })
    }

    /// The tiles [`ElevationField::range`] reads for this rectangle, in order, for prefetching. Every tile
    /// [`ElevationField::height_m`] needs inside the rectangle is among them.
    pub fn tiles_for(&self, x0: i32, z0: i32, x1: i32, z1: i32, zoom: u8) -> Vec<TileId> {
        let zoom = zoom.min(self.max_zoom);
        let Some(span) = self.span(x0, z0, x1, z1, zoom) else {
            return Vec::new();
        };
        let side = i64::from(TILE_SIZE);
        let mut ids = Vec::new();
        for &(xa, xb) in &span.xs {
            for ty in span.y.0 / side..=span.y.1 / side {
                for tx in xa / side..=xb / side {
                    ids.push(TileId {
                        z: zoom,
                        x: tx as u32,
                        y: ty as u32,
                    });
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Fractional world pixels of the point, with x not wrapped.
    fn pixel_of(&self, x: f64, z: f64, zoom: u8) -> (f64, f64) {
        let at = self.projection.to_lat_lon_unwrapped(x, z);
        latlon_to_pixel(at.lat, at.lon, zoom)
    }

    fn sample<E>(
        &self,
        x: f64,
        z: f64,
        zoom: u8,
        get: &mut dyn FnMut(TileId) -> Result<Option<Arc<Tile>>, E>,
    ) -> Result<Option<f32>, E> {
        let zoom = zoom.min(self.max_zoom);
        let (px, py) = self.pixel_of(x, z, zoom);
        if !(px.is_finite() && py.is_finite()) {
            return Ok(None);
        }
        let size = world_pixels(zoom) as i64;
        // Pixel `i` is centred at `i + 0.5`, so the point sits between pixels `floor(p - 0.5)` and the next.
        let (fx, fy) = (px - 0.5, py - 0.5);
        let (x_floor, y_floor) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x_floor, fy - y_floor);
        let xs = [
            (x_floor as i64).rem_euclid(size),
            (x_floor as i64 + 1).rem_euclid(size),
        ];
        let ys = [
            (y_floor as i64).clamp(0, size - 1),
            (y_floor as i64 + 1).clamp(0, size - 1),
        ];

        let mut held: Vec<(TileId, Arc<Tile>)> = Vec::with_capacity(4);
        let mut at = |gx: i64, gy: i64| -> Result<Option<f64>, E> {
            let side = i64::from(TILE_SIZE);
            let id = TileId {
                z: zoom,
                x: (gx / side) as u32,
                y: (gy / side) as u32,
            };
            let tile = match held.iter().find(|(held_id, _)| *held_id == id) {
                Some((_, tile)) => Arc::clone(tile),
                None => match get(id)? {
                    Some(tile) => {
                        held.push((id, Arc::clone(&tile)));
                        tile
                    }
                    None => return Ok(None),
                },
            };
            Ok(Some(f64::from(
                tile.height((gx % side) as usize, (gy % side) as usize),
            )))
        };
        let mut corner = |gx, gy| at(gx, gy);
        let (Some(h00), Some(h10), Some(h01), Some(h11)) = (
            corner(xs[0], ys[0])?,
            corner(xs[1], ys[0])?,
            corner(xs[0], ys[1])?,
            corner(xs[1], ys[1])?,
        ) else {
            return Ok(None);
        };
        let top = h00 + (h10 - h00) * tx;
        let bottom = h01 + (h11 - h01) * tx;
        Ok(Some((top + (bottom - top) * ty) as f32))
    }

    fn grid_with<E>(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        stride: f64,
        zoom: u8,
        get: &mut dyn FnMut(TileId) -> Result<Option<Arc<Tile>>, E>,
    ) -> Result<Option<Vec<f32>>, E> {
        // Tiles already resolved for this rectangle, so the store is asked once for each.
        let mut resolved: HashMap<TileId, Arc<Tile>> = HashMap::new();
        let mut cached_get = |id: TileId| -> Result<Option<Arc<Tile>>, E> {
            if let Some(tile) = resolved.get(&id) {
                return Ok(Some(Arc::clone(tile)));
            }
            let tile = get(id)?;
            if let Some(tile) = &tile {
                resolved.insert(id, Arc::clone(tile));
            }
            Ok(tile)
        };
        let mut out = Vec::with_capacity(width.saturating_mul(depth));
        for dz in 0..depth {
            for dx in 0..width {
                let x = f64::from(x0) + (dx as f64) * stride + 0.5;
                let z = f64::from(z0) + (dz as f64) * stride + 0.5;
                match self.sample(x, z, zoom, &mut cached_get)? {
                    Some(h) => out.push(h),
                    None => return Ok(None),
                }
            }
        }
        Ok(Some(out))
    }

    fn range_with<E>(
        &self,
        x0: i32,
        z0: i32,
        x1: i32,
        z1: i32,
        zoom: u8,
        get: &mut dyn FnMut(TileId) -> Result<Option<Arc<Tile>>, E>,
    ) -> Result<Option<(f32, f32)>, E> {
        let zoom = zoom.min(self.max_zoom);
        let Some(span) = self.span(x0, z0, x1, z1, zoom) else {
            return Ok(None);
        };
        let side = i64::from(TILE_SIZE);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &(xa, xb) in &span.xs {
            for ty in span.y.0 / side..=span.y.1 / side {
                for tx in xa / side..=xb / side {
                    let id = TileId {
                        z: zoom,
                        x: tx as u32,
                        y: ty as u32,
                    };
                    let Some(tile) = get(id)? else {
                        return Ok(None);
                    };
                    // The part of the span inside this tile, in the tile's own pixels.
                    let local = |from: i64, to: i64, origin: i64| {
                        (
                            (from.max(origin) - origin) as usize,
                            (to.min(origin + side - 1) - origin) as usize,
                        )
                    };
                    let (px0, px1) = local(xa, xb, tx * side);
                    let (py0, py1) = local(span.y.0, span.y.1, ty * side);
                    if let Some((a, b)) = tile.range(px0, py0, px1, py1) {
                        lo = lo.min(a);
                        hi = hi.max(b);
                    }
                }
            }
        }
        Ok((lo <= hi).then_some((lo, hi)))
    }

    /// The pixels a block rectangle's samples can blend, plus one pixel all round.
    fn span(&self, x0: i32, z0: i32, x1: i32, z1: i32, zoom: u8) -> Option<Span> {
        let (xa, xb) = (f64::from(x0.min(x1)), f64::from(x0.max(x1)) + 1.0);
        let (za, zb) = (f64::from(z0.min(z1)), f64::from(z0.max(z1)) + 1.0);
        // The mapping keeps each axis to its own pixel axis, so two opposite corners bound the rectangle.
        let (pxa, pya) = self.pixel_of(xa, za, zoom);
        let (pxb, pyb) = self.pixel_of(xb, zb, zoom);
        if !(pxa.is_finite() && pxb.is_finite() && pya.is_finite() && pyb.is_finite()) {
            return None;
        }
        let size = world_pixels(zoom) as i64;
        // A sample at pixel position `p` blends pixels `floor(p - 0.5)` and the next; one more pixel of margin.
        let first = |p: f64| (p - 0.5).floor() as i64 - 1;
        let last = |p: f64| (p - 0.5).floor() as i64 + 2;
        let (west, east) = (first(pxa.min(pxb)), last(pxa.max(pxb)));
        let xs = if east - west + 1 >= size {
            vec![(0, size - 1)]
        } else {
            let start = west.rem_euclid(size);
            let end = start + (east - west);
            if end < size {
                vec![(start, end)]
            } else {
                vec![(start, size - 1), (0, end - size)]
            }
        };
        let y = (
            first(pya.min(pyb)).clamp(0, size - 1),
            last(pya.max(pyb)).clamp(0, size - 1),
        );
        Some(Span { xs, y })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{MemoryFetcher, TempDir};

    /// Smooth and periodic in longitude, so it has no jump at the antimeridian.
    fn smooth(lat: f64, lon: f64) -> f64 {
        1000.0 + 400.0 * lat.to_radians().sin() + 300.0 * lon.to_radians().sin()
    }

    /// Changes sharply from one pixel to the next at the zooms the tests use.
    fn rough(lat: f64, lon: f64) -> f64 {
        1500.0 * (lat * 211.0).to_radians().sin() * (lon * 173.0).to_radians().cos()
            + 300.0 * (lat * 1000.0 + lon * 777.0).to_radians().sin()
    }

    struct Rig {
        fetcher: Arc<MemoryFetcher>,
        field: ElevationField,
        _dir: TempDir,
    }

    fn build(lat0: f64, lon0: f64, scale: f64, height: fn(f64, f64) -> f64) -> Rig {
        let dir = TempDir::new();
        let fetcher = Arc::new(MemoryFetcher::new(height));
        let store = Arc::new(TileStore::new(dir.path(), fetcher.clone(), 64));
        let field = ElevationField::new(
            store,
            Projection::local(lat0, lon0, scale).unwrap(),
            VerticalMap::new(63, scale),
        );
        Rig {
            fetcher,
            field,
            _dir: dir,
        }
    }

    /// A tiny deterministic generator for sample positions.
    fn unit(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn interpolation_across_a_tile_seam_matches_the_surface() {
        // The origin is the corner four tiles meet at, at zoom 8.
        let rig = build(0.0, 0.0, 100.0, smooth);
        let (mut x, mut worst) = (-60.0, 0.0f64);
        while x < 60.0 {
            for z in [-55.5, -3.2, 0.0, 4.4, 51.0] {
                let got = rig.field.height_m_blocking(x, z, 8).unwrap().unwrap();
                let at = rig.field.projection().to_lat_lon(x, z);
                worst = worst.max((f64::from(got) - smooth(at.lat, at.lon)).abs());
            }
            x += 3.7;
        }
        assert!(worst < 0.02, "worst error {worst} m");
        for (tx, ty) in [(127, 127), (128, 127), (127, 128), (128, 128)] {
            assert_eq!(
                rig.fetcher.fetches_of(TileId { z: 8, x: tx, y: ty }),
                1,
                "tile {tx},{ty}"
            );
        }
        assert_eq!(rig.fetcher.fetch_count(), 4);
        // Now everything is local, the non-blocking form gives the same answer.
        let a = rig.field.height_m(2.5, -1.5, 8).unwrap();
        assert_eq!(Some(a), rig.field.height_m_blocking(2.5, -1.5, 8).unwrap());
    }

    #[test]
    fn interpolation_across_the_antimeridian_matches_the_surface() {
        // 0.01° west of the antimeridian; 100 km east is about 0.9° past it.
        let rig = build(10.0, 179.99, 50.0, smooth);
        let mut worst = 0.0f64;
        for i in 0..=80 {
            let x = -500.0 + f64::from(i) * 50.0;
            let got = rig.field.height_m_blocking(x, 12.0, 10).unwrap().unwrap();
            let at = rig.field.projection().to_lat_lon(x, 12.0);
            worst = worst.max((f64::from(got) - smooth(at.lat, at.lon)).abs());
        }
        assert!(worst < 0.02, "worst error {worst} m");
        let columns: Vec<u32> = rig
            .field
            .tiles_for(-500, 12, 3500, 12, 10)
            .iter()
            .map(|t| t.x)
            .collect();
        assert!(
            columns.contains(&1023) && columns.contains(&0),
            "wrapped tile columns: {columns:?}"
        );
    }

    #[test]
    fn a_batched_read_matches_the_single_reads_point_for_point_across_a_seam() {
        // The origin is the corner four tiles meet at, so the rectangle reads all four.
        let rig = build(0.0, 0.0, 100.0, smooth);
        let (x0, z0, width, depth) = (-9, -6, 20, 13);
        let heights = rig
            .field
            .heights_m_blocking(x0, z0, width, depth, 8)
            .unwrap()
            .unwrap();
        let ys = rig.field.surface_ys(x0, z0, width, depth, 8).unwrap();
        assert_eq!((heights.len(), ys.len()), (width * depth, width * depth));
        for dz in 0..depth {
            for dx in 0..width {
                let (x, z) = (x0 + dx as i32, z0 + dz as i32);
                let single = rig
                    .field
                    .height_m(f64::from(x) + 0.5, f64::from(z) + 0.5, 8)
                    .unwrap();
                assert_eq!(heights[dz * width + dx], single, "height at {x},{z}");
                assert_eq!(
                    Some(ys[dz * width + dx]),
                    rig.field.surface_y(x, z, 8),
                    "surface at {x},{z}"
                );
            }
        }
        // The rectangle really crossed the seam: all four tiles were needed, one fetch each.
        assert_eq!(rig.fetcher.fetch_count(), 4);
        assert_eq!(
            rig.field
                .surface_ys_blocking(x0, z0, width, depth, 8)
                .unwrap(),
            Some(ys)
        );
    }

    #[test]
    fn a_strided_read_matches_the_single_reads_and_needs_every_tile() {
        let rig = build(0.0, 0.0, 100.0, smooth);
        let (x0, z0, n, stride) = (-70, -50, 16, 9);
        // Nothing local yet: no answer, and nothing fetched.
        assert_eq!(rig.field.surface_ys_strided(x0, z0, n, n, stride, 8), None);
        assert_eq!(rig.fetcher.fetch_count(), 0);
        for id in rig
            .field
            .tiles_for(x0, z0, x0 + stride * n as i32, z0 + stride * n as i32, 8)
        {
            rig.field.store().load(id).unwrap();
        }
        let ys = rig
            .field
            .surface_ys_strided(x0, z0, n, n, stride, 8)
            .unwrap();
        assert_eq!(ys.len(), n * n);
        for j in 0..n {
            for i in 0..n {
                let (x, z) = (x0 + i as i32 * stride, z0 + j as i32 * stride);
                assert_eq!(Some(ys[j * n + i]), rig.field.surface_y(x, z, 8), "{x},{z}");
            }
        }
        // A stride of one is the plain batch.
        assert_eq!(
            rig.field.surface_ys_strided(x0, z0, 5, 4, 1, 8),
            rig.field.surface_ys(x0, z0, 5, 4, 8)
        );
    }

    #[test]
    fn a_batched_read_with_a_tile_missing_gives_none_and_fetches_nothing() {
        let rig = build(0.0, 0.0, 100.0, smooth);
        rig.field
            .store()
            .load(TileId {
                z: 8,
                x: 127,
                y: 127,
            })
            .unwrap();
        assert!(rig.field.heights_m(-20, -20, 8, 8, 8).is_some());
        assert_eq!(rig.field.heights_m(-4, -4, 8, 8, 8), None);
        assert_eq!(rig.field.surface_ys(-4, -4, 8, 8, 8), None);
        assert_eq!(rig.fetcher.fetch_count(), 1);
        assert_eq!(rig.field.heights_m(0, 0, 0, 5, 8), Some(Vec::new()));
    }

    #[test]
    fn a_missing_tile_gives_none_never_a_number() {
        let rig = build(0.0, 0.0, 100.0, smooth);
        // Hold only the tile north-west of the origin's corner.
        rig.field
            .store()
            .load(TileId {
                z: 8,
                x: 127,
                y: 127,
            })
            .unwrap();
        // Deep inside it, the point is covered.
        assert!(rig.field.height_m(-30.0, -30.0, 8).is_some());
        // Near the corner a neighbouring tile is needed, and it is not local.
        assert_eq!(rig.field.height_m(2.0, 2.0, 8), None);
        assert_eq!(rig.field.surface_y(2, 2, 8), None);
        assert_eq!(rig.field.range(-10, -10, 10, 10, 8), None);
        // Nothing was fetched to find out.
        assert_eq!(rig.fetcher.fetch_count(), 1);
        // A point that is not a number has no height either.
        assert_eq!(rig.field.height_m(f64::NAN, 0.0, 8), None);
    }

    #[test]
    fn a_failing_fetcher_gives_an_error_from_the_blocking_form() {
        let rig = build(0.0, 0.0, 100.0, smooth);
        rig.fetcher
            .fail_with(Some(crate::source::FetchError::Temporary("offline".into())));
        assert!(rig.field.height_m_blocking(0.0, 0.0, 8).is_err());
        assert!(rig.field.range_blocking(0, 0, 5, 5, 8).is_err());
        assert_eq!(rig.field.height_m(0.0, 0.0, 8), None);
    }

    #[test]
    fn range_bounds_every_height_inside_small_and_large_rectangles() {
        // 60 m blocks at zoom 12 (38 m pixels): a block is bigger than a pixel. 5 m blocks: far smaller.
        for (scale, zoom) in [(60.0, 12u8), (5.0, 12), (400.0, 9)] {
            let rig = build(0.0, 0.0, scale, rough);
            let mut seed = 7u64;
            // Small rectangles, one across the seam at the origin, big ones, one wide enough to cover many tiles.
            for &(x0, z0, x1, z1) in &[
                (3, 4, 3, 4),
                (-2, -2, 2, 2),
                (10, 10, 12, 14),
                (-40, -35, 90, 77),
                (-300, -300, 300, 300),
            ] {
                let (lo, hi) = rig
                    .field
                    .range_blocking(x0, z0, x1, z1, zoom)
                    .unwrap()
                    .unwrap();
                assert!(lo <= hi);
                for _ in 0..400 {
                    let x = f64::from(x0) + unit(&mut seed) * f64::from(x1 - x0 + 1);
                    let z = f64::from(z0) + unit(&mut seed) * f64::from(z1 - z0 + 1);
                    let h = rig.field.height_m(x, z, zoom).unwrap();
                    assert!(
                        lo <= h && h <= hi,
                        "scale {scale}: {h} at ({x}, {z}) is outside {lo}..{hi} of rect {x0},{z0}..{x1},{z1}"
                    );
                }
            }
        }
    }

    #[test]
    fn range_bounds_heights_across_the_antimeridian_and_at_the_edge_of_the_map() {
        for (lat0, lon0) in [(5.0, 179.995), (84.9, 10.0)] {
            // Zoom 8 keeps the pole case, where a block spans many degrees of longitude, to a handful of tiles.
            let rig = build(lat0, lon0, 100.0, rough);
            let mut seed = 99u64;
            let (x0, z0, x1, z1) = (-300, -300, 700, 400);
            let (lo, hi) = rig
                .field
                .range_blocking(x0, z0, x1, z1, 8)
                .unwrap()
                .unwrap();
            for _ in 0..500 {
                let x = f64::from(x0) + unit(&mut seed) * f64::from(x1 - x0 + 1);
                let z = f64::from(z0) + unit(&mut seed) * f64::from(z1 - z0 + 1);
                let h = rig.field.height_m_blocking(x, z, 8).unwrap().unwrap();
                assert!(
                    lo <= h && h <= hi,
                    "{h} at ({x}, {z}) outside {lo}..{hi} near {lat0},{lon0}"
                );
            }
        }
    }

    #[test]
    fn range_is_tight_enough_to_be_useful() {
        // Height equals the longitude in degrees times ten: the range of a rectangle is its longitude span plus
        // about two pixels each side, nothing like the whole world's -1800..1800.
        let rig = build(0.0, 0.0, 100.0, |_, lon| lon * 10.0);
        let (lo, hi) = rig.field.range_blocking(0, 0, 99, 99, 9).unwrap().unwrap();
        let pixel_degrees = 360.0 / (256.0 * 512.0);
        let east_degrees = 100.0 * 100.0 / (crate::projection::EARTH_RADIUS_M.to_radians());
        assert!(
            f64::from(lo) > -2.5 * pixel_degrees * 10.0 && f64::from(lo) <= 0.0,
            "lo {lo}"
        );
        assert!(
            f64::from(hi) >= east_degrees * 10.0
                && f64::from(hi) < (east_degrees + 3.0 * pixel_degrees) * 10.0,
            "hi {hi}"
        );
        // A flat world has a flat range, however big the rectangle.
        let flat = build(0.0, 0.0, 100.0, |_, _| 42.5);
        assert_eq!(
            flat.field
                .range_blocking(-3000, -3000, 3000, 3000, 8)
                .unwrap(),
            Some((42.5, 42.5))
        );
    }

    #[test]
    fn zoom_for_picks_the_coarsest_zoom_that_is_fine_enough() {
        let at = |lat: f64| build(lat, 0.0, 1.0, smooth).field;
        let equator = at(0.0);
        // 156543 / 2^z m per pixel: 1223 (z7), 611.5 (z8), 305.7 (z9), 76.4 (z11), 4.78 (z15).
        assert_eq!(equator.zoom_for(1000.0), 8);
        assert_eq!(equator.zoom_for(611.5), 8);
        assert_eq!(equator.zoom_for(611.4), 9);
        assert_eq!(equator.zoom_for(100.0), 11);
        assert_eq!(equator.zoom_for(200_000.0), 0);
        assert_eq!(equator.zoom_for(0.5), 15);
        // At 60° pixels are half as wide on the ground: 782.7 (z6), 391.4 (z7), 195.7 (z8), 97.9 (z9), 76.4 (z10).
        let sixty = at(60.0);
        assert_eq!(sixty.zoom_for(100.0), 10);
        assert_eq!(sixty.zoom_for(1000.0), 7);
        assert!(sixty.zoom_for(100.0) < equator.zoom_for(100.0) + 1);
        // The ceiling is configurable; nonsense spacing gets the finest zoom.
        let capped = at(0.0).with_max_zoom(10);
        assert_eq!(capped.zoom_for(1.0), 10);
        assert_eq!(equator.zoom_for(f64::NAN), 15);
        assert_eq!(equator.zoom_for(-5.0), 15);
        assert_eq!(equator.with_max_zoom(99).max_zoom(), 15);
    }

    #[test]
    fn surface_y_is_the_scaled_elevation_rounded_down_at_the_block_centre() {
        let rig = build(20.0, 30.0, 2.0, |_, _| 123.7);
        // 63 + 123.7 / 2 = 124.85, and 0.0039 m of quantisation cannot reach 125.
        assert_eq!(rig.field.surface_y_blocking(5, -9, 12).unwrap(), Some(124));
        assert_eq!(rig.field.surface_y(5, -9, 12), Some(124));
        let sea = build(20.0, 30.0, 2.0, |_, _| 0.0);
        assert_eq!(sea.field.surface_y_blocking(0, 0, 12).unwrap(), Some(63));
        let trench = build(20.0, 30.0, 1.0, |_, _| -10.5);
        assert_eq!(trench.field.surface_y_blocking(0, 0, 12).unwrap(), Some(52));
    }

    #[test]
    fn tiles_for_lists_exactly_the_tiles_a_rectangle_reads() {
        // The origin at the middle of tile 4/5/6 and tiny blocks: one block reads that tile alone.
        let centre =
            crate::projection::pixel_to_latlon(5.0 * 256.0 + 128.0, 6.0 * 256.0 + 128.0, 4);
        let rig = build(centre.lat, centre.lon, 10.0, smooth);
        let one = TileId { z: 4, x: 5, y: 6 };
        assert_eq!(rig.field.tiles_for(0, 0, 0, 0, 4), vec![one]);
        // A rectangle across a seam lists both sides, sorted and without repeats.
        let seam = build(0.0, 0.0, 100.0, smooth);
        let ids = seam.field.tiles_for(-4, -4, 4, 4, 8);
        assert_eq!(ids.len(), 4);
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        // Loading exactly those makes range answer without another fetch.
        for id in &ids {
            seam.field.store().load(*id).unwrap();
        }
        assert!(seam.field.range(-4, -4, 4, 4, 8).is_some());
        assert_eq!(seam.fetcher.fetch_count(), 4);
    }
}

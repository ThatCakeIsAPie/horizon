//! Far-view reads that never wait: the columns of one octree node in a single batched read, and the list of
//! tiles workers found missing.
//!
//! A far node is 16 by 16 columns of voxels `step = 1 << level` blocks wide. [`EarthGenerator::node_columns`] reads
//! the surface of all 256 at once (one store lookup per tile, not per voxel); the voxels are then
//! [`EarthGenerator::far_voxel`] of each column's surface and the voxel's height. When a tile is missing the read
//! says [`NotReady`], and a worker calls [`EarthGenerator::request_tiles`] with the tiles it names and gives the
//! node up for later, so nothing built without its data is ever kept.
//!
//! With land cover the same holds for it: a node whose columns lack the class of a column is not ready, never
//! built from the elevation rule alone and kept. That choice is deliberate: a far node is cached on disk, so one
//! built before its class arrived would have to be thrown away and rebuilt when it did, and the player would see
//! the colour of the whole distance flip. What is missing is recorded ([`EarthGenerator::take_land_requests`]);
//! a place the source has no file for is no data at once (open sea), which is the elevation rule and final.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, PoisonError};

use horizon_earth::{Fetch, LandClass, TileId};

use super::far_trees::ColumnsDecor;
use super::generator::{Access, EarthGenerator, Gap, NotReady, Rect, node_box};
use super::surface::SurfaceKind;

/// How many requests wait at most; a long list means whatever fetches them is far behind.
const MAX_PENDING: usize = 8192;

/// A way to wake whatever takes the requests.
pub type Waker = Arc<dyn Fn() + Send + Sync>;

/// Tiles asked for and not yet taken.
#[derive(Default)]
pub struct Requests {
    pending: Mutex<HashSet<TileId>>,
    land: Mutex<HashSet<Fetch>>,
    climate: Mutex<HashSet<horizon_earth::ClimateFetch>>,
    waker: Mutex<Option<Waker>>,
}

/// The columns of a far node (16 by 16, `z << 4 | x`) or of a wider block of them: each one's surface block
/// height and kind of top, and with land cover its class.
#[derive(Clone, Debug)]
pub struct NodeColumns {
    /// Blocks per voxel.
    pub step: i32,
    /// Columns in a row.
    pub width: usize,
    /// The block the first column is sampled at.
    pub origin: [i32; 2],
    /// The final surface of each column: the top water block of a water column, the ground of the rest.
    pub surfaces: Vec<i32>,
    kinds: Vec<SurfaceKind>,
    /// The land class of each column; `None` without land cover.
    pub(super) classes: Option<Vec<LandClass>>,
    /// Trees and snow over the columns, once [`EarthGenerator::decorate_columns`] has made them.
    pub(super) decor: Option<Box<ColumnsDecor>>,
}

impl NodeColumns {
    /// The kind of top of column `i`.
    pub fn kind(&self, i: usize) -> SurfaceKind {
        self.kinds[i]
    }

    /// The land class of column `i`; `None` without land cover.
    pub fn class(&self, i: usize) -> Option<LandClass> {
        self.classes.as_ref().map(|c| c[i])
    }

    /// The block `(x, z)` column `i` is sampled at.
    pub(super) fn block_of(&self, i: usize) -> (i32, i32) {
        (
            self.origin[0].saturating_add((i % self.width) as i32 * self.step),
            self.origin[1].saturating_add((i / self.width) as i32 * self.step),
        )
    }
}

impl EarthGenerator {
    /// The surfaces alone of [`EarthGenerator::columns_at`]: what only the height of the ground decides (sky
    /// cover), which does not wait for land cover. With land cover the water rules move surfaces (a carved
    /// channel, a raised bank, the sea's top over a sunken column): when the land cover is already held the final
    /// surfaces are given, so a channel is not covered by the ground it was carved out of; when it is not, the
    /// elevation surfaces are (as without land cover).
    pub fn surfaces_at(
        &self,
        x0: i32,
        z0: i32,
        stride: i32,
        zoom_step: i32,
    ) -> Result<Vec<i32>, NotReady> {
        if self.land_cover_active()
            && let Ok(columns) = self.columns_at(x0, z0, stride, zoom_step)
        {
            return Ok(columns.surfaces);
        }
        let stride = stride.max(1);
        let (sx, sz) = (x0.saturating_add(stride / 2), z0.saturating_add(stride / 2));
        let zoom = self.zoom_for_step(zoom_step);
        self.field
            .surface_ys_strided(sx, sz, 16, 16, stride, zoom)
            .ok_or_else(|| {
                let span = 15 * stride;
                NotReady {
                    tiles: self.missing_in(Rect {
                        x0: sx,
                        z0: sz,
                        x1: sx.saturating_add(span),
                        z1: sz.saturating_add(span),
                        zoom,
                    }),
                }
            })
    }

    /// The columns of a block of 16 by 16 voxels `stride` blocks apart whose first voxel starts at the block
    /// `(x0, z0)`: each column is sampled at its voxel's middle block (`+ stride / 2`), at the zoom of a spacing
    /// of `zoom_step` blocks (and, with land cover, its class at the level of that spacing). Gives
    /// [`NotReady`] naming the tiles it lacks; when only land cover is missing the list is empty and what to
    /// fetch is in [`EarthGenerator::take_land_requests`].
    pub fn columns_at(
        &self,
        x0: i32,
        z0: i32,
        stride: i32,
        zoom_step: i32,
    ) -> Result<NodeColumns, NotReady> {
        self.columns_wide(x0, z0, 16, stride, zoom_step)
    }

    /// [`EarthGenerator::columns_at`] for `width` by `width` columns.
    pub(super) fn columns_wide(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        stride: i32,
        zoom_step: i32,
    ) -> Result<NodeColumns, NotReady> {
        let stride = stride.max(1);
        let (sx, sz) = (x0.saturating_add(stride / 2), z0.saturating_add(stride / 2));
        let zoom = self.zoom_for_step(zoom_step);
        let span = (width as i32 - 1) * stride;
        let rect = Rect {
            x0: sx,
            z0: sz,
            x1: sx.saturating_add(span),
            z1: sz.saturating_add(span),
            zoom,
        };
        let mut surfaces = self
            .field
            .surface_ys_strided(sx, sz, width, width, stride, zoom)
            .ok_or_else(|| NotReady {
                tiles: self.missing_in(rect),
            })?;
        let not_ready = |_| NotReady {
            tiles: self.missing_in(self.kind_rect(rect, stride)),
        };
        let (kinds, classes) = self
            .kinds_classes_core(
                Access::Local,
                [sx, sz],
                [width, width],
                stride,
                zoom_step,
                &mut surfaces,
            )
            .map_err(not_ready)?;
        Ok(NodeColumns {
            step: stride,
            width,
            origin: [sx, sz],
            surfaces,
            kinds,
            classes,
            decor: None,
        })
    }

    /// The columns of `width` by `depth` samples `stride` blocks apart from the block `(x0, z0)` (no half-step
    /// shift), read at the zoom and land cover level of `zoom_step`, loading what they need and waiting for it
    /// (as [`super::blocking`] does): what tools such as `examples/earth_probe.rs` show of the terrain. `None`
    /// when cancelled.
    pub fn columns_blocking(
        &self,
        [x0, z0]: [i32; 2],
        [width, depth]: [usize; 2],
        stride: i32,
        zoom_step: i32,
    ) -> Option<NodeColumns> {
        let stride = stride.max(1);
        let zoom = self.zoom_for_step(zoom_step);
        let span = |n: usize| (n.saturating_sub(1) as i64 * i64::from(stride)) as i32;
        let rect = Rect {
            x0,
            z0,
            x1: x0.saturating_add(span(width)),
            z1: z0.saturating_add(span(depth)),
            zoom,
        };
        self.wait_for(self.kind_rect(rect, stride), || {
            self.field
                .range_blocking(rect.x0, rect.z0, rect.x1, rect.z1, zoom)
                .map_err(Gap::Load)?;
            let mut surfaces = self
                .field
                .surface_ys_strided(x0, z0, width, depth, stride, zoom)
                .ok_or(Gap::Missing)?;
            let (kinds, classes) = self.kinds_classes_core(
                Access::Fetch,
                [x0, z0],
                [width, depth],
                stride,
                zoom_step,
                &mut surfaces,
            )?;
            Ok(NodeColumns {
                step: stride,
                width,
                origin: [x0, z0],
                surfaces,
                kinds,
                classes,
                decor: None,
            })
        })
    }

    /// [`EarthGenerator::columns_at`] for the far node at `level` whose column index is `(x, z)`: voxels `1 <<
    /// level` apart at that step's zoom, as [`EarthGenerator::block_at`] samples them.
    pub fn node_columns(&self, level: u32, x: i32, z: i32) -> Result<NodeColumns, NotReady> {
        let (min, _) = node_box(level, [x, 0, z]);
        let step = 1 << level.min(24);
        self.columns_at(min[0], min[2], step, step)
    }

    /// The block of the voxel in column `i` of `columns` whose lowest block is `lo`: the terrain's, with the
    /// column's trees and snow over it once they were made ([`EarthGenerator::decorate_columns`]).
    pub fn far_voxel(&self, columns: &NodeColumns, i: usize, lo: i32) -> u32 {
        let block = self.voxel_block(lo, columns.step, columns.surfaces[i], columns.kind(i));
        match &columns.decor {
            Some(decor) => self.decorated_voxel(columns, decor, i, lo, block),
            None => block,
        }
    }

    /// Asks for `tiles` to be fetched in the background (see [`EarthGenerator::take_requests`]); a tile asked
    /// for twice is listed once, and only a tile not already waiting wakes the taker.
    pub fn request_tiles(&self, tiles: &[TileId]) {
        if tiles.is_empty() {
            return;
        }
        let mut added = false;
        {
            let mut pending = self
                .requests
                .pending
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for &id in tiles {
                if pending.len() < MAX_PENDING {
                    added |= pending.insert(id);
                }
            }
        }
        // A tile already waiting has already woken the taker.
        if !added {
            return;
        }
        let waker = self
            .requests
            .waker
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(wake) = waker {
            wake();
        }
    }

    /// The tiles asked for since the last call, in tile order. Emptying the list is what takes them.
    pub fn take_requests(&self) -> Vec<TileId> {
        let mut taken: Vec<TileId> = self
            .requests
            .pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain()
            .collect();
        taken.sort_unstable();
        taken
    }

    /// Records land-cover fetches a read found missing, for whatever fetches them in the background (see
    /// [`EarthGenerator::take_land_requests`]); only a fetch not already waiting wakes the taker.
    pub(super) fn note_land(&self, fetches: Vec<Fetch>) {
        let mut added = false;
        {
            let mut pending = self
                .requests
                .land
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for fetch in fetches {
                if pending.len() < MAX_PENDING {
                    added |= pending.insert(fetch);
                }
            }
        }
        if !added {
            return;
        }
        let waker = self
            .requests
            .waker
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(wake) = waker {
            wake();
        }
    }

    /// The land-cover fetches asked for since the last call (by far-view reads and other local ones that lacked
    /// a class), in file and offset order. The downloader gives each to
    /// [`horizon_earth::LandCoverSource::satisfy`], and the readers that said "later" ask again.
    pub fn take_land_requests(&self) -> Vec<Fetch> {
        let mut taken: Vec<Fetch> = self
            .requests
            .land
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain()
            .collect();
        taken.sort_unstable_by_key(|f| (f.file, f.range.offset, f.range.len));
        taken
    }

    /// Records climate fetches a read found missing, like [`Self::note_land`] (see
    /// [`EarthGenerator::take_climate_requests`]).
    pub(super) fn note_climate(&self, fetch: horizon_earth::ClimateFetch) {
        let added = {
            let mut pending = self
                .requests
                .climate
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            pending.len() < MAX_PENDING && pending.insert(fetch)
        };
        if !added {
            return;
        }
        let waker = self
            .requests
            .waker
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(wake) = waker {
            wake();
        }
    }

    /// The climate fetches asked for since the last call, for [`horizon_earth::ClimateSource::satisfy`]; the
    /// readers that said "later" ask again.
    pub fn take_climate_requests(&self) -> Vec<horizon_earth::ClimateFetch> {
        let mut taken: Vec<horizon_earth::ClimateFetch> = self
            .requests
            .climate
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain()
            .collect();
        taken.sort_unstable_by_key(|f| (f.var.name(), f.range.offset, f.range.len));
        taken
    }

    /// Sets what [`EarthGenerator::request_tiles`] calls after it records a request, for the thread that takes
    /// them.
    pub fn set_request_waker(&self, waker: Waker) {
        *self
            .requests
            .waker
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(waker);
    }
}

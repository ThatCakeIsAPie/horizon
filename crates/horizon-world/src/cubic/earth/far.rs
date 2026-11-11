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

use horizon_earth::{Fetch, TileId};

use super::generator::{Access, EarthGenerator, NotReady, Rect, node_box};
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
    waker: Mutex<Option<Waker>>,
}

/// The 16 by 16 columns of a far node (`z << 4 | x`): each one's surface block height and kind of top.
#[derive(Clone, Debug)]
pub struct NodeColumns {
    /// Blocks per voxel.
    pub step: i32,
    pub surfaces: Vec<i32>,
    kinds: Vec<SurfaceKind>,
}

impl NodeColumns {
    /// The kind of top of column `i`.
    pub fn kind(&self, i: usize) -> SurfaceKind {
        self.kinds[i]
    }
}

impl EarthGenerator {
    /// The surfaces alone of [`EarthGenerator::columns_at`]: what only the height of the ground decides (sky
    /// cover), which does not wait for land cover.
    pub fn surfaces_at(
        &self,
        x0: i32,
        z0: i32,
        stride: i32,
        zoom_step: i32,
    ) -> Result<Vec<i32>, NotReady> {
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
        let stride = stride.max(1);
        let (sx, sz) = (x0.saturating_add(stride / 2), z0.saturating_add(stride / 2));
        let surfaces = self.surfaces_at(x0, z0, stride, zoom_step)?;
        let span = 15 * stride;
        let not_ready = |_| NotReady {
            tiles: self.missing_in(self.kind_rect(
                Rect {
                    x0: sx,
                    z0: sz,
                    x1: sx.saturating_add(span),
                    z1: sz.saturating_add(span),
                    zoom: self.zoom_for_step(zoom_step),
                },
                stride,
            )),
        };
        let kinds = self
            .kinds_core(
                Access::Local,
                [sx, sz],
                [16, 16],
                stride,
                zoom_step,
                &surfaces,
            )
            .map_err(not_ready)?;
        Ok(NodeColumns {
            step: stride,
            surfaces,
            kinds,
        })
    }

    /// [`EarthGenerator::columns_at`] for the far node at `level` whose column index is `(x, z)`: voxels `1 <<
    /// level` apart at that step's zoom, as [`EarthGenerator::block_at`] samples them.
    pub fn node_columns(&self, level: u32, x: i32, z: i32) -> Result<NodeColumns, NotReady> {
        let (min, _) = node_box(level, [x, 0, z]);
        let step = 1 << level.min(24);
        self.columns_at(min[0], min[2], step, step)
    }

    /// The block of the voxel in column `i` of `columns` whose lowest block is `lo`.
    pub fn far_voxel(&self, columns: &NodeColumns, i: usize, lo: i32) -> u32 {
        self.voxel_block(lo, columns.step, columns.surfaces[i], columns.kind(i))
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

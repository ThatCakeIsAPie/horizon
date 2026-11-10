//! The infallible forms of [`EarthGenerator`]'s queries, for callers that cannot say "not ready".
//!
//! Each loads the tiles it lacks through the store's fetcher. When a fetch fails it waits and tries again, the
//! wait starting at 250 ms and doubling up to 5 s, and logs once per streak of failures (and once when the
//! tiles come back). It gives up only when the cancel flag is set (see [`EarthGenerator::set_cancelled`]),
//! and then returns air, `Mixed` or nothing: terrain made after that must be thrown away by the caller.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use horizon_core::position::SectionPos;

use super::generator::{Access, EarthGenerator, Fill, Gap, Rect, no_data};

/// How often a waiting thread looks at the cancel flag.
const SLICE: Duration = Duration::from_millis(20);

impl EarthGenerator {
    /// Runs `op` (a read that may fetch) until it succeeds; `None` when cancelled first.
    fn wait_for<T>(&self, r: Rect, mut op: impl FnMut() -> Result<T, Gap>) -> Option<T> {
        let mut delay = self.retry.first;
        loop {
            if self.is_cancelled() {
                return None;
            }
            let why = match op() {
                Ok(value) => {
                    if self.failing.swap(false, Ordering::SeqCst) {
                        log::info!("Earth elevation tiles load again");
                    }
                    return Some(value);
                }
                Err(Gap::Load(e)) => e,
                Err(Gap::Missing) => no_data(),
            };
            if !self.failing.swap(true, Ordering::SeqCst) {
                let tiles = self.missing_in(r);
                log::warn!(
                    "Earth elevation tiles {tiles:?} did not load: {why}; retrying every {:?} up to {:?}",
                    self.retry.first,
                    self.retry.cap
                );
            }
            if !self.sleep(delay) {
                return None;
            }
            delay = (delay * 2).min(self.retry.cap);
        }
    }

    /// Sleeps `total`, waking early when cancelled; whether it was not.
    fn sleep(&self, total: Duration) -> bool {
        let end = Instant::now() + total;
        loop {
            if self.is_cancelled() {
                return false;
            }
            let left = end.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return true;
            }
            std::thread::sleep(left.min(SLICE));
        }
    }

    /// [`EarthGenerator::generate`], loading the tiles it needs. When cancelled, `out` is air and the cube is
    /// reported as uniform air.
    pub fn generate_blocking(&self, cube: SectionPos, out: &mut [u32; 4096]) -> Option<u32> {
        let r = Rect {
            x0: cube.x * 16,
            z0: cube.z * 16,
            x1: cube.x * 16 + 15,
            z1: cube.z * 16 + 15,
            zoom: self.base_zoom(),
        };
        match self.wait_for(r, || self.cube_core(Access::Fetch, cube, out)) {
            Some(uniform) => uniform,
            None => {
                let air = self.blocks().air;
                out.fill(air);
                Some(air)
            }
        }
    }

    /// [`EarthGenerator::block_at`], loading the tile it needs. Air when cancelled.
    pub fn block_at_blocking(&self, x: i32, y: i32, z: i32, step: i32) -> u32 {
        self.wait_for(self.point_rect(x, z, step), || {
            self.block_core(Access::Fetch, [x, y, z], step)
        })
        .unwrap_or(self.blocks().air)
    }

    /// [`EarthGenerator::first_at_column`], loading the tile it needs. `None` when cancelled.
    pub fn first_at_column_blocking(
        &self,
        x: i32,
        z: i32,
        ys: &[i32],
        step: i32,
        mut stop: impl FnMut(u32) -> bool,
    ) -> Option<(usize, u32)> {
        self.wait_for(self.point_rect(x, z, step), || {
            self.column_core(Access::Fetch, x, z, ys, step, &mut stop)
        })
        .flatten()
    }

    /// [`EarthGenerator::surface_y`], loading the tile it needs. `None` when cancelled.
    pub fn surface_y_blocking(&self, x: i32, z: i32) -> Option<i32> {
        self.wait_for(self.point_rect(x, z, 1), || {
            self.surface_core(Access::Fetch, x, z, 1)
        })
    }

    /// [`EarthGenerator::chunk_tops`], loading the tiles it needs. `None` when cancelled.
    pub fn chunk_tops_blocking(&self, cx: i32, cz: i32) -> Option<Vec<i32>> {
        let r = Rect {
            x0: cx * 16,
            z0: cz * 16,
            x1: cx * 16 + 15,
            z1: cz * 16 + 15,
            zoom: self.base_zoom(),
        };
        self.wait_for(r, || self.tops_core(Access::Fetch, cx, cz))
    }

    /// [`EarthGenerator::classify_box_at`], loading the tiles it needs. `Mixed` when cancelled.
    pub fn classify_box_blocking(&self, min: [i32; 3], max: [i32; 3], step: i32) -> Fill {
        let r = Rect {
            x0: min[0],
            z0: min[2],
            x1: max[0],
            z1: max[2],
            zoom: self.zoom_for_step(step),
        };
        self.wait_for(r, || self.box_core(Access::Fetch, min, max, step))
            .unwrap_or(Fill::Mixed)
    }

    /// [`EarthGenerator::classify_node`], loading the tiles it needs. `Mixed` when cancelled.
    pub fn classify_node_blocking(&self, level: u32, x: i32, y: i32, z: i32) -> Fill {
        self.wait_for(self.node_rect(level, x, z), || {
            self.node_core(Access::Fetch, level, [x, y, z])
        })
        .unwrap_or(Fill::Mixed)
    }
}

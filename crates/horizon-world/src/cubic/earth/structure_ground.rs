//! Vanilla structures on Earth: the ground they stand on, and the terrain adapted to them.
//!
//! Earth runs Cubic's structures ([`crate::cubic::custom::structures`]); this is what the generator gives them.
//!
//! * The heights a structure start reads come from [`EarthGenerator::chunk_ground`]: each column's topmost solid
//!   block (the ground; under water, the bed) and the top water block over it, from the same final surfaces and
//!   kinds the cubes are made of. So a village stands on the real ground at any height (a plain at 64, a pass at
//!   1,500, a coast at 2), a shipwreck lies on the real sea floor, and dry land under the sea level (with land
//!   cover) is ground, not sea.
//! * Vanilla's terrain adaptation (the beardifier: ground filled under a village house, cut away over it) is a
//!   density added to the terrain's. Earth's ground is read as a density of blocks under the surface
//!   (`top - y + 0.5`) and the beard added the way Custom Cubic adds it ([`crate::cubic::custom::adapt`]); a
//!   column's top moves up or down through the blocks joined to the ground that this turns solid or air
//!   ([`adapt_ground`]). The column is then made again with its
//!   top layers ([`CARRIED`] blocks: grass and the soil under it) moved to the new top: raised ground is filled
//!   with the block under those layers, lowered ground is cut to what was over the old top (air, or water).
//!   Away from structures every block is exactly the plain generator's.

use horizon_core::density::Beardifier;
use horizon_core::position::SectionPos;

use super::generator::{Access, EarthGenerator, Gap, NotReady, Rect};
use super::surface::SurfaceKind;
use crate::cubic::custom::{NO_TOP, adapt};

/// The blocks at the top of a column that move with it when a structure raises or lowers the ground: the top
/// block and the soil under it (grass over three dirt, then the first block of what lies under them).
pub const CARRIED: i32 = 5;

impl EarthGenerator {
    /// `[ground, water]` of each column of chunk (`cx`, `cz`) (`z << 4 | x`): the topmost solid block, and the
    /// top water block over it or [`NO_TOP`] for dry ground.
    pub(super) fn ground_core(&self, a: Access, cx: i32, cz: i32) -> Result<Vec<[i32; 2]>, Gap> {
        let mut surfaces = self.tops_core(a, cx, cz)?;
        let kinds = self.kinds_core(a, [cx * 16, cz * 16], [16, 16], 1, 1, &mut surfaces)?;
        Ok(surfaces
            .iter()
            .zip(kinds)
            .map(|(&s, kind)| match kind {
                SurfaceKind::Water { depth, .. } => [s.saturating_sub(depth as i32), s],
                SurfaceKind::Marsh => [s - 1, s],
                _ => [s, NO_TOP],
            })
            .collect())
    }

    fn chunk_rect(&self, cx: i32, cz: i32) -> Rect {
        self.kind_rect(
            Rect {
                x0: cx * 16,
                z0: cz * 16,
                x1: cx * 16 + 15,
                z1: cz * 16 + 15,
                zoom: self.base_zoom(),
            },
            1,
        )
    }

    /// The ground structures stand on in chunk (`cx`, `cz`): see [`EarthGenerator::ground_core`].
    pub fn chunk_ground(&self, cx: i32, cz: i32) -> Result<Vec<[i32; 2]>, NotReady> {
        let r = self.chunk_rect(cx, cz);
        self.ground_core(Access::Local, cx, cz)
            .map_err(|_| NotReady {
                tiles: self.missing_in(r),
            })
    }

    /// [`EarthGenerator::chunk_ground`], loading what it lacks. `None` when cancelled.
    pub fn chunk_ground_blocking(&self, cx: i32, cz: i32) -> Option<Vec<[i32; 2]>> {
        self.wait_for(self.chunk_rect(cx, cz), || {
            self.ground_core(Access::Fetch, cx, cz)
        })
    }

    /// [`EarthGenerator::generate_blocking`] with the terrain adapted to the structures nearby (`beard`, the
    /// chunk's): see the module docs. Columns the beard does not change are exactly the plain cube's.
    pub fn generate_adapted_blocking(
        &self,
        cube: SectionPos,
        out: &mut [u32; 4096],
        beard: &Beardifier,
    ) -> Option<u32> {
        let uniform = self.generate_blocking(cube, out);
        let [x0, y0, z0] = [cube.x * 16, cube.y * 16, cube.z * 16];
        let Some(near) = beard.restricted(
            [x0, crate::cubic::MIN_Y, z0],
            [x0 + 15, crate::cubic::MAX_Y, z0 + 15],
        ) else {
            return uniform;
        };
        // Nothing changes above the beard's reach (a column's new top is at most its top).
        if near.reach().is_none_or(|r| y0 > r.maximum()[1]) {
            return uniform;
        }
        let Some(ground) = self.chunk_ground_blocking(cube.x, cube.z) else {
            return uniform;
        };
        let old: [i32; 256] = std::array::from_fn(|i| ground[i][0]);
        let mut new = old;
        adapt_ground(&near, cube.x, cube.z, &mut new);
        let mut changed = false;
        for i in 0..256 {
            let (s, t) = (old[i], new[i]);
            if s == t || s == NO_TOP || t == NO_TOP {
                continue;
            }
            let lo = (s.min(t) - CARRIED + 1).max(y0);
            let hi = s.max(t).min(y0 + 15);
            if lo > hi {
                continue;
            }
            let from: Vec<i32> = (lo..=hi).map(|y| moved_from(y, s, t)).collect();
            let (x, z) = (x0 + (i & 15) as i32, z0 + (i >> 4) as i32);
            let mut blocks = Vec::with_capacity(from.len());
            self.first_at_column_blocking(x, z, &from, 1, |b| {
                blocks.push(b);
                false
            });
            if blocks.len() != from.len() {
                // Cancelled: the cube is thrown away anyway.
                continue;
            }
            for (y, block) in (lo..=hi).zip(blocks) {
                out[((y - y0) as usize) << 8 | i] = block;
            }
            changed = true;
        }
        if !changed {
            return uniform;
        }
        let first = out[0];
        out.iter().all(|&b| b == first).then_some(first)
    }
}

/// Where the block at height `y` of a column whose top `s` the structures moved to `t` comes from in the plain
/// column: over the new top, what was over the old one; the top [`CARRIED`] blocks moved by `t - s`; under them,
/// the block that was under the carried layers.
pub fn moved_from(y: i32, s: i32, t: i32) -> i32 {
    if y > t {
        s + 1
    } else if y - (t - s) > s - CARRIED {
        y - (t - s)
    } else {
        s - CARRIED
    }
}

/// Vanilla's terrain adaptation of chunk (`cx`, `cz`) whose columns' topmost solid blocks are `tops` (changed in
/// place): the ground is read as a density of blocks under the top, `beard` added, and each column's top raised
/// through the blocks over it that turn solid, or lowered through its own that turn air. Columns the beard does
/// not reach keep their top.
pub fn adapt_ground(beard: &Beardifier, cx: i32, cz: i32, tops: &mut [i32; 256]) {
    let (x0, z0) = (cx * 16, cz * 16);
    let Some(near) = beard.restricted(
        [x0, crate::cubic::MIN_Y, z0],
        [x0 + 15, crate::cubic::MAX_Y, z0 + 15],
    ) else {
        return;
    };
    let Some(reach) = near.reach() else {
        return;
    };
    let (low, high) = (reach.minimum(), reach.maximum());
    for (i, top) in tops.iter_mut().enumerate() {
        let (x, z) = (x0 + (i & 15) as i32, z0 + (i >> 4) as i32);
        let s = *top;
        // A beard under the ground leaves the top as it is (as Custom Cubic's adaptation does).
        if s == NO_TOP || s > high[1] || x < low[0] || x > high[0] || z < low[2] || z > high[2] {
            continue;
        }
        let solid = |y: i32| adapt(f64::from(s - y) + 0.5, near.sample([x, y, z])) > 0.0;
        // The top moves only through blocks joined to the ground: up while the block over it turns solid, down
        // while its own turns air. A lump the beard makes under a piece high over the ground is not ground.
        let mut t = s;
        if solid(s) {
            while t < high[1] && solid(t + 1) {
                t += 1;
            }
        } else {
            // Under the reach the beard adds nothing: the plain ground.
            while t >= low[1] && !solid(t) {
                t -= 1;
            }
        }
        *top = t;
    }
}

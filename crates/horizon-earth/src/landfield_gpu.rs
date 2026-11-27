//! A "class window plan" for [`LandCoverField::class_grid`]: everything a GPU kernel needs to compute the same
//! window of land classes, laid out as plain numbers, plus a pure-Rust evaluator that reads ONLY the plan and
//! gives, bit for bit, what `class_grid` gives. The evaluator is the reference algorithm the shader copies line
//! by line: explicit `f64` operations in the order the CPU does them, integers as `i64`/`u64`/`u32`, tables
//! searched linearly or indexed directly, no hash maps.
//!
//! What the plan holds, and why:
//!
//! * **Per window**: sizes, the metres of ground a sample stands for, the size in blocks of a pixel (`px`,
//!   `pz`), the warp octaves (amplitude times pixel size, the two salts), and the projection numbers
//!   (`lat0`, `lon0`, metres per block, metres per degree of longitude, the Earth radius, radians to degrees).
//! * **Per column and per row** ([`AxisSample`]): the block coordinate and, for each warp octave, the part of
//!   the lattice noise that depends on that axis alone: the smoothstep weight and the two lattice hash terms.
//!   A noise value at a sample is then `mix64(salt ^ column_term ^ row_term)` and a lerp, which is
//!   bit-identical to [`crate::warp::value_noise`] (the hash is an XOR of three independent terms).
//! * **Files** ([`PlanFile`]) and a dense **tile table** ([`ClassWindowPlan::tile_table`]) of the level each
//!   file is read at, with the **tiles** ([`PlanTile`]) the window (plus the warp's reach plus the neighbour
//!   pixels of the vote) can touch, decoded.
//!
//! What the evaluator answers for a place the plan does not cover (a file or a tile that is not in the plan) is
//! [`OUTSIDE_PLAN`]; the caller then falls back to the CPU. With a plan built by
//! [`LandCoverField::class_window_plan`] that does not happen unless the window's reach was underestimated
//! (the tests assert it never does).
//!
//! Where the plan is more conservative than `class_grid`: it returns `Err(fetch)` for a file index or a tile
//! that the window's bounding box (plus the warp's largest offset plus one block) touches in the plan's own
//! tile table, even if no sample would have read it; `class_grid` would only fail for a read that happens.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::{
    Fetch, LAND_WARP_COARSE_PX, LAND_WARP_COARSE_WAVELENGTH_PX, LAND_WARP_FINE_PX,
    LAND_WARP_FINE_WAVELENGTH_PX, LAND_WARP_MAX_PX, LandCoverField, LandTile, WARP_SALTS,
};
use crate::landcover::{FILE_DEGREES, LandClass, SPECK_GRID, SPECK_KEEP, WorldCoverFile, level_for};
use crate::projection::EARTH_RADIUS_M;
use crate::warp::{cell_hash, mix64};

/// What [`ClassWindowPlan::evaluate`] gives for a sample whose file or tile is not in the plan.
pub const OUTSIDE_PLAN: u8 = 255;
/// A [`ClassWindowPlan::tile_table`] entry for a tile that is not in the plan.
pub const NO_TILE: u32 = u32::MAX;

/// The multiplier of the column term of the lattice hash.
const LATTICE_COLUMN_MUL: u64 = 0x9E37_79B9_7F4A_7C15;
/// The addend of the row term of the lattice hash.
const LATTICE_ROW_ADD: u64 = 0xD1B5_4A32_D192_ED03;
/// XORed into a file's speck salt.
const SPECK_SALT: u64 = 0x5EC4_5BEC_0000_0001;

/// `[0, 1)` from a hash, as [`crate::warp`] makes it.
fn unit_from_hash(h: u64) -> f64 {
    (h >> 11) as f64 / (1_u64 << 53) as f64
}

/// [`crate::projection::wrap_longitude`], with `rem_euclid` written out.
fn wrap_longitude(lon: f64) -> f64 {
    let r = (lon + 180.0) % 360.0;
    let r = if r < 0.0 { r + 360.0 } else { r };
    r - 180.0
}

/// The part of one warp octave's lattice noise that depends on one axis: columns (x) and rows (z) each have
/// their own. The noise at `(column, row)` is [`octave_noise`] of the column's and the row's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisOctave {
    /// The lattice cell, `floor(u)` of the noise coordinate, as `i64`. Not used by the evaluator (its terms
    /// are), kept for checks.
    pub cell: i64,
    /// `fade(u - floor(u))` with `fade(t) = t * t * (3.0 - 2.0 * t)`.
    pub fade: f64,
    /// The hash term of lattice line `cell`. For a column: `(cell as u64).wrapping_mul(0x9E3779B97F4A7C15)`.
    /// For a row: `mix64((cell as u64).wrapping_add(0xD1B54A32D192ED03))`.
    pub term0: u64,
    /// The same for lattice line `cell + 1`.
    pub term1: u64,
}

impl AxisOctave {
    fn parts(u: f64) -> (i64, f64) {
        let floor = u.floor();
        let t = u - floor;
        (floor as i64, t * t * (3.0 - 2.0 * t))
    }

    /// The column's part of the noise at noise coordinate `u` (the x coordinate over the octave's
    /// wavelength in blocks).
    pub fn column(u: f64) -> Self {
        let (cell, fade) = Self::parts(u);
        Self {
            cell,
            fade,
            term0: (cell as u64).wrapping_mul(LATTICE_COLUMN_MUL),
            term1: (cell.wrapping_add(1) as u64).wrapping_mul(LATTICE_COLUMN_MUL),
        }
    }

    /// The row's part of the noise at noise coordinate `v` (the z coordinate over the wavelength in blocks).
    pub fn row(v: f64) -> Self {
        let (cell, fade) = Self::parts(v);
        Self {
            cell,
            fade,
            term0: mix64((cell as u64).wrapping_add(LATTICE_ROW_ADD)),
            term1: mix64((cell.wrapping_add(1) as u64).wrapping_add(LATTICE_ROW_ADD)),
        }
    }
}

/// Lattice value noise in `[-1, 1]` from the column's and the row's parts: [`crate::warp::value_noise`] bit for
/// bit.
pub fn octave_noise(col: &AxisOctave, row: &AxisOctave, salt: u64) -> f64 {
    let a = unit_from_hash(mix64(salt ^ col.term0 ^ row.term0));
    let b = unit_from_hash(mix64(salt ^ col.term1 ^ row.term0));
    let c = unit_from_hash(mix64(salt ^ col.term0 ^ row.term1));
    let d = unit_from_hash(mix64(salt ^ col.term1 ^ row.term1));
    let top = a + (b - a) * col.fade;
    let bottom = c + (d - c) * col.fade;
    2.0 * (top + (bottom - top) * row.fade) - 1.0
}

/// One column (or row) of the window: the block coordinate of the sample's centre and the two warp octaves'
/// parts (index 0 the coarse octave, 1 the fine).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisSample {
    /// `f64::from(x0) + (i as f64) * f64::from(stride) + 0.5` (z for a row).
    pub pos: f64,
    pub octaves: [AxisOctave; 2],
}

/// One warp octave of the window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctavePlan {
    /// The lattice spacing in pixels (informative: the columns' and rows' noise coordinates are made from it).
    pub wavelength_px: f64,
    /// `amplitude * px`: the offset in blocks along x is `amplitude_x * noise` (one multiply, as the CPU does).
    pub amplitude_x: f64,
    /// `amplitude * pz`.
    pub amplitude_z: f64,
    /// The salt of the noise that moves x.
    pub salt_x: u64,
    /// The salt of the noise that moves z.
    pub salt_z: u64,
}

/// A WorldCover file the window can touch, with the level it is read at. `absent` files (the source has none:
/// open sea) and files without the level read as no data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanFile {
    pub file: WorldCoverFile,
    /// `file.south` and `file.west` in degrees (multiples of 3).
    pub south: i16,
    pub west: i16,
    /// The source has no such file: every sample in it is no data.
    pub absent: bool,
    /// The level read (`level_for`), 0 the finest. 0 for an absent file.
    pub level_no: u32,
    /// The level's size in pixels. Zero (with the rest) for an absent file or a file with no such level: no
    /// data.
    pub width: u32,
    pub height: u32,
    /// The tiles' size in pixels.
    pub tile_width: u32,
    pub tile_height: u32,
    /// How many tile columns and rows the level has.
    pub tiles_across: u32,
    pub tiles_down: u32,
    /// `((south as u16 as u64) << 32 | (west as u16 as u64) << 16) ^ level_no`: the salt of the speck hash.
    pub speck_salt: u64,
    /// Where this file's `tiles_across * tiles_down` entries start in [`ClassWindowPlan::tile_table`]; the
    /// entry of tile `(col, row)` is `tile_table[tile_table_base + row * tiles_across + col]`.
    pub tile_table_base: u32,
}

impl PlanFile {
    /// Whether the file reads as no data throughout.
    pub fn is_no_data(&self) -> bool {
        self.absent || self.width == 0 || self.height == 0
    }
}

/// A decoded tile the plan holds.
#[derive(Clone, Debug)]
pub struct PlanTile {
    /// Index into [`ClassWindowPlan::files`].
    pub file: u32,
    /// The tile's number in the level (row-major from the top left).
    pub tile_no: u32,
    pub col: u32,
    pub row: u32,
    /// `None`: a tile the file leaves out or the source will not give: it reads as code 0 / no data.
    pub tile: Option<Arc<LandTile>>,
}

impl PlanTile {
    /// The tile's size in pixels (0 by 0 for a blank tile).
    pub fn width(&self) -> u32 {
        self.tile.as_ref().map_or(0, |t| t.width)
    }

    pub fn height(&self) -> u32 {
        self.tile.as_ref().map_or(0, |t| t.height)
    }

    /// The class codes of the pixels, row by row (`width * height` bytes; empty for a blank tile). These are
    /// WorldCover codes (0, 10..100), after the towns are given back and the specks softened.
    pub fn codes(&self) -> &[u8] {
        self.tile.as_ref().map_or(&[], |t| &t.pixels)
    }

    /// The specks of the tile: `(pixel index, the speck's own code)`, sorted by index.
    pub fn specks(&self) -> &[(u32, u8)] {
        self.tile.as_ref().map_or(&[], |t| &t.specks)
    }

    /// An identity of the decoded data (the address of the tile's allocation): equal for the same decoded
    /// tile, so a GPU cache keyed by `(file, level, tile_no)` can tell whether its copy is still this one.
    pub fn identity(&self) -> usize {
        self.tile.as_ref().map_or(0, |t| Arc::as_ptr(t) as usize)
    }

    /// The code at pixel `(x, y)` of the tile; `None` outside it (a blank tile has no pixels).
    pub fn code_at(&self, x: u32, y: u32) -> Option<u8> {
        self.tile.as_ref()?.code_at(x, y)
    }

    /// Writes the speck plane into `out` (`width * height` bytes): the speck's own code at the pixels that are
    /// specks, 0 elsewhere. Original codes of specks are always solid non-zero classes, so 0 means "not a
    /// speck".
    pub fn write_speck_plane(&self, out: &mut [u8]) {
        out.fill(0);
        for &(index, code) in self.specks() {
            if let Some(slot) = out.get_mut(index as usize) {
                *slot = code;
            }
        }
    }

    /// The speck plane as a vector, or empty when the tile has no specks.
    pub fn speck_plane(&self) -> Vec<u8> {
        if self.specks().is_empty() {
            return Vec::new();
        }
        let mut plane = vec![0; self.width() as usize * self.height() as usize];
        self.write_speck_plane(&mut plane);
        plane
    }
}

/// Everything needed to compute the classes of one window of [`LandCoverField::class_grid`] without the store.
#[derive(Clone, Debug)]
pub struct ClassWindowPlan {
    pub width: usize,
    pub depth: usize,
    pub metres_per_sample: f64,
    /// The size in blocks of a pixel, east-west and north-south (`LandCoverField::pixel_blocks`).
    pub px: f64,
    pub pz: f64,
    /// Index 0 the coarse warp octave, 1 the fine.
    pub octaves: [OctavePlan; 2],
    pub metres_per_block: f64,
    /// The projection's origin in degrees (longitude wrapped).
    pub lat0: f64,
    pub lon0: f64,
    /// `EARTH_RADIUS_M * lat0.to_radians().cos().to_radians()`.
    pub metres_per_degree_lon: f64,
    pub earth_radius_m: f64,
    /// `1.0_f64.to_degrees()`: `x.to_degrees()` is `x * radians_to_degrees`.
    pub radians_to_degrees: f64,
    /// `width` columns, each with its noise parts.
    pub columns: Vec<AxisSample>,
    /// `depth` rows.
    pub rows: Vec<AxisSample>,
    /// The files the window can touch, sorted by `(south, west)`.
    pub files: Vec<PlanFile>,
    /// For every non-absent file with a level, `tiles_across * tiles_down` entries (see
    /// [`PlanFile::tile_table_base`]): the index into [`ClassWindowPlan::tiles`], or [`NO_TILE`].
    pub tile_table: Vec<u32>,
    /// The tiles held, in table order of the files.
    pub tiles: Vec<PlanTile>,
}

impl LandCoverField {
    /// The plan for [`LandCoverField::class_grid`] of the same arguments (see [`ClassWindowPlan`]). `Err` with
    /// the first fetch that is missing: the index of a file the window may touch, or a tile of it (what is
    /// already held, in memory or on disk, is decoded and kept in the plan; nothing is fetched).
    pub fn class_window_plan(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        stride: i32,
        metres_per_sample: f64,
    ) -> Result<ClassWindowPlan, Fetch> {
        let (px, pz) = self.pixel_blocks(metres_per_sample);
        let octave = |wavelength_px: f64, amplitude: f64, salts: [u64; 2]| OctavePlan {
            wavelength_px,
            amplitude_x: amplitude * px,
            amplitude_z: amplitude * pz,
            salt_x: salts[0],
            salt_z: salts[1],
        };
        let octaves = [
            octave(
                LAND_WARP_COARSE_WAVELENGTH_PX,
                LAND_WARP_COARSE_PX,
                [WARP_SALTS[0], WARP_SALTS[1]],
            ),
            octave(
                LAND_WARP_FINE_WAVELENGTH_PX,
                LAND_WARP_FINE_PX,
                [WARP_SALTS[2], WARP_SALTS[3]],
            ),
        ];
        let axis = |origin: i32, count: usize, p: f64, make: fn(f64) -> AxisOctave| {
            (0..count)
                .map(|i| {
                    let pos = f64::from(origin) + (i as f64) * f64::from(stride) + 0.5;
                    let octave = |o: &OctavePlan| make(pos / (o.wavelength_px * p));
                    AxisSample {
                        pos,
                        octaves: [octave(&octaves[0]), octave(&octaves[1])],
                    }
                })
                .collect::<Vec<_>>()
        };
        let columns = axis(x0, width, px, AxisOctave::column);
        let rows = axis(z0, depth, pz, AxisOctave::row);

        let projection = self.projection();
        let lat0 = projection.reference_latitude();
        // The origin's longitude: at block (0, 0) the east offset is exactly 0.0, so this is `lon0 + 0.0`.
        let lon0 = projection.to_lat_lon_unwrapped(0.0, 0.0).lon;
        let mut plan = ClassWindowPlan {
            width,
            depth,
            metres_per_sample,
            px,
            pz,
            octaves,
            metres_per_block: projection.metres_per_block(),
            lat0,
            lon0,
            metres_per_degree_lon: EARTH_RADIUS_M * lat0.to_radians().cos().to_radians(),
            earth_radius_m: EARTH_RADIUS_M,
            radians_to_degrees: 1.0_f64.to_degrees(),
            columns,
            rows,
            files: Vec::new(),
            tile_table: Vec::new(),
            tiles: Vec::new(),
        };
        if width > 0 && depth > 0 {
            self.plan_files_and_tiles(&mut plan, metres_per_sample)?;
        }
        Ok(plan)
    }

    /// Fills the files, the tile table and the tiles of `plan`: the geographic box of the window (the samples'
    /// extent, plus the warp's largest offset and a block), the 3-degree cells of it as
    /// [`crate::landfield::LandCoverStore::needs`] walks them, and in each the tiles whose pixels the box
    /// touches, plus the pixel around (the vote reads one pixel past).
    fn plan_files_and_tiles(
        &self,
        plan: &mut ClassWindowPlan,
        metres_per_sample: f64,
    ) -> Result<(), Fetch> {
        let (first_x, last_x) = (plan.columns[0].pos, plan.columns[plan.width - 1].pos);
        let (first_z, last_z) = (plan.rows[0].pos, plan.rows[plan.depth - 1].pos);
        let (gx, gz) = (
            LAND_WARP_MAX_PX * plan.px + 1.0,
            LAND_WARP_MAX_PX * plan.pz + 1.0,
        );
        let (xa, xb) = (first_x.min(last_x) - gx, first_x.max(last_x) + gx);
        let (za, zb) = (first_z.min(last_z) - gz, first_z.max(last_z) + gz);
        let (xm, zm) = ((xa + xb) / 2.0, (za + zb) / 2.0);
        let mut south = f64::INFINITY;
        let mut north = f64::NEG_INFINITY;
        let mut west = f64::INFINITY;
        let mut east = f64::NEG_INFINITY;
        for (x, z) in [
            (xa, za),
            (xm, za),
            (xb, za),
            (xa, zm),
            (xb, zm),
            (xa, zb),
            (xm, zb),
            (xb, zb),
        ] {
            let at = self.projection().to_lat_lon_unwrapped(x, z);
            south = south.min(at.lat);
            north = north.max(at.lat);
            west = west.min(at.lon);
            east = east.max(at.lon);
        }
        if ![south, north, west, east].iter().all(|v| v.is_finite()) {
            return Ok(());
        }
        let (south, north) = (south.clamp(-90.0, 90.0), north.clamp(-90.0, 90.0));
        // At most the whole circle.
        let east = east.min(west + 359.999_999);
        let step = f64::from(FILE_DEGREES);
        let (k0, k1) = ((west / step).floor() as i64, (east / step).floor() as i64);
        let (r0, r1) = ((south / step).floor() as i64, (north / step).floor() as i64);

        struct Acc {
            file: WorldCoverFile,
            absent: bool,
            tiles: BTreeSet<(u32, u32)>,
        }
        let mut accs: Vec<Acc> = Vec::new();
        for r in r0..=r1 {
            for k in k0..=k1 {
                let Some(file) =
                    WorldCoverFile::containing(r as f64 * step + 1.5, k as f64 * step + 1.5)
                else {
                    continue;
                };
                // A pole's own row holds the pole: the 90th parallel is not a row of its own.
                if r as f64 * step >= 90.0 {
                    continue;
                }
                let at = match accs.iter().position(|a| a.file == file) {
                    Some(at) => at,
                    None => {
                        accs.push(Acc {
                            file,
                            absent: self.store.is_absent(file),
                            tiles: BTreeSet::new(),
                        });
                        accs.len() - 1
                    }
                };
                if accs[at].absent {
                    continue;
                }
                let index = self.store.index_of(file)?;
                let level_no = level_for(&index.levels, metres_per_sample);
                let Some(level) = index.levels.get(level_no) else {
                    continue;
                };
                if level.width == 0
                    || level.height == 0
                    || level.tile_width == 0
                    || level.tile_height == 0
                {
                    continue;
                }
                let cell_south = south.max(r as f64 * step);
                let cell_north = north.min(r as f64 * step + step);
                let cell_west = west.max(k as f64 * step);
                let cell_east = east.min(k as f64 * step + step);
                let (xa, ya) = file.pixel(cell_north, cell_west, level.width, level.height);
                let (xb, yb) = file.pixel(cell_south, cell_east, level.width, level.height);
                let (x_lo, x_hi) = (xa.min(xb).saturating_sub(1), xa.max(xb) + 1);
                let (y_lo, y_hi) = (ya.min(yb).saturating_sub(1), ya.max(yb) + 1);
                let (x_hi, y_hi) = (x_hi.min(level.width - 1), y_hi.min(level.height - 1));
                for row in y_lo / level.tile_height..=y_hi / level.tile_height {
                    for col in x_lo / level.tile_width..=x_hi / level.tile_width {
                        accs[at].tiles.insert((col, row));
                    }
                }
            }
        }
        accs.sort_by_key(|a| a.file);
        for acc in accs {
            let file_no = plan.files.len() as u32;
            let (south_i, west_i) = (acc.file.south, acc.file.west);
            let mut entry = PlanFile {
                file: acc.file,
                south: south_i,
                west: west_i,
                absent: acc.absent,
                level_no: 0,
                width: 0,
                height: 0,
                tile_width: 0,
                tile_height: 0,
                tiles_across: 0,
                tiles_down: 0,
                speck_salt: 0,
                tile_table_base: plan.tile_table.len() as u32,
            };
            if !acc.absent {
                let index = self.store.index_of(acc.file)?;
                let level_no = level_for(&index.levels, metres_per_sample);
                if let Some(level) = index.levels.get(level_no)
                    && level.width != 0
                    && level.height != 0
                    && level.tile_width != 0
                    && level.tile_height != 0
                {
                    entry.level_no = level_no as u32;
                    entry.width = level.width;
                    entry.height = level.height;
                    entry.tile_width = level.tile_width;
                    entry.tile_height = level.tile_height;
                    entry.tiles_across = level.tiles_across();
                    entry.tiles_down = level.tiles_down();
                    entry.speck_salt = ((u64::from(south_i as u16) << 32)
                        | (u64::from(west_i as u16) << 16))
                        ^ (level_no as u64);
                    let slots = entry.tiles_across as usize * entry.tiles_down as usize;
                    plan.tile_table.resize(plan.tile_table.len() + slots, NO_TILE);
                    for &(col, row) in &acc.tiles {
                        let Some(tile_no) = level.tile_index(col, row) else {
                            continue;
                        };
                        let tile = self.store.tile_of(acc.file, level_no, level, tile_no)?;
                        plan.tile_table[entry.tile_table_base as usize
                            + row as usize * entry.tiles_across as usize
                            + col as usize] = plan.tiles.len() as u32;
                        plan.tiles.push(PlanTile {
                            file: file_no,
                            tile_no: tile_no as u32,
                            col,
                            row,
                            tile,
                        });
                    }
                }
            }
            plan.files.push(entry);
        }
        Ok(())
    }
}

impl ClassWindowPlan {
    /// `(lat, lon)` of the warped world point `(x, z)`: [`crate::projection::Projection::to_lat_lon`] from the
    /// plan's numbers (longitude wrapped into `[-180, 180)`).
    pub fn lat_lon(&self, x: f64, z: f64) -> (f64, f64) {
        let east = x * self.metres_per_block;
        let north = -z * self.metres_per_block;
        let lat = self.lat0 + (north / self.earth_radius_m) * self.radians_to_degrees;
        let lon = self.lon0 + east / self.metres_per_degree_lon;
        (lat, wrap_longitude(lon))
    }

    /// The key of tile `k` of [`ClassWindowPlan::tiles`] in a cache that outlives the plan: the file, the level
    /// read and the tile number (pair it with [`PlanTile::identity`] to know the data is the same).
    pub fn tile_key(&self, k: usize) -> (WorldCoverFile, u32, u32) {
        let tile = &self.tiles[k];
        let file = &self.files[tile.file as usize];
        (file.file, file.level_no, tile.tile_no)
    }

    /// The class codes (`LandClass::code`) of the window, row by row, from the plan alone;
    /// [`OUTSIDE_PLAN`] for a sample whose file or tile the plan does not hold.
    pub fn evaluate(&self) -> Vec<u8> {
        let speck_planes: Vec<Vec<u8>> = self.tiles.iter().map(PlanTile::speck_plane).collect();
        let mut out = Vec::with_capacity(self.width.saturating_mul(self.depth).min(1 << 24));
        for row in &self.rows {
            for col in &self.columns {
                out.push(self.sample(col, row, &speck_planes).unwrap_or(OUTSIDE_PLAN));
            }
        }
        out
    }

    /// [`ClassWindowPlan::evaluate`] as classes; `None` when any sample is [`OUTSIDE_PLAN`].
    pub fn evaluate_classes(&self) -> Option<Vec<LandClass>> {
        self.evaluate()
            .into_iter()
            .map(LandClass::from_code)
            .collect()
    }

    /// One sample; `None` for outside the plan.
    fn sample(&self, col: &AxisSample, row: &AxisSample, speck_planes: &[Vec<u8>]) -> Option<u8> {
        let nodata = LandClass::NoData.code();
        // The warp: `LandCoverField::warped`.
        let [coarse, fine] = &self.octaves;
        let cx = coarse.amplitude_x
            * octave_noise(&col.octaves[0], &row.octaves[0], coarse.salt_x);
        let cz = coarse.amplitude_z
            * octave_noise(&col.octaves[0], &row.octaves[0], coarse.salt_z);
        let fx = fine.amplitude_x * octave_noise(&col.octaves[1], &row.octaves[1], fine.salt_x);
        let fz = fine.amplitude_z * octave_noise(&col.octaves[1], &row.octaves[1], fine.salt_z);
        let x = (col.pos + cx) + fx;
        let z = (row.pos + cz) + fz;
        let (lat, lon) = self.lat_lon(x, z);

        // `WorldCoverFile::containing`.
        if !(lat.is_finite() && lon.is_finite() && lat.abs() <= 90.0) {
            return Some(nodata);
        }
        let step = f64::from(FILE_DEGREES);
        let wrapped = wrap_longitude(lon);
        let file_south = (((lat / step).floor() * step).min(90.0 - step)) as i16;
        let file_west = (((wrapped / step).floor() * step).clamp(-180.0, 180.0 - step)) as i16;
        let file = self
            .files
            .iter()
            .find(|f| f.south == file_south && f.west == file_west)?;
        if file.is_no_data() {
            return Some(nodata);
        }

        // `WorldCoverFile::pixel` and `pixel_fraction`.
        let (level_w, level_h) = (file.width, file.height);
        let east_off = wrap_longitude(lon - f64::from(file.west));
        let fx_pixels = east_off / step * f64::from(level_w);
        let north_edge = f64::from(file.south) + f64::from(FILE_DEGREES);
        let fy_pixels = (north_edge - lat) / step * f64::from(level_h);
        let (cx_floor, cy_floor) = (fx_pixels.floor(), fy_pixels.floor());
        let pix_x = (cx_floor.max(0.0) as u32).min(level_w.saturating_sub(1));
        let pix_y = (cy_floor.max(0.0) as u32).min(level_h.saturating_sub(1));
        let fraction = |v: f64| {
            let f = v - v.floor();
            if f.is_finite() {
                f.clamp(0.0, 1.0 - f64::EPSILON)
            } else {
                0.0
            }
        };
        let (inside_x, inside_y) = (fraction(fx_pixels), fraction(fy_pixels));

        let (tile_col, tile_row) = (pix_x / file.tile_width, pix_y / file.tile_height);
        if tile_col >= file.tiles_across || tile_row >= file.tiles_down {
            return Some(nodata);
        }
        let slot = self.tile_slot(file, tile_col, tile_row);
        if slot == NO_TILE {
            return None;
        }
        let own = &self.tiles[slot as usize];
        if own.tile.is_none() {
            return Some(nodata);
        }
        let (lx, ly) = (pix_x % file.tile_width, pix_y % file.tile_height);

        // `LandTile::speck_kept`.
        let plane = &speck_planes[slot as usize];
        if !plane.is_empty() && lx < own.width() && ly < own.height() {
            let speck = plane[(ly * own.width() + lx) as usize];
            if speck != 0 {
                let cell = |f: f64| {
                    ((f * f64::from(SPECK_GRID)) as i64).clamp(0, i64::from(SPECK_GRID) - 1)
                };
                let key = cell(inside_y) * i64::from(SPECK_GRID) + cell(inside_x);
                let h = cell_hash(
                    i64::from(pix_x),
                    i64::from(pix_y),
                    key,
                    file.speck_salt ^ SPECK_SALT,
                );
                if h < SPECK_KEEP {
                    return Some(class_code(speck));
                }
            }
        }

        // The vote of the four pixels whose centres are nearest.
        let (u, v) = (inside_x - 0.5, inside_y - 0.5);
        let (step_x, tx) = (if u >= 0.0 { 1_i64 } else { -1 }, u.abs());
        let (step_y, ty) = (if v >= 0.0 { 1_i64 } else { -1 }, v.abs());
        let (own_x, own_y) = (i64::from(pix_x), i64::from(pix_y));
        let samples = [
            (own_x, own_y, (1.0 - tx) * (1.0 - ty)),
            (own_x + step_x, own_y, tx * (1.0 - ty)),
            (own_x, own_y + step_y, (1.0 - tx) * ty),
            (own_x + step_x, own_y + step_y, tx * ty),
        ];
        let mut votes: [(u8, f64); 4] = [(0, 0.0); 4];
        let mut used = 0;
        for (sx, sy, weight) in samples {
            let code = self.neighbour_code(file, sx, sy)?;
            match votes[..used].iter_mut().find(|(c, _)| *c == code) {
                Some((_, total)) => *total += weight,
                None => {
                    votes[used] = (code, weight);
                    used += 1;
                }
            }
        }
        let mut best = votes[0];
        for &(code, total) in &votes[1..used] {
            if total > best.1 + 1e-12 {
                best = (code, total);
            }
        }
        Some(class_code(best.0))
    }

    fn tile_slot(&self, file: &PlanFile, col: u32, row: u32) -> u32 {
        self.tile_table
            [file.tile_table_base as usize + row as usize * file.tiles_across as usize + col as usize]
    }

    /// The code of pixel `(x, y)` of the file's level, clamped to the level; 0 in a tile the level has not or
    /// the plan holds blank; `None` for a tile that is not in the plan.
    fn neighbour_code(&self, file: &PlanFile, x: i64, y: i64) -> Option<u8> {
        let x = x.clamp(0, i64::from(file.width) - 1) as u32;
        let y = y.clamp(0, i64::from(file.height) - 1) as u32;
        let (col, row) = (x / file.tile_width, y / file.tile_height);
        let (lx, ly) = (x % file.tile_width, y % file.tile_height);
        if col >= file.tiles_across || row >= file.tiles_down {
            return Some(0);
        }
        let slot = self.tile_slot(file, col, row);
        if slot == NO_TILE {
            return None;
        }
        Some(self.tiles[slot as usize].code_at(lx, ly).unwrap_or(0))
    }
}

/// `LandClass::from_code(code).unwrap_or(NoData).code()`.
fn class_code(code: u8) -> u8 {
    LandClass::from_code(code)
        .unwrap_or(LandClass::NoData)
        .code()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::landfield::{LandCoverStore, Lookup, Wanted};
    use crate::projection::Projection;
    use crate::testing::{TempDir, TiffSpec, encode_tiff, serve_range};
    use std::collections::HashSet;

    /// A deterministic generator for the many windows.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + self.below((hi - lo) as u64) as i64
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    fn scramble(x: u32, y: u32, level: usize) -> u64 {
        mix64((u64::from(x) << 32) ^ u64::from(y) ^ ((level as u64) << 56) ^ 0xABCD_0123)
    }

    /// Bands of classes (built-up land and no data among them) with one- and two-pixel specks of other classes
    /// scattered over them.
    fn code(level: usize, x: u32, y: u32) -> u8 {
        const BANDS: [u8; 10] = [30, 10, 40, 50, 30, 60, 80, 20, 0, 100];
        let speck = |x: u32, y: u32| scramble(x, y, level) % 67 == 0;
        let origin = if speck(x, y) {
            Some((x, y))
        } else if x > 0 && speck(x - 1, y) {
            Some((x - 1, y))
        } else {
            None
        };
        if let Some((ox, oy)) = origin {
            return [60, 10, 20, 100, 90][(scramble(ox, oy, level) / 67 % 5) as usize];
        }
        BANDS[((x / 37 + y / 41) as usize + level * 5) % BANDS.len()]
    }

    /// 600 by 600 pixels in 256-pixel tiles, then 300 and 150; with `blank`, tile 4 of the finest level is
    /// left out of the file.
    fn miniature(blank: bool) -> Vec<u8> {
        let mut spec = TiffSpec::new(600, 600, 256, 2);
        if blank {
            spec = spec.editing_tiles(|level, _offsets, counts| {
                if level == 0 {
                    counts[4] = 0;
                }
            });
        }
        encode_tiff(&spec, &code).bytes
    }

    /// A pretend server holding the miniature file for every place but the absent ones.
    struct Server {
        bytes: Vec<u8>,
        absent: HashSet<WorldCoverFile>,
    }

    impl Server {
        fn new(absent: &[(f64, f64)], blank: bool) -> Self {
            Self {
                bytes: miniature(blank),
                absent: absent
                    .iter()
                    .map(|&(lat, lon)| WorldCoverFile::containing(lat, lon).unwrap())
                    .collect(),
            }
        }

        fn serve(&self, store: &LandCoverStore, fetch: &Fetch) {
            if self.absent.contains(&fetch.file) {
                store.mark_absent(fetch.file);
            } else {
                store
                    .provide(fetch, serve_range(&self.bytes, fetch.range))
                    .expect("the store takes what the server sent");
            }
        }
    }

    fn field(lat0: f64, lon0: f64, metres_per_block: f64, level0_m: f64) -> (TempDir, LandCoverField) {
        let dir = TempDir::new();
        let projection = Projection::local(lat0, lon0, metres_per_block).unwrap();
        let field = LandCoverField::new(Arc::new(LandCoverStore::new(dir.path(), 256)), projection)
            .with_level0_pixel_metres(level0_m);
        (dir, field)
    }

    #[derive(Clone, Copy, Debug)]
    struct Window {
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        stride: i32,
        metres: f64,
    }

    /// What one window cost: the fetches `class_grid` needed and the ones only the plan asked for.
    #[derive(Default, Debug, Clone, Copy)]
    struct Cost {
        windows: usize,
        samples: usize,
        plan_only_fetches: usize,
    }

    impl Cost {
        fn add(&mut self, other: Cost) {
            self.windows += other.windows;
            self.samples += other.samples;
            self.plan_only_fetches += other.plan_only_fetches;
        }
    }

    fn grid(field: &LandCoverField, server: &Server, w: Window) -> Vec<LandClass> {
        for _ in 0..600 {
            match field.class_grid(w.x0, w.z0, w.width, w.depth, w.stride, w.metres) {
                Ok(grid) => return grid,
                Err(fetch) => server.serve(field.store(), &fetch),
            }
        }
        panic!("class_grid never finishes for {w:?}");
    }

    fn plan(field: &LandCoverField, server: &Server, w: Window) -> (ClassWindowPlan, usize) {
        let mut asked = 0;
        for _ in 0..600 {
            match field.class_window_plan(w.x0, w.z0, w.width, w.depth, w.stride, w.metres) {
                Ok(plan) => return (plan, asked),
                Err(fetch) => {
                    asked += 1;
                    server.serve(field.store(), &fetch);
                }
            }
        }
        panic!("the plan never finishes for {w:?}");
    }

    /// The plan's evaluation against `class_grid` and `class_grid_reference`, sample for sample.
    fn check(field: &LandCoverField, server: &Server, w: Window) -> (Cost, ClassWindowPlan) {
        let want = grid(field, server, w);
        let (plan, plan_only) = plan(field, server, w);
        let got = plan.evaluate();
        assert_eq!(got.len(), want.len(), "{w:?}");
        let mut outside = 0;
        for (n, (&g, &c)) in got.iter().zip(&want).enumerate() {
            if g == OUTSIDE_PLAN {
                outside += 1;
            } else if g != c.code() {
                panic!(
                    "{w:?}: sample {} ({}, {}) is {g}, class_grid says {:?}",
                    n,
                    n % w.width,
                    n / w.width,
                    c
                );
            }
        }
        assert_eq!(outside, 0, "{w:?}: samples outside the plan");
        assert_eq!(plan.evaluate_classes(), Some(want.clone()), "{w:?}");
        let reference = field
            .class_grid_reference(w.x0, w.z0, w.width, w.depth, w.stride, w.metres)
            .unwrap();
        assert_eq!(reference, want, "{w:?}");
        let cost = Cost {
            windows: 1,
            samples: want.len(),
            plan_only_fetches: plan_only,
        };
        (cost, plan)
    }

    const METRES: [f64; 8] = [10.0, 100.0, 556.0, 1200.0, 2300.0, 3000.0, 20_000.0, 1e9];
    const SIZES: [(usize, usize); 5] = [(38, 38), (1, 1), (7, 3), (64, 17), (3, 40)];
    const STRIDES: [i32; 9] = [1, 2, 3, 8, 16, 64, 100, 256, -7];

    /// Windows of every size and stride at random places within `reach` blocks of the origin.
    fn many(
        field: &LandCoverField,
        server: &Server,
        rng: &mut Rng,
        count: usize,
        reach: i64,
        strides: &[i32],
    ) -> Cost {
        let mut cost = Cost::default();
        for n in 0..count {
            let (width, depth) = SIZES[n % SIZES.len()];
            let stride = strides[(n / SIZES.len() + n) % strides.len()];
            let w = Window {
                x0: rng.range(-reach, reach) as i32 - (width as i32 * stride) / 2,
                z0: rng.range(-reach, reach) as i32 - (depth as i32 * stride) / 2,
                width,
                depth,
                stride,
                metres: METRES[rng.below(METRES.len() as u64) as usize],
            };
            cost.add(check(field, server, w).0);
        }
        cost
    }

    #[test]
    fn the_separated_lattice_noise_is_value_noise_bit_for_bit() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let salts = [
            0,
            1,
            super::WARP_SALTS[0],
            super::WARP_SALTS[1],
            super::WARP_SALTS[2],
            super::WARP_SALTS[3],
            u64::MAX,
        ];
        for n in 0..20_000 {
            let scale = [1.0, 40.0, 1e4, 1e7][n % 4];
            let (u, v) = ((rng.unit() - 0.5) * scale, (rng.unit() - 0.5) * scale);
            let salt = if n % 3 == 0 {
                rng.next()
            } else {
                salts[n % salts.len()]
            };
            let got = octave_noise(&AxisOctave::column(u), &AxisOctave::row(v), salt);
            let want = crate::warp::value_noise(u, v, salt);
            assert_eq!(got.to_bits(), want.to_bits(), "{u} {v} {salt}");
        }
        // Whole numbers, negative values and the lattice's own points.
        for u in -6..6 {
            for v in -6..6 {
                for d in [0.0, 0.5, 1.0 - f64::EPSILON] {
                    let (u, v) = (f64::from(u) + d, f64::from(v) + d);
                    let got = octave_noise(&AxisOctave::column(u), &AxisOctave::row(v), 77);
                    assert_eq!(got.to_bits(), crate::warp::value_noise(u, v, 77).to_bits());
                }
            }
        }
    }

    #[test]
    fn the_plans_projection_is_the_projections_bit_for_bit() {
        let mut rng = Rng(99);
        for (lat0, lon0, mpb) in [
            (28.5, 85.5, 10.0),
            (0.5, 179.99, 10.0),
            (-33.3, -179.5, 3.7),
            (85.0, 10.0, 100.0),
            (-84.9, -77.7, 1000.0),
            (0.0, 0.0, 1.0),
        ] {
            let (_dir, f) = field(lat0, lon0, mpb, 9.28);
            let plan = f.class_window_plan(0, 0, 0, 0, 1, 10.0).unwrap();
            for _ in 0..2000 {
                let scale = [100.0, 1e4, 1e6][rng.below(3) as usize];
                let (x, z) = ((rng.unit() - 0.5) * scale, (rng.unit() - 0.5) * scale);
                let at = f.projection().to_lat_lon(x, z);
                let (lat, lon) = plan.lat_lon(x, z);
                assert_eq!(lat.to_bits(), at.lat.to_bits(), "{lat0} {lon0} {x} {z}");
                assert_eq!(lon.to_bits(), at.lon.to_bits(), "{lat0} {lon0} {x} {z}");
            }
        }
    }

    #[test]
    fn a_window_inside_one_file_is_what_class_grid_gives() {
        let mut cost = Cost::default();
        for level0 in [556.0, 9.28] {
            let (_dir, f) = field(28.5, 85.5, 10.0, level0);
            let server = Server::new(&[], false);
            let mut rng = Rng(7 + level0 as u64);
            // Every size at every stride, around the origin.
            for &(width, depth) in &SIZES {
                for &stride in &STRIDES {
                    for metres in [10.0, 1200.0, 2300.0] {
                        let w = Window {
                            x0: -(width as i32 * stride) / 2,
                            z0: -(depth as i32 * stride) / 2,
                            width,
                            depth,
                            stride,
                            metres,
                        };
                        cost.add(check(&f, &server, w).0);
                    }
                }
            }
            cost.add(many(&f, &server, &mut rng, 150, 20_000, &STRIDES));
        }
        println!("one file: {cost:?}");
        assert!(cost.windows > 400);
    }

    #[test]
    fn the_plan_holds_what_the_window_reads_and_the_levels_follow_the_scale() {
        let (_dir, f) = field(28.5, 85.5, 10.0, 556.0);
        let server = Server::new(&[], false);
        let mut specks = 0;
        for (metres, level) in [(10.0, 0), (1200.0, 1), (2300.0, 2), (1e6, 2)] {
            let w = Window {
                x0: -800,
                z0: -800,
                width: 38,
                depth: 38,
                stride: 40,
                metres,
            };
            let (_, plan) = check(&f, &server, w);
            assert_eq!(plan.files.len(), 1);
            let file = plan.files[0];
            assert_eq!((file.south, file.west), (27, 84));
            assert_eq!(file.level_no, level);
            assert!(!file.absent);
            assert_eq!(file.tile_width, 256);
            assert_eq!(file.width, [600, 300, 150][level as usize]);
            assert_eq!(file.tiles_across, file.width.div_ceil(256));
            assert_eq!(file.speck_salt & 0xFFFF, u64::from(file.level_no));
            // The table indexes the tiles of the file.
            assert_eq!(
                plan.tile_table.len(),
                (file.tiles_across * file.tiles_down) as usize
            );
            let mut seen = 0;
            for (slot, &entry) in plan.tile_table.iter().enumerate() {
                if entry == NO_TILE {
                    continue;
                }
                seen += 1;
                let tile = &plan.tiles[entry as usize];
                assert_eq!(slot as u32, tile.row * file.tiles_across + tile.col);
                assert_eq!(tile.tile_no, tile.row * file.tiles_across + tile.col);
                assert_eq!(tile.codes().len(), 256 * 256);
                assert_eq!((tile.width(), tile.height()), (256, 256));
                let plane = tile.speck_plane();
                specks += tile.specks().len();
                if tile.specks().is_empty() {
                    assert!(plane.is_empty());
                } else {
                    assert_eq!(plane.len(), 256 * 256);
                    for &(index, own) in tile.specks() {
                        assert_eq!(plane[index as usize], own);
                        assert_ne!(own, 0);
                    }
                    assert_eq!(
                        plane.iter().filter(|&&c| c != 0).count(),
                        tile.specks().len()
                    );
                }
                assert_eq!(plan.tile_key(entry as usize).2, tile.tile_no);
                assert_ne!(tile.identity(), 0);
            }
            assert_eq!(seen, plan.tiles.len());
        }
        assert!(specks > 100, "the miniature file has specks: {specks}");
        // The same decoded tile is the same identity, as long as the store holds it.
        let w = Window {
            x0: 0,
            z0: 0,
            width: 4,
            depth: 4,
            stride: 1,
            metres: 10.0,
        };
        let (a, _) = plan(&f, &server, w);
        let (b, _) = plan(&f, &server, w);
        assert_eq!(a.tiles[0].identity(), b.tiles[0].identity());
    }

    #[test]
    fn windows_across_the_corner_of_four_files() {
        let mut cost = Cost::default();
        let mut most_files = 0;
        for (lat0, lon0, mpb, level0, strides) in [
            (29.99, 86.99, 10.0, 556.0, &[1, 8, 64, 256][..]),
            (30.0003, 87.0003, 10.0, 556.0, &[1, 8, 64, 256][..]),
            (29.9999, 86.9999, 10.0, 9.28, &[1, 8, 64, 256][..]),
            // Blocks of a kilometre: a window of strides up to 8 is already 300 km wide.
            (30.001, 86.999, 1000.0, 556.0, &[1, 3, 8][..]),
        ] {
            let (_dir, f) = field(lat0, lon0, mpb, level0);
            let server = Server::new(&[], false);
            let mut rng = Rng(31 + lon0 as u64);
            for &(width, depth) in &SIZES {
                for &stride in strides {
                    let w = Window {
                        x0: -(width as i32 * stride) / 2,
                        z0: -(depth as i32 * stride) / 2,
                        width,
                        depth,
                        stride,
                        metres: 100.0,
                    };
                    let (c, plan) = check(&f, &server, w);
                    cost.add(c);
                    most_files = most_files.max(plan.files.len());
                }
            }
            cost.add(many(&f, &server, &mut rng, 80, 3000, strides));
        }
        println!("corner: {cost:?}");
        assert_eq!(most_files, 4);
    }

    #[test]
    fn windows_with_an_absent_file_read_no_data_there() {
        let (_dir, f) = field(29.99, 86.99, 10.0, 556.0);
        // The file north-east of the corner is open sea.
        let server = Server::new(&[(31.0, 88.0)], false);
        let mut rng = Rng(5);
        let mut cost = Cost::default();
        let mut absent_seen = false;
        for stride in [8, 64, 256] {
            let w = Window {
                x0: -19 * stride,
                z0: -19 * stride,
                width: 38,
                depth: 38,
                stride,
                metres: 100.0,
            };
            let (c, plan) = check(&f, &server, w);
            cost.add(c);
            absent_seen |= plan.files.iter().any(|f| f.absent);
            let nodata = plan.evaluate().iter().filter(|&&c| c == 0).count();
            assert!(nodata > 0);
        }
        assert!(absent_seen);
        cost.add(many(&f, &server, &mut rng, 60, 3000, &STRIDES));
        println!("absent: {cost:?}");
    }

    #[test]
    fn windows_across_the_date_line() {
        let mut cost = Cost::default();
        for (lat0, lon0) in [(0.5, 179.99), (0.5, -179.99), (-0.001, 180.0), (45.2, -180.0)] {
            let (_dir, f) = field(lat0, lon0, 10.0, 556.0);
            let server = Server::new(&[], false);
            let mut rng = Rng(11);
            for &(width, depth) in &SIZES {
                for &stride in &[1, 16, 128] {
                    let w = Window {
                        x0: -(width as i32 * stride) / 2,
                        z0: -(depth as i32 * stride) / 2,
                        width,
                        depth,
                        stride,
                        metres: 200.0,
                    };
                    let (c, plan) = check(&f, &server, w);
                    cost.add(c);
                    let west: HashSet<i16> = plan.files.iter().map(|f| f.west).collect();
                    if width == 38 && stride == 128 && lat0 == 0.5 {
                        assert!(west.contains(&177) && west.contains(&-180), "{west:?}");
                    }
                }
            }
            cost.add(many(&f, &server, &mut rng, 60, 6000, &STRIDES));
        }
        println!("date line: {cost:?}");
    }

    #[test]
    fn windows_beyond_a_pole_read_no_data_and_near_it_are_files_of_the_top_row() {
        let mut cost = Cost::default();
        let mut some_no_file = false;
        for (lat0, lon0, z0) in [(85.0, 10.0, -6000), (-85.0, 10.0, 3600)] {
            let (_dir, f) = field(lat0, lon0, 100.0, 556.0);
            let server = Server::new(&[], false);
            for stride in [16, 64] {
                let w = Window {
                    x0: -1200,
                    z0,
                    width: 38,
                    depth: 38,
                    stride,
                    metres: 500.0,
                };
                let (c, plan) = check(&f, &server, w);
                cost.add(c);
                some_no_file |= plan.evaluate().contains(&0);
            }
            let mut rng = Rng(3);
            cost.add(many(&f, &server, &mut rng, 30, 8000, &[1, 8, 16, 64]));
        }
        println!("poles: {cost:?}");
        assert!(some_no_file);
    }

    #[test]
    fn a_tile_the_file_leaves_out_reads_no_data_in_the_plan_too() {
        let (_dir, f) = field(28.5, 85.5, 10.0, 556.0);
        let server = Server::new(&[], true);
        let mut rng = Rng(17);
        let mut cost = Cost::default();
        let mut blank = false;
        // Tile 4 of the finest level is column 1, row 1: pixels 256..512 both ways, near the middle.
        for stride in [1, 8, 40] {
            let w = Window {
                x0: -19 * stride,
                z0: -19 * stride,
                width: 38,
                depth: 38,
                stride,
                metres: 100.0,
            };
            let (c, plan) = check(&f, &server, w);
            cost.add(c);
            blank |= plan.tiles.iter().any(|t| t.tile.is_none());
        }
        cost.add(many(&f, &server, &mut rng, 80, 4000, &STRIDES));
        println!("blank tile: {cost:?}");
        assert!(blank);
    }

    #[test]
    fn a_plan_asks_for_the_index_then_the_tiles_and_gives_an_empty_one_for_nothing() {
        let (_dir, f) = field(28.5, 85.5, 10.0, 556.0);
        let server = Server::new(&[], false);
        let w = Window {
            x0: -20,
            z0: -20,
            width: 40,
            depth: 40,
            stride: 1,
            metres: 100.0,
        };
        let args = (w.x0, w.z0, w.width, w.depth, w.stride, w.metres);
        let Err(first) = f.class_window_plan(args.0, args.1, args.2, args.3, args.4, args.5) else {
            panic!("nothing is held")
        };
        assert_eq!(first.wanted, Wanted::Index);
        server.serve(f.store(), &first);
        let mut tiles = 0;
        loop {
            match f.class_window_plan(args.0, args.1, args.2, args.3, args.4, args.5) {
                Err(fetch) if matches!(fetch.wanted, Wanted::Index) => server.serve(f.store(), &fetch),
                Err(fetch) => {
                    assert!(matches!(fetch.wanted, Wanted::Tile { level: 0, .. }));
                    // Not held: asking again gives the same missing tile.
                    assert_eq!(
                        f.class_window_plan(args.0, args.1, args.2, args.3, args.4, args.5)
                            .err(),
                        Some(fetch)
                    );
                    tiles += 1;
                    server.serve(f.store(), &fetch);
                }
                Ok(plan) => {
                    assert!(!plan.tiles.is_empty());
                    break;
                }
            }
        }
        assert!(tiles >= 1);
        // A window of nothing needs nothing.
        let (_dir2, g) = field(28.5, 85.5, 10.0, 556.0);
        for (width, depth) in [(0, 5), (5, 0), (0, 0)] {
            let plan = g.class_window_plan(0, 0, width, depth, 1, 10.0).unwrap();
            assert!(plan.files.is_empty() && plan.tiles.is_empty());
            assert!(plan.evaluate().is_empty());
        }
        // The first fetch of a plan is one a lookup would also make.
        let want = g.class_at(0.5, 0.5, 100.0);
        assert!(matches!(want, Lookup::Need(_)));
    }
}

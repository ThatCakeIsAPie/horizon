//! The window fast path of [`super::LandCoverField::class_grid`]: bit for bit what
//! [`super::LandCoverField::class_grid_reference`] gives, with the per-window state resolved once.
//!
//! The reference makes one [`LandCoverField::class_at`] per sample: a warp of the point (four value noises),
//! the projection, then [`LandCoverStore::lookup_smooth`], which takes the store's lock a few times per sample
//! (is the file absent, its index, the tile). Nothing is fetched during a window, so what the store answered
//! for a file or a tile at the first sample it is the answer at every later one, and the window holds it:
//!
//! * the warp's lattice position, fade and hashes depend on the column alone (the `u` of the noise) or on the
//!   row alone (the `v`), so they are worked out once per column and per row, and the lattice values go in a
//!   table over the cells the window touches when the table is not bigger than the samples;
//! * the file (absent or not, index, level, speck salt) and the decoded tiles are asked for once per window.
//!
//! Every expression is the reference's, in the reference's order (Rust does not fuse a multiply and an add), and
//! the first missing byte range ends the window with the same [`Fetch`] the reference gives, because the
//! samples are read in the same order and each sample asks the store its questions in the same order.

use super::*;

// Timing helpers for the breakdown of a window's cost (examples/earth_class_profile in horizon-world).
#[path = "landfield_profile.rs"]
mod profile;

#[cfg(test)]
#[path = "landfield_window_tests.rs"]
mod tests;

/// Past these a window is not one the fast path prepares columns for: the reference is used (a window of this
/// size is never asked for, and the reference stops at its first missing tile without preparing anything).
const MAX_AXIS: usize = 1 << 20;
const MAX_SAMPLES: usize = 1 << 26;

const LATTICE_X: u64 = 0x9E37_79B9_7F4A_7C15;
const LATTICE_Z: u64 = 0xD1B5_4A32_D192_ED03;

/// The smoothstep of the noise, as `warp::value_noise` writes it.
fn fade(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// `warp::lattice` from the two halves of the hash (the column's and the row's).
fn lattice(salt: u64, hx: u64, hz: u64) -> f64 {
    let h = crate::warp::mix64(salt ^ hx ^ hz);
    (h >> 11) as f64 / (1_u64 << 53) as f64
}

/// The blend of `warp::value_noise` from its four corners and the faded fractions.
fn blend(a: f64, b: f64, c: f64, d: f64, u: f64, v: f64) -> f64 {
    let top = a + (b - a) * u;
    let bottom = c + (d - c) * u;
    2.0 * (top + (bottom - top) * v) - 1.0
}

/// Where a column (or row) is in the noise's lattice along its axis.
#[derive(Clone, Copy)]
struct Cell {
    /// The lattice index below the point.
    i: i64,
    /// The faded fraction of the point past it.
    t: f64,
    /// The axis's half of the lattice hash at `i` and at `i + 1`.
    h0: u64,
    h1: u64,
}

impl Cell {
    fn new(coord: f64, half: impl Fn(i64) -> u64) -> Self {
        let f = coord.floor();
        let t = coord - f;
        let i = f as i64;
        Self {
            i,
            t: fade(t),
            h0: half(i),
            h1: half(i.wrapping_add(1)),
        }
    }
}

fn column_half(i: i64) -> u64 {
    (i as u64).wrapping_mul(LATTICE_X)
}

fn row_half(i: i64) -> u64 {
    crate::warp::mix64((i as u64).wrapping_add(LATTICE_Z))
}

/// The lattice values of one salt over the cells a window touches.
struct Table {
    across: usize,
    values: Vec<f64>,
}

/// The tables of an octave's two noises, and per column and per row where its cell is in them.
struct Tables {
    noises: [Table; 2],
    at_col: Vec<usize>,
    at_row: Vec<usize>,
}

/// One octave of the warp: both directions' noises over the window's columns and rows.
struct Octave {
    /// The offsets' scale in blocks, `amplitude * pixel` (east-west, north-south).
    scale: (f64, f64),
    salts: [u64; 2],
    cols: Vec<Cell>,
    rows: Vec<Cell>,
    tables: Option<Tables>,
}

impl Octave {
    fn new(
        (wavelength, amplitude): (f64, f64),
        salts: [u64; 2],
        (px, pz): (f64, f64),
        xs: &[f64],
        zs: &[f64],
    ) -> Self {
        let cols: Vec<Cell> = xs
            .iter()
            .map(|&x| Cell::new(x / (wavelength * px), column_half))
            .collect();
        let rows: Vec<Cell> = zs
            .iter()
            .map(|&z| Cell::new(z / (wavelength * pz), row_half))
            .collect();
        let tables = Self::tables(&cols, &rows, salts);
        Self {
            scale: (amplitude * px, amplitude * pz),
            salts,
            cols,
            rows,
            tables,
        }
    }

    /// Tables of the lattice values when the cells the window spans are not many more than its samples, else
    /// none (each sample then hashes its own corners).
    fn tables(cols: &[Cell], rows: &[Cell], salts: [u64; 2]) -> Option<Tables> {
        let span = |cells: &[Cell]| {
            let lo = cells.iter().map(|c| i128::from(c.i)).min()?;
            let hi = cells.iter().map(|c| i128::from(c.i)).max()?;
            Some((lo, hi - lo + 2))
        };
        let ((x_lo, across), (z_lo, down)) = (span(cols)?, span(rows)?);
        let samples = cols.len() as i128 * rows.len() as i128;
        if across * down > 2 * samples + 64 {
            return None;
        }
        let (across, down) = (across as usize, down as usize);
        let halves_x: Vec<u64> = (0..across)
            .map(|k| column_half((x_lo + k as i128) as i64))
            .collect();
        let build = |salt: u64| {
            let mut values = Vec::with_capacity(across * down);
            for r in 0..down {
                let hz = row_half((z_lo + r as i128) as i64);
                values.extend(halves_x.iter().map(|&hx| lattice(salt, hx, hz)));
            }
            Table { across, values }
        };
        Some(Tables {
            noises: [build(salts[0]), build(salts[1])],
            at_col: cols.iter().map(|c| (i128::from(c.i) - x_lo) as usize).collect(),
            at_row: rows.iter().map(|c| (i128::from(c.i) - z_lo) as usize).collect(),
        })
    }

    /// The offset `(east-west, north-south)` of the warp at column `i`, row `j`.
    fn offset(&self, i: usize, j: usize) -> (f64, f64) {
        let (col, row) = (&self.cols[i], &self.rows[j]);
        let noise = |k: usize| match &self.tables {
            Some(tables) => {
                let table = &tables.noises[k];
                let at = tables.at_row[j] * table.across + tables.at_col[i];
                let v = &table.values;
                blend(
                    v[at],
                    v[at + 1],
                    v[at + table.across],
                    v[at + table.across + 1],
                    col.t,
                    row.t,
                )
            }
            None => {
                let salt = self.salts[k];
                blend(
                    lattice(salt, col.h0, row.h0),
                    lattice(salt, col.h1, row.h0),
                    lattice(salt, col.h0, row.h1),
                    lattice(salt, col.h1, row.h1),
                    col.t,
                    row.t,
                )
            }
        };
        (self.scale.0 * noise(0), self.scale.1 * noise(1))
    }
}

/// What the store said about a file, for one window.
struct FileSlot {
    file: WorldCoverFile,
    /// `None`: the file is absent (every read there is no data).
    ready: Option<Ready>,
}

struct Ready {
    index: Arc<TiffIndex>,
    level_no: usize,
    /// Which level of which file, for the hash that keeps a speck in part.
    salt: u64,
}

/// A tile the store gave (or `None`: blank, or left out by the file), for one window.
struct TileSlot {
    file: usize,
    no: usize,
    tile: Option<Arc<LandTile>>,
}

/// [`LandCoverStore::lookup_smooth`] for the samples of one window at one `metres`: the same reads in the same
/// order, but each file and each tile is asked of the store once.
struct Reader<'a> {
    store: &'a LandCoverStore,
    metres: f64,
    files: Vec<FileSlot>,
    last_file: usize,
    tiles: Vec<TileSlot>,
    last_tile: usize,
}

/// Where a tile is: the file's slot, the file, the level and the level's number.
struct Place<'l> {
    slot: usize,
    file: WorldCoverFile,
    level_no: usize,
    level: &'l Level,
}

impl<'a> Reader<'a> {
    fn new(store: &'a LandCoverStore, metres: f64) -> Self {
        Self {
            store,
            metres,
            files: Vec::new(),
            last_file: 0,
            tiles: Vec::new(),
            last_tile: 0,
        }
    }

    /// The slot of `file`, asking the store the first time.
    fn file_slot(&mut self, file: WorldCoverFile) -> Result<usize, Fetch> {
        if let Some(slot) = self.files.get(self.last_file)
            && slot.file == file
        {
            return Ok(self.last_file);
        }
        if let Some(at) = self.files.iter().position(|s| s.file == file) {
            self.last_file = at;
            return Ok(at);
        }
        let ready = if self.store.is_absent(file) {
            None
        } else {
            let index = self.store.index_of(file)?;
            let level_no = level_for(&index.levels, self.metres);
            let salt = ((u64::from(file.south as u16) << 32) | (u64::from(file.west as u16) << 16))
                ^ (level_no as u64);
            Some(Ready {
                index,
                level_no,
                salt,
            })
        };
        self.files.push(FileSlot { file, ready });
        self.last_file = self.files.len() - 1;
        Ok(self.last_file)
    }

    /// The slot of tile `no` of the file at `at`, asking the store the first time.
    fn tile_slot(
        store: &LandCoverStore,
        tiles: &mut Vec<TileSlot>,
        last: &mut usize,
        at: &Place<'_>,
        no: usize,
    ) -> Result<usize, Fetch> {
        if let Some(slot) = tiles.get(*last)
            && slot.file == at.slot
            && slot.no == no
        {
            return Ok(*last);
        }
        if let Some(found) = tiles
            .iter()
            .position(|s| s.file == at.slot && s.no == no)
        {
            *last = found;
            return Ok(found);
        }
        let tile = store.tile_of(at.file, at.level_no, at.level, no)?;
        tiles.push(TileSlot {
            file: at.slot,
            no,
            tile,
        });
        *last = tiles.len() - 1;
        Ok(*last)
    }

    /// What [`LandCoverStore::lookup_smooth`] gives at `(lat, lon)`, with [`Lookup::NoFile`] as no data and
    /// [`Lookup::Need`] as the `Err`.
    fn read(&mut self, lat: f64, lon: f64) -> Result<LandClass, Fetch> {
        let no_data = LandClass::NoData;
        let Some(file) = WorldCoverFile::containing(lat, lon) else {
            return Ok(no_data);
        };
        let fi = self.file_slot(file)?;
        let Some(ready) = &self.files[fi].ready else {
            return Ok(no_data);
        };
        let level_no = ready.level_no;
        let Some(level) = ready.index.levels.get(level_no) else {
            return Ok(no_data);
        };
        let (px, py) = file.pixel(lat, lon, level.width, level.height);
        let inside = file.pixel_fraction(lat, lon, level.width, level.height);
        let (col, row) = (px / level.tile_width, py / level.tile_height);
        let Some(tile_no) = level.tile_index(col, row) else {
            return Ok(no_data);
        };
        let salt = ready.salt;
        let store = self.store;
        let at = Place {
            slot: fi,
            file,
            level_no,
            level,
        };
        let own_slot = Self::tile_slot(store, &mut self.tiles, &mut self.last_tile, &at, tile_no)?;
        let (lx, ly) = (px % level.tile_width, py % level.tile_height);
        let place = (i64::from(px), i64::from(py), salt);
        match &self.tiles[own_slot].tile {
            None => return Ok(no_data),
            Some(tile) => {
                if let Some(kept) = tile.speck_kept(lx, ly, inside, place) {
                    return Ok(kept);
                }
            }
        }
        // The four pixels whose centres are nearest the point, with the bilinear weights of the point among
        // them: the own pixel, the one beside it, the one above or below, and the diagonal.
        let (u, v) = (inside.0 - 0.5, inside.1 - 0.5);
        let (step_x, tx) = (if u >= 0.0 { 1_i64 } else { -1 }, u.abs());
        let (step_y, ty) = (if v >= 0.0 { 1_i64 } else { -1 }, v.abs());
        let (own, here) = ((i64::from(px), i64::from(py)), (col, row));
        let (tiles, last_tile) = (&mut self.tiles, &mut self.last_tile);
        let mut code_at = |x: i64, y: i64| -> Result<u8, Fetch> {
            let x = x.clamp(0, i64::from(level.width) - 1) as u32;
            let y = y.clamp(0, i64::from(level.height) - 1) as u32;
            let there = (x / level.tile_width, y / level.tile_height);
            let (lx, ly) = (x % level.tile_width, y % level.tile_height);
            let slot = if there == here {
                own_slot
            } else {
                let Some(no) = level.tile_index(there.0, there.1) else {
                    return Ok(0);
                };
                Self::tile_slot(store, tiles, last_tile, &at, no)?
            };
            Ok(match &tiles[slot].tile {
                Some(tile) => tile.code_at(lx, ly).unwrap_or(0),
                None => 0,
            })
        };
        let samples = [
            (own.0, own.1, (1.0 - tx) * (1.0 - ty)),
            (own.0 + step_x, own.1, tx * (1.0 - ty)),
            (own.0, own.1 + step_y, (1.0 - tx) * ty),
            (own.0 + step_x, own.1 + step_y, tx * ty),
        ];
        // The own pixel first, so on a tie it wins.
        let mut votes: [(u8, f64); 4] = [(0, 0.0); 4];
        let mut used = 0;
        for (x, y, weight) in samples {
            let code = code_at(x, y)?;
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
        Ok(LandClass::from_code(best.0).unwrap_or(LandClass::NoData))
    }
}

impl LandCoverField {
    /// [`LandCoverField::class_grid`]: what [`LandCoverField::class_grid_reference`] gives, bit for bit (the
    /// same classes, or the same first missing [`Fetch`]), with the warp's per-column and per-row work done
    /// once and the store asked once per file and tile of the window instead of per sample.
    pub(super) fn class_grid_window(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        stride: i32,
        metres_per_sample: f64,
    ) -> Result<Vec<LandClass>, Fetch> {
        let samples = width.saturating_mul(depth);
        if width > MAX_AXIS || depth > MAX_AXIS || samples > MAX_SAMPLES {
            return self.class_grid_reference(x0, z0, width, depth, stride, metres_per_sample);
        }
        let mut out = Vec::with_capacity(samples.min(1 << 20));
        if samples == 0 {
            return Ok(out);
        }
        let pixel = self.pixel_blocks(metres_per_sample);
        let xs: Vec<f64> = (0..width)
            .map(|i| f64::from(x0) + (i as f64) * f64::from(stride) + 0.5)
            .collect();
        let zs: Vec<f64> = (0..depth)
            .map(|j| f64::from(z0) + (j as f64) * f64::from(stride) + 0.5)
            .collect();
        let coarse = Octave::new(
            (LAND_WARP_COARSE_WAVELENGTH_PX, LAND_WARP_COARSE_PX),
            [WARP_SALTS[0], WARP_SALTS[1]],
            pixel,
            &xs,
            &zs,
        );
        let fine = Octave::new(
            (LAND_WARP_FINE_WAVELENGTH_PX, LAND_WARP_FINE_PX),
            [WARP_SALTS[2], WARP_SALTS[3]],
            pixel,
            &xs,
            &zs,
        );
        let mut reader = Reader::new(&self.store, metres_per_sample);
        for (j, &z) in zs.iter().enumerate() {
            for (i, &x) in xs.iter().enumerate() {
                let (cx, cz) = coarse.offset(i, j);
                let (fx, fz) = fine.offset(i, j);
                let at = self.projection.to_lat_lon_flat(x + cx + fx, z + cz + fz);
                out.push(reader.read(at.lat, at.lon)?);
            }
        }
        Ok(out)
    }
}

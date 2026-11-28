//! Climate over the globe and over world blocks, read from the two CHELSA files the caller fetches in pieces.
//!
//! This follows [`crate::landfield`] and has the same contract: the store never fetches. A lookup that lacks data
//! answers with the [`ClimateFetch`] to make (which variable, which byte range), the caller hands the bytes back
//! with [`ClimateStore::provide`], and nothing is ever guessed. The differences are that there are only two files
//! (both cover the whole world, land and sea), their samples are 16-bit, and a value is the bilinear mix of the
//! four nearest pixel centres rather than one pixel, so a lookup may need up to four tiles of each file.
//!
//! What is held is bounded. Decoded tiles (512 KiB each at CHELSA's tiling) live in a memory cache of a fixed size
//! that drops the least recently used; the index of a file is about 60 KiB. Everything provided is also written under
//! a folder named by [`cache_key`], so a new rules version starts with an empty cache, and it is found again without
//! a fetch by the next run. The store is safe to share between threads.
//!
//! The store expects the full-resolution level to be the size of its [`ClimateGrid`] (CHELSA's 43200 by 20880 for
//! [`ClimateStore::new`]) and 16-bit samples; a file that is anything else is refused. Tests can use
//! [`ClimateStore::with_grid`] with a small synthetic file: the georeference is the same extent of 360 by 174
//! degrees whatever the size.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::climate::{
    CLIMATE_PREFIX_LEN, Climate, ClimateVar, EXTENT_LAT, EXTENT_LON, NORTH, SOUTH, WEST, cache_key,
    level_for, on_grid, pixel_position,
};
use crate::landfield::{GeoBox, MAX_INDEX_BYTES, ProvideError, Wanted};
use crate::projection::Projection;
use crate::source::write_atomically;
use crate::tiff::{ByteRange, Level, Parse, Piece, TiffError, TiffIndex, parse_index};

/// The size of the full-resolution level a store accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClimateGrid {
    pub width: u32,
    pub height: u32,
}

impl ClimateGrid {
    /// CHELSA V2.1's global grid: 30 arc-seconds.
    pub const CHELSA: ClimateGrid = ClimateGrid {
        width: 43200,
        height: 20880,
    };
}

/// A byte range of one of the files that a lookup needs and the store does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClimateFetch {
    pub var: ClimateVar,
    pub range: ByteRange,
    pub wanted: Wanted,
}

/// The answer to a climate question.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClimateLookup {
    /// The climate at the place.
    Climate(Climate),
    /// Not here yet: fetch this, give it to [`ClimateStore::provide`], and ask again.
    Need(ClimateFetch),
    /// There is none: the pixels are "no data", the tile is one the file leaves out, a file is absent at the
    /// source or refused, or the place is off the grid (north of about 84 N, south of about 90 S).
    NoData,
}

/// One decoded tile: raw 16-bit values, row by row.
#[derive(Debug, PartialEq, Eq)]
pub struct ClimateTile {
    pub width: u32,
    pub height: u32,
    values: Vec<u16>,
}

impl ClimateTile {
    /// The raw value at pixel `(x, y)` of the tile; `None` outside it.
    pub fn raw(&self, x: u32, y: u32) -> Option<u16> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.values
            .get(y as usize * self.width as usize + x as usize)
            .copied()
    }
}

/// Index bytes held for one file.
#[derive(Default)]
struct FileState {
    /// Pieces of the file from the start, as fetched: `(offset, bytes)`.
    pieces: Vec<(u64, Vec<u8>)>,
    /// The size of the file, once a fetch came back short.
    file_len: Option<u64>,
    /// The parsed index, once the pieces hold all of it.
    index: Option<Arc<TiffIndex>>,
    /// Whether the pieces were looked for on disk.
    from_disk: bool,
}

impl FileState {
    fn parse(&self, grid: ClimateGrid) -> Result<Parse, TiffError> {
        let pieces: Vec<Piece<'_>> = self
            .pieces
            .iter()
            .map(|(offset, bytes)| Piece {
                offset: *offset,
                bytes,
            })
            .collect();
        match parse_index(&pieces, self.file_len)? {
            Parse::Done(index) => {
                check_climate(&index, grid)?;
                Ok(Parse::Done(index))
            }
            need => Ok(need),
        }
    }

    fn held_bytes(&self) -> usize {
        self.pieces.iter().map(|(_, b)| b.len()).sum()
    }
}

/// A climate file is 16-bit samples and its full resolution is the grid.
fn check_climate(index: &TiffIndex, grid: ClimateGrid) -> Result<(), TiffError> {
    let Some(first) = index.levels.first() else {
        return Err(TiffError::Malformed("a file with no image"));
    };
    if (first.width, first.height) != (grid.width, grid.height) {
        return Err(TiffError::Unsupported(
            "a climate file whose full resolution is not the grid the store expects",
        ));
    }
    if index.levels.iter().any(|level| level.bits_per_sample != 16) {
        return Err(TiffError::Unsupported(
            "climate with anything but 16 bits per sample",
        ));
    }
    Ok(())
}

type TileKey = (ClimateVar, u8, u32);

struct Memory {
    files: HashMap<ClimateVar, FileState>,
    tiles: crate::fastmap::FastMap<TileKey, (Arc<ClimateTile>, u64)>,
    absent: HashSet<ClimateVar>,
    /// Tiles the source will not give this session: read as "no data", never stored.
    blank: HashSet<TileKey>,
    clock: u64,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// How a value is read from the pixels.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Read {
    Bilinear,
    Nearest,
}

/// The climate tiles of the globe: a bounded memory cache of decoded tiles over a disk cache of what the caller
/// fetched, with no fetcher of its own.
pub struct ClimateStore {
    dir: PathBuf,
    capacity: usize,
    grid: ClimateGrid,
    memory: Mutex<Memory>,
}

impl ClimateStore {
    /// A store of the real CHELSA grid that keeps files under `dir` (in a folder named by [`cache_key`]) and at
    /// most `memory_tiles` decoded tiles (each 512 KiB at CHELSA's tiling).
    pub fn new(dir: impl Into<PathBuf>, memory_tiles: usize) -> Self {
        Self::with_grid(dir, memory_tiles, ClimateGrid::CHELSA)
    }

    /// Like [`ClimateStore::new`] for files whose full resolution is `grid` (small synthetic ones in tests).
    pub fn with_grid(dir: impl Into<PathBuf>, memory_tiles: usize, grid: ClimateGrid) -> Self {
        Self {
            dir: dir.into().join(cache_key()),
            capacity: memory_tiles,
            grid,
            memory: Mutex::new(Memory {
                files: HashMap::new(),
                tiles: Default::default(),
                absent: HashSet::new(),
                blank: HashSet::new(),
                clock: 0,
            }),
        }
    }

    /// The grid this store accepts.
    pub fn grid(&self) -> ClimateGrid {
        self.grid
    }

    /// The folder holding what is kept of `var`.
    pub fn var_dir(&self, var: ClimateVar) -> PathBuf {
        self.dir.join(var.name())
    }

    /// Where the compressed bytes of a tile are kept on disk.
    pub fn tile_path(&self, var: ClimateVar, level: u8, tile: u32) -> PathBuf {
        self.var_dir(var).join(format!("tile-{level}-{tile}.bin"))
    }

    fn index_path(&self, var: ClimateVar, offset: u64) -> PathBuf {
        self.var_dir(var).join(format!("index-{offset}.bin"))
    }

    fn length_path(&self, var: ClimateVar, len: u64) -> PathBuf {
        self.var_dir(var).join(format!("length-{len}"))
    }

    fn absent_path(&self, var: ClimateVar) -> PathBuf {
        self.var_dir(var).join("absent")
    }

    /// Records that the source has no such file (a 404), so lookups answer [`ClimateLookup::NoData`]. It is kept
    /// on disk too, so the next run does not ask again.
    pub fn mark_absent(&self, var: ClimateVar) {
        lock(&self.memory).absent.insert(var);
        let _ = write_atomically(&self.absent_path(var), b"");
    }

    /// Records that the source will not give what `fetch` names (downloads are not approved, the server refuses):
    /// for the rest of this run the file (for an index piece) or the tile reads as "no data", so nothing waits for
    /// it. Nothing is written to disk, so another run asks again.
    pub fn mark_unavailable(&self, fetch: &ClimateFetch) {
        let mut memory = lock(&self.memory);
        match fetch.wanted {
            Wanted::Index => {
                memory.absent.insert(fetch.var);
            }
            Wanted::Tile { level, tile } => {
                memory.blank.insert((fetch.var, level, tile));
            }
        }
    }

    /// Whether `fetch` is still something the store lacks: a stale request (the bytes arrived meanwhile, the file
    /// turned out absent, a later piece of the index is the next one wanted) is not.
    pub fn is_missing(&self, fetch: &ClimateFetch) -> bool {
        if self.is_absent(fetch.var) {
            return false;
        }
        match self.index_of(fetch.var) {
            Err(next) => fetch.wanted == Wanted::Index && next.range == fetch.range,
            Ok(index) => {
                let Wanted::Tile { level, tile } = fetch.wanted else {
                    return false;
                };
                let Some(l) = index.levels.get(usize::from(level)) else {
                    return false;
                };
                self.tile_of_local(fetch.var, usize::from(level), l, tile as usize)
                    .is_err()
            }
        }
    }

    /// Whether `var` is known to be absent at the source, from memory or from the marker on disk.
    fn is_absent(&self, var: ClimateVar) -> bool {
        self.load_pieces_from_disk(var);
        lock(&self.memory).absent.contains(&var)
    }

    /// The climate at `(lat, lon)`: the bilinear mix of the four nearest pixel centres of each variable, read at
    /// the overview level whose pixels are no taller on the ground than `metres_per_sample` (the finest level
    /// when even that is coarser). Columns wrap at the date line, rows stop at the grid's top and bottom edge,
    /// pixels with no data are left out and the others' weights scaled up. Never fetches: it answers
    /// [`ClimateLookup::Need`] when it lacks bytes, the index of both files before any tile.
    pub fn lookup(&self, lat: f64, lon: f64, metres_per_sample: f64) -> ClimateLookup {
        self.lookup_with(lat, lon, metres_per_sample, Read::Bilinear)
    }

    /// Like [`ClimateStore::lookup`] but the value of the one pixel the place is in, not a mix (for checking
    /// against values read by hand).
    pub fn lookup_nearest(&self, lat: f64, lon: f64, metres_per_sample: f64) -> ClimateLookup {
        self.lookup_with(lat, lon, metres_per_sample, Read::Nearest)
    }

    fn lookup_with(&self, lat: f64, lon: f64, metres_per_sample: f64, read: Read) -> ClimateLookup {
        if !on_grid(lat, lon) || ClimateVar::ALL.iter().any(|v| self.is_absent(*v)) {
            return ClimateLookup::NoData;
        }
        let mut indexes = Vec::with_capacity(2);
        for var in ClimateVar::ALL {
            match self.index_of(var) {
                Err(fetch) => return ClimateLookup::Need(fetch),
                Ok(index) => indexes.push(index),
            }
        }
        let mut values = [0.0f64; 2];
        for (k, var) in ClimateVar::ALL.into_iter().enumerate() {
            let level_no = level_for(&indexes[k].levels, metres_per_sample);
            match self.sample(var, &indexes[k], level_no, lat, lon, read) {
                Err(fetch) => return ClimateLookup::Need(fetch),
                Ok(None) => return ClimateLookup::NoData,
                Ok(Some(value)) => values[k] = value,
            }
        }
        ClimateLookup::Climate(Climate {
            temperature_c: values[0] as f32,
            precipitation_mm: values[1] as f32,
        })
    }

    /// The value of one variable at a place, or the fetch for a tile it lacks. `Ok(None)` is no data.
    fn sample(
        &self,
        var: ClimateVar,
        index: &TiffIndex,
        level_no: usize,
        lat: f64,
        lon: f64,
        read: Read,
    ) -> Result<Option<f64>, ClimateFetch> {
        let Some(level) = index.levels.get(level_no) else {
            return Ok(None);
        };
        let Some((x, y)) = pixel_position(lat, lon, level.width, level.height) else {
            return Ok(None);
        };
        if read == Read::Nearest {
            let raw = self.raw_at(var, level_no, level, x.floor() as u32, y.floor() as u32)?;
            return Ok(raw.and_then(|raw| var.value_f64(raw)));
        }
        // Pixel centres are at whole numbers plus a half.
        let (fx, fy) = (x - 0.5, y - 0.5);
        let (cx, cy) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - cx, fy - cy);
        let (w, h) = (i64::from(level.width), i64::from(level.height));
        let cols = [cx as i64, cx as i64 + 1].map(|c| c.rem_euclid(w) as u32);
        let rows = [cy as i64, cy as i64 + 1].map(|r| r.clamp(0, h - 1) as u32);
        let (mut sum, mut total) = (0.0, 0.0);
        for (j, &row) in rows.iter().enumerate() {
            for (i, &col) in cols.iter().enumerate() {
                let weight =
                    (if i == 0 { 1.0 - tx } else { tx }) * (if j == 0 { 1.0 - ty } else { ty });
                if weight <= 0.0 {
                    continue;
                }
                if let Some(value) = self
                    .raw_at(var, level_no, level, col, row)?
                    .and_then(|raw| var.value_f64(raw))
                {
                    sum += weight * value;
                    total += weight;
                }
            }
        }
        Ok((total > 0.0).then(|| sum / total))
    }

    /// The raw value of pixel `(px, py)` of a level; `Ok(None)` where the file has no tile.
    fn raw_at(
        &self,
        var: ClimateVar,
        level_no: usize,
        level: &Level,
        px: u32,
        py: u32,
    ) -> Result<Option<u16>, ClimateFetch> {
        let (col, row) = (px / level.tile_width, py / level.tile_height);
        let Some(tile_no) = level.tile_index(col, row) else {
            return Ok(None);
        };
        Ok(self
            .tile_of(var, level_no, level, tile_no)?
            .and_then(|tile| tile.raw(px % level.tile_width, py % level.tile_height)))
    }

    /// The fetches still missing for lookups anywhere in `area` at `metres_per_sample`: the index pieces of files
    /// whose index is incomplete, and for files whose index is held, every tile the area touches (with the one
    /// pixel around it that a bilinear read reaches into). Asking again after providing gives the next round.
    /// Files marked absent are left out.
    pub fn needs(&self, area: GeoBox, metres_per_sample: f64) -> Vec<ClimateFetch> {
        let mut out: Vec<ClimateFetch> = Vec::new();
        let mut seen: HashSet<ClimateFetch> = HashSet::new();
        if ![area.south, area.north, area.west, area.east]
            .iter()
            .all(|v| v.is_finite())
        {
            return out;
        }
        let (south, north) = (area.south.min(area.north), area.south.max(area.north));
        if north < SOUTH || south > NORTH {
            return out;
        }
        let (south, north) = (south.max(SOUTH), north.min(NORTH));
        let (west, mut east) = (area.west, area.east);
        if east < west {
            east += 360.0 * ((west - east) / 360.0).ceil();
        }
        let vars: Vec<ClimateVar> = ClimateVar::ALL
            .into_iter()
            .filter(|v| !self.is_absent(*v))
            .collect();
        let mut held = Vec::new();
        for var in vars {
            match self.index_of(var) {
                Err(fetch) => {
                    if seen.insert(fetch) {
                        out.push(fetch);
                    }
                }
                Ok(index) => held.push((var, index)),
            }
        }
        for (var, index) in held {
            let level_no = level_for(&index.levels, metres_per_sample);
            let Some(level) = index.levels.get(level_no) else {
                continue;
            };
            let (w, h) = (f64::from(level.width), f64::from(level.height));
            // Pixel columns, unwrapped, one past each side for the bilinear neighbours.
            let lo = ((west - WEST) / EXTENT_LON * w - 0.5).floor() as i64;
            let hi = ((east - WEST) / EXTENT_LON * w - 0.5).floor() as i64 + 1;
            let cols = tile_columns(
                lo,
                hi - lo + 1,
                i64::from(level.width),
                i64::from(level.tile_width),
            );
            let top = (((NORTH - north) / EXTENT_LAT * h - 0.5).floor() as i64).max(0);
            let bottom = (((NORTH - south) / EXTENT_LAT * h - 0.5).floor() as i64 + 1)
                .min(i64::from(level.height) - 1);
            if bottom < top {
                continue;
            }
            let tile_h = i64::from(level.tile_height);
            for row in top / tile_h..=bottom / tile_h {
                for &col in &cols {
                    let Some(tile_no) = level.tile_index(col, row as u32) else {
                        continue;
                    };
                    if let Err(fetch) = self.tile_of_local(var, level_no, level, tile_no)
                        && seen.insert(fetch)
                    {
                        out.push(fetch);
                    }
                }
            }
        }
        out
    }

    /// Hands over the bytes of a [`ClimateFetch`]. For an index piece, any length up to the range's (shorter means
    /// the file ended there). For a tile, exactly the range's bytes, which must decode. Refused bytes (also a file
    /// that is not 16-bit or not the size of the store's grid) are never stored, here or on disk. A failure to
    /// write the disk cache does not fail the call: the data is still held in memory.
    pub fn provide(&self, fetch: &ClimateFetch, bytes: &[u8]) -> Result<(), ProvideError> {
        if bytes.is_empty() {
            return Err(ProvideError::Empty);
        }
        if bytes.len() as u64 > fetch.range.len {
            return Err(ProvideError::WrongLength);
        }
        match fetch.wanted {
            Wanted::Index => self.provide_index(fetch, bytes),
            Wanted::Tile { level, tile } => self.provide_tile(fetch, level, tile, bytes),
        }
    }

    fn provide_index(&self, fetch: &ClimateFetch, bytes: &[u8]) -> Result<(), ProvideError> {
        let var = fetch.var;
        let short = (bytes.len() as u64) < fetch.range.len;
        let end = fetch.range.offset.saturating_add(bytes.len() as u64);
        self.load_pieces_from_disk(var);
        {
            let mut memory = lock(&self.memory);
            let state = memory.files.entry(var).or_default();
            if state.held_bytes() + bytes.len() > MAX_INDEX_BYTES {
                return Err(ProvideError::TooLarge);
            }
            let backup = (state.pieces.clone(), state.file_len, state.index.take());
            state
                .pieces
                .retain(|(offset, _)| *offset != fetch.range.offset);
            state.pieces.push((fetch.range.offset, bytes.to_vec()));
            if short {
                state.file_len = Some(end);
            }
            match state.parse(self.grid) {
                Ok(Parse::Done(index)) => state.index = Some(Arc::new(index)),
                Ok(Parse::Need(_)) => {}
                Err(e) => {
                    // Refused bytes leave nothing behind.
                    (state.pieces, state.file_len, state.index) = backup;
                    return Err(ProvideError::Tiff(e));
                }
            }
        }
        let _ = write_atomically(&self.index_path(var, fetch.range.offset), bytes);
        if short {
            let _ = write_atomically(&self.length_path(var, end), b"");
        }
        Ok(())
    }

    fn provide_tile(
        &self,
        fetch: &ClimateFetch,
        level_no: u8,
        tile_no: u32,
        bytes: &[u8],
    ) -> Result<(), ProvideError> {
        let var = fetch.var;
        let Ok(index) = self.index_of(var) else {
            return Err(ProvideError::NoIndex);
        };
        let level = index
            .levels
            .get(usize::from(level_no))
            .ok_or(ProvideError::NotWanted)?;
        let range = match level.tile_range(tile_no as usize) {
            Ok(Some(range)) => range,
            _ => return Err(ProvideError::NotWanted),
        };
        if range != fetch.range {
            return Err(ProvideError::NotWanted);
        }
        if bytes.len() as u64 != range.len {
            return Err(ProvideError::WrongLength);
        }
        let values = level.decode_tile_u16(bytes).map_err(ProvideError::Tiff)?;
        let tile = Arc::new(ClimateTile {
            width: level.tile_width,
            height: level.tile_height,
            values,
        });
        let _ = write_atomically(&self.tile_path(var, level_no, tile_no), bytes);
        self.remember((var, level_no, tile_no), &tile);
        Ok(())
    }

    /// The parsed index of `var`, or the fetch for what it still lacks (the first one is
    /// [`CLIMATE_PREFIX_LEN`] bytes). Index bytes that no longer parse (a damaged cache file) are dropped, memory
    /// and disk, and asked for again.
    fn index_of(&self, var: ClimateVar) -> Result<Arc<TiffIndex>, ClimateFetch> {
        self.load_pieces_from_disk(var);
        let prefix = ClimateFetch {
            var,
            range: ByteRange {
                offset: 0,
                len: CLIMATE_PREFIX_LEN,
            },
            wanted: Wanted::Index,
        };
        let mut memory = lock(&self.memory);
        let state = memory.files.entry(var).or_default();
        if let Some(index) = &state.index {
            return Ok(Arc::clone(index));
        }
        if state.pieces.is_empty() {
            return Err(prefix);
        }
        match state.parse(self.grid) {
            Ok(Parse::Done(index)) => {
                let index = Arc::new(index);
                state.index = Some(Arc::clone(&index));
                Ok(index)
            }
            Ok(Parse::Need(range)) => Err(ClimateFetch { range, ..prefix }),
            Err(_) => {
                *state = FileState {
                    from_disk: true,
                    ..FileState::default()
                };
                drop(memory);
                let _ = fs::remove_dir_all(self.var_dir(var));
                Err(prefix)
            }
        }
    }

    /// Reads the index pieces the disk holds for `var`, once per run.
    fn load_pieces_from_disk(&self, var: ClimateVar) {
        {
            let mut memory = lock(&self.memory);
            let state = memory.files.entry(var).or_default();
            if state.from_disk {
                return;
            }
            state.from_disk = true;
        }
        if self.absent_path(var).exists() {
            lock(&self.memory).absent.insert(var);
        }
        let Ok(entries) = fs::read_dir(self.var_dir(var)) else {
            return;
        };
        let mut pieces = Vec::new();
        let mut file_len = None;
        let mut total = 0usize;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(rest) = name
                .strip_prefix("index-")
                .and_then(|n| n.strip_suffix(".bin"))
            {
                let Ok(offset) = rest.parse::<u64>() else {
                    continue;
                };
                if let Ok(bytes) = fs::read(entry.path()) {
                    total += bytes.len();
                    if total > MAX_INDEX_BYTES {
                        return;
                    }
                    pieces.push((offset, bytes));
                }
            } else if let Some(len) = name
                .strip_prefix("length-")
                .and_then(|n| n.parse::<u64>().ok())
            {
                file_len = Some(len);
            }
        }
        if pieces.is_empty() {
            return;
        }
        pieces.sort_by_key(|(offset, _)| *offset);
        let mut memory = lock(&self.memory);
        let state = memory.files.entry(var).or_default();
        if state.pieces.is_empty() {
            state.pieces = pieces;
            state.file_len = file_len;
        }
    }

    /// The decoded tile, from memory or disk, or the fetch for it. `Ok(None)` for a tile the file leaves out.
    fn tile_of(
        &self,
        var: ClimateVar,
        level_no: usize,
        level: &Level,
        tile_no: usize,
    ) -> Result<Option<Arc<ClimateTile>>, ClimateFetch> {
        let key = (var, level_no as u8, tile_no as u32);
        if let Some(tile) = self.in_memory(key) {
            return Ok(Some(tile));
        }
        if lock(&self.memory).blank.contains(&key) {
            return Ok(None);
        }
        let range = match level.tile_range(tile_no) {
            Ok(Some(range)) => range,
            Ok(None) => return Ok(None),
            Err(e) => {
                log::debug!("climate {var}: tile {tile_no} of level {level_no} is unusable: {e}");
                return Ok(None);
            }
        };
        if let Ok(bytes) = fs::read(self.tile_path(var, key.1, key.2))
            && bytes.len() as u64 == range.len
            && let Ok(values) = level.decode_tile_u16(&bytes)
        {
            let tile = Arc::new(ClimateTile {
                width: level.tile_width,
                height: level.tile_height,
                values,
            });
            self.remember(key, &tile);
            return Ok(Some(tile));
        }
        Err(ClimateFetch {
            var,
            range,
            wanted: Wanted::Tile {
                level: key.1,
                tile: key.2,
            },
        })
    }

    /// Like [`Self::tile_of`] but only says whether a fetch is missing, without decoding a tile that is on disk.
    fn tile_of_local(
        &self,
        var: ClimateVar,
        level_no: usize,
        level: &Level,
        tile_no: usize,
    ) -> Result<(), ClimateFetch> {
        let key = (var, level_no as u8, tile_no as u32);
        {
            let memory = lock(&self.memory);
            if memory.tiles.contains_key(&key) || memory.blank.contains(&key) {
                return Ok(());
            }
        }
        let range = match level.tile_range(tile_no) {
            Ok(Some(range)) => range,
            _ => return Ok(()),
        };
        let on_disk = fs::metadata(self.tile_path(var, key.1, key.2))
            .is_ok_and(|m| m.is_file() && m.len() == range.len);
        if on_disk {
            return Ok(());
        }
        Err(ClimateFetch {
            var,
            range,
            wanted: Wanted::Tile {
                level: key.1,
                tile: key.2,
            },
        })
    }

    fn in_memory(&self, key: TileKey) -> Option<Arc<ClimateTile>> {
        let mut memory = lock(&self.memory);
        memory.clock += 1;
        let now = memory.clock;
        let entry = memory.tiles.get_mut(&key)?;
        entry.1 = now;
        Some(Arc::clone(&entry.0))
    }

    fn remember(&self, key: TileKey, tile: &Arc<ClimateTile>) {
        if self.capacity == 0 {
            return;
        }
        let mut memory = lock(&self.memory);
        memory.clock += 1;
        let now = memory.clock;
        memory.tiles.insert(key, (Arc::clone(tile), now));
        while memory.tiles.len() > self.capacity {
            let oldest = memory
                .tiles
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(key, _)| *key);
            match oldest {
                Some(oldest) => memory.tiles.remove(&oldest),
                None => break,
            };
        }
    }

    /// How many decoded tiles are in memory.
    pub fn memory_tiles(&self) -> usize {
        lock(&self.memory).tiles.len()
    }
}

/// The tile columns that `count` pixel columns starting at `first` (which may be negative or past `width`: the
/// columns wrap) fall in, for tiles `tile` pixels wide on a level `width` pixels across.
fn tile_columns(first: i64, count: i64, width: i64, tile: i64) -> Vec<u32> {
    let mut cols: Vec<u32> = Vec::new();
    let mut remaining = count.clamp(0, width);
    let mut x = first.rem_euclid(width);
    while remaining > 0 {
        let col = x / tile;
        if !cols.contains(&(col as u32)) {
            cols.push(col as u32);
        }
        let end = ((col + 1) * tile).min(width);
        remaining -= end - x;
        x = end % width;
    }
    cols
}

/// Climate at world blocks: a [`ClimateStore`] placed on the globe by a [`Projection`], as
/// [`crate::LandCoverField`] places land cover.
#[derive(Clone)]
pub struct ClimateField {
    store: Arc<ClimateStore>,
    projection: Projection,
}

impl ClimateField {
    pub fn new(store: Arc<ClimateStore>, projection: Projection) -> Self {
        Self { store, projection }
    }

    pub fn store(&self) -> &Arc<ClimateStore> {
        &self.store
    }

    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    /// The climate at the world point `(x, z)` (fractional blocks; add 0.5 for a block's centre), read at the
    /// overview level that suits `metres_per_sample` of ground. See [`ClimateStore::lookup`].
    pub fn climate_at(&self, x: f64, z: f64, metres_per_sample: f64) -> ClimateLookup {
        let at = self.projection.to_lat_lon_flat(x, z);
        self.store.lookup(at.lat, at.lon, metres_per_sample)
    }

    /// The climate at the centre of block `(x, z)`, read as finely as the blocks are big.
    pub fn climate_at_block(&self, x: i32, z: i32) -> ClimateLookup {
        self.climate_at(
            f64::from(x) + 0.5,
            f64::from(z) + 0.5,
            self.projection.metres_per_block(),
        )
    }

    /// The fetches still missing for the block rectangle `x0..=x1` by `z0..=z1` (corners in either order) at
    /// `metres_per_sample`, for prefetching. The area is the lat/lon box around the rectangle's corners and
    /// edge midpoints, which for a local projection holds the rectangle.
    pub fn needs(
        &self,
        x0: i32,
        z0: i32,
        x1: i32,
        z1: i32,
        metres_per_sample: f64,
    ) -> Vec<ClimateFetch> {
        let (xa, xb) = (f64::from(x0.min(x1)), f64::from(x0.max(x1)) + 1.0);
        let (za, zb) = (f64::from(z0.min(z1)), f64::from(z0.max(z1)) + 1.0);
        let (xm, zm) = ((xa + xb) / 2.0, (za + zb) / 2.0);
        let mut south = f64::INFINITY;
        let mut north = f64::NEG_INFINITY;
        let mut west = f64::INFINITY;
        let mut east = f64::NEG_INFINITY;
        // Unwrapped longitudes, so a box across the date line is one range.
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
            let at = self.projection.to_lat_lon_unwrapped(x, z);
            south = south.min(at.lat);
            north = north.max(at.lat);
            west = west.min(at.lon);
            east = east.max(at.lon);
        }
        self.store.needs(
            GeoBox {
                south,
                north,
                west,
                east,
            },
            metres_per_sample,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::climate::NODATA;
    use crate::climatesource::{ClimateFetcher, ClimateSource};
    use crate::source::FetchError;
    use crate::testing::TiffSpec;
    use crate::testing::{MemoryClimate, SMALL_CLIMATE_GRID, TempDir, climate_file, serve_range};

    /// Raw temperature of a pixel of the miniature files (1 degree a pixel: 360 by 174, then 180 by 87, then 90
    /// by 43, in tiles of 64). A hole at one pixel and a block of four, and another level offset each.
    fn temp_raw(level: usize, x: u32, y: u32) -> u16 {
        if level == 0 && (x, y) == (100, 50) {
            return NODATA;
        }
        if level == 0 && (200..202).contains(&x) && (80..82).contains(&y) {
            return NODATA;
        }
        (3000 + x * 4 + y * 2 + level as u32 * 500) as u16
    }

    fn precip_raw(level: usize, x: u32, y: u32) -> u16 {
        if level == 0 && (x, y) == (300, 10) {
            return NODATA;
        }
        (1000 + x + y * 3 + level as u32 * 700) as u16
    }

    fn files() -> MemoryClimate {
        MemoryClimate::new()
            .with_raw(ClimateVar::Temperature, 64, 2, temp_raw)
            .with_raw(ClimateVar::Precipitation, 64, 2, precip_raw)
    }

    fn store(dir: &TempDir) -> ClimateStore {
        ClimateStore::with_grid(dir.path(), 16, SMALL_CLIMATE_GRID)
    }

    /// The place of the centre of pixel `(x, y)` of level 0.
    fn centre(x: u32, y: u32) -> (f64, f64) {
        (NORTH - (f64::from(y) + 0.5), WEST + f64::from(x) + 0.5)
    }

    /// The place of the corner shared by pixels `(x - 1, y - 1)` to `(x, y)` of level 0.
    fn corner(x: u32, y: u32) -> (f64, f64) {
        (NORTH - f64::from(y), WEST + f64::from(x))
    }

    fn value(var: ClimateVar, raw: u16) -> f64 {
        var.value_f64(raw).unwrap()
    }

    /// Answers a lookup by fetching what is missing from `files` until it is answered.
    fn resolve(
        store: &ClimateStore,
        files: &MemoryClimate,
        (lat, lon): (f64, f64),
        metres: f64,
        read: Read,
    ) -> ClimateLookup {
        for _ in 0..32 {
            match store.lookup_with(lat, lon, metres, read) {
                ClimateLookup::Need(fetch) => {
                    let bytes = files
                        .fetch(fetch.var, fetch.range)
                        .unwrap()
                        .expect("the file");
                    store.provide(&fetch, &bytes).expect("the store takes it");
                }
                done => return done,
            }
        }
        panic!("a lookup that never finishes");
    }

    fn assert_climate(got: ClimateLookup, temperature: f64, precipitation: f64) {
        let ClimateLookup::Climate(c) = got else {
            panic!("expected a climate, got {got:?}");
        };
        assert!(
            (f64::from(c.temperature_c) - temperature).abs() < 1e-3,
            "{} vs {temperature}",
            c.temperature_c
        );
        assert!(
            (f64::from(c.precipitation_mm) - precipitation).abs() < 1e-2,
            "{} vs {precipitation}",
            c.precipitation_mm
        );
    }

    fn pixel_values(x: u32, y: u32) -> (f64, f64) {
        (
            value(ClimateVar::Temperature, temp_raw(0, x, y)),
            value(ClimateVar::Precipitation, precip_raw(0, x, y)),
        )
    }

    #[test]
    fn a_lookup_asks_for_both_indexes_then_the_tiles_then_answers() {
        let dir = TempDir::new();
        let (store, files) = (store(&dir), files());
        let at = centre(130, 70);
        let mut asked = Vec::new();
        loop {
            match store.lookup_nearest(at.0, at.1, 1.0) {
                ClimateLookup::Need(fetch) => {
                    asked.push(fetch);
                    let bytes = files.fetch(fetch.var, fetch.range).unwrap().unwrap();
                    // The same question gives the same answer until something is provided.
                    assert_eq!(
                        store.lookup_nearest(at.0, at.1, 1.0),
                        ClimateLookup::Need(fetch)
                    );
                    store.provide(&fetch, &bytes).unwrap();
                }
                answer => {
                    let (t, p) = pixel_values(130, 70);
                    assert_climate(answer, t, p);
                    break;
                }
            }
        }
        let prefix = ByteRange {
            offset: 0,
            len: CLIMATE_PREFIX_LEN,
        };
        // Pixel (130, 70) of 360 is column 2 and row 1 of the 64-pixel tiles, six to a row: tile 8.
        let tile = Wanted::Tile { level: 0, tile: 8 };
        let expected = [
            (ClimateVar::Temperature, Wanted::Index),
            (ClimateVar::Precipitation, Wanted::Index),
            (ClimateVar::Temperature, tile),
            (ClimateVar::Precipitation, tile),
        ];
        assert_eq!(asked.len(), 4, "{asked:?}");
        for (fetch, (var, wanted)) in asked.iter().zip(expected) {
            assert_eq!((fetch.var, fetch.wanted), (var, wanted));
        }
        assert_eq!(asked[0].range, prefix);
        assert_eq!(files.fetch_count(), 4);
        // Held now: the next lookup nearby asks for nothing.
        let (lat, lon) = centre(135, 66);
        assert!(matches!(
            store.lookup(lat, lon, 1.0),
            ClimateLookup::Climate(_)
        ));
    }

    #[test]
    fn a_value_is_the_bilinear_mix_of_the_four_nearest_pixels_even_across_tiles() {
        let dir = TempDir::new();
        let (store, files) = (store(&dir), files());
        // The corner of pixels (127..=128, 63..=64): four different tiles.
        let got = resolve(&store, &files, corner(128, 64), 1.0, Read::Bilinear);
        let mean = |f: fn(u32, u32) -> f64| {
            [(127, 63), (128, 63), (127, 64), (128, 64)]
                .iter()
                .map(|&(x, y)| f(x, y))
                .sum::<f64>()
                / 4.0
        };
        assert_climate(
            got,
            mean(|x, y| pixel_values(x, y).0),
            mean(|x, y| pixel_values(x, y).1),
        );
        assert_eq!(
            files.fetch_count(),
            2 + 8,
            "two indexes and four tiles of each"
        );
        // A quarter of the way from pixel 130 to 131 along a row, level with the centres.
        let (lat, _) = centre(130, 70);
        let lon = WEST + 130.75;
        let (a, b) = (pixel_values(130, 70), pixel_values(131, 70));
        let got = resolve(&store, &files, (lat, lon), 1.0, Read::Bilinear);
        assert_climate(got, 0.75 * a.0 + 0.25 * b.0, 0.75 * a.1 + 0.25 * b.1);
        // At a pixel's centre it is the pixel.
        let got = resolve(&store, &files, centre(130, 70), 1.0, Read::Bilinear);
        assert_climate(got, a.0, a.1);
    }

    #[test]
    fn nearest_reads_the_pixel_the_place_is_in() {
        let dir = TempDir::new();
        let (store, files) = (store(&dir), files());
        let (lat, lon) = centre(10, 20);
        // Anywhere in the pixel, not only at its centre.
        for (dlat, dlon) in [(0.4, -0.4), (-0.4, 0.4), (0.0, 0.0)] {
            let got = resolve(&store, &files, (lat + dlat, lon + dlon), 1.0, Read::Nearest);
            let (t, p) = pixel_values(10, 20);
            assert_climate(got, t, p);
        }
    }

    #[test]
    fn the_columns_wrap_at_the_date_line_and_the_rows_stop_at_the_edges() {
        let dir = TempDir::new();
        let (store, files) = (store(&dir), files());
        let (lat, _) = centre(0, 70);
        // Just west of 180: halfway between the last column and the first.
        let lon = 179.999;
        let tx = lon - WEST - 359.5;
        let (a, b) = (pixel_values(359, 70), pixel_values(0, 70));
        let got = resolve(&store, &files, (lat, lon), 1.0, Read::Bilinear);
        assert_climate(
            got,
            (1.0 - tx) * a.0 + tx * b.0,
            (1.0 - tx) * a.1 + tx * b.1,
        );
        // And the same from the other side, just east of -180.
        let lon = -179.999;
        let tx = lon - WEST + 0.5;
        let got = resolve(&store, &files, (lat, lon), 1.0, Read::Bilinear);
        assert_climate(
            got,
            (1.0 - tx) * a.0 + tx * b.0,
            (1.0 - tx) * a.1 + tx * b.1,
        );
        // A tenth of a pixel inside the top and the bottom edge: only the edge row is mixed.
        let lon = WEST + 150.5;
        let got = resolve(&store, &files, (NORTH - 0.1, lon), 1.0, Read::Bilinear);
        let (t, p) = pixel_values(150, 0);
        assert_climate(got, t, p);
        let got = resolve(&store, &files, (SOUTH + 0.1, lon), 1.0, Read::Bilinear);
        let (t, p) = pixel_values(150, 173);
        assert_climate(got, t, p);
    }

    #[test]
    fn a_place_off_the_grid_is_no_data_and_asks_for_nothing() {
        let dir = TempDir::new();
        let store = store(&dir);
        for (lat, lon) in [
            (84.5, 0.0),
            (-90.001, 0.0),
            (89.0, 10.0),
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
        ] {
            assert_eq!(
                store.lookup(lat, lon, 1.0),
                ClimateLookup::NoData,
                "{lat} {lon}"
            );
        }
    }

    #[test]
    fn pixels_with_no_data_are_no_data_or_are_left_out_of_the_mix() {
        let dir = TempDir::new();
        let (store, files) = (store(&dir), files());
        // The pixel itself.
        let got = resolve(&store, &files, centre(100, 50), 1.0, Read::Nearest);
        assert_eq!(got, ClimateLookup::NoData);
        // Its neighbours leave it out and share its weight.
        // (Only the temperature has the hole: the precipitation is the mean of all four.)
        let near = [(101u32, 50u32), (100, 51), (101, 51)];
        let all = [(100u32, 50u32), (101, 50), (100, 51), (101, 51)];
        let mean = |pixels: &[(u32, u32)], f: fn(u32, u32) -> f64| {
            pixels.iter().map(|&(x, y)| f(x, y)).sum::<f64>() / pixels.len() as f64
        };
        let got = resolve(&store, &files, corner(101, 51), 1.0, Read::Bilinear);
        assert_climate(
            got,
            mean(&near, |x, y| {
                value(ClimateVar::Temperature, temp_raw(0, x, y))
            }),
            mean(&all, |x, y| {
                value(ClimateVar::Precipitation, precip_raw(0, x, y))
            }),
        );
        // All four with no data.
        let got = resolve(&store, &files, corner(201, 81), 1.0, Read::Bilinear);
        assert_eq!(got, ClimateLookup::NoData);
        // No data in either variable is no data.
        let got = resolve(&store, &files, centre(300, 10), 1.0, Read::Nearest);
        assert_eq!(got, ClimateLookup::NoData);
    }

    #[test]
    fn the_level_follows_the_requested_ground_size() {
        let dir = TempDir::new();
        let (store, files) = (store(&dir), files());
        let at = centre(130, 70);
        // 174, 87 and 43 rows: about 111, 222 and 450 km a pixel.
        for (metres, level, h) in [
            (10.0, 0, 174),
            (150_000.0, 0, 174),
            (250_000.0, 1, 87),
            (1e9, 2, 43),
        ] {
            let got = resolve(&store, &files, at, metres, Read::Nearest);
            let w = 360 >> level;
            let (x, y) = pixel_position(at.0, at.1, w, h).unwrap();
            let (x, y) = (x.floor() as u32, y.floor() as u32);
            assert_climate(
                got,
                value(ClimateVar::Temperature, temp_raw(level, x, y)),
                value(ClimateVar::Precipitation, precip_raw(level, x, y)),
            );
        }
    }

    #[test]
    fn what_was_fetched_is_read_from_disk_by_a_new_store_without_a_request() {
        let dir = TempDir::new();
        let files = files();
        let first = store(&dir);
        let at = corner(128, 64);
        let a = resolve(&first, &files, at, 1.0, Read::Bilinear);
        let made = files.fetch_count();
        assert!(made >= 10);
        assert!(first.tile_path(ClimateVar::Temperature, 0, 8).is_file());
        let second = store(&dir);
        assert_eq!(second.lookup(at.0, at.1, 1.0), a);
        assert_eq!(files.fetch_count(), made);
        assert!(
            dir.path()
                .join("chelsa-v2.1-1981-2010-rules1")
                .join("bio12")
                .is_dir(),
            "files live in a folder named by the dataset and the rules"
        );
    }

    #[test]
    fn the_memory_cache_is_bounded_and_evicted_tiles_come_back_from_disk() {
        let dir = TempDir::new();
        let files = files();
        let store = ClimateStore::with_grid(dir.path(), 3, SMALL_CLIMATE_GRID);
        let places = [
            centre(10, 10),
            centre(70, 10),
            centre(130, 10),
            centre(190, 10),
        ];
        for at in places {
            resolve(&store, &files, at, 1.0, Read::Nearest);
        }
        assert_eq!(store.memory_tiles(), 3);
        let made = files.fetch_count();
        for at in places {
            assert!(matches!(
                store.lookup_nearest(at.0, at.1, 1.0),
                ClimateLookup::Climate(_)
            ));
        }
        assert_eq!(files.fetch_count(), made);
        assert_eq!(store.memory_tiles(), 3);
    }

    #[test]
    fn needs_lists_the_indexes_then_every_tile_the_box_touches_across_the_date_line() {
        let dir = TempDir::new();
        let (store, files) = (store(&dir), files());
        // Longitudes 179 to 181 and latitudes 10 to 11: columns 359 and 0 on both sides, in tiles 11 and 6 (column 5 and 0 of row 1).
        let area = GeoBox {
            south: 10.0,
            north: 11.0,
            west: 179.0,
            east: 181.0,
        };
        let first = store.needs(area, 1.0);
        assert_eq!(first.len(), 2);
        assert!(first.iter().all(|f| f.wanted == Wanted::Index));
        for fetch in &first {
            store
                .provide(
                    fetch,
                    &files.fetch(fetch.var, fetch.range).unwrap().unwrap(),
                )
                .unwrap();
        }
        let tiles = store.needs(area, 1.0);
        let mut got: Vec<_> = tiles.iter().map(|f| (f.var.name(), f.wanted)).collect();
        got.sort_by_key(|(name, w)| match w {
            Wanted::Tile { tile, .. } => (*name, *tile),
            Wanted::Index => (*name, 0),
        });
        let t = |tile| Wanted::Tile { level: 0, tile };
        assert_eq!(
            got,
            vec![
                ("bio01", t(6)),
                ("bio01", t(11)),
                ("bio12", t(6)),
                ("bio12", t(11)),
            ]
        );
        for fetch in &tiles {
            assert!(store.is_missing(fetch));
            store
                .provide(
                    fetch,
                    &files.fetch(fetch.var, fetch.range).unwrap().unwrap(),
                )
                .unwrap();
            assert!(!store.is_missing(fetch));
        }
        assert!(store.needs(area, 1.0).is_empty());
        // Nothing on the grid: nothing to fetch.
        let polar = GeoBox {
            south: 85.0,
            north: 89.0,
            west: 0.0,
            east: 10.0,
        };
        assert!(store.needs(polar, 1.0).is_empty());
    }

    #[test]
    fn tile_columns_wrap_and_cover_a_box_once() {
        // 360 pixels in tiles of 64: six columns, the last one 40 wide.
        assert_eq!(tile_columns(0, 10, 360, 64), vec![0]);
        assert_eq!(tile_columns(60, 10, 360, 64), vec![0, 1]);
        assert_eq!(tile_columns(-2, 4, 360, 64), vec![5, 0]);
        assert_eq!(tile_columns(358, 4, 360, 64), vec![5, 0]);
        assert_eq!(tile_columns(-100, 5000, 360, 64), vec![4, 5, 0, 1, 2, 3]);
    }

    #[test]
    fn a_file_that_is_not_the_grid_or_not_16_bit_is_refused_and_nothing_is_stored() {
        let dir = TempDir::new();
        let store = store(&dir);
        let ClimateLookup::Need(index) = store.lookup(10.0, 10.0, 1.0) else {
            panic!("nothing is held yet")
        };
        let eight =
            crate::testing::encode_tiff(&TiffSpec::new(360, 174, 64, 2).bigtiff(), &|_, _, _| 1);
        let wrong_size = climate_file(400, 174, 64, 2, &|_, _, _| 1);
        for file in [eight.bytes, wrong_size.bytes] {
            assert!(matches!(
                store.provide(&index, serve_range(&file, index.range)),
                Err(ProvideError::Tiff(TiffError::Unsupported(_)))
            ));
        }
        assert_eq!(store.provide(&index, &[]), Err(ProvideError::Empty));
        assert!(!store.var_dir(index.var).join("index-0.bin").exists());
        assert_eq!(store.lookup(10.0, 10.0, 1.0), ClimateLookup::Need(index));
        // The real store wants the real size: a miniature is not that.
        let real = ClimateStore::new(dir.path().join("real"), 4);
        let ClimateLookup::Need(index) = real.lookup(10.0, 10.0, 1.0) else {
            panic!()
        };
        let small = files().file_bytes(ClimateVar::Temperature).unwrap();
        assert!(matches!(
            real.provide(&index, serve_range(&small, index.range)),
            Err(ProvideError::Tiff(TiffError::Unsupported(_)))
        ));
    }

    fn source(fetcher: Arc<MemoryClimate>, dir: &TempDir) -> ClimateSource {
        let store = Arc::new(store(dir));
        let projection = Projection::local(46.0, 7.5, 1000.0).unwrap();
        ClimateSource::new(ClimateField::new(store, projection), fetcher)
    }

    #[test]
    fn a_blocking_read_fetches_what_it_needs_then_stops_asking() {
        let dir = TempDir::new();
        let fetcher = Arc::new(files());
        let source = source(fetcher.clone(), &dir);
        let got = source
            .climate_at_blocking(10.5, 20.5, 1.0)
            .unwrap()
            .unwrap();
        let at = source.field().projection().to_lat_lon(10.5, 20.5);
        assert_eq!(
            source.field().store().lookup(at.lat, at.lon, 1.0),
            ClimateLookup::Climate(got)
        );
        let asked = fetcher.fetch_count();
        assert!(asked >= 4, "two indexes and a tile of each, got {asked}");
        assert_eq!(
            source.climate_at_blocking(10.5, 20.5, 1.0).unwrap(),
            Some(got)
        );
        assert_eq!(
            source
                .climate_at_latlon_blocking(at.lat, at.lon, 1.0)
                .unwrap(),
            Some(got)
        );
        assert_eq!(fetcher.fetch_count(), asked);
        assert!(matches!(
            source.field().climate_at_block(10, 20),
            ClimateLookup::Climate(_)
        ));
        // The field's prefetch list is empty once the blocks are held.
        assert!(source.field().needs(10, 20, 10, 20, 1.0).is_empty());
        // A place above the data is None without a fetch.
        assert_eq!(
            source.climate_at_latlon_blocking(85.0, 0.0, 1.0).unwrap(),
            None
        );
        assert_eq!(fetcher.fetch_count(), asked);
    }

    #[test]
    fn a_file_the_source_does_not_have_is_no_data_and_asked_for_once_even_by_the_next_run() {
        let dir = TempDir::new();
        let fetcher = Arc::new(MemoryClimate::new());
        let first = source(fetcher.clone(), &dir);
        for _ in 0..3 {
            assert_eq!(
                first.climate_at_latlon_blocking(10.0, 10.0, 1.0).unwrap(),
                None
            );
        }
        assert_eq!(fetcher.fetch_count(), 1);
        assert_eq!(fetcher.fetches_of(ClimateVar::Temperature), 1);
        let again = Arc::new(MemoryClimate::new());
        let second = source(again.clone(), &dir);
        assert_eq!(
            second.climate_at_latlon_blocking(10.0, 10.0, 1.0).unwrap(),
            None
        );
        assert_eq!(again.fetch_count(), 0);
    }

    #[test]
    fn a_refusal_reads_as_no_data_for_the_run_and_writes_nothing() {
        let dir = TempDir::new();
        let fetcher = Arc::new(files());
        fetcher.fail_with(Some(FetchError::Refused("not approved".into())));
        let source = source(fetcher.clone(), &dir);
        assert_eq!(
            source.climate_at_latlon_blocking(10.0, 10.0, 1.0).unwrap(),
            None
        );
        assert_eq!(fetcher.fetch_count(), 1);
        assert_eq!(
            source.field().store().lookup(10.0, 10.0, 1.0),
            ClimateLookup::NoData
        );
        // Another run (approved meanwhile) asks again.
        fetcher.fail_with(None);
        let next = self::source(fetcher.clone(), &dir);
        assert!(
            next.climate_at_latlon_blocking(10.0, 10.0, 1.0)
                .unwrap()
                .is_some()
        );
        // A failing fetcher is an error that marks nothing, and recovers.
        let other = TempDir::new();
        let third = self::source(fetcher.clone(), &other);
        fetcher.fail_with(Some(FetchError::Temporary("offline".into())));
        assert!(third.climate_at_latlon_blocking(10.0, 10.0, 1.0).is_err());
        fetcher.fail_with(None);
        assert!(
            third
                .climate_at_latlon_blocking(10.0, 10.0, 1.0)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn a_refused_tile_is_no_data_while_the_rest_is_read() {
        let dir = TempDir::new();
        let fetcher = Arc::new(files());
        let source = source(fetcher.clone(), &dir);
        let at = centre(130, 70);
        for _ in 0..2 {
            let ClimateLookup::Need(index) = source.field().store().lookup(at.0, at.1, 1.0) else {
                panic!("needs an index");
            };
            source.satisfy(&index).unwrap();
        }
        let ClimateLookup::Need(tile) = source.field().store().lookup(at.0, at.1, 1.0) else {
            panic!("needs a tile");
        };
        fetcher.fail_with(Some(FetchError::Refused("no".into())));
        assert!(matches!(source.satisfy(&tile), Err(FetchError::Refused(_))));
        assert_eq!(
            source.field().store().lookup(at.0, at.1, 1.0),
            ClimateLookup::NoData
        );
        // Another tile is still wanted normally.
        let other = centre(10, 10);
        assert!(matches!(
            source.field().store().lookup(other.0, other.1, 1.0),
            ClimateLookup::Need(_)
        ));
    }
}

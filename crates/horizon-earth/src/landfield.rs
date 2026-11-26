//! Land cover over the globe and over world blocks, read from WorldCover files the caller fetches in pieces.
//!
//! This mirrors [`crate::source`] and [`crate::height`] with one difference: the crate has no downloader, and
//! the pieces a WorldCover file is read from are byte ranges of a big file, so a lookup that lacks data does not
//! fetch it. It answers with the [`Fetch`] to make (which file, which byte range), and the caller hands the
//! bytes back with [`LandCoverStore::provide`]. Nothing is ever guessed: no class is answered for a place whose
//! tile is not held.
//!
//! What is held is bounded. Decoded tiles (about 1 MiB each) live in a memory cache of a fixed size that drops
//! the least recently used; the directory of each file is a few tens of kilobytes. Everything provided is also
//! written under a folder named by [`cache_key`], so a new rule version starts with an empty cache, and it is
//! found again without a fetch by the next run. The store is safe to share between threads.

use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::landcover::{
    LandClass, SPECK_GRID, SPECK_KEEP, WorldCoverFile, cache_key, level_for, soften_specks,
};
use crate::projection::Projection;
use crate::source::write_atomically;
use crate::tiff::{ByteRange, Level, Parse, Piece, TiffError, TiffIndex, parse_index};

/// The most bytes of a file's index (its directories and tile arrays) the store keeps in memory and on disk.
/// A real file's is about 28 KiB.
pub const MAX_INDEX_BYTES: usize = 4 << 20;
/// How many files' indexes the store holds in memory (more are read back from disk when needed).
pub const MAX_FILES_IN_MEMORY: usize = 256;

/// What a [`Fetch`] is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Wanted {
    /// A piece of the file's index: its first bytes, or a directory or array beyond them.
    Index,
    /// The compressed bytes of tile `tile` of level `level` (0 is full resolution).
    Tile { level: u8, tile: u32 },
}

/// A byte range of a file that a lookup needs and the store does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fetch {
    pub file: WorldCoverFile,
    pub range: ByteRange,
    pub wanted: Wanted,
}

/// The answer to a land-cover question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// The class at the place. [`LandClass::NoData`] is the file's own "no data" (or a tile it leaves out).
    Class(LandClass),
    /// Not here yet: fetch this, give it to [`LandCoverStore::provide`], and ask again.
    Need(Fetch),
    /// The caller said this file does not exist at the source ([`LandCoverStore::mark_absent`]); WorldCover
    /// has no file for places that are all sea. Also the answer for a position that is not on the globe.
    NoFile,
}

/// Why [`LandCoverStore::provide`] refused bytes (nothing was stored).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvideError {
    /// No bytes.
    Empty,
    /// More bytes than the range asked for, or a tile of the wrong length.
    WrongLength,
    /// The file's index is not known yet, so a tile cannot be placed.
    NoIndex,
    /// The tile is not one of the file, or its range is not the one the file's index gives.
    NotWanted,
    /// The bytes are not what a WorldCover file holds.
    Tiff(TiffError),
    /// The index would be larger than [`MAX_INDEX_BYTES`].
    TooLarge,
}

impl fmt::Display for ProvideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("no bytes were provided"),
            Self::WrongLength => f.write_str("the bytes are not the length of the range asked for"),
            Self::NoIndex => f.write_str("the file's index is not held yet"),
            Self::NotWanted => f.write_str("the file has no such tile at that range"),
            Self::Tiff(e) => e.fmt(f),
            Self::TooLarge => f.write_str("the file's index is larger than the store keeps"),
        }
    }
}

impl std::error::Error for ProvideError {}

/// One decoded tile: 8-bit class codes, row by row.
#[derive(Debug, PartialEq, Eq)]
pub struct LandTile {
    pub width: u32,
    pub height: u32,
    /// The classes after the towns are given back ([`crate::landcover::unbuild`]) and the specks take the class
    /// around them ([`soften_specks`]).
    pixels: Vec<u8>,
    /// The specks' own classes: `(pixel index, original code)`, sorted by index.
    specks: Vec<(u32, u8)>,
}

impl LandTile {
    /// The tile of `level` whose decoded class codes are `pixels`, with its built-up land given back to nature
    /// ([`crate::landcover::unbuild`]): no read ever sees a built-up class. Specks of one or two pixels take
    /// the class around them but remember their own ([`LandTile::class_mixed`]).
    fn decoded(level: &Level, mut pixels: Vec<u8>) -> Self {
        let (width, height) = (level.tile_width, level.tile_height);
        crate::landcover::unbuild(&mut pixels, width as usize, height as usize);
        let specks = soften_specks(&mut pixels, width as usize, height as usize);
        Self {
            width,
            height,
            pixels,
            specks,
        }
    }

    fn code_at(&self, x: u32, y: u32) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.pixels
            .get(y as usize * self.width as usize + x as usize)
            .copied()
    }

    /// The class at pixel `(x, y)` of the tile; [`LandClass::NoData`] for a value the dataset does not define.
    /// `None` outside the tile. A speck of one or two pixels reads as the class around it (see
    /// [`LandTile::class_mixed`] for the speck's own class kept in part).
    pub fn class(&self, x: u32, y: u32) -> Option<LandClass> {
        let code = self.code_at(x, y)?;
        Some(LandClass::from_code(code).unwrap_or(LandClass::NoData))
    }

    /// The class at the point `(fx, fy)` (each `0..1`) of pixel `(x, y)`. Away from specks it is
    /// [`LandTile::class`]. On a speck it is the speck's own class on [`SPECK_KEEP`] of the pixel's ground
    /// (cut into [`SPECK_GRID`] by [`SPECK_GRID`] cells, each kept or not by a hash of the cell and of `place`,
    /// which tells pixels of different levels or files apart) and the class around it elsewhere.
    pub fn class_mixed(
        &self,
        x: u32,
        y: u32,
        inside: (f64, f64),
        place: (i64, i64, u64),
    ) -> Option<LandClass> {
        match self.speck_kept(x, y, inside, place) {
            Some(class) => Some(class),
            None => self.class(x, y),
        }
    }

    /// The speck's own class when pixel `(x, y)` is a speck and the point `inside` it (see
    /// [`LandTile::class_mixed`]) is on the part of it that keeps it; `None` otherwise.
    fn speck_kept(
        &self,
        x: u32,
        y: u32,
        (fx, fy): (f64, f64),
        place: (i64, i64, u64),
    ) -> Option<LandClass> {
        if self.specks.is_empty() || x >= self.width || y >= self.height {
            return None;
        }
        let index = y * self.width + x;
        let at = self.specks.binary_search_by_key(&index, |&(i, _)| i).ok()?;
        let cell =
            |f: f64| ((f * f64::from(SPECK_GRID)) as i64).clamp(0, i64::from(SPECK_GRID) - 1);
        let key = cell(fy) * i64::from(SPECK_GRID) + cell(fx);
        let h = crate::warp::cell_hash(place.0, place.1, key, place.2 ^ 0x5EC4_5BEC_0000_0001);
        (h < SPECK_KEEP)
            .then(|| LandClass::from_code(self.specks[at].1).unwrap_or(LandClass::NoData))
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
    used: u64,
}

impl FileState {
    fn parse(&self) -> Result<Parse, TiffError> {
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
                check_land_cover(&index)?;
                Ok(Parse::Done(index))
            }
            need => Ok(need),
        }
    }

    fn held_bytes(&self) -> usize {
        self.pieces.iter().map(|(_, b)| b.len()).sum()
    }
}

/// Land cover is 8-bit class codes with no predictor, as WorldCover has them. The TIFF reader also reads other
/// sample formats (for climate files); a file of those is not land cover and is refused here.
fn check_land_cover(index: &TiffIndex) -> Result<(), TiffError> {
    for level in &index.levels {
        if level.bits_per_sample != 8 {
            return Err(TiffError::Unsupported(
                "land cover with anything but 8 bits per sample",
            ));
        }
        if level.predictor != 1 {
            return Err(TiffError::Unsupported("land cover with a predictor"));
        }
    }
    Ok(())
}

struct Memory {
    files: crate::fastmap::FastMap<WorldCoverFile, FileState>,
    tiles: crate::fastmap::FastMap<TileKey, (Arc<LandTile>, u64)>,
    absent: crate::fastmap::FastSet<WorldCoverFile>,
    /// Tiles the source will not give this session: read as "no data", never stored.
    blank: crate::fastmap::FastSet<TileKey>,
    clock: u64,
}

type TileKey = (WorldCoverFile, u8, u32);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A box on the globe for [`LandCoverStore::needs`]: latitudes, and longitudes that may run past 180 (west
/// 170, east 190 is the box across the date line). A box 360 degrees wide or more is the whole circle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoBox {
    pub south: f64,
    pub north: f64,
    pub west: f64,
    pub east: f64,
}

/// The land-cover tiles of the globe: a bounded memory cache of decoded tiles over a disk cache of what the
/// caller fetched, with no fetcher of its own.
pub struct LandCoverStore {
    dir: PathBuf,
    capacity: usize,
    memory: Mutex<Memory>,
}

impl LandCoverStore {
    /// A store that keeps files under `dir` (in a folder named by [`cache_key`]) and at most `memory_tiles`
    /// decoded tiles (each about 1 MiB at WorldCover's tiling).
    pub fn new(dir: impl Into<PathBuf>, memory_tiles: usize) -> Self {
        Self {
            dir: dir.into().join(cache_key()),
            capacity: memory_tiles,
            memory: Mutex::new(Memory {
                files: Default::default(),
                tiles: Default::default(),
                absent: Default::default(),
                blank: Default::default(),
                clock: 0,
            }),
        }
    }

    /// The folder holding what is kept of `file`.
    pub fn file_dir(&self, file: WorldCoverFile) -> PathBuf {
        self.dir.join(file.tile_name())
    }

    /// Where the compressed bytes of a tile are kept on disk.
    pub fn tile_path(&self, file: WorldCoverFile, level: u8, tile: u32) -> PathBuf {
        self.file_dir(file).join(format!("tile-{level}-{tile}.bin"))
    }

    fn index_path(&self, file: WorldCoverFile, offset: u64) -> PathBuf {
        self.file_dir(file).join(format!("index-{offset}.bin"))
    }

    fn length_path(&self, file: WorldCoverFile, len: u64) -> PathBuf {
        self.file_dir(file).join(format!("length-{len}"))
    }

    fn absent_path(&self, file: WorldCoverFile) -> PathBuf {
        self.file_dir(file).join("absent")
    }

    /// Records that the source has no such file (a 404: WorldCover has none for open sea), so lookups there
    /// answer [`Lookup::NoFile`]. It is kept on disk too, so the next run does not ask again.
    pub fn mark_absent(&self, file: WorldCoverFile) {
        lock(&self.memory).absent.insert(file);
        let _ = write_atomically(&self.absent_path(file), b"");
    }

    /// Records that the source will not give what `fetch` names (the player's approval was taken back, the
    /// server refuses): for the rest of this run the file (for an index piece) or the tile reads as "no data", so
    /// nothing waits for it. Nothing is written to disk, so another run asks again.
    pub fn mark_unavailable(&self, fetch: &Fetch) {
        let mut memory = lock(&self.memory);
        match fetch.wanted {
            Wanted::Index => {
                memory.absent.insert(fetch.file);
            }
            Wanted::Tile { level, tile } => {
                memory.blank.insert((fetch.file, level, tile));
            }
        }
    }

    /// Whether `fetch` is still something the store lacks: a stale request (the bytes arrived meanwhile, the file
    /// turned out absent, a later piece of the index is the next one wanted) is not, so whoever fetches in the
    /// background skips it instead of asking the source twice.
    pub fn is_missing(&self, fetch: &Fetch) -> bool {
        if self.is_absent(fetch.file) {
            return false;
        }
        let Err(next) = self.index_of(fetch.file) else {
            // The index is whole: only a tile can still be missing.
            let Wanted::Tile { level, tile } = fetch.wanted else {
                return false;
            };
            let Ok(index) = self.index_of(fetch.file) else {
                return false;
            };
            let Some(l) = index.levels.get(usize::from(level)) else {
                return false;
            };
            return self
                .tile_of_local(fetch.file, usize::from(level), l, tile as usize)
                .is_err();
        };
        fetch.wanted == Wanted::Index && next.range == fetch.range
    }

    /// Whether `file` is known to be absent at the source, from memory or from the marker on disk.
    fn is_absent(&self, file: WorldCoverFile) -> bool {
        self.load_pieces_from_disk(file);
        lock(&self.memory).absent.contains(&file)
    }

    /// The class at `(lat, lon)`, read at the overview level whose pixels are no taller on the ground than
    /// `metres_per_sample` (the finest level when even that is coarser). Never fetches: it answers
    /// [`Lookup::Need`] when it lacks bytes. This is the raw pixel grid (specks one or two pixels big are a mix
    /// of their own class and the land around, see [`LandTile::class_mixed`]);
    /// [`LandCoverStore::lookup_smooth`] and [`LandCoverField::class_at`] round its corners.
    pub fn lookup(&self, lat: f64, lon: f64, metres_per_sample: f64) -> Lookup {
        self.read(lat, lon, metres_per_sample, false)
    }

    /// Like [`LandCoverStore::lookup`] but the pixels are not squares: where a point is in the pixel, the class
    /// is the one holding most of a bilinear blend of the four pixels whose centres are nearest (the pixel's
    /// own class on a tie). A straight edge between two classes stays at the pixel line, but a corner of
    /// pixels is rounded and a staircase of one-pixel steps is a diagonal. It reads up to one pixel past the
    /// point, so it may need a neighbouring tile.
    pub fn lookup_smooth(&self, lat: f64, lon: f64, metres_per_sample: f64) -> Lookup {
        self.read(lat, lon, metres_per_sample, true)
    }

    fn read(&self, lat: f64, lon: f64, metres_per_sample: f64, smooth: bool) -> Lookup {
        let Some(file) = WorldCoverFile::containing(lat, lon) else {
            return Lookup::NoFile;
        };
        if self.is_absent(file) {
            return Lookup::NoFile;
        }
        let index = match self.index_of(file) {
            Err(fetch) => return Lookup::Need(fetch),
            Ok(index) => index,
        };
        let level_no = level_for(&index.levels, metres_per_sample);
        let Some(level) = index.levels.get(level_no) else {
            return Lookup::Class(LandClass::NoData);
        };
        let (px, py) = file.pixel(lat, lon, level.width, level.height);
        let inside = file.pixel_fraction(lat, lon, level.width, level.height);
        let (col, row) = (px / level.tile_width, py / level.tile_height);
        let Some(tile_no) = level.tile_index(col, row) else {
            return Lookup::Class(LandClass::NoData);
        };
        // Which pixel of which level of which file, for the hash that keeps a speck in part.
        let salt = ((u64::from(file.south as u16) << 32) | (u64::from(file.west as u16) << 16))
            ^ (level_no as u64);
        let tile = match self.tile_of(file, level_no, level, tile_no) {
            Ok(None) => return Lookup::Class(LandClass::NoData),
            Ok(Some(tile)) => tile,
            Err(fetch) => return Lookup::Need(fetch),
        };
        let (lx, ly) = (px % level.tile_width, py % level.tile_height);
        let place = (i64::from(px), i64::from(py), salt);
        if !smooth {
            return Lookup::Class(
                tile.class_mixed(lx, ly, inside, place)
                    .unwrap_or(LandClass::NoData),
            );
        }
        if let Some(kept) = tile.speck_kept(lx, ly, inside, place) {
            return Lookup::Class(kept);
        }
        // The four pixels whose centres are nearest the point, with the bilinear weights of the point among
        // them: the own pixel, the one beside it, the one above or below, and the diagonal.
        let (u, v) = (inside.0 - 0.5, inside.1 - 0.5);
        let (step_x, tx) = (if u >= 0.0 { 1_i64 } else { -1 }, u.abs());
        let (step_y, ty) = (if v >= 0.0 { 1_i64 } else { -1 }, v.abs());
        let (own, here) = ((i64::from(px), i64::from(py)), (col, row));
        let code_at = |x: i64, y: i64| -> Result<u8, Fetch> {
            let x = x.clamp(0, i64::from(level.width) - 1) as u32;
            let y = y.clamp(0, i64::from(level.height) - 1) as u32;
            let there = (x / level.tile_width, y / level.tile_height);
            let (lx, ly) = (x % level.tile_width, y % level.tile_height);
            if there == here {
                return Ok(tile.code_at(lx, ly).unwrap_or(0));
            }
            let Some(no) = level.tile_index(there.0, there.1) else {
                return Ok(0);
            };
            match self.tile_of(file, level_no, level, no)? {
                Some(other) => Ok(other.code_at(lx, ly).unwrap_or(0)),
                None => Ok(0),
            }
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
            let code = match code_at(x, y) {
                Ok(code) => code,
                Err(fetch) => return Lookup::Need(fetch),
            };
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
        Lookup::Class(LandClass::from_code(best.0).unwrap_or(LandClass::NoData))
    }

    /// The fetches still missing for lookups anywhere in `area` at `metres_per_sample`: first the index pieces of
    /// files whose index is incomplete, and for files whose index is held, the tiles the area touches. Asking
    /// again after providing gives the next round. Files marked absent are left out.
    pub fn needs(&self, area: GeoBox, metres_per_sample: f64) -> Vec<Fetch> {
        let mut out: Vec<Fetch> = Vec::new();
        let mut seen: HashSet<Fetch> = HashSet::new();
        let finite = [area.south, area.north, area.west, area.east]
            .iter()
            .all(|v| v.is_finite());
        if !finite {
            return out;
        }
        let (south, north) = (
            area.south.min(area.north).clamp(-90.0, 90.0),
            area.south.max(area.north).clamp(-90.0, 90.0),
        );
        let (west, mut east) = (area.west, area.east);
        if east < west {
            east += 360.0 * ((west - east) / 360.0).ceil();
        }
        // At most the whole circle: 120 cells of 3 degrees.
        east = east.min(west + 359.999_999);
        let step = f64::from(crate::landcover::FILE_DEGREES);
        let (k0, k1) = ((west / step).floor() as i64, (east / step).floor() as i64);
        let (r0, r1) = ((south / step).floor() as i64, (north / step).floor() as i64);
        for r in r0..=r1 {
            for k in k0..=k1 {
                let Some(file) =
                    WorldCoverFile::containing(r as f64 * step + 1.5, k as f64 * step + 1.5)
                else {
                    continue;
                };
                // A pole's own row holds the pole: the 90th parallel is not a row of its own.
                if r as f64 * step >= 90.0 || self.is_absent(file) {
                    continue;
                }
                // The box inside this cell, in the cell's own (unwrapped) coordinates.
                let cell_south = south.max(r as f64 * step);
                let cell_north = north.min(r as f64 * step + step);
                let cell_west = west.max(k as f64 * step);
                let cell_east = east.min(k as f64 * step + step);
                match self.index_of(file) {
                    Err(fetch) => {
                        if seen.insert(fetch) {
                            out.push(fetch);
                        }
                    }
                    Ok(index) => {
                        let level_no = level_for(&index.levels, metres_per_sample);
                        let Some(level) = index.levels.get(level_no) else {
                            continue;
                        };
                        let (x0, y0) = file.pixel(cell_north, cell_west, level.width, level.height);
                        let (x1, y1) = file.pixel(cell_south, cell_east, level.width, level.height);
                        for row in y0 / level.tile_height..=y1 / level.tile_height {
                            for col in x0 / level.tile_width..=x1 / level.tile_width {
                                let Some(tile_no) = level.tile_index(col, row) else {
                                    continue;
                                };
                                if let Err(fetch) =
                                    self.tile_of_local(file, level_no, level, tile_no)
                                    && seen.insert(fetch)
                                {
                                    out.push(fetch);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Hands over the bytes of a [`Fetch`]. For an index piece, any length up to the range's (shorter means the
    /// file ended there). For a tile, exactly the range's bytes, which must decode. Refused bytes are never
    /// stored, here or on disk. A failure to write the disk cache does not fail the call: the data is still
    /// held in memory.
    pub fn provide(&self, fetch: &Fetch, bytes: &[u8]) -> Result<(), ProvideError> {
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

    fn provide_index(&self, fetch: &Fetch, bytes: &[u8]) -> Result<(), ProvideError> {
        let file = fetch.file;
        let short = (bytes.len() as u64) < fetch.range.len;
        let end = fetch.range.offset.saturating_add(bytes.len() as u64);
        self.load_pieces_from_disk(file);
        {
            let mut memory = lock(&self.memory);
            memory.clock += 1;
            let now = memory.clock;
            let state = memory.files.entry(file).or_default();
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
            state.used = now;
            match state.parse() {
                Ok(Parse::Done(index)) => state.index = Some(Arc::new(index)),
                Ok(Parse::Need(_)) => {}
                Err(e) => {
                    // Refused bytes leave nothing behind.
                    (state.pieces, state.file_len, state.index) = backup;
                    return Err(ProvideError::Tiff(e));
                }
            }
        }
        let _ = write_atomically(&self.index_path(file, fetch.range.offset), bytes);
        if short {
            let _ = write_atomically(&self.length_path(file, end), b"");
        }
        self.trim_files();
        Ok(())
    }

    fn provide_tile(
        &self,
        fetch: &Fetch,
        level_no: u8,
        tile_no: u32,
        bytes: &[u8],
    ) -> Result<(), ProvideError> {
        let file = fetch.file;
        let Ok(index) = self.index_of(file) else {
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
        let pixels = level.decode_tile(bytes).map_err(ProvideError::Tiff)?;
        let tile = Arc::new(LandTile::decoded(level, pixels));
        let _ = write_atomically(&self.tile_path(file, level_no, tile_no), bytes);
        self.remember((file, level_no, tile_no), &tile);
        Ok(())
    }

    /// The parsed index of `file`, or the fetch for what it still lacks. Index bytes that no longer parse
    /// (a damaged cache file) are dropped, memory and disk, and asked for again.
    fn index_of(&self, file: WorldCoverFile) -> Result<Arc<TiffIndex>, Fetch> {
        self.load_pieces_from_disk(file);
        let mut memory = lock(&self.memory);
        memory.clock += 1;
        let now = memory.clock;
        let state = memory.files.entry(file).or_default();
        state.used = now;
        if let Some(index) = &state.index {
            return Ok(Arc::clone(index));
        }
        match state.parse() {
            Ok(Parse::Done(index)) => {
                let index = Arc::new(index);
                state.index = Some(Arc::clone(&index));
                Ok(index)
            }
            Ok(Parse::Need(range)) => Err(Fetch {
                file,
                range,
                wanted: Wanted::Index,
            }),
            Err(_) => {
                *state = FileState {
                    from_disk: true,
                    used: now,
                    ..FileState::default()
                };
                drop(memory);
                let _ = fs::remove_dir_all(self.file_dir(file));
                Err(Fetch {
                    file,
                    range: ByteRange {
                        offset: 0,
                        len: crate::tiff::PREFIX_LEN,
                    },
                    wanted: Wanted::Index,
                })
            }
        }
    }

    /// Reads the index pieces the disk holds for `file`, once per time the file is in memory.
    fn load_pieces_from_disk(&self, file: WorldCoverFile) {
        {
            let mut memory = lock(&self.memory);
            let state = memory.files.entry(file).or_default();
            if state.from_disk {
                return;
            }
            state.from_disk = true;
        }
        if self.absent_path(file).exists() {
            lock(&self.memory).absent.insert(file);
        }
        let Ok(entries) = fs::read_dir(self.file_dir(file)) else {
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
        let state = memory.files.entry(file).or_default();
        if state.pieces.is_empty() {
            state.pieces = pieces;
            state.file_len = file_len;
        }
    }

    /// The decoded tile, from memory or disk, or the fetch for it. `Ok(None)` for a tile the file leaves out.
    fn tile_of(
        &self,
        file: WorldCoverFile,
        level_no: usize,
        level: &Level,
        tile_no: usize,
    ) -> Result<Option<Arc<LandTile>>, Fetch> {
        let key = (file, level_no as u8, tile_no as u32);
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
                log::debug!(
                    "land cover {file}: tile {tile_no} of level {level_no} is unusable: {e}"
                );
                return Ok(None);
            }
        };
        if let Ok(bytes) = fs::read(self.tile_path(file, key.1, key.2))
            && bytes.len() as u64 == range.len
            && let Ok(pixels) = level.decode_tile(&bytes)
        {
            let tile = Arc::new(LandTile::decoded(level, pixels));
            self.remember(key, &tile);
            return Ok(Some(tile));
        }
        Err(Fetch {
            file,
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
        file: WorldCoverFile,
        level_no: usize,
        level: &Level,
        tile_no: usize,
    ) -> Result<(), Fetch> {
        let key = (file, level_no as u8, tile_no as u32);
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
        let on_disk = fs::metadata(self.tile_path(file, key.1, key.2))
            .is_ok_and(|m| m.is_file() && m.len() == range.len);
        if on_disk {
            return Ok(());
        }
        Err(Fetch {
            file,
            range,
            wanted: Wanted::Tile {
                level: key.1,
                tile: key.2,
            },
        })
    }

    fn in_memory(&self, key: TileKey) -> Option<Arc<LandTile>> {
        let mut memory = lock(&self.memory);
        memory.clock += 1;
        let now = memory.clock;
        let entry = memory.tiles.get_mut(&key)?;
        entry.1 = now;
        Some(Arc::clone(&entry.0))
    }

    fn remember(&self, key: TileKey, tile: &Arc<LandTile>) {
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

    /// Drops the least recently used files' index bytes from memory beyond [`MAX_FILES_IN_MEMORY`]; they come
    /// back from disk when asked for.
    fn trim_files(&self) {
        let mut memory = lock(&self.memory);
        while memory.files.len() > MAX_FILES_IN_MEMORY {
            let oldest = memory
                .files
                .iter()
                .min_by_key(|(_, state)| state.used)
                .map(|(file, _)| *file);
            match oldest {
                Some(oldest) => memory.files.remove(&oldest),
                None => break,
            };
        }
    }

    /// How many decoded tiles are in memory.
    pub fn memory_tiles(&self) -> usize {
        lock(&self.memory).tiles.len()
    }
}

/// Ground size of a pixel of the finest WorldCover level, north to south, in metres (the sphere's 3 degrees over
/// 36000 rows). The coarser levels are twice, four times, ... that.
pub const LEVEL0_PIXEL_M: f64 = 9.28;
/// The most overview levels above the finest that a file has (WorldCover's go to 594 m).
const MAX_LEVEL: i32 = 6;
/// The warp of [`LandCoverField::class_at`]: a coarse and a fine wave of offsets, each in pixels of the level
/// read. The wavelength is how far apart the offset's lattice is, the amplitude its largest size. Both
/// directions have their own noise. Together the offset is never over [`LAND_WARP_MAX_PX`] pixels, so the data
/// stays where it is to about a pixel, but a pixel's straight edges are bent into waves and its corners round.
pub const LAND_WARP_COARSE_WAVELENGTH_PX: f64 = 4.0;
pub const LAND_WARP_COARSE_PX: f64 = 0.8;
pub const LAND_WARP_FINE_WAVELENGTH_PX: f64 = 1.25;
pub const LAND_WARP_FINE_PX: f64 = 0.35;
/// The largest offset of the warp, in pixels.
pub const LAND_WARP_MAX_PX: f64 = LAND_WARP_COARSE_PX + LAND_WARP_FINE_PX;
/// Salts (fixed, so the warp is the same in every world) of the four noises.
const WARP_SALTS: [u64; 4] = [
    0x1A4D_0001_C0A7_5E01,
    0x1A4D_0002_C0A7_5E02,
    0x1A4D_0003_F1BE_5E03,
    0x1A4D_0004_F1BE_5E04,
];

/// Land cover at world blocks: a [`LandCoverStore`] placed on the globe by a [`Projection`], as
/// [`crate::ElevationField`] places the elevation tiles.
///
/// The classes are a grid of square pixels (about 9 m, 18 m, ... by level); read by nearest pixel they would
/// show as squares and staircases in the world, near and far. So every read goes through
/// [`LandCoverField::class_at`], which does two things. It rounds the pixels' corners
/// ([`LandCoverStore::lookup_smooth`]: a staircase of one-pixel steps becomes a diagonal), and it bends the grid
/// a little first: the point it asks the store about is moved by a smooth deterministic noise of up to
/// [`LAND_WARP_MAX_PX`] pixels, so straight pixel edges become wavy lines. Every reader (the ground, the far
/// view, biomes, vegetation) sees the same map.
#[derive(Clone)]
pub struct LandCoverField {
    store: Arc<LandCoverStore>,
    projection: Projection,
    /// Metres from north to south of a pixel of the finest level, for the size of the warp.
    level0_pixel_m: f64,
}

impl LandCoverField {
    pub fn new(store: Arc<LandCoverStore>, projection: Projection) -> Self {
        Self {
            store,
            projection,
            level0_pixel_m: LEVEL0_PIXEL_M,
        }
    }

    /// The same field with the finest level's pixels `metres` tall on the ground, for data that is not
    /// WorldCover's (a test's miniature files); the warp scales with it.
    pub fn with_level0_pixel_metres(mut self, metres: f64) -> Self {
        if metres.is_finite() && metres > 0.0 {
            self.level0_pixel_m = metres;
        }
        self
    }

    pub fn store(&self) -> &Arc<LandCoverStore> {
        &self.store
    }

    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    /// The size in blocks `(east-west, north-south)` of a pixel of the level a read at `metres_per_sample` takes
    /// (the coarsest whose pixel is no taller than that, like [`level_for`]). East-west pixels are narrower by
    /// the cosine of the latitude.
    fn pixel_blocks(&self, metres_per_sample: f64) -> (f64, f64) {
        let level = if metres_per_sample.is_finite() && metres_per_sample > self.level0_pixel_m {
            ((metres_per_sample / self.level0_pixel_m).log2().floor() as i32).clamp(0, MAX_LEVEL)
        } else {
            0
        };
        let tall =
            self.level0_pixel_m * f64::from(1_u32 << level) / self.projection.metres_per_block();
        let cos = self
            .projection
            .reference_latitude()
            .to_radians()
            .cos()
            .max(0.01);
        (tall * cos, tall)
    }

    /// The point `(x, z)` bent by the warp for a read at `metres_per_sample`: the offset field is noise over the
    /// world's own block coordinates, so it is smooth from block to block, and it is scaled to the pixels of the
    /// level that read takes.
    fn warped(&self, x: f64, z: f64, metres_per_sample: f64) -> (f64, f64) {
        let (px, pz) = self.pixel_blocks(metres_per_sample);
        let octave = |wavelength: f64, amplitude: f64, salts: [u64; 2]| {
            let (u, v) = (x / (wavelength * px), z / (wavelength * pz));
            (
                amplitude * px * crate::warp::value_noise(u, v, salts[0]),
                amplitude * pz * crate::warp::value_noise(u, v, salts[1]),
            )
        };
        let (cx, cz) = octave(
            LAND_WARP_COARSE_WAVELENGTH_PX,
            LAND_WARP_COARSE_PX,
            [WARP_SALTS[0], WARP_SALTS[1]],
        );
        let (fx, fz) = octave(
            LAND_WARP_FINE_WAVELENGTH_PX,
            LAND_WARP_FINE_PX,
            [WARP_SALTS[2], WARP_SALTS[3]],
        );
        (x + cx + fx, z + cz + fz)
    }

    /// The class at the world point `(x, z)` (fractional blocks; add 0.5 for a block's centre), read at the
    /// overview level that suits `metres_per_sample` of ground, after the warp and with the pixels' corners
    /// rounded (see [`LandCoverField`]). The store's [`LandCoverStore::lookup`] is the same read without either.
    pub fn class_at(&self, x: f64, z: f64, metres_per_sample: f64) -> Lookup {
        let (x, z) = self.warped(x, z, metres_per_sample);
        let at = self.projection.to_lat_lon(x, z);
        self.store.lookup_smooth(at.lat, at.lon, metres_per_sample)
    }

    /// The class at the centre of block `(x, z)`, read as finely as the blocks are big.
    pub fn class_at_block(&self, x: i32, z: i32) -> Lookup {
        self.class_at(
            f64::from(x) + 0.5,
            f64::from(z) + 0.5,
            self.projection.metres_per_block(),
        )
    }

    /// The classes at the centres of the blocks `x0..x0 + width` by `z0..z0 + depth`, `stride` blocks apart,
    /// row by row, each what [`LandCoverField::class_at`] gives. Where the source has no file the class is
    /// [`LandClass::NoData`]. `Err` with the first fetch that is missing when any tile is not held (nothing
    /// is fetched).
    pub fn class_grid(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        stride: i32,
        metres_per_sample: f64,
    ) -> Result<Vec<LandClass>, Fetch> {
        let mut out = Vec::with_capacity(width.saturating_mul(depth).min(1 << 20));
        for j in 0..depth {
            for i in 0..width {
                let x = f64::from(x0) + (i as f64) * f64::from(stride) + 0.5;
                let z = f64::from(z0) + (j as f64) * f64::from(stride) + 0.5;
                match self.class_at(x, z, metres_per_sample) {
                    Lookup::Class(class) => out.push(class),
                    Lookup::NoFile => out.push(LandClass::NoData),
                    Lookup::Need(fetch) => return Err(fetch),
                }
            }
        }
        Ok(out)
    }

    /// The fetches still missing for the block rectangle `x0..=x1` by `z0..=z1` (corners in either order) at
    /// `metres_per_sample`, for prefetching. The area is the lat/lon box around the rectangle's corners and
    /// edge midpoints, which for a local projection holds the rectangle.
    pub fn needs(&self, x0: i32, z0: i32, x1: i32, z1: i32, metres_per_sample: f64) -> Vec<Fetch> {
        // A read inside the rectangle may be bent out of it by up to the warp.
        let (px, pz) = self.pixel_blocks(metres_per_sample);
        // plus the one pixel past the point that the rounding of corners reads.
        let reach = LAND_WARP_MAX_PX + 1.0;
        let (gx, gz) = (reach * px + 1.0, reach * pz + 1.0);
        let (xa, xb) = (f64::from(x0.min(x1)) - gx, f64::from(x0.max(x1)) + 1.0 + gx);
        let (za, zb) = (f64::from(z0.min(z1)) - gz, f64::from(z0.max(z1)) + 1.0 + gz);
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
    use std::collections::HashMap;
    use crate::testing::{
        SyntheticTiff, TempDir, TiffSpec, encode_tiff, encode_tiff_u16, serve_range,
    };

    const CODES: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 0];

    /// The code of a pixel of the miniature files: bands that differ by level.
    fn code(level: usize, x: u32, y: u32) -> u8 {
        CODES[((x / 37 + y / 41) as usize + level * 5) % CODES.len()]
    }

    /// A WorldCover-like file in miniature: 600 by 600 pixels in 256-pixel tiles, then 300 and 150.
    fn miniature() -> SyntheticTiff {
        encode_tiff(&TiffSpec::new(600, 600, 256, 2), &code)
    }

    const SIZES: [u32; 3] = [600, 300, 150];

    /// The raw code the miniature file holds at a place.
    fn raw_code(lat: f64, lon: f64, level: usize) -> u8 {
        let file = WorldCoverFile::containing(lat, lon).unwrap();
        let size = SIZES[level];
        let (x, y) = file.pixel(lat, lon, size, size);
        code(level, x, y)
    }

    /// What the miniature file says at a place, worked out from the pixel arithmetic alone: the code of the
    /// pixel in its 256-pixel tile (padding is no data), with the tile's built-up land given back to nature.
    fn expected(lat: f64, lon: f64, level: usize) -> LandClass {
        let file = WorldCoverFile::containing(lat, lon).unwrap();
        let size = SIZES[level];
        let (x, y) = file.pixel(lat, lon, size, size);
        let (tx, ty) = (x / 256 * 256, y / 256 * 256);
        let mut tile: Vec<u8> = (0..256 * 256)
            .map(|i| {
                let (px, py) = (tx + i % 256, ty + i / 256);
                if px < size && py < size {
                    code(level, px, py)
                } else {
                    0
                }
            })
            .collect();
        crate::landcover::unbuild(&mut tile, 256, 256);
        LandClass::from_code(tile[((y - ty) * 256 + (x - tx)) as usize]).unwrap()
    }

    #[test]
    fn no_lookup_answers_built_up_land_at_any_level_or_place() {
        let dir = TempDir::new();
        let store = LandCoverStore::new(dir.path(), 16);
        let server = Server::with(&[HERE]);
        let (mut raw_built, mut seen) = (0, 0);
        for (level, metres) in [10.0, 1200.0, 2300.0].into_iter().enumerate() {
            for i in 0..30 {
                for j in 0..30 {
                    let (lat, lon) = (27.02 + f64::from(i) * 0.033, 84.02 + f64::from(j) * 0.033);
                    raw_built +=
                        usize::from(raw_code(lat, lon, level) == LandClass::BuiltUp.code());
                    let got = server.resolve(&store, lat, lon, metres);
                    assert_ne!(got, Lookup::Class(LandClass::BuiltUp), "{lat}, {lon}");
                    assert_eq!(
                        got,
                        Lookup::Class(expected(lat, lon, level)),
                        "{lat}, {lon}"
                    );
                    seen += 1;
                }
            }
        }
        // The file did hold built-up pixels where the lookups went.
        assert!(raw_built > 50, "{raw_built} of {seen}");
    }

    /// A pretend server: the files it has, and how often it was asked.
    struct Server {
        files: HashMap<WorldCoverFile, Vec<u8>>,
        asked: Mutex<Vec<Fetch>>,
    }

    impl Server {
        fn with(files: &[(f64, f64)]) -> Self {
            let bytes = miniature().bytes;
            Self {
                files: files
                    .iter()
                    .map(|&(lat, lon)| {
                        (WorldCoverFile::containing(lat, lon).unwrap(), bytes.clone())
                    })
                    .collect(),
                asked: Mutex::new(Vec::new()),
            }
        }

        fn asked(&self) -> Vec<Fetch> {
            lock(&self.asked).clone()
        }

        /// Answers lookups by fetching what is missing, until a class comes back.
        fn resolve(&self, store: &LandCoverStore, lat: f64, lon: f64, metres: f64) -> Lookup {
            for _ in 0..16 {
                match store.lookup(lat, lon, metres) {
                    Lookup::Need(fetch) => {
                        lock(&self.asked).push(fetch);
                        match self.files.get(&fetch.file) {
                            None => store.mark_absent(fetch.file),
                            Some(file) => {
                                let bytes = serve_range(file, fetch.range);
                                store
                                    .provide(&fetch, bytes)
                                    .expect("the store takes what the server sent");
                            }
                        }
                    }
                    done => return done,
                }
            }
            panic!("a lookup that never finishes");
        }
    }

    const HERE: (f64, f64) = (27.5, 84.5);

    #[test]
    fn a_lookup_asks_for_the_index_then_the_tile_then_answers() {
        let dir = TempDir::new();
        let store = LandCoverStore::new(dir.path(), 8);
        let Lookup::Need(first) = store.lookup(HERE.0, HERE.1, 100.0) else {
            panic!("nothing is held yet")
        };
        assert_eq!(first.wanted, Wanted::Index);
        assert_eq!(first.file.tile_name(), "N27E084");
        assert_eq!(
            first.range,
            ByteRange {
                offset: 0,
                len: crate::tiff::PREFIX_LEN
            }
        );
        // The same question gives the same answer until something is provided.
        assert_eq!(store.lookup(HERE.0, HERE.1, 100.0), Lookup::Need(first));

        let file = miniature().bytes;
        store
            .provide(&first, serve_range(&file, first.range))
            .unwrap();
        let Lookup::Need(second) = store.lookup(HERE.0, HERE.1, 100.0) else {
            panic!("the tile is not held")
        };
        // Pixel (100, 500) of 600 is in column 0 and row 1 of the 256-pixel tiles, three to a row: tile 3.
        assert_eq!(second.wanted, Wanted::Tile { level: 0, tile: 3 });
        store
            .provide(&second, serve_range(&file, second.range))
            .unwrap();
        assert_eq!(
            store.lookup(HERE.0, HERE.1, 100.0),
            Lookup::Class(expected(HERE.0, HERE.1, 0))
        );
    }

    #[test]
    fn the_overview_level_follows_the_requested_resolution() {
        let dir = TempDir::new();
        let store = LandCoverStore::new(dir.path(), 8);
        let server = Server::with(&[HERE]);
        // Levels of 600, 300 and 150 rows are about 557, 1113 and 2227 metres a pixel.
        for (metres, level) in [
            (10.0, 0),
            (556.0, 0),
            (560.0, 0),
            (1000.0, 0),
            (1200.0, 1),
            (2000.0, 1),
            (2300.0, 2),
            (1e6, 2),
        ] {
            for lon in [84.1, 84.5, 85.9] {
                let got = server.resolve(&store, HERE.0, lon, metres);
                assert_eq!(
                    got,
                    Lookup::Class(expected(HERE.0, lon, level)),
                    "{metres} m at {lon}"
                );
            }
        }
    }

    #[test]
    fn a_tile_is_fetched_once_and_a_new_store_finds_it_on_disk() {
        let dir = TempDir::new();
        let server = Server::with(&[HERE]);
        let store = LandCoverStore::new(dir.path(), 8);
        let a = server.resolve(&store, HERE.0, HERE.1, 100.0);
        let made = server.asked().len();
        assert_eq!(made, 2, "the index and one tile");
        // Another place in the same tile, and the same place again: nothing more is fetched.
        server.resolve(&store, HERE.0 + 0.05, HERE.1 + 0.05, 100.0);
        server.resolve(&store, HERE.0, HERE.1, 100.0);
        assert_eq!(server.asked().len(), made);
        // A fresh store over the same folder answers without the server.
        let fresh = LandCoverStore::new(dir.path(), 8);
        assert_eq!(fresh.lookup(HERE.0, HERE.1, 100.0), a);
        assert_eq!(server.asked().len(), made);
    }

    #[test]
    fn the_cache_folder_is_named_by_the_rules_version() {
        let dir = TempDir::new();
        let server = Server::with(&[HERE]);
        let store = LandCoverStore::new(dir.path(), 8);
        server.resolve(&store, HERE.0, HERE.1, 100.0);
        let key = cache_key();
        assert!(dir.path().join(&key).join("N27E084").is_dir(), "{key}");
        assert!(
            store
                .tile_path(WorldCoverFile::containing(HERE.0, HERE.1).unwrap(), 0, 3)
                .is_file()
        );
        // A store of another rules version would use another folder and so start empty: simulated by
        // looking in a sibling name.
        let other = LandCoverStore {
            dir: dir.path().join("worldcover-2021-v200-rules0"),
            ..LandCoverStore::new(dir.path(), 8)
        };
        assert!(matches!(
            other.lookup(HERE.0, HERE.1, 100.0),
            Lookup::Need(_)
        ));
    }

    #[test]
    fn the_memory_cache_is_bounded_and_evicted_tiles_come_back_from_disk() {
        let dir = TempDir::new();
        let server = Server::with(&[HERE]);
        let store = LandCoverStore::new(dir.path(), 2);
        // Three different tiles of level 0: columns 0, 1, 2 of the top row.
        let places = [(29.9, 84.1), (29.9, 85.5), (29.9, 86.9)];
        for (lat, lon) in places {
            server.resolve(&store, lat, lon, 100.0);
        }
        assert_eq!(store.memory_tiles(), 2);
        let made = server.asked().len();
        for (lat, lon) in places {
            assert_eq!(
                store.lookup(lat, lon, 100.0),
                Lookup::Class(expected(lat, lon, 0))
            );
        }
        assert_eq!(
            server.asked().len(),
            made,
            "evicted tiles are read back from disk"
        );
        assert_eq!(store.memory_tiles(), 2);
        // With no memory at all every lookup still works.
        let none = LandCoverStore::new(dir.path(), 0);
        assert_eq!(
            none.lookup(29.9, 84.1, 100.0),
            Lookup::Class(expected(29.9, 84.1, 0))
        );
        assert_eq!(none.memory_tiles(), 0);
    }

    #[test]
    fn refused_bytes_are_never_stored() {
        let dir = TempDir::new();
        let store = LandCoverStore::new(dir.path(), 8);
        let file = miniature().bytes;
        let Lookup::Need(index) = store.lookup(HERE.0, HERE.1, 100.0) else {
            panic!()
        };
        assert_eq!(store.provide(&index, &[]), Err(ProvideError::Empty));
        assert_eq!(
            store.provide(&index, &vec![0u8; index.range.len as usize + 1]),
            Err(ProvideError::WrongLength)
        );
        assert_eq!(
            store.provide(&index, b"GIF89a, not a TIFF at all"),
            Err(ProvideError::Tiff(TiffError::NotTiff))
        );
        // A BigTIFF with 16-bit samples and a predictor (a climate file) is not land cover.
        let climate = encode_tiff_u16(
            &TiffSpec::new(600, 600, 256, 2)
                .bigtiff()
                .sixteen_bit()
                .predictor(2),
            &|_, x, y| (x * 100 + y) as u16,
        );
        assert!(matches!(
            store.provide(&index, serve_range(&climate.bytes, index.range)),
            Err(ProvideError::Tiff(TiffError::Unsupported(_)))
        ));
        // So is an 8-bit file with a predictor, or a 16-bit classic one.
        for spec in [
            TiffSpec::new(600, 600, 256, 2).predictor(2),
            TiffSpec::new(600, 600, 256, 2).sixteen_bit(),
        ] {
            let other = encode_tiff(&spec, &code);
            assert!(matches!(
                store.provide(&index, serve_range(&other.bytes, index.range)),
                Err(ProvideError::Tiff(TiffError::Unsupported(_)))
            ));
        }
        assert!(!store.file_dir(index.file).join("index-0.bin").exists());
        // The store still asks for the same prefix.
        assert_eq!(store.lookup(HERE.0, HERE.1, 100.0), Lookup::Need(index));
        // Index pieces past the cap.
        let huge = Fetch {
            range: ByteRange {
                offset: 0,
                len: (MAX_INDEX_BYTES + 1) as u64,
            },
            ..index
        };
        assert_eq!(
            store.provide(&huge, &vec![0u8; MAX_INDEX_BYTES + 1]),
            Err(ProvideError::TooLarge)
        );

        // Tiles: before the index, with a range the index does not give, with the wrong length, damaged.
        let tile = Fetch {
            wanted: Wanted::Tile { level: 0, tile: 0 },
            range: ByteRange {
                offset: 1000,
                len: 10,
            },
            ..index
        };
        assert_eq!(store.provide(&tile, &[1; 10]), Err(ProvideError::NoIndex));
        store
            .provide(&index, serve_range(&file, index.range))
            .unwrap();
        assert_eq!(store.provide(&tile, &[1; 10]), Err(ProvideError::NotWanted));
        let Lookup::Need(real) = store.lookup(HERE.0, HERE.1, 100.0) else {
            panic!()
        };
        let good = serve_range(&file, real.range);
        assert_eq!(
            store.provide(&real, &good[..good.len() - 1]),
            Err(ProvideError::WrongLength)
        );
        let mut damaged = good.to_vec();
        let mid = damaged.len() / 2;
        damaged[mid] ^= 0xff;
        damaged[0] ^= 0xff;
        assert!(matches!(
            store.provide(&real, &damaged),
            Err(ProvideError::Tiff(_))
        ));
        for wrong in [
            Wanted::Tile { level: 9, tile: 0 },
            Wanted::Tile { level: 0, tile: 99 },
            Wanted::Tile { level: 1, tile: 0 },
        ] {
            assert_eq!(
                store.provide(
                    &Fetch {
                        wanted: wrong,
                        ..real
                    },
                    good
                ),
                Err(ProvideError::NotWanted)
            );
        }
        assert!(!store.tile_path(real.file, 0, 3).exists());
        assert_eq!(store.memory_tiles(), 0);
        // The honest bytes still go in.
        store.provide(&real, good).unwrap();
        assert_eq!(store.memory_tiles(), 1);
    }

    #[test]
    fn a_damaged_cache_is_fetched_again() {
        let dir = TempDir::new();
        let server = Server::with(&[HERE]);
        let store = LandCoverStore::new(dir.path(), 8);
        let want = server.resolve(&store, HERE.0, HERE.1, 100.0);
        let file = WorldCoverFile::containing(HERE.0, HERE.1).unwrap();
        // A damaged tile file: a fresh store asks for the tile, not the index.
        fs::write(store.tile_path(file, 0, 3), b"damaged").unwrap();
        let fresh = LandCoverStore::new(dir.path(), 8);
        assert!(matches!(
            fresh.lookup(HERE.0, HERE.1, 100.0),
            Lookup::Need(Fetch {
                wanted: Wanted::Tile { .. },
                ..
            })
        ));
        assert_eq!(server.resolve(&fresh, HERE.0, HERE.1, 100.0), want);
        // A damaged index file: the whole file's cache is dropped and the prefix asked for again.
        fs::write(
            fresh.file_dir(file).join("index-0.bin"),
            b"II*\0\x08\0\0\0\xff\xff",
        )
        .unwrap();
        let again = LandCoverStore::new(dir.path(), 8);
        let Lookup::Need(fetch) = again.lookup(HERE.0, HERE.1, 100.0) else {
            panic!()
        };
        assert_eq!(fetch.wanted, Wanted::Index);
        assert_eq!(fetch.range.offset, 0);
        assert_eq!(server.resolve(&again, HERE.0, HERE.1, 100.0), want);
    }

    #[test]
    fn absent_files_and_impossible_places_say_so() {
        let dir = TempDir::new();
        let store = LandCoverStore::new(dir.path(), 8);
        let server = Server::with(&[HERE]);
        // The sea: the server has no such file.
        assert_eq!(server.resolve(&store, 0.5, -29.0, 100.0), Lookup::NoFile);
        assert_eq!(store.lookup(0.5, -29.0, 100.0), Lookup::NoFile);
        assert_eq!(server.asked().len(), 1, "one try, then it is remembered");
        for (lat, lon) in [
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
            (91.0, 0.0),
            (0.0, f64::INFINITY),
        ] {
            assert_eq!(store.lookup(lat, lon, 100.0), Lookup::NoFile);
        }
        // Places with files still work.
        assert!(matches!(
            server.resolve(&store, HERE.0, HERE.1, 100.0),
            Lookup::Class(_)
        ));
    }

    #[test]
    fn a_tile_the_file_leaves_out_and_values_it_does_not_define_read_as_no_data() {
        let dir = TempDir::new();
        let spec = TiffSpec::new(600, 600, 256, 0).editing_tiles(|_, offsets, counts| {
            offsets[3] = 0;
            counts[3] = 0;
        });
        let file = encode_tiff(&spec, &|_, x, _| if x < 256 { 77 } else { 30 });
        let server = Server {
            files: HashMap::from([(
                WorldCoverFile::containing(HERE.0, HERE.1).unwrap(),
                file.bytes,
            )]),
            asked: Mutex::new(Vec::new()),
        };
        let store = LandCoverStore::new(dir.path(), 8);
        // Tile 3 is the second row's first column: (x 0-255, y 256-511): omitted. Pixel (100, 500).
        assert_eq!(
            server.resolve(&store, HERE.0, HERE.1, 10.0),
            Lookup::Class(LandClass::NoData)
        );
        // Tile 0: value 77, which WorldCover does not define.
        assert_eq!(
            server.resolve(&store, 29.9, 84.1, 10.0),
            Lookup::Class(LandClass::NoData)
        );
        // Tile 1: grass.
        assert_eq!(
            server.resolve(&store, 29.9, 85.5, 10.0),
            Lookup::Class(LandClass::Grass)
        );
    }

    #[test]
    fn places_across_the_date_line_and_the_poles_have_files() {
        let dir = TempDir::new();
        let store = LandCoverStore::new(dir.path(), 8);
        let server = Server::with(&[(0.5, 180.0), (0.5, 179.0), (89.0, 10.0), (-89.0, 10.0)]);
        for (lat, lon) in [
            (0.5, 180.0),
            (0.5, -179.9),
            (0.5, 179.9),
            (90.0, 10.0),
            (89.9, 10.0),
            (-90.0, 10.0),
            (-89.9, 10.0),
        ] {
            let got = server.resolve(&store, lat, lon, 100.0);
            assert_eq!(got, Lookup::Class(expected(lat, lon, 0)), "{lat}, {lon}");
        }
        // 180 east is 180 west: the same file and the same pixel.
        assert_eq!(
            server.resolve(&store, 0.5, 180.0, 100.0),
            server.resolve(&store, 0.5, -180.0, 100.0)
        );
    }

    #[test]
    fn many_threads_asking_get_the_same_answers() {
        let dir = TempDir::new();
        let server = Arc::new(Server::with(&[HERE]));
        let store = Arc::new(LandCoverStore::new(dir.path(), 2));
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let (server, store) = (Arc::clone(&server), Arc::clone(&store));
                std::thread::spawn(move || {
                    let mut out = Vec::new();
                    for i in 0..30 {
                        let lat = 27.1 + f64::from((i * 7 + t) % 29) * 0.1;
                        let lon = 84.1 + f64::from((i * 11 + t) % 28) * 0.1;
                        out.push((lat, lon, server.resolve(&store, lat, lon, 100.0)));
                    }
                    out
                })
            })
            .collect();
        for thread in threads {
            for (lat, lon, got) in thread.join().unwrap() {
                assert_eq!(got, Lookup::Class(expected(lat, lon, 0)), "{lat}, {lon}");
            }
        }
        assert!(store.memory_tiles() <= 2);
    }

    #[test]
    fn needs_lists_the_index_then_exactly_the_tiles_an_area_touches() {
        let dir = TempDir::new();
        let store = LandCoverStore::new(dir.path(), 16);
        let file = miniature().bytes;
        // Across the first and second file (84 to 87 and 87 to 90 east).
        let area = GeoBox {
            south: 27.5,
            north: 28.5,
            west: 86.0,
            east: 88.0,
        };
        let round = store.needs(area, 100.0);
        assert_eq!(round.len(), 2);
        assert!(round.iter().all(|f| f.wanted == Wanted::Index));
        let first = WorldCoverFile::containing(27.5, 86.5).unwrap();
        let second = WorldCoverFile::containing(27.5, 88.5).unwrap();
        assert_eq!(
            round.iter().map(|f| f.file).collect::<HashSet<_>>(),
            HashSet::from([first, second])
        );
        // The second file does not exist, the first does.
        store.mark_absent(second);
        let round = store.needs(area, 100.0);
        assert_eq!(round.len(), 1);
        store
            .provide(&round[0], serve_range(&file, round[0].range))
            .unwrap();
        // Level 0 tiles: x from 86.0 to 87.0 is pixels 400..600 (tile column 1 and 2), y from 28.5 to 27.5 is
        // pixels 300..500 (tile row 1): tiles 4 and 5, plus the columns the cell's end reaches.
        let round = store.needs(area, 100.0);
        let tiles: Vec<u32> = round
            .iter()
            .map(|f| match f.wanted {
                Wanted::Tile { level: 0, tile } => tile,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(tiles, [4, 5]);
        for fetch in &round {
            store
                .provide(fetch, serve_range(&file, fetch.range))
                .unwrap();
        }
        assert!(store.needs(area, 100.0).is_empty());
        // Coarser data is another level, with its own tiles.
        let coarse = store.needs(area, 1500.0);
        // Pixels 150 to 300 across and 150 to 250 down of a 300 by 300 level: columns 0 and 1 of the top row.
        let wanted: Vec<_> = coarse.iter().map(|f| f.wanted).collect();
        assert_eq!(
            wanted,
            [
                Wanted::Tile { level: 1, tile: 0 },
                Wanted::Tile { level: 1, tile: 1 }
            ]
        );
        // The area's tiles are on disk: a new store has nothing to fetch either.
        let fresh = LandCoverStore::new(dir.path(), 16);
        fresh.mark_absent(second);
        assert!(fresh.needs(area, 100.0).is_empty());
    }

    #[test]
    fn needs_copes_with_odd_boxes() {
        let dir = TempDir::new();
        let store = LandCoverStore::new(dir.path(), 4);
        let nan = GeoBox {
            south: f64::NAN,
            north: 1.0,
            west: 0.0,
            east: 1.0,
        };
        assert!(store.needs(nan, 100.0).is_empty());
        // A box across the date line is two files, with corners in either order.
        let across = GeoBox {
            south: 0.5,
            north: 0.6,
            west: 179.0,
            east: 181.0,
        };
        assert_eq!(store.needs(across, 100.0).len(), 2);
        let wrapped = GeoBox {
            west: 179.0,
            east: -179.0,
            ..across
        };
        assert_eq!(store.needs(wrapped, 100.0).len(), 2);
        // The whole world is the whole 120 by 60 grid, and no more.
        let world = GeoBox {
            south: -100.0,
            north: 100.0,
            west: -500.0,
            east: 500.0,
        };
        assert_eq!(store.needs(world, 100.0).len(), 120 * 60);
        // A box wholly past a pole has nothing.
        let past = GeoBox {
            south: 95.0,
            north: 99.0,
            west: 0.0,
            east: 1.0,
        };
        assert!(store.needs(past, 100.0).is_empty());
    }

    fn field(dir: &TempDir) -> LandCoverField {
        // The origin is in the middle of the first miniature file.
        let projection = Projection::local(28.5, 85.5, 10.0).unwrap();
        LandCoverField::new(Arc::new(LandCoverStore::new(dir.path(), 8)), projection)
    }

    #[test]
    fn the_field_places_blocks_on_the_globe() {
        let dir = TempDir::new();
        let field = field(&dir);
        let server = Server::with(&[(28.5, 85.5)]);
        let origin = field.projection().to_lat_lon(0.5, 0.5);
        let Lookup::Need(_) = field.class_at(0.5, 0.5, 20.0) else {
            panic!("nothing is held")
        };
        let want = server.resolve(field.store(), origin.lat, origin.lon, 20.0);
        assert_eq!(field.class_at(0.5, 0.5, 20.0), want);
        assert_eq!(
            field.class_at_block(0, 0),
            want,
            "blocks are 10 m, the finest level is coarser"
        );
        // 100 blocks east is a kilometre: another pixel, with the same arithmetic as the store's.
        let there = field.projection().to_lat_lon(100.5, 0.5);
        let got = server.resolve(field.store(), there.lat, there.lon, 20.0);
        assert_eq!(field.class_at(100.5, 0.5, 20.0), got);
    }

    #[test]
    fn a_grid_is_what_single_lookups_give_or_the_first_missing_fetch() {
        let dir = TempDir::new();
        let field = field(&dir);
        let server = Server::with(&[(28.5, 85.5)]);
        let Err(first) = field.class_grid(-8, -8, 16, 16, 1, 20.0) else {
            panic!("nothing is held")
        };
        assert_eq!(first.wanted, Wanted::Index);
        // Fetch until the grid is whole.
        let mut grid = None;
        for _ in 0..20 {
            match field.class_grid(-8, -8, 16, 16, 1, 20.0) {
                Ok(g) => {
                    grid = Some(g);
                    break;
                }
                Err(fetch) => {
                    let file = &server.files[&fetch.file];
                    field
                        .store()
                        .provide(&fetch, serve_range(file, fetch.range))
                        .unwrap();
                }
            }
        }
        let grid = grid.expect("the grid fills in");
        assert_eq!(grid.len(), 256);
        for j in 0..16 {
            for i in 0..16 {
                let want = field.class_at(f64::from(i - 8) + 0.5, f64::from(j - 8) + 0.5, 20.0);
                assert_eq!(Lookup::Class(grid[(j * 16 + i) as usize]), want);
            }
        }
        // Strided, and empty rectangles.
        let strided = field.class_grid(-80, -80, 4, 4, 40, 20.0).unwrap();
        assert_eq!(strided.len(), 16);
        assert_eq!(field.class_grid(0, 0, 0, 5, 1, 20.0), Ok(Vec::new()));
        // A file the source lacks reads as no data in a grid.
        let far = LandCoverField::new(
            Arc::clone(field.store()),
            Projection::local(0.5, -29.0, 10.0).unwrap(),
        );
        let probe = far.class_at(0.5, 0.5, 20.0);
        let Lookup::Need(need) = probe else { panic!() };
        far.store().mark_absent(need.file);
        assert_eq!(
            far.class_grid(0, 0, 3, 3, 1, 20.0),
            Ok(vec![LandClass::NoData; 9])
        );
    }

    #[test]
    fn the_field_lists_what_a_rectangle_needs() {
        let dir = TempDir::new();
        let field = field(&dir);
        let server = Server::with(&[(28.5, 85.5)]);
        let first = field.needs(-100, -100, 100, 100, 20.0);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].wanted, Wanted::Index);
        for fetch in &first {
            let file = &server.files[&fetch.file];
            field
                .store()
                .provide(fetch, serve_range(file, fetch.range))
                .unwrap();
        }
        // 200 blocks is 2 km, about 3 pixels of the miniature file: inside one or two tiles.
        let second = field.needs(-100, -100, 100, 100, 20.0);
        assert!(!second.is_empty() && second.len() <= 4, "{second:?}");
        assert!(
            second
                .iter()
                .all(|f| matches!(f.wanted, Wanted::Tile { level: 0, .. }))
        );
        for fetch in &second {
            let file = &server.files[&fetch.file];
            field
                .store()
                .provide(fetch, serve_range(file, fetch.range))
                .unwrap();
        }
        assert!(field.needs(-100, -100, 100, 100, 20.0).is_empty());
        // Corners in either order give the same list.
        assert_eq!(
            field.needs(100, 100, -100, -100, 20.0),
            field.needs(-100, -100, 100, 100, 20.0)
        );
    }

    #[test]
    fn a_poisoned_lock_does_not_stop_the_store() {
        let dir = TempDir::new();
        let store = Arc::new(LandCoverStore::new(dir.path(), 4));
        let s = Arc::clone(&store);
        let _ = std::thread::spawn(move || {
            let _guard = s.memory.lock().unwrap();
            panic!("a thread dies holding the lock");
        })
        .join();
        assert!(matches!(
            store.lookup(HERE.0, HERE.1, 100.0),
            Lookup::Need(_)
        ));
    }
}

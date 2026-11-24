//! Synthetic terrain for tests: no network, no real data. Available to this crate's tests and, with the `testing`
//! feature, to other crates' tests.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use crate::climate::ClimateVar;
use crate::climatefield::ClimateGrid;
use crate::climatesource::ClimateFetcher;
use crate::landcover::{FILE_DEGREES, WorldCoverFile};
use crate::landsource::LandCoverFetcher;
use crate::projection::pixel_to_latlon;
use crate::source::{FetchError, TileFetcher};
use crate::tiff::{ByteRange, Level};
use crate::tile::{TILE_SIZE, TileId};

/// A terrain as a function of `(latitude, longitude)` in degrees, giving metres.
pub type HeightFn = dyn Fn(f64, f64) -> f64 + Send + Sync;

/// The three terrarium bytes (R, G, B) of an elevation in metres, rounded to the nearest 1/256 m and clamped to
/// what the encoding can hold.
pub fn encode_height(elevation_m: f64) -> [u8; 3] {
    let raw = ((elevation_m + 32768.0) * 256.0)
        .round()
        .clamp(0.0, 16_777_215.0) as u32;
    [(raw >> 16) as u8, (raw >> 8) as u8, raw as u8]
}

/// A PNG of the given pixel bytes, in any colour type and depth (to build tiles the decoder must refuse).
pub fn encode_png(
    width: u32,
    height: u32,
    color: png::ColorType,
    depth: png::BitDepth,
    data: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, width, height);
    encoder.set_color(color);
    encoder.set_depth(depth);
    let mut writer = encoder.write_header().expect("PNG header");
    writer.write_image_data(data).expect("PNG data");
    writer.finish().expect("PNG end");
    out
}

/// The terrarium PNG of tile `id`, sampling `height` at the centre of each pixel.
pub fn encode_tile(id: TileId, height: &HeightFn) -> Vec<u8> {
    let side = TILE_SIZE;
    let mut rgb = Vec::with_capacity((side * side * 3) as usize);
    for py in 0..side {
        for px in 0..side {
            let x = f64::from(id.x * side + px) + 0.5;
            let y = f64::from(id.y * side + py) + 0.5;
            let at = pixel_to_latlon(x, y, id.z);
            rgb.extend(encode_height(height(at.lat, at.lon)));
        }
    }
    encode_png(side, side, png::ColorType::Rgb, png::BitDepth::Eight, &rgb)
}

/// A [`TileFetcher`] that serves synthetic tiles made from a height function, counts what it is asked for, and can
/// be told to fail.
pub struct MemoryFetcher {
    height: Box<HeightFn>,
    fetches: AtomicUsize,
    per_tile: Mutex<HashMap<TileId, usize>>,
    missing: Mutex<HashSet<TileId>>,
    failure: Mutex<Option<FetchError>>,
    delay: Mutex<Duration>,
}

impl MemoryFetcher {
    /// Serves tiles of the terrain `height(lat, lon)`.
    pub fn new(height: impl Fn(f64, f64) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            height: Box::new(height),
            fetches: AtomicUsize::new(0),
            per_tile: Mutex::default(),
            missing: Mutex::default(),
            failure: Mutex::new(None),
            delay: Mutex::new(Duration::ZERO),
        }
    }

    /// How many fetches were asked for, failed ones included.
    pub fn fetch_count(&self) -> usize {
        self.fetches.load(Ordering::SeqCst)
    }

    /// How many fetches asked for this tile.
    pub fn fetches_of(&self, id: TileId) -> usize {
        lock(&self.per_tile).get(&id).copied().unwrap_or(0)
    }

    /// Makes every fetch fail with `error` (or succeed again with `None`).
    pub fn fail_with(&self, error: Option<FetchError>) {
        *lock(&self.failure) = error;
    }

    /// Makes this one tile answer "not found".
    pub fn make_missing(&self, id: TileId) {
        lock(&self.missing).insert(id);
    }

    /// Makes every fetch take this long, so concurrent callers overlap.
    pub fn set_delay(&self, delay: Duration) {
        *lock(&self.delay) = delay;
    }
}

impl TileFetcher for MemoryFetcher {
    fn fetch(&self, id: TileId) -> Result<Vec<u8>, FetchError> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        *lock(&self.per_tile).entry(id).or_default() += 1;
        let delay = *lock(&self.delay);
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
        if let Some(error) = lock(&self.failure).clone() {
            return Err(error);
        }
        if lock(&self.missing).contains(&id) {
            return Err(FetchError::NotFound);
        }
        Ok(encode_tile(id, &*self.height))
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A fresh empty folder under the system's temporary directory, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("horizon-earth-test-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temporary folder");
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Default for TempDir {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A level of the given size and 256-pixel tiles with no tile data, for code that only looks at sizes.
pub fn empty_level(width: u32, height: u32) -> Level {
    Level::from_parts((width, height), (256, 256), Vec::new(), Vec::new())
}

/// One directory entry of a synthetic TIFF: its tag, its type (1 byte, 3 short, 4 long, 16 long8) and its values.
#[derive(Clone, Debug)]
pub struct TiffEntry {
    pub tag: u16,
    pub kind: u16,
    pub values: Vec<u32>,
}

impl TiffEntry {
    pub fn new(tag: u16, kind: u16, values: &[u32]) -> Self {
        Self {
            tag,
            kind,
            values: values.to_vec(),
        }
    }
}

/// Changes a level's directory entries before the file is laid out; gets the level number.
pub type EditEntries = dyn Fn(usize, &mut Vec<TiffEntry>);
/// Changes a level's tile offsets and byte counts after the tiles are placed; gets the level number.
pub type EditTiles = dyn Fn(usize, &mut Vec<u32>, &mut Vec<u32>);

/// What a synthetic tiled TIFF looks like. [`TiffSpec::new`] is a WorldCover-like file in miniature: classic,
/// little-endian, 8-bit, DEFLATE with no predictor, each overview half the size of the level before.
/// [`TiffSpec::bigtiff`], [`TiffSpec::sixteen_bit`] and [`TiffSpec::predictor`] make it CHELSA-like instead.
pub struct TiffSpec {
    pub big_endian: bool,
    /// A BigTIFF: 16-byte header, 64-bit counts, 20-byte entries, LONG8 tile arrays.
    pub bigtiff: bool,
    /// Bits per sample: 8 or 16.
    pub bits: u16,
    /// The predictor: 1 for none, 2 for horizontal differencing (the writer applies it).
    pub predictor: u16,
    pub width: u32,
    pub height: u32,
    /// Tile width and height.
    pub tile: u32,
    /// How many overview levels follow the full resolution.
    pub overviews: usize,
    /// The compression tag's value (8 is DEFLATE; the tiles are always written as zlib streams).
    pub compression: u16,
    pub edit_entries: Option<Box<EditEntries>>,
    pub edit_tiles: Option<Box<EditTiles>>,
}

impl TiffSpec {
    pub fn new(width: u32, height: u32, tile: u32, overviews: usize) -> Self {
        Self {
            big_endian: false,
            bigtiff: false,
            bits: 8,
            predictor: 1,
            width,
            height,
            tile,
            overviews,
            compression: 8,
            edit_entries: None,
            edit_tiles: None,
        }
    }

    pub fn big_endian(mut self) -> Self {
        self.big_endian = true;
        self
    }

    /// Writes a BigTIFF, with a fake GDAL ghost header after the file header as real cloud-optimised files have.
    pub fn bigtiff(mut self) -> Self {
        self.bigtiff = true;
        self
    }

    /// 16-bit samples (write them with [`encode_tiff_u16`]).
    pub fn sixteen_bit(mut self) -> Self {
        self.bits = 16;
        self
    }

    /// Sets the predictor tag (1 or 2); with 2 the writer stores row differences.
    pub fn predictor(mut self, predictor: u16) -> Self {
        self.predictor = predictor;
        self
    }

    pub fn editing_entries(mut self, f: impl Fn(usize, &mut Vec<TiffEntry>) + 'static) -> Self {
        self.edit_entries = Some(Box::new(f));
        self
    }

    pub fn editing_tiles(
        mut self,
        f: impl Fn(usize, &mut Vec<u32>, &mut Vec<u32>) + 'static,
    ) -> Self {
        self.edit_tiles = Some(Box::new(f));
        self
    }

    /// The size of a level: halved each time, never below 1.
    pub fn level_size(&self, level: usize) -> (u32, u32) {
        let shift = level.min(31) as u32;
        ((self.width >> shift).max(1), (self.height >> shift).max(1))
    }
}

/// A synthetic TIFF and how many bytes at its start hold all of its index (directories and tile arrays).
pub struct SyntheticTiff {
    pub bytes: Vec<u8>,
    pub index_len: usize,
}

/// Number writers for the layout of one file: byte order, and the size of counts and offsets.
#[derive(Clone, Copy)]
struct Writer {
    big_endian: bool,
    bigtiff: bool,
}

impl Writer {
    fn u16(self, out: &mut Vec<u8>, v: u16) {
        out.extend(if self.big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        });
    }

    fn u32(self, out: &mut Vec<u8>, v: u32) {
        out.extend(if self.big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        });
    }

    fn u64(self, out: &mut Vec<u8>, v: u64) {
        out.extend(if self.big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        });
    }

    /// A count or an offset: 8 bytes in a BigTIFF, 4 in a classic file.
    fn word(self, out: &mut Vec<u8>, v: u64) {
        if self.bigtiff {
            self.u64(out, v);
        } else {
            self.u32(out, v as u32);
        }
    }

    /// Bytes an entry holds in itself.
    fn inline(self) -> usize {
        if self.bigtiff { 8 } else { 4 }
    }

    /// Bytes of an entry, of the count before the entries, and of the next-directory offset after them.
    fn entry_len(self) -> usize {
        if self.bigtiff { 20 } else { 12 }
    }

    fn count_len(self) -> usize {
        if self.bigtiff { 8 } else { 2 }
    }

    fn next_len(self) -> usize {
        if self.bigtiff { 8 } else { 4 }
    }

    fn value_bytes(self, entry: &TiffEntry) -> Vec<u8> {
        let mut out = Vec::new();
        for &v in &entry.values {
            match type_size(entry.kind) {
                1 => out.push(v as u8),
                2 => self.u16(&mut out, v as u16),
                4 => self.u32(&mut out, v),
                _ => self.u64(&mut out, u64::from(v)),
            }
        }
        out
    }

    fn ifd_len(self, entries: &[TiffEntry]) -> usize {
        let extra: usize = entries
            .iter()
            .map(|e| e.values.len() * type_size(e.kind))
            .filter(|n| *n > self.inline())
            .map(|n| n + n % 2)
            .sum();
        self.count_len() + self.entry_len() * entries.len() + self.next_len() + extra
    }
}

fn type_size(kind: u16) -> usize {
    match kind {
        3 => 2,
        4 => 4,
        16 => 8,
        _ => 1,
    }
}

/// The text GDAL writes after a BigTIFF header in cloud-optimised files (preceded by its length).
const GHOST: &str = "GDAL_STRUCTURE_INFORMATION=\nLAYOUT=IFDS_BEFORE_DATA\nBLOCK_ORDER=ROW_MAJOR\nBLOCK_LEADER=SIZE_AS_UINT4\nBLOCK_TRAILER=LAST_4_BYTES_REPEATED\nKNOWN_INCOMPATIBLE_EDITION=NO\n";

/// A tiled, DEFLATE-compressed 8-bit TIFF: the full resolution then `spec.overviews` overviews, each directory
/// followed by its offset arrays, and then the tile data. `pixel(level, x, y)` gives the 8-bit value of a pixel;
/// the padding of edge tiles is 0. With `spec.bits` of 16 the values are stored as 16-bit samples.
pub fn encode_tiff(spec: &TiffSpec, pixel: &dyn Fn(usize, u32, u32) -> u8) -> SyntheticTiff {
    encode_tiff_u16(spec, &|level, x, y| u16::from(pixel(level, x, y)))
}

/// Like [`encode_tiff`] with 16-bit values: `pixel(level, x, y)` is stored as a 16-bit sample when
/// `spec.bits` is 16, and cut to its low 8 bits when it is 8. The predictor of the spec is applied to every
/// row of every tile (the padding included), with wrapping arithmetic; samples follow the spec's byte order.
pub fn encode_tiff_u16(spec: &TiffSpec, pixel: &dyn Fn(usize, u32, u32) -> u16) -> SyntheticTiff {
    let w = Writer {
        big_endian: spec.big_endian,
        bigtiff: spec.bigtiff,
    };
    let levels = spec.overviews + 1;
    let sixteen = spec.bits == 16;
    // Compressed tiles of every level.
    let mut compressed: Vec<Vec<Vec<u8>>> = Vec::new();
    for level in 0..levels {
        let (lw, lh) = spec.level_size(level);
        let (across, down) = (lw.div_ceil(spec.tile), lh.div_ceil(spec.tile));
        let mut tiles = Vec::new();
        for row in 0..down {
            for col in 0..across {
                let mut raw = Vec::with_capacity((spec.tile * spec.tile) as usize * 2);
                for y in 0..spec.tile {
                    let mut samples: Vec<u16> = (0..spec.tile)
                        .map(|x| {
                            let (px, py) = (col * spec.tile + x, row * spec.tile + y);
                            let v = if px < lw && py < lh {
                                pixel(level, px, py)
                            } else {
                                0
                            };
                            if sixteen { v } else { v & 0xff }
                        })
                        .collect();
                    if spec.predictor == 2 {
                        for i in (1..samples.len()).rev() {
                            let before = samples[i - 1];
                            samples[i] = if sixteen {
                                samples[i].wrapping_sub(before)
                            } else {
                                u16::from((samples[i] as u8).wrapping_sub(before as u8))
                            };
                        }
                    }
                    for v in samples {
                        if sixteen {
                            w.u16(&mut raw, v);
                        } else {
                            raw.push(v as u8);
                        }
                    }
                }
                tiles.push(miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6));
            }
        }
        compressed.push(tiles);
    }
    // The type of the tile arrays: LONG8 in a BigTIFF, LONG in a classic file.
    let array_kind = if spec.bigtiff { 16 } else { 4 };
    // Directory entries per level, without the tile arrays unless the edit hook supplies them.
    let mut entries: Vec<Vec<TiffEntry>> = Vec::new();
    for level in 0..levels {
        let (lw, lh) = spec.level_size(level);
        let mut list = Vec::new();
        if level > 0 {
            list.push(TiffEntry::new(254, 4, &[1]));
        }
        list.extend([
            TiffEntry::new(256, 4, &[lw]),
            TiffEntry::new(257, 4, &[lh]),
            TiffEntry::new(258, 3, &[u32::from(spec.bits)]),
            TiffEntry::new(259, 3, &[u32::from(spec.compression)]),
            TiffEntry::new(262, 3, &[3]),
            TiffEntry::new(277, 3, &[1]),
            TiffEntry::new(284, 3, &[1]),
            TiffEntry::new(317, 3, &[u32::from(spec.predictor)]),
            TiffEntry::new(322, 4, &[spec.tile]),
            TiffEntry::new(323, 4, &[spec.tile]),
            // Placeholders for the tile arrays: an entry the hook leaves as it is gets the real positions.
            TiffEntry::new(324, array_kind, &vec![0; compressed[level].len()]),
            TiffEntry::new(325, array_kind, &vec![0; compressed[level].len()]),
        ]);
        if sixteen {
            list.push(TiffEntry::new(339, 3, &[1]));
        }
        if let Some(edit) = &spec.edit_entries {
            edit(level, &mut list);
        }
        entries.push(list);
    }
    // Tile arrays: an entry the edit hook changed or removed is kept as it is, the others get the real positions
    // of the tiles (the placeholders above had the right sizes to lay the directories out with).
    let untouched =
        |e: &TiffEntry, tiles: usize| e.kind == array_kind && e.values == vec![0; tiles];
    let own: Vec<(bool, bool)> = entries
        .iter()
        .zip(&compressed)
        .map(|(list, tiles)| {
            let own = |tag| {
                !list
                    .iter()
                    .any(|e| e.tag == tag && untouched(e, tiles.len()))
            };
            (own(324), own(325))
        })
        .collect();
    // The header, and for a BigTIFF the ghost header (a length, the text, padded to an even length).
    let mut out = Vec::new();
    out.extend(if spec.big_endian { *b"MM" } else { *b"II" });
    w.u16(&mut out, if spec.bigtiff { 43 } else { 42 });
    if spec.bigtiff {
        w.u16(&mut out, 8);
        w.u16(&mut out, 0);
        let mut ghost = GHOST.as_bytes().to_vec();
        if ghost.len() % 2 == 1 {
            ghost.push(0);
        }
        w.u64(&mut out, 16 + 4 + ghost.len() as u64);
        w.u32(&mut out, ghost.len() as u32);
        out.extend(ghost);
    } else {
        w.u32(&mut out, 8);
    }
    let first_directory = out.len();
    let mut cursor = first_directory + entries.iter().map(|list| w.ifd_len(list)).sum::<usize>();
    let mut offsets: Vec<Vec<u32>> = Vec::new();
    let mut counts: Vec<Vec<u32>> = Vec::new();
    for tiles in &compressed {
        let mut o = Vec::new();
        let mut c = Vec::new();
        for tile in tiles {
            o.push(cursor as u32);
            c.push(tile.len() as u32);
            cursor += tile.len();
        }
        offsets.push(o);
        counts.push(c);
    }
    for level in 0..levels {
        if let Some(edit) = &spec.edit_tiles {
            edit(level, &mut offsets[level], &mut counts[level]);
        }
        for e in &mut entries[level] {
            if e.tag == 324 && !own[level].0 {
                e.values = offsets[level].clone();
            }
            if e.tag == 325 && !own[level].1 {
                e.values = counts[level].clone();
            }
        }
        entries[level].sort_by_key(|e| e.tag);
    }
    // Lay out each directory with its out-of-line values.
    for level in 0..levels {
        let list = &entries[level];
        let start = out.len();
        let fixed = w.count_len() + w.entry_len() * list.len() + w.next_len();
        let mut extra: Vec<u8> = Vec::new();
        if spec.bigtiff {
            w.u64(&mut out, list.len() as u64);
        } else {
            w.u16(&mut out, list.len() as u16);
        }
        for e in list {
            let bytes = w.value_bytes(e);
            w.u16(&mut out, e.tag);
            w.u16(&mut out, e.kind);
            w.word(&mut out, e.values.len() as u64);
            if bytes.len() <= w.inline() {
                out.extend(&bytes);
                out.extend(std::iter::repeat_n(0u8, w.inline() - bytes.len()));
            } else {
                w.word(&mut out, (start + fixed + extra.len()) as u64);
                extra.extend(&bytes);
                if extra.len() % 2 == 1 {
                    extra.push(0);
                }
            }
        }
        let next = if level + 1 < levels {
            (start + fixed + extra.len()) as u64
        } else {
            0
        };
        w.word(&mut out, next);
        out.extend(extra);
    }
    let index_len = out.len();
    for tiles in &compressed {
        for tile in tiles {
            out.extend(tile);
        }
    }
    SyntheticTiff {
        bytes: out,
        index_len,
    }
}

/// What an HTTP range request for `range` returns from a file of `file`: the bytes, cut short at the end of
/// the file.
pub fn serve_range(file: &[u8], range: ByteRange) -> &[u8] {
    let start = (range.offset as usize).min(file.len());
    let end = (range.end() as usize).min(file.len());
    &file[start..end]
}

/// A [`LandCoverFetcher`] over synthetic files held in memory: it serves byte ranges as a range request does,
/// answers "no such file" for a file it was not given, counts what it is asked for, and can be told to fail.
#[derive(Default)]
pub struct MemoryLandCover {
    files: HashMap<WorldCoverFile, Vec<u8>>,
    fetches: AtomicUsize,
    per_file: Mutex<HashMap<WorldCoverFile, usize>>,
    failure: Mutex<Option<FetchError>>,
}

impl MemoryLandCover {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serves `bytes` as `file`.
    pub fn with_file(mut self, file: WorldCoverFile, bytes: Vec<u8>) -> Self {
        self.files.insert(file, bytes);
        self
    }

    /// Serves a miniature `file`: `width` pixels square in tiles of `tile` with `overviews` overview levels,
    /// every pixel the code `class(latitude, longitude)` gives at the pixel's centre (overviews are sampled the
    /// same way, at their own pixel centres).
    pub fn with_classes(
        self,
        file: WorldCoverFile,
        width: u32,
        tile: u32,
        overviews: usize,
        class: impl Fn(f64, f64) -> u8,
    ) -> Self {
        let spec = TiffSpec::new(width, width, tile, overviews);
        let degrees = f64::from(FILE_DEGREES);
        let encoded = encode_tiff(&spec, &|level, x, y| {
            let (w, h) = spec.level_size(level);
            let lat = file.north() - (f64::from(y) + 0.5) / f64::from(h) * degrees;
            let lon = f64::from(file.west) + (f64::from(x) + 0.5) / f64::from(w) * degrees;
            class(lat, lon)
        });
        self.with_file(file, encoded.bytes)
    }

    /// The bytes of a file served, for a test that serves them another way.
    pub fn file_bytes(&self, file: WorldCoverFile) -> Option<Vec<u8>> {
        self.files.get(&file).cloned()
    }

    /// How many fetches were asked for, failed ones and ones for missing files included.
    pub fn fetch_count(&self) -> usize {
        self.fetches.load(Ordering::SeqCst)
    }

    /// How many fetches asked for this file.
    pub fn fetches_of(&self, file: WorldCoverFile) -> usize {
        lock(&self.per_file).get(&file).copied().unwrap_or(0)
    }

    /// Makes every fetch fail with `error` (or succeed again with `None`).
    pub fn fail_with(&self, error: Option<FetchError>) {
        *lock(&self.failure) = error;
    }
}

impl LandCoverFetcher for MemoryLandCover {
    fn fetch(&self, file: WorldCoverFile, range: ByteRange) -> Result<Option<Vec<u8>>, FetchError> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        *lock(&self.per_file).entry(file).or_default() += 1;
        if let Some(error) = lock(&self.failure).clone() {
            return Err(error);
        }
        Ok(self
            .files
            .get(&file)
            .map(|bytes| serve_range(bytes, range).to_vec()))
    }
}

/// The grid of the small synthetic climate files ([`climate_file`]): one pixel a degree.
pub const SMALL_CLIMATE_GRID: ClimateGrid = ClimateGrid {
    width: 360,
    height: 174,
};

/// A CHELSA-like file in miniature: BigTIFF, 16-bit, DEFLATE with predictor 2, `width` by `height` pixels in
/// tiles of `tile` with `overviews` overview levels, each half the size of the one before. `pixel(level, x, y)`
/// is the raw value of a pixel. Its extent is the whole grid, as with the real files.
pub fn climate_file(
    width: u32,
    height: u32,
    tile: u32,
    overviews: usize,
    pixel: &dyn Fn(usize, u32, u32) -> u16,
) -> SyntheticTiff {
    let spec = TiffSpec::new(width, height, tile, overviews)
        .bigtiff()
        .sixteen_bit()
        .predictor(2);
    encode_tiff_u16(&spec, pixel)
}

/// A [`ClimateFetcher`] over synthetic files held in memory: it serves byte ranges as a range request does,
/// answers "no such file" for a variable it was not given, counts what it is asked for, and can be told to fail.
#[derive(Default)]
pub struct MemoryClimate {
    files: HashMap<ClimateVar, Vec<u8>>,
    fetches: AtomicUsize,
    per_var: Mutex<HashMap<ClimateVar, usize>>,
    failure: Mutex<Option<FetchError>>,
}

impl MemoryClimate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serves `bytes` as the file of `var`.
    pub fn with_file(mut self, var: ClimateVar, bytes: Vec<u8>) -> Self {
        self.files.insert(var, bytes);
        self
    }

    /// Serves a miniature file of `var` of [`SMALL_CLIMATE_GRID`]'s size in tiles of `tile` with `overviews`
    /// overview levels; `raw(level, x, y)` is the raw value of a pixel.
    pub fn with_raw(
        self,
        var: ClimateVar,
        tile: u32,
        overviews: usize,
        raw: impl Fn(usize, u32, u32) -> u16,
    ) -> Self {
        let file = climate_file(
            SMALL_CLIMATE_GRID.width,
            SMALL_CLIMATE_GRID.height,
            tile,
            overviews,
            &raw,
        );
        self.with_file(var, file.bytes)
    }

    /// The bytes of a file served, for a test that serves them another way.
    pub fn file_bytes(&self, var: ClimateVar) -> Option<Vec<u8>> {
        self.files.get(&var).cloned()
    }

    /// How many fetches were asked for, failed ones and ones for missing files included.
    pub fn fetch_count(&self) -> usize {
        self.fetches.load(Ordering::SeqCst)
    }

    /// How many fetches asked for this variable's file.
    pub fn fetches_of(&self, var: ClimateVar) -> usize {
        lock(&self.per_var).get(&var).copied().unwrap_or(0)
    }

    /// Makes every fetch fail with `error` (or succeed again with `None`).
    pub fn fail_with(&self, error: Option<FetchError>) {
        *lock(&self.failure) = error;
    }
}

impl ClimateFetcher for MemoryClimate {
    fn fetch(&self, var: ClimateVar, range: ByteRange) -> Result<Option<Vec<u8>>, FetchError> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        *lock(&self.per_var).entry(var).or_default() += 1;
        if let Some(error) = lock(&self.failure).clone() {
            return Err(error);
        }
        Ok(self
            .files
            .get(&var)
            .map(|bytes| serve_range(bytes, range).to_vec()))
    }
}

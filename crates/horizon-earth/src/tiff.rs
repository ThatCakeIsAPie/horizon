//! A minimal TIFF reader for tiled land-cover rasters, as pure functions over byte slices.
//!
//! It reads exactly what the ESA WorldCover files use and nothing else: classic TIFF (not BigTIFF) in either byte
//! order, tiled, 8 bits, one sample per pixel, compressed with DEFLATE (a zlib stream) and no predictor, with a
//! chain of image directories (the full resolution first, then smaller overviews). Anything else is an error that
//! says what it is; nothing is guessed.
//!
//! The reader never sees the whole file. [`parse_index`] works on a prefix, or on pieces of the file the caller
//! holds, and when a directory or an array of tile offsets lies beyond them it says which byte range to fetch
//! next. Every offset and count in the input is bounds-checked, sizes are capped ([`MAX_DIRECTORIES`],
//! [`MAX_TILES_PER_LEVEL`], [`MAX_TOTAL_TILES`], [`MAX_DIMENSION`], [`MAX_TILE_SIDE`]), and nothing is allocated
//! from an unchecked length, so hostile bytes give an error, never a panic or a huge allocation.

use std::fmt;

/// Most image directories (levels) a file may have.
pub const MAX_DIRECTORIES: usize = 32;
/// Most entries one image directory may have.
pub const MAX_ENTRIES: usize = 256;
/// Largest width or height of a level, in pixels.
pub const MAX_DIMENSION: u32 = 1 << 20;
/// Largest tile width or height, in pixels (a tile is at most 16 MiB of 8-bit pixels).
pub const MAX_TILE_SIDE: u32 = 4096;
/// Most tiles one level may have.
pub const MAX_TILES_PER_LEVEL: u64 = 1 << 16;
/// Most tiles all the levels of a file may have together.
pub const MAX_TOTAL_TILES: u64 = 1 << 18;
/// How many bytes are asked for at once when a directory lies beyond the bytes held (its entry count is not yet
/// known). A real directory of about twenty entries fits.
pub const DIRECTORY_GUESS: u64 = 1024;
/// The bytes a caller should fetch first: the whole index of a WorldCover file lies within its first 32 KiB.
pub const PREFIX_LEN: u64 = 32 * 1024;

const HEADER_LEN: u64 = 8;

const TAG_WIDTH: u16 = 256;
const TAG_HEIGHT: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_PLANAR_CONFIG: u16 = 284;
const TAG_PREDICTOR: u16 = 317;
const TAG_TILE_WIDTH: u16 = 322;
const TAG_TILE_LENGTH: u16 = 323;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;
const TAG_SAMPLE_FORMAT: u16 = 339;

const TYPE_BYTE: u16 = 1;
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;

const COMPRESSION_DEFLATE: u32 = 8;
const COMPRESSION_DEFLATE_OLD: u32 = 32946;

/// Why bytes are not a TIFF this reader supports, or not a tile of one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TiffError {
    /// The bytes do not begin like a TIFF.
    NotTiff,
    /// A BigTIFF (64-bit offsets), which the reader does not support.
    BigTiff,
    /// Something the file points at lies beyond the end of the file.
    Truncated,
    /// The structure is malformed (a loop, a bad count, a value that does not fit).
    Malformed(&'static str),
    /// Valid TIFF, but not the kind this reader reads.
    Unsupported(&'static str),
    /// A size past the reader's caps.
    TooLarge(&'static str),
    /// A tile index outside the level.
    NoSuchTile,
    /// Tile bytes that do not decode to the tile they should.
    BadTile(&'static str),
}

impl fmt::Display for TiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotTiff => f.write_str("not a TIFF file"),
            Self::BigTiff => f.write_str("a BigTIFF file, which is not supported"),
            Self::Truncated => f.write_str("the TIFF file ends before something it points at"),
            Self::Malformed(why) => write!(f, "malformed TIFF: {why}"),
            Self::Unsupported(what) => write!(f, "unsupported TIFF: {what}"),
            Self::TooLarge(what) => write!(f, "TIFF too large: {what}"),
            Self::NoSuchTile => f.write_str("no such tile in this TIFF level"),
            Self::BadTile(why) => write!(f, "unusable TIFF tile: {why}"),
        }
    }
}

impl std::error::Error for TiffError {}

/// A run of bytes in a file: where it starts and how long it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ByteRange {
    pub offset: u64,
    pub len: u64,
}

impl ByteRange {
    /// The offset one past the last byte.
    pub fn end(self) -> u64 {
        self.offset.saturating_add(self.len)
    }

    /// The value of an HTTP `Range: bytes=` header for this range (the end is inclusive there).
    pub fn http_range(self) -> String {
        format!(
            "bytes={}-{}",
            self.offset,
            self.end().saturating_sub(1).max(self.offset)
        )
    }
}

/// Bytes of a file the caller holds, starting at `offset` in the file.
#[derive(Clone, Copy, Debug)]
pub struct Piece<'a> {
    pub offset: u64,
    pub bytes: &'a [u8],
}

/// How [`parse_index`] ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parse {
    /// The whole index was in the bytes given.
    Done(TiffIndex),
    /// Fetch this range, add it as a [`Piece`], and parse again.
    Need(ByteRange),
}

/// Byte order of a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    fn u16(self, b: [u8; 2]) -> u16 {
        match self {
            Self::Little => u16::from_le_bytes(b),
            Self::Big => u16::from_be_bytes(b),
        }
    }

    fn u32(self, b: [u8; 4]) -> u32 {
        match self {
            Self::Little => u32::from_le_bytes(b),
            Self::Big => u32::from_be_bytes(b),
        }
    }
}

/// One image directory of a file: its size, its tiling, and where its tiles are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    offsets: Vec<u32>,
    byte_counts: Vec<u32>,
}

impl Level {
    /// A level from its parts, unchecked (for tests of code that only looks at sizes).
    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn from_parts(
        (width, height): (u32, u32),
        (tile_width, tile_height): (u32, u32),
        offsets: Vec<u32>,
        byte_counts: Vec<u32>,
    ) -> Self {
        Self {
            width,
            height,
            tile_width,
            tile_height,
            offsets,
            byte_counts,
        }
    }

    /// Tiles along a row.
    pub fn tiles_across(&self) -> u32 {
        self.width.div_ceil(self.tile_width)
    }

    /// Tile rows.
    pub fn tiles_down(&self) -> u32 {
        self.height.div_ceil(self.tile_height)
    }

    /// How many tiles the level has.
    pub fn tile_count(&self) -> usize {
        self.offsets.len()
    }

    /// The index of the tile in column `col` and row `row` (row-major, from the top left), or `None` outside
    /// the level.
    pub fn tile_index(&self, col: u32, row: u32) -> Option<usize> {
        if col >= self.tiles_across() || row >= self.tiles_down() {
            return None;
        }
        Some(row as usize * self.tiles_across() as usize + col as usize)
    }

    /// Pixels in one decoded tile (edge tiles are stored whole, padded).
    pub fn tile_pixels(&self) -> usize {
        self.tile_width as usize * self.tile_height as usize
    }

    /// The most compressed bytes a tile of this level may have: DEFLATE may expand incompressible data a little.
    pub fn max_tile_bytes(&self) -> u64 {
        let raw = self.tile_pixels() as u64;
        raw + raw / 16 + 1024
    }

    /// Where the compressed bytes of tile `index` are in the file. `None` is a tile the file leaves out (its
    /// byte count is 0), which holds no data. An error when the index is outside the level or the file's
    /// numbers cannot be right.
    pub fn tile_range(&self, index: usize) -> Result<Option<ByteRange>, TiffError> {
        let (Some(&offset), Some(&len)) = (self.offsets.get(index), self.byte_counts.get(index))
        else {
            return Err(TiffError::NoSuchTile);
        };
        if len == 0 {
            return Ok(None);
        }
        if u64::from(len) > self.max_tile_bytes() {
            return Err(TiffError::TooLarge("a tile's compressed size"));
        }
        if u64::from(offset) < HEADER_LEN {
            return Err(TiffError::Malformed("a tile starts inside the header"));
        }
        Ok(Some(ByteRange {
            offset: u64::from(offset),
            len: u64::from(len),
        }))
    }

    /// The pixels of one tile from its compressed bytes: `tile_width * tile_height` bytes, row by row. Output
    /// is never larger than that, whatever the input claims.
    pub fn decode_tile(&self, bytes: &[u8]) -> Result<Vec<u8>, TiffError> {
        let want = self.tile_pixels();
        if bytes.is_empty() {
            return Err(TiffError::BadTile("no bytes"));
        }
        if bytes.len() as u64 > self.max_tile_bytes() {
            return Err(TiffError::TooLarge("a tile's compressed size"));
        }
        let pixels = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(bytes, want)
            .map_err(|_| TiffError::BadTile("the DEFLATE data is damaged or larger than a tile"))?;
        if pixels.len() != want {
            return Err(TiffError::BadTile(
                "the tile decodes to the wrong number of pixels",
            ));
        }
        Ok(pixels)
    }
}

/// The directory of a TIFF file: its byte order and its levels, full resolution first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiffIndex {
    pub order: ByteOrder,
    pub levels: Vec<Level>,
}

/// The bytes the caller holds, and whether the end of the file is known.
struct Held<'a> {
    pieces: &'a [Piece<'a>],
    file_len: Option<u64>,
}

enum Got<'a> {
    Bytes(&'a [u8]),
    Need(ByteRange),
}

impl<'a> Held<'a> {
    /// `len` bytes at `offset`: the bytes when one piece holds them all, otherwise the range to fetch (at least
    /// `ask` bytes, so a probe can take more than it needs). An error when the file is known to end first.
    fn get(&self, offset: u64, len: u64, ask: u64) -> Result<Got<'a>, TiffError> {
        let end = offset.checked_add(len).ok_or(TiffError::Truncated)?;
        if self.file_len.is_some_and(|total| end > total) {
            return Err(TiffError::Truncated);
        }
        for piece in self.pieces {
            let start = piece.offset;
            let piece_end = start.saturating_add(piece.bytes.len() as u64);
            if start <= offset && end <= piece_end {
                let from = (offset - start) as usize;
                if let Some(bytes) = piece.bytes.get(from..from + len as usize) {
                    return Ok(Got::Bytes(bytes));
                }
            }
        }
        let want = ask.max(len);
        let mut range = ByteRange { offset, len: want };
        if let Some(total) = self.file_len {
            range.len = range.len.min(total - offset);
        }
        Ok(Got::Need(range))
    }
}

/// One 12-byte directory entry.
#[derive(Clone, Copy)]
struct Entry {
    tag: u16,
    kind: u16,
    count: u32,
    field: [u8; 4],
}

impl Entry {
    fn type_size(&self) -> Option<u64> {
        match self.kind {
            TYPE_BYTE => Some(1),
            TYPE_SHORT => Some(2),
            TYPE_LONG => Some(4),
            _ => None,
        }
    }

    /// The single value of an entry holding one SHORT, LONG or BYTE.
    fn scalar(&self, order: ByteOrder) -> Result<u32, TiffError> {
        if self.count != 1 {
            return Err(TiffError::Malformed(
                "a tag that holds one value holds several",
            ));
        }
        let f = self.field;
        match self.kind {
            TYPE_BYTE => Ok(u32::from(f[0])),
            TYPE_SHORT => Ok(u32::from(order.u16([f[0], f[1]]))),
            TYPE_LONG => Ok(order.u32(f)),
            _ => Err(TiffError::Malformed(
                "a number tag of a type that is not a number",
            )),
        }
    }
}

/// A directory's numbers, before its arrays are read.
struct Raw {
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    offsets: Entry,
    byte_counts: Entry,
}

fn read_u32(bytes: &[u8], at: usize, order: ByteOrder) -> Result<u32, TiffError> {
    let b = bytes
        .get(at..at + 4)
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .ok_or(TiffError::Truncated)?;
    Ok(order.u32(b))
}

fn read_u16(bytes: &[u8], at: usize, order: ByteOrder) -> Result<u16, TiffError> {
    let b = bytes
        .get(at..at + 2)
        .and_then(|s| <[u8; 2]>::try_from(s).ok())
        .ok_or(TiffError::Truncated)?;
    Ok(order.u16(b))
}

/// Reads the directory of the file from the bytes held. The result is the whole index, or the range to fetch
/// before asking again. `file_len` is the file's size when known (a fetch that came back short means the file
/// ended): then reaching past it is [`TiffError::Truncated`] instead of a request for more.
///
/// Start with a prefix of [`PREFIX_LEN`] bytes. A caller that keeps adding what is asked for ends within a few
/// rounds: each [`Parse::Need`] range lies beyond what was held, and the number of directories and tiles is
/// capped.
pub fn parse_index(pieces: &[Piece<'_>], file_len: Option<u64>) -> Result<Parse, TiffError> {
    let held = Held { pieces, file_len };
    let header = match held.get(0, HEADER_LEN, PREFIX_LEN)? {
        Got::Bytes(b) => b,
        Got::Need(range) => return Ok(Parse::Need(range)),
    };
    let order = match &header[0..2] {
        b"II" => ByteOrder::Little,
        b"MM" => ByteOrder::Big,
        _ => return Err(TiffError::NotTiff),
    };
    match read_u16(header, 2, order)? {
        42 => {}
        43 => return Err(TiffError::BigTiff),
        _ => return Err(TiffError::NotTiff),
    }
    let mut next = u64::from(read_u32(header, 4, order)?);
    let mut visited: Vec<u64> = Vec::new();
    let mut levels: Vec<Level> = Vec::new();
    let mut total_tiles = 0u64;
    while next != 0 {
        if levels.len() >= MAX_DIRECTORIES {
            return Err(TiffError::TooLarge("too many image directories"));
        }
        if next < HEADER_LEN {
            return Err(TiffError::Malformed("a directory starts inside the header"));
        }
        if visited.contains(&next) {
            return Err(TiffError::Malformed("the directories form a loop"));
        }
        visited.push(next);
        let (entries, after) = match read_directory(&held, next, order)? {
            Directory::Done(entries, after) => (entries, after),
            Directory::Need(range) => return Ok(Parse::Need(range)),
        };
        let raw = interpret(&entries, order)?;
        let tiles = tile_count(&raw)?;
        total_tiles += tiles;
        if total_tiles > MAX_TOTAL_TILES {
            return Err(TiffError::TooLarge("too many tiles in the file"));
        }
        let offsets = match read_array(&held, raw.offsets, tiles, order, false)? {
            Array::Done(values) => values,
            Array::Need(range) => return Ok(Parse::Need(range)),
        };
        let byte_counts = match read_array(&held, raw.byte_counts, tiles, order, true)? {
            Array::Done(values) => values,
            Array::Need(range) => return Ok(Parse::Need(range)),
        };
        if let Some(previous) = levels.last()
            && (raw.width > previous.width || raw.height > previous.height)
        {
            return Err(TiffError::Unsupported(
                "a later image directory larger than the one before it (not an overview chain)",
            ));
        }
        levels.push(Level {
            width: raw.width,
            height: raw.height,
            tile_width: raw.tile_width,
            tile_height: raw.tile_height,
            offsets,
            byte_counts,
        });
        next = after;
    }
    if levels.is_empty() {
        return Err(TiffError::Malformed("no image directory"));
    }
    Ok(Parse::Done(TiffIndex { order, levels }))
}

enum Directory {
    Done(Vec<Entry>, u64),
    Need(ByteRange),
}

fn read_directory(held: &Held<'_>, at: u64, order: ByteOrder) -> Result<Directory, TiffError> {
    let count_bytes = match held.get(at, 2, DIRECTORY_GUESS)? {
        Got::Bytes(b) => b,
        Got::Need(range) => return Ok(Directory::Need(range)),
    };
    let count = usize::from(read_u16(count_bytes, 0, order)?);
    if count == 0 {
        return Err(TiffError::Malformed("an empty image directory"));
    }
    if count > MAX_ENTRIES {
        return Err(TiffError::TooLarge(
            "too many entries in an image directory",
        ));
    }
    let size = 2 + 12 * count as u64 + 4;
    let body = match held.get(at, size, size)? {
        Got::Bytes(b) => b,
        Got::Need(range) => return Ok(Directory::Need(range)),
    };
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let base = 2 + 12 * i;
        let field = body
            .get(base + 8..base + 12)
            .and_then(|s| <[u8; 4]>::try_from(s).ok())
            .ok_or(TiffError::Truncated)?;
        entries.push(Entry {
            tag: read_u16(body, base, order)?,
            kind: read_u16(body, base + 2, order)?,
            count: read_u32(body, base + 4, order)?,
            field,
        });
    }
    let after = u64::from(read_u32(body, 2 + 12 * count, order)?);
    Ok(Directory::Done(entries, after))
}

/// Checks the tags that say what the pixels are, and gathers the ones that say where they are.
fn interpret(entries: &[Entry], order: ByteOrder) -> Result<Raw, TiffError> {
    let (mut width, mut height, mut tile_width, mut tile_height) = (None, None, None, None);
    let (mut offsets, mut byte_counts) = (None, None);
    let mut compression = None;
    let mut seen: Vec<u16> = Vec::with_capacity(entries.len());
    for e in entries {
        if seen.contains(&e.tag) {
            return Err(TiffError::Malformed("a tag appears twice in a directory"));
        }
        seen.push(e.tag);
        match e.tag {
            TAG_WIDTH => width = Some(e.scalar(order)?),
            TAG_HEIGHT => height = Some(e.scalar(order)?),
            TAG_TILE_WIDTH => tile_width = Some(e.scalar(order)?),
            TAG_TILE_LENGTH => tile_height = Some(e.scalar(order)?),
            TAG_TILE_OFFSETS => offsets = Some(*e),
            TAG_TILE_BYTE_COUNTS => byte_counts = Some(*e),
            TAG_COMPRESSION => compression = Some(e.scalar(order)?),
            TAG_BITS_PER_SAMPLE => {
                if e.scalar(order)? != 8 {
                    return Err(TiffError::Unsupported("anything but 8 bits per sample"));
                }
            }
            TAG_SAMPLES_PER_PIXEL => {
                if e.scalar(order)? != 1 {
                    return Err(TiffError::Unsupported("more than one sample per pixel"));
                }
            }
            TAG_PLANAR_CONFIG => {
                if e.scalar(order)? != 1 {
                    return Err(TiffError::Unsupported(
                        "planar configuration other than chunky",
                    ));
                }
            }
            TAG_PREDICTOR => {
                if e.scalar(order)? != 1 {
                    return Err(TiffError::Unsupported("a predictor"));
                }
            }
            TAG_SAMPLE_FORMAT => {
                if e.scalar(order)? != 1 {
                    return Err(TiffError::Unsupported(
                        "samples that are not unsigned integers",
                    ));
                }
            }
            TAG_STRIP_OFFSETS => {
                return Err(TiffError::Unsupported("strips (the image is not tiled)"));
            }
            _ => {}
        }
    }
    match compression {
        Some(COMPRESSION_DEFLATE | COMPRESSION_DEFLATE_OLD) => {}
        Some(_) => return Err(TiffError::Unsupported("a compression other than DEFLATE")),
        None => return Err(TiffError::Unsupported("no compression tag")),
    }
    let missing = TiffError::Malformed("a tag the reader needs is missing");
    let (width, height) = (
        width.ok_or(missing.clone())?,
        height.ok_or(missing.clone())?,
    );
    let (tile_width, tile_height) = (
        tile_width.ok_or(TiffError::Unsupported("the image is not tiled"))?,
        tile_height.ok_or(TiffError::Unsupported("the image is not tiled"))?,
    );
    if width == 0 || height == 0 || tile_width == 0 || tile_height == 0 {
        return Err(TiffError::Malformed("a size of 0"));
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(TiffError::TooLarge("the image's width or height"));
    }
    if tile_width > MAX_TILE_SIDE || tile_height > MAX_TILE_SIDE {
        return Err(TiffError::TooLarge("a tile's width or height"));
    }
    Ok(Raw {
        width,
        height,
        tile_width,
        tile_height,
        offsets: offsets.ok_or(missing.clone())?,
        byte_counts: byte_counts.ok_or(missing)?,
    })
}

/// How many tiles the directory's size says it has, within the caps.
fn tile_count(raw: &Raw) -> Result<u64, TiffError> {
    let across = u64::from(raw.width.div_ceil(raw.tile_width));
    let down = u64::from(raw.height.div_ceil(raw.tile_height));
    let tiles = across * down;
    if tiles > MAX_TILES_PER_LEVEL {
        return Err(TiffError::TooLarge("too many tiles in a level"));
    }
    Ok(tiles)
}

enum Array {
    Done(Vec<u32>),
    Need(ByteRange),
}

/// The `expected` numbers of an offsets or byte-counts entry. Its own count must agree, so an array's length
/// is never taken from the file.
fn read_array(
    held: &Held<'_>,
    entry: Entry,
    expected: u64,
    order: ByteOrder,
    allow_short: bool,
) -> Result<Array, TiffError> {
    let size = match entry.type_size() {
        Some(4) => 4,
        Some(2) if allow_short => 2,
        _ => return Err(TiffError::Malformed("a tile array of the wrong type")),
    };
    if u64::from(entry.count) != expected {
        return Err(TiffError::Malformed(
            "a tile array whose count differs from the tile count",
        ));
    }
    let count = expected as usize;
    if expected * size <= 4 {
        // Held in the entry itself, left-justified.
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            let at = i * size as usize;
            values.push(match size {
                2 => u32::from(read_u16(&entry.field, at, order)?),
                _ => read_u32(&entry.field, at, order)?,
            });
        }
        return Ok(Array::Done(values));
    }
    let at = u64::from(order.u32(entry.field));
    if at < HEADER_LEN {
        return Err(TiffError::Malformed(
            "a tile array starts inside the header",
        ));
    }
    let bytes = match held.get(at, expected * size, expected * size)? {
        Got::Bytes(b) => b,
        Got::Need(range) => return Ok(Array::Need(range)),
    };
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let at = i * size as usize;
        values.push(match size {
            2 => u32::from(read_u16(bytes, at, order)?),
            _ => read_u32(bytes, at, order)?,
        });
    }
    Ok(Array::Done(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{SyntheticTiff, TiffEntry, TiffSpec, encode_tiff, serve_range};

    /// A pixel value that differs by level, column and row.
    fn pattern(level: usize, x: u32, y: u32) -> u8 {
        ((level as u32 * 37 + x * 7 + y * 13) % 251) as u8
    }

    fn sample(spec: &TiffSpec) -> SyntheticTiff {
        encode_tiff(spec, &pattern)
    }

    fn miniature() -> TiffSpec {
        TiffSpec::new(600, 400, 256, 2)
    }

    fn parse_all(file: &[u8]) -> Result<TiffIndex, TiffError> {
        match parse_index(
            &[Piece {
                offset: 0,
                bytes: file,
            }],
            Some(file.len() as u64),
        )? {
            Parse::Done(index) => Ok(index),
            Parse::Need(range) => panic!("the whole file was held, yet {range:?} was asked for"),
        }
    }

    /// Reads the index the way a caller does: a prefix of `first` bytes, then whatever is asked for. Gives the
    /// index and how many times more was asked for.
    fn parse_by_fetching(file: &[u8], first: usize) -> Result<(TiffIndex, usize), TiffError> {
        let first = first.min(file.len());
        let mut owned: Vec<(u64, Vec<u8>)> = vec![(0, file[..first].to_vec())];
        let mut file_len = (first == file.len()).then_some(file.len() as u64);
        let mut rounds = 0;
        loop {
            let pieces: Vec<Piece<'_>> = owned
                .iter()
                .map(|(offset, bytes)| Piece {
                    offset: *offset,
                    bytes,
                })
                .collect();
            match parse_index(&pieces, file_len)? {
                Parse::Done(index) => return Ok((index, rounds)),
                Parse::Need(range) => {
                    rounds += 1;
                    assert!(rounds < 64, "no progress: {range:?}");
                    let got = serve_range(file, range);
                    if (got.len() as u64) < range.len {
                        file_len = Some(file.len() as u64);
                    }
                    if got.is_empty() {
                        return Err(TiffError::Truncated);
                    }
                    owned.push((range.offset, got.to_vec()));
                }
            }
        }
    }

    /// The position of the 12-byte entry with `tag` in the directory at `ifd`.
    fn entry_at(bytes: &[u8], ifd: usize, tag: u16, big: bool) -> usize {
        let n = if big {
            u16::from_be_bytes([bytes[ifd], bytes[ifd + 1]])
        } else {
            u16::from_le_bytes([bytes[ifd], bytes[ifd + 1]])
        } as usize;
        (0..n)
            .map(|i| ifd + 2 + 12 * i)
            .find(|&at| {
                let t = [bytes[at], bytes[at + 1]];
                (if big {
                    u16::from_be_bytes(t)
                } else {
                    u16::from_le_bytes(t)
                }) == tag
            })
            .expect("the tag is in the directory")
    }

    fn patch32(bytes: &mut [u8], at: usize, v: u32, big: bool) {
        bytes[at..at + 4].copy_from_slice(&if big {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        });
    }

    fn patch16(bytes: &mut [u8], at: usize, v: u16, big: bool) {
        bytes[at..at + 2].copy_from_slice(&if big {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        });
    }

    fn expect_pixels(level: &Level, lvl: usize, col: u32, row: u32, got: &[u8]) {
        assert_eq!(got.len(), level.tile_pixels());
        for y in 0..level.tile_height {
            for x in 0..level.tile_width {
                let (px, py) = (col * level.tile_width + x, row * level.tile_height + y);
                let want = if px < level.width && py < level.height {
                    pattern(lvl, px, py)
                } else {
                    0
                };
                assert_eq!(
                    got[(y * level.tile_width + x) as usize],
                    want,
                    "level {lvl} tile ({col},{row}) pixel ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn a_file_round_trips_in_both_byte_orders() {
        for big in [false, true] {
            let spec = if big {
                miniature().big_endian()
            } else {
                miniature()
            };
            let file = sample(&spec);
            let index = parse_all(&file.bytes).unwrap();
            assert_eq!(
                index.order,
                if big {
                    ByteOrder::Big
                } else {
                    ByteOrder::Little
                }
            );
            let sizes: Vec<_> = index.levels.iter().map(|l| (l.width, l.height)).collect();
            assert_eq!(sizes, [(600, 400), (300, 200), (150, 100)]);
            assert_eq!(index.levels[0].tiles_across(), 3);
            assert_eq!(index.levels[0].tiles_down(), 2);
            assert_eq!(index.levels[0].tile_count(), 6);
            assert_eq!(index.levels[1].tile_count(), 2);
            assert_eq!(index.levels[2].tile_count(), 1);
            for (lvl, level) in index.levels.iter().enumerate() {
                for row in 0..level.tiles_down() {
                    for col in 0..level.tiles_across() {
                        let at = level.tile_index(col, row).unwrap();
                        let range = level
                            .tile_range(at)
                            .unwrap()
                            .expect("every tile is present");
                        let bytes = serve_range(&file.bytes, range);
                        assert_eq!(bytes.len() as u64, range.len);
                        let pixels = level.decode_tile(bytes).unwrap();
                        expect_pixels(level, lvl, col, row, &pixels);
                    }
                }
            }
        }
    }

    #[test]
    fn the_index_lies_within_the_length_the_writer_reports() {
        let file = sample(&miniature());
        let held = &file.bytes[..file.index_len];
        assert!(matches!(
            parse_index(
                &[Piece {
                    offset: 0,
                    bytes: held
                }],
                None
            ),
            Ok(Parse::Done(_))
        ));
        // Without the last index byte the final array is not whole.
        let short = &file.bytes[..file.index_len - 1];
        assert!(matches!(
            parse_index(
                &[Piece {
                    offset: 0,
                    bytes: short
                }],
                None
            ),
            Ok(Parse::Need(_))
        ));
    }

    #[test]
    fn a_short_prefix_says_what_to_fetch_next_and_ends_with_the_same_index() {
        for big in [false, true] {
            let spec = if big {
                miniature().big_endian()
            } else {
                miniature()
            };
            let file = sample(&spec);
            let whole = parse_all(&file.bytes).unwrap();
            for first in [
                0,
                1,
                7,
                8,
                9,
                20,
                100,
                200,
                300,
                file.index_len - 1,
                file.index_len,
                file.bytes.len(),
            ] {
                let (index, rounds) = parse_by_fetching(&file.bytes, first)
                    .unwrap_or_else(|e| panic!("prefix {first}: {e}"));
                assert_eq!(index, whole, "prefix {first}");
                if first >= file.index_len {
                    assert_eq!(rounds, 0, "prefix {first}");
                }
            }
        }
    }

    #[test]
    fn the_first_request_is_the_prefix_and_later_ones_are_exact() {
        let file = sample(&miniature());
        let Ok(Parse::Need(first)) = parse_index(&[], None) else {
            panic!("an empty prefix needs the prefix")
        };
        assert_eq!(
            first,
            ByteRange {
                offset: 0,
                len: PREFIX_LEN
            }
        );
        // The header only: the first directory is at byte 8.
        let header = Piece {
            offset: 0,
            bytes: &file.bytes[..8],
        };
        let Ok(Parse::Need(next)) = parse_index(&[header], None) else {
            panic!("a header alone is not enough")
        };
        assert_eq!(next.offset, 8);
        assert_eq!(next.len, DIRECTORY_GUESS);
        // With the first directory (12 entries) held but not its offset array, the array is asked for
        // exactly: 6 offsets of 4 bytes.
        let held = Piece {
            offset: 0,
            bytes: &file.bytes[..8 + 2 + 12 * 12 + 4],
        };
        let Ok(Parse::Need(array)) = parse_index(&[held], None) else {
            panic!("the arrays are beyond the bytes held")
        };
        assert_eq!(array.len, 24);
    }

    #[test]
    fn a_file_that_ends_early_is_truncated_not_endless() {
        let file = sample(&miniature());
        // The caller was told the file is this long: asking for more would never end.
        for cut in [9, 40, 120, file.index_len - 1] {
            let held = &file.bytes[..cut];
            let r = parse_index(
                &[Piece {
                    offset: 0,
                    bytes: held,
                }],
                Some(cut as u64),
            );
            assert_eq!(r, Err(TiffError::Truncated), "cut at {cut}");
        }
        assert_eq!(
            parse_by_fetching(&file.bytes[..file.index_len - 1], 100).err(),
            Some(TiffError::Truncated)
        );
        // A file of 3 bytes is not even a header.
        assert_eq!(
            parse_index(
                &[Piece {
                    offset: 0,
                    bytes: b"II*"
                }],
                Some(3)
            ),
            Err(TiffError::Truncated)
        );
    }

    #[test]
    fn pieces_may_arrive_in_any_order_and_overlap() {
        let file = sample(&miniature());
        let whole = parse_all(&file.bytes).unwrap();
        let n = file.bytes.len();
        let pieces = [
            Piece {
                offset: 40,
                bytes: &file.bytes[40..n.min(file.index_len + 10)],
            },
            Piece {
                offset: 0,
                bytes: &file.bytes[..60],
            },
            Piece {
                offset: 0,
                bytes: &file.bytes[..200],
            },
        ];
        assert_eq!(parse_index(&pieces, None), Ok(Parse::Done(whole)));
        // A piece that claims an offset beyond u64 does not panic.
        let odd = [
            Piece {
                offset: u64::MAX - 2,
                bytes: &[1, 2, 3, 4, 5],
            },
            Piece {
                offset: 0,
                bytes: &file.bytes[..16],
            },
        ];
        assert!(parse_index(&odd, None).is_ok());
    }

    #[test]
    fn the_http_range_header_has_an_inclusive_end() {
        assert_eq!(
            ByteRange {
                offset: 0,
                len: 32768
            }
            .http_range(),
            "bytes=0-32767"
        );
        assert_eq!(
            ByteRange {
                offset: 100,
                len: 1
            }
            .http_range(),
            "bytes=100-100"
        );
        assert_eq!(
            ByteRange {
                offset: u64::MAX,
                len: 10
            }
            .end(),
            u64::MAX
        );
    }

    #[test]
    fn bytes_that_are_not_a_tiff_are_refused() {
        for bytes in [
            &b"GIF89a\x01\x00\x01\x00"[..],
            b"PK\x03\x04aaaaaaaa",
            b"XXXXXXXXXXXX",
            &[0u8; 64],
        ] {
            assert_eq!(
                parse_index(&[Piece { offset: 0, bytes }], None),
                Err(TiffError::NotTiff)
            );
        }
        // The right letters and the wrong magic number.
        assert_eq!(
            parse_index(
                &[Piece {
                    offset: 0,
                    bytes: b"II\x2b\x00\x08\x00\x00\x00"
                }],
                None
            ),
            Err(TiffError::BigTiff)
        );
        assert_eq!(
            parse_index(
                &[Piece {
                    offset: 0,
                    bytes: b"MM\x00\x2b\x00\x00\x00\x08"
                }],
                None
            ),
            Err(TiffError::BigTiff)
        );
        assert_eq!(
            parse_index(
                &[Piece {
                    offset: 0,
                    bytes: b"II\x2c\x00\x08\x00\x00\x00"
                }],
                None
            ),
            Err(TiffError::NotTiff)
        );
        // "II" with the byte order of the other kind: magic 42 read the wrong way is 10752.
        assert_eq!(
            parse_index(
                &[Piece {
                    offset: 0,
                    bytes: b"MM\x2a\x00\x08\x00\x00\x00"
                }],
                None
            ),
            Err(TiffError::NotTiff)
        );
    }

    #[test]
    fn a_first_directory_that_is_missing_or_inside_the_header_is_malformed() {
        let mut file = sample(&miniature()).bytes;
        for bad in [0u32, 4, 7] {
            patch32(&mut file, 4, bad, false);
            assert!(
                matches!(
                    parse_index(
                        &[Piece {
                            offset: 0,
                            bytes: &file
                        }],
                        None
                    ),
                    Err(TiffError::Malformed(_))
                ),
                "{bad}"
            );
        }
        // Past the end of a file of known length.
        patch32(&mut file, 4, 0xFFFF_FFF0, false);
        assert_eq!(
            parse_index(
                &[Piece {
                    offset: 0,
                    bytes: &file
                }],
                Some(file.len() as u64)
            ),
            Err(TiffError::Truncated)
        );
        // Past the end of bytes that may be a prefix: it is asked for, in a range that does not overflow.
        let Ok(Parse::Need(range)) = parse_index(
            &[Piece {
                offset: 0,
                bytes: &file,
            }],
            None,
        ) else {
            panic!("a far directory is fetched")
        };
        assert_eq!(range.offset, 0xFFFF_FFF0);
    }

    #[test]
    fn directories_that_loop_are_refused() {
        let mut file = sample(&miniature()).bytes;
        let n = u16::from_le_bytes([file[8], file[9]]) as usize;
        // The first directory's next pointer points back at itself.
        patch32(&mut file, 8 + 2 + 12 * n, 8, false);
        assert!(matches!(parse_all(&file), Err(TiffError::Malformed(why)) if why.contains("loop")));
        // The second points back at the first.
        let mut file = sample(&miniature()).bytes;
        let first_next = {
            let n = u16::from_le_bytes([file[8], file[9]]) as usize;
            u32::from_le_bytes(file[8 + 2 + 12 * n..8 + 2 + 12 * n + 4].try_into().unwrap())
                as usize
        };
        let n2 = u16::from_le_bytes([file[first_next], file[first_next + 1]]) as usize;
        patch32(&mut file, first_next + 2 + 12 * n2, 8, false);
        assert!(matches!(parse_all(&file), Err(TiffError::Malformed(_))));
    }

    #[test]
    fn too_many_directories_are_refused() {
        let file = sample(&TiffSpec::new(64, 64, 64, MAX_DIRECTORIES + 2));
        assert!(matches!(
            parse_all(&file.bytes),
            Err(TiffError::TooLarge(_))
        ));
        // The most the reader takes still works.
        let file = sample(&TiffSpec::new(64, 64, 64, MAX_DIRECTORIES - 1));
        assert_eq!(
            parse_all(&file.bytes).unwrap().levels.len(),
            MAX_DIRECTORIES
        );
    }

    #[test]
    fn a_directory_with_no_entries_or_too_many_is_refused() {
        let mut file = sample(&miniature()).bytes;
        patch16(&mut file, 8, 0, false);
        assert!(matches!(parse_all(&file), Err(TiffError::Malformed(_))));
        for count in [(MAX_ENTRIES + 1) as u16, 65535] {
            let mut file = sample(&miniature()).bytes;
            patch16(&mut file, 8, count, false);
            assert!(
                matches!(parse_all(&file), Err(TiffError::TooLarge(_))),
                "{count}"
            );
        }
        // A count that fits the cap but not the file.
        let mut file = sample(&miniature()).bytes;
        patch16(&mut file, 8, 200, false);
        file.truncate(400);
        assert_eq!(parse_all(&file), Err(TiffError::Truncated));
    }

    fn with_entries(
        edit: impl Fn(usize, &mut Vec<TiffEntry>) + 'static,
    ) -> Result<TiffIndex, TiffError> {
        parse_all(&sample(&miniature().editing_entries(edit)).bytes)
    }

    fn set(entries: &mut [TiffEntry], tag: u16, kind: u16, values: &[u32]) {
        let e = entries.iter_mut().find(|e| e.tag == tag).expect("tag");
        e.kind = kind;
        e.values = values.to_vec();
    }

    fn remove(entries: &mut Vec<TiffEntry>, tag: u16) {
        entries.retain(|e| e.tag != tag);
    }

    #[test]
    fn what_the_reader_does_not_read_is_named() {
        let cases: Vec<(&str, Box<dyn Fn(usize, &mut Vec<TiffEntry>)>)> = vec![
            ("LZW", Box::new(|_, e| set(e, 259, 3, &[5]))),
            ("no compression", Box::new(|_, e| set(e, 259, 3, &[1]))),
            ("JPEG", Box::new(|_, e| set(e, 259, 3, &[7]))),
            ("a predictor", Box::new(|_, e| set(e, 317, 3, &[2]))),
            ("16 bits", Box::new(|_, e| set(e, 258, 3, &[16]))),
            ("1 bit", Box::new(|_, e| set(e, 258, 3, &[1]))),
            ("three samples", Box::new(|_, e| set(e, 277, 3, &[3]))),
            ("planar", Box::new(|_, e| set(e, 284, 3, &[2]))),
            (
                "float samples",
                Box::new(|_, e| e.push(TiffEntry::new(339, 3, &[3]))),
            ),
            (
                "strips",
                Box::new(|_, e| e.push(TiffEntry::new(273, 4, &[8]))),
            ),
            ("no compression tag", Box::new(|_, e| remove(e, 259))),
            (
                "not tiled",
                Box::new(|_, e| {
                    remove(e, 322);
                    remove(e, 323);
                }),
            ),
        ];
        for (name, edit) in cases {
            let r = parse_all(&sample(&miniature().editing_entries(edit)).bytes);
            assert!(matches!(r, Err(TiffError::Unsupported(_))), "{name}: {r:?}");
        }
    }

    #[test]
    fn the_old_deflate_code_is_accepted_and_the_samples_may_say_unsigned() {
        let r = with_entries(|_, e| {
            set(e, 259, 3, &[32946]);
            e.push(TiffEntry::new(339, 3, &[1]));
        });
        assert!(r.is_ok(), "{r:?}");
    }

    #[test]
    fn malformed_tags_are_refused() {
        let missing = |tag: u16| with_entries(move |_, e| remove(e, tag));
        for tag in [256, 257, 324, 325] {
            assert!(
                matches!(missing(tag), Err(TiffError::Malformed(_))),
                "tag {tag}"
            );
        }
        assert!(matches!(
            with_entries(|_, e| e.push(TiffEntry::new(256, 4, &[5]))),
            Err(TiffError::Malformed(_))
        ));
        // A size tag with two values, or of a type that is no number.
        assert!(matches!(
            with_entries(|_, e| set(e, 256, 4, &[5, 6])),
            Err(TiffError::Malformed(_))
        ));
        assert!(matches!(
            with_entries(|_, e| set(e, 256, 2, &[5])),
            Err(TiffError::Malformed(_))
        ));
        // Offsets of the wrong type, or too few.
        assert!(matches!(
            with_entries(|_, e| set(e, 324, 3, &[8; 6])),
            Err(TiffError::Malformed(_))
        ));
        assert!(matches!(
            with_entries(|_, e| set(e, 324, 4, &[8; 5])),
            Err(TiffError::Malformed(_))
        ));
        assert!(matches!(
            with_entries(|_, e| set(e, 325, 4, &[8; 7])),
            Err(TiffError::Malformed(_))
        ));
    }

    #[test]
    fn zero_and_oversized_dimensions_are_refused() {
        for (tag, value) in [(256u16, 0u32), (257, 0), (322, 0), (323, 0)] {
            let r = with_entries(move |_, e| set(e, tag, 4, &[value]));
            assert!(
                matches!(r, Err(TiffError::Malformed(_))),
                "{tag}={value}: {r:?}"
            );
        }
        for (tag, value) in [
            (256u16, MAX_DIMENSION + 1),
            (257, u32::MAX),
            (322, MAX_TILE_SIDE + 1),
            (323, u32::MAX),
        ] {
            let r = with_entries(move |_, e| set(e, tag, 4, &[value]));
            assert!(
                matches!(r, Err(TiffError::TooLarge(_))),
                "{tag}={value}: {r:?}"
            );
        }
    }

    #[test]
    fn a_level_with_too_many_tiles_is_refused_before_anything_is_allocated() {
        // 1,048,576 by 1,048,576 pixels in 16-pixel tiles is 4 billion tiles.
        let r = with_entries(|_, e| {
            set(e, 256, 4, &[MAX_DIMENSION]);
            set(e, 257, 4, &[MAX_DIMENSION]);
            set(e, 322, 4, &[16]);
            set(e, 323, 4, &[16]);
        });
        assert!(matches!(r, Err(TiffError::TooLarge(_))), "{r:?}");
        // Exactly the cap per level passes the size check (the arrays then disagree with it).
        let r = with_entries(|_, e| {
            set(e, 256, 4, &[256 * 16]);
            set(e, 257, 4, &[256 * 16]);
            set(e, 322, 4, &[16]);
            set(e, 323, 4, &[16]);
        });
        assert!(matches!(r, Err(TiffError::Malformed(_))), "{r:?}");
    }

    #[test]
    fn a_file_with_too_many_tiles_in_all_is_refused() {
        // Five levels of the per-level cap each (the arrays are real, so each directory reads whole).
        let r = parse_all(
            &sample(&TiffSpec::new(600, 400, 256, 4).editing_entries(|_, e| {
                set(e, 256, 4, &[4096]);
                set(e, 257, 4, &[4096]);
                set(e, 322, 4, &[16]);
                set(e, 323, 4, &[16]);
                set(e, 324, 4, &vec![8; MAX_TILES_PER_LEVEL as usize]);
                set(e, 325, 4, &vec![100; MAX_TILES_PER_LEVEL as usize]);
            }))
            .bytes,
        );
        assert!(
            matches!(r, Err(TiffError::TooLarge(why)) if why.contains("in the file")),
            "{r:?}"
        );
    }

    #[test]
    fn a_later_level_larger_than_the_one_before_is_refused() {
        let r = with_entries(|level, e| {
            if level == 1 {
                // 601 pixels in two 301-pixel tiles: as many tiles as the arrays hold, but wider than level 0.
                set(e, 256, 4, &[601]);
                set(e, 322, 4, &[301]);
            }
        });
        assert!(matches!(r, Err(TiffError::Unsupported(_))), "{r:?}");
    }

    #[test]
    fn arrays_that_point_outside_the_file_or_into_the_header_are_refused() {
        for big in [false, true] {
            let spec = if big {
                miniature().big_endian()
            } else {
                miniature()
            };
            let file = sample(&spec).bytes;
            let at = entry_at(&file, 8, 324, big) + 8;
            let mut inside = file.clone();
            patch32(&mut inside, at, 4, big);
            assert!(matches!(parse_all(&inside), Err(TiffError::Malformed(_))));
            let mut beyond = file.clone();
            patch32(&mut beyond, at, u32::MAX - 3, big);
            assert_eq!(parse_all(&beyond), Err(TiffError::Truncated));
            let mut off_end = file.clone();
            patch32(&mut off_end, at, file.len() as u32 - 10, big);
            assert_eq!(parse_all(&off_end), Err(TiffError::Truncated));
        }
    }

    #[test]
    fn a_lying_count_in_a_directory_entry_never_sizes_an_allocation() {
        let mut file = sample(&miniature()).bytes;
        let at = entry_at(&file, 8, 324, false);
        patch32(&mut file, at + 4, u32::MAX, false);
        assert!(matches!(parse_all(&file), Err(TiffError::Malformed(_))));
        let mut file = sample(&miniature()).bytes;
        let at = entry_at(&file, 8, 256, false);
        patch32(&mut file, at + 4, u32::MAX, false);
        assert!(matches!(parse_all(&file), Err(TiffError::Malformed(_))));
    }

    #[test]
    fn tile_ranges_are_checked() {
        let sizes = |offset: u32, count: u32| {
            let spec = miniature().editing_tiles(move |level, offsets, counts| {
                if level == 0 {
                    offsets[1] = offset;
                    counts[1] = count;
                }
            });
            let index = parse_all(&sample(&spec).bytes).unwrap();
            index.levels[0].tile_range(1)
        };
        // A tile with no bytes is a tile the file leaves out.
        assert_eq!(sizes(0, 0), Ok(None));
        assert_eq!(sizes(500, 0), Ok(None));
        assert_eq!(
            sizes(500, 100),
            Ok(Some(ByteRange {
                offset: 500,
                len: 100
            }))
        );
        assert!(matches!(sizes(3, 100), Err(TiffError::Malformed(_))));
        assert!(matches!(sizes(0, 100), Err(TiffError::Malformed(_))));
        assert!(matches!(sizes(500, u32::MAX), Err(TiffError::TooLarge(_))));
        assert!(
            matches!(sizes(u32::MAX, 100), Ok(Some(r)) if r.end() == u64::from(u32::MAX) + 100)
        );
        let index = parse_all(&sample(&miniature()).bytes).unwrap();
        assert_eq!(index.levels[0].tile_range(6), Err(TiffError::NoSuchTile));
        assert_eq!(
            index.levels[0].tile_range(usize::MAX),
            Err(TiffError::NoSuchTile)
        );
        assert_eq!(index.levels[0].tile_index(3, 0), None);
        assert_eq!(index.levels[0].tile_index(0, 2), None);
        assert_eq!(index.levels[0].tile_index(2, 1), Some(5));
    }

    fn level_of(file: &SyntheticTiff) -> Level {
        parse_all(&file.bytes).unwrap().levels.remove(0)
    }

    fn deflate(raw: &[u8]) -> Vec<u8> {
        miniz_oxide::deflate::compress_to_vec_zlib(raw, 6)
    }

    #[test]
    fn bad_tile_bytes_are_refused() {
        let level = level_of(&sample(&miniature()));
        let want = level.tile_pixels();
        assert!(matches!(level.decode_tile(&[]), Err(TiffError::BadTile(_))));
        assert!(matches!(
            level.decode_tile(b"not zlib at all"),
            Err(TiffError::BadTile(_))
        ));
        assert!(matches!(
            level.decode_tile(&[0x78, 0x9c]),
            Err(TiffError::BadTile(_))
        ));
        let good = deflate(&vec![7u8; want]);
        assert_eq!(level.decode_tile(&good).unwrap(), vec![7u8; want]);
        // Cut short, or with a damaged checksum.
        assert!(matches!(
            level.decode_tile(&good[..good.len() / 2]),
            Err(TiffError::BadTile(_))
        ));
        let mut flipped = good.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 0xff;
        assert!(matches!(
            level.decode_tile(&flipped),
            Err(TiffError::BadTile(_))
        ));
        // A different number of pixels than the tile has: fewer, or more.
        assert!(matches!(
            level.decode_tile(&deflate(&vec![1u8; want - 1])),
            Err(TiffError::BadTile(_))
        ));
        assert!(matches!(
            level.decode_tile(&deflate(&vec![1u8; want + 1])),
            Err(TiffError::BadTile(_))
        ));
        assert!(matches!(
            level.decode_tile(&deflate(&[])),
            Err(TiffError::BadTile(_))
        ));
    }

    #[test]
    fn a_decompression_bomb_gives_an_error_not_a_huge_buffer() {
        let level = level_of(&sample(&miniature()));
        // 8 MiB of zeros compresses to a few kilobytes; the tile is allowed 64 KiB of pixels.
        let bomb = deflate(&vec![0u8; 8 << 20]);
        assert!(
            bomb.len() as u64 <= level.max_tile_bytes(),
            "the test bomb is small"
        );
        assert!(level.decode_tile(&bomb).is_err());
        // Over the compressed cap: refused unread.
        let big = vec![0x55u8; level.max_tile_bytes() as usize + 1];
        assert!(matches!(
            level.decode_tile(&big),
            Err(TiffError::TooLarge(_))
        ));
    }

    #[test]
    fn damaged_files_never_panic_and_never_ask_for_more_than_they_say() {
        let file = sample(&miniature().big_endian());
        let index = &file.bytes[..file.index_len];
        // Every byte of the index in turn, in every damaged form.
        for at in 0..index.len() {
            for flip in [0xffu8, 0x01, 0x80] {
                let mut damaged = index.to_vec();
                damaged[at] ^= flip;
                let _ = parse_index(
                    &[Piece {
                        offset: 0,
                        bytes: &damaged,
                    }],
                    None,
                );
                let _ = parse_index(
                    &[Piece {
                        offset: 0,
                        bytes: &damaged,
                    }],
                    Some(damaged.len() as u64),
                );
            }
        }
        // And little-endian too, with every byte set to the extremes.
        let file = sample(&miniature());
        let index = &file.bytes[..file.index_len];
        for at in 0..index.len() {
            for value in [0u8, 0xff, 0x7f] {
                let mut damaged = index.to_vec();
                damaged[at] = value;
                let _ = parse_index(
                    &[Piece {
                        offset: 0,
                        bytes: &damaged,
                    }],
                    None,
                );
            }
        }
        // Every truncation.
        for end in 0..index.len() {
            let _ = parse_index(
                &[Piece {
                    offset: 0,
                    bytes: &index[..end],
                }],
                None,
            );
            let _ = parse_index(
                &[Piece {
                    offset: 0,
                    bytes: &index[..end],
                }],
                Some(end as u64),
            );
        }
    }

    #[test]
    fn random_bytes_never_panic() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..400 {
            let len = (next() % 600) as usize;
            let mut bytes: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            // Half of them start like a TIFF, so the parser goes further than the header.
            if round % 2 == 0 && bytes.len() >= 8 {
                bytes[..4].copy_from_slice(if round % 4 == 0 {
                    b"II*\x00"
                } else {
                    b"MM\x00*"
                });
                let at = (next() % 200 + 8) as u32;
                let word = if round % 4 == 0 {
                    at.to_le_bytes()
                } else {
                    at.to_be_bytes()
                };
                bytes[4..8].copy_from_slice(&word);
            }
            let held = [Piece {
                offset: 0,
                bytes: &bytes,
            }];
            let _ = parse_index(&held, None);
            let _ = parse_index(&held, Some(bytes.len() as u64));
            // As tile bytes too.
            let _ = crate::testing::empty_level(256, 256).decode_tile(&bytes);
        }
    }

    #[test]
    fn a_single_tile_file_keeps_its_offset_in_the_entry() {
        // One tile: the offset and the byte count are held in the entry itself, not in an array.
        for big in [false, true] {
            let spec = TiffSpec::new(100, 100, 128, 0);
            let spec = if big { spec.big_endian() } else { spec };
            let file = sample(&spec);
            let index = parse_all(&file.bytes).unwrap();
            let level = &index.levels[0];
            assert_eq!(level.tile_count(), 1);
            let range = level.tile_range(0).unwrap().unwrap();
            let pixels = level.decode_tile(serve_range(&file.bytes, range)).unwrap();
            expect_pixels(level, 0, 0, 0, &pixels);
        }
    }

    #[test]
    fn byte_counts_may_be_shorts() {
        let spec = miniature().editing_entries(|level, e| {
            if level == 0 {
                // Placeholder counts that fit in 16 bits; the real ones are put in by hand below.
                set(e, 325, 3, &[100, 200, 300, 400, 500, 600]);
            }
        });
        let index = parse_all(&sample(&spec).bytes).unwrap();
        assert_eq!(index.levels[0].tile_range(2).unwrap().unwrap().len, 300);
    }
}

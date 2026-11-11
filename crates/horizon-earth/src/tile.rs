//! Tile identities and the decoding of terrarium elevation tiles.
//!
//! A terrarium tile is a 256×256 PNG whose colour channels hold the elevation in metres:
//! `R·256 + G + B/256 − 32768`.

use std::fmt;
use std::io::Cursor;

/// Pixels along one side of a tile.
pub const TILE_SIZE: u32 = 256;
/// The finest zoom the dataset has.
pub const MAX_ZOOM: u8 = 15;
/// Cells along one side of a tile's coarse min/max grid.
pub const GRID: usize = 16;
/// Pixels along one side of a grid cell.
pub const CELL: usize = TILE_SIZE as usize / GRID;

/// A Web Mercator tile: zoom `z`, column `x` counted from the west, row `y` counted from the north.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

/// Why a tile identity is not one the dataset has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileIdError {
    ZoomTooHigh(u8),
    OutOfRange(TileId),
}

impl fmt::Display for TileIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZoomTooHigh(z) => {
                write!(f, "zoom {z} is above the dataset's maximum of {MAX_ZOOM}")
            }
            Self::OutOfRange(id) => write!(
                f,
                "tile {}/{}/{} is outside its zoom's grid",
                id.z, id.x, id.y
            ),
        }
    }
}

impl std::error::Error for TileIdError {}

impl TileId {
    /// A tile, checked: the zoom is 0 to 15 and `x` and `y` are below 2^zoom.
    pub fn new(z: u8, x: u32, y: u32) -> Result<Self, TileIdError> {
        let id = Self { z, x, y };
        id.validate().map(|()| id)
    }

    /// Whether this is a tile the dataset has.
    pub fn validate(self) -> Result<(), TileIdError> {
        if self.z > MAX_ZOOM {
            return Err(TileIdError::ZoomTooHigh(self.z));
        }
        let side = 1u32 << self.z;
        if self.x >= side || self.y >= side {
            return Err(TileIdError::OutOfRange(self));
        }
        Ok(())
    }

    /// The tile one zoom out that contains this one; `None` at zoom 0.
    pub fn parent(self) -> Option<Self> {
        (self.z > 0).then(|| Self {
            z: self.z - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    /// The four tiles one zoom in that this one is made of (north-west, north-east, south-west, south-east);
    /// `None` at the finest zoom.
    pub fn children(self) -> Option<[Self; 4]> {
        if self.z >= MAX_ZOOM {
            return None;
        }
        let (z, x, y) = (self.z + 1, self.x * 2, self.y * 2);
        Some([
            Self { z, x, y },
            Self { z, x: x + 1, y },
            Self { z, x, y: y + 1 },
            Self {
                z,
                x: x + 1,
                y: y + 1,
            },
        ])
    }
}

/// Decodes one terrarium pixel to metres.
pub fn decode_elevation(r: u8, g: u8, b: u8) -> f32 {
    // 24 bits scaled by 1/256 are exact in an f32, so the formula loses nothing.
    let raw = (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
    raw as f32 / 256.0 - 32768.0
}

/// Why a PNG is not a usable terrarium tile.
#[derive(Debug)]
pub enum TileError {
    /// The bytes are not a PNG the decoder can read.
    Png(png::DecodingError),
    /// The image is not 256×256 pixels.
    WrongSize { width: u32, height: u32 },
    /// The image is not 8-bit RGB or RGBA.
    WrongFormat {
        color: png::ColorType,
        depth: png::BitDepth,
    },
}

impl fmt::Display for TileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Png(e) => write!(f, "not a readable PNG: {e}"),
            Self::WrongSize { width, height } => write!(
                f,
                "a tile is {TILE_SIZE}×{TILE_SIZE} pixels, this is {width}×{height}"
            ),
            Self::WrongFormat { color, depth } => {
                write!(
                    f,
                    "a terrarium tile is 8-bit RGB or RGBA, this is {depth:?} {color:?}"
                )
            }
        }
    }
}

impl std::error::Error for TileError {}

impl From<png::DecodingError> for TileError {
    fn from(e: png::DecodingError) -> Self {
        Self::Png(e)
    }
}

/// A decoded tile: 256×256 heights in metres, their extremes, and a coarse grid of extremes for fast range queries.
#[derive(Debug)]
pub struct Tile {
    heights: Vec<f32>,
    min: f32,
    max: f32,
    cell_min: [f32; GRID * GRID],
    cell_max: [f32; GRID * GRID],
    repaired: usize,
}

impl Tile {
    /// Decodes the bytes of a terrarium PNG (8-bit RGB or RGBA, 256×256) taken at zoom `zoom`; anything else is
    /// an error. Implausible pixels are repaired first ([`crate::clean`]), so the extremes, the coarse grid and
    /// every reader see cleaned data; the zoom scales what counts as implausible.
    pub fn decode(png_bytes: &[u8], zoom: u8) -> Result<Self, TileError> {
        let mut reader = png::Decoder::new(Cursor::new(png_bytes)).read_info()?;
        let info = reader.info();
        let (width, height) = (info.width, info.height);
        let (color, depth) = (info.color_type, info.bit_depth);
        if (width, height) != (TILE_SIZE, TILE_SIZE) {
            return Err(TileError::WrongSize { width, height });
        }
        let channels = match (color, depth) {
            (png::ColorType::Rgb, png::BitDepth::Eight) => 3,
            (png::ColorType::Rgba, png::BitDepth::Eight) => 4,
            _ => return Err(TileError::WrongFormat { color, depth }),
        };
        let mut data = vec![
            0;
            reader
                .output_buffer_size()
                .ok_or(TileError::WrongSize { width, height })?
        ];
        let frame = reader.next_frame(&mut data)?;
        let mut heights: Vec<f32> = data[..frame.buffer_size()]
            .chunks_exact(channels)
            .map(|p| decode_elevation(p[0], p[1], p[2]))
            .collect();
        let repaired = crate::clean::clean_tile(&mut heights, zoom);
        let mut tile = Self::from_heights(heights);
        tile.repaired = repaired;
        Ok(tile)
    }

    /// How many pixels the decoder repaired as implausible.
    pub fn repaired(&self) -> usize {
        self.repaired
    }

    fn from_heights(heights: Vec<f32>) -> Self {
        debug_assert_eq!(heights.len(), (TILE_SIZE * TILE_SIZE) as usize);
        let side = TILE_SIZE as usize;
        let mut cell_min = [f32::INFINITY; GRID * GRID];
        let mut cell_max = [f32::NEG_INFINITY; GRID * GRID];
        for (i, &h) in heights.iter().enumerate() {
            let cell = (i / side / CELL) * GRID + (i % side) / CELL;
            cell_min[cell] = cell_min[cell].min(h);
            cell_max[cell] = cell_max[cell].max(h);
        }
        let min = cell_min.iter().copied().fold(f32::INFINITY, f32::min);
        let max = cell_max.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Self {
            heights,
            min,
            max,
            cell_min,
            cell_max,
            repaired: 0,
        }
    }

    /// The height in metres of pixel `(px, py)`, counted from the tile's north-west corner.
    ///
    /// # Panics
    /// If either coordinate is 256 or more.
    pub fn height(&self, px: usize, py: usize) -> f32 {
        assert!(
            px < TILE_SIZE as usize && py < TILE_SIZE as usize,
            "pixel ({px}, {py}) is outside the tile"
        );
        self.heights[py * TILE_SIZE as usize + px]
    }

    /// The lowest height in the tile, in metres.
    pub fn min(&self) -> f32 {
        self.min
    }

    /// The highest height in the tile, in metres.
    pub fn max(&self) -> f32 {
        self.max
    }

    /// The lowest and highest height among the pixels `x0..=x1` by `y0..=y1` (clamped to the tile), reading
    /// whole grid cells where the rectangle covers them. `None` when the rectangle is empty.
    pub fn range(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Option<(f32, f32)> {
        let last = TILE_SIZE as usize - 1;
        let (x1, y1) = (x1.min(last), y1.min(last));
        if x0 > x1 || y0 > y1 {
            return None;
        }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for cy in y0 / CELL..=y1 / CELL {
            for cx in x0 / CELL..=x1 / CELL {
                let (cell_x0, cell_y0) = (cx * CELL, cy * CELL);
                let (cell_x1, cell_y1) = (cell_x0 + CELL - 1, cell_y0 + CELL - 1);
                if x0 <= cell_x0 && cell_x1 <= x1 && y0 <= cell_y0 && cell_y1 <= y1 {
                    lo = lo.min(self.cell_min[cy * GRID + cx]);
                    hi = hi.max(self.cell_max[cy * GRID + cx]);
                    continue;
                }
                for py in y0.max(cell_y0)..=y1.min(cell_y1) {
                    for px in x0.max(cell_x0)..=x1.min(cell_x1) {
                        let h = self.heights[py * TILE_SIZE as usize + px];
                        lo = lo.min(h);
                        hi = hi.max(h);
                    }
                }
            }
        }
        Some((lo, hi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{encode_height, encode_png};

    fn tile_from(f: impl Fn(usize, usize) -> f64) -> Tile {
        let mut rgb = Vec::new();
        for y in 0..256 {
            for x in 0..256 {
                rgb.extend(encode_height(f(x, y)));
            }
        }
        Tile::decode(
            &encode_png(256, 256, png::ColorType::Rgb, png::BitDepth::Eight, &rgb),
            0,
        )
        .unwrap()
    }

    #[test]
    fn the_terrarium_formula_matches_the_published_examples() {
        assert_eq!(decode_elevation(128, 0, 0), 0.0);
        assert_eq!(decode_elevation(0, 0, 0), -32768.0);
        // 127·256 + 244 + 128/256 − 32768 = −11.5: below sea level, with a fraction.
        assert_eq!(decode_elevation(127, 244, 128), -11.5);
        assert_eq!(decode_elevation(129, 0, 255), 256.0 + 255.0 / 256.0);
    }

    #[test]
    fn heights_survive_encoding_and_decoding_to_a_256th_of_a_metre() {
        let samples = [
            -10_994.25,
            -32.0,
            -0.00390625,
            0.0,
            0.5,
            1.0 / 3.0,
            8848.86,
            30_000.125,
        ];
        let tile = tile_from(|x, _| samples[x % samples.len()]);
        for (x, want) in samples.iter().enumerate() {
            let got = f64::from(tile.height(x, 7));
            assert!(
                (got - want).abs() <= 1.0 / 512.0 + 1e-9,
                "{want} came back as {got}"
            );
        }
    }

    #[test]
    fn a_tile_knows_its_extremes_and_those_of_each_region() {
        // A ramp west to east, with one deep pit and one tall spike.
        let tile = tile_from(|x, y| match (x, y) {
            (200, 30) => -500.0,
            (17, 250) => 4000.0,
            _ => x as f64,
        });
        assert_eq!(tile.min(), -500.0);
        assert_eq!(tile.max(), 4000.0);
        assert_eq!(tile.range(0, 0, 255, 255), Some((-500.0, 4000.0)));
        // A whole-cell rectangle away from both: only the ramp.
        assert_eq!(tile.range(32, 64, 63, 127), Some((32.0, 63.0)));
        // A ragged rectangle that clips the spike's cell but not the spike.
        assert_eq!(tile.range(20, 240, 40, 255), Some((20.0, 40.0)));
        // And one that holds the spike alone.
        assert_eq!(tile.range(17, 250, 17, 250), Some((4000.0, 4000.0)));
        assert_eq!(tile.range(10, 10, 9, 20), None);
    }

    #[test]
    fn range_agrees_with_a_scan_of_every_pixel() {
        let tile = tile_from(|x, y| ((x * 31 + y * 17) % 1000) as f64 - 300.0);
        for &(x0, y0, x1, y1) in &[
            (0, 0, 255, 255),
            (5, 9, 100, 200),
            (16, 16, 31, 31),
            (15, 15, 32, 32),
            (100, 100, 100, 100),
        ] {
            let mut want = (f32::INFINITY, f32::NEG_INFINITY);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    want = (want.0.min(tile.height(x, y)), want.1.max(tile.height(x, y)));
                }
            }
            assert_eq!(
                tile.range(x0, y0, x1, y1),
                Some(want),
                "rect {x0},{y0}..{x1},{y1}"
            );
        }
    }

    #[test]
    fn an_rgba_tile_decodes_the_same_as_an_rgb_one() {
        let mut rgba = Vec::new();
        for i in 0..256 * 256 {
            let [r, g, b] = encode_height(i as f64 / 7.0 - 100.0);
            rgba.extend([r, g, b, 255]);
        }
        let tile = Tile::decode(
            &encode_png(256, 256, png::ColorType::Rgba, png::BitDepth::Eight, &rgba),
            0,
        )
        .unwrap();
        let [r, g, b] = encode_height(5.0 / 7.0 - 100.0);
        let want = decode_elevation(r, g, b);
        assert_eq!(tile.height(5, 0), want);
    }

    #[test]
    fn a_png_that_is_not_an_8_bit_rgb_tile_is_refused_with_a_reason() {
        let grey = encode_png(
            256,
            256,
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            &vec![0; 256 * 256],
        );
        assert!(matches!(
            Tile::decode(&grey, 0),
            Err(TileError::WrongFormat { .. })
        ));
        let deep = encode_png(
            256,
            256,
            png::ColorType::Rgb,
            png::BitDepth::Sixteen,
            &vec![0; 256 * 256 * 6],
        );
        assert!(matches!(
            Tile::decode(&deep, 0),
            Err(TileError::WrongFormat { .. })
        ));
        let small = encode_png(
            64,
            64,
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &vec![0; 64 * 64 * 3],
        );
        let err = Tile::decode(&small, 0).unwrap_err();
        assert!(matches!(
            err,
            TileError::WrongSize {
                width: 64,
                height: 64
            }
        ));
        assert!(err.to_string().contains("64×64"), "{err}");
        assert!(matches!(
            Tile::decode(b"definitely not a png", 0),
            Err(TileError::Png(_))
        ));
        assert!(matches!(Tile::decode(&[], 0), Err(TileError::Png(_))));
    }

    #[test]
    fn tile_ids_know_their_parents_and_children() {
        let id = TileId::new(5, 17, 9).unwrap();
        assert_eq!(id.parent(), Some(TileId { z: 4, x: 8, y: 4 }));
        assert_eq!(TileId { z: 0, x: 0, y: 0 }.parent(), None);
        let kids = id.children().unwrap();
        assert_eq!(kids[0], TileId { z: 6, x: 34, y: 18 });
        assert_eq!(kids[3], TileId { z: 6, x: 35, y: 19 });
        assert!(kids.iter().all(|k| k.parent() == Some(id)));
        assert_eq!(TileId { z: 15, x: 0, y: 0 }.children(), None);
    }

    #[test]
    fn tile_ids_outside_the_dataset_are_invalid() {
        assert!(TileId::new(0, 0, 0).is_ok());
        assert!(TileId::new(15, 32_767, 32_767).is_ok());
        assert_eq!(TileId::new(16, 0, 0), Err(TileIdError::ZoomTooHigh(16)));
        assert!(matches!(
            TileId::new(3, 8, 0),
            Err(TileIdError::OutOfRange(_))
        ));
        assert!(matches!(
            TileId::new(0, 0, 1),
            Err(TileIdError::OutOfRange(_))
        ));
    }
}

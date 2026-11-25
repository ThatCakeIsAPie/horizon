//! Ore richness from real data: how much richer or poorer in an ore a place on Earth is than the average place.
//!
//! The data is a global grid shipped inside the binary (`data/ore_richness.bin`, made by
//! `tools/earth_ore_richness.py`): one byte per ore per cell, holding the base-2 logarithm of the richness. It is
//! decoded once, on first use, and a lookup is the Catmull-Rom bicubic mix of the 4 by 4 nearest cell centres taken in
//! that logarithmic space, so the answer is smooth in latitude and longitude and shows no grid edges.
//!
//! # Asset format
//!
//! All integers are little endian. A header of the magic `HZOR`, a `u32` version (1), `u16` columns and rows, `f32`
//! west longitude, north latitude and cell size in degrees, then a `u8` ore count followed by each ore's name (a `u8`
//! length and ASCII bytes), and a `u32` raw length. The rest is one zlib stream of that many bytes: ore by ore, rows
//! from north to south, columns from west to east. Each row is delta coded (a cell holds its difference from the
//! cell to its west, modulo 256; the first holds itself). A value `q` means richness `2^((q - 128) / 32)`, so 128 is
//! an average place (and also the ocean and anywhere with no information).
//!
//! The grid is taken to wrap around in longitude and to stop at its first and last row in latitude. A lookup never
//! fails: an ore the grid does not name, a place that is not a number, or an asset that cannot be decoded all answer 1.

use std::sync::OnceLock;

use miniz_oxide::inflate::decompress_to_vec_zlib_with_limit;

/// The version of the richness data and rules (part of an Earth world's
/// cache key when its ores follow the map).
pub const ORE_RICHNESS_RULES: u32 = 1;

/// The smallest richness a lookup answers.
const MIN_RICHNESS: f64 = 0.3;
/// The largest richness a lookup answers.
const MAX_RICHNESS: f64 = 5.0;
/// The stored value that means a richness of exactly 1.
const NEUTRAL_Q: f64 = 128.0;
/// How many stored steps make a doubling of richness.
const Q_PER_OCTAVE: f64 = 32.0;

/// The magic bytes that start the asset.
const MAGIC: &[u8; 4] = b"HZOR";
/// The only asset format version understood.
const FORMAT_VERSION: u32 = 1;

/// The grid shipped with the game.
static ASSET: &[u8] = include_bytes!("../data/ore_richness.bin");

/// The decoded shipped grid, or `None` when it could not be decoded (warned about once).
static GRID: OnceLock<Option<Grid>> = OnceLock::new();

/// The richness multiplier of `ore` (short name: `iron`, `copper`, `coal`,
/// `gold`, `redstone`, `lapis`, `diamond`, `emerald`; anything else = 1) at a
/// place, smooth in latitude and longitude (no grid edges), clamped to
/// 0.3 ..= 5.0. 1 = an average place.
pub fn richness(ore: &str, latitude: f64, longitude: f64) -> f64 {
    let grid = GRID.get_or_init(|| match decode(ASSET) {
        Ok(grid) => Some(grid),
        Err(why) => {
            log::warn!("the ore richness asset cannot be decoded, richness is 1 everywhere: {why}");
            None
        }
    });
    grid.as_ref()
        .map_or(1.0, |grid| grid.sample(ore, latitude, longitude))
}

/// The size in bytes of the embedded asset.
pub fn asset_len() -> usize {
    ASSET.len()
}

/// A decoded richness grid: the stored value of every ore in every cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// How many columns (west to east).
    pub cols: usize,
    /// How many rows (north to south).
    pub rows: usize,
    /// The west edge of the grid in degrees of longitude.
    pub lon_west: f64,
    /// The north edge of the grid in degrees of latitude.
    pub lat_north: f64,
    /// The side of a cell in degrees.
    pub cell: f64,
    /// The names of the ores, in the order of `data`.
    pub names: Vec<String>,
    /// The stored values: ore by ore, rows north to south, columns west to east.
    pub data: Vec<u8>,
}

/// A little endian reader over the asset header.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or("the asset ends inside its header")?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.array()?))
    }
}

/// Decodes an asset (see the module documentation for the format), checking every size it declares.
pub fn decode(bytes: &[u8]) -> Result<Grid, String> {
    let mut r = Reader { bytes, at: 0 };
    if &r.array::<4>()? != MAGIC {
        return Err("not an ore richness asset (bad magic)".into());
    }
    let version = r.u32()?;
    if version != FORMAT_VERSION {
        return Err(format!("unsupported asset version {version}"));
    }
    let cols = usize::from(r.u16()?);
    let rows = usize::from(r.u16()?);
    let lon_west = f64::from(r.f32()?);
    let lat_north = f64::from(r.f32()?);
    let cell = f64::from(r.f32()?);
    if cols == 0 || rows == 0 {
        return Err("the grid is empty".into());
    }
    if !(lon_west.is_finite() && lat_north.is_finite() && cell.is_finite() && cell > 0.0) {
        return Err("the grid placement is not valid".into());
    }
    let count = usize::from(r.u8()?);
    if count == 0 {
        return Err("the asset names no ore".into());
    }
    let mut names = Vec::with_capacity(count);
    for _ in 0..count {
        let len = usize::from(r.u8()?);
        let name = std::str::from_utf8(r.take(len)?).map_err(|_| "an ore name is not text")?;
        names.push(name.to_owned());
    }
    let raw_len = r.u32()? as usize;
    let expected = count
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(cols))
        .ok_or("the grid is too large")?;
    if raw_len != expected {
        return Err(format!(
            "the asset declares {raw_len} values but its grid holds {expected}"
        ));
    }
    let mut data = decompress_to_vec_zlib_with_limit(&bytes[r.at..], raw_len)
        .map_err(|e| format!("the asset's data cannot be inflated: {e:?}"))?;
    if data.len() != raw_len {
        return Err(format!(
            "the asset inflates to {} values, not {raw_len}",
            data.len()
        ));
    }
    for row in data.chunks_exact_mut(cols) {
        let mut sum = 0u8;
        for q in row {
            sum = sum.wrapping_add(*q);
            *q = sum;
        }
    }
    Ok(Grid {
        cols,
        rows,
        lon_west,
        lat_north,
        cell,
        names,
        data,
    })
}

/// The Catmull-Rom spline through `p[1]` (at 0) and `p[2]` (at 1), shaped by its outer neighbours, at `t` in 0 ..= 1.
fn catmull_rom(p: [f64; 4], t: f64) -> f64 {
    let [p0, p1, p2, p3] = p;
    0.5 * (2.0 * p1
        + (p2 - p0) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (3.0 * (p1 - p2) + p3 - p0) * t * t * t)
}

impl Grid {
    /// The richness of `ore` at a place, from the bicubic mix of the 4 by 4 nearest cell centres in log space, clamped
    /// to 0.3 ..= 5. An ore the grid does not name, or a place that is not finite, is 1.
    pub fn sample(&self, ore: &str, latitude: f64, longitude: f64) -> f64 {
        if !(latitude.is_finite() && longitude.is_finite()) {
            return 1.0;
        }
        let Some(index) = self.names.iter().position(|name| name == ore) else {
            return 1.0;
        };
        let size = self.rows * self.cols;
        let plane = &self.data[index * size..][..size];

        // Position in cell units, where whole numbers are cell centres.
        let x = (longitude - self.lon_west) / self.cell - 0.5;
        let y = (self.lat_north - latitude) / self.cell - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);

        let col_of = |offset: f64| (x0 + offset).rem_euclid(self.cols as f64) as usize;
        let row_of = |offset: f64| (y0 + offset).clamp(0.0, (self.rows - 1) as f64) as usize;
        let cols = [col_of(-1.0), col_of(0.0), col_of(1.0), col_of(2.0)];

        let mut column = [0.0; 4];
        for (slot, offset) in column.iter_mut().zip([-1.0, 0.0, 1.0, 2.0]) {
            let row = &plane[row_of(offset) * self.cols..][..self.cols];
            *slot = catmull_rom(cols.map(|c| f64::from(row[c])), tx);
        }
        let q = catmull_rom(column, ty);
        ((q - NEUTRAL_Q) / Q_PER_OCTAVE)
            .exp2()
            .clamp(MIN_RICHNESS, MAX_RICHNESS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use miniz_oxide::deflate::compress_to_vec_zlib;

    const NAMES: [&str; 8] = [
        "iron", "copper", "coal", "gold", "redstone", "lapis", "diamond", "emerald",
    ];

    /// Where the raw length sits in an asset with one ore named `iron`: magic, version, size, placement, count, name length, name.
    const RAW_LEN_AT: usize = 4 + 4 + 2 + 2 + 12 + 1 + 1 + 4;

    /// Builds an asset per the format: `planes[i]` holds ore `i`'s values, rows north to south.
    fn encode(names: &[&str], cols: usize, rows: usize, cell: f32, planes: &[Vec<u8>]) -> Vec<u8> {
        let mut out = b"HZOR".to_vec();
        out.extend(1u32.to_le_bytes());
        out.extend((cols as u16).to_le_bytes());
        out.extend((rows as u16).to_le_bytes());
        out.extend((-180.0f32).to_le_bytes());
        out.extend(90.0f32.to_le_bytes());
        out.extend(cell.to_le_bytes());
        out.push(names.len() as u8);
        for name in names {
            out.push(name.len() as u8);
            out.extend(name.as_bytes());
        }
        let mut raw = Vec::new();
        for plane in planes {
            assert_eq!(plane.len(), cols * rows);
            for row in plane.chunks(cols) {
                let mut previous = 0u8;
                for &q in row {
                    raw.push(q.wrapping_sub(previous));
                    previous = q;
                }
            }
        }
        out.extend((raw.len() as u32).to_le_bytes());
        out.extend(compress_to_vec_zlib(&raw, 6));
        out
    }

    /// A global grid of 8 by 4 cells of 45 degrees holding `value(col, row)` for one ore called `"iron"`.
    fn grid_of(value: impl Fn(usize, usize) -> u8) -> Grid {
        let plane = (0..4)
            .flat_map(|r| (0..8).map(move |c| (c, r)))
            .map(|(c, r)| value(c, r))
            .collect();
        decode(&encode(&["iron"], 8, 4, 45.0, &[plane])).unwrap()
    }

    /// The centre of cell (col, row) of the grids above, as (latitude, longitude).
    fn centre(col: usize, row: usize) -> (f64, f64) {
        (
            90.0 - (row as f64 + 0.5) * 45.0,
            -180.0 + (col as f64 + 0.5) * 45.0,
        )
    }

    fn richness_of_q(q: f64) -> f64 {
        ((q - 128.0) / 32.0).exp2()
    }

    #[test]
    fn decode_round_trips_including_wrapping_deltas() {
        let plane: Vec<u8> = vec![
            250, 5, 5, 0, 255, 1, 128, 129, // wraps up and down
            0, 255, 0, 255, 0, 255, 0, 255, //
            10, 20, 30, 40, 50, 60, 70, 80, //
            200, 100, 50, 25, 12, 6, 3, 1,
        ];
        let copper: Vec<u8> = plane.iter().map(|q| q.wrapping_mul(3)).collect();
        let bytes = encode(
            &["iron", "copper"],
            8,
            4,
            45.0,
            &[plane.clone(), copper.clone()],
        );
        let grid = decode(&bytes).unwrap();
        assert_eq!((grid.cols, grid.rows), (8, 4));
        assert_eq!(
            (grid.lon_west, grid.lat_north, grid.cell),
            (-180.0, 90.0, 45.0)
        );
        assert_eq!(grid.names, ["iron", "copper"]);
        assert_eq!(&grid.data[..32], &plane[..]);
        assert_eq!(&grid.data[32..], &copper[..]);
    }

    #[test]
    fn centre_of_a_cell_gives_its_value_on_a_constant_grid() {
        let grid = grid_of(|_, _| 160);
        for (col, row) in [(0, 0), (3, 2), (7, 3)] {
            let (lat, lon) = centre(col, row);
            let got = grid.sample("iron", lat, lon);
            assert!((got - richness_of_q(160.0)).abs() < 1e-12, "{got}");
        }
    }

    #[test]
    fn centre_of_a_cell_gives_its_value_on_a_ramp() {
        let grid = grid_of(|c, r| (120 + 4 * c - 6 * r) as u8);
        for (col, row) in [(0, 1), (3, 2), (7, 1), (4, 0)] {
            let (lat, lon) = centre(col, row);
            let got = grid.sample("iron", lat, lon);
            let want = richness_of_q((120 + 4 * col - 6 * row) as f64);
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    #[test]
    fn a_linear_ramp_is_reproduced_between_cells() {
        // q grows 4 per column and falls 6 per row.
        let grid = grid_of(|c, r| (120 + 4 * c - 6 * r) as u8);
        let q_at = |lat: f64, lon: f64| {
            let x = (lon + 180.0) / 45.0 - 0.5;
            let y = (90.0 - lat) / 45.0 - 0.5;
            120.0 + 4.0 * x - 6.0 * y
        };
        // Far enough from the seam and the polar rows that no neighbour is wrapped or clamped.
        for (lat, lon) in [(10.0, -30.0), (0.0, 0.0), (20.0, 30.0), (-7.0, 15.5)] {
            let got = grid.sample("iron", lat, lon);
            let want = richness_of_q(q_at(lat, lon));
            assert!((got - want).abs() < 1e-9, "({lat}, {lon}): {got} vs {want}");
        }
    }

    #[test]
    fn samples_are_continuous_across_cell_edges_and_the_seam() {
        let grid = grid_of(|c, r| ((c * 37 + r * 91 + c * r * 11) % 256) as u8);
        let step = 1e-4;
        let mut worst = 0.0f64;
        // Along a latitude, all the way around and over the seam at 180.
        let mut previous = grid.sample("iron", 12.3, -200.0);
        for i in 1..=4_000_000 {
            let now = grid.sample("iron", 12.3, -200.0 + f64::from(i) * step);
            worst = worst.max((now - previous).abs());
            previous = now;
        }
        // Along a meridian, from beyond the north pole to beyond the south.
        let mut previous = grid.sample("iron", 100.0, 77.7);
        for i in 1..=2_000_000 {
            let now = grid.sample("iron", 100.0 - f64::from(i) * step, 77.7);
            worst = worst.max((now - previous).abs());
            previous = now;
        }
        assert!(worst < 1e-2, "largest jump {worst}");
        let seam =
            grid.sample("iron", 40.0, 180.0 - 1e-9) - grid.sample("iron", 40.0, -180.0 + 1e-9);
        assert!(seam.abs() < 1e-6, "{seam}");
    }

    #[test]
    fn results_are_clamped() {
        assert_eq!(grid_of(|_, _| 0).sample("iron", 10.0, 10.0), 0.3);
        assert_eq!(grid_of(|_, _| 255).sample("iron", 10.0, 10.0), 5.0);
    }

    #[test]
    fn unknown_ore_and_bad_places_are_neutral() {
        let grid = grid_of(|_, _| 200);
        assert_eq!(grid.sample("unobtainium", 0.0, 0.0), 1.0);
        assert_eq!(richness("unobtainium", 10.0, 10.0), 1.0);
        for (lat, lon) in [
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
            (f64::INFINITY, 0.0),
            (0.0, f64::NEG_INFINITY),
        ] {
            assert_eq!(grid.sample("iron", lat, lon), 1.0);
            assert_eq!(richness("iron", lat, lon), 1.0);
        }
    }

    #[test]
    fn the_poles_and_far_longitudes_do_not_panic() {
        let grid = grid_of(|c, r| (c * 30 + r * 7) as u8);
        for (lat, lon) in [
            (90.0, 0.0),
            (-90.0, 0.0),
            (90.0, 540.0),
            (-90.0, -720.0),
            (0.0, 540.0),
            (1e6, 1e12),
            (-1e300, -1e300),
        ] {
            let got = grid.sample("iron", lat, lon);
            assert!((MIN_RICHNESS..=MAX_RICHNESS).contains(&got), "{got}");
        }
        // A longitude a whole number of turns away is the same place.
        let a = grid.sample("iron", 20.0, 33.0);
        let b = grid.sample("iron", 20.0, 33.0 + 720.0);
        assert!((a - b).abs() < 1e-9);
    }

    #[test]
    fn corrupt_or_truncated_assets_are_errors() {
        let good = encode(&["iron"], 8, 4, 45.0, &[vec![140u8; 32]]);
        assert!(decode(&good).is_ok());
        // Every truncation fails cleanly.
        for len in 0..good.len() {
            assert!(decode(&good[..len]).is_err(), "length {len}");
        }
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(decode(&bad_magic).is_err());
        let mut bad_version = good.clone();
        bad_version[4] = 2;
        assert!(decode(&bad_version).is_err());
        let mut bad_len = good.clone();
        bad_len[RAW_LEN_AT] ^= 1;
        assert!(decode(&bad_len).is_err());
        // Garbage in place of the compressed part.
        let mut garbage = good.clone();
        garbage[RAW_LEN_AT + 4..].fill(0xFF);
        assert!(decode(&garbage).is_err());
        assert!(decode(&[]).is_err());
        assert!(decode(&[0xFF; 100]).is_err());
    }

    #[test]
    fn the_embedded_asset_decodes_and_names_the_eight_ores() {
        assert!(asset_len() > 0);
        let grid = decode(ASSET).expect("the shipped asset decodes");
        assert_eq!(grid.names, NAMES);
        for ore in NAMES {
            let got = richness(ore, 45.0, 10.0);
            assert!((MIN_RICHNESS..=MAX_RICHNESS).contains(&got));
        }
    }

    #[test]
    fn the_shipped_grid_has_its_stated_size() {
        // Update with the asset whenever `tools/earth_ore_richness.py` is rerun.
        assert_eq!(asset_len(), 600_646);
        let grid = decode(ASSET).unwrap();
        assert_eq!((grid.cols, grid.rows), (1440, 720));
        assert_eq!(
            (grid.lon_west, grid.lat_north, grid.cell),
            (-180.0, 90.0, 0.25)
        );
        assert_eq!(grid.data.len(), 8 * 1440 * 720);
    }

    /// Famous places, as test cases only: the code and the script know no places, these come out of the data.
    #[test]
    fn famous_places_in_the_shipped_grid_behave() {
        let rich = [
            ("iron", -22.5, 118.5, 2.0),     // Pilbara
            ("iron", 47.5, -92.5, 2.0),      // Mesabi Range
            ("copper", -24.3, -69.1, 1.8),   // Escondida
            ("gold", -26.2, 27.9, 1.8),      // Witwatersrand
            ("coal", 44.0, -105.5, 1.3),     // Powder River Basin
            ("coal", -32.5, 151.0, 1.8),     // Hunter Valley
            ("diamond", -28.7, 24.8, 2.0),   // Kimberley
            ("emerald", 5.5, -74.1, 2.0),    // Muzo
            ("redstone", 38.77, -4.84, 1.5), // Almadén
            ("lapis", 36.2, 70.8, 1.5),      // Sar-e-Sang
        ];
        for (ore, lat, lon, floor) in rich {
            let got = richness(ore, lat, lon);
            assert!(got > floor, "{ore} at ({lat}, {lon}): {got}");
        }
        // Mid-Pacific: every ore is exactly average.
        for ore in NAMES {
            assert!((richness(ore, 0.0, -150.0) - 1.0).abs() < 1e-9, "{ore}");
        }
        // Greenland's ice: no data, close to average, not barren.
        for ore in NAMES {
            let got = richness(ore, 72.0, -40.0);
            assert!((0.8..1.25).contains(&got), "{ore}: {got}");
        }
        // A well-surveyed but metal-poor plain is not rich just because it is well surveyed.
        for ore in ["iron", "copper", "gold"] {
            let got = richness(ore, 40.0, -100.0);
            assert!(got < 0.8, "{ore}: {got}");
        }
    }
}

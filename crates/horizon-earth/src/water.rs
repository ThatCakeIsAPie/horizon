//! Rivers and lakes from global datasets: the water atlas.
//!
//! The atlas holds two kinds of feature, in degrees and metres, with no game types:
//!
//! * [`RiverReach`]: a river line from upstream to downstream, with the water surface elevation at each point
//!   (never rising downstream, and continuous where one reach flows into the next) and the long-term average
//!   discharge, from which the generator makes the river's width and depth.
//! * [`Lake`]: a lake or reservoir: its shoreline's bounding box, its surface elevation, area and mean depth. The
//!   shape itself comes from the land cover's water class; the record gives the flat level.
//!
//! Features are indexed by 1 by 1 degree cells ([`CELL_DEG`]): a feature is listed in every cell its bounding box
//! touches, and a query returns each feature whose cells meet the query box once. Queries never wait and never
//! fail: a place with no feature returns nothing.
//!
//! [`WaterAtlas::from_parts`] builds an atlas from features in memory (tests, tools). [`WaterAtlas::embedded`] is
//! the atlas shipped in the game: `data/water_atlas.bin`, built by `tools/earth_water.py` from RiverATLAS and
//! HydroLAKES (see `docs/earth.md`, "Water atlas"), embedded in the binary. [`WaterAtlas::decode`] reads such a file
//! (its format is in [`crate::waterfile`]): it checks the header and the cell index at once, and inflates and
//! decodes a cell's block only the first time a query meets that cell, then keeps it. A block that does not decode
//! is logged once and reads as an empty cell; nothing in a bad file can make the game panic.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, OnceLock};

use crate::projection::LatLon;
use crate::waterfile::{self, Entry};

/// The version of the atlas's data and of what its fields mean. It goes up when the shipped data or its rules
/// change, so far-view caches made from an older atlas are not used.
pub const WATER_RULES: u32 = 1;

/// The side of an index cell, in degrees.
pub const CELL_DEG: f64 = 1.0;

/// A river reach: a line of points from upstream to downstream.
#[derive(Clone, Debug, PartialEq)]
pub struct RiverReach {
    /// The reach's id in the source data (unique in an atlas).
    pub id: u32,
    /// The points, upstream first, in degrees (longitude in -180..=180; a reach never crosses the date line).
    pub points: Vec<LatLon>,
    /// The water surface elevation at each point, in metres above sea level: one per point, never rising from one
    /// point to the next, and the last equal to the first of the reach it flows into.
    pub levels_m: Vec<f32>,
    /// The long-term average discharge, in cubic metres a second (at the reach's downstream end).
    pub discharge_m3s: f32,
    /// The id of the reach this one flows into; `None` at a coast or an inland sink.
    pub next_down: Option<u32>,
}

impl RiverReach {
    /// The bounding box of the points: `(south, west, north, east)`.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        bounds_of(self.points.iter().copied())
    }
}

/// A lake or reservoir.
#[derive(Clone, Debug, PartialEq)]
pub struct Lake {
    /// The lake's id in the source data (unique in an atlas).
    pub id: u32,
    /// The bounding box of the shoreline, in degrees.
    pub south: f64,
    pub west: f64,
    pub north: f64,
    pub east: f64,
    /// The water surface elevation, in metres above sea level.
    pub level_m: f32,
    /// The surface area, in square kilometres.
    pub area_km2: f32,
    /// The mean depth, in metres.
    pub depth_avg_m: f32,
    /// A reservoir (a dammed lake) rather than a natural lake.
    pub reservoir: bool,
}

impl Lake {
    /// The bounding box: `(south, west, north, east)`.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        (self.south, self.west, self.north, self.east)
    }

    /// Whether `(lat, lon)` is in the bounding box.
    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        lat >= self.south && lat <= self.north && lon >= self.west && lon <= self.east
    }
}

/// Why an atlas file could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaterAtlasError(pub String);

impl fmt::Display for WaterAtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "water atlas: {}", self.0)
    }
}

impl std::error::Error for WaterAtlasError {}

/// The features listed in one index cell.
#[derive(Default, Debug)]
struct Cell {
    reaches: Vec<Arc<RiverReach>>,
    lakes: Vec<Arc<Lake>>,
}

impl Cell {
    fn from_features(reaches: Vec<RiverReach>, lakes: Vec<Lake>) -> Self {
        Self {
            reaches: reaches.into_iter().map(Arc::new).collect(),
            lakes: lakes.into_iter().map(Arc::new).collect(),
        }
    }
}

/// An atlas file: its bytes, its index, and each cell once decoded.
struct FileCells {
    bytes: Cow<'static, [u8]>,
    units: f64,
    index: Vec<Entry>,
    cells: Vec<OnceLock<Cell>>,
}

impl fmt::Debug for FileCells {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileCells")
            .field("bytes", &self.bytes.len())
            .field("cells", &self.index.len())
            .finish()
    }
}

/// Where an atlas's cells come from.
#[derive(Debug)]
enum Cells {
    Memory(HashMap<(i32, i32), Cell>),
    File(FileCells),
}

impl Default for Cells {
    fn default() -> Self {
        Self::Memory(HashMap::new())
    }
}

/// Rivers and lakes, indexed by cell.
#[derive(Default, Debug)]
pub struct WaterAtlas {
    cells: Cells,
    reach_count: usize,
    lake_count: usize,
}

/// The atlas shipped in the game (see the module doc).
static EMBEDDED: &[u8] = include_bytes!("../data/water_atlas.bin");

impl WaterAtlas {
    /// An atlas of `reaches` and `lakes` held in memory.
    pub fn from_parts(reaches: Vec<RiverReach>, lakes: Vec<Lake>) -> Self {
        let reach_count = reaches.iter().map(|r| r.id).collect::<HashSet<_>>().len();
        let lake_count = lakes.iter().map(|l| l.id).collect::<HashSet<_>>().len();
        let mut cells: HashMap<(i32, i32), Cell> = HashMap::new();
        for reach in reaches {
            let reach = Arc::new(reach);
            for key in cells_of(reach.bounds()) {
                cells
                    .entry(key)
                    .or_default()
                    .reaches
                    .push(Arc::clone(&reach));
            }
        }
        for lake in lakes {
            let lake = Arc::new(lake);
            for key in cells_of(lake.bounds()) {
                cells.entry(key).or_default().lakes.push(Arc::clone(&lake));
            }
        }
        Self {
            cells: Cells::Memory(cells),
            reach_count,
            lake_count,
        }
    }

    /// The atlas shipped in the game, decoded once; `None` when the game has none (or it does not decode, which
    /// is logged once).
    pub fn embedded() -> Option<&'static WaterAtlas> {
        static ATLAS: OnceLock<Option<WaterAtlas>> = OnceLock::new();
        ATLAS
            .get_or_init(|| match WaterAtlas::decode(EMBEDDED) {
                Ok(atlas) if !atlas.is_empty() => Some(atlas),
                Ok(_) => None,
                Err(e) => {
                    log::warn!("the embedded water atlas does not decode: {e}");
                    None
                }
            })
            .as_ref()
    }

    /// An atlas read from a file's bytes (see [`crate::waterfile`]). The header and index are checked now; each
    /// cell is decoded on the first query that meets it.
    pub fn decode(bytes: impl Into<Cow<'static, [u8]>>) -> Result<Self, WaterAtlasError> {
        let bytes = bytes.into();
        let header = waterfile::read_header(&bytes)?;
        let cells = (0..header.index.len()).map(|_| OnceLock::new()).collect();
        Ok(Self {
            cells: Cells::File(FileCells {
                bytes,
                units: header.units,
                index: header.index,
                cells,
            }),
            reach_count: header.reaches as usize,
            lake_count: header.lakes as usize,
        })
    }

    /// The bytes of a file holding `reaches` and `lakes`, which [`WaterAtlas::decode`] reads back: coordinates
    /// rounded to the file's units (1/9600 degree), levels to whole decimetres (a level that would rise is written
    /// flat), quantities to log codes (0.07 %), each id once.
    pub fn encode(reaches: &[RiverReach], lakes: &[Lake]) -> Vec<u8> {
        waterfile::write(reaches, lakes)
    }

    /// Whether the atlas holds no feature at all.
    pub fn is_empty(&self) -> bool {
        match &self.cells {
            Cells::Memory(cells) => cells.is_empty(),
            Cells::File(file) => file.index.is_empty(),
        }
    }

    /// The number of distinct reaches (as the file's header says, for a decoded atlas).
    pub fn reach_count(&self) -> usize {
        self.reach_count
    }

    /// The number of distinct lakes (as the file's header says, for a decoded atlas).
    pub fn lake_count(&self) -> usize {
        self.lake_count
    }

    /// Every non-empty cell, as `(floor(lat), floor(lon))` in degrees, in order.
    pub fn cell_keys(&self) -> Vec<(i32, i32)> {
        let mut keys: Vec<(i32, i32)> = match &self.cells {
            Cells::Memory(cells) => cells.keys().copied().collect(),
            Cells::File(file) => file.index.iter().map(|e| e.key).collect(),
        };
        keys.sort_unstable();
        keys
    }

    /// The features of one cell, decoding it on first use.
    fn cell(&self, key: (i32, i32)) -> Option<&Cell> {
        match &self.cells {
            Cells::Memory(cells) => cells.get(&key),
            Cells::File(file) => {
                let i = file.index.binary_search_by_key(&key, |e| e.key).ok()?;
                Some(file.cells[i].get_or_init(|| {
                    match waterfile::read_cell(&file.bytes, &file.index[i], file.units) {
                        Ok((reaches, lakes)) => Cell::from_features(reaches, lakes),
                        Err(e) => {
                            log::warn!("water atlas cell {key:?} does not decode: {e}");
                            Cell::default()
                        }
                    }
                }))
            }
        }
    }

    /// The features of one cell of a decoded file, decoded afresh and not kept (whole-atlas checks); `None` for an
    /// in-memory atlas or a cell not in the file.
    pub fn read_cell(
        &self,
        key: (i32, i32),
    ) -> Option<Result<(Vec<RiverReach>, Vec<Lake>), WaterAtlasError>> {
        match &self.cells {
            Cells::Memory(_) => None,
            Cells::File(file) => {
                let i = file.index.binary_search_by_key(&key, |e| e.key).ok()?;
                Some(waterfile::read_cell(
                    &file.bytes,
                    &file.index[i],
                    file.units,
                ))
            }
        }
    }

    /// Every reach listed in a cell the box `south..=north` by `west..=east` (degrees) meets, once each, in id
    /// order. The caller measures the actual distance.
    pub fn reaches_in(&self, south: f64, west: f64, north: f64, east: f64) -> Vec<Arc<RiverReach>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for key in cells_of((south, west, north, east)) {
            if let Some(cell) = self.cell(key) {
                for r in &cell.reaches {
                    if seen.insert(r.id) {
                        out.push(Arc::clone(r));
                    }
                }
            }
        }
        out.sort_unstable_by_key(|r| r.id);
        out
    }

    /// Every lake listed in a cell the box meets, once each, in id order.
    pub fn lakes_in(&self, south: f64, west: f64, north: f64, east: f64) -> Vec<Arc<Lake>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for key in cells_of((south, west, north, east)) {
            if let Some(cell) = self.cell(key) {
                for l in &cell.lakes {
                    if seen.insert(l.id) {
                        out.push(Arc::clone(l));
                    }
                }
            }
        }
        out.sort_unstable_by_key(|l| l.id);
        out
    }
}

/// The bounding box of `points`: `(south, west, north, east)`.
fn bounds_of(points: impl Iterator<Item = LatLon>) -> (f64, f64, f64, f64) {
    let mut b = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in points {
        b.0 = b.0.min(p.lat);
        b.1 = b.1.min(p.lon);
        b.2 = b.2.max(p.lat);
        b.3 = b.3.max(p.lon);
    }
    b
}

/// The index cells a box `(south, west, north, east)` meets. A box whose west is east of its east wraps the date
/// line. Latitudes are clamped to the poles.
pub(crate) fn cells_of((south, west, north, east): (f64, f64, f64, f64)) -> Vec<(i32, i32)> {
    if !(south.is_finite() && west.is_finite() && north.is_finite() && east.is_finite())
        || south > north
    {
        return Vec::new();
    }
    let cell = |v: f64| (v / CELL_DEG).floor() as i32;
    let lat0 = cell(south.clamp(-90.0, 90.0));
    let lat1 = cell(north.clamp(-90.0, 90.0));
    let lon_spans: Vec<(f64, f64)> = if west <= east {
        vec![(west.max(-180.0), east.min(180.0))]
    } else {
        vec![(west.max(-180.0), 180.0), (-180.0, east.min(180.0))]
    };
    let mut out = Vec::new();
    for (w, e) in lon_spans {
        for la in lat0..=lat1 {
            for lo in cell(w)..=cell(e) {
                out.push((la, lo));
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "water_tests.rs"]
mod file_tests;

#[cfg(test)]
#[path = "water_embedded_tests.rs"]
mod embedded_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn reach(id: u32, pts: &[(f64, f64)]) -> RiverReach {
        RiverReach {
            id,
            points: pts.iter().map(|&(lat, lon)| LatLon { lat, lon }).collect(),
            levels_m: (0..pts.len()).map(|i| 100.0 - i as f32).collect(),
            discharge_m3s: 10.0,
            next_down: None,
        }
    }

    #[test]
    fn a_query_finds_each_feature_once_in_the_cells_it_meets() {
        let atlas = WaterAtlas::from_parts(
            vec![
                reach(1, &[(10.5, 20.5), (11.5, 21.5)]),
                reach(2, &[(40.2, -3.3), (40.3, -3.2)]),
            ],
            vec![Lake {
                id: 7,
                south: 10.1,
                west: 20.1,
                north: 10.2,
                east: 20.2,
                level_m: 50.0,
                area_km2: 1.0,
                depth_avg_m: 4.0,
                reservoir: false,
            }],
        );
        let r = atlas.reaches_in(10.0, 20.0, 12.0, 22.0);
        assert_eq!(r.iter().map(|r| r.id).collect::<Vec<_>>(), vec![1]);
        assert_eq!(atlas.lakes_in(10.0, 20.0, 10.5, 20.5).len(), 1);
        assert!(atlas.reaches_in(-5.0, -5.0, -4.0, -4.0).is_empty());
        assert_eq!(atlas.reaches_in(40.0, -4.0, 40.5, -3.0)[0].id, 2);
    }

    #[test]
    fn a_box_over_the_date_line_wraps() {
        let atlas = WaterAtlas::from_parts(vec![reach(3, &[(0.5, -179.5), (0.6, -179.4)])], vec![]);
        assert_eq!(atlas.reaches_in(0.0, 179.0, 1.0, -179.0).len(), 1);
    }
}

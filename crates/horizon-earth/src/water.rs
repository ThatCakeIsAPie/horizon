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
//! the atlas shipped in the game (see `tools/earth_water.py` and `docs/earth.md`, "Water").

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::projection::LatLon;

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

/// The features listed in one index cell.
#[derive(Default, Debug)]
struct Cell {
    reaches: Vec<Arc<RiverReach>>,
    lakes: Vec<Arc<Lake>>,
}

/// Rivers and lakes, indexed by cell.
#[derive(Default, Debug)]
pub struct WaterAtlas {
    cells: HashMap<(i32, i32), Cell>,
}

impl WaterAtlas {
    /// An atlas of `reaches` and `lakes` held in memory.
    pub fn from_parts(reaches: Vec<RiverReach>, lakes: Vec<Lake>) -> Self {
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
        Self { cells }
    }

    /// The atlas shipped in the game, decoded once; `None` when the game has none (or it does not decode, which
    /// is logged once).
    pub fn embedded() -> Option<&'static WaterAtlas> {
        None
    }

    /// Whether the atlas holds no feature at all.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Every reach listed in a cell the box `south..=north` by `west..=east` (degrees) meets, once each, in id
    /// order. The caller measures the actual distance.
    pub fn reaches_in(&self, south: f64, west: f64, north: f64, east: f64) -> Vec<Arc<RiverReach>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for key in cells_of((south, west, north, east)) {
            if let Some(cell) = self.cells.get(&key) {
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
            if let Some(cell) = self.cells.get(&key) {
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
fn cells_of((south, west, north, east): (f64, f64, f64, f64)) -> Vec<(i32, i32)> {
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

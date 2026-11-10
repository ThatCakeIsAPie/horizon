//! What the top of an Earth column is made of: grass, bare stone, snow or sand.
//!
//! The rule depends on the column's real elevation (in metres, before the world's scale and exaggeration) and
//! its own latitude, and is the same for near generation and the far view:
//!
//! * **Permanent snow line**: [`SNOW_LINE_LOW_LATITUDE_M`] (5,500 m) up to [`SNOW_LINE_FLAT_TO_DEG`] (30°)
//!   of latitude, then falling linearly to 0 m at [`SNOW_LINE_ZERO_AT_DEG`] (80°), by absolute latitude.
//! * Elevation at or above the snow line: [`SNOW_DEPTH`] (2) blocks of `snow_block` over stone.
//! * Within [`ALPINE_BAND_M`] (1,500 m) below the snow line: bare stone at the surface, no soil.
//! * Below that: grass over three dirt.
//! * Under water (the surface below the sea level): sand over stone, whatever the elevation.
//!
//! A column's elevation is read back from its surface block height `s` (`VerticalMap::elevation_for_y(s)`), so
//! the material is a function of `s` and the latitude alone, and a block rectangle's materials follow from its
//! range of surfaces.

use horizon_earth::VerticalMap;

/// The snow line up to [`SNOW_LINE_FLAT_TO_DEG`] of latitude, in metres.
pub const SNOW_LINE_LOW_LATITUDE_M: f64 = 5500.0;
/// The absolute latitude up to which the snow line is flat.
pub const SNOW_LINE_FLAT_TO_DEG: f64 = 30.0;
/// The absolute latitude where the snow line reaches 0 m.
pub const SNOW_LINE_ZERO_AT_DEG: f64 = 80.0;
/// How far under the snow line the ground is bare stone, in metres.
pub const ALPINE_BAND_M: f64 = 1500.0;
/// Blocks of snow on the ground above the snow line.
pub const SNOW_DEPTH: i64 = 2;
/// The version of this rule: it goes up when what the same column is made of changes, so caches of far terrain
/// made under the old one are not used.
pub const RULES: u32 = 2;
/// Blocks of soil under the grass: stone is every block below `s - SOIL_DEPTH`.
pub const SOIL_DEPTH: i64 = 3;

/// The permanent snow line at `latitude` degrees, in metres of real elevation.
pub fn snow_line_m(latitude: f64) -> f64 {
    let lat = latitude.abs();
    if lat <= SNOW_LINE_FLAT_TO_DEG {
        SNOW_LINE_LOW_LATITUDE_M
    } else if lat >= SNOW_LINE_ZERO_AT_DEG {
        0.0
    } else {
        SNOW_LINE_LOW_LATITUDE_M * (SNOW_LINE_ZERO_AT_DEG - lat)
            / (SNOW_LINE_ZERO_AT_DEG - SNOW_LINE_FLAT_TO_DEG)
    }
}

/// What the top blocks of a column are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    /// A grass block over three dirt.
    Grass,
    /// Bare stone to the top.
    Alpine,
    /// Two blocks of snow over stone.
    Snow,
    /// Under the sea: sand over stone.
    Sand,
}

impl SurfaceKind {
    /// The kind of a column of real `elevation_m` at `latitude` degrees, on land.
    pub fn on_land(elevation_m: f64, latitude: f64) -> Self {
        let line = snow_line_m(latitude);
        if elevation_m >= line {
            Self::Snow
        } else if elevation_m >= line - ALPINE_BAND_M {
            Self::Alpine
        } else {
            Self::Grass
        }
    }

    /// The kind of a column whose surface block is `s`, at `latitude` degrees.
    pub fn of_column(s: i32, latitude: f64, sea_level: i32, vertical: &VerticalMap) -> Self {
        if s < sea_level {
            Self::Sand
        } else {
            Self::on_land(vertical.elevation_for_y(f64::from(s)), latitude)
        }
    }
}

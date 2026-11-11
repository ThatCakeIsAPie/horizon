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
//! With land cover (see [`SurfaceKind::on_classified`]) a column whose class is known takes its top from the
//! class instead, and a column without data keeps the rule above. `docs/earth.md` has the table.
//!
//! A column's elevation is read back from its surface block height `s` (`VerticalMap::elevation_for_y(s)`), so
//! the material is a function of `s` and the latitude alone, and a block rectangle's materials follow from its
//! range of surfaces.

use horizon_earth::{LandClass, VerticalMap};

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
/// The version of the land-cover surface rule and of the biome table ([`super::biome`]): it goes up when what a
/// column with a known class is made of changes. It is in the terrain key only while land cover is on.
pub const LAND_RULES: u32 = 1;
/// With land cover, the lowest elevation (m) where the latitude snow line still turns a column that is not
/// snow or ice on the map white. A map made from one summer's imagery misses seasonal snow on high peaks (Everest
/// stays white), but the line falls to sea level at the poles, where lowland the map calls bare or moss is not
/// under permanent snow.
pub const LAND_SNOW_FLOOR_M: f64 = 2500.0;
/// Blocks a lake column's ground lies under its surface (the water is that deep).
pub const LAKE_DEPTH: i64 = 3;
/// Blocks a marsh column's ground lies under its surface (one block of water).
pub const MARSH_DEPTH: i64 = 1;
/// A bare column is flat enough for sand when no neighbour a [`SLOPE_BASE_M`] away is higher or lower than this
/// share of that distance (0.6 is about 31 degrees).
pub const STEEP_GRADIENT: f64 = 0.6;
/// The ground distance, in metres, over which the slope of a bare column is measured: about one elevation pixel.
/// At a coarse step the distance is the step, so near and far agree where they can.
pub const SLOPE_BASE_M: f64 = 30.0;
/// Bare flat ground is gravel instead of sand from this absolute latitude (glacial outwash, tundra and
/// moraines, not dunes).
pub const GRAVEL_FROM_LATITUDE: f64 = 50.0;
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
    /// Under the sea: sand over stone, with water over it up to the sea level. Only this kind is flooded.
    Sand,
    /// Dry sand, four blocks: bare flat ground, even below the sea level.
    Dune,
    /// A gravel block over stone: built-up ground and cold bare flat ground.
    Gravel,
    /// A moss block over three dirt.
    Moss,
    /// A lake or river: three blocks of water, ending at the surface block, over a sand bed.
    Lake,
    /// A marsh or mangrove: one block of water, at the surface block, over mud.
    Marsh,
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

    /// The kind of a column whose class is `class`, from the map: `elevation_m` is its real elevation and
    /// `below_sea` whether its surface is under the sea level. `steep` says whether the ground around it is steep;
    /// it is asked only for bare ground that nothing else decides, and its error is passed on. A class of
    /// [`LandClass::NoData`] gets the rule without land cover ([`SurfaceKind::of_column`]).
    ///
    /// * snow and ice: snow, and so is any column at or above the latitude snow line or
    ///   [`LAND_SNOW_FLOOR_M`], whichever is higher (a lake that high is frozen);
    /// * tree, shrub, grass and crop: grass over dirt (no block fits crops better: farmland dries out without
    ///   water);
    /// * bare: stone within the alpine band under the snow line, else stone where steep, else sand, or gravel
    ///   from [`GRAVEL_FROM_LATITUDE`];
    /// * water: a lake, or the seabed (flooded to the sea level) when the surface is under the sea;
    /// * wetland and mangrove: a marsh; moss and lichen: moss; built-up: gravel.
    ///
    /// Only [`SurfaceKind::Sand`] is flooded: a known class that is not water stays dry under the sea level.
    pub fn on_classified<E>(
        class: LandClass,
        elevation_m: f64,
        latitude: f64,
        below_sea: bool,
        steep: impl FnOnce() -> Result<bool, E>,
    ) -> Result<Self, E> {
        let line = snow_line_m(latitude);
        if class == LandClass::SnowIce || elevation_m >= line.max(LAND_SNOW_FLOOR_M) {
            return Ok(Self::Snow);
        }
        Ok(match class {
            LandClass::NoData if below_sea => Self::Sand,
            LandClass::NoData => Self::on_land(elevation_m, latitude),
            LandClass::Tree | LandClass::Shrub | LandClass::Grass | LandClass::Crop => Self::Grass,
            LandClass::Bare => {
                if elevation_m >= line - ALPINE_BAND_M || steep()? {
                    Self::Alpine
                } else if latitude.abs() >= GRAVEL_FROM_LATITUDE {
                    Self::Gravel
                } else {
                    Self::Dune
                }
            }
            LandClass::SnowIce => Self::Snow,
            LandClass::Water if below_sea => Self::Sand,
            LandClass::Water => Self::Lake,
            LandClass::Wetland | LandClass::Mangrove => Self::Marsh,
            LandClass::MossLichen => Self::Moss,
            LandClass::BuiltUp => Self::Gravel,
        })
    }

    /// How many blocks under the surface block the ground of this kind lies: the water of a lake or marsh fills
    /// the space between.
    pub fn sink(self) -> i64 {
        match self {
            Self::Lake => LAKE_DEPTH,
            Self::Marsh => MARSH_DEPTH,
            _ => 0,
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

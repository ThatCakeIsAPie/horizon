//! What the top of an Earth column is made of: grass, bare stone, snow, sand, or water over a bed.
//!
//! The rule depends on the column's real elevation (in metres, before the world's scale and exaggeration) and
//! its own latitude, and is the same for near generation and the far view:
//!
//! * **Permanent snow line**: [`SNOW_LINE_LOW_LATITUDE_M`] (5,500 m) up to [`SNOW_LINE_FLAT_TO_DEG`] (30°)
//!   of latitude, then falling linearly to 0 m at [`SNOW_LINE_ZERO_AT_DEG`] (80°), by absolute latitude.
//! * Elevation at or above the snow line: [`SNOW_DEPTH`] (2) blocks of `snow_block` over stone.
//! * Within [`ALPINE_BAND_M`] (1,500 m) below the snow line: bare stone at the surface, no soil.
//! * Below that: grass over three dirt.
//! * Under the sea (the elevation surface below the sea level): water up to the block under the sea level over a
//!   sand bed, whatever the elevation ([`SurfaceKind::sea_column`]).
//!
//! With land cover (see [`SurfaceKind::on_classified`]) a column whose class is known takes its top from the
//! class instead, and a column without data keeps the rule above. `docs/earth.md` has the table.
//!
//! With the real climate on as well, open ground (bare, shrub, grass) in a dry climate
//! ([`DRY_ARIDITY`](super::dry::DRY_ARIDITY)) takes the dry places' tops ([`super::dry`]):
//!
//! | Class, in a dry climate | Steep | Gentle |
//! |---|---|---|
//! | bare | [`SurfaceKind::Banded`] (terracotta bands) | hot (from 18 C): sand dunes; else [`SurfaceKind::Barren`] (red sand, coarse dirt) |
//! | shrub | [`SurfaceKind::Banded`] | [`SurfaceKind::Scrub`] (grass block, coarse dirt, red sand mixed block by block) |
//! | grass | [`SurfaceKind::Banded`] | grass over dirt, as in a wet climate |
//!
//! **Steep snow is rock, snow/ice is a glacier, wet bare ground is basalt.** Ground that would be snow (land cover
//! snow/ice, or at or above the snow line) whose slope is [`SNOW_STEEP_GRADIENT`] or more is bare stone
//! ([`SurfaceKind::Alpine`]). Snow/ice class ground under that is a [`SurfaceKind::Glacier`] (snow over packed
//! ice) when gentle (under [`STEEP_GRADIENT`]) and an [`SurfaceKind::Icefall`] (packed ice over blue ice) when
//! steeper. Snow above the snow line on other classes stays snow over stone. Gentle bare ground below
//! [`GRAVEL_FROM_LATITUDE`] in a known wet climate is [`SurfaceKind::Basalt`] (young volcanic rock: basalt,
//! smooth basalt, tuff, blackstone), not sand; sand stays for hot dry ground and for ground without climate.
//!
//! **Frozen wetland is tundra.** Herbaceous wetland in a known climate under [`FROZEN_BELOW_C`] (the frozen row,
//! whose biome snows and whose still water the snow pass freezes) is grass over dirt, not a marsh: one block of
//! water over every wetland column of the Arctic coastal plain froze into a single sheet of ice. The tundra's
//! open water is mapped as water (its thaw lakes) and stays lakes.
//!
//! Snow and the alpine band still win first; tree cover, crops, water and wetland are unchanged.
//!
//! **Moss is a mix** ([`super::cover`]): a moss and lichen column is [`SurfaceKind::Moss`], mostly moss block
//! with some grass and bare stone, gravel and coarse dirt. Built-up land has no kind of its own: the decoded
//! land cover has no built-up class (the land around a town takes its place, `horizon_earth::unbuild`).
//! Under the top of a dry kind the ground is terracotta bands by absolute height down to
//! [`DRY_ROCK_DEPTH`](super::dry::DRY_ROCK_DEPTH) blocks, then stone.
//!
//! **Water is part of the column.** A column is a pair `(s, kind)`. For a water column ([`SurfaceKind::Water`]) `s`
//! is the TOP WATER BLOCK, not the ground: the kind says how many blocks of water there are (`depth`, at least 1)
//! and what the bed is made of ([`Bed`]), so the ground (the top of the bed) is `s - depth`. Blocks above `s` are
//! air; blocks `s - depth + 1 ..= s` are water; the bed block is at `s - depth`, and the same material goes
//! [`SOIL_DEPTH`] blocks further down (stone for a stone bed), then stone. The sea is such a column with its top
//! at the block under the sea level ([`SurfaceKind::sea_column`]); a lake is one with [`LAKE_DEPTH`] blocks of
//! water. Everything that reads a column (near cubes, far voxels, far trees) takes `(s, kind)` together, so rules
//! that move a column's top (water held over low ground, carved or raised banks) only have to change `s`.
//!
//! A column's elevation is read back from its surface block height `s` (`VerticalMap::elevation_for_y(s)`), so
//! the material is a function of `s` and the latitude alone, and a block rectangle's materials follow from its
//! range of surfaces.

use horizon_earth::{LandClass, VerticalMap};

use super::biome::FROZEN_BELOW_C;
use super::cover::{MossTop, moss_top};
use super::dry::{BARREN_MIX, BASALT_MIX, BasaltTop, DESERT_MIX, DryColumn, DryTop, SCRUB_MIX};
use super::water::Bed;

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
pub const LAND_RULES: u32 = 8;
/// With land cover, the lowest elevation (m) where the latitude snow line still turns a column that is not
/// snow or ice on the map white. A map made from one summer's imagery misses seasonal snow on high peaks (Everest
/// stays white), but the line falls to sea level at the poles, where lowland the map calls bare or moss is not
/// under permanent snow.
pub const LAND_SNOW_FLOOR_M: f64 = 2500.0;
/// Blocks of water in a lake column (its ground lies this far under its top water block).
pub const LAKE_DEPTH: i64 = 3;
/// Blocks of water in a sea column whose elevation surface is at or just under the top of the sea: the sea always
/// has at least one block of water over its bed.
pub const MIN_WATER_DEPTH: i64 = 1;
/// Blocks a marsh column's ground lies under its surface (one block of water).
pub const MARSH_DEPTH: i64 = 1;
/// A bare column is flat enough for sand when no neighbour a [`SLOPE_BASE_M`] away is higher or lower than this
/// share of that distance (0.6 is about 31 degrees).
pub const STEEP_GRADIENT: f64 = 0.6;
/// Snow does not hold on ground this steep (0.84 is about 40 degrees): a column that would be snow shows bare
/// stone there. Between [`STEEP_GRADIENT`] and this, snow stays (and snow/ice class ground shows ice).
pub const SNOW_STEEP_GRADIENT: f64 = 0.84;
/// Blocks of snow on top of a glacier.
pub const GLACIER_SNOW_DEPTH: i64 = 1;
/// How many blocks under the surface block the ice of a glacier or icefall reaches (stone below that).
pub const GLACIER_ICE_DEPTH: i64 = 6;
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

/// How steep the ground around a column is, by the largest height difference to a neighbour one slope base away
/// ([`STEEP_GRADIENT`], [`SNOW_STEEP_GRADIENT`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Slope {
    /// Under [`STEEP_GRADIENT`].
    Gentle,
    /// From [`STEEP_GRADIENT`], under [`SNOW_STEEP_GRADIENT`]: too steep for sand, snow still holds.
    Steep,
    /// From [`SNOW_STEEP_GRADIENT`]: bare rock, snow does not hold.
    Rock,
}

impl Slope {
    /// The slope of a gradient (rise over run).
    pub fn of_gradient(gradient: f64) -> Self {
        if gradient >= SNOW_STEEP_GRADIENT {
            Self::Rock
        } else if gradient >= STEEP_GRADIENT {
            Self::Steep
        } else {
            Self::Gentle
        }
    }

    /// Whether bare ground here is too steep for sand ([`STEEP_GRADIENT`] or more).
    pub fn is_steep(self) -> bool {
        self >= Self::Steep
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
    /// Dry sand, four blocks: bare flat ground, even below the sea level.
    Dune,
    /// A gravel block over stone: cold bare flat ground.
    Gravel,
    /// Moss and lichen: the given top ([`super::cover::MOSS_MIX`]), over three dirt under moss, grass and coarse
    /// dirt, over stone under stone and gravel.
    Moss(MossTop),
    /// The sea, a lake or a river: `depth` blocks of water (at least 1) ending at the surface block `s`, over a
    /// bed of `bed` whose top block is `s - depth`.
    Water { depth: u32, bed: Bed },
    /// A marsh or mangrove: one block of water, at the surface block, over mud.
    Marsh,
    /// Steep rock in a dry climate: terracotta bands by absolute height ([`super::dry`]), down to
    /// [`DRY_ROCK_DEPTH`](super::dry::DRY_ROCK_DEPTH) blocks under the surface, stone below.
    Banded,
    /// Sparse ground of a dry climate under shrub: the top is the given block (a fixed hash of the
    /// column picks it), terracotta bands under it.
    Scrub(DryTop),
    /// Bare flat ground of a dry climate that is not hot: red sand or coarse dirt over terracotta bands.
    Barren(DryTop),
    /// A glacier: snow land cover on gentle ground, [`GLACIER_SNOW_DEPTH`] block of snow over packed ice
    /// ([`GLACIER_ICE_DEPTH`] blocks), stone below.
    Glacier,
    /// An icefall: snow land cover on steeper ground shows ice at the top: packed ice over blue ice
    /// ([`GLACIER_ICE_DEPTH`] blocks), stone below.
    Icefall,
    /// Young bare rock: bare gentle ground in a wet climate. The top is a basalt-family block (a fixed hash
    /// of the column picks it), stone below.
    Basalt(BasaltTop),
}

impl SurfaceKind {
    /// What a water-class column under the sea is before its surface is known: one block of water over sand. The
    /// generator replaces it by [`SurfaceKind::sea_column`], which has the depth.
    pub const SEA: Self = Self::Water {
        depth: MIN_WATER_DEPTH as u32,
        bed: Bed::Sand,
    };

    /// The sea over a column whose elevation surface is `s0` under `sea_level`: the column `(s, kind)` with the
    /// top water block `s = sea_level - 1` and a sand bed at `min(s0, s - 1)`, so the water is
    /// `s - bed` blocks deep (at least 1).
    pub fn sea_column(s0: i32, sea_level: i32) -> (i32, Self) {
        let top = i64::from(sea_level) - 1;
        let ground = i64::from(s0).min(top - MIN_WATER_DEPTH);
        let depth = u32::try_from(top - ground).unwrap_or(u32::MAX);
        (
            top as i32,
            Self::Water {
                depth,
                bed: Bed::Sand,
            },
        )
    }

    /// Whether the column holds water over a bed (the sea, a lake, a river); a marsh is not one.
    pub fn is_water(self) -> bool {
        matches!(self, Self::Water { .. })
    }

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
    /// `below_sea` whether its surface is under the sea level. `slope` says how steep the ground around it is
    /// ([`Slope`]); it is asked only for columns that nothing else decides, and its error is passed on. A class of
    /// [`LandClass::NoData`] gets the rule without land cover ([`SurfaceKind::of_column`]).
    ///
    /// * snow and ice, and any column at or above the latitude snow line or [`LAND_SNOW_FLOOR_M`], whichever is
    ///   higher (a lake that high is frozen): on [`Slope::Rock`] ground bare stone; else snow, except that the
    ///   snow/ice class is a [`SurfaceKind::Glacier`] on gentle ground and an [`SurfaceKind::Icefall`] on
    ///   steep ground;
    /// * tree and crop: grass over dirt (no block fits crops better: farmland dries out without water);
    /// * shrub and grass: grass over dirt, but in a dry climate ([`DryColumn::is_dry`]): steep ground is
    ///   [`SurfaceKind::Banded`], gentle shrub [`SurfaceKind::Scrub`] (gentle grass stays grass);
    /// * bare: stone within the alpine band under the snow line, else (steep) stone, or in a dry climate
    ///   [`SurfaceKind::Banded`], else sand, or gravel from [`GRAVEL_FROM_LATITUDE`]; gentle dry bare ground
    ///   is sand (dunes) when hot ([`HOT_C`](super::dry::HOT_C)), else [`SurfaceKind::Barren`]; gentle bare
    ///   ground in a known wet climate is [`SurfaceKind::Basalt`];
    /// * water: a lake ([`SurfaceKind::Water`], [`LAKE_DEPTH`] blocks over sand), or [`SurfaceKind::SEA`] when the
    ///   surface is under the sea (no data under the sea is the same); the caller turns it into the real sea
    ///   column with [`SurfaceKind::sea_column`];
    /// * wetland and mangrove: a marsh, except wetland in a frozen climate (a known mean annual temperature under
    ///   [`FROZEN_BELOW_C`]): grass over dirt;
    /// * moss and lichen: [`SurfaceKind::Moss`], its mix picked by a hash of the block column `at`.
    ///
    /// `dry` is the climate at the column with its hash; without it (`None`: no climate, or no data there) the
    /// rules are exactly those without the dry places, without wet bare rock and without frozen wetland.
    ///
    /// A known class that is not water stays dry under the sea level.
    pub fn on_classified<E>(
        class: LandClass,
        elevation_m: f64,
        latitude: f64,
        below_sea: bool,
        dry: Option<DryColumn>,
        at: [i32; 2],
        slope: impl FnOnce() -> Result<Slope, E>,
    ) -> Result<Self, E> {
        let line = snow_line_m(latitude);
        if class == LandClass::SnowIce || elevation_m >= line.max(LAND_SNOW_FLOOR_M) {
            // A frozen lake is a flat sheet, whatever is around it.
            if class == LandClass::Water {
                return Ok(Self::Snow);
            }
            return Ok(match (slope()?, class) {
                (Slope::Rock, _) => Self::Alpine,
                (Slope::Gentle, LandClass::SnowIce) => Self::Glacier,
                (Slope::Steep, LandClass::SnowIce) => Self::Icefall,
                _ => Self::Snow,
            });
        }
        Ok(match class {
            LandClass::NoData if below_sea => Self::SEA,
            LandClass::NoData => Self::on_land(elevation_m, latitude),
            LandClass::Tree | LandClass::Crop => Self::Grass,
            // Decoded land cover never holds built-up land (it is the natural land around it); if one is named, it is
            // plain grassland, as no town is.
            // Grassland keeps its grass in a dry climate (only a wall too steep for soil is rock); shrubland is the
            // sparse ground, a mix picked block by block.
            LandClass::Shrub | LandClass::Grass | LandClass::BuiltUp => {
                match dry.filter(DryColumn::is_dry) {
                    None => Self::Grass,
                    Some(_) if slope()?.is_steep() => Self::Banded,
                    Some(d) if class == LandClass::Shrub => {
                        let mix = if d.is_desert() { DESERT_MIX } else { SCRUB_MIX };
                        Self::Scrub(mix.pick(d.bucket))
                    }
                    Some(_) => Self::Grass,
                }
            }
            LandClass::Bare => {
                let climate = dry;
                let dry = dry.filter(DryColumn::is_dry);
                if elevation_m >= line - ALPINE_BAND_M {
                    Self::Alpine
                } else if slope()?.is_steep() {
                    if dry.is_some() {
                        Self::Banded
                    } else {
                        Self::Alpine
                    }
                } else if latitude.abs() >= GRAVEL_FROM_LATITUDE {
                    Self::Gravel
                } else {
                    match (dry, climate) {
                        (Some(d), _) if !d.is_hot() => Self::Barren(BARREN_MIX.pick(d.bucket)),
                        (None, Some(wet)) => Self::Basalt(BASALT_MIX.pick(wet.bucket)),
                        _ => Self::Dune,
                    }
                }
            }
            LandClass::SnowIce => Self::Snow,
            LandClass::Water if below_sea => Self::SEA,
            LandClass::Water => Self::Water {
                depth: LAKE_DEPTH as u32,
                bed: Bed::Sand,
            },
            // Tundra the map calls wetland is waterlogged ground, not open water: its biome snows and the snow pass
            // would freeze a marsh into one sheet of ice. Its open water is mapped as water (thaw lakes).
            LandClass::Wetland if dry.is_some_and(|c| c.temperature_c < FROZEN_BELOW_C) => {
                Self::Grass
            }
            LandClass::Wetland | LandClass::Mangrove => Self::Marsh,
            LandClass::MossLichen => Self::Moss(moss_top(at[0], at[1])),
        })
    }

    /// How many blocks under the surface block the ground of this kind lies: the water of a water column or marsh
    /// fills the space between.
    pub fn sink(self) -> i64 {
        match self {
            Self::Water { depth, .. } => i64::from(depth),
            Self::Marsh => MARSH_DEPTH,
            _ => 0,
        }
    }

    /// Whether the kind is one of the dry places' ([`SurfaceKind::Banded`], [`SurfaceKind::Scrub`],
    /// [`SurfaceKind::Barren`]): terracotta bands under the top, down to [`super::dry::DRY_ROCK_DEPTH`].
    pub fn is_dry(self) -> bool {
        matches!(self, Self::Banded | Self::Scrub(_) | Self::Barren(_))
    }

    /// The column `(s, kind)` whose elevation surface block is `s0`, at `latitude` degrees: the sea
    /// ([`SurfaceKind::sea_column`]) under the sea level, else the kind of land of that elevation on `s0`.
    pub fn of_column(
        s0: i32,
        latitude: f64,
        sea_level: i32,
        vertical: &VerticalMap,
    ) -> (i32, Self) {
        if s0 < sea_level {
            Self::sea_column(s0, sea_level)
        } else {
            (
                s0,
                Self::on_land(vertical.elevation_for_y(f64::from(s0)), latitude),
            )
        }
    }
}

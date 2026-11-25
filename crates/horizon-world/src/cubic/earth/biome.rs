//! The biome of an Earth column when land cover is on: a small table from class, latitude and elevation.
//!
//! Without land cover every column is `plains` ([`super::BIOME`]). With it the biome decides grass, foliage and
//! water colour and which mobs spawn, so it has to be the same wherever it is asked: it is a function of the
//! 4 by 4 block column alone, read at one fixed ground resolution ([`BIOME_SAMPLE_M`]) whatever the detail level
//! or the cube asking. That resolution is coarse (a few hundred metres, a coarse overview level of the map and a
//! coarse zoom of the elevation), so a far view needs only a handful of tiles for it, never the fine ones.
//!
//! | Class (then latitude `l`, elevation `e`) | Biome |
//! |---|---|
//! | snow and ice, or `e` at or above the snow line ([`SNOW_FLOOR`](super::surface::LAND_SNOW_FLOOR_M)) | `frozen_peaks` from 3,000 m, `snowy_slopes` from 1,500 m, else `snowy_plains` |
//! | water | `ocean` at or under sea level, else `river` |
//! | no data | `ocean` under sea level, else `plains` |
//! | tree | `jungle` under 12 degrees, `forest` under 55, `taiga` under 66, else `snowy_taiga` |
//! | shrub | `snowy_plains` from 66 degrees, `savanna` under 35, else `plains` |
//! | grass | `snowy_plains` from 66 degrees, `meadow` from 1,500 m, `savanna` under 20, else `plains` |
//! | crop, built-up | `plains` |
//! | bare | `stony_peaks` from 2,000 m, `desert` under 40 degrees, else `windswept_gravelly_hills` |
//! | wetland, mangrove | `swamp`, `mangrove_swamp` |
//! | moss and lichen | `snowy_plains` (tundra) |
//!
//! ## With the real climate
//!
//! A world with `climate` on (and land cover) reads the CHELSA mean annual temperature `T` (degrees C) and
//! annual precipitation `P` (mm) at the column ([`horizon_earth::ClimateSource`]), at the same 500 m. Land cover
//! still decides the kind of ground; the climate picks the biome inside it with vanilla's own overworld tables
//! (`OverworldBiomeBuilder`: a temperature row and a humidity column):
//!
//! - temperature row: frozen under 0 C, cold under 8, temperate under 18, warm from 18; hot (vanilla's desert
//!   row) when warm and dry (humidity 0 or 1).
//! - humidity column from a Koppen-style aridity ratio `r = P / (20 (T + 7))`: 0 under 0.5, 1 under 1, 2 under 2,
//!   3 under 3.5, else 4.
//!
//! | Row | Middle (humidity 0 to 4) | Variant | Plateau (from 1,500 m) |
//! |---|---|---|---|
//! | frozen | snowy plains x3, snowy taiga, taiga | ice spikes, -, snowy taiga, -, - | snowy plains x3, snowy taiga x2 |
//! | cold | plains, plains, forest, taiga, old growth spruce taiga | -, -, -, -, old growth pine taiga | meadow, meadow, forest, taiga, old growth spruce taiga |
//! | temperate | flower forest, plains, forest, birch forest, dark forest | sunflower plains, -, -, old growth birch forest, - | meadow x4, dark forest |
//! | warm | savanna, savanna, forest, jungle, jungle | -, -, plains, sparse jungle, bamboo jungle | savanna plateau x2, forest, forest, jungle |
//! | hot | desert x5 | - | badlands x3, wooded badlands x2 |
//!
//! The variant is taken in about a third of 2 km cells (a fixed hash of the cell). The pick must fit the land
//! class, else the class's own choice for the row is used:
//!
//! | Class | Fits | Otherwise |
//! |---|---|---|
//! | tree | the forests, taigas and jungles, wooded badlands; savanna when warm or hot; always snowy taiga when frozen | frozen snowy taiga, cold taiga, temperate forest, warm jungle (humidity 3+) or savanna, hot savanna; temperate with `r` of 6 or more: old growth spruce taiga (temperate rainforest) |
//! | grass, no data on land | plains, sunflower plains, snowy plains, savanna, savanna plateau, meadow | frozen snowy plains, cold and temperate plains (meadow from 1,500 m), warm and hot savanna |
//! | shrub | as grass, and desert | as grass, but desert when humidity is 0 and the row is temperate or warmer (dry places: see below) |
//! | crop | plains, sunflower plains | frozen snowy plains, warm and hot savanna, else plains |
//! | bare | - | in a dry climate (see below) desert (badlands from 1,000 m when temperate or warmer); else stony peaks from 2,000 m, windswept gravelly hills |
//! | water at or under sea level, no data under it | - | by temperature: warm ocean from 24 C, lukewarm, ocean, cold, frozen ocean (the deep kinds under 200 m) |
//! | water above sea level | - | river, frozen river when frozen |
//! | wetland | - | swamp; snowy plains when frozen |
//! | mangrove, moss and lichen, snow and ice / snow line | - | as without climate |
//!
//! ## Dry places
//!
//! Bare, shrub and grass ground in a dry climate (aridity `r` under [`DRY_ARIDITY`](super::dry::DRY_ARIDITY),
//! not frozen) ignores the humidity column above:
//!
//! | Ground | Biome |
//! |---|---|
//! | rugged (relief from [`RUGGED_M`](super::dry::RUGGED_M) = 150 m) | `badlands`; `eroded_badlands` from [`ERODED_M`](super::dry::ERODED_M) = 400 m |
//! | bare, flat | `desert`; `badlands` from 1,000 m when temperate or warmer |
//! | shrub, grass, flat, temperate or warmer (8 C and up) | `desert` when `r` is under [`DESERT_ARIDITY`](super::dry::DESERT_ARIDITY) (0.5), else `savanna` (`savanna_plateau` from 1,500 m) |
//! | shrub, grass, flat, under 8 C | as the table above |
//!
//! Relief is the elevation range of the 500 m cell's middle and its four neighbours half a cell (250 m) away,
//! read at the biome's own zoom. Tree cover keeps the table (wooded badlands on a hot plateau). Wet places are
//! unchanged.
//!
//! Where the climate has no data (no file, a refused host, a "no data" pixel), land uses the table without
//! climate above; seas and rivers take a temperature from latitude and height (`27 - 0.0074 lat^2` C, minus
//! 6.5 C a kilometre above the sea).
//!
//! [`EarthBiomes`] gives this to the cubic generator as its biome source.

use std::sync::Arc;

use horizon_earth::{Climate, ClimateLookup, LandClass, LoadError, Lookup};

use super::dry::{DESERT_ARIDITY, ERODED_M, RUGGED_M, SAVANNA_FROM_C, aridity, is_dry};
use super::generator::{Access, EarthGenerator, Gap, NotReady, Rect};
use super::surface::{LAND_SNOW_FLOOR_M, snow_line_m};
use crate::cubic::custom::biomes::BiomeColumns;
use crate::worldgen::biome::BiomeId;

/// The ground resolution, in metres, the biome of a column is read at.
pub const BIOME_SAMPLE_M: f64 = 500.0;

/// The version of the climate biome rules above; in the terrain key while climate is on.
pub const CLIMATE_BIOME_RULES: u32 = 2;

/// The ground size, in metres, of the cells a vanilla variant biome is chosen in.
const VARIANT_CELL_M: f64 = 2000.0;
/// Cells out of 1,000 that take the variant.
const VARIANT_SHARE: u64 = 333;

/// The biomes of the table, `plains` first (it is what a cancelled read gives). New ones go at the end, so the
/// ids of the old ones stay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Biome {
    Plains,
    Ocean,
    River,
    Desert,
    Savanna,
    Forest,
    Jungle,
    Taiga,
    SnowyTaiga,
    SnowyPlains,
    SnowySlopes,
    FrozenPeaks,
    StonyPeaks,
    Meadow,
    WindsweptGravellyHills,
    Swamp,
    MangroveSwamp,
    SunflowerPlains,
    FlowerForest,
    BirchForest,
    OldGrowthBirchForest,
    DarkForest,
    OldGrowthSpruceTaiga,
    OldGrowthPineTaiga,
    SparseJungle,
    BambooJungle,
    SavannaPlateau,
    Badlands,
    WoodedBadlands,
    IceSpikes,
    WarmOcean,
    LukewarmOcean,
    DeepLukewarmOcean,
    DeepOcean,
    ColdOcean,
    DeepColdOcean,
    FrozenOcean,
    DeepFrozenOcean,
    FrozenRiver,
    ErodedBadlands,
}

impl Biome {
    /// Every biome, in id order.
    pub const ALL: [Biome; 40] = [
        Self::Plains,
        Self::Ocean,
        Self::River,
        Self::Desert,
        Self::Savanna,
        Self::Forest,
        Self::Jungle,
        Self::Taiga,
        Self::SnowyTaiga,
        Self::SnowyPlains,
        Self::SnowySlopes,
        Self::FrozenPeaks,
        Self::StonyPeaks,
        Self::Meadow,
        Self::WindsweptGravellyHills,
        Self::Swamp,
        Self::MangroveSwamp,
        Self::SunflowerPlains,
        Self::FlowerForest,
        Self::BirchForest,
        Self::OldGrowthBirchForest,
        Self::DarkForest,
        Self::OldGrowthSpruceTaiga,
        Self::OldGrowthPineTaiga,
        Self::SparseJungle,
        Self::BambooJungle,
        Self::SavannaPlateau,
        Self::Badlands,
        Self::WoodedBadlands,
        Self::IceSpikes,
        Self::WarmOcean,
        Self::LukewarmOcean,
        Self::DeepLukewarmOcean,
        Self::DeepOcean,
        Self::ColdOcean,
        Self::DeepColdOcean,
        Self::FrozenOcean,
        Self::DeepFrozenOcean,
        Self::FrozenRiver,
        Self::ErodedBadlands,
    ];

    /// Whether the far view and the snow rules treat the biome as one that snows: vanilla's own temperature
    /// is under 0.15 and it has precipitation (the frozen biomes and the snowy ones).
    pub fn snows(self) -> bool {
        matches!(
            self,
            Self::SnowyTaiga
                | Self::SnowyPlains
                | Self::SnowySlopes
                | Self::FrozenPeaks
                | Self::IceSpikes
                | Self::FrozenOcean
                | Self::DeepFrozenOcean
                | Self::FrozenRiver
        )
    }

    /// Whether the biome is one of the dry places' where only a share of the columns carry plants
    /// ([`PLANT_SHARE`](super::dry::PLANT_SHARE)): deserts, badlands and savannas.
    pub fn few_plants(self) -> bool {
        matches!(
            self,
            Self::Desert
                | Self::Badlands
                | Self::ErodedBadlands
                | Self::Savanna
                | Self::SavannaPlateau
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Plains => "minecraft:plains",
            Self::Ocean => "minecraft:ocean",
            Self::River => "minecraft:river",
            Self::Desert => "minecraft:desert",
            Self::Savanna => "minecraft:savanna",
            Self::Forest => "minecraft:forest",
            Self::Jungle => "minecraft:jungle",
            Self::Taiga => "minecraft:taiga",
            Self::SnowyTaiga => "minecraft:snowy_taiga",
            Self::SnowyPlains => "minecraft:snowy_plains",
            Self::SnowySlopes => "minecraft:snowy_slopes",
            Self::FrozenPeaks => "minecraft:frozen_peaks",
            Self::StonyPeaks => "minecraft:stony_peaks",
            Self::Meadow => "minecraft:meadow",
            Self::WindsweptGravellyHills => "minecraft:windswept_gravelly_hills",
            Self::Swamp => "minecraft:swamp",
            Self::MangroveSwamp => "minecraft:mangrove_swamp",
            Self::SunflowerPlains => "minecraft:sunflower_plains",
            Self::FlowerForest => "minecraft:flower_forest",
            Self::BirchForest => "minecraft:birch_forest",
            Self::OldGrowthBirchForest => "minecraft:old_growth_birch_forest",
            Self::DarkForest => "minecraft:dark_forest",
            Self::OldGrowthSpruceTaiga => "minecraft:old_growth_spruce_taiga",
            Self::OldGrowthPineTaiga => "minecraft:old_growth_pine_taiga",
            Self::SparseJungle => "minecraft:sparse_jungle",
            Self::BambooJungle => "minecraft:bamboo_jungle",
            Self::SavannaPlateau => "minecraft:savanna_plateau",
            Self::Badlands => "minecraft:badlands",
            Self::WoodedBadlands => "minecraft:wooded_badlands",
            Self::IceSpikes => "minecraft:ice_spikes",
            Self::WarmOcean => "minecraft:warm_ocean",
            Self::LukewarmOcean => "minecraft:lukewarm_ocean",
            Self::DeepLukewarmOcean => "minecraft:deep_lukewarm_ocean",
            Self::DeepOcean => "minecraft:deep_ocean",
            Self::ColdOcean => "minecraft:cold_ocean",
            Self::DeepColdOcean => "minecraft:deep_cold_ocean",
            Self::FrozenOcean => "minecraft:frozen_ocean",
            Self::DeepFrozenOcean => "minecraft:deep_frozen_ocean",
            Self::FrozenRiver => "minecraft:frozen_river",
            Self::ErodedBadlands => "minecraft:eroded_badlands",
        }
    }

    /// Whether a column of `class` at `elevation_m` and `latitude` is snow (the snow class or the snow line).
    fn snowy(class: LandClass, elevation_m: f64, latitude: f64) -> Option<Self> {
        let snowy = class == LandClass::SnowIce
            || elevation_m >= snow_line_m(latitude).max(LAND_SNOW_FLOOR_M);
        snowy.then_some({
            if elevation_m >= 3000.0 {
                Self::FrozenPeaks
            } else if elevation_m >= 1500.0 {
                Self::SnowySlopes
            } else {
                Self::SnowyPlains
            }
        })
    }
    /// The biome of a column of `class` at real `elevation_m` and `latitude` degrees (see the table above).
    pub fn of(class: LandClass, elevation_m: f64, latitude: f64) -> Self {
        let lat = latitude.abs();
        if let Some(snow) = Self::snowy(class, elevation_m, latitude) {
            return snow;
        }
        match class {
            LandClass::Water => {
                if elevation_m <= 0.0 {
                    Self::Ocean
                } else {
                    Self::River
                }
            }
            LandClass::NoData => {
                if elevation_m < 0.0 {
                    Self::Ocean
                } else {
                    Self::Plains
                }
            }
            LandClass::Tree => {
                if lat < 12.0 {
                    Self::Jungle
                } else if lat < 55.0 {
                    Self::Forest
                } else if lat < 66.0 {
                    Self::Taiga
                } else {
                    Self::SnowyTaiga
                }
            }
            LandClass::Shrub => {
                if lat >= 66.0 {
                    Self::SnowyPlains
                } else if lat < 35.0 {
                    Self::Savanna
                } else {
                    Self::Plains
                }
            }
            LandClass::Grass => {
                if lat >= 66.0 {
                    Self::SnowyPlains
                } else if elevation_m >= 1500.0 {
                    Self::Meadow
                } else if lat < 20.0 {
                    Self::Savanna
                } else {
                    Self::Plains
                }
            }
            LandClass::Crop | LandClass::BuiltUp | LandClass::SnowIce => Self::Plains,
            LandClass::Bare => {
                if elevation_m >= 2000.0 {
                    Self::StonyPeaks
                } else if lat < 40.0 {
                    Self::Desert
                } else {
                    Self::WindsweptGravellyHills
                }
            }
            LandClass::Wetland => Self::Swamp,
            LandClass::Mangrove => Self::MangroveSwamp,
            LandClass::MossLichen => Self::SnowyPlains,
        }
    }

    /// The biome of a column of `class` at real `elevation_m` and `latitude` with the real `climate` there (the
    /// tables in the module docs). `variant` says whether the column's 2 km cell takes vanilla's variant biome.
    /// Without a climate, land is [`Biome::of`] exactly and water takes a temperature from latitude and height.
    pub fn with_climate(
        class: LandClass,
        elevation_m: f64,
        latitude: f64,
        climate: Option<Climate>,
        variant: bool,
    ) -> Self {
        Self::with_relief(class, elevation_m, latitude, climate, variant, 0.0)
    }

    /// [`Biome::with_climate`] with the `relief_m` of the ground around the column (the elevation range of its
    /// cell's middle and four neighbours half a cell away): dry rugged ground is badlands
    /// ([`RUGGED_M`], [`ERODED_M`]).
    pub fn with_relief(
        class: LandClass,
        elevation_m: f64,
        latitude: f64,
        climate: Option<Climate>,
        variant: bool,
        relief_m: f64,
    ) -> Self {
        let Some(climate) = climate else {
            let t = fallback_temperature_c(elevation_m, latitude);
            return match class {
                LandClass::Water if elevation_m <= 0.0 => ocean(t, elevation_m),
                LandClass::NoData if elevation_m < 0.0 => ocean(t, elevation_m),
                LandClass::Water if Self::snowy(class, elevation_m, latitude).is_none() && t < 0.0 => {
                    Self::FrozenRiver
                }
                _ => Self::of(class, elevation_m, latitude),
            };
        };
        if let Some(snow) = Self::snowy(class, elevation_m, latitude) {
            return snow;
        }
        let t = f64::from(climate.temperature_c);
        let p = f64::from(climate.precipitation_mm).max(0.0);
        let r = aridity(t, p);
        let hum = humidity_index(r);
        let row = temperature_row(t, hum);
        let plateau = elevation_m >= 1500.0;
        let pick = vanilla(row, hum, plateau, variant);
        use Biome::*;
        // The dry places: open ground in a dry climate is badlands where rugged, desert or savanna where flat.
        if is_dry(t, r) && matches!(class, LandClass::Bare | LandClass::Shrub | LandClass::Grass) {
            if relief_m >= ERODED_M {
                return ErodedBadlands;
            } else if relief_m >= RUGGED_M {
                return Badlands;
            }
            match class {
                LandClass::Bare => {
                    return if elevation_m >= 1000.0 && row >= Row::Temperate {
                        Badlands
                    } else {
                        Desert
                    };
                }
                // Cold dry open ground (under the temperate row) stays as the table gives it.
                _ if t < SAVANNA_FROM_C => {}
                _ if r < DESERT_ARIDITY => return Desert,
                _ if plateau => return SavannaPlateau,
                _ => return Savanna,
            }
        }
        match class {
            LandClass::Water if elevation_m <= 0.0 => ocean(t, elevation_m),
            LandClass::NoData if elevation_m < 0.0 => ocean(t, elevation_m),
            LandClass::Water => {
                if row == Row::Frozen {
                    FrozenRiver
                } else {
                    River
                }
            }
            LandClass::Tree => {
                if row == Row::Temperate && r >= 6.0 {
                    return OldGrowthSpruceTaiga;
                }
                // Below 0 C the ground is under snow much of the year, and cold air makes almost any rain
                // "humid", which in vanilla's row is plain taiga: a frozen forest is snowy taiga.
                if row == Row::Frozen {
                    return SnowyTaiga;
                }
                let fits = matches!(
                    pick,
                    Forest
                        | FlowerForest
                        | BirchForest
                        | OldGrowthBirchForest
                        | DarkForest
                        | Taiga
                        | OldGrowthSpruceTaiga
                        | OldGrowthPineTaiga
                        | SnowyTaiga
                        | Jungle
                        | SparseJungle
                        | BambooJungle
                        | WoodedBadlands
                ) || (matches!(pick, Savanna | SavannaPlateau) && row >= Row::Warm);
                if fits {
                    pick
                } else {
                    match row {
                        Row::Frozen => SnowyTaiga,
                        Row::Cold => Taiga,
                        Row::Temperate => Forest,
                        Row::Warm if hum >= 3 => Jungle,
                        Row::Warm | Row::Hot => Savanna,
                    }
                }
            }
            LandClass::Grass | LandClass::NoData | LandClass::Shrub => {
                let open = matches!(
                    pick,
                    Plains | SunflowerPlains | SnowyPlains | Savanna | SavannaPlateau | Meadow
                );
                if class == LandClass::Shrub && hum == 0 && row >= Row::Temperate {
                    Desert
                } else if open {
                    pick
                } else {
                    match row {
                        Row::Frozen => SnowyPlains,
                        Row::Cold | Row::Temperate if plateau => Meadow,
                        Row::Cold | Row::Temperate => Plains,
                        Row::Warm | Row::Hot => Savanna,
                    }
                }
            }
            LandClass::Crop => match row {
                Row::Frozen => SnowyPlains,
                Row::Warm | Row::Hot => Savanna,
                _ if pick == SunflowerPlains => SunflowerPlains,
                _ => Plains,
            },
            LandClass::BuiltUp => match row {
                Row::Frozen => SnowyPlains,
                _ => Plains,
            },
            LandClass::Bare => {
                if elevation_m >= 2000.0 {
                    StonyPeaks
                } else {
                    WindsweptGravellyHills
                }
            }
            LandClass::Wetland => {
                if row == Row::Frozen {
                    SnowyPlains
                } else {
                    Swamp
                }
            }
            LandClass::Mangrove => MangroveSwamp,
            LandClass::MossLichen | LandClass::SnowIce => SnowyPlains,
        }
    }
}

/// A rough mean annual temperature where the climate has no data: about 27 C at the equator and -20 C at 80
/// degrees, 6.5 C colder a kilometre up.
pub fn fallback_temperature_c(elevation_m: f64, latitude: f64) -> f64 {
    27.0 - 0.0074 * latitude * latitude - 6.5 * elevation_m.max(0.0) / 1000.0
}

/// The sea at mean air temperature `t`, deep under 200 m.
fn ocean(t: f64, elevation_m: f64) -> Biome {
    let deep = elevation_m < -200.0;
    match (t, deep) {
        (t, _) if t >= 24.0 => Biome::WarmOcean,
        (t, false) if t >= 18.0 => Biome::LukewarmOcean,
        (t, true) if t >= 18.0 => Biome::DeepLukewarmOcean,
        (t, false) if t >= 8.0 => Biome::Ocean,
        (t, true) if t >= 8.0 => Biome::DeepOcean,
        (t, false) if t >= 0.0 => Biome::ColdOcean,
        (t, true) if t >= 0.0 => Biome::DeepColdOcean,
        (_, false) => Biome::FrozenOcean,
        (_, true) => Biome::DeepFrozenOcean,
    }
}

/// Vanilla's temperature rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Row {
    Frozen,
    Cold,
    Temperate,
    Warm,
    Hot,
}

fn humidity_index(r: f64) -> usize {
    if r < 0.5 {
        0
    } else if r < 1.0 {
        1
    } else if r < 2.0 {
        2
    } else if r < 3.5 {
        3
    } else {
        4
    }
}

fn temperature_row(t: f64, hum: usize) -> Row {
    if t < 0.0 {
        Row::Frozen
    } else if t < 8.0 {
        Row::Cold
    } else if t < 18.0 {
        Row::Temperate
    } else if hum <= 1 {
        Row::Hot
    } else {
        Row::Warm
    }
}

/// Vanilla's middle (or plateau) biome for a row and humidity, or its variant when `variant` and there is one.
fn vanilla(row: Row, hum: usize, plateau: bool, variant: bool) -> Biome {
    use Biome::*;
    const MIDDLE: [[Biome; 5]; 5] = [
        [SnowyPlains, SnowyPlains, SnowyPlains, SnowyTaiga, Taiga],
        [Plains, Plains, Forest, Taiga, OldGrowthSpruceTaiga],
        [FlowerForest, Plains, Forest, BirchForest, DarkForest],
        [Savanna, Savanna, Forest, Jungle, Jungle],
        [Desert, Desert, Desert, Desert, Desert],
    ];
    const VARIANT: [[Option<Biome>; 5]; 5] = [
        [Some(IceSpikes), None, Some(SnowyTaiga), None, None],
        [None, None, None, None, Some(OldGrowthPineTaiga)],
        [Some(SunflowerPlains), None, None, Some(OldGrowthBirchForest), None],
        [None, None, Some(Plains), Some(SparseJungle), Some(BambooJungle)],
        [None, None, None, None, None],
    ];
    const PLATEAU: [[Biome; 5]; 5] = [
        [SnowyPlains, SnowyPlains, SnowyPlains, SnowyTaiga, SnowyTaiga],
        [Meadow, Meadow, Forest, Taiga, OldGrowthSpruceTaiga],
        [Meadow, Meadow, Meadow, Meadow, DarkForest],
        [SavannaPlateau, SavannaPlateau, Forest, Forest, Jungle],
        [Badlands, Badlands, Badlands, WoodedBadlands, WoodedBadlands],
    ];
    let (t, h) = (row as usize, hum.min(4));
    if plateau {
        return PLATEAU[t][h];
    }
    match VARIANT[t][h] {
        Some(v) if variant => v,
        _ => MIDDLE[t][h],
    }
}

/// Whether the cell of about 2 km (a fixed step of latitude and longitude, so it is the same at every scale)
/// holding `lat`, `lon` takes the variant biome: a fixed hash of the cell.
pub fn variant_cell(lat: f64, lon: f64) -> bool {
    let step = VARIANT_CELL_M / 111_320.0;
    let (cx, cz) = ((lon / step).floor() as i64, (lat / step).floor() as i64);
    let mut h = (cx as u64).rotate_left(32) ^ (cz as u64) ^ 0x5DEE_CE66_D1CE_4E5B;
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (h ^ (h >> 31)) % 1000 < VARIANT_SHARE
}

/// The elevation (m) a river channel's biome is read at, at least: water above the sea is a river.
const RIVER_BIOME_MIN_M: f64 = 1.0;

impl EarthGenerator {
    /// The biome of the 4 by 4 column `(qx, qz)` (block `x >> 2`), read at its middle: the class and the height
    /// at [`BIOME_SAMPLE_M`] of ground. Without land cover, `plains`.
    pub(super) fn biome_core(&self, a: Access, qx: i32, qz: i32) -> Result<Biome, Gap> {
        let Some(land) = self.land_source() else {
            return Ok(Biome::Plains);
        };
        let (x, z) = (f64::from(qx) * 4.0 + 2.0, f64::from(qz) * 4.0 + 2.0);
        let zoom = self.field.zoom_for(BIOME_SAMPLE_M);
        // All are asked for before any failure is reported, so one pass names everything a column lacks.
        let class = match a {
            Access::Local => match land.field().class_at(x, z, BIOME_SAMPLE_M) {
                Lookup::Class(class) => Ok(class),
                Lookup::NoFile => Ok(LandClass::NoData),
                Lookup::Need(fetch) => {
                    self.note_land(vec![fetch]);
                    Err(Gap::Land)
                }
            },
            Access::Fetch => land
                .class_at_blocking(x, z, BIOME_SAMPLE_M)
                .map_err(|e| Gap::Load(LoadError::Fetch(e))),
        };
        let climate = match self.climate_source() {
            None => Ok(None),
            Some(source) => match a {
                Access::Local => match source.field().climate_at(x, z, BIOME_SAMPLE_M) {
                    ClimateLookup::Climate(c) => Ok(Some(c)),
                    ClimateLookup::NoData => Ok(None),
                    ClimateLookup::Need(fetch) => {
                        self.note_climate(fetch);
                        Err(Gap::Land)
                    }
                },
                Access::Fetch => source
                    .climate_at_blocking(x, z, BIOME_SAMPLE_M)
                    .map_err(|e| Gap::Load(LoadError::Fetch(e))),
            },
        };
        let height = match a {
            Access::Local => self.field.height_m(x, z, zoom),
            Access::Fetch => self
                .field
                .height_m_blocking(x, z, zoom)
                .map_err(Gap::Load)?,
        };
        let class = class?;
        let climate = climate?;
        let elevation = f64::from(height.ok_or(Gap::Missing)?);
        // A river channel of the water atlas through the middle block is a river (frozen by the same rule as
        // water above the sea), whatever the map says there.
        let (class, elevation) = if self
            .river_channel_at(qx.saturating_mul(4) + 2, qz.saturating_mul(4) + 2)
            .is_some()
        {
            (LandClass::Water, elevation.max(RIVER_BIOME_MIN_M))
        } else {
            (class, elevation)
        };
        let at = self.field.projection().to_lat_lon(x, z);
        if !self.climate_wanted() {
            return Ok(Biome::of(class, elevation, at.lat));
        }
        // Only dry open ground reads the neighbours' heights, for how rugged it is.
        let dry = climate.is_some_and(|c| {
            let t = f64::from(c.temperature_c);
            is_dry(t, aridity(t, f64::from(c.precipitation_mm)))
        });
        let relief =
            if dry && matches!(class, LandClass::Bare | LandClass::Shrub | LandClass::Grass) {
                self.relief_m(a, x, z, zoom, elevation)?
            } else {
                0.0
            };
        Ok(Biome::with_relief(
            class,
            elevation,
            at.lat,
            climate,
            variant_cell(at.lat, at.lon),
            relief,
        ))
    }

    /// The relief around the point `(x, z)` whose height at `zoom` is `middle` (m): the range of that height
    /// and the heights half a [`BIOME_SAMPLE_M`] cell away to the north, south, east and west.
    fn relief_m(&self, a: Access, x: f64, z: f64, zoom: u8, middle: f64) -> Result<f64, Gap> {
        let half = self.biome_half_cell();
        let (mut lo, mut hi) = (middle, middle);
        for (dx, dz) in [(half, 0.0), (-half, 0.0), (0.0, half), (0.0, -half)] {
            let h = match a {
                Access::Local => self.field.height_m(x + dx, z + dz, zoom),
                Access::Fetch => self
                    .field
                    .height_m_blocking(x + dx, z + dz, zoom)
                    .map_err(Gap::Load)?,
            };
            let h = f64::from(h.ok_or(Gap::Missing)?);
            lo = lo.min(h);
            hi = hi.max(h);
        }
        Ok(hi - lo)
    }

    /// Half a [`BIOME_SAMPLE_M`] cell, in blocks.
    fn biome_half_cell(&self) -> f64 {
        BIOME_SAMPLE_M / 2.0 / self.field.projection().metres_per_block()
    }

    /// The block rectangle a biome read of the 4 by 4 column `(qx, qz)` covers: the column itself, grown by half
    /// a cell where the biome reads its neighbours' heights (a world with climate).
    fn biome_rect(&self, qx: i32, qz: i32) -> Rect {
        let grow = if self.climate_wanted() && self.climate_source().is_some() {
            self.biome_half_cell().ceil().min(f64::from(1_u32 << 24)) as i32 + 1
        } else {
            0
        };
        Rect {
            x0: qx.saturating_mul(4).saturating_sub(grow),
            z0: qz.saturating_mul(4).saturating_sub(grow),
            x1: qx.saturating_mul(4).saturating_add(3).saturating_add(grow),
            z1: qz.saturating_mul(4).saturating_add(3).saturating_add(grow),
            zoom: self.field.zoom_for(BIOME_SAMPLE_M),
        }
    }

    /// [`EarthGenerator::biome_core`], loading what it lacks. `plains` when cancelled.
    pub fn biome_at_quart_blocking(&self, qx: i32, qz: i32) -> Biome {
        let r = self.biome_rect(qx, qz);
        self.wait_for(r, || self.biome_core(Access::Fetch, qx, qz))
            .unwrap_or(Biome::Plains)
    }

    /// Whether the biomes of the 16 by 16 columns `stride` blocks apart from `(x0, z0)` can be read without
    /// loading anything. When not, what is missing is requested like any other read's. A far view calls this
    /// before it asks for the biomes of a node, so it never waits for them.
    pub fn biomes_ready(&self, x0: i32, z0: i32, stride: i32) -> Result<(), NotReady> {
        if self.land_source().is_none() {
            return Ok(());
        }
        let stride = stride.max(1);
        let mut last = None;
        for j in 0..16 {
            for i in 0..16 {
                let qx = x0.saturating_add(i * stride) >> 2;
                let qz = z0.saturating_add(j * stride) >> 2;
                if last == Some((qx, qz)) {
                    continue;
                }
                last = Some((qx, qz));
                if self.biome_core(Access::Local, qx, qz).is_err() {
                    return Err(NotReady {
                        tiles: self.missing_in(self.biome_rect(qx, qz)),
                    });
                }
            }
        }
        Ok(())
    }
}

/// The biome source of an Earth world with land cover: the table above over a shared generator. A call may load
/// what it needs and wait, as the generator's blocking forms do.
pub struct EarthBiomes {
    generator: Arc<EarthGenerator>,
}

impl EarthBiomes {
    pub fn new(generator: Arc<EarthGenerator>) -> Self {
        Self { generator }
    }
}

impl BiomeColumns for EarthBiomes {
    fn biome(&self, quart_x: i32, quart_z: i32) -> BiomeId {
        let biome = self.generator.biome_at_quart_blocking(quart_x, quart_z);
        BiomeId(Biome::ALL.iter().position(|&b| b == biome).unwrap_or(0) as u32)
    }

    fn biome_name(&self, id: BiomeId) -> &str {
        Biome::ALL
            .get(id.0 as usize)
            .map_or(Biome::Plains.name(), |b| b.name())
    }

    fn possible_biomes(&self) -> Vec<BiomeId> {
        (0..Biome::ALL.len() as u32).map(BiomeId).collect()
    }
}

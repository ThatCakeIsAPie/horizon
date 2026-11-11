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
//! [`EarthBiomes`] gives this to the cubic generator as its biome source.

use std::sync::Arc;

use horizon_earth::{LandClass, LoadError, Lookup};

use super::generator::{Access, EarthGenerator, Gap, NotReady, Rect};
use super::surface::{LAND_SNOW_FLOOR_M, snow_line_m};
use crate::cubic::custom::biomes::BiomeColumns;
use crate::worldgen::biome::BiomeId;

/// The ground resolution, in metres, the biome of a column is read at.
pub const BIOME_SAMPLE_M: f64 = 500.0;

/// The biomes of the table, `plains` first (it is what a cancelled read gives).
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
}

impl Biome {
    /// Every biome, in id order.
    pub const ALL: [Biome; 17] = [
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
    ];

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
        }
    }

    /// The biome of a column of `class` at real `elevation_m` and `latitude` degrees (see the table above).
    pub fn of(class: LandClass, elevation_m: f64, latitude: f64) -> Self {
        let lat = latitude.abs();
        let snowy = class == LandClass::SnowIce
            || elevation_m >= snow_line_m(latitude).max(LAND_SNOW_FLOOR_M);
        if snowy {
            return if elevation_m >= 3000.0 {
                Self::FrozenPeaks
            } else if elevation_m >= 1500.0 {
                Self::SnowySlopes
            } else {
                Self::SnowyPlains
            };
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
}

impl EarthGenerator {
    /// The biome of the 4 by 4 column `(qx, qz)` (block `x >> 2`), read at its middle: the class and the height
    /// at [`BIOME_SAMPLE_M`] of ground. Without land cover, `plains`.
    fn biome_core(&self, a: Access, qx: i32, qz: i32) -> Result<Biome, Gap> {
        let Some(land) = self.land_source() else {
            return Ok(Biome::Plains);
        };
        let (x, z) = (f64::from(qx) * 4.0 + 2.0, f64::from(qz) * 4.0 + 2.0);
        let zoom = self.field.zoom_for(BIOME_SAMPLE_M);
        // Both are asked for before either failure is reported, so one pass names everything a column lacks.
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
        let height = match a {
            Access::Local => self.field.height_m(x, z, zoom),
            Access::Fetch => self
                .field
                .height_m_blocking(x, z, zoom)
                .map_err(Gap::Load)?,
        };
        let class = class?;
        let elevation = f64::from(height.ok_or(Gap::Missing)?);
        let latitude = self.field.projection().to_lat_lon(x, z).lat;
        Ok(Biome::of(class, elevation, latitude))
    }

    /// [`EarthGenerator::biome_core`], loading what it lacks. `plains` when cancelled.
    pub fn biome_at_quart_blocking(&self, qx: i32, qz: i32) -> Biome {
        let r = Rect {
            x0: qx.saturating_mul(4),
            z0: qz.saturating_mul(4),
            x1: qx.saturating_mul(4).saturating_add(3),
            z1: qz.saturating_mul(4).saturating_add(3),
            zoom: self.field.zoom_for(BIOME_SAMPLE_M),
        };
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
                    let r = Rect {
                        x0: qx.saturating_mul(4),
                        z0: qz.saturating_mul(4),
                        x1: qx.saturating_mul(4).saturating_add(3),
                        z1: qz.saturating_mul(4).saturating_add(3),
                        zoom: self.field.zoom_for(BIOME_SAMPLE_M),
                    };
                    return Err(NotReady {
                        tiles: self.missing_in(r),
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

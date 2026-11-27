//! Real-world terrain from public elevation data (the terrarium encoding of the AWS Terrain Tiles).
//!
//! This crate knows nothing about cubes, columns, the game or HTTP. It turns world block coordinates into
//! latitude and longitude ([`projection`]), decodes elevation tiles ([`tile`]), keeps them on disk and in memory
//! with a fetcher the caller supplies ([`source`]), and answers height and height-range questions ([`height`]).
//!
//! Land cover (ESA WorldCover) has the same shape: a TIFF reader over byte slices ([`tiff`]), the classes and file
//! addressing ([`landcover`]) and a cache of decoded tiles that says which byte range to fetch when it lacks one
//! ([`landfield`]); [`landsource`] adds the small trait a downloader implements to answer what the store lacks. It
//! needs no network and no real data.
//!
//! Climate (CHELSA V2.1, 1981-2010, CC0) has that shape too: the two bioclimatic files for mean annual
//! temperature (`bio01`) and annual precipitation (`bio12`) are world-wide 30 arc-second GeoTIFFs, read by byte
//! ranges, never downloaded whole. [`climate`] has the addressing and the georeference, [`climatefield`] the cache
//! and the field, [`climatesource`] the fetcher trait. A place gets the temperature in degrees Celsius and the
//! precipitation in millimetres a year, the bilinear mix of the four nearest pixels, land and sea alike.
//!
//! Water (rivers and lakes) is a small preprocessed atlas shipped inside the game ([`water`], format in
//! [`waterfile`]): river lines with discharge and water levels from RiverATLAS, lake boxes and levels from
//! HydroLAKES, decoded one 1-degree cell at a time on first use.
//!
//! Missing data is never invented: every query that needs a tile the store does not hold says so with `None`.
#![forbid(unsafe_code)]

pub mod clean;
pub mod climate;
pub mod climatefield;
pub mod climatesource;
pub mod fastmap;
pub mod height;
pub mod landcover;
pub mod landfield;
pub mod landsource;
#[cfg(test)]
mod landwarp_tests;
pub mod ore_richness;
pub mod projection;
pub mod source;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod tiff;
pub mod tile;
mod warp;
pub mod water;
pub mod waterfile;
pub mod window;

pub use climate::{CLIMATE_RULES, Climate, ClimateVar};
pub use climatefield::{ClimateFetch, ClimateField, ClimateGrid, ClimateLookup, ClimateStore};
pub use climatesource::{ClimateFetcher, ClimateSource};
pub use height::ElevationField;
pub use landcover::{LAND_COVER_RULES, LandClass, WorldCoverFile};
pub use landfield::gpu::{
    AxisOctave, AxisSample, ClassWindowPlan, NO_TILE, OUTSIDE_PLAN, OctavePlan, PlanFile, PlanTile,
    octave_noise,
};
pub use landfield::{Fetch, GeoBox, LandCoverField, LandCoverStore, Lookup, Wanted};
pub use landsource::{LandCoverFetcher, LandCoverSource};
pub use projection::{LatLon, LocalTangent, Projection, VerticalMap};
pub use source::{FetchError, LoadError, SEA_FLOOR_M, TileFetcher, TileStore};
pub use tile::{Tile, TileError, TileId};
pub use window::{AxisIndex, AxisSample, WindowPlan, WindowPlanError};
pub use water::{Lake, RiverReach, WATER_RULES, WaterAtlas, WaterAtlasError};

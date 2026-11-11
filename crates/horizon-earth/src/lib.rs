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
//! Missing data is never invented: every query that needs a tile the store does not hold says so with `None`.
#![forbid(unsafe_code)]

pub mod clean;
pub mod height;
pub mod landcover;
pub mod landfield;
pub mod landsource;
pub mod projection;
pub mod source;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod tiff;
pub mod tile;

pub use height::ElevationField;
pub use landcover::{LAND_COVER_RULES, LandClass, WorldCoverFile};
pub use landfield::{Fetch, GeoBox, LandCoverField, LandCoverStore, Lookup, Wanted};
pub use landsource::{LandCoverFetcher, LandCoverSource};
pub use projection::{LatLon, LocalTangent, Projection, VerticalMap};
pub use source::{FetchError, LoadError, TileFetcher, TileStore};
pub use tile::{Tile, TileError, TileId};

//! Real-world terrain from public elevation data (the terrarium encoding of the AWS Terrain Tiles).
//!
//! This crate knows nothing about cubes, columns, the game or HTTP. It turns world block coordinates into
//! latitude and longitude ([`projection`]), decodes elevation tiles ([`tile`]), keeps them on disk and in memory
//! with a fetcher the caller supplies ([`source`]), and answers height and height-range questions ([`height`]).
//!
//! Land cover (ESA WorldCover) will have the same shape; its TIFF reader over byte slices is [`tiff`]. It needs no
//! network and no real data, and nothing yet reads it.
//!
//! Missing data is never invented: every query that needs a tile the store does not hold says so with `None`.
#![forbid(unsafe_code)]

pub mod clean;
pub mod height;
pub mod projection;
pub mod source;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod tiff;
pub mod tile;

pub use height::ElevationField;
pub use projection::{LatLon, LocalTangent, Projection, VerticalMap};
pub use source::{FetchError, LoadError, TileFetcher, TileStore};
pub use tile::{Tile, TileError, TileId};

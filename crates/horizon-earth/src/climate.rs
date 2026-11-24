//! Climate for the globe: where the CHELSA V2.1 files are, how to read them, and the arithmetic from a place to
//! a pixel. Pure functions and constants, no I/O.
//!
//! The data is CHELSA V2.1, climatologies 1981-2010, two of its bioclimatic variables: `bio01` (mean annual
//! near-surface air temperature) and `bio12` (annual precipitation). Each is one cloud-optimised GeoTIFF for the
//! whole world, land and sea, 43200 by 20880 pixels of 30 arc-seconds (about 900 m) with overviews, 16-bit
//! unsigned, DEFLATE with predictor 2. A raw value `v` is `v * 0.1 - 273.15` degrees Celsius for `bio01` (the
//! file's own scale and offset) and `v` millimetres a year for `bio12` (it has none: checked against the file's
//! pixels, not assumed); 65535 is "no data". The files are far too big to download whole
//! (150 MB and 350 MB), so they are read by byte ranges ([`crate::landfield`] explains how).
//!
//! Licence: CC0 1.0. Please cite: Karger, D.N., Conrad, O., Böhner, J., Kawohl, T., Kreft, H., Soria-Auza, R.W.,
//! Zimmermann, N.E., Linder, H.P., Kessler, M. (2017) Climatologies at high resolution for the earth's land
//! surface areas. Scientific Data 4, 170122. Data: EnviDat, doi:10.16904/envidat.228.v2.1.
//!
//! The grid is placed on the globe by constants, as GDAL reads it: the top left corner of pixel (0, 0) is at
//! longitude [`WEST`], latitude [`NORTH`]; the extent is 360 degrees wide and 174 tall, and every level (an
//! overview too) spreads that extent over its own pixels. Rows run north to south. Nothing north of
//! [`NORTH`] (about 84 N) or south of [`SOUTH`] has data.

use std::fmt;

use crate::landcover::METRES_PER_DEGREE;
use crate::tiff::Level;

/// Version of everything that turns a CHELSA file's bytes into climate for a place: the file naming, the pixel
/// and level arithmetic, the value scaling and the interpolation. Anything built from climate must key on it,
/// and the on-disk cache keeps each version in its own folder (see [`cache_key`]).
pub const CLIMATE_RULES: u32 = 1;

/// The name of the dataset and its version in cache keys: it changes if the data's edition or period does.
pub const DATASET: &str = "chelsa-v2.1-1981-2010";

/// The text that keys a climate cache on disk and in built data: the dataset and [`CLIMATE_RULES`].
pub fn cache_key() -> String {
    format!("{DATASET}-rules{CLIMATE_RULES}")
}

/// The first bytes of a file asked for: every directory and tile array of a real CHELSA file lies within them.
pub const CLIMATE_PREFIX_LEN: u64 = 64 * 1024;

/// The longitude of the west edge of pixel column 0, in degrees.
pub const WEST: f64 = -180.000_138_888_85;
/// The latitude of the north edge of pixel row 0, in degrees.
pub const NORTH: f64 = 83.999_860_415_15;
/// Degrees of longitude the grid spans.
pub const EXTENT_LON: f64 = 360.0;
/// Degrees of latitude the grid spans.
pub const EXTENT_LAT: f64 = 174.0;
/// The latitude of the south edge of the last pixel row, in degrees.
pub const SOUTH: f64 = NORTH - EXTENT_LAT;

/// The raw value of a pixel with no data.
pub const NODATA: u16 = 65535;

/// Which of the two variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClimateVar {
    /// Mean annual near-surface air temperature (`bio01`).
    Temperature,
    /// Annual precipitation (`bio12`).
    Precipitation,
}

impl ClimateVar {
    pub const ALL: [ClimateVar; 2] = [ClimateVar::Temperature, ClimateVar::Precipitation];

    /// CHELSA's short name for it: `bio01` or `bio12`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Temperature => "bio01",
            Self::Precipitation => "bio12",
        }
    }

    /// The path of its file below the dataset's base address (which the caller owns).
    pub fn path(self) -> &'static str {
        match self {
            Self::Temperature => "bio01/1981-2010/CHELSA_bio01_1981-2010_V.2.1.tif",
            Self::Precipitation => "bio12/1981-2010/CHELSA_bio12_1981-2010_V.2.1.tif",
        }
    }

    /// The value of a raw pixel: degrees Celsius (tenths of a kelvin less 273.15) or millimetres a year (the raw
    /// value). `None` for no data.
    pub fn value(self, raw: u16) -> Option<f32> {
        self.value_f64(raw).map(|v| v as f32)
    }

    pub(crate) fn value_f64(self, raw: u16) -> Option<f64> {
        if raw == NODATA {
            return None;
        }
        Some(match self {
            Self::Temperature => f64::from(raw) * 0.1 - 273.15,
            Self::Precipitation => f64::from(raw),
        })
    }
}

impl fmt::Display for ClimateVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The climate of a place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Climate {
    /// Mean annual air temperature, degrees Celsius.
    pub temperature_c: f32,
    /// Annual precipitation, millimetres.
    pub precipitation_mm: f32,
}

/// Whether `(lat, lon)` is a place the grid covers: finite, and between [`SOUTH`] and [`NORTH`].
pub fn on_grid(lat: f64, lon: f64) -> bool {
    lat.is_finite() && lon.is_finite() && lat <= NORTH && lat > SOUTH
}

/// The continuous pixel position of `(lat, lon)` on a level of `width` by `height` pixels: `(x, y)` where the
/// pixel `(i, j)` covers `i..i + 1` and `j..j + 1` (so its centre is `i + 0.5`). The column wraps around the date
/// line into `0..width`. `None` for a latitude outside the grid or input that is not finite.
pub fn pixel_position(lat: f64, lon: f64, width: u32, height: u32) -> Option<(f64, f64)> {
    if !on_grid(lat, lon) || width == 0 || height == 0 {
        return None;
    }
    let (w, h) = (f64::from(width), f64::from(height));
    let mut x = ((lon - WEST) / EXTENT_LON * w).rem_euclid(w);
    if x >= w {
        // rem_euclid can round up to the modulus for a tiny negative input.
        x = 0.0;
    }
    let y = (NORTH - lat) / EXTENT_LAT * h;
    (y >= 0.0 && y < h).then_some((x, y))
}

/// Metres of ground one pixel row of `level` covers.
pub fn level_metres_per_pixel(level: &Level) -> f64 {
    EXTENT_LAT / f64::from(level.height.max(1)) * METRES_PER_DEGREE
}

/// The index of the coarsest level whose pixels are no taller on the ground than `metres_per_sample`, or the
/// finest (0) when even that is coarser than asked, or when the request is not a positive number. `levels` runs
/// from the finest to the coarsest. Level 0 is about 927 m a pixel, so anything finer than about 1.85 km reads it.
pub fn level_for(levels: &[Level], metres_per_sample: f64) -> usize {
    if metres_per_sample.is_nan() || metres_per_sample <= 0.0 {
        return 0;
    }
    levels
        .iter()
        .rposition(|level| level_metres_per_pixel(level) <= metres_per_sample)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::empty_level;

    #[test]
    fn the_cache_key_names_the_dataset_and_the_rules() {
        assert_eq!(cache_key(), "chelsa-v2.1-1981-2010-rules1");
        assert!(!cache_key().contains(['/', '#', ' ']));
    }

    #[test]
    fn raw_values_scale_to_degrees_and_millimetres_and_65535_is_no_data() {
        assert_eq!(ClimateVar::Temperature.value(2998), Some(26.65));
        assert_eq!(ClimateVar::Precipitation.value(2449), Some(2449.0));
        assert_eq!(ClimateVar::Temperature.value(NODATA), None);
        assert_eq!(ClimateVar::Precipitation.value(NODATA), None);
        assert_eq!(ClimateVar::Temperature.name(), "bio01");
        assert!(ClimateVar::Precipitation.path().contains("bio12"));
    }

    #[test]
    fn places_map_to_pixels_at_the_corners_and_wrap_at_the_date_line() {
        let (w, h) = (43200, 20880);
        let (x, y) = pixel_position(NORTH - 1e-9, WEST + 1e-9, w, h).unwrap();
        assert!(x < 1e-3 && y < 1e-3, "{x} {y}");
        let (x, y) = pixel_position(SOUTH + 1e-9, 179.9998, w, h).unwrap();
        assert!(x > 43199.9 && y > 20879.9, "{x} {y}");
        // 180 and -180 are the same meridian, a hair east of the west edge.
        let (a, _) = pixel_position(0.0, 180.0, w, h).unwrap();
        let (b, _) = pixel_position(0.0, -180.0, w, h).unwrap();
        assert!((a - b).abs() < 1e-6 && a < 0.1, "{a} {b}");
        // The hand-checked Amazon pixel.
        let (x, y) = pixel_position(-3.1, -60.0, w, h).unwrap();
        assert_eq!((x.floor() as u32, y.floor() as u32), (14400, 10451));
        // Not on the grid.
        assert_eq!(pixel_position(84.5, 0.0, w, h), None);
        assert_eq!(pixel_position(-90.001, 0.0, w, h), None);
        assert_eq!(pixel_position(f64::NAN, 0.0, w, h), None);
        assert_eq!(pixel_position(0.0, f64::INFINITY, w, h), None);
    }

    #[test]
    fn the_level_follows_the_requested_ground_size() {
        let levels: Vec<Level> = [20880, 10440, 5220, 2610]
            .into_iter()
            .map(|h| empty_level(h * 2, h))
            .collect();
        // About 927, 1854, 3708 and 7416 m a pixel.
        assert_eq!(level_for(&levels, 10.0), 0);
        assert_eq!(level_for(&levels, 1000.0), 0);
        assert_eq!(level_for(&levels, 1800.0), 0);
        assert_eq!(level_for(&levels, 1900.0), 1);
        assert_eq!(level_for(&levels, 4000.0), 2);
        assert_eq!(level_for(&levels, 1e7), 3);
        assert_eq!(level_for(&levels, f64::NAN), 0);
        assert_eq!(level_for(&levels, -1.0), 0);
        assert_eq!(level_for(&[], 100.0), 0);
        let m = level_metres_per_pixel(&levels[0]);
        assert!((m - 927.0).abs() < 1.0, "{m}");
    }
}

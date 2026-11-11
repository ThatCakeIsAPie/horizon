//! Land cover from ESA WorldCover 2021 (v200): the classes, which file holds a place, which pixel, and which
//! level of the file's overviews suits a requested ground resolution.
//!
//! WorldCover is one file per 3° by 3° of the globe, named by its south-west corner, at 10 m per pixel with
//! overviews down to about 590 m. A pixel's value is its class code. This module only does the arithmetic and
//! names things; reading the files is [`crate::tiff`] and the cache is [`crate::landfield`].

use std::fmt;

use crate::tiff::Level;

/// Degrees of latitude and longitude a file spans.
pub const FILE_DEGREES: i32 = 3;

/// Metres of ground in one degree of latitude (and of longitude at the equator) on the sphere the projection uses.
pub const METRES_PER_DEGREE: f64 = crate::projection::EARTH_RADIUS_M * std::f64::consts::PI / 180.0;

/// Version of everything that turns a WorldCover file's bytes into classes for a place: the file naming, the
/// pixel and level arithmetic, the class table and the decoding. Anything built from land cover (the far-view
/// cache) must key on it, as with [`crate::clean::CLEAN_RULES`], and the on-disk cache keeps each version in
/// its own folder (see [`cache_key`]).
pub const LAND_COVER_RULES: u32 = 1;

/// The name of the dataset and its version in cache keys: it changes if the data's edition or version does.
pub const DATASET: &str = "worldcover-2021-v200";

/// The text that keys a land-cover cache on disk and in built data: the dataset and [`LAND_COVER_RULES`].
pub fn cache_key() -> String {
    format!("{DATASET}-rules{LAND_COVER_RULES}")
}

/// What the ground is, by WorldCover's classes. The numbers are the pixel values in the files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LandClass {
    /// 0: the file has no data here.
    NoData,
    /// 10: tree cover.
    Tree,
    /// 20: shrubland.
    Shrub,
    /// 30: grassland.
    Grass,
    /// 40: cropland.
    Crop,
    /// 50: built-up.
    BuiltUp,
    /// 60: bare or sparsely vegetated.
    Bare,
    /// 70: snow and ice.
    SnowIce,
    /// 80: permanent water bodies.
    Water,
    /// 90: herbaceous wetland.
    Wetland,
    /// 95: mangroves.
    Mangrove,
    /// 100: moss and lichen.
    MossLichen,
}

impl LandClass {
    /// Every class, in code order.
    pub const ALL: [LandClass; 12] = [
        Self::NoData,
        Self::Tree,
        Self::Shrub,
        Self::Grass,
        Self::Crop,
        Self::BuiltUp,
        Self::Bare,
        Self::SnowIce,
        Self::Water,
        Self::Wetland,
        Self::Mangrove,
        Self::MossLichen,
    ];

    /// The class a pixel value stands for, or `None` for a value the dataset does not define.
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::NoData,
            10 => Self::Tree,
            20 => Self::Shrub,
            30 => Self::Grass,
            40 => Self::Crop,
            50 => Self::BuiltUp,
            60 => Self::Bare,
            70 => Self::SnowIce,
            80 => Self::Water,
            90 => Self::Wetland,
            95 => Self::Mangrove,
            100 => Self::MossLichen,
            _ => return None,
        })
    }

    /// The pixel value of the class.
    pub fn code(self) -> u8 {
        match self {
            Self::NoData => 0,
            Self::Tree => 10,
            Self::Shrub => 20,
            Self::Grass => 30,
            Self::Crop => 40,
            Self::BuiltUp => 50,
            Self::Bare => 60,
            Self::SnowIce => 70,
            Self::Water => 80,
            Self::Wetland => 90,
            Self::Mangrove => 95,
            Self::MossLichen => 100,
        }
    }

    /// A short English name, for logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::NoData => "no data",
            Self::Tree => "tree cover",
            Self::Shrub => "shrubland",
            Self::Grass => "grassland",
            Self::Crop => "cropland",
            Self::BuiltUp => "built-up",
            Self::Bare => "bare or sparse vegetation",
            Self::SnowIce => "snow and ice",
            Self::Water => "water",
            Self::Wetland => "herbaceous wetland",
            Self::Mangrove => "mangroves",
            Self::MossLichen => "moss and lichen",
        }
    }
}

/// One WorldCover file, by its south-west corner in degrees (both multiples of 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorldCoverFile {
    /// Latitude of the south edge, -90 to 87.
    pub south: i16,
    /// Longitude of the west edge, -180 to 177.
    pub west: i16,
}

impl WorldCoverFile {
    /// The file that holds the point, with longitude wrapped into `[-180, 180)`. The north pole is in the
    /// northernmost row and the south pole in the southernmost. `None` when either number is not finite or the
    /// latitude is beyond ±90.
    pub fn containing(lat: f64, lon: f64) -> Option<Self> {
        if !(lat.is_finite() && lon.is_finite() && lat.abs() <= 90.0) {
            return None;
        }
        let lon = crate::projection::wrap_longitude(lon);
        let step = f64::from(FILE_DEGREES);
        // The pole itself is the top edge of the row below it.
        let south = ((lat / step).floor() * step).min(90.0 - step);
        let west = ((lon / step).floor() * step).clamp(-180.0, 180.0 - step);
        Some(Self {
            south: south as i16,
            west: west as i16,
        })
    }

    /// The file's name stem as the bucket spells it: `N27E084`, `S03W003`, `N00E000`.
    pub fn tile_name(self) -> String {
        format!(
            "{}{:02}{}{:03}",
            if self.south < 0 { 'S' } else { 'N' },
            self.south.unsigned_abs(),
            if self.west < 0 { 'W' } else { 'E' },
            self.west.unsigned_abs()
        )
    }

    /// The path of the file's map inside the dataset:
    /// `v200/2021/map/ESA_WorldCover_10m_2021_v200_N27E084_Map.tif`. It is a path, not an address: the host
    /// comes from the caller.
    pub fn path(self) -> String {
        format!(
            "v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif",
            self.tile_name()
        )
    }

    /// The latitude of the north edge.
    pub fn north(self) -> f64 {
        f64::from(self.south) + f64::from(FILE_DEGREES)
    }

    /// The longitude of the east edge.
    pub fn east(self) -> f64 {
        f64::from(self.west) + f64::from(FILE_DEGREES)
    }

    /// The pixel of a level `width` by `height` that holds the point, counted from the file's north-west corner,
    /// treating each pixel as the area it covers. A point on or past an edge gets the nearest pixel inside, so
    /// the result is always a pixel of the level. The point should be in this file (see
    /// [`WorldCoverFile::containing`]); longitude is taken relative to the file's west edge, wrapped to -180..180, so a point just west of the file gets column 0.
    pub fn pixel(self, lat: f64, lon: f64, width: u32, height: u32) -> (u32, u32) {
        let step = f64::from(FILE_DEGREES);
        let east = crate::projection::wrap_longitude(lon - f64::from(self.west));
        let x = (east / step * f64::from(width)).floor();
        let y = ((self.north() - lat) / step * f64::from(height)).floor();
        // `as` saturates and turns NaN into 0, and the clamp does the rest.
        (
            (x.max(0.0) as u32).min(width.saturating_sub(1)),
            (y.max(0.0) as u32).min(height.saturating_sub(1)),
        )
    }
}

impl fmt::Display for WorldCoverFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.tile_name())
    }
}

/// Ground size of one pixel of a level, north to south, in metres. (East to west it is smaller by the cosine
/// of the latitude.)
pub fn level_metres_per_pixel(level: &Level) -> f64 {
    f64::from(FILE_DEGREES) * METRES_PER_DEGREE / f64::from(level.height.max(1))
}

/// The index of the coarsest level whose pixels are no taller on the ground than `metres_per_sample`, or the
/// finest (0) when even that is coarser than asked, or when the request is not a positive number. `levels` runs
/// from the finest to the coarsest, as a file's directories do. Returns 0 for no levels.
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

    fn file(lat: f64, lon: f64) -> String {
        WorldCoverFile::containing(lat, lon).unwrap().tile_name()
    }

    #[test]
    fn the_class_codes_are_the_datasets() {
        let codes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100];
        for (class, code) in LandClass::ALL.iter().zip(codes) {
            assert_eq!(class.code(), code);
            assert_eq!(LandClass::from_code(code), Some(*class));
            assert!(!class.name().is_empty());
        }
        for code in 0..=255u8 {
            let known = codes.contains(&code);
            assert_eq!(LandClass::from_code(code).is_some(), known, "{code}");
        }
    }

    #[test]
    fn names_follow_the_south_west_corner() {
        // The research note's example file holds the Himalayas around Everest (27.99 N, 86.93 E): N27E084.
        assert_eq!(file(27.99, 86.93), "N27E084");
        assert_eq!(file(48.5, 9.5), "N48E009");
        assert_eq!(
            WorldCoverFile::containing(27.99, 86.93).unwrap().path(),
            "v200/2021/map/ESA_WorldCover_10m_2021_v200_N27E084_Map.tif"
        );
    }

    #[test]
    fn the_equator_and_the_prime_meridian_belong_to_the_north_east_file() {
        assert_eq!(file(0.0, 0.0), "N00E000");
        assert_eq!(file(2.999, 2.999), "N00E000");
        assert_eq!(file(3.0, 3.0), "N03E003");
        assert_eq!(file(-0.0001, -0.0001), "S03W003");
        assert_eq!(file(-3.0, -3.0), "S03W003");
        assert_eq!(file(-3.0001, -3.0001), "S06W006");
    }

    #[test]
    fn negative_latitudes_and_longitudes_name_their_south_and_west_edges() {
        assert_eq!(file(-33.9, 18.4), "S36E018");
        assert_eq!(file(40.7, -74.0), "N39W075");
        assert_eq!(file(-12.0, -77.0), "S12W078");
        assert_eq!(file(-12.0, -75.0), "S12W075");
    }

    #[test]
    fn the_date_line_wraps() {
        assert_eq!(file(0.0, 180.0), "N00W180");
        assert_eq!(file(0.0, -180.0), "N00W180");
        assert_eq!(file(0.0, 179.999), "N00E177");
        assert_eq!(file(0.0, 181.0), "N00W180");
        assert_eq!(file(0.0, -181.0), "N00E177");
        assert_eq!(file(0.0, 540.0), "N00W180");
    }

    #[test]
    fn the_poles_are_in_the_last_rows() {
        assert_eq!(file(90.0, 0.0), "N87E000");
        assert_eq!(file(89.999, 10.0), "N87E009");
        assert_eq!(file(-90.0, 0.0), "S90E000");
        assert_eq!(file(-89.999, -10.0), "S90W012");
        assert_eq!(file(87.0, 0.0), "N87E000");
        assert_eq!(file(86.999, 0.0), "N84E000");
        assert_eq!(file(-87.0, 0.0), "S87E000");
    }

    #[test]
    fn places_that_are_not_places_have_no_file() {
        assert!(WorldCoverFile::containing(90.1, 0.0).is_none());
        assert!(WorldCoverFile::containing(-90.1, 0.0).is_none());
        assert!(WorldCoverFile::containing(f64::NAN, 0.0).is_none());
        assert!(WorldCoverFile::containing(0.0, f64::INFINITY).is_none());
        assert!(WorldCoverFile::containing(f64::NEG_INFINITY, 0.0).is_none());
    }

    #[test]
    fn every_file_name_has_the_bucket_shape() {
        for lat in (-90..90).step_by(3) {
            for lon in (-180..180).step_by(3) {
                let at =
                    WorldCoverFile::containing(f64::from(lat) + 1.5, f64::from(lon) + 1.5).unwrap();
                assert_eq!((i32::from(at.south), i32::from(at.west)), (lat, lon));
                let name = at.tile_name();
                assert_eq!(name.len(), 7, "{name}");
                let b = name.as_bytes();
                assert!(b[0] == b'N' || b[0] == b'S');
                assert!(b[3] == b'E' || b[3] == b'W');
                assert!(name[1..3].bytes().all(|c| c.is_ascii_digit()));
                assert!(name[4..7].bytes().all(|c| c.is_ascii_digit()));
            }
        }
    }

    #[test]
    fn pixels_are_counted_from_the_north_west_corner() {
        let f = WorldCoverFile::containing(27.5, 84.5).unwrap();
        assert_eq!((f.south, f.west), (27, 84));
        // The north-west corner of the file is its first pixel; the south-east corner its last.
        assert_eq!(f.pixel(30.0, 84.0, 36000, 36000), (0, 0));
        assert_eq!(f.pixel(27.0, 87.0, 36000, 36000), (35999, 35999));
        // The middle of the file.
        assert_eq!(f.pixel(28.5, 85.5, 36000, 36000), (18000, 18000));
        // A tenth of a pixel in from the north-west corner stays in pixel 0, a pixel and a bit moves on.
        let tenth = 3.0 / 36000.0;
        assert_eq!(
            f.pixel(30.0 - tenth * 0.1, 84.0 + tenth * 0.1, 36000, 36000),
            (0, 0)
        );
        assert_eq!(
            f.pixel(30.0 - tenth * 1.1, 84.0 + tenth * 1.1, 36000, 36000),
            (1, 1)
        );
        // The same point at an overview that is a quarter of the size.
        assert_eq!(f.pixel(28.5, 85.5, 9000, 9000), (4500, 4500));
    }

    #[test]
    fn pixels_beyond_an_edge_are_clamped_into_the_level() {
        let f = WorldCoverFile::containing(27.5, 84.5).unwrap();
        assert_eq!(f.pixel(31.0, 83.0, 100, 50), (0, 0));
        assert_eq!(f.pixel(20.0, 90.0, 100, 50), (99, 49));
        assert_eq!(f.pixel(f64::NAN, f64::NAN, 100, 50), (0, 0));
        assert_eq!(f.pixel(28.0, 85.0, 1, 1), (0, 0));
    }

    #[test]
    fn pixels_across_the_date_line_stay_in_the_west_file() {
        let f = WorldCoverFile::containing(0.0, 180.0).unwrap();
        assert_eq!(f.west, -180);
        // 180 east is the file's west edge, 179 east its far (east) edge's neighbour.
        assert_eq!(f.pixel(1.0, 180.0, 300, 300), (0, 200));
        assert_eq!(f.pixel(1.0, -179.0, 300, 300), (100, 200));
        let last = WorldCoverFile::containing(0.0, 179.0).unwrap();
        assert_eq!(last.west, 177);
        assert_eq!(last.pixel(1.0, 179.9, 300, 300), (290, 200));
    }

    #[test]
    fn pixels_at_the_poles_are_in_range() {
        let north = WorldCoverFile::containing(90.0, 0.0).unwrap();
        assert_eq!(north.pixel(90.0, 1.0, 300, 300).1, 0);
        let south = WorldCoverFile::containing(-90.0, 0.0).unwrap();
        assert_eq!(south.pixel(-90.0, 1.0, 300, 300).1, 299);
    }

    fn level(height: u32) -> Level {
        crate::testing::empty_level(height, height)
    }

    #[test]
    fn the_level_is_the_coarsest_that_is_fine_enough() {
        // 36000, 18000, ... 562 rows: about 9.3, 18.6, 37, 74, 148, 297 and 594 metres.
        let levels: Vec<Level> = [36000, 18000, 9000, 4500, 2250, 1125, 562]
            .map(level)
            .to_vec();
        assert_eq!(level_for(&levels, 10.0), 0);
        assert_eq!(level_for(&levels, 9.3), 0);
        assert_eq!(level_for(&levels, 20.0), 1);
        assert_eq!(level_for(&levels, 40.0), 2);
        assert_eq!(level_for(&levels, 100.0), 3);
        assert_eq!(level_for(&levels, 200.0), 4);
        assert_eq!(level_for(&levels, 300.0), 5);
        assert_eq!(level_for(&levels, 600.0), 6);
        assert_eq!(level_for(&levels, 100_000.0), 6);
        // Finer than the data has: the finest.
        assert_eq!(level_for(&levels, 1.0), 0);
        assert_eq!(level_for(&levels, 0.0), 0);
        assert_eq!(level_for(&levels, -5.0), 0);
        assert_eq!(level_for(&levels, f64::NAN), 0);
        assert_eq!(level_for(&levels, f64::INFINITY), 6);
        assert_eq!(level_for(&[], 100.0), 0);
        assert_eq!(level_for(&levels[..1], 5000.0), 0);
    }

    #[test]
    fn the_cache_key_names_the_dataset_and_the_rules() {
        let key = cache_key();
        assert!(key.starts_with("worldcover-2021-v200"), "{key}");
        assert!(key.ends_with(&format!("rules{LAND_COVER_RULES}")), "{key}");
        assert!(!key.contains(['/', '#', ' ']));
    }
}

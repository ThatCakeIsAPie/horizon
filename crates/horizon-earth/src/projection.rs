//! Web Mercator tile maths, the mapping from world blocks to latitude and longitude, and the vertical scale.
//!
//! World axes follow Minecraft: +x is east and +z is south. Block `(x, z)` covers the area from `(x, z)` to
//! `(x + 1, z + 1)`; the world origin `(0, 0)` is the corner the projection is anchored on.

use std::f64::consts::PI;
use std::fmt;

use crate::tile::{TILE_SIZE, TileId};

/// The radius the local mapping uses for metres on the ground (the WGS 84 equatorial radius).
pub const EARTH_RADIUS_M: f64 = 6_378_137.0;
/// The northern and southern limit of Web Mercator, in degrees of latitude.
pub const MAX_LATITUDE: f64 = 85.051_128_779_806_59;
/// The lowest ground the world has (the Mariana Trench), in metres.
pub const MIN_ELEVATION_M: f64 = -11_000.0;
/// The highest ground the world has (Everest), in metres.
pub const MAX_ELEVATION_M: f64 = 8_849.0;

/// A position on the globe, in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatLon {
    pub lat: f64,
    pub lon: f64,
}

/// Clamps a latitude to the range Web Mercator can show.
pub fn clamp_latitude(lat: f64) -> f64 {
    lat.clamp(-MAX_LATITUDE, MAX_LATITUDE)
}

/// Brings a longitude into `[-180, 180)`.
pub fn wrap_longitude(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

/// The latitude of the poles, in degrees.
pub const POLE_LATITUDE: f64 = 90.0;

/// Folds a latitude that is past a pole back onto the globe, and says whether the fold put the point on the far side
/// (its longitude is then 180 degrees further round). A latitude of 100 comes back as 80 on the opposite meridian,
/// 190 as -10 on it, 360 as 0 on the original one, so the map is continuous through both poles however far the
/// input goes. A latitude within `[-90, 90]` (or not a finite number) is returned as it is, exactly.
pub(crate) fn fold_latitude(lat: f64) -> (f64, bool) {
    if !(lat.abs() > POLE_LATITUDE && lat.is_finite()) {
        return (lat, false);
    }
    let q = lat + POLE_LATITUDE;
    let n = (q / 180.0).floor();
    let r = (q - n * 180.0).clamp(0.0, 180.0);
    if n.rem_euclid(2.0) == 1.0 {
        (
            (POLE_LATITUDE - r).clamp(-POLE_LATITUDE, POLE_LATITUDE),
            true,
        )
    } else {
        (
            (r - POLE_LATITUDE).clamp(-POLE_LATITUDE, POLE_LATITUDE),
            false,
        )
    }
}

/// Folds a latitude and longitude, in degrees, onto the globe: a latitude past +-90 comes back down on the other side
/// of the pole with the longitude 180 degrees further round, so the mapping is continuous through the poles, and
/// the longitude then wraps into `[-180, 180)`. Any finite input is valid, however many times it laps the globe.
pub fn fold_globe(lat: f64, lon: f64) -> LatLon {
    let (lat, far_side) = fold_latitude(lat);
    LatLon {
        lat,
        lon: wrap_longitude(if far_side { lon + 180.0 } else { lon }),
    }
}

/// The width, in pixels, of the whole world at `zoom` (it is as tall as it is wide).
pub fn world_pixels(zoom: u8) -> f64 {
    f64::from(TILE_SIZE) * (1u64 << zoom) as f64
}

/// Latitude and longitude to fractional pixel coordinates over the whole world at `zoom`: `(0, 0)` is the north-west
/// corner of tile `(0, 0)`, and pixel `i` spans `[i, i + 1)`. Latitude is clamped to the Mercator limit.
pub fn latlon_to_pixel(lat: f64, lon: f64, zoom: u8) -> (f64, f64) {
    (pixel_x(lon, zoom), pixel_y(lat, zoom))
}

/// The pixel column of a longitude ([`latlon_to_pixel`]'s `x`, which depends on the longitude alone).
pub(crate) fn pixel_x(lon: f64, zoom: u8) -> f64 {
    let size = world_pixels(zoom);
    (lon + 180.0) / 360.0 * size
}

/// The pixel row of a latitude ([`latlon_to_pixel`]'s `y`, which depends on the latitude alone).
pub(crate) fn pixel_y(lat: f64, zoom: u8) -> f64 {
    let size = world_pixels(zoom);
    (1.0 - clamp_latitude(lat).to_radians().tan().asinh() / PI) / 2.0 * size
}

/// The inverse of [`latlon_to_pixel`], with the longitude in `[-180, 180)` when the pixel is inside the world.
pub fn pixel_to_latlon(x: f64, y: f64, zoom: u8) -> LatLon {
    let size = world_pixels(zoom);
    let lon = x / size * 360.0 - 180.0;
    let lat = (PI * (1.0 - 2.0 * y / size)).sinh().atan().to_degrees();
    LatLon { lat, lon }
}

/// Latitude and longitude to the tile that holds the point and the fractional pixel position inside that tile.
/// The longitude wraps and the latitude is clamped, so the tile is always valid.
pub fn latlon_to_tile_pixel(lat: f64, lon: f64, zoom: u8) -> (TileId, f64, f64) {
    let size = world_pixels(zoom);
    let (x, y) = latlon_to_pixel(lat, wrap_longitude(lon), zoom);
    let x = x.rem_euclid(size);
    let y = y.clamp(0.0, size - 1e-9);
    let tile = f64::from(TILE_SIZE);
    let (tx, ty) = ((x / tile).floor(), (y / tile).floor());
    let id = TileId {
        z: zoom,
        x: tx as u32,
        y: ty as u32,
    };
    (id, x - tx * tile, y - ty * tile)
}

/// Why a projection could not be built.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectionError {
    /// The scale must be a positive, finite number of metres per block.
    BadScale(f64),
    /// The origin must lie inside Web Mercator's latitude range and have a finite longitude.
    BadOrigin { lat: f64, lon: f64 },
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadScale(s) => {
                write!(f, "scale must be positive and finite, got {s} m per block")
            }
            Self::BadOrigin { lat, lon } => {
                write!(
                    f,
                    "origin ({lat}, {lon}) must be within ±{MAX_LATITUDE:.4}° latitude"
                )
            }
        }
    }
}

impl std::error::Error for ProjectionError {}

/// A flat map laid over the globe around one origin: east metres = R·cos(lat0)·Δlon, north metres = R·Δlat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalTangent {
    origin: LatLon,
    metres_per_block: f64,
}

impl LocalTangent {
    /// A mapping whose world origin is at `lat0`, `lon0` (degrees) and whose blocks are `metres_per_block` wide.
    pub fn new(lat0: f64, lon0: f64, metres_per_block: f64) -> Result<Self, ProjectionError> {
        if !(metres_per_block.is_finite() && metres_per_block > 0.0) {
            return Err(ProjectionError::BadScale(metres_per_block));
        }
        if !(lat0.abs() <= MAX_LATITUDE && lon0.is_finite()) {
            return Err(ProjectionError::BadOrigin {
                lat: lat0,
                lon: lon0,
            });
        }
        let origin = LatLon {
            lat: lat0,
            lon: wrap_longitude(lon0),
        };
        Ok(Self {
            origin,
            metres_per_block,
        })
    }

    fn metres_per_degree_lon(&self) -> f64 {
        EARTH_RADIUS_M * self.origin.lat.to_radians().cos().to_radians()
    }
}

/// How world blocks map to the globe. One kind exists today; a whole-globe mapping can be added as another
/// variant without changing the callers, which only use the methods below.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Tangent(LocalTangent),
}

impl Projection {
    /// A local tangent mapping; see [`LocalTangent::new`].
    pub fn local(lat0: f64, lon0: f64, metres_per_block: f64) -> Result<Self, ProjectionError> {
        LocalTangent::new(lat0, lon0, metres_per_block).map(Self::Tangent)
    }

    /// The world point `(x, z)` (fractional blocks) to latitude and longitude on the globe: the latitude is folded
    /// over the poles ([`fold_globe`]: past +-90 it comes back down on the other side, continuous through the
    /// pole, with the longitude 180 degrees further round) and the longitude wraps into `[-180, 180)`, so far from
    /// the origin the point still lands on the globe, however many laps round it is. The latitude is in
    /// `[-90, 90]`, so it can pass the Mercator limit (85.05) near the poles.
    pub fn to_lat_lon(&self, x: f64, z: f64) -> LatLon {
        let at = self.to_lat_lon_unwrapped(x, z);
        fold_globe(at.lat, at.lon)
    }

    /// The point as the datasets with a southern and northern edge see it: the latitude **not folded** (it may pass
    /// +-90, where those datasets have no data) and the longitude wrapped into `[-180, 180)`. This is what
    /// [`Projection::to_lat_lon`] was before the fold; the land cover and climate readers keep it, because their
    /// separable window plans cannot express the half-lap longitude the fold adds.
    pub fn to_lat_lon_flat(&self, x: f64, z: f64) -> LatLon {
        let at = self.to_lat_lon_unwrapped(x, z);
        LatLon {
            lat: at.lat,
            lon: wrap_longitude(at.lon),
        }
    }

    /// The raw mapping, **not folded**: the latitude may pass +-90 (and keeps counting) and the longitude keeps
    /// counting past +-180 degrees, so two points can be compared to see how far apart they are, and a box of
    /// blocks becomes a box of degrees. Use [`Projection::to_lat_lon`] for a place on the globe.
    pub fn to_lat_lon_unwrapped(&self, x: f64, z: f64) -> LatLon {
        match self {
            Self::Tangent(t) => {
                let east = x * t.metres_per_block;
                let north = -z * t.metres_per_block;
                LatLon {
                    lat: t.origin.lat + (north / EARTH_RADIUS_M).to_degrees(),
                    lon: t.origin.lon + east / t.metres_per_degree_lon(),
                }
            }
        }
    }

    /// The inverse of [`Projection::to_lat_lon`] near the origin, taking the short way round the globe for
    /// longitude. Once the map folds over the poles and laps, a point of the globe has many world positions; this
    /// gives the one nearest the origin's meridian and latitude.
    pub fn from_lat_lon(&self, lat: f64, lon: f64) -> (f64, f64) {
        match self {
            Self::Tangent(t) => {
                let east = wrap_longitude(lon - t.origin.lon) * t.metres_per_degree_lon();
                let north = (lat - t.origin.lat).to_radians() * EARTH_RADIUS_M;
                (east / t.metres_per_block, -north / t.metres_per_block)
            }
        }
    }

    /// Metres on the ground that one block spans.
    pub fn metres_per_block(&self) -> f64 {
        match self {
            Self::Tangent(t) => t.metres_per_block,
        }
    }

    /// The latitude at which ground resolution is judged (the origin's).
    pub fn reference_latitude(&self) -> f64 {
        match self {
            Self::Tangent(t) => t.origin.lat,
        }
    }
}

/// How elevation maps to block height: `y = sea_level_y + elevation · exaggeration / metres_per_block`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerticalMap {
    pub sea_level_y: i32,
    pub metres_per_block: f64,
    pub exaggeration: f64,
}

impl VerticalMap {
    /// True scale (an exaggeration of 1.0): a block is `metres_per_block` tall, like it is wide.
    pub fn new(sea_level_y: i32, metres_per_block: f64) -> Self {
        Self {
            sea_level_y,
            metres_per_block,
            exaggeration: 1.0,
        }
    }

    /// The same map with heights multiplied by `exaggeration`.
    pub fn with_exaggeration(mut self, exaggeration: f64) -> Self {
        self.exaggeration = exaggeration;
        self
    }

    /// Fractional block height of an elevation in metres.
    pub fn y_for_elevation(&self, elevation_m: f64) -> f64 {
        f64::from(self.sea_level_y) + elevation_m * self.exaggeration / self.metres_per_block
    }

    /// The elevation in metres at a fractional block height: the inverse of [`VerticalMap::y_for_elevation`].
    pub fn elevation_for_y(&self, y: f64) -> f64 {
        (y - f64::from(self.sea_level_y)) * self.metres_per_block / self.exaggeration
    }

    /// The block a surface at `elevation_m` tops: the block height rounded down, so 0 m is block `sea_level_y`.
    pub fn block_y(&self, elevation_m: f64) -> i32 {
        self.y_for_elevation(elevation_m).floor() as i32
    }

    /// The lowest and highest block heights (inclusive) the whole Earth needs, from the deepest trench to the
    /// highest summit, so a caller can check a scale against a world's height limits.
    pub fn required_y_range(&self) -> (i32, i32) {
        (self.block_y(MIN_ELEVATION_M), self.block_y(MAX_ELEVATION_M))
    }

    /// Whether the whole Earth fits between `min_y` and `max_y`, both inclusive block heights.
    pub fn fits(&self, min_y: i32, max_y: i32) -> bool {
        let (low, high) = self.required_y_range();
        min_y <= low && high <= max_y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Great-circle distance in metres, by the haversine formula.
    fn haversine(a: LatLon, b: LatLon) -> f64 {
        let (p1, p2) = (a.lat.to_radians(), b.lat.to_radians());
        let dlat = p2 - p1;
        let dlon = (b.lon - a.lon).to_radians();
        let h = (dlat / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dlon / 2.0).sin().powi(2);
        2.0 * EARTH_RADIUS_M * h.sqrt().asin()
    }

    #[test]
    fn pixel_coordinates_round_trip_through_latitude_and_longitude() {
        for zoom in [0u8, 3, 9, 15] {
            for (lat, lon) in [
                (0.0, 0.0),
                (48.8566, 2.3522),
                (-33.9, 151.2),
                (84.0, -179.5),
                (-60.0, 179.9),
            ] {
                let (x, y) = latlon_to_pixel(lat, lon, zoom);
                let back = pixel_to_latlon(x, y, zoom);
                assert!(
                    (back.lat - lat).abs() < 1e-9,
                    "lat {lat} at zoom {zoom}: {}",
                    back.lat
                );
                assert!(
                    (back.lon - lon).abs() < 1e-9,
                    "lon {lon} at zoom {zoom}: {}",
                    back.lon
                );
            }
        }
    }

    #[test]
    fn the_world_corners_land_on_the_tile_grid() {
        let (x, y) = latlon_to_pixel(MAX_LATITUDE, -180.0, 2);
        assert!(x.abs() < 1e-9 && y.abs() < 1e-9);
        let (x, y) = latlon_to_pixel(0.0, 0.0, 2);
        assert!((x - 512.0).abs() < 1e-9 && (y - 512.0).abs() < 1e-9);
        // Beyond the limit the latitude is clamped, not extrapolated.
        let (_, y) = latlon_to_pixel(89.9, 0.0, 2);
        assert!(y.abs() < 1e-9);
    }

    #[test]
    fn a_point_is_found_in_the_right_tile_pixel() {
        // Zoom 1 has four tiles; 10° west and 10° south of the origin is in the south-west one (0, 1).
        let (id, px, py) = latlon_to_tile_pixel(-10.0, -10.0, 1);
        assert_eq!(id, TileId { z: 1, x: 0, y: 1 });
        assert!((0.0..256.0).contains(&px) && (0.0..256.0).contains(&py));
        // The antimeridian wraps: 190° east is 170° west.
        let (wrapped, ..) = latlon_to_tile_pixel(0.0, 190.0, 1);
        assert_eq!(wrapped.x, 0);
    }

    #[test]
    fn world_blocks_round_trip_through_latitude_and_longitude() {
        let p = Projection::local(47.3, 8.5, 2.5).unwrap();
        for (x, z) in [
            (0.0, 0.0),
            (1000.0, -2500.0),
            (-123_456.0, 98_765.5),
            (0.25, 0.75),
        ] {
            let ll = p.to_lat_lon(x, z);
            let (bx, bz) = p.from_lat_lon(ll.lat, ll.lon);
            assert!(
                (bx - x).abs() < 1e-6 && (bz - z).abs() < 1e-6,
                "({x}, {z}) came back as ({bx}, {bz})"
            );
        }
    }

    #[test]
    fn east_is_plus_x_and_south_is_plus_z() {
        let p = Projection::local(10.0, 20.0, 1.0).unwrap();
        let o = p.to_lat_lon(0.0, 0.0);
        assert!((o.lat - 10.0).abs() < 1e-12 && (o.lon - 20.0).abs() < 1e-12);
        assert!(p.to_lat_lon(100.0, 0.0).lon > 20.0);
        assert!(p.to_lat_lon(0.0, 100.0).lat < 10.0);
    }

    #[test]
    fn a_thousand_blocks_measure_a_thousand_scaled_metres_at_mid_latitude() {
        for scale in [1.0, 10.0] {
            let p = Projection::local(45.0, 10.0, scale).unwrap();
            let origin = p.to_lat_lon(0.0, 0.0);
            let expected = 1000.0 * scale;
            for (x, z) in [(1000.0, 0.0), (0.0, 1000.0), (0.0, -1000.0)] {
                let measured = haversine(origin, p.to_lat_lon(x, z));
                assert!(
                    (measured / expected - 1.0).abs() < 2e-3,
                    "scale {scale}: ({x}, {z}) measured {measured} m, wanted about {expected} m"
                );
            }
        }
    }

    #[test]
    fn longitude_wraps_across_the_antimeridian_and_comes_back() {
        let p = Projection::local(0.0, 179.9, 100.0).unwrap();
        // 100 blocks of 100 m is 10 km, about 0.09°: past 180° east.
        let ll = p.to_lat_lon(2000.0, 0.0);
        assert!((-180.0..-170.0).contains(&ll.lon), "lon {}", ll.lon);
        let (x, _) = p.from_lat_lon(ll.lat, ll.lon);
        assert!((x - 2000.0).abs() < 1e-6);
    }

    /// The unit vector of a point on the globe.
    fn unit_vector(p: LatLon) -> [f64; 3] {
        let (la, lo) = (p.lat.to_radians(), p.lon.to_radians());
        [la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()]
    }

    fn chord(a: LatLon, b: LatLon) -> f64 {
        let (u, v) = (unit_vector(a), unit_vector(b));
        ((u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2) + (u[2] - v[2]).powi(2)).sqrt()
    }

    #[test]
    fn folding_over_a_pole_comes_down_the_other_side() {
        let at = fold_globe;
        // Inside the globe nothing changes (the longitude only wraps).
        assert_eq!(
            at(45.0, 10.0),
            LatLon {
                lat: 45.0,
                lon: 10.0
            }
        );
        assert_eq!(at(90.0, 10.0).lat, 90.0);
        assert_eq!(at(-90.0, 10.0).lat, -90.0);
        assert_eq!(at(0.0, 190.0).lon, -170.0);
        // Past the north pole: down the opposite meridian.
        let p = at(100.0, 30.0);
        assert!(
            (p.lat - 80.0).abs() < 1e-12 && (p.lon + 150.0).abs() < 1e-12,
            "{p:?}"
        );
        // Past the south pole.
        let p = at(-100.0, 30.0);
        assert!(
            (p.lat + 80.0).abs() < 1e-12 && (p.lon + 150.0).abs() < 1e-12,
            "{p:?}"
        );
        // Half a lap over both poles is the antipode's latitude, a full lap is the same place.
        let p = at(180.0, 30.0);
        assert!(
            p.lat.abs() < 1e-12 && (p.lon + 150.0).abs() < 1e-12,
            "{p:?}"
        );
        let p = at(360.0, 30.0);
        assert!(p.lat.abs() < 1e-12 && (p.lon - 30.0).abs() < 1e-12, "{p:?}");
        let p = at(270.0, 30.0);
        assert!((p.lat + 90.0).abs() < 1e-12, "{p:?}");
        // Always on the globe, however far.
        for lat in [1e3, -1e3, 123_456.7, -9.9e6, 2.4e7] {
            let p = at(lat, 1e6 + 0.25);
            assert!(
                (-90.0..=90.0).contains(&p.lat) && (-180.0..180.0).contains(&p.lon),
                "{lat}: {p:?}"
            );
        }
        assert!(at(f64::NAN, 0.0).lat.is_nan());
    }

    #[test]
    fn the_folded_globe_is_continuous_through_both_poles_and_many_laps() {
        // Walk the latitude in small steps over the poles and several laps: consecutive points stay close on the
        // sphere (no jump at a pole, at the fold, or at the antimeridian).
        let step = 0.05;
        for lon in [0.0, 30.0, 179.99, -179.99] {
            let mut prev = fold_globe(-1000.0, lon);
            let mut lat = -1000.0 + step;
            let mut worst = 0.0f64;
            while lat < 1000.0 {
                let now = fold_globe(lat, lon);
                worst = worst.max(chord(prev, now));
                prev = now;
                lat += step;
            }
            assert!(
                worst < step.to_radians() * 1.01,
                "lon {lon}: jump of chord {worst}"
            );
        }
        // The same through the projection: walking north (negative z) over the pole and on round the globe.
        let p = Projection::local(60.0, 20.0, 1000.0).unwrap();
        let mut prev = p.to_lat_lon(0.0, 0.0);
        let mut worst = 0.0f64;
        let mut past_pole = false;
        for i in 1..=200_000 {
            let now = p.to_lat_lon(5.0, -f64::from(i) * 2.0);
            assert!((-90.0..=90.0).contains(&now.lat) && (-180.0..180.0).contains(&now.lon));
            past_pole |= p.to_lat_lon_unwrapped(5.0, -f64::from(i) * 2.0).lat > 90.0;
            worst = worst.max(chord(prev, now));
            prev = now;
        }
        assert!(past_pole);
        // 2 blocks of 1 km on a sphere of 6378 km is 0.0003 radians, and the pole only swings the longitude.
        assert!(worst < 0.001, "worst chord {worst}");
    }

    #[test]
    fn the_unwrapped_mapping_is_the_raw_one_and_the_wrapped_one_is_folded() {
        let p = Projection::local(60.0, 20.0, 1000.0).unwrap();
        // 4,000 km north of 60 degrees is past the pole.
        let raw = p.to_lat_lon_unwrapped(0.0, -4_000.0);
        assert!(raw.lat > 90.0 && (raw.lon - 20.0).abs() < 1e-9, "{raw:?}");
        let folded = p.to_lat_lon(0.0, -4_000.0);
        assert_eq!(folded, fold_globe(raw.lat, raw.lon));
        assert!(
            folded.lat <= 90.0 && (folded.lon + 160.0).abs() < 1e-9,
            "{folded:?}"
        );
        // Inside the globe the two agree.
        let near = p.to_lat_lon(100.0, 100.0);
        assert_eq!(near.lat, p.to_lat_lon_unwrapped(100.0, 100.0).lat);
    }

    #[test]
    fn a_projection_rejects_a_nonsense_scale_or_origin() {
        assert!(Projection::local(0.0, 0.0, 0.0).is_err());
        assert!(Projection::local(0.0, 0.0, f64::NAN).is_err());
        assert!(Projection::local(89.0, 0.0, 1.0).is_err());
        assert!(Projection::local(0.0, f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn sea_level_and_elevation_map_to_blocks_and_back() {
        let v = VerticalMap::new(63, 2.0);
        assert_eq!(v.block_y(0.0), 63);
        assert_eq!(v.block_y(100.0), 113);
        assert_eq!(v.block_y(-1.0), 62);
        assert!((v.elevation_for_y(v.y_for_elevation(1234.5)) - 1234.5).abs() < 1e-9);
        let tall = v.with_exaggeration(2.0);
        assert_eq!(tall.block_y(100.0), 163);
        assert!((tall.elevation_for_y(tall.y_for_elevation(-77.0)) + 77.0).abs() < 1e-9);
    }

    #[test]
    fn the_whole_earth_needs_a_height_range_that_scales_with_blocks() {
        // At 1 m per block: 8849 m above and 11000 m below sea level 0.
        let one = VerticalMap::new(0, 1.0);
        assert_eq!(one.required_y_range(), (-11_000, 8_849));
        assert!(one.fits(-11_000, 8_849));
        assert!(!one.fits(-64, 320));
        // At 100 m per block it needs 110 blocks down and 88 up, so a 384-block world holds it.
        let coarse = VerticalMap::new(120, 100.0);
        assert_eq!(coarse.required_y_range(), (10, 208));
        assert!(coarse.fits(-64, 320));
    }
}

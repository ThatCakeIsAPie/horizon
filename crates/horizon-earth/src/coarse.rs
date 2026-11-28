//! The coarse global data set: which tiles a whole-world view needs, which of the world's below-sea-level pixels
//! are sea, and the land class of a column known only from its elevation and climate.
//!
//! A far node whose voxels are kilometres wide never needs fine data. It reads the whole-world terrarium tiles of
//! zooms `0..=`[`COARSE_PREFETCH_ZOOM`] (341 small tiles, [`coarse_tiles`]), a [`SeaMask`] built once from the zoom
//! [`SEA_MASK_ZOOM`] ones, and classifies each column from data alone ([`coarse_class`]).

use std::sync::Arc;

use crate::landcover::LandClass;
use crate::projection::{LatLon, latlon_to_pixel, pixel_to_latlon, wrap_longitude};
use crate::tile::{TILE_SIZE, Tile, TileId};

/// The finest zoom of the coarse set: the whole world at zooms `0..=COARSE_PREFETCH_ZOOM` (341 tiles) is what a
/// coarse far node reads.
pub const COARSE_PREFETCH_ZOOM: u8 = 4;
/// The zoom whose pixel grid the [`SeaMask`] is built on (4096 by 4096 pixels at zoom 4, about 10 km at the equator).
pub const SEA_MASK_ZOOM: u8 = 4;
/// A below-sea-level region is sea when its deepest pixel is deeper than this many metres: oceans and deep
/// landlocked seas qualify, dry basins and polder pockets (a few tens of metres down) do not.
pub const SEA_BASIN_MIN_DEPTH_M: f64 = 150.0;
/// The finest zoom [`SeaMask::build`] accepts (a larger one is lowered to it): the pixel grid then still fits
/// 32-bit indices, and a mask of it is already 128 MiB.
pub const SEA_MASK_MAX_ZOOM: u8 = 7;

/// A column colder than this (mean annual temperature, degrees Celsius) is an ice sheet.
pub const ICE_SHEET_BELOW_C: f64 = -17.0;
/// A column colder than this, and not an ice sheet, is tundra: moss and lichen.
pub const TUNDRA_BELOW_C: f64 = -9.0;
/// Koppen desert threshold: a column is bare when precipitation (mm a year) is below
/// `DESERT_MM_PER_C * max(t, 0) + DESERT_BASE_MM`.
pub const DESERT_MM_PER_C: f64 = 10.0;
/// The constant term of the desert threshold in millimetres a year, see [`DESERT_MM_PER_C`].
pub const DESERT_BASE_MM: f64 = 140.0;
/// Koppen steppe threshold: below `STEPPE_MM_PER_C * max(t, 0) + STEPPE_BASE_MM` (and above the desert one) the
/// column is steppe.
pub const STEPPE_MM_PER_C: f64 = 20.0;
/// The constant term of the steppe threshold in millimetres a year, see [`STEPPE_MM_PER_C`].
pub const STEPPE_BASE_MM: f64 = 280.0;
/// Steppe at least this warm (degrees Celsius) is shrubland; cooler steppe is grass.
pub const STEPPE_SHRUB_FROM_C: f64 = 12.0;
/// Humid columns at least this warm (degrees Celsius) are tropical savanna when they are not wet enough for forest.
pub const SAVANNA_FROM_C: f64 = 22.0;
/// A warm humid column with less rain than this (millimetres a year) is savanna grass, with more it is forest.
pub const SAVANNA_BELOW_MM: f64 = 1200.0;
/// Land above this elevation (metres) is bare rock, unless it is ice or water.
pub const SNOWLINE_M: f64 = 5500.0;

/// Every tile of zooms `0..=`[`COARSE_PREFETCH_ZOOM`], zoom ascending, and within a zoom the nearest to `centre`
/// first (great-circle distance to the tile's centre; ties by tile id). 341 tiles.
pub fn coarse_tiles(centre: LatLon) -> Vec<TileId> {
    let mut out = Vec::with_capacity(341);
    for z in 0..=COARSE_PREFETCH_ZOOM {
        let side = 1u32 << z;
        let mut ring: Vec<(f64, TileId)> = (0..side)
            .flat_map(|y| (0..side).map(move |x| TileId { z, x, y }))
            .map(|id| {
                let tile = f64::from(TILE_SIZE);
                let at = pixel_to_latlon(
                    (f64::from(id.x) + 0.5) * tile,
                    (f64::from(id.y) + 0.5) * tile,
                    z,
                );
                (great_circle_rad(centre, at), id)
            })
            .collect();
        ring.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        out.extend(ring.into_iter().map(|(_, id)| id));
    }
    out
}

/// The great-circle angle in radians between two points (haversine); a point that is not a number is infinitely far.
fn great_circle_rad(a: LatLon, b: LatLon) -> f64 {
    let (p1, p2) = (a.lat.to_radians(), b.lat.to_radians());
    let dlon = (b.lon - a.lon).to_radians();
    let h = ((p2 - p1) / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dlon / 2.0).sin().powi(2);
    if h.is_nan() {
        return f64::INFINITY;
    }
    2.0 * h.clamp(0.0, 1.0).sqrt().asin()
}

/// Which pixels of the world grid at one zoom are sea: below-sea-level pixels whose connected region (4-neighbours,
/// wrapping round the antimeridian, not over the poles) reaches deeper than `-`[`SEA_BASIN_MIN_DEPTH_M`], grown by
/// one pixel. One bit per pixel (2 MiB at zoom 4).
#[derive(Clone, Debug)]
pub struct SeaMask {
    zoom: u8,
    /// Pixels along one side of the world grid.
    side: usize,
    /// `side * side` bits, row by row from the north, one `u64` word per 64 pixels (a row is whole words).
    bits: Vec<u64>,
    sea_pixels: usize,
}

impl SeaMask {
    /// Builds the mask of the world grid at `zoom` (lowered to [`SEA_MASK_MAX_ZOOM`]) from the tiles `tile` hands
    /// out (it is asked once for every tile of the zoom, and never fetches anything here). `Err` lists every tile
    /// it lacks, and then no mask is built.
    ///
    /// The rule: pixels below 0 m form 4-connected regions (the grid wraps in x, not in y). A region is sea when
    /// its deepest pixel is below `-`[`SEA_BASIN_MIN_DEPTH_M`]: the oceans, and enclosed deep basins (a deep
    /// landlocked sea). A shallow enclosed depression is land. The sea is then grown by one pixel in all eight
    /// directions, so a coastline pixel whose bilinear neighbours are sea counts as sea too.
    pub fn build(
        zoom: u8,
        tile: &mut dyn FnMut(TileId) -> Option<Arc<Tile>>,
    ) -> Result<SeaMask, Vec<TileId>> {
        let zoom = zoom.min(SEA_MASK_MAX_ZOOM);
        let tiles_per_side = 1u32 << zoom;
        let mut held = Vec::with_capacity((tiles_per_side * tiles_per_side) as usize);
        let mut missing = Vec::new();
        for y in 0..tiles_per_side {
            for x in 0..tiles_per_side {
                let id = TileId { z: zoom, x, y };
                match tile(id) {
                    Some(t) => held.push(t),
                    None => missing.push(id),
                }
            }
        }
        if !missing.is_empty() {
            return Err(missing);
        }

        let tile_px = TILE_SIZE as usize;
        let side = tile_px << zoom;
        let words_per_row = side / 64;
        let mut below = vec![0u64; words_per_row * side];
        let mut deep = vec![0u64; words_per_row * side];
        let deep_limit = -(SEA_BASIN_MIN_DEPTH_M as f32);
        for ty in 0..tiles_per_side as usize {
            for tx in 0..tiles_per_side as usize {
                let heights = held[ty * tiles_per_side as usize + tx].heights();
                for py in 0..tile_px {
                    let row = &heights[py * tile_px..(py + 1) * tile_px];
                    let gy = ty * tile_px + py;
                    let gx0 = tx * tile_px;
                    // A tile row is 256 pixels: four whole words.
                    let base = gy * words_per_row + gx0 / 64;
                    for (w, chunk) in row.chunks_exact(64).enumerate() {
                        let (mut b, mut d) = (0u64, 0u64);
                        for (bit, &h) in chunk.iter().enumerate() {
                            b |= u64::from(h < 0.0) << bit;
                            d |= u64::from(h < deep_limit) << bit;
                        }
                        below[base + w] = b;
                        deep[base + w] = d;
                    }
                }
            }
        }
        drop(held);

        // Flood every region from its deep pixels through the below-sea-level ones.
        let mut sea = vec![0u64; words_per_row * side];
        let get = |bits: &[u64], i: usize| bits[i >> 6] >> (i & 63) & 1 != 0;
        let mut stack: Vec<u32> = Vec::new();
        for (w, &word) in deep.iter().enumerate() {
            let mut rest = word;
            while rest != 0 {
                let i = w * 64 + rest.trailing_zeros() as usize;
                rest &= rest - 1;
                if get(&sea, i) {
                    continue;
                }
                sea[i >> 6] |= 1u64 << (i & 63);
                stack.push(i as u32);
                while let Some(at) = stack.pop() {
                    let at = at as usize;
                    let (y, x) = (at / side, at % side);
                    let west = if x == 0 { at + side - 1 } else { at - 1 };
                    let east = if x + 1 == side { at + 1 - side } else { at + 1 };
                    let mut visit = |n: usize| {
                        if get(&below, n) && !get(&sea, n) {
                            sea[n >> 6] |= 1u64 << (n & 63);
                            stack.push(n as u32);
                        }
                    };
                    visit(west);
                    visit(east);
                    if y > 0 {
                        visit(at - side);
                    }
                    if y + 1 < side {
                        visit(at + side);
                    }
                }
            }
        }
        drop((below, deep));

        // Grow by one pixel: rows first (north, self, south), then a one-bit shift each way with wrap.
        let mut grown = vec![0u64; words_per_row * side];
        let mut row = vec![0u64; words_per_row];
        for y in 0..side {
            for (w, slot) in row.iter_mut().enumerate() {
                let mut v = sea[y * words_per_row + w];
                if y > 0 {
                    v |= sea[(y - 1) * words_per_row + w];
                }
                if y + 1 < side {
                    v |= sea[(y + 1) * words_per_row + w];
                }
                *slot = v;
            }
            for w in 0..words_per_row {
                let before = row[(w + words_per_row - 1) % words_per_row];
                let after = row[(w + 1) % words_per_row];
                let v = row[w];
                grown[y * words_per_row + w] = v | (v << 1 | before >> 63) | (v >> 1 | after << 63);
            }
        }
        let sea_pixels = grown.iter().map(|w| w.count_ones() as usize).sum();
        Ok(SeaMask {
            zoom,
            side,
            bits: grown,
            sea_pixels,
        })
    }

    /// The zoom of the pixel grid the mask is on.
    pub fn zoom(&self) -> u8 {
        self.zoom
    }

    /// The fraction of the world grid's pixels that are sea (after the one-pixel growth), `0.0..=1.0`.
    pub fn sea_share(&self) -> f64 {
        self.sea_pixels as f64 / (self.side as f64 * self.side as f64)
    }

    /// Whether the pixel holding this point is sea. The latitude is clamped to the Mercator limit and the longitude
    /// wrapped; a point that is not a number is not sea.
    pub fn is_sea(&self, lat: f64, lon: f64) -> bool {
        if !(lat.is_finite() && lon.is_finite()) {
            return false;
        }
        let (px, py) = latlon_to_pixel(lat, wrap_longitude(lon), self.zoom);
        let last = self.side - 1;
        let x = (px.floor().max(0.0) as usize).min(last);
        let y = (py.floor().max(0.0) as usize).min(last);
        let i = y * self.side + x;
        self.bits[i >> 6] >> (i & 63) & 1 != 0
    }
}

/// The land class of a coarse column from data alone (no land cover): `water` first, then ice sheets and tundra by
/// temperature, then Koppen's desert and steppe thresholds and the savanna split by precipitation (without
/// precipitation: grass), and bare rock above [`SNOWLINE_M`] unless the column is ice or water.
pub fn coarse_class(
    elevation_m: f64,
    temperature_c: f64,
    precipitation_mm: Option<f64>,
    water: bool,
) -> LandClass {
    if water {
        return LandClass::Water;
    }
    let class = if temperature_c < ICE_SHEET_BELOW_C {
        LandClass::SnowIce
    } else if temperature_c < TUNDRA_BELOW_C {
        LandClass::MossLichen
    } else if let Some(p) = precipitation_mm.filter(|p| p.is_finite() && temperature_c.is_finite())
    {
        let warm = temperature_c.max(0.0);
        if p < DESERT_MM_PER_C * warm + DESERT_BASE_MM {
            LandClass::Bare
        } else if p < STEPPE_MM_PER_C * warm + STEPPE_BASE_MM {
            if temperature_c >= STEPPE_SHRUB_FROM_C {
                LandClass::Shrub
            } else {
                LandClass::Grass
            }
        } else if temperature_c >= SAVANNA_FROM_C && p < SAVANNA_BELOW_MM {
            LandClass::Grass
        } else {
            LandClass::Tree
        }
    } else {
        LandClass::Grass
    };
    if elevation_m > SNOWLINE_M && class != LandClass::SnowIce {
        LandClass::Bare
    } else {
        class
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::encode_tile;
    use std::collections::HashMap;
    use std::time::Instant;

    fn tiles_of(zoom: u8, height: &crate::testing::HeightFn) -> HashMap<TileId, Arc<Tile>> {
        let mut out = HashMap::new();
        for y in 0..1u32 << zoom {
            for x in 0..1u32 << zoom {
                let id = TileId { z: zoom, x, y };
                out.insert(
                    id,
                    Arc::new(Tile::decode(&encode_tile(id, height), zoom).unwrap()),
                );
            }
        }
        out
    }

    fn mask_of(zoom: u8, height: &crate::testing::HeightFn) -> SeaMask {
        let tiles = tiles_of(zoom, height);
        SeaMask::build(zoom, &mut |id| tiles.get(&id).cloned()).unwrap()
    }

    fn inside(v: f64, a: f64, b: f64) -> bool {
        (a..=b).contains(&v)
    }

    /// Land at 300 m with: an ocean, a shallow shelf touching it, a deep landlocked basin, a shallow dry basin, a
    /// sea across the antimeridian whose deep part is only on its west side, and a deep polar trench not joined to
    /// a shallow one at the other end of the map.
    fn world(lat: f64, lon: f64) -> f64 {
        let ocean = inside(lon, -100.0, -40.0) && inside(lat, -30.0, 30.0);
        let shelf = inside(lon, -40.0, -30.0) && inside(lat, -10.0, 10.0);
        let deep_basin = inside(lon, 20.0, 40.0) && inside(lat, 40.0, 50.0);
        let dry_basin = inside(lon, 60.0, 80.0) && inside(lat, 40.0, 50.0);
        let seam_deep = inside(lon, -180.0, -176.0) && inside(lat, 55.0, 65.0);
        let seam_shallow = inside(lon, 176.0, 180.0) && inside(lat, 55.0, 65.0);
        let north_deep = lat > 80.0;
        let south_shallow = lat < -80.0;
        if ocean {
            -3000.0
        } else if deep_basin || seam_deep {
            -400.0
        } else if shelf || dry_basin || seam_shallow || south_shallow {
            -50.0
        } else if north_deep {
            -400.0
        } else {
            300.0
        }
    }

    #[test]
    fn the_coarse_set_has_341_tiles_in_zoom_then_distance_order() {
        let centre = LatLon {
            lat: 47.0,
            lon: 8.0,
        };
        let tiles = coarse_tiles(centre);
        assert_eq!(tiles.len(), 341);
        assert_eq!(tiles.iter().filter(|t| t.z == 4).count(), 256);
        assert!(tiles.windows(2).all(|w| w[0].z <= w[1].z));
        let mut unique = tiles.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 341);
        assert!(
            unique
                .iter()
                .all(|t| t.validate().is_ok() && t.z <= COARSE_PREFETCH_ZOOM)
        );
        assert_eq!(tiles[0], TileId { z: 0, x: 0, y: 0 });
        // Zoom 4: the tile holding the centre (8°E is column 8, 47°N row 5) is first.
        let first = tiles.iter().find(|t| t.z == 4).unwrap();
        assert_eq!(*first, TileId { z: 4, x: 8, y: 5 });
        // Within the zoom the distance never decreases.
        let dist = |id: &TileId| {
            let at = pixel_to_latlon(
                (f64::from(id.x) + 0.5) * 256.0,
                (f64::from(id.y) + 0.5) * 256.0,
                id.z,
            );
            great_circle_rad(centre, at)
        };
        let z4: Vec<_> = tiles.iter().filter(|t| t.z == 4).collect();
        assert!(z4.windows(2).all(|w| dist(w[0]) <= dist(w[1])));
        // The far side of the world comes last: tile 4/0/15 is near the south-west, 4/8/10 is nearer.
        let pos = |id: TileId| tiles.iter().position(|t| *t == id).unwrap();
        assert!(pos(TileId { z: 4, x: 8, y: 6 }) < pos(TileId { z: 4, x: 0, y: 15 }));
        // A centre past the Mercator limit or the antimeridian still gives every tile.
        for c in [
            LatLon {
                lat: 89.0,
                lon: 190.0,
            },
            LatLon {
                lat: -120.0,
                lon: -540.0,
            },
        ] {
            assert_eq!(coarse_tiles(c).len(), 341);
        }
    }

    #[test]
    fn the_sea_mask_marks_oceans_and_deep_basins_and_not_shallow_depressions() {
        let mask = mask_of(2, &world);
        assert_eq!(mask.zoom(), 2);
        assert!(mask.is_sea(0.0, -70.0), "ocean");
        assert!(mask.is_sea(-20.0, -90.0), "ocean corner area");
        assert!(
            mask.is_sea(0.0, -35.0),
            "a shallow shelf touching the ocean is sea"
        );
        assert!(mask.is_sea(45.0, 30.0), "a deep landlocked basin is sea");
        assert!(
            !mask.is_sea(45.0, 70.0),
            "a shallow enclosed depression is land"
        );
        assert!(!mask.is_sea(45.0, -10.0), "plain land");
        assert!(!mask.is_sea(-60.0, 100.0), "plain land");
        // The dilation reaches one pixel (0.35 degrees here) past the coast, not further.
        assert!(mask.is_sea(0.0, -39.8));
        assert!(!mask.is_sea(0.0, -20.0));
        let share = mask.sea_share();
        assert!(share > 0.02 && share < 0.4, "sea share {share}");
    }

    #[test]
    fn a_region_across_the_antimeridian_is_one_region_and_the_grid_does_not_wrap_over_the_poles() {
        let mask = mask_of(2, &world);
        // The deep part is west of 180 (negative longitudes), the shallow part east: joined through the wrap.
        assert!(mask.is_sea(60.0, -178.0), "deep side");
        assert!(mask.is_sea(60.0, 178.0), "shallow side joined by the wrap");
        // Longitude wraps in the query too.
        assert_eq!(mask.is_sea(60.0, 178.0), mask.is_sea(60.0, 178.0 - 360.0));
        assert_eq!(mask.is_sea(60.0, 182.0), mask.is_sea(60.0, -178.0));
        assert_eq!(mask.is_sea(0.0, -70.0 + 720.0), mask.is_sea(0.0, -70.0));
        // The deep far north is sea, the shallow far south is not joined to it: y does not wrap.
        assert!(mask.is_sea(84.0, 100.0));
        assert!(!mask.is_sea(-84.0, 100.0));
        // The latitude is clamped, not an error.
        assert!(mask.is_sea(89.9, 0.0));
        assert!(!mask.is_sea(-89.9, 0.0));
        assert!(!mask.is_sea(f64::NAN, 0.0));
    }

    #[test]
    fn a_continent_that_straddles_the_antimeridian_stays_land_in_a_world_of_sea() {
        // Sea everywhere (-3000 m) except a land block either side of 180 degrees: both halves are land.
        let height = |lat: f64, lon: f64| {
            if inside(lat, -30.0, 30.0) && (lon >= 165.0 || lon <= -165.0) {
                200.0
            } else {
                -3000.0
            }
        };
        let mask = mask_of(2, &height);
        assert!(!mask.is_sea(0.0, 179.0));
        assert!(!mask.is_sea(0.0, -179.0));
        assert!(!mask.is_sea(0.0, 180.0));
        assert!(mask.is_sea(0.0, 100.0));
        assert!(mask.is_sea(45.0, 179.0));
    }

    #[test]
    fn building_without_every_tile_lists_what_is_missing() {
        let mut tiles = tiles_of(2, &world);
        let gone = [TileId { z: 2, x: 0, y: 0 }, TileId { z: 2, x: 3, y: 3 }];
        for id in gone {
            tiles.remove(&id);
        }
        let err = SeaMask::build(2, &mut |id| tiles.get(&id).cloned()).unwrap_err();
        assert_eq!(err, gone.to_vec());
        let none = SeaMask::build(0, &mut |_| None).unwrap_err();
        assert_eq!(none, vec![TileId { z: 0, x: 0, y: 0 }]);
    }

    #[test]
    fn the_sea_mask_of_zoom_4_builds_fast() {
        let height = |lat: f64, lon: f64| {
            // Continents and basins from smooth waves, a few deep pits.
            let a = (lat * 0.11).sin() * (lon * 0.07).cos() * 2500.0;
            let b = (lat * 0.9).sin() * (lon * 1.3).sin() * 150.0;
            a + b - 200.0
        };
        let tiles = tiles_of(SEA_MASK_ZOOM, &height);
        let start = Instant::now();
        let mask = SeaMask::build(SEA_MASK_ZOOM, &mut |id| tiles.get(&id).cloned()).unwrap();
        let took = start.elapsed();
        eprintln!(
            "sea mask zoom {}: {} tiles, built in {:.1} ms, sea share {:.3}, {} KiB",
            mask.zoom(),
            tiles.len(),
            took.as_secs_f64() * 1000.0,
            mask.sea_share(),
            mask.bits.len() * 8 / 1024
        );
        assert!(mask.sea_share() > 0.1 && mask.sea_share() < 0.9);
        let limit = if cfg!(debug_assertions) { 5.0 } else { 1.0 };
        assert!(took.as_secs_f64() < limit, "built in {took:?}");
    }

    #[test]
    fn coarse_classes_follow_the_rules() {
        use LandClass::*;
        // Water wins over everything.
        assert_eq!(coarse_class(6000.0, -30.0, Some(10.0), true), Water);
        // Ice sheets, tundra.
        assert_eq!(coarse_class(2500.0, -20.0, Some(300.0), false), SnowIce);
        assert_eq!(coarse_class(100.0, -17.0, Some(300.0), false), MossLichen);
        assert_eq!(coarse_class(100.0, -10.0, None, false), MossLichen);
        // Desert: p < 10 t + 140.
        assert_eq!(coarse_class(300.0, 25.0, Some(100.0), false), Bare);
        assert_eq!(coarse_class(300.0, 20.0, Some(339.0), false), Bare);
        assert_eq!(coarse_class(300.0, 20.0, Some(341.0), false), Shrub);
        // Steppe: grass when cool, shrub when warm (12 and above).
        assert_eq!(coarse_class(300.0, 8.0, Some(300.0), false), Grass);
        assert_eq!(coarse_class(300.0, 12.0, Some(400.0), false), Shrub);
        assert_eq!(coarse_class(300.0, 11.9, Some(400.0), false), Grass);
        // Below freezing uses t' = 0: the thresholds are 140 and 280.
        assert_eq!(coarse_class(300.0, -5.0, Some(139.0), false), Bare);
        assert_eq!(coarse_class(300.0, -5.0, Some(279.0), false), Grass);
        assert_eq!(coarse_class(300.0, -5.0, Some(281.0), false), Tree);
        // Humid: forest, except warm and not wet enough for it (savanna).
        assert_eq!(coarse_class(300.0, 10.0, Some(800.0), false), Tree);
        assert_eq!(coarse_class(300.0, 22.0, Some(1199.0), false), Grass);
        assert_eq!(coarse_class(300.0, 22.0, Some(1200.0), false), Tree);
        assert_eq!(coarse_class(300.0, 27.0, Some(2500.0), false), Tree);
        assert_eq!(coarse_class(300.0, 21.9, Some(900.0), false), Tree);
        // No precipitation: grass.
        assert_eq!(coarse_class(300.0, 15.0, None, false), Grass);
        // Above the snowline: bare rock, but ice stays ice.
        assert_eq!(coarse_class(5501.0, 5.0, Some(900.0), false), Bare);
        assert_eq!(coarse_class(5500.0, 5.0, Some(900.0), false), Tree);
        assert_eq!(coarse_class(6000.0, -10.0, Some(900.0), false), Bare);
        assert_eq!(coarse_class(6000.0, -25.0, Some(900.0), false), SnowIce);
        assert_eq!(coarse_class(6000.0, 5.0, None, true), Water);
    }
}

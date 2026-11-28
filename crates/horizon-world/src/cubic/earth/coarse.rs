//! The far view's coarse levels: columns of far nodes whose voxels are kilometres wide, read from a small coarse
//! global set instead of the fine data.
//!
//! A far node thousands of kilometres away has voxels far bigger than any pixel of the fine data (WorldCover's
//! coarsest overview is about 600 m, the biomes read the map at 500 m), and reading that data for it would touch a
//! land-cover file per voxel. So from [`coarse_from_m`] of ground per voxel on, [`EarthGenerator::columns_wide`]
//! hands the read to [`EarthGenerator::columns_coarse`], which reads:
//!
//! * the elevation at the zoom of the voxel's size (whole-world zooms 0 to 4 are the coarse set, the next few zooms
//!   are small rings the prefetcher keeps);
//! * the sea from a [`SeaMask`] built once from the zoom-[`SEA_MASK_ZOOM`] tiles: below the sea level and part of
//!   an ocean or deep basin;
//! * big lakes of the water atlas, flat at their recorded level;
//! * the climate at the overview of the voxel's size;
//! * and a land class made from those ([`horizon_earth::coarse_class`]), from which the kind of top and the biome
//!   follow by the fine path's own rules.
//!
//! Nothing here waits: a node whose data is not local is "later" ([`NotReady`]) and what it lacks is asked for, as
//! for every far read.

use std::sync::{Arc, Mutex, OnceLock};

use horizon_earth::{
    ClimateFetch, ClimateLookup, GeoBox, Lake, LandClass, SEA_MASK_ZOOM, SeaMask, TileId,
    coarse_class, coarse_tiles,
};

use super::biome::{Biome, fallback_temperature_c, variant_cell};
use super::dry::DryColumn;
use super::far::NodeColumns;
use super::generator::{EarthGenerator, NotReady, Rect};
use super::surface::{Slope, SurfaceKind};
use super::water::Bed;

/// Voxels at least this many metres of ground wide take the coarse path (level 12 at one metre a block: 7 times
/// WorldCover's coarsest overview). `HORIZON_EARTH_COARSE_M` overrides it; 0 turns the coarse path off.
pub const COARSE_FROM_M: f64 = 4096.0;

/// A lake of the water atlas is drawn at the coarse levels when its area is at least this share of a voxel's.
pub const COARSE_LAKE_SHARE: f64 = 0.25;
/// ... and at least this many square kilometres.
pub const COARSE_LAKE_MIN_KM2: f64 = 50.0;
/// Nodes whose box spans more degrees than this (either way) look up no lakes: the query would decode most of
/// the atlas, and lakes are a voxel or two at that size.
pub const COARSE_LAKE_QUERY_MAX_DEG: f64 = 12.0;
/// A column inside a lake's box is the lake's water when its elevation is at most this far over the lake level.
pub const COARSE_LAKE_OVER_M: f64 = 1.0;
/// The ground per sample (metres) the whole-world climate is fetched at when a world opens: the coarsest overview.
pub const COARSE_CLIMATE_PREFETCH_M: f64 = 1.0e6;

/// The coarse threshold in use, in metres of ground per voxel (0 = off).
pub fn coarse_from_m() -> f64 {
    static FROM: OnceLock<f64> = OnceLock::new();
    *FROM.get_or_init(|| {
        std::env::var("HORIZON_EARTH_COARSE_M")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(COARSE_FROM_M)
    })
}

/// What a generator keeps for its coarse levels: the sea mask, once built.
#[derive(Default)]
pub(super) struct CoarseState {
    mask: OnceLock<Arc<SeaMask>>,
    building: Mutex<()>,
    /// The threshold, set by tests; `None` is [`coarse_from_m`].
    from_m: OnceLock<f64>,
}

impl EarthGenerator {
    /// The coarse threshold of this generator, in metres per voxel (0 = off).
    fn coarse_threshold_m(&self) -> f64 {
        self.coarse
            .from_m
            .get()
            .copied()
            .unwrap_or_else(coarse_from_m)
    }

    /// Makes the coarse path start at `metres` of ground per voxel (0 = off), whatever the env says: for tests
    /// and tools. Only the first call counts, and only before the generator was asked anything.
    #[doc(hidden)]
    pub fn set_coarse_from_m(&self, metres: f64) {
        let _ = self.coarse.from_m.set(metres.max(0.0));
    }

    /// Whether columns `step` blocks apart take the coarse path: land cover is on and the step is at least the
    /// coarse threshold of ground. Without land cover the elevation-only far view is cheap at every level and
    /// stays as it is.
    pub fn is_coarse_step(&self, step: i32) -> bool {
        let from = self.coarse_threshold_m();
        from > 0.0
            && self.land_cover_active()
            && f64::from(step.max(1)) * self.field.projection().metres_per_block() >= from
    }

    /// Builds the sea mask once from the zoom-[`SEA_MASK_ZOOM`] tiles already in the store. Never fetches:
    /// `Err` lists the tiles still missing (empty while another thread is building it). Cheap once built.
    pub fn prepare_coarse(&self) -> Result<(), Vec<TileId>> {
        if self.coarse.mask.get().is_some() {
            return Ok(());
        }
        let Ok(_building) = self.coarse.building.try_lock() else {
            return Err(Vec::new());
        };
        if self.coarse.mask.get().is_some() {
            return Ok(());
        }
        let store = self.field.store().clone();
        let started = std::time::Instant::now();
        let mask = SeaMask::build(SEA_MASK_ZOOM, &mut |id| store.cached(id))?;
        log::info!(
            "Horizon: Earth coarse sea mask built in {} ms ({:.0}% sea)",
            started.elapsed().as_millis(),
            mask.sea_share() * 100.0
        );
        let _ = self.coarse.mask.set(Arc::new(mask));
        Ok(())
    }

    /// Whether the sea mask is built.
    pub fn coarse_ready(&self) -> bool {
        self.coarse.mask.get().is_some()
    }

    /// The coarse set's tiles ([`coarse_tiles`]) the store does not hold, nearest the block `centre_block` first.
    pub fn coarse_tiles_missing(&self, centre_block: (i32, i32)) -> Vec<TileId> {
        let at = self.field.projection().to_lat_lon(
            f64::from(centre_block.0) + 0.5,
            f64::from(centre_block.1) + 0.5,
        );
        let store = self.field.store();
        coarse_tiles(at)
            .into_iter()
            .filter(|&id| store.cached(id).is_none())
            .collect()
    }

    /// What the climate store still lacks for whole-world reads at its coarsest overview (empty without climate):
    /// fetched when a world opens. Finer overviews are asked for by the nodes that read them.
    pub fn coarse_climate_needs(&self) -> Vec<ClimateFetch> {
        if !self.climate_wanted() {
            return Vec::new();
        }
        let Some(source) = self.climate_source() else {
            return Vec::new();
        };
        source.field().store().needs(
            GeoBox {
                south: -90.0,
                north: 90.0,
                west: -180.0,
                east: 180.0,
            },
            COARSE_CLIMATE_PREFETCH_M,
        )
    }

    /// The columns of `width` by `width` voxels `stride` blocks apart from the block `(x0, z0)` on the coarse path
    /// (see the module docs): sampled at the voxels' middles as [`EarthGenerator::columns_at`] samples them.
    pub(super) fn columns_coarse(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        stride: i32,
    ) -> Result<NodeColumns, NotReady> {
        let stride = stride.max(1);
        let (sx, sz) = (x0.saturating_add(stride / 2), z0.saturating_add(stride / 2));
        let zoom = self.zoom_for_step(stride);
        let span = (width as i32 - 1).saturating_mul(stride);
        let rect = Rect {
            x0: sx,
            z0: sz,
            x1: sx.saturating_add(span),
            z1: sz.saturating_add(span),
            zoom,
        };
        let surfaces0 = self
            .field
            .surface_ys_strided(sx, sz, width, width, stride, zoom)
            .ok_or_else(|| NotReady {
                tiles: self.missing_in(rect),
            })?;
        if self.prepare_coarse().is_err() {
            let missing = self.coarse_tiles_missing((sx, sz));
            self.request_tiles(&missing);
            return Err(NotReady { tiles: missing });
        }
        let mask = self.coarse.mask.get().expect("built above").clone();
        let projection = *self.field.projection();
        let vertical = *self.field.vertical();
        let metres = f64::from(stride) * projection.metres_per_block();
        let n = width * width;
        let at = |i: usize| -> (i32, i32) {
            (
                sx.saturating_add((i % width) as i32 * stride),
                sz.saturating_add((i / width) as i32 * stride),
            )
        };
        let geo: Vec<_> = (0..n)
            .map(|i| {
                let (x, z) = at(i);
                projection.to_lat_lon(f64::from(x) + 0.5, f64::from(z) + 0.5)
            })
            .collect();
        // The climate of every column at the overview of the voxel's size; all are asked for before a gap is
        // reported, so one pass names everything the node lacks.
        let mut climates = Vec::with_capacity(n);
        let mut lacking = false;
        if let (true, Some(source)) = (self.climate_wanted(), self.climate_source()) {
            let store = source.field().store();
            for p in &geo {
                match store.lookup(p.lat, p.lon, metres) {
                    ClimateLookup::Climate(c) => climates.push(Some(c)),
                    ClimateLookup::NoData => climates.push(None),
                    ClimateLookup::Need(fetch) => {
                        self.note_climate(fetch);
                        lacking = true;
                        climates.push(None);
                    }
                }
            }
        } else {
            climates.resize(n, None);
        }
        if lacking {
            return Err(NotReady { tiles: Vec::new() });
        }
        let lakes = self.coarse_lakes(&geo, metres);
        let elevation: Vec<f64> = surfaces0
            .iter()
            .map(|&s| vertical.elevation_for_y(f64::from(s)))
            .collect();
        let mut surfaces = Vec::with_capacity(n);
        let mut kinds = Vec::with_capacity(n);
        let mut classes = Vec::with_capacity(n);
        let mut biomes = Vec::with_capacity(n);
        for i in 0..n {
            let (x, z) = at(i);
            let (s0, e, p) = (surfaces0[i], elevation[i], geo[i]);
            let climate = climates[i];
            let lake = lakes.iter().find(|l| {
                p.lat >= l.south
                    && p.lat <= l.north
                    && lon_in(p.lon, l.west, l.east)
                    && e <= f64::from(l.level_m) + COARSE_LAKE_OVER_M
            });
            let sea = lake.is_none() && e < 0.0 && mask.is_sea(p.lat, p.lon);
            let t = climate.map_or_else(
                || fallback_temperature_c(e, p.lat),
                |c| f64::from(c.temperature_c),
            );
            let precipitation = climate.map(|c| f64::from(c.precipitation_mm));
            let class = coarse_class(e, t, precipitation, sea || lake.is_some());
            let (s, kind) = if let Some(lake) = lake {
                let top = vertical.block_y(f64::from(lake.level_m)).saturating_sub(1);
                let ground = s0.min(top.saturating_sub(1));
                (
                    top,
                    SurfaceKind::Water {
                        depth: u32::try_from(i64::from(top) - i64::from(ground)).unwrap_or(1),
                        bed: Bed::Sand,
                    },
                )
            } else if sea {
                SurfaceKind::sea_column(s0, self.sea_level())
            } else {
                let dry = match (class, climate) {
                    (
                        LandClass::Bare | LandClass::Shrub | LandClass::Grass | LandClass::Wetland,
                        Some(c),
                    ) if self.climate_wanted() => Some(DryColumn::new(c, x, z)),
                    _ => None,
                };
                let slope = || -> Result<Slope, std::convert::Infallible> {
                    Ok(Slope::of_gradient(
                        neighbour_rise(&surfaces0, width, i) / f64::from(stride),
                    ))
                };
                let kind = SurfaceKind::on_classified(
                    class,
                    e,
                    p.lat,
                    s0 < self.sea_level(),
                    dry,
                    [x, z],
                    slope,
                )
                .unwrap_or(SurfaceKind::Grass);
                (s0, kind)
            };
            let biome = Biome::with_relief(
                class,
                if sea { e.min(0.0) } else { e },
                p.lat,
                climate,
                variant_cell(p.lat, p.lon),
                0.0,
            );
            surfaces.push(s);
            kinds.push(kind);
            classes.push(class);
            biomes.push(biome);
        }
        Ok(NodeColumns::coarse(
            stride,
            width,
            [sx, sz],
            surfaces,
            kinds,
            classes,
            biomes,
        ))
    }

    /// The atlas lakes big enough to show at voxels `metres` wide among the columns at `geo`: none without an
    /// atlas or when the columns span more than [`COARSE_LAKE_QUERY_MAX_DEG`].
    fn coarse_lakes(&self, geo: &[horizon_earth::LatLon], metres: f64) -> Vec<Arc<Lake>> {
        let Some(atlas) = self.water_atlas() else {
            return Vec::new();
        };
        let (mut s, mut n) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in geo {
            s = s.min(p.lat);
            n = n.max(p.lat);
        }
        // Longitudes are wrapped: take the box around the first column, unwrapping the rest near it.
        let lon0 = geo.first().map_or(0.0, |p| p.lon);
        let (mut w, mut e) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in geo {
            let d = horizon_earth::projection::wrap_longitude(p.lon - lon0);
            w = w.min(lon0 + d);
            e = e.max(lon0 + d);
        }
        let margin = metres / 111_320.0;
        if !(s.is_finite() && n.is_finite())
            || n - s + 2.0 * margin > COARSE_LAKE_QUERY_MAX_DEG
            || e - w + 2.0 * margin > COARSE_LAKE_QUERY_MAX_DEG
        {
            return Vec::new();
        }
        let voxel_km2 = (metres / 1000.0).powi(2);
        let min_km2 = (voxel_km2 * COARSE_LAKE_SHARE).max(COARSE_LAKE_MIN_KM2);
        let mut lakes = atlas.lakes_in(s - margin, w - margin, n + margin, e + margin);
        lakes.retain(|l| f64::from(l.area_km2) >= min_km2);
        lakes
    }
}

/// Whether the wrapped longitude `lon` lies in the box `west..=east` (degrees; the box may cross the antimeridian
/// or be unwrapped past it).
fn lon_in(lon: f64, west: f64, east: f64) -> bool {
    let d = (lon - west).rem_euclid(360.0);
    d <= (east - west).max(0.0)
}

/// The largest height difference (blocks) between column `i` of a `width`-wide grid and its four neighbours that
/// are in the grid.
fn neighbour_rise(surfaces: &[i32], width: usize, i: usize) -> f64 {
    let (cx, cz) = (i % width, i / width);
    let s = i64::from(surfaces[i]);
    let mut most = 0_i64;
    let mut take = |j: usize| most = most.max((i64::from(surfaces[j]) - s).abs());
    if cx > 0 {
        take(i - 1);
    }
    if cx + 1 < width {
        take(i + 1);
    }
    if cz > 0 {
        take(i - width);
    }
    if cz + 1 < width && i + width < surfaces.len() {
        take(i + width);
    }
    most as f64
}

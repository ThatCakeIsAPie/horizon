//! The Earth generator: a column's surface height from elevation tiles, and stone, soil, sand and water around it.
//!
//! A column is `(s, kind)`: its surface block height `s` and the [`SurfaceKind`] of its top. For a dry column `s` is
//! the elevation surface (the DEM) and the ground; for a water column `s` is the TOP WATER BLOCK. The sea level
//! `sea` is where the sea stops: the sea's top block is `sea - 1`.
//!
//! * dry columns, by kind (see [`super::surface`] for the snow line and the alpine band): `y <= s - 4` is stone
//!   (for the dry places' kinds, `y < s - 32`); grass over three dirt (`s - 3 <= y < s` dirt, a grass block at
//!   `y == s`); bare stone up to `s`; or snow for `s - 1 <= y <= s` over stone; air above; the dry places of a
//!   world with climate ([`super::dry`]) have terracotta bands by absolute height from 32 blocks under `s` up to
//!   their top;
//! * water columns ([`SurfaceKind::Water`] with `depth` blocks of water and a [`Bed`](super::water::Bed)): the
//!   ground is `g = s - depth`; water for `g < y <= s`, air above; the bed block for `g - 3 <= y <= g` (one
//!   block for a stone bed, which is stone all the way down), stone below. Under the sea (`s0 < sea`, `s0` the
//!   elevation surface) the column is `s = sea - 1`, `g = min(s0, s - 1)` over sand
//!   ([`SurfaceKind::sea_column`]).
//!
//! A far voxel (a sample `step > 1` blocks tall, from `lo` to `lo + step - 1`) that holds the surface block is
//! the surface block, so distant terrain shows one material per column and does not stripe with the height of
//! the voxel grid; the voxels above and below it are what the rule gives at their own top (below the ground) and
//! bottom (above it) block.
//!
//! With land cover on and a source handed in ([`EarthGenerator::with_land_cover`]) a column whose class is known
//! takes its top from the class ([`SurfaceKind::on_classified`]): a lake or marsh is water up to the surface
//! block over a bed under it, and a known class that is not water stays dry under the sea level. The sea fills
//! only columns whose class is water or unknown. The rules may move a column's `s` (the sea's top is under the
//! elevation of a sunken column), so every read that gives kinds gives the final surfaces with them
//! ([`EarthGenerator::kinds_classes_core`]); a box classification widens its margins by the most the rules carve
//! or raise ([`MAX_CARVE`], [`MAX_RAISE`]). Columns read their class at the zoom of their step, like their
//! height; bare ground also reads its neighbours' heights for the slope. Without it every rule above is exactly
//! what it was.
//!
//! Every query comes in two forms. The plain one reads only tiles the store already holds and gives
//! [`NotReady`], naming the tiles it lacks, when one is missing: it never invents a height, because a cube that
//! is saved is permanent. The `*_blocking` forms (see [`super::blocking`]) fetch what is missing and wait.
//!
//! Boxes and far nodes are classified from the elevation range over their footprint alone, so a cube of sky or
//! rock costs no per-block work. The classification is conservative: `Mixed` may hide a uniform box, but a
//! box called uniform is the same block everywhere.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use horizon_core::position::SectionPos;
use horizon_earth::fastmap::FastMap;
use horizon_earth::{
    ClimateLookup, ElevationField, Fetch, FetchError, LandClass, LandCoverSource, LoadError,
    TileId, TileStore,
};

use super::biome::BIOME_SAMPLE_M;
use super::cover::MossTop;
use super::dry::{
    BAND_COUNT, BARREN_FAR, BasaltTop, DRY_ROCK_DEPTH, DryColumn, DryTop, SCRUB_FAR, band_index,
    band_table,
};
use super::far_trees::MAX_TREE_UP;
use super::settings::EarthSettings;
use super::surface::{
    GLACIER_ICE_DEPTH, GLACIER_SNOW_DEPTH, SLOPE_BASE_M, SNOW_DEPTH, SOIL_DEPTH, Slope, SurfaceKind,
};
use super::water::{Bed, MAX_CARVE, MAX_RAISE, WINDOW_PAD};
use crate::cubic::custom::replacers::resolve_block;
use crate::cubic::custom::settings::BlockStateDesc;
use crate::registry::BlockRegistry;

/// The one biome an Earth world has without land cover (see [`super::biome`] for the biomes with it).
pub const BIOME: &str = "plains";

/// The block state ids the generator places.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EarthBlocks {
    pub air: u32,
    pub stone: u32,
    pub dirt: u32,
    pub grass_block: u32,
    pub sand: u32,
    pub water: u32,
    pub snow_block: u32,
    pub gravel: u32,
    pub mud: u32,
    pub clay: u32,
    pub moss_block: u32,
    /// The dry places' ground and the seven terracotta blocks of the bands ([`super::dry`]).
    pub red_sand: u32,
    pub coarse_dirt: u32,
    pub terracotta: u32,
    pub orange_terracotta: u32,
    pub yellow_terracotta: u32,
    pub brown_terracotta: u32,
    pub red_terracotta: u32,
    pub white_terracotta: u32,
    pub light_gray_terracotta: u32,
    /// The glacier ice and the wet bare rock tops ([`SurfaceKind::Glacier`], [`SurfaceKind::Basalt`]).
    pub packed_ice: u32,
    pub blue_ice: u32,
    pub basalt: u32,
    pub smooth_basalt: u32,
    pub tuff: u32,
    pub blackstone: u32,
    /// What the far view's trees and snow are made of.
    pub far: super::far_trees::FarBlocks,
}

impl EarthBlocks {
    /// The default state of each block in `registry`; fails naming the block it does not have.
    pub fn from_registry(registry: &BlockRegistry) -> Result<Self, String> {
        let get = |name: &str| resolve_block(registry, &BlockStateDesc::new(name));
        Ok(Self {
            air: get("air")?,
            stone: get("stone")?,
            dirt: get("dirt")?,
            grass_block: get("grass_block")?,
            sand: get("sand")?,
            water: get("water")?,
            snow_block: get("snow_block")?,
            gravel: get("gravel")?,
            mud: get("mud")?,
            clay: get("clay")?,
            moss_block: get("moss_block")?,
            red_sand: get("red_sand")?,
            coarse_dirt: get("coarse_dirt")?,
            terracotta: get("terracotta")?,
            orange_terracotta: get("orange_terracotta")?,
            yellow_terracotta: get("yellow_terracotta")?,
            brown_terracotta: get("brown_terracotta")?,
            red_terracotta: get("red_terracotta")?,
            white_terracotta: get("white_terracotta")?,
            light_gray_terracotta: get("light_gray_terracotta")?,
            packed_ice: get("packed_ice")?,
            blue_ice: get("blue_ice")?,
            basalt: get("basalt")?,
            smooth_basalt: get("smooth_basalt")?,
            tuff: get("tuff")?,
            blackstone: get("blackstone")?,
            far: super::far_trees::FarBlocks::from_registry(registry)?,
        })
    }
}

/// What a box of blocks is made of, as far as the elevation bounds prove it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    /// Every block is air.
    Air,
    /// Every block is stone.
    Stone,
    /// Every block is water.
    Water,
    /// Not proven uniform (it may still be).
    Mixed,
}

/// Elevation tiles (or land cover) a query needs are not in the store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotReady {
    /// The elevation tiles to fetch before asking again, in tile order. It can list a neighbour the answer would
    /// not have read, never fewer than it needs. It is empty when only land cover is missing: that is recorded
    /// in [`EarthGenerator::take_land_requests`].
    pub tiles: Vec<TileId>,
}

impl fmt::Display for NotReady {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.tiles.is_empty() {
            return f.write_str("land cover is not ready");
        }
        f.write_str("elevation tiles are not ready:")?;
        for (i, t) in self.tiles.iter().take(8).enumerate() {
            write!(
                f,
                "{} {}/{}/{}",
                if i == 0 { "" } else { "," },
                t.z,
                t.x,
                t.y
            )?;
        }
        if self.tiles.len() > 8 {
            write!(f, " and {} more", self.tiles.len() - 8)?;
        }
        Ok(())
    }
}

impl std::error::Error for NotReady {}

/// How the blocking forms retry a failed fetch.
#[derive(Clone, Copy, Debug)]
pub(super) struct Retry {
    pub first: Duration,
    pub cap: Duration,
}

/// A block rectangle (inclusive corners) read at one zoom.
#[derive(Clone, Copy, Debug)]
pub(super) struct Rect {
    pub x0: i32,
    pub z0: i32,
    pub x1: i32,
    pub z1: i32,
    pub zoom: u8,
}

/// Whether a read may call the fetcher.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    /// Only tiles already local.
    Local,
    /// Fetch what is missing.
    Fetch,
}

/// Why a read gave nothing.
pub(super) enum Gap {
    /// A tile is not local (a local read), or the position has no data.
    Missing,
    /// The fetcher failed.
    Load(LoadError),
    /// Land cover the source does not hold yet (a local read); the fetches are recorded.
    Land,
}

/// How many blocks a sea column's bed lies under its elevation surface at most without land cover: the bed is
/// under the block of water, so an elevation at the top of the sea has its bed one block lower.
const SEA_BED_SLACK: i64 = 1;

/// A generator of Earth terrain over a tile store.
pub struct EarthGenerator {
    pub(super) field: ElevationField,
    blocks: EarthBlocks,
    /// The terracotta bands of the dry places, by [`band_index`] of the absolute height.
    bands: Box<[u32; BAND_COUNT]>,
    sea_level: i32,
    base_zoom: u8,
    pub(super) cancel: Arc<AtomicBool>,
    pub(super) retry: Retry,
    /// Whether the last blocking read failed, so a streak of failures logs once.
    pub(super) failing: AtomicBool,
    /// Tiles far-view workers found missing, for whatever fetches tiles in the background.
    pub(super) requests: super::far::Requests,
    /// Whether the settings ask for land cover.
    land_cover: bool,
    /// The land cover the generator was handed, kept only while the settings ask for it.
    land: Option<Arc<LandCoverSource>>,
    /// Whether the settings ask for climate.
    climate_on: bool,
    /// The climate the generator was handed, kept only while the settings ask for it (and land cover is on).
    climate: Option<Arc<horizon_earth::ClimateSource>>,
    /// The water atlas and the river indexes built from it ([`super::water_gen`]).
    pub(super) water: super::water_gen::WaterState,
    /// Biomes the far view already read ([`super::biome`]).
    pub(super) biome_memo: super::biome::BiomeMemo,
    /// The surfaces and kinds of the cube columns already made, so the cubes stacked in a column share them.
    pub(super) column_memo: ColumnMemo,
    /// The hook that may compute padded windows elsewhere, and its switch ([`super::gpu_window`]).
    pub(super) window_gpu: super::gpu_window::WindowGpuSlot,
}

/// The final surfaces and the kinds of one cube column's 16 by 16 blocks (`z << 4 | x`).
pub(super) type CubeColumn = Arc<(Vec<i32>, Vec<SurfaceKind>)>;

/// Cube columns already worked out, by cube `(x, z)`: the surfaces and kinds do not depend on the cube's height,
/// and a column has about five mixed cubes stacked. Only answers that were made are kept, so a column whose data
/// is missing is asked for again. Sharded so the workers do not queue on one lock.
pub(super) struct ColumnMemo {
    shards: [Mutex<FastMap<(i32, i32), CubeColumn>>; COLUMN_MEMO_SHARDS],
}

const COLUMN_MEMO_SHARDS: usize = 16;
/// Columns kept per shard before it starts over (8192 in all, a few KB each).
const COLUMN_MEMO_SHARD_CAP: usize = 512;

impl Default for ColumnMemo {
    fn default() -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(FastMap::default())),
        }
    }
}

impl ColumnMemo {
    fn shard(&self, x: i32, z: i32) -> MutexGuard<'_, FastMap<(i32, i32), CubeColumn>> {
        let h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (z as u32).wrapping_mul(0x85EB_CA6B);
        self.shards[(h >> 24) as usize % COLUMN_MEMO_SHARDS]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn get(&self, x: i32, z: i32) -> Option<CubeColumn> {
        self.shard(x, z).get(&(x, z)).cloned()
    }

    fn put(&self, x: i32, z: i32, column: CubeColumn) {
        let mut shard = self.shard(x, z);
        if shard.len() >= COLUMN_MEMO_SHARD_CAP {
            shard.clear();
        }
        shard.insert((x, z), column);
    }

    /// Forgets every column.
    pub(super) fn clear(&self) {
        for shard in &self.shards {
            shard.lock().unwrap_or_else(PoisonError::into_inner).clear();
        }
    }
}

impl EarthGenerator {
    /// A generator for `settings` over `store` placing `blocks`. Fails with a message naming a bad setting.
    pub fn new(
        settings: &EarthSettings,
        store: Arc<TileStore>,
        blocks: EarthBlocks,
    ) -> Result<Self, String> {
        let field = settings.elevation_field(store)?;
        Ok(Self {
            base_zoom: field.zoom_for(settings.metres_per_block),
            field,
            bands: band_table(&blocks),
            blocks,
            sea_level: settings.sea_level_y,
            cancel: Arc::new(AtomicBool::new(false)),
            retry: Retry {
                first: Duration::from_millis(250),
                cap: Duration::from_secs(5),
            },
            failing: AtomicBool::new(false),
            requests: super::far::Requests::default(),
            land_cover: settings.land_cover,
            land: None,
            climate_on: settings.land_cover && settings.climate,
            climate: None,
            water: super::water_gen::WaterState::default(),
            biome_memo: super::biome::BiomeMemo::default(),
            column_memo: ColumnMemo::default(),
            window_gpu: super::gpu_window::WindowGpuSlot::default(),
        })
    }

    /// The same generator reading land cover from `source` (built by [`EarthSettings::land_cover_source`] so it
    /// shares the projection). It has no effect unless the settings say `landCover`: with it off, or without
    /// this call, the terrain is the elevation-only terrain exactly.
    pub fn with_land_cover(mut self, source: Arc<LandCoverSource>) -> Self {
        if self.land_cover {
            self.land = Some(source);
            self.column_memo.clear();
        }
        self
    }

    /// The same generator reading the climate from `source` (built by [`EarthSettings::climate_source`]). It has
    /// no effect unless the settings say both `landCover` and `climate`: without it the biomes are the land-cover
    /// table by latitude ([`super::biome::Biome::of`]) exactly.
    pub fn with_climate(mut self, source: Arc<horizon_earth::ClimateSource>) -> Self {
        if self.climate_on {
            self.climate = Some(source);
            self.column_memo.clear();
        }
        self
    }

    /// Whether the settings ask for climate (with land cover). With no source handed in, the biomes use the
    /// climate fallback ([`super::biome::Biome::with_climate`] without a climate).
    pub fn climate_wanted(&self) -> bool {
        self.climate_on
    }

    /// The climate source in use, for whatever fetches in the background.
    pub fn climate_source(&self) -> Option<&Arc<horizon_earth::ClimateSource>> {
        self.climate.as_ref()
    }

    /// Whether columns follow land cover: the settings ask for it and a source was handed in.
    pub fn land_cover_active(&self) -> bool {
        self.land.is_some()
    }

    /// The land cover source in use, for whatever fetches in the background.
    pub fn land_source(&self) -> Option<&Arc<LandCoverSource>> {
        self.land.as_ref()
    }

    /// The same generator whose blocking forms retry after `first`, doubling up to `cap`.
    pub fn with_retry(mut self, first: Duration, cap: Duration) -> Self {
        self.retry = Retry { first, cap };
        self
    }

    /// The same generator sharing `flag` as its cancel flag (see [`EarthGenerator::set_cancelled`]).
    pub fn with_cancel_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel = flag;
        self
    }

    /// The cancel flag, to share with whatever shuts the level down.
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    /// Sets or clears the cancel flag. While it is set the blocking forms stop waiting for tiles and return
    /// air (or `Mixed`, or nothing) at once.
    ///
    /// **Whatever a blocking form produced after the flag was set is not terrain and must be discarded**: do not
    /// save it, mesh it or cache it. Set it only when the level is shutting down.
    pub fn set_cancelled(&self, cancelled: bool) {
        self.cancel.store(cancelled, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    pub fn field(&self) -> &ElevationField {
        &self.field
    }

    pub fn blocks(&self) -> &EarthBlocks {
        &self.blocks
    }

    /// The sea level (`seaLevelY`): water fills the columns below it.
    pub fn sea_level(&self) -> i32 {
        self.sea_level
    }

    /// The biome of every column.
    pub fn biome(&self) -> &'static str {
        BIOME
    }

    /// The zoom of one-block samples: `zoom_for(metresPerBlock)`.
    pub fn base_zoom(&self) -> u8 {
        self.base_zoom
    }

    /// The zoom samples spaced `step` blocks apart read: `zoom_for(step * metresPerBlock)`. Steps below 1 count
    /// as 1.
    pub fn zoom_for_step(&self, step: i32) -> u8 {
        let metres = f64::from(step.max(1)) * self.field.projection().metres_per_block();
        self.field.zoom_for(metres)
    }

    /// The block at height `y` of a column whose surface is `s` and whose top is of `kind`. The ground (the top
    /// of the soil or bed) is `s` for most kinds and `kind.sink()` blocks lower under the water of a water column
    /// or marsh, which reaches the surface block `s`: `s` is the top water block there.
    pub fn block_for(&self, y: i32, s: i32, kind: SurfaceKind) -> u32 {
        let (y, s) = (i64::from(y), i64::from(s));
        let b = &self.blocks;
        let ground = s - kind.sink();
        let depth = match kind {
            // A stone bed is stone from its top block down.
            SurfaceKind::Water {
                bed: Bed::Stone, ..
            } => 0,
            _ if kind.is_dry() => DRY_ROCK_DEPTH,
            SurfaceKind::Glacier | SurfaceKind::Icefall => GLACIER_ICE_DEPTH,
            _ => SOIL_DEPTH,
        };
        if y < ground - depth {
            return b.stone;
        }
        if y > ground {
            return if y <= s { b.water } else { b.air };
        }
        match kind {
            SurfaceKind::Water { bed, .. } => match bed {
                Bed::Sand => b.sand,
                Bed::Gravel => b.gravel,
                Bed::Mud => b.mud,
                Bed::Clay => b.clay,
                Bed::Stone => b.stone,
            },
            SurfaceKind::Dune => b.sand,
            SurfaceKind::Grass if y == ground => b.grass_block,
            SurfaceKind::Grass => b.dirt,
            SurfaceKind::Alpine => b.stone,
            SurfaceKind::Snow if y > ground - SNOW_DEPTH => b.snow_block,
            SurfaceKind::Snow => b.stone,
            SurfaceKind::Gravel if y == ground => b.gravel,
            SurfaceKind::Gravel => b.stone,
            SurfaceKind::Glacier if y > ground - GLACIER_SNOW_DEPTH => b.snow_block,
            SurfaceKind::Glacier => b.packed_ice,
            SurfaceKind::Icefall if y == ground => b.packed_ice,
            SurfaceKind::Icefall => b.blue_ice,
            SurfaceKind::Basalt(top) if y == ground => match top {
                BasaltTop::Basalt => b.basalt,
                BasaltTop::SmoothBasalt => b.smooth_basalt,
                BasaltTop::Tuff => b.tuff,
                BasaltTop::Blackstone => b.blackstone,
            },
            SurfaceKind::Basalt(_) => b.stone,
            SurfaceKind::Moss(top) if y == ground => match top {
                MossTop::Moss => b.moss_block,
                MossTop::Grass => b.grass_block,
                MossTop::Stone => b.stone,
                MossTop::Gravel => b.gravel,
                MossTop::CoarseDirt => b.coarse_dirt,
            },
            SurfaceKind::Moss(MossTop::Stone | MossTop::Gravel) => b.stone,
            SurfaceKind::Moss(_) => b.dirt,
            SurfaceKind::Marsh => b.mud,
            SurfaceKind::Scrub(top) | SurfaceKind::Barren(top) if y == ground => {
                self.dry_top_block(top)
            }
            SurfaceKind::Banded | SurfaceKind::Scrub(_) | SurfaceKind::Barren(_) => {
                self.bands[band_index(y)]
            }
        }
    }

    /// The block of the far voxel whose lowest block is `lo` and which is `step` blocks tall, in a column whose
    /// surface is `s` and top of `kind`. A voxel holding the surface block is the surface block; one wholly
    /// below it is what its top block is, one wholly above it is what its bottom block is (air, `s` being the
    /// top of the water too). `step` 1 is [`EarthGenerator::block_for`].
    ///
    /// The dry kinds differ: their terracotta is the band of the voxel's middle height (a function of the height
    /// alone, so the bands stay level whatever the surfaces of the columns), and a mixed top is the mix's main
    /// block ([`SCRUB_FAR`], [`BARREN_FAR`]), not one column's pick standing for the whole voxel.
    ///
    /// On land the voxel holding the surface is ground only when the ground fills at least half of it
    /// ([`EarthGenerator::far_surface`]), so the far ground lies within half a voxel of the real one.
    pub fn voxel_block(&self, lo: i32, step: i32, s: i32, kind: SurfaceKind) -> u32 {
        if step <= 1 {
            return self.block_for(lo, s, kind);
        }
        // The voxels are chosen against the rounded surface `drawn`, then read `lift` blocks higher, so the
        // voxel holding `drawn` is the surface block and the ones under it are what lies under the surface.
        let drawn = Self::far_surface(s, step, kind);
        let lift = i64::from(s) - i64::from(drawn);
        let top = i64::from(lo) + i64::from(step) - 1;
        let surface = i64::from(drawn);
        let y = if i64::from(lo) <= surface && surface <= top {
            s
        } else if top < surface {
            (top + lift) as i32
        } else {
            (i64::from(lo) + lift).min(i64::from(i32::MAX)) as i32
        };
        if kind.is_dry() && y <= s && i64::from(y) >= i64::from(s) - DRY_ROCK_DEPTH {
            return match kind {
                SurfaceKind::Scrub(_) if y == s => self.dry_top_block(SCRUB_FAR),
                SurfaceKind::Barren(_) if y == s => self.dry_top_block(BARREN_FAR),
                _ => self.bands[band_index(i64::from(lo) + i64::from(step / 2))],
            };
        }
        self.block_for(y, s, kind)
    }

    /// The block of a dry top.
    fn dry_top_block(&self, top: DryTop) -> u32 {
        match top {
            DryTop::GrassBlock => self.blocks.grass_block,
            DryTop::CoarseDirt => self.blocks.coarse_dirt,
            DryTop::RedSand => self.blocks.red_sand,
        }
    }

    /// The height whose voxel a far grid of `step` draws as the surface of a land column whose surface block is
    /// `s`: the voxel holding `s` is ground only when the ground fills at least half of it (`s` in its upper
    /// half), else the voxel under it is the top. A voxel holding the surface drawn whole stood up to a whole
    /// voxel above the real ground: a sandy shore a block over the sea stood as a box a voxel tall out of the
    /// water. Water and marsh keep their surface: it is the water's top, level with the near water.
    pub fn far_surface(s: i32, step: i32, kind: SurfaceKind) -> i32 {
        if step <= 2 || kind.is_water() || kind == SurfaceKind::Marsh {
            s
        } else {
            s.saturating_sub(step / 2 - 1)
        }
    }

    /// The voxel of a step-`step` sample containing height `y` (`y` itself for step 1) starts here.
    pub fn voxel_floor(y: i32, step: i32) -> i32 {
        if step <= 1 {
            y
        } else {
            y.div_euclid(step) * step
        }
    }

    /// The column `(s, kind)` at the point `(x, z)` whose elevation surface block is `s0`, without land cover: the
    /// sea under the sea level ([`SurfaceKind::sea_column`], its top block `sea - 1`), else the kind by its
    /// elevation and the latitude of the point itself (not of the world's origin).
    pub fn surface_column(&self, x: i32, z: i32, s0: i32) -> (i32, SurfaceKind) {
        SurfaceKind::of_column(
            s0,
            self.latitude_at(x, z),
            self.sea_level,
            self.field.vertical(),
        )
    }

    /// The latitude and longitude (degrees) of the centre of block `(x, z)`.
    pub fn lat_lon_at(&self, x: i32, z: i32) -> (f64, f64) {
        let p = self
            .field
            .projection()
            .to_lat_lon(f64::from(x) + 0.5, f64::from(z) + 0.5);
        (p.lat, p.lon)
    }

    /// The latitude of the centre of block `(x, z)`.
    pub(super) fn latitude_at(&self, x: i32, z: i32) -> f64 {
        self.field
            .projection()
            .to_lat_lon(f64::from(x) + 0.5, f64::from(z) + 0.5)
            .lat
    }

    /// The ground distance (in blocks) a bare column's slope is measured over for samples `stride` blocks apart:
    /// [`SLOPE_BASE_M`] of ground, and at least the stride.
    pub(super) fn slope_base(&self, stride: i32) -> i32 {
        let metres = self.field.projection().metres_per_block();
        let blocks = (SLOPE_BASE_M / metres).round().clamp(1.0, 64.0) as i32;
        blocks.max(stride.max(1))
    }

    /// How steep the ground at block `(x, z)` (surface `s`) is: the largest height difference to a neighbour one
    /// [`EarthGenerator::slope_base`] away north, south, east or west, as a share of that distance
    /// ([`Slope::of_gradient`]), the heights read at the zoom of `zoom_step`. A neighbour whose tile is missing
    /// is a gap, never a guess.
    pub(super) fn slope_at(
        &self,
        a: Access,
        x: i32,
        z: i32,
        s: i32,
        stride: i32,
        zoom_step: i32,
    ) -> Result<Slope, Gap> {
        let d = self.slope_base(stride);
        let mut most = 0_i64;
        for (dx, dz) in [(d, 0), (-d, 0), (0, d), (0, -d)] {
            let n = self.surface_core(a, x.saturating_add(dx), z.saturating_add(dz), zoom_step)?;
            most = most.max((i64::from(n) - i64::from(s)).abs());
        }
        Ok(Slope::of_gradient(most as f64 / f64::from(d)))
    }

    /// Whether the ground at block `(x, z)` is steep for bare ground ([`Slope::is_steep`]).
    pub(super) fn is_steep(
        &self,
        a: Access,
        x: i32,
        z: i32,
        s: i32,
        stride: i32,
        zoom_step: i32,
    ) -> Result<bool, Gap> {
        Ok(self.slope_at(a, x, z, s, stride, zoom_step)?.is_steep())
    }

    /// The class of the column `(i, j)` of each of `width` by `depth` columns `stride` blocks apart from
    /// `(x0, z0)`, read at the zoom of `zoom_step`. A local read that lacks data records what to fetch
    /// ([`EarthGenerator::take_land_requests`]) and gives [`Gap::Land`].
    pub(super) fn classes_core(
        &self,
        a: Access,
        land: &LandCoverSource,
        [x0, z0]: [i32; 2],
        [width, depth]: [usize; 2],
        stride: i32,
        zoom_step: i32,
    ) -> Result<Vec<LandClass>, Gap> {
        let metres = f64::from(zoom_step.max(1)) * self.field.projection().metres_per_block();
        match a {
            Access::Local => land
                .field()
                .class_grid(x0, z0, width, depth, stride, metres)
                .map_err(|first| {
                    let span = |n: usize| (n.saturating_sub(1) as i64 * i64::from(stride)) as i32;
                    let mut wanted = vec![first];
                    wanted.extend(land.field().needs(
                        x0,
                        z0,
                        x0.saturating_add(span(width)),
                        z0.saturating_add(span(depth)),
                        metres,
                    ));
                    self.note_land(wanted);
                    Gap::Land
                }),
            Access::Fetch => land
                .class_grid_blocking(x0, z0, width, depth, stride, metres)
                .map_err(|e| Gap::Load(LoadError::Fetch(e))),
        }
    }

    /// The kinds of `width` by `depth` columns `stride` blocks apart from `(x0, z0)` whose elevation surfaces are
    /// `surfaces`, row by row: by land cover where the generator has it, else by
    /// [`EarthGenerator::surface_column`]. `zoom_step` is the spacing whose zoom and land-cover level the columns
    /// are read at. On success `surfaces` holds the FINAL surfaces (a water column's top block, a moved bank);
    /// on a gap it is left as it was.
    pub(super) fn kinds_core(
        &self,
        a: Access,
        origin: [i32; 2],
        size: [usize; 2],
        stride: i32,
        zoom_step: i32,
        surfaces: &mut [i32],
    ) -> Result<Vec<SurfaceKind>, Gap> {
        Ok(self
            .kinds_classes_core(a, origin, size, stride, zoom_step, surfaces)?
            .0)
    }

    /// [`EarthGenerator::kinds_core`] and the classes the kinds were made from (`None` without land cover).
    pub(super) fn kinds_classes_core(
        &self,
        a: Access,
        [x0, z0]: [i32; 2],
        [width, depth]: [usize; 2],
        stride: i32,
        zoom_step: i32,
        surfaces: &mut [i32],
    ) -> Result<(Vec<SurfaceKind>, Option<Vec<LandClass>>), Gap> {
        let at = |i: usize| {
            (
                x0.saturating_add((i % width) as i32 * stride),
                z0.saturating_add((i / width) as i32 * stride),
            )
        };
        let (columns, classes) = match &self.land {
            None => {
                let columns = surfaces
                    .iter()
                    .enumerate()
                    .map(|(i, &s0)| {
                        let (x, z) = at(i);
                        self.surface_column(x, z, s0)
                    })
                    .collect::<Vec<_>>();
                (columns, None)
            }
            Some(land) => {
                // The water rules read a padded window around the columns ([`super::water_gen`]).
                let water =
                    self.water_columns(a, land, [x0, z0], [width, depth], stride, zoom_step)?;
                let columns = water
                    .columns
                    .iter()
                    .zip(&water.classes)
                    .enumerate()
                    .map(|(i, (&cw, &class))| {
                        let (x, z) = at(i);
                        self.water_column(a, [x, z], cw, class, stride, zoom_step)
                    })
                    .collect::<Result<Vec<_>, Gap>>()?;
                (columns, Some(water.classes))
            }
        };
        let mut kinds = Vec::with_capacity(columns.len());
        for (slot, (s, kind)) in surfaces.iter_mut().zip(columns) {
            *slot = s;
            kinds.push(kind);
        }
        Ok((kinds, classes))
    }

    /// The column `(s, kind)` of the one column at `(x, z)` whose elevation surface is `s0`, samples `step`
    /// blocks apart.
    pub(super) fn kind_core(
        &self,
        a: Access,
        x: i32,
        z: i32,
        s0: i32,
        step: i32,
    ) -> Result<(i32, SurfaceKind), Gap> {
        if self.land.is_none() {
            return Ok(self.surface_column(x, z, s0));
        }
        let mut s = [s0];
        let kind = self.kinds_core(a, [x, z], [1, 1], step, step, &mut s)?[0];
        Ok((s[0], kind))
    }

    /// The column `(s, kind)` of a column of known `class` whose elevation surface is `s0`: what the class makes
    /// of it ([`SurfaceKind::on_classified`]), and under the sea level a water class is the sea
    /// ([`SurfaceKind::sea_column`]: its top is the block under the sea level, not `s0`).
    pub(super) fn land_column(
        &self,
        a: Access,
        [x, z]: [i32; 2],
        s0: i32,
        stride: i32,
        zoom_step: i32,
        class: LandClass,
    ) -> Result<(i32, SurfaceKind), Gap> {
        let elevation = self.field.vertical().elevation_for_y(f64::from(s0));
        // Only open ground and wetland read the climate: what a tree, crop, water or snow column is does not
        // depend on it.
        let dry = if matches!(
            class,
            LandClass::Bare | LandClass::Shrub | LandClass::Grass | LandClass::Wetland
        ) {
            self.dry_column(a, x, z)?
        } else {
            None
        };
        let kind = SurfaceKind::on_classified(
            class,
            elevation,
            self.latitude_at(x, z),
            s0 < self.sea_level,
            dry,
            [x, z],
            || self.slope_at(a, x, z, s0, stride, zoom_step),
        )?;
        Ok(if kind.is_water() && s0 < self.sea_level {
            SurfaceKind::sea_column(s0, self.sea_level)
        } else {
            (s0, kind)
        })
    }

    /// What the dry places' rules need of the column `(x, z)`: the climate at it, read as the biome reads it (at
    /// [`BIOME_SAMPLE_M`]). `None` without climate, or where the climate has no data (today's rules). A local
    /// read that lacks the climate records what to fetch and gives [`Gap::Land`].
    pub(super) fn dry_column(&self, a: Access, x: i32, z: i32) -> Result<Option<DryColumn>, Gap> {
        if !self.climate_wanted() {
            return Ok(None);
        }
        let Some(source) = self.climate_source() else {
            return Ok(None);
        };
        let (fx, fz) = (f64::from(x) + 0.5, f64::from(z) + 0.5);
        let climate = match a {
            Access::Local => match source.field().climate_at(fx, fz, BIOME_SAMPLE_M) {
                ClimateLookup::Climate(c) => Some(c),
                ClimateLookup::NoData => None,
                ClimateLookup::Need(fetch) => {
                    self.note_climate(fetch);
                    return Err(Gap::Land);
                }
            },
            Access::Fetch => source
                .climate_at_blocking(fx, fz, BIOME_SAMPLE_M)
                .map_err(|e| Gap::Load(LoadError::Fetch(e)))?,
        };
        Ok(climate.map(|c| DryColumn::new(c, x, z)))
    }

    /// `r` grown by what the kinds of its columns read beyond it (the water rules' window of
    /// [`WINDOW_PAD`] samples, and a bare column's neighbours), for the lists of tiles a read is not ready for.
    /// Unchanged without land cover.
    pub(super) fn kind_rect(&self, r: Rect, stride: i32) -> Rect {
        if self.land.is_none() {
            return r;
        }
        let d = Self::window_grow(stride).saturating_add(self.slope_base(stride));
        Rect {
            x0: r.x0.saturating_sub(d),
            z0: r.z0.saturating_sub(d),
            x1: r.x1.saturating_add(d),
            z1: r.z1.saturating_add(d),
            zoom: r.zoom,
        }
    }

    /// The land-cover fetches still missing for a read of the block rectangle `x0..=x1` by `z0..=z1` with samples
    /// `step` blocks apart, for prefetching; empty without land cover. Asking again after providing gives the
    /// next round (see [`horizon_earth::LandCoverStore::needs`]).
    pub fn land_needs(&self, x0: i32, z0: i32, x1: i32, z1: i32, step: i32) -> Vec<Fetch> {
        let Some(land) = &self.land else {
            return Vec::new();
        };
        let metres = f64::from(step.max(1)) * self.field.projection().metres_per_block();
        // The water rules read the classes of a padded window around what is asked.
        let g = Self::window_grow(step);
        let (x0, x1) = (x0.min(x1), x0.max(x1));
        let (z0, z1) = (z0.min(z1), z0.max(z1));
        land.field().needs(
            x0.saturating_sub(g),
            z0.saturating_sub(g),
            x1.saturating_add(g),
            z1.saturating_add(g),
            metres,
        )
    }

    /// The blocks the water rules' window reaches beyond columns `stride` blocks apart.
    pub(super) fn window_grow(stride: i32) -> i32 {
        (WINDOW_PAD as i32).saturating_mul(stride.max(1))
    }

    /// The block a [`Fill`] stands for; `None` for `Mixed`.
    pub fn fill_block(&self, fill: Fill) -> Option<u32> {
        match fill {
            Fill::Air => Some(self.blocks.air),
            Fill::Stone => Some(self.blocks.stone),
            Fill::Water => Some(self.blocks.water),
            Fill::Mixed => None,
        }
    }

    /// What blocks `ymin..=ymax` are in every column whose surface is somewhere in `s_lo..=s_hi`.
    fn classify_heights(&self, ymin: i32, ymax: i32, s_lo: i32, s_hi: i32) -> Fill {
        let (ymin, ymax) = (i64::from(ymin), i64::from(ymax));
        let (s_lo, s_hi) = (i64::from(s_lo), i64::from(s_hi));
        let sea = i64::from(self.sea_level);
        // The ground may lie under the elevation surface: by one block in a sea column whose elevation is at the
        // top of the sea (the water takes the block over the bed), and with land cover by up to MAX_CARVE (a lake
        // or river bed, a shelf, a carved bank), and what is under the sea level is water only where the map says
        // so, which the heights do not prove. With climate the dry places have terracotta bands down to
        // DRY_ROCK_DEPTH under the ground. Land the rules raise puts its top up to MAX_RAISE over the surface.
        let land = self.land.is_some();
        let (carve, raise) = if land {
            (MAX_CARVE, MAX_RAISE)
        } else {
            (SEA_BED_SLACK, 0)
        };
        let rock = if land && self.climate.is_some() {
            SOIL_DEPTH.max(DRY_ROCK_DEPTH).max(GLACIER_ICE_DEPTH)
        } else if land {
            SOIL_DEPTH.max(GLACIER_ICE_DEPTH)
        } else {
            SOIL_DEPTH
        };
        if ymax < s_lo - carve - rock {
            Fill::Stone
        } else if ymin > s_hi + raise && ymin >= sea {
            Fill::Air
        } else if !land && ymin > s_hi && ymax < sea {
            Fill::Water
        } else {
            Fill::Mixed
        }
    }

    // --- Tiles -----------------------------------------------------------------------------------------------

    /// The tiles a read of the block rectangle `x0..=x1` by `z0..=z1` (corners in either order) at one-block
    /// resolution needs and the store does not hold, for prefetching.
    pub fn missing_tiles(&self, x0: i32, z0: i32, x1: i32, z1: i32) -> Vec<TileId> {
        self.missing_in(self.kind_rect(
            Rect {
                x0,
                z0,
                x1,
                z1,
                zoom: self.base_zoom,
            },
            1,
        ))
    }

    /// [`EarthGenerator::missing_tiles`] for the footprint of one cube.
    pub fn missing_tiles_for_cube(&self, cube: SectionPos) -> Vec<TileId> {
        self.missing_in(self.kind_rect(cube_rect(cube, self.base_zoom), 1))
    }

    /// The tiles missing for a far node of the octree at `level` (a cube-aligned box of side `16 << level`
    /// blocks, as [`EarthGenerator::classify_node`] takes), at the zoom of that level's step. Its blocks all
    /// read the same tiles whatever their height, so there is no `y`.
    pub fn missing_tiles_for_node(&self, level: u32, x: i32, z: i32) -> Vec<TileId> {
        self.missing_in(self.kind_rect(self.node_rect(level, x, z), 1 << level.min(24)))
    }

    /// The tiles missing for a read of the block rectangle `x0..=x1` by `z0..=z1` (corners in either order) with
    /// samples `step` blocks apart, at the zoom of that step: what the far view asks for beyond one-block
    /// resolution. `step` 1 is [`EarthGenerator::missing_tiles`].
    pub fn missing_tiles_for_step(
        &self,
        x0: i32,
        z0: i32,
        x1: i32,
        z1: i32,
        step: i32,
    ) -> Vec<TileId> {
        self.missing_in(self.kind_rect(
            Rect {
                x0,
                z0,
                x1,
                z1,
                zoom: self.zoom_for_step(step),
            },
            step,
        ))
    }

    pub(super) fn missing_in(&self, r: Rect) -> Vec<TileId> {
        self.field
            .tiles_for(r.x0, r.z0, r.x1, r.z1, r.zoom)
            .into_iter()
            .filter(|&id| self.field.store().cached(id).is_none())
            .collect()
    }

    pub(super) fn point_rect(&self, x: i32, z: i32, step: i32) -> Rect {
        Rect {
            x0: x,
            z0: z,
            x1: x,
            z1: z,
            zoom: self.zoom_for_step(step),
        }
    }

    /// [`EarthGenerator::point_rect`] with what the column's kind reads around it, for the reads that give one.
    pub(super) fn kind_point_rect(&self, x: i32, z: i32, step: i32) -> Rect {
        self.kind_rect(self.point_rect(x, z, step), step)
    }

    /// The footprint of a node and the zoom of its step; its far faces are included, where voxel corners sit.
    pub(super) fn node_rect(&self, level: u32, x: i32, z: i32) -> Rect {
        let (min, max) = node_box(level, [x, 0, z]);
        Rect {
            x0: min[0],
            z0: min[2],
            x1: max[0],
            z1: max[2],
            zoom: self.zoom_for_step(1 << level.min(24)),
        }
    }

    // --- Reads, by access ------------------------------------------------------------------------------------

    fn range_of(&self, a: Access, r: Rect) -> Result<(f32, f32), Gap> {
        let got = match a {
            Access::Local => self.field.range(r.x0, r.z0, r.x1, r.z1, r.zoom),
            Access::Fetch => self
                .field
                .range_blocking(r.x0, r.z0, r.x1, r.z1, r.zoom)
                .map_err(Gap::Load)?,
        };
        got.ok_or(Gap::Missing)
    }

    fn surfaces_of(&self, a: Access, r: Rect) -> Result<Vec<i32>, Gap> {
        let (w, d) = (
            (i64::from(r.x1) - i64::from(r.x0) + 1) as usize,
            (i64::from(r.z1) - i64::from(r.z0) + 1) as usize,
        );
        let got = match a {
            Access::Local => self.field.surface_ys(r.x0, r.z0, w, d, r.zoom),
            Access::Fetch => self
                .field
                .surface_ys_blocking(r.x0, r.z0, w, d, r.zoom)
                .map_err(Gap::Load)?,
        };
        got.ok_or(Gap::Missing)
    }

    pub(super) fn surface_core(&self, a: Access, x: i32, z: i32, step: i32) -> Result<i32, Gap> {
        let zoom = self.zoom_for_step(step);
        let got = match a {
            Access::Local => self.field.surface_y(x, z, zoom),
            Access::Fetch => self
                .field
                .surface_y_blocking(x, z, zoom)
                .map_err(Gap::Load)?,
        };
        got.ok_or(Gap::Missing)
    }

    pub(super) fn block_core(&self, a: Access, [x, y, z]: [i32; 3], step: i32) -> Result<u32, Gap> {
        let s0 = self.surface_core(a, x, z, step)?;
        let (s, kind) = self.kind_core(a, x, z, s0, step)?;
        Ok(self.voxel_block(Self::voxel_floor(y, step), step, s, kind))
    }

    pub(super) fn column_core(
        &self,
        a: Access,
        x: i32,
        z: i32,
        ys: &[i32],
        step: i32,
        mut stop: impl FnMut(u32) -> bool,
    ) -> Result<Option<(usize, u32)>, Gap> {
        if ys.is_empty() {
            return Ok(None);
        }
        let s0 = self.surface_core(a, x, z, step)?;
        let (s, kind) = self.kind_core(a, x, z, s0, step)?;
        for (i, &y) in ys.iter().enumerate() {
            let block = self.voxel_block(Self::voxel_floor(y, step), step, s, kind);
            if stop(block) {
                return Ok(Some((i, block)));
            }
        }
        Ok(None)
    }

    pub(super) fn box_core(
        &self,
        a: Access,
        min: [i32; 3],
        max: [i32; 3],
        step: i32,
    ) -> Result<Fill, Gap> {
        self.box_core_up(a, min, max, step, 0)
    }

    /// [`EarthGenerator::box_core`] where the air counts only from `up` blocks over the highest ground: what
    /// trees and snow put over it are not proven away by the heights.
    pub(super) fn box_core_up(
        &self,
        a: Access,
        min: [i32; 3],
        max: [i32; 3],
        step: i32,
        up: i32,
    ) -> Result<Fill, Gap> {
        let r = Rect {
            x0: min[0],
            z0: min[2],
            x1: max[0],
            z1: max[2],
            zoom: self.zoom_for_step(step),
        };
        let (lo, hi) = self.range_of(a, r)?;
        // The voxels the box touches, whole: a voxel is one block, whatever part of it the box holds.
        let step = step.max(1);
        let ymin = Self::voxel_floor(min[1].min(max[1]), step);
        let ymax = Self::voxel_floor(min[1].max(max[1]), step).saturating_add(step - 1);
        let s_hi = self.vertical_block(hi).saturating_add(up);
        // A land voxel holding the surface is drawn as air when the ground fills less than half of it
        // ([`Self::far_surface`]): solid only under the lowest top that can be drawn.
        let s_lo = Self::far_surface(self.vertical_block(lo), step, SurfaceKind::Grass);
        Ok(self.classify_heights(ymin, ymax, s_lo, s_hi))
    }

    pub(super) fn node_core(
        &self,
        a: Access,
        level: u32,
        [x, y, z]: [i32; 3],
        up: i32,
    ) -> Result<Fill, Gap> {
        let (min, mut max) = node_box(level, [x, y, z]);
        // Its voxels are the `16 << level` blocks from `min`; the far face is only read for the columns.
        max[1] = min[1].saturating_add((16_i32 << level.min(24)) - 1);
        self.box_core_up(a, min, max, 1 << level.min(24), up)
    }

    pub(super) fn cube_core(
        &self,
        a: Access,
        cube: SectionPos,
        out: &mut [u32; 4096],
    ) -> Result<Option<u32>, Gap> {
        let r = cube_rect(cube, self.base_zoom);
        let y0 = cube.y * 16;
        let (lo, hi) = self.range_of(a, r)?;
        let fill = self.classify_heights(
            y0,
            y0 + 15,
            self.vertical_block(lo),
            self.vertical_block(hi),
        );
        if let Some(id) = self.fill_block(fill) {
            out.fill(id);
            return Ok(Some(id));
        }
        let column = match self.column_memo.get(cube.x, cube.z) {
            Some(column) => column,
            None => {
                let mut surfaces = self.surfaces_of(a, r)?;
                let kinds = self.kinds_core(a, [r.x0, r.z0], [16, 16], 1, 1, &mut surfaces)?;
                let column = Arc::new((surfaces, kinds));
                self.column_memo.put(cube.x, cube.z, Arc::clone(&column));
                column
            }
        };
        let (surfaces, kinds) = &*column;
        for dy in 0..16 {
            for (column, (&s, &kind)) in surfaces.iter().zip(kinds).enumerate() {
                out[dy << 8 | column] = self.block_for(y0 + dy as i32, s, kind);
            }
        }
        let first = out[0];
        Ok(out.iter().all(|&b| b == first).then_some(first))
    }

    pub(super) fn tops_core(&self, a: Access, cx: i32, cz: i32) -> Result<Vec<i32>, Gap> {
        self.surfaces_of(
            a,
            Rect {
                x0: cx * 16,
                z0: cz * 16,
                x1: cx * 16 + 15,
                z1: cz * 16 + 15,
                zoom: self.base_zoom,
            },
        )
    }

    fn vertical_block(&self, elevation_m: f32) -> i32 {
        self.field.vertical().block_y(f64::from(elevation_m))
    }

    // --- The fallible, non-blocking forms --------------------------------------------------------------------

    fn ready<T>(&self, r: Rect, got: Result<T, Gap>) -> Result<T, NotReady> {
        got.map_err(|_| NotReady {
            tiles: self.missing_in(r),
        })
    }

    /// Fills `out` (`y << 8 | z << 4 | x`) with the cube's blocks. `Some(id)` when the whole cube is one block
    /// (`out` is filled either way); in the common cases of sky and rock the bounds say so without a per-block
    /// read.
    pub fn generate(
        &self,
        cube: SectionPos,
        out: &mut [u32; 4096],
    ) -> Result<Option<u32>, NotReady> {
        self.ready(
            self.kind_rect(cube_rect(cube, self.base_zoom), 1),
            self.cube_core(Access::Local, cube, out),
        )
    }

    /// The block at `(x, y, z)`, sampled at the zoom of a sample spacing of `step` blocks (1 is exactly what
    /// [`EarthGenerator::generate`] places).
    pub fn block_at(&self, x: i32, y: i32, z: i32, step: i32) -> Result<u32, NotReady> {
        self.ready(
            self.kind_point_rect(x, z, step),
            self.block_core(Access::Local, [x, y, z], step),
        )
    }

    /// The blocks of one column at the heights `ys` in order until `stop` holds: the index and the block. The
    /// same answers as [`EarthGenerator::block_at`] with the column's height read once.
    pub fn first_at_column(
        &self,
        x: i32,
        z: i32,
        ys: &[i32],
        step: i32,
        stop: impl FnMut(u32) -> bool,
    ) -> Result<Option<(usize, u32)>, NotReady> {
        self.ready(
            self.kind_point_rect(x, z, step),
            self.column_core(Access::Local, x, z, ys, step, stop),
        )
    }

    /// The exact surface block height of the column `(x, z)`: the top block of the ground (sand or grass, under
    /// water too), read from the data however high it is.
    pub fn surface_y(&self, x: i32, z: i32) -> Result<i32, NotReady> {
        self.ready(
            self.point_rect(x, z, 1),
            self.surface_core(Access::Local, x, z, 1),
        )
    }

    /// The surface of each column of chunk (`cx`, `cz`), indexed `z << 4 | x`.
    pub fn chunk_tops(&self, cx: i32, cz: i32) -> Result<Vec<i32>, NotReady> {
        let r = Rect {
            x0: cx * 16,
            z0: cz * 16,
            x1: cx * 16 + 15,
            z1: cz * 16 + 15,
            zoom: self.base_zoom,
        };
        self.ready(r, self.tops_core(Access::Local, cx, cz))
    }

    /// What the blocks `min..=max` (inclusive) are, as one-block samples make them, when the elevation bounds
    /// prove it.
    pub fn classify_box(&self, min: [i32; 3], max: [i32; 3]) -> Result<Fill, NotReady> {
        self.classify_box_at(min, max, 1)
    }

    /// [`EarthGenerator::classify_box`] for blocks sampled with a spacing of `step`: it reads the zoom that
    /// [`EarthGenerator::block_at`] reads for that step.
    pub fn classify_box_at(
        &self,
        min: [i32; 3],
        max: [i32; 3],
        step: i32,
    ) -> Result<Fill, NotReady> {
        let r = Rect {
            x0: min[0],
            z0: min[2],
            x1: max[0],
            z1: max[2],
            zoom: self.zoom_for_step(step),
        };
        self.ready(r, self.box_core(Access::Local, min, max, step))
    }

    /// Classifies a node of the far-terrain octree: the cube-aligned box of side `16 << level` blocks whose
    /// minimum corner is `(x, y, z) * side`, and its far faces. Its voxels are sampled with a step of
    /// `1 << level`, so it reads the same zoom as [`EarthGenerator::block_at`] with that step, and the two agree.
    pub fn classify_node(&self, level: u32, x: i32, y: i32, z: i32) -> Result<Fill, NotReady> {
        self.ready(
            self.node_rect(level, x, z),
            self.node_core(Access::Local, level, [x, y, z], 0),
        )
    }

    /// A block height no column of the far node column (`level`, `x`, `z`) has its final surface under, at the
    /// zoom of the node's step: the lowest elevation over its footprint less what the water rules may carve
    /// ([`MAX_CARVE`]; one block for a sea bed without land cover). The bound [`EarthGenerator::classify_node`]
    /// proves stone with, so a node's sky cover can be known without reading its columns.
    pub fn ground_floor(&self, level: u32, x: i32, z: i32) -> Result<i32, NotReady> {
        let r = self.node_rect(level, x, z);
        let (lo, _) = self.ready(r, self.range_of(Access::Local, r))?;
        let carve = if self.land.is_some() {
            MAX_CARVE
        } else {
            SEA_BED_SLACK
        };
        let floor = i64::from(self.vertical_block(lo)) - carve;
        Ok(floor.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
    }

    /// [`EarthGenerator::classify_node`] for a world whose far view draws trees and snow: air is proven only
    /// from [`MAX_TREE_UP`] blocks over the highest ground, so a node trees reach into is not pruned.
    pub fn classify_node_above(
        &self,
        level: u32,
        x: i32,
        y: i32,
        z: i32,
    ) -> Result<Fill, NotReady> {
        self.ready(
            self.node_rect(level, x, z),
            self.node_core(Access::Local, level, [x, y, z], MAX_TREE_UP),
        )
    }

    /// [`EarthGenerator::classify_box_at`] with the margin of [`EarthGenerator::classify_node_above`].
    pub fn classify_box_above(
        &self,
        min: [i32; 3],
        max: [i32; 3],
        step: i32,
    ) -> Result<Fill, NotReady> {
        let r = Rect {
            x0: min[0],
            z0: min[2],
            x1: max[0],
            z1: max[2],
            zoom: self.zoom_for_step(step),
        };
        self.ready(
            r,
            self.box_core_up(Access::Local, min, max, step, MAX_TREE_UP),
        )
    }
}

/// The footprint of a cube's blocks.
fn cube_rect(cube: SectionPos, zoom: u8) -> Rect {
    Rect {
        x0: cube.x * 16,
        z0: cube.z * 16,
        x1: cube.x * 16 + 15,
        z1: cube.z * 16 + 15,
        zoom,
    }
}

/// The blocks of a far node, `min` to `max` inclusive of its far faces.
pub(super) fn node_box(level: u32, [x, y, z]: [i32; 3]) -> ([i32; 3], [i32; 3]) {
    let side = 16_i64 << level.min(24);
    let at = |v: i32| -> i32 { (i64::from(v) * side).clamp(-(1 << 30), 1 << 30) as i32 };
    let min = [at(x), at(y), at(z)];
    let max = min.map(|v| (i64::from(v) + side).min(i64::from(i32::MAX)) as i32);
    (min, max)
}

/// The error of a position that has no elevation, for the blocking forms (it cannot happen for finite settings).
pub(super) fn no_data() -> LoadError {
    LoadError::Fetch(FetchError::Other(
        "no elevation data for this position".into(),
    ))
}

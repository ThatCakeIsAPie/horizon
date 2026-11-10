//! Synthetic terrain for tests: no network, no real data. Available to this crate's tests and, with the `testing`
//! feature, to other crates' tests.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use crate::projection::pixel_to_latlon;
use crate::source::{FetchError, TileFetcher};
use crate::tile::{TILE_SIZE, TileId};

/// A terrain as a function of `(latitude, longitude)` in degrees, giving metres.
pub type HeightFn = dyn Fn(f64, f64) -> f64 + Send + Sync;

/// The three terrarium bytes (R, G, B) of an elevation in metres, rounded to the nearest 1/256 m and clamped to
/// what the encoding can hold.
pub fn encode_height(elevation_m: f64) -> [u8; 3] {
    let raw = ((elevation_m + 32768.0) * 256.0)
        .round()
        .clamp(0.0, 16_777_215.0) as u32;
    [(raw >> 16) as u8, (raw >> 8) as u8, raw as u8]
}

/// A PNG of the given pixel bytes, in any colour type and depth (to build tiles the decoder must refuse).
pub fn encode_png(
    width: u32,
    height: u32,
    color: png::ColorType,
    depth: png::BitDepth,
    data: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, width, height);
    encoder.set_color(color);
    encoder.set_depth(depth);
    let mut writer = encoder.write_header().expect("PNG header");
    writer.write_image_data(data).expect("PNG data");
    writer.finish().expect("PNG end");
    out
}

/// The terrarium PNG of tile `id`, sampling `height` at the centre of each pixel.
pub fn encode_tile(id: TileId, height: &HeightFn) -> Vec<u8> {
    let side = TILE_SIZE;
    let mut rgb = Vec::with_capacity((side * side * 3) as usize);
    for py in 0..side {
        for px in 0..side {
            let x = f64::from(id.x * side + px) + 0.5;
            let y = f64::from(id.y * side + py) + 0.5;
            let at = pixel_to_latlon(x, y, id.z);
            rgb.extend(encode_height(height(at.lat, at.lon)));
        }
    }
    encode_png(side, side, png::ColorType::Rgb, png::BitDepth::Eight, &rgb)
}

/// A [`TileFetcher`] that serves synthetic tiles made from a height function, counts what it is asked for, and can
/// be told to fail.
pub struct MemoryFetcher {
    height: Box<HeightFn>,
    fetches: AtomicUsize,
    per_tile: Mutex<HashMap<TileId, usize>>,
    missing: Mutex<HashSet<TileId>>,
    failure: Mutex<Option<FetchError>>,
    delay: Mutex<Duration>,
}

impl MemoryFetcher {
    /// Serves tiles of the terrain `height(lat, lon)`.
    pub fn new(height: impl Fn(f64, f64) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            height: Box::new(height),
            fetches: AtomicUsize::new(0),
            per_tile: Mutex::default(),
            missing: Mutex::default(),
            failure: Mutex::new(None),
            delay: Mutex::new(Duration::ZERO),
        }
    }

    /// How many fetches were asked for, failed ones included.
    pub fn fetch_count(&self) -> usize {
        self.fetches.load(Ordering::SeqCst)
    }

    /// How many fetches asked for this tile.
    pub fn fetches_of(&self, id: TileId) -> usize {
        lock(&self.per_tile).get(&id).copied().unwrap_or(0)
    }

    /// Makes every fetch fail with `error` (or succeed again with `None`).
    pub fn fail_with(&self, error: Option<FetchError>) {
        *lock(&self.failure) = error;
    }

    /// Makes this one tile answer "not found".
    pub fn make_missing(&self, id: TileId) {
        lock(&self.missing).insert(id);
    }

    /// Makes every fetch take this long, so concurrent callers overlap.
    pub fn set_delay(&self, delay: Duration) {
        *lock(&self.delay) = delay;
    }
}

impl TileFetcher for MemoryFetcher {
    fn fetch(&self, id: TileId) -> Result<Vec<u8>, FetchError> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        *lock(&self.per_tile).entry(id).or_default() += 1;
        let delay = *lock(&self.delay);
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
        if let Some(error) = lock(&self.failure).clone() {
            return Err(error);
        }
        if lock(&self.missing).contains(&id) {
            return Err(FetchError::NotFound);
        }
        Ok(encode_tile(id, &*self.height))
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A fresh empty folder under the system's temporary directory, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("horizon-earth-test-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temporary folder");
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Default for TempDir {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

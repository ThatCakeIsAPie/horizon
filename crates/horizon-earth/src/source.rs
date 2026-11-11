//! Where tiles come from: a fetcher the caller supplies, and a store that keeps what it fetched on disk and in
//! memory. The crate has no HTTP client; the caller's [`TileFetcher`] does the downloading.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::tile::{Tile, TileError, TileId, TileIdError};

/// Why a fetcher could not give a tile's bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchError {
    /// The source has no such tile.
    NotFound,
    /// Worth trying again later: a timeout, a busy server, no connection.
    Temporary(String),
    /// Anything else.
    Other(String),
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => f.write_str("the tile does not exist at the source"),
            Self::Temporary(why) => write!(f, "temporary failure fetching a tile: {why}"),
            Self::Other(why) => write!(f, "could not fetch a tile: {why}"),
        }
    }
}

impl std::error::Error for FetchError {}

/// Gives the PNG bytes of a tile. The caller implements it over whatever it downloads with.
pub trait TileFetcher: Send + Sync {
    /// The bytes of the terrarium PNG for `id`. May block.
    fn fetch(&self, id: TileId) -> Result<Vec<u8>, FetchError>;
}

/// Why [`TileStore::load`] gave no tile.
#[derive(Debug)]
pub enum LoadError {
    /// The id is not a tile the dataset has.
    InvalidId(TileIdError),
    /// The fetcher failed.
    Fetch(FetchError),
    /// The fetcher's bytes were not a usable tile (nothing was stored).
    Decode(TileError),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId(e) => e.fmt(f),
            Self::Fetch(e) => e.fmt(f),
            Self::Decode(e) => write!(f, "the fetched tile is unusable: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

struct Memory {
    tiles: HashMap<TileId, (Arc<Tile>, u64)>,
    clock: u64,
}

/// Decodes a tile's PNG at its zoom, logging once when pixels were repaired.
fn decode_tile(id: TileId, bytes: &[u8]) -> Result<Tile, TileError> {
    let tile = Tile::decode(bytes, id.z)?;
    if tile.repaired() > 0 {
        log::debug!(
            "earth tile {}/{}/{}: repaired {} implausible pixels",
            id.z,
            id.x,
            id.y,
            tile.repaired()
        );
    }
    Ok(tile)
}

/// A disk cache of tile PNGs laid out `{z}/{x}/{y}.png`, with a bounded memory cache of decoded tiles in front.
///
/// Safe to share between threads. Two threads that ask for the same missing tile make one fetch.
pub struct TileStore {
    dir: PathBuf,
    fetcher: Arc<dyn TileFetcher>,
    capacity: usize,
    memory: Mutex<Memory>,
    inflight: Mutex<HashMap<TileId, Arc<Mutex<()>>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl TileStore {
    /// A store that keeps files under `dir` and at most `memory_tiles` decoded tiles (each about 260 KiB), and
    /// fetches what it lacks through `fetcher`.
    pub fn new(
        dir: impl Into<PathBuf>,
        fetcher: Arc<dyn TileFetcher>,
        memory_tiles: usize,
    ) -> Self {
        Self {
            dir: dir.into(),
            fetcher,
            capacity: memory_tiles,
            memory: Mutex::new(Memory {
                tiles: HashMap::new(),
                clock: 0,
            }),
            inflight: Mutex::default(),
        }
    }

    /// Where the PNG of `id` lives on disk: `{dir}/{z}/{x}/{y}.png`.
    pub fn path_of(&self, id: TileId) -> PathBuf {
        self.dir
            .join(id.z.to_string())
            .join(id.x.to_string())
            .join(format!("{}.png", id.y))
    }

    /// Whether the tile is in memory or has a file on disk (the file is not checked for damage).
    pub fn is_local(&self, id: TileId) -> bool {
        lock(&self.memory).tiles.contains_key(&id) || self.path_of(id).is_file()
    }

    /// The tile if it is in memory or on disk; never calls the fetcher. A damaged file counts as absent.
    pub fn cached(&self, id: TileId) -> Option<Arc<Tile>> {
        id.validate().ok()?;
        if let Some(tile) = self.in_memory(id) {
            return Some(tile);
        }
        let bytes = fs::read(self.path_of(id)).ok()?;
        let tile = Arc::new(decode_tile(id, &bytes).ok()?);
        self.remember(id, &tile);
        Some(tile)
    }

    /// The tile, fetched when it is not local, and stored on disk and in memory. Blocks while it fetches. A file
    /// on disk that does not decode is replaced by a fresh fetch. A failure to write the file does not fail the
    /// load: the tile still serves from memory.
    pub fn load(&self, id: TileId) -> Result<Arc<Tile>, LoadError> {
        id.validate().map_err(LoadError::InvalidId)?;
        if let Some(tile) = self.cached(id) {
            return Ok(tile);
        }
        let gate = Arc::clone(lock(&self.inflight).entry(id).or_default());
        let result = {
            let _turn = lock(&gate);
            // Whoever held the gate before us may have fetched it already.
            match self.cached(id) {
                Some(tile) => Ok(tile),
                None => self.fetch_and_store(id),
            }
        };
        let mut inflight = lock(&self.inflight);
        drop(gate);
        // With our handle gone, the map's is the last one: nobody else is waiting on this gate.
        if inflight.get(&id).is_some_and(|g| Arc::strong_count(g) == 1) {
            inflight.remove(&id);
        }
        result
    }

    fn fetch_and_store(&self, id: TileId) -> Result<Arc<Tile>, LoadError> {
        let bytes = self.fetcher.fetch(id).map_err(LoadError::Fetch)?;
        let tile = Arc::new(decode_tile(id, &bytes).map_err(LoadError::Decode)?);
        let _ = write_atomically(&self.path_of(id), &bytes);
        self.remember(id, &tile);
        Ok(tile)
    }

    fn in_memory(&self, id: TileId) -> Option<Arc<Tile>> {
        let mut memory = lock(&self.memory);
        memory.clock += 1;
        let now = memory.clock;
        let entry = memory.tiles.get_mut(&id)?;
        entry.1 = now;
        Some(Arc::clone(&entry.0))
    }

    fn remember(&self, id: TileId, tile: &Arc<Tile>) {
        if self.capacity == 0 {
            return;
        }
        let mut memory = lock(&self.memory);
        memory.clock += 1;
        let now = memory.clock;
        memory.tiles.insert(id, (Arc::clone(tile), now));
        while memory.tiles.len() > self.capacity {
            let oldest = memory
                .tiles
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(id, _)| *id);
            match oldest {
                Some(oldest) => memory.tiles.remove(&oldest),
                None => break,
            };
        }
    }
}

/// Writes `bytes` to `path` through a temporary file in the same folder and a rename, so a reader never sees a
/// half-written file.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("a tile path has a folder"))?;
    fs::create_dir_all(dir)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temp = dir.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let written =
        fs::File::create(&temp).and_then(|mut f| f.write_all(bytes).and_then(|()| f.sync_all()));
    match written.and_then(|()| fs::rename(&temp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&temp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{MemoryFetcher, TempDir};
    use std::time::Duration;

    fn hills(lat: f64, lon: f64) -> f64 {
        100.0 + 50.0 * lat.to_radians().sin() + 20.0 * lon.to_radians().cos()
    }

    fn make(dir: &TempDir, memory: usize) -> (Arc<MemoryFetcher>, TileStore) {
        let fetcher = Arc::new(MemoryFetcher::new(hills));
        (
            Arc::clone(&fetcher),
            TileStore::new(dir.path(), fetcher, memory),
        )
    }

    const A: TileId = TileId { z: 4, x: 5, y: 6 };

    #[test]
    fn a_second_load_does_no_fetch() {
        let dir = TempDir::new();
        let (fetcher, store) = make(&dir, 8);
        let first = store.load(A).unwrap();
        let second = store.load(A).unwrap();
        assert_eq!(fetcher.fetch_count(), 1);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn a_fetched_tile_is_written_as_z_x_y_png() {
        let dir = TempDir::new();
        let (_, store) = make(&dir, 8);
        store.load(A).unwrap();
        assert!(dir.path().join("4/5/6.png").is_file());
        // No temporary files are left behind.
        let leftovers: Vec<_> = fs::read_dir(dir.path().join("4/5"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(leftovers, vec![std::ffi::OsString::from("6.png")]);
    }

    #[test]
    fn cached_never_fetches_and_finds_only_local_tiles() {
        let dir = TempDir::new();
        let (fetcher, store) = make(&dir, 8);
        assert!(store.cached(A).is_none());
        assert_eq!(fetcher.fetch_count(), 0);
        store.load(A).unwrap();
        assert!(store.cached(A).is_some());
        assert_eq!(fetcher.fetch_count(), 1);
    }

    #[test]
    fn a_new_store_finds_what_an_earlier_one_saved_on_disk() {
        let dir = TempDir::new();
        let (_, first) = make(&dir, 8);
        first.load(A).unwrap();
        let (fetcher, second) = make(&dir, 8);
        assert!(second.is_local(A));
        assert!(second.cached(A).is_some());
        second.load(A).unwrap();
        assert_eq!(fetcher.fetch_count(), 0);
    }

    #[test]
    fn a_corrupt_file_on_disk_is_refetched_and_replaced() {
        let dir = TempDir::new();
        let (fetcher, store) = make(&dir, 8);
        let path = store.path_of(A);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"this is not a png").unwrap();
        assert!(
            store.cached(A).is_none(),
            "a damaged file is as good as absent"
        );
        assert_eq!(fetcher.fetch_count(), 0);
        let tile = store.load(A).unwrap();
        assert_eq!(fetcher.fetch_count(), 1);
        assert!(tile.max() > 0.0);
        // The file is whole again: a fresh store reads it without fetching.
        let (again, fresh) = make(&dir, 8);
        assert!(fresh.cached(A).is_some());
        assert_eq!(again.fetch_count(), 0);
    }

    #[test]
    fn concurrent_loads_of_one_tile_fetch_once() {
        let dir = TempDir::new();
        let (fetcher, store) = make(&dir, 8);
        fetcher.set_delay(Duration::from_millis(100));
        let store = Arc::new(store);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || store.load(A).unwrap())
            })
            .collect();
        let tiles: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(fetcher.fetch_count(), 1);
        assert!(tiles.iter().all(|t| Arc::ptr_eq(t, &tiles[0])));
        assert!(
            lock(&store.inflight).is_empty(),
            "the per-tile gates are cleaned up"
        );
    }

    #[test]
    fn concurrent_loads_of_different_tiles_each_fetch_once() {
        let dir = TempDir::new();
        let (fetcher, store) = make(&dir, 8);
        let store = Arc::new(store);
        let ids = [A, TileId { z: 4, x: 6, y: 6 }, TileId { z: 4, x: 7, y: 6 }];
        let threads: Vec<_> = (0..9)
            .map(|i| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || store.load(ids[i % 3]).unwrap())
            })
            .collect();
        threads.into_iter().for_each(|t| drop(t.join().unwrap()));
        assert!(ids.iter().all(|&id| fetcher.fetches_of(id) == 1));
    }

    #[test]
    fn a_fetch_failure_is_reported_and_nothing_is_stored() {
        let dir = TempDir::new();
        let (fetcher, store) = make(&dir, 8);
        fetcher.fail_with(Some(FetchError::Temporary("timed out".into())));
        let err = store.load(A).unwrap_err();
        assert!(
            matches!(err, LoadError::Fetch(FetchError::Temporary(_))),
            "{err}"
        );
        assert!(store.cached(A).is_none());
        assert!(!store.path_of(A).exists());
        // Once the source recovers, the same tile loads.
        fetcher.fail_with(None);
        assert!(store.load(A).is_ok());
        fetcher.make_missing(TileId { z: 4, x: 0, y: 0 });
        let gone = store.load(TileId { z: 4, x: 0, y: 0 }).unwrap_err();
        assert!(matches!(gone, LoadError::Fetch(FetchError::NotFound)));
    }

    #[test]
    fn bytes_that_are_not_a_tile_are_never_stored() {
        struct Garbage;
        impl TileFetcher for Garbage {
            fn fetch(&self, _: TileId) -> Result<Vec<u8>, FetchError> {
                Ok(b"<html>an error page</html>".to_vec())
            }
        }
        let dir = TempDir::new();
        let store = TileStore::new(dir.path(), Arc::new(Garbage), 4);
        assert!(matches!(store.load(A), Err(LoadError::Decode(_))));
        assert!(!store.path_of(A).exists());
    }

    #[test]
    fn an_invalid_tile_id_is_refused_without_a_fetch() {
        let dir = TempDir::new();
        let (fetcher, store) = make(&dir, 8);
        let bad = TileId { z: 3, x: 8, y: 0 };
        assert!(matches!(store.load(bad), Err(LoadError::InvalidId(_))));
        assert!(store.cached(bad).is_none());
        assert_eq!(fetcher.fetch_count(), 0);
    }

    #[test]
    fn the_memory_cache_is_bounded_and_drops_the_least_recently_used() {
        let dir = TempDir::new();
        let (fetcher, store) = make(&dir, 2);
        let ids = [A, TileId { z: 4, x: 6, y: 6 }, TileId { z: 4, x: 7, y: 6 }];
        store.load(ids[0]).unwrap();
        store.load(ids[1]).unwrap();
        store.cached(ids[0]); // ids[1] is now the least recently used
        store.load(ids[2]).unwrap();
        assert_eq!(lock(&store.memory).tiles.len(), 2);
        assert!(lock(&store.memory).tiles.contains_key(&ids[0]));
        assert!(!lock(&store.memory).tiles.contains_key(&ids[1]));
        // The evicted tile is still on disk, so it comes back without a fetch.
        assert!(store.cached(ids[1]).is_some());
        assert_eq!(fetcher.fetch_count(), 3);
    }
}

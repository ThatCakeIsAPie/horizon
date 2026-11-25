//! Land cover with a way to fetch what is missing: the seam the game's future downloader plugs into.
//!
//! [`LandCoverStore`] never fetches; it says which byte range of which file it lacks. [`LandCoverFetcher`] is the
//! one small thing a caller implements to answer such a request (a file and a byte range in, bytes or "this file
//! does not exist" out), and [`LandCoverSource`] puts a [`LandCoverField`] and a fetcher together the way
//! [`crate::TileStore`] holds its fetcher: the plain reads give the [`Fetch`] that is missing, and the blocking
//! reads fetch it, hand it to the store and ask again.
//!
//! A file that does not exist at the source (WorldCover has none for open ocean) is not an error: the fetcher
//! answers `Ok(None)`, the store remembers it ([`LandCoverStore::mark_absent`]) and every lookup there is
//! [`Lookup::NoFile`] from then on, so nothing is asked twice and nothing waits.

use std::sync::Arc;

use crate::landcover::{LandClass, WorldCoverFile};
use crate::landfield::{Fetch, LandCoverField, Lookup};
use crate::source::FetchError;
use crate::tiff::ByteRange;

/// How many fetches one blocking read may make before it gives up (a read needs an index and a tile or two per
/// file; far more means the fetcher hands back bytes that never complete the answer).
const MAX_FETCHES_PER_READ: usize = 64;

/// Gives bytes of WorldCover files. The caller implements it over whatever it downloads with.
pub trait LandCoverFetcher: Send + Sync {
    /// `range` of `file` (see [`WorldCoverFile::path`] for its place in the dataset), as an HTTP range request
    /// gives it: cut short at the end of the file. `Ok(None)` when the file does not exist at the source. May
    /// block.
    fn fetch(&self, file: WorldCoverFile, range: ByteRange) -> Result<Option<Vec<u8>>, FetchError>;
}

/// Land cover on world blocks plus the fetcher that fills it.
#[derive(Clone)]
pub struct LandCoverSource {
    field: LandCoverField,
    fetcher: Arc<dyn LandCoverFetcher>,
}

impl LandCoverSource {
    pub fn new(field: LandCoverField, fetcher: Arc<dyn LandCoverFetcher>) -> Self {
        Self { field, fetcher }
    }

    pub fn field(&self) -> &LandCoverField {
        &self.field
    }

    /// Fetches `fetch` and hands it to the store; a file the fetcher says does not exist is marked absent
    /// (kept on disk). A fetcher that refuses ([`FetchError::Refused`]: downloads are off or not approved, the
    /// server says no) is answered as no data for the rest of the run ([`LandCoverStore::mark_unavailable`]),
    /// so a read never waits for what will not come, and the reason is the caller's to log (this still returns
    /// the error once, for it). Any other error means nothing was stored (and nothing was marked), so asking
    /// again retries.
    pub fn satisfy(&self, fetch: &Fetch) -> Result<(), FetchError> {
        match self.fetcher.fetch(fetch.file, fetch.range) {
            Ok(None) | Err(FetchError::NotFound) => {
                self.field.store().mark_absent(fetch.file);
                Ok(())
            }
            Err(refused @ FetchError::Refused(_)) => {
                self.field.store().mark_unavailable(fetch);
                Err(refused)
            }
            Ok(Some(bytes)) => self
                .field
                .store()
                .provide(fetch, &bytes)
                .map_err(|e| FetchError::Other(format!("land cover bytes refused: {e}"))),
            Err(e) => Err(e),
        }
    }

    /// The class at the world point `(x, z)` at `metres_per_sample` of ground, fetching what is missing.
    pub fn class_at_blocking(
        &self,
        x: f64,
        z: f64,
        metres_per_sample: f64,
    ) -> Result<LandClass, FetchError> {
        self.resolve(|| match self.field.class_at(x, z, metres_per_sample) {
            Lookup::Class(class) => Ok(class),
            Lookup::NoFile => Ok(LandClass::NoData),
            Lookup::Need(fetch) => Err(fetch),
        })
    }

    /// [`LandCoverField::class_grid`], fetching what is missing. Where there is no file the class is
    /// [`LandClass::NoData`].
    pub fn class_grid_blocking(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        stride: i32,
        metres_per_sample: f64,
    ) -> Result<Vec<LandClass>, FetchError> {
        self.resolve(|| {
            self.field
                .class_grid(x0, z0, width, depth, stride, metres_per_sample)
        })
    }

    /// Runs `read` until it gives an answer, fetching each [`Fetch`] it names.
    fn resolve<T>(&self, mut read: impl FnMut() -> Result<T, Fetch>) -> Result<T, FetchError> {
        for _ in 0..MAX_FETCHES_PER_READ {
            match read() {
                Ok(value) => return Ok(value),
                // A refusal marked what was asked for as no data: the next read answers without it.
                Err(fetch) => match self.satisfy(&fetch) {
                    Ok(()) | Err(FetchError::Refused(_)) => {}
                    Err(e) => return Err(e),
                },
            }
        }
        Err(FetchError::Other(
            "land cover: the fetched bytes never completed the answer".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::landfield::LandCoverStore;
    use crate::projection::Projection;
    use crate::testing::{MemoryLandCover, TempDir};

    const FILE: WorldCoverFile = WorldCoverFile { south: 45, west: 6 };

    fn source(fetcher: Arc<MemoryLandCover>, dir: &TempDir) -> LandCoverSource {
        let store = Arc::new(LandCoverStore::new(dir.path(), 16));
        let projection = Projection::local(46.0, 7.5, 1.0).unwrap();
        LandCoverSource::new(LandCoverField::new(store, projection), fetcher)
    }

    #[test]
    fn a_blocking_read_fetches_the_index_and_the_tile_it_needs_then_stops_asking() {
        let dir = TempDir::new();
        let fetcher = Arc::new(MemoryLandCover::new().with_classes(FILE, 256, 64, 2, |_, _| 40));
        let source = source(fetcher.clone(), &dir);
        let grid = source.class_grid_blocking(0, 0, 4, 4, 1, 1.0).unwrap();
        assert_eq!(grid, vec![LandClass::Crop; 16]);
        let asked = fetcher.fetch_count();
        assert!(asked >= 2, "an index and a tile, got {asked}");
        source.class_grid_blocking(0, 0, 4, 4, 1, 1.0).unwrap();
        assert_eq!(fetcher.fetch_count(), asked);
    }

    #[test]
    fn a_file_the_source_does_not_have_is_no_data_and_is_asked_for_once() {
        let dir = TempDir::new();
        let fetcher = Arc::new(MemoryLandCover::new());
        let source = source(fetcher.clone(), &dir);
        for _ in 0..3 {
            let class = source.class_at_blocking(0.5, 0.5, 1.0).unwrap();
            assert_eq!(class, LandClass::NoData);
        }
        assert_eq!(fetcher.fetches_of(FILE), 1);
        // The plain read agrees without asking.
        assert_eq!(
            source.field().class_at_block(0, 0),
            Lookup::NoFile,
            "marked absent"
        );
    }

    #[test]
    fn a_failing_fetcher_is_an_error_that_marks_nothing_absent_and_recovers() {
        let dir = TempDir::new();
        let fetcher = Arc::new(MemoryLandCover::new().with_classes(FILE, 256, 64, 2, |_, _| 10));
        let source = source(fetcher.clone(), &dir);
        fetcher.fail_with(Some(FetchError::Temporary("offline".into())));
        assert!(source.class_at_blocking(0.5, 0.5, 1.0).is_err());
        assert!(matches!(
            source.field().class_at_block(0, 0),
            Lookup::Need(_)
        ));
        fetcher.fail_with(None);
        assert_eq!(
            source.class_at_blocking(0.5, 0.5, 1.0).unwrap(),
            LandClass::Tree
        );
    }

    #[test]
    fn a_file_the_source_does_not_have_is_remembered_on_disk_for_the_next_run() {
        let dir = TempDir::new();
        let fetcher = Arc::new(MemoryLandCover::new());
        let first = source(fetcher.clone(), &dir);
        assert_eq!(first.class_at_blocking(0.5, 0.5, 1.0).unwrap(), LandClass::NoData);
        assert_eq!(fetcher.fetch_count(), 1);
        // A new store over the same folder asks nothing: its plain read already says "no file".
        let again = Arc::new(MemoryLandCover::new());
        let second = source(again.clone(), &dir);
        assert_eq!(second.field().class_at_block(0, 0), Lookup::NoFile);
        assert_eq!(second.class_at_blocking(0.5, 0.5, 1.0).unwrap(), LandClass::NoData);
        assert_eq!(again.fetch_count(), 0);
    }

    #[test]
    fn what_was_fetched_is_read_from_disk_by_the_next_run_without_a_request() {
        let dir = TempDir::new();
        let fetcher = Arc::new(MemoryLandCover::new().with_classes(FILE, 256, 64, 2, |_, _| 80));
        let first = source(fetcher.clone(), &dir);
        let grid = first.class_grid_blocking(0, 0, 8, 8, 1, 1.0).unwrap();
        assert_eq!(grid, vec![LandClass::Water; 64]);
        assert!(fetcher.fetch_count() >= 2);
        let again = Arc::new(MemoryLandCover::new());
        let second = source(again.clone(), &dir);
        assert_eq!(second.class_grid_blocking(0, 0, 8, 8, 1, 1.0).unwrap(), grid);
        assert_eq!(again.fetch_count(), 0);
    }

    #[test]
    fn a_refusal_reads_as_no_data_for_the_run_and_nothing_waits_or_is_written() {
        let dir = TempDir::new();
        let fetcher = Arc::new(MemoryLandCover::new().with_classes(FILE, 256, 64, 2, |_, _| 10));
        fetcher.fail_with(Some(FetchError::Refused("not approved".into())));
        let source = source(fetcher.clone(), &dir);
        // The blocking read answers instead of failing or looping: the file's index is "no data".
        assert_eq!(source.class_at_blocking(0.5, 0.5, 1.0).unwrap(), LandClass::NoData);
        assert_eq!(fetcher.fetch_count(), 1);
        // The caller that fetches in the background hears the refusal once, then the same answer.
        assert_eq!(source.field().class_at_block(0, 0), Lookup::NoFile);
        // Another run (the player approved meanwhile) asks again: nothing was written.
        fetcher.fail_with(None);
        let next = self::source(fetcher.clone(), &dir);
        assert_eq!(next.class_at_blocking(0.5, 0.5, 1.0).unwrap(), LandClass::Tree);
    }

    #[test]
    fn a_refused_tile_is_no_data_while_the_rest_of_the_file_is_read() {
        let dir = TempDir::new();
        let fetcher = Arc::new(MemoryLandCover::new().with_classes(FILE, 256, 64, 2, |_, _| 10));
        let source = source(fetcher.clone(), &dir);
        // The index first, then refuse the tile.
        let Lookup::Need(index) = source.field().class_at_block(0, 0) else {
            panic!("needs the index");
        };
        source.satisfy(&index).unwrap();
        let Lookup::Need(tile) = source.field().class_at_block(0, 0) else {
            panic!("needs the tile");
        };
        fetcher.fail_with(Some(FetchError::Refused("no".into())));
        assert!(matches!(source.satisfy(&tile), Err(FetchError::Refused(_))));
        assert_eq!(source.field().class_at_block(0, 0), Lookup::Class(LandClass::NoData));
    }
}

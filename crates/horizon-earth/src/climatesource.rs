//! Climate with a way to fetch what is missing: the seam a downloader plugs into.
//!
//! [`ClimateStore`](crate::ClimateStore) never fetches; it says which byte range of which file it lacks. [`ClimateFetcher`] is the one
//! small thing a caller implements to answer such a request (a variable and a byte range in, bytes or "this file
//! does not exist" out), and [`ClimateSource`] puts a [`ClimateField`] and a fetcher together the way
//! [`crate::LandCoverSource`] does: the blocking reads fetch what is missing, hand it to the store and ask again.
//!
//! A file that does not exist at the source is not an error: the fetcher answers `Ok(None)`, the store remembers
//! it ([`mark_absent`](crate::ClimateStore::mark_absent)) and every lookup is [`ClimateLookup::NoData`] from then on, so nothing is
//! asked twice and nothing waits.

use std::sync::Arc;

use crate::climate::{Climate, ClimateVar};
use crate::climatefield::{ClimateFetch, ClimateField, ClimateLookup};
use crate::source::FetchError;
use crate::tiff::ByteRange;

/// How many fetches one blocking read may make before it gives up. A read needs two indexes (a few pieces if the
/// first bytes are short) and up to four tiles of each file; far more means the fetcher hands back bytes that
/// never complete the answer.
const MAX_FETCHES_PER_READ: usize = 64;

/// Gives bytes of the CHELSA files. The caller implements it over whatever it downloads with.
pub trait ClimateFetcher: Send + Sync {
    /// `range` of the file of `var` (see [`ClimateVar::path`] for its place in the dataset), as an HTTP range
    /// request gives it: cut short at the end of the file. `Ok(None)` when the file does not exist at the source.
    /// May block.
    fn fetch(&self, var: ClimateVar, range: ByteRange) -> Result<Option<Vec<u8>>, FetchError>;
}

/// Climate on world blocks plus the fetcher that fills it.
#[derive(Clone)]
pub struct ClimateSource {
    field: ClimateField,
    fetcher: Arc<dyn ClimateFetcher>,
}

impl ClimateSource {
    pub fn new(field: ClimateField, fetcher: Arc<dyn ClimateFetcher>) -> Self {
        Self { field, fetcher }
    }

    pub fn field(&self) -> &ClimateField {
        &self.field
    }

    /// Fetches `fetch` and hands it to the store; a file the fetcher says does not exist is marked absent (kept
    /// on disk). A fetcher that refuses ([`FetchError::Refused`]) is answered as no data for the rest of the run
    /// ([`mark_unavailable`](crate::ClimateStore::mark_unavailable)), so a read never waits for what will not come; this still returns the
    /// error once, for the caller to log. Any other error means nothing was stored (and nothing was marked), so
    /// asking again retries.
    pub fn satisfy(&self, fetch: &ClimateFetch) -> Result<(), FetchError> {
        match self.fetcher.fetch(fetch.var, fetch.range) {
            Ok(None) | Err(FetchError::NotFound) => {
                self.field.store().mark_absent(fetch.var);
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
                .map_err(|e| FetchError::Other(format!("climate bytes refused: {e}"))),
            Err(e) => Err(e),
        }
    }

    /// The climate at the world point `(x, z)` at `metres_per_sample` of ground, fetching what is missing.
    /// `None` where there is no data.
    pub fn climate_at_blocking(
        &self,
        x: f64,
        z: f64,
        metres_per_sample: f64,
    ) -> Result<Option<Climate>, FetchError> {
        self.resolve(|| self.field.climate_at(x, z, metres_per_sample))
    }

    /// The climate at `(lat, lon)` at `metres_per_sample` of ground, fetching what is missing. `None` where
    /// there is no data.
    pub fn climate_at_latlon_blocking(
        &self,
        lat: f64,
        lon: f64,
        metres_per_sample: f64,
    ) -> Result<Option<Climate>, FetchError> {
        self.resolve(|| self.field.store().lookup(lat, lon, metres_per_sample))
    }

    /// Runs `read` until it gives an answer, fetching each [`ClimateFetch`] it names.
    fn resolve(
        &self,
        mut read: impl FnMut() -> ClimateLookup,
    ) -> Result<Option<Climate>, FetchError> {
        for _ in 0..MAX_FETCHES_PER_READ {
            match read() {
                ClimateLookup::Climate(climate) => return Ok(Some(climate)),
                ClimateLookup::NoData => return Ok(None),
                // A refusal marked what was asked for as no data: the next read answers without it.
                ClimateLookup::Need(fetch) => match self.satisfy(&fetch) {
                    Ok(()) | Err(FetchError::Refused(_)) => {}
                    Err(e) => return Err(e),
                },
            }
        }
        Err(FetchError::Other(
            "climate: the fetched bytes never completed the answer".into(),
        ))
    }
}

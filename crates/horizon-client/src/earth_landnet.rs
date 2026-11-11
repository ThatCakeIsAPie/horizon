//! Where an Earth world's land cover comes from: an HTTP [`LandCoverFetcher`] over the public WorldCover
//! bucket, and the disk cache the store keeps what it fetched in.
//!
//! WorldCover 2021 v200 is a set of 3 by 3 degree GeoTIFFs of a hundred megabytes each. The game never
//! downloads one whole: the store asks for the first 32 KiB of a file (its index) and then for the byte range
//! of each tile it reads, and this fetcher answers each with an HTTP range request.
//!
//! The address is not in the world's settings (a world file only says `landCover`): it is the folder the
//! game's build offers (`Hooks::earth_land_cover_address`, set by the official executable from
//! `horizon-addresses`), or the one the player gives in `HORIZON_EARTH_LAND_COVER_URL` (like the other
//! `HORIZON_*` service overrides). Without either there is no land cover source and a world that asks for it
//! runs without. Everything else follows the elevation downloader (`earth_net`): the same HTTP stack with the
//! same timeouts, a request carries only `User-Agent: Horizon/<version>` and a `Range`, nothing is contacted
//! unless the world allows downloads **and the host of the address is on the player's approved list**
//! (`earth_consent`), and a redirect may lead only to an approved host. A 404 is a file that does not exist
//! (open sea has none): `Ok(None)`. A refusal ([`FetchError::Refused`]) reads as "no data" for the run.
//!
//! What was fetched is kept under `.horizon-cache/earth-landcover/<hash of the address>/<dataset and rules
//! version>/<file name>/` in the game folder (see `horizon_earth::LandCoverStore`): the file's index pieces and
//! each compressed tile, plus a marker for a file the source does not have. A second run at the same place reads
//! all of it from disk and makes no request.
use crate::earth_consent::Approved;
use crate::earth_net::{Http, address_key, allowed, host_of};
use horizon_earth::tiff::ByteRange;
use horizon_earth::{FetchError, LandCoverFetcher, LandCoverSource, LandCoverStore, WorldCoverFile};
use horizon_world::cubic::earth::EarthSettings;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

/// The folder under the game folder where land cover is cached, one subfolder per address.
pub const CACHE_DIR: &str = ".horizon-cache/earth-landcover";
/// The environment variable that gives the address when the build offers none (or another one).
pub const ADDRESS_ENV: &str = "HORIZON_EARTH_LAND_COVER_URL";
/// The most bytes one range response may have: a file's index is 32 KiB and a compressed tile about a
/// megabyte at most.
pub const MAX_RANGE_BYTES: usize = 8 << 20;
/// Decoded land cover tiles a world keeps in memory (about 1 MiB each).
pub const MEMORY_TILES: usize = 48;
/// Range requests in flight at once, from every thread together (the elevation downloader's threads each
/// make one at a time).
pub const MAX_PARALLEL: usize = 3;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const READ_TIMEOUT: Duration = Duration::from_secs(15);
const TOTAL_TIMEOUT: Duration = Duration::from_secs(40);

/// The address land cover is read from: the player's override, else the build's offer.
pub fn address() -> Option<String> {
    std::env::var(ADDRESS_ENV)
        .ok()
        .map(|a| a.trim().to_owned())
        .filter(|a| !a.is_empty())
        .or_else(|| horizon_hooks::engine::earth_land_cover_address().map(str::to_owned))
}

/// The game-wide folder of the land cover cache: [`CACHE_DIR`] in the game folder (the working directory).
pub fn cache_root() -> PathBuf {
    std::env::current_dir().unwrap_or_default().join(CACHE_DIR)
}

/// `base` as the folder address of the dataset: a usable `https://` (or local `http://`) address, ending in `/`.
fn folder(base: &str) -> Result<String, String> {
    let base = base.trim();
    allowed(base).map_err(|why| why.to_string())?;
    host_of(base)?;
    if base.contains(['{', '}', '?', '#']) {
        return Err("the land cover address must be a plain folder address".to_owned());
    }
    Ok(if base.ends_with('/') {
        base.to_owned()
    } else {
        format!("{base}/")
    })
}

/// The host a world's land cover would be downloaded from, when the world wants it and downloads and an address
/// exists: what the player approves.
pub fn host_for(settings: &EarthSettings, address: Option<&str>) -> Option<String> {
    if !settings.land_cover || !settings.allow_downloads {
        return None;
    }
    host_of(&folder(address?).ok()?).ok()
}

/// A counting gate: at most `limit` holders at once.
struct Gate {
    limit: usize,
    busy: Mutex<usize>,
    freed: Condvar,
}

struct Permit<'a>(&'a Gate);

impl Gate {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            busy: Mutex::new(0),
            freed: Condvar::new(),
        }
    }

    fn enter(&self) -> Permit<'_> {
        let mut busy = self.busy.lock().unwrap_or_else(PoisonError::into_inner);
        while *busy >= self.limit {
            busy = self
                .freed
                .wait(busy)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *busy += 1;
        Permit(self)
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        *self.0.busy.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
        self.0.freed.notify_one();
    }
}

/// Fetches byte ranges of WorldCover files over HTTP, when the world allows downloads and the player approved
/// the host.
pub struct HttpLandCover {
    /// The folder address to download from, or why nothing is downloaded.
    source: Result<String, String>,
    http: Http,
    gate: Gate,
}

impl HttpLandCover {
    /// A fetcher for `settings` (the world's `landCover` and `allowDownloads`), the folder `address` (the
    /// build's or the player's) and the hosts `approved`.
    pub fn new(settings: &EarthSettings, address: Option<&str>, approved: &Approved) -> Self {
        let source = if !settings.land_cover {
            Err("this world does not use land cover (landCover is false)".to_owned())
        } else if !settings.allow_downloads {
            Err("land cover downloads are turned off for this world (allowDownloads is false)"
                .to_owned())
        } else {
            match address {
                None => Err(format!(
                    "this build of the game has no land cover source (set {ADDRESS_ENV} to a WorldCover \
                     folder address)"
                )),
                Some(address) => match folder(address) {
                    Err(why) => Err(format!("the land cover address is not usable: {why}")),
                    Ok(base) => match host_of(&base) {
                        Ok(host) if approved.contains(&host) => Ok(base),
                        Ok(host) => Err(format!(
                            "you have not approved downloads from {host} (the land cover server): turn on \
                             \"Allow downloads\" in Create World's Earth settings, or press the Allow button \
                             on the world's Edit screen"
                        )),
                        Err(why) => Err(format!("the land cover address is not usable: {why}")),
                    },
                },
            }
        };
        Self::over(
            source,
            approved.clone(),
            CONNECT_TIMEOUT,
            READ_TIMEOUT,
            TOTAL_TIMEOUT,
            MAX_PARALLEL,
        )
    }

    fn over(
        source: Result<String, String>,
        approved: Approved,
        connect: Duration,
        read: Duration,
        total: Duration,
        parallel: usize,
    ) -> Self {
        Self {
            source,
            http: Http::new(approved, connect, read, total, MAX_RANGE_BYTES),
            gate: Gate::new(parallel.max(1)),
        }
    }

    /// Whether this fetcher downloads at all; when not, [`Self::refusal`] says why.
    pub fn downloads(&self) -> bool {
        self.source.is_ok()
    }

    /// Why nothing is downloaded, when so.
    pub fn refusal(&self) -> Option<&str> {
        self.source.as_ref().err().map(String::as_str)
    }

    fn get(
        &self,
        base: &str,
        file: WorldCoverFile,
        range: ByteRange,
    ) -> Result<Option<Vec<u8>>, FetchError> {
        if range.len == 0 || range.len > MAX_RANGE_BYTES as u64 {
            return Err(FetchError::Other(format!(
                "a range of {} bytes is not one the game asks for",
                range.len
            )));
        }
        let url = format!("{base}{}", file.path());
        let last = range.offset.saturating_add(range.len - 1);
        let _permit = self.gate.enter();
        let reply = match self
            .http
            .get(&url, "image/tiff, */*;q=0.5", Some((range.offset, last)))
        {
            Ok(reply) => reply,
            // The file is not there: open sea has none.
            Err(FetchError::NotFound) => return Ok(None),
            Err(e) => return Err(e),
        };
        if reply.status == 206 {
            // `bytes first-last/total`: the start must be the one asked for, or the bytes are another piece.
            let first = reply
                .content_range
                .as_deref()
                .and_then(|r| r.trim().strip_prefix("bytes "))
                .and_then(|r| r.split(['-', '/']).next())
                .and_then(|n| n.trim().parse::<u64>().ok());
            if first != Some(range.offset) {
                return Err(FetchError::Other(format!(
                    "the server answered another byte range ({:?}) than asked for ({}-{last})",
                    reply.content_range, range.offset
                )));
            }
            return Ok(Some(reply.body));
        }
        // A server that ignores `Range` sends the whole file; it was capped at the range limit, so what arrived
        // is a whole small file at most.
        let start = usize::try_from(range.offset)
            .unwrap_or(usize::MAX)
            .min(reply.body.len());
        let end = usize::try_from(last.saturating_add(1))
            .unwrap_or(usize::MAX)
            .min(reply.body.len());
        Ok(Some(reply.body[start..end].to_vec()))
    }
}

impl LandCoverFetcher for HttpLandCover {
    fn fetch(&self, file: WorldCoverFile, range: ByteRange) -> Result<Option<Vec<u8>>, FetchError> {
        match &self.source {
            Ok(base) => self.get(base, file, range),
            Err(why) => Err(FetchError::Refused(why.clone())),
        }
    }
}

/// What an Earth world reads land cover with, or why it has none.
pub struct EarthLand {
    /// The source the generator reads, `None` when this world gets no land cover.
    pub source: Option<Arc<LandCoverSource>>,
    /// Why there is none (for the log).
    pub why_not: Option<String>,
    /// Where this world's land cover is cached.
    pub dir: PathBuf,
}

/// The land cover of a world with `settings`, caching under `root` (see [`cache_root`]), downloading from the
/// build's or the player's address for the hosts this player approved. Says in the log whether land cover is
/// used.
pub fn open(settings: &EarthSettings, root: &Path) -> EarthLand {
    open_with(settings, root, address().as_deref(), &Approved::for_player())
}

/// [`open`] for the folder `address` and the hosts `approved`.
pub fn open_with(
    settings: &EarthSettings,
    root: &Path,
    address: Option<&str>,
    approved: &Approved,
) -> EarthLand {
    let dir = root.join(address_key(address.unwrap_or_default()));
    let none = |why: String| EarthLand {
        source: None,
        why_not: Some(why),
        dir: dir.clone(),
    };
    if !settings.land_cover {
        return none("this world does not use land cover".to_owned());
    }
    let fetcher = HttpLandCover::new(settings, address, approved);
    if let Some(why) = fetcher.refusal() {
        if let Some(host) = crate::earth_consent::pending_land_host_of(settings, address, approved)
        {
            log::warn!(
                "Horizon: this Earth world wants land cover from {host}, which the player has not approved"
            );
            crate::earth_consent::note_unapproved(&host);
        }
        log::warn!(
            "Horizon: Earth land cover is off for this world (it runs as if landCover were false): {why}"
        );
        return none(why.to_owned());
    }
    let store = Arc::new(LandCoverStore::new(&dir, MEMORY_TILES));
    match settings.land_cover_source(store, Arc::new(fetcher)) {
        Ok(source) => {
            log::info!("Horizon: Earth land cover is downloaded as needed");
            EarthLand {
                source: Some(source),
                why_not: None,
                dir,
            }
        }
        Err(why) => {
            log::warn!("Horizon: Earth land cover is off for this world: {why}");
            none(why)
        }
    }
}

#[cfg(test)]
mod tests;

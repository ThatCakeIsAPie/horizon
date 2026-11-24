//! Where an Earth world's climate comes from: an HTTP [`ClimateFetcher`] over the public CHELSA V2.1 bucket, and
//! the disk cache the store keeps what it fetched in.
//!
//! CHELSA V2.1 (1981-2010, CC0 1.0) serves mean annual temperature (`bio01`) and annual precipitation (`bio12`)
//! as two global cloud-optimised GeoTIFFs of a few hundred megabytes. The game never downloads one whole: the
//! store asks for the first 64 KiB of a file (its index) and then for the byte range of each 512 by 512 tile it
//! reads, and this fetcher answers each with an HTTP range request.
//!
//! It works like the land cover downloader (`earth_landnet`), whose HTTP code it uses: the address is the
//! build's (`Hooks::earth_climate_address`, from `horizon-addresses`) or `HORIZON_EARTH_CLIMATE_URL`; nothing is
//! contacted unless the world allows downloads and the player approved the host (`earth_consent`); a refusal
//! reads as "no data" for the run, and the biomes fall back to the table without climate.
//!
//! What was fetched is kept under `.horizon-cache/earth-climate/<hash of the address>/<dataset and rules
//! version>/<variable>/` in the game folder (see `horizon_earth::ClimateStore`). The whole world's climate is a
//! few tiles: a second run at the same place makes no request.
use crate::earth_consent::Approved;
use crate::earth_landnet::{HttpLandCover, folder};
use crate::earth_net::{address_key, host_of};
use horizon_earth::tiff::ByteRange;
use horizon_earth::{ClimateFetcher, ClimateSource, ClimateStore, ClimateVar, FetchError};
use horizon_world::cubic::earth::EarthSettings;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// The folder under the game folder where climate is cached, one subfolder per address.
pub const CACHE_DIR: &str = ".horizon-cache/earth-climate";
/// The environment variable that gives the address when the build offers none (or another one).
pub const ADDRESS_ENV: &str = "HORIZON_EARTH_CLIMATE_URL";
/// Decoded climate tiles a world keeps in memory (512 KiB each).
pub const MEMORY_TILES: usize = 16;
const MAX_PARALLEL: usize = 3;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const READ_TIMEOUT: Duration = Duration::from_secs(15);
const TOTAL_TIMEOUT: Duration = Duration::from_secs(40);

/// The address climate is read from: the player's override, else the build's offer.
pub fn address() -> Option<String> {
    std::env::var(ADDRESS_ENV)
        .ok()
        .map(|a| a.trim().to_owned())
        .filter(|a| !a.is_empty())
        .or_else(|| horizon_hooks::engine::earth_climate_address().map(str::to_owned))
}

/// The game-wide folder of the climate cache: [`CACHE_DIR`] in the game folder (the working directory).
pub fn cache_root() -> PathBuf {
    std::env::current_dir().unwrap_or_default().join(CACHE_DIR)
}

/// The host a world's climate would be downloaded from, when the world wants it (with land cover) and downloads
/// and an address exists: what the player approves.
pub fn host_for(settings: &EarthSettings, address: Option<&str>) -> Option<String> {
    if !settings.climate || !settings.land_cover || !settings.allow_downloads {
        return None;
    }
    host_of(&folder(address?).ok()?).ok()
}

/// Fetches byte ranges of the CHELSA files over HTTP, when the world allows downloads and the player approved the
/// host.
pub struct HttpClimate(HttpLandCover);

impl HttpClimate {
    /// A fetcher for `settings`, the folder `address` and the hosts `approved`.
    pub fn new(settings: &EarthSettings, address: Option<&str>, approved: &Approved) -> Self {
        let source = if !(settings.climate && settings.land_cover) {
            Err("this world does not use the climate (climate is false)".to_owned())
        } else if !settings.allow_downloads {
            Err("climate downloads are turned off for this world (allowDownloads is false)".to_owned())
        } else {
            match address.map(folder) {
                None => Err(format!(
                    "this build of the game has no climate source (set {ADDRESS_ENV} to the CHELSA bioclim \
                     folder address)"
                )),
                Some(Err(why)) => Err(format!("the climate address is not usable: {why}")),
                Some(Ok(base)) => match host_of(&base) {
                    Ok(host) if approved.contains(&host) => Ok(base),
                    Ok(host) => Err(format!(
                        "you have not approved downloads from {host} (the climate server)"
                    )),
                    Err(why) => Err(format!("the climate address is not usable: {why}")),
                },
            }
        };
        Self(HttpLandCover::over(
            source,
            approved.clone(),
            CONNECT_TIMEOUT,
            READ_TIMEOUT,
            TOTAL_TIMEOUT,
            MAX_PARALLEL,
        ))
    }

    /// Why nothing is downloaded, when so.
    pub fn refusal(&self) -> Option<&str> {
        self.0.refusal()
    }
}

impl ClimateFetcher for HttpClimate {
    fn fetch(&self, var: ClimateVar, range: ByteRange) -> Result<Option<Vec<u8>>, FetchError> {
        self.0.get_path(var.path(), range)
    }
}

/// The climate of a world with `settings`, caching under `root` (see [`cache_root`]), downloading from the build's
/// or the player's address for the hosts this player approved. `None` (logged) when the world gets none: its
/// biomes then use the table without climate.
pub fn open(settings: &EarthSettings, root: &Path) -> Option<Arc<ClimateSource>> {
    open_with(settings, root, address().as_deref(), &Approved::for_player())
}

/// [`open`] for the folder `address` and the hosts `approved`.
pub fn open_with(
    settings: &EarthSettings,
    root: &Path,
    address: Option<&str>,
    approved: &Approved,
) -> Option<Arc<ClimateSource>> {
    if !(settings.climate && settings.land_cover) {
        return None;
    }
    let fetcher = HttpClimate::new(settings, address, approved);
    if let Some(why) = fetcher.refusal() {
        if let Some(host) = host_for(settings, address).filter(|h| !approved.contains(h)) {
            log::warn!(
                "Horizon: this Earth world wants climate from {host}, which the player has not approved"
            );
            crate::earth_consent::note_unapproved(&host);
        }
        log::warn!("Horizon: Earth climate is off for this world (biomes by latitude): {why}");
        return None;
    }
    let dir = root.join(address_key(address.unwrap_or_default()));
    let store = Arc::new(ClimateStore::new(&dir, MEMORY_TILES));
    match settings.climate_source(store, Arc::new(fetcher)) {
        Ok(source) => {
            log::info!("Horizon: Earth climate is downloaded as needed");
            Some(source)
        }
        Err(why) => {
            log::warn!("Horizon: Earth climate is off for this world: {why}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> EarthSettings {
        let mut s = EarthSettings::at(46.5, 8.0);
        s.land_cover = true;
        s.climate = true;
        s.allow_downloads = true;
        s
    }

    #[test]
    fn the_climate_host_is_asked_only_for_a_world_that_uses_it() {
        let address = Some("https://climate.example.org/chelsa/");
        assert_eq!(
            host_for(&settings(), address).as_deref(),
            Some("climate.example.org")
        );
        for off in [
            |s: &mut EarthSettings| s.climate = false,
            |s: &mut EarthSettings| s.land_cover = false,
            |s: &mut EarthSettings| s.allow_downloads = false,
        ] {
            let mut s = settings();
            off(&mut s);
            assert_eq!(host_for(&s, address), None);
        }
        assert_eq!(host_for(&settings(), None), None);
    }

    #[test]
    fn an_unapproved_host_gives_no_climate_and_contacts_nothing() {
        let dir = std::env::temp_dir().join(format!("horizon-climatenet-{}", std::process::id()));
        let address = Some("https://climate.example.org/chelsa/");
        let fetcher = HttpClimate::new(&settings(), address, &Approved::of(&[]));
        assert!(fetcher.refusal().unwrap().contains("climate.example.org"));
        assert!(matches!(
            fetcher.fetch(ClimateVar::Temperature, ByteRange { offset: 0, len: 16 }),
            Err(FetchError::Refused(_))
        ));
        assert!(open_with(&settings(), &dir, address, &Approved::of(&[])).is_none());
        let approved = Approved::of(&["climate.example.org"]);
        assert!(open_with(&settings(), &dir, address, &approved).is_some());
    }
}

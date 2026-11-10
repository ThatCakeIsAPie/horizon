//! Where an Earth world's elevation tiles come from: an HTTP [`TileFetcher`] over the address the world names,
//! and the tile cache it keeps on disk.
//!
//! The address is the world's `tileUrl` setting, a template with `{z}`, `{x}` and `{y}`, and nothing else:
//! the game has no built-in tile server (docs/layers.md, "Built-in addresses"). Nothing is downloaded unless
//! the world's `allowDownloads` is true and the template is an `https://` address (or `http://` to
//! `127.0.0.1`, `localhost` or `[::1]`, for testing). Otherwise every fetch fails with a message that says why,
//! and the world runs on the tiles already cached.
//!
//! A request carries only a neutral `User-Agent` (`Horizon/<version>`): no cookies, no identifiers of the
//! player, no referrer. A redirect is followed (up to three hops) only to an address the same rules allow, so
//! an `https://` address never leads to plain `http://`.
//!
//! Tiles are cached under `.horizon-cache/earth-tiles/<hash of the address>/{z}/{x}/{y}.png`, in the game
//! folder like the other game-wide caches, so worlds share what they download and two servers never mix tiles.
use horizon_earth::{FetchError, TileFetcher, TileId, TileStore};
use horizon_world::cubic::earth::EarthSettings;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// The folder under the game folder where tiles are cached, one subfolder per tile address.
pub const CACHE_DIR: &str = ".horizon-cache/earth-tiles";
/// The most bytes one tile response may have. A terrarium tile is about 100 KB.
pub const MAX_TILE_BYTES: usize = 4 << 20;
/// Decoded tiles a world keeps in memory (about 260 KiB each).
pub const MEMORY_TILES: usize = 192;
/// Redirects followed per tile.
const MAX_REDIRECTS: usize = 3;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const READ_TIMEOUT: Duration = Duration::from_secs(15);
const TOTAL_TIMEOUT: Duration = Duration::from_secs(40);

/// The URL of tile `id` under `template`: its `{z}`, `{x}` and `{y}` replaced.
pub fn tile_url(template: &str, id: TileId) -> String {
    template
        .replace("{z}", &id.z.to_string())
        .replace("{x}", &id.x.to_string())
        .replace("{y}", &id.y.to_string())
}

/// The `scheme://authority` of `url`, split into the scheme and the host (without port), when the shape is
/// plausible: no spaces or control characters, no user name or password.
fn origin_of(url: &str) -> Result<(&str, &str), String> {
    if url.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err("the address has spaces or control characters".into());
    }
    let (scheme, rest) = url
        .split_once("://")
        .ok_or("the address must start with https://")?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.contains('@') {
        return Err("the address must not hold a user name or password".into());
    }
    let host = if authority.starts_with('[') {
        authority
            .split_once(']')
            .map_or(authority, |(inside, _)| &authority[..=inside.len()])
    } else {
        authority.split(':').next().unwrap_or_default()
    };
    if host.is_empty() {
        return Err("the address has no host".into());
    }
    Ok((scheme, host))
}

/// Whether the game may download from `url`: `https://` anywhere, plain `http://` only to this machine.
fn allowed(url: &str) -> Result<(), String> {
    let (scheme, host) = origin_of(url)?;
    match scheme {
        "https" => Ok(()),
        "http" if matches!(host, "127.0.0.1" | "localhost" | "[::1]") => Ok(()),
        "http" => Err(
            "only https:// addresses are allowed (http:// only to 127.0.0.1 or localhost)".into(),
        ),
        _ => Err("the address must start with https://".into()),
    }
}

/// Checks a tile address template: one the game may download from, with `{z}`, `{x}` and `{y}` in the path
/// or query (not in the host) and no other `{...}`. The error says what is wrong.
pub fn check_template(template: &str) -> Result<(), String> {
    for placeholder in ["{z}", "{x}", "{y}"] {
        if !template.contains(placeholder) {
            return Err(format!("the tile address needs {placeholder} in it"));
        }
    }
    let rest = template
        .replace("{z}", "")
        .replace("{x}", "")
        .replace("{y}", "");
    if rest.contains(['{', '}']) {
        return Err("the tile address has a {...} other than {z}, {x} and {y}".into());
    }
    let authority_end = template.split_once("://").map_or(0, |(scheme, rest)| {
        scheme.len() + 3 + rest.find(['/', '?', '#']).unwrap_or(rest.len())
    });
    if template[..authority_end].contains('{') {
        return Err("the host of the tile address cannot hold {z}, {x} or {y}".into());
    }
    allowed(template)
}

/// Fetches tiles over HTTP from the address a world names, when it allows downloads.
pub struct HttpTileFetcher {
    /// The template to download from, or why nothing is downloaded.
    source: Result<String, String>,
    agent: ureq::Agent,
    max_bytes: usize,
}

impl HttpTileFetcher {
    /// A fetcher for the world's `tileUrl` and `allowDownloads`.
    pub fn new(settings: &EarthSettings) -> Self {
        let source = if !settings.allow_downloads {
            Err("tile downloads are turned off for this world (allowDownloads is false)".to_owned())
        } else if settings.tile_url.trim().is_empty() {
            Err("this world names no tile server (tileUrl is empty)".to_owned())
        } else {
            let template = settings.tile_url.trim();
            check_template(template)
                .map(|()| template.to_owned())
                .map_err(|why| format!("tileUrl is not usable: {why}"))
        };
        Self::over(
            source,
            CONNECT_TIMEOUT,
            READ_TIMEOUT,
            TOTAL_TIMEOUT,
            MAX_TILE_BYTES,
        )
    }

    fn over(
        source: Result<String, String>,
        connect: Duration,
        read: Duration,
        total: Duration,
        max_bytes: usize,
    ) -> Self {
        let agent = ureq::AgentBuilder::new()
            .user_agent(&format!("Horizon/{}", horizon_core::release::VERSION))
            .timeout_connect(connect)
            .timeout_read(read)
            .timeout(total)
            // Redirects are followed by hand, to addresses the same rules allow.
            .redirects(0)
            .build();
        Self {
            source,
            agent,
            max_bytes,
        }
    }

    /// The same fetcher with other timeouts and size cap, for tests.
    #[cfg(test)]
    fn limits(self, timeout: Duration, max_bytes: usize) -> Self {
        Self::over(self.source, timeout, timeout, timeout, max_bytes)
    }

    /// Whether this fetcher downloads at all; when not, [`Self::refusal`] says why.
    pub fn downloads(&self) -> bool {
        self.source.is_ok()
    }

    /// Why nothing is downloaded, when so.
    pub fn refusal(&self) -> Option<&str> {
        self.source.as_ref().err().map(String::as_str)
    }

    fn get(&self, template: &str, id: TileId) -> Result<Vec<u8>, FetchError> {
        let mut url = tile_url(template, id);
        for _ in 0..=MAX_REDIRECTS {
            let response = match self
                .agent
                .get(&url)
                .set("Accept", "image/png, */*;q=0.5")
                .call()
            {
                Ok(response) => response,
                Err(ureq::Error::Status(code, _)) => return Err(status_error(code)),
                Err(ureq::Error::Transport(t)) => return Err(transport_error(&t)),
            };
            match response.status() {
                200 => return self.body(response),
                301 | 302 | 303 | 307 | 308 => {
                    let next = response
                        .header("Location")
                        .ok_or_else(|| FetchError::Other("a redirect without a Location".into()))?;
                    url = resolve(&url, next)?;
                    allowed(&url).map_err(|why| {
                        FetchError::Other(format!(
                            "the server redirected to an address not allowed: {why}"
                        ))
                    })?;
                }
                other => return Err(status_error(other)),
            }
        }
        Err(FetchError::Other("too many redirects".into()))
    }

    fn body(&self, response: ureq::Response) -> Result<Vec<u8>, FetchError> {
        let too_big = || {
            FetchError::Other(format!(
                "the tile is larger than {} bytes, which is not a tile",
                self.max_bytes
            ))
        };
        if response
            .header("Content-Length")
            .and_then(|n| n.trim().parse::<usize>().ok())
            .is_some_and(|n| n > self.max_bytes)
        {
            return Err(too_big());
        }
        let mut bytes = Vec::new();
        response
            .into_reader()
            .take(self.max_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| FetchError::Temporary(format!("the download broke off: {e}")))?;
        if bytes.len() > self.max_bytes {
            return Err(too_big());
        }
        Ok(bytes)
    }
}

impl TileFetcher for HttpTileFetcher {
    fn fetch(&self, id: TileId) -> Result<Vec<u8>, FetchError> {
        match &self.source {
            Ok(template) => self.get(template, id),
            Err(why) => Err(FetchError::Other(why.clone())),
        }
    }
}

/// An HTTP status that is not success, as a fetch error: a missing tile, a busy or broken server, or a
/// refusal.
fn status_error(code: u16) -> FetchError {
    match code {
        404 | 410 => FetchError::NotFound,
        408 | 425 | 429 | 500..=599 => {
            FetchError::Temporary(format!("the server answered HTTP {code}"))
        }
        _ => FetchError::Other(format!("the server answered HTTP {code}")),
    }
}

fn transport_error(error: &ureq::Transport) -> FetchError {
    use ureq::ErrorKind;
    match error.kind() {
        ErrorKind::InvalidUrl
        | ErrorKind::UnknownScheme
        | ErrorKind::InsecureRequestHttpsOnly
        | ErrorKind::TooManyRedirects
        | ErrorKind::BadStatus
        | ErrorKind::BadHeader
        | ErrorKind::ProxyConnect
        | ErrorKind::ProxyUnauthorized
        | ErrorKind::InvalidProxyUrl => FetchError::Other(error.to_string()),
        // Timeouts, no connection, a name that does not resolve, a dropped link.
        _ => FetchError::Temporary(error.to_string()),
    }
}

/// `location` (a redirect's target) as an absolute URL, relative to `base`.
fn resolve(base: &str, location: &str) -> Result<String, FetchError> {
    if location.contains("://") {
        return Ok(location.to_owned());
    }
    let (scheme, rest) = base
        .split_once("://")
        .ok_or_else(|| FetchError::Other("a redirect from a malformed address".into()))?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    match location.strip_prefix('/') {
        Some(path) if !path.starts_with('/') => Ok(format!("{scheme}://{authority}/{path}")),
        _ => Err(FetchError::Other(format!(
            "a redirect to {location:?}, which is not an address"
        ))),
    }
}

/// The subfolder of the tile cache for a tile address: a hash of it, so tiles of two servers never mix. A
/// world with no address reads the `none` folder, which holds only what was put there by hand.
pub fn address_key(template: &str) -> String {
    let template = template.trim();
    if template.is_empty() {
        return "none".into();
    }
    let digest = Sha256::digest(template.as_bytes());
    digest[..10].iter().map(|b| format!("{b:02x}")).collect()
}

/// The game-wide folder of the tile cache: [`CACHE_DIR`] in the game folder (the working directory).
pub fn cache_root() -> PathBuf {
    std::env::current_dir().unwrap_or_default().join(CACHE_DIR)
}

/// What an Earth world reads tiles with.
pub struct EarthTiles {
    /// The cache and the fetcher behind it.
    pub store: Arc<TileStore>,
    /// Whether tiles missing from the cache are downloaded.
    pub downloads: bool,
    /// Where this world's tiles are cached.
    pub dir: PathBuf,
}

/// The tile store of a world with `settings`, caching under `root` (see [`cache_root`]). Says in the log
/// whether tiles are downloaded.
pub fn open(settings: &EarthSettings, root: &Path) -> EarthTiles {
    let fetcher = HttpTileFetcher::new(settings);
    let downloads = fetcher.downloads();
    match fetcher.refusal() {
        None => log::info!("Horizon: Earth tiles are downloaded as needed"),
        Some(why) if settings.allow_downloads => {
            log::warn!("Horizon: Earth tiles are not downloaded: {why}")
        }
        Some(why) => log::info!("Horizon: Earth runs on cached tiles only: {why}"),
    }
    let dir = root.join(address_key(&settings.tile_url));
    EarthTiles {
        store: Arc::new(TileStore::new(&dir, Arc::new(fetcher), MEMORY_TILES)),
        downloads,
        dir,
    }
}

#[cfg(test)]
mod tests;

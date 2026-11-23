//! The tile servers a player has approved for Earth worlds: a small file of its own in the player's folder
//! (`horizon-earth-servers.txt`, next to `servers.dat` and `horizon-player.txt`, so `--player-dir` keeps it
//! per player), one host a line.
//!
//! Download consent belongs to the player, not to the world: a world file can say `allowDownloads: true` and
//! name any address, but the game contacts a host only when it is in this list too (`earth_net`). A host is
//! added when the player turns on "Allow downloads" for its address on Earth's settings page (Create World) or
//! from the world's Edit screen, and removed the same way. A world from someone else therefore runs on the
//! tiles already on this computer until the player says otherwise.
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The file's name in the player's folder.
pub const FILE: &str = "horizon-earth-servers.txt";
const HEADER: &str = "# Servers this player lets Earth worlds download elevation tiles and land cover from, one host a line.\n\
# A world downloads only when it allows downloads and its server's host is listed here.\n\
# Lines starting with # are ignored. Delete a line to take the approval back.\n";

/// The hosts a player approved, and the file they are kept in.
#[derive(Clone, Debug, Default)]
pub struct Approved {
    hosts: BTreeSet<String>,
    /// `None`: a list that is not kept (tests).
    path: Option<PathBuf>,
}

/// `host` as the list keeps it: lower case, a name, an IPv4 address or a bracketed IPv6 one, nothing else.
pub fn normalize(host: &str) -> Option<String> {
    let host = host.trim().to_ascii_lowercase();
    let plain = !host.is_empty()
        && host
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-');
    let v6 = host.len() > 2
        && host.starts_with('[')
        && host.ends_with(']')
        && host[1..host.len() - 1]
            .bytes()
            .all(|b| b.is_ascii_hexdigit() || b == b':' || b == b'.');
    (plain || v6).then_some(host)
}

fn read(path: &Path) -> BTreeSet<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .filter_map(normalize)
        .collect()
}

impl Approved {
    /// The list kept in `dir`; empty when there is no file.
    pub fn load(dir: &Path) -> Self {
        let path = dir.join(FILE);
        Self {
            hosts: read(&path),
            path: Some(path),
        }
    }

    /// This player's list ([`crate::logging::player_dir`]).
    pub fn for_player() -> Self {
        Self::load(&crate::logging::player_dir())
    }

    /// A list of `hosts` that is never written, for tests.
    #[cfg(test)]
    pub fn of(hosts: &[&str]) -> Self {
        Self {
            hosts: hosts.iter().filter_map(|h| normalize(h)).collect(),
            path: None,
        }
    }

    pub fn contains(&self, host: &str) -> bool {
        normalize(host).is_some_and(|h| self.hosts.contains(&h))
    }

    pub fn hosts(&self) -> impl Iterator<Item = &str> {
        self.hosts.iter().map(String::as_str)
    }

    /// Approves `host` and keeps the list. Another game's additions to the file since it was read stay.
    pub fn approve(&mut self, host: &str) -> std::io::Result<()> {
        let host = normalize(host).ok_or_else(|| invalid(host))?;
        self.refresh();
        self.hosts.insert(host);
        self.save()
    }

    /// Takes the approval of `host` back and keeps the list.
    pub fn revoke(&mut self, host: &str) -> std::io::Result<()> {
        let host = normalize(host).ok_or_else(|| invalid(host))?;
        self.refresh();
        self.hosts.remove(&host);
        self.save()
    }

    fn refresh(&mut self) {
        if let Some(path) = &self.path {
            self.hosts.extend(read(path));
        }
    }

    fn save(&self) -> std::io::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut text = String::from(HEADER);
        for host in &self.hosts {
            text.push_str(host);
            text.push('\n');
        }
        // Written beside the file and moved over it, so a crash never leaves half a list.
        let temp = path.with_extension("tmp");
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        std::fs::rename(&temp, path)
    }
}

fn invalid(host: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!("{host:?} is not a host name"),
    )
}

/// The world in `world_root` wants downloads from this host and the player has not approved it: the host to
/// tell them about. `None` for other world types, a world that allows no downloads, an address the game would
/// not use anyway, and an approved host.
pub fn pending_host(world_root: &Path, approved: &Approved) -> Option<String> {
    let (kind, json) = crate::cubic_settings::load_world_generation(world_root)?;
    if kind != crate::cubic_settings::CubicKind::Earth {
        return None;
    }
    let settings = horizon_world::cubic::earth::EarthSettings::from_json(&json).ok()?;
    pending_host_of(&settings, approved)
}

/// [`pending_host`] for settings already read.
pub fn pending_host_of(
    settings: &horizon_world::cubic::earth::EarthSettings,
    approved: &Approved,
) -> Option<String> {
    if !settings.allow_downloads {
        return None;
    }
    let url = settings.tile_url.trim();
    crate::earth_net::check_template(url).ok()?;
    let host = crate::earth_net::host_of(url).ok()?;
    (!approved.contains(&host)).then_some(host)
}

/// The land cover server of a world with `settings` that wants land cover and downloads, whose host the player
/// has not approved: the host to tell them about. `address` is the land cover folder the build or the player
/// gives (`earth_landnet::address`). The land cover server is a host of its own: approving the tile server does
/// not approve it.
pub fn pending_land_host_of(
    settings: &horizon_world::cubic::earth::EarthSettings,
    address: Option<&str>,
    approved: &Approved,
) -> Option<String> {
    let host = crate::earth_landnet::host_for(settings, address)?;
    (!approved.contains(&host)).then_some(host)
}

/// Every host a world with `settings` wants to download from: the tile server's, then the land cover server's
/// (when the world uses land cover and the game has an address for it). Hosts the game would not use are left
/// out.
pub fn wanted_hosts(
    settings: &horizon_world::cubic::earth::EarthSettings,
    land_address: Option<&str>,
) -> Vec<String> {
    let mut hosts = Vec::new();
    let url = settings.tile_url.trim();
    if settings.allow_downloads
        && crate::earth_net::check_template(url).is_ok()
        && let Ok(host) = crate::earth_net::host_of(url)
    {
        hosts.push(host);
    }
    if let Some(host) = crate::earth_landnet::host_for(settings, land_address)
        && !hosts.contains(&host)
    {
        hosts.push(host);
    }
    hosts
}

/// Hosts the player was told about this run, and the one not yet told.
static NOTICE: std::sync::Mutex<(Vec<String>, Option<String>)> =
    std::sync::Mutex::new((Vec::new(), None));

/// A world was opened that wants downloads from the unapproved `host`: the game says so in chat, once per host
/// a run ([`take_notice`]).
pub fn note_unapproved(host: &str) {
    let mut notice = NOTICE.lock().unwrap_or_else(|e| e.into_inner());
    if !notice.0.iter().any(|h| h == host) {
        notice.0.push(host.to_owned());
        notice.1 = Some(host.to_owned());
    }
}

/// The host a world wanted that the player has not been told about yet.
pub fn take_notice() -> Option<String> {
    NOTICE.lock().unwrap_or_else(|e| e.into_inner()).1.take()
}

#[cfg(test)]
mod tests;

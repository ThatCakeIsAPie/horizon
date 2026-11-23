//! Edit World's button for an Earth world that wants to download tiles: it names the tile server's host and
//! approves it for this player (or takes the approval back) with one press. The world file's own
//! `allowDownloads` is not consent (`earth_consent`); this is the way to give it for a world that is already made.
use crate::earth_consent::Approved;
use crate::{cubic_settings, lang};
use std::path::Path;

/// The hosts the Earth world in `root` wants to download from: its tile server (when it wants downloads and names
/// a usable address) and, when it uses land cover, the land cover server.
fn wanted_hosts(root: &Path) -> Vec<String> {
    let Some((kind, json)) = cubic_settings::load_world_generation(root) else {
        return Vec::new();
    };
    if kind != cubic_settings::CubicKind::Earth {
        return Vec::new();
    }
    let Ok(settings) = horizon_world::cubic::earth::EarthSettings::from_json(&json) else {
        return Vec::new();
    };
    crate::earth_consent::wanted_hosts(&settings, crate::earth_landnet::address().as_deref())
}

fn text(key: &str, fallback: &str, hosts: &[String]) -> String {
    hosts.iter().fold(lang::get_or(key, fallback), |text, host| {
        text.replacen("%s", host, 1)
    })
}

/// The label and whether it is enabled of the world's button in the slot of Convert to Cubic Chunks, which an
/// Earth world (already cubic) has no use for: Convert's own for any other world.
pub(super) fn convert_button(root: &Path, locked: bool) -> (String, bool) {
    let hosts = wanted_hosts(root);
    if !hosts.is_empty() {
        let approved = Approved::for_player();
        let all = hosts.iter().all(|h| approved.contains(h));
        let key = match (hosts.len() > 1, all) {
            (false, true) => (
                "horizon.cubic.earth.edit.revoke",
                "Tiles allowed: %s (stop)",
            ),
            (false, false) => ("horizon.cubic.earth.edit.approve", "Allow tiles from %s"),
            (true, true) => (
                "horizon.cubic.earth.edit.revoke.land",
                "Downloads allowed: %s, %s (stop)",
            ),
            (true, false) => (
                "horizon.cubic.earth.edit.approve.land",
                "Allow downloads from %s and %s",
            ),
        };
        return (text(key.0, key.1, &hosts), true);
    }
    (
        lang::get_or("horizon.cubic.convert.button", "Convert to Cubic Chunks..."),
        !locked && !crate::cubic::is_cubic_world(root),
    )
}

/// The button was pressed: approves the world's servers (all or none: they are one consent to the world's
/// downloads), or takes the approval back. Whether it was an Earth world's (else the press is Convert's).
pub(super) fn press(root: &Path) -> bool {
    let hosts = wanted_hosts(root);
    if hosts.is_empty() {
        return false;
    }
    let mut approved = Approved::for_player();
    let revoke = hosts.iter().all(|h| approved.contains(h));
    for host in &hosts {
        let done = if revoke {
            approved.revoke(host)
        } else {
            approved.approve(host)
        };
        if let Err(e) = done {
            log::warn!("Horizon: could not keep the approval list: {e}");
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_world_with_land_cover_names_the_land_cover_host_too_when_the_game_has_one() {
        let dir = horizon_earth::testing::TempDir::new();
        let root = dir.path().join("w");
        world(&root, "earth", true, "https://tiles.example.com/{z}/{x}/{y}.png");
        let mut settings = horizon_world::cubic::earth::EarthSettings::at(1.0, 2.0);
        settings.tile_url = "https://tiles.example.com/{z}/{x}/{y}.png".into();
        settings.allow_downloads = true;
        settings.land_cover = true;
        let hosts = |s: &horizon_world::cubic::earth::EarthSettings, a: Option<&str>| {
            crate::earth_consent::wanted_hosts(s, a)
        };
        assert_eq!(
            hosts(&settings, Some("https://land.example.org/")),
            vec!["tiles.example.com", "land.example.org"]
        );
        // No address in this build, land cover off, or downloads off: only what the world can really use.
        assert_eq!(hosts(&settings, None), vec!["tiles.example.com"]);
        settings.land_cover = false;
        assert_eq!(
            hosts(&settings, Some("https://land.example.org/")),
            vec!["tiles.example.com"]
        );
        settings.land_cover = true;
        settings.allow_downloads = false;
        assert!(hosts(&settings, Some("https://land.example.org/")).is_empty());
    }

    fn world(dir: &Path, kind: &str, allow: bool, url: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("horizon_cubic.json"),
            format!(r#"{{"cubic":true,"type":"{kind}"}}"#),
        )
        .unwrap();
        std::fs::write(
            dir.join("cubicgen.json"),
            format!(
                r#"{{"originLatitude":1,"originLongitude":2,"tileUrl":"{url}","allowDownloads":{allow}}}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn only_an_earth_world_that_wants_downloads_offers_its_host() {
        let dir = horizon_earth::testing::TempDir::new();
        let url = "https://Tiles.Example.com/{z}/{x}/{y}.png";
        let at = |name: &str| dir.path().join(name);
        world(&at("wants"), "earth", true, url);
        world(&at("off"), "earth", false, url);
        world(&at("custom"), "custom", true, url);
        world(
            &at("insecure"),
            "earth",
            true,
            "http://example.com/{z}/{x}/{y}",
        );
        assert_eq!(wanted_hosts(&at("wants")), vec!["tiles.example.com"]);
        for other in ["off", "custom", "insecure", "missing"] {
            assert!(wanted_hosts(&at(other)).is_empty(), "{other}");
        }
        // Another world type's press is the Convert button's.
        assert!(!press(&at("custom")));
    }
}

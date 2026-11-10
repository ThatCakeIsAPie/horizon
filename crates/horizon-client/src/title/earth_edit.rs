//! Edit World's button for an Earth world that wants to download tiles: it names the tile server's host and
//! approves it for this player (or takes the approval back) with one press. The world file's own
//! `allowDownloads` is not consent (`earth_consent`); this is the way to give it for a world that is already made.
use crate::earth_consent::Approved;
use crate::{cubic_settings, earth_net, lang};
use std::path::Path;

/// The tile server host of the Earth world in `root`, when the world wants downloads and names a usable address.
fn wanted_host(root: &Path) -> Option<String> {
    let (kind, json) = cubic_settings::load_world_generation(root)?;
    if kind != cubic_settings::CubicKind::Earth {
        return None;
    }
    let settings = horizon_world::cubic::earth::EarthSettings::from_json(&json).ok()?;
    let url = settings.tile_url.trim();
    (settings.allow_downloads && earth_net::check_address(url).is_ok())
        .then(|| earth_net::host_of(url).ok())?
}

fn text(key: &str, fallback: &str, host: &str) -> String {
    lang::get_or(key, fallback).replacen("%s", host, 1)
}

/// The label and whether it is enabled of the world's button in the slot of Convert to Cubic Chunks, which an
/// Earth world (already cubic) has no use for: Convert's own for any other world.
pub(super) fn convert_button(root: &Path, locked: bool) -> (String, bool) {
    if let Some(host) = wanted_host(root) {
        let key = if Approved::for_player().contains(&host) {
            (
                "horizon.cubic.earth.edit.revoke",
                "Tiles allowed: %s (stop)",
            )
        } else {
            ("horizon.cubic.earth.edit.approve", "Allow tiles from %s")
        };
        return (text(key.0, key.1, &host), true);
    }
    (
        lang::get_or("horizon.cubic.convert.button", "Convert to Cubic Chunks..."),
        !locked && !crate::cubic::is_cubic_world(root),
    )
}

/// The button was pressed: approves the world's tile server, or takes the approval back. Whether it was an Earth
/// world's (else the press is Convert's).
pub(super) fn press(root: &Path) -> bool {
    let Some(host) = wanted_host(root) else {
        return false;
    };
    let mut approved = Approved::for_player();
    let done = if approved.contains(&host) {
        approved.revoke(&host)
    } else {
        approved.approve(&host)
    };
    if let Err(e) = done {
        log::warn!("Horizon: could not keep the approval list: {e}");
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            wanted_host(&at("wants")).as_deref(),
            Some("tiles.example.com")
        );
        for other in ["off", "custom", "insecure", "missing"] {
            assert_eq!(wanted_host(&at(other)), None, "{other}");
        }
        // Another world type's press is the Convert button's.
        assert!(!press(&at("custom")));
    }
}

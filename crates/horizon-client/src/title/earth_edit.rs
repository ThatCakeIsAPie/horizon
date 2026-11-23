//! Edit World on an Earth world: when the world downloads from hosts the player has not approved, Edit World asks
//! the consent question first (`Mode::EarthConsent`); this names the hosts. There is no approve/stop button: the
//! world file's own `allowDownloads` is not consent (`earth_consent`), and Yes is.
use crate::earth_consent::Approved;
use crate::{cubic_settings, lang};
use std::path::Path;

/// The hosts the Earth world in `root` wants to download from: its tile server (when it wants downloads and names
/// a usable address) and, when it uses land cover, the land cover server. A world keeps what its file says: an old
/// world with no land cover does not get it now, because switching its terrain mid-world would leave seams.
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

/// The hosts the Earth world in `root` wants that the player has not approved: what the consent question names.
/// Empty for another kind of world.
pub(super) fn pending_hosts(root: &Path, approved: &Approved) -> Vec<String> {
    let mut hosts = wanted_hosts(root);
    hosts.retain(|h| !approved.contains(h));
    hosts
}

/// The label and whether it is enabled of the world's button in the slot of Convert to Cubic Chunks, which an
/// Earth world (already cubic) has no use for: it stays off there like for any cubic world.
pub(super) fn convert_button(root: &Path, locked: bool) -> (String, bool) {
    (
        lang::get_or("horizon.cubic.convert.button", "Convert to Cubic Chunks..."),
        !locked && !crate::cubic::is_cubic_world(root),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_world_with_land_cover_names_the_land_cover_host_too_when_the_game_has_one() {
        let dir = horizon_earth::testing::TempDir::new();
        let root = dir.path().join("w");
        world(
            &root,
            "earth",
            true,
            "https://tiles.example.com/{z}/{x}/{y}.png",
        );
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
        // A world kept from before, and not approved: the question names its host; approved: none.
        let none = Approved::of(&[]);
        assert_eq!(
            pending_hosts(&at("wants"), &none),
            vec!["tiles.example.com"]
        );
        assert!(pending_hosts(&at("wants"), &Approved::of(&["tiles.example.com"])).is_empty());
        assert!(pending_hosts(&at("custom"), &none).is_empty());
    }
}

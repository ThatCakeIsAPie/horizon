//! The settings of an Earth world: where on the globe world block (0, 0) sits, the scale, and what to fetch.
//!
//! The JSON uses camelCase names like the Custom Cubic settings; unknown fields are ignored so a newer file
//! still loads. `originLatitude` and `originLongitude` have no default: a world has to say where it is.

use horizon_earth::projection::MAX_LATITUDE;
use serde::{Deserialize, Serialize};

use crate::cubic::{MAX_Y, MIN_Y};

/// The settings of an Earth world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EarthSettings {
    /// Degrees north of the equator at world block (0, 0).
    pub origin_latitude: f64,
    /// Degrees east of Greenwich at world block (0, 0).
    pub origin_longitude: f64,
    /// Metres of ground that one block spans, in width and (before the exaggeration) in height.
    #[serde(default = "default_metres_per_block")]
    pub metres_per_block: f64,
    /// The block height of sea level: 0 m of elevation, and the surface of the water.
    #[serde(default)]
    pub sea_level_y: i32,
    /// A multiplier on every height, to make relief taller than life.
    #[serde(default = "default_exaggeration")]
    pub vertical_exaggeration: f64,
    /// The finest tile zoom to read, 0 to 15 (15 is about 4.8 m per pixel at the equator).
    #[serde(default = "default_max_zoom")]
    pub max_zoom: u8,
    /// Where the game client fetches tiles from, with `{z}`, `{x}` and `{y}` for the tile. This crate only
    /// stores it; it has no default, so the client has to be given one.
    #[serde(default)]
    pub tile_url: String,
    /// Whether the client may download tiles it does not have.
    #[serde(default)]
    pub allow_downloads: bool,
    /// Whether structures (villages and the like) generate.
    #[serde(default)]
    pub structures: bool,
    /// Whether the surface and the biomes follow land cover (ESA WorldCover classes) where the generator is
    /// given a source of it ([`crate::cubic::earth::EarthGenerator::with_land_cover`]). Off by default, and
    /// then, or with no source, the terrain is the elevation-only terrain exactly. The file leaves the name out
    /// while it is off, so a world made without land cover keeps the text (and the cache key) it always had.
    #[serde(default, skip_serializing_if = "is_false")]
    pub land_cover: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn default_metres_per_block() -> f64 {
    1.0
}

fn default_exaggeration() -> f64 {
    1.0
}

fn default_max_zoom() -> u8 {
    15
}

impl EarthSettings {
    /// The settings of a world at `origin_latitude`, `origin_longitude`, with every other value its default.
    pub fn at(origin_latitude: f64, origin_longitude: f64) -> Self {
        Self {
            origin_latitude,
            origin_longitude,
            metres_per_block: default_metres_per_block(),
            sea_level_y: 0,
            vertical_exaggeration: default_exaggeration(),
            max_zoom: default_max_zoom(),
            tile_url: String::new(),
            allow_downloads: false,
            structures: false,
            land_cover: false,
        }
    }

    /// Checks the numbers a generator cannot work with, naming the setting that is wrong.
    pub fn validate(&self) -> Result<(), String> {
        let lat = self.origin_latitude;
        if !(lat.is_finite() && lat.abs() <= MAX_LATITUDE) {
            return Err(format!(
                "originLatitude must be from -{MAX_LATITUDE:.4} to {MAX_LATITUDE:.4} degrees (the limit of web maps), not {lat}"
            ));
        }
        let lon = self.origin_longitude;
        if !(lon.is_finite() && (-180.0..=180.0).contains(&lon)) {
            return Err(format!(
                "originLongitude must be from -180 to 180 degrees, not {lon}"
            ));
        }
        let scale = self.metres_per_block;
        if !(scale.is_finite() && scale > 0.0) {
            return Err(format!(
                "metresPerBlock must be a number above 0, not {scale}"
            ));
        }
        if !(MIN_Y..=MAX_Y).contains(&self.sea_level_y) {
            return Err(format!(
                "seaLevelY must be from {MIN_Y} to {MAX_Y}, not {}",
                self.sea_level_y
            ));
        }
        let exaggeration = self.vertical_exaggeration;
        if !(exaggeration.is_finite() && exaggeration > 0.0) {
            return Err(format!(
                "verticalExaggeration must be a number above 0, not {exaggeration}"
            ));
        }
        if self.max_zoom > horizon_earth::tile::MAX_ZOOM {
            return Err(format!(
                "maxZoom must be from 0 to {}, not {}",
                horizon_earth::tile::MAX_ZOOM,
                self.max_zoom
            ));
        }
        Ok(())
    }

    /// The elevation field these settings describe over `store`: the projection from world blocks to the
    /// globe, the height map and the finest zoom. The generator reads its terrain through it, and so does
    /// whoever needs a surface height before a generator exists.
    pub fn elevation_field(
        &self,
        store: std::sync::Arc<horizon_earth::TileStore>,
    ) -> Result<horizon_earth::ElevationField, String> {
        self.validate()?;
        let projection = horizon_earth::Projection::local(
            self.origin_latitude,
            self.origin_longitude,
            self.metres_per_block,
        )
        .map_err(|e| e.to_string())?;
        let vertical = horizon_earth::VerticalMap::new(self.sea_level_y, self.metres_per_block)
            .with_exaggeration(self.vertical_exaggeration);
        Ok(
            horizon_earth::ElevationField::new(store, projection, vertical)
                .with_max_zoom(self.max_zoom),
        )
    }

    /// The land cover these settings describe over `store`, filled by `fetcher`: the same projection as
    /// [`EarthSettings::elevation_field`], so a block has the class of the ground its height is read from.
    pub fn land_cover_source(
        &self,
        store: std::sync::Arc<horizon_earth::LandCoverStore>,
        fetcher: std::sync::Arc<dyn horizon_earth::LandCoverFetcher>,
    ) -> Result<std::sync::Arc<horizon_earth::LandCoverSource>, String> {
        self.validate()?;
        let projection = horizon_earth::Projection::local(
            self.origin_latitude,
            self.origin_longitude,
            self.metres_per_block,
        )
        .map_err(|e| e.to_string())?;
        Ok(std::sync::Arc::new(horizon_earth::LandCoverSource::new(
            horizon_earth::LandCoverField::new(store, projection),
            fetcher,
        )))
    }

    /// Parses and validates the JSON of a world's Earth settings. Unknown fields are ignored; a missing origin
    /// is an error.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let settings: Self =
            serde_json::from_str(text).map_err(|e| format!("invalid Earth settings: {e}"))?;
        settings.validate()?;
        Ok(settings)
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub fn to_json_compact(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// A text that changes with whatever decides what the terrain is made of: these settings and the version of
    /// the surface rule ([`super::surface::RULES`]) and the elevation cleaning rule
    /// ([`horizon_earth::clean::CLEAN_RULES`]). Far-view caches are keyed by it, so nodes built under an
    /// older rule are not shown. With land cover on it also carries the version of the land-cover data rules
    /// ([`horizon_earth::LAND_COVER_RULES`]) and of the land-cover surface and biome rules
    /// ([`super::surface::LAND_RULES`]), so a world made with land cover never shares a cache with one made
    /// without; with it off the text is what it was before land cover existed.
    pub fn terrain_key(&self) -> String {
        let mut key = format!(
            "{}#surface{}#clean{}",
            self.to_json_compact(),
            super::surface::RULES,
            horizon_earth::clean::CLEAN_RULES
        );
        if self.land_cover {
            key.push_str(&format!(
                "#landcover{}.{}",
                horizon_earth::LAND_COVER_RULES,
                super::surface::LAND_RULES
            ));
        }
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_terrain_key_carries_the_elevation_cleaning_version() {
        let key = EarthSettings::at(46.0, 7.5).terrain_key();
        assert!(
            key.ends_with(&format!("#clean{}", horizon_earth::clean::CLEAN_RULES)),
            "{key}"
        );
    }

    #[test]
    fn a_file_with_only_an_origin_gets_every_default() {
        let s = EarthSettings::from_json(r#"{"originLatitude": 46.0, "originLongitude": 7.5}"#)
            .unwrap();
        assert_eq!(s.origin_latitude, 46.0);
        assert_eq!(s.origin_longitude, 7.5);
        assert_eq!(s.metres_per_block, 1.0);
        assert_eq!(s.sea_level_y, 0);
        assert_eq!(s.vertical_exaggeration, 1.0);
        assert_eq!(s.max_zoom, 15);
        assert_eq!(s.tile_url, "");
        assert!(!s.allow_downloads && !s.structures && !s.land_cover);
        assert_eq!(s, EarthSettings::at(46.0, 7.5));
    }

    #[test]
    fn settings_survive_a_json_round_trip_in_camel_case() {
        let s = EarthSettings {
            origin_latitude: -33.8688,
            origin_longitude: 151.2093,
            metres_per_block: 2.5,
            sea_level_y: 63,
            vertical_exaggeration: 1.75,
            max_zoom: 12,
            tile_url: "tiles/{z}/{x}/{y}.png".into(),
            allow_downloads: true,
            structures: true,
            land_cover: true,
        };
        for text in [s.to_json_pretty(), s.to_json_compact()] {
            assert_eq!(EarthSettings::from_json(&text).unwrap(), s);
        }
        let value: serde_json::Value = serde_json::from_str(&s.to_json_compact()).unwrap();
        for key in [
            "originLatitude",
            "originLongitude",
            "metresPerBlock",
            "seaLevelY",
            "verticalExaggeration",
            "maxZoom",
            "tileUrl",
            "allowDownloads",
            "structures",
            "landCover",
        ] {
            assert!(value.get(key).is_some(), "missing {key}");
        }
    }

    #[test]
    fn land_cover_is_off_by_default_and_left_out_of_the_file_while_off() {
        let off = EarthSettings::at(46.0, 7.5);
        assert!(!off.land_cover);
        assert!(!off.to_json_compact().contains("landCover"));
        let on = EarthSettings::from_json(
            r#"{"originLatitude": 46.0, "originLongitude": 7.5, "landCover": true}"#,
        )
        .unwrap();
        assert!(on.land_cover);
        assert!(on.to_json_compact().contains("\"landCover\":true"));
        assert!(
            EarthSettings::from_json(
                r#"{"originLatitude": 46.0, "originLongitude": 7.5, "landCover": "yes"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn the_terrain_key_changes_with_land_cover_and_carries_its_rules_only_then() {
        let off = EarthSettings::at(46.0, 7.5);
        let mut on = off.clone();
        on.land_cover = true;
        assert_ne!(off.terrain_key(), on.terrain_key());
        let rules = format!(
            "#landcover{}.{}",
            horizon_earth::LAND_COVER_RULES,
            super::super::surface::LAND_RULES
        );
        assert!(on.terrain_key().ends_with(&rules), "{}", on.terrain_key());
        assert!(!off.terrain_key().contains("landcover"));
        // The key of a world without land cover is what it was before the setting existed.
        assert_eq!(
            off.terrain_key(),
            format!(
                "{}#surface{}#clean{}",
                off.to_json_compact(),
                super::super::surface::RULES,
                horizon_earth::clean::CLEAN_RULES
            )
        );
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let s = EarthSettings::from_json(
            r#"{"originLatitude": 1, "originLongitude": 2, "somethingNew": [1, 2], "structures": true}"#,
        )
        .unwrap();
        assert!(s.structures);
        assert_eq!(s.origin_longitude, 2.0);
    }

    #[test]
    fn bad_values_are_rejected_with_the_name_of_the_setting() {
        let with = |field: &str, value: &str| {
            let mut json = serde_json::json!({"originLatitude": 10, "originLongitude": 20});
            json[field] = serde_json::from_str(value).unwrap();
            EarthSettings::from_json(&json.to_string())
        };
        for (field, value) in [
            ("originLatitude", "90"),
            ("originLatitude", "-85.06"),
            ("originLongitude", "181"),
            ("originLongitude", "-200.5"),
            ("metresPerBlock", "0"),
            ("metresPerBlock", "-2"),
            ("verticalExaggeration", "0"),
            ("verticalExaggeration", "-1"),
            ("maxZoom", "16"),
            ("seaLevelY", "2000000000"),
        ] {
            let error = with(field, value).expect_err(&format!("{field}: {value}"));
            assert!(error.contains(field), "{field}: {value} gave {error}");
        }
        // The edge of what web maps show is fine.
        assert!(with("originLatitude", "85.0511").is_ok());
        assert!(with("metresPerBlock", "0.25").is_ok());
    }

    #[test]
    fn a_missing_origin_and_non_numbers_are_errors() {
        assert!(EarthSettings::from_json("{}").is_err());
        assert!(EarthSettings::from_json(r#"{"originLatitude": 1}"#).is_err());
        assert!(
            EarthSettings::from_json(r#"{"originLatitude": "north", "originLongitude": 0}"#)
                .is_err()
        );
        assert!(EarthSettings::from_json("not json").is_err());
        let mut s = EarthSettings::at(0.0, 0.0);
        s.origin_latitude = f64::NAN;
        assert!(s.validate().is_err());
        s = EarthSettings::at(0.0, 0.0);
        s.metres_per_block = f64::INFINITY;
        assert!(s.validate().is_err());
    }
}

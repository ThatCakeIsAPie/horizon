//! Earth's page of the Customize screen: where on the globe world block (0, 0) is, how many metres a block
//! spans, the tile server's address, and whether tiles may be downloaded. Plain text boxes; the other settings
//! (sea level, vertical exaggeration, finest zoom, structures) keep the values they had, and the Presets overlay
//! edits them as JSON.
use super::{Builder, Item, Kind, ROW, State, WIDTH, Widget, toggle, tr};
use crate::cubic_settings;
use horizon_world::cubic::earth::EarthSettings;

/// The box ids the clipboard and the caret know the page's text boxes by.
const FIELD: &str = "cubic.earth";
/// Language keys of the boxes' labels: latitude, longitude, scale, tile address.
const LABELS: [&str; 4] = [
    "horizon.cubic.earth.latitude",
    "horizon.cubic.earth.longitude",
    "horizon.cubic.earth.scale",
    "horizon.cubic.earth.url",
];
/// The most characters each box takes.
const MAX: [usize; 4] = [24, 24, 24, 512];
/// Characters of a warning line that fit the page's column.
const LINE: usize = 54;

/// Earth's settings as the page holds them: the text of each box, and the switch.
#[derive(Clone, Debug)]
pub(super) struct Draft {
    latitude: String,
    longitude: String,
    scale: String,
    url: String,
    allow_downloads: bool,
    /// The settings the page does not show, as they were read.
    rest: EarthSettings,
    /// The box that has the keys.
    pub(super) focus: Option<usize>,
}

impl Default for Draft {
    fn default() -> Self {
        Self::of(&cubic_settings::default_earth())
    }
}

impl Draft {
    fn of(settings: &EarthSettings) -> Self {
        Self {
            latitude: settings.origin_latitude.to_string(),
            longitude: settings.origin_longitude.to_string(),
            scale: settings.metres_per_block.to_string(),
            url: settings.tile_url.clone(),
            allow_downloads: settings.allow_downloads,
            rest: settings.clone(),
            focus: None,
        }
    }

    /// The page for the settings text `text` (empty is the starting place); why it is not Earth's.
    pub(super) fn from_json(text: &str) -> Result<Self, String> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        EarthSettings::from_json(text).map(|settings| Self::of(&settings))
    }

    /// What the boxes say as settings, or what is wrong with them.
    pub(super) fn settings(&self) -> Result<EarthSettings, String> {
        let number = |text: &str, name: &str| {
            text.trim()
                .replace(',', ".")
                .parse::<f64>()
                .map_err(|_| format!("{name} is not a number: {:?}", text.trim()))
        };
        let mut settings = self.rest.clone();
        settings.origin_latitude = number(&self.latitude, "originLatitude")?;
        settings.origin_longitude = number(&self.longitude, "originLongitude")?;
        settings.metres_per_block = number(&self.scale, "metresPerBlock")?;
        settings.tile_url = self.url.trim().to_owned();
        settings.allow_downloads = self.allow_downloads;
        settings.validate()?;
        if !settings.tile_url.is_empty() {
            crate::earth_net::check_template(&settings.tile_url)
                .map_err(|why| format!("tileUrl: {why}"))?;
        }
        Ok(settings)
    }

    /// The settings as JSON, pretty or compact; empty while the boxes hold something refused.
    pub(super) fn json(&self, pretty: bool) -> String {
        self.settings()
            .map(|s| {
                if pretty {
                    s.to_json_pretty()
                } else {
                    s.to_json_compact()
                }
            })
            .unwrap_or_default()
    }

    /// Box `n` as the caret draws it.
    pub(super) fn field(&self, n: usize) -> (crate::clipboard::FieldId, &str, usize) {
        let text = match n {
            0 => &self.latitude,
            1 => &self.longitude,
            2 => &self.scale,
            _ => &self.url,
        };
        ((FIELD, n), text, MAX[n.min(3)])
    }

    /// The box that has the keys, for typing and the clipboard.
    pub(super) fn focused_text(
        &mut self,
    ) -> Option<(crate::clipboard::FieldId, &mut String, usize)> {
        let n = self.focus?;
        let text = match n {
            0 => &mut self.latitude,
            1 => &mut self.longitude,
            2 => &mut self.scale,
            _ => &mut self.url,
        };
        Some(((FIELD, n), text, MAX[n.min(3)]))
    }
}

/// The name of box `n`, for the narrator.
pub(super) fn label(n: usize) -> String {
    tr(LABELS[n.min(3)])
}

fn field(b: &mut Builder, n: usize) {
    b.widgets.push(Widget {
        rect: [0.0, b.y, WIDTH, 20.0],
        item: Item {
            kind: Kind::Field(n),
            label: String::new(),
            enabled: true,
            set: None,
        },
    });
    b.y += ROW;
}

/// `text` in lines of at most `width` characters, broken between words.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        match lines.last_mut() {
            Some(line) if line.chars().count() + 1 + word.chars().count() <= width => {
                line.push(' ');
                line.push_str(word);
            }
            _ => lines.push(word.to_owned()),
        }
    }
    lines
}

pub(super) fn page(b: &mut Builder, state: &State) {
    let earth = &state.earth;
    b.note(tr("horizon.cubic.earth.help.1"));
    b.note(tr("horizon.cubic.earth.help.2"));
    b.heading(tr("horizon.cubic.earth.place"));
    for n in 0..3 {
        b.note(label(n));
        field(b, n);
    }
    b.heading(tr("horizon.cubic.earth.tiles"));
    b.note(label(3));
    field(b, 3);
    b.note(tr("horizon.cubic.earth.url.help"));
    b.space(4.0);
    b.row(vec![toggle(
        "horizon.cubic.earth.downloads",
        earth.allow_downloads,
        |s, on| s.earth.allow_downloads = on,
    )]);
    if earth.allow_downloads && earth.url.trim().is_empty() {
        b.note(tr("horizon.cubic.earth.no_url"));
    }
    if let Err(why) = earth.settings() {
        b.space(4.0);
        b.text(Kind::Warning, tr("horizon.cubic.earth.invalid"));
        for line in wrap(&why, LINE) {
            b.text(Kind::Warning, line);
        }
    }
}

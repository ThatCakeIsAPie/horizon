//! Earth's page of the Customize screen: where on the globe world block (0, 0) is (typed, or one of a few
//! places), how many metres a block spans, the tile server's address, and whether tiles may be downloaded
//! from it. A second view of the same page shows the data's required credits.
//!
//! The other settings (sea level, vertical exaggeration, finest zoom, structures) keep the values they had,
//! and the Presets overlay edits them as JSON.
//!
//! **Consent.** Download consent is the player's, not the world's (`earth_consent`): turning "Allow
//! downloads" on says which host will be contacted, and pressing Done adds that host to the player's approved
//! list. A world file that says `allowDownloads: true` for a host the player has not approved shows the
//! switch off here.
use super::{Builder, Item, Kind, ROW, State, WIDTH, Widget, button, disabled, toggle, tr};
use crate::earth_consent::Approved;
use crate::{cubic_settings, earth_net, text::Text};
use horizon_earth::VerticalMap;
use horizon_world::cubic::earth::EarthSettings;
use horizon_world::cubic::{MAX_Y, MIN_Y};

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
/// Characters of a text line that fit the page's column.
const LINE: usize = 54;

/// Places the page fills the origin with: the language key of the name, degrees north and east (the
/// summit, the village or the landmark itself).
pub(super) const PLACES: [(&str, f64, f64); 8] = [
    ("horizon.cubic.earth.place.everest", 27.9881, 86.9250),
    ("horizon.cubic.earth.place.grand_canyon", 36.0544, -112.1401),
    ("horizon.cubic.earth.place.matterhorn", 45.9763, 7.6586),
    ("horizon.cubic.earth.place.fuji", 35.3606, 138.7274),
    ("horizon.cubic.earth.place.yosemite", 37.7456, -119.5936),
    ("horizon.cubic.earth.place.monaco", 43.7384, 7.4246),
    ("horizon.cubic.earth.place.uluru", -25.3444, 131.0369),
    ("horizon.cubic.earth.place.machu_picchu", -13.1631, -72.5450),
];

/// The terrain data's required attribution, as its publisher words it
/// (github.com/tilezen/joerd, docs/attribution.md, "Required attribution"): shown as it is, in English.
const ATTRIBUTION: &str = include_str!("earth/attribution.txt");

/// The land cover data's required attribution, as ESA words it (esa-worldcover.org/en/data-access): shown as it
/// is, in English.
const LAND_ATTRIBUTION: &str = include_str!("earth/land_attribution.txt");

/// What the page knows of the player and the game, apart from the world.
#[derive(Clone, Debug)]
pub(super) struct Context {
    /// The tile address the box starts with: the game's own offer (`Hooks::earth_tile_address`), if any.
    pub(super) default_url: Option<String>,
    /// The land cover folder the game offers (`earth_landnet::address`), if any.
    pub(super) land_address: Option<String>,
    /// The hosts this player approved.
    pub(super) approved: Approved,
}

impl Context {
    /// This game and this player.
    fn game() -> Self {
        Self {
            default_url: horizon_hooks::engine::earth_tile_address().map(str::to_owned),
            land_address: crate::earth_landnet::address(),
            approved: Approved::for_player(),
        }
    }
}

/// A box's value the settings refuse, as a translatable message.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Problem {
    pub(super) key: &'static str,
    args: Vec<String>,
}

impl Problem {
    fn new(key: &'static str, args: &[&str]) -> Self {
        Self {
            key,
            args: args.iter().map(|a| (*a).to_owned()).collect(),
        }
    }
    /// The message in the player's language.
    pub(super) fn text(&self) -> String {
        let args: Vec<Text> = self.args.iter().cloned().map(Text::literal).collect();
        Text::translate(self.key, &args).plain()
    }
}

/// Earth's settings as the page holds them: the text of each box, and the switch.
#[derive(Clone, Debug)]
pub(super) struct Draft {
    latitude: String,
    longitude: String,
    scale: String,
    url: String,
    /// The switch as the player left it.
    allow_downloads: bool,
    /// The host the player turned the switch on for, on this page.
    consent: Option<String>,
    /// Whether the world uses land cover.
    land_cover: bool,
    /// The settings the page does not show, as they were read.
    rest: EarthSettings,
    ctx: Context,
    /// The box that has the keys.
    pub(super) focus: Option<usize>,
    /// The credits are shown instead of the settings.
    credits: bool,
}

impl Default for Draft {
    fn default() -> Self {
        Self::start(Context::game())
    }
}

impl Draft {
    /// The starting page: the default place, the offered tile address, downloads off.
    fn start(ctx: Context) -> Self {
        let mut draft = Self::of(&cubic_settings::default_earth(), ctx);
        if let Some(url) = draft.ctx.default_url.clone() {
            draft.url = url;
        }
        // A new world uses land cover when this game has somewhere to get it (an existing world keeps what its
        // file says, which is off unless it names `landCover`).
        draft.land_cover = draft.land_host().is_some();
        draft
    }

    fn of(settings: &EarthSettings, ctx: Context) -> Self {
        let mut draft = Self {
            latitude: settings.origin_latitude.to_string(),
            longitude: settings.origin_longitude.to_string(),
            scale: settings.metres_per_block.to_string(),
            url: settings.tile_url.clone(),
            allow_downloads: settings.allow_downloads,
            consent: None,
            land_cover: settings.land_cover,
            rest: settings.clone(),
            ctx,
            focus: None,
            credits: false,
        };
        // A world's own `allowDownloads` is not consent: the switch shows on only for an approved host.
        draft.allow_downloads &= draft
            .host()
            .is_some_and(|h| draft.ctx.approved.contains(&h));
        draft
    }

    /// The page for the settings text `text` (empty is the starting place), keeping this page's context;
    /// why it is not Earth's.
    pub(super) fn reload(&self, text: &str) -> Result<Self, String> {
        let ctx = self.ctx.clone();
        if text.trim().is_empty() {
            return Ok(Self::start(ctx));
        }
        EarthSettings::from_json(text).map(|settings| Self::of(&settings, ctx))
    }

    /// The tile server's host, when the address is one the game may use.
    fn host(&self) -> Option<String> {
        let url = self.url.trim();
        earth_net::check_address(url).ok()?;
        earth_net::host_of(url).ok()
    }

    /// The land cover server's host, when this game has an address for it that it may use.
    fn land_host(&self) -> Option<String> {
        let mut settings = self.rest.clone();
        settings.land_cover = true;
        settings.allow_downloads = true;
        crate::earth_landnet::host_for(&settings, self.ctx.land_address.as_deref())
    }

    /// Whether land cover is on: the switch is, and the game has a source for it.
    fn land_on(&self) -> bool {
        self.land_cover && self.land_host().is_some()
    }

    /// The land cover switch.
    fn toggle_land_cover(&mut self) {
        self.land_cover = !self.land_on() && self.land_host().is_some();
    }

    /// Whether downloads are on: the switch is, for a host the player approved before or just now.
    fn downloads_on(&self) -> bool {
        self.allow_downloads
            && self.host().is_some_and(|h| {
                self.consent.as_ref() == Some(&h) || self.ctx.approved.contains(&h)
            })
    }

    /// The switch: turning it on is the player's consent to downloads from the host the page shows.
    fn toggle_downloads(&mut self) {
        if self.downloads_on() {
            self.allow_downloads = false;
            self.consent = None;
        } else if let Some(host) = self.host() {
            self.allow_downloads = true;
            self.consent = Some(host);
        }
    }

    /// Done: the host the player turned downloads on for joins their approved list. Nothing else is.
    pub(super) fn approve(&mut self) {
        let Some(host) = self.host().filter(|_| self.downloads_on()) else {
            return;
        };
        if let Err(e) = self.ctx.approved.approve(&host) {
            log::warn!("Horizon: could not keep the approval of {host} for Earth tiles: {e}");
        }
        // The land cover server is a host of its own, approved the same way, by the same press of Done.
        if self.land_on()
            && let Some(land) = self.land_host()
            && let Err(e) = self.ctx.approved.approve(&land)
        {
            log::warn!("Horizon: could not keep the approval of {land} for Earth land cover: {e}");
        }
    }

    /// Fills the origin from place `n` of [`PLACES`].
    fn pick(&mut self, n: usize) {
        if let Some(&(_, lat, lon)) = PLACES.get(n) {
            self.latitude = lat.to_string();
            self.longitude = lon.to_string();
        }
    }

    /// What the boxes say as settings, or what is wrong with them.
    fn check(&self) -> Result<EarthSettings, Problem> {
        let number = |text: &str, label: usize| {
            text.trim()
                .replace(',', ".")
                .parse::<f64>()
                .ok()
                .filter(|n| n.is_finite())
                .ok_or_else(|| {
                    Problem::new(
                        "horizon.cubic.earth.err.number",
                        &[&label_of(label), text.trim()],
                    )
                })
        };
        let mut settings = self.rest.clone();
        settings.origin_latitude = number(&self.latitude, 0)?;
        settings.origin_longitude = number(&self.longitude, 1)?;
        settings.metres_per_block = number(&self.scale, 2)?;
        settings.tile_url = self.url.trim().to_owned();
        settings.allow_downloads = self.downloads_on();
        settings.land_cover = self.land_on();
        let limit = horizon_earth::projection::MAX_LATITUDE;
        if settings.origin_latitude.abs() > limit {
            return Err(Problem::new(
                "horizon.cubic.earth.err.latitude",
                &[&format!("{limit:.4}")],
            ));
        }
        if settings.origin_longitude.abs() > 180.0 {
            return Err(Problem::new("horizon.cubic.earth.err.longitude", &[]));
        }
        if settings.metres_per_block <= 0.0 {
            return Err(Problem::new("horizon.cubic.earth.err.scale", &[]));
        }
        // Whatever else the settings refuse (only the Presets overlay can set it), in the generator's words.
        settings
            .validate()
            .map_err(|why| Problem::new("horizon.cubic.earth.err.other", &[&why]))?;
        if !settings.tile_url.is_empty() {
            earth_net::check_address(&settings.tile_url)
                .map_err(|why| Problem::new(why.key(), &[why.placeholder().unwrap_or_default()]))?;
        }
        Ok(settings)
    }

    /// What the boxes say as settings, or what is wrong with them, in the player's language.
    pub(super) fn settings(&self) -> Result<EarthSettings, String> {
        self.check().map_err(|problem| problem.text())
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

    /// Escape: the box that has the keys lets go, else the credits close. Whether either was so.
    pub(super) fn escape(&mut self) -> bool {
        self.focus.take().is_some() || std::mem::take(&mut self.credits)
    }
}

impl State {
    /// Done on Create World's Customize screen: an Earth world's tile server, when the player turned
    /// downloads on for it, becomes an approved host (nothing else does).
    /// The settings text Done hands back, after it.
    pub fn finish(&mut self) -> String {
        if self.kind == cubic_settings::CubicKind::Earth {
            self.earth.approve();
        }
        self.json()
    }
}

/// The name of box `n`, for the narrator.
pub(super) fn label(n: usize) -> String {
    label_of(n)
}

fn label_of(n: usize) -> String {
    tr(LABELS[n.min(3)])
}

/// The page's title.
pub(super) fn title(draft: &Draft) -> String {
    tr(if draft.credits {
        "horizon.cubic.earth.credits.title"
    } else {
        "horizon.cubic.earth.title"
    })
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

/// `n` as a short number for a sentence: `1`, `2.5`, `0.25`.
fn plain(n: f64) -> String {
    let text = format!("{n:.3}");
    text.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// "1 block = N metres", with "1 metre" in the singular: the wording follows the number as the page shows it,
/// so a scale that rounds to 1 reads as 1. A language that has the plural sentence but no singular one keeps
/// the plural, as it did before the singular existed.
pub(super) fn scale_sentence(scale: f64) -> String {
    let shown = plain(scale);
    let one = "horizon.cubic.earth.scale.is.one";
    let key = if shown == "1" && crate::lang::has(one) {
        one
    } else {
        "horizon.cubic.earth.scale.is"
    };
    sentence(key, &[shown])
}

fn sentence(key: &str, args: &[String]) -> String {
    let args: Vec<Text> = args.iter().cloned().map(Text::literal).collect();
    Text::translate(key, &args).plain()
}

pub(super) fn page(b: &mut Builder, state: &State) {
    let earth = &state.earth;
    if earth.credits {
        return credits_page(b);
    }
    b.note(tr("horizon.cubic.earth.help.1"));
    b.note(tr("horizon.cubic.earth.help.2"));
    // What is wrong goes at the top, where Done's being unavailable is explained without scrolling.
    if let Err(problem) = earth.check() {
        b.text(Kind::Warning, tr("horizon.cubic.earth.invalid"));
        for line in wrap(&problem.text(), LINE) {
            b.text(Kind::Warning, line);
        }
    }
    b.heading(tr("horizon.cubic.earth.place"));
    for n in 0..2 {
        b.note(label(n));
        field(b, n);
    }
    b.heading(tr("horizon.cubic.earth.places"));
    for pair in PLACES.chunks(2) {
        let items = pair
            .iter()
            .map(|&(key, ..)| {
                let n = PLACES.iter().position(|p| p.0 == key).unwrap_or_default();
                button(tr(key), move |s| s.earth.pick(n))
            })
            .collect();
        b.row(items);
    }
    b.heading(tr("horizon.cubic.earth.scale.heading"));
    b.note(label(2));
    field(b, 2);
    scale_notes(b, earth);
    b.heading(tr("horizon.cubic.earth.tiles"));
    b.note(label(3));
    field(b, 3);
    b.note(tr("horizon.cubic.earth.url.help"));
    downloads(b, earth);
    land_cover(b, earth);
    b.heading(tr("horizon.cubic.earth.credits.heading"));
    b.note(tr("horizon.cubic.earth.credits.short"));
    b.row(vec![button(tr("horizon.cubic.earth.credits.open"), |s| {
        s.earth.credits = true;
        s.earth.focus = None;
        s.scroll = crate::ui::Scroll::default();
    })]);
}

/// "1 block = N metres", what the whole Earth needs at this scale, and a warning when it does not fit.
fn scale_notes(b: &mut Builder, earth: &Draft) {
    let Some(scale) = earth
        .scale
        .trim()
        .replace(',', ".")
        .parse::<f64>()
        .ok()
        .filter(|n| *n > 0.0 && n.is_finite())
    else {
        return;
    };
    b.note(scale_sentence(scale));
    let vertical = VerticalMap::new(earth.rest.sea_level_y, scale)
        .with_exaggeration(earth.rest.vertical_exaggeration);
    let (low, high) = vertical.required_y_range();
    b.note(sentence(
        "horizon.cubic.earth.scale.needs",
        &[low.to_string(), high.to_string()],
    ));
    if !vertical.fits(MIN_Y, MAX_Y) {
        for line in wrap(
            &sentence(
                "horizon.cubic.earth.scale.too_tall",
                &[MIN_Y.to_string(), MAX_Y.to_string()],
            ),
            LINE,
        ) {
            b.text(Kind::Warning, line);
        }
    }
}

/// The host the page will contact, and the switch.
fn downloads(b: &mut Builder, earth: &Draft) {
    b.space(4.0);
    match earth.host() {
        Some(host) => {
            b.text(
                Kind::Heading,
                sentence("horizon.cubic.earth.server_is", std::slice::from_ref(&host)),
            );
            let key = if earth.downloads_on() {
                "horizon.cubic.earth.downloads.on"
            } else {
                "horizon.cubic.earth.downloads.off"
            };
            for line in wrap(&sentence(key, &[host]), LINE) {
                b.note(line);
            }
        }
        None => b.note(tr("horizon.cubic.earth.no_url")),
    }
    let on = earth.downloads_on();
    b.row(vec![disabled(
        toggle("horizon.cubic.earth.downloads", on, |s, _| {
            s.earth.toggle_downloads();
        }),
        earth.host().is_none(),
    )]);
}

/// The land cover switch, and the second host the page will contact for it.
fn land_cover(b: &mut Builder, earth: &Draft) {
    b.heading(tr("horizon.cubic.earth.land_cover.heading"));
    for line in wrap(&tr("horizon.cubic.earth.land_cover.help"), LINE) {
        b.note(line);
    }
    let host = earth.land_host();
    match &host {
        None => b.note(tr("horizon.cubic.earth.land_cover.no_source")),
        Some(host) if earth.land_on() => {
            let key = if earth.downloads_on() {
                "horizon.cubic.earth.land_cover.on"
            } else {
                "horizon.cubic.earth.land_cover.needs_downloads"
            };
            for line in wrap(&sentence(key, std::slice::from_ref(host)), LINE) {
                b.note(line);
            }
        }
        Some(_) => {}
    }
    b.row(vec![disabled(
        toggle("horizon.cubic.earth.land_cover", earth.land_on(), |s, _| {
            s.earth.toggle_land_cover();
        }),
        host.is_none(),
    )]);
}

/// The credits view: the data source and the attribution it requires.
fn credits_page(b: &mut Builder) {
    b.row(vec![button(tr("horizon.cubic.earth.credits.back"), |s| {
        s.earth.credits = false;
        s.scroll = crate::ui::Scroll::default();
    })]);
    b.heading(tr("horizon.cubic.earth.credits.source"));
    for line in wrap(&tr("horizon.cubic.earth.credits.about"), LINE) {
        b.note(line);
    }
    b.heading(tr("horizon.cubic.earth.credits.required"));
    for entry in ATTRIBUTION.split("\n* ") {
        let entry = entry.trim_start_matches("* ");
        let lines = wrap(entry, LINE - 2);
        for (i, line) in lines.iter().enumerate() {
            b.note(format!("{}{line}", if i == 0 { "* " } else { "  " }));
        }
    }
    b.heading(tr("horizon.cubic.earth.credits.land_source"));
    for line in wrap(&tr("horizon.cubic.earth.credits.land_about"), LINE) {
        b.note(line);
    }
    b.heading(tr("horizon.cubic.earth.credits.land_required"));
    for line in wrap(LAND_ATTRIBUTION.trim(), LINE) {
        b.note(line);
    }
    for line in wrap(&tr("horizon.cubic.earth.credits.land_licence"), LINE) {
        b.note(line);
    }
    b.space(4.0);
    for line in wrap(&tr("horizon.cubic.earth.credits.idea"), LINE) {
        b.note(line);
    }
}

#[cfg(test)]
pub(super) fn attribution() -> &'static str {
    ATTRIBUTION
}

#[cfg(test)]
pub(super) fn land_attribution() -> &'static str {
    LAND_ATTRIBUTION
}

#[cfg(test)]
impl Draft {
    /// A page over `approved` and an offered address, for tests.
    pub(super) fn with(default_url: Option<&str>, approved: Approved) -> Self {
        Self::with_land(default_url, None, approved)
    }
    /// [`Draft::with`] for a game that offers land cover at `land_address`.
    pub(super) fn with_land(
        default_url: Option<&str>,
        land_address: Option<&str>,
        approved: Approved,
    ) -> Self {
        Self::start(Context {
            default_url: default_url.map(str::to_owned),
            land_address: land_address.map(str::to_owned),
            approved,
        })
    }
    pub(super) fn approved(&self) -> &Approved {
        &self.ctx.approved
    }
    pub(super) fn url(&self) -> &str {
        &self.url
    }
    pub(super) fn credits_open(&self) -> bool {
        self.credits
    }
}

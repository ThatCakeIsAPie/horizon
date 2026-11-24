//! The Ores page: what it shows of each world type's settings, and what its
//! controls write.
use super::super::{
    CANCEL, DONE, Kind, NEXT, Overlay, PREV, Page, PageId, Set, buttons, content_height, list_area,
    page as build_page, press, wheel,
};
use super::*;
use crate::cubic_settings::{self, CubicKind};
use crate::lang::{Language, ResourceSource, scoped};
use horizon_world::cubic::custom::settings::UserFunction;
use horizon_world::cubic::earth::EarthSettings;
use std::path::Path;

const SIZE: (f32, f32) = (480.0, 270.0);

/// The vanilla strings the labels use, under Horizon's bundled ones.
struct Strings;
impl ResourceSource for Strings {
    fn read(&self, path: &str) -> Option<Vec<u8>> {
        (path == "assets/minecraft/lang/en_us.json").then(|| {
            br#"{"options.generic_value": "%s: %s", "options.on": "ON", "options.off": "OFF"}"#
                .to_vec()
        })
    }
}
fn english<R>(f: impl FnOnce() -> R) -> R {
    scoped(Language::load(&[&Strings], "en_us"), f).0
}

fn state(kind: CubicKind, json: &str) -> State {
    State::new(Path::new("/nonexistent/horizon-test-resources"), kind, json)
}
fn open(s: &mut State) {
    let at = s
        .pages()
        .iter()
        .position(|p| *p == PageId::Ores)
        .unwrap_or_else(|| panic!("{:?} has an Ores page", s.kind));
    s.switch_page(at);
    assert_eq!(s.page_id(), Some(PageId::Ores));
}
/// The open page's widgets, labelled in English.
fn pg(s: &State) -> Page {
    english(|| build_page(s))
}
fn label_of(s: &State, index: usize) -> String {
    pg(s).widgets[index].item.label.clone()
}
fn enabled_of(s: &State, index: usize) -> bool {
    pg(s).widgets[index].item.enabled
}
/// Presses widget `index` as a click does: a greyed one does nothing.
fn tap(s: &mut State, index: usize) {
    let page = pg(s);
    let widget = &page.widgets[index];
    assert!(widget.item.enabled, "{:?} is greyed", widget.item.label);
    let set: Set = widget.item.set.clone().expect("a control");
    set(s, 0.0);
}
/// Drags slider `index` to `pos` (0 to 1 along its track).
fn drag(s: &mut State, index: usize, pos: f32) {
    let page = pg(s);
    let widget = &page.widgets[index];
    assert!(
        matches!(widget.item.kind, Kind::Slider(_)),
        "{:?} is a slider",
        widget.item.label
    );
    assert!(widget.item.enabled, "{:?} is greyed", widget.item.label);
    let set: Set = widget.item.set.clone().expect("a control");
    set(s, pos);
}
fn at_int(value: i32, min: i32, max: i32) -> f32 {
    Mapping::Int { min, max }.pos(f64::from(value))
}

/// The first widget at or after `from` whose label starts with `prefix`, the
/// `n`th such (from 0), as an index of the page.
fn find(s: &State, from: usize, to: usize, prefix: &str, n: usize) -> usize {
    let page = pg(s);
    let to = to.min(page.widgets.len());
    (from..to)
        .filter(|&i| page.widgets[i].item.label.starts_with(prefix))
        .nth(n)
        .unwrap_or_else(|| {
            let all: Vec<_> = page.widgets.iter().map(|w| w.item.label.clone()).collect();
            panic!("no {n}th {prefix:?} in {from}..{to} of {all:#?}")
        })
}
/// Where ore `i`'s heading is, and where its rows end.
fn ore_span(s: &State, i: usize) -> (usize, usize) {
    let page = pg(s);
    let heading = |n: usize| {
        page.widgets.iter().position(|w| {
            w.item.kind == Kind::Heading && w.item.label.starts_with(&format!("{}. ", n + 1))
        })
    };
    let start = heading(i).unwrap_or_else(|| panic!("an ore {i}"));
    let end = heading(i + 1)
        .or_else(|| meteors_start(s))
        .unwrap_or(page.widgets.len());
    (start, end)
}
fn meteors_start(s: &State) -> Option<usize> {
    pg(s)
        .widgets
        .iter()
        .position(|w| w.item.kind == Kind::Heading && w.item.label == "Meteors")
}
/// The `n`th widget of ore `i` that starts with `prefix`.
fn ore_nth(s: &State, i: usize, prefix: &str, n: usize) -> usize {
    let (a, b) = ore_span(s, i);
    find(s, a, b, prefix, n)
}
fn ore(s: &State, i: usize, prefix: &str) -> usize {
    ore_nth(s, i, prefix, 0)
}
/// The `n`th widget of the meteors section that starts with `prefix`.
fn met_nth(s: &State, prefix: &str, n: usize) -> usize {
    let from = meteors_start(s).expect("the meteors section");
    find(s, from, usize::MAX, prefix, n)
}
fn met(s: &State, prefix: &str) -> usize {
    met_nth(s, prefix, 0)
}
fn press_ore(s: &mut State, i: usize, prefix: &str) {
    let at = ore(s, i, prefix);
    tap(s, at);
}
fn press_first(s: &mut State, prefix: &str) {
    let at = find(s, 0, usize::MAX, prefix, 0);
    tap(s, at);
}
fn ore_count(s: &State) -> usize {
    pg(s)
        .widgets
        .iter()
        .filter(|w| {
            w.item.kind == Kind::Heading
                && w.item
                    .label
                    .split_once(". ")
                    .is_some_and(|(n, _)| n.parse::<usize>().is_ok())
        })
        .count()
}
fn ores_of(s: &State) -> OreGenSettings {
    view(s).expect("the world has ore settings")
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn entry(name: &str) -> OreEntry {
    OreEntry {
        blockstate: BlockStateDesc::new(name),
        spawn_size: 8,
        rich: RichZone {
            min: 0.0,
            max: 64.0,
            intensity: 1.0,
            ..RichZone::default()
        },
        ..OreEntry::default()
    }
}
fn meteors() -> MeteorSettings {
    MeteorSettings {
        enabled: true,
        per_square_km: 4.0,
        min_diameter: 5,
        max_diameter: 25,
        size_curve: UserFunction::new(&[(5.0, 0.6), (7.0, 1.0), (12.0, 0.35), (25.0, 0.03)]),
        size_floor: 0.02,
        surface_share: 0.6,
        sunk_share: 0.25,
        buried_share: 0.15,
        buried_depth: 50.0,
        mix_share: 0.2,
        kinds: vec![MeteorKind::default()],
    }
}
/// Three ores and known meteors.
fn known() -> OreGenSettings {
    OreGenSettings {
        ores: vec![entry("coal_ore"), entry("iron_ore"), entry("gold_ore")],
        meteors: meteors(),
        ..OreGenSettings::default()
    }
}
/// A world of `kind` with these ore settings.
fn with_ores(kind: CubicKind, ores: OreGenSettings) -> State {
    if kind == CubicKind::Earth {
        let mut earth = cubic_settings::default_earth();
        earth.ore_gen = Some(ores);
        state(kind, &earth.to_json_compact())
    } else {
        let mut c = CustomGeneratorSettings::defaults();
        c.ore_gen = Some(ores);
        state(kind, &c.to_json_compact())
    }
}
/// An old preset: the ore lists, no `oreGen`.
fn old_preset() -> CustomGeneratorSettings {
    let mut c = CustomGeneratorSettings::defaults();
    c.ore_gen = None;
    if c.standard_ores.is_empty() {
        c.standard_ores
            .push(horizon_world::cubic::custom::settings::StandardOreConfig::default());
    }
    c
}
fn earth_without_ores() -> State {
    let mut earth = cubic_settings::default_earth();
    earth.ore_gen = None;
    state(CubicKind::Earth, &earth.to_json_compact())
}
fn earth_json_ores(s: &State) -> Option<OreGenSettings> {
    EarthSettings::from_json(&s.json()).unwrap().ore_gen
}

// ---------------------------------------------------------------------------
// Opening and leaving
// ---------------------------------------------------------------------------

#[test]
fn opening_and_leaving_an_old_preset_changes_nothing() {
    let preset = old_preset();
    for kind in [CubicKind::Custom, CubicKind::Hybrid] {
        let mut s = state(kind, &preset.to_json_compact());
        let before = s.json();
        assert!(!before.contains("oreGen"), "an old preset has no oreGen");
        assert_eq!(s.settings.ore_gen, None);
        open(&mut s);
        // It shows the old lists converted, a heading each.
        assert_eq!(
            ore_count(&s),
            preset.standard_ores.len() + preset.periodic_gaussian_ores.len()
        );
        // Building it, scrolling it, and walking over every page and back.
        let _ = build_page(&s);
        wheel(&mut s, -30.0, (240.0, 100.0), SIZE);
        for page in 0..s.page_count() {
            s.switch_page(page);
            let _ = build_page(&s);
        }
        open(&mut s);
        assert_eq!(s.json(), before, "{kind:?}: the JSON is byte-identical");
        assert_eq!(
            s.settings,
            CustomGeneratorSettings::from_json(&before).unwrap()
        );
        // An open and cancelled picker writes nothing either.
        s.open_picker(Target::OreBlock(0));
        press(&mut s, CANCEL);
        s.open_picker(Target::OreBiomes(0));
        press(&mut s, CANCEL);
        assert_eq!(s.json(), before);
    }
}

#[test]
fn an_edit_writes_ore_gen_clears_the_old_lists_and_round_trips() {
    let preset = old_preset();
    let converted = OreGenSettings::from_legacy(
        &preset.standard_ores,
        &preset.periodic_gaussian_ores,
        preset.expected_base_height,
        preset.expected_height_variation,
    );
    assert!(!converted.ores.is_empty());
    let mut s = state(CubicKind::Custom, &preset.to_json_compact());
    open(&mut s);
    let size = ore(&s, 0, "Vein size: ");
    drag(&mut s, size, at_int(12, 1, 64));
    let written = s.settings.ore_gen.clone().expect("oreGen is written");
    assert!(s.settings.standard_ores.is_empty() && s.settings.periodic_gaussian_ores.is_empty());
    assert_eq!(written.ores[0].spawn_size, 12);
    // The rest is the converted lists, untouched.
    assert_eq!(written.ores.len(), converted.ores.len());
    assert_eq!(written.ores[1..], converted.ores[1..]);
    assert_eq!(
        OreEntry {
            spawn_size: converted.ores[0].spawn_size,
            ..written.ores[0].clone()
        },
        converted.ores[0]
    );
    assert!(s.json().contains("\"oreGen\""));
    // Valid, back through the JSON and through a new screen.
    s.settings.validate().unwrap();
    assert_eq!(
        CustomGeneratorSettings::from_json(&s.json()).unwrap(),
        s.settings
    );
    let again = state(CubicKind::Custom, &s.json());
    assert_eq!(again.settings, s.settings);
    assert_eq!(ores_of(&again), written);
}

#[test]
fn a_custom_world_with_ore_gen_shows_it_and_leaves_the_old_lists_alone() {
    let mut s = with_ores(CubicKind::Custom, known());
    let before = s.json();
    open(&mut s);
    assert_eq!(ore_count(&s), 3);
    let _ = build_page(&s);
    assert_eq!(s.json(), before);
    // Editing keeps what the world had and writes only oreGen.
    press_first(&mut s, "Ores: ");
    assert!(!ores_of(&s).enabled);
    assert_eq!(ores_of(&s).ores, known().ores);
    assert_eq!(
        CustomGeneratorSettings::from_json(&s.json()).unwrap(),
        s.settings
    );
}

#[test]
fn hybrid_reaches_the_page_too() {
    let mut s = state(CubicKind::Hybrid, &old_preset().to_json_compact());
    assert!(s.pages().contains(&PageId::Ores));
    open(&mut s);
    let toggle = find(&s, 0, usize::MAX, "Ores: ", 0);
    assert_eq!(label_of(&s, toggle), "Ores: ON");
    tap(&mut s, toggle);
    assert!(!ores_of(&s).enabled);
    assert!(s.json().contains("\"oreGen\""));
    s.settings.validate().unwrap();
    assert_eq!(state(CubicKind::Hybrid, &s.json()).settings, s.settings);
}

// ---------------------------------------------------------------------------
// The list
// ---------------------------------------------------------------------------

#[test]
fn add_duplicate_move_delete_and_reset_edit_the_list() {
    let mut s = with_ores(CubicKind::Custom, known());
    open(&mut s);
    let [coal, iron, gold] = [0, 1, 2].map(|i| ores_of(&s).ores[i].clone());
    // Add ore: a new entry at the end.
    press_first(&mut s, "Add ore");
    let list = ores_of(&s).ores;
    assert_eq!(list.len(), 4);
    assert_eq!(list[3].blockstate.name, "minecraft:iron_ore");
    assert_eq!(ore_count(&s), 4);
    // Up is greyed on the first, Down on the last.
    assert!(!enabled_of(&s, ore(&s, 0, "Up")));
    assert!(!enabled_of(&s, ore(&s, 3, "Down")));
    assert!(enabled_of(&s, ore(&s, 0, "Down")) && enabled_of(&s, ore(&s, 3, "Up")));
    // Down on the first swaps it with the second; Up on that undoes it.
    press_ore(&mut s, 0, "Down");
    assert_eq!(
        ores_of(&s).ores[..3],
        [iron.clone(), coal.clone(), gold.clone()]
    );
    press_ore(&mut s, 1, "Up");
    assert_eq!(
        ores_of(&s).ores[..3],
        [coal.clone(), iron.clone(), gold.clone()]
    );
    // Duplicate puts a copy right after.
    press_ore(&mut s, 1, "Duplicate");
    let list = ores_of(&s).ores;
    assert_eq!(list.len(), 5);
    assert_eq!(list[1..3], [iron.clone(), iron.clone()]);
    // Delete removes that entry.
    press_ore(&mut s, 2, "selectServer.delete");
    assert_eq!(ores_of(&s).ores[..3], [coal, iron, gold]);
    assert_eq!(ores_of(&s).ores.len(), 4);
    // The list round trips, and the meteors went through untouched.
    assert_eq!(
        CustomGeneratorSettings::from_json(&s.json()).unwrap(),
        s.settings
    );
    assert_eq!(ores_of(&s).meteors, meteors());
}

#[test]
fn reset_to_defaults_restores_the_ores_but_not_the_meteors() {
    let mut s = with_ores(CubicKind::Custom, known());
    edit(&mut s, |o| {
        o.enabled = false;
        o.richness_strength = 0.5;
        o.meteors.per_square_km = 9.0;
    });
    open(&mut s);
    press_first(&mut s, "Reset to defaults");
    let defaults = OreGenSettings::custom_defaults();
    let now = ores_of(&s);
    assert_eq!(now.ores, defaults.ores);
    assert_eq!(
        (now.enabled, now.richness_map, now.richness_strength),
        (
            defaults.enabled,
            defaults.richness_map,
            defaults.richness_strength
        )
    );
    assert_eq!(
        now.meteors.per_square_km, 9.0,
        "the meteors have their own reset"
    );
    // On an old preset it replaces the converted lists the same way.
    let mut old = state(CubicKind::Custom, &old_preset().to_json_compact());
    open(&mut old);
    press_first(&mut old, "Reset to defaults");
    assert_eq!(ores_of(&old).ores, defaults.ores);
    assert!(old.settings.standard_ores.is_empty());
}

#[test]
fn the_ore_entry_edits_its_block_size_and_biomes() {
    let mut s = with_ores(CubicKind::Custom, known());
    open(&mut s);
    assert_eq!(label_of(&s, ore(&s, 1, "Block: ")), "Block: iron_ore");
    // The block picker starts on the entry's block and Done writes the choice.
    press_ore(&mut s, 1, "Block: ");
    let Overlay::Picker(p) = &mut s.overlay else {
        panic!("the block picker opens")
    };
    assert_eq!(p.target, Target::OreBlock(1));
    assert_eq!(
        p.selected.iter().collect::<Vec<_>>(),
        ["minecraft:iron_ore"]
    );
    p.selected = ["minecraft:copper_ore".to_owned()].into_iter().collect();
    press(&mut s, DONE);
    assert_eq!(ores_of(&s).ores[1].blockstate.name, "minecraft:copper_ore");
    assert_eq!(ores_of(&s).ores[0].blockstate.name, "minecraft:coal_ore");
    // Vein size is a whole number 1 to 64.
    let size = ore(&s, 1, "Vein size: ");
    drag(&mut s, size, 1.0);
    assert_eq!(ores_of(&s).ores[1].spawn_size, 64);
    drag(&mut s, size, 0.0);
    assert_eq!(ores_of(&s).ores[1].spawn_size, 1);
    // Biomes: every biome shows All; leaving one out restricts the ore.
    assert_eq!(label_of(&s, ore(&s, 1, "Biomes: ")), "Biomes: All");
    press_ore(&mut s, 1, "Biomes: ");
    let Overlay::Picker(p) = &mut s.overlay else {
        panic!("the biome picker opens")
    };
    assert_eq!(p.target, Target::OreBiomes(1));
    assert!(p.multi && p.selected.len() == p.rows.len());
    p.selected.remove("minecraft:plains");
    press(&mut s, DONE);
    let restricted = ores_of(&s).ores[1].biomes.clone().unwrap();
    assert!(!restricted.is_empty() && !restricted.iter().any(|b| b.name() == "minecraft:plains"));
    assert_eq!(
        label_of(&s, ore(&s, 1, "Biomes: ")),
        format!("Biomes: {}", restricted.len())
    );
    assert_eq!(
        ores_of(&s).ores[0].biomes,
        None,
        "the other ores keep every biome"
    );
    // Choosing every biome again lifts the restriction.
    press_ore(&mut s, 1, "Biomes: ");
    press_first_all(&mut s);
    assert_eq!(ores_of(&s).ores[1].biomes, None);
}
/// The open picker's All button, then Done.
fn press_first_all(s: &mut State) {
    press(s, super::super::ALL);
    press(s, DONE);
}

// ---------------------------------------------------------------------------
// Tiers
// ---------------------------------------------------------------------------

#[test]
fn tier_one_bands_have_open_ends_and_keep_min_below_max() {
    let mut s = with_ores(CubicKind::Custom, known());
    open(&mut s);
    // Static height: Y from -512 to 1024; the low end of Min and the high
    // end of Max are open.
    let min = ore(&s, 0, "Min Y: ");
    let max = ore(&s, 0, "Max Y: ");
    assert_eq!(label_of(&s, min), "Min Y: 0");
    drag(&mut s, min, 0.0);
    assert_eq!(ores_of(&s).ores[0].rich.min, f64::NEG_INFINITY);
    assert_eq!(label_of(&s, min), "Min Y: unbounded");
    drag(&mut s, max, 1.0);
    assert_eq!(ores_of(&s).ores[0].rich.max, f64::INFINITY);
    assert_eq!(label_of(&s, max), "Max Y: unbounded");
    drag(&mut s, min, at_int(300, -512, 1024));
    drag(&mut s, max, at_int(100, -512, 1024));
    let r = ores_of(&s).ores[0].rich.clone();
    assert_eq!((r.min, r.max), (300.0, 300.0), "Max stops at Min");
    drag(&mut s, min, at_int(900, -512, 1024));
    assert_eq!(ores_of(&s).ores[0].rich.min, 300.0, "Min stops at Max");
    drag(&mut s, max, at_int(500, -512, 1024));
    drag(&mut s, min, at_int(-200, -512, 1024));
    let r = ores_of(&s).ores[0].rich.clone();
    assert_eq!((r.min, r.max), (-200.0, 500.0));
    assert_eq!(
        ores_of(&s).ores[1].rich.min,
        0.0,
        "the next ore is untouched"
    );
    s.settings.validate().unwrap();
}

#[test]
fn the_mode_cycles_and_each_shows_its_own_fields() {
    let mut s = with_ores(CubicKind::Custom, known());
    open(&mut s);
    let mode = ore(&s, 0, "Mode: ");
    assert_eq!(label_of(&s, mode), "Mode: Static height");
    let min = ore(&s, 0, "Min Y: ");
    drag(&mut s, min, at_int(30, -512, 1024));
    // Below ground: depths 0 to 512; changing the mode resets the band.
    tap(&mut s, mode);
    assert_eq!(label_of(&s, mode), "Mode: Below ground");
    assert_eq!(ores_of(&s).ores[0].rich.mode, RichMode::BelowGround);
    assert_eq!(ores_of(&s).ores[0].rich.min, f64::NEG_INFINITY);
    let depth = ore(&s, 0, "Min depth: ");
    assert_eq!(label_of(&s, depth), "Min depth: unbounded");
    drag(&mut s, depth, at_int(20, 0, 512));
    assert_eq!(ores_of(&s).ores[0].rich.min, 20.0);
    let at = ore(&s, 0, "Max depth: ");
    drag(&mut s, at, 1.0);
    assert_eq!(ores_of(&s).ores[0].rich.max, f64::INFINITY);
    drag(&mut s, depth, 0.0);
    assert_eq!(ores_of(&s).ores[0].rich.min, f64::NEG_INFINITY);
    // 3D noise: scale, threshold, seed offset; no band.
    tap(&mut s, mode);
    assert_eq!(label_of(&s, mode), "Mode: 3D noise");
    let (a, b) = ore_span(&s, 0);
    assert!(
        pg(&s).widgets[a..b]
            .iter()
            .all(|w| { !w.item.label.starts_with("Min ") && !w.item.label.starts_with("Max ") })
    );
    let scale = ore(&s, 0, "Noise scale: ");
    drag(&mut s, scale, 0.0);
    assert_eq!(ores_of(&s).ores[0].rich.noise_scale, 4.0);
    drag(&mut s, scale, 1.0);
    assert_eq!(ores_of(&s).ores[0].rich.noise_scale, 512.0);
    let threshold = ore(&s, 0, "Threshold: ");
    drag(&mut s, threshold, 0.7);
    assert!((ores_of(&s).ores[0].rich.noise_threshold - 0.4).abs() < 1e-9);
    assert_eq!(label_of(&s, threshold), "Threshold: 0.4");
    drag(&mut s, threshold, 0.0);
    assert_eq!(ores_of(&s).ores[0].rich.noise_threshold, -1.0);
    let seed = ore(&s, 0, "Seed offset: ");
    drag(&mut s, seed, 1.0);
    assert_eq!(ores_of(&s).ores[0].rich.noise_seed, 1000);
    drag(&mut s, seed, 0.0);
    assert_eq!(ores_of(&s).ores[0].rich.noise_seed, -1000);
    // Back round to static height.
    tap(&mut s, mode);
    assert_eq!(ores_of(&s).ores[0].rich.mode, RichMode::StaticHeight);
    assert_eq!(label_of(&s, ore(&s, 0, "Mode: ")), "Mode: Static height");
    ore(&s, 0, "Min Y: ");
    assert_eq!(
        CustomGeneratorSettings::from_json(&s.json()).unwrap(),
        s.settings
    );
}

#[test]
fn shape_and_intensity_are_tier_ones_and_show_hundredths() {
    let mut s = with_ores(CubicKind::Custom, known());
    open(&mut s);
    let shape = ore(&s, 2, "Shape: ");
    assert_eq!(label_of(&s, shape), "Shape: Flat");
    tap(&mut s, shape);
    assert_eq!(ores_of(&s).ores[2].rich.shape, Shape::Triangle);
    tap(&mut s, shape);
    assert_eq!(ores_of(&s).ores[2].rich.shape, Shape::Bell);
    assert_eq!(label_of(&s, shape), "Shape: Bell");
    tap(&mut s, shape);
    assert_eq!(ores_of(&s).ores[2].rich.shape, Shape::Flat);
    let intensity = ore(&s, 2, "Intensity: ");
    assert_eq!(label_of(&s, intensity), "Intensity: 1.00");
    drag(&mut s, intensity, Mapping::Cubic { max: 32.0 }.pos(2.5));
    assert_eq!(ores_of(&s).ores[2].rich.intensity, 2.5);
    assert_eq!(label_of(&s, intensity), "Intensity: 2.50");
    drag(&mut s, intensity, 1.0);
    assert_eq!(ores_of(&s).ores[2].rich.intensity, 32.0);
    drag(&mut s, intensity, 0.0);
    assert_eq!(ores_of(&s).ores[2].rich.intensity, 0.0);
}

#[test]
fn a_tier_that_is_off_is_greyed_not_hidden() {
    let mut s = with_ores(CubicKind::Custom, known());
    open(&mut s);
    let tier = |s: &State, n: &str| ore(s, 0, &format!("Tier {n}: "));
    assert_eq!(label_of(&s, tier(&s, "1")), "Tier 1: ON");
    assert_eq!(label_of(&s, tier(&s, "2")), "Tier 2: OFF");
    let tier_one = ["Mode: ", "Shape: ", "Min Y: ", "Max Y: ", "Intensity: "];
    let tier_two = ["Rate: ", "Range: "];
    for p in tier_one {
        assert!(enabled_of(&s, ore(&s, 0, p)), "{p}");
    }
    for p in tier_two {
        assert!(
            !enabled_of(&s, ore(&s, 0, p)),
            "{p} is greyed with tier 2 off"
        );
    }
    let at = tier(&s, "1");
    tap(&mut s, at);
    assert!(!ores_of(&s).ores[0].rich.enabled);
    for p in tier_one {
        assert!(
            !enabled_of(&s, ore(&s, 0, p)),
            "{p} is greyed with tier 1 off"
        );
    }
    let at = tier(&s, "2");
    tap(&mut s, at);
    assert!(ores_of(&s).ores[0].background.enabled);
    for p in tier_two {
        assert!(enabled_of(&s, ore(&s, 0, p)), "{p}");
    }
    // The second ore is another entry.
    assert!(ores_of(&s).ores[1].rich.enabled && !ores_of(&s).ores[1].background.enabled);
}

#[test]
fn tier_two_has_a_rate_and_a_range() {
    let mut s = with_ores(CubicKind::Custom, known());
    open(&mut s);
    let at = ore(&s, 1, "Tier 2: ");
    tap(&mut s, at);
    let rate = ore(&s, 1, "Rate: ");
    assert_eq!(label_of(&s, rate), "Rate: 0.00");
    drag(&mut s, rate, Mapping::Cubic { max: 4.0 }.pos(0.5));
    assert_eq!(ores_of(&s).ores[1].background.rate, 0.5);
    assert_eq!(label_of(&s, rate), "Rate: 0.50");
    drag(&mut s, rate, 1.0);
    assert_eq!(ores_of(&s).ores[1].background.rate, 4.0);
    // Everywhere has no bounds; Static height and Below ground add Min and Max.
    let range = ore(&s, 1, "Range: ");
    assert_eq!(label_of(&s, range), "Range: Everywhere");
    let band_rows = |s: &State, p: &str| {
        let (a, b) = ore_span(s, 1);
        pg(s).widgets[a..b]
            .iter()
            .filter(|w| w.item.label.starts_with(p))
            .count()
    };
    assert_eq!(band_rows(&s, "Min "), 1, "tier 1's Min only");
    tap(&mut s, range);
    assert_eq!(
        ores_of(&s).ores[1].background.range,
        RangeMode::StaticHeight
    );
    assert_eq!(band_rows(&s, "Min Y: "), 2);
    let (min, max) = (ore_nth(&s, 1, "Min Y: ", 1), ore_nth(&s, 1, "Max Y: ", 1));
    drag(&mut s, min, at_int(-100, -512, 1024));
    drag(&mut s, max, at_int(-40, -512, 1024));
    let bg = ores_of(&s).ores[1].background.clone();
    assert_eq!((bg.min, bg.max), (-100.0, -40.0));
    assert_eq!(
        ores_of(&s).ores[1].rich.min,
        0.0,
        "tier 1's band is its own"
    );
    drag(&mut s, min, at_int(500, -512, 1024));
    assert_eq!(ores_of(&s).ores[1].background.min, -40.0);
    let at = ore(&s, 1, "Range: ");
    tap(&mut s, at);
    assert_eq!(ores_of(&s).ores[1].background.range, RangeMode::BelowGround);
    assert_eq!(ores_of(&s).ores[1].background.min, f64::NEG_INFINITY);
    assert_eq!(band_rows(&s, "Min depth: "), 1);
    let at = ore(&s, 1, "Range: ");
    tap(&mut s, at);
    assert_eq!(ores_of(&s).ores[1].background.range, RangeMode::Everywhere);
    assert_eq!(band_rows(&s, "Min "), 1);
    assert_eq!(
        CustomGeneratorSettings::from_json(&s.json()).unwrap(),
        s.settings
    );
}

// ---------------------------------------------------------------------------
// Earth
// ---------------------------------------------------------------------------

#[test]
fn earth_reaches_the_page_by_next_and_without_ore_gen_shows_ores_off() {
    let mut s = earth_without_ores();
    assert_eq!(s.page_count(), 2);
    assert_eq!(s.page_id(), Some(PageId::Earth));
    let ids: Vec<u8> = buttons(SIZE.0, SIZE.1, &s).iter().map(|b| b.id).collect();
    assert!(ids.contains(&NEXT) && ids.contains(&PREV));
    let before = s.json();
    assert!(!before.contains("oreGen"));
    assert_eq!(press(&mut s, NEXT), super::super::Exit::Stay);
    assert_eq!(s.page_id(), Some(PageId::Ores));
    let widgets = pg(&s).widgets;
    assert_eq!(widgets.len(), 1, "only the switch");
    assert_eq!(widgets[0].item.label, "Ores: OFF");
    assert_eq!(s.json(), before, "looking changes nothing");
    assert_eq!(s.earth.ore_gen(), &None);
    press(&mut s, PREV);
    assert_eq!(s.page_id(), Some(PageId::Earth));
    assert_eq!(s.json(), before);
}

#[test]
fn turning_earths_ores_on_fills_the_earth_defaults_and_edits_round_trip() {
    let mut s = earth_without_ores();
    open(&mut s);
    tap(&mut s, 0);
    let mut expected = OreGenSettings::earth_defaults();
    expected.enabled = true;
    assert_eq!(ores_of(&s), expected);
    assert_eq!(earth_json_ores(&s), Some(expected.clone()));
    assert_eq!(pg(&s).widgets[0].item.label, "Ores: ON");
    // The page now has the richness map and strength first, then the ores.
    let map = find(&s, 0, usize::MAX, "Richness map: ", 0);
    assert_eq!(
        label_of(&s, map),
        format!(
            "Richness map: {}",
            if expected.richness_map { "ON" } else { "OFF" }
        )
    );
    tap(&mut s, map);
    assert_eq!(ores_of(&s).richness_map, !expected.richness_map);
    let strength = find(&s, 0, usize::MAX, "Richness strength: ", 0);
    drag(&mut s, strength, 0.5);
    assert_eq!(ores_of(&s).richness_strength, 1.0);
    drag(&mut s, strength, 0.3);
    assert_eq!(ores_of(&s).richness_strength, 0.6);
    assert_eq!(label_of(&s, strength), "Richness strength: 0.6");
    drag(&mut s, strength, 1.0);
    assert_eq!(ores_of(&s).richness_strength, 2.0);
    drag(&mut s, strength, 0.0);
    assert_eq!(ores_of(&s).richness_strength, 0.0);
    // An ore edit goes through the Earth JSON and back through a new screen.
    let size = ore(&s, 0, "Vein size: ");
    drag(&mut s, size, at_int(20, 1, 64));
    assert_eq!(ores_of(&s).ores[0].spawn_size, 20);
    let json_ores = earth_json_ores(&s).expect("oreGen is in the file");
    assert_eq!(json_ores, ores_of(&s));
    let again = state(CubicKind::Earth, &s.json());
    assert_eq!(ores_of(&again), json_ores);
    // Reset to defaults is Earth's.
    press_first(&mut s, "Reset to defaults");
    assert_eq!(ores_of(&s).ores, OreGenSettings::earth_defaults().ores);
    assert_eq!(ores_of(&s).richness_strength, expected.richness_strength);
}

#[test]
fn earths_ore_toggle_off_keeps_the_page_and_custom_has_no_richness_rows() {
    let mut s = with_ores(CubicKind::Earth, known());
    open(&mut s);
    press_first(&mut s, "Ores: ");
    assert!(!ores_of(&s).enabled);
    assert!(s.earth.ore_gen().is_some());
    assert_eq!(ore_count(&s), 3, "off is a switch, the list stays");
    let c = with_ores(CubicKind::Custom, known());
    let mut c = c;
    open(&mut c);
    assert!(
        pg(&c)
            .widgets
            .iter()
            .all(|w| !w.item.label.starts_with("Richness"))
    );
    assert!(
        pg(&s)
            .widgets
            .iter()
            .any(|w| w.item.label.starts_with("Richness map"))
    );
}

// ---------------------------------------------------------------------------
// Meteors
// ---------------------------------------------------------------------------

fn meteor_state() -> State {
    let mut s = with_ores(CubicKind::Custom, known());
    open(&mut s);
    s
}
fn m(s: &State) -> MeteorSettings {
    ores_of(s).meteors
}
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} is not {b}");
}

#[test]
fn the_meteor_switch_and_frequency_land_in_the_json() {
    let mut s = meteor_state();
    let toggle = met(&s, "Meteors: ");
    assert_eq!(label_of(&s, toggle), "Meteors: ON");
    let rate = met(&s, "Per km²: ");
    assert_eq!(label_of(&s, rate), "Per km²: 4");
    drag(&mut s, rate, Mapping::Cubic { max: 32.0 }.pos(10.0));
    near(m(&s).per_square_km, 10.0);
    drag(&mut s, rate, 1.0);
    assert_eq!(m(&s).per_square_km, 32.0);
    drag(&mut s, rate, 0.0);
    assert_eq!(m(&s).per_square_km, 0.0);
    assert_eq!(label_of(&s, rate), "Per km²: OFF");
    // The switch greys every other meteor control but the resets.
    tap(&mut s, toggle);
    assert!(!m(&s).enabled);
    for p in [
        "Per km²: ",
        "Min size: ",
        "Max size: ",
        "Surface: ",
        "Sunk: ",
        "Buried: ",
        "Buried depth: ",
        "Mix share: ",
        "Size floor: ",
        "Add point",
        "Add kind",
        "Size: ",
        "Weight: ",
        "Ore: ",
        "Second: ",
    ] {
        assert!(
            !enabled_of(&s, met(&s, p)),
            "{p} is greyed with meteors off"
        );
    }
    assert!(enabled_of(&s, met(&s, "Reset meteors")));
    assert!(enabled_of(&s, met(&s, "Meteors: ")));
    let at = met(&s, "Meteors: ");
    tap(&mut s, at);
    assert!(m(&s).enabled);
    assert_eq!(
        CustomGeneratorSettings::from_json(&s.json()).unwrap(),
        s.settings
    );
}

#[test]
fn meteor_diameters_keep_min_below_max() {
    let mut s = meteor_state();
    let (min, max) = (met(&s, "Min size: "), met(&s, "Max size: "));
    assert_eq!(label_of(&s, min), "Min size: 5");
    drag(&mut s, min, at_int(10, 3, 25));
    assert_eq!(m(&s).min_diameter, 10);
    drag(&mut s, max, at_int(6, 3, 25));
    assert_eq!((m(&s).min_diameter, m(&s).max_diameter), (10, 10));
    drag(&mut s, max, at_int(20, 3, 25));
    drag(&mut s, min, 1.0);
    assert_eq!((m(&s).min_diameter, m(&s).max_diameter), (20, 20));
    drag(&mut s, min, 0.0);
    assert_eq!(m(&s).min_diameter, 3);
    drag(&mut s, max, 1.0);
    assert_eq!(m(&s).max_diameter, 25);
}

#[test]
fn the_size_spline_has_a_row_a_point() {
    let mut s = meteor_state();
    let points = |s: &State| m(s).size_curve.0.clone();
    let sizes = |s: &State| points(s).iter().map(|p| p.y).collect::<Vec<_>>();
    assert_eq!(sizes(&s), [5.0, 7.0, 12.0, 25.0]);
    // A point's size stops at its neighbours, so the curve stays sorted.
    let second = met_nth(&s, "Size: ", 1);
    assert_eq!(label_of(&s, second), "Size: 7");
    drag(&mut s, second, at_int(25, 3, 25));
    assert_eq!(sizes(&s), [5.0, 12.0, 12.0, 25.0]);
    drag(&mut s, second, at_int(3, 3, 25));
    assert_eq!(sizes(&s), [5.0, 5.0, 12.0, 25.0]);
    drag(&mut s, second, at_int(9, 3, 25));
    assert_eq!(sizes(&s), [5.0, 9.0, 12.0, 25.0]);
    // Its weight is 0 to 1.
    let weight = met_nth(&s, "Weight: ", 1);
    drag(&mut s, weight, 0.35);
    near(points(&s)[1].v, 0.35);
    drag(&mut s, weight, 1.0);
    assert_eq!(points(&s)[1].v, 1.0);
    // Add point appends one past the last, sorted.
    let at = met(&s, "Add point");
    tap(&mut s, at);
    assert_eq!(points(&s).len(), 5);
    assert!(sizes(&s).windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(points(&s)[4].y, 25.0);
    assert_eq!(points(&s)[4].v, 0.03);
    // x deletes a point.
    let at = met_nth(&s, "x", 0);
    tap(&mut s, at);
    assert_eq!(sizes(&s), [9.0, 12.0, 25.0, 25.0]);
    // The last point cannot be deleted.
    for _ in 0..3 {
        let at = met_nth(&s, "x", 0);
        tap(&mut s, at);
    }
    assert_eq!(points(&s).len(), 1);
    assert!(!enabled_of(&s, met_nth(&s, "x", 0)));
    let at = met(&s, "Add point");
    tap(&mut s, at);
    assert_eq!(points(&s).len(), 2);
    // The floor.
    let floor = met(&s, "Size floor: ");
    drag(&mut s, floor, Mapping::Cubic { max: 1.0 }.pos(0.5));
    near(m(&s).size_floor, 0.5);
    assert_eq!(label_of(&s, floor), "Size floor: 0.5");
    assert_eq!(
        CustomGeneratorSettings::from_json(&s.json()).unwrap(),
        s.settings
    );
}

#[test]
fn the_meteor_shares_depth_and_mix_land_in_the_json() {
    let mut s = meteor_state();
    let at = met(&s, "Surface: ");
    drag(&mut s, at, 0.3);
    near(m(&s).surface_share, 0.3);
    let at = met(&s, "Sunk: ");
    drag(&mut s, at, 0.2);
    near(m(&s).sunk_share, 0.2);
    let at = met(&s, "Buried: ");
    drag(&mut s, at, 1.0);
    near(m(&s).buried_share, 1.0);
    let depth = met(&s, "Buried depth: ");
    assert_eq!(label_of(&s, depth), "Buried depth: 50");
    drag(&mut s, depth, at_int(80, 0, 128));
    assert_eq!(m(&s).buried_depth, 80.0);
    drag(&mut s, depth, 1.0);
    assert_eq!(m(&s).buried_depth, 128.0);
    let mix = met(&s, "Mix share: ");
    drag(&mut s, mix, 0.4);
    near(m(&s).mix_share, 0.2);
    drag(&mut s, mix, 1.0);
    near(m(&s).mix_share, 0.5);
    let back = CustomGeneratorSettings::from_json(&s.json()).unwrap();
    assert_eq!(back, s.settings);
    assert_eq!(back.ore_gen.unwrap().meteors, m(&s));
}

#[test]
fn meteor_kinds_add_cycle_and_delete() {
    let mut s = meteor_state();
    assert_eq!(label_of(&s, met(&s, "Ore: ")), "Ore: Iron");
    assert_eq!(label_of(&s, met(&s, "Second: ")), "Second: None");
    let at = met(&s, "Ore: ");
    tap(&mut s, at);
    assert_eq!(m(&s).kinds[0].ore, "copper");
    let at = met(&s, "Second: ");
    tap(&mut s, at);
    assert_eq!(m(&s).kinds[0].second, "iron");
    assert_eq!(label_of(&s, met(&s, "Second: ")), "Second: Iron");
    // Second goes round the eight ores and ends on none; Ore never does.
    for _ in 0..8 {
        let at = met(&s, "Second: ");
        tap(&mut s, at);
    }
    assert_eq!(m(&s).kinds[0].second, "");
    for _ in 0..8 {
        let at = met(&s, "Ore: ");
        tap(&mut s, at);
    }
    assert_eq!(m(&s).kinds[0].ore, "copper");
    // A new kind is an ore no kind has.
    let at = met(&s, "Add kind");
    tap(&mut s, at);
    assert_eq!(m(&s).kinds.len(), 2);
    assert_eq!(m(&s).kinds[1].ore, "iron");
    let at = met(&s, "Add kind");
    tap(&mut s, at);
    assert_eq!(m(&s).kinds[2].ore, "coal");
    // Its weight is 0 to 10.
    let kind_weight = met_nth(&s, "Weight: ", m(&s).size_curve.0.len() + 2);
    drag(&mut s, kind_weight, 0.25);
    near(m(&s).kinds[2].weight, 2.5);
    drag(&mut s, kind_weight, 1.0);
    assert_eq!(m(&s).kinds[2].weight, 10.0);
    // Delete the middle one: the kinds that are left keep their order.
    let ores_before: Vec<_> = m(&s).kinds.iter().map(|k| k.ore.clone()).collect();
    let spline = m(&s).size_curve.0.len();
    let at = met_nth(&s, "x", spline + 1);
    tap(&mut s, at);
    let ores_after: Vec<_> = m(&s).kinds.iter().map(|k| k.ore.clone()).collect();
    assert_eq!(ores_after, [ores_before[0].clone(), ores_before[2].clone()]);
    assert_eq!(
        CustomGeneratorSettings::from_json(&s.json()).unwrap(),
        s.settings
    );
}

#[test]
fn reset_meteors_restores_the_meteor_defaults_only() {
    let mut s = meteor_state();
    edit(&mut s, |o| {
        o.meteors.per_square_km = 9.0;
        o.meteors.kinds.clear();
        o.meteors.size_curve.0.truncate(1);
        o.enabled = false;
    });
    let at = met(&s, "Reset meteors");
    tap(&mut s, at);
    assert_eq!(m(&s), MeteorSettings::default());
    assert!(!ores_of(&s).enabled, "the ores keep their own switch");
    assert_eq!(ores_of(&s).ores, known().ores);
}

// ---------------------------------------------------------------------------
// The page as a whole
// ---------------------------------------------------------------------------

#[test]
fn no_row_of_the_page_is_a_note() {
    let mut states = vec![
        state(CubicKind::Custom, &old_preset().to_json_compact()),
        with_ores(CubicKind::Hybrid, known()),
        with_ores(CubicKind::Earth, known()),
        earth_without_ores(),
    ];
    // A noise ore, a below-ground one and a ranged tier 2 (the rows that differ).
    let mut odd = known();
    odd.ores[0].rich.mode = RichMode::Noise;
    odd.ores[1].rich.mode = RichMode::BelowGround;
    odd.ores[2].background = Background {
        enabled: true,
        rate: 0.5,
        range: RangeMode::BelowGround,
        min: 10.0,
        max: 60.0,
    };
    odd.meteors.enabled = false;
    states.push(with_ores(CubicKind::Custom, odd));
    for mut s in states {
        open(&mut s);
        for w in pg(&s).widgets {
            assert!(
                !matches!(w.item.kind, Kind::Note | Kind::Warning),
                "{:?}: {:?} is helper text",
                s.kind,
                w.item.label
            );
            if w.item.kind == Kind::Heading {
                assert!(!w.item.label.is_empty());
            }
        }
    }
}

#[test]
fn every_label_has_an_english_and_a_portuguese_string() {
    let mut odd = known();
    odd.ores[0].rich.mode = RichMode::Noise;
    odd.ores[1].rich.mode = RichMode::BelowGround;
    odd.ores[1].background.range = RangeMode::StaticHeight;
    odd.ores[2].background.range = RangeMode::BelowGround;
    odd.meteors.kinds.push(MeteorKind {
        ore: "gold".into(),
        second: "emerald".into(),
        weight: 1.0,
    });
    for code in ["en_us", "pt_br"] {
        for kind in [CubicKind::Custom, CubicKind::Earth] {
            let mut s = with_ores(kind, odd.clone());
            open(&mut s);
            let (widgets, missed) = scoped(Language::load(&[&Strings], code), || build_page(&s));
            let labels: Vec<_> = widgets
                .widgets
                .iter()
                .map(|w| w.item.label.clone())
                .collect();
            assert!(
                missed.iter().all(|k| k.starts_with("selectServer.")),
                "{code} {kind:?}: {missed:?}"
            );
            assert!(labels.len() > 20);
        }
    }
    // Both files hold the same new keys.
    let keys = |text: &str| {
        let map: std::collections::BTreeMap<String, String> = serde_json::from_str(text).unwrap();
        map.into_iter()
            .filter(|(k, _)| k.starts_with("horizon.cubic.oregen."))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let en = keys(include_str!("../../../assets/horizon/lang/en_us.json"));
    let pt = keys(include_str!("../../../assets/horizon/lang/pt_br.json"));
    assert!(en.len() > 50);
    assert_eq!(en.keys().collect::<Vec<_>>(), pt.keys().collect::<Vec<_>>());
    assert!(pt.values().all(|v| !v.is_empty()));
}

#[test]
fn the_long_page_scrolls_and_every_control_stays_in_its_column() {
    let mut s = state(CubicKind::Custom, &old_preset().to_json_compact());
    open(&mut s);
    let widgets = build_page(&s);
    let (top, bottom, _) = list_area(&s, SIZE);
    assert!(
        content_height(&widgets) > (bottom - top) * 2.0,
        "the page is taller than the list"
    );
    wheel(&mut s, 1000.0, (240.0, 100.0), SIZE);
    let one = s.scroll.amount;
    wheel(&mut s, -1000.0, (240.0, 100.0), SIZE);
    let other = s.scroll.amount;
    assert!(one.max(other) > 0.0 && one != other, "the wheel scrolls it");
    // Every control is inside the column; the rows do not overlap.
    for w in &widgets.widgets {
        assert!(
            w.rect[0] >= 0.0 && w.rect[0] + w.rect[2] <= WIDTH + 0.01,
            "{:?}",
            w.item.label
        );
    }
    for kind in [CubicKind::Custom, CubicKind::Earth] {
        let mut s = with_ores(kind, known());
        open(&mut s);
        let page = build_page(&s);
        let mut by_row: std::collections::BTreeMap<i32, Vec<[f32; 4]>> = Default::default();
        for w in page.widgets.iter().filter(|w| w.item.kind != Kind::Heading) {
            by_row.entry(w.rect[1] as i32).or_default().push(w.rect);
        }
        for row in by_row.values() {
            for pair in row.windows(2) {
                assert!(
                    pair[0][0] + pair[0][2] <= pair[1][0],
                    "controls overlap in a row"
                );
            }
        }
    }
}

#[test]
fn the_ores_page_of_an_edit_never_breaks_the_other_pages() {
    // The Lock of the Advanced page rewrites the expected heights on each
    // edit; an ore edit must not disturb the lists it no longer reads.
    let mut s = state(CubicKind::Custom, &old_preset().to_json_compact());
    open(&mut s);
    press_first(&mut s, "Add ore");
    let lakes = s.settings.lakes.clone();
    let caves = s.settings.caves.clone();
    for page in 0..s.page_count() {
        s.switch_page(page);
        let _ = build_page(&s);
    }
    assert_eq!(
        (s.settings.lakes.clone(), s.settings.caves.clone()),
        (lakes, caves)
    );
    s.settings.validate().unwrap();
}

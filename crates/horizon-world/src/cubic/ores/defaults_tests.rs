//! Tests of the ore defaults (`custom_defaults`, `earth_defaults`), of
//! `CustomGeneratorSettings::defaults()` carrying them, and of
//! `OreGenSettings::validate`.

use super::defaults::BIOME_MULTIPLIERS;
use super::*;
use crate::cubic::custom::settings::{
    BADLANDS_BIOMES, CustomGeneratorSettings, MOUNTAIN_BIOMES, legacy_vanilla_ores,
};
use crate::cubic::earth::settings::EarthSettings;

const ORES: [&str; 8] = [
    "coal", "iron", "copper", "gold", "redstone", "diamond", "lapis", "emerald",
];

fn short(e: &OreEntry) -> &str {
    e.blockstate
        .name
        .strip_prefix("minecraft:")
        .and_then(|n| n.strip_suffix("_ore"))
        .unwrap_or("")
}

fn of<'a>(o: &'a OreGenSettings, ore: &str) -> Vec<&'a OreEntry> {
    o.ores.iter().filter(|e| short(e) == ore).collect()
}

/// The entries with a background on.
fn with_background(o: &OreGenSettings) -> Vec<&OreEntry> {
    o.ores.iter().filter(|e| e.background.enabled).collect()
}

fn peak(e: &OreEntry) -> f64 {
    e.rich.intensity * e.rich.probability
}

// --- defaults() -----------------------------------------------------------------------------

#[test]
fn defaults_carry_oregen_and_empty_legacy_lists() {
    for s in [
        CustomGeneratorSettings::defaults(),
        CustomGeneratorSettings::defaults_without_structures(),
    ] {
        assert!(s.standard_ores.is_empty());
        assert!(s.periodic_gaussian_ores.is_empty());
        assert_eq!(s.ore_gen, Some(OreGenSettings::custom_defaults()));
        s.validate().unwrap();
    }
}

#[test]
fn the_legacy_lists_are_still_there_to_build_from() {
    let (standard, periodic) = legacy_vanilla_ores();
    assert_eq!(standard.len(), 16);
    assert_eq!(periodic.len(), 10);
    let o = OreGenSettings::from_legacy(&standard, &periodic, 64.0, 64.0);
    assert_eq!(o.ores.len(), 26);
}

#[test]
fn defaults_round_trip_through_json() {
    let s = CustomGeneratorSettings::defaults();
    let back = CustomGeneratorSettings::from_json(&s.to_json_pretty()).unwrap();
    assert_eq!(back.ore_gen, s.ore_gen);
    assert!(back.standard_ores.is_empty());
    assert!(back.periodic_gaussian_ores.is_empty());
    // (The lakes' probabilities come back one ulp off, so the whole struct
    // is compared without them.)
    let mut a = back.clone();
    let mut b = s.clone();
    a.lakes.clear();
    b.lakes.clear();
    assert_eq!(a, b);
    back.validate().unwrap();

    let e = OreGenSettings::earth_defaults();
    let mut earth = EarthSettings::at(10.0, 20.0);
    earth.ore_gen = Some(e.clone());
    let back = EarthSettings::from_json(&earth.to_json_pretty()).unwrap();
    assert_eq!(back.ore_gen, Some(e));
}

// --- custom_defaults ------------------------------------------------------------------------

#[test]
fn every_ore_has_one_background_of_eight_percent_of_its_main_band() {
    let o = OreGenSettings::custom_defaults();
    assert!(o.enabled);
    assert!(o.meteors.enabled);
    for ore in ORES {
        let main: Vec<_> = of(&o, ore)
            .into_iter()
            .filter(|e| e.background.enabled)
            .collect();
        assert_eq!(main.len(), 1, "{ore}: {} backgrounds", main.len());
        let e = main[0];
        let b = &e.background;
        assert!(
            b.rate >= 0.05 * peak(e) && b.rate <= 0.10 * peak(e),
            "{ore}"
        );
        assert!((b.rate - 0.08 * peak(e)).abs() < 1e-12, "{ore}");
        assert_eq!(b.range, RangeMode::Everywhere, "{ore}");
    }
    // Nothing else (stone patches, the other bands) has one.
    assert_eq!(with_background(&o).len(), 8);

    // And the main bands are these.
    let band = |ore: &str| {
        let e = with_background(&o)
            .into_iter()
            .find(|e| short(e) == ore)
            .unwrap();
        (e.rich.min, e.rich.max, e.rich.shape, e.spawn_size)
    };
    assert_eq!(band("coal"), (0.0, 192.0, Shape::Bell, 17));
    assert_eq!(band("iron"), (-24.0, 56.0, Shape::Bell, 9));
    assert_eq!(band("copper"), (-16.0, 112.0, Shape::Bell, 10));
    assert_eq!(band("gold"), (-96.0, -32.0, Shape::Bell, 9));
    assert_eq!(band("redstone"), (-96.0, -32.0, Shape::Bell, 8));
    assert_eq!(band("diamond"), (-144.0, 16.0, Shape::Bell, 8));
    assert_eq!(band("lapis"), (-64.0, 64.0, Shape::Flat, 7));
    assert_eq!(band("emerald"), (-16.0, 480.0, Shape::Bell, 3));
}

#[test]
fn multipliers_never_vanish_an_ore_and_emerald_keeps_the_mountains() {
    let o = OreGenSettings::custom_defaults();
    for e in &o.ores {
        for (biome, m) in &e.biome_multipliers {
            assert!(biome.starts_with("minecraft:"), "{biome}");
            assert!(*m >= 0.5, "{} {biome} {m}", e.blockstate.name);
        }
    }
    for (ore, m, biomes) in BIOME_MULTIPLIERS {
        assert!(*m >= 0.5, "{ore}");
        for e in of(&o, ore) {
            for b in *biomes {
                assert_eq!(e.biome_multipliers[&format!("minecraft:{b}")], *m);
            }
        }
    }
    let emerald = of(&o, "emerald");
    assert_eq!(emerald.len(), 1);
    let e = emerald[0];
    assert_eq!(e.biomes, None);
    assert_eq!(e.biome_multipliers.len(), MOUNTAIN_BIOMES.len());
    for b in MOUNTAIN_BIOMES {
        assert_eq!(e.biome_multipliers[&format!("minecraft:{b}")], 10.0);
    }
    // Vanilla's rate in the mountains: 10 x the base entry.
    let (_, periodic) = legacy_vanilla_ores();
    let vanilla = periodic.last().unwrap();
    let vanilla_peak = f64::from(vanilla.spawn_tries) * vanilla.spawn_probability;
    assert!((10.0 * peak(e) - vanilla_peak).abs() < 1e-9 * vanilla_peak);
    // The badlands' extra gold stays biome-filtered.
    let badlands_gold = of(&o, "gold")
        .into_iter()
        .filter(|e| e.biomes.is_some())
        .count();
    assert_eq!(badlands_gold, 1);
    assert_eq!(BADLANDS_BIOMES.len(), 3);
}

#[test]
fn without_the_extras_the_bands_are_exactly_vanillas() {
    let (standard, periodic) = legacy_vanilla_ores();
    let vanilla = OreGenSettings::from_legacy(&standard, &periodic, 64.0, 64.0);
    let mut o = OreGenSettings::custom_defaults();
    assert_eq!(o.ores.len(), vanilla.ores.len());
    for (e, v) in o.ores.iter_mut().zip(&vanilla.ores) {
        e.background = Background::default();
        e.biome_multipliers.clear();
        if short(e) == "emerald" {
            // Back to intensity 100 in the mountains only.
            e.rich.intensity = v.rich.intensity;
            e.rich.probability = v.rich.probability;
            e.biomes = v.biomes.clone();
        }
    }
    assert_eq!(o.ores, vanilla.ores);
}

// --- earth_defaults -------------------------------------------------------------------------

#[test]
fn earth_bands_are_depths_under_the_ground() {
    let o = OreGenSettings::earth_defaults();
    assert!(o.richness_map);
    assert_eq!(o.richness_strength, 1.0);
    assert!(o.meteors.enabled);
    assert!(!o.ores.is_empty());
    for e in &o.ores {
        let r = &e.rich;
        let name = &e.blockstate.name;
        assert_eq!(r.mode, RichMode::BelowGround, "{name}");
        assert!(r.min >= 0.0 && r.max >= 0.0 && r.min <= r.max, "{name}");
        if let Some(p) = r.peak {
            assert!(p >= r.min && p <= r.max, "{name}: peak {p}");
        }
    }
    // Every ore keeps its background (the main bands are all below ground).
    assert_eq!(with_background(&o).len(), 8);
    for ore in ORES {
        assert!(
            of(&o, ore).iter().any(|e| e.background.enabled),
            "{ore} lost its background"
        );
    }

    // Diamond: Y -144..16 is depth 48..208, peak at Y -64 = depth 128.
    let diamond = with_background(&o)
        .into_iter()
        .find(|e| short(e) == "diamond")
        .unwrap();
    assert_eq!((diamond.rich.min, diamond.rich.max), (48.0, 208.0));
    assert_eq!(diamond.rich.peak, Some(128.0));
    // Coal: Y 0..192 is depth -128..64: cut at the surface.
    let coal = with_background(&o)
        .into_iter()
        .find(|e| short(e) == "coal")
        .unwrap();
    assert_eq!((coal.rich.min, coal.rich.max), (0.0, 64.0));
    assert_eq!(coal.rich.peak, Some(0.0));
    // The bands wholly above ground (coal Y 136..320, iron 80..384) are
    // dropped; iron's other flat band (Y -64..72) stays, cut at depth 0.
    assert_eq!(of(&o, "coal").len(), 1);
    assert_eq!(of(&o, "iron").len(), 2);
    assert!(o.ores.iter().all(|e| e.rich.max >= 0.0));
    // Stone patches come along, below ground.
    for patch in ["dirt", "gravel", "granite", "diorite", "andesite", "tuff"] {
        assert!(
            o.ores
                .iter()
                .any(|e| e.blockstate.name == format!("minecraft:{patch}")),
            "{patch}"
        );
    }
}

// --- validate -------------------------------------------------------------------------------

#[test]
fn the_defaults_pass_validation() {
    OreGenSettings::custom_defaults().validate().unwrap();
    OreGenSettings::earth_defaults().validate().unwrap();
    OreGenSettings::default().validate().unwrap();
}

#[test]
fn bad_values_are_refused_naming_the_setting() {
    let bad = |what: &str, f: &dyn Fn(&mut OreGenSettings)| {
        let mut o = OreGenSettings::custom_defaults();
        f(&mut o);
        let e = o.validate().expect_err(what);
        assert!(e.contains(what), "{what}: {e}");
    };
    bad("oreGen.ores[3].rich.intensity", &|o| {
        o.ores[3].rich.intensity = 5000.0;
    });
    bad("oreGen.ores[3].rich.intensity", &|o| {
        o.ores[3].rich.intensity = f64::NAN;
    });
    bad("oreGen.ores[3].rich.intensity", &|o| {
        o.ores[3].rich.intensity = -1.0;
    });
    bad("oreGen.ores[4].spawnSize", &|o| o.ores[4].spawn_size = 65);
    bad("oreGen.ores[4].spawnSize", &|o| o.ores[4].spawn_size = 0);
    bad("oreGen.ores[1].rich.probability", &|o| {
        o.ores[1].rich.probability = 1.5;
    });
    bad("oreGen.ores[2].rich.min", &|o| {
        o.ores[2].rich.min = 10.0;
        o.ores[2].rich.max = 0.0;
    });
    bad("oreGen.ores[2].rich.spread", &|o| {
        o.ores[2].rich.spread = Some(0.0);
    });
    bad("oreGen.ores[2].rich.repeat", &|o| {
        o.ores[2].rich.repeat = -3.0
    });
    bad("oreGen.ores[2].rich.noiseScale", &|o| {
        o.ores[2].rich.noise_scale = 0.0;
    });
    bad("oreGen.ores[2].rich.noiseThreshold", &|o| {
        o.ores[2].rich.noise_threshold = f64::INFINITY;
    });
    bad("oreGen.ores[5].background.rate", &|o| {
        o.ores[5].background.rate = 9999.0;
    });
    bad("oreGen.ores[5].background.min", &|o| {
        o.ores[5].background.min = 5.0;
        o.ores[5].background.max = -5.0;
    });
    bad("oreGen.ores[0].biomeMultipliers[minecraft:plains]", &|o| {
        o.ores[0]
            .biome_multipliers
            .insert("minecraft:plains".into(), -0.1);
    });
    bad("oreGen.richnessStrength", &|o| o.richness_strength = 3.0);
    bad("oreGen.version", &|o| o.version = 2);
    bad("oreGen.version", &|o| o.version = -1);

    // Infinite bands are fine.
    let mut o = OreGenSettings::custom_defaults();
    o.ores[0].rich.min = f64::NEG_INFINITY;
    o.ores[0].rich.max = f64::INFINITY;
    o.validate().unwrap();
}

#[test]
fn a_cwg_json_with_a_bad_oregen_fails_validation() {
    let mut s = CustomGeneratorSettings::defaults();
    s.validate().unwrap();
    s.ore_gen.as_mut().unwrap().ores[3].rich.intensity = 5000.0;
    let parsed = CustomGeneratorSettings::from_json(&s.to_json_compact()).unwrap();
    let e = parsed.validate().unwrap_err();
    assert!(e.contains("oreGen.ores[3].rich.intensity"), "{e}");

    // Also written by hand, with a missing optional part.
    let json = r#"{ "oreGen": { "ores": [ { "blockstate": "coal_ore", "spawnSize": 99 } ] } }"#;
    let parsed = CustomGeneratorSettings::from_json(json).unwrap();
    let e = parsed.validate().unwrap_err();
    assert!(e.contains("oreGen.ores[0].spawnSize"), "{e}");
}

#[test]
fn earth_settings_validate_their_oregen() {
    let mut s = EarthSettings::at(10.0, 20.0);
    s.ore_gen = Some(OreGenSettings::earth_defaults());
    s.validate().unwrap();
    s.ore_gen.as_mut().unwrap().richness_strength = 5.0;
    let e = s.validate().unwrap_err();
    assert!(e.contains("oreGen.richnessStrength"), "{e}");
    let e = EarthSettings::from_json(&s.to_json_compact()).unwrap_err();
    assert!(e.contains("oreGen.richnessStrength"), "{e}");
}

// --- the tables, for the report -------------------------------------------------------------

#[test]
#[ignore = "prints the default tables: cargo test -- --ignored --nocapture print_default_tables"]
fn print_default_tables() {
    let show = |title: &str, o: &OreGenSettings| {
        println!("== {title}");
        for e in &o.ores {
            let r = &e.rich;
            println!(
                "{:<26} {:>6} {:>7} {:>7} {:?} peak {:?} sd {:?} int {:.4} p {:.4} (={:.4}) bg {} biomes {} mult {}",
                e.blockstate.name,
                e.spawn_size,
                r.min,
                r.max,
                r.shape,
                r.peak,
                r.spread,
                r.intensity,
                r.probability,
                peak(e),
                if e.background.enabled {
                    format!("{:.4}", e.background.rate)
                } else {
                    "-".into()
                },
                e.biomes.as_ref().map_or(0, Vec::len),
                e.biome_multipliers.len(),
            );
        }
    };
    show("custom", &OreGenSettings::custom_defaults());
    show("earth", &OreGenSettings::earth_defaults());
}

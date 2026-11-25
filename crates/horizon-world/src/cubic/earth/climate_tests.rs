//! Real climate in Earth's biomes: the mapping, the fallback, the setting and the far view's readiness.
use std::sync::Arc;

use horizon_earth::testing::{MemoryClimate, SMALL_CLIMATE_GRID, TempDir};
use horizon_earth::{Climate, ClimateFetcher, ClimateStore, ClimateVar, LandClass};

use super::biome::{Biome, CLIMATE_BIOME_RULES, EarthBiomes, fallback_temperature_c};
use super::land_tests::LandRig;
use super::*;
use crate::cubic::custom::biomes::BiomeColumns;

fn c(temperature_c: f32, precipitation_mm: f32) -> Option<Climate> {
    Some(Climate {
        temperature_c,
        precipitation_mm,
    })
}

#[test]
fn real_places_get_the_biomes_they_look_like() {
    use Biome::*;
    let of = |class, elevation, lat, climate| Biome::with_climate(class, elevation, lat, climate, false);
    // Amazon rainforest: hot and wet tree cover is jungle.
    assert_eq!(of(LandClass::Tree, 60.0, -3.0, c(26.7, 2449.0)), Jungle);
    // Sahara: hot, bone-dry bare ground is desert, a high bare plateau badlands.
    assert_eq!(of(LandClass::Bare, 400.0, 23.0, c(24.0, 6.5)), Desert);
    assert_eq!(of(LandClass::Bare, 1200.0, 23.0, c(22.0, 30.0)), Badlands);
    // Hot and dry grass is desert (dry places); wetter than that (r of 1 or more) it is savanna.
    assert_eq!(of(LandClass::Grass, 450.0, 17.0, c(28.0, 160.0)), Desert);
    assert_eq!(of(LandClass::Grass, 450.0, 12.0, c(26.0, 800.0)), Savanna);
    // Canadian boreal forest: below 0 C, taiga with snow.
    assert_eq!(of(LandClass::Tree, 400.0, 55.0, c(-0.9, 505.0)), SnowyTaiga);
    assert_eq!(of(LandClass::Tree, 300.0, 50.0, c(2.0, 600.0)), Taiga);
    // Pacific Northwest rainforest: mild and very wet is old growth spruce.
    assert_eq!(of(LandClass::Tree, 200.0, 47.9, c(10.0, 2500.0)), OldGrowthSpruceTaiga);
    // A wet tropical highland (Costa Rica's cloud forest, 17 C) is jungle, a drier one forest: never spruce.
    assert_eq!(of(LandClass::Tree, 1500.0, 10.3, c(17.0, 3000.0)), Jungle);
    assert_eq!(of(LandClass::Tree, 1500.0, -10.0, c(16.0, 500.0)), Forest);
    // A temperate wood of middling rain is a plain forest; wetter is birch, wetter again dark forest.
    assert_eq!(of(LandClass::Tree, 200.0, 48.0, c(10.0, 600.0)), Forest);
    assert_eq!(of(LandClass::Tree, 200.0, 48.0, c(10.0, 1000.0)), BirchForest);
    // A dry temperate wood is vanilla's flower forest.
    assert_eq!(of(LandClass::Tree, 200.0, 40.0, c(14.0, 150.0)), FlowerForest);
    // Alpine meadows from 1,500 m.
    assert_eq!(of(LandClass::Grass, 1800.0, 46.5, c(4.0, 1500.0)), Meadow);
    // Farmland: plains in a mild climate, savanna in a warm one.
    assert_eq!(of(LandClass::Crop, 100.0, 50.0, c(9.0, 700.0)), Plains);
    assert_eq!(of(LandClass::Crop, 100.0, 15.0, c(26.0, 900.0)), Savanna);
    // The sea by its temperature.
    assert_eq!(of(LandClass::Water, -30.0, 0.0, c(26.0, 1500.0)), WarmOcean);
    assert_eq!(of(LandClass::Water, -3000.0, 30.0, c(20.0, 800.0)), DeepLukewarmOcean);
    assert_eq!(of(LandClass::Water, -30.0, 45.0, c(12.0, 800.0)), Ocean);
    assert_eq!(of(LandClass::Water, -30.0, 58.0, c(4.0, 800.0)), ColdOcean);
    assert_eq!(of(LandClass::NoData, -900.0, 75.0, c(-8.0, 300.0)), DeepFrozenOcean);
    // Rivers and lakes above the sea freeze where it is below 0 C.
    assert_eq!(of(LandClass::Water, 300.0, 55.0, c(-1.0, 500.0)), FrozenRiver);
    assert_eq!(of(LandClass::Water, 300.0, 45.0, c(9.0, 500.0)), River);
    // Wetlands and mangroves stay as they are.
    assert_eq!(of(LandClass::Wetland, 5.0, 30.0, c(20.0, 1300.0)), Swamp);
    assert_eq!(of(LandClass::Mangrove, 1.0, 5.0, c(27.0, 2000.0)), MangroveSwamp);
    // Snow and the snow line win over the climate.
    assert_eq!(of(LandClass::Grass, 8849.0, 28.0, c(-20.0, 500.0)), FrozenPeaks);
    // A variant cell takes vanilla's variant where there is one.
    assert_eq!(
        Biome::with_climate(LandClass::Tree, 200.0, 48.0, c(10.0, 1000.0), true),
        OldGrowthBirchForest
    );
    assert_eq!(
        Biome::with_climate(LandClass::Tree, 60.0, -3.0, c(26.7, 2449.0), true),
        BambooJungle
    );
}

#[test]
fn without_climate_data_land_is_the_latitude_table_and_the_sea_takes_a_temperature() {
    let classes = [
        LandClass::Tree,
        LandClass::Shrub,
        LandClass::Grass,
        LandClass::Crop,
        LandClass::BuiltUp,
        LandClass::Bare,
        LandClass::SnowIce,
        LandClass::Wetland,
        LandClass::Mangrove,
        LandClass::MossLichen,
        LandClass::NoData,
    ];
    for class in classes {
        for elevation in [1.0, 300.0, 1600.0, 2600.0, 5000.0] {
            for lat in [-70.0, -40.0, -5.0, 0.0, 20.0, 47.0, 62.0, 75.0] {
                for variant in [false, true] {
                    assert_eq!(
                        Biome::with_climate(class, elevation, lat, None, variant),
                        Biome::of(class, elevation, lat),
                        "{class:?} at {elevation} m, latitude {lat}"
                    );
                }
            }
        }
    }
    // About 27 C at the equator, freezing near the poles, 6.5 C colder a kilometre up.
    assert!((fallback_temperature_c(0.0, 0.0) - 27.0).abs() < 1e-9);
    assert!(fallback_temperature_c(0.0, 70.0) < 0.0);
    assert!((fallback_temperature_c(1000.0, 0.0) - 20.5).abs() < 1e-9);
    use Biome::*;
    assert_eq!(Biome::with_climate(LandClass::Water, -50.0, 2.0, None, false), WarmOcean);
    assert_eq!(Biome::with_climate(LandClass::Water, -50.0, 45.0, None, false), Ocean);
    assert_eq!(Biome::with_climate(LandClass::Water, -50.0, 80.0, None, false), FrozenOcean);
    assert_eq!(Biome::with_climate(LandClass::Water, 200.0, 70.0, None, false), FrozenRiver);
}

#[test]
fn every_biome_is_a_vanilla_name_once() {
    let names: std::collections::BTreeSet<_> = Biome::ALL.iter().map(|b| b.name()).collect();
    assert_eq!(names.len(), Biome::ALL.len());
    // Every pick the climate can make is in the list the biome source declares.
    let climates = [(-15.0, 200.0), (-2.0, 900.0), (4.0, 300.0), (12.0, 80.0), (12.0, 3000.0), (22.0, 1200.0)];
    let more = [(26.0, 3000.0), (28.0, 50.0), (20.0, 400.0)];
    for class in [
        LandClass::Tree,
        LandClass::Shrub,
        LandClass::Grass,
        LandClass::Crop,
        LandClass::BuiltUp,
        LandClass::Bare,
        LandClass::Water,
        LandClass::Wetland,
        LandClass::NoData,
    ] {
        for elevation in [-500.0, -10.0, 50.0, 1200.0, 1700.0, 2500.0] {
            for &(t, p) in climates.iter().chain(&more) {
                for variant in [false, true] {
                    let b = Biome::with_climate(class, elevation, 30.0, c(t, p), variant);
                    assert!(Biome::ALL.contains(&b));
                }
            }
        }
    }
}

#[test]
fn the_climate_setting_is_off_unless_written_and_is_in_the_key_only_with_land_cover() {
    let plain = EarthSettings::at(46.5, 8.0);
    assert!(!plain.climate);
    assert!(!plain.to_json_compact().contains("climate"));
    assert!(!plain.terrain_key().contains("#climate"));
    let mut on = plain.clone();
    on.land_cover = true;
    on.climate = true;
    let json = on.to_json_compact();
    assert!(json.contains("\"climate\":true"), "{json}");
    assert!(EarthSettings::from_json(&json).unwrap().climate);
    let wanted = format!("#climate{}.{CLIMATE_BIOME_RULES}", horizon_earth::CLIMATE_RULES);
    assert!(on.terrain_key().ends_with(&wanted), "{}", on.terrain_key());
    let mut no_land = on.clone();
    no_land.land_cover = false;
    assert!(!no_land.terrain_key().contains("#climate"));
    let mut off = on.clone();
    off.climate = false;
    assert_ne!(off.terrain_key(), on.terrain_key());
}

/// A rig of tree cover at `(lat, 8)` with a whole-world climate of `t` C and `p` mm.
fn climate_rig(lat: f64, t: f32, p: f32) -> (LandRig, Arc<MemoryClimate>, Arc<EarthGenerator>, TempDir) {
    let mut rig = LandRig::level(lat, 200.0, 10);
    rig.settings.climate = true;
    let raw_t = ((t + 273.15) * 10.0).round() as u16;
    let raw_p = p.round() as u16;
    let climate = Arc::new(
        MemoryClimate::new()
            .with_raw(ClimateVar::Temperature, 64, 1, move |_, _, _| raw_t)
            .with_raw(ClimateVar::Precipitation, 64, 1, move |_, _, _| raw_p),
    );
    let dir = TempDir::new();
    let store = Arc::new(ClimateStore::with_grid(dir.path(), 8, SMALL_CLIMATE_GRID));
    let fetcher: Arc<dyn ClimateFetcher> = climate.clone();
    let source = rig.settings.climate_source(store, fetcher).unwrap();
    let generator = Arc::new(
        EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks)
            .unwrap()
            .with_land_cover(rig.source.clone())
            .with_climate(source),
    );
    (rig, climate, generator, dir)
}

#[test]
fn the_biomes_follow_the_climate_and_the_far_view_waits_for_it_without_loading() {
    let (rig, climate, g, _dir) = climate_rig(LAT_TEST, 26.5, 2400.0);
    // The far view's form never fetches: it names the climate it lacks and says "not ready".
    assert!(g.biomes_ready(0, 0, 1).is_err());
    assert_eq!(climate.fetch_count(), 0);
    for _ in 0..50 {
        match g.biomes_ready(0, 0, 1) {
            Ok(()) => break,
            Err(missing) => {
                for &id in &missing.tiles {
                    rig.store.load(id).unwrap();
                }
                for fetch in g.take_land_requests() {
                    rig.source.satisfy(&fetch).unwrap();
                }
                for fetch in g.take_climate_requests() {
                    g.climate_source().unwrap().satisfy(&fetch).unwrap();
                }
            }
        }
    }
    assert!(g.biomes_ready(0, 0, 1).is_ok());
    assert!(climate.fetch_count() > 0);
    let biomes = EarthBiomes::new(g.clone());
    let name = biomes.biome_name(biomes.biome(0, 0)).to_string();
    assert!(name.contains("jungle"), "{name}");
    // Cold and dry elsewhere gives another biome: the climate picks, not the latitude.
    let (_rig, _c, cold, _d) = climate_rig(LAT_TEST, -3.0, 400.0);
    let b = EarthBiomes::new(cold);
    assert_eq!(b.biome_name(b.biome(0, 0)), "minecraft:snowy_taiga");
}

#[test]
fn a_world_without_the_climate_setting_ignores_a_source() {
    let (mut rig, _climate, _g, _dir) = climate_rig(LAT_TEST, 26.5, 2400.0);
    rig.settings.climate = false;
    let g = EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks)
        .unwrap()
        .with_land_cover(rig.source.clone());
    assert!(!g.climate_wanted() && g.climate_source().is_none());
}

/// A latitude where the latitude table alone says forest (not jungle, not taiga).
const LAT_TEST: f64 = 40.0;

/// Prints the real climate and the biome per land class at places, from a game's downloaded climate:
/// `HORIZON_CLIMATE_CACHE=<game>/.horizon-cache/earth-climate/<hash>/<rules> cargo test ... climate_probe -- --ignored --nocapture`
#[test]
#[ignore = "reads climate a game downloaded"]
fn climate_probe() {
    let Ok(dir) = std::env::var("HORIZON_CLIMATE_CACHE") else {
        return;
    };
    let store = ClimateStore::new(dir, 64);
    let places = [
        ("amazon", -2.93, -59.97, 60.0),
        ("sahara", 16.97, 7.99, 500.0),
        ("boreal", 58.0, -122.5, 400.0),
        ("pnw", 47.86, -123.93, 180.0),
        ("alps", 46.59, 7.91, 2000.0),
    ];
    for (name, lat, lon, elevation) in places {
        let climate = match store.lookup(lat, lon, 1000.0) {
            horizon_earth::ClimateLookup::Climate(c) => Some(c),
            other => {
                println!("{name}: {other:?}");
                None
            }
        };
        let mut line = format!("{name} {climate:?}:");
        for class in [LandClass::Tree, LandClass::Shrub, LandClass::Grass, LandClass::Crop, LandClass::Bare] {
            let b = Biome::with_climate(class, elevation, lat, climate, false);
            line += &format!(" {class:?}={}", b.name().trim_start_matches("minecraft:"));
        }
        println!("{line}");
    }
}

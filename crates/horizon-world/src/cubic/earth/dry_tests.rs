//! Tests of the dry places' rules ([`super::dry`]) on synthetic data: the surface kinds, the terracotta bands,
//! the biomes, the plants and the generator over a synthetic climate. No network, no real data.

use std::sync::Arc;

use horizon_core::position::SectionPos;
use horizon_earth::testing::{MemoryClimate, SMALL_CLIMATE_GRID, TempDir};
use horizon_earth::{Climate, ClimateFetcher, ClimateStore, ClimateVar, LandClass};

use super::biome::Biome;
use super::dry::{
    BAND_BLOCKS, BAND_COUNT, BARREN_MIX, DESERT_ARIDITY, DESERT_MIX, DRY_ARIDITY, DRY_ROCK_DEPTH,
    DryColumn, DryTop, ERODED_M, HOT_C, PLANT_SHARE, RUGGED_M, SCRUB_MIX, aridity, band_index,
    band_table, column_bucket,
};
use super::generator::Access;
use super::land_tests::LandRig;
use super::surface::{LAKE_DEPTH, SOIL_DEPTH, Slope, SurfaceKind};
use super::tests::LON0;
use super::vegetation::{FeatureKind, allowed, allowed_in};
use super::water::Bed;
use super::*;

const NO_SLOPE: fn() -> Result<Slope, ()> = || Ok(Slope::Gentle);
const SLOPE: fn() -> Result<Slope, ()> = || Ok(Slope::Steep);

/// A dry column of temperature `t` and aridity `r` (its hash decides the top, `bucket`).
fn col(t: f64, r: f64, bucket: u16) -> Option<DryColumn> {
    Some(DryColumn::of(t, r, bucket))
}

fn kind(
    class: LandClass,
    lat: f64,
    dry: Option<DryColumn>,
    steep: fn() -> Result<Slope, ()>,
) -> SurfaceKind {
    SurfaceKind::on_classified(class, 1200.0, lat, false, dry, [0, 0], steep).unwrap()
}

fn climate(t: f32, p: f32) -> Option<Climate> {
    Some(Climate {
        temperature_c: t,
        precipitation_mm: p,
    })
}

// --- The constants ---------------------------------------------------------------------------------------------

#[test]
fn the_rule_constants_are_what_the_docs_say() {
    assert_eq!(DRY_ARIDITY, 1.3);
    assert_eq!(DESERT_ARIDITY, 0.5);
    assert_eq!(HOT_C, 18.0);
    assert_eq!(BAND_BLOCKS, 4);
    assert_eq!(DRY_ROCK_DEPTH, 32);
    assert_eq!((RUGGED_M, ERODED_M), (150.0, 400.0));
    assert_eq!(PLANT_SHARE, 300);
    // The aridity ratio: 400 mm at 10 C is 400 / 340.
    assert!((aridity(10.0, 400.0) - 400.0 / 340.0).abs() < 1e-12);
    assert!(aridity(-30.0, 100.0).is_finite());
}

#[test]
fn the_cache_keys_know_the_dry_rules() {
    let mut settings = EarthSettings::at(36.0, -112.0);
    settings.land_cover = true;
    settings.vegetation = true;
    settings.climate = true;
    let key = settings.terrain_key();
    assert!(key.contains("#landcover"), "{key}");
    assert_eq!(surface::LAND_RULES, 10);
    assert_eq!(biome::CLIMATE_BIOME_RULES, 4);
    assert_eq!(vegetation::VEGETATION_RULES, 2);
    assert!(key.contains(&format!(".{}", surface::LAND_RULES)), "{key}");
    assert!(key.contains("#vegetation2"), "{key}");
    assert!(
        key.ends_with(&format!(".{}", biome::CLIMATE_BIOME_RULES)),
        "{key}"
    );
}

// --- The surface kinds -----------------------------------------------------------------------------------------

#[test]
fn dry_bare_steep_ground_is_banded_and_gentle_ground_depends_on_the_heat() {
    let dry = col(22.0, 0.3, 500);
    assert_eq!(kind(LandClass::Bare, 36.0, dry, SLOPE), SurfaceKind::Banded);
    // Gentle: hot is sand dunes (the Sahara), not hot is barren red ground.
    assert_eq!(
        kind(LandClass::Bare, 23.0, dry, NO_SLOPE),
        SurfaceKind::Dune
    );
    let cool = col(HOT_C - 1.0, 0.3, 500);
    assert!(matches!(
        kind(LandClass::Bare, 36.0, cool, NO_SLOPE),
        SurfaceKind::Barren(_)
    ));
    // The hot edge itself is hot.
    assert_eq!(
        kind(LandClass::Bare, 23.0, col(HOT_C, 0.3, 0), NO_SLOPE),
        SurfaceKind::Dune
    );
    // The alpine band under the snow line still wins; so does the cold latitude gravel.
    assert_eq!(
        SurfaceKind::on_classified(LandClass::Bare, 4500.0, 36.0, false, dry, [0, 0], SLOPE)
            .unwrap(),
        SurfaceKind::Alpine
    );
    assert_eq!(
        kind(LandClass::Bare, 55.0, cool, NO_SLOPE),
        SurfaceKind::Gravel
    );
}

#[test]
fn dry_shrub_is_scrub_where_gentle_and_dry_grass_stays_grass_both_banded_where_steep() {
    let dry = col(12.0, 0.7, 0);
    for class in [LandClass::Shrub, LandClass::Grass] {
        assert_eq!(kind(class, 36.0, dry, SLOPE), SurfaceKind::Banded);
    }
    assert!(matches!(
        kind(LandClass::Shrub, 36.0, dry, NO_SLOPE),
        SurfaceKind::Scrub(_)
    ));
    assert_eq!(
        kind(LandClass::Grass, 36.0, dry, NO_SLOPE),
        SurfaceKind::Grass
    );
}

#[test]
fn the_scrub_mix_follows_the_stated_shares_per_column() {
    let count = |mix_r: f64| {
        let (mut grass, mut coarse, mut sand) = (0, 0, 0);
        for x in -150..150 {
            for z in -100..100 {
                let c = DryColumn::new(
                    Climate {
                        temperature_c: 15.0,
                        precipitation_mm: (mix_r * 20.0 * 22.0) as f32,
                    },
                    x,
                    z,
                );
                match kind(LandClass::Shrub, 36.0, Some(c), NO_SLOPE) {
                    SurfaceKind::Scrub(DryTop::GrassBlock) => grass += 1,
                    SurfaceKind::Scrub(DryTop::CoarseDirt) => coarse += 1,
                    SurfaceKind::Scrub(DryTop::RedSand) => sand += 1,
                    other => panic!("{other:?}"),
                }
            }
        }
        let total = f64::from(grass + coarse + sand);
        [grass, coarse, sand].map(|n| f64::from(n) / total)
    };
    let near = |got: [f64; 3], want: [f64; 3]| {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 0.02, "{got:?} vs {want:?}");
        }
    };
    // Desert (r under 0.5): 15 % grass block, 45 % coarse dirt, 40 % red sand.
    near(count(0.3), [0.15, 0.45, 0.40]);
    // Dry scrub (r from 0.5 to 1): 40 %, 40 %, 20 %.
    near(count(0.8), [0.40, 0.40, 0.20]);
    // The shares are the constants'.
    assert_eq!((DESERT_MIX.grass_block, DESERT_MIX.coarse_dirt), (150, 450));
    assert_eq!((SCRUB_MIX.grass_block, SCRUB_MIX.coarse_dirt), (400, 400));
    // Barren ground: 55 % red sand, 45 % coarse dirt, never grass.
    let mut tops = [0_u32; 3];
    for x in 0..300 {
        for z in 0..100 {
            let c = DryColumn::new(climate(10.0, 50.0).unwrap(), x, z);
            let top = BARREN_MIX.pick(c.bucket);
            tops[top as usize] += 1;
        }
    }
    assert_eq!(tops[DryTop::GrassBlock as usize], 0);
    let sand = f64::from(tops[DryTop::RedSand as usize]) / 30_000.0;
    assert!((sand - 0.55).abs() < 0.02, "{sand}");
}

#[test]
fn the_column_hash_is_fixed_and_stays_in_range() {
    assert_eq!(column_bucket(5, -9, 3), column_bucket(5, -9, 3));
    assert_ne!(
        (0..50).map(|x| column_bucket(x, 7, 1)).collect::<Vec<_>>(),
        (0..50).map(|x| column_bucket(x, 7, 2)).collect::<Vec<_>>()
    );
    for x in -50..50 {
        assert!(column_bucket(x, x * 3, 9) < 1000);
    }
}

#[test]
fn trees_and_the_rest_keep_their_tops_in_a_dry_climate() {
    let dry = col(22.0, 0.3, 0);
    // Rim forest: tree cover stays grass over dirt, whatever the climate.
    assert_eq!(
        kind(LandClass::Tree, 36.0, dry, NO_SLOPE),
        SurfaceKind::Grass
    );
    assert_eq!(kind(LandClass::Tree, 36.0, dry, SLOPE), SurfaceKind::Grass);
    assert_eq!(
        kind(LandClass::Crop, 36.0, dry, NO_SLOPE),
        SurfaceKind::Grass
    );
    assert_eq!(
        kind(LandClass::Wetland, 36.0, dry, NO_SLOPE),
        SurfaceKind::Marsh
    );
    assert_eq!(
        kind(LandClass::Mangrove, 36.0, dry, NO_SLOPE),
        SurfaceKind::Marsh
    );
    assert_eq!(
        kind(LandClass::MossLichen, 36.0, dry, NO_SLOPE),
        SurfaceKind::Moss(super::cover::moss_top(0, 0))
    );
    assert_eq!(
        kind(LandClass::BuiltUp, 36.0, dry, NO_SLOPE),
        kind(LandClass::Grass, 36.0, dry, NO_SLOPE)
    );
    assert_eq!(
        kind(LandClass::Water, 36.0, dry, NO_SLOPE),
        SurfaceKind::Water {
            depth: LAKE_DEPTH as u32,
            bed: Bed::Sand
        }
    );
    assert_eq!(
        kind(LandClass::SnowIce, 36.0, dry, NO_SLOPE),
        SurfaceKind::Glacier
    );
    // Snow wins over everything.
    assert_eq!(
        SurfaceKind::on_classified(LandClass::Bare, 8000.0, 30.0, false, dry, [0, 0], SLOPE)
            .unwrap(),
        SurfaceKind::Snow
    );
}

#[test]
fn wet_climate_and_no_climate_give_the_old_kinds_except_wet_bare_rock_and_frozen_wetland() {
    // T 9, P 2,800: r = 8.75, and a frozen-dry mix (T -5, P 50) is not "dry" either.
    let wet = DryColumn::new(climate(9.0, 2800.0).unwrap(), 4, 4);
    assert!(!wet.is_dry());
    let frozen = DryColumn::new(climate(-5.0, 20.0).unwrap(), 4, 4);
    assert!(!frozen.is_dry());
    // The edge: r equal to the dry threshold is not dry.
    assert!(!DryColumn::of(12.0, DRY_ARIDITY, 0).is_dry());
    for class in LandClass::ALL {
        for (elevation, lat) in [
            (40.0, 10.0),
            (800.0, 36.0),
            (1500.0, 47.0),
            (3000.0, 52.0),
            (-5.0, 5.0),
        ] {
            for steep in [NO_SLOPE, SLOPE] {
                let none = SurfaceKind::on_classified(
                    class,
                    elevation,
                    lat,
                    elevation < 0.0,
                    None,
                    [0, 0],
                    steep,
                )
                .unwrap();
                for other in [wet, frozen] {
                    let got = SurfaceKind::on_classified(
                        class,
                        elevation,
                        lat,
                        elevation < 0.0,
                        Some(other),
                        [0, 0],
                        steep,
                    )
                    .unwrap();
                    // Gentle bare ground that was sand is wet bare rock in a known wet (or frozen) climate.
                    if class == LandClass::Bare && none == SurfaceKind::Dune {
                        assert!(matches!(got, SurfaceKind::Basalt(_)), "{got:?}");
                        continue;
                    }
                    // Wetland in a frozen climate is tundra ground, not a marsh (rules_tests).
                    if class == LandClass::Wetland && none == SurfaceKind::Marsh && other == frozen
                    {
                        assert_eq!(got, SurfaceKind::Grass);
                        continue;
                    }
                    assert_eq!(got, none, "{class:?} at {elevation} m, {lat}");
                }
                assert!(!none.is_dry(), "{class:?} {none:?}");
            }
        }
    }
    // And the old kinds themselves.
    assert_eq!(
        kind(LandClass::Shrub, 36.0, None, SLOPE),
        SurfaceKind::Grass
    );
    assert_eq!(
        kind(LandClass::Grass, 36.0, None, NO_SLOPE),
        SurfaceKind::Grass
    );
    assert_eq!(
        kind(LandClass::Bare, 36.0, None, SLOPE),
        SurfaceKind::Alpine
    );
    assert_eq!(
        kind(LandClass::Bare, 36.0, None, NO_SLOPE),
        SurfaceKind::Dune
    );
    assert_eq!(
        kind(LandClass::Bare, 55.0, None, NO_SLOPE),
        SurfaceKind::Gravel
    );
}

// --- The bands -------------------------------------------------------------------------------------------------

#[test]
fn the_band_table_is_vanillas_clay_bands_by_absolute_height_in_steps_of_four_blocks() {
    let rig = LandRig::level(36.0, 1500.0, 20);
    let b = rig.blocks;
    let table = band_table(&b);
    let terracotta = [
        b.terracotta,
        b.orange_terracotta,
        b.yellow_terracotta,
        b.brown_terracotta,
        b.red_terracotta,
        b.white_terracotta,
        b.light_gray_terracotta,
    ];
    assert_eq!(table.len(), BAND_COUNT);
    assert!(table.iter().all(|id| terracotta.contains(id)));
    // Every one of the seven kinds shows up, and it is the same table every time.
    for id in terracotta {
        assert!(table.contains(&id), "{id}");
    }
    assert_eq!(table, band_table(&b));
    // Four blocks per band, repeating after 192 bands, below zero too.
    assert_eq!(band_index(0), band_index(3));
    assert_ne!(band_index(3), band_index(4));
    assert_eq!(band_index(4), 1);
    assert_eq!(band_index(-1), BAND_COUNT - 1);
    assert_eq!(band_index(4 * 192), 0);
    assert_eq!(band_index(-4 * 192 - 1), BAND_COUNT - 1);
    assert!(table.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn banded_ground_has_the_same_band_at_the_same_height_whatever_the_surface() {
    let rig = LandRig::level(36.0, 1500.0, 60);
    let g = &rig.generator;
    let b = rig.blocks;
    let table = band_table(&b);
    for s in [100, 101, 137, 250, -40, 3000] {
        for y in s - 40..=s + 3 {
            let got = g.block_for(y, s, SurfaceKind::Banded);
            let want = if y > s {
                b.air
            } else if y < s - 32 {
                b.stone
            } else {
                table[band_index(i64::from(y))]
            };
            assert_eq!(got, want, "surface {s} height {y}");
            // A scrub or barren top has the same bands under its top block.
            for top in [
                SurfaceKind::Scrub(DryTop::RedSand),
                SurfaceKind::Barren(DryTop::CoarseDirt),
            ] {
                if y < s && y >= s - 32 {
                    assert_eq!(g.block_for(y, s, top), want);
                }
            }
        }
    }
    // The tops.
    for (top, want) in [
        (DryTop::GrassBlock, b.grass_block),
        (DryTop::CoarseDirt, b.coarse_dirt),
        (DryTop::RedSand, b.red_sand),
    ] {
        assert_eq!(g.block_for(77, 77, SurfaceKind::Scrub(top)), want);
        assert_eq!(g.block_for(78, 77, SurfaceKind::Scrub(top)), b.air);
    }
    assert_eq!(
        g.block_for(77, 77, SurfaceKind::Barren(DryTop::RedSand)),
        b.red_sand
    );
    // Far voxels: the voxel holding a mixed top is the mix's main block, one below is the band at its middle.
    assert_eq!(
        g.voxel_block(72, 8, 77, SurfaceKind::Scrub(DryTop::RedSand)),
        b.coarse_dirt
    );
    assert_eq!(
        g.voxel_block(72, 8, 77, SurfaceKind::Barren(DryTop::CoarseDirt)),
        b.red_sand
    );
    assert_eq!(
        g.voxel_block(48, 8, 77, SurfaceKind::Banded),
        table[band_index(52)]
    );
    // Other kinds keep their three blocks of soil.
    assert_eq!(
        g.block_for(77 - SOIL_DEPTH as i32 - 1, 77, SurfaceKind::Grass),
        b.stone
    );
    assert_eq!(
        g.block_for(77 - SOIL_DEPTH as i32, 77, SurfaceKind::Grass),
        b.dirt
    );
}

// --- The biomes ------------------------------------------------------------------------------------------------

#[test]
fn dry_rugged_ground_is_badlands_and_very_rugged_ground_is_eroded_badlands() {
    use Biome::*;
    let of =
        |class, elevation, c, relief| Biome::with_relief(class, elevation, 36.0, c, false, relief);
    for class in [LandClass::Bare, LandClass::Shrub, LandClass::Grass] {
        // A rim at 380 mm and 11 C: r = 380 / 360, so dryness comes from a lower figure below.
        let c = climate(12.0, 250.0);
        assert_eq!(of(class, 800.0, c, RUGGED_M), Badlands, "{class:?}");
        assert_eq!(of(class, 800.0, c, ERODED_M - 1.0), Badlands, "{class:?}");
        assert_eq!(of(class, 800.0, c, ERODED_M), ErodedBadlands, "{class:?}");
        assert_ne!(of(class, 800.0, c, RUGGED_M - 1.0), Badlands, "{class:?}");
        // Hot and dry as well.
        assert_eq!(of(class, 800.0, climate(24.0, 90.0), 500.0), ErodedBadlands);
    }
    // Trees keep the table, rugged or not; so does frozen ground (not dry).
    for (class, elevation, c) in [
        (LandClass::Tree, 800.0, climate(12.0, 250.0)),
        (LandClass::Tree, 1800.0, climate(24.0, 90.0)),
        (LandClass::Shrub, 800.0, climate(-4.0, 30.0)),
    ] {
        assert_eq!(
            of(class, elevation, c, 900.0),
            Biome::with_climate(class, elevation, 36.0, c, false),
            "{class:?} {elevation}"
        );
    }
    // Snow still wins.
    assert_eq!(
        of(LandClass::Bare, 8000.0, climate(12.0, 100.0), 900.0),
        FrozenPeaks
    );
}

#[test]
fn dry_flat_shrub_and_grass_are_desert_or_savanna_by_the_heat() {
    use Biome::*;
    let of = |class, elevation, c| Biome::with_relief(class, elevation, 36.0, c, false, 10.0);
    for class in [LandClass::Shrub, LandClass::Grass] {
        // Very dry (r under 0.5): desert, temperate or hot.
        assert_eq!(of(class, 300.0, climate(12.0, 150.0)), Desert, "{class:?}");
        assert_eq!(of(class, 300.0, climate(24.0, 90.0)), Desert, "{class:?}");
        // Dry but not desert: savanna, hot or temperate (dry savanna such as the Serengeti, r about 1.1).
        assert_eq!(of(class, 300.0, climate(22.0, 500.0)), Savanna, "{class:?}");
        assert_eq!(of(class, 300.0, climate(12.0, 300.0)), Savanna, "{class:?}");
        assert_eq!(
            of(class, 1800.0, climate(12.0, 300.0)),
            SavannaPlateau,
            "{class:?}"
        );
        // Cold dry ground stays as the table has it.
        assert_eq!(of(class, 300.0, climate(4.0, 100.0)), Plains, "{class:?}");
        assert_eq!(of(class, 1800.0, climate(4.0, 100.0)), Meadow, "{class:?}");
    }
    // Dry bare ground is desert, and badlands from 1,000 m when temperate or warmer.
    assert_eq!(of(LandClass::Bare, 300.0, climate(24.0, 6.0)), Desert);
    assert_eq!(of(LandClass::Bare, 1200.0, climate(12.0, 100.0)), Badlands);
    assert_eq!(of(LandClass::Bare, 1200.0, climate(4.0, 100.0)), Desert);
    assert_eq!(
        of(LandClass::Bare, 1200.0, climate(-4.0, 100.0)),
        WindsweptGravellyHills
    );
    assert_eq!(
        of(LandClass::Bare, 2500.0, climate(-4.0, 100.0)),
        StonyPeaks
    );
}

#[test]
fn wet_places_and_a_missing_climate_get_the_old_biomes_whatever_the_relief() {
    let classes = [
        LandClass::Tree,
        LandClass::Shrub,
        LandClass::Grass,
        LandClass::Crop,
        LandClass::Bare,
        LandClass::Wetland,
        LandClass::NoData,
    ];
    for class in classes {
        for elevation in [50.0, 800.0, 1700.0] {
            for lat in [-35.0, 5.0, 36.0, 60.0] {
                for variant in [false, true] {
                    // Wet (T 9 C, P 2,800 mm), and cold wet.
                    for c in [
                        climate(9.0, 2800.0),
                        climate(3.0, 1500.0),
                        climate(26.0, 2400.0),
                    ] {
                        let flat = Biome::with_climate(class, elevation, lat, c, variant);
                        for relief in [0.0, 200.0, 5000.0] {
                            assert_eq!(
                                Biome::with_relief(class, elevation, lat, c, variant, relief),
                                flat,
                                "{class:?} {elevation} {lat}"
                            );
                        }
                    }
                    // No climate: the latitude table, whatever the relief.
                    for relief in [0.0, 5000.0] {
                        assert_eq!(
                            Biome::with_relief(class, elevation, lat, None, variant, relief),
                            Biome::with_climate(class, elevation, lat, None, variant)
                        );
                    }
                }
            }
        }
    }
    // A few fixed wet examples.
    use Biome::*;
    assert_eq!(
        Biome::with_climate(LandClass::Tree, 200.0, 47.9, climate(9.0, 2800.0), false),
        OldGrowthSpruceTaiga
    );
    assert_eq!(
        Biome::with_climate(LandClass::Grass, 200.0, 47.9, climate(9.0, 2800.0), false),
        Plains
    );
    assert_eq!(
        Biome::with_climate(LandClass::Shrub, 200.0, 47.9, climate(9.0, 2800.0), false),
        Plains
    );
}

#[test]
fn eroded_badlands_is_the_last_biome_and_the_old_ids_stay() {
    assert_eq!(Biome::ALL.len(), 40);
    assert_eq!(Biome::ALL[39], Biome::ErodedBadlands);
    assert_eq!(Biome::ALL[38], Biome::FrozenRiver);
    assert_eq!(Biome::ErodedBadlands.name(), "minecraft:eroded_badlands");
    assert!(!Biome::ErodedBadlands.snows());
    for b in [
        Biome::Desert,
        Biome::Badlands,
        Biome::ErodedBadlands,
        Biome::Savanna,
        Biome::SavannaPlateau,
    ] {
        assert!(b.few_plants(), "{b:?}");
    }
    for b in [
        Biome::Plains,
        Biome::Forest,
        Biome::WoodedBadlands,
        Biome::Jungle,
        Biome::Meadow,
    ] {
        assert!(!b.few_plants(), "{b:?}");
    }
}

// --- The plants ------------------------------------------------------------------------------------------------

#[test]
fn grass_allows_dry_features_and_dry_biomes_thin_the_plants_on_shrub_and_grass() {
    for (x, z) in [(0, 0), (-7, 31), (1000, -40)] {
        assert!(allowed(LandClass::Grass, FeatureKind::Dry, x, z));
        assert!(!allowed(LandClass::Grass, FeatureKind::Water, x, z));
    }
    let share = |class, kind, biome| {
        let mut yes = 0;
        for x in -150..150 {
            for z in -100..100 {
                yes += i32::from(allowed_in(class, kind, x, z, biome));
            }
        }
        f64::from(yes) / 60_000.0
    };
    for class in [LandClass::Shrub, LandClass::Grass] {
        for biome in [
            Biome::Desert,
            Biome::Badlands,
            Biome::ErodedBadlands,
            Biome::Savanna,
            Biome::SavannaPlateau,
        ] {
            let s = share(class, FeatureKind::Plant, Some(biome));
            assert!((s - 0.3).abs() < 0.02, "{class:?} {biome:?} {s}");
            // Dry features are not thinned.
            assert_eq!(share(class, FeatureKind::Dry, Some(biome)), 1.0);
        }
        // Wet biomes, no biome: every column.
        for biome in [Some(Biome::Plains), Some(Biome::Forest), None] {
            assert_eq!(
                share(class, FeatureKind::Plant, biome),
                1.0,
                "{class:?} {biome:?}"
            );
        }
    }
    // Other classes do not thin their plants, and the biome-blind form is the plain table.
    assert_eq!(
        share(LandClass::Crop, FeatureKind::Plant, Some(Biome::Desert)),
        1.0
    );
    assert_eq!(
        share(LandClass::Tree, FeatureKind::Plant, Some(Biome::Desert)),
        1.0
    );
    assert!(allowed(LandClass::Grass, FeatureKind::Plant, 3, 3));
}

// --- The generator over a synthetic climate --------------------------------------------------------------------

/// A land-cover rig of one class over `elevation` with a whole-world climate of `t` C and `p` mm, and the
/// generator reading it.
pub(super) fn dry_rig(
    lat: f64,
    elevation: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
    code: u8,
    t: f32,
    p: f32,
) -> (LandRig, Arc<EarthGenerator>, TempDir) {
    let mut rig = LandRig::new(EarthSettings::at(lat, LON0), elevation, move |_, _| code);
    rig.settings.climate = true;
    let (raw_t, raw_p) = (((t + 273.15) * 10.0).round() as u16, p.round() as u16);
    let climate = Arc::new(
        MemoryClimate::new()
            .with_raw(ClimateVar::Temperature, 64, 1, move |_, _, _| raw_t)
            .with_raw(ClimateVar::Precipitation, 64, 1, move |_, _, _| raw_p),
    );
    let dir = TempDir::new();
    let store = Arc::new(ClimateStore::with_grid(dir.path(), 8, SMALL_CLIMATE_GRID));
    let fetcher: Arc<dyn ClimateFetcher> = climate;
    let source = rig.settings.climate_source(store, fetcher).unwrap();
    let generator = Arc::new(
        EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks)
            .unwrap()
            .with_land_cover(rig.source.clone())
            .with_climate(source),
    );
    (rig, generator, dir)
}

/// 1 block of height per block of ground to the north (steep), around `base` metres.
pub(super) fn slope(
    lat0: f64,
    base: f64,
    gradient: f64,
) -> impl Fn(f64, f64) -> f64 + Send + Sync + 'static {
    move |lat, _| base + (lat - lat0) * 111_194.9 * gradient
}

pub(super) fn kind_at(g: &EarthGenerator, x: i32, z: i32) -> (i32, SurfaceKind) {
    let s = g.surface_y_blocking(x, z).unwrap();
    g.kind_core(Access::Fetch, x, z, s, 1).ok().unwrap()
}

#[test]
fn a_steep_bare_wall_in_a_dry_climate_is_level_terracotta_down_to_the_rock_depth() {
    let (rig, g, _dir) = dry_rig(36.0, slope(36.0, 800.0, 1.0), 60, 22.0, 90.0);
    let b = rig.blocks;
    let table = band_table(&b);
    // Two columns on the wall, a few blocks apart: different surfaces, the same band at the same height.
    let (s1, k1) = kind_at(&g, 0, 0);
    let (s2, k2) = kind_at(&g, 0, 10);
    assert_eq!((k1, k2), (SurfaceKind::Banded, SurfaceKind::Banded));
    assert_ne!(s1, s2);
    for (z, s) in [(0, s1), (10, s2)] {
        for y in s - 36..=s + 2 {
            let got = g.block_at_blocking(0, y, z, 1);
            let want = if y > s {
                b.air
            } else if y < s - 32 {
                b.stone
            } else {
                table[band_index(i64::from(y))]
            };
            assert_eq!(got, want, "column 0,{z} surface {s} height {y}");
        }
    }
    // Not the old grey rock: the wall (and its top) is terracotta.
    assert_ne!(g.block_at_blocking(0, s1, 0, 1), b.stone);
    // The cube whose blocks are all well under the surface of flat ground is not proven stone any more.
}

#[test]
fn the_section_fill_shortcut_does_not_hide_the_bands_under_dry_ground() {
    let (rig, g, _dir) = dry_rig(36.0, |_, _| 1500.0, 20, 12.0, 300.0);
    let b = rig.blocks;
    let table = band_table(&b);
    let (s, k) = kind_at(&g, 0, 0);
    assert!(matches!(k, SurfaceKind::Scrub(_)), "{k:?}");
    let y0 = (s - 28).div_euclid(16) * 16;
    let min = [0, y0, 0];
    let max = [15, y0 + 15, 15];
    // The cube holds the bands, so neither world proves it stone (a water rule may carve the ground by
    // MAX_CARVE, and the bands reach DRY_ROCK_DEPTH under it).
    assert_eq!(
        rig.generator.classify_box_blocking(min, max, 1),
        Fill::Mixed
    );
    assert_eq!(g.classify_box_blocking(min, max, 1), Fill::Mixed);
    // A cube under the carve margin is proven stone without climate, but not with it: the dry kinds' bands may
    // reach that deep under a ground that was carved.
    let mid = (s - 60).div_euclid(16) * 16;
    let (mid_min, mid_max) = ([0, mid, 0], [15, mid + 15, 15]);
    assert_eq!(
        rig.generator.classify_box_blocking(mid_min, mid_max, 1),
        Fill::Stone
    );
    assert_eq!(g.classify_box_blocking(mid_min, mid_max, 1), Fill::Mixed);
    let mut out = Box::new([0_u32; 4096]);
    let uniform = g.generate_blocking(SectionPos::new(0, y0.div_euclid(16), 0), &mut out);
    assert_eq!(uniform, None);
    for dy in 0..16 {
        let y = y0 + dy;
        let want = if y < s - 32 {
            b.stone
        } else {
            table[band_index(i64::from(y))]
        };
        assert_eq!(out[(dy as usize) << 8 | 5 << 4 | 7], want, "height {y}");
    }
    // Well below the bands it is stone again.
    let deep = (s - 80).div_euclid(16) * 16;
    assert_eq!(
        g.classify_box_blocking([0, deep, 0], [15, deep + 15, 15], 1),
        Fill::Stone
    );
}

#[test]
fn a_dry_scrub_ground_is_sparse_and_its_columns_are_what_the_far_view_shows() {
    let (rig, g, _dir) = dry_rig(36.0, |_, _| 1500.0, 20, 12.0, 100.0);
    let b = rig.blocks;
    let mut tops = std::collections::BTreeSet::new();
    for x in 0..40 {
        for z in 0..10 {
            let (s, k) = kind_at(&g, x, z);
            assert!(matches!(k, SurfaceKind::Scrub(_)), "{k:?}");
            let top = g.block_at_blocking(x, s, z, 1);
            assert!([b.grass_block, b.coarse_dirt, b.red_sand].contains(&top));
            tops.insert(top);
            // No dirt under it: terracotta.
            assert_ne!(g.block_at_blocking(x, s - 1, z, 1), b.dirt);
        }
    }
    assert_eq!(tops.len(), 3, "all three tops are used");
    // The far view reads the same kinds: its voxel of a column is the block the near rule gives.
    let cols = loop {
        match g.columns_at(0, 0, 4, 4) {
            Ok(cols) => break cols,
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
    };
    for i in [0, 5, 17, 100, 255] {
        assert!(matches!(cols.kind(i), SurfaceKind::Scrub(_)));
        let (x, z) = cols.block_of(i);
        assert_eq!(
            cols.kind(i),
            kind_at(&g, x, z).1,
            "near and far kinds of column {i}"
        );
        let s = cols.surfaces[i];
        for lo in [s - 40, s - 12, s - 3, s, s + 4] {
            let lo = EarthGenerator::voxel_floor(lo, 4);
            assert_eq!(
                g.far_voxel(&cols, i, lo),
                g.block_at_blocking(x, lo, z, 4),
                "column {i} voxel {lo}"
            );
        }
    }
}

#[test]
fn a_wet_climate_or_none_makes_the_old_terrain_block_for_block() {
    let (rig, wet, _dir) = dry_rig(45.0, super::tests::terrain, 20, 9.0, 2800.0);
    // The same world with a climate source but no data is today's terrain too: a plain land-cover generator.
    let plain = &rig.generator;
    let mut checked = 0;
    for cx in [-3, 0, 20, 70, 100, 150] {
        for cz in [-1, 0] {
            let s = plain.surface_y_blocking(cx * 16 + 8, cz * 16 + 8).unwrap();
            for cy in (s >> 4) - 3..=(s >> 4) + 1 {
                let mut want = Box::new([0_u32; 4096]);
                let mut got = Box::new([0_u32; 4096]);
                let a = plain.generate_blocking(SectionPos::new(cx, cy, cz), &mut want);
                let b = wet.generate_blocking(SectionPos::new(cx, cy, cz), &mut got);
                assert_eq!(&want[..], &got[..], "cube {cx},{cy},{cz}");
                assert_eq!(a, b);
                checked += 1;
            }
        }
    }
    assert!(checked > 30);
    // Bare ground too: steep stays stone, flat stays sand.
    let (rig, wet, _dir) = dry_rig(45.0, slope(45.0, 500.0, 1.0), 60, 9.0, 2800.0);
    for z in [0, 40] {
        let s = rig.generator.surface_y_blocking(0, z).unwrap();
        for y in s - 40..=s + 1 {
            assert_eq!(
                wet.block_at_blocking(0, y, z, 1),
                rig.generator.block_at_blocking(0, y, z, 1),
                "{y}"
            );
        }
        assert_eq!(wet.block_at_blocking(0, s, z, 1), rig.blocks.stone);
    }
}

#[test]
fn the_biome_of_a_dry_canyon_wall_is_eroded_badlands_and_of_a_dry_plain_is_desert() {
    // 1 m of height per metre of ground: the neighbours half a 500 m cell away differ by 500 m.
    let (_rig, wall, _dir) = dry_rig(36.0, slope(36.0, 900.0, 1.0), 60, 14.0, 120.0);
    assert_eq!(wall.biome_at_quart_blocking(0, 0), Biome::ErodedBadlands);
    // 0.5 m per metre: a relief of 250 m is rugged but not eroded.
    let (_rig, hills, _dir) = dry_rig(36.0, slope(36.0, 900.0, 0.5), 60, 14.0, 120.0);
    assert_eq!(hills.biome_at_quart_blocking(0, 0), Biome::Badlands);
    // A gentle slope: desert bare ground (lower than 1,000 m).
    let (_rig, flat, _dir) = dry_rig(36.0, slope(36.0, 300.0, 0.1), 60, 14.0, 120.0);
    assert_eq!(flat.biome_at_quart_blocking(0, 0), Biome::Desert);
    // Shrub on a flat dry plateau: savanna in a temperate climate that is not desert dry.
    let (_rig, shrub, _dir) = dry_rig(36.0, |_, _| 400.0, 20, 14.0, 330.0);
    assert_eq!(shrub.biome_at_quart_blocking(0, 0), Biome::Savanna);
    // A wet world with the same wall: the old biome, rugged or not.
    let (_rig, wet, _dir) = dry_rig(36.0, slope(36.0, 900.0, 1.0), 20, 9.0, 2800.0);
    let want = Biome::with_climate(LandClass::Shrub, 900.0, 36.0, climate(9.0, 2800.0), false);
    assert_eq!(wet.biome_at_quart_blocking(0, 0), want);
}

#[test]
fn the_far_view_is_ready_for_dry_biomes_only_when_the_neighbours_are_loaded() {
    let (rig, g, _dir) = dry_rig(36.0, slope(36.0, 900.0, 1.0), 60, 14.0, 120.0);
    let mut rounds = 0;
    loop {
        match g.biomes_ready(0, 0, 1) {
            Ok(()) => break,
            Err(missing) => {
                rounds += 1;
                assert!(rounds < 50, "never ready");
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
    assert!(rounds > 0);
    // Ready means the local read answers without fetching.
    assert_eq!(
        g.biome_core(Access::Local, 0, 0).ok(),
        Some(Biome::ErodedBadlands)
    );
}

/// The columns of the far read at `(x, z)` with `step`, fetching what it lacks.
fn far_columns(
    rig: &LandRig,
    g: &EarthGenerator,
    x: i32,
    z: i32,
    step: i32,
) -> super::far::NodeColumns {
    loop {
        match g.columns_at(x, z, step, step) {
            Ok(cols) => return cols,
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
}

#[test]
fn grass_and_forest_never_get_dry_patch_blocks_near_or_far() {
    // Every dry climate the rules tell apart: desert and scrub, hot and cool.
    for (t, r) in [(10.0, 1.2), (10.0, 0.3), (25.0, 0.9), (25.0, 0.1)] {
        for bucket in [0, 100, 450, 700, 999] {
            let dry = col(t, r, bucket);
            for class in [LandClass::Grass, LandClass::Tree] {
                assert_eq!(
                    kind(class, 36.0, dry, NO_SLOPE),
                    SurfaceKind::Grass,
                    "{class:?} {t} {r}"
                );
            }
        }
    }
    // Through the generator: grassland (30) and forest (10) in the South Rim's climate, near and at far steps.
    for code in [30, 10] {
        let (rig, g, _dir) = dry_rig(36.0, |_, _| 2100.0, code, 10.0, 430.0);
        let b = rig.blocks;
        for x in 0..24 {
            for z in 0..8 {
                let (s, k) = kind_at(&g, x, z);
                assert_eq!(k, SurfaceKind::Grass, "class {code} at {x},{z}");
                assert_eq!(g.block_at_blocking(x, s, z, 1), b.grass_block);
            }
        }
        for step in [2, 8, 32] {
            let cols = far_columns(&rig, &g, 0, 0, step);
            for i in [0, 9, 77, 200, 255] {
                assert_eq!(cols.kind(i), SurfaceKind::Grass, "class {code} step {step}");
                let s = cols.surfaces[i];
                let top = EarthGenerator::far_surface(s, step, SurfaceKind::Grass);
                let lo = EarthGenerator::voxel_floor(top, step);
                assert_eq!(g.far_voxel(&cols, i, lo), b.grass_block);
            }
        }
    }
}

#[test]
fn dry_shrub_is_a_fine_mix_block_by_block_not_patches() {
    let climate = climate(10.0, 430.0).unwrap();
    let top = |x: i32, z: i32| match kind_of(LandClass::Shrub, climate, x, z) {
        SurfaceKind::Scrub(t) => t,
        other => panic!("{other:?}"),
    };
    // Neighbouring blocks differ about as often as three shares mixed at random do (1 - 0.4² - 0.4² - 0.2² =
    // 0.64): no run of the same block over a land-cover pixel.
    let (mut pairs, mut differ) = (0, 0);
    for x in 0..120 {
        for z in 0..120 {
            pairs += 2;
            differ +=
                usize::from(top(x, z) != top(x + 1, z)) + usize::from(top(x, z) != top(x, z + 1));
        }
    }
    let share = differ as f64 / pairs as f64;
    assert!((share - 0.64).abs() < 0.03, "{share}");
    // Every 4 by 4 square of blocks, wherever it is, holds more than one block, and no 30-block pixel is one colour.
    for x0 in (-64..64).step_by(4) {
        for z0 in (-64..64).step_by(4) {
            let first = top(x0, z0);
            assert!(
                (0..4).any(|dx| (0..4).any(|dz| top(x0 + dx, z0 + dz) != first)),
                "square at {x0},{z0}"
            );
        }
    }
}

/// The kind of a gentle column of `class` at block `(x, z)` with `climate`.
fn kind_of(class: LandClass, climate: Climate, x: i32, z: i32) -> SurfaceKind {
    SurfaceKind::on_classified(
        class,
        1200.0,
        36.0,
        false,
        Some(DryColumn::new(climate, x, z)),
        [x, z],
        NO_SLOPE,
    )
    .unwrap()
}

#[test]
fn far_band_colour_depends_on_the_height_alone_and_stays_level_at_every_step() {
    let rig = LandRig::level(36.0, 1500.0, 60);
    let g = &rig.generator;
    let b = rig.blocks;
    let table = band_table(&b);
    let kinds = [
        SurfaceKind::Banded,
        SurfaceKind::Scrub(DryTop::GrassBlock),
        SurfaceKind::Scrub(DryTop::RedSand),
        SurfaceKind::Barren(DryTop::CoarseDirt),
    ];
    for level in 1..=6 {
        let step = 1 << level;
        for lo in (-256..256).step_by(step as usize) {
            let want = table[band_index(i64::from(lo) + i64::from(step / 2))];
            // Columns whose surfaces lie anywhere in this voxel or up to the rock depth over it (a wall seen from
            // the side): the banded voxel is the same block for each, so a row of voxels is one level band.
            // (A surface in the voxel's lower half is drawn in the voxel under it: `far_surface`.)
            let first = lo + if step > 2 { step / 2 - 1 } else { 0 };
            for s in first..lo + step + DRY_ROCK_DEPTH as i32 - step {
                for kind in kinds {
                    if EarthGenerator::far_surface(s, step, kind) < lo + step
                        && !matches!(kind, SurfaceKind::Banded)
                    {
                        continue;
                    }
                    assert_eq!(
                        g.voxel_block(lo, step, s, kind),
                        want,
                        "step {step} voxel {lo} surface {s} {kind:?}"
                    );
                }
            }
            // Where a voxel lies within one band, it is the band the near blocks have.
            if step <= BAND_BLOCKS as i32 {
                for y in lo..lo + step {
                    assert_eq!(want, g.block_for(y, lo + step + 1, SurfaceKind::Banded));
                }
            }
        }
        // The top voxel of a mixed dry top is the mix's main block whichever block the column picked.
        for t in [DryTop::GrassBlock, DryTop::CoarseDirt, DryTop::RedSand] {
            let s = 100;
            let top = EarthGenerator::far_surface(s, step, SurfaceKind::Scrub(t));
            let lo = EarthGenerator::voxel_floor(top, step);
            assert_eq!(
                g.voxel_block(lo, step, s, SurfaceKind::Scrub(t)),
                b.coarse_dirt
            );
            assert_eq!(
                g.voxel_block(lo, step, s, SurfaceKind::Barren(t)),
                b.red_sand
            );
        }
    }
    assert_eq!(
        (super::dry::SCRUB_FAR, super::dry::BARREN_FAR),
        (DryTop::CoarseDirt, DryTop::RedSand)
    );
}

/// The far ground lies within half a voxel of the real one: the voxel holding a land surface is ground only when
/// the ground fills at least half of it. A voxel drawn whole for a shore a block over the sea stood a whole voxel
/// tall out of the water (Lyle's floating sand box).
#[test]
fn far_land_stands_at_most_half_a_voxel_above_the_ground() {
    let (rig, g, _dir) = dry_rig(36.0, |_, _| 2100.0, 30, 10.0, 430.0);
    let b = rig.blocks;
    for step in [2, 4, 8, 16, 32, 64] {
        for s in -70..200 {
            for kind in [SurfaceKind::Grass, SurfaceKind::Dune, SurfaceKind::Gravel] {
                let top = EarthGenerator::far_surface(s, step, kind);
                let lo = EarthGenerator::voxel_floor(top, step);
                // The voxel drawn as the top is the surface block; the one over it is air.
                assert_eq!(
                    g.voxel_block(lo, step, s, kind),
                    g.block_for(s, s, kind),
                    "{kind:?} s {s} step {step}"
                );
                assert_eq!(g.voxel_block(lo + step, step, s, kind), b.air);
                // Its top face is within half a voxel of the real top face (s + 1).
                let face = lo + step;
                assert!(
                    (face - (s + 1)).abs() <= step / 2,
                    "{kind:?} s {s} step {step}: face {face}"
                );
            }
            // Water keeps its level: the far water is the near water's top.
            let (top, sea) = SurfaceKind::sea_column(s.min(60), 63);
            assert_eq!(EarthGenerator::far_surface(top, step, sea), top);
        }
    }
}

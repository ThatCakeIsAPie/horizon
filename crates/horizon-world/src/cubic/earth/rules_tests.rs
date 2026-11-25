//! Tests of the general surface rules on made-up data: steep snow is rock, snow/ice is a glacier, wet bare ground
//! is basalt ([`super::surface`]). No network, no real data.

use horizon_earth::LandClass;

use super::dry::{BASALT_MIX, BasaltTop, DryColumn};
use super::dry_tests::{dry_rig, kind_at, slope};
use super::land_tests::LandRig;
use super::surface::{
    GLACIER_ICE_DEPTH, GLACIER_SNOW_DEPTH, SNOW_STEEP_GRADIENT, STEEP_GRADIENT, Slope, SurfaceKind,
};
use super::tests::LON0;
use super::*;

const SNOW_ICE: u8 = 70;
const BARE: u8 = 60;

fn on(
    class: LandClass,
    metres: f64,
    lat: f64,
    dry: Option<DryColumn>,
    slope: Slope,
) -> SurfaceKind {
    SurfaceKind::on_classified(class, metres, lat, false, dry, [0, 0], || {
        Ok::<_, ()>(slope)
    })
    .unwrap()
}

// --- The kinds -------------------------------------------------------------------------------------------------

#[test]
fn the_slope_classes_follow_the_two_gradients() {
    assert_eq!((STEEP_GRADIENT, SNOW_STEEP_GRADIENT), (0.6, 0.84));
    assert_eq!(Slope::of_gradient(0.0), Slope::Gentle);
    assert_eq!(Slope::of_gradient(0.59), Slope::Gentle);
    assert_eq!(Slope::of_gradient(0.6), Slope::Steep);
    assert_eq!(Slope::of_gradient(0.83), Slope::Steep);
    assert_eq!(Slope::of_gradient(0.84), Slope::Rock);
    assert!(!Slope::Gentle.is_steep() && Slope::Steep.is_steep() && Slope::Rock.is_steep());
}

#[test]
fn snow_above_the_snow_line_is_stone_on_rock_steep_ground_and_snow_otherwise() {
    // 6,000 m at 30 degrees is over the snow line (5,500 m); the map calls it bare, not snow.
    for class in [LandClass::Bare, LandClass::Grass, LandClass::NoData] {
        assert_eq!(
            on(class, 6000.0, 30.0, None, Slope::Rock),
            SurfaceKind::Alpine,
            "{class:?}"
        );
        for gentle in [Slope::Gentle, Slope::Steep] {
            assert_eq!(on(class, 6000.0, 30.0, None, gentle), SurfaceKind::Snow);
        }
    }
    // A frozen lake is flat whatever is around it.
    assert_eq!(
        on(LandClass::Water, 4000.0, 45.0, None, Slope::Rock),
        SurfaceKind::Snow
    );
}

#[test]
fn snow_ice_class_is_a_glacier_when_gentle_an_icefall_when_steeper_and_stone_on_rock() {
    assert_eq!(
        on(LandClass::SnowIce, 3000.0, 45.0, None, Slope::Gentle),
        SurfaceKind::Glacier
    );
    assert_eq!(
        on(LandClass::SnowIce, 3000.0, 45.0, None, Slope::Steep),
        SurfaceKind::Icefall
    );
    assert_eq!(
        on(LandClass::SnowIce, 3000.0, 45.0, None, Slope::Rock),
        SurfaceKind::Alpine
    );
    // Also low down at the poles, and at the top of the world.
    assert_eq!(
        on(LandClass::SnowIce, 20.0, 78.0, None, Slope::Gentle),
        SurfaceKind::Glacier
    );
    assert_eq!(
        on(LandClass::SnowIce, 8849.0, 27.99, None, Slope::Gentle),
        SurfaceKind::Glacier
    );
}

#[test]
fn bare_wet_ground_is_basalt_dry_hot_ground_is_sand_and_no_climate_is_unchanged() {
    let wet = Some(DryColumn::of(24.0, 6.0, 10));
    assert_eq!(
        on(LandClass::Bare, 300.0, 20.0, wet, Slope::Gentle),
        SurfaceKind::Basalt(BasaltTop::Basalt)
    );
    // Cold and wet is not dry either; the hash picks the top.
    let cold = Some(DryColumn::of(-3.0, 6.0, 999));
    assert_eq!(
        on(LandClass::Bare, 300.0, 20.0, cold, Slope::Gentle),
        SurfaceKind::Basalt(BasaltTop::Blackstone)
    );
    // Dry and hot: sand. Dry and cool: barren red ground, as before.
    let desert = Some(DryColumn::of(25.0, 0.2, 10));
    assert_eq!(
        on(LandClass::Bare, 300.0, 20.0, desert, Slope::Gentle),
        SurfaceKind::Dune
    );
    assert!(matches!(
        on(
            LandClass::Bare,
            300.0,
            36.0,
            Some(DryColumn::of(12.0, 0.2, 10)),
            Slope::Gentle
        ),
        SurfaceKind::Barren(_)
    ));
    // No climate data: sand, as before.
    assert_eq!(
        on(LandClass::Bare, 300.0, 20.0, None, Slope::Gentle),
        SurfaceKind::Dune
    );
    // Steep wet stays stone; cold latitude stays gravel; the alpine band stays stone.
    assert_eq!(
        on(LandClass::Bare, 300.0, 20.0, wet, Slope::Steep),
        SurfaceKind::Alpine
    );
    assert_eq!(
        on(LandClass::Bare, 300.0, 55.0, wet, Slope::Gentle),
        SurfaceKind::Gravel
    );
    assert_eq!(
        on(LandClass::Bare, 4500.0, 20.0, wet, Slope::Gentle),
        SurfaceKind::Alpine
    );
    // Other classes in a wet climate are untouched.
    assert_eq!(
        on(LandClass::Grass, 300.0, 20.0, wet, Slope::Gentle),
        SurfaceKind::Grass
    );
}

#[test]
fn the_basalt_mix_picks_by_the_hash_of_the_column() {
    let mut counts = [0_u32; 4];
    for bucket in 0..1000 {
        counts[match BASALT_MIX.pick(bucket) {
            BasaltTop::Basalt => 0,
            BasaltTop::SmoothBasalt => 1,
            BasaltTop::Tuff => 2,
            BasaltTop::Blackstone => 3,
        }] += 1;
    }
    assert_eq!(counts, [700, 140, 90, 70]);
}

// --- The blocks ------------------------------------------------------------------------------------------------

/// The column at `(0, 0)` of `rig` has snow on top, then packed ice, then stone.
#[test]
fn a_gentle_snow_ice_column_is_a_glacier_snow_over_packed_ice_over_stone() {
    let rig = LandRig::level(45.0, 3000.0, SNOW_ICE);
    let b = rig.blocks;
    rig.check(0, 0, |y, s| {
        if y > s {
            b.air
        } else if y > s - GLACIER_SNOW_DEPTH as i32 {
            b.snow_block
        } else if y >= s - GLACIER_ICE_DEPTH as i32 {
            b.packed_ice
        } else {
            b.stone
        }
    });
}

#[test]
fn a_moderately_steep_snow_ice_column_is_an_icefall_packed_ice_over_blue_ice() {
    // 0.7 m of height per metre of ground: steeper than bare ground allows, not rock-steep.
    let rig = LandRig::new(
        EarthSettings::at(45.0, LON0),
        slope(45.0, 3000.0, 0.7),
        |_, _| SNOW_ICE,
    );
    let b = rig.blocks;
    rig.check(0, 0, |y, s| {
        if y > s {
            b.air
        } else if y == s {
            b.packed_ice
        } else if y >= s - GLACIER_ICE_DEPTH as i32 {
            b.blue_ice
        } else {
            b.stone
        }
    });
}

#[test]
fn a_rock_steep_snow_ice_column_is_stone_to_the_top() {
    let rig = LandRig::new(
        EarthSettings::at(45.0, LON0),
        slope(45.0, 3000.0, 1.2),
        |_, _| SNOW_ICE,
    );
    let b = rig.blocks;
    rig.check(0, 0, |y, s| if y <= s { b.stone } else { b.air });
}

#[test]
fn bare_ground_above_the_snow_line_is_snow_when_gentle_and_stone_when_steep() {
    // 4,500 m at 45 degrees is over the snow line (3,850 m), and the map says bare.
    let flat = LandRig::level(45.0, 4500.0, BARE);
    let b = flat.blocks;
    flat.check(0, 0, |y, s| {
        if y > s {
            b.air
        } else if y >= s - 1 {
            b.snow_block
        } else {
            b.stone
        }
    });
    let wall = LandRig::new(
        EarthSettings::at(45.0, LON0),
        slope(45.0, 4500.0, 1.2),
        |_, _| BARE,
    );
    wall.check(0, 0, |y, s| if y <= s { b.stone } else { b.air });
    // A middle slope keeps its snow.
    let mid = LandRig::new(
        EarthSettings::at(45.0, LON0),
        slope(45.0, 4500.0, 0.7),
        |_, _| BARE,
    );
    mid.check(0, 0, |y, s| {
        if y > s {
            b.air
        } else if y >= s - 1 {
            b.snow_block
        } else {
            b.stone
        }
    });
}

#[test]
fn bare_ground_in_a_wet_climate_is_basalt_rock_and_in_a_dry_hot_one_sand() {
    let (rig, wet, _dir) = dry_rig(10.0, |_, _| 300.0, BARE, 25.0, 2500.0);
    let b = rig.blocks;
    let mut tops = std::collections::BTreeSet::new();
    for x in 0..40 {
        let (s, kind) = kind_at(&wet, x, 0);
        assert!(matches!(kind, SurfaceKind::Basalt(_)), "{kind:?}");
        let top = wet.block_at_blocking(x, s, 0, 1);
        assert!(
            [b.basalt, b.smooth_basalt, b.tuff, b.blackstone].contains(&top),
            "{top}"
        );
        tops.insert(top);
        assert_eq!(wet.block_at_blocking(x, s - 1, 0, 1), b.stone);
        assert_eq!(wet.block_at_blocking(x, s - 5, 0, 1), b.stone);
        assert_eq!(wet.block_at_blocking(x, s + 1, 0, 1), b.air);
        // The far view reads the same blocks.
        assert_eq!(wet.voxel_block(s, 1, s, kind), top);
        assert_eq!(wet.voxel_block(s - 3, 4, s, kind), top);
    }
    assert!(tops.contains(&b.basalt) && tops.len() > 1, "{tops:?}");
    // Dry and hot: sand dunes.
    let (rig, dry, _dir) = dry_rig(10.0, |_, _| 300.0, BARE, 25.0, 60.0);
    let (s, kind) = kind_at(&dry, 0, 0);
    assert_eq!(kind, SurfaceKind::Dune);
    assert_eq!(dry.block_at_blocking(0, s, 0, 1), rig.blocks.sand);
    // No climate at all (a land-cover rig): sand, as before.
    let rig = LandRig::level(10.0, 300.0, BARE);
    let b = rig.blocks;
    rig.check(0, 0, |y, s| {
        if y > s {
            b.air
        } else if y > s - 4 {
            b.sand
        } else {
            b.stone
        }
    });
}

#[test]
fn a_steep_bare_wet_wall_stays_stone() {
    let (rig, wet, _dir) = dry_rig(10.0, slope(10.0, 300.0, 1.0), BARE, 25.0, 2500.0);
    let (s, kind) = kind_at(&wet, 0, 0);
    assert_eq!(kind, SurfaceKind::Alpine);
    assert_eq!(wet.block_at_blocking(0, s, 0, 1), rig.blocks.stone);
}

// --- Built-up ground and moss -----------------------------------------------------------------------------------

#[test]
fn built_up_ground_is_a_patchwork_of_parcels_with_a_quarter_of_parks() {
    use super::cover::{CityTop, PARCEL_BLOCKS, city_top};
    use std::collections::HashMap;
    let mut counts: HashMap<CityTop, u32> = HashMap::new();
    let n = 60;
    for pz in -n..n {
        for px in -n..n {
            let (x, z) = (px * PARCEL_BLOCKS, pz * PARCEL_BLOCKS);
            let top = city_top(x, z, false);
            // One parcel shares one top; the same column always gets the same one.
            assert_eq!(city_top(x + PARCEL_BLOCKS - 1, z + 3, false), top);
            assert_eq!(city_top(x, z, false), top);
            *counts.entry(top).or_default() += 1;
            // A dry climate turns the yards to sand and nothing else.
            let dry = city_top(x, z, true);
            if top == CityTop::Grass {
                assert_eq!(dry, CityTop::Sand);
            } else {
                assert_eq!(dry, top);
            }
        }
    }
    let total = (2 * n * 2 * n) as f64;
    assert!(counts.len() >= 6, "{counts:?}");
    assert!(!counts.contains_key(&CityTop::Sand));
    let grass = f64::from(counts[&CityTop::Grass]) / total;
    assert!((0.18..=0.32).contains(&grass), "{grass}");
    // No one grey covers most of the ground: the far view is not one flat colour.
    assert!(
        counts.values().all(|&c| f64::from(c) / total < 0.3),
        "{counts:?}"
    );
    // The class gives that kind, wet or dry.
    let wet = DryColumn::of(12.0, 3.0, 0);
    let dry = DryColumn::of(25.0, 0.3, 0);
    for (x, z) in [(0, 0), (17, -40), (800, 333)] {
        let at = |d| {
            SurfaceKind::on_classified(LandClass::BuiltUp, 50.0, 40.0, false, d, [x, z], || {
                Ok::<_, ()>(Slope::Gentle)
            })
            .unwrap()
        };
        assert_eq!(at(Some(wet)), SurfaceKind::BuiltUp(city_top(x, z, false)));
        assert_eq!(at(None), SurfaceKind::BuiltUp(city_top(x, z, false)));
        assert_eq!(at(Some(dry)), SurfaceKind::BuiltUp(city_top(x, z, true)));
    }
}

#[test]
fn moss_and_lichen_is_mostly_moss_with_some_grass_and_bare_ground() {
    use super::cover::{MossTop, moss_top};
    use std::collections::HashMap;
    let mut counts: HashMap<MossTop, u32> = HashMap::new();
    for z in 0..100 {
        for x in 0..100 {
            let top = moss_top(x, z);
            assert_eq!(moss_top(x, z), top);
            *counts.entry(top).or_default() += 1;
        }
    }
    assert_eq!(
        on(LandClass::MossLichen, 300.0, 64.0, None, Slope::Gentle),
        SurfaceKind::Moss(moss_top(0, 0))
    );
    let share = |t| f64::from(counts.get(&t).copied().unwrap_or(0)) / 10_000.0;
    assert!((0.45..=0.65).contains(&share(MossTop::Moss)), "{counts:?}");
    assert!((0.08..=0.25).contains(&share(MossTop::Grass)), "{counts:?}");
    let bare = share(MossTop::Stone) + share(MossTop::Gravel) + share(MossTop::CoarseDirt);
    assert!((0.2..=0.4).contains(&bare), "{counts:?}");
}

#[test]
fn street_trees_grow_only_on_grass_parcels() {
    use super::cover::{CityTop, city_top};
    use super::vegetation::{FeatureKind, allowed};
    let (mut trees, mut grass) = (0, 0);
    for z in 0..200 {
        for x in 0..200 {
            let on_grass = city_top(x, z, false) == CityTop::Grass;
            grass += u32::from(on_grass);
            if allowed(LandClass::BuiltUp, FeatureKind::Tree, x, z) {
                assert!(on_grass, "{x} {z}");
                trees += 1;
            }
            assert!(!allowed(LandClass::BuiltUp, FeatureKind::Plant, x, z));
        }
    }
    let share = f64::from(trees) / f64::from(grass);
    assert!((0.2..=0.4).contains(&share), "{share}");
}

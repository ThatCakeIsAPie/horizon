//! Tests of the far view's trees and snow over synthetic level ground (no network): where they grow, how thick,
//! of which kind, when a node is not ready, and that without them nothing changes.

use std::collections::HashMap;
use std::sync::Arc;

use horizon_core::position::SectionPos;
use horizon_earth::testing::{MemoryClimate, SMALL_CLIMATE_GRID, TempDir};
use horizon_earth::{ClimateFetcher, ClimateStore, ClimateVar};

use super::far_trees::{FarDecor, Species, style};
use super::land_tests::LandRig;
use super::*;

const TREES: FarDecor = FarDecor {
    trees: true,
    snow: false,
};
const SNOWY: FarDecor = FarDecor {
    trees: true,
    snow: true,
};

/// The WorldCover codes of the classes the tests use.
const TREE: u8 = 10;
const SHRUB: u8 = 20;
const GRASS: u8 = 30;
const CROP: u8 = 40;
const BUILT: u8 = 50;
const BARE: u8 = 60;
const SNOW_ICE: u8 = 70;
const WATER: u8 = 80;
const WETLAND: u8 = 90;
const MANGROVE: u8 = 95;

/// A rig and the generator the far view reads it with (the rig's own, or one with a climate), and what it takes to
/// have the data a read lacks, as the downloader does.
struct World<'a> {
    rig: &'a LandRig,
    g: &'a EarthGenerator,
}

impl World<'_> {
    /// `read` once the tiles, land cover and climate it lacks are loaded.
    fn ready<T>(&self, read: impl Fn(&EarthGenerator) -> Result<T, NotReady>) -> T {
        for _ in 0..300 {
            match read(self.g) {
                Ok(value) => return value,
                Err(missing) => {
                    for &id in &missing.tiles {
                        self.rig.store.load(id).unwrap();
                    }
                    for fetch in self.g.take_land_requests() {
                        self.rig.source.satisfy(&fetch).unwrap();
                    }
                    for fetch in self.g.take_climate_requests() {
                        self.g.climate_source().unwrap().satisfy(&fetch).unwrap();
                    }
                }
            }
        }
        panic!("a read never became ready");
    }

    fn ground(&self) -> i32 {
        self.rig.generator.surface_y_blocking(8, 8).unwrap()
    }
}

fn plain(rig: &LandRig) -> World<'_> {
    World {
        rig,
        g: &rig.generator,
    }
}

/// A rig of `class` over level ground of `metres` at `lat` with a whole-world climate of `t` C and 900 mm, and
/// the generator that reads it.
fn climate_world(
    lat: f64,
    metres: f64,
    class: u8,
    t: f32,
) -> (LandRig, Arc<EarthGenerator>, TempDir) {
    let mut rig = LandRig::level(lat, metres, class);
    rig.settings.climate = true;
    let raw_t = ((t + 273.15) * 10.0).round() as u16;
    let climate = Arc::new(
        MemoryClimate::new()
            .with_raw(ClimateVar::Temperature, 64, 1, move |_, _, _| raw_t)
            .with_raw(ClimateVar::Precipitation, 64, 1, move |_, _, _| 900),
    );
    let dir = TempDir::new();
    let store = Arc::new(ClimateStore::with_grid(dir.path(), 8, SMALL_CLIMATE_GRID));
    let fetcher: Arc<dyn ClimateFetcher> = climate;
    let source = rig.settings.climate_source(store, fetcher).unwrap();
    let g = Arc::new(
        EarthGenerator::new(&rig.settings, rig.store.clone(), rig.blocks)
            .unwrap()
            .with_land_cover(rig.source.clone())
            .with_climate(source),
    );
    (rig, g, dir)
}

/// Square blocks of cubes of the far view over `n` by `n` cubes, from the ground's cube up to the sky.
struct Scan {
    n: i32,
    y0: i32,
    cubes: Vec<Box<[u32; 4096]>>,
    blocks: EarthBlocks,
}

impl Scan {
    /// The cubes `0..n` by `0..n` and `cy0..=cy1` as the far view makes them with `decor`.
    fn of(w: &World, n: i32, cy0: i32, cy1: i32, decor: FarDecor) -> Self {
        let mut cubes = Vec::new();
        for cy in cy0..=cy1 {
            for cz in 0..n {
                for cx in 0..n {
                    let at = SectionPos::new(cx, cy, cz);
                    cubes.push(w.ready(|g| {
                        let mut out = Box::new([0u32; 4096]);
                        g.generate(at, &mut out)?;
                        g.decorate_cube(at, &mut out, decor)?;
                        Ok(out)
                    }));
                }
            }
        }
        Self {
            n,
            y0: cy0,
            cubes,
            blocks: w.rig.blocks,
        }
    }

    /// Ground level cubes: from the one under the surface of the rig to three above.
    fn ground(w: &World, n: i32, decor: FarDecor) -> Self {
        let s = w.ground();
        Self::of(w, n, (s >> 4) - 1, (s >> 4) + 3, decor)
    }

    fn at(&self, x: i32, y: i32, z: i32) -> u32 {
        let (cx, cy, cz) = (x >> 4, (y >> 4) - self.y0, z >> 4);
        let cube = (cy * self.n + cz) * self.n + cx;
        self.cubes[cube as usize][((y & 15) << 8 | (z & 15) << 4 | (x & 15)) as usize]
    }

    fn side(&self) -> i32 {
        self.n * 16
    }

    fn top_y(&self) -> i32 {
        (self.y0 + self.cubes.len() as i32 / (self.n * self.n)) * 16 - 1
    }

    /// How many blocks of each id there are.
    fn counts(&self) -> HashMap<u32, usize> {
        let mut out = HashMap::new();
        for cube in &self.cubes {
            for &b in cube.iter() {
                *out.entry(b).or_default() += 1;
            }
        }
        out
    }

    fn count_of(&self, ids: &[u32]) -> usize {
        let counts = self.counts();
        ids.iter()
            .map(|id| counts.get(id).copied().unwrap_or(0))
            .sum()
    }

    /// The highest block in the column that is not air, and its height.
    fn top(&self, x: i32, z: i32) -> (i32, u32) {
        (self.y0 * 16..=self.top_y())
            .rev()
            .map(|y| (y, self.at(x, y, z)))
            .find(|&(_, b)| b != self.blocks.air)
            .unwrap()
    }

    /// How many trunks: columns with a log just over the ground.
    fn trunks(&self, ground: i32) -> usize {
        let logs = &self.blocks.far.logs;
        (0..self.side())
            .flat_map(|z| (0..self.side()).map(move |x| (x, z)))
            .filter(|&(x, z)| logs.contains(&self.at(x, ground + 1, z)))
            .count()
    }
}

fn leaves_of(rig: &LandRig, species: Species) -> u32 {
    rig.blocks.far.leaves[species as usize]
}

/// A far node of `level` over the rig's level ground, with `decor` made, and the block of each voxel from the
/// ground voxel up, per column.
fn node_tops(w: &World, level: u32, x: i32, z: i32, decor: FarDecor) -> Vec<Vec<u32>> {
    let cols = w.ready(|g| {
        let mut cols = g.node_columns(level, x, z)?;
        g.decorate_columns(&mut cols, decor)?;
        Ok(cols)
    });
    let step = 1i32 << level;
    (0..256)
        .map(|i| {
            let first = EarthGenerator::voxel_floor(cols.surfaces[i], step);
            (0..40)
                .map(|v| w.g.far_voxel(&cols, i, first + v * step))
                .collect()
        })
        .collect()
}

/// The columns of `tops` that have a voxel of one of `ids` over the ground voxel.
fn columns_with(tops: &[Vec<u32>], ids: &[u32]) -> usize {
    tops.iter()
        .filter(|c| c.iter().any(|b| ids.contains(b)))
        .count()
}

fn all_leaves(rig: &LandRig) -> Vec<u32> {
    rig.blocks.far.leaves.to_vec()
}

#[test]
fn trees_stand_on_tree_cover_with_their_canopy_across_cube_borders_and_not_on_farmland_or_bare_ground()
 {
    let rig = LandRig::level(45.0, 100.0, TREE);
    let w = plain(&rig);
    let scan = Scan::ground(&w, 4, TREES);
    let s = w.ground();
    let leaves = all_leaves(&rig);
    assert!(scan.count_of(&leaves) > 500, "{}", scan.count_of(&leaves));
    let trunks = scan.trunks(s);
    assert!(trunks > 50, "{trunks}");
    // Every trunk has a leaf over it and a leaf or log beside its top, whichever cube holds them.
    let logs = rig.blocks.far.logs;
    let mut checked = 0;
    for z in 1..scan.side() - 1 {
        for x in 1..scan.side() - 1 {
            if !logs.contains(&scan.at(x, s + 1, z)) {
                continue;
            }
            let top = (s + 1..=scan.top_y())
                .take_while(|&y| logs.contains(&scan.at(x, y, z)))
                .last()
                .unwrap();
            assert!(
                leaves.contains(&scan.at(x, top + 1, z)),
                "no crown over {x},{z}"
            );
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let b = scan.at(x + dx, top, z + dz);
                assert!(
                    leaves.contains(&b) || logs.contains(&b),
                    "no canopy beside {x},{z}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 40);
    // The ground is what the terrain made: trees only go into air.
    let bare = Scan::ground(&w, 4, FarDecor::default());
    for z in 0..scan.side() {
        for x in 0..scan.side() {
            assert_eq!(scan.at(x, s, z), bare.at(x, s, z));
            assert_eq!(scan.at(x, s - 1, z), bare.at(x, s - 1, z));
        }
    }
    // Farmland, bare ground, ice and open water grow none (a city's street trees: `rules_tests`).
    for class in [CROP, BARE, SNOW_ICE, WATER] {
        let rig = LandRig::level(45.0, 100.0, class);
        let scan = Scan::ground(&plain(&rig), 3, TREES);
        assert_eq!(scan.count_of(&all_leaves(&rig)), 0, "class {class}");
        assert_eq!(scan.count_of(&rig.blocks.far.logs), 0, "class {class}");
    }
    // A city's street trees stand on its grass parcels only.
    let rig = LandRig::level(45.0, 100.0, BUILT);
    let w = plain(&rig);
    let scan = Scan::ground(&w, 8, TREES);
    let s = w.ground();
    assert!(scan.trunks(s) > 0);
    for z in 0..scan.side() {
        for x in 0..scan.side() {
            if rig.blocks.far.logs.contains(&scan.at(x, s + 1, z)) {
                assert_eq!(scan.at(x, s, z), rig.blocks.grass_block, "{x},{z}");
            }
        }
    }
}

#[test]
fn the_density_of_trees_follows_the_land_class() {
    // Level 2: a canopy per column, over 4 by 4 nodes (a 256 block square).
    let columns = |class: u8| {
        let rig = LandRig::level(45.0, 100.0, class);
        let w = plain(&rig);
        let leaves = all_leaves(&rig);
        let mut canopy = 0;
        for z in 0..4 {
            for x in 0..4 {
                canopy += columns_with(&node_tops(&w, 2, x, z, TREES), &leaves);
            }
        }
        canopy as f64 / (16.0 * 256.0)
    };
    let forest = style(Biome::Forest).unwrap().cover();
    let plains = style(Biome::Plains).unwrap().cover();
    let (tree, shrub, grass, crop) = (columns(TREE), columns(SHRUB), columns(GRASS), columns(CROP));
    // Tree cover takes the whole cover of its style, shrub a fifth of a thin one, grass a fiftieth of it.
    assert!((tree - forest).abs() < 0.05, "{tree} for {forest}");
    assert!(
        (shrub - 0.2 * plains).abs() < 0.01,
        "{shrub} for {}",
        0.2 * plains
    );
    assert!(
        grass < shrub / 3.0 && grass < 0.006,
        "grass {grass}, shrub {shrub}"
    );
    assert!(
        tree > 20.0 * shrub && shrub > grass,
        "{tree} {shrub} {grass}"
    );
    assert_eq!(crop, 0.0);
    // Level 0 agrees: trunks over 128 blocks square, one per cell of the spacing, and a fifth of that on shrub.
    let trunks = |class: u8| {
        let rig = LandRig::level(45.0, 100.0, class);
        let w = plain(&rig);
        Scan::ground(&w, 8, TREES).trunks(w.ground())
    };
    let (t, sh) = (trunks(TREE), trunks(SHRUB));
    let want = 128.0 * 128.0 / 25.0;
    assert!(
        (t as f64) > 0.6 * want && (t as f64) < 1.2 * want,
        "{t} trunks for {want}"
    );
    assert!(sh * 8 < t && sh > 0, "{t} {sh}");
}

#[test]
fn the_kind_of_tree_follows_the_biome() {
    use Species::*;
    // (latitude, class, the species its biome grows): the latitude table makes jungle, forest, taiga, savanna,
    // swamp and mangrove swamp of these.
    let cases = [
        (5.0, TREE, vec![Jungle]),
        (45.0, TREE, vec![Oak, Birch]),
        (60.0, TREE, vec![Spruce]),
        (10.0, SHRUB, vec![Acacia]),
        (45.0, WETLAND, vec![Oak]),
        (5.0, MANGROVE, vec![Mangrove]),
    ];
    for (lat, class, want) in cases {
        let rig = LandRig::level(lat, 100.0, class);
        let w = plain(&rig);
        let scan = Scan::ground(&w, 6, TREES);
        let tops = node_tops(&w, 3, 0, 0, TREES);
        for species in Species::ALL {
            let n = scan.count_of(&[leaves_of(&rig, species)]);
            let coarse = columns_with(&tops, &[leaves_of(&rig, species)]);
            if want.contains(&species) {
                assert!(n > 0, "lat {lat} class {class}: no {species:?} leaves");
            } else {
                assert_eq!(n, 0, "lat {lat} class {class}: {species:?} leaves");
                assert_eq!(coarse, 0, "lat {lat} class {class}: {species:?} at level 3");
            }
        }
        assert!(
            columns_with(&tops, &all_leaves(&rig)) > 0,
            "lat {lat} class {class}"
        );
    }
    // Biomes without a tree give none.
    for biome in [
        Biome::Desert,
        Biome::Ocean,
        Biome::StonyPeaks,
        Biome::Badlands,
        Biome::IceSpikes,
    ] {
        assert!(style(biome).is_none(), "{biome:?}");
    }
    // Every tree stays within the reach the classification allows.
    for biome in Biome::ALL {
        if let Some(st) = style(biome) {
            assert!(
                st.height.1 + 4 <= super::far_trees::MAX_TREE_UP,
                "{biome:?}"
            );
            assert!(st.radius <= super::far_trees::MAX_RADIUS, "{biome:?}");
        }
    }
}

#[test]
fn a_node_without_its_class_or_climate_is_not_ready_and_waits_for_nothing() {
    // Class: heights are held, land cover is not.
    let rig = LandRig::level(45.0, 100.0, TREE);
    let g = &rig.generator;
    let at = SectionPos::new(0, rig.generator.surface_y_blocking(8, 8).unwrap() >> 4, 0);
    let mut out = Box::new([0u32; 4096]);
    let err = loop {
        match g
            .generate(at, &mut out)
            .and_then(|_| g.decorate_cube(at, &mut out, TREES))
        {
            Err(e) if !e.tiles.is_empty() => {
                for t in e.tiles {
                    rig.store.load(t).unwrap();
                }
            }
            Err(e) => break e,
            Ok(_) => panic!("decorated without land cover"),
        }
    };
    assert!(err.tiles.is_empty());
    assert_eq!(rig.land.fetch_count(), 0);
    assert!(!g.take_land_requests().is_empty());

    // Coarse: the class is held (the columns came), the biome is not.
    let mut cols = plain(&rig).ready(|g| g.node_columns(3, 0, 0));
    let mut waited = false;
    for _ in 0..40 {
        match g.decorate_columns(&mut cols, TREES) {
            Ok(()) => break,
            Err(e) => {
                waited = true;
                for t in e.tiles {
                    rig.store.load(t).unwrap();
                }
                for fetch in g.take_land_requests() {
                    rig.source.satisfy(&fetch).unwrap();
                }
            }
        }
    }
    assert!(waited, "the biome was ready without being loaded");
    assert!(g.decorate_columns(&mut cols, TREES).is_ok());
    assert!(cols.decor.is_some());

    // Climate: the class is held, the climate is not.
    let (rig, g, _dir) = climate_world(40.0, 100.0, TREE, 10.0);
    let at = SectionPos::new(0, rig.generator.surface_y_blocking(8, 8).unwrap() >> 4, 0);
    let mut seen = false;
    for _ in 0..60 {
        match g
            .generate(at, &mut out)
            .and_then(|_| g.decorate_cube(at, &mut out, TREES))
        {
            Ok(_) => break,
            Err(e) => {
                for &t in &e.tiles {
                    rig.store.load(t).unwrap();
                }
                for fetch in g.take_land_requests() {
                    rig.source.satisfy(&fetch).unwrap();
                }
                let asked = g.take_climate_requests();
                seen |= !asked.is_empty();
                for fetch in asked {
                    g.climate_source().unwrap().satisfy(&fetch).unwrap();
                }
            }
        }
    }
    assert!(seen, "the climate was never asked for");
    assert!(g.decorate_cube(at, &mut out, TREES).is_ok());
}

#[test]
fn snow_lies_on_cold_ground_and_leaves_and_ice_on_cold_water_and_nothing_in_a_warm_biome() {
    // Snowy taiga (tree cover past 66 degrees): a snow layer on every top, the grass under it snowy.
    let rig = LandRig::level(70.0, 100.0, TREE);
    let w = plain(&rig);
    let b = rig.blocks;
    let s = w.ground();
    let scan = Scan::ground(&w, 4, SNOWY);
    let leaves = all_leaves(&rig);
    let (mut on_leaves, mut on_ground) = (0, 0);
    for z in 0..scan.side() {
        for x in 0..scan.side() {
            let (y, top) = scan.top(x, z);
            assert_eq!(top, b.far.snow, "column {x},{z} ends in {top} at {y}");
            let under = scan.at(x, y - 1, z);
            if leaves.contains(&under) || b.far.logs.contains(&under) {
                on_leaves += 1;
            } else {
                assert_eq!(under, b.far.snowy_grass, "column {x},{z}");
                assert_eq!(y, s + 1);
                on_ground += 1;
            }
        }
    }
    assert!(
        on_leaves > 200 && on_ground > 200,
        "{on_leaves} {on_ground}"
    );
    // Snow off: the same trees, no snow.
    let trees_only = Scan::ground(&w, 4, TREES);
    // (The synthetic registry has no `snowy` property: its snowy grass is plain grass.)
    assert_eq!(trees_only.count_of(&[b.far.snow, b.far.ice]), 0);
    // A warm biome has none.
    let warm = LandRig::level(45.0, 100.0, TREE);
    let scan = Scan::ground(&plain(&warm), 4, SNOWY);
    let b = warm.blocks;
    assert_eq!(scan.count_of(&[b.far.snow, b.far.ice, b.snow_block]), 0);

    // Cold still water is ice (the frozen ocean the climate makes), warm water stays water.
    let (frozen, g, _d) = climate_world(70.0, -50.0, WATER, -10.0);
    let w = World {
        rig: &frozen,
        g: &g,
    };
    let b = frozen.blocks;
    let sea = g.sea_level();
    let scan = Scan::of(&w, 2, (sea >> 4) - 1, sea >> 4, SNOWY);
    for z in 0..scan.side() {
        for x in 0..scan.side() {
            assert_eq!(scan.at(x, sea - 1, z), b.far.ice, "{x},{z}");
            assert_eq!(scan.at(x, sea - 2, z), b.water, "{x},{z}");
            assert_eq!(scan.at(x, sea, z), b.air);
        }
    }
    let (warm_sea, g, _d) = climate_world(5.0, -50.0, WATER, 26.0);
    let w = World {
        rig: &warm_sea,
        g: &g,
    };
    let scan = Scan::of(&w, 2, (sea >> 4) - 1, sea >> 4, SNOWY);
    assert_eq!(scan.count_of(&[b.far.ice, b.far.snow]), 0);
    assert!(scan.count_of(&[b.water]) > 0);
}

#[test]
fn coarse_snow_whitens_cold_tops_speckles_canopies_and_freezes_cold_water() {
    let rig = LandRig::level(70.0, 100.0, TREE);
    let w = plain(&rig);
    let b = rig.blocks;
    let leaves = all_leaves(&rig);
    // Level 2: a snow block on every ground top, and about half of the canopy tops.
    let tops = node_tops(&w, 2, 0, 0, SNOWY);
    let (mut white_ground, mut white_canopy, mut green_canopy) = (0, 0, 0);
    for col in &tops {
        // The highest voxel that is not air.
        let top = *col.iter().rev().find(|&&v| v != b.air).unwrap();
        let canopy = col.iter().any(|v| leaves.contains(v));
        match (canopy, top) {
            // A snow layer (light passes it), not a block; on a one-voxel tip it is the column's only canopy.
            (_, t) if t == b.far.snow => white_canopy += 1,
            (false, t) => {
                assert_eq!(t, b.snow_block);
                white_ground += 1;
            }
            (true, t) => {
                assert!(leaves.contains(&t));
                green_canopy += 1;
            }
        }
    }
    assert!(
        white_ground > 50 && white_canopy > 20 && green_canopy > 20,
        "{white_ground} {white_canopy} {green_canopy}"
    );
    // Without snow nothing is white, and warm ground has none anywhere.
    assert_eq!(
        columns_with(&node_tops(&w, 2, 0, 0, TREES), &[b.snow_block, b.far.snow]),
        0
    );
    let warm = LandRig::level(45.0, 100.0, TREE);
    let ww = plain(&warm);
    assert_eq!(
        columns_with(
            &node_tops(&ww, 2, 0, 0, SNOWY),
            &[b.snow_block, b.far.snow, b.far.ice]
        ),
        0
    );
    // Cold water: the top water voxel is ice.
    let (frozen, g, _d) = climate_world(70.0, -50.0, WATER, -10.0);
    let fw = World {
        rig: &frozen,
        g: &g,
    };
    assert_eq!(
        columns_with(&node_tops(&fw, 2, 0, 0, SNOWY), &[b.far.ice]),
        256
    );
    assert_eq!(
        columns_with(&node_tops(&fw, 2, 0, 0, FarDecor::default()), &[b.far.ice]),
        0
    );
    // A voxel wider than the canopy is thin: the ground voxel itself is leaves where a forest stands.
    let coarse = node_tops(&ww, 5, 0, 0, TREES);
    assert!(columns_with(&coarse, &all_leaves(&warm)) > 200);
}

#[test]
fn with_trees_and_snow_off_the_far_node_is_what_it_was() {
    let rig = LandRig::level(70.0, 100.0, TREE);
    let w = plain(&rig);
    let g = &rig.generator;
    let s = w.ground();
    // Cubes and columns are untouched, and nothing is reported changed.
    for cy in (s >> 4) - 1..=(s >> 4) + 2 {
        let at = SectionPos::new(1, cy, 2);
        let mut made = w.ready(|g| {
            let mut out = Box::new([0u32; 4096]);
            g.generate(at, &mut out)?;
            Ok(out)
        });
        let want = made.clone();
        assert!(!g.decorate_cube(at, &mut made, FarDecor::default()).unwrap());
        assert_eq!(&made[..], &want[..]);
    }
    for level in [2, 4, 6] {
        let mut cols = w.ready(|g| g.node_columns(level, 0, 0));
        g.decorate_columns(&mut cols, FarDecor::default()).unwrap();
        assert!(cols.decor.is_none());
        for i in 0..256 {
            for y in [s - 20, s - 1, s, s + 1, s + 20] {
                let terrain = g.voxel_block(y, cols.step, cols.surfaces[i], cols.kind(i));
                assert_eq!(g.far_voxel(&cols, i, y), terrain);
            }
        }
    }
    // A world without land cover (an old one) is untouched whatever is asked.
    let mut off = rig.settings.clone();
    off.land_cover = false;
    let old = EarthGenerator::new(&off, rig.store.clone(), rig.blocks).unwrap();
    let at = SectionPos::new(0, s >> 4, 0);
    let ow = World { rig: &rig, g: &old };
    let mut out = ow.ready(|g| {
        let mut cube = Box::new([0u32; 4096]);
        g.generate(at, &mut cube)?;
        Ok(cube)
    });
    let want = out.clone();
    assert!(!old.decorate_cube(at, &mut out, SNOWY).unwrap());
    assert_eq!(&out[..], &want[..]);
    let mut cols = ow.ready(|g| g.node_columns(3, 0, 0));
    old.decorate_columns(&mut cols, SNOWY).unwrap();
    assert!(cols.decor.is_none());
}

#[test]
fn air_is_proven_only_from_the_height_trees_reach() {
    let rig = LandRig::level(45.0, 100.0, TREE);
    let w = plain(&rig);
    let s = w.ground();
    // A node of 64 blocks whose bottom is a little over the ground is sky to the plain classification and not to
    // this one; higher up they agree, and below the ground they always do.
    let level = 2;
    let probe = |y: i32| {
        w.ready(|g| {
            let a = g.classify_node(level, 0, y, 0)?;
            let b = g.classify_node_above(level, 0, y, 0)?;
            Ok((a, b))
        })
    };
    let low = (s + 2).div_euclid(64) + 1;
    assert!(low * 64 > s && low * 64 <= s + MAX_TREE_UP);
    assert_eq!(probe(low), (Fill::Air, Fill::Mixed));
    assert_eq!(probe(low + 2), (Fill::Air, Fill::Air));
    let (plain_class, above) = probe(s.div_euclid(64) - 3);
    assert_eq!(plain_class, above);
    assert_eq!(plain_class, Fill::Stone);
}

//! Tests of how land cover reads at world blocks: the pixel grid is bent by a smooth warp so a pixel's straight
//! edges are no longer straight and its corners no longer right angles, and specks of one or two pixels are a
//! light mix, not a solid slab. Miniature WorldCover files (600 pixels over 3 degrees, 556 m a pixel) with
//! blocks 1/9 of a pixel, which is what real data (9.28 m pixels) and one-metre blocks look like.

use std::sync::Arc;

use crate::landcover::{FILE_DEGREES, LandClass, METRES_PER_DEGREE, WorldCoverFile};
use crate::landfield::{LandCoverField, LandCoverStore, Lookup, Wanted};
use crate::landsource::LandCoverSource;
use crate::projection::Projection;
use crate::testing::{MemoryLandCover, TempDir, serve_range};

const SIZE: u32 = 600;
const DEGREES_PER_PIXEL: f64 = FILE_DEGREES as f64 / SIZE as f64;
/// Blocks across a pixel, north to south.
const BLOCKS: f64 = 9.0;
const FILE: WorldCoverFile = WorldCoverFile { south: 0, west: 0 };
/// The file's pixel `(300, 300)` has its north-west corner at the block origin of the projection below.
const ORIGIN: f64 = 1.5;

fn pixel_metres() -> f64 {
    f64::from(FILE_DEGREES) * METRES_PER_DEGREE / f64::from(SIZE)
}

fn metres_per_block() -> f64 {
    pixel_metres() / BLOCKS
}

/// The pixel `(column, row)` of the file at a place, as the data counts them.
fn pixel_of(lat: f64, lon: f64) -> (i64, i64) {
    (
        ((lon - f64::from(FILE.west)) / DEGREES_PER_PIXEL).floor() as i64,
        ((FILE.north() - lat) / DEGREES_PER_PIXEL).floor() as i64,
    )
}

struct Rig {
    source: Arc<LandCoverSource>,
    _dir: TempDir,
}

impl Rig {
    /// A file whose pixel `(column, row)` is `class(column, row)`.
    fn new(class: impl Fn(i64, i64) -> LandClass) -> Self {
        let land = MemoryLandCover::new().with_classes(FILE, SIZE, 256, 1, |lat, lon| {
            let (c, r) = pixel_of(lat, lon);
            class(c, r).code()
        });
        let dir = TempDir::new();
        let store = Arc::new(LandCoverStore::new(dir.path(), 16));
        let projection = Projection::local(ORIGIN, ORIGIN, metres_per_block()).unwrap();
        let field = LandCoverField::new(store, projection).with_level0_pixel_metres(pixel_metres());
        Self {
            source: Arc::new(LandCoverSource::new(field, Arc::new(land))),
            _dir: dir,
        }
    }

    fn field(&self) -> &LandCoverField {
        self.source.field()
    }

    /// The class at the centre of block `(x, z)`, fetching what is missing.
    fn block(&self, x: i32, z: i32) -> LandClass {
        self.source
            .class_at_blocking(f64::from(x) + 0.5, f64::from(z) + 0.5, metres_per_block())
            .unwrap()
    }

    /// The class the store says at the centre of block `(x, z)` with no warp, once the tiles are held.
    fn unwarped(&self, x: i32, z: i32) -> LandClass {
        let at = self
            .field()
            .projection()
            .to_lat_lon(f64::from(x) + 0.5, f64::from(z) + 0.5);
        match self
            .field()
            .store()
            .lookup(at.lat, at.lon, metres_per_block())
        {
            Lookup::Class(c) => c,
            other => panic!("not held: {other:?}"),
        }
    }

    /// The pixel column coordinate, fractional, of the west edge of block `x` on row `z`.
    fn column_of(&self, x: i32, z: i32) -> f64 {
        let at = self
            .field()
            .projection()
            .to_lat_lon(f64::from(x), f64::from(z) + 0.5);
        (at.lon - f64::from(FILE.west)) / DEGREES_PER_PIXEL
    }

    /// The block rectangle `(x0, z0, x1, z1)` (end exclusive) of blocks whose centres lie in pixel `(c, r)`.
    fn footprint(&self, c: i64, r: i64) -> (i32, i32, i32, i32) {
        let p = self.field().projection();
        let lon = |c: i64| f64::from(FILE.west) + c as f64 * DEGREES_PER_PIXEL;
        let lat = |r: i64| FILE.north() - r as f64 * DEGREES_PER_PIXEL;
        let (xa, za) = p.from_lat_lon(lat(r), lon(c));
        let (xb, zb) = p.from_lat_lon(lat(r + 1), lon(c + 1));
        // Block centres are at +0.5.
        let edge = |v: f64| (v - 0.5).ceil() as i32;
        (edge(xa), edge(za), edge(xb), edge(zb))
    }
}

/// For each row `z` in `rows`, the first block `x` of `scan(z)` that `read` says is `class`.
fn first_switches(
    rows: std::ops::Range<i32>,
    scan: impl Fn(i32) -> std::ops::Range<i32>,
    read: impl Fn(i32, i32) -> LandClass,
    class: LandClass,
) -> Vec<(i32, i32)> {
    rows.filter_map(|z| scan(z).find(|&x| read(x, z) == class).map(|x| (z, x)))
        .collect()
}

#[test]
fn a_straight_boundary_between_two_pixel_columns_is_a_wavy_line_not_a_pixel_edge() {
    // Grass west of pixel column 300, trees east of it: unbent, the line is x = 0 for every row.
    let rig = Rig::new(|c, _| {
        if c < 300 {
            LandClass::Grass
        } else {
            LandClass::Tree
        }
    });
    let switches = first_switches(
        -100..100,
        |_| -30..30,
        |x, z| rig.block(x, z),
        LandClass::Tree,
    );
    assert_eq!(switches.len(), 200);
    let xs: Vec<i32> = switches.iter().map(|&(_, x)| x).collect();
    let distinct: std::collections::BTreeSet<i32> = xs.iter().copied().collect();
    assert!(
        distinct.len() >= 8,
        "only {} positions: {xs:?}",
        distinct.len()
    );
    // Few rows sit on the pixel line itself.
    let on_line = xs.iter().filter(|&&x| x == 0).count();
    assert!(on_line < 50, "{on_line} of 200 rows on the pixel line");
    // The line is smooth: neighbouring rows differ by a few blocks at most, nearly everywhere.
    let calm = xs.windows(2).filter(|w| (w[1] - w[0]).abs() <= 3).count();
    assert!(calm >= 190, "{calm} of 199 steps are calm: {xs:?}");
    // And the data stays where it is, within about a pixel and a half.
    assert!(
        xs.iter().all(|&x| f64::from(x.abs()) <= 1.5 * BLOCKS),
        "{xs:?}"
    );
    // The boundary is not a straight line either: it wanders to both sides.
    assert!(
        xs.iter().any(|&x| x < -1) && xs.iter().any(|&x| x > 1),
        "{xs:?}"
    );
}

#[test]
fn a_staircase_of_pixels_is_not_a_staircase_of_blocks() {
    // Trees where the column is past the row: a one-pixel staircase along the diagonal.
    let rig = Rig::new(|c, r| {
        if c > r {
            LandClass::Tree
        } else {
            LandClass::Grass
        }
    });
    let on_edge = |switches: &[(i32, i32)]| {
        let near = switches
            .iter()
            .filter(|&&(z, x)| {
                let u = rig.column_of(x, z);
                (u - u.round()).abs() * BLOCKS <= 0.6
            })
            .count();
        near as f64 / switches.len() as f64
    };
    let warped = first_switches(
        -80..80,
        |z| z - 40..z + 40,
        |x, z| rig.block(x, z),
        LandClass::Tree,
    );
    assert_eq!(warped.len(), 160);
    // The same walk over the raw pixels: every row's switch is on a pixel edge, which is what the test measures.
    let raw = first_switches(
        -80..80,
        |z| z - 40..z + 40,
        |x, z| rig.unwarped(x, z),
        LandClass::Tree,
    );
    let (raw_share, warped_share) = (on_edge(&raw), on_edge(&warped));
    assert!(raw_share > 0.95, "raw {raw_share}");
    assert!(warped_share < 0.4, "warped {warped_share}");
    // Steps of nine blocks (a pixel's height) are gone: the switch wanders by a block or two a row rather than
    // holding for a pixel and jumping by a pixel.
    let xs = |switches: &[(i32, i32)]| switches.iter().map(|&(_, x)| x).collect::<Vec<i32>>();
    let (warped_xs, raw_xs) = (xs(&warped), xs(&raw));
    let jumps = |xs: &[i32]| xs.windows(2).filter(|w| (w[1] - w[0]).abs() >= 6).count();
    let longest = |xs: &[i32]| {
        let (mut best, mut run) = (1, 1);
        for w in xs.windows(2) {
            run = if w[0] == w[1] { run + 1 } else { 1 };
            best = best.max(run);
        }
        best
    };
    assert!(jumps(&raw_xs) >= 15, "{}", jumps(&raw_xs));
    assert!(longest(&raw_xs) >= 8, "{}", longest(&raw_xs));
    assert!(
        jumps(&warped_xs) * 3 < jumps(&raw_xs),
        "{} jumps against {}: {warped_xs:?}",
        jumps(&warped_xs),
        jumps(&raw_xs)
    );
    assert!(
        longest(&warped_xs) < longest(&raw_xs),
        "{}: {warped_xs:?}",
        longest(&warped_xs)
    );
}

#[test]
fn a_lone_pixel_is_scattered_blocks_and_a_patch_stays_solid_inside() {
    // Single bare pixels in grass on a lattice, and a 3 by 3 bare patch elsewhere.
    let lattice = [282_i64, 294, 306, 318];
    let rig = Rig::new(|c, r| {
        let speck = lattice.contains(&c) && lattice.contains(&r);
        let patch = (270..=272).contains(&c) && (270..=272).contains(&r);
        if speck || patch {
            LandClass::Bare
        } else {
            LandClass::Grass
        }
    });
    let share = |c: i64, r: i64| {
        let (x0, z0, x1, z1) = rig.footprint(c, r);
        let mut bare = 0_usize;
        let mut all = 0_usize;
        for z in z0..z1 {
            for x in x0..x1 {
                all += 1;
                bare += usize::from(rig.block(x, z) == LandClass::Bare);
            }
        }
        assert!(all >= 70, "footprint of {all} blocks");
        bare as f64 / all as f64
    };
    let mut shares = Vec::new();
    for &c in &lattice {
        for &r in &lattice {
            shares.push(share(c, r));
        }
    }
    let mean = shares.iter().sum::<f64>() / shares.len() as f64;
    assert!((0.1..0.5).contains(&mean), "mean {mean}: {shares:?}");
    for s in &shares {
        assert!((0.03..0.65).contains(s), "{s} in {shares:?}");
    }
    // The patch is bare in its middle pixel, most of it, and the grass beside it is not.
    let middle = share(271, 271);
    assert!(middle > 0.8, "{middle}");
    let beside = share(275, 271);
    assert!(beside < 0.05, "{beside}");
}

#[test]
fn a_speck_read_is_its_own_class_or_the_class_around_it_and_nothing_else() {
    let rig = Rig::new(|c, r| {
        if c % 10 == 5 && r % 10 == 5 {
            LandClass::Bare
        } else {
            LandClass::Crop
        }
    });
    let mut seen = std::collections::BTreeSet::new();
    for z in (-90..90).step_by(3) {
        for x in (-90..90).step_by(3) {
            seen.insert(rig.block(x, z));
        }
    }
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        vec![LandClass::Crop, LandClass::Bare]
    );
}

#[test]
fn near_and_far_reads_of_the_class_are_the_same_function() {
    // A busy map: classes by a hash of the pixel in a few classes, in blobs of about 3 pixels.
    let classes = [
        LandClass::Tree,
        LandClass::Grass,
        LandClass::Bare,
        LandClass::Crop,
        LandClass::Water,
    ];
    let rig = Rig::new(|c, r| {
        let h = crate::warp::cell_hash(c.div_euclid(3), r.div_euclid(3), 0, 77);
        classes[(h * classes.len() as f64) as usize]
    });
    let block = metres_per_block();
    for metres in [block, 4.0 * block, 20.0 * block, 80.0 * block] {
        let stride = (metres / block).round() as i32;
        let (x0, z0) = (-4 * stride, -3 * stride);
        let grid = rig
            .source
            .class_grid_blocking(x0, z0, 12, 9, stride, metres)
            .unwrap();
        let direct = rig
            .field()
            .class_grid(x0, z0, 12, 9, stride, metres)
            .unwrap();
        assert_eq!(grid, direct, "stride {stride}");
        for j in 0..9 {
            for i in 0..12 {
                let (x, z) = (
                    f64::from(x0 + i * stride) + 0.5,
                    f64::from(z0 + j * stride) + 0.5,
                );
                let at = rig.field().class_at(x, z, metres);
                let got = rig.source.class_at_blocking(x, z, metres).unwrap();
                assert_eq!(at, Lookup::Class(got), "stride {stride}");
                assert_eq!(grid[(j * 12 + i) as usize], got);
                if stride == 1 {
                    assert_eq!(rig.field().class_at_block(x0 + i, z0 + j), at);
                }
            }
        }
    }
    // The warp is in it: the bent map is not the raw one, and it has not moved the data far.
    let (mut same, mut all) = (0, 0);
    for z in -60..60 {
        for x in -60..60 {
            all += 1;
            same += usize::from(rig.block(x, z) == rig.unwarped(x, z));
        }
    }
    assert!(same < all, "the warp changed nothing");
    assert!(
        same * 100 > all * 60,
        "{same} of {all} agree: the warp moves data too far"
    );
}

#[test]
fn the_warp_is_the_same_wherever_and_whenever_it_is_asked() {
    let make = || {
        Rig::new(|c, _| {
            if c % 2 == 0 {
                LandClass::Tree
            } else {
                LandClass::Grass
            }
        })
    };
    let (a, b) = (make(), make());
    for z in (-50..50).step_by(7) {
        for x in (-50..50).step_by(5) {
            assert_eq!(a.block(x, z), b.block(x, z));
            assert_eq!(a.block(x, z), a.block(x, z));
        }
    }
}

#[test]
fn the_prefetch_reaches_as_far_as_the_warp_bends_a_read() {
    let rig = Rig::new(|_, _| LandClass::Grass);
    let bytes = MemoryLandCover::new()
        .with_classes(FILE, SIZE, 256, 1, |_, _| 30)
        .file_bytes(FILE)
        .unwrap();
    // Give the store the file's index.
    for _ in 0..6 {
        let wanted = rig.field().needs(-390, 0, -380, 10, metres_per_block());
        let Some(index) = wanted.into_iter().find(|f| f.wanted == Wanted::Index) else {
            break;
        };
        rig.field()
            .store()
            .provide(&index, serve_range(&bytes, index.range))
            .unwrap();
    }
    // Pixel column 256 (where the second tile column starts) is at block -396: this rectangle is in tile
    // column 1 on the raw grid, and within the warp's reach of column 0.
    let mut tiles: Vec<u32> = rig
        .field()
        .needs(-390, 0, -380, 10, metres_per_block())
        .into_iter()
        .filter_map(|f| match f.wanted {
            Wanted::Tile { tile, .. } => Some(tile),
            Wanted::Index => None,
        })
        .collect();
    tiles.sort_unstable();
    assert_eq!(tiles, vec![3, 4], "{tiles:?}");
    // A rectangle far from any tile edge needs one tile.
    assert_eq!(rig.field().needs(0, 0, 10, 10, metres_per_block()).len(), 1);
}

/// The store's rounded read at `(lat, lon)`, fetching what is missing.
fn smooth(rig: &Rig, lat: f64, lon: f64) -> LandClass {
    for _ in 0..20 {
        match rig
            .field()
            .store()
            .lookup_smooth(lat, lon, metres_per_block())
        {
            Lookup::Class(c) => return c,
            Lookup::Need(fetch) => rig.source.satisfy(&fetch).unwrap(),
            Lookup::NoFile => return LandClass::NoData,
        }
    }
    panic!("never completes");
}

/// The place `(lat, lon)` at `(fx, fy)` inside pixel `(c, r)`.
fn place(c: i64, r: i64, fx: f64, fy: f64) -> (f64, f64) {
    (
        FILE.north() - (r as f64 + fy) * DEGREES_PER_PIXEL,
        f64::from(FILE.west) + (c as f64 + fx) * DEGREES_PER_PIXEL,
    )
}

#[test]
fn the_corner_of_a_block_of_pixels_is_rounded_and_its_straight_edge_is_not() {
    let rig = Rig::new(|c, r| {
        if c >= 300 && r >= 300 {
            LandClass::Tree
        } else {
            LandClass::Grass
        }
    });
    let at = |fx, fy| {
        let (lat, lon) = place(300, 300, fx, fy);
        (
            smooth(&rig, lat, lon),
            rig.field().store().lookup(lat, lon, metres_per_block()),
        )
    };
    // Right at the outer corner of the tree pixel the raw map says tree and the rounded one grass.
    assert_eq!(
        at(0.05, 0.05),
        (LandClass::Grass, Lookup::Class(LandClass::Tree))
    );
    // The pixel's middle and the middle of its edges are tree in both.
    for (fx, fy) in [(0.5, 0.5), (0.5, 0.05), (0.05, 0.5), (0.9, 0.9)] {
        assert_eq!(at(fx, fy).0, LandClass::Tree, "{fx},{fy}");
    }
    // Beside the edge it is grass in both.
    let (lat, lon) = place(299, 305, 0.5, 0.5);
    assert_eq!(smooth(&rig, lat, lon), LandClass::Grass);
}

#[test]
fn rounding_reads_across_tile_edges_as_it_does_inside_a_tile() {
    // A one-pixel staircase that is the same everywhere along its diagonal: read at the corner where four
    // tiles meet (pixel 256, 256) and in the middle of a tile (pixel 300, 300), the rounded classes agree
    // at every point.
    let rig = Rig::new(|c, r| {
        if c > r {
            LandClass::Tree
        } else {
            LandClass::Grass
        }
    });
    let mut differ = 0;
    let mut trees = 0;
    for i in 0..30 {
        for j in 0..30 {
            let (fx, fy) = (f64::from(i) / 10.0 - 1.0, f64::from(j) / 10.0 - 1.0);
            let (a, b) = (
                place(300, 300, fx + 0.02, fy + 0.03),
                place(256, 256, fx + 0.02, fy + 0.03),
            );
            let (inside, corner) = (smooth(&rig, a.0, a.1), smooth(&rig, b.0, b.1));
            differ += usize::from(inside != corner);
            trees += usize::from(inside == LandClass::Tree);
        }
    }
    assert_eq!(differ, 0);
    assert!(trees > 200 && trees < 700, "{trees}");
}

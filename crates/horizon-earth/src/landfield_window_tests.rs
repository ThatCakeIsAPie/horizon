//! The window fast path of `class_grid` against the per-sample reference, bit for bit.
//!
//! Miniature WorldCover files (600 pixels over 3 degrees, small tiles) are served to a store a fetch at a time,
//! and after every step the fast path and the reference are asked for the same windows (many strides, sizes,
//! origins on file, tile and date-line edges, scales between levels). Both must give the same classes, or the
//! same first missing [`Fetch`], whatever the store holds: nothing, part of an index, some tiles, tiles blank,
//! files absent, all of it, with a memory cache that holds everything or next to nothing.

use std::collections::{BTreeSet, HashMap};

#[allow(unused_imports)]
use super::super::*;
use super::Octave;
use crate::landcover::{FILE_DEGREES, METRES_PER_DEGREE};
use crate::testing::{TempDir, TiffSpec, encode_tiff, serve_range};

const SIZE: u32 = 600;
/// Blocks across a pixel of the finest level, north to south.
const BLOCKS: f64 = 9.0;

/// Ground size of a pixel of the miniature files' finest level.
fn pixel_metres() -> f64 {
    f64::from(FILE_DEGREES) * METRES_PER_DEGREE / f64::from(SIZE)
}

fn metres_per_block() -> f64 {
    pixel_metres() / BLOCKS
}

/// Scales of reads: finer than the finest level, on and between levels, and values that are not scales.
const METRES: [f64; 17] = [
    10.0,
    61.8,
    300.0,
    556.0,
    560.0,
    700.0,
    1000.0,
    1200.0,
    1500.0,
    2000.0,
    2300.0,
    4000.0,
    1e6,
    f64::NAN,
    f64::INFINITY,
    0.0,
    -5.0,
];

const STRIDES: [i32; 17] = [
    1, 2, 3, 5, 7, 8, 13, 16, 31, 64, 100, 128, 255, 256, 0, -1, -7,
];
const SMALL_STRIDES: [i32; 10] = [1, 2, 3, 5, 7, 8, 13, 16, 31, 64];
const SIDES: [usize; 7] = [0, 1, 2, 7, 16, 38, 38];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        crate::warp::mix64(self.0)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

/// The codes the miniature files hold. Built-up land (50) is given back to nature by the store.
const CODES: [u8; 11] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100];

/// A pixel of a miniature file: bands that differ by level and file, with specks of one or two pixels in them.
fn code(seed: u64, level: usize, x: u32, y: u32) -> u8 {
    let band = CODES[((x / 23 + y / 29) as usize + level * 3 + seed as usize) % CODES.len()];
    // Pairs of pixels side by side share a hash: a speck of one or two pixels now and then.
    let h = crate::warp::mix64(
        (u64::from(x / 2) << 32) ^ u64::from(y) ^ ((level as u64) << 56) ^ seed,
    );
    if h % 53 == 0 {
        CODES[((h >> 20) % CODES.len() as u64) as usize]
    } else if h % 211 == 1 {
        0
    } else {
        band
    }
}

fn file_bytes(file: WorldCoverFile, tile: u32) -> Vec<u8> {
    let seed = u64::from(file.south as u16) * 7 + u64::from(file.west as u16);
    encode_tiff(&TiffSpec::new(SIZE, SIZE, tile, 2), &|level, x, y| {
        code(seed, level, x, y)
    })
    .bytes
}

/// One window of `class_grid`.
#[derive(Clone, Copy, Debug)]
struct Win {
    x0: i32,
    z0: i32,
    width: usize,
    depth: usize,
    stride: i32,
    metres: f64,
}

#[derive(Default)]
struct Stats {
    windows: usize,
    ok_windows: usize,
    index_errs: usize,
    tile_errs: usize,
    /// Samples of the windows that gave classes.
    samples: usize,
    classes: BTreeSet<LandClass>,
    no_data: usize,
}

/// A source and a store's way of filling: what the source has and how it answers what the store asks for.
struct Scenario {
    name: &'static str,
    /// The lat/lon of the block origin of the projection.
    origin: (f64, f64),
    /// The files the source has (`(south, west)`); a fetch for any other is answered "no such file".
    files: &'static [(i16, i16)],
    /// Files whose index is never given (the fetch stays missing).
    never_indexed: &'static [(i16, i16)],
    capacity: usize,
    tile: u32,
    windows: usize,
    rounds: usize,
    /// Windows of a few samples only (for a store that reads tiles back from disk at every sample).
    small: bool,
    /// How far from the origin, in blocks, windows start.
    reach: i32,
    /// Of a thousand tile fetches, how many are never answered, and how many are answered "blank".
    leave_permille: u64,
    blank_permille: u64,
}

impl Scenario {
    fn files(&self) -> HashMap<WorldCoverFile, Vec<u8>> {
        self.files
            .iter()
            .map(|&(south, west)| {
                let file = WorldCoverFile { south, west };
                (file, file_bytes(file, self.tile))
            })
            .collect()
    }

    fn field(&self, dir: &TempDir) -> LandCoverField {
        let store = Arc::new(LandCoverStore::new(dir.path(), self.capacity));
        let projection =
            Projection::local(self.origin.0, self.origin.1, metres_per_block()).unwrap();
        LandCoverField::new(store, projection).with_level0_pixel_metres(pixel_metres())
    }

    /// Block coordinates of the edges of the files and of their tiles at every level.
    fn edges(&self, field: &LandCoverField) -> (Vec<f64>, Vec<f64>) {
        let (mut xs, mut zs) = (Vec::new(), Vec::new());
        let degrees = f64::from(FILE_DEGREES);
        for &(south, west) in self.files {
            let file = WorldCoverFile { south, west };
            for size in [SIZE, SIZE / 2, SIZE / 4] {
                let mut k = 0;
                loop {
                    let at = (k * self.tile).min(size);
                    let step = f64::from(at) / f64::from(size) * degrees;
                    xs.push(
                        field
                            .projection()
                            .from_lat_lon(self.origin.0, f64::from(file.west) + step)
                            .0,
                    );
                    zs.push(
                        field
                            .projection()
                            .from_lat_lon(file.north() - step, self.origin.1)
                            .1,
                    );
                    if at == size {
                        break;
                    }
                    k += 1;
                }
            }
        }
        (xs, zs)
    }

    fn windows(&self, rng: &mut Rng, (edge_x, edge_z): (&[f64], &[f64])) -> Vec<Win> {
        let mut out = Vec::new();
        for _ in 0..self.windows {
            let (width, depth) = if self.small {
                (rng.below(8) as usize, rng.below(8) as usize)
            } else if rng.below(2) == 0 {
                (rng.pick(&SIDES), rng.pick(&SIDES))
            } else {
                (rng.below(38) as usize + 1, rng.below(38) as usize + 1)
            };
            let stride = if self.small {
                rng.pick(&SMALL_STRIDES)
            } else {
                rng.pick(&STRIDES)
            };
            let metres = rng.pick(&METRES);
            let reach = i64::from(self.reach);
            // Random, or so that the window straddles one of the edges.
            let origin = |rng: &mut Rng, edges: &[f64], cells: usize, on_edge: bool| -> i32 {
                if on_edge && !edges.is_empty() {
                    let span = (cells as i64 * i64::from(stride).abs()).max(1);
                    let off = rng.below(span as u64) as f64;
                    (rng.pick(edges) - off).floor() as i32
                } else {
                    (rng.below(2 * reach as u64 + 1) as i64 - reach) as i32
                }
            };
            let mode = rng.below(3);
            let x0 = origin(rng, edge_x, width, mode >= 1);
            let z0 = origin(rng, edge_z, depth, mode == 2);
            out.push(Win {
                x0,
                z0,
                width,
                depth,
                stride,
                metres,
            });
        }
        out
    }

    /// Answers a fetch the way this scenario's source does. Whether the store changed.
    fn answer(
        &self,
        store: &LandCoverStore,
        files: &HashMap<WorldCoverFile, Vec<u8>>,
        f: &Fetch,
    ) -> bool {
        let Some(bytes) = files.get(&f.file) else {
            store.mark_absent(f.file);
            return true;
        };
        let key = (f.file.south, f.file.west);
        if self.never_indexed.contains(&key) && f.wanted == Wanted::Index {
            return false;
        }
        if let Wanted::Tile { level, tile } = f.wanted {
            let h = crate::warp::mix64(
                (u64::from(f.file.south as u16) << 40)
                    ^ (u64::from(f.file.west as u16) << 24)
                    ^ (u64::from(level) << 20)
                    ^ u64::from(tile),
            ) % 1000;
            if h < self.leave_permille {
                return false;
            }
            if h < self.leave_permille + self.blank_permille {
                store.mark_unavailable(f);
                return true;
            }
        }
        store
            .provide(f, serve_range(bytes, f.range))
            .expect("the store takes what the source sent");
        true
    }

    /// Runs the scenario: windows compared at every state of a store that is filled a fetch at a time.
    fn run(&self) -> Stats {
        let dir = TempDir::new();
        let field = self.field(&dir);
        let files = self.files();
        let mut rng = Rng(0xC1A5_5000 ^ (self.name.len() as u64 * 7919));
        let edges = self.edges(&field);
        let wins = self.windows(&mut rng, (&edges.0, &edges.1));
        let mut stats = Stats::default();
        let mut turn = 0_usize;
        let mut compare = |win: Win, stats: &mut Stats| -> Result<Vec<LandClass>, Fetch> {
            turn += 1;
            let a = (win.x0, win.z0, win.width, win.depth, win.stride, win.metres);
            // Either first: only the store's use counters could tell, and the answers must not.
            let (fast, slow);
            if turn % 2 == 0 {
                fast = field.class_grid(a.0, a.1, a.2, a.3, a.4, a.5);
                slow = field.class_grid_reference(a.0, a.1, a.2, a.3, a.4, a.5);
            } else {
                slow = field.class_grid_reference(a.0, a.1, a.2, a.3, a.4, a.5);
                fast = field.class_grid(a.0, a.1, a.2, a.3, a.4, a.5);
            }
            if fast != slow {
                let first = match (&fast, &slow) {
                    (Ok(x), Ok(y)) => (0..x.len().max(y.len()))
                        .find(|&k| x.get(k) != y.get(k))
                        .map(|k| format!("sample {k}: fast {:?}, reference {:?}", x.get(k), y.get(k))),
                    _ => None,
                };
                panic!(
                    "{}: window {win:?} differs ({first:?})\nfast: {:?}\nreference: {:?}",
                    self.name,
                    fast.as_ref().map(|v| v.len()),
                    slow.as_ref().map(|v| v.len()),
                );
            }
            stats.windows += 1;
            match &slow {
                Ok(classes) => {
                    stats.ok_windows += 1;
                    stats.samples += classes.len();
                    stats.classes.extend(classes.iter().copied());
                    stats.no_data += classes.iter().filter(|&&c| c == LandClass::NoData).count();
                }
                Err(f) if f.wanted == Wanted::Index => stats.index_errs += 1,
                Err(_) => stats.tile_errs += 1,
            }
            slow
        };
        let store = Arc::clone(field.store());
        // Rounds over the windows, each one answering the first thing its window is missing.
        for _ in 0..self.rounds {
            for &win in &wins {
                if let Err(f) = compare(win, &mut stats) {
                    self.answer(&store, &files, &f);
                }
            }
        }
        // Then until every window is whole or waits for something that never comes.
        for &win in &wins {
            for _ in 0..2000 {
                match compare(win, &mut stats) {
                    Err(f) if self.answer(&store, &files, &f) => {}
                    _ => break,
                }
            }
        }
        // And all of them once more, in the state that is left.
        for &win in &wins {
            compare(win, &mut stats).ok();
        }
        eprintln!(
            "[{}] {} windows compared ({} with classes, {} stopped by a missing index, {} by a missing \
             tile), {} samples compared, {} of them no data, classes seen {}, tiles in memory {}",
            self.name,
            stats.windows,
            stats.ok_windows,
            stats.index_errs,
            stats.tile_errs,
            stats.samples,
            stats.no_data,
            stats.classes.len(),
            store.memory_tiles(),
        );
        stats
    }
}

/// Four files meeting at 3 degrees north, 87 east; the one north-east of the meeting point does not exist.
const JUNCTION: [(i16, i16); 3] = [(0, 84), (3, 84), (0, 87)];

#[test]
fn window_fast_path_is_bit_identical_to_the_reference_across_four_files() {
    let stats = Scenario {
        name: "four files",
        origin: (3.0, 87.0),
        files: &JUNCTION,
        never_indexed: &[],
        capacity: 4096,
        tile: 64,
        windows: 700,
        rounds: 8,
        small: false,
        reach: 7000,
        leave_permille: 0,
        blank_permille: 0,
    }
    .run();
    assert!(stats.windows >= 7000, "{}", stats.windows);
    assert!(stats.ok_windows > 3000 && stats.samples > 500_000);
    assert!(stats.index_errs > 10 && stats.tile_errs > 100);
    // Land of many classes, and the missing file's no data.
    assert!(stats.classes.len() >= 8, "{:?}", stats.classes);
    assert!(stats.no_data > 1000);
}

#[test]
fn window_fast_path_is_bit_identical_when_tiles_are_blank_or_never_come() {
    let stats = Scenario {
        name: "blank and missing tiles",
        origin: (3.0, 87.0),
        files: &JUNCTION,
        never_indexed: &[(0, 87)],
        capacity: 4096,
        tile: 64,
        windows: 600,
        rounds: 8,
        small: false,
        reach: 7000,
        leave_permille: 70,
        blank_permille: 90,
    }
    .run();
    assert!(stats.windows >= 6000, "{}", stats.windows);
    assert!(stats.ok_windows > 1000);
    // Windows stopped on a missing index (of the file that never has one) and on tiles that never come.
    assert!(stats.index_errs > 500 && stats.tile_errs > 500);
}

#[test]
fn window_fast_path_is_bit_identical_across_the_date_line() {
    let stats = Scenario {
        name: "date line",
        origin: (1.5, 179.6),
        files: &[(0, 177), (0, -180)],
        never_indexed: &[],
        capacity: 8,
        tile: 64,
        windows: 700,
        rounds: 6,
        small: false,
        reach: 6000,
        leave_permille: 30,
        blank_permille: 30,
    }
    .run();
    assert!(stats.windows >= 4000, "{}", stats.windows);
    assert!(stats.ok_windows > 1500);
    assert!(stats.classes.len() >= 8, "{:?}", stats.classes);
}

#[test]
fn window_fast_path_is_bit_identical_across_the_equator_and_the_prime_meridian() {
    let stats = Scenario {
        name: "origin of the grid",
        origin: (0.0, 0.0),
        files: &[(-3, -3), (-3, 0), (0, -3), (0, 0)],
        never_indexed: &[],
        capacity: 4096,
        tile: 32,
        windows: 500,
        rounds: 8,
        small: false,
        reach: 6500,
        leave_permille: 0,
        blank_permille: 20,
    }
    .run();
    assert!(stats.windows >= 5000, "{}", stats.windows);
    assert!(stats.ok_windows > 2000 && stats.samples > 400_000);
}

#[test]
fn window_fast_path_is_bit_identical_at_the_pole() {
    // The last row of files, whose top edge is the pole: reads past it are no data, and the longitude runs
    // through a degree in a hundred and fifty blocks.
    let stats = Scenario {
        name: "pole",
        origin: (85.0, 10.0),
        files: &[(84, 6), (84, 9), (84, 12), (87, 6), (87, 9), (87, 12)],
        never_indexed: &[],
        capacity: 4096,
        tile: 64,
        windows: 600,
        rounds: 6,
        small: false,
        reach: 9500,
        leave_permille: 20,
        blank_permille: 20,
    }
    .run();
    assert!(stats.windows >= 3600, "{}", stats.windows);
    assert!(stats.ok_windows > 1000);
    assert!(stats.no_data > 1000);
}

#[test]
fn window_fast_path_is_bit_identical_when_the_memory_cache_holds_next_to_nothing() {
    // Tiles come back from disk, at every sample of the reference; the windows are small for that.
    for (name, capacity) in [
        ("memory of none", 0),
        ("memory of one", 1),
        ("memory of three", 3),
    ] {
        let stats = Scenario {
            name,
            origin: (3.0, 87.0),
            files: &JUNCTION,
            never_indexed: &[],
            capacity,
            tile: 64,
            windows: 120,
            rounds: 4,
            small: true,
            reach: 3500,
            leave_permille: 0,
            blank_permille: 30,
        }
        .run();
        assert!(stats.windows >= 480, "{name}: {}", stats.windows);
        assert!(stats.ok_windows > 100, "{name}");
    }
}

#[test]
fn window_fast_path_is_bit_identical_with_specks_in_the_tiles() {
    let dir = TempDir::new();
    let scenario = Scenario {
        name: "specks",
        origin: (3.0, 87.0),
        files: &JUNCTION,
        never_indexed: &[],
        capacity: 4096,
        tile: 64,
        windows: 150,
        rounds: 6,
        small: false,
        reach: 6000,
        leave_permille: 0,
        blank_permille: 0,
    };
    let field = scenario.field(&dir);
    let files = scenario.files();
    let store = Arc::clone(field.store());
    let mut rng = Rng(99);
    let edges = scenario.edges(&field);
    let wins = scenario.windows(&mut rng, (&edges.0, &edges.1));
    // Everything the windows touch is held first, so every window gives classes.
    for win in &wins {
        for _ in 0..2000 {
            let at = field.class_grid_reference(
                win.x0, win.z0, win.width, win.depth, win.stride, win.metres,
            );
            match at {
                Err(f) => {
                    scenario.answer(&store, &files, &f);
                }
                Ok(_) => break,
            }
        }
    }
    let speckled = lock(&store.memory)
        .tiles
        .values()
        .filter(|(tile, _)| !tile.specks.is_empty())
        .count();
    assert!(speckled > 20, "{speckled} tiles with specks");
    let mut samples = 0;
    for win in &wins {
        let fast = field.class_grid(win.x0, win.z0, win.width, win.depth, win.stride, win.metres);
        let slow = field.class_grid_reference(
            win.x0, win.z0, win.width, win.depth, win.stride, win.metres,
        );
        assert_eq!(fast, slow, "{win:?}");
        samples += slow.map_or(0, |v| v.len());
    }
    assert!(samples > 20_000, "{samples}");
    eprintln!(
        "[specks] {} windows, {samples} samples, {speckled} tiles with specks",
        wins.len()
    );
}

/// The warp of a window, a column and a row at a time, is [`LandCoverField::warped`] to the last bit, with
/// the lattice in a table and without.
#[test]
fn window_warp_is_bit_identical_to_the_per_sample_warp() {
    let dir = TempDir::new();
    let field = {
        let store = Arc::new(LandCoverStore::new(dir.path(), 1));
        let projection = Projection::local(30.0, 84.0, metres_per_block()).unwrap();
        LandCoverField::new(store, projection).with_level0_pixel_metres(pixel_metres())
    };
    let mut rng = Rng(5);
    let (mut tabled, mut direct, mut samples) = (0, 0, 0);
    for _ in 0..1500 {
        let x0 = rng.below(400_001) as i32 - 200_000;
        let z0 = rng.below(400_001) as i32 - 200_000;
        let (width, depth) = (rng.below(40) as usize + 1, rng.below(40) as usize + 1);
        let stride = rng.pick(&STRIDES);
        let metres = rng.pick(&METRES);
        let pixel = field.pixel_blocks(metres);
        let xs: Vec<f64> = (0..width)
            .map(|i| f64::from(x0) + (i as f64) * f64::from(stride) + 0.5)
            .collect();
        let zs: Vec<f64> = (0..depth)
            .map(|j| f64::from(z0) + (j as f64) * f64::from(stride) + 0.5)
            .collect();
        let coarse = Octave::new(
            (LAND_WARP_COARSE_WAVELENGTH_PX, LAND_WARP_COARSE_PX),
            [WARP_SALTS[0], WARP_SALTS[1]],
            pixel,
            &xs,
            &zs,
        );
        let fine = Octave::new(
            (LAND_WARP_FINE_WAVELENGTH_PX, LAND_WARP_FINE_PX),
            [WARP_SALTS[2], WARP_SALTS[3]],
            pixel,
            &xs,
            &zs,
        );
        for octave in [&coarse, &fine] {
            if octave.tables.is_some() {
                tabled += 1;
            } else {
                direct += 1;
            }
        }
        for (j, &z) in zs.iter().enumerate() {
            for (i, &x) in xs.iter().enumerate() {
                let (cx, cz) = coarse.offset(i, j);
                let (fx, fz) = fine.offset(i, j);
                let got = (x + cx + fx, z + cz + fz);
                let want = field.warped(x, z, metres);
                assert_eq!(
                    (got.0.to_bits(), got.1.to_bits()),
                    (want.0.to_bits(), want.1.to_bits()),
                    "({x}, {z}) at {metres} m, stride {stride}"
                );
                samples += 1;
            }
        }
    }
    eprintln!("[warp] {samples} samples, {tabled} octaves with a table, {direct} without");
    assert!(tabled > 300 && direct > 300 && samples > 100_000);
}

#[test]
fn window_of_no_samples_or_a_huge_one_is_what_the_reference_gives() {
    let dir = TempDir::new();
    let scenario = Scenario {
        name: "empty",
        origin: (3.0, 87.0),
        files: &JUNCTION,
        never_indexed: &[],
        capacity: 16,
        tile: 64,
        windows: 0,
        rounds: 0,
        small: false,
        reach: 0,
        leave_permille: 0,
        blank_permille: 0,
    };
    let field = scenario.field(&dir);
    for (w, d) in [(0, 0), (0, 9), (9, 0)] {
        assert_eq!(field.class_grid(0, 0, w, d, 1, 100.0), Ok(Vec::new()));
        assert_eq!(
            field.class_grid(0, 0, w, d, 1, 100.0),
            field.class_grid_reference(0, 0, w, d, 1, 100.0)
        );
    }
    // Nothing is held, so a window of any size stops at the first missing piece, as the reference does.
    for (w, d) in [(usize::MAX, 3), (1 << 21, 1 << 21), (5, usize::MAX)] {
        let fast = field.class_grid(0, 0, w, d, 1, 100.0);
        assert!(matches!(fast, Err(f) if f.wanted == Wanted::Index));
        assert_eq!(fast, field.class_grid_reference(0, 0, w, d, 1, 100.0));
    }
}

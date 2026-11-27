//! Breaks down the cost of `LandCoverField::class_grid` per 38x38 window (the water window of a 16x16 node:
//! 16 + 2 * WINDOW_PAD) on real, cached data, and checks `class_grid` against `class_grid_reference`.
//! Usage: earth_class_profile <lat> <lon> <cache dir> [rounds]
//!
//! Never downloads: the cache must hold the nodes `earth_far_profile` loads (levels 2..=8, 4x4 nodes each);
//! windows the store lacks bytes for are skipped (and counted). Single thread.
//! Stages (`class_grid_stage`): 0 loop only, 1 warp, 2 + projection, 3 + store lookup up to the own tile,
//! 4 + speck and 4-pixel vote (= the whole reference read), 5 = 3 and 6 = 4 with file/index/tile resolved once
//! per window.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use horizon_earth::tiff::ByteRange;
use horizon_earth::{
    FetchError, LandCoverFetcher, LandCoverStore, TileFetcher, TileId, TileStore, WorldCoverFile,
};
use horizon_world::cubic::custom::testing::default_registry;
use horizon_world::cubic::earth::{EarthBlocks, EarthGenerator, EarthSettings};

const TILE_URL: &str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
const LAND_URL: &str = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/";
/// water.rs: `WINDOW_PAD = PAD + CONTAINMENT_RADIUS` (11 as of writing).
const WINDOW_PAD: i32 = 11;
const SIDE: usize = 16 + 2 * WINDOW_PAD as usize;

/// `curl` the url (a byte range when given): `Ok(None)` on 404. Only here to build the stores the way
/// `earth_far_profile` does; this example never calls a fetching API.
fn curl(url: &str, range: Option<ByteRange>) -> Result<Option<Vec<u8>>, FetchError> {
    let mut cmd = Command::new("curl");
    cmd.args(["-sS", "-L", "--max-time", "120", "-w", "\n%{http_code}"]);
    if let Some(r) = range {
        cmd.arg("-r").arg(format!("{}-{}", r.offset, r.offset + r.len.max(1) - 1));
    }
    let out = cmd.arg(url).output().map_err(|e| FetchError::Temporary(e.to_string()))?;
    if !out.status.success() {
        return Err(FetchError::Temporary(String::from_utf8_lossy(&out.stderr).into_owned()));
    }
    let body = out.stdout;
    let cut = body.iter().rposition(|&b| b == b'\n').unwrap_or(0);
    let code: u32 = String::from_utf8_lossy(&body[cut + 1..]).trim().parse().unwrap_or(0);
    let mut bytes = body;
    bytes.truncate(cut);
    match code {
        200 | 206 => Ok(Some(bytes)),
        404 | 403 => Ok(None),
        _ => Err(FetchError::Temporary(format!("HTTP {code} for {url}"))),
    }
}

struct Curl;

impl TileFetcher for Curl {
    fn fetch(&self, id: TileId) -> Result<Vec<u8>, FetchError> {
        let url = TILE_URL
            .replace("{z}", &id.z.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{y}", &id.y.to_string());
        curl(&url, None)?.ok_or(FetchError::NotFound)
    }
}

impl LandCoverFetcher for Curl {
    fn fetch(&self, file: WorldCoverFile, range: ByteRange) -> Result<Option<Vec<u8>>, FetchError> {
        curl(&format!("{LAND_URL}{}", file.path()), Some(range))
    }
}

/// The first folder under `dir`, or `dir/probe`.
fn keyed(dir: &Path) -> PathBuf {
    std::fs::read_dir(dir)
        .ok()
        .and_then(|mut it| it.find_map(|e| e.ok().filter(|e| e.path().is_dir()).map(|e| e.path())))
        .unwrap_or_else(|| dir.join("probe"))
}

/// Milliseconds per window of `f` over `windows`: the best of 5 batches of `rounds` passes (after one warm-up
/// pass), so other processes' load on the machine shows less.
fn time(windows: &[(i32, i32)], rounds: usize, mut f: impl FnMut(i32, i32)) -> f64 {
    for &(x, z) in windows {
        f(x, z);
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..rounds {
            for &(x, z) in windows {
                f(x, z);
            }
        }
        best = best.min(t.elapsed().as_secs_f64() * 1000.0 / (rounds * windows.len()) as f64);
    }
    best
}

const NAMES: [&str; 9] = [
    "reference", "class_grid", "s0 loop", "s1 warp", "s2 +proj", "s3 +tile_of", "s4 +vote", "s5 s3 cached", "s6 s4 cached",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let lat: f64 = args[0].parse().unwrap();
    let lon: f64 = args[1].parse().unwrap();
    let cache = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| "earth-probe-cache".into()));
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let json = format!(
        r#"{{"originLatitude": {lat}, "originLongitude": {lon}, "tileUrl": "{TILE_URL}", "allowDownloads": true,
            "landCover": true, "vegetation": true, "climate": true, "snow": true}}"#
    );
    let settings = EarthSettings::from_json(&json).expect("settings");
    let tiles = Arc::new(TileStore::new(keyed(&cache.join("earth-tiles")), Arc::new(Curl), 512));
    let land = settings
        .land_cover_source(Arc::new(LandCoverStore::new(keyed(&cache.join("earth-landcover")), 256)), Arc::new(Curl))
        .expect("land cover");
    let blocks = EarthBlocks::from_registry(default_registry()).expect("blocks");
    let mut generator = EarthGenerator::new(&settings, tiles, blocks).expect("generator").with_land_cover(land);
    if let Some(atlas) = horizon_earth::WaterAtlas::embedded() {
        generator = generator.with_water(atlas);
    }
    let source = generator.land_source().expect("land source").clone();
    let field = source.field();
    let mpb = field.projection().metres_per_block();
    println!("spot {lat} {lon}  metres/block {mpb:.4}  window {SIDE}x{SIDE}  rounds {rounds}");

    let (mut n_win, mut n_err, mut n_mismatch, mut n_samples) = (0usize, 0usize, 0usize, 0usize);
    let mut rows: Vec<(u32, usize, [f64; 9])> = Vec::new();
    for level in 2..=8u32 {
        let size = 16i32 << level;
        let stride = 1i32 << level;
        let metres = f64::from(stride) * mpb;
        let mut ok: Vec<(i32, i32)> = Vec::new();
        for j in -2..2 {
            for i in -2..2 {
                let (sx, sz) = (i * size + stride / 2, j * size + stride / 2);
                let (px, pz) = (sx - WINDOW_PAD * stride, sz - WINDOW_PAD * stride);
                n_win += 1;
                let r = field.class_grid_reference(px, pz, SIDE, SIDE, stride, metres);
                let c = field.class_grid(px, pz, SIDE, SIDE, stride, metres);
                if r != c {
                    n_mismatch += 1;
                    eprintln!("MISMATCH level {level} window ({px}, {pz}): {:?} vs {:?}", r.as_ref().err(), c.as_ref().err());
                }
                match r {
                    Ok(v) => {
                        n_samples += v.len();
                        ok.push((px, pz));
                    }
                    Err(_) => n_err += 1,
                }
            }
        }
        if ok.is_empty() {
            println!("level {level}: no window held");
            continue;
        }
        let mut ms = [0.0f64; 9];
        ms[0] = time(&ok, rounds, |x, z| {
            black_box(field.class_grid_reference(x, z, SIDE, SIDE, stride, metres)).ok();
        });
        ms[1] = time(&ok, rounds, |x, z| {
            black_box(field.class_grid(x, z, SIDE, SIDE, stride, metres)).ok();
        });
        for stage in 0..=4u8 {
            ms[2 + stage as usize] = time(&ok, rounds, |x, z| {
                assert_ne!(black_box(field.class_grid_stage(x, z, SIDE, SIDE, stride, metres, stage)), u64::MAX);
            });
        }
        // stage 5 and 6 are 3 and 4 with a per-window cache; the checksums must agree with the uncached ones.
        for &(x, z) in &ok {
            let a = field.class_grid_stage(x, z, SIDE, SIDE, stride, metres, 3);
            let b = field.class_grid_stage(x, z, SIDE, SIDE, stride, metres, 5);
            let c = field.class_grid_stage(x, z, SIDE, SIDE, stride, metres, 4);
            let d = field.class_grid_stage(x, z, SIDE, SIDE, stride, metres, 6);
            assert!(a == b && c == d, "cached stage checksum differs at level {level} ({x}, {z})");
        }
        ms[7] = time(&ok, rounds, |x, z| {
            black_box(field.class_grid_stage(x, z, SIDE, SIDE, stride, metres, 5));
        });
        ms[8] = time(&ok, rounds, |x, z| {
            black_box(field.class_grid_stage(x, z, SIDE, SIDE, stride, metres, 6));
        });
        rows.push((level, ok.len(), ms));
    }

    print!("\n{:>5} {:>4}", "level", "n");
    for n in NAMES {
        print!(" {n:>12}");
    }
    println!("   (ms/window)");
    let mut avg = [0.0f64; 9];
    for (level, n, ms) in &rows {
        print!("{level:>5} {n:>4}");
        for (k, v) in ms.iter().enumerate() {
            print!(" {v:>12.3}");
            avg[k] += v / rows.len() as f64;
        }
        println!();
    }
    print!("{:>5} {:>4}", "avg", "");
    for v in avg {
        print!(" {v:>12.3}");
    }
    println!();
    println!(
        "breakdown (avg over levels, ms/window): total {:.3} = loop {:.3} + warp {:.3} + projection {:.3} + lookups/tile_of/locks {:.3} + speck+vote {:.3}  (s4 {:.3}); cached lookups (s5-s2) {:.3}",
        avg[0],
        avg[2],
        avg[3] - avg[2],
        avg[4] - avg[3],
        avg[5] - avg[4],
        avg[6] - avg[5],
        avg[6],
        avg[7] - avg[4]
    );
    println!(
        "class_grid vs reference: {:.3} vs {:.3} ms/window = {:.2}x",
        avg[1],
        avg[0],
        avg[0] / avg[1]
    );
    println!("windows {n_win}  skipped(Err) {n_err}  samples {n_samples}  mismatches {n_mismatch}");
    if n_mismatch != 0 {
        std::process::exit(1);
    }
}

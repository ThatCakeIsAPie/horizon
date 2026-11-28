//! Measures the far view's coarse levels (voxels of 4 km and up) on real data: the bytes of the whole-world coarse
//! set, the time to decode it and build the sea mask, and the CPU time of a coarse node at levels 12 to 22.
//!
//! ```sh
//! cargo run --release -p horizon-world --example earth_coarse_profile -- <lat> <lon> <cache dir>
//! ```
//!
//! `<cache dir>` holds `earth-tiles/<key>` and `earth-climate/<key>` as a game folder's `.horizon-cache` does (use
//! a copy: what is missing is downloaded into it with `curl`).

#[allow(unused_imports)]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

#[allow(unused_imports)]
use horizon_earth::LatLon;
use horizon_earth::tiff::ByteRange;
use horizon_earth::{
    ClimateFetcher, ClimateStore, ClimateVar, FetchError, LandCoverFetcher, LandCoverStore,
    TileFetcher, TileId, TileStore, WorldCoverFile,
};
use horizon_world::cubic::custom::testing::default_registry;
use horizon_world::cubic::earth::{EarthBlocks, EarthGenerator, EarthSettings};

const TILE_URL: &str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
const LAND_URL: &str = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/";
const CLIMATE_URL: &str = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim/";

/// `curl` the url (a byte range when given): `Ok(None)` on 404.
fn curl(url: &str, range: Option<ByteRange>) -> Result<Option<Vec<u8>>, FetchError> {
    let mut cmd = Command::new("curl");
    cmd.args(["-sS", "-L", "--max-time", "120", "-w", "\n%{http_code}"]);
    if let Some(r) = range {
        cmd.arg("-r")
            .arg(format!("{}-{}", r.offset, r.offset + r.len.max(1) - 1));
    }
    let out = cmd
        .arg(url)
        .output()
        .map_err(|e| FetchError::Temporary(e.to_string()))?;
    if !out.status.success() {
        return Err(FetchError::Temporary(
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ));
    }
    let body = out.stdout;
    let cut = body.iter().rposition(|&b| b == b'\n').unwrap_or(0);
    let code: u32 = String::from_utf8_lossy(&body[cut + 1..])
        .trim()
        .parse()
        .unwrap_or(0);
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

impl ClimateFetcher for Curl {
    fn fetch(&self, var: ClimateVar, range: ByteRange) -> Result<Option<Vec<u8>>, FetchError> {
        curl(&format!("{CLIMATE_URL}{}", var.path()), Some(range))
    }
}

/// The first folder under `dir`, or `dir/probe`.
fn keyed(dir: &Path) -> PathBuf {
    std::fs::read_dir(dir)
        .ok()
        .and_then(|mut it| it.find_map(|e| e.ok().filter(|e| e.path().is_dir()).map(|e| e.path())))
        .unwrap_or_else(|| dir.join("probe"))
}

/// The bytes of the tiles of zooms `0..=zoom` under the tile cache `dir`, and how many there are.
fn bytes_of_zooms(dir: &Path, zoom: u8) -> (u64, usize) {
    let mut bytes = 0;
    let mut n = 0;
    for z in 0..=zoom {
        let Ok(xs) = std::fs::read_dir(dir.join(z.to_string())) else {
            continue;
        };
        for x in xs.flatten() {
            let Ok(ys) = std::fs::read_dir(x.path()) else {
                continue;
            };
            for y in ys.flatten() {
                if let Ok(m) = y.metadata() {
                    bytes += m.len();
                    n += 1;
                }
            }
        }
    }
    (bytes, n)
}

/// The bytes of every file under `dir`.
fn bytes_under(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                total += bytes_under(&p);
            } else if let Ok(m) = e.metadata() {
                total += m.len();
            }
        }
    }
    total
}

fn generator(settings: &EarthSettings, tile_dir: &Path, climate_dir: &Path) -> EarthGenerator {
    let tiles = Arc::new(TileStore::new(tile_dir, Arc::new(Curl), 4096));
    let land = settings
        .land_cover_source(
            Arc::new(LandCoverStore::new(tile_dir.join("unused-landcover"), 16)),
            Arc::new(Curl),
        )
        .expect("land cover");
    let climate = settings
        .climate_source(Arc::new(ClimateStore::new(climate_dir, 64)), Arc::new(Curl))
        .expect("climate");
    let blocks = EarthBlocks::from_registry(default_registry()).expect("blocks");
    let mut g = EarthGenerator::new(settings, tiles, blocks)
        .expect("generator")
        .with_land_cover(land)
        .with_climate(climate);
    if let Some(atlas) = horizon_earth::WaterAtlas::embedded() {
        g = g.with_water(atlas);
    }
    g
}

/// Asks `read` until it answers, fetching the tiles and climate it names; the tile ids it named.
fn loaded<T>(
    g: &EarthGenerator,
    read: impl Fn() -> Result<T, horizon_world::cubic::earth::NotReady>,
) -> (Option<T>, Vec<TileId>) {
    let mut named = Vec::new();
    for _ in 0..40 {
        match read() {
            Ok(v) => return (Some(v), named),
            Err(missing) => {
                for &id in &missing.tiles {
                    named.push(id);
                    let _ = g.field().store().load(id);
                }
                if let Some(c) = g.climate_source() {
                    for fetch in g.take_climate_requests() {
                        let _ = c.satisfy(&fetch);
                    }
                }
            }
        }
    }
    (None, named)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let lat: f64 = args[0].parse().unwrap();
    let lon: f64 = args[1].parse().unwrap();
    let cache = PathBuf::from(
        args.get(2)
            .cloned()
            .unwrap_or_else(|| "earth-coarse-cache".into()),
    );
    let json = format!(
        r#"{{"originLatitude": {lat}, "originLongitude": {lon}, "tileUrl": "{TILE_URL}", "allowDownloads": true,
            "landCover": true, "vegetation": true, "climate": true, "snow": true}}"#
    );
    let settings = EarthSettings::from_json(&json).expect("settings");
    let tile_dir = keyed(&cache.join("earth-tiles"));
    let climate_dir = keyed(&cache.join("earth-climate"));
    let (before_bytes, before_n) = bytes_of_zooms(&tile_dir, horizon_earth::COARSE_PREFETCH_ZOOM);
    let climate_before = bytes_under(&climate_dir);

    // The coarse set, as the prefetcher fetches it at world load.
    let g = generator(&settings, &tile_dir, &climate_dir);
    let t = std::time::Instant::now();
    let missing = g.coarse_tiles_missing((0, 0));
    let n_missing = missing.len();
    for id in missing {
        g.field().store().load(id).expect("coarse tile");
    }
    let climate_needs = g.coarse_climate_needs();
    let n_climate = climate_needs.len();
    if let Some(c) = g.climate_source() {
        for fetch in &climate_needs {
            let _ = c.satisfy(fetch);
        }
    }
    let fetched = t.elapsed();
    let (bytes, n) = bytes_of_zooms(&tile_dir, horizon_earth::COARSE_PREFETCH_ZOOM);
    let climate_after = bytes_under(&climate_dir);
    println!(
        "coarse tiles: {n} tiles, {:.2} MB on disk (fetched {n_missing} now, {:.2} MB; {before_n} were cached), climate pieces {n_climate} ({:.2} MB fetched), {:.1} s",
        bytes as f64 / 1e6,
        (bytes - before_bytes) as f64 / 1e6,
        (climate_after - climate_before) as f64 / 1e6,
        fetched.as_secs_f64()
    );

    // Decode from disk into a cold generator and build the sea mask.
    let g = generator(&settings, &tile_dir, &climate_dir);
    let t = std::time::Instant::now();
    for id in horizon_earth::coarse_tiles(horizon_earth::LatLon { lat, lon }) {
        g.field().store().load(id).expect("cached coarse tile");
    }
    let decode = t.elapsed();
    let t = std::time::Instant::now();
    g.prepare_coarse().expect("sea mask");
    let mask = t.elapsed();
    println!(
        "decode {:.0} ms ({:.2} ms a tile), sea mask {:.0} ms",
        decode.as_secs_f64() * 1000.0,
        decode.as_secs_f64() * 1000.0 / n.max(1) as f64,
        mask.as_secs_f64() * 1000.0
    );

    // Nodes of every coarse level in a ring around the origin, out to a few thousand km.
    for level in [12u32, 14, 16, 18, 20, 22] {
        let size = 16i64 << level;
        let mut nodes = Vec::new();
        for k in 0..16i64 {
            let a = k as f64 / 16.0 * std::f64::consts::TAU;
            let r = (size as f64 * 3.0).min(6.0e6);
            let (x, z) = ((a.cos() * r) as i64, (a.sin() * r) as i64);
            nodes.push((x.div_euclid(size) as i32, z.div_euclid(size) as i32));
        }
        let mut tiles = std::collections::BTreeSet::new();
        let mut ok = 0;
        for &(nx, nz) in &nodes {
            let (cols, named) = loaded(&g, || g.node_columns(level, nx, nz));
            tiles.extend(named);
            ok += usize::from(cols.is_some());
        }
        let mut times = Vec::new();
        for _ in 0..5 {
            for &(nx, nz) in &nodes {
                let t = std::time::Instant::now();
                let r = g.node_columns(level, nx, nz);
                let e = t.elapsed();
                if r.is_ok() {
                    times.push(e.as_secs_f64() * 1000.0);
                }
            }
        }
        times.sort_by(f64::total_cmp);
        let median = times.get(times.len() / 2).copied().unwrap_or(f64::NAN);
        let worst = times.last().copied().unwrap_or(f64::NAN);
        let zooms: std::collections::BTreeSet<u8> = tiles.iter().map(|t| t.z).collect();
        println!(
            "level {level}: coarse {} zoom {} | {ok}/16 ready | {:.3} ms/node median, {:.3} worst | {} more tiles named (zooms {zooms:?})",
            g.is_coarse_step(1 << level),
            g.zoom_for_step(1 << level),
            median,
            worst,
            tiles.len()
        );
    }
}

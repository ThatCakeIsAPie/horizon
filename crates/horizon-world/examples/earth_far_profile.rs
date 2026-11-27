//! Times the far view's column reads (EarthGenerator::columns_at) per level and per step, on real data.
//! Usage: earth_far_profile <lat> <lon> <cache dir>
//!
//! Headless look at Earth terrain: the generator a world with
//! `{originLatitude, originLongitude, tileUrl: terrarium, allowDownloads, landCover, vegetation, climate, snow}`
//! makes, printed as ASCII maps of surface kinds (near rules and far rules) and land cover classes around the
//! origin, plus a few columns in detail. A dev tool: it downloads with `curl` and caches like the game does.
//!
//! ```sh
//! cargo run --release -p horizon-world --example earth_probe -- <lat> <lon> [--cache DIR] [--size 4000] [--cell 50]
//! ```
//!
//! `--cache DIR` holds `earth-tiles/<key>`, `earth-landcover/<key>` and `earth-climate/<key>` as a game folder's
//! `.horizon-cache` does (the first folder under each is used, or `probe` when there is none).

use std::path::{Path, PathBuf};
#[allow(unused_imports)]
use std::io::Write;
use std::process::Command;
use std::sync::Arc;

use horizon_earth::tiff::ByteRange;
use horizon_earth::{
    ClimateFetcher, ClimateStore, ClimateVar, FetchError, LandCoverFetcher,
    LandCoverStore, TileFetcher, TileId, TileStore, WorldCoverFile,
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


fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let lat: f64 = args[0].parse().unwrap();
    let lon: f64 = args[1].parse().unwrap();
    let cache = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| "earth-probe-cache".into()));
    let json = format!(
        r#"{{"originLatitude": {lat}, "originLongitude": {lon}, "tileUrl": "{TILE_URL}", "allowDownloads": true,
            "landCover": true, "vegetation": true, "climate": true, "snow": true}}"#
    );
    let settings = EarthSettings::from_json(&json).expect("settings");
    let tiles = Arc::new(TileStore::new(keyed(&cache.join("earth-tiles")), Arc::new(Curl), 512));
    let land = settings
        .land_cover_source(Arc::new(LandCoverStore::new(keyed(&cache.join("earth-landcover")), 256)), Arc::new(Curl))
        .expect("land cover");
    let climate = settings
        .climate_source(Arc::new(ClimateStore::new(keyed(&cache.join("earth-climate")), 64)), Arc::new(Curl))
        .expect("climate");
    let blocks = EarthBlocks::from_registry(default_registry()).expect("blocks");
    let mut generator = EarthGenerator::new(&settings, tiles, blocks)
        .expect("generator")
        .with_land_cover(land.clone())
        .with_climate(climate.clone());
    if let Some(atlas) = horizon_earth::WaterAtlas::embedded() {
        generator = generator.with_water(atlas);
    }
    let g = generator;
    let mut keys = Vec::new();
    for level in 2..=8u32 {
        let size = 16i32 << level;
        for j in -2..2 {
            for i in -2..2 {
                keys.push((level, i * size, j * size));
            }
        }
    }
    // Load.
    let t = std::time::Instant::now();
    for &(level, x, z) in &keys {
        let step = 1 << level;
        let _ = g.columns_blocking([x + step / 2, z + step / 2], [16, 16], step, step);
    }
    eprintln!("loaded {} nodes in {:?}", keys.len(), t.elapsed());
    let mut total = std::collections::BTreeMap::<u32, (std::time::Duration, usize)>::new();
    let mut st_sum = horizon_world::cubic::earth::far_profile::Stages::default();
    let mut n_ok = 0;
    for _round in 0..3 {
        for &(level, x, z) in &keys {
            let step = 1 << level;
            let t = std::time::Instant::now();
            let ok = g.columns_at(x, z, step, step).is_ok();
            let e = t.elapsed();
            if ok {
                let s = total.entry(level).or_default();
                s.0 += e;
                s.1 += 1;
            }
            if let Some(st) = g.columns_at_stages(x, z, step, step) {
                n_ok += 1;
                st_sum.surfaces += st.surfaces;
                st_sum.window_dem += st.window_dem;
                st_sum.window_class += st.window_class;
                st_sum.rivers += st.rivers;
                st_sum.lakes += st.lakes;
                st_sum.water_rules += st.water_rules;
                st_sum.kinds += st.kinds;
                st_sum.window_samples = st.window_samples;
            }
        }
    }
    for (level, (d, n)) in &total {
        println!("level {level}: columns_at {:.3} ms/node ({n})", d.as_secs_f64() * 1000.0 / *n as f64);
    }
    let per = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / n_ok.max(1) as f64;
    println!(
        "stages ms/node over {n_ok}: surfaces {:.3} window_dem {:.3} window_class {:.3} rivers {:.3} water_rules {:.3} kinds {:.3} (window {} samples)",
        per(st_sum.surfaces), per(st_sum.window_dem), per(st_sum.window_class), per(st_sum.rivers),
        per(st_sum.water_rules), per(st_sum.kinds), st_sum.window_samples
    );
}

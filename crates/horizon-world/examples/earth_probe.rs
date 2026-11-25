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
use std::process::Command;
use std::sync::Arc;

use horizon_earth::tiff::ByteRange;
use horizon_earth::{
    ClimateFetcher, ClimateStore, ClimateVar, FetchError, LandClass, LandCoverFetcher,
    LandCoverStore, TileFetcher, TileId, TileStore, WorldCoverFile,
};
use horizon_world::cubic::custom::testing::default_registry;
use horizon_world::cubic::earth::{EarthBlocks, EarthGenerator, EarthSettings, SurfaceKind};

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

/// The character of a column: water at the sea's top (`sea_top`) is the sea.
fn kind_char(s: i32, kind: SurfaceKind, sea_top: i32) -> char {
    match kind {
        SurfaceKind::Water { .. } if s == sea_top => '=',
        SurfaceKind::Water { .. } => '~',
        SurfaceKind::Marsh => 'm',
        SurfaceKind::Grass => 'g',
        SurfaceKind::Alpine => 'A',
        SurfaceKind::Snow => 'S',
        SurfaceKind::Dune => 's',
        SurfaceKind::Gravel => '.',
        SurfaceKind::Moss(_) => 'M',
        SurfaceKind::BuiltUp(_) => 'u',
        SurfaceKind::Banded => 'b',
        SurfaceKind::Scrub(_) => 'c',
        SurfaceKind::Barren(_) => 'r',
        SurfaceKind::Glacier => 'I',
        SurfaceKind::Icefall => 'i',
        SurfaceKind::Basalt(_) => 'k',
    }
}

fn class_char(class: LandClass) -> char {
    match class {
        LandClass::NoData => ' ',
        LandClass::Tree => 'T',
        LandClass::Shrub => 'h',
        LandClass::Grass => 'g',
        LandClass::Crop => 'c',
        LandClass::BuiltUp => '#',
        LandClass::Bare => 'b',
        LandClass::SnowIce => 'I',
        LandClass::Water => '~',
        LandClass::Wetland => 'w',
        LandClass::Mangrove => 'v',
        LandClass::MossLichen => 'M',
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let (mut cache, mut size, mut cell) = (PathBuf::from("earth-probe-cache"), 4000i32, 50i32);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cache" => cache = PathBuf::from(it.next().expect("--cache DIR")),
            "--size" => size = it.next().expect("--size N").parse().expect("--size N"),
            "--cell" => cell = it.next().expect("--cell N").parse().expect("--cell N"),
            other => pos.push(other.parse::<f64>().expect("lat lon")),
        }
    }
    let [lat, lon] = pos[..] else {
        eprintln!("usage: earth_probe <lat> <lon> [--cache DIR] [--size BLOCKS] [--cell BLOCKS]");
        std::process::exit(2);
    };
    let json = format!(
        r#"{{"originLatitude": {lat}, "originLongitude": {lon}, "tileUrl": "{TILE_URL}", "allowDownloads": true,
            "landCover": true, "vegetation": true, "climate": true, "snow": true}}"#
    );
    let settings = EarthSettings::from_json(&json).expect("settings");
    let tiles = Arc::new(TileStore::new(
        keyed(&cache.join("earth-tiles")),
        Arc::new(Curl),
        512,
    ));
    let land = settings
        .land_cover_source(
            Arc::new(LandCoverStore::new(
                keyed(&cache.join("earth-landcover")),
                256,
            )),
            Arc::new(Curl),
        )
        .expect("land cover");
    let climate = settings
        .climate_source(
            Arc::new(ClimateStore::new(keyed(&cache.join("earth-climate")), 64)),
            Arc::new(Curl),
        )
        .expect("climate");
    let blocks = EarthBlocks::from_registry(default_registry()).expect("blocks");
    let mut generator = EarthGenerator::new(&settings, tiles, blocks)
        .expect("generator")
        .with_land_cover(land.clone())
        .with_climate(climate.clone());
    if let Some(atlas) = horizon_earth::WaterAtlas::embedded() {
        generator = generator.with_water(atlas);
    }
    let generator = Arc::new(generator);
    let n = (size / cell).max(1) as usize;
    let x0 = -(n as i32 / 2) * cell;
    println!(
        "Earth probe at {lat}, {lon}: {n}x{n} cells of {cell} blocks (1 block = {} m), sea level y {}, north up",
        settings.metres_per_block, settings.sea_level_y
    );

    let (nw, se) = (
        generator.lat_lon_at(x0, x0),
        generator.lat_lon_at(x0 + (n as i32 - 1) * cell, x0 + (n as i32 - 1) * cell),
    );
    println!(
        "first row/column at lat {:.4} lon {:.4}, last at lat {:.4} lon {:.4}",
        nw.0, nw.1, se.0, se.1
    );

    // Near: each cell's middle column as a near cube makes it (stride 1).
    let mut near = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let (x, z) = (x0 + i as i32 * cell, x0 + j as i32 * cell);
            let c = generator
                .columns_blocking([x, z], [1, 1], 1, 1)
                .expect("columns");
            near.push((c.surfaces[0], c.kind(0), c.class(0)));
        }
    }
    // Far: the same lattice as one far read at the cell's spacing.
    let far = generator
        .columns_blocking([x0, x0], [n, n], cell, cell)
        .expect("far columns");
    // The 10 m map at each cell.
    let class10: Vec<LandClass> = (0..n * n)
        .map(|k| {
            let (x, z) = (x0 + (k % n) as i32 * cell, x0 + (k / n) as i32 * cell);
            land.class_at_blocking(f64::from(x) + 0.5, f64::from(z) + 0.5, 1.0)
                .unwrap_or(LandClass::NoData)
        })
        .collect();

    let legend = "kinds: = water at sea level, ~ other water, m marsh, g grass, . gravel, s sand, A stone, S snow, I glacier, i icefall, M moss, k basalt, c scrub, r barren, b banded";
    println!("{legend}");
    println!(
        "classes: ~ water, # built-up, T tree, h shrub, g grass, c crop, b bare, I snow/ice, w wetland, v mangrove, M moss, ' ' no data"
    );
    let center = n / 2;
    let sea_top = settings.sea_level_y - 1;
    let mark = |row: usize, col: usize, ch: char| {
        if row == center && col == center {
            '@'
        } else {
            ch
        }
    };
    println!(
        "\n{:<w$}  {:<w$}  {:<w$}",
        "NEAR kinds",
        "FAR kinds",
        "WorldCover 10 m",
        w = n
    );
    for j in 0..n {
        let a: String = (0..n)
            .map(|i| {
                mark(
                    j,
                    i,
                    kind_char(near[j * n + i].0, near[j * n + i].1, sea_top),
                )
            })
            .collect();
        let b: String = (0..n)
            .map(|i| {
                mark(
                    j,
                    i,
                    kind_char(far.surfaces[j * n + i], far.kind(j * n + i), sea_top),
                )
            })
            .collect();
        let c: String = (0..n)
            .map(|i| mark(j, i, class_char(class10[j * n + i])))
            .collect();
        println!("{a}  {b}  {c}");
    }
    println!("\nColumns (x, z): s, kind, class, elevation surface m, biome, mean temp C");
    let offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];
    let q = (n as i32 / 4).max(1);
    for (di, dj) in offsets {
        let (i, j) = (
            (center as i32 + di * q) as usize,
            (center as i32 + dj * q) as usize,
        );
        let (x, z) = (x0 + i as i32 * cell, x0 + j as i32 * cell);
        let (s, kind, class) = near[j * n + i];
        let s0 = generator.surface_y_blocking(x, z).unwrap_or(i32::MIN);
        let elevation = generator.field().vertical().elevation_for_y(f64::from(s0));
        let biome = generator.biome_at_quart_blocking(x >> 2, z >> 2);
        let t = climate
            .climate_at_blocking(f64::from(x) + 0.5, f64::from(z) + 0.5, 1000.0)
            .ok()
            .flatten()
            .map(|c| c.temperature_c);
        println!(
            "  ({x:6}, {z:6}): s {s:4} {kind:?} {class:?}, dem y {s0} = {elevation:.1} m, {}, {t:?}",
            biome.name()
        );
    }
}

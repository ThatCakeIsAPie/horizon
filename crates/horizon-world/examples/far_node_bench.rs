//! Times what one Earth far node "built from surfaces" costs, stage by stage, over real decoded tiles.
//!
//! ```sh
//! cargo run --profile fast -p horizon-world --example far_node_bench -- <lat> <lon> [--cache DIR] [--reps N] [--levels 2,4,6]
//! ```
//!
//! `--cache DIR` holds `earth-tiles/<key>`, `earth-landcover/<key>` and `earth-climate/<key>` like a game folder's
//! `.horizon-cache`; what is missing is downloaded with `curl` once (warm-up, not timed). Per level it makes the
//! nodes around the origin (3 by 3, at the ground's height) and prints per-stage ms per node (the best of
//! `--reps`) and a checksum of each node's contents, so two builds can be compared for identical output.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use horizon_earth::tiff::ByteRange;
use horizon_earth::{
    ClimateFetcher, ClimateStore, ClimateVar, FetchError, LandCoverFetcher, LandCoverStore,
    TileFetcher, TileId, TileStore, WorldCoverFile,
};
use horizon_world::cubic::custom::testing::default_registry;
use horizon_world::cubic::earth::{EarthBlocks, EarthGenerator, EarthSettings, FarDecor, NotReady};

const TILE_URL: &str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
const LAND_URL: &str = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/";
const CLIMATE_URL: &str = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim/";

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

fn keyed(dir: &Path) -> PathBuf {
    std::fs::read_dir(dir)
        .ok()
        .and_then(|mut it| it.find_map(|e| e.ok().filter(|e| e.path().is_dir()).map(|e| e.path())))
        .unwrap_or_else(|| dir.join("probe"))
}

#[derive(Default, Clone, Copy)]
struct Stages {
    classify: f64,
    biomes_ready: f64,
    columns: f64,
    decorate: f64,
    biome_fn: f64,
    voxels: f64,
}

impl Stages {
    fn total(&self) -> f64 {
        self.classify + self.biomes_ready + self.columns + self.decorate + self.biome_fn + self.voxels
    }
    fn min(&mut self, o: &Stages) {
        self.classify = self.classify.min(o.classify);
        self.biomes_ready = self.biomes_ready.min(o.biomes_ready);
        self.columns = self.columns.min(o.columns);
        self.decorate = self.decorate.min(o.decorate);
        self.biome_fn = self.biome_fn.min(o.biome_fn);
        self.voxels = self.voxels.min(o.voxels);
    }
    fn add(&mut self, o: &Stages) {
        self.classify += o.classify;
        self.biomes_ready += o.biomes_ready;
        self.columns += o.columns;
        self.decorate += o.decorate;
        self.biome_fn += o.biome_fn;
        self.voxels += o.voxels;
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let mut cache = PathBuf::from("earth-probe-cache");
    let mut reps = 5usize;
    let mut levels = vec![2u32, 4, 6];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cache" => cache = PathBuf::from(it.next().expect("--cache DIR")),
            "--reps" => reps = it.next().expect("--reps N").parse().expect("--reps N"),
            "--levels" => {
                levels = it
                    .next()
                    .expect("--levels a,b")
                    .split(',')
                    .map(|v| v.parse().expect("level"))
                    .collect()
            }
            other => pos.push(other.parse::<f64>().expect("lat lon")),
        }
    }
    let [lat, lon] = pos[..] else {
        eprintln!("usage: far_node_bench <lat> <lon> [--cache DIR] [--reps N] [--levels 2,4,6]");
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
        4096,
    ));
    let land = settings
        .land_cover_source(
            Arc::new(LandCoverStore::new(keyed(&cache.join("earth-landcover")), 256)),
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
    let mut generator = EarthGenerator::new(&settings, tiles.clone(), blocks)
        .expect("generator")
        .with_land_cover(land.clone())
        .with_climate(climate.clone());
    if let Some(atlas) = horizon_earth::WaterAtlas::embedded() {
        generator = generator.with_water(atlas);
    }
    let g = Arc::new(generator);
    let decor = FarDecor {
        trees: true,
        snow: true,
    };

    // Loads what a read lacks (as the downloader does) until it answers.
    let ready = |read: &dyn Fn() -> Result<(), NotReady>| {
        for _ in 0..400 {
            match read() {
                Ok(()) => return,
                Err(missing) => {
                    for &id in &missing.tiles {
                        let _ = tiles.load(id);
                    }
                    for fetch in g.take_land_requests() {
                        let _ = land.satisfy(&fetch);
                    }
                    for fetch in g.take_climate_requests() {
                        let _ = climate.satisfy(&fetch);
                    }
                }
            }
        }
        panic!("a read never became ready");
    };

    println!("far node bench at {lat}, {lon}; best of {reps}; ms per node");
    for &level in &levels {
        let step = 1i32 << level;
        let side = 16 * step;
        let mut sum = Stages::default();
        let mut count = 0;
        let mut sum_hash = 0u64;
        for dz in -1..=1 {
            for dx in -1..=1 {
                let (nx, nz) = (dx, dz);
                // The node at the ground's height.
                let ground = g
                    .surface_y_blocking(nx * side + side / 2, nz * side + side / 2)
                    .unwrap_or(0);
                let ny = ground.div_euclid(side);
                // Warm up: tiles, land cover, climate for every read the node makes.
                ready(&|| {
                    g.classify_node_above(level, nx, ny, nz)?;
                    g.biomes_ready(nx * side + step / 2, nz * side + step / 2, step)?;
                    g.biomes_ready(nx * side, nz * side, step)?;
                    let mut cols = g.node_columns(level, nx, nz)?;
                    g.decorate_columns(&mut cols, decor)
                });
                let fill = g.classify_node_above(level, nx, ny, nz).unwrap();
                if g.fill_block(fill).is_some() {
                    continue;
                }
                let mut best = Stages {
                    classify: f64::MAX,
                    biomes_ready: f64::MAX,
                    columns: f64::MAX,
                    decorate: f64::MAX,
                    biome_fn: f64::MAX,
                    voxels: f64::MAX,
                };
                let mut hash = 0u64;
                for _ in 0..reps {
                    let mut s = Stages::default();
                    g.clear_far_memos();
                    let t = Instant::now();
                    let _ = g.classify_node_above(level, nx, ny, nz).unwrap();
                    s.classify = ms(t);
                    let t = Instant::now();
                    g.biomes_ready(nx * side + step / 2, nz * side + step / 2, step).unwrap();
                    g.biomes_ready(nx * side, nz * side, step).unwrap();
                    s.biomes_ready = ms(t);
                    let t = Instant::now();
                    let mut cols = g.node_columns(level, nx, nz).unwrap();
                    s.columns = ms(t);
                    let t = Instant::now();
                    g.decorate_columns(&mut cols, decor).unwrap();
                    s.decorate = ms(t);
                    // The client's biome closure: one biome read per column (the string table lookup is not here).
                    let t = Instant::now();
                    let biomes: Vec<u16> = (0..256)
                        .map(|i| {
                            let x = nx * side + (i & 15) * step + step / 2;
                            let z = nz * side + (i >> 4) * step + step / 2;
                            g.biome_at_quart_blocking(x >> 2, z >> 2) as u16
                        })
                        .collect();
                    s.biome_fn = ms(t);
                    let t = Instant::now();
                    let mut h = DefaultHasher::new();
                    for y in 0..16 {
                        for z in 0..16 {
                            for x in 0..16 {
                                let lo = (ny * side).saturating_add(y * step);
                                g.far_voxel(&cols, z << 4 | x, lo).hash(&mut h);
                            }
                        }
                    }
                    s.voxels = ms(t);
                    cols.surfaces.hash(&mut h);
                    for i in 0..256 {
                        format!("{:?}{:?}", cols.kind(i), cols.class(i)).hash(&mut h);
                    }
                    biomes.hash(&mut h);
                    hash = h.finish();
                    best.min(&s);
                }
                sum.add(&best);
                count += 1;
                sum_hash ^= hash;
                println!(
                    "  level {level} node ({nx},{ny},{nz}): {:6.2} ms  classify {:.2} ready {:.2} columns {:.2} decorate {:.2} biome_fn {:.2} voxels {:.2}  hash {hash:016x}",
                    best.total(),
                    best.classify,
                    best.biomes_ready,
                    best.columns,
                    best.decorate,
                    best.biome_fn,
                    best.voxels
                );
            }
        }
        if count > 0 {
            let n = f64::from(count);
            println!(
                "level {level}: {count} nodes, avg {:.2} ms  classify {:.2} ready {:.2} columns {:.2} decorate {:.2} biome_fn {:.2} voxels {:.2}  hash {sum_hash:016x}",
                sum.total() / n,
                sum.classify / n,
                sum.biomes_ready / n,
                sum.columns / n,
                sum.decorate / n,
                sum.biome_fn / n,
                sum.voxels / n
            );
        }
    }
}

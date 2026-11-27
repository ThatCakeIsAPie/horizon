//! Earth land classes: GPU (`gfx::earth_class::EarthClassGpu`) vs CPU (`LandCoverField::class_grid`).
//!
//! ```sh
//! cargo build --profile fast -p horizon-client-render --example earth_class_compare
//! cd /dev/shm/earthgpu && $CARGO_TARGET_DIR/fast/examples/earth_class_compare compare [N_PER_SPOT] [CACHE]
//! $CARGO_TARGET_DIR/fast/examples/earth_class_compare probe [ROWS]
//! ```
//!
//! `compare` (the default) builds the land-cover field the game builds (EarthSettings defaults, the store under
//! `CACHE/earth-landcover/<key>`, nothing is downloaded) at three spots, picks random far-view node windows
//! (levels 2..=8: stride `1 << level`, 38 by 38 samples, the node's 16 by 16 plus 11 on each side, metres per
//! sample `stride * metres_per_block`, as `columns_at` reads them), and compares per sample the classes of
//! `class_grid`, `ClassWindowPlan::evaluate` (CPU, from the plan) and the GPU kernel; then times them.
//!
//! `probe` reports whether the adapter has `SHADER_F64` and `SHADER_INT64` and how exactly WGSL f64 / u64
//! operations match Rust bit for bit. `EARTH_CLASS_WGSL_DIR=<dir with the .wgsl files>` reads the shaders
//! from disk instead of the binary.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use horizon_client_render::gfx::earth_class::{
    AdapterCaps, ClassifyTimings, Division, EarthClassGpu, headless_adapter, headless_f64_device,
    probe_exactness,
};
use horizon_earth::tiff::ByteRange;
use horizon_earth::{
    ClassWindowPlan, FetchError, LandCoverFetcher, LandCoverField, LandCoverStore, OUTSIDE_PLAN,
    WorldCoverFile,
};
use horizon_world::cubic::earth::EarthSettings;

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "compare".into());
    match mode.as_str() {
        "probe" => {
            let rows = args
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1usize << 20);
            probe(rows);
        }
        "compare" => {
            let n = args.next().and_then(|s| s.parse().ok()).unwrap_or(300usize);
            let cache = PathBuf::from(
                args.next()
                    .unwrap_or_else(|| "/dev/shm/earthgpu/cache".into()),
            );
            compare(n, &cache);
        }
        other => {
            eprintln!("unknown mode {other:?}; modes: compare [N_PER_SPOT] [CACHE] | probe [ROWS]");
            std::process::exit(2);
        }
    }
}

fn probe(rows: usize) {
    let Some(adapter) = headless_adapter() else {
        eprintln!("no GPU adapter");
        std::process::exit(1);
    };
    println!("{}\n", AdapterCaps::of(&adapter));
    drop(adapter);
    let Some((_adapter, device, queue)) = headless_f64_device() else {
        std::process::exit(1);
    };
    println!("probing with {rows} rows per test (division: {}x)...\n", 10);
    let report = probe_exactness(&device, &queue, rows);
    println!("{report}");
    println!(
        "\nall verdict tests exact: {}",
        if report.all_verdicts_exact() { "YES" } else { "NO" }
    );
}

// ------------------------------------------------------------------ real data

struct Spot {
    name: &'static str,
    lat: f64,
    lon: f64,
}

const SPOTS: [Spot; 3] = [
    Spot { name: "grand-canyon", lat: 36.0544, lon: -112.1401 },
    Spot { name: "monaco", lat: 43.7333, lon: 7.4200 },
    Spot { name: "new-york", lat: 40.71, lon: -74.0 },
];

/// Never fetches: the cache must hold what is read.
struct Offline;

impl LandCoverFetcher for Offline {
    fn fetch(&self, _: WorldCoverFile, _: ByteRange) -> Result<Option<Vec<u8>>, FetchError> {
        Err(FetchError::Temporary("offline".into()))
    }
}

/// The first folder under `dir` (the cache key folder), as earth_far_profile finds it.
fn keyed(dir: &Path) -> PathBuf {
    std::fs::read_dir(dir)
        .ok()
        .and_then(|mut it| it.find_map(|e| e.ok().filter(|e| e.path().is_dir()).map(|e| e.path())))
        .unwrap_or_else(|| dir.join("probe"))
}

/// The land-cover field a world at the spot reads (the settings' projection over the cached store).
fn land_field(spot: &Spot, cache: &Path) -> LandCoverField {
    let json = format!(
        r#"{{"originLatitude": {}, "originLongitude": {}, "tileUrl": "https://example.invalid/{{z}}/{{x}}/{{y}}.png",
            "allowDownloads": false, "landCover": true}}"#,
        spot.lat, spot.lon
    );
    let settings = EarthSettings::from_json(&json).expect("settings");
    let store = Arc::new(LandCoverStore::new(keyed(&cache.join("earth-landcover")), 256));
    settings
        .land_cover_source(store, Arc::new(Offline))
        .expect("land cover source")
        .field()
        .clone()
}

#[derive(Clone, Copy, Debug)]
struct Window {
    level: u32,
    x0: i32,
    z0: i32,
    stride: i32,
    metres: f64,
}

const SIDE: usize = 38;
const PAD: i32 = 11;

impl Window {
    /// The node at `(nx, nz)` (block origin) of `level`, as `columns_at` reads its class window.
    fn of_node(level: u32, nx: i32, nz: i32, metres_per_block: f64) -> Self {
        let stride = 1 << level;
        let (sx, sz) = (nx + stride / 2, nz + stride / 2);
        Self {
            level,
            x0: sx - PAD * stride,
            z0: sz - PAD * stride,
            stride,
            metres: f64::from(stride) * metres_per_block,
        }
    }

    fn cpu(&self, field: &LandCoverField) -> Option<Vec<u8>> {
        field
            .class_grid(self.x0, self.z0, SIDE, SIDE, self.stride, self.metres)
            .ok()
            .map(|v| v.iter().map(|c| c.code()).collect())
    }

    fn cpu_reference(&self, field: &LandCoverField) -> Option<Vec<u8>> {
        field
            .class_grid_reference(self.x0, self.z0, SIDE, SIDE, self.stride, self.metres)
            .ok()
            .map(|v| v.iter().map(|c| c.code()).collect())
    }

    fn plan(&self, field: &LandCoverField) -> Option<ClassWindowPlan> {
        field
            .class_window_plan(self.x0, self.z0, SIDE, SIDE, self.stride, self.metres)
            .ok()
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

/// Random node windows around the origin, levels 2..=8 in turn: node origins on the stride grid inside the
/// 4 by 4 nodes around the origin of each level (the area the far view of a fresh world loads first, which
/// earth_far_profile warmed the cache with).
fn random_windows(n: usize, seed: u64, metres_per_block: f64) -> Vec<Window> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|k| {
            let level = 2 + (k % 7) as u32;
            let stride = 1 << level;
            // In strides: a node spans 16; the 4 by 4 nodes span -32..32.
            let nx = rng.range(-32, 17) * stride;
            let nz = rng.range(-32, 17) * stride;
            Window::of_node(level, nx, nz, metres_per_block)
        })
        .collect()
}

// ------------------------------------------------------------------ compare

#[derive(Default, Clone, Copy)]
struct Count {
    windows: usize,
    samples: usize,
    bad_samples: usize,
    bad_windows: usize,
    outside: usize,
}

impl Count {
    fn add(&mut self, cpu: &[u8], other: &[u8]) {
        self.windows += 1;
        self.samples += cpu.len();
        let bad = cpu.iter().zip(other).filter(|(a, b)| a != b).count();
        self.bad_samples += bad;
        self.bad_windows += usize::from(bad > 0);
        self.outside += other.iter().filter(|&&c| c == OUTSIDE_PLAN).count();
    }

    fn merge(&mut self, c: &Count) {
        self.windows += c.windows;
        self.samples += c.samples;
        self.bad_samples += c.bad_samples;
        self.bad_windows += c.bad_windows;
        self.outside += c.outside;
    }
}

fn ms(d: Duration, n: usize) -> f64 {
    d.as_secs_f64() * 1000.0 / n.max(1) as f64
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

struct Case {
    spot: usize,
    window: Window,
    cpu: Vec<u8>,
    plan: ClassWindowPlan,
}

fn print_counts(title: &str, counts: &BTreeMap<(usize, u32), Count>) -> Count {
    println!("\n{title}, per spot and level:");
    println!(
        "{:<14} {:>5} {:>8} {:>9} {:>12} {:>12} {:>8}",
        "spot", "level", "windows", "samples", "bad samples", "bad windows", "outside"
    );
    let mut total = Count::default();
    for ((s, level), c) in counts {
        println!(
            "{:<14} {:>5} {:>8} {:>9} {:>12} {:>12} {:>8}",
            SPOTS[*s].name, level, c.windows, c.samples, c.bad_samples, c.bad_windows, c.outside
        );
        total.merge(c);
    }
    println!(
        "TOTAL: {} of {} samples differ, {} of {} windows, {} samples outside the plan",
        total.bad_samples, total.samples, total.bad_windows, total.windows, total.outside
    );
    total
}

fn compare(n_per_spot: usize, cache: &Path) {
    let Some(adapter) = headless_adapter() else {
        eprintln!("no GPU adapter");
        std::process::exit(1);
    };
    println!("{}\n", AdapterCaps::of(&adapter));
    drop(adapter);
    let Some((_adapter, device, queue)) = headless_f64_device() else {
        std::process::exit(1);
    };
    let Some(mut gpu) = EarthClassGpu::new(&device, &queue) else {
        eprintln!("the land-class kernel did not build");
        std::process::exit(1);
    };
    println!(
        "tile pool: {} slots of {} pixels",
        gpu.pool_slots(),
        EarthClassGpu::SLOT_PIXELS
    );

    let mut cases: Vec<Case> = Vec::new();
    let (mut cpu_time, mut ref_time, mut plan_time, mut eval_time) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let (mut timed, mut plan_err) = (0usize, 0usize);
    let mut eval_counts: BTreeMap<(usize, u32), Count> = BTreeMap::new();
    let mut ref_bad = 0usize;
    let mut first_eval_bad: Vec<String> = Vec::new();
    for (s, spot) in SPOTS.iter().enumerate() {
        let field = land_field(spot, cache);
        let mpb = field.projection().metres_per_block();
        let windows = random_windows(n_per_spot, 0xC1A5_5000 + s as u64, mpb);
        let (mut sk, mut pe) = (0, 0);
        for w in windows {
            // Warm (tiles decoded into the store's memory), then timed.
            if w.cpu(&field).is_none() {
                sk += 1;
                continue;
            }
            let t = Instant::now();
            let cpu = w.cpu(&field).expect("held");
            cpu_time += t.elapsed();
            let t = Instant::now();
            let reference = w.cpu_reference(&field).expect("held");
            ref_time += t.elapsed();
            ref_bad += usize::from(reference != cpu);
            timed += 1;
            let t = Instant::now();
            let plan = w.plan(&field);
            plan_time += t.elapsed();
            let Some(plan) = plan else {
                pe += 1;
                continue;
            };
            let t = Instant::now();
            let eval = plan.evaluate();
            eval_time += t.elapsed();
            eval_counts.entry((s, w.level)).or_default().add(&cpu, &eval);
            if eval != cpu && first_eval_bad.len() < 5 {
                first_eval_bad.push(format!("{} {:?}", spot.name, w));
            }
            cases.push(Case {
                spot: s,
                window: w,
                cpu,
                plan,
            });
        }
        println!(
            "{}: {} windows asked, {} skipped (class_grid Err: not in the cache), {} plan Err (plan more \
             conservative than class_grid), metres/block {}",
            spot.name, n_per_spot, sk, pe, mpb
        );
        plan_err += pe;
    }
    println!(
        "\nCPU (one thread, warm store), {} windows: class_grid {:.3} ms/window, class_grid_reference {:.3} \
         ms/window ({} windows differ from class_grid), class_window_plan {:.3} ms/window, plan.evaluate {:.3} \
         ms/window",
        timed,
        ms(cpu_time, timed),
        ms(ref_time, timed),
        ref_bad,
        ms(plan_time, timed),
        ms(eval_time, cases.len()),
    );
    print_counts("plan.evaluate (CPU) vs class_grid", &eval_counts);
    for l in &first_eval_bad {
        println!("  evaluate differs: {l}");
    }

    // GPU: cold pool, everything in batches of 256.
    let plans: Vec<&ClassWindowPlan> = cases.iter().map(|c| &c.plan).collect();
    let mut gpu_codes: Vec<Option<Vec<u8>>> = Vec::with_capacity(plans.len());
    gpu.clear_pool();
    let mut wall = Duration::ZERO;
    let mut cold = ClassifyTimings::default();
    for chunk in plans.chunks(256) {
        let t = Instant::now();
        let (codes, timings) = gpu.classify_timed(chunk);
        wall += t.elapsed();
        cold += timings;
        gpu_codes.extend(codes);
    }
    let mut counts: BTreeMap<(usize, u32), Count> = BTreeMap::new();
    let mut declined = 0;
    let mut shown = 0;
    for (case, got) in cases.iter().zip(&gpu_codes) {
        let Some(got) = got else {
            declined += 1;
            continue;
        };
        counts
            .entry((case.spot, case.window.level))
            .or_default()
            .add(&case.cpu, got);
        for (k, (a, b)) in case.cpu.iter().zip(got).enumerate() {
            if a != b && shown < 10 {
                shown += 1;
                println!(
                    "  MISMATCH {} {:?} sample i={} j={}: cpu {} gpu {} (evaluate {})",
                    SPOTS[case.spot].name,
                    case.window,
                    k % SIDE,
                    k / SIDE,
                    a,
                    b,
                    case.plan.evaluate()[k]
                );
            }
        }
    }
    let total = print_counts("GPU (native division) vs class_grid", &counts);
    println!("windows the GPU declined (CPU fallback): {declined}");
    println!(
        "\ncold pool, batches of 256: {:.4} ms/window wall over {} windows; {} tiles uploaded ({:.1} MiB) in \
         {:.1} ms (pack + queue write; {:.2} ms/tile), job packing {:.1} ms, GPU write + dispatch + readback \
         {:.1} ms, {} dispatches",
        ms(wall, plans.len()),
        plans.len(),
        cold.tiles_uploaded,
        cold.upload_bytes as f64 / f64::from(1 << 20),
        cold.upload.as_secs_f64() * 1000.0,
        ms(cold.upload, cold.tiles_uploaded),
        cold.pack.as_secs_f64() * 1000.0,
        cold.gpu.as_secs_f64() * 1000.0,
        cold.dispatches
    );

    println!("\nwarm pool, GPU wall per window (median over batches of one spot; native division):");
    for batch in [1usize, 16, 64, 256] {
        let (per, uploads) = time_batches(&mut gpu, &cases, batch);
        println!(
            "  batch {batch:>3}: {per:.4} ms/window = {:.3} ms/batch ({uploads} tile uploads in the timed runs)",
            per * batch as f64
        );
    }

    // The fma division.
    if gpu.set_division(Division::Fma) {
        let mut fma_counts: BTreeMap<(usize, u32), Count> = BTreeMap::new();
        for (chunk, part) in plans.chunks(256).zip(cases.chunks(256)) {
            for (got, case) in gpu.classify(chunk).iter().zip(part) {
                if let Some(got) = got {
                    fma_counts
                        .entry((case.spot, case.window.level))
                        .or_default()
                        .add(&case.cpu, got);
                }
            }
        }
        let fma_total = fma_counts.values().fold(Count::default(), |mut a, c| {
            a.merge(c);
            a
        });
        let (per, _) = time_batches(&mut gpu, &cases, 256);
        println!(
            "\nfma division (div_cr): {} of {} samples differ ({} windows); batch 256 warm {per:.4} ms/window",
            fma_total.bad_samples, fma_total.samples, fma_total.bad_windows
        );
        gpu.set_division(Division::Native);
    } else {
        println!("\nthe fma-division kernel did not build");
    }
    println!(
        "\nsummary: {} windows ({} samples) compared: GPU {} bad samples, {} bad windows; plan Err {}, \
         declined {}",
        total.windows, total.samples, total.bad_samples, total.bad_windows, plan_err, declined
    );
}

/// Median wall ms per window of `classify` over batches of `batch` windows of one spot (after a warm-up pass
/// over the same batches); and the tile uploads seen during the timed runs.
fn time_batches(gpu: &mut EarthClassGpu, cases: &[Case], batch: usize) -> (f64, usize) {
    let mut per = Vec::new();
    let mut uploads = 0;
    for s in 0..SPOTS.len() {
        let plans: Vec<&ClassWindowPlan> =
            cases.iter().filter(|c| c.spot == s).map(|c| &c.plan).collect();
        // At most 64 batches per spot (batch 1 would take long over all).
        let chunks: Vec<&[&ClassWindowPlan]> = plans.chunks(batch).take(64).collect();
        for chunk in &chunks {
            let _ = gpu.classify(chunk);
        }
        for chunk in &chunks {
            let t = Instant::now();
            let (_, timings) = gpu.classify_timed(chunk);
            per.push(ms(t.elapsed(), chunk.len()));
            uploads += timings.tiles_uploaded;
        }
    }
    (median(&mut per), uploads)
}

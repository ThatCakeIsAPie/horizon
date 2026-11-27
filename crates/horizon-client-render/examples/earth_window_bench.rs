//! Times and checks the GPU window DEM of the Earth far view (`gfx::earth_window::EarthWindowGpu`) against the CPU.
//!
//! ```sh
//! cd /dev/shm/earthgpu && earth_window_bench <cache dir> [bench|compare|all] [--nodes N] [--bench-nodes N] [--reps N] [--no-f64] [--igpu] [--linger-us N]
//! ```
//!
//! The generator is built from the warm tile cache like `horizon-world`'s `earth_far_profile` (three spots: the
//! Grand Canyon, Monaco, New York). `bench` times, per node, the padded window's elevation by the old per-sample
//! CPU code, the separable CPU plan and the GPU, then the whole `columns_at` with the hook off and on, with one
//! thread and with six at once; each as the worker thread's CPU time, the time it was blocked waiting for the
//! GPU, and the wall time. `compare` checks the GPU's heights against the plan's CPU heights over random nodes
//! (levels 2..8) and the final columns (surfaces, kinds, classes) of `columns_at` with the hook on against off.

use std::cell::Cell;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use horizon_client_render::gfx::earth_window::{EarthWindowGpu, EarthWindowOptions};
use horizon_earth::tiff::ByteRange;
use horizon_earth::{
    ClimateFetcher, ClimateStore, ClimateVar, FetchError, LandCoverFetcher, LandCoverStore,
    TileFetcher, TileId, TileStore, WorldCoverFile,
};
use horizon_world::cubic::custom::testing::default_registry;
use horizon_world::cubic::earth::water::WINDOW_PAD;
use horizon_world::cubic::earth::{
    EarthBlocks, EarthGenerator, EarthSettings, WindowGpu, WindowJob, WindowResult, WindowTicket,
};

const TILE_URL: &str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
const LAND_URL: &str = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/";
const CLIMATE_URL: &str = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim/";

const SPOTS: [(&str, f64, f64); 3] = [
    ("GrandCanyon", 36.0544, -112.1401),
    ("Monaco", 43.7333, 7.4200),
    ("NewYork", 40.71, -74.0),
];

// ------------------------------------------------------------------ the generator (as earth_far_profile)

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

fn generator(lat: f64, lon: f64, cache: &Path) -> EarthGenerator {
    let json = format!(
        r#"{{"originLatitude": {lat}, "originLongitude": {lon}, "tileUrl": "{TILE_URL}", "allowDownloads": true,
            "landCover": true, "vegetation": true, "climate": true, "snow": true}}"#
    );
    let settings = EarthSettings::from_json(&json).expect("settings");
    // A memory big enough to hold every tile the nodes touch: the plans need them all cached at once.
    let tiles = Arc::new(TileStore::new(
        keyed(&cache.join("earth-tiles")),
        Arc::new(Curl),
        4096,
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
    let mut g = EarthGenerator::new(&settings, tiles, blocks)
        .expect("generator")
        .with_land_cover(land)
        .with_climate(climate);
    if let Some(atlas) = horizon_earth::WaterAtlas::embedded() {
        g = g.with_water(atlas);
    }
    g
}

// ------------------------------------------------------------------ nodes

#[derive(Clone, Copy, Debug)]
struct Node {
    spot: usize,
    level: u32,
    i: i32,
    j: i32,
}

impl Node {
    fn step(&self) -> i32 {
        1 << self.level
    }
    /// The node's first block.
    fn origin(&self) -> (i32, i32) {
        let size = 16i32 << self.level;
        (self.i * size, self.j * size)
    }
    /// The padded window: (x0, z0, side, stride).
    fn window(&self) -> (i32, i32, usize, i32) {
        let step = self.step();
        let (x0, z0) = self.origin();
        let pad = WINDOW_PAD as i32 * step;
        (
            x0 + step / 2 - pad,
            z0 + step / 2 - pad,
            16 + 2 * WINDOW_PAD,
            step,
        )
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

/// `count` distinct random nodes of every level 2..=8 (about) around spot `spot`; the profiler's own keys first.
fn pick_nodes(spot: usize, count: usize, seed: u64) -> Vec<Node> {
    let mut rng = Rng(seed ^ (spot as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for level in 2..=8u32 {
        for j in -2..2 {
            for i in -2..2 {
                seen.insert((level, i, j));
                out.push(Node { spot, level, i, j });
            }
        }
    }
    while out.len() < count {
        let level = 2 + (rng.next() % 7) as u32;
        let (i, j) = (rng.range(-8, 8), rng.range(-8, 8));
        if seen.insert((level, i, j)) {
            out.push(Node { spot, level, i, j });
        }
    }
    out
}

/// Loads each node's tiles and land cover like the profiler; the nodes that could not be loaded are dropped.
fn load(g: &EarthGenerator, nodes: Vec<Node>) -> Vec<Node> {
    let t = Instant::now();
    let total = nodes.len();
    let kept: Vec<Node> = nodes
        .into_iter()
        .filter(|n| {
            let step = n.step();
            let (x, z) = n.origin();
            g.columns_blocking([x + step / 2, z + step / 2], [16, 16], step, step)
                .is_some()
        })
        .collect();
    // Keep only nodes that now read without waiting.
    let kept: Vec<Node> = kept
        .into_iter()
        .filter(|n| {
            let (x, z) = n.origin();
            g.columns_at(x, z, n.step(), n.step()).is_ok()
        })
        .collect();
    eprintln!(
        "  loaded {} of {total} nodes in {:.1?}",
        kept.len(),
        t.elapsed()
    );
    kept
}

// ------------------------------------------------------------------ clocks and the timing hook

fn thread_cpu_ns() -> u64 {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}

fn process_cpu_ns() -> u64 {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}

thread_local! {
    /// Nanoseconds this thread spent in `wait()` of a GPU ticket.
    static BLOCKED_NS: Cell<u64> = const { Cell::new(0) };
}

/// The GPU hook, counting the time each thread waits for its tickets.
struct Timed(Arc<EarthWindowGpu>);

struct TimedTicket(Box<dyn WindowTicket>);

impl WindowTicket for TimedTicket {
    fn wait(self: Box<Self>) -> Option<WindowResult> {
        let t = Instant::now();
        let r = self.0.wait();
        BLOCKED_NS.with(|b| b.set(b.get() + t.elapsed().as_nanos() as u64));
        r
    }
}

impl WindowGpu for Timed {
    fn submit(&self, job: WindowJob) -> Option<Box<dyn WindowTicket>> {
        let t = self.0.submit(job)?;
        Some(Box::new(TimedTicket(t)))
    }
}

// ------------------------------------------------------------------ bench

#[derive(Clone, Copy, Debug, Default)]
struct Acc {
    n: u64,
    cpu_ns: u64,
    blocked_ns: u64,
    wall_ns: u64,
}

impl Acc {
    fn add(&mut self, o: Acc) {
        self.n += o.n;
        self.cpu_ns += o.cpu_ns;
        self.blocked_ns += o.blocked_ns;
        self.wall_ns += o.wall_ns;
    }
}

struct Run {
    per_spot: Vec<Acc>,
    /// Elapsed wall time of the whole pass, and the whole process's CPU time in it.
    elapsed: Duration,
    process_cpu: Duration,
    nodes: usize,
}

impl Run {
    fn total(&self) -> Acc {
        let mut a = Acc::default();
        for s in &self.per_spot {
            a.add(*s);
        }
        a
    }
}

/// One pass of `work` over `items` on `threads` threads (item `k` on thread `k % threads`).
fn pass(items: &[Node], spots: usize, threads: usize, work: &(dyn Fn(&Node) + Sync)) -> Run {
    let barrier = std::sync::Barrier::new(threads + 1);
    let p0 = process_cpu_ns();
    let t0 = Instant::now();
    let per_thread: Vec<Vec<Acc>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let barrier = &barrier;
                s.spawn(move || {
                    let mut acc = vec![Acc::default(); spots];
                    barrier.wait();
                    for node in items.iter().skip(t).step_by(threads) {
                        let b0 = BLOCKED_NS.with(Cell::get);
                        let c0 = thread_cpu_ns();
                        let w0 = Instant::now();
                        work(node);
                        let w = w0.elapsed().as_nanos() as u64;
                        let c = thread_cpu_ns() - c0;
                        let b = BLOCKED_NS.with(Cell::get) - b0;
                        acc[node.spot].add(Acc {
                            n: 1,
                            cpu_ns: c,
                            blocked_ns: b,
                            wall_ns: w,
                        });
                    }
                    acc
                })
            })
            .collect();
        barrier.wait();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let elapsed = t0.elapsed();
    let process_cpu = Duration::from_nanos(process_cpu_ns() - p0);
    let mut per_spot = vec![Acc::default(); spots];
    for acc in per_thread {
        for (s, a) in acc.into_iter().enumerate() {
            per_spot[s].add(a);
        }
    }
    Run {
        per_spot,
        elapsed,
        process_cpu,
        nodes: items.len(),
    }
}

/// A warm-up pass, then `reps` timed passes; the one with the median elapsed time.
fn measure(
    items: &[Node],
    spots: usize,
    threads: usize,
    reps: usize,
    work: &(dyn Fn(&Node) + Sync),
) -> Run {
    pass(items, spots, threads, work);
    let mut runs: Vec<Run> = (0..reps)
        .map(|_| pass(items, spots, threads, work))
        .collect();
    runs.sort_by_key(|r| r.elapsed);
    runs.swap_remove(runs.len() / 2)
}

fn ms(ns: u64, n: u64) -> f64 {
    ns as f64 / 1e6 / n.max(1) as f64
}

fn print_row(label: &str, threads: usize, r: &Run) {
    let t = r.total();
    println!(
        "{label:<34} {threads}T  thread-cpu {:7.3}  blocked {:7.3}  wall(latency) {:7.3}  elapsed/node {:7.3}  process-cpu/node {:7.3}  (ms/node, {} nodes)",
        ms(t.cpu_ns, t.n),
        ms(t.blocked_ns, t.n),
        ms(t.wall_ns, t.n),
        r.elapsed.as_secs_f64() * 1e3 / r.nodes.max(1) as f64,
        r.process_cpu.as_secs_f64() * 1e3 / r.nodes.max(1) as f64,
        t.n,
    );
}

fn bench(gens: &[Arc<EarthGenerator>], hook: &Arc<Timed>, nodes: &[Node], reps: usize) {
    let spots = gens.len();
    // Interleave the spots so every thread sees all of them.
    let mut items: Vec<Node> = Vec::new();
    let longest = (0..spots)
        .map(|s| nodes.iter().filter(|n| n.spot == s).count())
        .max()
        .unwrap_or(0);
    for k in 0..longest {
        for s in 0..spots {
            if let Some(n) = nodes.iter().filter(|n| n.spot == s).nth(k) {
                items.push(*n);
            }
        }
    }
    let old = |n: &Node| {
        let (x0, z0, side, stride) = n.window();
        let g = &gens[n.spot];
        let zoom = g.zoom_for_step(n.step());
        std::hint::black_box(
            g.field()
                .heights_strided_reference(x0, z0, side, side, stride, zoom),
        );
    };
    let cpu = |n: &Node| {
        let (x0, z0, side, stride) = n.window();
        let g = &gens[n.spot];
        let zoom = g.zoom_for_step(n.step());
        let plan = g
            .field()
            .window_plan(x0, z0, side, side, stride, zoom)
            .unwrap();
        std::hint::black_box(plan.heights_m());
    };
    let gpu = |n: &Node| {
        let (x0, z0, side, stride) = n.window();
        let g = &gens[n.spot];
        let zoom = g.zoom_for_step(n.step());
        let plan = Arc::new(
            g.field()
                .window_plan(x0, z0, side, side, stride, zoom)
                .unwrap(),
        );
        let t = hook.submit(WindowJob { plan }).expect("gpu job");
        std::hint::black_box(t.wait().expect("gpu result"));
    };
    let cols = |n: &Node| {
        let (x, z) = n.origin();
        std::hint::black_box(gens[n.spot].columns_at(x, z, n.step(), n.step()).unwrap());
    };
    let before = hook.0.stats();
    for threads in [1usize, 6] {
        println!("--- window DEM alone, {threads} thread(s)");
        let r = measure(&items, spots, threads, reps, &old);
        print_row("old CPU (per-sample reference)", threads, &r);
        let r = measure(&items, spots, threads, reps, &cpu);
        print_row("separable CPU (plan + heights_m)", threads, &r);
        let r = measure(&items, spots, threads, reps, &gpu);
        print_row("GPU (plan + submit + wait)", threads, &r);
        println!("--- whole columns_at, {threads} thread(s)");
        for g in gens {
            g.set_window_gpu_enabled(false);
        }
        let r = measure(&items, spots, threads, reps, &cols);
        print_row("columns_at hook off", threads, &r);
        for g in gens {
            g.set_window_gpu_enabled(true);
        }
        let r = measure(&items, spots, threads, reps, &cols);
        print_row("columns_at hook on", threads, &r);
        for g in gens {
            g.set_window_gpu_enabled(false);
        }
    }
    let after = hook.0.stats();
    let batches = after.batches - before.batches;
    let jobs = after.jobs - before.jobs;
    println!(
        "gpu batcher: {batches} batches, {jobs} jobs ({:.2} jobs/batch), {:.3} ms/batch inside the batcher, {} uploads, {} evictions",
        jobs as f64 / batches.max(1) as f64,
        (after.gpu_ns - before.gpu_ns) as f64 / 1e6 / batches.max(1) as f64,
        after.uploads,
        after.evictions
    );
}

// ------------------------------------------------------------------ compare

#[derive(Default)]
struct Cmp {
    windows: u64,
    samples: u64,
    bit_diff: u64,
    max_abs: f64,
    block_y_diff: u64,
    gpu_none: u64,
    nodes: u64,
    cols_nodes_diff: u64,
    columns: u64,
    cols_diff: u64,
    surfaces_diff: u64,
    kinds_diff: u64,
    classes_diff: u64,
    skipped: u64,
}

impl Cmp {
    fn add(&mut self, o: &Cmp) {
        self.windows += o.windows;
        self.samples += o.samples;
        self.bit_diff += o.bit_diff;
        self.max_abs = self.max_abs.max(o.max_abs);
        self.block_y_diff += o.block_y_diff;
        self.gpu_none += o.gpu_none;
        self.nodes += o.nodes;
        self.cols_nodes_diff += o.cols_nodes_diff;
        self.columns += o.columns;
        self.cols_diff += o.cols_diff;
        self.surfaces_diff += o.surfaces_diff;
        self.kinds_diff += o.kinds_diff;
        self.classes_diff += o.classes_diff;
        self.skipped += o.skipped;
    }
}

fn compare(gens: &[Arc<EarthGenerator>], hook: &Arc<Timed>, nodes: &[Node]) {
    // Six threads share the nodes (so the batcher sees concurrent jobs); the on/off columns comparison of a node
    // runs under a per-generator lock because the hook switch is the generator's.
    let threads = 6;
    let total = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut c = Cmp::default();
                    for n in nodes
                        .iter()
                        .enumerate()
                        .filter(|(k, n)| (k + n.spot) % threads == t)
                    {
                        compare_node(gens, hook, n.1, &mut c);
                    }
                    c
                })
            })
            .collect();
        let mut total = Cmp::default();
        for h in handles {
            total.add(&h.join().unwrap());
        }
        total
    });
    let c = total;
    println!(
        "window heights: {} windows, {} samples; GPU None {}, skipped {}; bitwise-different heights {} ({:.6} %), max abs diff {:.9} m; block_y mismatches {} ({:.6} %)",
        c.windows,
        c.samples,
        c.gpu_none,
        c.skipped,
        c.bit_diff,
        100.0 * c.bit_diff as f64 / c.samples.max(1) as f64,
        c.max_abs,
        c.block_y_diff,
        100.0 * c.block_y_diff as f64 / c.samples.max(1) as f64,
    );
    println!(
        "columns_at hook on vs off: {} nodes, {} columns; nodes differing {}; columns differing {} (surfaces {}, kinds {}, classes {})",
        c.nodes,
        c.columns,
        c.cols_nodes_diff,
        c.cols_diff,
        c.surfaces_diff,
        c.kinds_diff,
        c.classes_diff
    );
}

/// Compares one node: the window heights (concurrent), then the whole columns with the hook on against off.
fn compare_node(gens: &[Arc<EarthGenerator>], hook: &Timed, n: &Node, c: &mut Cmp) {
    static LOCKS: [std::sync::Mutex<()>; 3] = [
        std::sync::Mutex::new(()),
        std::sync::Mutex::new(()),
        std::sync::Mutex::new(()),
    ];
    // Window heights run concurrently (they exercise the batcher); the columns comparison runs under the lock.
    let g = &gens[n.spot];
    let (x0, z0, side, stride) = n.window();
    let zoom = g.zoom_for_step(n.step());
    let Ok(plan) = g.field().window_plan(x0, z0, side, side, stride, zoom) else {
        c.skipped += 1;
        return;
    };
    let cpu = plan.heights_m();
    let Some(gpu) = hook
        .submit(WindowJob {
            plan: Arc::new(plan),
        })
        .and_then(|t| t.wait())
    else {
        c.gpu_none += 1;
        return;
    };
    c.windows += 1;
    assert_eq!(gpu.heights_m.len(), cpu.len());
    for (a, b) in gpu.heights_m.iter().zip(&cpu) {
        c.samples += 1;
        if a.to_bits() != b.to_bits() {
            c.bit_diff += 1;
            c.max_abs = c.max_abs.max((f64::from(*a) - f64::from(*b)).abs());
        }
        if g.field().block_y_of(*a) != g.field().block_y_of(*b) {
            c.block_y_diff += 1;
        }
    }
    let _guard = LOCKS[n.spot % 3].lock().unwrap();
    let (x, z) = n.origin();
    g.set_window_gpu_enabled(false);
    let off = g.columns_at(x, z, n.step(), n.step());
    g.set_window_gpu_enabled(true);
    let on = g.columns_at(x, z, n.step(), n.step());
    g.set_window_gpu_enabled(false);
    let (Ok(off), Ok(on)) = (off, on) else {
        c.skipped += 1;
        return;
    };
    c.nodes += 1;
    let mut node_differs = false;
    for i in 0..off.surfaces.len() {
        c.columns += 1;
        let s = off.surfaces[i] != on.surfaces[i];
        let k = off.kind(i) != on.kind(i);
        let cl = off.class(i) != on.class(i);
        c.surfaces_diff += u64::from(s);
        c.kinds_diff += u64::from(k);
        c.classes_diff += u64::from(cl);
        if s || k || cl {
            c.cols_diff += 1;
            node_differs = true;
        }
    }
    c.cols_nodes_diff += u64::from(node_differs);
}

// ------------------------------------------------------------------ main

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cache = PathBuf::from(args.first().cloned().unwrap_or_else(|| "cache".into()));
    let mode = args.get(1).map_or("all", String::as_str).to_string();
    let flag_value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|p| args.get(p + 1))
            .and_then(|v| v.parse::<usize>().ok())
    };
    let per_spot = flag_value("--nodes").unwrap_or(800);
    let bench_per_spot = flag_value("--bench-nodes").unwrap_or(250);
    let reps = flag_value("--reps").unwrap_or(3);
    let allow_f64 = !args.iter().any(|a| a == "--no-f64");
    let linger = Duration::from_micros(flag_value("--linger-us").unwrap_or(0) as u64);

    // A headless device, with SHADER_F64 when the adapter has it.
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle().with_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        // As the game: the discrete GPU when there is one (`--igpu` for the default pick).
        power_preference: if args.iter().any(|a| a == "--igpu") {
            wgpu::PowerPreference::None
        } else {
            wgpu::PowerPreference::HighPerformance
        },
        ..Default::default()
    }))
    .expect("an adapter");
    let info = adapter.get_info();
    let features = adapter.features() & wgpu::Features::SHADER_F64;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("earth_window_bench"),
        required_features: features,
        ..Default::default()
    }))
    .expect("a device");
    println!(
        "adapter: {} ({:?}, {:?}); adapter SHADER_F64: {}; device SHADER_F64: {}; max storage binding {} MiB",
        info.name,
        info.device_type,
        info.backend,
        adapter.features().contains(wgpu::Features::SHADER_F64),
        device.features().contains(wgpu::Features::SHADER_F64),
        device.limits().max_storage_buffer_binding_size >> 20,
    );
    let gpu = EarthWindowGpu::with_options(device, queue, EarthWindowOptions { allow_f64, linger })
        .expect("EarthWindowGpu (no compute support?)");
    println!(
        "EarthWindowGpu: uses_f64 = {}, pool {} tile slots, linger {:?}",
        gpu.uses_f64(),
        gpu.pool_slots(),
        linger
    );
    let hook = Arc::new(Timed(gpu));

    let mut gens: Vec<Arc<EarthGenerator>> = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    let n_nodes = if mode == "bench" {
        bench_per_spot
    } else {
        per_spot.max(bench_per_spot)
    };
    for (s, (name, lat, lon)) in SPOTS.iter().enumerate() {
        eprintln!("{name}: loading");
        let g = generator(*lat, *lon, &cache);
        let picked = load(&g, pick_nodes(s, n_nodes, 12345));
        g.set_window_gpu(Some(hook.clone() as Arc<dyn WindowGpu>));
        g.set_window_gpu_enabled(false);
        gens.push(Arc::new(g));
        nodes.extend(picked);
    }
    println!("{} nodes over {} spots", nodes.len(), SPOTS.len());

    if mode == "bench" || mode == "all" {
        let bench_nodes: Vec<Node> = (0..SPOTS.len())
            .flat_map(|s| {
                nodes
                    .iter()
                    .filter(move |n| n.spot == s)
                    .take(bench_per_spot)
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();
        println!(
            "== bench ({} nodes, warm-up + {reps} passes, median pass)",
            bench_nodes.len()
        );
        bench(&gens, &hook, &bench_nodes, reps);
    }
    if mode == "compare" || mode == "all" {
        println!("== compare ({} nodes)", nodes.len());
        compare(&gens, &hook, &nodes);
        let s = hook.0.stats();
        println!(
            "gpu: {} batches, {} jobs, {} uploads, {} evictions; alive {}",
            s.batches,
            s.jobs,
            s.uploads,
            s.evictions,
            hook.0.is_alive()
        );
    }
}

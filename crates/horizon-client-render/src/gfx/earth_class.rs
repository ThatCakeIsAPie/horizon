//! Probe for bit-exact f64 / u64 on the GPU (the Earth land-class kernel).
//!
//! The land class of a lon/lat window is computed on the CPU in IEEE f64 and
//! u64; a GPU port is only worth having if it matches bit for bit. This module
//! creates a headless device with `SHADER_F64` and `SHADER_INT64`, runs the
//! small kernels of `earth_class_probe.wgsl` over millions of random operands
//! and compares every result bitwise with Rust (`probe_exactness`). The exact
//! helper functions the probe validates live in `earth_class.wgsl`, to be
//! reused by the real kernel.
//!
//! Run it with `cargo run --profile fast -p horizon-client-render --example
//! earth_class_compare` (probe mode).

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

#[path = "earth_class_gpu.rs"]
mod gpu;
pub use gpu::{ClassEncoded, ClassifyTimings, Division, EarthClassGpu, kernel_source};

/// Shader source: the exact helpers, then the probe kernels.
const HELPERS_WGSL: &str = include_str!("earth_class.wgsl");
const PROBE_WGSL: &str = include_str!("earth_class_probe.wgsl");

/// Rows per dispatch (a power of two; 64 invocations per workgroup).
pub const CHUNK: usize = 1 << 20;
/// u64 slots each probe row writes (`OUT_STRIDE` in the probe shader).
pub const OUT_STRIDE: usize = 8;
/// How much more the division tests run than `n` (>= 1e7 rows for n >= 1e6).
pub const DIV_FACTOR: usize = 10;

const ENTRY_POINTS: [&str; 11] = [
    "k_plain",
    "k_contract",
    "k_naive_madd",
    "k_naive_msub",
    "k_naive_lerp",
    "k_expr",
    "k_fma",
    "k_div",
    "k_u64",
    "k_conv",
    "k_rem",
];

/// The probe's shader source. `EARTH_CLASS_WGSL_DIR=<dir>` reads both files
/// from disk instead (shader edits without rebuilding).
pub fn probe_source() -> String {
    if let Ok(dir) = std::env::var("EARTH_CLASS_WGSL_DIR") {
        let read = |name: &str| std::fs::read_to_string(std::path::Path::new(&dir).join(name));
        if let (Ok(a), Ok(b)) = (read("earth_class.wgsl"), read("earth_class_probe.wgsl")) {
            return format!("{a}\n{b}");
        }
    }
    format!("{HELPERS_WGSL}\n{PROBE_WGSL}")
}

// ------------------------------------------------------------------ adapter

/// What the adapter is and whether it has the two features.
#[derive(Clone, Debug)]
pub struct AdapterCaps {
    pub name: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub device_type: String,
    pub shader_f64: bool,
    pub shader_int64: bool,
}

impl AdapterCaps {
    pub fn of(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let features = adapter.features();
        Self {
            name: info.name,
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            driver_info: info.driver_info,
            device_type: format!("{:?}", info.device_type),
            shader_f64: features.contains(wgpu::Features::SHADER_F64),
            shader_int64: features.contains(wgpu::Features::SHADER_INT64),
        }
    }

    /// The features missing for exact f64 / u64 work.
    pub fn missing(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if !self.shader_f64 {
            missing.push("SHADER_F64");
        }
        if !self.shader_int64 {
            missing.push("SHADER_INT64");
        }
        missing
    }
}

impl fmt::Display for AdapterCaps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "adapter      : {} ({})", self.name, self.device_type)?;
        writeln!(f, "backend      : {}", self.backend)?;
        writeln!(f, "driver       : {} {}", self.driver, self.driver_info)?;
        writeln!(f, "SHADER_F64   : {}", self.shader_f64)?;
        write!(f, "SHADER_INT64 : {}", self.shader_int64)
    }
}

/// The default adapter (high performance) of an instance honouring the
/// `WGPU_*` environment variables.
pub fn headless_adapter() -> Option<wgpu::Adapter> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle().with_env());
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()
}

/// A headless device with `SHADER_F64 | SHADER_INT64`, or `None` (a line on
/// stderr says what is missing) unless the adapter has both.
pub fn headless_f64_device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let Some(adapter) = headless_adapter() else {
        eprintln!("earth_class: no GPU adapter");
        return None;
    };
    let caps = AdapterCaps::of(&adapter);
    let missing = caps.missing();
    if !missing.is_empty() {
        eprintln!(
            "earth_class: {} / {} lacks {}",
            caps.name,
            caps.backend,
            missing.join(" + ")
        );
        return None;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("earth class f64"),
        required_features: wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(|e| eprintln!("earth_class: request_device failed: {e}"))
    .ok()?;
    Some((adapter, device, queue))
}

// ------------------------------------------------------------------- report

/// One compared quantity.
#[derive(Clone, Debug)]
pub struct TestResult {
    /// Which of the probe's numbered questions ("i": diagnostics).
    pub group: &'static str,
    pub name: String,
    /// Rows compared.
    pub n: u64,
    /// Rows whose bits differ from Rust.
    pub mismatches: u64,
    /// A few mismatching rows.
    pub examples: Vec<String>,
    /// `true` for diagnostics whose mismatch count is not a verdict.
    pub info: bool,
}

impl TestResult {
    pub fn exact(&self) -> bool {
        self.n > 0 && self.mismatches == 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProbeReport {
    pub tests: Vec<TestResult>,
    /// Pipeline or run failures (a kernel that did not compile, ...).
    pub errors: Vec<String>,
    pub elapsed: Duration,
}

impl ProbeReport {
    pub fn get(&self, name: &str) -> Option<&TestResult> {
        self.tests.iter().find(|t| t.name == name)
    }

    /// Whether the named test ran and matched Rust on every row.
    pub fn exact(&self, name: &str) -> bool {
        self.get(name).is_some_and(TestResult::exact)
    }

    /// Whether every non-diagnostic test matched and nothing failed.
    pub fn all_verdicts_exact(&self) -> bool {
        self.errors.is_empty() && self.tests.iter().filter(|t| !t.info).all(TestResult::exact)
    }
}

impl fmt::Display for ProbeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self.tests.iter().map(|t| t.name.len()).max().unwrap_or(10);
        writeln!(
            f,
            "{:<5} {:<width$} {:>12} {:>12}  result",
            "group", "test", "rows", "mismatches"
        )?;
        for t in &self.tests {
            let verdict = if t.n == 0 {
                "NOT RUN"
            } else if t.mismatches == 0 {
                "exact"
            } else if t.info {
                "(info)"
            } else {
                "DIFFERS"
            };
            writeln!(
                f,
                "{:<5} {:<width$} {:>12} {:>12}  {}",
                t.group, t.name, t.n, t.mismatches, verdict
            )?;
            for e in &t.examples {
                writeln!(f, "        e.g. {e}")?;
            }
        }
        for e in &self.errors {
            writeln!(f, "ERROR: {e}")?;
        }
        write!(f, "elapsed: {:.1}s", self.elapsed.as_secs_f64())
    }
}

// ------------------------------------------------------------------- random

/// splitmix64 (the operand generator; the same function the shader mixes with).
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.0)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// A finite f64 with a random sign and mantissa and an unbiased exponent
    /// in `lo..=hi`.
    pub fn wide(&mut self, lo: i32, hi: i32) -> f64 {
        let r = self.next();
        let span = (hi - lo + 1) as u64;
        let e = lo + ((r >> 52) % span) as i32;
        f64::from_bits((r & 0x800F_FFFF_FFFF_FFFF) | (((e + 1023) as u64) << 52))
    }
}

/// The splitmix64 finalizer, in Rust (the reference of the shader's `mix64`).
pub fn mix64(x: u64) -> u64 {
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ------------------------------------------------------------------- runner

struct Runner<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: HashMap<&'static str, Result<wgpu::ComputePipeline, String>>,
    params: wgpu::Buffer,
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    c: wgpu::Buffer,
    out: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl<'a> Runner<'a> {
    fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> Result<Self, String> {
        let source = probe_source();
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("earth class probe"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
        let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("earth class probe"),
            entries: &[
                entry(0, true),
                entry(1, true),
                entry(2, true),
                entry(3, true),
                entry(4, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("earth class probe"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        if let Some(e) = pollster::block_on(validation.pop()) {
            let _ = pollster::block_on(internal.pop());
            return Err(format!("shader module: {e}"));
        }
        let mut pipelines = HashMap::new();
        for name in ENTRY_POINTS {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            });
            let result = match pollster::block_on(scope.pop()) {
                Some(e) => Err(e.to_string()),
                None => Ok(pipeline),
            };
            pipelines.insert(name, result);
        }
        if let Some(e) = pollster::block_on(internal.pop()) {
            return Err(format!("internal error: {e}"));
        }
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let row = (CHUNK * 8) as u64;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        Ok(Self {
            device,
            queue,
            layout,
            pipelines,
            params: buffer("probe params", 16, storage),
            a: buffer("probe a", row, storage),
            b: buffer("probe b", row, storage),
            c: buffer("probe c", row, storage),
            out: buffer(
                "probe out",
                row * OUT_STRIDE as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: buffer(
                "probe staging",
                row * OUT_STRIDE as u64,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        })
    }

    /// Runs `entry` on the rows (a, b, c) (bit patterns, at most [`CHUNK`]);
    /// `OUT_STRIDE` result bit patterns per row. `w` is the shader's width.
    fn run(
        &self,
        entry: &str,
        a: &[u64],
        b: &[u64],
        c: &[u64],
        w: u32,
    ) -> Result<Vec<u64>, String> {
        let pipeline = match self.pipelines.get(entry) {
            Some(Ok(p)) => p,
            Some(Err(e)) => return Err(format!("{entry}: pipeline failed: {e}")),
            None => return Err(format!("{entry}: unknown entry point")),
        };
        let n = a.len();
        assert!(n <= CHUNK && b.len() == n && c.len() == n);
        // n, flag (never 12345), w, zero (always 0).
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&[n as u32, 0, w, 0]));
        self.queue.write_buffer(&self.a, 0, bytemuck::cast_slice(a));
        self.queue.write_buffer(&self.b, 0, bytemuck::cast_slice(b));
        self.queue.write_buffer(&self.c, 0, bytemuck::cast_slice(c));
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.out.as_entire_binding() },
            ],
        });
        let bytes = (n * OUT_STRIDE * 8) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("probe") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(entry),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out, 0, &self.staging, 0, bytes);
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(120)),
            })
            .map_err(|e| format!("{entry}: poll: {e:?}"))?;
        rx.recv_timeout(Duration::from_secs(120))
            .map_err(|e| format!("{entry}: map timeout: {e}"))?
            .map_err(|e| format!("{entry}: map: {e}"))?;
        let words = {
            let view = self
                .staging
                .slice(..bytes)
                .get_mapped_range()
                .map_err(|e| format!("{entry}: mapped range: {e}"))?;
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        self.staging.unmap();
        Ok(words)
    }
}

// ------------------------------------------------------------------- checks

type Expect = Box<dyn Fn(u64, u64, u64) -> Option<u64>>;

/// One output slot of a kernel compared with a Rust expression; `None` from
/// the expression skips a row.
struct Check {
    group: &'static str,
    name: String,
    slot: usize,
    expect: Expect,
    info: bool,
    /// Compare ±0 as equal.
    zero_blind: bool,
    n: u64,
    mismatches: u64,
    examples: Vec<String>,
}

impl Check {
    fn raw(
        group: &'static str,
        name: impl Into<String>,
        slot: usize,
        expect: impl Fn(u64, u64, u64) -> Option<u64> + 'static,
    ) -> Self {
        Self {
            group,
            name: name.into(),
            slot,
            expect: Box::new(expect),
            info: false,
            zero_blind: false,
            n: 0,
            mismatches: 0,
            examples: Vec::new(),
        }
    }

    /// An f64 expression of the row's (a, b, c).
    fn f(
        group: &'static str,
        name: impl Into<String>,
        slot: usize,
        expect: impl Fn(f64, f64, f64) -> f64 + 'static,
    ) -> Self {
        Self::raw(group, name, slot, move |a, b, c| {
            Some(expect(f64::from_bits(a), f64::from_bits(b), f64::from_bits(c)).to_bits())
        })
    }

    fn info(mut self) -> Self {
        self.info = true;
        self
    }

    fn zero_blind(mut self) -> Self {
        self.zero_blind = true;
        self
    }

    fn result(self) -> TestResult {
        TestResult {
            group: self.group,
            name: self.name,
            n: self.n,
            mismatches: self.mismatches,
            examples: self.examples,
            info: self.info,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Operands {
    Floats,
    Ints,
}

fn show(bits: u64, kind: Operands) -> String {
    match kind {
        Operands::Floats => format!("{:e}[{bits:016x}]", f64::from_bits(bits)),
        Operands::Ints => format!("{bits:016x}"),
    }
}

type RowGen<'a> = &'a mut dyn FnMut(&mut Rng, usize) -> (u64, u64, u64);

/// Runs `entry` over `rows` generated rows (in chunks) and feeds every check.
fn run_set(
    runner: &Runner<'_>,
    entry: &str,
    rows: usize,
    w: u32,
    kind: Operands,
    rng: &mut Rng,
    gen_row: RowGen<'_>,
    checks: &mut [Check],
) -> Result<(), String> {
    let mut done = 0;
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    while done < rows {
        let len = (rows - done).min(CHUNK);
        a.clear();
        b.clear();
        c.clear();
        for i in 0..len {
            let (x, y, z) = gen_row(rng, done + i);
            a.push(x);
            b.push(y);
            c.push(z);
        }
        let out = runner.run(entry, &a, &b, &c, w)?;
        for check in checks.iter_mut() {
            for i in 0..len {
                let Some(want) = (check.expect)(a[i], b[i], c[i]) else {
                    continue;
                };
                check.n += 1;
                let got = out[i * OUT_STRIDE + check.slot];
                let same = got == want
                    || (check.zero_blind && (got | want) << 1 == 0);
                if !same {
                    check.mismatches += 1;
                    if check.examples.len() < 3 {
                        check.examples.push(format!(
                            "a={} b={} c={} got={} want={}",
                            show(a[i], kind),
                            show(b[i], kind),
                            show(c[i], kind),
                            show(got, kind),
                            show(want, kind)
                        ));
                    }
                }
            }
        }
        done += len;
    }
    Ok(())
}

fn finish(r: Result<(), String>, checks: Vec<Check>) -> Result<Vec<Check>, String> {
    r.map(|()| checks)
}

// ------------------------------------------------------------- the questions

/// Runs every probe over `n` random rows per test (the division tests over
/// `DIV_FACTOR * n`) and compares the GPU's bits with Rust's. `device` must
/// have `SHADER_F64 | SHADER_INT64` (see [`headless_f64_device`]).
pub fn probe_exactness(device: &wgpu::Device, queue: &wgpu::Queue, n: usize) -> ProbeReport {
    let start = Instant::now();
    let mut report = ProbeReport::default();
    let runner = match Runner::new(device, queue) {
        Ok(r) => r,
        Err(e) => {
            report.errors.push(e);
            report.elapsed = start.elapsed();
            return report;
        }
    };
    let n = n.max(1);
    let mut rng = Rng(0x00C0_FFEE_1234_5678);
    type Probe = fn(&Runner<'_>, usize, &mut Rng) -> Result<Vec<Check>, String>;
    let probes: [Probe; 8] = [
        probe_plain,
        probe_contraction,
        probe_expr,
        probe_fma,
        probe_division,
        probe_u64,
        probe_conversions,
        probe_rem,
    ];
    for probe in probes {
        match probe(&runner, n, &mut rng) {
            Ok(checks) => report.tests.extend(checks.into_iter().map(Check::result)),
            Err(e) => report.errors.push(e),
        }
    }
    report.elapsed = start.elapsed();
    report
}

/// 1. Plain f64 add, sub, mul.
fn probe_plain(r: &Runner<'_>, n: usize, rng: &mut Rng) -> Result<Vec<Check>, String> {
    let mut checks = vec![
        Check::f("1", "plain a+b", 0, |a, b, _| a + b),
        Check::f("1", "plain a-b", 1, |a, b, _| a - b),
        Check::f("1", "plain a*b", 2, |a, b, _| a * b),
    ];
    let mut gen_row = |rng: &mut Rng, i: usize| {
        let (a, b) = match i % 4 {
            0 => (rng.wide(-60, 60), rng.wide(-60, 60)),
            1 => {
                // Close exponents: cancellation and sticky bits.
                let a = rng.wide(-20, 20);
                let b = f64::from_bits(
                    (rng.next() & 0x800F_FFFF_FFFF_FFFF) | (a.to_bits() & 0x7FF0_0000_0000_0000),
                );
                (a, b)
            }
            2 => (rng.range(-1e7, 1e7), rng.range(-1e7, 1e7)),
            _ => (rng.range(-1.0, 1.0), rng.range(-200.0, 200.0)),
        };
        (a.to_bits(), b.to_bits(), 0)
    };
    let res = run_set(r, "k_plain", n, 0, Operands::Floats, rng, &mut gen_row, &mut checks);
    finish(res, checks)
}

/// 2. FMA contraction of `a*b + c`, `a*b - c` and `a + (b - a) * u`: the
/// naive expression alone in its kernel (one use of the product) and behind
/// the opaque guards (in `k_contract`, where the products are shared).
fn probe_contraction(r: &Runner<'_>, n: usize, rng: &mut Rng) -> Result<Vec<Check>, String> {
    let mut all = Vec::new();
    // c near -a*b, where fused and separate roundings differ most.
    let gen_madd = |rng: &mut Rng, i: usize| {
        let a = if i % 2 == 0 { rng.wide(-30, 30) } else { rng.range(-1e4, 1e4) };
        let b = if i % 2 == 0 { rng.wide(-30, 30) } else { rng.range(-1e4, 1e4) };
        let p = a * b;
        let c = match i % 4 {
            0 | 1 => -p * (1.0 + rng.range(-1e-9, 1e-9)),
            2 => -f64::from_bits(p.to_bits().wrapping_add(rng.next() % 64)),
            _ => p * rng.range(-2.0, 2.0),
        };
        (a.to_bits(), b.to_bits(), c.to_bits())
    };
    let mut set = vec![
        Check::f("2", "mul_nc(a,b)+c", 2, |a, b, c| a * b + c),
        Check::f("2", "mulx_nc(a,b)+c", 4, |a, b, c| a * b + c),
        Check::f("2", "mul_nc(a,b)-c", 7, |a, b, c| a * b - c),
        Check::f("i", "rust: fused != separate (a*b+c)", 2, |a, b, c| a.mul_add(b, c)).info(),
    ];
    let res = run_set(r, "k_contract", n, 0, Operands::Floats, rng, &mut { gen_madd }, &mut set);
    all.extend(finish(res, set)?);
    for (entry, name, op) in [
        ("k_naive_madd", "naive a*b+c (one use)", false),
        ("k_naive_msub", "naive a*b-c (one use)", true),
    ] {
        let mut set = vec![
            // Contraction is expected here: a diagnostic, not a verdict.
            Check::f("2", name, 0, move |a, b, c| if op { a * b - c } else { a * b + c }).info(),
            Check::f(
                "i",
                format!("{name} vs fused"),
                0,
                move |a, b, c| if op { a.mul_add(b, -c) } else { a.mul_add(b, c) },
            )
            .info(),
        ];
        let res = run_set(r, entry, n, 0, Operands::Floats, rng, &mut { gen_madd }, &mut set);
        all.extend(finish(res, set)?);
    }

    // a + (b - a) * u with u in [0, 1), realistic ranges.
    let gen_lerp = |rng: &mut Rng, i: usize| {
        let (a, b) = if i % 2 == 0 {
            (rng.range(-180.0, 180.0), rng.range(-180.0, 180.0))
        } else {
            (rng.range(-1e7, 1e7), rng.range(-1e7, 1e7))
        };
        (a.to_bits(), b.to_bits(), rng.unit().to_bits())
    };
    let mut set = vec![
        Check::f("2", "lerp mul_nc", 3, |a, b, u| a + (b - a) * u),
        Check::f("2", "lerp mulx_nc", 5, |a, b, u| a + (b - a) * u),
    ];
    let res = run_set(r, "k_contract", n, 0, Operands::Floats, rng, &mut { gen_lerp }, &mut set);
    all.extend(finish(res, set)?);
    let mut set = vec![
        Check::f("2", "naive lerp a+(b-a)*u (one use)", 0, |a, b, u| a + (b - a) * u).info(),
        Check::f("i", "naive lerp (one use) vs fused", 0, |a, b, u| (b - a).mul_add(u, a)).info(),
    ];
    let res = run_set(r, "k_naive_lerp", n, 0, Operands::Floats, rng, &mut { gen_lerp }, &mut set);
    all.extend(finish(res, set)?);
    Ok(all)
}

/// 2b. Several products in one expression, naive and guarded.
fn probe_expr(r: &Runner<'_>, n: usize, rng: &mut Rng) -> Result<Vec<Check>, String> {
    let mut checks = vec![
        Check::f("2", "naive a*b+c*a", 0, |a, b, c| a * b + c * a).info(),
        Check::f("2", "guarded a*b+c*a", 1, |a, b, c| a * b + c * a),
        Check::f("2", "naive a*b-b*c", 2, |a, b, c| a * b - b * c).info(),
        Check::f("2", "guarded a*b-b*c", 3, |a, b, c| a * b - b * c),
        Check::f("2", "naive horner ((ab+c)a+b)c+a", 4, |a, b, c| ((a * b + c) * a + b) * c + a).info(),
        Check::f("2", "guarded horner", 5, |a, b, c| ((a * b + c) * a + b) * c + a),
        Check::f("2", "lerp_nc(a,b,c)", 6, |a, b, c| a + (b - a) * c),
        Check::f("2", "muladd_nc(a,b,c)", 7, |a, b, c| a * b + c),
    ];
    let mut gen_row = |rng: &mut Rng, i: usize| {
        let (a, b, c) = if i % 2 == 0 {
            (rng.range(-180.0, 180.0), rng.range(-90.0, 90.0), rng.range(-1.0, 1.0))
        } else {
            (rng.wide(-20, 20), rng.wide(-20, 20), rng.wide(-20, 20))
        };
        (a.to_bits(), b.to_bits(), c.to_bits())
    };
    let res = run_set(r, "k_expr", n, 0, Operands::Floats, rng, &mut gen_row, &mut checks);
    finish(res, checks)
}

/// 6. WGSL `fma()` on f64 is a real fused multiply-add.
fn probe_fma(r: &Runner<'_>, n: usize, rng: &mut Rng) -> Result<Vec<Check>, String> {
    let mut checks = vec![
        Check::f("6", "fma(a,b,c) == f64::mul_add", 0, |a, b, c| a.mul_add(b, c)),
        Check::f("6", "fma_exact(a,b,c) == f64::mul_add", 1, |a, b, c| a.mul_add(b, c)),
    ];
    let mut gen_row = |rng: &mut Rng, i: usize| {
        let a = rng.wide(-30, 30);
        let b = rng.wide(-30, 30);
        let c = if i % 2 == 0 {
            -(a * b) * (1.0 + rng.range(-1e-9, 1e-9))
        } else {
            rng.wide(-60, 60)
        };
        (a.to_bits(), b.to_bits(), c.to_bits())
    };
    let res = run_set(r, "k_fma", n, 0, Operands::Floats, rng, &mut gen_row, &mut checks);
    finish(res, checks)
}

/// 3. Division: native `/`, `div_cr`, `div_by` with a precomputed reciprocal.
fn probe_division(r: &Runner<'_>, n: usize, rng: &mut Rng) -> Result<Vec<Check>, String> {
    let rows = n * DIV_FACTOR;
    let mut all = Vec::new();
    struct DivSet {
        label: &'static str,
        fixed: Option<f64>,
        range: Option<(f64, f64)>,
    }
    let sets = [
        DivSet { label: "general", fixed: None, range: None },
        DivSet { label: "|a|<1e7 / 6378137", fixed: Some(6378137.0), range: None },
        DivSet { label: "|a|<1e7 / 3", fixed: Some(3.0), range: None },
        DivSet { label: "|a|<1e7 / 92000.12345", fixed: Some(92000.12345), range: None },
        DivSet { label: "|a|<1e7 / b in 7e4..1.12e5", fixed: None, range: Some((7e4, 1.12e5)) },
    ];
    for set in sets {
        let general = set.fixed.is_none() && set.range.is_none();
        let label = set.label;
        let mut checks = vec![
            Check::f("3", format!("native a/b [{label}]"), 0, |a, b, _| a / b),
            Check::f("3", format!("div_cr(a,b) [{label}]"), 1, |a, b, _| a / b),
            Check::f("3", format!("div_by(a,b,1/b) [{label}]"), 2, |a, b, _| a / b),
        ];
        if general || set.range.is_some() {
            checks.push(Check::f("3", format!("native 1.0/b [{label}]"), 4, |_, b, _| 1.0 / b));
            checks.push(Check::f("3", format!("recip_newton(b) [{label}]"), 5, |_, b, _| 1.0 / b).info());
        }
        if general {
            checks.push(Check::f("i", "a*(1/b) vs a/b (naive)", 3, |a, b, _| a / b).info());
        }
        let (fixed, range) = (set.fixed, set.range);
        let mut gen_row = move |rng: &mut Rng, i: usize| {
            let (a, b) = if general {
                let a = match i % 8 {
                    0..=4 => rng.wide(-100, 100),
                    5 => rng.range(-1e7, 1e7),
                    6 => rng.range(-1e7, 1e7).round(),
                    _ => rng.wide(-5, 5),
                };
                let sign = rng.next() & 0x8000_0000_0000_0000;
                let exp = (rng.next() % 200 + 923) << 52;
                let b = match i % 64 {
                    // Mantissa 0 and all ones: the awkward divisors.
                    1 => f64::from_bits(sign | exp),
                    2 => f64::from_bits(sign | exp | 0x000F_FFFF_FFFF_FFFF),
                    _ => rng.wide(-100, 100),
                };
                (a, b)
            } else {
                let a = if i % 2 == 0 {
                    rng.range(-1e7, 1e7)
                } else {
                    rng.range(-1e7, 1e7).round()
                };
                let b = match (fixed, range) {
                    (Some(b), _) => b,
                    (_, Some((lo, hi))) => rng.range(lo, hi),
                    _ => unreachable!(),
                };
                (a, b)
            };
            (a.to_bits(), b.to_bits(), (1.0 / b).to_bits())
        };
        let res = run_set(r, "k_div", rows, 0, Operands::Floats, rng, &mut gen_row, &mut checks);
        all.extend(finish(res, checks)?);
    }
    Ok(all)
}

/// 4. u64 multiply, shifts, xor, add, mix64, the 53-bit unit float.
fn probe_u64(r: &Runner<'_>, n: usize, rng: &mut Rng) -> Result<Vec<Check>, String> {
    const G: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut checks = vec![
        Check::raw("4", "u64 a*0x9E3779B97F4A7C15 (wrapping)", 0, |a, _, _| {
            Some(a.wrapping_mul(G))
        }),
        Check::raw("4", "mix64(a) (splitmix finalizer)", 1, |a, _, _| Some(mix64(a))),
        Check::raw("4", "f64(mix64(a)>>11)*2^-53", 2, |a, _, _| {
            Some(((mix64(a) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)).to_bits())
        }),
        Check::raw("4", "u64 a*b (wrapping)", 3, |a, b, _| Some(a.wrapping_mul(b))),
        Check::raw("4", "shifts >>11 >>27 >>30 >>31 and ^", 4, |a, b, _| {
            Some((a >> 11) ^ (b >> 27) ^ (a >> 31) ^ (b >> 30))
        }),
        Check::raw("4", "(a+b wrapping) ^ (a>>11)", 5, |a, b, _| {
            Some(a.wrapping_add(b) ^ (a >> 11))
        }),
        Check::raw("4", "u64 as f64 (all 64 bits, RNE)", 6, |a, _, _| Some((a as f64).to_bits())),
        Check::raw("4", "literal 0x1p-53lf", 7, |_, _, _| {
            Some((1.0 / (1u64 << 53) as f64).to_bits())
        }),
    ];
    let mut gen_row = |rng: &mut Rng, i: usize| {
        let special = [
            0,
            1,
            u64::MAX,
            1 << 63,
            1 << 53,
            (1 << 53) + 1,
            u64::MAX - 1,
            0x8000_0000_0000_0001,
        ];
        if i < special.len() * 2 {
            return (special[i % special.len()], special[(i / 2) % special.len()], 0);
        }
        (rng.next(), rng.next(), 0)
    };
    let res = run_set(r, "k_u64", n, 0, Operands::Ints, rng, &mut gen_row, &mut checks);
    finish(res, checks)
}

fn compare_mask(a: f64, b: f64) -> u64 {
    u64::from(a < b)
        | u64::from(a <= b) << 1
        | u64::from(a > b) << 2
        | u64::from(a >= b) << 3
        | u64::from(a == b) << 4
        | u64::from(a != b) << 5
}

/// 5a. floor, f64 -> i64 / u32 conversions, abs, comparisons.
fn probe_conversions(r: &Runner<'_>, n: usize, rng: &mut Rng) -> Result<Vec<Check>, String> {
    const W: u32 = 4096;
    let in_range = |a: f64| !a.is_nan() && a.abs() < 9.2e18;
    let mut checks = vec![
        Check::raw("5", "floor(f64)", 0, |a, _, _| {
            let a = f64::from_bits(a);
            (!a.is_nan()).then(|| a.floor().to_bits())
        }),
        Check::raw("5", "floor_i64 as u64 (saturating, NaN->0)", 1, |a, _, _| {
            Some((f64::from_bits(a).floor() as i64) as u64)
        }),
        Check::raw("5", "clamp_px: (a.max(0) as u32).min(w-1)", 2, |a, _, _| {
            let a = f64::from_bits(a);
            (!a.is_nan()).then(|| u64::from((a.max(0.0) as u32).min(W - 1)))
        }),
        Check::raw("5", "abs(f64)", 3, |a, _, _| {
            let a = f64::from_bits(a);
            (!a.is_nan()).then(|| a.abs().to_bits())
        }),
        Check::raw("5", "compare < <= > >= == != (non-NaN)", 4, |a, b, _| {
            let (a, b) = (f64::from_bits(a), f64::from_bits(b));
            (!a.is_nan() && !b.is_nan()).then(|| compare_mask(a, b))
        }),
        // The GPU does not preserve NaN semantics: a diagnostic.
        Check::raw("5", "compare ops on NaN rows (NaN unsupported)", 4, |a, b, _| {
            let (a, b) = (f64::from_bits(a), f64::from_bits(b));
            (a.is_nan() || b.is_nan()).then(|| compare_mask(a, b))
        })
        .info(),
        Check::raw("5", "plain i64(floor(a)) in range", 5, move |a, _, _| {
            let a = f64::from_bits(a);
            in_range(a).then(|| (a.floor() as i64) as u64)
        }),
        Check::raw("5", "f64(i64(floor(a))) in range", 6, move |a, _, _| {
            let a = f64::from_bits(a);
            in_range(a).then(|| ((a.floor() as i64) as f64).to_bits())
        }),
    ];
    let specials = [
        0.0,
        -0.0,
        1e19,
        -1e19,
        9223372036854775808.0,
        -9223372036854775808.0,
        18446744073709551616.0,
        4294967295.0,
        4294967296.0,
        4294967295.5,
        4294967294.5,
        -1.0,
        -0.5,
        0.5,
        4095.5,
        4096.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        5e-324,
        9007199254740993.0,
    ];
    let mut gen_row = |rng: &mut Rng, i: usize| {
        if i < specials.len() * 2 {
            return (
                specials[i % specials.len()].to_bits(),
                specials[(i / 2) % specials.len()].to_bits(),
                0,
            );
        }
        let a = match i % 8 {
            0 => rng.range(-1e12, 1e12),
            1 => rng.range(-1e6, 1e6).floor() + 0.5,
            2 => rng.range(-1e9, 1e9).round(),
            3 => rng.range(-10.0, 5000.0),
            4 => rng.range(-1e-9, 1e-9),
            5 => {
                let k = rng.range(-1e6, 1e6).round();
                f64::from_bits(k.to_bits().wrapping_add(rng.next() % 3).wrapping_sub(1))
            }
            6 => rng.wide(-5, 60),
            _ => rng.range(-2e9, 8e9),
        };
        let b = match i % 4 {
            0 => a,
            1 => f64::from_bits(a.to_bits().wrapping_add(rng.next() % 3).wrapping_sub(1)),
            _ => rng.range(-1e12, 1e12),
        };
        (a.to_bits(), b.to_bits(), 0)
    };
    let res = run_set(r, "k_conv", n, W, Operands::Floats, rng, &mut gen_row, &mut checks);
    finish(res, checks)
}

/// 5b. fmod, rem_euclid(360) and longitude wrapping for values in [-720, 720].
fn probe_rem(r: &Runner<'_>, n: usize, rng: &mut Rng) -> Result<Vec<Check>, String> {
    let wrap = |l: f64| (l + 180.0).rem_euclid(360.0) - 180.0;
    let mut checks = vec![
        Check::f("5", "rem_euclid_small(a,360)", 0, |a, _, _| a.rem_euclid(360.0)),
        Check::f("5", "fmod_small(a,360)", 2, |a, _, _| a % 360.0),
        Check::f("5", "wrap_longitude(l)", 3, move |a, _, _| wrap(a)),
        Check::f("5", "native a % 360 (WGSL %)", 1, |a, _, _| a % 360.0).info(),
        Check::f("5", "native a % 360, +-0 equal", 1, |a, _, _| a % 360.0).zero_blind().info(),
        Check::f("5", "native rem_euclid via %", 4, |a, _, _| a.rem_euclid(360.0)).info(),
        Check::f("5", "native rem_euclid via %, +-0 equal", 4, |a, _, _| a.rem_euclid(360.0))
            .zero_blind()
            .info(),
        Check::f("5", "native wrap via %", 5, move |a, _, _| wrap(a)).info(),
    ];
    let mut gen_row = |rng: &mut Rng, i: usize| {
        let a = match i % 8 {
            0 | 1 => rng.range(-720.0, 720.0),
            2 => rng.range(-2880.0, 2880.0).round() * 0.25,
            3 => {
                rng.range(-2.0, 2.0).round() * 360.0
                    + [0.0, 0.0, -0.0, 180.0, -180.0][i / 8 % 5]
            }
            4 => {
                let k = rng.range(-2.0, 2.0).round() * 360.0;
                f64::from_bits(k.to_bits().wrapping_add(rng.next() % 5).wrapping_sub(2))
            }
            5 => rng.range(-1.0, 1.0) * 10f64.powi(-((rng.next() % 300) as i32) - 5),
            6 => rng.range(-190.0, 190.0),
            _ => rng.range(-540.0, 540.0),
        };
        (a.to_bits(), 0, 0)
    };
    let res = run_set(r, "k_rem", n, 0, Operands::Floats, rng, &mut gen_row, &mut checks);
    finish(res, checks)
}

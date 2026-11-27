//! The Earth far view's padded-window elevation on the GPU: a [`WindowGpu`] for `horizon_world`'s `EarthGenerator`.
//!
//! Far workers (several at once) hand over [`WindowJob`]s: a separable sampling plan plus the elevation tiles it
//! reads. A batcher thread collects the jobs that are waiting, uploads the tiles the GPU lacks into a pool of tile
//! slots (an LRU keyed by [`TileId`]; a tile a batch uses is never evicted for that batch), writes one job buffer,
//! runs **one** compute dispatch that does every sample of every job, copies the heights into one readback buffer,
//! maps it once and gives each waiting worker its heights. The CPU keeps being the reference: any failure (no
//! compute, pool too small for the job, a lost device) answers `None` and the worker computes the window itself.
//!
//! Precision: with `SHADER_F64` on the device the shader does the bilinear mix in `f64` exactly as
//! `WindowPlan::heights_m` does (products kept out of fused multiply-adds), so the heights are the CPU's; without
//! it the mix is `f32`, off by a few millimetres at most (an elevation tile is metres-quantised, the fraction is
//! what loses bits).
//!
//! One instance serves one tile source: the pool is keyed by tile id alone.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use horizon_earth::{Tile, TileId, WindowPlan};
use horizon_world::cubic::earth::{WindowGpu, WindowJob, WindowResult, WindowTicket};

const SHADER: &str = include_str!("earth_window.wgsl");

/// Floats in one tile slot (256 x 256).
const TILE_FLOATS: usize = 256 * 256;
const TILE_BYTES: u64 = TILE_FLOATS as u64 * 4;
/// Tile slots in the pool at most (64 MiB), fewer when the device's storage binding is smaller.
const POOL_SLOTS: u32 = 256;
/// Fewest slots worth having (one job can read a handful of tiles).
const MIN_SLOTS: u32 = 16;
/// Jobs in one dispatch at most.
const MAX_JOBS: usize = 128;
/// Samples in one dispatch at most (8 MiB of heights).
const MAX_SAMPLES: usize = 1 << 21;
/// Storage buffers the shader binds.
const STORAGE_BUFFERS: u32 = 5;
/// The readback and the wait for the GPU give up after this.
const GPU_TIMEOUT: Duration = Duration::from_secs(30);

/// Counters of what the batcher did since it started.
#[derive(Clone, Copy, Debug, Default)]
pub struct EarthWindowStats {
    /// Dispatches (each with one readback).
    pub batches: u64,
    /// Jobs answered from the GPU.
    pub jobs: u64,
    /// Samples computed.
    pub samples: u64,
    /// Tiles uploaded into the pool.
    pub uploads: u64,
    /// Tiles that left the pool to make room.
    pub evictions: u64,
    /// Time from starting a batch (after collecting the jobs) to having its heights, nanoseconds.
    pub gpu_ns: u64,
}

#[derive(Default)]
struct Counters {
    batches: AtomicU64,
    jobs: AtomicU64,
    samples: AtomicU64,
    uploads: AtomicU64,
    evictions: AtomicU64,
    gpu_ns: AtomicU64,
}

/// How an [`EarthWindowGpu`] is set up.
#[derive(Clone, Copy, Debug)]
pub struct EarthWindowOptions {
    /// Use `f64` in the shader when the device has `SHADER_F64`.
    pub allow_f64: bool,
    /// After the first job of a batch, wait this long for more before dispatching (zero: take what is already
    /// queued only; jobs arriving while a batch is on the GPU make the next batch anyway).
    pub linger: Duration,
}

impl Default for EarthWindowOptions {
    fn default() -> Self {
        Self {
            allow_f64: true,
            linger: Duration::ZERO,
        }
    }
}

struct Msg {
    job: WindowJob,
    reply: mpsc::Sender<Option<WindowResult>>,
}

struct Ticket {
    rx: mpsc::Receiver<Option<WindowResult>>,
}

impl WindowTicket for Ticket {
    fn wait(self: Box<Self>) -> Option<WindowResult> {
        self.rx.recv().ok().flatten()
    }
}

/// Computes padded-window elevations on the GPU. See the module docs.
pub struct EarthWindowGpu {
    /// Taken (closing the channel) when the sampler is dropped, so the batcher thread ends.
    tx: Option<mpsc::Sender<Msg>>,
    thread: Option<thread::JoinHandle<()>>,
    alive: Arc<AtomicBool>,
    counters: Arc<Counters>,
    f64: bool,
    slots: u32,
}

impl EarthWindowGpu {
    /// The sampler on this device and queue, or `None` when the device cannot run it (too few storage buffers
    /// per stage, a binding too small for a useful pool, or the shader does not build).
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Option<Arc<Self>> {
        Self::with_options(device, queue, EarthWindowOptions::default())
    }

    /// [`EarthWindowGpu::new`] with options.
    pub fn with_options(
        device: wgpu::Device,
        queue: wgpu::Queue,
        options: EarthWindowOptions,
    ) -> Option<Arc<Self>> {
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < STORAGE_BUFFERS {
            return None;
        }
        let by_limit = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / TILE_BYTES;
        let slots = u32::try_from(by_limit).unwrap_or(u32::MAX).min(POOL_SLOTS);
        if slots < MIN_SLOTS {
            return None;
        }
        let want_f64 = options.allow_f64 && device.features().contains(wgpu::Features::SHADER_F64);
        let (pipeline, f64) = match want_f64.then(|| build_pipeline(&device, true)).flatten() {
            Some(p) => (p, true),
            None => (build_pipeline(&device, false)?, false),
        };
        let pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Earth window tile pool"),
            size: TILE_BYTES * u64::from(slots),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Earth window globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let alive = Arc::new(AtomicBool::new(true));
        let counters = Arc::new(Counters::default());
        let (tx, rx) = mpsc::channel();
        let batcher = Batcher {
            device,
            queue,
            pipeline,
            f64,
            pool,
            globals,
            growable: None,
            map: HashMap::new(),
            slot_id: vec![None; slots as usize],
            last_used: vec![0; slots as usize],
            epoch_of: vec![0; slots as usize],
            epoch: 0,
            tick: 0,
            uploads: Vec::new(),
            alive: alive.clone(),
            counters: counters.clone(),
            linger: options.linger,
        };
        let thread = thread::Builder::new()
            .name("earth-window-gpu".into())
            .spawn(move || batcher.run(rx))
            .ok()?;
        Some(Arc::new(Self {
            tx: Some(tx),
            thread: Some(thread),
            alive,
            counters,
            f64,
            slots,
        }))
    }

    /// Whether the shader mixes in `f64` (the CPU's arithmetic) rather than `f32`.
    pub fn uses_f64(&self) -> bool {
        self.f64
    }

    /// Tile slots in the pool.
    pub fn pool_slots(&self) -> u32 {
        self.slots
    }

    /// Whether the batcher still runs (it stops for good when the device fails).
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    /// What the batcher did so far.
    pub fn stats(&self) -> EarthWindowStats {
        let c = &self.counters;
        EarthWindowStats {
            batches: c.batches.load(Ordering::Relaxed),
            jobs: c.jobs.load(Ordering::Relaxed),
            samples: c.samples.load(Ordering::Relaxed),
            uploads: c.uploads.load(Ordering::Relaxed),
            evictions: c.evictions.load(Ordering::Relaxed),
            gpu_ns: c.gpu_ns.load(Ordering::Relaxed),
        }
    }
}

impl Drop for EarthWindowGpu {
    /// Ends the batcher (after the batch it is on) and waits for it, so the device is gone before the caller's
    /// process or window is.
    fn drop(&mut self) {
        self.tx.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl WindowGpu for EarthWindowGpu {
    fn submit(&self, job: WindowJob) -> Option<Box<dyn WindowTicket>> {
        if !self.alive.load(Ordering::Relaxed) || !plan_is_sound(&job.plan) {
            return None;
        }
        let (reply, rx) = mpsc::channel();
        self.tx.as_ref()?.send(Msg { job, reply }).ok()?;
        Some(Box::new(Ticket { rx }))
    }
}

/// The plan's lists agree with each other (the shader trusts them).
fn plan_is_sound(p: &WindowPlan) -> bool {
    let (nx, ny) = (p.tile_xs.len(), p.tile_ys.len());
    p.width > 0
        && p.depth > 0
        && p.cols.len() == p.width
        && p.rows.len() == p.depth
        && p.tiles.len() == nx * ny
        && nx <= usize::from(u16::MAX)
        && ny <= usize::from(u16::MAX)
        && p.tiles
            .iter()
            .all(|(_, t)| t.heights().len() == TILE_FLOATS)
        && p.cols
            .iter()
            .all(|c| usize::from(c.a.tile) < nx && usize::from(c.b.tile) < nx)
        && p.rows
            .iter()
            .all(|r| usize::from(r.a.tile) < ny && usize::from(r.b.tile) < ny)
}

/// The compute pipeline with `T` as `f64` or `f32`; `None` when it does not validate.
fn build_pipeline(device: &wgpu::Device, f64: bool) -> Option<wgpu::ComputePipeline> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let source = format!("alias T = {};\n{SHADER}", if f64 { "f64" } else { "f32" });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Earth window"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Earth window"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let internal = pollster::block_on(internal.pop());
    let validation = pollster::block_on(validation.pop());
    if let Some(e) = internal.or(validation) {
        log::warn!("Horizon renderer: Earth window GPU (f64 {f64}): {e}");
        return None;
    }
    Some(pipeline)
}

/// Buffers that grow with the batch.
struct Growable {
    hdrs: wgpu::Buffer,
    data: wgpu::Buffer,
    ts: wgpu::Buffer,
    out: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Capacities: data words, ts values, samples.
    data_cap: usize,
    ts_cap: usize,
    out_cap: usize,
}

struct Batcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    f64: bool,
    pool: wgpu::Buffer,
    globals: wgpu::Buffer,
    growable: Option<Growable>,
    /// Tile -> slot.
    map: HashMap<TileId, u32>,
    slot_id: Vec<Option<TileId>>,
    /// Per slot: when it was last used (a batch counter value), and the batch that last pinned it.
    last_used: Vec<u64>,
    epoch_of: Vec<u64>,
    epoch: u64,
    tick: u64,
    /// Tiles to write into the pool before the next dispatch.
    uploads: Vec<(u32, Arc<Tile>)>,
    alive: Arc<AtomicBool>,
    counters: Arc<Counters>,
    linger: Duration,
}

impl Batcher {
    fn run(mut self, rx: mpsc::Receiver<Msg>) {
        let mut pending: VecDeque<Msg> = VecDeque::new();
        loop {
            if pending.is_empty() {
                match rx.recv() {
                    Ok(m) => pending.push_back(m),
                    Err(_) => return,
                }
            }
            let deadline = Instant::now() + self.linger;
            while pending.len() < MAX_JOBS {
                match rx.try_recv() {
                    Ok(m) => pending.push_back(m),
                    Err(mpsc::TryRecvError::Disconnected) => break,
                    Err(mpsc::TryRecvError::Empty) => {
                        let left = deadline.saturating_duration_since(Instant::now());
                        if left.is_zero() {
                            break;
                        }
                        match rx.recv_timeout(left) {
                            Ok(m) => pending.push_back(m),
                            Err(_) => break,
                        }
                    }
                }
            }
            if !self.alive.load(Ordering::Relaxed) {
                for m in pending.drain(..) {
                    let _ = m.reply.send(None);
                }
                continue;
            }
            self.batch(&mut pending);
        }
    }

    /// Takes the jobs at the front of `pending` that fit one dispatch, runs them and answers them.
    fn batch(&mut self, pending: &mut VecDeque<Msg>) {
        let started = Instant::now();
        self.epoch += 1;
        let mut jobs: Vec<(Msg, Vec<u32>)> = Vec::new();
        let mut samples = 0usize;
        while let Some(msg) = pending.front() {
            if jobs.len() >= MAX_JOBS {
                break;
            }
            let n = msg.job.plan.width * msg.job.plan.depth;
            if !jobs.is_empty() && samples + n > MAX_SAMPLES {
                break;
            }
            if n > MAX_SAMPLES {
                let msg = pending.pop_front().expect("front exists");
                let _ = msg.reply.send(None);
                continue;
            }
            match self.assign(&msg.job.plan) {
                Some(slots) => {
                    samples += n;
                    jobs.push((pending.pop_front().expect("front exists"), slots));
                }
                // The pool is full of tiles this batch uses: next batch, or never when even alone it does not fit.
                None if jobs.is_empty() => {
                    let msg = pending.pop_front().expect("front exists");
                    let _ = msg.reply.send(None);
                }
                None => break,
            }
        }
        if jobs.is_empty() {
            return;
        }
        match self.dispatch(&jobs, samples) {
            Some(all) => {
                let mut at = 0;
                for (msg, _) in &jobs {
                    let n = msg.job.plan.width * msg.job.plan.depth;
                    let _ = msg.reply.send(Some(WindowResult {
                        heights_m: all[at..at + n].to_vec(),
                    }));
                    at += n;
                }
                let c = &self.counters;
                c.batches.fetch_add(1, Ordering::Relaxed);
                c.jobs.fetch_add(jobs.len() as u64, Ordering::Relaxed);
                c.samples.fetch_add(samples as u64, Ordering::Relaxed);
                c.gpu_ns
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            None => {
                self.alive.store(false, Ordering::Relaxed);
                for (msg, _) in jobs {
                    let _ = msg.reply.send(None);
                }
            }
        }
    }

    /// The pool slots of the plan's tiles (in `plan.tiles` order), loading the missing ones; `None` when there is
    /// no slot left that this batch does not already use.
    fn assign(&mut self, plan: &WindowPlan) -> Option<Vec<u32>> {
        let mut slots = Vec::with_capacity(plan.tiles.len());
        for (id, tile) in &plan.tiles {
            self.tick += 1;
            let slot = match self.map.get(id) {
                Some(&s) => s,
                None => {
                    let s = self.victim()?;
                    if let Some(old) = self.slot_id[s as usize].take() {
                        self.map.remove(&old);
                        self.counters.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                    self.map.insert(*id, s);
                    self.slot_id[s as usize] = Some(*id);
                    self.uploads.push((s, tile.clone()));
                    s
                }
            };
            self.epoch_of[slot as usize] = self.epoch;
            self.last_used[slot as usize] = self.tick;
            slots.push(slot);
        }
        Some(slots)
    }

    /// An empty slot, else the least recently used one this batch does not use.
    fn victim(&self) -> Option<u32> {
        let mut best: Option<(u64, u32)> = None;
        for s in 0..self.slot_id.len() {
            if self.slot_id[s].is_none() {
                return Some(s as u32);
            }
            if self.epoch_of[s] == self.epoch {
                continue;
            }
            if best.is_none_or(|(t, _)| self.last_used[s] < t) {
                best = Some((self.last_used[s], s as u32));
            }
        }
        best.map(|(_, s)| s)
    }

    /// Makes the growable buffers big enough.
    fn ensure(&mut self, data_words: usize, ts_values: usize, samples: usize) {
        let fits = self.growable.as_ref().is_some_and(|g| {
            g.data_cap >= data_words && g.ts_cap >= ts_values && g.out_cap >= samples
        });
        if fits {
            return;
        }
        let old = self.growable.take();
        let cap =
            |have: usize, need: usize, floor: usize| have.max(need.next_power_of_two()).max(floor);
        let data_cap = cap(old.as_ref().map_or(0, |g| g.data_cap), data_words, 4096);
        let ts_cap = cap(old.as_ref().map_or(0, |g| g.ts_cap), ts_values, 4096);
        let out_cap = cap(old.as_ref().map_or(0, |g| g.out_cap), samples, 1 << 16);
        let buffer = |label, size: u64, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage_in = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let hdrs = buffer("Earth window jobs", (MAX_JOBS * 32) as u64, storage_in);
        let data = buffer("Earth window axes", data_cap as u64 * 4, storage_in);
        let ts_size = if self.f64 { 8 } else { 4 };
        let ts = buffer(
            "Earth window fractions",
            ts_cap as u64 * ts_size,
            storage_in,
        );
        let out = buffer(
            "Earth window heights",
            out_cap as u64 * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = buffer(
            "Earth window readback",
            out_cap as u64 * 4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let layout = self.pipeline.get_bind_group_layout(0);
        fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Earth window"),
            layout: &layout,
            entries: &[
                entry(0, &self.globals),
                entry(1, &hdrs),
                entry(2, &data),
                entry(3, &ts),
                entry(4, &self.pool),
                entry(5, &out),
            ],
        });
        self.growable = Some(Growable {
            hdrs,
            data,
            ts,
            out,
            readback,
            bind_group,
            data_cap,
            ts_cap,
            out_cap,
        });
    }

    /// Writes the jobs, dispatches once and reads every height back (`samples` of them, job after job).
    fn dispatch(&mut self, jobs: &[(Msg, Vec<u32>)], samples: usize) -> Option<Vec<f32>> {
        let mut hdrs: Vec<u32> = Vec::with_capacity(jobs.len() * 8);
        let mut data: Vec<u32> = Vec::new();
        let mut ts32: Vec<f32> = Vec::new();
        let mut ts64: Vec<f64> = Vec::new();
        let mut start = 0usize;
        for (msg, slots) in jobs {
            let p = &msg.job.plan;
            let ts_off = if self.f64 { ts64.len() } else { ts32.len() };
            hdrs.extend_from_slice(&[
                p.width as u32,
                p.depth as u32,
                start as u32,
                data.len() as u32,
                p.tile_xs.len() as u32,
                start as u32,
                ts_off as u32,
                0,
            ]);
            let pack = |a: horizon_earth::AxisIndex| u32::from(a.tile) << 16 | u32::from(a.local);
            for axis in p.cols.iter().chain(p.rows.iter()) {
                data.push(pack(axis.a));
                data.push(pack(axis.b));
                if self.f64 {
                    ts64.push(axis.t);
                } else {
                    ts32.push(axis.t as f32);
                }
            }
            data.extend_from_slice(slots);
            start += p.width * p.depth;
        }
        let ts_values = ts64.len().max(ts32.len());
        self.ensure(data.len(), ts_values, samples);
        let g = self.growable.as_ref()?;
        let uploads = std::mem::take(&mut self.uploads);
        for (slot, tile) in &uploads {
            self.queue.write_buffer(
                &self.pool,
                TILE_BYTES * u64::from(*slot),
                bytemuck::cast_slice(tile.heights()),
            );
        }
        self.counters
            .uploads
            .fetch_add(uploads.len() as u64, Ordering::Relaxed);
        let globals = [0u32, jobs.len() as u32, samples as u32, 0];
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::cast_slice(&globals));
        self.queue
            .write_buffer(&g.hdrs, 0, bytemuck::cast_slice(&hdrs));
        self.queue
            .write_buffer(&g.data, 0, bytemuck::cast_slice(&data));
        if self.f64 {
            self.queue
                .write_buffer(&g.ts, 0, bytemuck::cast_slice(&ts64));
        } else {
            self.queue
                .write_buffer(&g.ts, 0, bytemuck::cast_slice(&ts32));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Earth window"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Earth window"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &g.bind_group, &[]);
            pass.dispatch_workgroups(samples.div_ceil(64) as u32, 1, 1);
        }
        let bytes = samples as u64 * 4;
        encoder.copy_buffer_to_buffer(&g.out, 0, &g.readback, 0, bytes);
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        g.readback
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        let polled = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(GPU_TIMEOUT),
        });
        if polled.is_err() || !matches!(rx.recv_timeout(GPU_TIMEOUT), Ok(Ok(()))) {
            return None;
        }
        let heights = {
            let view = g.readback.slice(..bytes).get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        g.readback.unmap();
        Some(heights)
    }
}

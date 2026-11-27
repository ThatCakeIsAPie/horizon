//! The Earth far view's land classes on the GPU: [`horizon_earth::ClassWindowPlan::evaluate`] (which is
//! `LandCoverField::class_grid` bit for bit) as a wgpu compute kernel (`earth_class_kernel.wgsl`, after the
//! exact helpers of `earth_class.wgsl`). Needs `SHADER_F64 | SHADER_INT64`.
//!
//! Tiles live in a pool of fixed-size slots on the GPU (codes and speck codes, 16 bits a pixel), kept by an
//! LRU keyed by `(file, level, tile)` and the decoded tile's identity. A batch of windows is one upload of
//! the tiles it lacks, one buffer of job records, one dispatch and one readback.

use std::borrow::Cow;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use horizon_earth::landcover::SPECK_KEEP;
use horizon_earth::{ClassWindowPlan, LandClass, WorldCoverFile};

const KERNEL_WGSL: &str = include_str!("earth_class_kernel.wgsl");

/// The speck salt's constant (`LandTile::speck_kept`).
const SPECK_SALT: u64 = 0x5EC4_5BEC_0000_0001;
const HEADER_WORDS: usize = 16;
const AXIS_WORDS: usize = 7;
const FILE_WORDS: usize = 7;
const BLANK_SLOT: u32 = u32::MAX;
const PARAM_WORDS: usize = 8 + 256;
/// Windows a dispatch takes at most (the y dimension of the dispatch).
const MAX_WINDOWS_PER_DISPATCH: usize = 65_535;

/// How the kernel divides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Division {
    /// The native f64 `/` (correctly rounded on the drivers probed; see `probe_exactness`).
    Native,
    /// Correctly rounded division from fma (`div_cr`).
    Fma,
}

/// The kernel's source for a division mode. `EARTH_CLASS_WGSL_DIR=<dir>` reads the `.wgsl` files from disk.
pub fn kernel_source(division: Division) -> String {
    let (mut helpers, mut kernel) = (super::HELPERS_WGSL.to_string(), KERNEL_WGSL.to_string());
    if let Ok(dir) = std::env::var("EARTH_CLASS_WGSL_DIR") {
        let read = |name: &str| std::fs::read_to_string(std::path::Path::new(&dir).join(name));
        if let (Ok(a), Ok(b)) = (read("earth_class.wgsl"), read("earth_class_kernel.wgsl")) {
            (helpers, kernel) = (a, b);
        }
    }
    if division == Division::Fma {
        kernel = kernel.replace(
            "const USE_DIV_CR: bool = false;",
            "const USE_DIV_CR: bool = true;",
        );
    }
    format!("{helpers}\n{kernel}")
}

/// Where the time of a [`EarthClassGpu::classify_timed`] went.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassifyTimings {
    /// Tiles copied into the pool, their bytes, and the time to pack and queue them.
    pub tiles_uploaded: usize,
    pub upload_bytes: u64,
    pub upload: Duration,
    /// Packing the job records (CPU).
    pub pack: Duration,
    /// Writing the jobs, the dispatch, the wait and the readback.
    pub gpu: Duration,
    /// Dispatches made (more than one when the batch needs more tiles than the pool holds).
    pub dispatches: usize,
}

impl std::ops::AddAssign for ClassifyTimings {
    fn add_assign(&mut self, o: Self) {
        self.tiles_uploaded += o.tiles_uploaded;
        self.upload_bytes += o.upload_bytes;
        self.upload += o.upload;
        self.pack += o.pack;
        self.gpu += o.gpu;
        self.dispatches += o.dispatches;
    }
}

type TileKey = (WorldCoverFile, u32, u32);

struct SlotEntry {
    slot: u32,
    identity: usize,
    used: u64,
}

/// A growable GPU buffer.
struct Grow {
    buffer: wgpu::Buffer,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl Grow {
    fn new(device: &wgpu::Device, label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Self {
            buffer: Self::make(device, label, usage, 1 << 16),
            usage,
            label,
        }
    }

    fn make(device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    fn fit(&mut self, device: &wgpu::Device, bytes: u64) {
        if self.buffer.size() < bytes {
            let size = bytes.next_power_of_two().max(1 << 16);
            self.buffer = Self::make(device, self.label, self.usage, size);
        }
    }
}

/// The land-class kernel with its tile pool.
pub struct EarthClassGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    division: Division,
    params: wgpu::Buffer,
    pool: wgpu::Buffer,
    slots: usize,
    slot_pixels: usize,
    lru: HashMap<TileKey, SlotEntry>,
    owner: Vec<Option<TileKey>>,
    clock: u64,
    windows: Grow,
    jobs: Grow,
    out: Grow,
    staging: Grow,
}

impl EarthClassGpu {
    /// Tile pixels a pool slot holds (WorldCover's tiles are 1024 by 1024).
    pub const SLOT_PIXELS: usize = 1024 * 1024;
    /// The pool's default size: 128 slots of 2 MiB.
    pub const POOL_BYTES: u64 = 256 << 20;

    /// The kernel on `device` (which must have been made with `SHADER_F64 | SHADER_INT64`, e.g. by
    /// [`super::headless_f64_device`]); `None` without them or when the kernel does not build.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        Self::with_pool(device, queue, Self::POOL_BYTES, Division::Native)
    }

    /// [`EarthClassGpu::new`] with a pool of about `pool_bytes` and the division mode.
    pub fn with_pool(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pool_bytes: u64,
        division: Division,
    ) -> Option<Self> {
        let needed = wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64;
        if !device.features().contains(needed) {
            log::warn!("earth class gpu: the device lacks SHADER_F64 | SHADER_INT64");
            return None;
        }
        let limits = device.limits();
        let slot_bytes = (Self::SLOT_PIXELS * 2) as u64;
        let cap = pool_bytes
            .min(u64::from(limits.max_storage_buffer_binding_size))
            .min(limits.max_buffer_size);
        let slots = (cap / slot_bytes) as usize;
        if slots == 0 {
            return None;
        }
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
            label: Some("earth class"),
            entries: &[
                entry(0, true),
                entry(1, true),
                entry(2, true),
                entry(3, true),
                entry(4, false),
            ],
        });
        let pipeline = Self::pipeline(device, &layout, division)?;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("earth class params"),
            size: (PARAM_WORDS * 8) as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let mut words = vec![0u64; PARAM_WORDS];
        words[0] = 0; // nc_flag: never 12345
        words[1] = 0; // nc_zero
        words[2] = 1e-12_f64.to_bits();
        words[3] = SPECK_KEEP.to_bits();
        words[4] = (1.0 - f64::EPSILON).to_bits();
        words[5] = (Self::SLOT_PIXELS / 2) as u64;
        for code in 0..256usize {
            words[8 + code] = u64::from(
                LandClass::from_code(code as u8)
                    .unwrap_or(LandClass::NoData)
                    .code(),
            );
        }
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&words));
        let pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("earth class tile pool"),
            size: slots as u64 * slot_bytes,
            usage: storage,
            mapped_at_creation: false,
        });
        Some(Self {
            device: device.clone(),
            queue: queue.clone(),
            layout,
            pipeline,
            division,
            params,
            pool,
            slots,
            slot_pixels: Self::SLOT_PIXELS,
            lru: HashMap::new(),
            owner: vec![None; slots],
            clock: 0,
            windows: Grow::new(device, "earth class windows", storage),
            jobs: Grow::new(device, "earth class jobs", storage),
            out: Grow::new(
                device,
                "earth class out",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: Grow::new(
                device,
                "earth class staging",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        })
    }

    fn pipeline(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        division: Division,
    ) -> Option<wgpu::ComputePipeline> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("earth class kernel"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(kernel_source(division))),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("earth class"),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("earth class"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("classify"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            log::error!("earth class kernel: {e}");
            eprintln!("earth class kernel: {e}");
            return None;
        }
        Some(pipeline)
    }

    /// Rebuilds the pipeline for another division mode.
    pub fn set_division(&mut self, division: Division) -> bool {
        if division == self.division {
            return true;
        }
        match Self::pipeline(&self.device, &self.layout, division) {
            Some(p) => {
                self.pipeline = p;
                self.division = division;
                true
            }
            None => false,
        }
    }

    pub fn division(&self) -> Division {
        self.division
    }

    /// Slots in the tile pool.
    pub fn pool_slots(&self) -> usize {
        self.slots
    }

    /// Forgets every pooled tile (the next batch uploads all it needs).
    pub fn clear_pool(&mut self) {
        self.lru.clear();
        self.owner.iter_mut().for_each(|o| *o = None);
    }

    /// The class codes ([`LandClass::code`]) of each window, row by row; [`horizon_earth::OUTSIDE_PLAN`] for
    /// a sample whose file or tile the plan does not hold. `None` for a window the kernel cannot take (a tile larger than a
    /// pool slot, more tiles than the pool) or when the GPU fails: the caller uses the CPU.
    pub fn classify(&mut self, plans: &[&ClassWindowPlan]) -> Vec<Option<Vec<u8>>> {
        self.classify_timed(plans).0
    }

    /// [`EarthClassGpu::classify`] with where the time went.
    pub fn classify_timed(
        &mut self,
        plans: &[&ClassWindowPlan],
    ) -> (Vec<Option<Vec<u8>>>, ClassifyTimings) {
        let mut out: Vec<Option<Vec<u8>>> = vec![None; plans.len()];
        let mut timings = ClassifyTimings::default();
        // Group the windows into dispatches whose tiles fit in the pool together.
        let mut group: Vec<usize> = Vec::new();
        let mut group_keys: HashMap<TileKey, ()> = HashMap::new();
        for (w, plan) in plans.iter().enumerate() {
            if !self.supported(plan) {
                continue;
            }
            let keys = Self::held_keys(plan);
            let new = keys.iter().filter(|k| !group_keys.contains_key(k)).count();
            if keys.len() > self.slots {
                continue;
            }
            if group_keys.len() + new > self.slots || group.len() >= MAX_WINDOWS_PER_DISPATCH {
                self.run_group(plans, &group, &mut out, &mut timings);
                group.clear();
                group_keys.clear();
            }
            for k in keys {
                group_keys.insert(k, ());
            }
            group.push(w);
        }
        if !group.is_empty() {
            self.run_group(plans, &group, &mut out, &mut timings);
        }
        (out, timings)
    }

    fn supported(&self, plan: &ClassWindowPlan) -> bool {
        plan.width > 0
            && plan.depth > 0
            && plan.width <= u32::MAX as usize
            && plan.tiles.iter().all(|t| {
                t.tile.is_none() || (t.width() as usize * t.height() as usize) <= self.slot_pixels
            })
    }

    /// The pool keys of the tiles a plan holds data for (blank tiles need no slot).
    fn held_keys(plan: &ClassWindowPlan) -> Vec<TileKey> {
        (0..plan.tiles.len())
            .filter(|&k| plan.tiles[k].tile.is_some())
            .map(|k| plan.tile_key(k))
            .collect()
    }

    /// Makes sure the tile is in a slot (uploading it when not); its slot.
    fn ensure_tile(
        &mut self,
        plan: &ClassWindowPlan,
        k: usize,
        stamp: u64,
        timings: &mut ClassifyTimings,
    ) -> u32 {
        let key = plan.tile_key(k);
        let tile = &plan.tiles[k];
        let identity = tile.identity();
        let held = self.lru.get_mut(&key).map(|e| {
            e.used = stamp;
            (e.slot, e.identity == identity)
        });
        // The same key with other data (decoded again): upload into its own slot. Otherwise a free slot, or
        // the least recently used one not used by this dispatch.
        let slot = match (held, self.owner.iter().position(Option::is_none)) {
            (Some((slot, true)), _) => return slot,
            (Some((slot, false)), _) => slot as usize,
            (None, Some(s)) => s,
            (None, None) => {
                let (victim, _) = self
                    .lru
                    .iter()
                    .filter(|(_, e)| e.used < stamp)
                    .min_by_key(|(_, e)| e.used)
                    .map(|(k, e)| (*k, e.slot))
                    .expect("a group never needs more tiles than the pool holds");
                let e = self.lru.remove(&victim).expect("victim");
                e.slot as usize
            }
        };
        if let Some(old) = self.owner[slot].take() {
            self.lru.remove(&old);
        }
        let t = Instant::now();
        let (codes, specks) = (tile.codes(), tile.specks());
        let mut pixels: Vec<u16> = codes.iter().map(|&c| u16::from(c)).collect();
        for &(index, code) in specks {
            if let Some(p) = pixels.get_mut(index as usize) {
                *p |= u16::from(code) << 8;
            }
        }
        if pixels.len() % 2 == 1 {
            pixels.push(0);
        }
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        self.queue.write_buffer(
            &self.pool,
            slot as u64 * (self.slot_pixels * 2) as u64,
            bytes,
        );
        timings.upload += t.elapsed();
        timings.upload_bytes += bytes.len() as u64;
        timings.tiles_uploaded += 1;
        self.owner[slot] = Some(key);
        self.lru.insert(
            key,
            SlotEntry {
                slot: slot as u32,
                identity,
                used: stamp,
            },
        );
        slot as u32
    }

    fn run_group(
        &mut self,
        plans: &[&ClassWindowPlan],
        group: &[usize],
        out: &mut [Option<Vec<u8>>],
        timings: &mut ClassifyTimings,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("earth class"),
            });
        let encoded = self.encode_group(plans, group, &mut encoder, timings);
        let t = Instant::now();
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging_slice(&encoded)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        let ok = self
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(60)),
            })
            .is_ok()
            && matches!(rx.recv_timeout(Duration::from_secs(60)), Ok(Ok(())));
        self.read_group(plans, encoded, ok, out);
        timings.gpu += t.elapsed();
        timings.dispatches += 1;
    }

    /// Encodes one batch of windows into the caller's `encoder` (for a caller that submits it together with its
    /// own work and maps [`EarthClassGpu::staging_slice`] itself): the windows whose tiles fit in the pool
    /// together, in order; the others are left out (their answer from [`EarthClassGpu::finish_encoded`] is
    /// `None`). `None` when no window can go.
    pub fn encode(
        &mut self,
        plans: &[&ClassWindowPlan],
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<ClassEncoded> {
        let mut group: Vec<usize> = Vec::new();
        let mut keys_in: HashMap<TileKey, ()> = HashMap::new();
        for (w, plan) in plans.iter().enumerate() {
            if !self.supported(plan) || group.len() >= MAX_WINDOWS_PER_DISPATCH {
                continue;
            }
            let keys = Self::held_keys(plan);
            let new = keys.iter().filter(|k| !keys_in.contains_key(k)).count();
            if keys_in.len() + new > self.slots {
                continue;
            }
            for k in keys {
                keys_in.insert(k, ());
            }
            group.push(w);
        }
        if group.is_empty() {
            return None;
        }
        let mut timings = ClassifyTimings::default();
        Some(self.encode_group(plans, &group, encoder, &mut timings))
    }

    /// The readback buffer of an encoded batch, to map once the caller's submission is done.
    pub fn staging_slice(&self, encoded: &ClassEncoded) -> wgpu::BufferSlice<'_> {
        self.staging.buffer.slice(..encoded.bytes.max(4))
    }

    /// The codes of an encoded batch after its staging buffer was mapped (`ok`) or failed to (`!ok`): one entry
    /// per plan given to [`EarthClassGpu::encode`], `None` for a window left out or on failure. Unmaps.
    pub fn finish_encoded(
        &mut self,
        plans: &[&ClassWindowPlan],
        encoded: ClassEncoded,
        ok: bool,
    ) -> Vec<Option<Vec<u8>>> {
        let mut out = vec![None; plans.len()];
        self.read_group(plans, encoded, ok, &mut out);
        out
    }

    /// Uploads the tiles of `group`, writes its job records and encodes the dispatch and the copy to staging.
    fn encode_group(
        &mut self,
        plans: &[&ClassWindowPlan],
        group: &[usize],
        encoder: &mut wgpu::CommandEncoder,
        timings: &mut ClassifyTimings,
    ) -> ClassEncoded {
        self.clock += 1;
        let stamp = self.clock;
        // Tiles first (their slots go into the jobs).
        let mut slots_of: Vec<Vec<u32>> = Vec::with_capacity(group.len());
        for &w in group {
            let plan = plans[w];
            let slots = (0..plan.tiles.len())
                .map(|k| {
                    if plan.tiles[k].tile.is_none() {
                        BLANK_SLOT
                    } else {
                        self.ensure_tile(plan, k, stamp, timings)
                    }
                })
                .collect();
            slots_of.push(slots);
        }
        let t = Instant::now();
        let mut jobs: Vec<u64> = Vec::new();
        let mut windows: Vec<u64> = Vec::with_capacity(group.len() * 2);
        let mut total = 0usize;
        let mut max_samples = 0usize;
        for (g, &w) in group.iter().enumerate() {
            let plan = plans[w];
            let samples = plan.width * plan.depth;
            windows.push(jobs.len() as u64);
            windows.push(total as u64 | ((samples as u64) << 32));
            pack_job(plan, &slots_of[g], &mut jobs);
            total += samples;
            max_samples = max_samples.max(samples);
        }
        timings.pack += t.elapsed();

        let device = self.device.clone();
        self.windows.fit(&device, (windows.len() * 8) as u64);
        self.jobs.fit(&device, (jobs.len() * 8) as u64);
        self.out.fit(&device, (total * 4) as u64);
        self.staging.fit(&device, (total * 4) as u64);
        self.queue
            .write_buffer(&self.windows.buffer, 0, bytemuck::cast_slice(&windows));
        self.queue
            .write_buffer(&self.jobs.buffer, 0, bytemuck::cast_slice(&jobs));
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("earth class"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.windows.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.jobs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.pool.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.out.buffer.as_entire_binding(),
                },
            ],
        });
        let bytes = (total * 4) as u64;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("earth class"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((max_samples as u32).div_ceil(64), group.len() as u32, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out.buffer, 0, &self.staging.buffer, 0, bytes);
        ClassEncoded {
            group: group.to_vec(),
            total,
            bytes,
        }
    }

    /// Reads a mapped batch into `out` (or, on failure, drops the pool whose state is unknown), and unmaps.
    fn read_group(
        &mut self,
        plans: &[&ClassWindowPlan],
        encoded: ClassEncoded,
        ok: bool,
        out: &mut [Option<Vec<u8>>],
    ) {
        let ClassEncoded {
            group,
            total,
            bytes,
        } = encoded;
        if ok {
            let codes: Vec<u32> = {
                let view = self.staging.buffer.slice(..bytes.max(4)).get_mapped_range();
                match view {
                    Ok(view) => bytemuck::cast_slice::<u8, u32>(&view).to_vec(),
                    Err(_) => Vec::new(),
                }
            };
            self.staging.buffer.unmap();
            if codes.len() >= total {
                let mut at = 0;
                for &w in &group {
                    let n = plans[w].width * plans[w].depth;
                    out[w] = Some(codes[at..at + n].iter().map(|&c| c as u8).collect());
                    at += n;
                }
            }
        } else {
            // Leave the windows `None` (the CPU answers them); drop the pool, its state is unknown.
            self.clear_pool();
        }
    }
}

/// A batch [`EarthClassGpu::encode`] put into a caller's command encoder.
pub struct ClassEncoded {
    /// Indexes of the plans in the batch, in output order.
    group: Vec<usize>,
    /// Samples in the batch.
    total: usize,
    /// Bytes of codes to read back.
    bytes: u64,
}

/// Appends a plan's job record (see the layout in `earth_class_kernel.wgsl`); `slots[k]` is the pool slot of
/// `plan.tiles[k]` (or [`BLANK_SLOT`]).
fn pack_job(plan: &ClassWindowPlan, slots: &[u32], jobs: &mut Vec<u64>) {
    let base = jobs.len();
    let pair = |lo: u32, hi: u32| u64::from(lo) | (u64::from(hi) << 32);
    let [coarse, fine] = &plan.octaves;
    // Tile entries grouped by file, in file order (the plan's tiles are in table order of the files).
    let mut entries: Vec<Vec<usize>> = vec![Vec::new(); plan.files.len()];
    for (k, t) in plan.tiles.iter().enumerate() {
        entries[t.file as usize].push(k);
    }
    jobs.extend_from_slice(&[
        pair(plan.width as u32, plan.depth as u32),
        pair(plan.files.len() as u32, plan.tiles.len() as u32),
        coarse.amplitude_x.to_bits(),
        coarse.amplitude_z.to_bits(),
        fine.amplitude_x.to_bits(),
        fine.amplitude_z.to_bits(),
        coarse.salt_x,
        coarse.salt_z,
        fine.salt_x,
        fine.salt_z,
        plan.metres_per_block.to_bits(),
        plan.lat0.to_bits(),
        plan.lon0.to_bits(),
        plan.metres_per_degree_lon.to_bits(),
        plan.earth_radius_m.to_bits(),
        plan.radians_to_degrees.to_bits(),
    ]);
    debug_assert_eq!(jobs.len() - base, HEADER_WORDS);
    for axis in plan.columns.iter().chain(&plan.rows) {
        jobs.push(axis.pos.to_bits());
        for o in &axis.octaves {
            jobs.extend_from_slice(&[o.fade.to_bits(), o.term0, o.term1]);
        }
    }
    let mut first = 0u32;
    for (f, file) in plan.files.iter().enumerate() {
        let count = entries[f].len() as u32;
        jobs.extend_from_slice(&[
            pair(i32::from(file.south) as u32, i32::from(file.west) as u32),
            pair(u32::from(file.is_no_data()), file.level_no),
            pair(file.width, file.height),
            pair(file.tile_width, file.tile_height),
            pair(file.tiles_across, file.tiles_down),
            file.speck_salt ^ SPECK_SALT,
            pair(first, count),
        ]);
        first += count;
    }
    debug_assert_eq!(
        jobs.len() - base,
        HEADER_WORDS + (plan.width + plan.depth) * AXIS_WORDS + plan.files.len() * FILE_WORDS
    );
    for list in &entries {
        for &k in list {
            let t = &plan.tiles[k];
            jobs.push(pair(t.col, t.row));
            jobs.push(u64::from(slots[k]));
        }
    }
}

//! Earth's sub-pixel far tiles rasterized in compute into the terrain visibility buffer
//! (part C of the farworld batch; `far_raster.wgsl` has the passes).
//!
//! Per far_cull phase, before the phase's ID pass, [`FarRaster::dispatch`] takes the far tiles
//! of far_cull's draw lists whose voxel spans at most [`FAR_RASTER_PX`] pixels at the tile's
//! nearest point ([`raster_small`], the WGSL's `far_raster_small`) out of the hardware draws
//! (their instance count becomes 0; their boxes stay in far_cull's lists and Hi-Z test) and
//! plots their quads with 64-bit atomics (depth bits high, a compact ID low). In the ID pass,
//! after far_cull's draws, [`FarRaster::ids`] draws the alpha-tested quads it left over with the
//! ID pipeline and merges the plots over the screen tiles they touched: depth through the ID
//! pipeline's depth test, the far tile ID (`vis_terrain::id::arena(FAR_TILE, ..)`) the hardware
//! draw writes. So the resolve sees the same IDs, and the Hi-Z pyramid built after phase 1 holds
//! the compute tiles' depth. Measured on a GTX 1070 Ti (cproto, horizon-farworld-bc-notes.md
//! "C0 method pick"): the far pass 0.70 vs 0.74 ms at lod 4096, 0.82 vs 0.98 ms at 16384.
//!
//! Needs `SHADER_INT64` and `SHADER_INT64_ATOMIC_MIN_MAX` ([`features`]); without them the far
//! tiles stay hardware drawn (the 32-bit two-pass scheme measured slower than hardware).
//! Default off ([`DEFAULT_ON`]: slower in the game on the GTX 1070 Ti), `HORIZON_FAR_RASTER=1` on,
//! `0` off (no pipelines); `raster=0/1` in the `HORIZON_VISBUFFER_TOGGLE` file flips it in a session.
//!
//! The compact ID (`vis_terrain::id::RASTER`'s 32 bits, never in the IDs texture): arena page
//! << 26 | the page's compute quad, which the merge turns into its draw and corner 0's vertex
//! through the page's prefix of compute quads (`fr_starts`) and its heads (`fr_heads`).

use glam::Mat4;

/// The env knob turning the compute raster off (`0`).
pub const ENV: &str = "HORIZON_FAR_RASTER";
/// A far tile goes to compute when its voxel spans at most this many pixels at its nearest
/// point (`far_raster.wgsl`'s `FAR_RASTER_PX`). Both paths sample the same pixel centres with
/// the same fill rule, so a tile crossing it does not pop: no hysteresis.
#[cfg_attr(not(test), allow(dead_code))] // The WGSL's mirror, pinned by a test.
pub const FAR_RASTER_PX: f32 = 1.25;
/// Big quads and leftover draws a page holds.
const BIG_CAPACITY: u32 = 1 << 16;
const LEFTOVER_CAPACITY: u32 = 1 << 14;
/// Compute quads a head covers, threads of the small pass, groups of the big one.
const HEAD_QUADS: u64 = 128;
#[allow(dead_code)] // far_raster.wgsl's small-pass workgroup size (written there).
const SMALL_WG: u32 = 128;
const BIG_GROUPS: u32 = 512;
const MASK_TILE: u32 = 16;
/// Arena pages an ID names (`vis_terrain::id::MAX_PAGES`).
const PAGES: u32 = 64;
/// The dynamic offset step of the per-dispatch uniform.
const PASS_STRIDE: u64 = 256;

/// The switch: a voxel of `voxel` blocks at `nearest` blocks spans `voxel * focal / nearest`
/// pixels; `far_raster.wgsl`'s `far_raster_small` is the same.
#[cfg_attr(not(test), allow(dead_code))] // The WGSL's mirror, pinned by a test.
pub fn raster_small(voxel: f32, nearest: f32, focal: f32) -> bool {
    voxel > 0.0 && voxel * focal <= FAR_RASTER_PX * nearest
}

/// Pixels a block spans at a distance of one block, as the shader reads it off the view
/// projection (its y row's length, the projection's y scale) for a target `height` pixels high.
#[cfg_attr(not(test), allow(dead_code))] // The WGSL's mirror, pinned by a test.
pub fn focal(view_proj: Mat4, height: u32) -> f32 {
    let m = view_proj.to_cols_array_2d();
    glam::Vec3::new(m[0][1], m[1][1], m[2][1]).length() * height as f32 * 0.5
}

/// The device features it needs, where the adapter has them all.
pub fn features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let needed = wgpu::Features::SHADER_INT64 | wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX;
    if adapter.features().contains(needed) {
        needed
    } else {
        wgpu::Features::empty()
    }
}

/// Whether frames on the visibility buffer use it when the toggle file does not say: off, measured
/// slower in the game on the GTX 1070 Ti (horizon-farworld-bc-notes.md "C built"). On with
/// `HORIZON_FAR_RASTER=1`.
pub const DEFAULT_ON: bool = false;

/// `HORIZON_FAR_RASTER=1` asks for it (see [`DEFAULT_ON`]); `0` turns it off, pipelines too.
pub fn env_allows() -> bool {
    std::env::var(ENV).as_deref() != Ok("0")
}

/// The raster's default for this run: `HORIZON_FAR_RASTER=1` on, else [`DEFAULT_ON`].
pub fn env_default() -> bool {
    std::env::var(ENV).as_deref() == Ok("1") || DEFAULT_ON
}

/// terrain.wgsl's items the passes read the corners with, verbatim: the arena page binding,
/// the vertex bases, `ArenaCorner` and its functions, `fade_depth` and what it calls.
const TERRAIN_ITEMS: [&str; 12] = [
    "@group(1) @binding(19) var<storage, read> arena_page",
    "const PACKED_BASE",
    "const FAR_BASE",
    "const TILED_UV_STEPS",
    "struct ArenaCorner",
    "fn arena_vertex(",
    "fn corner_offset(",
    "fn packed_corner(",
    "fn far_corner(",
    "fn instance_step(",
    "fn fade_code(",
    "fn fade_depth(",
];

/// The item of `source` starting at the line that starts with `head`: to the end of the line
/// (`;`) or of its braces.
fn wgsl_item<'a>(source: &'a str, head: &str) -> Option<&'a str> {
    let start = source
        .match_indices(head)
        .map(|(i, _)| i)
        .find(|&i| i == 0 || source.as_bytes()[i - 1] == b'\n')?;
    let rest = &source[start..];
    let line_end = rest.find('\n').unwrap_or(rest.len());
    let brace = rest.find('{');
    if brace.is_none_or(|b| b > line_end && rest[..line_end].trim_end().ends_with(';')) {
        return Some(&rest[..line_end]);
    }
    let mut depth = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// What terrain.wgsl's `fade_depth` reads of `globals`, for the modules that take it without the
/// world pass's `Globals`: `world_view` zero, the main view's branch (the raster only runs in the
/// main view's ID pass; a mod's world view with a clip plane keeps hardware raster).
pub const FADE_PRELUDE: &str = "struct FrFadeGlobals { world_view: vec4<f32> }\n\
                                var<private> globals: FrFadeGlobals;\n";

/// The passes' module: terrain.wgsl's corner items, then `far_raster.wgsl`.
pub fn shader_source() -> String {
    let terrain = super::terrain_shader();
    let mut out = String::from(FADE_PRELUDE);
    for head in TERRAIN_ITEMS {
        let item = wgsl_item(&terrain, head)
            .unwrap_or_else(|| panic!("terrain.wgsl lost `{head}` (far_raster.rs)"));
        out.push_str(item);
        out.push('\n');
    }
    out + include_str!("far_raster.wgsl")
}

/// What a frame's dispatch reads of far_cull (`FarCull::raster_inputs`).
pub struct Inputs<'a> {
    /// The world pass's globals (terrain.wgsl `Globals`, its view projection first).
    pub globals: &'a wgpu::Buffer,
    pub params: &'a wgpu::Buffer,
    pub records: &'a wgpu::Buffer,
    pub instances: &'a wgpu::Buffer,
    pub draws: &'a wgpu::Buffer,
    pub counts: &'a wgpu::Buffer,
    /// The pages with records: (arena page, first draw of its region, the region's size).
    pub pages: &'a [(u32, u32, u32)],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PageInfo {
    region: [u32; 4],
    lists: [u32; 4],
}

struct Buffers {
    vis: wgpu::Buffer,
    mask: wgpu::Buffer,
    tile_list: wgpu::Buffer,
    size: [u32; 2],
}

/// The compute raster's pipelines and buffers.
pub struct FarRaster {
    prep: wgpu::ComputePipeline,
    small: wgpu::ComputePipeline,
    big: wgpu::ComputePipeline,
    tiles_reset: wgpu::ComputePipeline,
    tiles: wgpu::ComputePipeline,
    merge: wgpu::RenderPipeline,
    main_layout: wgpu::BindGroupLayout,
    page_layout: wgpu::BindGroupLayout,
    tiles_layout: wgpu::BindGroupLayout,
    merge_layout: wgpu::BindGroupLayout,
    frame: wgpu::Buffer,
    passes: wgpu::Buffer,
    page_infos: wgpu::Buffer,
    meta: wgpu::Buffer,
    dispatch: wgpu::Buffer,
    tile_args: wgpu::Buffer,
    dispatch_group: wgpu::BindGroup,
    starts: Option<wgpu::Buffer>,
    heads: Option<wgpu::Buffer>,
    big_list: Option<wgpu::Buffer>,
    leftovers: Option<wgpu::Buffer>,
    targets: Option<Buffers>,
    /// The frame's groups, kept while their buffers stay.
    main_group: Option<(Vec<wgpu::Buffer>, wgpu::BindGroup)>,
    merge_group: Option<(Vec<wgpu::Buffer>, wgpu::BindGroup)>,
    tiles_group: Option<(Vec<wgpu::Buffer>, wgpu::BindGroup)>,
    page_groups: Vec<(u32, wgpu::Buffer, wgpu::BindGroup)>,
    /// This frame's pages (arena page, its leftovers' first draw).
    pages: Vec<(u32, u32)>,
    /// Phases dispatched and not merged yet (their plots are still in `vis`).
    pending: std::cell::Cell<[bool; 2]>,
    heads_per_page: u64,
}

fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn uniform(dynamic: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: dynamic,
        min_binding_size: None,
    }
}

fn layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}

fn buffer(device: &wgpu::Device, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(16),
        usage,
        mapped_at_creation: false,
    })
}

/// `slot` holds a buffer of at least `bytes` (grown by powers of two).
fn ensure(
    device: &wgpu::Device,
    slot: &mut Option<wgpu::Buffer>,
    label: &str,
    bytes: u64,
    usage: wgpu::BufferUsages,
) {
    if slot.as_ref().is_some_and(|b| b.size() >= bytes) {
        return;
    }
    *slot = Some(buffer(device, label, bytes.next_power_of_two(), usage));
}

fn group_of(
    device: &wgpu::Device,
    cache: &mut Option<(Vec<wgpu::Buffer>, wgpu::BindGroup)>,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let key: Vec<wgpu::Buffer> = entries.iter().map(|(_, b)| (*b).clone()).collect();
    if let Some((k, group)) = cache
        && *k == key
    {
        return group.clone();
    }
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries
            .iter()
            .map(|&(binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    *cache = Some((key, group.clone()));
    group
}

impl FarRaster {
    /// `None` without the device features, with `HORIZON_FAR_RASTER=0`, or when the pipelines
    /// fail. `ids_depth`: the ID pass's depth format.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        let needed = wgpu::Features::SHADER_INT64 | wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX;
        if !env_allows() || !device.features().contains(needed) {
            return None;
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let made = Self::make(device);
        let error = pollster::block_on(scope.pop());
        match (made, error) {
            (made, None) => {
                log::info!("Horizon: sub-pixel far tiles rasterized in compute ({ENV})");
                Some(made)
            }
            (_, Some(error)) => {
                log::warn!("Horizon: the far tiles' compute raster failed, left out: {error}");
                None
            }
        }
    }

    fn make(device: &wgpu::Device) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Far raster"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let merge_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Far raster merge"),
            source: wgpu::ShaderSource::Wgsl(include_str!("far_raster_merge.wgsl").into()),
        });
        let main_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Far raster"),
            entries: &[
                layout_entry(0, c, uniform(false)),
                layout_entry(1, c, uniform(false)),
                layout_entry(2, c, storage(true)),
                layout_entry(3, c, storage(true)),
                layout_entry(4, c, storage(false)),
                layout_entry(5, c, storage(true)),
                layout_entry(6, c, storage(true)),
                layout_entry(7, c, storage(false)),
                layout_entry(8, c, storage(false)),
                layout_entry(9, c, storage(false)),
                layout_entry(10, c, storage(false)),
                layout_entry(11, c, storage(false)),
                layout_entry(12, c, storage(false)),
                layout_entry(13, c, storage(false)),
                layout_entry(14, c, uniform(false)),
            ],
        });
        let page_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Far raster page"),
            entries: &[
                layout_entry(super::flat::ARENA_BINDING, c, storage(true)),
                layout_entry(40, c, uniform(true)),
            ],
        });
        let dispatch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Far raster dispatch"),
            entries: &[layout_entry(0, c, storage(false))],
        });
        let tiles_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Far raster tiles"),
            entries: &[
                layout_entry(0, c, storage(false)),
                layout_entry(1, c, storage(false)),
                layout_entry(2, c, storage(false)),
            ],
        });
        let v = wgpu::ShaderStages::VERTEX;
        let f = wgpu::ShaderStages::FRAGMENT;
        let merge_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Far raster merge"),
            entries: &[
                layout_entry(10, v | f, uniform(false)),
                layout_entry(11, f, storage(false)),
                layout_entry(12, v, storage(true)),
                layout_entry(13, f, storage(true)),
                layout_entry(14, f, storage(true)),
                layout_entry(15, f, storage(true)),
                layout_entry(16, f, storage(true)),
            ],
        });
        let pipeline_layout = |groups: &[&wgpu::BindGroupLayout]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Far raster"),
                bind_group_layouts: &groups.iter().map(|g| Some(*g)).collect::<Vec<_>>(),
                ..Default::default()
            })
        };
        let prep_layout = pipeline_layout(&[&main_layout, &page_layout, &dispatch_layout]);
        let raster_layout = pipeline_layout(&[&main_layout, &page_layout]);
        let tiles_pipeline_layout = pipeline_layout(&[&tiles_layout]);
        let compute = |layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let merge = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Far raster merge"),
            layout: Some(&pipeline_layout(&[&merge_layout])),
            vertex: wgpu::VertexState {
                module: &merge_module,
                entry_point: Some("vs_fr_merge"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &merge_module,
                entry_point: Some("fs_fr_merge"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: super::vis_terrain::IDS,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let s = wgpu::BufferUsages::STORAGE;
        let dst = wgpu::BufferUsages::COPY_DST;
        let passes = buffer(
            device,
            "Far raster passes",
            u64::from(PAGES) * 2 * PASS_STRIDE,
            wgpu::BufferUsages::UNIFORM | dst,
        );
        let dispatch = buffer(device, "Far raster dispatch", u64::from(PAGES) * 16, s | wgpu::BufferUsages::INDIRECT);
        let dispatch_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Far raster dispatch"),
            layout: &dispatch_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: dispatch.as_entire_binding(),
            }],
        });
        Self {
            prep: compute(&prep_layout, &module, "fr_prep"),
            small: compute(&raster_layout, &module, "fr_small"),
            big: compute(&raster_layout, &module, "fr_big_quads"),
            tiles_reset: compute(&tiles_pipeline_layout, &merge_module, "fr_tiles_reset"),
            tiles: compute(&tiles_pipeline_layout, &merge_module, "fr_tiles"),
            merge,
            main_layout,
            page_layout,
            tiles_layout,
            merge_layout,
            frame: buffer(device, "Far raster frame", 16, wgpu::BufferUsages::UNIFORM | dst),
            passes,
            page_infos: buffer(device, "Far raster pages", u64::from(PAGES) * 32, s | dst),
            meta: buffer(device, "Far raster counts", u64::from(PAGES) * 16, s | wgpu::BufferUsages::INDIRECT),
            dispatch,
            tile_args: buffer(device, "Far raster tile draw", 16, s | wgpu::BufferUsages::INDIRECT),
            dispatch_group,
            starts: None,
            heads: None,
            big_list: None,
            leftovers: None,
            targets: None,
            main_group: None,
            merge_group: None,
            tiles_group: None,
            page_groups: Vec::new(),
            pages: Vec::new(),
            pending: std::cell::Cell::new([false; 2]),
            heads_per_page: 0,
        }
    }

    /// Before the frame's passes: the buffers and groups for `inputs`, IDs `size` pixels,
    /// arena pages of `page_vertices` vertices (`arena(page)` their buffers). `false`: nothing
    /// to raster this frame.
    pub fn prepare<'b>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &Inputs<'_>,
        size: [u32; 2],
        page_vertices: u64,
        arena: impl Fn(u32) -> &'b wgpu::Buffer,
    ) -> bool {
        self.pages.clear();
        if inputs.pages.is_empty() || inputs.pages.iter().any(|&(p, _, _)| p >= PAGES) {
            return false;
        }
        let s = wgpu::BufferUsages::STORAGE;
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            let pixels = u64::from(size[0]) * u64::from(size[1]);
            let tiles = u64::from(size[0].div_ceil(MASK_TILE) * size[1].div_ceil(MASK_TILE));
            self.targets = Some(Buffers {
                vis: buffer(device, "Far raster plots", pixels * 8, s),
                mask: buffer(device, "Far raster tile marks", tiles * 4, s),
                tile_list: buffer(device, "Far raster tiles", tiles * 4, s),
                size,
            });
            self.pending.set([false; 2]);
        } else if self.pending.get() != [false; 2] {
            // A dispatch that never merged (a frame given up): its plots and marks go.
            let t = self.targets.as_ref().expect("made above");
            encoder.clear_buffer(&t.vis, 0, None);
            encoder.clear_buffer(&t.mask, 0, None);
            self.pending.set([false; 2]);
        }
        let ords = inputs.pages.len() as u64;
        let draws: u64 = inputs.pages.iter().map(|&(_, _, n)| u64::from(n)).sum();
        self.heads_per_page = page_vertices.div_ceil(HEAD_QUADS) + 1;
        ensure(device, &mut self.starts, "Far raster starts", (draws + ords + 1) * 4, s);
        ensure(device, &mut self.heads, "Far raster heads", ords * self.heads_per_page * 4, s);
        ensure(device, &mut self.big_list, "Far raster big quads", ords * u64::from(BIG_CAPACITY) * 4, s);
        ensure(
            device,
            &mut self.leftovers,
            "Far raster leftover draws",
            ords * u64::from(LEFTOVER_CAPACITY) * 20,
            s | wgpu::BufferUsages::INDIRECT,
        );
        let mut infos = [PageInfo::default(); PAGES as usize];
        // Each page's prefix of compute quads: a word a draw and one past its last.
        let mut starts_at = 0u32;
        for (ord, &(page, first, n)) in inputs.pages.iter().enumerate() {
            let ord = ord as u32;
            infos[page as usize] = PageInfo {
                region: [first, n, starts_at, ord * self.heads_per_page as u32],
                lists: [ord * BIG_CAPACITY, ord * LEFTOVER_CAPACITY, 0, 0],
            };
            self.pages.push((page, ord * LEFTOVER_CAPACITY));
            starts_at += n + 1;
        }
        queue.write_buffer(&self.page_infos, 0, bytemuck::cast_slice(&infos));
        queue.write_buffer(
            &self.frame,
            0,
            bytemuck::cast_slice(&[size[0], size[1], BIG_CAPACITY, LEFTOVER_CAPACITY]),
        );
        let mut passes = vec![0u32; (u64::from(PAGES) * 2 * PASS_STRIDE / 4) as usize];
        for &(page, _, _) in inputs.pages {
            for phase in 0..2u32 {
                let at = ((page * 2 + phase) as u64 * PASS_STRIDE / 4) as usize;
                passes[at] = page;
                passes[at + 1] = phase;
            }
        }
        queue.write_buffer(&self.passes, 0, bytemuck::cast_slice(&passes));
        let t = self.targets.as_ref().expect("made above");
        let (starts, heads) = (self.starts.as_ref().unwrap(), self.heads.as_ref().unwrap());
        let (big_list, leftovers) = (self.big_list.as_ref().unwrap(), self.leftovers.as_ref().unwrap());
        group_of(
            device,
            &mut self.main_group,
            "Far raster",
            &self.main_layout,
            &[
                (0, inputs.params),
                (1, &self.frame),
                (2, inputs.instances),
                (3, inputs.records),
                (4, inputs.draws),
                (5, inputs.counts),
                (6, &self.page_infos),
                (7, starts),
                (8, heads),
                (9, &self.meta),
                (10, big_list),
                (11, &t.vis),
                (12, &t.mask),
                (13, leftovers),
                (14, inputs.globals),
            ],
        );
        group_of(
            device,
            &mut self.merge_group,
            "Far raster merge",
            &self.merge_layout,
            &[
                (10, &self.frame),
                (11, &t.vis),
                (12, &t.tile_list),
                (13, starts),
                (14, heads),
                (15, inputs.draws),
                (16, &self.page_infos),
            ],
        );
        group_of(
            device,
            &mut self.tiles_group,
            "Far raster tiles",
            &self.tiles_layout,
            &[(0, &t.mask), (1, &t.tile_list), (2, &self.tile_args)],
        );
        for &(page, _, _) in inputs.pages {
            let arena = arena(page);
            if self.page_groups.iter().any(|(p, b, _)| *p == page && b == arena) {
                continue;
            }
            self.page_groups.retain(|(p, _, _)| *p != page);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Far raster page"),
                layout: &self.page_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: super::flat::ARENA_BINDING,
                        resource: arena.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 40,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.passes,
                            offset: 0,
                            size: wgpu::BufferSize::new(16),
                        }),
                    },
                ],
            });
            self.page_groups.push((page, arena.clone(), group));
        }
        true
    }

    fn page_group(&self, page: u32) -> Option<&wgpu::BindGroup> {
        self.page_groups.iter().find(|(p, _, _)| *p == page).map(|(_, _, g)| g)
    }

    /// Phase `phase`'s compute passes (0: after far_cull's lists, 1: after its test), before
    /// that phase's ID pass, which must call [`Self::ids`].
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        phase: u32,
        mut timing: Option<&mut super::pack_timing::PassTiming>,
    ) {
        let (Some((_, main)), Some((_, tiles_group)), Some(t)) =
            (&self.main_group, &self.tiles_group, &self.targets)
        else {
            return;
        };
        let offset = |page: u32| ((page * 2 + phase) as u64 * PASS_STRIDE) as u32;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Horizon far raster lists"),
                timestamp_writes: timing.as_deref_mut().and_then(|t| t.compute_writes("Far raster lists")),
            });
            pass.set_pipeline(&self.prep);
            pass.set_bind_group(0, main, &[]);
            pass.set_bind_group(2, &self.dispatch_group, &[]);
            for &(page, _) in &self.pages {
                if let Some(group) = self.page_group(page) {
                    pass.set_bind_group(1, group, &[offset(page)]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Horizon far raster"),
            timestamp_writes: timing.as_deref_mut().and_then(|t| t.compute_writes("Far raster")),
        });
        pass.set_bind_group(0, main, &[]);
        pass.set_pipeline(&self.small);
        for &(page, _) in &self.pages {
            if let Some(group) = self.page_group(page) {
                pass.set_bind_group(1, group, &[offset(page)]);
                pass.dispatch_workgroups_indirect(&self.dispatch, u64::from(page) * 16);
            }
        }
        pass.set_pipeline(&self.big);
        for &(page, _) in &self.pages {
            if let Some(group) = self.page_group(page) {
                pass.set_bind_group(1, group, &[offset(page)]);
                pass.dispatch_workgroups(BIG_GROUPS, 1, 1);
            }
        }
        let tiles = t.size[0].div_ceil(MASK_TILE) * t.size[1].div_ceil(MASK_TILE);
        pass.set_bind_group(0, tiles_group, &[]);
        pass.set_pipeline(&self.tiles_reset);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.tiles);
        pass.dispatch_workgroups(tiles.div_ceil(64), 1, 1);
        let mut pending = self.pending.get();
        pending[phase as usize & 1] = true;
        self.pending.set(pending);
    }

    /// In phase `phase`'s ID pass, after far_cull's draws, with the far tiles' ID pipeline
    /// set and the quad indices bound: the alpha-tested quads the dispatch left over (each
    /// page with `group(page)`, the instances at vertex slot 1), then the merge. Leaves its own
    /// pipeline and group 0 set (the caller sets its own again).
    pub fn ids<'a>(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        phase: u32,
        group: impl Fn(u32) -> Option<&'a wgpu::BindGroup>,
        instances: &wgpu::Buffer,
    ) {
        let mut pending = self.pending.get();
        if !std::mem::take(&mut pending[phase as usize & 1]) {
            return;
        }
        self.pending.set(pending);
        let (Some((_, merge)), Some(leftovers)) = (&self.merge_group, &self.leftovers) else {
            return;
        };
        pass.set_vertex_buffer(1, instances.slice(..));
        for &(page, first) in &self.pages {
            if let Some(group) = group(page) {
                pass.set_bind_group(1, group, &[]);
                pass.multi_draw_indexed_indirect_count(
                    leftovers,
                    u64::from(first) * 20,
                    &self.meta,
                    (u64::from(page) * 4 + 2) * 4,
                    LEFTOVER_CAPACITY,
                );
            }
        }
        pass.set_pipeline(&self.merge);
        pass.set_bind_group(0, merge, &[]);
        pass.draw_indirect(&self.tile_args, 0);
    }
}

#[cfg(test)]
#[path = "far_raster_tests.rs"]
mod tests;

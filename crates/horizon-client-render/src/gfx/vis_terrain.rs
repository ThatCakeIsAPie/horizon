//! The terrain visibility buffer on the frames the renderer draws per section (Vulkan here,
//! any backend outside flat frames): solid terrain's draws go into an ID pass instead of the
//! world pass, with the same draw calls, culling phases and Hi-Z, through ID pipelines that run
//! each solid family's own vertex function and write only depth and a 64-bit quad ID
//! (`vis_terrain.wgsl`; layout in [`id`]); cutout, fading and fading-in quads keep their alpha
//! test and dithers there. Then one full-screen resolve draw per family (and arena page) shades
//! each pixel once: it rebuilds the pixel's triangle with the family's own corner function,
//! interpolates perspective-correctly, works out the UV's screen gradients and runs
//! terrain.wgsl's solid shading. Translucent terrain and everything after it stay forward, in a
//! world pass that loads the colour and the depth.
//!
//! Families: near sections and CPU-culled far tiles (`id::SECTION`, `Renderer::instances`),
//! far_cull's far tiles (`id::FAR_TILE`), cube_far's GPU-meshed nodes (`id::CUBE`), surface
//! tiles (`id::SURFACE`); `id::RASTER` is left for part C's compute raster, which merges its
//! IDs and depth into [`VisTerrain::ids_view`] and the world depth before the resolve. A family
//! without its pipelines (a device that rejects them) draws forward after the resolve.
//!
//! The flat frames' visibility buffer (`visbuffer.rs`, `HORIZON_VISBUFFER=1`) stays as it was;
//! [`frame_uses`] decides per frame, `HORIZON_VISBUFFER=0` turns this path off, and
//! `HORIZON_VISBUFFER_TOGGLE=<file>` flips it within a session ([`Toggle`]).
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::pack_timing::{self, CountedPass, PassTiming};

/// The IDs' format: two words a pixel ([`id`]).
pub const IDS: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
/// How often the toggle file is read again.
pub const TOGGLE_INTERVAL: Duration = Duration::from_secs(1);
/// The env knob turning the path off (`0`); `1` also asks for the flat frames' visbuffer.
pub const ENV: &str = "HORIZON_VISBUFFER";
/// The env knob naming the toggle file ([`Toggle`]).
pub const TOGGLE_ENV: &str = "HORIZON_VISBUFFER_TOGGLE";
/// The arena families' group 1 binding holding the draw's family and page (`vis_page`).
pub const PAGE_TAG_BINDING: u32 = 39;

/// The shared WGSL (the triangle math, ID wire, ID fragment stage, resolve helpers), after a
/// family's module.
pub const COMMON: &str = concat!(
    include_str!("vis_terrain_math.wgsl"),
    include_str!("vis_terrain.wgsl")
);
/// The arena families' entry points, after [`COMMON`] on terrain.wgsl.
pub const SECTION: &str = include_str!("vis_terrain_section.wgsl");
/// cube_far's, after [`COMMON`] on `cube_far::shader_source`.
pub const CUBE: &str = include_str!("vis_terrain_cube.wgsl");
/// The surface tiles', after [`COMMON`] on `surface_tile::draw_shader_source`.
pub const SURFACE: &str = include_str!("vis_terrain_surface.wgsl");

/// The ID bit layout, as `vis_terrain.wgsl` packs it (its `VIS_*` constants; a test holds the
/// two together). Two words `[x, y]`; `[0, 0]` is no terrain.
///
/// - y bits 31..29: the family ([`SECTION`](id::SECTION) 1 .. [`RASTER`](id::RASTER) 5; 6, 7
///   spare).
/// - Arena families (1, 2): y bits 28..23 the arena page, 22..0 the draw's instance; x the
///   vertex index of the quad's corner 0 (a plain vertex, `PACKED_BASE + 4 slot` or
///   `FAR_BASE + 4 slot`): corner c is `x + c`. Corner 0 is the provoking vertex of both of
///   the quad's triangles, so the flat ID carries it; the resolve tells the triangles apart
///   by the diagonal.
/// - Node families (3, 4): y bits 28..0 the node or tile slot; x the triangle's provoking
///   vertex, `6 quad` (corners 0 1 2) or `6 quad + 3` (corners 0 2 3): its vertices are
///   `x, x + 1, x + 2`.
/// - Family 5 (part C): y bits 28..0 and x are C's to define.
#[allow(dead_code)] // The whole layout: part C's raster and the tests read the rest.
pub mod id {
    pub const FAMILY_SHIFT: u32 = 29;
    pub const SECTION: u32 = 1;
    pub const FAR_TILE: u32 = 2;
    pub const CUBE: u32 = 3;
    pub const SURFACE: u32 = 4;
    pub const RASTER: u32 = 5;
    pub const PAGE_SHIFT: u32 = 23;
    pub const INSTANCE_MASK: u32 = (1 << PAGE_SHIFT) - 1;
    pub const LOW_MASK: u32 = (1 << FAMILY_SHIFT) - 1;
    pub const TAG_MASK: u32 = !INSTANCE_MASK;
    /// Arena pages an ID names.
    pub const MAX_PAGES: u32 = 1 << (FAMILY_SHIFT - PAGE_SHIFT);
    /// Instances an arena ID names.
    pub const MAX_INSTANCES: u32 = INSTANCE_MASK + 1;

    /// What an ID names.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Quad {
        None,
        /// An arena quad: its family, page, draw instance and corner 0's vertex index.
        Arena {
            family: u32,
            page: u32,
            instance: u32,
            corner0: u32,
        },
        /// A node family's triangle: family, node (or tile slot), quad, and whether it is
        /// the quad's second triangle (corners 0 2 3).
        Node {
            family: u32,
            node: u32,
            quad: u32,
            second: bool,
        },
    }

    /// An arena family's page tag (`vis_page.x`, the ID's y without the instance).
    pub const fn tag(family: u32, page: u32) -> u32 {
        (family << FAMILY_SHIFT) | (page << PAGE_SHIFT)
    }

    /// `vs_vis_section`'s ID at the provoking vertex `corner0` (corner 0's index).
    pub fn arena(family: u32, page: u32, instance: u32, corner0: u32) -> [u32; 2] {
        debug_assert!(page < MAX_PAGES && instance < MAX_INSTANCES);
        [corner0, tag(family, page) | (instance & INSTANCE_MASK)]
    }

    /// `vs_vis_cube`'s and `vs_vis_surface`'s ID at a triangle's provoking vertex.
    pub fn node(family: u32, node: u32, provoking: u32) -> [u32; 2] {
        [provoking, (family << FAMILY_SHIFT) | (node & LOW_MASK)]
    }

    pub fn unpack([x, y]: [u32; 2]) -> Quad {
        let family = y >> FAMILY_SHIFT;
        match family {
            0 => Quad::None,
            SECTION | FAR_TILE => Quad::Arena {
                family,
                page: (y & LOW_MASK) >> PAGE_SHIFT,
                instance: y & INSTANCE_MASK,
                corner0: x,
            },
            _ => Quad::Node {
                family,
                node: y & LOW_MASK,
                quad: x / 6,
                second: x % 6 == 3,
            },
        }
    }
}

/// What a frame's choice of solid terrain path turns on.
#[derive(Clone, Copy, Debug, Default)]
pub struct Facts {
    /// The renderer made the path's pipelines (`HORIZON_VISBUFFER` is not `0`, the device
    /// took them).
    pub built: bool,
    /// The toggle file's choice, else on.
    pub wanted: bool,
    /// The frame draws flat or through mesh shaders (the flat visbuffer's frames).
    pub flat_frame: bool,
    /// Solid draws read their page themselves (`Flat::page_group`).
    pub pulled: bool,
    pub msaa: bool,
    pub title: bool,
    /// A shader pack (Iris/OptiFine or Horizon's) has terrain programs or draws terrain.
    pub pack_terrain: bool,
    /// One of Horizon's own packs gives a material stage, a surface target or a shadow pass.
    pub material: bool,
    pub surface: bool,
    pub shadow: bool,
    /// Terrain looks (`looks::LookGroups`) or an external executor take the terrain.
    pub looks: bool,
    pub executor: bool,
    /// GPU culling draws the sections (its own instances: drawn forward).
    pub gpu_culling: bool,
    /// The arena's pages: one past the highest.
    pub pages: u32,
}

/// Whether a frame's solid terrain goes through the visibility buffer: wherever its pipelines
/// exist and nothing else owns the solid terrain or its look.
pub fn frame_uses(f: &Facts) -> bool {
    f.built
        && f.wanted
        && f.pulled
        && !f.flat_frame
        && !f.msaa
        && !f.title
        && !f.pack_terrain
        && !f.material
        && !f.surface
        && !f.shadow
        && !f.looks
        && !f.executor
        && !f.gpu_culling
        && f.pages <= id::MAX_PAGES
}

/// `HORIZON_VISBUFFER` leaves the path on (anything but `0`).
pub fn env_allows() -> bool {
    std::env::var(ENV).as_deref() != Ok("0")
}

/// The dev knob for same-session A/B checks: `HORIZON_VISBUFFER_TOGGLE=<file>`, read again
/// about once a second ([`TOGGLE_INTERVAL`]). Its words: `0`/`1` (or `vis=0`/`vis=1`) the old
/// path or this one, `raster=0`/`raster=1` part C's compute raster. No file, or a word
/// missing, leaves that choice to its default.
pub struct Toggle {
    path: Option<PathBuf>,
    next: Option<Instant>,
    vis: Option<bool>,
    raster: Option<bool>,
}

impl Toggle {
    pub fn from_env() -> Self {
        Self {
            path: std::env::var_os(TOGGLE_ENV)
                .filter(|p| !p.is_empty())
                .map(PathBuf::from),
            next: None,
            vis: None,
            raster: None,
        }
    }

    /// Reads the file again once its interval has passed.
    pub fn poll(&mut self, now: Instant) {
        let Some(path) = &self.path else {
            return;
        };
        if self.next.is_some_and(|next| now < next) {
            return;
        }
        self.next = Some(now + TOGGLE_INTERVAL);
        let (vis, raster) = std::fs::read_to_string(path)
            .map(|text| parse_toggle(&text))
            .unwrap_or((None, None));
        if vis != self.vis || raster != self.raster {
            log::info!("Horizon: {TOGGLE_ENV}: vis {vis:?}, raster {raster:?}");
        }
        self.vis = vis;
        self.raster = raster;
    }

    /// The file's choice of the visibility buffer, if it makes one.
    pub fn vis(&self) -> Option<bool> {
        self.vis
    }

    /// The file's choice of part C's compute raster, if it makes one.
    #[allow(dead_code)] // Part C's.
    pub fn raster(&self) -> Option<bool> {
        self.raster
    }
}

/// A toggle file's (vis, raster) choices.
pub fn parse_toggle(text: &str) -> (Option<bool>, Option<bool>) {
    let flag = |v: &str| match v.trim() {
        "0" | "off" | "false" => Some(false),
        "1" | "on" | "true" => Some(true),
        _ => None,
    };
    let (mut vis, mut raster) = (None, None);
    for word in text.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
        match word.split_once('=') {
            Some(("vis", v)) => vis = flag(v).or(vis),
            Some(("raster", v)) => raster = flag(v).or(raster),
            Some(_) => {}
            None => vis = flag(word).or(vis),
        }
    }
    (vis, raster)
}

/// A family's module source: its own source, [`COMMON`], then its entry points.
pub fn source(family: &str, entries: &str) -> String {
    format!("{family}\n{COMMON}\n{entries}")
}

/// The resolve's group 2: the IDs and an instance buffer (`vis_terrain.wgsl`).
pub fn group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Terrain visibility resolve"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: fragment,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: fragment,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// One solid family's ID pipeline (the family's solid state, IDs instead of colour) and its
/// resolve pipelines (full screen, into the world's colour), without HDR output's lighting
/// and, where the display has HDR output, with it (as `SolidPipeline`).
pub struct FamilyPipelines {
    pub ids: wgpu::RenderPipeline,
    pub resolve: wgpu::RenderPipeline,
    pub resolve_hdr: Option<wgpu::RenderPipeline>,
}

/// What [`family_pipelines`] builds a family's pipelines from.
pub struct FamilyDesc<'a> {
    pub label: &'a str,
    /// The family's module, [`source`].
    pub source: String,
    /// terrain.wgsl's group 0, the family's group 1 (both visible to the fragment stage).
    pub terrain: &'a wgpu::BindGroupLayout,
    pub group1: &'a wgpu::BindGroupLayout,
    /// The family's solid vertex buffers (the origins at slot 1 for the arena families).
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub cull: Option<wgpu::Face>,
    pub id_entry: &'a str,
    pub resolve_entry: &'a str,
    /// The family's solid fragment constants without HDR output's lighting, and with it.
    pub constants: &'a [(&'a str, f64)],
    pub hdr_constants: Option<&'a [(&'a str, f64)]>,
    /// The world's colour format.
    pub format: wgpu::TextureFormat,
}

/// A family's pipelines, or `None` where the device rejects them (the family then draws
/// forward; `log`ged once).
pub fn family_pipelines(device: &wgpu::Device, d: &FamilyDesc<'_>) -> Option<FamilyPipelines> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(d.label),
        source: wgpu::ShaderSource::Wgsl(d.source.as_str().into()),
    });
    let group2 = group_layout(device);
    let id_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(d.label),
        bind_group_layouts: &[Some(d.terrain), Some(d.group1)],
        ..Default::default()
    });
    let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(d.label),
        bind_group_layouts: &[Some(d.terrain), Some(d.group1), Some(&group2)],
        ..Default::default()
    });
    let ids = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(d.label),
        layout: Some(&id_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some(d.id_entry),
            compilation_options: Default::default(),
            buffers: d.buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: d.cull,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_vis_terrain_id"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: d.constants,
                ..Default::default()
            },
            targets: &[Some(wgpu::ColorTargetState {
                format: IDS,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let resolve = |constants: &[(&str, f64)]| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(d.label),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_vis_terrain_resolve"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(d.resolve_entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                targets: &[Some(wgpu::ColorTargetState {
                    format: d.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let pipelines = FamilyPipelines {
        ids,
        resolve: resolve(d.constants),
        resolve_hdr: d.hdr_constants.map(resolve),
    };
    if let Some(error) = pollster::block_on(scope.pop()) {
        log::warn!(
            "Horizon renderer: {} not drawn through the visibility buffer ({error})",
            d.label
        );
        return None;
    }
    Some(pipelines)
}

impl FamilyPipelines {
    /// The resolve pipeline: with HDR output's lighting where `hdr` and made.
    pub fn resolve(&self, hdr: bool) -> &wgpu::RenderPipeline {
        match &self.resolve_hdr {
            Some(lit) if hdr => lit,
            _ => &self.resolve,
        }
    }

    /// A node family's resolve draw: group 1 its own, group 2 [`VisTerrain::group`]'s.
    pub fn resolve_draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        hdr: bool,
        group1: &wgpu::BindGroup,
        group2: &wgpu::BindGroup,
    ) {
        pass.set_pipeline(self.resolve(hdr));
        pass.set_bind_group(1, group1, &[]);
        pass.set_bind_group(2, group2, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// An arena page's group 1 for one arena family: the page at `flat::ARENA_BINDING` and its
/// tag at [`PAGE_TAG_BINDING`].
struct PageGroup {
    family: u32,
    page: u32,
    buffer: wgpu::Buffer,
    group: wgpu::BindGroup,
}

/// The renderer's part: the arena families' pipelines, their page groups, the IDs and the
/// resolve's groups.
pub struct VisTerrain {
    sections: FamilyPipelines,
    page_layout: wgpu::BindGroupLayout,
    group_layout: wgpu::BindGroupLayout,
    /// The page tags' uniforms by (family, page), made as pages come.
    tags: Vec<((u32, u32), wgpu::Buffer)>,
    pages: Vec<PageGroup>,
    ids: Option<(wgpu::TextureView, [u32; 2])>,
    ids_generation: u64,
    /// The resolve's group 2 per instance buffer, with the IDs of `ids_generation`.
    groups: Vec<(wgpu::Buffer, u64, wgpu::BindGroup)>,
    /// The size the IDs had no room at (#13624), not tried again until it changes.
    no_room: Option<[u32; 2]>,
}

/// What [`VisTerrain::new`] takes from the renderer's own pipelines.
pub struct SectionSetup<'a> {
    pub terrain: &'a wgpu::BindGroupLayout,
    /// The pulled solid pipelines' vertex buffers (the origins at slot 1).
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub format: wgpu::TextureFormat,
    /// The solid pipelines' fragment constants without HDR output's lighting, and with it
    /// where the display has HDR output (`SolidPipeline`'s two).
    pub plain_constants: &'a [(&'a str, f64)],
    pub hdr_constants: Option<&'a [(&'a str, f64)]>,
}

impl VisTerrain {
    pub fn new(device: &wgpu::Device, s: &SectionSetup<'_>) -> Option<Self> {
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let page_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain visibility pages"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: super::flat::ARENA_BINDING,
                    visibility: vf,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: PAGE_TAG_BINDING,
                    visibility: vf,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sections = family_pipelines(
            device,
            &FamilyDesc {
                label: "Terrain visibility sections",
                source: source(&super::terrain_shader(), SECTION),
                terrain: s.terrain,
                group1: &page_layout,
                buffers: s.buffers,
                cull: Some(wgpu::Face::Back),
                id_entry: "vs_vis_section",
                resolve_entry: "fs_vis_resolve_section",
                constants: s.plain_constants,
                hdr_constants: s.hdr_constants,
                format: s.format,
            },
        )?;
        log::info!("Horizon: solid terrain through the terrain visibility buffer ({ENV})");
        Some(Self {
            sections,
            page_layout,
            group_layout: group_layout(device),
            tags: Vec::new(),
            pages: Vec::new(),
            ids: None,
            ids_generation: 0,
            groups: Vec::new(),
            no_room: None,
        })
    }

    /// The arena families' ID pipeline (their draws bind [`Self::page_group`]s).
    pub fn section_ids(&self) -> &wgpu::RenderPipeline {
        &self.sections.ids
    }

    /// Makes both arena families' group of every page in `pages` (the arena's, with their
    /// buffers), and drops the groups of pages that went or changed buffer.
    pub fn bind_pages<'b>(
        &mut self,
        device: &wgpu::Device,
        pages: impl Iterator<Item = (u32, &'b wgpu::Buffer)>,
    ) {
        let pages: Vec<(u32, &wgpu::Buffer)> = pages.collect();
        self.pages
            .retain(|g| pages.iter().any(|&(p, b)| p == g.page && *b == g.buffer));
        for &(page, buffer) in &pages {
            for family in [id::SECTION, id::FAR_TILE] {
                if self
                    .pages
                    .iter()
                    .any(|g| g.family == family && g.page == page)
                {
                    continue;
                }
                let tag = match self.tags.iter().find(|(k, _)| *k == (family, page)) {
                    Some((_, b)) => b.clone(),
                    None => {
                        let b = wgpu::util::DeviceExt::create_buffer_init(
                            device,
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("Terrain visibility page tag"),
                                contents: bytemuck::cast_slice(&[
                                    id::tag(family, page),
                                    0u32,
                                    0,
                                    0,
                                ]),
                                usage: wgpu::BufferUsages::UNIFORM,
                            },
                        );
                        self.tags.push(((family, page), b.clone()));
                        b
                    }
                };
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Terrain visibility page"),
                    layout: &self.page_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: super::flat::ARENA_BINDING,
                            resource: buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: PAGE_TAG_BINDING,
                            resource: tag.as_entire_binding(),
                        },
                    ],
                });
                self.pages.push(PageGroup {
                    family,
                    page,
                    buffer: buffer.clone(),
                    group,
                });
            }
        }
    }

    /// An arena family's group 1 of `page` ([`Self::bind_pages`]).
    pub fn page_group(&self, family: u32, page: u32) -> Option<&wgpu::BindGroup> {
        self.pages
            .iter()
            .find(|g| g.family == family && g.page == page)
            .map(|g| &g.group)
    }

    /// Makes the IDs the world depth's `size`: `false` where there is no room for them
    /// (#13624), and the frame draws its solid terrain forward.
    pub fn fit(&mut self, device: &wgpu::Device, size: [u32; 2]) -> bool {
        use super::frame_targets::{Kind, made};
        if self.ids.as_ref().is_some_and(|(_, s)| *s == size) {
            return true;
        }
        if self.no_room == Some(size) {
            return false;
        }
        self.ids = None;
        self.groups.clear();
        self.ids_generation = self.ids_generation.wrapping_add(1);
        self.ids = made(device, Kind::Visibility, || {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Terrain visibility IDs"),
                size: wgpu::Extent3d {
                    width: size[0].max(1),
                    height: size[1].max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: IDS,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            (texture.create_view(&Default::default()), size)
        });
        self.no_room = self.ids.is_none().then_some(size);
        self.ids.is_some()
    }

    /// The IDs: part C writes its own into them between the ID passes and the resolve
    /// (STORAGE_BINDING usage; the world depth is the depth they hold).
    #[allow(dead_code)] // Part C's.
    pub fn ids_view(&self) -> Option<&wgpu::TextureView> {
        self.ids.as_ref().map(|(view, _)| view)
    }

    /// The ID pass over the world's `depth`: the IDs cleared (`clear`: phase 1) or kept.
    pub fn id_pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        clear: bool,
        mut timing: Option<&mut PassTiming>,
    ) -> CountedPass<'e> {
        let (ids, _) = self.ids.as_ref().expect("fitted before the world pass");
        let name = if clear {
            "Visibility IDs"
        } else {
            "Visibility IDs after culling"
        };
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Horizon terrain visibility IDs"),
            timestamp_writes: timing.as_deref_mut().and_then(|t| t.writes(name)),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ids,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: if clear {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pack_timing::counted(timing, pass, super::pixel_count(depth), "solid terrain ids")
    }

    /// The resolve's pass: the world's colour kept, no depth.
    pub fn resolve_pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        pixels: u64,
        mut timing: Option<&mut PassTiming>,
    ) -> CountedPass<'e> {
        let name = "Visibility resolve";
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Horizon terrain visibility resolve"),
            timestamp_writes: timing.as_deref_mut().and_then(|t| t.writes(name)),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pack_timing::counted(timing, pass, pixels, "resolve")
    }

    /// The resolve's group 2 with `instances` (kept per buffer while the IDs stay).
    pub fn group(&mut self, device: &wgpu::Device, instances: &wgpu::Buffer) -> wgpu::BindGroup {
        let generation = self.ids_generation;
        if let Some((_, _, group)) = self
            .groups
            .iter()
            .find(|(b, g, _)| b == instances && *g == generation)
        {
            return group.clone();
        }
        let (ids, _) = self.ids.as_ref().expect("fitted before the resolve");
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain visibility resolve"),
            layout: &self.group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ids),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instances.as_entire_binding(),
                },
            ],
        });
        // The instance buffers grow, and a few live at once: old ones' groups go.
        self.groups.retain(|(_, g, _)| *g == generation);
        if self.groups.len() >= 4 {
            self.groups.remove(0);
        }
        self.groups
            .push((instances.clone(), generation, group.clone()));
        group
    }

    /// The arena family `family`'s pixels of `pages` resolved; `group` is [`Self::group`]
    /// with the instances its draws read, `hdr` HDR output's lighting.
    pub fn resolve_arena(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        hdr: bool,
        family: u32,
        pages: impl Iterator<Item = u32>,
        group: &wgpu::BindGroup,
    ) {
        pass.set_pipeline(self.sections.resolve(hdr));
        pass.set_bind_group(2, group, &[]);
        for page in pages {
            if let Some(page_group) = self.page_group(family, page) {
                pass.set_bind_group(1, page_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }
}

/// The pages of `pages` once each, in order of first appearance.
pub fn distinct_pages(pages: impl Iterator<Item = u32>) -> Vec<u32> {
    let mut seen = Vec::new();
    for page in pages {
        if !seen.contains(&page) {
            seen.push(page);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str) {
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!("{}", e.emit_to_string(source));
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    }

    fn family_sources() -> [String; 3] {
        [
            source(&super::super::terrain_shader(), SECTION),
            source(&super::super::cube_far::shader_source(), CUBE),
            source(&super::super::surface_tile::draw_shader_source(), SURFACE),
        ]
    }

    #[test]
    fn vis_terrain_shaders_validate_for_every_family() {
        for source in family_sources() {
            validate(&source);
        }
    }

    #[test]
    fn vis_terrain_id_stage_fits_the_weakest_gpus() {
        for source in family_sources() {
            for i in super::super::inter_stage_tests::interfaces(&source) {
                assert!(i.total() <= super::super::inter_stage_tests::LIMIT, "{i:?}");
            }
        }
    }

    /// The WGSL's `VIS_*` constants are the Rust layout's.
    #[test]
    fn vis_terrain_wgsl_constants_match_the_rust_layout() {
        let value = |name: &str| -> u32 {
            let at = COMMON
                .find(&format!("const {name}: u32 = "))
                .unwrap_or_else(|| panic!("{name}"));
            let rest = &COMMON[at..];
            let text = rest[rest.find('=').unwrap() + 1..rest.find("u;").unwrap()].trim();
            match text.strip_prefix("0x") {
                Some(hex) => u32::from_str_radix(hex, 16).unwrap(),
                None => text.parse().unwrap(),
            }
        };
        assert_eq!(value("VIS_FAMILY_SHIFT"), id::FAMILY_SHIFT);
        assert_eq!(value("VIS_SECTION"), id::SECTION);
        assert_eq!(value("VIS_FAR_TILE"), id::FAR_TILE);
        assert_eq!(value("VIS_CUBE"), id::CUBE);
        assert_eq!(value("VIS_SURFACE"), id::SURFACE);
        assert_eq!(value("VIS_RASTER"), id::RASTER);
        assert_eq!(value("VIS_PAGE_SHIFT"), id::PAGE_SHIFT);
        assert_eq!(value("VIS_INSTANCE_MASK"), id::INSTANCE_MASK);
        assert_eq!(value("VIS_LOW_MASK"), id::LOW_MASK);
        assert_eq!(value("VIS_TAG_MASK"), id::TAG_MASK);
    }

    #[test]
    fn vis_terrain_ids_round_trip_per_family_at_their_limits() {
        // terrain.wgsl's vertex bases: plain, packed records and far records.
        const PACKED_BASE: u32 = 0x1000_0000;
        const FAR_BASE: u32 = 0x2000_0000;
        // The most slots a 256 MB page holds (24 bytes a slot).
        let max_slot = (256u32 << 20) / 24;
        for family in [id::SECTION, id::FAR_TILE] {
            for page in [0, 1, id::MAX_PAGES - 1] {
                for instance in [0, 12345, id::MAX_INSTANCES - 1] {
                    for corner0 in [
                        0,
                        6,
                        4 * 777 + 2,
                        PACKED_BASE + 4 * max_slot,
                        FAR_BASE + 4 * max_slot,
                    ] {
                        let packed = id::arena(family, page, instance, corner0);
                        assert_ne!(packed, [0, 0]);
                        assert_eq!(packed[1] & id::TAG_MASK, id::tag(family, page));
                        assert_eq!(
                            id::unpack(packed),
                            id::Quad::Arena {
                                family,
                                page,
                                instance,
                                corner0
                            }
                        );
                    }
                }
            }
        }
        for family in [id::CUBE, id::SURFACE] {
            for node in [0, 99, id::LOW_MASK] {
                for quad in [0, 5, u32::MAX / 6 - 1] {
                    for (k, second) in [(0, false), (3, true)] {
                        let packed = id::node(family, node, quad * 6 + k);
                        assert_ne!(packed, [0, 0]);
                        assert_eq!(
                            id::unpack(packed),
                            id::Quad::Node {
                                family,
                                node,
                                quad,
                                second
                            }
                        );
                    }
                }
            }
        }
        assert_eq!(id::unpack([0, 0]), id::Quad::None);
        // Tags of different families or pages never meet, and every page an ID names fits.
        assert_ne!(id::tag(id::SECTION, 5), id::tag(id::FAR_TILE, 5));
        assert_ne!(id::tag(id::SECTION, 5), id::tag(id::SECTION, 6));
        assert!(id::MAX_PAGES >= super::super::visbuffer::MAX_PAGES);
    }

    #[test]
    fn vis_terrain_path_choice() {
        let on = Facts {
            built: true,
            wanted: true,
            pulled: true,
            pages: 3,
            ..Default::default()
        };
        assert!(frame_uses(&on));
        let off = |f: Facts| assert!(!frame_uses(&f), "{f:?}");
        off(Facts {
            pack_terrain: true,
            ..on
        });
        off(Facts {
            material: true,
            ..on
        });
        off(Facts {
            surface: true,
            ..on
        });
        off(Facts { shadow: true, ..on });
        off(Facts { msaa: true, ..on });
        off(Facts { title: true, ..on });
        off(Facts {
            flat_frame: true,
            ..on
        });
        off(Facts { looks: true, ..on });
        off(Facts {
            executor: true,
            ..on
        });
        off(Facts {
            gpu_culling: true,
            ..on
        });
        off(Facts { built: false, ..on });
        off(Facts {
            wanted: false,
            ..on
        });
        off(Facts {
            pulled: false,
            ..on
        });
        off(Facts {
            pages: id::MAX_PAGES + 1,
            ..on
        });
        assert!(frame_uses(&Facts {
            pages: id::MAX_PAGES,
            ..on
        }));
    }

    #[test]
    fn vis_terrain_toggle_file_words() {
        assert_eq!(parse_toggle("1\n"), (Some(true), None));
        assert_eq!(parse_toggle("0"), (Some(false), None));
        assert_eq!(parse_toggle("vis=1 raster=0"), (Some(true), Some(false)));
        assert_eq!(parse_toggle("raster=1"), (None, Some(true)));
        assert_eq!(parse_toggle("1,raster=1"), (Some(true), Some(true)));
        assert_eq!(parse_toggle(""), (None, None));
        assert_eq!(parse_toggle("junk"), (None, None));
    }

    #[test]
    fn vis_terrain_distinct_pages_keep_their_order() {
        assert_eq!(
            distinct_pages([2, 2, 0, 2, 1, 0].into_iter()),
            vec![2, 0, 1]
        );
    }
}

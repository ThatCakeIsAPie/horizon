//! far_raster.rs: the switch rule, and the compute raster against the hardware raster on a
//! test field of far tiles (GPU readback on the real device; run filtered, one at a time).
use super::*;
use crate::gfx::test_gpu;
use glam::{Mat4, Vec3};

#[test]
fn far_raster_switch_rule_by_on_screen_size() {
    // A voxel of 1 block at 177 blocks with a focal of 221.7 px spans 1.2525 px.
    let f = 128.0 / 30f32.to_radians().tan();
    assert!(raster_small(1.0, 178.0, f));
    assert!(!raster_small(1.0, 176.0, f));
    assert!(raster_small(16.0, 16.0 * 178.0, f));
    assert!(!raster_small(16.0, 16.0 * 176.0, f));
    // No voxel size (records without one): hardware.
    assert!(!raster_small(0.0, 1e9, f));
    // At the threshold exactly: compute.
    assert!(raster_small(1.25, 1.0, 1.0));
    // The shader's constant is this one.
    let wgsl = include_str!("far_raster.wgsl");
    assert!(wgsl.contains(&format!("const FAR_RASTER_PX: f32 = {FAR_RASTER_PX:?};")));
    // The focal length read off a view projection is the projection's y scale times half the
    // height, whatever the view's rotation.
    let proj = Mat4::perspective_infinite_reverse_rh(60f32.to_radians(), 1.0, 0.05);
    let view = Mat4::look_at_rh(Vec3::new(3.0, 40.0, 1.0), Vec3::new(-20.0, 5.0, -90.0), Vec3::Y);
    assert!((focal(proj * view, 256) - f).abs() < 1e-3 * f);
}

#[test]
fn far_raster_shader_takes_terrain_items() {
    let source = shader_source();
    for head in TERRAIN_ITEMS {
        assert_eq!(source.matches(head).count(), 1, "{head}");
    }
    assert!(source.contains("fn fr_prep("));
    let module = naga::front::wgsl::parse_str(&source).expect("far_raster.wgsl parses");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::SHADER_INT64 | naga::valid::Capabilities::SHADER_INT64_ATOMIC_MIN_MAX,
    );
    validator.validate(&module).expect("far_raster.wgsl validates");
    let merge = naga::front::wgsl::parse_str(include_str!("far_raster_merge.wgsl")).expect("merge parses");
    validator.validate(&merge).expect("merge validates");
}

const SIZE: u32 = 256;

/// The test hardware raster: the far tiles' corners as `far_raster.wgsl` reads them, the ID
/// the visibility buffer's far tiles get (corner 0 provokes both triangles).
const HW: &str = r#"
struct Tp { view_proj: mat4x4<f32>, }
@group(0) @binding(0) var<uniform> tp: Tp;
@group(0) @binding(1) var<storage, read> tinst: array<vec4<f32>>;
struct V {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) id: vec2<u32>,
}
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> V {
    let c = arena_vertex(v);
    let o = tinst[i];
    return V(fade_depth(tp.view_proj * vec4(bitcast<vec3<f32>>(c.p) + o.xyz, 1.0), o.w),
        vec2(v, 0x40000000u | i));
}
@fragment fn fs(x: V) -> @location(0) vec2<u32> {
    return x.id;
}
"#;

struct Field {
    vertices: Vec<u32>,
    records: Vec<u32>,
    instances: Vec<[f32; 4]>,
    draws: Vec<[u32; 5]>,
    max_quads: u32,
}

/// Tiles of 16 x 16 blocks of a stepped heightfield (tops and the sides between columns, each
/// quad in both windings: back faces are culled), receding from 40 to 420 blocks; a twentieth
/// of the quads alpha-tested (no OPAQUE_LAYER).
fn field(cam: Vec3) -> Field {
    let height = |x: i32, z: i32| -> i32 {
        let (x, z) = (x as f32, z as f32);
        ((x * 0.21).sin() * 4.0 + (z * 0.13).cos() * 5.0 + ((x + z) * 0.05).sin() * 9.0).floor() as i32
    };
    let mut f = Field {
        vertices: Vec::new(),
        records: Vec::new(),
        instances: Vec::new(),
        draws: Vec::new(),
        max_quads: 0,
    };
    let mut n_quad = 0u32;
    for tz in 0..24 {
        for tx in -2..2 {
            let origin = [tx * 16, 0, -40 - tz * 16];
            let base = (f.vertices.len() / 6) as u32;
            let mut quads = 0u32;
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            let mut quad = |f: &mut Field, p: [Vec3; 4]| {
                for order in [[0, 1, 2, 3], [0, 3, 2, 1]] {
                    let layer = if n_quad % 20 == 7 { 3u32 } else { 3 | 0x8000 };
                    n_quad += 1;
                    for &k in &order {
                        let v = p[k];
                        lo = lo.min(v);
                        hi = hi.max(v);
                        f.vertices.extend([v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), 0, layer, 0xffff_ffff]);
                    }
                    quads += 1;
                }
            };
            for z in 0..16 {
                for x in 0..16 {
                    let (wx, wz) = (origin[0] + x, origin[2] + z);
                    let h = height(wx, wz) as f32;
                    let (x0, z0) = (x as f32, z as f32);
                    quad(&mut f, [
                        Vec3::new(x0, h, z0),
                        Vec3::new(x0 + 1.0, h, z0),
                        Vec3::new(x0 + 1.0, h, z0 + 1.0),
                        Vec3::new(x0, h, z0 + 1.0),
                    ]);
                    // The sides down to the +X and +Z neighbours.
                    let hx = height(wx + 1, wz) as f32;
                    if hx != h {
                        quad(&mut f, [
                            Vec3::new(x0 + 1.0, h, z0),
                            Vec3::new(x0 + 1.0, hx, z0),
                            Vec3::new(x0 + 1.0, hx, z0 + 1.0),
                            Vec3::new(x0 + 1.0, h, z0 + 1.0),
                        ]);
                    }
                    let hz = height(wx, wz + 1) as f32;
                    if hz != h {
                        quad(&mut f, [
                            Vec3::new(x0, h, z0 + 1.0),
                            Vec3::new(x0 + 1.0, h, z0 + 1.0),
                            Vec3::new(x0 + 1.0, hz, z0 + 1.0),
                            Vec3::new(x0, hz, z0 + 1.0),
                        ]);
                    }
                }
            }
            let id = f.draws.len() as u32;
            let rel = Vec3::new(origin[0] as f32, origin[1] as f32, origin[2] as f32) - cam;
            // A whole tile's instance: the UV step, no fade.
            f.instances.push([rel.x, rel.y, rel.z, 1.0 / 32.0]);
            let mut rec = [0u32; 28];
            rec[..4].copy_from_slice(&origin.map(|v| v as u32).into_iter().chain([0]).collect::<Vec<_>>());
            rec[4..8].copy_from_slice(&[base, quads, 1, 1]);
            rec[8..11].copy_from_slice(&lo.to_array().map(f32::to_bits));
            rec[12..15].copy_from_slice(&hi.to_array().map(f32::to_bits));
            f.records.extend(rec);
            f.draws.push([quads * 6, 1, 0, base, id]);
            f.max_quads = f.max_quads.max(quads);
        }
    }
    f
}

fn init(device: &wgpu::Device, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    wgpu::util::DeviceExt::create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        },
    )
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, aspect: wgpu::TextureAspect) -> Vec<u32> {
    let bpp = if aspect == wgpu::TextureAspect::DepthOnly { 4 } else { 8 };
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(SIZE * SIZE * bpp),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &out,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * bpp),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    out.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let words = bytemuck::cast_slice(&out.slice(..).get_mapped_range().expect("mapped")).to_vec();
    words
}

#[test]
fn far_raster_matches_hardware_raster() {
    let _gpu = test_gpu::lock();
    let Some(adapter) = test_gpu::adapter() else { return };
    let needed = wgpu::Features::SHADER_INT64
        | wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX
        | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
        | wgpu::Features::INDIRECT_FIRST_INSTANCE;
    if !adapter.features().contains(needed) {
        eprintln!("skipped: the adapter ({:?}) lacks {:?}", adapter.get_info(), needed - adapter.features());
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: needed,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("GPU test device creation failed");
    let raster = FarRaster::new(&device).expect("the compute raster's pipelines");

    let cam = Vec3::new(0.5, 30.0, 0.5);
    let proj = Mat4::perspective_infinite_reverse_rh(60f32.to_radians(), 1.0, 0.05);
    let view_proj = proj * Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.02, -0.12, -1.0), Vec3::Y);
    let f = field(cam);
    let s = wgpu::BufferUsages::STORAGE;
    let arena = init(&device, "arena", bytemuck::cast_slice(&f.vertices), s);
    let records = init(&device, "records", bytemuck::cast_slice(&f.records), s);
    let instances = init(&device, "instances", bytemuck::cast_slice(&f.instances), s | wgpu::BufferUsages::VERTEX);
    let draws_words: Vec<u32> = f.draws.iter().flatten().copied().collect();
    let draws = init(
        &device,
        "draws",
        bytemuck::cast_slice(&draws_words),
        s | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
    );
    let n = f.draws.len() as u32;
    let counts = init(&device, "counts", bytemuck::cast_slice(&[0, 0, 0, 0, 0, 0, 0, 0, n, n, 0, 0]), s);
    let mut params = vec![0u32; 56];
    params[2] = 1;
    params[40..56].copy_from_slice(&view_proj.to_cols_array().map(f32::to_bits));
    let params = init(&device, "params", bytemuck::cast_slice(&params), wgpu::BufferUsages::UNIFORM);
    let indices: Vec<u32> = (0..f.max_quads).flat_map(|q| [0, 1, 2, 0, 2, 3].map(|i| 4 * q + i)).collect();
    let indices = init(&device, "indices", bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX);
    let tp = init(
        &device,
        "tp",
        bytemuck::cast_slice(&view_proj.to_cols_array()),
        wgpu::BufferUsages::UNIFORM,
    );

    // The hardware raster pipeline.
    let terrain = super::super::terrain_shader();
    let items: String = TERRAIN_ITEMS.iter().map(|h| wgsl_item(&terrain, h).unwrap().to_owned() + "\n").collect();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("far raster test hw"),
        source: wgpu::ShaderSource::Wgsl((super::FADE_PRELUDE.to_owned() + &items + HW).into()),
    });
    let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[layout_entry(0, vf, uniform(false)), layout_entry(1, vf, storage(true))],
    });
    let g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[layout_entry(super::super::flat::ARENA_BINDING, vf, storage(true))],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&g0), Some(&g1)],
        ..Default::default()
    });
    let hw = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("far raster test hw"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
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
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(super::super::vis_terrain::IDS.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g0,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: instances.as_entire_binding() },
        ],
    });
    let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g1,
        entries: &[wgpu::BindGroupEntry {
            binding: super::super::flat::ARENA_BINDING,
            resource: arena.as_entire_binding(),
        }],
    });
    let target = |format: wgpu::TextureFormat| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };

    // One frame: hardware only (`compute` false), or the compute raster's switch.
    let frame = |raster: Option<&mut FarRaster>| -> (Vec<u32>, Vec<u32>) {
        let ids = target(super::super::vis_terrain::IDS);
        let depth = target(wgpu::TextureFormat::Depth32Float);
        let mut encoder = device.create_command_encoder(&Default::default());
        let raster = raster.map(|r| {
            let inputs = Inputs {
                globals: &tp,
                params: &params,
                records: &records,
                instances: &instances,
                draws: &draws,
                counts: &counts,
                pages: &[(0, 0, n)],
            };
            assert!(r.prepare(&device, &queue, &mut encoder, &inputs, [SIZE, SIZE], 1 << 20, |_| &arena));
            r.dispatch(&mut encoder, 0, None);
            &*r
        });
        {
            let ids_view = ids.create_view(&Default::default());
            let depth_view = depth.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &ids_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&hw);
            pass.set_bind_group(0, &group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.multi_draw_indexed_indirect_count(&draws, 0, &counts, 8 * 4, n);
            if let Some(r) = raster {
                r.ids(&mut pass, 0, |_| Some(&group1), &instances);
            }
        }
        queue.submit([encoder.finish()]);
        (
            read(&device, &queue, &ids, wgpu::TextureAspect::All),
            read(&device, &queue, &depth, wgpu::TextureAspect::DepthOnly),
        )
    };
    let (hw_ids, hw_depth) = frame(None);
    let mut raster = raster;
    let (c_ids, c_depth) = frame(Some(&mut raster));
    // The draws the switch took.
    let taken = {
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: draws.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&draws, 0, &out, 0, draws.size());
        queue.submit([encoder.finish()]);
        out.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let words: Vec<u32> = bytemuck::cast_slice(&out.slice(..).get_mapped_range().expect("mapped")).to_vec();
        words.chunks(5).filter(|d| d[1] == 0).count()
    };
    let (mut covered, mut holes, mut extra, mut same, mut both) = (0, 0, 0, 0, 0);
    let mut ulps = Vec::new();
    for p in 0..(SIZE * SIZE) as usize {
        let h = [hw_ids[2 * p], hw_ids[2 * p + 1]];
        let c = [c_ids[2 * p], c_ids[2 * p + 1]];
        covered += usize::from(h[1] != 0);
        match (h[1] != 0, c[1] != 0) {
            (true, false) => holes += 1,
            (false, true) => extra += 1,
            (true, true) => {
                both += 1;
                same += usize::from(h == c);
                ulps.push((i64::from(hw_depth[p]) - i64::from(c_depth[p])).unsigned_abs());
            }
            _ => {}
        }
    }
    ulps.sort_unstable();
    let p99 = ulps.get(ulps.len() * 99 / 100).copied().unwrap_or(0);
    let max = ulps.last().copied().unwrap_or(0);
    eprintln!(
        "far raster vs hardware: {taken} of {n} tiles in compute, {covered} px covered, \
         {holes} holes, {extra} extra, {same} of {both} same quad, depth ulps p99 {p99} max {max}"
    );
    assert!(taken > 0 && taken < n as usize, "the field crosses the switch");
    assert!(covered > (SIZE * SIZE / 4) as usize, "the field covers the view");
    assert!(holes + extra <= covered / 10_000, "coverage");
    assert!(same * 1000 >= both * 999, "same quads");
    assert!(p99 <= 2 && max <= 64, "depth");
}

//! The terrain visibility buffer's resolve against the rasteriser (gfx/vis_terrain.rs): quads
//! of every corner order drawn forward write their interpolated UV and its screen gradients;
//! the same quads' IDs, rebuilt by the resolve's own math (`vis_terrain_math.wgsl`: the
//! triangle from the diagonal test, perspective-correct weights, the gradients of the 2x2
//! quad's neighbours), must give each pixel the same values.
use super::test_gpu;
use super::vis_terrain;
use glam::{Mat4, Vec3, Vec4};

const SIZE: u32 = 64;

/// The test's shaders: forward, ID and resolve over one array of clip-space corners (six
/// vertices a quad: corners 0 1 2 0 2 3, as cube_far and the surface tiles draw), each
/// with a UV.
const HARNESS: &str = r#"
struct Corner { clip: vec4<f32>, uv: vec4<f32> }
@group(0) @binding(0) var<storage, read> corners: array<Corner>;
@group(0) @binding(1) var ids: texture_2d<u32>;

fn corner_of(k: u32) -> u32 {
    var c = k;
    if k >= 3u {
        c = select(0u, k - 2u, k > 3u);
    }
    return c;
}
fn vertex_corner(vertex: u32) -> Corner {
    return corners[(vertex / 6u) * 4u + corner_of(vertex % 6u)];
}

struct Forward { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_forward(@builtin(vertex_index) v: u32) -> Forward {
    let c = vertex_corner(v);
    return Forward(c.clip, c.uv.xy);
}
struct Out { @location(0) a: vec4<f32>, @location(1) b: vec4<f32> }
@fragment fn fs_forward(in: Forward) -> Out {
    return Out(vec4(in.uv, dpdxFine(in.uv)), vec4(dpdyFine(in.uv), 0.0, 1.0));
}

struct Id { @builtin(position) clip: vec4<f32>, @location(0) @interpolate(flat) id: u32 }
@vertex fn vs_id(@builtin(vertex_index) v: u32) -> Id {
    return Id(vertex_corner(v).clip, v);
}
@fragment fn fs_id(in: Id) -> @location(0) vec2<u32> {
    return vec2(in.id, 1u);
}

@vertex fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4(f32(i32(i & 1u) * 4 - 1), f32(i32(i >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_resolve(@builtin(position) at: vec4<f32>) -> Out {
    let id = textureLoad(ids, vec2<i32>(floor(at.xy)), 0).xy;
    if id.y == 0u { discard; }
    let size = vec2<f32>(textureDimensions(ids));
    // The node families' rule: the provoking vertex starts the triangle.
    let a = vertex_corner(id.x);
    let b = vertex_corner(id.x + 1u);
    let c = vertex_corner(id.x + 2u);
    let w = vis_weights(at.xy, size, a.clip, b.clip, c.clip);
    return Out(vec4(vis_at2(w, a.uv.xy, b.uv.xy, c.uv.xy), vis_dx2(w, a.uv.xy, b.uv.xy, c.uv.xy)),
        vec4(vis_dy2(w, a.uv.xy, b.uv.xy, c.uv.xy), 0.0, 1.0));
}
// The arena families' rule: corner 0 provokes both triangles, the diagonal test picks one.
@fragment fn fs_resolve_arena(@builtin(position) at: vec4<f32>) -> Out {
    let id = textureLoad(ids, vec2<i32>(floor(at.xy)), 0).xy;
    if id.y == 0u { discard; }
    let size = vec2<f32>(textureDimensions(ids));
    let first = (id.x / 6u) * 4u;
    let a = corners[first];
    var b = corners[first + 1u];
    var c = corners[first + 2u];
    if vis_second(vis_ndc_in(at.xy, size), a.clip, b.clip, c.clip) {
        b = c;
        c = corners[first + 3u];
    }
    let w = vis_weights(at.xy, size, a.clip, b.clip, c.clip);
    return Out(vec4(vis_at2(w, a.uv.xy, b.uv.xy, c.uv.xy), vis_dx2(w, a.uv.xy, b.uv.xy, c.uv.xy)),
        vec4(vis_dy2(w, a.uv.xy, b.uv.xy, c.uv.xy), 0.0, 1.0));
}
"#;

/// Quads in perspective, tilted, near and far, some overlapping, with UVs in blocks (a
/// block a unit), as clip-space corners and UVs (x, y, 0, 0).
fn quads() -> Vec<[f32; 8]> {
    let proj = Mat4::perspective_rh(70f32.to_radians(), 1.0, 0.05, 1000.0);
    let view = Mat4::look_at_rh(Vec3::new(0.3, 1.7, 0.0), Vec3::new(2.0, 0.2, -9.0), Vec3::Y);
    let clip = proj * view;
    let mut out = Vec::new();
    let mut quad = |p: [Vec3; 4], uv: [[f32; 2]; 4]| {
        for (p, uv) in p.iter().zip(uv) {
            let c = clip * Vec4::new(p.x, p.y, p.z, 1.0);
            out.push([c.x, c.y, c.z, c.w, uv[0], uv[1], 0.0, 0.0]);
        }
    };
    // A floor of 16 x 16 blocks receding into the distance, one quad a block row.
    for row in 0..12 {
        let z0 = -1.0 - row as f32 * 2.5;
        let z1 = z0 - 2.5;
        quad(
            [
                Vec3::new(-6.0, 0.0, z0),
                Vec3::new(10.0, 0.0, z0),
                Vec3::new(10.0, 0.0, z1),
                Vec3::new(-6.0, 0.0, z1),
            ],
            [[0.0, 0.0], [16.0, 0.0], [16.0, 2.5], [0.0, 2.5]],
        );
    }
    // A wall at an angle, a slanted roof, and a small far quad: each corner order.
    quad(
        [
            Vec3::new(1.0, 0.0, -5.0),
            Vec3::new(1.0, 0.0, -9.0),
            Vec3::new(1.0, 3.0, -9.0),
            Vec3::new(1.0, 3.0, -5.0),
        ],
        [[0.0, 3.0], [4.0, 3.0], [4.0, 0.0], [0.0, 0.0]],
    );
    quad(
        [
            Vec3::new(4.0, 2.0, -12.0),
            Vec3::new(4.0, 3.5, -16.0),
            Vec3::new(-1.0, 3.5, -16.0),
            Vec3::new(-1.0, 2.0, -12.0),
        ],
        [[5.0, 1.0], [5.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
    );
    quad(
        [
            Vec3::new(3.0, 0.5, -40.0),
            Vec3::new(3.6, 0.5, -40.0),
            Vec3::new(3.6, 1.1, -40.0),
            Vec3::new(3.0, 1.1, -40.0),
        ],
        [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
    );
    out
}

fn target(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vis terrain test target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    texel: u32,
) -> Vec<u8> {
    let row = SIZE * texel;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * SIZE),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
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
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap().to_vec();
    data
}

#[test]
fn vis_terrain_resolve_rebuilds_what_the_rasteriser_interpolates() {
    let _gpu = test_gpu::lock();
    let Some((_adapter, device, queue)) = test_gpu::optional_device() else {
        eprintln!("skipped: no GPU");
        return;
    };
    let source = format!("{}\n{HARNESS}", include_str!("vis_terrain_math.wgsl"));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vis terrain test"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let corners = quads();
    let quads = corners.len() as u32 / 4;
    let buffer = wgpu::util::DeviceExt::create_buffer_init(
        &device,
        &wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&corners),
            usage: wgpu::BufferUsages::STORAGE,
        },
    );
    let fragment = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: fragment,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: fragment,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let float = wgpu::TextureFormat::Rgba32Float;
    let floats = [
        Some(wgpu::ColorTargetState::from(float)),
        Some(wgpu::ColorTargetState::from(float)),
    ];
    let ids_target = [Some(wgpu::ColorTargetState::from(vis_terrain::IDS))];
    let pipeline = |vs: &str, fs: &str, targets: &[Option<wgpu::ColorTargetState>], depth: bool| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(fs),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(vs),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: depth.then(|| wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let forward = pipeline("vs_forward", "fs_forward", &floats, true);
    let id = pipeline("vs_id", "fs_id", &ids_target, true);
    let resolves = [
        pipeline("vs_full", "fs_resolve", &floats, false),
        pipeline("vs_full", "fs_resolve_arena", &floats, false),
    ];
    let ids = target(&device, vis_terrain::IDS);
    let ids_view = ids.create_view(&Default::default());
    let group_with = |view: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        })
    };
    // The forward and ID draws must not bind the ID texture they (may) render into.
    let spare = target(&device, vis_terrain::IDS);
    let spare_view = spare.create_view(&Default::default());
    let draw_group = group_with(&spare_view);
    let resolve_group = group_with(&ids_view);
    let depth = target(&device, wgpu::TextureFormat::Depth32Float);
    let depth_view = depth.create_view(&Default::default());
    let draw = |pipeline: &wgpu::RenderPipeline,
                group: &wgpu::BindGroup,
                views: &[&wgpu::TextureView],
                depth: bool,
                vertices: u32| {
        let mut encoder = device.create_command_encoder(&Default::default());
        let attachments: Vec<_> = views
            .iter()
            .map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &attachments,
                depth_stencil_attachment: depth.then(|| wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..vertices, 0..1);
        }
        queue.submit([encoder.finish()]);
    };
    let (fa, fb) = (target(&device, float), target(&device, float));
    let (fa_view, fb_view) = (
        fa.create_view(&Default::default()),
        fb.create_view(&Default::default()),
    );
    draw(&forward, &draw_group, &[&fa_view, &fb_view], true, quads * 6);
    let forward_px = [
        read(&device, &queue, &fa, 16),
        read(&device, &queue, &fb, 16),
    ];
    // The ID pass gets a target of its own: the resolve reads it.
    let (ia, ib) = (target(&device, float), target(&device, float));
    let (ia_view, ib_view) = (
        ia.create_view(&Default::default()),
        ib.create_view(&Default::default()),
    );
    draw(&id, &draw_group, &[&ids_view], true, quads * 6);
    let id_px = read(&device, &queue, &ids, 8);
    let covered = id_px.chunks(8).filter(|t| t[4] != 0).count();
    assert!(
        covered > (SIZE * SIZE) as usize / 3,
        "the quads cover {covered} pixels"
    );
    for (r, resolve) in resolves.iter().enumerate() {
        draw(resolve, &resolve_group, &[&ia_view, &ib_view], false, 3);
        let resolved = [
            read(&device, &queue, &ia, 16),
            read(&device, &queue, &ib, 16),
        ];
        let f32s = |b: &[u8]| -> Vec<f32> { bytemuck::cast_slice(b).to_vec() };
        let (fwd_a, fwd_b) = (f32s(&forward_px[0]), f32s(&forward_px[1]));
        let (res_a, res_b) = (f32s(&resolved[0]), f32s(&resolved[1]));
        let mut worst = (0.0f32, 0.0f32);
        let mut gradient_misses = 0;
        for p in 0..(SIZE * SIZE) as usize {
            if id_px[p * 8 + 4] == 0 {
                continue;
            }
            let uv = |v: &[f32]| [v[p * 4], v[p * 4 + 1]];
            let (fu, ru) = (uv(&fwd_a), uv(&res_a));
            let grads = |a: &[f32], b: &[f32]| [a[p * 4 + 2], a[p * 4 + 3], b[p * 4], b[p * 4 + 1]];
            let (fg, rg) = (grads(&fwd_a, &fwd_b), grads(&res_a, &res_b));
            // In pixels: the rasteriser snaps the corners to 1/256 px (8 sub-pixel bits), the
            // resolve does not, so the UVs may differ by that much of a pixel's UV step.
            let step = fg.iter().fold(1e-3f32, |m, g| m.max(g.abs()));
            worst.0 = worst
                .0
                .max((fu[0] - ru[0]).abs().max((fu[1] - ru[1]).abs()) / step);
            for k in 0..4 {
                let err = (fg[k] - rg[k]).abs() / fg[k].abs().max(0.05);
                worst.1 = worst.1.max(err);
                // Edge pixels whose 2x2 quad straddles another quad differ by design
                // (the rasteriser's helper takes this triangle's plane too, but depth may
                // have kept another quad's pixel next to it): they must stay rare.
                if err > 0.02 {
                    gradient_misses += 1;
                }
            }
        }
        eprintln!(
            "resolve {r}: {covered} pixels, worst UV error {:.2e} px, worst gradient error {:.2e}, {gradient_misses} gradient misses",
            worst.0, worst.1
        );
        assert!(worst.0 < 1.0 / 64.0, "resolve {r}: UV off by {:.2e} px", worst.0);
        // A 2x2 quad on a triangle edge may take its neighbour differently: a few in a
        // thousand (a fraction of a mip level there), never a pattern.
        assert!(
            gradient_misses * 200 <= covered * 4,
            "resolve {r}: {gradient_misses} gradients off (worst {:.2e})",
            worst.1
        );
    }
}

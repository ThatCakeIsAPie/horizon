
// The terrain visibility buffer (gfx/vis_terrain.rs), appended after a solid family's own
// module (terrain.wgsl; cube_far's; the surface tiles') and before the family's entry points
// (gfx/vis_terrain_section.wgsl, vis_terrain_cube.wgsl, vis_terrain_surface.wgsl).
//
// The ID pass runs the family's own vertex function (the very clip position, `fade_depth` and
// `cf_by_level` depth bias of its solid pipeline) and writes, per pixel, the depth and a 64-bit
// ID of the quad it shows; quads with a cutout, a fade or a chunk fade-in keep the alpha test
// and dithers `solid_uncut` and `solid_color` make. The resolve, one full-screen draw per
// family (and arena page), rebuilds the pixel's triangle from the ID with the family's own
// corner function, interpolates its `TerrainOut` perspective-correctly, works out the UV's
// screen gradients and runs `fs_solid`'s shading past its discards (`sprite_at`,
// `solid_finish_at`, `solid_light`): emission, HDR marks and fog come along.
//
// ID layout (Rg32Uint; (0, 0): no terrain), mirrored by vis_terrain.rs's `id` module:
//   y bits 31..29: the family (VIS_SECTION 1, VIS_FAR_TILE 2, VIS_CUBE 3, VIS_SURFACE 4,
//   VIS_RASTER 5 for part C's compute raster; 0 none, 6 and 7 spare).
//   Families 1, 2 (arena quads): y bits 28..23 the arena page, bits 22..0 the draw's instance
//   (`Renderer::instances` for 1, far_cull's instances for 2); x the vertex index of the
//   quad's corner 0 (the provoking vertex of both its triangles, so the flat ID carries it):
//   a plain vertex, PACKED_BASE + 4 slot or FAR_BASE + 4 slot; corner c is x + c.
//   Families 3, 4: y bits 28..0 the node (cube_far) or tile slot (surface tiles); x the
//   vertex index of the triangle's provoking vertex, 6 quad + 0 for its first triangle
//   (corners 0 1 2) or 6 quad + 3 for its second (corners 0 2 3).
const VIS_FAMILY_SHIFT: u32 = 29u;
const VIS_SECTION: u32 = 1u;
const VIS_FAR_TILE: u32 = 2u;
const VIS_CUBE: u32 = 3u;
const VIS_SURFACE: u32 = 4u;
const VIS_RASTER: u32 = 5u;
const VIS_PAGE_SHIFT: u32 = 23u;
const VIS_INSTANCE_MASK: u32 = 0x7fffffu;
const VIS_LOW_MASK: u32 = 0x1fffffffu;
// Family and page: what one arena page's resolve draw takes.
const VIS_TAG_MASK: u32 = 0xff800000u;

fn vis_family(id: vec2<u32>) -> u32 {
    return id.y >> VIS_FAMILY_SHIFT;
}

// Group 2 of the resolve: the IDs, and the origins of the draws' instances for the arena
// families (`Renderer::instances`, or far_cull's).
@group(2) @binding(0) var vis_ids: texture_2d<u32>;
@group(2) @binding(1) var<storage, read> vis_instances: array<vec4<f32>>;

// What the ID pass's vertex stage hands its fragment stage: the ID, and what the cutout,
// fade and fade-in tests read.
struct VisIdWire {
    @builtin(position) clip: vec4<f32>,
    // The ID, the layer and the fade code (`TerrainOut::fade`).
    @location(0) @interpolate(flat) id: vec4<u32>,
    // `local` and `visibility`.
    @location(1) local: vec4<f32>,
    // `uv` and the colour's alpha.
    @location(2) uv: vec4<f32>,
}
fn vis_id_wire(t: TerrainOut, id: vec2<u32>) -> VisIdWire {
    var w: VisIdWire;
    w.clip = t.clip;
    w.id = vec4(id, t.layer, t.fade);
    w.local = vec4(t.local, t.visibility);
    w.uv = vec4(t.uv, t.color.a, 0.0);
    return w;
}
// `solid_uncut`'s and `solid_color`'s discards, in their order, then the ID. A quad with none
// of them (no fade, fully faded in, forceOpaque) takes no branch that discards.
@fragment fn fs_vis_terrain_id(w: VisIdWire) -> @location(0) vec2<u32> {
    let dx = dpdx(w.local.xyz);
    let dy = dpdy(w.local.xyz);
    let du = dpdx(w.uv.xy);
    let dv = dpdy(w.uv.xy);
    let layer = w.id.z;
    if faded(w.id.w, w.local.xyz, dx, dy) { discard; }
    if w.local.w < 1.0 && dithered_out(w.local.w, w.local.xyz, dx, dy) { discard; }
    // Bit 15 of the layer marks opaque leaves (forceOpaque): no alpha cutout.
    if (layer & 0x8000u) == 0u {
        let alpha = sprite_at(w.uv.xy, layer & 0x7fffu, du, dv).a * w.uv.z;
        if alpha < 0.5 { discard; }
    }
    return w.id.xy;
}

// The resolve's full-screen triangle.
@vertex fn vs_vis_terrain_resolve(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2(f32(i32(i & 1u) * 4 - 1), f32(i32(i >> 1u) * 4 - 1));
    return vec4(p, 0.0, 1.0);
}

fn vis_load(at: vec2<f32>) -> vec2<u32> {
    return textureLoad(vis_ids, vec2<i32>(floor(at)), 0).xy;
}

// A framebuffer position's NDC (the world pass draws to the whole target).
fn vis_ndc(at: vec2<f32>) -> vec2<f32> {
    return vis_ndc_in(at, vec2<f32>(textureDimensions(vis_ids)));
}

// The pixel's `TerrainOut` on the triangle a b c with weights `w`: every interpolated field
// perspective-correct, the flat ones from corner 0 (the provoking vertex of both triangles).
fn vis_lerp(a: TerrainOut, b: TerrainOut, c: TerrainOut, w: vec3<f32>) -> TerrainOut {
    var o = a;
    o.uv = w.x * a.uv + w.y * b.uv + w.z * c.uv;
    o.color = w.x * a.color + w.y * b.color + w.z * c.color;
    o.spherical = w.x * a.spherical + w.y * b.spherical + w.z * c.spherical;
    o.cylindrical = w.x * a.cylindrical + w.y * b.cylindrical + w.z * c.cylindrical;
    o.scene = w.x * a.scene + w.y * b.scene + w.z * c.scene;
    o.local = w.x * a.local + w.y * b.local + w.z * c.local;
    o.visibility = w.x * a.visibility + w.y * b.visibility + w.z * c.visibility;
    o.height = w.x * a.height + w.y * b.height + w.z * c.height;
    return o;
}

// `fs_solid` past its discards (the ID pass made them) for the pixel at framebuffer position
// `at` on the triangle a b c: the UV's screen gradients as dpdxFine/dpdyFine give them (the
// other pixel of this one's 2x2 quad in its row and in its column, odd minus even, on the
// same triangle's plane), then `sprite_at`, `solid_finish_at` and `solid_light`.
fn vis_shade(at: vec2<f32>, a: TerrainOut, b: TerrainOut, c: TerrainOut) -> vec4<f32> {
    let w = vis_weights(at, vec2<f32>(textureDimensions(vis_ids)), a.clip, b.clip, c.clip);
    let t = vis_lerp(a, b, c, w.here);
    let du = vis_dx2(w, a.uv, b.uv, c.uv);
    let dv = vis_dy2(w, a.uv, b.uv, c.uv);
    let texel = sprite_at(t.uv, t.layer & 0x7fffu, du, dv);
    let color = texel * t.color;
    return solid_light(solid_finish_at(t, texel, color, du, dv));
}

// Earth's sub-pixel far tiles rasterized in compute into the terrain visibility buffer
// (gfx/far_raster.rs). far_raster.rs prepends terrain.wgsl's own corner functions to this module:
// the arena page binding, ArenaCorner, arena_vertex, corner_offset, packed_corner, far_corner,
// PACKED_BASE, FAR_BASE, TILED_UV_STEPS, instance_step, fade_code and fade_depth, verbatim.
//
// Per phase of far_cull's lists (phase 1, then phase 2 after its Hi-Z test), before that
// phase's ID pass:
// - `fr_prep`, a workgroup per arena page: walks the page's far_cull draws; a draw whose far
//   tile passes the switch (`far_raster_small`: its voxel spans at most FAR_RASTER_PX pixels at
//   the tile's nearest point) goes to compute: its instance count becomes 0 (the hardware ID
//   draw skips it, its box stays in far_cull's lists and Hi-Z test as before) and its quads
//   join the page's compute list: `fr_starts` (exclusive prefix of the draws' compute quads),
//   `fr_heads` (the draw holding every HEAD_QUADS-th quad), the small pass's dispatch.
// - `fr_small`, a thread per compute quad: the quad's corners as the ID pass's vertex stage
//   makes them (fade_depth(view_proj * (position + origin), w), the viewport transform), snapped
//   to the 1/256 px grid (Vulkan's subPixelPrecisionBits 8); integer edge functions with the
//   top-left rule at pixel centres for triangles (0 1 2) and (0 2 3); back faces culled (the
//   solid ID pipeline's Back, CCW front); depth on the triangle's screen-space plane. Quads
//   covering up to SMALL_PIXELS pixel-centre bounds plot in the thread, bigger ones go to the
//   page's big list, alpha-tested ones (no OPAQUE_LAYER bit) to the page's leftover draws for
//   the hardware ID pipeline (their sprite's alpha test stays the ID pass's). Marks the
//   MASK_TILE screen tiles it touches.
// - `fr_big`: a workgroup per big quad (persistent groups), 64 threads over its pixels, i64
//   edges once the quad is too wide for i32.
// - `fr_tiles_reset` + `fr_tiles`: the marked tiles' list, an instanced indirect draw.
// Then in the phase's ID pass, after far_cull's draws: the leftover draws, then the merge
// (`vs_fr_merge` + `fs_fr_merge` over the marked tiles): frag_depth = the plot's depth with the
// ID pipeline's depth test, colour = the far tile ID the hardware draw writes
// (vis_terrain.rs `id::arena(FAR_TILE, page, instance, corner 0's vertex)`), and the pixel is
// cleared for the next phase. Hi-Z is built from that depth, so compute tiles occlude.
//
// Plot: atomicMax(vis[pixel], depth bits << 32 | compact id): reversed Z, nearer is larger,
// positive floats order as their bits. Compact id (vis_terrain.rs `id::RASTER`'s 32 bits):
// arena page << 26 | the page's compute quad. The merge expands it through fr_heads/fr_starts
// to the draw and the quad's corner 0 vertex.

// --- the switch (far_raster.rs `raster_small` is its mirror) ---
// On-screen size of a voxel of `voxel` blocks at `nearest` blocks: voxel * focal / nearest.
const FAR_RASTER_PX: f32 = 1.25;
fn far_raster_small(voxel: f32, nearest: f32, focal: f32) -> bool {
    return voxel > 0.0 && voxel * focal <= FAR_RASTER_PX * nearest;
}

const FR_HEAD_QUADS: u32 = 128u;
const FR_SMALL_WG: u32 = 128u;
const FR_BIG_GROUPS: u32 = 512u;
const FR_SMALL_PIXELS: u32 = 32u;
const FR_MASK_TILE: u32 = 16u;
const FR_SUBPIXEL: f32 = 256.0;
const FR_SUBPIXEL_BITS: u32 = 8u;
const FR_HALF: i32 = 128;
const FR_MIN_W: f32 = 0.05;
// Corners further out than this (pixels) are left to the hardware (never in practice).
const FR_GUARD_PX: f32 = 2097152.0;
// i32 edge functions hold while a quad spans fewer subpixels than this.
const FR_EXACT_EXTENT: i32 = 30000;
const FR_PAGE_SHIFT: u32 = 26u;
const FR_QUAD_MASK: u32 = 0x3ffffffu;
const FR_OPAQUE_LAYER: u32 = 0x8000u;
// far_cull.rs STATS.
const FR_STATS: u32 = 8u;
// vis_terrain.rs id::FAR_TILE << FAMILY_SHIFT, PAGE_SHIFT.
const FR_FAR_TAG: u32 = 0x40000000u;
const FR_ID_PAGE_SHIFT: u32 = 23u;
const FR_INSTANCE_MASK: u32 = 0x7fffffu;

struct FrParams {
    ids: u32,
    updates: u32,
    pages: u32,
    flags: u32,
    cam_int: vec4<i32>,
    cam_frac: vec4<f32>,
    eye: vec4<f32>,
    planes: array<vec4<f32>, 6>,
    view_proj: mat4x4<f32>,
}

struct FrRec {
    origin: vec4<i32>,
    a: vec4<u32>,
    bmin: vec4<f32>,
    bmax: vec4<f32>,
    ends: array<u32, 6>,
    keys: array<u32, 6>,
}

// terrain.wgsl's Globals, its first field (the world pass's own matrix: the corners are the
// ID pass's bit for bit).
struct FrGlobals {
    view_proj: mat4x4<f32>,
}

struct FrFrame {
    // The IDs' width and height, big list and leftover capacity a page.
    size: vec4<u32>,
}

struct FrPage {
    // Its far_cull region (first draw, size), its first fr_starts word, its first fr_heads word.
    region: vec4<u32>,
    // Its big list's and leftover draws' first entry, spare.
    lists: vec4<u32>,
}

@group(0) @binding(0) var<uniform> fr_params: FrParams;
@group(0) @binding(1) var<uniform> fr_frame: FrFrame;
@group(0) @binding(2) var<storage, read> fr_instances: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> fr_records: array<FrRec>;
@group(0) @binding(4) var<storage, read_write> fr_draws: array<u32>;
@group(0) @binding(5) var<storage, read> fr_counts: array<u32>;
@group(0) @binding(6) var<storage, read> fr_pages: array<FrPage>;
@group(0) @binding(7) var<storage, read_write> fr_starts: array<u32>;
@group(0) @binding(8) var<storage, read_write> fr_heads: array<u32>;
// Per arena page: big quads, compute quads, leftover draws, spare.
@group(0) @binding(9) var<storage, read_write> fr_meta: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> fr_big: array<u32>;
@group(0) @binding(11) var<storage, read_write> fr_vis: array<atomic<u64>>;
@group(0) @binding(12) var<storage, read_write> fr_mask: array<u32>;
@group(0) @binding(13) var<storage, read_write> fr_leftovers: array<u32>;
@group(0) @binding(14) var<uniform> fr_globals: FrGlobals;
// The page and phase this dispatch is for.
@group(1) @binding(40) var<uniform> fr_pass: vec4<u32>;
// Per arena page the small pass's dispatch (x, y, z, spare).
@group(2) @binding(0) var<storage, read_write> fr_dispatch: array<u32>;

fn fr_focal() -> f32 {
    let m = fr_globals.view_proj;
    return length(vec3(m[0][1], m[1][1], m[2][1])) * f32(fr_frame.size.y) * 0.5;
}

// The tile of record `id` goes to compute.
fn fr_compute_tile(id: u32) -> bool {
    let r = fr_records[id];
    let o = fr_instances[id];
    // Fading or dithered instances keep the hardware's discards.
    if o.w <= 0.0 || (bitcast<u32>(o.w) & 0x7fffffu) != 0u {
        return false;
    }
    let eye = fr_params.eye.xyz;
    let lo = o.xyz + r.bmin.xyz - eye;
    let hi = o.xyz + r.bmax.xyz - eye;
    let nearest = length(max(max(lo, -hi), vec3(0.0)));
    return far_raster_small(f32(r.a.w), nearest, fr_focal());
}

var<workgroup> fr_scan: array<u32, 256>;

@compute @workgroup_size(256)
fn fr_prep(@builtin(local_invocation_index) lane: u32) {
    let page = fr_pass.x;
    let p = fr_pages[page];
    let n = min(fr_counts[FR_STATS + fr_pass.y * fr_params.pages + page], p.region.y);
    var base = 0u;
    for (var c = 0u; c < n; c += 256u) {
        let i = c + lane;
        var q = 0u;
        if i < n {
            let d = (p.region.x + i) * 5u;
            if fr_draws[d + 1u] != 0u && fr_compute_tile(fr_draws[d + 4u]) {
                q = fr_draws[d] / 6u;
                fr_draws[d + 1u] = 0u;
            }
        }
        fr_scan[lane] = q;
        workgroupBarrier();
        for (var o = 1u; o < 256u; o <<= 1u) {
            var add = 0u;
            if lane >= o {
                add = fr_scan[lane - o];
            }
            workgroupBarrier();
            fr_scan[lane] += add;
            workgroupBarrier();
        }
        let end = base + fr_scan[lane];
        let start = end - q;
        if i < n {
            fr_starts[p.region.z + i] = start;
            for (var b = (start + FR_HEAD_QUADS - 1u) / FR_HEAD_QUADS; b * FR_HEAD_QUADS < end; b++) {
                fr_heads[p.region.w + b] = i;
            }
        }
        base += fr_scan[255];
        workgroupBarrier();
    }
    if lane == 0u {
        fr_starts[p.region.z + n] = base;
        let groups = (base + FR_SMALL_WG - 1u) / FR_SMALL_WG;
        fr_dispatch[page * 4u] = min(groups, 65535u);
        fr_dispatch[page * 4u + 1u] = select(1u, (groups + 65534u) / 65535u, groups > 0u);
        fr_dispatch[page * 4u + 2u] = 1u;
        atomicStore(&fr_meta[page * 4u], 0u);
        atomicStore(&fr_meta[page * 4u + 1u], base);
        atomicStore(&fr_meta[page * 4u + 2u], 0u);
    }
}

// The draw (its index in the page's region) holding compute quad q of `page`, and q's place in it.
struct FrRun {
    draw: u32,
    quad: u32,
}
fn fr_run(p: FrPage, q: u32) -> FrRun {
    var i = fr_heads[p.region.w + q / FR_HEAD_QUADS];
    while fr_starts[p.region.z + i + 1u] <= q {
        i++;
    }
    return FrRun(p.region.x + i, q - fr_starts[p.region.z + i]);
}

// Its corner 0's vertex (the ID's x) and the draw's instance.
fn fr_vertex(r: FrRun) -> u32 {
    return fr_draws[r.draw * 5u + 3u] + 4u * r.quad;
}

fn fr_corner(vertex: u32) -> ArenaCorner {
    if vertex < PACKED_BASE {
        return arena_vertex(vertex);
    }
    var slot = (vertex - PACKED_BASE) >> 2u;
    if vertex >= FAR_BASE {
        slot = (vertex - FAR_BASE) >> 2u;
    }
    let a = arena_page[slot * 3u];
    let b = arena_page[slot * 3u + 1u];
    let c = arena_page[slot * 3u + 2u];
    if (c.y & 0x80000000u) != 0u {
        return arena_vertex(slot + a.x + (vertex & 3u));
    }
    if vertex >= FAR_BASE {
        return far_corner(vec3(a.x, a.y, b.x), vec3(b.y, c.x, c.y), vertex & 3u);
    }
    return packed_corner(vec3(a.x, a.y, b.x), vec3(b.y, c.x, c.y), vertex & 3u);
}

struct FrCorners {
    p: array<vec2<i32>, 4>,
    z: vec4<f32>,
    ok: bool,
    opaque: bool,
}

fn fr_corners(vertex: u32, origin: vec4<f32>) -> FrCorners {
    var out: FrCorners;
    out.ok = true;
    let size = vec2<f32>(fr_frame.size.xy);
    for (var k = 0u; k < 4u; k++) {
        let c = fr_corner(vertex + k);
        if k == 0u {
            out.opaque = (c.r.y & FR_OPAQUE_LAYER) != 0u;
        }
        let clip = fade_depth(fr_globals.view_proj * vec4(bitcast<vec3<f32>>(c.p) + origin.xyz, 1.0),
            origin.w);
        if !(clip.w > FR_MIN_W) {
            out.ok = false;
            return out;
        }
        let ndc = clip.xyz / clip.w;
        let fb = vec2((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y);
        if any(abs(fb) > vec2(FR_GUARD_PX)) {
            out.ok = false;
            return out;
        }
        out.p[k] = vec2<i32>(round(fb * FR_SUBPIXEL));
        out.z[k] = ndc.z;
    }
    return out;
}

fn fr_top_left(p0: vec2<i32>, p1: vec2<i32>) -> bool {
    let d = p1 - p0;
    return d.y < 0 || (d.y == 0 && d.x > 0);
}
fn fr_edge(p0: vec2<i32>, p1: vec2<i32>, s: vec2<i32>) -> i32 {
    let d = p1 - p0;
    return d.x * (s.y - p0.y) - d.y * (s.x - p0.x);
}
fn fr_edge64(p0: vec2<i32>, p1: vec2<i32>, s: vec2<i32>) -> i64 {
    let d = p1 - p0;
    return i64(d.x) * i64(s.y - p0.y) - i64(d.y) * i64(s.x - p0.x);
}

struct FrTri {
    a: vec2<i32>,
    b: vec2<i32>,
    c: vec2<i32>,
    z: vec3<f32>,
    bias: vec3<i32>,
    inv_area: f32,
    live: bool,
}

// Corners i0 i1 i2 relative to o. Front faces (CCW in NDC) have a negative area in framebuffer
// space; they are turned round so that inside is non-negative. Back faces stay dead.
fn fr_setup(k: FrCorners, i0: u32, i1: u32, i2: u32, o: vec2<i32>, wide: bool) -> FrTri {
    var t: FrTri;
    t.a = k.p[i0] - o;
    var b = k.p[i1] - o;
    var c = k.p[i2] - o;
    var zb = k.z[i1];
    var zc = k.z[i2];
    var front: bool;
    var mag: f32;
    if wide {
        let a64 = fr_edge64(t.a, b, c);
        front = a64 < 0;
        mag = f32(-a64);
    } else {
        let a32 = fr_edge(t.a, b, c);
        front = a32 < 0;
        mag = f32(-a32);
    }
    t.live = front;
    if front {
        let tb = b;
        b = c;
        c = tb;
        let tz = zb;
        zb = zc;
        zc = tz;
    }
    t.b = b;
    t.c = c;
    t.z = vec3(k.z[i0], zb, zc);
    t.bias = vec3(select(-1, 0, fr_top_left(b, c)), select(-1, 0, fr_top_left(c, t.a)),
        select(-1, 0, fr_top_left(t.a, b)));
    t.inv_area = 1.0 / max(mag, 1.0);
    return t;
}

fn fr_cover(t: FrTri, s: vec2<i32>) -> f32 {
    let w0 = fr_edge(t.b, t.c, s);
    let w1 = fr_edge(t.c, t.a, s);
    let w2 = fr_edge(t.a, t.b, s);
    if !t.live || w0 + t.bias.x < 0 || w1 + t.bias.y < 0 || w2 + t.bias.z < 0 {
        return -1.0;
    }
    return clamp((t.z.x * f32(w0) + t.z.y * f32(w1) + t.z.z * f32(w2)) * t.inv_area, 0.0, 1.0);
}
fn fr_cover64(t: FrTri, s: vec2<i32>) -> f32 {
    let w0 = fr_edge64(t.b, t.c, s);
    let w1 = fr_edge64(t.c, t.a, s);
    let w2 = fr_edge64(t.a, t.b, s);
    if !t.live || w0 + i64(t.bias.x) < 0 || w1 + i64(t.bias.y) < 0 || w2 + i64(t.bias.z) < 0 {
        return -1.0;
    }
    return clamp((t.z.x * f32(w0) + t.z.y * f32(w1) + t.z.z * f32(w2)) * t.inv_area, 0.0, 1.0);
}

// The pixel centres (inclusive) the corners' bounds hold, within the screen.
fn fr_range(k: FrCorners) -> vec4<i32> {
    let mn = min(min(k.p[0], k.p[1]), min(k.p[2], k.p[3]));
    let mx = max(max(k.p[0], k.p[1]), max(k.p[2], k.p[3]));
    let lo = (mn - vec2(FR_HALF) + vec2(255)) >> vec2(FR_SUBPIXEL_BITS);
    let hi = (mx - vec2(FR_HALF)) >> vec2(FR_SUBPIXEL_BITS);
    return vec4(max(lo, vec2(0)), min(hi, vec2<i32>(fr_frame.size.xy) - vec2(1)));
}
fn fr_bounds(k: FrCorners) -> vec4<i32> {
    let mn = min(min(k.p[0], k.p[1]), min(k.p[2], k.p[3]));
    let mx = max(max(k.p[0], k.p[1]), max(k.p[2], k.p[3]));
    return vec4(mn, mx - mn);
}

fn fr_plot(x: i32, y: i32, z: f32, id: u32) {
    atomicMax(&fr_vis[u32(y) * fr_frame.size.x + u32(x)], (u64(bitcast<u32>(z)) << 32u) | u64(id));
}

fn fr_mark(r: vec4<i32>) {
    let lo = vec2<u32>(r.xy) / FR_MASK_TILE;
    let hi = vec2<u32>(r.zw) / FR_MASK_TILE;
    let row = (fr_frame.size.x + FR_MASK_TILE - 1u) / FR_MASK_TILE;
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            fr_mask[y * row + x] = 1u;
        }
    }
}

// Every pixel of the quad in one thread (`lane` of `lanes`).
fn fr_raster(k: FrCorners, r: vec4<i32>, id: u32, lane: u32, lanes: u32) {
    let ext = fr_bounds(k);
    let o = ext.xy;
    let wide = max(ext.z, ext.w) > FR_EXACT_EXTENT;
    let t0 = fr_setup(k, 0u, 1u, 2u, o, wide);
    let t1 = fr_setup(k, 0u, 2u, 3u, o, wide);
    if !t0.live && !t1.live {
        return;
    }
    let w = u32(r.z - r.x + 1);
    let n = w * u32(r.w - r.y + 1);
    for (var i = lane; i < n; i += lanes) {
        let px = vec2(i32(i % w), i32(i / w)) + r.xy;
        let s = (px << vec2(FR_SUBPIXEL_BITS)) + vec2(FR_HALF) - o;
        var z: f32;
        if wide {
            z = fr_cover64(t0, s);
            if z < 0.0 {
                z = fr_cover64(t1, s);
            }
        } else {
            z = fr_cover(t0, s);
            if z < 0.0 {
                z = fr_cover(t1, s);
            }
        }
        if z >= 0.0 {
            fr_plot(px.x, px.y, z, id);
        }
    }
}

@compute @workgroup_size(FR_SMALL_WG)
fn fr_small(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let page = fr_pass.x;
    let p = fr_pages[page];
    let q = (group.y * 65535u + group.x) * FR_SMALL_WG + lane;
    if q >= atomicLoad(&fr_meta[page * 4u + 1u]) {
        return;
    }
    let run = fr_run(p, q);
    let vertex = fr_vertex(run);
    let instance = fr_draws[run.draw * 5u + 4u];
    let k = fr_corners(vertex, fr_instances[instance]);
    if !k.ok {
        return;
    }
    if !k.opaque {
        // The hardware ID pipeline's alpha test: a draw of its own.
        let at = atomicAdd(&fr_meta[page * 4u + 2u], 1u);
        if at < fr_frame.size.w {
            let d = (p.lists.y + at) * 5u;
            fr_leftovers[d] = 6u;
            fr_leftovers[d + 1u] = 1u;
            fr_leftovers[d + 2u] = 0u;
            fr_leftovers[d + 3u] = vertex;
            fr_leftovers[d + 4u] = instance;
            return;
        }
    }
    let r = fr_range(k);
    if r.x > r.z || r.y > r.w {
        return;
    }
    fr_mark(r);
    let id = (page << FR_PAGE_SHIFT) | q;
    if u32(r.z - r.x + 1) * u32(r.w - r.y + 1) > FR_SMALL_PIXELS {
        let at = atomicAdd(&fr_meta[page * 4u], 1u);
        if at < fr_frame.size.z {
            fr_big[p.lists.x + at] = q;
            return;
        }
    }
    fr_raster(k, r, id, 0u, 1u);
}

@compute @workgroup_size(64)
fn fr_big_quads(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let page = fr_pass.x;
    let p = fr_pages[page];
    let count = min(atomicLoad(&fr_meta[page * 4u]), fr_frame.size.z);
    for (var b = group.x; b < count; b += FR_BIG_GROUPS) {
        let q = fr_big[p.lists.x + b];
        let run = fr_run(p, q);
        let k = fr_corners(fr_vertex(run), fr_instances[fr_draws[run.draw * 5u + 4u]]);
        fr_raster(k, fr_range(k), (page << FR_PAGE_SHIFT) | q, lane, 64u);
    }
}

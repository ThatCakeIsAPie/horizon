// far_raster.wgsl's tile list and merge (gfx/far_raster.rs).

struct FrFrame {
    size: vec4<u32>,
}

struct FrPage {
    region: vec4<u32>,
    lists: vec4<u32>,
}

const FR_MASK_TILE: u32 = 16u;
const FR_HEAD_QUADS: u32 = 128u;
const FR_PAGE_SHIFT: u32 = 26u;
const FR_QUAD_MASK: u32 = 0x3ffffffu;
const FR_FAR_TAG: u32 = 0x40000000u;
const FR_ID_PAGE_SHIFT: u32 = 23u;
const FR_INSTANCE_MASK: u32 = 0x7fffffu;

// --- the marked tiles as an instanced indirect draw (6 vertices a tile) ---
@group(0) @binding(0) var<storage, read_write> fr_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> fr_tile_list: array<u32>;
// draw indirect: vertex count, instance count, first vertex, first instance
@group(0) @binding(2) var<storage, read_write> fr_tile_args: array<atomic<u32>>;

@compute @workgroup_size(1)
fn fr_tiles_reset() {
    atomicStore(&fr_tile_args[0], 6u);
    atomicStore(&fr_tile_args[1], 0u);
    atomicStore(&fr_tile_args[2], 0u);
    atomicStore(&fr_tile_args[3], 0u);
}

@compute @workgroup_size(64)
fn fr_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= arrayLength(&fr_mask) || fr_mask[t] == 0u {
        return;
    }
    fr_mask[t] = 0u;
    fr_tile_list[atomicAdd(&fr_tile_args[1], 1u)] = t;
}

// --- the merge, in the phase's ID pass over the marked tiles ---
@group(0) @binding(10) var<uniform> fm_frame: FrFrame;
// far_raster.wgsl's fr_vis: depth bits high, compact id low (little-endian: x id, y depth).
@group(0) @binding(11) var<storage, read_write> fm_vis: array<vec2<u32>>;
@group(0) @binding(12) var<storage, read> fm_tiles: array<u32>;
@group(0) @binding(13) var<storage, read> fm_starts: array<u32>;
@group(0) @binding(14) var<storage, read> fm_heads: array<u32>;
@group(0) @binding(15) var<storage, read> fm_draws: array<u32>;
@group(0) @binding(16) var<storage, read> fm_pages: array<FrPage>;

@vertex
fn vs_fr_merge(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> @builtin(position) vec4<f32> {
    let size = fm_frame.size.xy;
    let row = (size.x + FR_MASK_TILE - 1u) / FR_MASK_TILE;
    let t = fm_tiles[i];
    var corner = vec2(0u, 0u);
    if v == 1u || v == 3u {
        corner = vec2(1u, 0u);
    } else if v == 2u || v == 5u {
        corner = vec2(0u, 1u);
    } else if v == 4u {
        corner = vec2(1u, 1u);
    }
    let px = min((vec2(t % row, t / row) + corner) * FR_MASK_TILE, size);
    return vec4(f32(px.x) / f32(size.x) * 2.0 - 1.0, 1.0 - f32(px.y) / f32(size.y) * 2.0, 0.0, 1.0);
}

struct FrMerged {
    @builtin(frag_depth) depth: f32,
    @location(0) id: vec2<u32>,
}

@fragment
fn fs_fr_merge(@builtin(position) at: vec4<f32>) -> FrMerged {
    let i = u32(at.y) * fm_frame.size.x + u32(at.x);
    let v = fm_vis[i];
    if v.y == 0u {
        discard;
    }
    fm_vis[i] = vec2(0u);
    let page = v.x >> FR_PAGE_SHIFT;
    let q = v.x & FR_QUAD_MASK;
    let p = fm_pages[page];
    var run = fm_heads[p.region.w + q / FR_HEAD_QUADS];
    while fm_starts[p.region.z + run + 1u] <= q {
        run++;
    }
    let d = (p.region.x + run) * 5u;
    let vertex = fm_draws[d + 3u] + 4u * (q - fm_starts[p.region.z + run]);
    let instance = fm_draws[d + 4u];
    return FrMerged(bitcast<f32>(v.y),
        vec2(vertex, FR_FAR_TAG | (page << FR_ID_PAGE_SHIFT) | (instance & FR_INSTANCE_MASK)));
}

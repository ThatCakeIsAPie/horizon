// Earth padded-window elevation: the bilinear mix of horizon-earth's `WindowPlan::heights_m`, many windows (jobs) in one
// dispatch. One thread per sample; `T` is `f64` when the device has SHADER_F64 (the CPU's arithmetic exactly) and `f32`
// otherwise. The host prepends `alias T = f64;` or `alias T = f32;`.
//
// Layout (u32 words unless noted):
//   hdrs[8 * job + 0..8]: width, depth, out_off (sample index in `out`), data_off (word index in `data`), nx (tile
//     columns of the job), sample_start (first flat sample of the job), ts_off (index in `ts`), unused.
//   data[data_off ..]: width column axes then depth row axes, two words each (a, b), each `tile << 16 | local`;
//     then the job's tile-slot table, row-major `[ty][tx]`, one pool slot per word.
//   ts[ts_off ..]: the width column fractions then the depth row fractions.
//   pool: tile slots of 65536 f32 each (256 x 256, index `y * 256 + x`).

struct Globals {
    // Always 0; the compiler cannot know, so it cannot fuse `a * b + c` into an fma (the CPU rounds the product).
    flag: u32,
    jobs: u32,
    total: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> hdrs: array<u32>;
@group(0) @binding(2) var<storage, read> data: array<u32>;
@group(0) @binding(3) var<storage, read> ts: array<T>;
@group(0) @binding(4) var<storage, read> pool: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

const NOFUSE: u32 = 12345u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= g.total) {
        return;
    }
    // The job holding sample i: the last whose sample_start <= i.
    var lo = 0u;
    var hi = g.jobs;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (hdrs[mid * 8u + 5u] <= i) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let h = lo * 8u;
    let width = hdrs[h];
    let depth = hdrs[h + 1u];
    let out_off = hdrs[h + 2u];
    let data_off = hdrs[h + 3u];
    let nx = hdrs[h + 4u];
    let ts_off = hdrs[h + 6u];
    let local = i - hdrs[h + 5u];
    let col = local % width;
    let row = local / width;

    let ca = data[data_off + 2u * col];
    let cb = data[data_off + 2u * col + 1u];
    let ra = data[data_off + 2u * (width + row)];
    let rb = data[data_off + 2u * (width + row) + 1u];
    let table = data_off + 2u * (width + depth);
    let tx = ts[ts_off + col];
    let ty = ts[ts_off + width + row];

    let s00 = data[table + (ra >> 16u) * nx + (ca >> 16u)];
    let s10 = data[table + (ra >> 16u) * nx + (cb >> 16u)];
    let s01 = data[table + (rb >> 16u) * nx + (ca >> 16u)];
    let s11 = data[table + (rb >> 16u) * nx + (cb >> 16u)];
    let h00 = T(pool[s00 * 65536u + (ra & 0xffffu) * 256u + (ca & 0xffffu)]);
    let h10 = T(pool[s10 * 65536u + (ra & 0xffffu) * 256u + (cb & 0xffffu)]);
    let h01 = T(pool[s01 * 65536u + (rb & 0xffffu) * 256u + (ca & 0xffffu)]);
    let h11 = T(pool[s11 * 65536u + (rb & 0xffffu) * 256u + (cb & 0xffffu)]);

    let zero = T(0.0);
    let p0 = (h10 - h00) * tx;
    let top = h00 + select(p0, zero, g.flag == NOFUSE);
    let p1 = (h11 - h01) * tx;
    let bottom = h01 + select(p1, zero, g.flag == NOFUSE);
    let d = bottom - top;
    let p2 = d * ty;
    let v = top + select(p2, zero, g.flag == NOFUSE);
    out[out_off + local] = f32(v);
}

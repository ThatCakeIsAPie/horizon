// GPU-driven culling of Earth's CPU-meshed far tiles (gfx/far_cull.rs).
//
// `apply` writes the records that changed (and marks each seen).
// `cull`: one invocation per record id keeps the records whose box meets the frustum,
// writes its camera-relative instance (origin, UV step) and the runs of its solid quads
// that can face the eye (mesher/sides.rs `visible_from`, as gpu_culling.wgsl does),
// and decides its phase: 1 (drawn before the depth pyramid exists) when the frame does
// not occlude or the last test saw its box, else a phase 2 candidate.
// `count`, `scan`, `scatter` (phase 1) and `count2`, `scan2`, `scatter2` (phase 2) turn
// the runs into draws, deterministically: a page's draws are in record id order, then
// run order. `count`: one workgroup per block of 64 ids counts its draws per page.
// `scan`: one workgroup per page turns those counts into where each block's draws start
// in the page's region of `draws`, and writes the page's draw count. `scatter`: one
// workgroup per block writes its ids' draws there. No atomic decides an order.
// `test` (after phase 1 drew and the pyramid is built): the conservative Hi-Z test of
// every frustum-visible box, as gpu_occlusion.wgsl `test_boxes`; `hidden[id]` is stored
// for the next frame, and the candidates seen now are phase 2's draws.

struct Params {
    // Ids to test, records `apply` writes, the arena pages the counts and regions cover,
    // and FLAGS.
    ids: u32,
    updates: u32,
    pages: u32,
    flags: u32,
    // The camera: its block (floor, per axis) and the f32 remainder in 0..1.
    cam_int: vec4<i32>,
    cam_frac: vec4<f32>,
    // The eye less the camera (xyz) and the instances' UV step (w).
    eye: vec4<f32>,
    // The frustum's planes (gfx/frustum.rs), camera-relative, facing in.
    planes: array<vec4<f32>, 6>,
    // The world pass's view projection, TAA's jitter included.
    view_proj: mat4x4<f32>,
}

// far_cull.rs `Record`.
struct Rec {
    // The node's origin (blocks) and its arena page.
    origin: vec4<i32>,
    // The base vertex of its first solid quad, its solid quads, its flags (ACTIVE), spare.
    a: vec4<u32>,
    // Its mesh's bounds around the origin.
    bmin: vec4<f32>,
    bmax: vec4<f32>,
    // Where its first six buckets end (mesher/sides.rs `BUCKETS`): +X, +Z, the other
    // bucket, +Y, -Z, -X; and what the eye's keys are compared with
    // (gpu_culling.rs `Record::facing`).
    ends: array<u32, 6>,
    keys: array<u32, 6>,
}

// far_cull.rs `Update`: the record for id `id.x`.
struct Update {
    id: vec4<u32>,
    rec: Rec,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> records: array<Rec>;
@group(0) @binding(2) var<storage, read> updates: array<Update>;
// Per id the instance it is drawn with: camera-relative origin and UV step.
@group(0) @binding(3) var<storage, read_write> instances: array<vec4<f32>>;
// Per id RUN_WORDS: (page << PAGE_SHIFT) | flags | its draws, then each draw's first quad
// and quads.
@group(0) @binding(4) var<storage, read_write> runs: array<u32>;
// Per id: 1 when the last test did not see its box (0 for a new record).
@group(0) @binding(5) var<storage, read_write> hidden: array<u32>;
// DrawIndexedIndirectArgs, five words each.
@group(0) @binding(6) var<storage, read_write> draws: array<u32>;
// The statistics (STATS), each page's draws in phase 1 then phase 2, then per phase, page
// and block of ids the block's draws (`count`), turned into where they start (`scan`).
@group(0) @binding(7) var<storage, read_write> counts: array<atomic<u32>>;
// Each page's region of `draws`: its first draw and its size.
@group(0) @binding(8) var<storage, read> regions: array<u32>;

@group(1) @binding(0) var hiz_depth: texture_2d<f32>;

const ROW: u32 = 65535u;
const BLOCK: u32 = 64u;
// far_cull.rs `RUNS` and `RUN_WORDS`.
const RUNS: u32 = 4u;
const RUN_WORDS: u32 = 9u;
// Statistics words: records seen by the frustum, boxes the test did not see, draws and
// triangles per phase, boxes tested, and the triangles of every solid quad of the records
// drawn (every face direction: what the facing runs leave out, geometry_stats).
const STATS: u32 = 8u;
const VISIBLE: u32 = 0u;
const OCCLUDED: u32 = 1u;
const DRAWS: u32 = 2u;
const TRIANGLES: u32 = 4u;
const TESTED: u32 = 6u;
const ALL_TRIANGLES: u32 = 7u;
// In `params.flags`: solid quads facing away from the eye are left out; the frame occludes.
const EYE: u32 = 1u;
const OCCLUDE: u32 = 2u;
// In a record's `a.z`.
const ACTIVE: u32 = 1u;
// In a run word 0: the draws (bits 0..2), phase 1 draws it, phase 2 may, its box is tested,
// phase 2 draws it; its page above PAGE_SHIFT.
const N_MASK: u32 = 7u;
const DRAW1: u32 = 8u;
const CANDIDATE: u32 = 16u;
const TEST: u32 = 32u;
const DRAW2: u32 = 64u;
const PAGE_SHIFT: u32 = 8u;
// The frustum test's box padding: a block's 1/64 and a share of the box's largest
// coordinate, far past what f32 rounding moves a camera-relative corner.
const FRUSTUM_PAD: f32 = 0.015625;
const FRUSTUM_REL: f32 = 0.000003814697265625;
// The occlusion test's box padding, a block and rounding.
const TEST_PAD: f32 = 1.015625;

fn order_key(x: f32) -> u32 {
    var bits = bitcast<u32>(x);
    if (bits << 1u) == 0u {
        bits = 0u;
    }
    if bits < 0x80000000u {
        return bits | 0x80000000u;
    }
    return ~bits;
}

fn region(page: u32) -> vec2<u32> {
    return vec2<u32>(regions[2u * page], regions[2u * page + 1u]);
}

fn page_count(page: u32, phase: u32) -> u32 {
    return STATS + phase * params.pages + page;
}

// Where block 0's count of (phase, page) is; the blocks follow.
fn table_base(page: u32, phase: u32) -> u32 {
    let blocks = (params.ids + BLOCK - 1u) / BLOCK;
    return STATS + 2u * params.pages + (phase * params.pages + page) * blocks;
}

// mesher/sides.rs `Sides::visible_from` for bucket `b` of record `id` (+X, +Z, the other
// bucket, +Y, -Z, -X, -Y), with `eye` the keys of the eye less its origin.
fn seen(id: u32, b: u32, eye: vec3<u32>) -> bool {
    if b == 2u {
        return true;
    }
    let t = records[id].keys[b - select(0u, 1u, b > 2u)];
    switch b {
        case 0u: { return eye.x > t; }
        case 1u: { return eye.z > t; }
        case 3u: { return eye.y > t; }
        case 4u: { return eye.z < t; }
        case 5u: { return eye.x < t; }
        default: { return eye.y < t; }
    }
}

fn bucket_end(id: u32, b: u32) -> u32 {
    if b == 6u {
        return records[id].a.y;
    }
    return records[id].ends[b];
}

// Whether the camera-relative box meets the frustum: gfx/view_frustum.rs `intersects`,
// with the box grown (FRUSTUM_PAD, FRUSTUM_REL) so rounding never rejects a box the CPU keeps.
fn in_frustum(lo: vec3<f32>, hi: vec3<f32>) -> bool {
    let size = max(max(max(abs(lo.x), abs(lo.y)), max(abs(lo.z), abs(hi.x))),
        max(abs(hi.y), abs(hi.z)));
    let pad = FRUSTUM_PAD + FRUSTUM_REL * size;
    let mn = lo - vec3(pad);
    let mx = hi + vec3(pad);
    for (var j = 0u; j < 6u; j++) {
        let p = params.planes[j];
        let far = select(mn, mx, p.xyz >= vec3(0.0));
        if dot(p.xyz, far) + p.w < 0.0 {
            return false;
        }
    }
    return true;
}

// One invocation per changed record; the CPU lists each id once.
@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < params.updates {
        let u = updates[gid.x];
        records[u.id.x] = u.rec;
        hidden[u.id.x] = 0u;
    }
}

var<workgroup> wg_visible: atomic<u32>;

// Whether record `id` is in the frustum; writes its instance and its runs word.
fn cull_one(id: u32) -> bool {
    let at = id * RUN_WORDS;
    runs[at] = 0u;
    let r = records[id];
    if (r.a.z & ACTIVE) == 0u {
        return false;
    }
    let page = u32(r.origin.w);
    if page >= params.pages {
        return false;
    }
    let o = vec3<f32>(r.origin.xyz - params.cam_int.xyz) - params.cam_frac.xyz;
    if !in_frustum(o + r.bmin.xyz, o + r.bmax.xyz) {
        return false;
    }
    instances[id] = vec4<f32>(o, params.eye.w);
    // gfx.rs `facing_runs`: the runs of its buckets that can face the eye, an empty
    // bucket breaking none, a draw each; without an eye, its solid quads.
    var n = 0u;
    if (params.flags & EYE) == 0u {
        if r.a.y > 0u {
            runs[at + 1u] = 0u;
            runs[at + 2u] = r.a.y;
            n = 1u;
        }
    } else {
        let d = params.eye.xyz - o;
        let eye = vec3<u32>(order_key(d.x), order_key(d.y), order_key(d.z));
        var start = 0u;
        var first = 0u;
        var open = false;
        for (var b = 0u; b < 7u; b++) {
            let end = bucket_end(id, b);
            if end == start {
                continue;
            }
            let shown = seen(id, b, eye);
            if shown && !open {
                open = true;
                first = start;
            } else if !shown && open {
                if n < RUNS {
                    runs[at + 1u + 2u * n] = first;
                    runs[at + 2u + 2u * n] = start - first;
                    n += 1u;
                }
                open = false;
            }
            start = end;
        }
        if open && n < RUNS {
            runs[at + 1u + 2u * n] = first;
            runs[at + 2u + 2u * n] = start - first;
            n += 1u;
        }
    }
    if n > 0u {
        var flags = DRAW1;
        if (params.flags & OCCLUDE) != 0u {
            flags = TEST | select(DRAW1, CANDIDATE, hidden[id] != 0u);
        }
        runs[at] = (page << PAGE_SHIFT) | flags | n;
    }
    return true;
}

@compute @workgroup_size(64)
fn cull(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let id = (group.y * ROW + group.x) * BLOCK + lane;
    if lane == 0u {
        atomicStore(&wg_visible, 0u);
    }
    workgroupBarrier();
    if id < params.ids && cull_one(id) {
        atomicAdd(&wg_visible, 1u);
    }
    workgroupBarrier();
    if lane == 0u {
        let v = atomicLoad(&wg_visible);
        if v > 0u {
            atomicAdd(&counts[VISIBLE], v);
        }
    }
}

// The runs word of `id` if it draws in `phase` (0 or 1), else 0.
fn drawn_in(id: u32, phase: u32) -> u32 {
    if id >= params.ids {
        return 0u;
    }
    let w = runs[id * RUN_WORDS];
    if (w & N_MASK) == 0u || (w & select(DRAW1, DRAW2, phase == 1u)) == 0u {
        return 0u;
    }
    return w;
}

var<workgroup> held: array<u32, 64>;
var<workgroup> held_tris: array<u32, 64>;
var<workgroup> held_all: array<u32, 64>;

// One workgroup per block of ids: each page's draws in the block, written where the
// block's count of the page goes (stored once, by the first id of that page in it).
fn count_block(block: u32, lane: u32, phase: u32) {
    let id = block * BLOCK + lane;
    let w = drawn_in(id, phase);
    held[lane] = w;
    var tris = 0u;
    let n = w & N_MASK;
    for (var k = 0u; k < n; k++) {
        tris += runs[id * RUN_WORDS + 2u + 2u * k] * 2u;
    }
    held_tris[lane] = tris;
    held_all[lane] = 0u;
    if n > 0u {
        held_all[lane] = records[id].a.y * 2u;
    }
    workgroupBarrier();
    if n > 0u {
        let page = w >> PAGE_SHIFT;
        var leader = true;
        var total = 0u;
        for (var other = 0u; other < BLOCK; other++) {
            let h = held[other];
            if (h & N_MASK) > 0u && (h >> PAGE_SHIFT) == page {
                total += h & N_MASK;
                if other < lane {
                    leader = false;
                }
            }
        }
        if leader && page < params.pages {
            atomicStore(&counts[table_base(page, phase) + block], total);
        }
    }
    if lane == 0u {
        var draws_here = 0u;
        var tris_here = 0u;
        var all_here = 0u;
        for (var other = 0u; other < BLOCK; other++) {
            draws_here += held[other] & N_MASK;
            tris_here += held_tris[other];
            all_here += held_all[other];
        }
        if draws_here > 0u {
            atomicAdd(&counts[TRIANGLES + phase], tris_here);
            atomicAdd(&counts[ALL_TRIANGLES], all_here);
        }
    }
}

var<workgroup> partial: array<u32, 256>;

// One workgroup per page: an exclusive scan of the page's blocks' counts, in place, and
// the page's draw count, clamped to its region.
fn scan_page(page: u32, lane: u32, phase: u32) {
    let blocks = (params.ids + BLOCK - 1u) / BLOCK;
    let base = table_base(page, phase);
    var running = 0u;
    for (var tile = 0u; tile < blocks; tile += 256u) {
        let b = tile + lane;
        var v = 0u;
        if b < blocks {
            v = atomicLoad(&counts[base + b]);
        }
        partial[lane] = v;
        workgroupBarrier();
        for (var step = 1u; step < 256u; step <<= 1u) {
            var before = 0u;
            if lane >= step {
                before = partial[lane - step];
            }
            workgroupBarrier();
            partial[lane] += before;
            workgroupBarrier();
        }
        if b < blocks && v > 0u {
            atomicStore(&counts[base + b], running + partial[lane] - v);
        }
        running += partial[255];
        workgroupBarrier();
    }
    if lane == 0u {
        let total = min(running, region(page).y);
        atomicStore(&counts[page_count(page, phase)], total);
        atomicAdd(&counts[DRAWS + phase], total);
    }
}

// One workgroup per block of ids: each id's draws after those of the block's earlier ids
// of its page, in run order.
fn scatter_block(block: u32, lane: u32, phase: u32) {
    let id = block * BLOCK + lane;
    let w = drawn_in(id, phase);
    held[lane] = w;
    workgroupBarrier();
    let n = w & N_MASK;
    if n == 0u {
        return;
    }
    let page = w >> PAGE_SHIFT;
    var slot = atomicLoad(&counts[table_base(page, phase) + block]);
    for (var other = 0u; other < lane; other++) {
        let h = held[other];
        if (h & N_MASK) > 0u && (h >> PAGE_SHIFT) == page {
            slot += h & N_MASK;
        }
    }
    let r = region(page);
    let at = id * RUN_WORDS;
    let base = records[id].a.x;
    for (var k = 0u; k < n; k++) {
        if slot + k >= r.y {
            return;
        }
        // A run of its solid quads, with its own instance.
        let d = (r.x + slot + k) * 5u;
        draws[d] = runs[at + 2u + 2u * k] * 6u;
        draws[d + 1u] = 1u;
        draws[d + 2u] = 0u;
        draws[d + 3u] = base + runs[at + 1u + 2u * k] * 4u;
        draws[d + 4u] = id;
    }
}

@compute @workgroup_size(64)
fn count(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    count_block(group.y * ROW + group.x, lane, 0u);
}

@compute @workgroup_size(256)
fn scan(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    scan_page(group.x, lane, 0u);
}

@compute @workgroup_size(64)
fn scatter(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    scatter_block(group.y * ROW + group.x, lane, 0u);
}

@compute @workgroup_size(64)
fn count2(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    count_block(group.y * ROW + group.x, lane, 1u);
}

@compute @workgroup_size(256)
fn scan2(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    scan_page(group.x, lane, 1u);
}

@compute @workgroup_size(64)
fn scatter2(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    scatter_block(group.y * ROW + group.x, lane, 1u);
}

var<workgroup> wg_tested: atomic<u32>;
var<workgroup> wg_hidden: atomic<u32>;

// gpu_occlusion.wgsl `test_boxes` for record `id`'s box (its mesh's bounds grown by TEST_PAD
// around its instance's origin): whether it may be seen. A corner at or behind the near plane
// or past it, or one that is not a number, counts as seen; a box off screen as not seen; else
// the box's footprint's 2x2 texels of the reversed-Z min pyramid are compared with its largest depth.
fn box_seen(id: u32) -> bool {
    let o = instances[id].xyz;
    let mn = o + records[id].bmin.xyz - vec3(TEST_PAD);
    let mx = o + records[id].bmax.xyz + vec3(TEST_PAD);
    var lo = vec2(1e30);
    var hi = vec2(-1e30);
    var max_z = -1e30;
    for (var c = 0u; c < 8u; c++) {
        let corner = select(mn, mx, (vec3(c) & vec3(1u, 2u, 4u)) != vec3(0u));
        let p = params.view_proj * vec4(corner, 1.0);
        if !(p.w > 1e-6 && p.z < p.w) {
            return true;
        }
        let inv_w = 1.0 / p.w;
        let ndc = p.xy * inv_w;
        let z = p.z * inv_w;
        lo = min(lo, ndc);
        hi = max(hi, ndc);
        max_z = max(max_z, z);
    }
    if hi.x < -1.0 || lo.x > 1.0 || hi.y < -1.0 || lo.y > 1.0 || max_z < 0.0 {
        return false;
    }

    let ndc_lo = max(lo, vec2(-1.0));
    let ndc_hi = min(hi, vec2(1.0));
    let uv_min = vec2(ndc_lo.x * 0.5 + 0.5, 0.5 - ndc_hi.y * 0.5);
    let uv_max = vec2(ndc_hi.x * 0.5 + 0.5, 0.5 - ndc_lo.y * 0.5);

    let max_mip = i32(textureNumLevels(hiz_depth)) - 1;
    let hiz_base = textureDimensions(hiz_depth, 0);
    let hiz_size = vec2<f32>(hiz_base);
    let rect = max((uv_max - uv_min) * hiz_size, vec2(0.0));
    let max_extent = max(rect.x, rect.y);
    var mip = clamp(i32(ceil(log2(max(max_extent, 1.0)))), 0, max_mip);

    if mip > 0 {
        let fine = mip - 1;
        let fine_size = vec2<i32>(max(hiz_base >> vec2<u32>(u32(fine)), vec2<u32>(1u)));
        let f_lo = clamp(vec2<i32>(floor(uv_min * vec2<f32>(fine_size))), vec2(0), fine_size - vec2(1));
        let f_hi = clamp(vec2<i32>(floor(uv_max * vec2<f32>(fine_size))), vec2(0), fine_size - vec2(1));
        if (f_hi.x - f_lo.x <= 1) && (f_hi.y - f_lo.y <= 1) {
            mip = fine;
        }
    }

    var mip_size = vec2<i32>(max(hiz_base >> vec2<u32>(u32(mip)), vec2<u32>(1u)));
    var lo_px = clamp(vec2<i32>(floor(uv_min * vec2<f32>(mip_size))), vec2(0), mip_size - vec2(1));
    var hi_px = clamp(vec2<i32>(floor(uv_max * vec2<f32>(mip_size))), vec2(0), mip_size - vec2(1));
    if ((hi_px.x - lo_px.x > 1) || (hi_px.y - lo_px.y > 1)) && mip < max_mip {
        mip += 1;
        mip_size = vec2<i32>(max(hiz_base >> vec2<u32>(u32(mip)), vec2<u32>(1u)));
        lo_px = clamp(vec2<i32>(floor(uv_min * vec2<f32>(mip_size))), vec2(0), mip_size - vec2(1));
        hi_px = clamp(vec2<i32>(floor(uv_max * vec2<f32>(mip_size))), vec2(0), mip_size - vec2(1));
    }

    let d00 = textureLoad(hiz_depth, vec2<i32>(lo_px.x, lo_px.y), mip).r;
    let d10 = textureLoad(hiz_depth, vec2<i32>(hi_px.x, lo_px.y), mip).r;
    let d01 = textureLoad(hiz_depth, vec2<i32>(lo_px.x, hi_px.y), mip).r;
    let d11 = textureLoad(hiz_depth, vec2<i32>(hi_px.x, hi_px.y), mip).r;
    let occluder_min_z = min(min(d00, d10), min(d01, d11));
    return !(max_z < occluder_min_z);
}

// One invocation per id: those the frustum kept and `cull` listed for the test (TEST) store
// whether their box is seen; a candidate seen now draws in phase 2.
@compute @workgroup_size(64)
fn test(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let id = (group.y * ROW + group.x) * BLOCK + lane;
    if lane == 0u {
        atomicStore(&wg_tested, 0u);
        atomicStore(&wg_hidden, 0u);
    }
    workgroupBarrier();
    if id < params.ids {
        let at = id * RUN_WORDS;
        let w = runs[at];
        if (w & TEST) != 0u {
            let box_is_seen = box_seen(id);
            hidden[id] = select(1u, 0u, box_is_seen);
            atomicAdd(&wg_tested, 1u);
            if !box_is_seen {
                atomicAdd(&wg_hidden, 1u);
            }
            if (w & CANDIDATE) != 0u && box_is_seen {
                runs[at] = w | DRAW2;
            }
        }
    }
    workgroupBarrier();
    if lane == 0u {
        let t = atomicLoad(&wg_tested);
        if t > 0u {
            atomicAdd(&counts[TESTED], t);
        }
        let h = atomicLoad(&wg_hidden);
        if h > 0u {
            atomicAdd(&counts[OCCLUDED], h);
        }
    }
}

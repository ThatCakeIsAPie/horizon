// Culls the far cube nodes against the frustum and the camera's side of each
// direction, writes their instance data (camera-relative origin and fade code,
// as gfx/lod.rs `instance`) and compacts one indirect draw per (node, direction,
// pass): non-indexed, six vertices a quad, the node in `first_instance`.
//
// Hook for Hi-Z occlusion: gpu_occlusion.wgsl's test of the node's box against the
// previous frame's depth pyramid goes where `visible` is decided; not wired yet.

const NODE_WORDS: u32 = 32u;
const NOT_SET: u32 = 0xffffffffu;
const TILE_STEP: f32 = 0.03125;
// Nodes the table holds (cube_far.rs `MAX_NODES`): a node's instance slots are its own and,
// past that many each, a shell dithering out and a shell dithering in.
const MAX_NODES: u32 = 262144u;
// Added to a shell's fade code (terrain.wgsl's `REVEAL`).
const REVEAL: u32 = 8192u;
// The coarsest node level that gets a shell (gfx/reveal.rs's `MAX_SHELL_LEVEL`, keep equal).
const MAX_SHELL_LEVEL: u32 = 2u;

struct Frame {
    planes: array<vec4<f32>, 6>,
    cam_int: vec4<i32>,
    cam_frac: vec4<f32>,
    now: u32,
    nodes: u32,
    // The sea's surface (sea level + 1), in blocks.
    water_top: i32,
    // The fade time in ms (`lod::fade_time`).
    fade_ms: u32,
    // The windows near cubes come in and leave in (gfx/reveal.rs): per window the
    // camera-relative box's low corner and how far along it is, then its high corner and 1
    // when cubes came in. How many are set.
    reveal: array<vec4<f32>, 128>,
    reveal_count: vec4<u32>,
}
@group(0) @binding(0) var<uniform> frame: Frame;
// Per node: origin xyz (blocks), level; shown ms, dying ms, flags (bit 0 live, bit 1 alone),
// spare; twelve (first quad, quads) ranges, index dir * 2 + translucent.
@group(0) @binding(1) var<storage, read> nodes: array<u32>;
@group(0) @binding(2) var<storage, read_write> instances: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> solid_args: array<u32>;
@group(0) @binding(4) var<storage, read_write> translucent_args: array<u32>;
// Draws written per pass; the count draws read them.
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>, 2>;

fn elapsed(now: u32, since: u32) -> f32 {
    return f32(select(0u, now - since, now > since)) / f32(frame.fade_ms);
}
fn fade_in(shown: u32, level: u32, now: u32) -> u32 {
    let step = u32(elapsed(now, shown) * 64.0) + 1u;
    if step >= 64u {
        return 0u;
    }
    return 1u + level * 129u + step;
}
fn fade_out(since: u32, level: u32, now: u32) -> u32 {
    return 1u + level * 129u + 64u + min(u32(elapsed(now, since) * 64.0) + 1u, 64u);
}

@compute @workgroup_size(64)
fn cull_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let node = id.x;
    if node >= frame.nodes {
        return;
    }
    let base = node * NODE_WORDS;
    if (nodes[base + 6u] & 1u) == 0u {
        return;
    }
    let shown = nodes[base + 4u];
    if shown == NOT_SET {
        return;
    }
    let level = nodes[base + 3u];
    let origin = vec3<i32>(bitcast<i32>(nodes[base]), bitcast<i32>(nodes[base + 1u]),
        bitcast<i32>(nodes[base + 2u]));
    let lo = vec3<f32>(origin - frame.cam_int.xyz) - frame.cam_frac.xyz;
    let hi = lo + f32(16u << level);
    var visible = true;
    for (var i = 0u; i < 6u; i++) {
        let pl = frame.planes[i];
        let far = select(lo, hi, pl.xyz >= vec3<f32>(0.0));
        if dot(pl.xyz, far) + pl.w < 0.0 {
            visible = false;
        }
    }
    if !visible {
        return;
    }
    var code = fade_in(shown, level, frame.now);
    let dying = nodes[base + 5u];
    if dying != NOT_SET && (nodes[base + 6u] & 2u) != 0u {
        code = fade_out(dying, level, frame.now);
    }
    // A node faded out entirely draws nothing.
    if code != 0u && (code - 1u) % 129u == 128u {
        return;
    }
    let w = select(-(f32(code) + TILE_STEP), TILE_STEP, code == 0u);
    instances[node * 2u] = vec4<f32>(lo, w);
    // The voxel size, and the sea's surface in node-local blocks (cube_far.wgsl
    // draws water up to it), clamped where it is far outside the node anyway.
    let water = clamp(frame.water_top - origin.y, -1, (16 << level) + 1);
    instances[node * 2u + 1u] = vec4<f32>(f32(1u << level), f32(water), 0.0, 0.0);
    // The camera sits at the origin: a direction shows only when it can face it.
    // Down, Up, North, South, West, East.
    var front = 0u;
    front |= select(0u, 1u, 0.0 < hi.y);
    front |= select(0u, 2u, 0.0 > lo.y);
    front |= select(0u, 4u, 0.0 < hi.z);
    front |= select(0u, 8u, 0.0 > lo.z);
    front |= select(0u, 16u, 0.0 < hi.x);
    front |= select(0u, 32u, 0.0 > lo.x);
    emit(base, node, front, 3u);
    // A node standing whole is drawn again, a shell, where near cubes come in or leave in
    // front of it: one for each kind, at its own depth (terrain.wgsl's `fade_depth`).
    if code == 0u && level <= MAX_SHELL_LEVEL && frame.reveal_count.x > 0u {
        let margin = vec3<f32>(f32(min(1u << level, 4u)));
        var kinds = 0u;
        for (var i = 0u; i < min(frame.reveal_count.x, 64u); i++) {
            let a = frame.reveal[i * 2u];
            let b = frame.reveal[i * 2u + 1u];
            if all(lo <= b.xyz + margin) && all(hi >= a.xyz - margin) {
                kinds |= select(2u, 1u, b.w > 0.5);
            }
        }
        for (var kind = 0u; kind < 2u; kind++) {
            if (kinds & (1u << kind)) == 0u {
                continue;
            }
            let out = kind == 0u;
            let shell_code = REVEAL + 1u + level * 129u + select(0u, 64u, out);
            let slot = MAX_NODES * (1u + kind) + node;
            instances[slot * 2u] = vec4<f32>(lo, -(f32(shell_code) + TILE_STEP));
            instances[slot * 2u + 1u] = vec4<f32>(f32(1u << level), f32(water), 0.0, 0.0);
            emit(base, slot, front, 1u);
        }
    }
}

// Appends the node's draws for the directions in `front`: its solid pass, and its water
// when `passes` has bit 1 (a shell has none: the mesh drawn whole has the same).
fn emit(base: u32, slot: u32, front: u32, passes: u32) {
    for (var dir = 0u; dir < 6u; dir++) {
        if (front & (1u << dir)) == 0u {
            continue;
        }
        for (var pass_ = 0u; pass_ < 2u; pass_++) {
            if (passes & (1u << pass_)) == 0u {
                continue;
            }
            let first = nodes[base + 8u + (dir * 2u + pass_) * 2u];
            let count = nodes[base + 9u + (dir * 2u + pass_) * 2u];
            if count == 0u {
                continue;
            }
            let draw = atomicAdd(&counts[pass_], 1u);
            let at = draw * 4u;
            if pass_ == 0u {
                if draw >= frame.nodes * 12u {
                    continue;
                }
                solid_args[at] = count * 6u;
                solid_args[at + 1u] = 1u;
                solid_args[at + 2u] = first * 6u;
                solid_args[at + 3u] = slot;
            } else {
                if draw >= frame.nodes * 6u {
                    continue;
                }
                translucent_args[at] = count * 6u;
                translucent_args[at + 1u] = 1u;
                translucent_args[at + 2u] = first * 6u;
                translucent_args[at + 3u] = slot;
            }
        }
    }
}

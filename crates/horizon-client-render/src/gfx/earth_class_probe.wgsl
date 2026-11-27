// Probe kernels for earth_class.wgsl's exact helpers (concatenated after it).
// Every entry point reads rows of (a, b, c) bit patterns and writes
// OUT_STRIDE result bit patterns per row; the host compares them with Rust.

struct Params {
    n: u32,
    // Never 12345 at run time (the opaque guard of mul_nc).
    flag: u32,
    // A width for clamp_px.
    w: u32,
    // Always 0 at run time (the opaque guard of mulx_nc).
    zero: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> in_a: array<u64>;
@group(0) @binding(2) var<storage, read> in_b: array<u64>;
@group(0) @binding(3) var<storage, read> in_c: array<u64>;
@group(0) @binding(4) var<storage, read_write> out: array<u64>;

const OUT_STRIDE: u32 = 8u;

fn nc_flag() -> u32 { return params.flag; }
fn nc_zero() -> u32 { return params.zero; }

fn put(i: u32, slot: u32, v: f64) { out[i * OUT_STRIDE + slot] = bitcast<u64>(v); }
fn put_u(i: u32, slot: u32, v: u64) { out[i * OUT_STRIDE + slot] = v; }

// 1. plain add / sub / mul
@compute @workgroup_size(64)
fn k_plain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    let b = bitcast<f64>(in_b[i]);
    put(i, 0u, a + b);
    put(i, 1u, a - b);
    put(i, 2u, a * b);
}

// 2. contraction: naive expressions and the guarded ones
@compute @workgroup_size(64)
fn k_contract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    let b = bitcast<f64>(in_b[i]);
    let c = bitcast<f64>(in_c[i]);
    put(i, 0u, a * b + c);
    put(i, 1u, a + (b - a) * c);
    put(i, 2u, mul_nc(a, b) + c);
    put(i, 3u, a + mul_nc(b - a, c));
    put(i, 4u, mulx_nc(a, b) + c);
    put(i, 5u, a + mulx_nc(b - a, c));
    put(i, 6u, a * b - c);
    put(i, 7u, mul_nc(a, b) - c);
}

// 2a. Each naive expression alone in its kernel: the product has exactly one
// use, so a fused multiply-add is possible (in k_contract the products are
// shared between several results).
@compute @workgroup_size(64)
fn k_naive_madd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    let b = bitcast<f64>(in_b[i]);
    let c = bitcast<f64>(in_c[i]);
    put(i, 0u, a * b + c);
}

@compute @workgroup_size(64)
fn k_naive_msub(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    let b = bitcast<f64>(in_b[i]);
    let c = bitcast<f64>(in_c[i]);
    put(i, 0u, a * b - c);
}

@compute @workgroup_size(64)
fn k_naive_lerp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    let b = bitcast<f64>(in_b[i]);
    let u = bitcast<f64>(in_c[i]);
    put(i, 0u, a + (b - a) * u);
}

// 6. fma
@compute @workgroup_size(64)
fn k_fma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    let b = bitcast<f64>(in_b[i]);
    let c = bitcast<f64>(in_c[i]);
    put(i, 0u, fma(a, b, c));
    put(i, 1u, fma_exact(a, b, c));
}

// 2b. expressions with several products: sums and differences of products,
// Horner; naive and with every product guarded
@compute @workgroup_size(64)
fn k_expr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    let b = bitcast<f64>(in_b[i]);
    let c = bitcast<f64>(in_c[i]);
    put(i, 0u, a * b + c * a);
    put(i, 1u, mul_nc(a, b) + mul_nc(c, a));
    put(i, 2u, a * b - b * c);
    put(i, 3u, mul_nc(a, b) - mul_nc(b, c));
    put(i, 4u, ((a * b + c) * a + b) * c + a);
    put(i, 5u, mul_nc(mul_nc(mul_nc(a, b) + c, a) + b, c) + a);
    put(i, 6u, lerp_nc(a, b, c));
    put(i, 7u, muladd_nc(a, b, c));
}

// 3. division; c holds the CPU's 1 / b
@compute @workgroup_size(64)
fn k_div(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    let b = bitcast<f64>(in_b[i]);
    let rc = bitcast<f64>(in_c[i]);
    put(i, 0u, a / b);
    put(i, 1u, div_cr(a, b));
    put(i, 2u, div_by(a, b, rc));
    put(i, 3u, a * rc);
    put(i, 4u, 1.0lf / b);
    put(i, 5u, recip_newton(b));
}

// 4. u64 and literals
@compute @workgroup_size(64)
fn k_u64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = in_a[i];
    let b = in_b[i];
    put_u(i, 0u, a * SM_GOLDEN);
    put_u(i, 1u, mix64(a));
    put(i, 2u, unit53(mix64(a)));
    put_u(i, 3u, a * b);
    put_u(i, 4u, (a >> 11u) ^ (b >> 27u) ^ (a >> 31u) ^ (b >> 30u));
    put_u(i, 5u, (a + b) ^ (a >> 11u));
    put(i, 6u, f64(a));
    put(i, 7u, 0x1p-53lf);
}

// 5. floor and conversions
@compute @workgroup_size(64)
fn k_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    let b = bitcast<f64>(in_b[i]);
    put(i, 0u, floor(a));
    put_u(i, 1u, i64_as_u64(floor_i64(a)));
    put_u(i, 2u, u64(clamp_px(a, params.w)));
    put(i, 3u, abs(a));
    var m = 0u;
    if a < b { m |= 1u; }
    if a <= b { m |= 2u; }
    if a > b { m |= 4u; }
    if a >= b { m |= 8u; }
    if a == b { m |= 16u; }
    if a != b { m |= 32u; }
    put_u(i, 4u, u64(m));
    // Plain conversion (undefined out of range; compared only inside it).
    put_u(i, 5u, bitcast<u64>(i64(floor(a))));
    put(i, 6u, f64(i64(floor(a))));
}

// 5. fmod, rem_euclid, longitude wrapping
@compute @workgroup_size(64)
fn k_rem(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    let a = bitcast<f64>(in_a[i]);
    put(i, 0u, rem_euclid_small(a, 360.0lf));
    put(i, 1u, a % 360.0lf);
    put(i, 2u, fmod_small(a, 360.0lf));
    put(i, 3u, wrap_longitude(a));
    let r = a % 360.0lf;
    put(i, 4u, select(r, r + 360.0lf, r < 0.0lf));
    let r2 = (a + 180.0lf) % 360.0lf;
    put(i, 5u, select(r2, r2 + 360.0lf, r2 < 0.0lf) - 180.0lf);
}

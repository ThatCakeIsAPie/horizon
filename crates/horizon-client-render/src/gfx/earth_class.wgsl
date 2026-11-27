// Exact f64 / u64 helpers for the Earth land-class kernel: results that match
// the CPU (Rust, IEEE-754 binary64, round-to-nearest-even, no contraction)
// bit for bit. Needs the device features SHADER_F64 and SHADER_INT64.
//
// This file holds ONLY helpers (no bindings, no entry points). The shader that
// includes it (earth_class_probe.wgsl, the class kernel) is concatenated after
// it and must define
//     fn nc_flag() -> u32     // a runtime value that is never 12345u
//     fn nc_zero() -> u32     // a runtime value that is always 0u
// (read from a storage buffer: the compiler cannot see through them).
// `earth_class::probe_exactness` measures every claim made below.
//
// Syntax that works in the vendored naga (WGSL front end), checked on the GPU:
//   f64 literals   1.0lf  0.5lf  1e-9lf  0x1p-53lf      (suffix `lf`)
//   i64 literals   1li    -5li                          (suffix `li`)
//   u64 literals   1lu    0x9E3779B97F4A7C15lu          (suffix `lu`)
//   casts          f64(x) i64(x) u64(x) f32(x) u32(x)   (value conversions)
//   bitcasts       bitcast<u64>(f64), bitcast<f64>(u64), bitcast<u64>(i64),
//                  bitcast<i64>(u64)    -- same width only. bitcast between
//                  vec2<u32> and f64/u64 is NOT available: build with
//                  u64(lo) | (u64(hi) << 32u).
//   shifts         u64 >> u32 (the shift count must be u32)
//   builtins       fma, floor, abs, max, min work on f64.
// Storage buffers hold the data as `array<u64>` (bit patterns).

// ---------------------------------------------------------------- bit access

fn f64_from_bits(b: u64) -> f64 { return bitcast<f64>(b); }
fn f64_bits(x: f64) -> u64 { return bitcast<u64>(x); }
fn u64_from_pair(lo: u32, hi: u32) -> u64 { return u64(lo) | (u64(hi) << 32u); }

const SIGN64: u64 = 0x8000000000000000lu;

// -------------------------------------------------- products that never fuse
//
// The driver may contract `a * b + c` into one fused multiply-add (one
// rounding), which differs from Rust's two roundings. A product is kept apart
// from the sum that follows by routing it through an operation the compiler
// cannot see through. Use mul_nc / mulx_nc for EVERY product that feeds an
// addition or subtraction; write the add/sub with a plain `+` / `-`.
//
//   p = mul_nc(a, b);  r = p + c;      // == Rust `a * b + c`
//   mul_nc:  select(a * b, 0.0lf, nc_flag() == 12345u)   -- value is always a*b
//   mulx_nc: bitcast<f64>(bitcast<u64>(a * b) ^ u64(nc_zero()))

fn mul_nc(a: f64, b: f64) -> f64 {
    return select(a * b, 0.0lf, nc_flag() == 12345u);
}

fn mulx_nc(a: f64, b: f64) -> f64 {
    return bitcast<f64>(bitcast<u64>(a * b) ^ u64(nc_zero()));
}

// a + (b - a) * u  with the product rounded separately (Rust `a + (b - a) * u`).
fn lerp_nc(a: f64, b: f64, u: f64) -> f64 {
    return a + mul_nc(b - a, u);
}

// a * b + c with two roundings (Rust `a * b + c`).
fn muladd_nc(a: f64, b: f64, c: f64) -> f64 {
    return mul_nc(a, b) + c;
}

// One rounding (Rust `f64::mul_add`).
fn fma_exact(a: f64, b: f64, c: f64) -> f64 {
    return fma(a, b, c);
}

// ----------------------------------------------------------------- division
//
// Correctly rounded a / b from fma, for the case the native `/` is not exact.
// b must be finite, normal, with 2^-100 < |b| < 2^100 (the seed goes through
// f32); a finite, and a / b neither overflowing nor subnormal.

// Reciprocal of b correctly rounded in all but rare cases: f32 seed, three
// Newton steps (each squares the relative error: 2^-23 -> 2^-46 -> 2^-92).
fn recip_newton(b: f64) -> f64 {
    var r = f64(1.0 / f32(b));
    r = fma(fma(-b, r, 1.0lf), r, r);
    r = fma(fma(-b, r, 1.0lf), r, r);
    r = fma(fma(-b, r, 1.0lf), r, r);
    return r;
}

// Quotient of a by b given r ~ 1 / b (Markstein): q = a * r, exact remainder
// rem = a - q * b (one fma), q' = q + rem * r (one fma).
fn div_by(a: f64, b: f64, r: f64) -> f64 {
    let q = a * r;
    let rem = fma(-q, b, a);
    return fma(rem, r, q);
}

fn div_cr(a: f64, b: f64) -> f64 {
    return div_by(a, b, recip_newton(b));
}

// ------------------------------------------------------------------- u64 hash

const SM_GOLDEN: u64 = 0x9E3779B97F4A7C15lu;
const SM_M1: u64 = 0xBF58476D1CE4E5B9lu;
const SM_M2: u64 = 0x94D049BB133111EBlu;

// splitmix64 finalizer (wrapping u64 multiply, native on the GPU).
fn mix64(x: u64) -> u64 {
    var z = x;
    z = (z ^ (z >> 30u)) * SM_M1;
    z = (z ^ (z >> 27u)) * SM_M2;
    return z ^ (z >> 31u);
}

// Uniform in [0, 1): (h >> 11) as f64 * 2^-53. Exact.
fn unit53(h: u64) -> f64 {
    return f64(h >> 11u) * 0x1p-53lf;
}

// ------------------------------------------------- f64 <-> integer conversions

// NaN test on the bits. The GPU does not preserve NaN semantics: `x != x` is
// false and every comparison with a NaN is false (measured), so never test
// for NaN with a comparison.
fn is_nan64(x: f64) -> bool {
    let b = bitcast<u64>(x) & ~SIGN64;
    return b > 0x7FF0000000000000lu;
}

// Rust `x as u32`: truncates, saturates, NaN -> 0.
fn f64_to_u32_sat(x: f64) -> u32 {
    if is_nan64(x) || !(x > 0.0lf) { return 0u; }
    if x >= 4294967295.0lf { return 0xFFFFFFFFu; }
    return u32(x);
}

// Rust `x as i64`: truncates, saturates, NaN -> 0.
fn f64_to_i64_sat(x: f64) -> i64 {
    if is_nan64(x) { return 0li; }
    if x >= 9223372036854775808.0lf { return 9223372036854775807li; }
    if x <= -9223372036854775808.0lf { return -9223372036854775807li - 1li; }
    return i64(x);
}

// Rust `x.floor() as i64`.
fn floor_i64(x: f64) -> i64 { return f64_to_i64_sat(floor(x)); }

// Rust `(i as u64)` for i: i64 -- a bit cast.
fn i64_as_u64(i: i64) -> u64 { return bitcast<u64>(i); }

// Rust `(x.max(0.0) as u32).min(w - 1)` (w >= 1).
fn clamp_px(x: f64, w: u32) -> u32 {
    return min(f64_to_u32_sat(max(x, 0.0lf)), w - 1u);
}

// ------------------------------------------------------------- fmod, wrapping

// Value of `a` with the sign of `s` (also for zeros): Rust `a.copysign(s)`.
fn copysign64(a: f64, s: f64) -> f64 {
    return bitcast<f64>((bitcast<u64>(a) & ~SIGN64) | (bitcast<u64>(s) & SIGN64));
}

// Rust `a % m` (C fmod, exact) for m > 0 and |a| < 64 * m, by repeated exact
// subtraction: x - m is exact for x >= m whenever m is a multiple of ulp(x)
// (m = 360 is a multiple of 8, so every x < 2^55). Keeps -0.0 for negative
// multiples of m, like fmod.
fn fmod_small(a: f64, m: f64) -> f64 {
    var r = abs(a);
    for (var i = 0u; i < 64u; i++) {
        if r < m { break; }
        r = r - m;
    }
    return copysign64(r, a);
}

// Rust `a.rem_euclid(m)` for m > 0, |a| < 64 * m.
fn rem_euclid_small(a: f64, m: f64) -> f64 {
    let r = fmod_small(a, m);
    if r < 0.0lf { return r + m; }
    return r;
}

// Rust `(l + 180.0).rem_euclid(360.0) - 180.0`, |l| < 64 * 360 - 180.
fn wrap_longitude(l: f64) -> f64 {
    return rem_euclid_small(l + 180.0lf, 360.0lf) - 180.0lf;
}

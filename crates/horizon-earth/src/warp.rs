//! Smooth deterministic 2D noise for bending the pixel grid of raster data.
//!
//! Raster land cover is a grid of squares; read by nearest pixel it gives right-angle steps as wide as a pixel.
//! Reading it at a point moved by a smooth, small, fixed offset field turns straight pixel edges into wavy lines
//! and corners into bends, without moving any data further than about a pixel. The noise is a lattice of hashed
//! values blended by smoothstep, so it is continuous (no per-block speckle), the same everywhere for everyone
//! (no world seed, a fixed salt per use) and needs no state.

/// The splitmix64 finalizer: a good 64-bit mix of its input.
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A number in `[0, 1)` from the lattice point `(ix, iz)` and a salt.
fn lattice(ix: i64, iz: i64, salt: u64) -> f64 {
    let h = mix64(
        salt ^ (ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ mix64((iz as u64).wrapping_add(0xD1B5_4A32_D192_ED03)),
    );
    (h >> 11) as f64 / (1_u64 << 53) as f64
}

/// A number in `[0, 1)` from three integers and a salt: a hash of cells, for [`crate::landfield`]'s speck mix.
pub(crate) fn cell_hash(a: i64, b: i64, c: i64, salt: u64) -> f64 {
    let h = mix64(
        salt ^ (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ mix64((b as u64).wrapping_add(0xD1B5_4A32_D192_ED03))
            ^ mix64((c as u64).wrapping_mul(0xA24B_AED4_963E_E407) ^ 0x1656_67B1_9E37_79F9),
    );
    (h >> 11) as f64 / (1_u64 << 53) as f64
}

/// Lattice value noise at `(x, z)` (lattice spacing 1) in `[-1, 1]`: continuous, with the slope at most 3
/// per lattice step.
pub(crate) fn value_noise(x: f64, z: f64, salt: u64) -> f64 {
    let (fx, fz) = (x.floor(), z.floor());
    let (tx, tz) = (x - fx, z - fz);
    let (ix, iz) = (fx as i64, fz as i64);
    let fade = |t: f64| t * t * (3.0 - 2.0 * t);
    let (u, v) = (fade(tx), fade(tz));
    let a = lattice(ix, iz, salt);
    let b = lattice(ix + 1, iz, salt);
    let c = lattice(ix, iz + 1, salt);
    let d = lattice(ix + 1, iz + 1, salt);
    let top = a + (b - a) * u;
    let bottom = c + (d - c) * u;
    2.0 * (top + (bottom - top) * v) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_noise_is_in_range_continuous_and_repeatable() {
        let mut max_step: f64 = 0.0;
        let mut previous = value_noise(-50.0, 3.3, 7);
        for i in -499..500 {
            let x = f64::from(i) * 0.1;
            let n = value_noise(x, 3.3, 7);
            assert!((-1.0..=1.0).contains(&n));
            assert_eq!(n, value_noise(x, 3.3, 7));
            max_step = max_step.max((n - previous).abs());
            previous = n;
        }
        // Slope at most 3 per lattice step, so 0.3 per 0.1.
        assert!(max_step <= 0.3 + 1e-9, "{max_step}");
        // Different salts are different fields.
        assert_ne!(value_noise(0.4, 0.6, 1), value_noise(0.4, 0.6, 2));
    }

    #[test]
    fn the_noise_is_spread_over_its_range() {
        let values: Vec<f64> = (0..4000)
            .map(|i| value_noise(f64::from(i) * 0.37, f64::from(i % 17) * 0.91, 11))
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!(mean.abs() < 0.1, "{mean}");
        assert!(values.iter().any(|&v| v > 0.6) && values.iter().any(|&v| v < -0.6));
    }
}

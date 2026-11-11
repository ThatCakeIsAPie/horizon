//! Repair of implausible pixels in a decoded elevation tile.
//!
//! The source elevation data is stitched from 1°×1° cells, and the seams carry bad pixels: a whole row or
//! column, sometimes only part of one, thousands of metres off (a 4,475 m pit one pixel wide in terrain at
//! 7,280 m was measured at 28°N). Left alone, it renders as a dead-straight trench or wall across the world.
//!
//! A pixel is bad when it stands out from its neighbours on both sides along one axis by far more than terrain
//! can, while those two neighbours agree with each other. Runs of up to [`MAX_RUN`] such pixels are replaced by
//! interpolation between the good pixels on either side. At a tile's edge there is only one side, so the two
//! inner pixels are extrapolated, under a stricter limit. Every limit scales with the ground size of a pixel
//! at the tile's zoom, so real peaks and canyons at coarse zooms are never touched.

use crate::tile::TILE_SIZE;

/// Version of the cleaning rule. Anything built from cleaned tiles (the far-view cache) must key on it, so a
/// change to the rule or its thresholds invalidates what was built under the old one.
pub const CLEAN_RULES: u32 = 1;

/// The widest run of bad pixels (across the line) that is repaired. In the real tiles along 28°N the defect is
/// one pixel wide in some tiles and two in others (with different depths), and never more. Wider runs begin to
/// look like real landforms at fine zoom, and a missed rare defect is better than a damaged ridge.
pub const MAX_RUN: usize = 2;

/// A pixel is bad when it is off the line through its neighbours by more than this many pixel widths of ground
/// (about 150 m at zoom 15). No real landform is a one-pixel slot that deep.
const DEPTH_PIXELS: f64 = 32.0;
/// The two neighbours must agree within this many pixel widths of ground across two pixel steps (about 38 m at
/// zoom 15, a slope of 4 m per metre), i.e. be on the same ordinary slope: a cliff or a coast has different
/// neighbours and is never a "slot". A run of `n` pixels has `n + 1` steps between its neighbours and gets
/// `n + 1` halves of it.
const AGREE_PIXELS: f64 = 8.0;
/// At a tile's edge there is no second neighbour to confirm a slot, so the deviation must be this many pixel
/// widths (about 460 m at zoom 15), three times the interior limit.
const EDGE_PIXELS: f64 = 96.0;

const SIDE: usize = TILE_SIZE as usize;

/// Metres of ground in one pixel at `zoom`, at the equator. Mercator pixels are narrower toward the poles, so
/// this overstates the ground size there, which only makes every limit more lenient.
pub fn ground_size_m(zoom: u8) -> f64 {
    40_075_016.685_578_5 / (f64::from(TILE_SIZE) * f64::from(1u32 << zoom.min(30)))
}

/// Repairs implausible pixels of a row-major 256×256 height grid in place, columns first and then rows, and
/// returns how many pixels it changed.
pub fn clean_tile(heights: &mut [f32], zoom: u8) -> usize {
    debug_assert_eq!(heights.len(), SIDE * SIDE);
    let g = ground_size_m(zoom);
    let limits = Limits {
        depth: (DEPTH_PIXELS * g) as f32,
        agree: (AGREE_PIXELS * g) as f32,
        edge: (EDGE_PIXELS * g) as f32,
    };
    // A pixel that both passes repair (where a bad row crosses a bad column) is counted once.
    let mut touched = vec![false; SIDE * SIDE];
    let mut line = [0.0f32; SIDE];
    for stride in [SIDE, 1] {
        // stride SIDE walks down a column (the line's positions are rows), stride 1 along a row.
        for k in 0..SIDE {
            let base = if stride == SIDE { k } else { k * SIDE };
            for (i, v) in line.iter_mut().enumerate() {
                *v = heights[base + i * stride];
            }
            let n = clean_line(&mut line, limits);
            if n > 0 {
                for (i, v) in line.iter().enumerate() {
                    let at = base + i * stride;
                    touched[at] |= heights[at] != *v;
                    heights[at] = *v;
                }
            }
        }
    }
    touched.iter().filter(|&&t| t).count()
}

#[derive(Clone, Copy)]
struct Limits {
    depth: f32,
    agree: f32,
    edge: f32,
}

/// Repairs the bad runs of one line of heights and returns how many pixels changed.
fn clean_line(line: &mut [f32; SIDE], lim: Limits) -> usize {
    let mut repaired = 0;
    let mut i = 0;
    while i < SIDE {
        let mut hit = 0;
        for len in 1..=MAX_RUN.min(SIDE - 2) {
            if i + len <= SIDE && run_is_bad(line, i, len, lim) {
                hit = len;
                break;
            }
        }
        if hit == 0 {
            i += 1;
            continue;
        }
        for k in 0..hit {
            line[i + k] = expected(line, i, hit, k);
        }
        repaired += hit;
        i += hit;
    }
    repaired
}

/// Where the pixel `k` of the run `start..start + len` would be if the terrain were smooth: interpolated between
/// the pixels either side, or extrapolated from the two nearest on the one side that exists.
fn expected(line: &[f32; SIDE], start: usize, len: usize, k: usize) -> f32 {
    let end = start + len;
    if start == 0 {
        let (p1, p2) = (line[end], line[end + 1]);
        p1 + (p1 - p2) * (len - k) as f32
    } else if end == SIDE {
        let (p1, p2) = (line[start - 1], line[start - 2]);
        p1 + (p1 - p2) * (k + 1) as f32
    } else {
        let (a, d) = (line[start - 1], line[end]);
        a + (d - a) * (k + 1) as f32 / (len + 1) as f32
    }
}

fn run_is_bad(line: &[f32; SIDE], start: usize, len: usize, lim: Limits) -> bool {
    let end = start + len;
    // Neighbours agree when their height difference is a steep but real slope: `agree` per pixel step.
    let per_step = lim.agree / 2.0;
    let (anchors_agree, limit) = if start == 0 {
        if end + 1 >= SIDE {
            return false;
        }
        ((line[end] - line[end + 1]).abs() <= per_step, lim.edge)
    } else if end == SIDE {
        (
            (line[start - 1] - line[start - 2]).abs() <= per_step,
            lim.edge,
        )
    } else {
        // The anchors are `len + 1` steps apart.
        (
            (line[start - 1] - line[end]).abs() <= per_step * (len + 1) as f32,
            lim.depth,
        )
    };
    if !anchors_agree {
        return false;
    }
    // Every pixel of the run must be off the smooth line by more than the limit, all on the same side.
    let mut sign = 0.0f32;
    for k in 0..len {
        let dev = line[start + k] - expected(line, start, len, k);
        if !dev.is_finite() || dev.abs() <= limit || (sign != 0.0 && dev.signum() != sign) {
            return false;
        }
        sign = dev.signum();
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{encode_height, encode_png};
    use crate::tile::Tile;

    /// A smooth slope: about 0.8 m per pixel down the tile and 0.3 m along it.
    fn slope(x: usize, y: usize) -> f32 {
        7000.0 + 0.8 * y as f32 + 0.3 * x as f32
    }

    fn grid() -> Vec<f32> {
        (0..SIDE * SIDE)
            .map(|i| slope(i % SIDE, i / SIDE))
            .collect()
    }

    fn at(g: &mut [f32], x: usize, y: usize) -> &mut f32 {
        &mut g[y * SIDE + x]
    }

    fn assert_on_slope(g: &[f32], pixels: &[(usize, usize)]) {
        for &(x, y) in pixels {
            let got = g[y * SIDE + x];
            assert!(
                (got - slope(x, y)).abs() < 3.0,
                "({x}, {y}) is {got}, the slope says {}",
                slope(x, y)
            );
        }
    }

    #[test]
    fn the_measured_bad_row_is_repaired_to_the_slope() {
        let mut g = grid();
        for x in 0..SIDE {
            *at(&mut g, x, 109) -= 4475.0;
        }
        assert_eq!(clean_tile(&mut g, 15), SIDE);
        let all: Vec<_> = (0..SIDE).map(|x| (x, 109)).collect();
        assert_on_slope(&g, &all);
        // Nothing else moved.
        let mut want = grid();
        for x in 0..SIDE {
            *at(&mut want, x, 109) = g[109 * SIDE + x];
        }
        assert_eq!(g, want);
    }

    #[test]
    fn a_decoded_tile_has_no_trace_of_the_pit_in_its_extremes() {
        let mut rgb = Vec::new();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let h = if y == 109 {
                    2805.4
                } else {
                    f64::from(slope(x, y))
                };
                rgb.extend(encode_height(h));
            }
        }
        let png = encode_png(256, 256, png::ColorType::Rgb, png::BitDepth::Eight, &rgb);
        let tile = Tile::decode(&png, 15).unwrap();
        assert_eq!(tile.repaired(), SIDE);
        assert!(tile.min() > 6999.0, "min {}", tile.min());
        let (lo, _) = tile.range(0, 100, 255, 120).unwrap();
        assert!(lo > 7079.0, "range min {lo}");
        // The same bytes at zoom 8 are real terrain as far as the rule can tell.
        let coarse = Tile::decode(&png, 8).unwrap();
        assert_eq!(coarse.repaired(), 0);
        assert!(coarse.min() < 2806.0);
    }

    #[test]
    fn a_bad_column_a_spike_and_a_partial_row_are_repaired() {
        let mut g = grid();
        for y in 0..SIDE {
            *at(&mut g, 77, y) -= 3000.0;
        }
        for x in 0..SIDE {
            *at(&mut g, x, 200) += 2500.0;
        }
        for x in 40..140 {
            *at(&mut g, x, 30) -= 900.0;
        }
        let n = clean_tile(&mut g, 15);
        // The column crosses both rows; those two pixels are counted once each.
        assert_eq!(n, 256 + 256 + 100 - 2);
        let mut want = grid();
        for y in 0..SIDE {
            *at(&mut want, 77, y) = slope(77, y);
        }
        for x in 0..SIDE {
            *at(&mut want, x, 200) = slope(x, 200);
        }
        for x in 40..140 {
            *at(&mut want, x, 30) = slope(x, 30);
        }
        for (i, (a, b)) in g.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 3.0, "pixel {i}: {a} vs {b}");
        }
    }

    #[test]
    fn a_two_pixel_wide_run_is_repaired_and_a_three_wide_one_is_not() {
        let mut g = grid();
        for x in 0..SIDE {
            *at(&mut g, x, 50) -= 4000.0;
            *at(&mut g, x, 51) -= 4100.0;
        }
        assert_eq!(clean_tile(&mut g, 15), 2 * SIDE);
        let both: Vec<_> = (0..SIDE).flat_map(|x| [(x, 50), (x, 51)]).collect();
        assert_on_slope(&g, &both);

        let mut wide = grid();
        for y in 50..53 {
            for x in 0..SIDE {
                *at(&mut wide, x, y) -= 4000.0;
            }
        }
        assert_eq!(clean_tile(&mut wide, 15), 0);
    }

    #[test]
    fn a_bad_first_or_last_row_or_column_cannot_hide_on_the_edge() {
        for (what, dy) in [("first row", 0), ("last row", 255)] {
            let mut g = grid();
            for x in 0..SIDE {
                *at(&mut g, x, dy) -= 4475.0;
            }
            assert_eq!(clean_tile(&mut g, 15), SIDE, "{what}");
            let line: Vec<_> = (0..SIDE).map(|x| (x, dy)).collect();
            assert_on_slope(&g, &line);
        }
        for x0 in [0, 255] {
            let mut g = grid();
            for y in 0..SIDE {
                *at(&mut g, x0, y) += 4000.0;
            }
            assert_eq!(clean_tile(&mut g, 15), SIDE, "column {x0}");
            let line: Vec<_> = (0..SIDE).map(|y| (x0, y)).collect();
            assert_on_slope(&g, &line);
        }
        // Two bad rows on the last edge.
        let mut g = grid();
        for x in 0..SIDE {
            *at(&mut g, x, 254) -= 4475.0;
            *at(&mut g, x, 255) -= 4475.0;
        }
        assert_eq!(clean_tile(&mut g, 15), 2 * SIDE);
        let line: Vec<_> = (0..SIDE).flat_map(|x| [(x, 254), (x, 255)]).collect();
        assert_on_slope(&g, &line);
    }

    #[test]
    fn a_cliff_a_coast_and_a_shallow_slot_are_left_alone() {
        // A 300 m cliff: a step, not a slot.
        let mut cliff = grid();
        for y in 128..SIDE {
            for x in 0..SIDE {
                *at(&mut cliff, x, y) += 300.0;
            }
        }
        let before = cliff.clone();
        assert_eq!(clean_tile(&mut cliff, 15), 0);
        assert_eq!(cliff, before);

        // A coast: land 400 m up on one side, sea at exactly 0 m on the other, in both directions.
        let mut coast: Vec<f32> = (0..SIDE * SIDE)
            .map(|i| {
                if i % SIDE < 100 {
                    0.0
                } else {
                    400.0 + (i / SIDE) as f32
                }
            })
            .collect();
        for y in 0..SIDE {
            for x in 0..SIDE {
                if y > 200 {
                    *at(&mut coast, x, y) = 0.0;
                }
            }
        }
        let before = coast.clone();
        assert_eq!(clean_tile(&mut coast, 15), 0);
        assert_eq!(coast, before);

        // A slot 100 m deep is within what the data could hold at zoom 15.
        let mut slot = grid();
        for x in 0..SIDE {
            *at(&mut slot, x, 90) -= 100.0;
        }
        let before = slot.clone();
        assert_eq!(clean_tile(&mut slot, 15), 0);
        assert_eq!(slot, before);
    }

    #[test]
    fn the_zoom_15_defect_pattern_is_real_terrain_at_zoom_10() {
        // A one-pixel canyon 4,475 m deep and a one-pixel ridge 3,000 m high, at zoom 10 (150 m a pixel).
        let mut g = grid();
        for x in 0..SIDE {
            *at(&mut g, x, 109) -= 4475.0;
            *at(&mut g, x, 180) += 3000.0;
        }
        let before = g.clone();
        assert_eq!(clean_tile(&mut g, 10), 0);
        assert_eq!(g, before);
        // At zoom 13 (19 m a pixel) the same pattern is still a defect.
        assert_eq!(clean_tile(&mut g, 13), 2 * SIDE);
    }

    #[test]
    fn random_rough_terrain_is_never_changed() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        for (zoom, amplitude) in [(15, 60.0), (14, 120.0), (12, 600.0), (9, 3000.0)] {
            let mut g: Vec<f32> = (0..SIDE * SIDE)
                .map(|_| 1500.0 + amplitude * (next() - 0.5))
                .collect();
            let before = g.clone();
            assert_eq!(clean_tile(&mut g, zoom), 0, "zoom {zoom}");
            assert_eq!(g, before);
        }
    }
}

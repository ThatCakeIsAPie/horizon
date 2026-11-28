//! A full greedy merge of a near section's solid quads that keeps the picture identical:
//! whole block faces on the grid, with one light, colour, sprite, tint, look and sprite
//! orientation each, merge into rectangles of equal faces of one plane. Faces sharing a cell
//! of their plane with another quad (overlays, layered models) never merge, nor do faces a
//! shader pack meshes (`TerrainSource` names each face's own block). `count` measures what it
//! would save (`geometry_stats`); `merge` gives the rectangles (tests: the merge is measured,
//! not drawn, as it saves under 10% of near quads, horizon-farworld-d-notes.md).
use super::{QuadLook, TerrainSource, TerrainVertex, UV_STEPS, packed::UNKNOWN_TINT};
use std::cell::RefCell;

/// One merged rectangle of a plane: `axis` across it, its `plane` coordinate, the cells
/// `[u0, w0, u1, w1)` it covers (u, w the plane's other axes as `mesher::packed` names them),
/// and the input quad it repeats.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub axis: u8,
    pub plane: u8,
    pub cells: [u8; 4],
    pub quad: u32,
}

/// What `merge` makes of a mesh: `kept` input quads unchanged, `rects` the merged faces
/// (each covering at least two cells), and how many input quads could merge at all.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Default, Debug)]
pub(crate) struct Merged {
    pub kept: Vec<u32>,
    pub rects: Vec<Rect>,
    pub candidates: usize,
}
#[cfg(test)]
impl Merged {
    /// The quads the mesh would have merged.
    pub fn quads(&self) -> usize {
        self.kept.len() + self.rects.len()
    }
}

/// A face that can merge: its group (all must be equal to merge) and cell.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    /// axis * 34 + facing * 17 + plane.
    plane: u16,
    layer: u16,
    light: u16,
    /// Where the corners' uv corners sit (8 bits) and the corner order (3 bits).
    orient: u16,
    color: u32,
    tint: u32,
    look: [u32; 2],
    /// u * 16 + w.
    cell: u8,
    quad: u32,
}

#[derive(Default)]
struct Scratch {
    candidates: Vec<Candidate>,
    /// Quads on each (plane, cell): more than one keeps them all apart.
    taken: Vec<u8>,
}
thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::default();
}

/// The plane's two other axes, as `mesher::packed` orders them.
fn plane_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    }
}

/// `quad` as an axis plane face on the block grid: (axis, plane, facing positive, the cell's
/// low (u, w) corner, each corner's (du, dw) step from it), whether it covers exactly one
/// cell. `None` off the grid or without one facing.
#[allow(clippy::type_complexity)]
fn grid_face(quad: &[TerrainVertex]) -> Option<(usize, u8, bool, [u8; 2], [[u8; 2]; 4], bool)> {
    let mut p = [[0u8; 3]; 4];
    for (to, v) in p.iter_mut().zip(quad) {
        for (c, &x) in to.iter_mut().zip(&v.pos) {
            if !(0.0..=16.0).contains(&x) {
                return None;
            }
            let n = x as u8;
            if x.to_bits() != f32::from(n).to_bits() {
                return None;
            }
            *c = n;
        }
    }
    let axis = (0..3).find(|&a| p.iter().all(|c| c[a] == p[0][a]))?;
    let (u, w) = plane_axes(axis);
    let lo = [p.iter().map(|c| c[u]).min()?, p.iter().map(|c| c[w]).min()?];
    let hi = [p.iter().map(|c| c[u]).max()?, p.iter().map(|c| c[w]).max()?];
    let d = |i: usize, j: usize| {
        let (a, b) = (p[i], p[j]);
        [
            f32::from(b[u]) - f32::from(a[u]),
            f32::from(b[w]) - f32::from(a[w]),
        ]
    };
    let (e1, e2, e3) = (d(0, 1), d(0, 2), d(0, 3));
    let first = e1[0] * e2[1] - e1[1] * e2[0];
    let second = e2[0] * e3[1] - e2[1] * e3[0];
    let positive = first > 0.0 && second > 0.0;
    let negative = first < 0.0 && second < 0.0;
    if positive == negative {
        return None;
    }
    let unit = hi[0] == lo[0] + 1 && hi[1] == lo[1] + 1 && lo[0] < 16 && lo[1] < 16;
    let mut steps = [[0u8; 2]; 4];
    for (s, c) in steps.iter_mut().zip(&p) {
        *s = [c[u] - lo[0], c[w] - lo[1]];
    }
    Some((axis, p[0][axis], positive, lo, steps, unit))
}

/// The orientation code of a unit face whose uvs are a whole sprite: bit 2k, 2k+1 = the u, v
/// (0 or 1) at the cell corner k = (du, dw) in (0,0), (1,0), (0,1), (1,1); bits 8..11 the
/// corner of the cell vertex 0 sits on and the winding. `None` when the uvs are not the
/// sprite's corners or do not map affinely (tiling would then differ).
fn orientation(quad: &[TerrainVertex], steps: &[[u8; 2]; 4]) -> Option<u16> {
    const ONE: u16 = UV_STEPS as u16;
    let mut at = [[u8::MAX; 2]; 4];
    for (v, s) in quad.iter().zip(steps) {
        let bit = |x: u16| match x {
            0 => Some(0u8),
            ONE => Some(1u8),
            _ => None,
        };
        let k = usize::from(s[0] | s[1] << 1);
        at[k] = [bit(v.uv[0])?, bit(v.uv[1])?];
    }
    // Every corner once, and uv(1,1) = uv(1,0) + uv(0,1) - uv(0,0) on each axis, with the
    // two edges along different sprite axes.
    if at.iter().any(|c| c[0] == u8::MAX) {
        return None;
    }
    for c in 0..2 {
        let (o, a, b, ab) = (
            at[0][c] as i8,
            at[1][c] as i8,
            at[2][c] as i8,
            at[3][c] as i8,
        );
        if ab != a + b - o {
            return None;
        }
    }
    let du = [at[1][0] != at[0][0], at[1][1] != at[0][1]];
    let dw = [at[2][0] != at[0][0], at[2][1] != at[0][1]];
    if du == dw || !(du[0] ^ du[1]) || !(dw[0] ^ dw[1]) {
        return None;
    }
    let mut code = 0u16;
    for (k, c) in at.iter().enumerate() {
        code |= u16::from(c[0] | c[1] << 1) << (2 * k);
    }
    // Vertex 0's cell corner and the next one's: the corner order round the face.
    let next = usize::from(steps[1][0] | steps[1][1] << 1);
    let start = usize::from(steps[0][0] | steps[0][1] << 1);
    Some(code | (start as u16) << 8 | (next as u16) << 10)
}

#[cfg(test)]
/// The merge of `solid`'s quads (4 vertices each). `tints`, `sources`, `looks` per quad, each
/// may be shorter than the quads (missing: no tint said, no source, no look) or empty.
pub(crate) fn merge(
    solid: &[TerrainVertex],
    tints: &[u32],
    sources: &[TerrainSource],
    looks: &[QuadLook],
) -> Merged {
    let mut out = Merged::default();
    run(solid, tints, sources, looks, Some(&mut out));
    out
}

/// `merge`'s (candidates, quads after) without building it.
pub(crate) fn count(
    solid: &[TerrainVertex],
    tints: &[u32],
    sources: &[TerrainSource],
    looks: &[QuadLook],
) -> (usize, usize) {
    run(solid, tints, sources, looks, None)
}

const PLANES: usize = 3 * 2 * 17;

fn run(
    solid: &[TerrainVertex],
    tints: &[u32],
    sources: &[TerrainSource],
    looks: &[QuadLook],
    mut out: Option<&mut Merged>,
) -> (usize, usize) {
    let quads = solid.len() / 4;
    SCRATCH.with_borrow_mut(|s| {
        let Scratch { candidates, taken } = s;
        candidates.clear();
        taken.clear();
        taken.resize(PLANES * 256, 0);
        let mut kept = 0usize;
        let mut keep = |q: usize, out: &mut Option<&mut Merged>| {
            kept += 1;
            if let Some(o) = out.as_deref_mut() {
                o.kept.push(q as u32);
            }
        };
        for q in 0..quads {
            let quad = &solid[q * 4..q * 4 + 4];
            let Some((axis, plane, positive, lo, steps, unit)) = grid_face(quad) else {
                keep(q, &mut out);
                continue;
            };
            let key = axis * 34 + usize::from(positive) * 17 + usize::from(plane);
            if !unit {
                // Mark every cell it touches, so no face there merges.
                let (u0, w0) = (lo[0], lo[1]);
                let (mut u1, mut w1) = (u0, w0);
                for s in &steps {
                    u1 = u1.max(u0 + s[0]);
                    w1 = w1.max(w0 + s[1]);
                }
                for u in u0..u1.max(u0 + 1).min(16) {
                    for w in w0..w1.max(w0 + 1).min(16) {
                        taken[key * 256 + usize::from(u) * 16 + usize::from(w)] = 2;
                    }
                }
                keep(q, &mut out);
                continue;
            }
            let cell = lo[0] * 16 + lo[1];
            let slot = &mut taken[key * 256 + usize::from(cell)];
            *slot = slot.saturating_add(1);
            let v = quad[0];
            let same = quad[1..]
                .iter()
                .all(|c| c.layer == v.layer && c.light == v.light && c.color == v.color);
            let orient = if same {
                orientation(quad, &steps)
            } else {
                None
            };
            let (Some(orient), true) = (orient, sources.get(q).is_none()) else {
                keep(q, &mut out);
                continue;
            };
            candidates.push(Candidate {
                plane: key as u16,
                layer: v.layer,
                light: v.light,
                orient,
                color: v.color,
                tint: tints.get(q).copied().unwrap_or(UNKNOWN_TINT),
                look: looks.get(q).map_or([0; 2], |l| l.0),
                cell,
                quad: q as u32,
            });
        }
        // Faces sharing a cell with another quad of their plane stay as they are.
        candidates.retain(|c| {
            let alone = taken[usize::from(c.plane) * 256 + usize::from(c.cell)] == 1;
            if !alone {
                keep(c.quad as usize, &mut out);
            }
            alone
        });
        let total = candidates.len();
        candidates.sort_unstable();
        let mut after = 0usize;
        let mut i = 0;
        while i < candidates.len() {
            let head = candidates[i];
            let same = |c: &Candidate| {
                c.plane == head.plane
                    && c.layer == head.layer
                    && c.light == head.light
                    && c.orient == head.orient
                    && c.color == head.color
                    && c.tint == head.tint
                    && c.look == head.look
            };
            let mut j = i + 1;
            while j < candidates.len() && same(&candidates[j]) {
                j += 1;
            }
            let group = &candidates[i..j];
            if group.len() == 1 {
                after += 1;
                if let Some(o) = out.as_deref_mut() {
                    o.kept.push(head.quad);
                }
                i = j;
                continue;
            }
            let mut grid = [0u16; 16];
            for c in group {
                grid[usize::from(c.cell >> 4)] |= 1 << (c.cell & 15);
            }
            let axis = (head.plane / 34) as u8;
            let plane = (head.plane % 17) as u8;
            after += best(grid, |r| {
                if let Some(o) = out.as_deref_mut() {
                    let [u0, w0, u1, w1] = r;
                    let at = group.partition_point(|c| c.cell < u0 * 16 + w0);
                    let quad = group[at].quad;
                    if u1 - u0 == 1 && w1 - w0 == 1 {
                        o.kept.push(quad);
                    } else {
                        o.rects.push(Rect {
                            axis,
                            plane,
                            cells: r,
                            quad,
                        });
                    }
                }
            });
            i = j;
        }
        if let Some(o) = out.as_deref_mut() {
            o.candidates = total;
        }
        (total, kept + after)
    })
}

/// Greedy rectangles of `grid` (row u = bits w): rows first, or the transposed way when that
/// leaves fewer. Calls `emit([u0, w0, u1, w1))` for each of the fewer; returns how many.
fn best(grid: [u16; 16], mut emit: impl FnMut([u8; 4])) -> usize {
    let mut t = [0u16; 16];
    for (u, row) in grid.iter().enumerate() {
        for w in 0..16 {
            if row & (1 << w) != 0 {
                t[w] |= 1 << u;
            }
        }
    }
    let a = greedy(grid, |_| {});
    let b = greedy(t, |_| {});
    if b < a {
        greedy(t, |[w0, u0, w1, u1]| emit([u0, w0, u1, w1]))
    } else {
        greedy(grid, emit)
    }
}

/// Takes the first set cell, widens along w, then grows along u while the rows hold it.
fn greedy(mut grid: [u16; 16], mut emit: impl FnMut([u8; 4])) -> usize {
    let mut n = 0;
    for u0 in 0..16 {
        while grid[u0] != 0 {
            let w0 = grid[u0].trailing_zeros();
            let run = (grid[u0] >> w0).trailing_ones();
            let mask = (((1u32 << run) - 1) << w0) as u16;
            let mut u1 = u0 + 1;
            while u1 < 16 && grid[u1] & mask == mask {
                u1 += 1;
            }
            for row in &mut grid[u0..u1] {
                *row &= !mask;
            }
            emit([u0 as u8, w0 as u8, u1 as u8, (w0 + run) as u8]);
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// A unit face of cell (u, w) of `plane` on `axis`, facing positive or not, its uv corner
    /// (0,0) at the cell's low corner, u along the plane's u axis.
    fn face(
        axis: usize,
        plane: u8,
        positive: bool,
        u: u8,
        w: u8,
        attr: [u32; 3],
    ) -> [TerrainVertex; 4] {
        let (ua, wa) = plane_axes(axis);
        let corner = |du: u8, dw: u8| {
            let mut pos = [0.0f32; 3];
            pos[axis] = f32::from(plane);
            pos[ua] = f32::from(u + du);
            pos[wa] = f32::from(w + dw);
            TerrainVertex {
                pos,
                uv: [u16::from(du) * 32768, u16::from(dw) * 32768],
                layer: attr[0] as u16,
                light: attr[1] as u16,
                color: attr[2],
            }
        };
        // (u, w) then a positive turn faces +axis (u x w = axis).
        let ring = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
        if positive {
            ring
        } else {
            [ring[0], ring[3], ring[2], ring[1]]
        }
    }

    fn mesh(faces: &[[TerrainVertex; 4]]) -> Vec<TerrainVertex> {
        faces.iter().flatten().copied().collect()
    }

    /// Each covered cell of the mesh: (axis, plane, facing, cell) -> (layer, light, color,
    /// tint, look, uv at the cell's low corner mod 1, uv step along u and w).
    type Cells = BTreeMap<
        (u8, u8, bool, u8, u8),
        Vec<(u16, u16, u32, u32, [u32; 2], [i32; 2], [[i32; 2]; 2])>,
    >;

    fn cells_of_quad(
        q: &[TerrainVertex],
        tint: u32,
        look: [u32; 2],
        to: &mut Cells,
        rect: Option<[u8; 4]>,
    ) {
        let (axis, plane, positive, lo, steps, _) = grid_face(q).unwrap();
        let one = 32768i32;
        let k = |s: [u8; 2]| {
            q[steps.iter().position(|&t| t == s).unwrap()]
                .uv
                .map(i32::from)
        };
        let (o, a, b) = (k([0, 0]), k([1, 0]), k([0, 1]));
        let du = [(a[0] - o[0]) / one, (a[1] - o[1]) / one];
        let dw = [(b[0] - o[0]) / one, (b[1] - o[1]) / one];
        let [u0, w0, u1, w1] = rect.unwrap_or([lo[0], lo[1], lo[0] + 1, lo[1] + 1]);
        for u in u0..u1 {
            for w in w0..w1 {
                let (iu, iw) = (i32::from(u - u0), i32::from(w - w0));
                let uv = [
                    (o[0] / one + du[0] * iu + dw[0] * iw).rem_euclid(1),
                    (o[1] / one + du[1] * iu + dw[1] * iw).rem_euclid(1),
                ];
                to.entry((axis as u8, plane, positive, u, w))
                    .or_default()
                    .push((q[0].layer, q[0].light, q[0].color, tint, look, uv, [du, dw]));
            }
        }
    }

    /// The cells the mesh shows, before and after `merge`.
    fn coverage(
        solid: &[TerrainVertex],
        tints: &[u32],
        looks: &[QuadLook],
    ) -> (Cells, Cells, Merged) {
        let m = merge(solid, tints, &[], looks);
        let attr = |q: usize| {
            (
                tints.get(q).copied().unwrap_or(UNKNOWN_TINT),
                looks.get(q).map_or([0; 2], |l| l.0),
            )
        };
        let mut before = Cells::new();
        for q in 0..solid.len() / 4 {
            let (t, l) = attr(q);
            cells_of_quad(&solid[q * 4..q * 4 + 4], t, l, &mut before, None);
        }
        let mut after = Cells::new();
        for &q in &m.kept {
            let (t, l) = attr(q as usize);
            let q = q as usize;
            cells_of_quad(&solid[q * 4..q * 4 + 4], t, l, &mut after, None);
        }
        for r in &m.rects {
            let (t, l) = attr(r.quad as usize);
            let q = r.quad as usize;
            cells_of_quad(&solid[q * 4..q * 4 + 4], t, l, &mut after, Some(r.cells));
        }
        for v in before.values_mut().chain(after.values_mut()) {
            v.sort();
        }
        (before, after, m)
    }

    #[test]
    fn greedy_keeps_coverage_identical() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for _ in 0..200 {
            let mut faces = Vec::new();
            let mut tints = Vec::new();
            let mut looks = Vec::new();
            let mut used = BTreeSet::new();
            for _ in 0..600 {
                let axis = next(3) as usize;
                let plane = next(3) as u8 + 7;
                let positive = next(2) == 0;
                let (u, w) = (next(16) as u8, next(16) as u8);
                // A few overlays: the same cell twice.
                if !used.insert((axis, plane, positive, u, w)) && next(4) != 0 {
                    continue;
                }
                let attr = [
                    next(2) as u32 + 1,
                    next(2) as u32,
                    0xff80_8080 | next(2) as u32,
                ];
                faces.push(face(axis, plane, positive, u, w, attr));
                tints.push(next(2) as u32);
                looks.push(QuadLook([next(2) as u32, 0]));
            }
            let solid = mesh(&faces);
            let (before, after, m) = coverage(&solid, &tints, &looks);
            assert_eq!(before, after);
            assert!(m.quads() <= faces.len());
            assert_eq!(
                m.kept.len()
                    + m.rects
                        .iter()
                        .map(|r| {
                            let [u0, w0, u1, w1] = r.cells;
                            usize::from(u1 - u0) * usize::from(w1 - w0)
                        })
                        .sum::<usize>(),
                faces.len()
            );
        }
    }

    #[test]
    fn greedy_never_merges_different_attributes() {
        let base = [3, 7, 0xffff_ffff];
        let pairs: [([u32; 3], u32, u32); 5] = [
            ([4, 7, 0xffff_ffff], 0, 0),
            ([3, 8, 0xffff_ffff], 0, 0),
            ([3, 7, 0xfffe_ffff], 0, 0),
            (base, 1, 0),
            (base, 0, 1),
        ];
        for (other, tint, look) in pairs {
            let faces = [face(1, 5, true, 2, 2, base), face(1, 5, true, 3, 2, other)];
            let m = merge(
                &mesh(&faces),
                &[0, tint],
                &[],
                &[QuadLook([0, 0]), QuadLook([look, 0])],
            );
            assert_eq!(m.quads(), 2, "{other:?} {tint} {look}");
            assert!(m.rects.is_empty());
        }
        // Equal: they merge.
        let faces = [face(1, 5, true, 2, 2, base), face(1, 5, true, 3, 2, base)];
        assert_eq!(merge(&mesh(&faces), &[], &[], &[]).quads(), 1);
        // A source each (a shader pack's mesh): never.
        let src = TerrainSource::new(1, [2, 5, 2], 0, false);
        assert_eq!(merge(&mesh(&faces), &[], &[src, src], &[]).quads(), 2);
        // The sprite turned on one of them.
        let mut turned = face(1, 5, true, 3, 2, base);
        for v in &mut turned {
            v.uv = [v.uv[1], v.uv[0]];
        }
        let faces = [face(1, 5, true, 2, 2, base), turned];
        assert_eq!(merge(&mesh(&faces), &[], &[], &[]).quads(), 2);
        // Facing apart.
        let faces = [face(1, 5, true, 2, 2, base), face(1, 5, false, 3, 2, base)];
        assert_eq!(merge(&mesh(&faces), &[], &[], &[]).quads(), 2);
    }

    #[test]
    fn greedy_passes_other_quads_through() {
        let base = [3, 7, 0xffff_ffff];
        let mut ao = face(1, 5, true, 1, 1, base);
        ao[2].color = 0xff80_8080;
        let mut half = face(1, 5, true, 2, 1, base);
        for v in &mut half {
            v.pos[0] = v.pos[0].min(1.5);
        }
        let over = [
            face(1, 5, true, 4, 1, base),
            face(1, 5, true, 4, 1, [9, 7, 0xffff_ffff]),
        ];
        let mut partial_uv = face(1, 5, true, 6, 1, base);
        partial_uv[1].uv[0] = 16384;
        let mut faces = vec![ao, half, partial_uv];
        faces.extend(over);
        // Neighbours of every one, which could merge with them were they candidates.
        for u in [0u8, 3, 5, 7] {
            faces.push(face(1, 5, true, u, 1, base));
        }
        let m = merge(&mesh(&faces), &[], &[], &[]);
        assert_eq!(m.candidates, 4);
        assert_eq!(m.quads(), faces.len());
        for q in 0..5 {
            assert!(m.kept.contains(&q));
        }
    }

    #[test]
    fn greedy_flat_top_is_one_quad() {
        let mut faces = Vec::new();
        for u in 0..16 {
            for w in 0..16 {
                faces.push(face(1, 16, true, u, w, [5, 15, 0xffff_ffff]));
            }
        }
        let m = merge(&mesh(&faces), &[], &[], &[]);
        assert_eq!(m.quads(), 1);
        assert_eq!(m.rects[0].cells, [0, 0, 16, 16]);
        assert_eq!(count(&mesh(&faces), &[], &[], &[]), (256, 1));
        // An L: two rects, either way round.
        let l: Vec<_> = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]
            .iter()
            .map(|&(u, w)| face(0, 3, false, u, w, [5, 15, 0xffff_ffff]))
            .collect();
        assert_eq!(merge(&mesh(&l), &[], &[], &[]).quads(), 2);
    }

    #[test]
    fn greedy_rects_face_their_inputs() {
        for axis in 0..3 {
            for positive in [true, false] {
                let faces = [
                    face(axis, 4, positive, 0, 0, [1, 1, 1]),
                    face(axis, 4, positive, 1, 0, [1, 1, 1]),
                ];
                let solid = mesh(&faces);
                let m = merge(&solid, &[], &[], &[]);
                let r = m.rects[0];
                let q = &solid[r.quad as usize * 4..][..4];
                let (a, _, p, _, _, _) = grid_face(q).unwrap();
                assert_eq!((usize::from(r.axis), r.plane, p), (axis, 4, positive));
                // Independent: the cross product of the face's first two edges.
                let e =
                    |i: usize| -> [f32; 3] { std::array::from_fn(|c| q[i].pos[c] - q[0].pos[c]) };
                let (x, y) = (e(1), e(2));
                let n = [
                    x[1] * y[2] - x[2] * y[1],
                    x[2] * y[0] - x[0] * y[2],
                    x[0] * y[1] - x[1] * y[0],
                ];
                assert_eq!(n[a] > 0.0, positive);
            }
        }
    }
}

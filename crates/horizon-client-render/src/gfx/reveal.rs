//! Windows in which far cube nodes dither in front of near cubes that come in whole or
//! are held whole while they leave.
//!
//! A near cube never dithers (the holes of its pattern showed its own inside, caves and
//! all): it comes in whole over the far node that covered its place, and the far node
//! dithers out in front of it, so every hole of the pattern shows finished terrain, the
//! cube or the node's own mesh drawn behind. When cubes leave, they are held whole and
//! the node dithers in in front of them (`Renderer::open_cube_window`).
//!
//! The unit is the cube, not the node: cubes of a cubic world come in over seconds, and a
//! window per node that started over for each one would cover the cubes it had shown
//! already (a flash). Each cube has a window of its own, a box in the world with its start;
//! a shell's pixels look up the window they lie in (terrain.wgsl's `reveal_window`), so
//! each part of the shell dithers on the clock of the cube behind it, and a pixel outside
//! every window is not drawn: behind it is the same mesh, whole. Cubes that come in
//! together (the windows of one frame, next to one another) share one window.
use glam::DVec3;
use std::time::Instant;

/// A shell's fade code (terrain.wgsl's `REVEAL`): the node's level and whether it dithers out
/// (over cubes that came in) or in (over cubes held while they leave); the pixels take their
/// progress from the windows.
pub(super) fn shell_code(level: u8, out: bool) -> u32 {
    REVEAL + 1 + u32::from(level) * 129 + if out { 64 } else { 0 }
}
/// The coarsest far cube node (level: voxel of `2^level` blocks) that gets a shell. A shell is
/// drawn up to 3/8 of a voxel (at most 8% of the distance) toward the camera so it wins over
/// the cubes behind it; for a node many levels coarser than the terrain it hands over to,
/// that is a block-shaped slab standing in front of correct near terrain for the whole
/// window, and cubes keep arriving while far refinement lags. Levels 0 to 2 (voxels of 1, 2
/// and 4 blocks, the cap of `margin` and of terrain.wgsl's `reveal_leaves`) still look like the
/// terrain they hand over to; coarser nodes get no window and no shell: the cube is whole at
/// once over them, as before the dither. Keep `MAX_SHELL_LEVEL` in cube_cull.wgsl equal.
pub(super) const MAX_SHELL_LEVEL: u8 = 2;
/// How far a window is grown for a shell's pixels, in blocks (terrain.wgsl's `reveal_leaves`).
pub(super) fn margin(level: u8) -> f64 {
    f64::from(1u32 << level.min(2))
}
pub(super) const REVEAL: u32 = 1 << 13;

/// Windows the shaders read at once; when more are open the oldest, nearly over, close.
pub(super) const MAX_WINDOWS: usize = 64;

/// How near in time (ms) and space (blocks) a cube must be to an open window of the
/// same kind to share it, and the longest side a shared window grows to.
const MERGE_MS: u128 = 120;
const MERGE_GAP: f64 = 16.0;
const MERGE_SIDE: f64 = 96.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Window {
    pub lo: DVec3,
    pub hi: DVec3,
    pub since: Instant,
    /// Cubes came in (the shell dithers out), else they leave (it dithers in).
    pub out: bool,
}

/// What the shaders read: per window `[lo.xyz, progress]` then `[hi.xyz, out]`, relative to
/// the camera, and how many are set.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct Table {
    pub boxes: [[f32; 4]; 2 * MAX_WINDOWS],
    pub count: [u32; 4],
}

#[derive(Default)]
pub(super) struct Reveal {
    windows: Vec<Window>,
    /// Counts for `HORIZON_FADE_STATS`: windows opened with a shell to dither, closed
    /// early for room, and by kind.
    pub opened: [u32; 2],
    pub evicted: u32,
}

impl Reveal {
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
    pub fn len(&self) -> usize {
        self.windows.len()
    }
    #[cfg(test)]
    pub fn windows(&self) -> &[Window] {
        &self.windows
    }
    /// Drops the windows `fade` long gone.
    pub fn expire(&mut self, now: Instant, fade: std::time::Duration) {
        self.windows
            .retain(|w| now.saturating_duration_since(w.since) < fade);
    }
    pub fn clear(&mut self) {
        self.windows.clear();
    }
    /// A cube's box comes in (`out`) or leaves at `now`; the start of the window that
    /// carries it. A window of the same kind that began just before and lies next to it
    /// is joined (`MERGE_MS`), growing to hold it: the cubes of a burst share a clock.
    pub fn open(&mut self, lo: DVec3, hi: DVec3, now: Instant, out: bool) -> Instant {
        let joined = self.windows.iter_mut().rev().find(|w| {
            w.out == out
                && now.saturating_duration_since(w.since).as_millis() <= MERGE_MS
                && (lo - MERGE_GAP).cmple(w.hi).all()
                && (hi + MERGE_GAP).cmpge(w.lo).all()
                && (hi.max(w.hi) - lo.min(w.lo)).max_element() <= MERGE_SIDE
        });
        if let Some(w) = joined {
            w.lo = w.lo.min(lo);
            w.hi = w.hi.max(hi);
            return w.since;
        }
        if self.windows.len() >= MAX_WINDOWS {
            // The oldest is the nearest to over.
            let oldest = self
                .windows
                .iter()
                .enumerate()
                .min_by_key(|(_, w)| w.since)
                .map(|(i, _)| i)
                .expect("full");
            self.windows.swap_remove(oldest);
            self.evicted += 1;
        }
        self.opened[usize::from(out)] += 1;
        self.windows.push(Window {
            lo,
            hi,
            since: now,
            out,
        });
        now
    }
    /// Whether a window of this kind, grown by `margin` blocks, meets the box.
    pub fn touches(&self, lo: DVec3, hi: DVec3, out: bool, margin: f64) -> bool {
        self.windows
            .iter()
            .any(|w| w.out == out && lo.cmple(w.hi + margin).all() && hi.cmpge(w.lo - margin).all())
    }
    /// The windows of both kinds that meet the box: `[out, in]`.
    pub fn touching(&self, lo: DVec3, hi: DVec3, margin: f64) -> [bool; 2] {
        [
            self.touches(lo, hi, true, margin),
            self.touches(lo, hi, false, margin),
        ]
    }
    /// The windows as the shaders read them this frame.
    pub fn table(&self, camera: DVec3, now: Instant, fade: std::time::Duration) -> Table {
        let mut table = Table::zeroed();
        for (i, w) in self.windows.iter().take(MAX_WINDOWS).enumerate() {
            let progress = (now.saturating_duration_since(w.since).as_secs_f32()
                / fade.as_secs_f32())
            .min(1.0);
            let lo = (w.lo - camera).as_vec3();
            let hi = (w.hi - camera).as_vec3();
            table.boxes[i * 2] = [lo.x, lo.y, lo.z, progress];
            table.boxes[i * 2 + 1] = [hi.x, hi.y, hi.z, f32::from(u8::from(w.out))];
            table.count[0] = i as u32 + 1;
        }
        table
    }
}

impl Table {
    pub fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn only_fine_nodes_get_shells_and_the_culling_pass_agrees() {
        assert!(MAX_SHELL_LEVEL <= 2, "margin() and reveal_leaves stop growing at 4 blocks");
        let wgsl = include_str!("cube_cull.wgsl");
        assert!(
            wgsl.contains(&format!("const MAX_SHELL_LEVEL: u32 = {MAX_SHELL_LEVEL}u;")),
            "cube_cull.wgsl's MAX_SHELL_LEVEL differs"
        );
    }

    fn cube(x: f64) -> (DVec3, DVec3) {
        (DVec3::new(x, 0.0, 0.0), DVec3::new(x + 16.0, 16.0, 16.0))
    }

    #[test]
    fn cubes_of_one_burst_share_a_window_and_late_ones_get_their_own() {
        let t0 = Instant::now();
        let mut reveal = Reveal::default();
        let (lo, hi) = cube(0.0);
        assert_eq!(reveal.open(lo, hi, t0, true), t0);
        // Next to it, 50 ms later: joined, on the first one's clock.
        let (lo, hi) = cube(16.0);
        assert_eq!(
            reveal.open(lo, hi, t0 + Duration::from_millis(50), true),
            t0
        );
        assert_eq!(reveal.len(), 1);
        assert_eq!(reveal.windows()[0].hi.x, 32.0);
        // A cube a second later has a window of its own and starts at its own time: the
        // first one's shell, half dithered out by then, is not covered again.
        let later = t0 + Duration::from_millis(1000);
        let (lo, hi) = cube(32.0);
        assert_eq!(reveal.open(lo, hi, later, true), later);
        assert_eq!(reveal.len(), 2);
        // A cube leaving is another kind: never joins an arrival.
        let (lo, hi) = cube(48.0);
        reveal.open(lo, hi, later, false);
        assert_eq!(reveal.len(), 3);
        // Far apart: its own.
        let (lo, hi) = cube(1000.0);
        reveal.open(lo, hi, later, true);
        assert_eq!(reveal.len(), 4);
    }

    #[test]
    fn windows_end_and_the_table_carries_their_progress() {
        let t0 = Instant::now();
        let fade = Duration::from_millis(600);
        let mut reveal = Reveal::default();
        let (lo, hi) = cube(0.0);
        reveal.open(lo, hi, t0, true);
        let (lo, hi) = cube(500.0);
        reveal.open(lo, hi, t0 + Duration::from_millis(300), false);
        let table = reveal.table(
            DVec3::new(8.0, 8.0, 8.0),
            t0 + Duration::from_millis(300),
            fade,
        );
        assert_eq!(table.count[0], 2);
        assert_eq!(table.boxes[0], [-8.0, -8.0, -8.0, 0.5]);
        assert_eq!(table.boxes[1], [8.0, 8.0, 8.0, 1.0]);
        assert_eq!(table.boxes[2][3], 0.0);
        assert_eq!(table.boxes[3][3], 0.0, "the one leaving");
        reveal.expire(t0 + Duration::from_millis(700), fade);
        assert_eq!(reveal.len(), 1, "the first ended");
        assert!(reveal.touches(
            DVec3::new(500.0, 0.0, 0.0),
            DVec3::new(510.0, 8.0, 8.0),
            false,
            0.0
        ));
        assert!(!reveal.touches(
            DVec3::new(500.0, 0.0, 0.0),
            DVec3::new(510.0, 8.0, 8.0),
            true,
            0.0
        ));
    }

    #[test]
    fn a_full_table_closes_the_oldest_window() {
        let t0 = Instant::now();
        let mut reveal = Reveal::default();
        for i in 0..MAX_WINDOWS + 3 {
            let (lo, hi) = cube(i as f64 * 1000.0);
            reveal.open(lo, hi, t0 + Duration::from_millis(i as u64 * 200), true);
        }
        assert_eq!(reveal.len(), MAX_WINDOWS);
        assert_eq!(reveal.evicted, 3);
        assert!(
            reveal
                .windows()
                .iter()
                .all(|w| w.since >= t0 + Duration::from_millis(600))
        );
    }
}

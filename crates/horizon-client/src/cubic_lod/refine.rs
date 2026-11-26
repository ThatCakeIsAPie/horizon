//! The far view's second pass: coarse first, refined later (docs/earth.md, "Far view").
//!
//! An Earth node with trees is first made without them (its ground, colour and snow: the terrain as it was
//! before far trees, as fast), so the distance is never empty while trees are worked out. Each such node is
//! noted here as coarse; once every planned node is drawn the producer asks for the nodes to make again
//! ([`LateContent`]), and this hands out the coarse ones nearest the camera and in view first, a few at a
//! time, so they go behind all other far work. A node made again is made whole (with its trees) and replaces
//! the coarse one in place. Whole nodes are kept in the far cache as final, coarse ones as coarse
//! (`lod::cube_cache`): a revisit draws what was refined whole at once, and the rest coarse at once, refined again.
//!
//! Coarse first is only for where nothing is drawn: the distance is never empty, so a hole (and the coverage
//! nodes over it, or every node in a big fill, a join or a teleport) is made coarse at once. Anywhere else the
//! node being made replaces something that is drawn and stays drawn until the new node is ready, so it is
//! made whole at once: no blank frame, and one build instead of a coarse one and a whole one. The producer
//! tells which nodes must come coarse first ([`LateContent::coarse_first`]); until it has, everything does.
//! A node the producer is about to split into its children is not made whole now ([`LateContent::not_now`]):
//! it goes back among the coarse ones, and is not handed out again for a short while.

use crate::lod::cube::{CubeKey, LateContent, Look, job_score};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Nodes handed out at a time: the next batch waits until these are made, so the order follows the camera.
const BATCH: usize = 48;
/// A node handed out but not made in this long was not made again (the plan dropped it): forgotten.
const HANDED_FOR: Duration = Duration::from_secs(30);
/// A node given back ([`LateContent::not_now`]) is not handed out again for this long. Without it the
/// producer, which splits the node a moment later, would be asked for it again at once and give it back
/// again: a loop that fills every batch with the same nodes and starves the others. After it, if the node is
/// still drawn (it did not split), it is made whole as any other.
const DEFER_FOR: Duration = Duration::from_millis(500);

#[derive(Default)]
struct State {
    /// Drawn coarse, not handed out yet.
    coarse: HashSet<CubeKey>,
    /// Handed out to be made whole, not made yet, and when.
    handed: HashMap<CubeKey, Instant>,
    /// Made whole: made whole straight away when asked for again.
    whole: HashSet<CubeKey>,
    /// Given back by the producer (it is about to split them) and when: not handed out for [`DEFER_FOR`].
    deferred: HashMap<CubeKey, Instant>,
    /// Nodes that must be made coarse first (nothing is drawn over their place), as the producer said.
    urgent: HashSet<CubeKey>,
    /// Every node must be made coarse first (a big fill, a join, a teleport, the coverage phase).
    all: bool,
    /// The producer has told [`State::urgent`] and [`State::all`] at least once; until then every node is
    /// made coarse first (nothing is known of what is drawn).
    heard: bool,
    camera: [f64; 3],
    forward: [f64; 3],
    max_distance: f64,
}

/// The coarse nodes and the order they are refined in, shared by every fork of the sampler.
pub struct Refine {
    state: Mutex<State>,
    /// Make a node whole at once where something is drawn over its place ([`Refine::pass`]).
    whole_when_covered: bool,
}

/// Whether [`Refine::pass`] makes nodes whole at once where something is drawn over their place
/// (`HORIZON_FAR_WHOLE_WHEN_COVERED`). Off: measured on a 60 s forest walk (6 workers) it cut the far jobs
/// from 28.5k to 20.4k but each whole build costs more, so the workers stayed as busy, the far view fell
/// 3.7k nodes behind the walk (219 with it off) and the first fill took 220 s instead of 109 s.
pub const WHOLE_WHEN_COVERED: u64 = 0;

impl Default for Refine {
    fn default() -> Self {
        Self {
            state: Mutex::default(),
            whole_when_covered: crate::lod::cube::far_knob(
                "HORIZON_FAR_WHOLE_WHEN_COVERED",
                WHOLE_WHEN_COVERED,
            ) != 0,
        }
    }
}

/// How node `key` should be made now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Coarse,
    Whole,
}

impl Refine {
    /// Whether node `key` is made coarse (first) or whole: whole when it was handed out, was made whole
    /// before, or something is drawn over its place already.
    ///
    /// Distance is never empty, so where there is a hole a node comes coarse first (the producer lists those:
    /// `urgent`, or everything with `all`). Elsewhere a drawn piece is swapped only when its replacement is
    /// ready, so the node being made now replaces one that stays drawn until it is done: making it whole at
    /// once costs no blank frame and saves the second build. (`HORIZON_FAR_WHOLE_WHEN_COVERED`, default off,
    /// see [`WHOLE_WHEN_COVERED`]; off: coarse first always, as before.) Before the producer has said
    /// anything, nothing is known to be drawn, so it is coarse.
    pub fn pass(&self, key: CubeKey) -> Pass {
        let s = self.state.lock().unwrap();
        if s.handed.contains_key(&key) || s.whole.contains(&key) {
            return Pass::Whole;
        }
        if self.whole_when_covered && s.heard && !s.all && !s.urgent.contains(&key) {
            Pass::Whole
        } else {
            Pass::Coarse
        }
    }

    /// Node `key` was made: coarse (`whole` false, to be refined), or whole. `made` false when a whole build
    /// could not be made yet (data missing): it goes back to the coarse ones and is handed out again.
    pub fn made(&self, key: CubeKey, whole: bool, made: bool) {
        use crate::lod::cube::profile;
        use std::sync::atomic::Ordering::Relaxed;
        let mut s = self.state.lock().unwrap();
        match (whole, made) {
            (false, true) => {
                s.coarse.insert(key);
                profile::R_COARSE.fetch_add(1, Relaxed);
            }
            (true, true) => {
                let refresh = s.handed.remove(&key).is_some();
                s.coarse.remove(&key);
                s.deferred.remove(&key);
                s.whole.insert(key);
                let counter = if refresh {
                    &profile::R_WHOLE_REFRESH
                } else {
                    &profile::R_WHOLE_DIRECT
                };
                counter.fetch_add(1, Relaxed);
            }
            (true, false) => {
                if s.handed.remove(&key).is_some() {
                    s.coarse.insert(key);
                }
            }
            (false, false) => {}
        }
    }

    /// Coarse nodes still to refine and nodes handed out.
    #[cfg(test)]
    pub fn counts(&self) -> (usize, usize) {
        let s = self.state.lock().unwrap();
        (s.coarse.len(), s.handed.len())
    }

    /// Boxes the producer gave back ([`LateContent::not_now`]): the handed-out nodes among them go back to the
    /// coarse ones, deferred from `now`.
    fn defer(s: &mut State, boxes: &[([i32; 3], [i32; 3])], now: Instant) {
        for (min, max) in boxes {
            if let Some(key) = CubeKey::of_box(*min, *max)
                && s.handed.remove(&key).is_some()
            {
                s.coarse.insert(key);
                s.deferred.insert(key, now);
            }
        }
    }

    /// The next coarse nodes to refine: by the far view's work order (the crosshair first, then the near).
    fn next(s: &mut State, now: Instant) -> Vec<CubeKey> {
        s.handed
            .retain(|_, at| now.duration_since(*at) < HANDED_FOR);
        let (camera, forward) = (s.camera, s.forward);
        // Coarse nodes left far behind (a teleport) are out of the plan: forgotten, made coarse
        // again if they come back.
        let reach = s.max_distance * 2.0;
        if reach > 0.0 {
            s.coarse.retain(|key| {
                let (o, size) = (key.origin(), f64::from(key.size()));
                let d: [f64; 3] = std::array::from_fn(|i| f64::from(o[i]) + size / 2.0 - camera[i]);
                d[0].abs().max(d[2].abs()) <= reach + size
            });
        }
        let room = BATCH.saturating_sub(s.handed.len());
        // Nodes given back a moment ago are skipped (see `DEFER_FOR`).
        s.deferred
            .retain(|_, at| now.duration_since(*at) < DEFER_FOR);
        if room == 0 || s.coarse.is_empty() {
            return Vec::new();
        }
        // The far view's work order (`job_score`: the crosshair first, half as much the near), highest first.
        let look = Look {
            forward,
            ..Look::default()
        };
        let score = |key: &CubeKey| -job_score(camera, look, *key);
        let mut keys: Vec<(f64, CubeKey)> = s
            .coarse
            .iter()
            .filter(|k| !s.deferred.contains_key(k))
            .map(|k| (score(k), *k))
            .collect();
        if keys.is_empty() {
            return Vec::new();
        }
        let room = room.min(keys.len());
        keys.select_nth_unstable_by(room - 1, |a, b| a.0.total_cmp(&b.0));
        keys.truncate(room);
        keys.sort_by(|a, b| a.0.total_cmp(&b.0));
        let keys: Vec<CubeKey> = keys.into_iter().map(|(_, k)| k).collect();
        for key in &keys {
            s.coarse.remove(key);
            s.handed.insert(*key, now);
        }
        keys
    }
}

impl LateContent for Refine {
    fn view(&self, camera: [f64; 3], forward: [f64; 3], max_distance: f64, _complete: bool) {
        let mut s = self.state.lock().unwrap();
        s.camera = camera;
        s.forward = forward;
        s.max_distance = max_distance;
    }

    /// A handed-out node the producer has no drawn node for (the plan dropped it since it was made coarse, as
    /// the nodes near the player are once their cubes load) is never made: it is forgotten at once, so it does
    /// not hold a place in the batch until [`HANDED_FOR`] runs out. Before this, such nodes held the batch
    /// nearly full and the far view stayed without trees.
    fn not_drawn(&self, boxes: &[([i32; 3], [i32; 3])]) {
        let mut s = self.state.lock().unwrap();
        for (min, max) in boxes {
            if let Some(key) = CubeKey::of_box(*min, *max) {
                s.handed.remove(&key);
            }
        }
    }

    fn exact(&self) -> bool {
        true
    }

    fn wants_coarse_first(&self) -> bool {
        self.whole_when_covered
    }

    /// Nodes with nothing drawn over their place (or every node: `all`) are made coarse first, the rest whole
    /// at once (see [`Refine::pass`]).
    fn coarse_first(&self, keys: &[CubeKey], all: bool) {
        let mut s = self.state.lock().unwrap();
        s.urgent = keys.iter().copied().collect();
        s.all = all;
        s.heard = true;
    }

    /// Handed-out nodes the producer is about to split: not made now, back among the coarse ones, and not
    /// handed out again for [`DEFER_FOR`] (they are made whole then if they are still drawn).
    fn not_now(&self, boxes: &[([i32; 3], [i32; 3])]) {
        let mut s = self.state.lock().unwrap();
        Self::defer(&mut s, boxes, Instant::now());
    }

    fn take_refreshes(&self) -> Vec<([i32; 3], [i32; 3])> {
        let mut s = self.state.lock().unwrap();
        Self::next(&mut s, Instant::now())
            .into_iter()
            .map(|key| {
                let (o, size) = (key.origin(), key.size());
                (o, std::array::from_fn(|i| o[i].saturating_add(size - 1)))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_nodes_are_refined_nearest_and_in_view_first_a_batch_at_a_time() {
        let refine = Refine::default();
        // A row of level-2 nodes (64 blocks) along x, either side of the camera at 0.
        let keys: Vec<CubeKey> = (-100..100).map(|x| CubeKey::new(2, x, 0, 0)).collect();
        for &key in &keys {
            assert_eq!(refine.pass(key), Pass::Coarse);
            refine.made(key, false, true);
        }
        // The camera at x = 10 in node 0, looking along +x.
        refine.view([10.0, 32.0, 32.0], [1.0, 0.0, 0.0], 4096.0, true);
        let first = refine.take_refreshes();
        assert_eq!(first.len(), BATCH);
        // The far view's work order: the node round the camera and the one touching it (10 blocks behind, its
        // edge under the crosshair), then the ones ahead, nearest first; the rest behind come after all ahead.
        let order: Vec<i32> = first.iter().map(|(min, _)| min[0] >> 6).collect();
        let expected: Vec<i32> = [0, -1].into_iter().chain(1..=46).collect();
        assert_eq!(order, expected);
        assert_eq!(first[0].1, [63, 63, 63], "the box is the node's blocks");
        // Nothing more until the batch is made.
        assert!(refine.take_refreshes().is_empty());
        let handed = CubeKey::new(2, 0, 0, 0);
        assert_eq!(refine.pass(handed), Pass::Whole);
        // A whole build that could not be made goes back to be handed out again.
        refine.made(handed, true, false);
        assert_eq!(refine.pass(handed), Pass::Coarse);
        assert_eq!(refine.take_refreshes(), vec![([0, 0, 0], [63, 63, 63])]);
        // Made whole: whole from now on, and the next batch comes.
        for (min, _) in &first {
            refine.made(CubeKey::new(2, min[0] >> 6, 0, 0), true, true);
        }
        assert_eq!(refine.pass(handed), Pass::Whole);
        assert_eq!(refine.take_refreshes().len(), BATCH);
        assert_eq!(refine.counts(), (keys.len() - 2 * BATCH, BATCH));
        // Handed out and never made (the plan dropped them): forgotten after a while.
        let mut s = refine.state.lock().unwrap();
        let later = Instant::now() + HANDED_FOR;
        assert_eq!(Refine::next(&mut s, later).len(), BATCH);
        // A teleport: coarse nodes far beyond the view are forgotten too.
        s.camera = [1e6, 0.0, 0.0];
        s.max_distance = 4096.0;
        assert!(Refine::next(&mut s, later).is_empty());
        assert!(s.coarse.is_empty());
    }

    #[test]
    fn nodes_the_plan_dropped_free_their_place_in_the_batch_at_once() {
        let refine = Refine::default();
        let keys: Vec<CubeKey> = (0..200).map(|x| CubeKey::new(2, x, 0, 0)).collect();
        for &key in &keys {
            refine.made(key, false, true);
        }
        refine.view([0.0, 32.0, 32.0], [1.0, 0.0, 0.0], 1e5, true);
        let first = refine.take_refreshes();
        assert_eq!(first.len(), BATCH);
        // The producer has no drawn node for any of them (they left the plan): the next batch comes at once,
        // without waiting for them to time out.
        refine.not_drawn(&first);
        assert_eq!(refine.counts(), (200 - BATCH, 0));
        let second = refine.take_refreshes();
        assert_eq!(second.len(), BATCH);
        assert!(
            second.iter().all(|b| !first.contains(b)),
            "new nodes, the dropped ones are forgotten"
        );
        // A box that is no node's extent changes nothing.
        refine.not_drawn(&[([1, 0, 0], [64, 63, 63])]);
        assert_eq!(refine.counts().1, BATCH);
    }

    fn view(refine: &Refine) {
        refine.view([0.0, 32.0, 32.0], [1.0, 0.0, 0.0], 1e5, true);
    }

    #[test]
    fn whole_when_covered_is_off_by_default_so_every_node_comes_coarse_first() {
        assert_eq!(WHOLE_WHEN_COVERED, 0);
        let refine = Refine {
            whole_when_covered: false,
            ..Refine::default()
        };
        assert!(!refine.wants_coarse_first());
        let (hole, other) = (CubeKey::new(2, 0, 0, 0), CubeKey::new(2, 1, 0, 0));
        refine.coarse_first(&[hole], false);
        assert_eq!(refine.pass(hole), Pass::Coarse);
        assert_eq!(
            refine.pass(other),
            Pass::Coarse,
            "coarse first though covered"
        );
    }

    #[test]
    fn nodes_are_coarse_first_only_where_the_producer_says_nothing_is_drawn() {
        let refine = Refine {
            whole_when_covered: true,
            ..Refine::default()
        };
        assert!(refine.exact());
        assert!(refine.wants_coarse_first());
        let (hole, other, third) = (
            CubeKey::new(2, 0, 0, 0),
            CubeKey::new(2, 1, 0, 0),
            CubeKey::new(3, 5, 0, 0),
        );
        // The producer has not said what is drawn: everything comes coarse first.
        for key in [hole, other, third] {
            assert_eq!(refine.pass(key), Pass::Coarse);
        }
        // Only the hole is bare: the rest replaces what is drawn and is made whole at once.
        refine.coarse_first(&[hole], false);
        assert_eq!(refine.pass(hole), Pass::Coarse);
        assert_eq!(refine.pass(other), Pass::Whole);
        assert_eq!(refine.pass(third), Pass::Whole);
        // A new list replaces the old one.
        refine.coarse_first(&[other], false);
        assert_eq!(refine.pass(hole), Pass::Whole);
        assert_eq!(refine.pass(other), Pass::Coarse);
        // A big fill: every node coarse first.
        refine.coarse_first(&[], true);
        for key in [hole, other, third] {
            assert_eq!(refine.pass(key), Pass::Coarse);
        }
        // Handed-out and made-whole nodes are whole whatever the producer says.
        refine.made(hole, false, true);
        view(&refine);
        assert_eq!(refine.take_refreshes().len(), 1);
        assert_eq!(refine.pass(hole), Pass::Whole);
        refine.made(other, true, true);
        assert_eq!(refine.pass(other), Pass::Whole);
        refine.coarse_first(&[hole, other, third], false);
        assert_eq!(refine.pass(hole), Pass::Whole);
        assert_eq!(refine.pass(other), Pass::Whole);
        assert_eq!(refine.pass(third), Pass::Coarse);
    }

    #[test]
    fn a_node_made_whole_directly_is_never_handed_out_to_be_made_again() {
        let refine = Refine::default();
        let (direct, coarse) = (CubeKey::new(2, 3, 0, 0), CubeKey::new(2, 4, 0, 0));
        refine.coarse_first(&[], false);
        // Whole straight away, though it was noted coarse earlier (before the producer spoke).
        refine.made(direct, false, true);
        refine.made(coarse, false, true);
        refine.made(direct, true, true);
        view(&refine);
        assert_eq!(refine.take_refreshes(), vec![([256, 0, 0], [319, 63, 63])]);
        assert_eq!(refine.pass(direct), Pass::Whole);
        // And one never noted coarse is not handed out either.
        let fresh = CubeKey::new(2, 9, 0, 0);
        refine.made(fresh, true, true);
        refine.made(CubeKey::new(2, 10, 0, 0), false, true);
        assert_eq!(refine.counts(), (1, 1));
        assert_eq!(
            refine.take_refreshes(),
            vec![([640, 0, 0], [703, 63, 63])],
            "the coarse node, not the whole ones"
        );
    }

    #[test]
    fn a_node_given_back_is_handed_out_again_only_after_the_defer_time() {
        let refine = Refine::default();
        let (a, b) = (CubeKey::new(2, 0, 0, 0), CubeKey::new(2, 1, 0, 0));
        for key in [a, b] {
            refine.made(key, false, true);
        }
        view(&refine);
        let boxes = refine.take_refreshes();
        assert_eq!(boxes.len(), 2);
        assert_eq!(refine.counts(), (0, 2));
        // Boxes that are no node's extent, and nodes not handed out, change nothing.
        refine.not_now(&[([1, 0, 0], [64, 63, 63]), ([640, 0, 0], [703, 63, 63])]);
        assert_eq!(refine.counts(), (0, 2));
        // `a` is given back: it is coarse again, not handed out; `b` stays handed out.
        let t0 = Instant::now();
        {
            let mut s = refine.state.lock().unwrap();
            Refine::defer(&mut s, &[boxes[0]], t0);
            assert_eq!((s.coarse.len(), s.handed.len()), (1, 1));
            s.handed.clear();
            assert!(
                Refine::next(&mut s, t0).is_empty(),
                "not handed out again at once"
            );
            assert!(
                Refine::next(&mut s, t0 + DEFER_FOR / 2).is_empty(),
                "nor within the defer time"
            );
            let again = Refine::next(&mut s, t0 + DEFER_FOR);
            assert_eq!(again.len(), 1, "handed out again after it");
            assert!(s.deferred.is_empty(), "the old entry is dropped");
        }
        // The public hook does the same with the clock.
        let refine = Refine::default();
        refine.made(a, false, true);
        view(&refine);
        let boxes = refine.take_refreshes();
        refine.not_now(&boxes);
        assert_eq!(refine.counts(), (1, 0));
        assert!(refine.take_refreshes().is_empty());
        // Other nodes are not held back by it.
        refine.made(b, false, true);
        assert_eq!(refine.take_refreshes(), vec![([64, 0, 0], [127, 63, 63])]);
    }
}

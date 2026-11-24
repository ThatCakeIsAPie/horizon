//! The far view's second pass: coarse first, refined later (docs/earth.md, "Far view").
//!
//! An Earth node with trees is first made without them (its ground, colour and snow: the terrain as it was
//! before far trees, as fast), so the distance is never empty while trees are worked out. Each such node is
//! noted here as coarse; once every planned node is drawn the producer asks for the nodes to make again
//! ([`LateContent`]), and this hands out the coarse ones nearest the camera and in view first, a few at a
//! time, so they go behind all other far work. A node made again is made whole (with its trees) and replaces
//! the coarse one in place. Only whole nodes are kept in the far cache, so a revisit draws them refined at once.

use crate::lod::cube::{CubeKey, LateContent};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Nodes handed out at a time: the next batch waits until these are made, so the order follows the camera.
const BATCH: usize = 48;
/// A node handed out but not made in this long was not made again (the plan dropped it): forgotten.
const HANDED_FOR: Duration = Duration::from_secs(30);

#[derive(Default)]
struct State {
    /// Drawn coarse, not handed out yet.
    coarse: HashSet<CubeKey>,
    /// Handed out to be made whole, not made yet, and when.
    handed: HashMap<CubeKey, Instant>,
    /// Made whole: made whole straight away when asked for again.
    whole: HashSet<CubeKey>,
    camera: [f64; 3],
    forward: [f64; 3],
    max_distance: f64,
}

/// The coarse nodes and the order they are refined in, shared by every fork of the sampler.
#[derive(Default)]
pub struct Refine {
    state: Mutex<State>,
}

/// How node `key` should be made now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Coarse,
    Whole,
}

impl Refine {
    /// Whether node `key` is made coarse (first) or whole (it was handed out, or was made whole before).
    pub fn pass(&self, key: CubeKey) -> Pass {
        let s = self.state.lock().unwrap();
        if s.handed.contains_key(&key) || s.whole.contains(&key) {
            Pass::Whole
        } else {
            Pass::Coarse
        }
    }

    /// Node `key` was made: coarse (`whole` false, to be refined), or whole. `made` false when a whole build
    /// could not be made yet (data missing): it goes back to the coarse ones and is handed out again.
    pub fn made(&self, key: CubeKey, whole: bool, made: bool) {
        let mut s = self.state.lock().unwrap();
        match (whole, made) {
            (false, true) => {
                s.coarse.insert(key);
            }
            (true, true) => {
                s.handed.remove(&key);
                s.coarse.remove(&key);
                s.whole.insert(key);
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

    /// The next coarse nodes to refine: nearest first, those in front of the camera before those behind.
    fn next(s: &mut State, now: Instant) -> Vec<CubeKey> {
        s.handed.retain(|_, at| now.duration_since(*at) < HANDED_FOR);
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
        if room == 0 || s.coarse.is_empty() {
            return Vec::new();
        }
        let score = |key: &CubeKey| {
            let (o, size) = (key.origin(), f64::from(key.size()));
            let d: [f64; 3] = std::array::from_fn(|i| f64::from(o[i]) + size / 2.0 - camera[i]);
            let distance = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let along = d[0] * forward[0] + d[1] * forward[1] + d[2] * forward[2];
            // Behind the camera (beyond the node's own size): as if three times as far.
            if along < -size {
                distance * 3.0
            } else {
                distance
            }
        };
        let mut keys: Vec<(f64, CubeKey)> = s.coarse.iter().map(|k| (score(k), *k)).collect();
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
        // Nearest first; a node behind the camera counts three times its distance (node -2, 106
        // blocks behind, comes after node 4, 278 ahead, and before node 5, 342 ahead).
        let order: Vec<i32> = first.iter().take(8).map(|(min, _)| min[0] >> 6).collect();
        assert_eq!(order, [0, -1, 1, 2, 3, 4, -2, 5]);
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
}

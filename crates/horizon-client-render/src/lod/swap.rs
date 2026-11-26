//! When far cube nodes swap: which new nodes show and which replaced ones
//! go, decided per set of overlapping nodes instead of once the whole queue
//! is in. Pure bookkeeping: the renderers (`gfx::lod`, `gfx::cube_far`) tell
//! [`Swap`] what happens to their nodes and apply what [`Swap::step`] returns.
//!
//! The single-cut rule: no two drawn nodes ever overlap (one holding the
//! other), in any frame. Two far surfaces over the same ground drawn at once
//! fight for the same pixels (z-fighting), so a set swaps in one frame and no
//! replaced node stays drawn beside what replaces it, not even fading out.
//!
//! The rules, each step:
//! - A removed node goes as soon as no node still waiting (queued, not
//!   installed yet) overlaps its box: whatever replaces it is in.
//! - An installed node that is not shown yet shows as soon as no shown node
//!   overlaps its box (the ones going this step do not count). Coarser nodes
//!   are looked at first.
//! - A node going under something that shows this step is dropped at once
//!   and what shows over it is drawn whole: the swap is atomic, no frame with
//!   both and none with neither. A node going with nothing over it (out of
//!   range, or replaced by air) fades out alone; a node showing over nothing
//!   fades in. A fade never overlaps another drawn node: one showing over a
//!   node still fading out cuts that fade short.
//! - A node never shows beside a live node holding it (or inside it), even
//!   when nothing else waits: it waits until that node is removed.
use super::cube::{CubeKey, MAX_CUBE_LEVEL};
use crate::fast_hash::{FxHashMap, FxHashSet};

/// A set of nodes that answers "does any of them overlap this node's box" by
/// looking at the node and its ancestors only.
#[derive(Default)]
pub struct KeyIndex {
    keys: FxHashSet<CubeKey>,
    /// For each node that is, or holds, a key: how many keys lie at or below it.
    under: FxHashMap<CubeKey, u32>,
}

impl KeyIndex {
    /// Adds `key`; false if it was in already.
    pub fn add(&mut self, key: CubeKey) -> bool {
        if !self.keys.insert(key) {
            return false;
        }
        let mut node = key;
        loop {
            *self.under.entry(node).or_insert(0) += 1;
            if node.level >= MAX_CUBE_LEVEL {
                return true;
            }
            node = node.parent();
        }
    }

    /// Removes `key`; false if it was not in.
    pub fn remove(&mut self, key: CubeKey) -> bool {
        if !self.keys.remove(&key) {
            return false;
        }
        let mut node = key;
        loop {
            if let Some(count) = self.under.get_mut(&node) {
                *count -= 1;
                if *count == 0 {
                    self.under.remove(&node);
                }
            }
            if node.level >= MAX_CUBE_LEVEL {
                return true;
            }
            node = node.parent();
        }
    }

    pub fn contains(&self, key: CubeKey) -> bool {
        self.keys.contains(&key)
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.under.clear();
    }

    pub fn keys(&self) -> impl Iterator<Item = CubeKey> + '_ {
        self.keys.iter().copied()
    }

    /// Whether `key`, something inside it or something holding it is in.
    pub fn overlaps(&self, key: CubeKey) -> bool {
        if self.under.contains_key(&key) {
            return true;
        }
        let mut node = key;
        while node.level < MAX_CUBE_LEVEL {
            node = node.parent();
            if self.keys.contains(&node) {
                return true;
            }
        }
        false
    }

    /// How many keys overlap `key` other than `key` itself: the ones holding
    /// it and the ones inside it.
    pub fn overlapping(&self, key: CubeKey) -> usize {
        let inside = self.under.get(&key).copied().unwrap_or(0) as usize
            - usize::from(self.keys.contains(&key));
        let mut holding = 0;
        let mut node = key;
        while node.level < MAX_CUBE_LEVEL {
            node = node.parent();
            holding += usize::from(self.keys.contains(&node));
        }
        inside + holding
    }
}

/// The overlap counter: of the far nodes a renderer draws (shown, or fading
/// in or out), how many pairs overlap, one holding the other. The single-cut
/// rule keeps it at 0 every frame; anything else is two far surfaces drawn
/// over the same ground (z-fighting). Kept up as nodes start and stop drawing.
#[derive(Default)]
pub struct Overlaps {
    drawn: KeyIndex,
    pairs: usize,
    /// Frames counted ([`Overlaps::frame`]), those with an overlap, and the
    /// most pairs in one.
    frames: u64,
    frames_over: u64,
    max: usize,
}

/// What [`Overlaps`] counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OverlapStats {
    /// Overlapping pairs drawn now.
    pub now: usize,
    /// The most in one frame so far.
    pub max: usize,
    /// Frames counted, and those that drew an overlap.
    pub frames: u64,
    pub frames_over: u64,
    /// Nodes drawn now.
    pub drawn: usize,
}

impl Overlaps {
    /// `key` draws (`true`) or not.
    pub fn set(&mut self, key: CubeKey, drawn: bool) {
        if drawn == self.drawn.contains(key) {
            return;
        }
        if drawn {
            self.pairs += self.drawn.overlapping(key);
            self.drawn.add(key);
        } else {
            self.drawn.remove(key);
            self.pairs -= self.drawn.overlapping(key);
        }
    }

    /// Nothing draws any more (the counts so far stay).
    pub fn clear(&mut self) {
        self.drawn.clear();
        self.pairs = 0;
    }

    /// Overlapping pairs drawn now.
    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// A frame is drawn with the nodes as they are.
    pub fn frame(&mut self) {
        self.frames += 1;
        if self.pairs > 0 {
            self.frames_over += 1;
            self.max = self.max.max(self.pairs);
        }
    }

    pub fn stats(&self) -> OverlapStats {
        OverlapStats {
            now: self.pairs,
            max: self.max,
            frames: self.frames,
            frames_over: self.frames_over,
            drawn: self.drawn.len(),
        }
    }
}

impl OverlapStats {
    /// Both renderers' counts together.
    pub fn plus(self, other: Self) -> Self {
        Self {
            now: self.now + other.now,
            max: self.max.max(other.max),
            frames: self.frames.max(other.frames),
            frames_over: self.frames_over.max(other.frames_over),
            drawn: self.drawn + other.drawn,
        }
    }
}

/// What a [`Swap::step`] decided.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Step {
    /// Installed nodes to show now, whole: they replace nodes going now.
    pub show: Vec<CubeKey>,
    /// Installed nodes to show now over nothing drawn: they fade in.
    pub fade_in: Vec<CubeKey>,
    /// Nodes to drop now with no fade: replaced this step, a fade cut short,
    /// or never shown.
    pub drop: Vec<CubeKey>,
    /// Nodes to fade out: nothing is drawn over their place. A node that was
    /// never shown, or that is not installed, is to be dropped instead.
    pub fade: Vec<CubeKey>,
}

/// The swap state of one set of far nodes.
#[derive(Default)]
pub struct Swap {
    /// Queued or in the mesher, not installed.
    waiting: KeyIndex,
    /// Installed, shown and not going.
    shown: KeyIndex,
    /// Installed, not shown yet.
    unshown: FxHashSet<CubeKey>,
    /// The plan dropped them: they go once what replaces them is in.
    removed: FxHashSet<CubeKey>,
    /// Fading out alone: still drawn until the renderer drops them (`forget`).
    fading: KeyIndex,
    /// Something changed since the last step.
    dirty: bool,
}

impl Swap {
    /// `key` waits for installing (a newer mesh of it waits again). It is no
    /// longer removed.
    pub fn queued(&mut self, key: CubeKey) {
        self.removed.remove(&key);
        self.waiting.add(key);
        self.dirty = true;
    }

    /// `key` is not waiting any more without being installed: its mesh was
    /// dropped, failed or discarded.
    pub fn unwait(&mut self, key: CubeKey) {
        if self.waiting.remove(key) {
            self.dirty = true;
        }
    }

    /// The plan dropped `key`. It does not stop waiting if it is (a node in
    /// the mesher cannot be called back): `unwait` does that.
    pub fn retire(&mut self, key: CubeKey) {
        self.removed.insert(key);
        self.dirty = true;
    }

    /// `key` is installed: it stops waiting and replaces an older node of the
    /// key (one fading out is gone). `shown`: it keeps the older one's place
    /// in the scene already.
    pub fn installed(&mut self, key: CubeKey, shown: bool) {
        self.waiting.remove(key);
        self.shown.remove(key);
        self.unshown.remove(&key);
        self.fading.remove(key);
        if shown {
            self.shown.add(key);
        } else {
            self.unshown.insert(key);
        }
        self.dirty = true;
    }

    /// The installed node of `key` is gone (or was never there), or its fade
    /// out is over.
    pub fn forget(&mut self, key: CubeKey) {
        self.shown.remove(key);
        self.unshown.remove(&key);
        self.fading.remove(key);
        self.dirty = true;
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        self.waiting.clear();
        self.shown.clear();
        self.unshown.clear();
        self.removed.clear();
        self.fading.clear();
        self.dirty = false;
    }

    /// Whether nothing waits, hides or goes: the state is at rest.
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.unshown.is_empty() && self.removed.is_empty()
    }

    /// Nodes queued or in the mesher.
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// The nodes to show, drop and fade this frame. Call every frame: when
    /// nothing changed it returns at once.
    pub fn step(&mut self) -> Step {
        if !std::mem::take(&mut self.dirty) {
            return Step::default();
        }
        let mut step = Step::default();
        // Removed nodes whose replacements are all in leave the shown set
        // first, so what replaces them shows in the same step.
        let mut going = KeyIndex::default();
        for key in std::mem::take(&mut self.removed) {
            let was_shown = self.shown.contains(key);
            if !was_shown && !self.waiting.contains(key) {
                // Never shown and not coming in: nothing to swap.
                self.unshown.remove(&key);
                step.drop.push(key);
            } else if self.waiting.overlaps(key) {
                self.removed.insert(key);
            } else {
                self.shown.remove(key);
                self.unshown.remove(&key);
                going.add(key);
            }
        }
        // Coarser nodes first: of two unshown nodes that overlap, the bigger
        // shows and the other waits for it to go.
        let mut unshown: Vec<CubeKey> = self.unshown.iter().copied().collect();
        unshown.sort_unstable_by_key(|key| (std::cmp::Reverse(key.level), key.x, key.y, key.z));
        let mut showing = KeyIndex::default();
        for key in unshown {
            if self.shown.overlaps(key) {
                continue;
            }
            self.unshown.remove(&key);
            self.shown.add(key);
            showing.add(key);
            if going.overlaps(key) || self.fading.overlaps(key) {
                step.show.push(key);
            } else {
                step.fade_in.push(key);
            }
        }
        // What goes under something showing now is dropped in this very step;
        // the rest fades out alone.
        for key in going.keys() {
            if showing.overlaps(key) {
                step.drop.push(key);
            } else {
                self.fading.add(key);
                step.fade.push(key);
            }
        }
        // A fade under something showing now is cut short.
        if !showing.is_empty() {
            let cut: Vec<CubeKey> = self
                .fading
                .keys()
                .filter(|k| showing.overlaps(*k))
                .collect();
            for key in cut {
                self.fading.remove(key);
                step.drop.push(key);
            }
        }
        step.show.sort_unstable();
        step.fade_in.sort_unstable();
        step.drop.sort_unstable();
        step.fade.sort_unstable();
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn k(level: u8, x: i32, y: i32, z: i32) -> CubeKey {
        CubeKey::new(level, x, y, z)
    }

    /// Far from anything in the tests.
    fn far(n: i32) -> CubeKey {
        k(3, 1000 + n, 0, 1000)
    }

    #[test]
    fn the_index_sees_ancestors_descendants_and_itself() {
        let mut index = KeyIndex::default();
        let node = k(3, 5, 2, 7);
        assert!(!index.overlaps(node));
        assert!(index.add(node));
        assert!(!index.add(node), "a double add is ignored");
        assert_eq!(index.len(), 1);
        assert!(index.overlaps(node));
        // Inside it, at any depth.
        assert!(index.overlaps(node.children()[5]));
        assert!(index.overlaps(node.children()[5].children()[2]));
        // Holding it, at any height.
        assert!(index.overlaps(node.parent()));
        assert!(index.overlaps(node.parent().parent().parent()));
        // Beside it.
        for child in node.parent().children() {
            if child != node {
                assert!(!index.overlaps(child), "{child:?}");
                assert!(!index.overlaps(child.children()[0]));
            }
        }
        assert!(!index.overlaps(k(3, 6, 2, 7)));
        assert!(!index.overlaps(k(3, 5, 3, 7)));
        assert!(!index.overlaps(k(3, 5, 2, 6)));
        assert!(!index.overlaps(k(4, 5, 2, 7)));
    }

    #[test]
    fn the_index_nests_negative_coordinates() {
        let mut index = KeyIndex::default();
        let node = k(2, -1, -3, -8);
        index.add(node);
        assert!(index.overlaps(node.parent()));
        assert!(index.overlaps(node.children()[7]));
        assert!(!index.overlaps(k(2, -2, -3, -8)));
        assert!(!index.overlaps(k(2, 0, -3, -8)));
        assert!(index.overlaps(k(3, -1, -2, -4)));
        assert!(!index.overlaps(k(3, 0, -2, -4)));
    }

    #[test]
    fn the_index_forgets_removed_keys() {
        let mut index = KeyIndex::default();
        let (parent, child) = (k(4, 1, 1, 1), k(2, 4, 4, 4));
        index.add(parent);
        index.add(child);
        assert!(index.overlaps(child.children()[0]));
        assert!(index.remove(parent));
        assert!(!index.remove(parent), "a missing remove is ignored");
        assert!(!index.overlaps(k(2, 5, 4, 4)), "the parent is gone");
        assert!(index.overlaps(k(5, 0, 0, 0)), "the child is still under it");
        assert!(index.remove(child));
        assert!(index.is_empty());
        assert!(!index.overlaps(k(5, 0, 0, 0)));
        assert!(!index.overlaps(child));
        assert!(index.under.is_empty(), "no counts are left behind");
    }

    #[test]
    fn the_index_reaches_the_top_level() {
        let mut index = KeyIndex::default();
        let top = k(MAX_CUBE_LEVEL, 0, 0, 0);
        index.add(top);
        assert!(index.overlaps(k(0, 5, 5, 5)));
        index.remove(top);
        index.add(k(0, 5, 5, 5));
        assert!(index.overlaps(top));
        assert!(index.remove(k(0, 5, 5, 5)));
        assert!(index.under.is_empty());
    }

    #[test]
    fn overlapping_counts_holders_and_insiders_but_not_itself() {
        let mut index = KeyIndex::default();
        let node = k(3, 5, 2, 7);
        index.add(node);
        index.add(node.parent().parent());
        index.add(node.children()[1]);
        index.add(node.children()[1].children()[4]);
        index.add(k(3, 6, 2, 7));
        assert_eq!(index.overlapping(node), 3);
        assert_eq!(index.overlapping(node.children()[2]), 2, "two above it");
        assert_eq!(index.overlapping(k(3, 9, 9, 9)), 0);
    }

    /// Whether `a` holds `b` (strictly).
    fn holds(a: CubeKey, b: CubeKey) -> bool {
        a.level > b.level && {
            let up = a.level - b.level;
            CubeKey::new(a.level, b.x >> up, b.y >> up, b.z >> up) == a
        }
    }

    fn brute_pairs(drawn: &FxHashSet<CubeKey>) -> usize {
        let mut n = 0;
        for &a in drawn {
            for &b in drawn {
                n += usize::from(holds(a, b));
            }
        }
        n
    }

    /// A tiny deterministic random source.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn chance(&mut self, percent: u64) -> bool {
            self.next() % 100 < percent
        }
    }

    #[test]
    fn the_overlap_counter_matches_a_brute_count() {
        let mut rng = Lcg(7);
        let mut overlaps = Overlaps::default();
        let mut drawn = FxHashSet::default();
        let root = k(4, 0, 0, 0);
        for _ in 0..2000 {
            // A random node of the root's tree, levels 1 to 4.
            let mut node = root;
            for _ in 0..rng.below(4) {
                node = node.children()[rng.below(8)];
            }
            let on = rng.chance(50);
            overlaps.set(node, on);
            if on {
                drawn.insert(node);
            } else {
                drawn.remove(&node);
            }
            assert_eq!(overlaps.pairs(), brute_pairs(&drawn));
            overlaps.frame();
        }
        let stats = overlaps.stats();
        assert!(stats.max > 0 && stats.frames_over > 0 && stats.frames == 2000);
        overlaps.clear();
        assert_eq!(overlaps.pairs(), 0);
        assert_eq!(overlaps.stats().max, stats.max, "the counts so far stay");
    }

    /// A renderer's far nodes as it applies the steps: installed ones, the
    /// drawn ones among them (shown or fading) and those fading out. Every
    /// step it checks that no two drawn nodes overlap.
    #[derive(Default)]
    struct Scene {
        installed: FxHashSet<CubeKey>,
        drawn: FxHashSet<CubeKey>,
        fading: FxHashSet<CubeKey>,
        overlaps: Overlaps,
        steps: usize,
    }

    impl Scene {
        fn apply(&mut self, swap: &mut Swap) -> Step {
            let step = swap.step();
            for key in &step.drop {
                self.drawn.remove(key);
                self.fading.remove(key);
                self.installed.remove(key);
            }
            for key in &step.fade {
                if self.drawn.contains(key) {
                    self.fading.insert(*key);
                } else {
                    self.installed.remove(key);
                }
            }
            for key in step.show.iter().chain(&step.fade_in) {
                assert!(self.installed.contains(key), "{key:?} shows uninstalled");
                assert!(self.drawn.insert(*key), "{key:?} shows twice");
            }
            for key in &step.show {
                assert!(!step.fade_in.contains(key), "{key:?} both whole and fading");
            }
            self.check();
            step
        }

        fn check(&mut self) {
            for &key in &self.installed {
                self.overlaps.set(key, self.drawn.contains(&key));
            }
            let stale: Vec<CubeKey> = self.overlaps.drawn.keys().collect();
            for key in stale {
                if !self.drawn.contains(&key) {
                    self.overlaps.set(key, false);
                }
            }
            self.overlaps.frame();
            self.steps += 1;
            assert_eq!(
                brute_pairs(&self.drawn),
                0,
                "step {}: two drawn far nodes overlap: {:?}",
                self.steps,
                self.drawn
            );
            assert_eq!(self.overlaps.pairs(), 0);
        }

        /// A mesh of `key` comes in (`mesh`), or comes in empty.
        fn install(&mut self, swap: &mut Swap, key: CubeKey, mesh: bool) {
            // A node fading out is replaced, not kept (`upload_cube_lod`).
            let shown = self.drawn.contains(&key) && !self.fading.contains(&key);
            if self.fading.remove(&key) {
                self.drawn.remove(&key);
            }
            if mesh {
                self.installed.insert(key);
                swap.installed(key, shown);
            } else {
                self.installed.remove(&key);
                self.drawn.remove(&key);
                swap.unwait(key);
                swap.forget(key);
            }
        }

        /// Every fade out is over.
        fn finish_fades(&mut self, swap: &mut Swap) {
            for key in std::mem::take(&mut self.fading) {
                self.drawn.remove(&key);
                self.installed.remove(&key);
                swap.forget(key);
            }
        }
    }

    /// A shown node `R` replaced by its eight children and some unrelated
    /// nodes elsewhere, all queued.
    fn parent_with_children(swap: &mut Swap, scene: &mut Scene) -> (CubeKey, [CubeKey; 8]) {
        let parent = k(4, 3, 1, -2);
        swap.queued(parent);
        scene.install(swap, parent, true);
        assert_eq!(
            scene.apply(swap).fade_in,
            [parent],
            "over nothing: fades in"
        );
        let children = parent.children();
        for child in children {
            swap.queued(child);
        }
        for n in 0..5 {
            swap.queued(far(n));
        }
        swap.retire(parent);
        assert_eq!(scene.apply(swap), Step::default(), "nothing is in yet");
        (parent, children)
    }

    #[test]
    fn a_parent_swaps_when_its_own_children_are_in() {
        let mut swap = Swap::default();
        let mut scene = Scene::default();
        let (parent, children) = parent_with_children(&mut swap, &mut scene);
        for child in &children[..4] {
            scene.install(&mut swap, *child, true);
        }
        let step = scene.apply(&mut swap);
        assert_eq!(step, Step::default(), "four children are not the set");
        assert!(scene.drawn.contains(&parent));
        for child in &children[4..] {
            scene.install(&mut swap, *child, true);
        }
        // The far nodes are still waiting.
        assert_eq!(swap.waiting(), 5);
        let step = scene.apply(&mut swap);
        assert_eq!(
            step.drop,
            [parent],
            "dropped at once, no fade over its children"
        );
        assert!(step.fade.is_empty());
        let mut all = children.to_vec();
        all.sort_unstable();
        assert_eq!(
            step.show, all,
            "the children show whole the frame the parent goes"
        );
        assert!(step.fade_in.is_empty());
        assert_eq!(swap.step(), Step::default());
    }

    #[test]
    fn children_in_any_order_and_empty_ones_swap_together() {
        for seed in 0..40 {
            let mut rng = Lcg(seed);
            let mut swap = Swap::default();
            let mut scene = Scene::default();
            let (parent, children) = parent_with_children(&mut swap, &mut scene);
            let mut order = children.to_vec();
            for i in (1..order.len()).rev() {
                order.swap(i, rng.below(i + 1));
            }
            let empty: Vec<bool> = (0..8).map(|_| rng.chance(30)).collect();
            for (n, child) in order.iter().enumerate() {
                scene.install(&mut swap, *child, !empty[n]);
                let step = scene.apply(&mut swap);
                if n < 7 {
                    assert!(scene.drawn.contains(&parent), "seed {seed}: no hole");
                    assert!(step.show.is_empty() && step.fade_in.is_empty());
                } else {
                    assert!(!scene.drawn.contains(&parent));
                    let meshed: FxHashSet<CubeKey> = order
                        .iter()
                        .zip(&empty)
                        .filter(|(_, e)| !**e)
                        .map(|(c, _)| *c)
                        .collect();
                    assert_eq!(scene.drawn, meshed, "seed {seed}: the set at once");
                    if meshed.is_empty() {
                        assert_eq!(step.fade, [parent], "replaced by air: fades alone");
                    } else {
                        assert_eq!(step.drop, [parent]);
                        assert_eq!(step.show.len(), meshed.len(), "whole, no fade");
                    }
                }
            }
        }
    }

    #[test]
    fn a_merge_drops_the_children_the_frame_the_parent_shows() {
        let mut swap = Swap::default();
        let mut scene = Scene::default();
        let parent = k(5, 0, 0, 0);
        let children = parent.children();
        for child in children {
            swap.queued(child);
            scene.install(&mut swap, child, true);
        }
        scene.apply(&mut swap);
        swap.queued(parent);
        swap.queued(far(0));
        for child in children {
            swap.retire(child);
        }
        assert_eq!(scene.apply(&mut swap), Step::default());
        scene.install(&mut swap, parent, true);
        let step = scene.apply(&mut swap);
        assert_eq!(step.drop.len(), 8);
        assert!(step.fade.is_empty());
        assert_eq!(step.show, [parent]);
    }

    #[test]
    fn an_unshown_node_under_a_live_ancestor_stays_hidden() {
        let mut swap = Swap::default();
        let mut scene = Scene::default();
        let parent = k(4, 0, 0, 0);
        swap.queued(parent);
        scene.install(&mut swap, parent, true);
        scene.apply(&mut swap);
        // A child arrives, the parent is not removed, and something else waits.
        let child = parent.children()[3];
        swap.queued(child);
        swap.queued(far(0));
        scene.install(&mut swap, child, true);
        assert_eq!(scene.apply(&mut swap), Step::default());
        // Another node elsewhere does show.
        let other = far(1);
        swap.queued(other);
        scene.install(&mut swap, other, true);
        assert_eq!(scene.apply(&mut swap).fade_in, [other]);
        // The child shows once the parent goes.
        swap.retire(parent);
        let step = scene.apply(&mut swap);
        assert_eq!(step.drop, [parent]);
        assert_eq!(step.show, [child]);
    }

    #[test]
    fn nothing_waiting_never_shows_a_node_beside_a_live_one() {
        // The planner has not released the coarse node yet (its other finer
        // nodes are still being made): its installed children stay hidden
        // even with the renderer's queue empty.
        let mut swap = Swap::default();
        let mut scene = Scene::default();
        let parent = k(4, 0, 0, 0);
        let child = parent.children()[0];
        swap.queued(parent);
        scene.install(&mut swap, parent, true);
        scene.apply(&mut swap);
        swap.queued(child);
        scene.install(&mut swap, child, true);
        assert_eq!(swap.waiting(), 0);
        assert_eq!(scene.apply(&mut swap), Step::default());
        swap.retire(parent);
        let step = scene.apply(&mut swap);
        assert_eq!((step.drop, step.show), (vec![parent], vec![child]));
        assert!(swap.is_idle());
    }

    #[test]
    fn a_node_showing_over_a_fade_cuts_it_short() {
        let mut swap = Swap::default();
        let mut scene = Scene::default();
        let node = k(3, 0, 0, 0);
        swap.queued(node);
        scene.install(&mut swap, node, true);
        scene.apply(&mut swap);
        // Out of range: it fades out alone.
        swap.retire(node);
        let step = scene.apply(&mut swap);
        assert_eq!(step.fade, [node]);
        assert!(scene.fading.contains(&node));
        // Back in range as children while it still fades.
        let child = node.children()[2];
        swap.queued(child);
        scene.install(&mut swap, child, true);
        let step = scene.apply(&mut swap);
        assert_eq!(step.drop, [node], "the fade is cut short");
        assert_eq!(step.show, [child], "and the child is whole at once");
        scene.fading.remove(&node);
        // A node fading in over nothing; its own key comes back while fading.
        let other = far(3);
        swap.queued(other);
        scene.install(&mut swap, other, true);
        scene.apply(&mut swap);
        swap.retire(other);
        assert_eq!(scene.apply(&mut swap).fade, [other]);
        swap.queued(other);
        scene.install(&mut swap, other, true);
        let step = scene.apply(&mut swap);
        assert_eq!(step.fade_in, [other], "the fading one was replaced");
    }

    #[test]
    fn a_removed_node_that_never_showed_goes_at_once() {
        let mut swap = Swap::default();
        let node = k(3, 0, 0, 0);
        swap.queued(node);
        swap.queued(far(0));
        swap.installed(node, false);
        swap.retire(node);
        assert_eq!(swap.step().drop, [node]);
        assert!(swap.step().show.is_empty());
    }

    #[test]
    fn a_removed_node_in_the_mesher_stays_waiting_until_installed() {
        let mut swap = Swap::default();
        let node = k(3, 0, 0, 0);
        swap.queued(node);
        swap.retire(node);
        assert_eq!(swap.waiting(), 1);
        assert_eq!(swap.step(), Step::default());
        swap.installed(node, false);
        assert_eq!(swap.step().drop, [node]);
        assert!(swap.is_idle());
    }

    #[test]
    fn a_queued_node_is_not_removed_any_more() {
        let mut swap = Swap::default();
        let node = k(3, 0, 0, 0);
        swap.queued(node);
        swap.installed(node, false);
        swap.step();
        swap.queued(far(0));
        swap.retire(node);
        swap.queued(node);
        swap.installed(node, true);
        assert_eq!(swap.step(), Step::default());
    }

    /// The plan's leaves change at random (splits, merges, refine and merge
    /// again while meshes are in flight), meshes come back out of order and
    /// some empty, fades end at random: no step ever draws two overlapping
    /// nodes, and once all is in the drawn nodes are the plan's.
    #[test]
    fn no_step_draws_two_overlapping_nodes_through_random_refines_and_merges() {
        for seed in 0..60 {
            let mut rng = Lcg(seed * 7919 + 1);
            let mut swap = Swap::default();
            let mut scene = Scene::default();
            let root = k(5, 0, 0, 0);
            // Whether a node's mesh is empty: fixed per node.
            let empty = |key: CubeKey| {
                let h = (key.x as u64)
                    .wrapping_mul(73856093)
                    .wrapping_add((key.y as u64).wrapping_mul(19349663))
                    .wrapping_add((key.z as u64).wrapping_mul(83492791))
                    .wrapping_add(u64::from(key.level) * 2654435761)
                    .wrapping_add(seed);
                h % 5 == 0
            };
            let mut plan: FxHashSet<CubeKey> = [root].into_iter().collect();
            let mut in_flight: Vec<CubeKey> = vec![root];
            swap.queued(root);
            for _round in 0..120 {
                // The planner changes its mind now and then.
                if rng.chance(40) {
                    let leaves: Vec<CubeKey> = {
                        let mut v: Vec<CubeKey> = plan.iter().copied().collect();
                        v.sort_unstable();
                        v
                    };
                    let leaf = leaves[rng.below(leaves.len())];
                    if leaf.level > 1 && rng.chance(65) {
                        // Split a leaf into its eight.
                        plan.remove(&leaf);
                        for child in leaf.children() {
                            plan.insert(child);
                            swap.queued(child);
                            in_flight.push(child);
                        }
                        swap.retire(leaf);
                    } else if leaf.level < root.level {
                        // Merge the parent's subtree back into it.
                        let parent = leaf.parent();
                        let under: Vec<CubeKey> =
                            plan.iter().copied().filter(|&l| holds(parent, l)).collect();
                        for l in under {
                            plan.remove(&l);
                            swap.retire(l);
                        }
                        plan.insert(parent);
                        swap.queued(parent);
                        in_flight.push(parent);
                    }
                }
                // The mesher hands back some meshes, in any order.
                for _ in 0..rng.below(4) {
                    if in_flight.is_empty() {
                        break;
                    }
                    let key = in_flight.swap_remove(rng.below(in_flight.len()));
                    // A key queued twice is installed once per queue; the
                    // swap tracks it waiting once.
                    if in_flight.contains(&key) {
                        continue;
                    }
                    scene.install(&mut swap, key, !empty(key));
                }
                scene.apply(&mut swap);
                if rng.chance(30) {
                    scene.finish_fades(&mut swap);
                }
            }
            // Everything comes in.
            while let Some(key) = in_flight.pop() {
                if !in_flight.contains(&key) {
                    scene.install(&mut swap, key, !empty(key));
                }
                scene.apply(&mut swap);
            }
            scene.apply(&mut swap);
            scene.finish_fades(&mut swap);
            scene.apply(&mut swap);
            let want: FxHashSet<CubeKey> = plan.iter().copied().filter(|&l| !empty(l)).collect();
            assert_eq!(scene.drawn, want, "seed {seed}: the plan is drawn at rest");
            assert_eq!(scene.overlaps.stats().frames_over, 0);
        }
    }
}

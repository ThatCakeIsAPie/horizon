//! When far cube nodes swap: which new nodes show and which replaced ones
//! fade, decided per set of overlapping nodes instead of once the whole
//! queue is in. Pure bookkeeping: the renderers (`gfx::lod`, `gfx::cube_far`)
//! tell [`Swap`] what happens to their nodes and apply what [`Swap::step`]
//! returns.
//!
//! The rules, each frame:
//! - A removed node starts to fade as soon as no node still waiting (queued,
//!   not installed yet) overlaps its box: whatever replaces it is in.
//! - An installed node that is not shown yet shows as soon as no shown, live
//!   node other than itself overlaps its box (the removed ones fading this
//!   frame do not count).
//! - When nothing waits at all, every unshown node shows and every removed
//!   node fades, so nothing stays hidden or drawn forever.
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
}

/// What a [`Swap::step`] decided.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Step {
    /// Installed nodes to show now.
    pub show: Vec<CubeKey>,
    /// Nodes to fade out now. A node that was never shown, or that is not
    /// installed, is to be dropped (or ignored) with no fade.
    pub fade: Vec<CubeKey>,
}

/// The swap state of one set of far nodes.
#[derive(Default)]
pub struct Swap {
    /// Queued or in the mesher, not installed.
    waiting: KeyIndex,
    /// Installed, shown and not fading out.
    shown: KeyIndex,
    /// Installed, not shown yet.
    unshown: FxHashSet<CubeKey>,
    /// The plan dropped them: they fade once what replaces them is in.
    removed: FxHashSet<CubeKey>,
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
    /// key. `shown`: it keeps the older one's place in the scene already.
    pub fn installed(&mut self, key: CubeKey, shown: bool) {
        self.waiting.remove(key);
        self.shown.remove(key);
        self.unshown.remove(&key);
        if shown {
            self.shown.add(key);
        } else {
            self.unshown.insert(key);
        }
        self.dirty = true;
    }

    /// The installed node of `key` is gone (or was never there).
    pub fn forget(&mut self, key: CubeKey) {
        self.shown.remove(key);
        self.unshown.remove(&key);
        self.dirty = true;
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        self.waiting.clear();
        self.shown.clear();
        self.unshown.clear();
        self.removed.clear();
        self.dirty = false;
    }

    /// Whether nothing waits, hides or fades: the state is at rest.
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.unshown.is_empty() && self.removed.is_empty()
    }

    /// Nodes queued or in the mesher.
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// The nodes to fade and show this frame. Call every frame: when nothing
    /// changed it returns at once.
    pub fn step(&mut self) -> Step {
        if !std::mem::take(&mut self.dirty) {
            return Step::default();
        }
        let net = self.waiting.is_empty();
        let mut step = Step::default();
        // Fade first, so the children of a parent fading now show now.
        let removed = std::mem::take(&mut self.removed);
        for key in removed {
            // Never shown and not coming in: nothing to fade.
            let hidden = !self.shown.contains(key) && !self.waiting.contains(key);
            if net || hidden || !self.waiting.overlaps(key) {
                self.shown.remove(key);
                self.unshown.remove(&key);
                step.fade.push(key);
            } else {
                self.removed.insert(key);
            }
        }
        // Coarser nodes first: of two unshown nodes that overlap, the bigger
        // shows and the other waits for it to go.
        let mut unshown: Vec<CubeKey> = self.unshown.iter().copied().collect();
        unshown.sort_unstable_by_key(|key| (std::cmp::Reverse(key.level), key.x, key.y, key.z));
        for key in unshown {
            if net || !self.shown.overlaps(key) {
                self.unshown.remove(&key);
                self.shown.add(key);
                step.show.push(key);
            }
        }
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

    /// A shown node `R` replaced by its eight children and some unrelated
    /// nodes elsewhere, all queued.
    fn parent_with_children(swap: &mut Swap) -> (CubeKey, [CubeKey; 8]) {
        let parent = k(4, 3, 1, -2);
        swap.queued(parent);
        swap.installed(parent, false);
        assert_eq!(swap.step().show, [parent]);
        let children = parent.children();
        for child in children {
            swap.queued(child);
        }
        for n in 0..5 {
            swap.queued(far(n));
        }
        swap.retire(parent);
        assert_eq!(swap.step(), Step::default(), "nothing is in yet");
        (parent, children)
    }

    #[test]
    fn a_parent_swaps_when_its_own_children_are_in() {
        let mut swap = Swap::default();
        let (parent, children) = parent_with_children(&mut swap);
        for child in &children[..4] {
            swap.installed(*child, false);
        }
        let step = swap.step();
        assert_eq!(step, Step::default(), "four children are not the set");
        for child in &children[4..] {
            swap.installed(*child, false);
        }
        // The far nodes are still waiting.
        assert_eq!(swap.waiting(), 5);
        let step = swap.step();
        assert_eq!(step.fade, [parent]);
        let mut shown = step.show.clone();
        shown.sort_unstable();
        let mut all = children.to_vec();
        all.sort_unstable();
        assert_eq!(shown, all, "the children show the frame the parent fades");
        assert_eq!(swap.step(), Step::default());
    }

    #[test]
    fn a_waiting_node_above_holds_its_children_back() {
        // A merge: the children go once the parent is in.
        let mut swap = Swap::default();
        let parent = k(5, 0, 0, 0);
        let children = parent.children();
        for child in children {
            swap.queued(child);
            swap.installed(child, false);
        }
        swap.step();
        swap.queued(parent);
        swap.queued(far(0));
        for child in children {
            swap.retire(child);
        }
        assert_eq!(swap.step(), Step::default());
        swap.installed(parent, false);
        let step = swap.step();
        assert_eq!(step.fade.len(), 8);
        assert_eq!(step.show, [parent]);
    }

    #[test]
    fn an_unshown_node_under_a_live_ancestor_stays_hidden() {
        let mut swap = Swap::default();
        let parent = k(4, 0, 0, 0);
        swap.queued(parent);
        swap.installed(parent, false);
        swap.step();
        // A child arrives, the parent is not removed, and something else waits.
        let child = parent.children()[3];
        swap.queued(child);
        swap.queued(far(0));
        swap.installed(child, false);
        assert_eq!(swap.step(), Step::default());
        // Another node elsewhere does show.
        let other = far(1);
        swap.queued(other);
        swap.installed(other, false);
        assert_eq!(swap.step().show, [other]);
        // The child shows once the parent goes.
        swap.retire(parent);
        let step = swap.step();
        assert_eq!(step.fade, [parent]);
        assert_eq!(step.show, [child]);
    }

    #[test]
    fn nothing_waiting_shows_and_fades_everything() {
        let mut swap = Swap::default();
        let parent = k(4, 0, 0, 0);
        let child = parent.children()[0];
        swap.queued(parent);
        swap.installed(parent, false);
        swap.step();
        swap.queued(far(0));
        swap.queued(child);
        swap.installed(child, false);
        assert_eq!(swap.step(), Step::default(), "blocked by the live parent");
        // The safety net: nothing waits, so the child shows beside the parent.
        swap.unwait(far(0));
        let step = swap.step();
        assert_eq!(step.show, [child]);
        // A removed node fades with nothing replacing it.
        swap.retire(parent);
        swap.queued(far(1));
        swap.retire(far(1));
        swap.unwait(far(1));
        let step = swap.step();
        assert_eq!(step.fade, [far(1), parent]);
        assert!(swap.is_idle());
    }

    #[test]
    fn a_removed_node_that_never_showed_goes_at_once() {
        let mut swap = Swap::default();
        let node = k(3, 0, 0, 0);
        swap.queued(node);
        swap.queued(far(0));
        swap.installed(node, false);
        swap.retire(node);
        assert_eq!(swap.step().fade, [node]);
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
        assert_eq!(swap.step().fade, [node]);
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
}

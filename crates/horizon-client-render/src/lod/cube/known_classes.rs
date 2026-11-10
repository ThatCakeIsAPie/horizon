//! The classes the classifier thread has found, kept in two generations.

use super::{CubeKey, NodeClass};
use crate::fast_hash::FxHashMap;

/// Classes kept in each generation of [`KnownClasses`].
const CLASS_NODES: usize = 1 << 18;

/// The classes found so far, in two generations of at most [`CLASS_NODES`]:
/// a full one becomes the older and a new one starts, so the classes of the
/// last answers are always all there. Emptied whole when full, as it was, it
/// kept only the newest descent's at Ultra (one answer holds 150,000 or so),
/// and the plan swung between that one's detail and the union's: a still
/// camera's far view went back and forth between levels, the descent asked
/// again at each plan's detail, for good.
#[derive(Default)]
pub(super) struct KnownClasses {
    newer: FxHashMap<CubeKey, NodeClass>,
    older: FxHashMap<CubeKey, NodeClass>,
}

impl KnownClasses {
    pub(super) fn get(&self, key: &CubeKey) -> Option<NodeClass> {
        self.newer.get(key).or_else(|| self.older.get(key)).copied()
    }

    /// Files `found`, starting a new generation first when they would not
    /// fit in this one.
    pub(super) fn extend(&mut self, found: impl ExactSizeIterator<Item = (CubeKey, NodeClass)>) {
        if self.newer.len() + found.len() > CLASS_NODES {
            self.older = std::mem::take(&mut self.newer);
        }
        self.newer.extend(found);
    }

    pub(super) fn insert(&mut self, key: CubeKey, class: NodeClass) {
        if self.newer.len() >= CLASS_NODES {
            self.older = std::mem::take(&mut self.newer);
        }
        self.newer.insert(key, class);
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.newer.len() + self.older.len()
    }
}

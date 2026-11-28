//! Counters of how much geometry the meshers make and the renderer draws, for measuring
//! merges and face-direction skipping (`HORIZON_GEOMETRY_STATS=1`; off, each hook costs one
//! relaxed load). `horizon-ctl status` shows them under `geometry` (`snapshot`).
//!
//! Near sections (`Mesher::mesh_reusing` on the meshing workers): sections meshed, their
//! solid quads, the quads a full greedy merge of them would leave (`mesher::greedy`'s count),
//! and the time meshing and that count took. Far nodes (`lod::mesh::mesh_cube`): nodes,
//! their unmerged voxel faces, the rects today's merge leaves, the quads emitted, the faces on
//! the node's own boundary (its closed shell) and the time. Drawn: per frame, the near and
//! far triangles drawn and what they would be with every face direction drawn.
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

static ON: OnceLock<bool> = OnceLock::new();
static FORCED: AtomicBool = AtomicBool::new(false);

/// Whether the counters run: `HORIZON_GEOMETRY_STATS=1`, or `force` (tests).
#[inline]
pub fn on() -> bool {
    *ON.get_or_init(|| std::env::var("HORIZON_GEOMETRY_STATS").is_ok_and(|v| v == "1"))
        || FORCED.load(Ordering::Relaxed)
}

/// Turns the counters on whatever the environment says.
pub fn force(on: bool) {
    FORCED.store(on, Ordering::Relaxed);
}

/// A counter.
pub struct Counter(AtomicU64);
impl Counter {
    const fn new() -> Self {
        Self(AtomicU64::new(0))
    }
    #[inline]
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }
    pub fn set(&self, n: u64) {
        self.0.store(n, Ordering::Relaxed);
    }
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Near sections meshed (with at least one solid quad), their solid quads, the quads a full
/// greedy merge would leave of them, nanoseconds meshing them, nanoseconds counting the merge.
pub static NEAR_SECTIONS: Counter = Counter::new();
pub static NEAR_QUADS: Counter = Counter::new();
pub static NEAR_GREEDY_QUADS: Counter = Counter::new();
pub static NEAR_MESH_NS: Counter = Counter::new();
pub static NEAR_GREEDY_NS: Counter = Counter::new();
/// Of `NEAR_QUADS`, those a greedy merge could take at all (whole unit faces with equal
/// corners): the merge's ceiling.
pub static NEAR_MERGEABLE_QUADS: Counter = Counter::new();
/// Near sections whose solid quads were merged when meshed (`mesher::greedy` built in), the
/// quads before and after.
pub static NEAR_MERGED_SECTIONS: Counter = Counter::new();
pub static NEAR_MERGED_BEFORE: Counter = Counter::new();
pub static NEAR_MERGED_AFTER: Counter = Counter::new();

/// Far cube nodes meshed on the CPU, their unmerged voxel faces, the rects today's merge
/// leaves, the quads emitted, those of them on the node's boundary planes, a plain one-pass
/// greedy merge's rects (for comparison), nanoseconds meshing.
pub static FAR_NODES: Counter = Counter::new();
pub static FAR_FACES: Counter = Counter::new();
pub static FAR_MERGED_RECTS: Counter = Counter::new();
pub static FAR_QUADS: Counter = Counter::new();
pub static FAR_SHELL_QUADS: Counter = Counter::new();
pub static FAR_GREEDY_RECTS: Counter = Counter::new();
pub static FAR_MESH_NS: Counter = Counter::new();

/// The last frame's drawn triangles: near sections (CPU path) as drawn and with every
/// direction; far tiles on the GPU (`far_cull`'s statistics, a few frames late) as drawn
/// and with every direction of the kept records.
pub static FRAME_NEAR_TRIANGLES: Counter = Counter::new();
pub static FRAME_NEAR_ALL_TRIANGLES: Counter = Counter::new();
pub static FRAME_FAR_TRIANGLES: Counter = Counter::new();
pub static FRAME_FAR_ALL_TRIANGLES: Counter = Counter::new();

/// Every counter by name, for `horizon-ctl status` (`geometry`); `None` when off.
pub fn snapshot() -> Option<serde_json::Value> {
    if !on() {
        return None;
    }
    let ms = |c: &Counter| c.get() as f64 / 1e6;
    Some(serde_json::json!({
        "near_sections": NEAR_SECTIONS.get(),
        "near_quads": NEAR_QUADS.get(),
        "near_greedy_quads": NEAR_GREEDY_QUADS.get(),
        "near_mergeable_quads": NEAR_MERGEABLE_QUADS.get(),
        "near_mesh_ms": ms(&NEAR_MESH_NS),
        "near_greedy_ms": ms(&NEAR_GREEDY_NS),
        "near_merged_sections": NEAR_MERGED_SECTIONS.get(),
        "near_merged_before": NEAR_MERGED_BEFORE.get(),
        "near_merged_after": NEAR_MERGED_AFTER.get(),
        "far_nodes": FAR_NODES.get(),
        "far_faces": FAR_FACES.get(),
        "far_merged_rects": FAR_MERGED_RECTS.get(),
        "far_quads": FAR_QUADS.get(),
        "far_shell_quads": FAR_SHELL_QUADS.get(),
        "far_greedy_rects": FAR_GREEDY_RECTS.get(),
        "far_mesh_ms": ms(&FAR_MESH_NS),
        "frame_near_triangles": FRAME_NEAR_TRIANGLES.get(),
        "frame_near_all_triangles": FRAME_NEAR_ALL_TRIANGLES.get(),
        "frame_far_triangles": FRAME_FAR_TRIANGLES.get(),
        "frame_far_all_triangles": FRAME_FAR_ALL_TRIANGLES.get(),
    }))
}

//! Far cube meshes of realistic sample nodes (rolling hills, a canyon, a sea with a shore, a
//! forest, at several levels): the merged mesh draws exactly the surface of the unmerged one, covers
//! every exposed voxel side once, and has fewer quads than the old two-pass merge. `measure_*`
//! (ignored) prints quads and milliseconds a node.
use super::*;
use crate::lod::cube::CubeKey;
use std::time::Instant;

const AIR: BlockStateId = BlockStateId(0);
const STONE: BlockStateId = BlockStateId(1);
const GRASS: BlockStateId = BlockStateId(2);
const WATER: BlockStateId = BlockStateId(3);
const DIRT: BlockStateId = BlockStateId(4);
const SAND: BlockStateId = BlockStateId(5);
const LOG: BlockStateId = BlockStateId(6);
const LEAVES: BlockStateId = BlockStateId(7);
const SEA: f64 = 63.0;

fn hash(x: i64, z: i64, seed: u64) -> u64 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ (z as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^ (h >> 32)
}
fn unit(x: i64, z: i64, seed: u64) -> f64 {
    (hash(x, z, seed) >> 11) as f64 / (1u64 << 53) as f64
}
/// Smooth value noise, one cell per `wave` blocks.
fn noise(x: f64, z: f64, wave: f64, seed: u64) -> f64 {
    let (fx, fz) = (x / wave, z / wave);
    let (ix, iz) = (fx.floor(), fz.floor());
    let (tx, tz) = (fx - ix, fz - iz);
    let (sx, sz) = (tx * tx * (3.0 - 2.0 * tx), tz * tz * (3.0 - 2.0 * tz));
    let v = |dx: i64, dz: i64| unit(ix as i64 + dx, iz as i64 + dz, seed);
    let a = v(0, 0) + (v(1, 0) - v(0, 0)) * sx;
    let b = v(0, 1) + (v(1, 1) - v(0, 1)) * sx;
    a + (b - a) * sz
}
fn fbm(x: f64, z: f64) -> f64 {
    noise(x, z, 256.0, 1) * 60.0
        + noise(x, z, 128.0, 2) * 30.0
        + noise(x, z, 64.0, 3) * 15.0
        + noise(x, z, 32.0, 4) * 7.0
        + noise(x, z, 8.0, 5) * 2.0
}

#[derive(Clone, Copy)]
enum Land {
    Hills,
    Canyon,
    Sea,
    Forest,
}
impl Land {
    fn height(self, x: f64, z: f64) -> f64 {
        match self {
            Land::Hills | Land::Forest => 40.0 + fbm(x, z),
            // Terraces of 14 blocks with sheer sides.
            Land::Canyon => {
                let h = 20.0 + noise(x, z, 200.0, 7) * 150.0 + noise(x, z, 40.0, 8) * 20.0;
                (h / 14.0).floor() * 14.0
            }
            Land::Sea => 20.0 + fbm(x, z) * 0.7,
        }
    }
}

/// A node of `land` at `level` holding the terrain's surface near block column `at`, whole voxels
/// of grass (sand by the sea), dirt and stone, water to the sea level (`Sea`), trees (`Forest`).
fn sample(land: Land, level: u8, at: [f64; 2], sea: Option<i32>) -> CubeNode {
    let scale = f64::from(1u32 << level);
    let size = 16.0 * scale;
    let (kx, kz) = ((at[0] / size).floor() as i32, (at[1] / size).floor() as i32);
    let h = land.height(at[0], at[1]);
    let ky = match land {
        // The sea level in the node's upper part.
        Land::Sea => (SEA / size).floor() as i32,
        _ => (h / size).floor() as i32,
    };
    let key = CubeKey::new(level, kx, ky, kz);
    let [ox, oy, oz] = key.origin();
    let voxel = |x: usize, y: usize, z: usize| {
        let (bx, bz) = (
            f64::from(ox) + x as f64 * scale,
            f64::from(oz) + z as f64 * scale,
        );
        let by = f64::from(oy) + y as f64 * scale;
        let h = land.height(bx + scale / 2.0, bz + scale / 2.0);
        let biome = (noise(bx, bz, 300.0, 9) * 3.0) as u16 + 1;
        let depth = h - by;
        let state = if depth <= 0.0 {
            if sea.is_some_and(|top| by < f64::from(top)) {
                WATER
            } else {
                AIR
            }
        } else if depth <= scale {
            if h < SEA + 2.0 { SAND } else { GRASS }
        } else if depth <= scale + 4.0 {
            if h < SEA + 2.0 { SAND } else { DIRT }
        } else {
            STONE
        };
        let state = match (land, state) {
            (Land::Forest, AIR) => tree(bx, by, bz, h, scale).unwrap_or(AIR),
            (_, s) => s,
        };
        Voxel { state, biome }
    };
    CubeNode::from_fn(key, voxel)
}
/// A tree of 1-in-~60 columns of 5 blocks' spacing, drawn at the voxels it covers.
fn tree(bx: f64, by: f64, bz: f64, ground: f64, scale: f64) -> Option<BlockStateId> {
    let cell = 6.0;
    let (cx, cz) = ((bx / cell).floor() as i64, (bz / cell).floor() as i64);
    for (dx, dz) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
        let (tx, tz) = (cx + dx, cz + dz);
        if unit(tx, tz, 11) > 0.45 {
            continue;
        }
        let px = (tx as f64 + 0.5) * cell + (unit(tx, tz, 12) - 0.5) * 3.0;
        let pz = (tz as f64 + 0.5) * cell + (unit(tx, tz, 13) - 0.5) * 3.0;
        let base = ground;
        let top = base + 5.0 + unit(tx, tz, 14) * 3.0;
        let (ex, ez) = ((bx + scale / 2.0 - px).abs(), (bz + scale / 2.0 - pz).abs());
        let y = by + scale / 2.0;
        if ex < scale.max(1.0) / 1.0 * 0.6 && ez < scale.max(1.0) * 0.6 && y > base && y < top {
            return Some(LOG);
        }
        let radius = 2.6 - (y - (top - 3.0)).max(0.0) * 0.6;
        if y > top - 3.5 && y < top + 1.5 && ex.hypot(ez) <= radius.max(0.5) {
            return Some(LEAVES);
        }
    }
    None
}

/// Sample nodes: (name, node, sea surface).
fn samples() -> Vec<(String, CubeNode, Option<i32>)> {
    let mut out = Vec::new();
    for (land, name, at) in [
        (Land::Hills, "hills", [1234.0, 777.0]),
        (Land::Hills, "hills2", [-9000.0, 4096.0]),
        (Land::Canyon, "canyon", [5200.0, -3100.0]),
        (Land::Forest, "forest", [310.0, 2200.0]),
    ] {
        for level in [0, 1, 2, 4, 6] {
            if matches!(land, Land::Forest) && level > 2 {
                continue;
            }
            out.push((
                format!("{name} L{level}"),
                sample(land, level, at, None),
                None,
            ));
        }
    }
    // The sea: a node whose top is the sea's surface (the sea ends there), and one it cuts.
    for (level, top) in [(2, 64), (3, 68), (4, 64)] {
        let node = sample(Land::Sea, level, [800.0, 800.0], Some(top));
        out.push((format!("sea L{level}"), node, Some(top)));
    }
    out
}

/// A sample's meshing: the test materials (a grass top is another layer than its sides, a biome
/// tints), or the real ones when `HORIZON_TEST_RESOURCES` points at a client's resources.
struct Scene {
    real: Option<(std::sync::Arc<WorkerContext>, [BlockStateId; 8])>,
    mesher: std::cell::RefCell<Mesher>,
}
impl Scene {
    fn synthetic() -> Self {
        Self {
            real: None,
            mesher: Default::default(),
        }
    }
    /// The client's real far styles and tints, when `HORIZON_TEST_RESOURCES` is set.
    fn real() -> Option<Self> {
        use crate::{assets::Assets, biome::Biomes, light::LightTables, mesher::MeshTables};
        use horizon_assets::resource_source::ResourceRoot;
        let root = std::path::PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES")?);
        let assets = std::sync::Arc::new(Assets::load(&root).ok()?);
        let render =
            std::sync::Arc::new(assets.reload(&ResourceRoot::directory(root.clone())).ok()?);
        let facts = crate::blocks::BlockFacts::load(&root, assets.registry).ok()?;
        let biomes =
            std::sync::Arc::new(Biomes::load(&ResourceRoot::directory(root.clone())).ok()?);
        let tables = std::sync::Arc::new(
            MeshTables::new(
                &assets,
                &render,
                &facts,
                crate::models::load_visibility(&root, assets.registry).ok()?,
                None,
            )
            .ok()?,
        );
        let context = std::sync::Arc::new(WorkerContext {
            light: std::sync::Arc::new(LightTables::load(&root, assets.registry, &facts).ok()?),
            tables,
            assets: assets.clone(),
            render,
            biomes,
            hidden_states: std::sync::Arc::from([]),
            part_tables: Default::default(),
        });
        let registry = &context.assets.registry;
        let id = |name: &str| registry.default_state(name);
        let ids = [
            id("minecraft:air")?,
            id("minecraft:stone")?,
            id("minecraft:grass_block")?,
            id("minecraft:water")?,
            id("minecraft:dirt")?,
            id("minecraft:sand")?,
            id("minecraft:oak_log")?,
            id("minecraft:oak_leaves")?,
        ];
        Some(Self {
            real: Some((context, ids)),
            mesher: Default::default(),
        })
    }
    fn material(&self, span: &Span, direction: Direction, at: [i32; 2]) -> Material<()> {
        if let Some((context, ids)) = &self.real {
            let mut span = *span;
            span.state = ids[span.state.0 as usize];
            return self
                .mesher
                .borrow_mut()
                .material(&span, direction, at, context);
        }
        let water = span.state == WATER;
        let layer = match (span.state, direction) {
            (GRASS, Direction::Up) => 20,
            (GRASS, Direction::Down) => DIRT.0 as u16,
            (GRASS, _) => 21,
            (s, _) => s.0 as u16,
        };
        let tint = matches!(span.state, GRASS | LEAVES);
        let rgb = if tint {
            0x4090_20 + u32::from(span.biome) * 0x0003_0201
        } else {
            0xffffff
        };
        let shade = (direction_shade(direction) * 255.0) as u32;
        Material {
            faces: [
                Some(Paint {
                    layer,
                    light: span.light,
                    color: (rgb & 0xffffff) | shade << 24,
                    translucent: water,
                    look: (),
                }),
                None,
            ],
            fluid: u8::from(water),
            opaque: !water,
        }
    }
    fn through(&self, state: BlockStateId) -> bool {
        match &self.real {
            Some((context, ids)) => lets_sky_through(context, ids[state.0 as usize]),
            None => state == LEAVES,
        }
    }
    fn faces(&self, node: &CubeNode, sea: Option<i32>) -> Faces<()> {
        voxel_faces(
            node,
            AIR,
            sea,
            |s, d, at| self.material(s, d, at),
            |s| self.through(s),
        )
    }
    fn mesh(&self, node: &CubeNode, sea: Option<i32>) -> SectionMesh {
        let mut mesh = SectionMesh::default();
        mesh_voxels(
            node,
            AIR,
            sea,
            |s, d, at| self.material(s, d, at),
            |s| self.through(s),
            &mut mesh,
        );
        mesh.sort_solid();
        mesh
    }
}

/// A far quad decoded from its vertices: what it merges by, and its rect.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct Quad {
    axis: usize,
    positive: bool,
    plane: f32,
    layer: u16,
    color: u32,
    light: u16,
    translucent: bool,
    rect: [f32; 4],
}
fn quads_of(mesh: &SectionMesh) -> Vec<Quad> {
    let mut out = Vec::new();
    for (vertices, translucent) in [(&mesh.solid, false), (&mesh.translucent, true)] {
        for q in vertices.as_chunks::<4>().0 {
            let p = |i: usize| glam::Vec3::from(q[i].pos);
            let normal = (p(1) - p(0)).cross(p(2) - p(0));
            let axis = (0..3)
                .find(|&a| q.iter().all(|v| v.pos[a] == q[0].pos[a]))
                .expect("a flat quad");
            let (a, b) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let range = |i: usize| {
                let lo = q.iter().map(|v| v.pos[i]).fold(f32::INFINITY, f32::min);
                let hi = q.iter().map(|v| v.pos[i]).fold(f32::NEG_INFINITY, f32::max);
                (lo, hi)
            };
            let (ua, ub) = (range(a), range(b));
            // The sprite repeats per block from the quad's corner: its UVs are whole blocks.
            for v in q {
                for uv in v.uv {
                    assert_eq!(uv % 32, 0, "a tiled UV is whole blocks");
                }
            }
            out.push(Quad {
                axis,
                positive: normal[axis] > 0.0,
                plane: q[0].pos[axis],
                layer: q[0].layer,
                color: q[0].color,
                light: q[0].light,
                translucent,
                rect: [ua.0, ub.0, ua.1, ub.1],
            });
        }
    }
    out
}

/// The unmerged mesh: every rect on its own.
fn unmerged(faces: &Faces<()>) -> SectionMesh {
    let mut mesh = SectionMesh::default();
    for (&key, rects) in faces {
        for &rect in rects {
            emit(&mut mesh, key, rect);
        }
    }
    mesh.finish_sprite_layers();
    mesh
}
/// The mesh the old two-pass `merge` makes.
fn two_pass(faces: &Faces<()>) -> SectionMesh {
    let mut mesh = SectionMesh::default();
    for (&key, rects) in faces {
        let mut rects = rects.clone();
        merge(&mut rects);
        for rect in rects {
            emit(&mut mesh, key, rect);
        }
    }
    mesh.finish_sprite_layers();
    mesh
}

type Key = (usize, bool, u32, u16, u32, u16, bool);
fn group(quads: &[Quad]) -> BTreeMap<Key, Vec<[f32; 4]>> {
    let mut by: BTreeMap<Key, Vec<[f32; 4]>> = BTreeMap::new();
    for q in quads {
        let key = (
            q.axis,
            q.positive,
            q.plane.to_bits(),
            q.layer,
            q.color,
            q.light,
            q.translucent,
        );
        by.entry(key).or_default().push(q.rect);
    }
    by
}
/// How many of `rects` cover each cell of the grid the rects of `a` and `b` cut a plane into.
fn coverage(rects: &[[f32; 4]], xs: &[f32], ys: &[f32]) -> Vec<u16> {
    let at = |v: &[f32], x: f32| v.iter().position(|&c| c == x).unwrap();
    let mut count = vec![0u16; xs.len() * ys.len()];
    for r in rects {
        for j in at(ys, r[1])..at(ys, r[3]) {
            for i in at(xs, r[0])..at(xs, r[2]) {
                count[j * xs.len() + i] += 1;
            }
        }
    }
    count
}
fn sorted(mut v: Vec<f32>) -> Vec<f32> {
    v.sort_by(f32::total_cmp);
    v.dedup();
    v
}
/// Whether `a` draws the surface `b` does, every point once (no overlap, no hole in what `b` draws).
fn same_surface(a: &[Quad], b: &[Quad]) -> Result<(), String> {
    let (ga, gb) = (group(a), group(b));
    if ga.keys().ne(gb.keys()) {
        return Err(format!(
            "other plane/material sets: {} vs {} keys",
            ga.len(),
            gb.len()
        ));
    }
    for (key, ra) in &ga {
        let rb = &gb[key];
        let xs = sorted(ra.iter().chain(rb).flat_map(|r| [r[0], r[2]]).collect());
        let ys = sorted(ra.iter().chain(rb).flat_map(|r| [r[1], r[3]]).collect());
        let (ca, cb) = (coverage(ra, &xs, &ys), coverage(rb, &xs, &ys));
        if ca != cb {
            return Err(format!("coverage differs on plane {key:?}"));
        }
        if ca.iter().any(|&c| c > 1) {
            return Err(format!("a point drawn twice on plane {key:?}"));
        }
    }
    Ok(())
}

#[test]
fn merged_surface_is_the_unmerged_surface() {
    let scene = Scene::synthetic();
    for (name, node, sea) in samples() {
        let faces = scene.faces(&node, sea);
        assert!(
            node.as_uniform().is_none(),
            "{name}: the node holds one voxel"
        );
        let reference = quads_of(&unmerged(&faces));
        let merged = quads_of(&scene.mesh(&node, sea));
        assert!(!reference.is_empty(), "{name}: nothing drawn");
        if let Err(e) = same_surface(&merged, &reference) {
            panic!("{name}: {e}");
        }
        // The old merge is held to the same.
        if let Err(e) = same_surface(&quads_of(&two_pass(&faces)), &reference) {
            panic!("{name}: two-pass: {e}");
        }
    }
}

/// Every exposed voxel side of the node, by the rules of `mesh_voxels` written out plainly: a
/// non-air voxel's side is drawn toward air, a voxel that does not hide it, and the node's edge
/// (a fluid's only inside the node).
fn exposed(node: &CubeNode) -> std::collections::BTreeSet<(usize, bool, i32, i32, i32)> {
    let opaque = |s: BlockStateId| s != AIR && s != WATER;
    let fluid = |s: BlockStateId| s == WATER;
    let mut set = std::collections::BTreeSet::new();
    for y in 0..16usize {
        for z in 0..16usize {
            for x in 0..16usize {
                let v = node.get(x, y, z).state;
                if v == AIR {
                    continue;
                }
                for axis in 0..3 {
                    for positive in [false, true] {
                        let mut n = [x as i32, y as i32, z as i32];
                        n[axis] += if positive { 1 } else { -1 };
                        let inside = n.iter().all(|c| (0..16).contains(c));
                        let shown = if inside {
                            let next = node.get(n[0] as usize, n[1] as usize, n[2] as usize).state;
                            next != v
                                && (next == AIR || !(opaque(next) || (fluid(next) && fluid(v))))
                        } else {
                            !fluid(v)
                        };
                        if shown {
                            let plane = [x, y, z][axis] as i32 + i32::from(positive);
                            let mut cell = [x as i32, y as i32, z as i32];
                            cell[axis] = plane;
                            set.insert((axis, positive, cell[0], cell[1], cell[2]));
                        }
                    }
                }
            }
        }
    }
    set
}

#[test]
fn every_exposed_side_is_covered_exactly_once() {
    let scene = Scene::synthetic();
    for (name, node, sea) in samples() {
        // Sea nodes cut a fluid at its surface (fractional): their sides are checked above.
        if sea.is_some() {
            continue;
        }
        let want = exposed(&node);
        let scale = (1u32 << node.key.level) as f32;
        let mut got = std::collections::BTreeMap::<_, u32>::new();
        let mesh = scene.mesh(&node, sea);
        for q in quads_of(&mesh) {
            let (a, b) = match q.axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let (ca, cb) = (
                (q.rect[0] / scale) as i32..(q.rect[2] / scale) as i32,
                (q.rect[1] / scale) as i32..(q.rect[3] / scale) as i32,
            );
            for ib in cb.clone() {
                for ia in ca.clone() {
                    let mut cell = [0i32; 3];
                    cell[q.axis] = (q.plane / scale) as i32;
                    cell[a] = ia;
                    cell[b] = ib;
                    *got.entry((q.axis, q.positive, cell[0], cell[1], cell[2]))
                        .or_default() += 1;
                }
            }
        }
        for (cell, n) in &got {
            assert_eq!(*n, 1, "{name}: {cell:?} drawn {n} times");
        }
        let missing: Vec<_> = want
            .iter()
            .filter(|c| !got.contains_key(*c))
            .take(5)
            .collect();
        assert!(missing.is_empty(), "{name}: holes at {missing:?}");
        let extra: Vec<_> = got.keys().filter(|c| !want.contains(*c)).take(5).collect();
        assert!(
            extra.is_empty(),
            "{name}: faces nothing exposes at {extra:?}"
        );
    }
}

fn quads_count(m: &SectionMesh) -> usize {
    (m.solid.len() + m.translucent.len()) / 4
}

#[test]
fn merging_leaves_far_fewer_quads_than_one_a_side() {
    let scene = Scene::synthetic();
    let (mut cells, mut old, mut new) = (0, 0, 0);
    for (name, node, sea) in samples() {
        let faces = scene.faces(&node, sea);
        let one_each = faces.values().map(Vec::len).sum::<usize>();
        let (o, n) = (
            quads_count(&two_pass(&faces)),
            quads_count(&scene.mesh(&node, sea)),
        );
        println!("{name:14} cells {one_each:5} two-pass {o:5} now {n:5}");
        assert!(n <= o, "{name}: {n} quads after {o} of the two-pass merge");
        cells += one_each;
        old += o;
        new += n;
    }
    println!("total: cells {cells}, two-pass {old}, now {new}");
    assert!(new * 2 < cells);
}

/// The mesh as before: the two-pass merge.
fn legacy(scene: &Scene, node: &CubeNode, sea: Option<i32>) -> SectionMesh {
    let mut mesh = two_pass(&scene.faces(node, sea));
    mesh.sort_solid();
    mesh
}
fn average_ms(runs: u32, mut f: impl FnMut()) -> f64 {
    f();
    let started = Instant::now();
    for _ in 0..runs {
        f();
    }
    started.elapsed().as_secs_f64() * 1e3 / f64::from(runs)
}
/// Returns (tris before, tris now).
fn time(name: &str, scene: &Scene, node: &CubeNode, sea: Option<i32>, runs: u32) -> (usize, usize) {
    let mesh = scene.mesh(node, sea);
    let old = legacy(scene, node, sea);
    let now = average_ms(runs, || drop(std::hint::black_box(scene.mesh(node, sea))));
    let before = average_ms(runs, || {
        drop(std::hint::black_box(legacy(scene, node, sea)))
    });
    let faces = average_ms(runs, || drop(std::hint::black_box(scene.faces(node, sea))));
    println!(
        "{name:14} tris {:5} -> {:5}  ms/node {before:7.3} -> {now:7.3}  (face collection {faces:6.3})",
        quads_count(&old) * 2,
        quads_count(&mesh) * 2,
    );
    (quads_count(&old) * 2, quads_count(&mesh) * 2)
}

fn measure(scene: &Scene) {
    let (mut before, mut now, mut count) = (0, 0, 0);
    for (name, node, sea) in samples() {
        let (b, n) = time(&name, scene, &node, sea, 100);
        before += b;
        now += n;
        count += 1;
    }
    println!(
        "mean tris/node over {count} nodes: {} -> {}",
        before / count,
        now / count
    );
}

#[test]
#[ignore = "measurement: cargo test --profile fast -p horizon-client-render --lib lod::mesh::sample_tests::measure -- --ignored --nocapture"]
fn measure_synthetic_nodes() {
    measure(&Scene::synthetic());
}

#[test]
#[ignore = "measurement with the real styles: needs HORIZON_TEST_RESOURCES"]
fn measure_nodes_with_the_real_styles() {
    let Some(scene) = Scene::real() else {
        println!("HORIZON_TEST_RESOURCES not set or unreadable: skipped");
        return;
    };
    measure(&scene);
}

#[test]
fn merged_cells_are_the_cells_in_no_more_rects_than_the_row_runs() {
    let mut seed = 1u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let (mut old, mut new) = (0, 0);
    for case in 0..400 {
        let density = 1 + case % 9;
        let unit = [1.0f32, 2.0, 64.0][case as usize % 3];
        let cells: Vec<(u32, u32)> = (0..16)
            .flat_map(|v| (0..16).map(move |u| (u, v)))
            .filter(|_| next() % 10 < density)
            .collect();
        if cells.is_empty() {
            continue;
        }
        let rect = |&(u, v): &(u32, u32)| Rect {
            u0: u as f32 * unit,
            v0: v as f32 * unit,
            u1: (u + 1) as f32 * unit,
            v1: (v + 1) as f32 * unit,
        };
        let mut a: Vec<Rect> = cells.iter().map(rect).collect();
        let mut b = a.clone();
        merge(&mut b);
        assert!(merge_cells(&mut a, unit));
        // Every cell once, nothing else.
        let mut seen = std::collections::BTreeMap::<(u32, u32), u32>::new();
        for r in &a {
            for v in (r.v0 / unit) as u32..(r.v1 / unit) as u32 {
                for u in (r.u0 / unit) as u32..(r.u1 / unit) as u32 {
                    *seen.entry((u, v)).or_default() += 1;
                }
            }
        }
        assert_eq!(seen.len(), cells.len(), "case {case}");
        assert!(cells.iter().all(|c| seen[c] == 1), "case {case}");
        assert!(
            a.len() <= b.len(),
            "case {case}: {} rects, merge: {}",
            a.len(),
            b.len()
        );
        old += b.len();
        new += a.len();
    }
    println!("random planes: {old} rects with merge, {new} with merge_cells");
    assert!(new < old);
}

#[test]
fn merge_cells_leaves_fluid_cuts_to_merge() {
    // A cell cut short by a fluid surface is not a whole cell.
    let mut rects = vec![
        Rect {
            u0: 0.0,
            v0: 0.0,
            u1: 8.0,
            v1: 4.0,
        },
        Rect {
            u0: 8.0,
            v0: 0.0,
            u1: 16.0,
            v1: 8.0,
        },
    ];
    let before = rects.clone();
    assert!(!merge_cells(&mut rects, 8.0));
    assert_eq!(rects, before);
}

#[test]
#[ignore = "measurement: how many quads lie on the node's own border planes"]
fn measure_border_quads() {
    let scene = Scene::synthetic();
    let (mut all, mut border, mut one_cell) = (0, 0, 0);
    for (name, node, sea) in samples() {
        let size = (16u32 << node.key.level) as f32;
        let scale = (1u32 << node.key.level) as f32;
        let quads = quads_of(&scene.mesh(&node, sea));
        let on_border = quads
            .iter()
            .filter(|q| q.plane == 0.0 || q.plane == size)
            .count();
        let small = quads
            .iter()
            .filter(|q| (q.rect[2] - q.rect[0]) <= scale && (q.rect[3] - q.rect[1]) <= scale)
            .count();
        println!(
            "{name:14} quads {:4} on a border plane {on_border:4} one-cell {small:4}",
            quads.len()
        );
        all += quads.len();
        border += on_border;
        one_cell += small;
    }
    println!(
        "total {all}: border {border} ({:.0}%), one-cell {one_cell} ({:.0}%)",
        border as f64 * 100.0 / all as f64,
        one_cell as f64 * 100.0 / all as f64
    );
}

// The terrain visibility buffer's triangle math (gfx/vis_terrain.rs): it stands alone, so
// the resolve's reconstruction is tested against the rasteriser itself (vis_terrain.rs).

// A framebuffer position's NDC in a target of `size`.
fn vis_ndc_in(at: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return vec2(at.x / size.x * 2.0 - 1.0, 1.0 - at.y / size.y * 2.0);
}

// visbuffer.wgsl's `vis_bary` and `vis_rows`: perspective-correct barycentrics of the point at
// NDC `p` on the triangle with clip-space corners (x, y, w) a, b, c (homogeneous 2D
// rasterisation, Olano and Greer; corners behind the camera need no clipping).
fn vis_terrain_bary(rows: mat3x3<f32>, p: vec2<f32>) -> vec3<f32> {
    let l = vec3(p, 1.0) * rows;
    let sum = l.x + l.y + l.z;
    if abs(sum) < 1e-30 {
        return vec3(-1.0);
    }
    return l / sum;
}
fn vis_terrain_rows(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> mat3x3<f32> {
    return mat3x3(cross(b, c), cross(c, a), cross(a, b));
}

// Whether the pixel at NDC `p` lies in an arena quad's second triangle, 0 2 3, rather than
// 0 1 2: corner 1's weight in 0 1 2 is negative there, the sign of its edge function over
// the triangle's determinant (visbuffer.wgsl's test). A degenerate 0 1 2 leaves it to 0 2 3.
fn vis_second(p: vec2<f32>, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>) -> bool {
    let diagonal = cross(c2.xyw, c0.xyw);
    let det = dot(c1.xyw, diagonal);
    return dot(vec3(p, 1.0), diagonal) * det < 0.0 || det == 0.0;
}

// The weights of a pixel at framebuffer position `at` on the triangle with clip-space corners
// a b c in a target of `size`, and of its neighbours in its 2x2 quad's row and column, with
// the signs that make neighbour minus pixel odd minus even (as dpdxFine/dpdyFine pair them;
// the neighbours lie on the same triangle's plane, as the rasteriser's helper pixels do).
struct VisWeights {
    here: vec3<f32>,
    x: vec3<f32>,
    y: vec3<f32>,
    toward: vec2<f32>,
}
fn vis_weights(at: vec2<f32>, size: vec2<f32>, a: vec4<f32>, b: vec4<f32>,
    c: vec4<f32>) -> VisWeights {
    let rows = vis_terrain_rows(a.xyw, b.xyw, c.xyw);
    let odd = vec2<f32>(vec2<i32>(floor(at)) & vec2(1));
    let toward = vec2(1.0) - 2.0 * odd;
    return VisWeights(vis_terrain_bary(rows, vis_ndc_in(at, size)),
        vis_terrain_bary(rows, vis_ndc_in(at + vec2(toward.x, 0.0), size)),
        vis_terrain_bary(rows, vis_ndc_in(at + vec2(0.0, toward.y), size)), toward);
}
// A vec2 with corner values va vb vc at the pixel, and its screen gradients.
fn vis_at2(w: VisWeights, va: vec2<f32>, vb: vec2<f32>, vc: vec2<f32>) -> vec2<f32> {
    return w.here.x * va + w.here.y * vb + w.here.z * vc;
}
fn vis_dx2(w: VisWeights, va: vec2<f32>, vb: vec2<f32>, vc: vec2<f32>) -> vec2<f32> {
    return (w.x.x * va + w.x.y * vb + w.x.z * vc - vis_at2(w, va, vb, vc)) * w.toward.x;
}
fn vis_dy2(w: VisWeights, va: vec2<f32>, vb: vec2<f32>, vc: vec2<f32>) -> vec2<f32> {
    return (w.y.x * va + w.y.y * vb + w.y.z * vc - vis_at2(w, va, vb, vc)) * w.toward.y;
}

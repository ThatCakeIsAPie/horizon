
// GPU-meshed far nodes (cube_far.wgsl's `cf_quad_vertex`) in the terrain visibility buffer
// (vis_terrain.wgsl): VIS_CUBE, the node, the triangle's provoking vertex.
@vertex fn vs_vis_cube(@builtin(vertex_index) vertex: u32,
    @builtin(instance_index) node: u32) -> VisIdWire {
    return vis_id_wire(cf_quad_vertex(vertex, node),
        vec2(vertex, (VIS_CUBE << VIS_FAMILY_SHIFT) | (node & VIS_LOW_MASK)));
}

@fragment fn fs_vis_resolve_cube(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let id = vis_load(at.xy);
    if vis_family(id) != VIS_CUBE { discard; }
    let node = id.y & VIS_LOW_MASK;
    // The triangle's three vertices (corners 0 1 2, or 0 2 3 from the quad's fourth vertex).
    return vis_shade(at.xy, cf_quad_vertex(id.x, node), cf_quad_vertex(id.x + 1u, node),
        cf_quad_vertex(id.x + 2u, node));
}

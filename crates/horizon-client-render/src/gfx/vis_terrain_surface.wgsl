
// Surface tiles (surface_draw.wgsl's `st_surface_vertex`) in the terrain visibility buffer
// (vis_terrain.wgsl): VIS_SURFACE, the tile's pool slot, the triangle's provoking vertex.
@vertex fn vs_vis_surface(@builtin(vertex_index) vertex: u32,
    @builtin(instance_index) slot: u32) -> VisIdWire {
    return vis_id_wire(st_surface_vertex(vertex, slot),
        vec2(vertex, (VIS_SURFACE << VIS_FAMILY_SHIFT) | (slot & VIS_LOW_MASK)));
}

@fragment fn fs_vis_resolve_surface(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let id = vis_load(at.xy);
    if vis_family(id) != VIS_SURFACE { discard; }
    let slot = id.y & VIS_LOW_MASK;
    // The triangle's three vertices (corners 0 1 2, or 0 2 3 from the quad's fourth vertex).
    return vis_shade(at.xy, st_surface_vertex(id.x, slot), st_surface_vertex(id.x + 1u, slot),
        st_surface_vertex(id.x + 2u, slot));
}


// The arena families of the terrain visibility buffer (vis_terrain.wgsl): near sections
// (VIS_SECTION, origins from `Renderer::instances`) and Earth's CPU-meshed far tiles drawn by
// far_cull (VIS_FAR_TILE, origins from its instances), both through `vs_section`'s
// `section_corner` + `corner_vertex` on the arena page of group 1, whose `vis_page` holds the
// family and page (VIS_TAG_MASK bits: vis_terrain.rs's `PageGroups`).
@group(1) @binding(39) var<uniform> vis_page: vec4<u32>;

@vertex fn vs_vis_section(@builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32, @location(5) origin: vec4<f32>) -> VisIdWire {
    return vis_id_wire(corner_vertex(section_corner(vertex).c, origin),
        vec2(vertex, vis_page.x | (instance & VIS_INSTANCE_MASK)));
}

fn vis_section_corner(vertex: u32, origin: vec4<f32>) -> TerrainOut {
    return corner_vertex(section_corner(vertex).c, origin);
}

@fragment fn fs_vis_resolve_section(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let id = vis_load(at.xy);
    // A tag has a family, so pixels with no terrain never match.
    if (id.y & VIS_TAG_MASK) != vis_page.x { discard; }
    let origin = vis_instances[id.y & VIS_INSTANCE_MASK];
    let a = vis_section_corner(id.x, origin);
    var b = vis_section_corner(id.x + 1u, origin);
    var c = vis_section_corner(id.x + 2u, origin);
    if vis_second(vis_ndc(at.xy), a.clip, b.clip, c.clip) {
        b = c;
        c = vis_section_corner(id.x + 3u, origin);
    }
    return vis_shade(at.xy, a, b, c);
}

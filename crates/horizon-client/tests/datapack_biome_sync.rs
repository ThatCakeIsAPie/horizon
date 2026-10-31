//! A data-pack biome is in the synchronized registry, sent in full, at the id
//! the chunk encoder and the client's table give it.
use horizon_client::{
    biome::Biomes,
    chunk::LevelShapes,
    lan::data::{BIOME_REGISTRY, SyncData},
};
use serde_json::Value;
use std::path::PathBuf;

/// Removes the folder when the test ends, passed or not.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

#[test]
fn a_pack_biome_is_synchronized_with_the_id_the_encoder_uses() {
    let root = std::env::temp_dir().join(format!("horizon-sync-biomes-{}", std::process::id()));
    let _scratch = Scratch(root.clone());
    let maps = root.join("assets/minecraft/textures/colormap");
    let biome_dir = root.join("data/minecraft/worldgen/biome");
    std::fs::create_dir_all(&maps).unwrap();
    std::fs::create_dir_all(&biome_dir).unwrap();
    std::fs::write(root.join("registries.json"), "{}").unwrap();
    for name in ["grass", "foliage", "dry_foliage"] {
        let file = std::fs::File::create(maps.join(format!("{name}.png"))).unwrap();
        let mut png = png::Encoder::new(file, 1, 1);
        png.set_color(png::ColorType::Rgb);
        png.set_depth(png::BitDepth::Eight);
        let mut writer = png.write_header().unwrap();
        writer.write_image_data(&[1, 2, 3]).unwrap();
    }
    for name in ["plains", "river"] {
        std::fs::write(biome_dir.join(format!("{name}.json")), "{}").unwrap();
    }
    let base = Biomes::load(&root).unwrap();
    let pack: Value = serde_json::from_str(r##"{"effects":{"grass_color":"#123456"}}"##).unwrap();
    let world = base.with_packs([("minecraft:meadow", &pack)]);
    let sync =
        SyncData::load_with(&root, &horizon_world::loot::PackOverlay::NONE, Some(&world)).unwrap();
    let registry = &sync.dynamic.registries[BIOME_REGISTRY];
    let names: Vec<_> = registry.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        ["minecraft:meadow", "minecraft:plains", "minecraft:river"]
    );
    // The encoder numbers a biome by its place here: none may be missing.
    assert!(world.list.iter().all(|b| names.contains(&b.name.as_str())));
    let sent = sync.registry_packets(true, &LevelShapes::VANILLA);
    let biomes = sent.iter().find(|r| r.registry == BIOME_REGISTRY).unwrap();
    assert!(biomes.entries[0].data.is_some(), "the pack's biome is sent");
    assert!(biomes.entries[1].data.is_none(), "a known one is not");
}

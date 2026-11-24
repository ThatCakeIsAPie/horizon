# Cubic ores

How Cubic worlds (Custom Cubic, Hybrid outside the vanilla band, Earth) place ores, stone patches and deepslate.
The settings are the `oreGen` object of the world's generator settings (`cubicgen.json` for Custom / Hybrid, the
Earth settings for Earth). The code is in `crates/horizon-world/src/cubic/ores/` (engine, defaults, settings).

## Two tiers

Every entry of `oreGen.ores` is one block (an ore or a stone patch) with two tiers:

* **Tier 1, the rich zone** (`rich`): where the ore is plentiful. `intensity` is veins per cube at the peak; its
  fractional part is the chance of one more vein (0.4 = 40%). `probability` scales every try. Three modes:
  * `staticHeight`: a band of absolute Y, `min`..`max`.
  * `belowGround`: a band of depth under the local ground, `min`..`max` blocks down. Custom Cubic reads the ground
    from its own terrain estimate (cached per column), Earth from the real surface.
  * `noise`: pockets where a 3D noise is above `noiseThreshold` (`noiseScale` blocks per unit, `noiseSeed` added
    to the world seed), anywhere in stone, optionally kept inside `min`..`max` Y.

  The band's `shape`: `flat`, `triangle` (1 at `peak`, 0 at the ends) or `bell` (a Gaussian around `peak` with
  `spread`; `repeat` > 0 repeats it every `repeat` blocks, as CWG's periodic ores). `peak` defaults to the middle
  of the band, `spread` to a sixth of it. The noise mode ignores the shape.
* **Tier 2, the background** (`background`): a constant `rate` (veins per cube) outside the rich zone, `range`
  `everywhere`, or only in a `staticHeight` Y range or a `belowGround` depth range. Off when `enabled` is false.

Other fields of an entry: `blockstate`, `spawnSize` (vein size), `biomes` (only these biomes; none = all),
`biomeMultipliers` (`"minecraft:desert": 0.5` scales both tiers there), `placeBlockWhen` / `generateWhen` (CWG's
conditions). Without `placeBlockWhen` an ore only replaces stone, granite, diorite, andesite, deepslate and tuff;
in deepslate and tuff it becomes the `deepslate_` variant when one exists. Ores never go into air, water, lava,
dirt, gravel or anything else.

World-wide fields: `enabled` (false = no ores at all), `meteors` (ore meteors, see the meteors doc),
`richnessMap` and `richnessStrength` (Earth: both tiers are multiplied by the real-world richness of the ore at
that latitude and longitude, raised to `richnessStrength`, kept within 0.3..5; stone patches are not affected),
`version` (1).

Each cube's veins are drawn in the 2x2x2 population window like CWG's ores. New randomness (fractional veins,
below-ground, noise and background tries, meteors) uses randoms of its own, seeded from the world seed, the entry
and the cube, never the cube's population random.

## Defaults

**Custom Cubic and Hybrid** (`OreGenSettings::custom_defaults`, what `CustomGeneratorSettings::defaults()` now
carries; its old `standardOres` / `periodicGaussianOres` lists are empty):

* Tier 1: vanilla's bands (26.x) as static-height bands, converted from the old CWG lists.
* Emerald: a tenth of vanilla's veins everywhere, x10 (vanilla's number) in the mountain biomes. The badlands'
  extra gold band stays.
* Background: 8% of the main band's peak, everywhere, for each of the eight ores (coal, iron, copper, gold,
  redstone, diamond, lapis, emerald). Stone patches have none.
* Biome multipliers (every other biome is 1; none is below 0.5, and the background is everywhere, so every ore can
  be found in every biome):
  * coal 0.5 in desert and the badlands;
  * iron 0.6 in the oceans, 1.5 in the windswept hills and forest;
  * copper 1.5 in swamps, 0.6 in snowy plains, ice spikes, snowy taiga and snowy beach;
  * gold 0.5 in snowy plains, ice spikes, snowy taiga, frozen river and snowy beach;
  * redstone 1.5 in desert and the badlands, 0.6 in jungles and swamps;
  * lapis 0.5 in plains, sunflower plains and savannas;
  * diamond 0.6 in the oceans.
* Meteors on (their defaults).

**Earth** (`OreGenSettings::earth_defaults`, new Earth worlds): the same entries with every band moved under the
ground, taking vanilla's ground as Y 64: depth = 64 - Y. So diamonds are 48..208 blocks under the ground (peak
128), coal from the ground down to 64, iron's main band 8..88, gold and redstone 96..160, lapis 0..128. Bands
that would be above the ground are cut at the ground or dropped. The stone patches (dirt, gravel, granite, diorite,
andesite, tuff) come along by depth the same way. Same background, richness map on (strength 1), meteors on.

Earth also gets deepslate: stone 64 to 72 blocks under the ground turns to deepslate with a growing chance, and all
stone deeper than 72 is deepslate. Order per cube: deepslate, stone patches and ores, then the vegetation, then
meteors. The ores have their own random, so the vegetation is the same as without ores.

## Old worlds and presets (the legacy rule)

* A CWG JSON without `oreGen` generates exactly as before (its `standardOres` and `periodicGaussianOres` lists).
* `OreGenSettings::from_legacy` turns those lists into `oreGen` entries that generate bit-identically: flat and
  repeating-bell entries draw from the cube's population random in the old order.
* An Earth world without `oreGen` gets no ores and no deepslate, as before.
* `oreGen.version` is 1. The CWG settings `version` stays 9: `oreGen` is optional and additive.

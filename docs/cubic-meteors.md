# Ore meteors (Cubic)

Ore meteors are balls of ore with a deepslate shell and a raw-ore-block core. They land rarely on the surface, half sunk into it, or lie buried. They are placed while a cube is populated, in every Cubic world type that has `oreGen` (Custom Cubic, Hybrid outside vanilla's band, Earth). The code is `crates/horizon-world/src/cubic/ores/meteors.rs`; the settings are `oreGen.meteors` (`MeteorSettings` in `ores/settings.rs`).

## What one looks like

From the inside out:

- **Core**: `raw_<ore>_block` (`raw_iron_block`, `raw_copper_block`, `raw_gold_block`), or `<ore>_block` where there is no raw block (`coal_block`, `redstone_block`, `lapis_block`, `diamond_block`, `emerald_block`). The core radius is `max(0.5, 0.28 R)`.
- **Ore layer**: `<ore>_ore` (stone variant). With chance `mixShare` (0.2) a block is the second ore instead (80 / 20).
- **Shell**: `1 + 0.06 R` blocks thick, `deepslate`. 15 % of its blocks are `deepslate_<ore>_ore` flecks of the main ore.

`R` is half the diameter. The outer radius varies by up to +-10 % with direction (smooth noise on the sphere), so no meteor is a perfect ball. Bedrock is never overwritten.

Surface and sunk meteors of diameter 9 or more also dig a crater: the columns around the ball are lowered by up to 3 blocks (keeping their top block, so grass stays grass), with a one-block rim at the edge. In water the crater stays full of water. Vegetation (grass, flowers, logs, leaves, kelp, and so on) inside the footprint is removed above the ground. This is a name list, not every plant. Trees that stand partly outside the footprint can lose part of their canopy: the clearing is thin on purpose, not a tree-aware cut.

## Placement

- The world is cut into regions of 32 x 32 block columns. A region holds at most one meteor, with chance `perSquareKm * 1024 / 1,000,000` (a 1,000 x 1,000 area is 976 regions, so about `perSquareKm` meteors there).
- Everything a meteor writes stays inside its region's columns, so meteors of different regions never touch.
- Its plan (kind, diameter, how it sits, where in the region, the shape noise) comes from a small seeded random of its own, from the world seed and the region only. It never uses the cube's population random, so adding meteors does not change any other generation.
- **Kind**: weighted among `kinds`. **Diameter**: `minDiameter` to `maxDiameter`, weighted by `sizeCurve` at that diameter, never below `sizeFloor` of the curve's peak (so the big ones keep a tail). **Placement**: surface, sunk or buried by `surfaceShare`, `sunkShare`, `buriedShare`.
- Surface: the centre sits `-0.1 R` to `+0.4 R` above the ground (resting on it, partly sunk). Sunk: `0.4 R` to `0.9 R` below the ground (top poking out a little or flush). Buried: the centre is `R + 3` (a little more, to keep the whole top three blocks under) down to `buriedDepth` below the ground.

### Who places it

A cube's population may write only into the 2 x 2 x 2 cubes from it up (blocks `c*16` to `c*16 + 31` on each axis). A meteor is placed whole by exactly one cube, its owner: the cube with even X and Z over the region, at the Y whose window is centred nearest the meteor. Every other cube returns at once (odd X or Z, no meteor in the region: one hash and no call of the ground estimate; wrong Y: only the plan). The result is the same whatever order cubes are populated in.

The ground estimate (`ground(x, z)` = Y of the top solid block) is only used to plan: where the meteor goes vertically and which cube owns it. When the owner populates, it looks at the real ground at five columns (the centre and `R/2` away on both axes) in its window and uses the median, falling back to the estimate if it finds none ("estimate off: still place whole"). Then it clamps the centre so the body fits in the window. A 25-wide meteor (up to 27.5 blocks with the noise) leaves only a few blocks of slack in a 32-block window, so a giant can sit a few blocks off its ideal height (up to about 6). Giants are rare (`sizeFloor`, default 2 % of the peak).

## Defaults

| setting | default |
|---|---|
| `enabled` | true |
| `perSquareKm` | 5 |
| `minDiameter`, `maxDiameter` | 5, 25 |
| `sizeCurve` | (5, 0.6) (7, 1.0) (10, 0.5) (16, 0.12) (25, 0.02) |
| `sizeFloor` | 0.02 |
| `surfaceShare`, `sunkShare`, `buriedShare` | 0.6, 0.25, 0.15 |
| `buriedDepth` | 50 |
| `mixShare` | 0.2 |
| `kinds` (ore + second, weight) | iron + copper 30, copper + iron 25, coal + iron 20, gold + redstone 8, redstone + lapis 7, lapis + redstone 6, diamond + lapis 2, emerald + diamond 2 |

Why 5 per km^2: one meteor per 200,000 m^2, a mean spacing of about 450 blocks. Someone noticing things 48 blocks either side (a 96-wide strip) passes one per about 2,100 blocks walked, and about 85 % of them are visible (surface or sunk). That is the odd one per session: rarer than a steady supply, a find rather than a resource.

## Settings checks

`MeteorSettings::validate` returns an error naming the setting for: a diameter outside 3 to 25 or `min > max`; `perSquareKm` not finite, below 0 or above 1000; `sizeFloor` outside 0 to 1; a non-finite `sizeCurve` point; a negative or non-finite share, or all three 0; `buriedDepth` not finite, below 0 or above 1000; `mixShare` outside 0 to 1; a negative or non-finite kind weight; an empty ore name. No kinds, or all weights 0, is valid: meteors are then inactive. `Meteors::new` validates and resolves the blocks of every kind when `enabled` (a missing block is an error naming it); when not enabled it builds an inactive `Meteors` without looking at the registry.

## Far view

Meteors are population, so the far terrain (distant or LOD tiles) does not show them. In this first version they appear only near the player. Showing them far away is a future item.

## Thin

- A tree partly outside the footprint can keep a cut canopy.
- Logs and leaves above the top of the window (32 blocks) are not cleared.
- Vegetation placed later by a neighbouring cube's population can still land on or in the crater area.
- The plant list is by name and not exhaustive.
- Giants can sit up to about 6 blocks off their ideal height because of the window clamp.
- In Custom Cubic the ground estimate can be off (overhangs). The meteor then rests on the real ground found in the window, or falls back to the estimate.
- Meteor blocks are normal blocks, so shader packs see them like any terrain.

## Tests

`ores/meteor_tests.rs`, on a synthetic flat world: whole across cube boundaries and independent of population order; all writes inside the population window (also with a ground estimate 20 blocks off); determinism; layer order and flecks; mix share; size histogram against the spline; placement shares and depths; off switches; settings validation; no air under the body, and an ASCII cross-section (`cargo test -p horizon-world --lib cubic::ores cross_section -- --nocapture`).

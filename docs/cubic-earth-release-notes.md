**Cubic + Earth for Horizon.** A world type with no height limit (Cubic), and real-world terrain on top of it (Earth).

## Which zip

| System | File |
| --- | --- |
| Linux | `horizon-cubic-earth-<version>-linux-x64.zip` |
| Windows | `horizon-cubic-earth-<version>-windows-x64.zip` |
| Mac, Apple Silicon | `horizon-cubic-earth-<version>-macos-arm64.zip` |
| Mac, Intel | `horizon-cubic-earth-<version>-macos-x64.zip` (may be missing) |

Do not unzip it. The launcher reads the zip.

## Install in brief

1. Install the Horizon launcher, **0.2.70 or newer**.
2. Install the Horizon **Nightly of 2026-11-22 (f1bb66f) or newer**. Play it once. You need your own Minecraft: Java Edition.
3. Download the zip for your system.
4. Versions → **Import a game build…** → pick the zip → accept that it is unsigned.
5. Press Play. In Create New World → World, press **Cubic Chunks** until it reads **Earth** (or **Custom Cubic**).

Full guide, with pictures: https://github.com/ThatCakeIsAPie/horizon/blob/earth/up/docs/play-cubic-earth.md

## Please note

- The program is **unsigned**. Windows or macOS may warn you. The guide shows what to do.
- **macOS is untested.** The Windows build is made by CI but not play-tested. Linux was tested start to finish.
- Back up your worlds first. Plain Horizon players cannot join Cubic worlds, and you cannot join theirs.
- Earth worlds have no trees or plants yet.
- Earth asks before it downloads anything ("Allow Earth downloads?"). Cancel downloads nothing.

## Data credits

- Terrain: Terrain Tiles on AWS Open Data. It uses data from many public sources, including Copernicus data funded by the European Union (EU-DEM) and the U.S. Geological Survey.
- Land cover: © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium (CC BY 4.0).
- Earth was inspired by the Terrarium mod by gegy. It shares no code with it.

The full required credit list is in the guide and in the game's **Data credits…** button.

## Licence and source

Horizon is LGPL-3.0-or-later. Older code stays under MIT. Source: https://github.com/ThatCakeIsAPie/horizon (branch `earth/up`; this release names its tag). Not affiliated with Mojang or Microsoft. No Minecraft files are in the zip.

# Earth: the real world as terrain

Earth is a world type of Cubic Chunks. Its land is the real one: the height of every column comes from public
elevation data for a place on the globe you choose, at a scale you choose (1 block = 1 metre by default). A world
at the summit of Mount Everest at 1 block per metre opens on a surface at about y 8,850 and the far view shows the
valleys around it.

Earth is Horizon's own design, in MIT-licensed code. It was **inspired by the Terrarium mod by gegy; Earth shares
no code with it.**

## Create an Earth world

1. Singleplayer, Create New World.
2. Press the Cubic Chunks button until it reads Earth.
3. Press Customize. The Earth page opens.
4. Pick a place (Mount Everest, Grand Canyon, Matterhorn, Mount Fuji, Yosemite Valley, Monaco coast, Uluru, Machu
   Picchu), or type a latitude and longitude. Pick the scale.
5. To download tiles, turn on **Allow downloads**. The page shows the tile server's host when you do. Press Done.
6. Create New World.

The page keeps its fields and buttons reachable by mouse and keyboard like the other Customize pages: click a box to
type into it, Escape lets go of it, and Done and Cancel are in the footer.

## The settings

| Setting | Meaning |
|---|---|
| Origin latitude, longitude | The place at world block (0, 0), in degrees (latitude -85.0511 to 85.0511, the limit of web maps; longitude -180 to 180). Places fill both. |
| Scale | Metres of ground per block, in width and height: "1 block = N metres". The page shows the heights the whole Earth would need at that scale (the deepest trench to the highest summit) and warns when they do not fit the world's height range. |
| Tile server address | A template with `{z}`, `{x}` and `{y}` for the tile. With Horizon's official game it starts as AWS Terrain Tiles (below); the game built alone starts with the box empty. `https://` only (`http://` only to this computer, for testing). |
| Allow downloads | Whether the game may download tiles it does not have, from the host shown. Off by default. |

The Presets overlay shows the same settings as JSON, with four that have no box: `seaLevelY` (the block height of
sea level, 0 by default), `verticalExaggeration` (1), `maxZoom` (the finest tile zoom, 15) and `structures`.
Everything is saved in the world's `cubicgen.json`.

## Where tiles are kept

Tiles are cached under `.horizon-cache/earth-tiles/<hash of the tile address>/{z}/{x}/{y}.png` in the game folder.
Worlds share what they download, and two tile servers never mix. A world with downloads off (or an unapproved
server) runs on whatever is in its folder: copy that folder to move or share the data.

## Who decides about downloads: the player

A world is a file, and a file from someone else could name any server. So **downloading is the player's consent, not
the world's**. The game downloads tiles only when both are true:

- the world allows downloads (`allowDownloads` in its `cubicgen.json`), and
- the host of its tile address is on the player's approved list.

The list is the file `horizon-earth-servers.txt` in the player's folder (the game folder, or `--player-dir`), one
host a line; lines starting with `#` are ignored, and deleting a line takes the approval back.

- **Approving.** Turning on Allow downloads on the Earth page shows "Tile server: host", and pressing Done adds
  that host. Cancel adds nothing.
- **A world that arrives with downloads on.** Its switch shows off on the page. When you open it, the game runs on
  cached tiles, says in chat which host the world wants and that you have not approved it, and logs it. To approve:
  Singleplayer, select the world, Edit, and press the "Allow tiles from <host>" button, which approves that host
  (it then reads "Tiles allowed: <host> (stop)", and pressing it again takes the approval back).
- **Redirects.** A server may redirect a tile request only to another approved host.
- A dedicated server reads the list from its own folder: its owner edits the file.

A request carries only `User-Agent: Horizon/<version>`: no cookies, no identifier of the player, no referrer.

## The data and its attribution

Elevation comes from **Terrain Tiles**, a global dataset of bare-earth heights tiled for easy use, built by Mapzen
from many public sources and hosted on AWS Open Data (https://registry.opendata.aws/terrain-tiles/; managed by
Mapzen, a Linux Foundation project). The game reads the *terrarium* PNG tiles: the height of each pixel is stored in
its colour. Official Horizon builds offer
`https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`.

The dataset's publisher requires this attribution
(https://github.com/tilezen/joerd/blob/master/docs/attribution.md, "Required attribution"). The Earth page's
"Data credits..." button shows it in the game; it is also in `THIRD_PARTY_NOTICES.md`:

```
* ArcticDEM terrain data DEM(s) were created from DigitalGlobe, Inc., imagery and
  funded under National Science Foundation awards 1043681, 1559691, and 1542736;
* Australia terrain data © Commonwealth of Australia (Geoscience Australia) 2017;
* Austria terrain data © offene Daten Österreichs – Digitales Geländemodell (DGM)
  Österreich;
* Canada terrain data contains information licensed under the Open Government
  Licence – Canada;
* Europe terrain data produced using Copernicus data and information funded by the
  European Union - EU-DEM layers;
* Global ETOPO1 terrain data U.S. National Oceanic and Atmospheric Administration
* Mexico terrain data source: INEGI, Continental relief, 2016;
* New Zealand terrain data Copyright 2011 Crown copyright (c) Land Information New
  Zealand and the New Zealand Government (All rights reserved);
* Norway terrain data © Kartverket;
* United Kingdom terrain data © Environment Agency copyright and/or database right
  2015. All rights reserved;
* United States 3DEP (formerly NED) and global GMTED2010 and SRTM terrain data
  courtesy of the U.S. Geological Survey.
```

The publisher's page also gives the per-source details (licenses, credit lines) behind each item.

## Known limits

- **Elevation only, one biome.** There is no land cover, no rivers or roads, no buildings. The surface is grass,
  bare stone or snow by height and latitude; every column has the same biome, and vanilla structures are off
  (`structures` stays false).
- **Land below sea level floods.** The sea is at `seaLevelY`: any ground lower than 0 m has water over it, so a
  dry depression such as the Dead Sea shore, or the Netherlands, is under water.
- **The data is coarse.** The tiles have about one point per 30 metres in most places (finer in a few). Closer
  than that at 1 block = 1 metre, terrain is smooth.
- **The projection is local.** Blocks map to the globe by a projection tangent at the origin, so far from it
  (hundreds of kilometres at a large scale) distances and directions drift from the real ones.
- Seas are not charted in detail: ETOPO1 is coarse.
- **Data cleaning.** The source stitches 1°×1° cells, and along those lines (28°N, for one) whole rows or columns
  of pixels are thousands of metres off, which showed up as a straight trench. When a tile is decoded, a pixel
  (or a run of up to two) that differs from its neighbours on both sides along a row or column by more than 32
  pixel-widths of ground, while the neighbours agree with each other, is replaced by interpolating between them.
  At a tile's edge, where one side is in another tile, the two inner pixels are extrapolated and the limit is
  three times higher. The limits scale with the zoom, so real peaks and canyons at coarse zooms stay. Cliffs,
  coasts and anything wider than two pixels are never touched, so a rare defect can slip through. Far-view
  caches are keyed by the rule's version (`CLEAN_RULES`).

## The design in brief

- **`crates/horizon-earth`** (engine, no game or network in it): the projection from blocks to latitude and
  longitude, terrarium tile decoding, a tile store that caches on disk and in memory behind a fetcher the caller
  supplies, and the height field. Missing data is never invented: queries say which tiles they need.
- **`crates/horizon-world/src/cubic/earth/`**: `EarthSettings` and the generator that turns heights into columns
  of soil, stone, sand and water, with blocking and batched (far view) forms.
- **The downloader** (`earth_net.rs`): the only code that contacts a tile server: https only, only the world's
  address, only when allowed and approved (`earth_consent.rs`).
- **The prefetcher** (`earth_prefetch.rs`): works ahead of the players and fetches the tiles the generator and
  the far view will need, with gaps between requests and back-off for tiles that fail.
- **The far view** (`cubic_lod/earth.rs`): far nodes are built from the elevation tiles alone, never waiting for a
  download; a node whose data is missing is built later.
- **The page** (`title/cubic_customize/earth.rs`) and its default address: the engine holds no built-in address
  (docs/layers.md), so the address comes from `horizon-addresses` through the hook `Hooks::earth_tile_address`,
  which the official executable registers (`crates/horizon-earth-source`).

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
Everything is saved in the world's `cubicgen.json`. A fifth, `landCover` (off; see Land cover below), is left out of
the file while it is off.

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

## Land cover: used by the generator, off by default, no downloader yet

The next data layer is land cover from **ESA WorldCover 2021 v200**: 10 m classes (tree cover, shrubland, grassland,
cropland, built-up, bare, snow and ice, water, wetland, mangroves, moss and lichen) under CC BY 4.0, in 3° by 3°
GeoTIFF files on a public bucket, read with HTTP range requests. The generator can use it; nothing in the game
fetches it yet. In `crates/horizon-earth`:

- **`tiff.rs`**: a minimal TIFF reader over byte slices. It reads exactly what those files are (classic TIFF in
  either byte order, tiled, 8 bits, one sample, DEFLATE with no predictor, a chain of overview levels) and refuses
  anything else by name. From a prefix of the file (the first 32 KiB holds the whole index) it gives the levels,
  or says which byte range to fetch next; for a level and a tile index it gives that tile's byte range, and it
  decodes a tile's bytes to pixels. Every offset and count is bounds-checked, sizes are capped, and a bad file is an
  error, never a panic or a huge allocation. DEFLATE is `miniz_oxide`, which the `png` crate and the client already
  use (no new crate in `Cargo.lock`).
- **`landcover.rs`**: the `LandClass` type (the dataset's raw codes), the file for a latitude and longitude (named by
  its south-west corner, `N27E084`; the date line wraps and each pole is in its row's file), the pixel inside it,
  the choice of overview level for a ground resolution, and `LAND_COVER_RULES`, the version of all of that, which
  names the on-disk cache folder (`worldcover-2021-v200-rules1`) the way `CLEAN_RULES` keys the elevation cleaning.
- **`landfield.rs`**: `LandCoverStore`, a bounded memory cache of decoded tiles over a disk cache, with no fetcher.
  A lookup that lacks data answers with the `Fetch` to make (file, byte range, index or tile); the caller fetches
  the bytes and hands them to `provide`, which refuses anything that does not decode and stores nothing then.
  `LandCoverField` places it on world blocks through a `Projection`, like the elevation field, with `needs` to list
  what a rectangle will want for prefetching.
- **`landsource.rs`**: the seam a downloader plugs into. `LandCoverFetcher::fetch(file, range)` gives the bytes of
  a byte range of a file (cut short at the end of the file, as an HTTP range request does), `Ok(None)` when the
  file does not exist at the source (open sea has none), or an error. `LandCoverSource` is a field plus a fetcher:
  its blocking reads fetch what the store names, hand it over and ask again; a file the fetcher says is absent is
  marked absent and counts as no data from then on, so it is not an error and nothing retries it.
- **`testing.rs`**: a writer of synthetic tiled TIFFs and `MemoryLandCover`, a fetcher over them, so the tests need
  no real data.

### What the generator does with it

Set `"landCover": true` in the world's `cubicgen.json` (there is no box for it yet; the file leaves the name out
while it is off) **and** hand the generator a source (`EarthGenerator::with_land_cover`, built by
`EarthSettings::land_cover_source`; `CustomCubicGenerator::earth_with_land_cover` for the game's cubic level). With
either missing the terrain is the elevation-only terrain, block for block (a test compares them). Where a column's
class is known it replaces the elevation rule for the top of the column (`surface.rs`); no data keeps the old rule.
The class is read at the zoom of the sampling step, so a far voxel uses a coarser overview level than a block.

| Class | Top of the column (over stone) | Biome |
|---|---|---|
| snow and ice, or a column at or above the snow line (the latitude line, never below 2,500 m) | two snow blocks | `frozen_peaks` from 3,000 m, `snowy_slopes` from 1,500 m, else `snowy_plains` |
| tree cover | grass over three dirt | `jungle` under 12° latitude, `forest` under 55°, `taiga` under 66°, else `snowy_taiga` |
| shrubland | grass over three dirt | `savanna` under 35°, `plains`, `snowy_plains` from 66° |
| grassland | grass over three dirt | `meadow` from 1,500 m, `savanna` under 20°, `plains`, `snowy_plains` from 66° |
| cropland | grass over three dirt (farmland dries out without water, and nothing else fits) | `plains` |
| built-up | one gravel block (buildings come much later) | `plains` |
| bare | stone inside the 1,500 m alpine band under the snow line; else stone where steep (a neighbour 30 m away differs by 0.6 of the distance or more); else sand, or gravel from 50° latitude | `stony_peaks` from 2,000 m, `desert` under 40°, else `windswept_gravelly_hills` |
| water | a lake or river: three blocks of water ending at the surface block, over a sand bed. Under the sea level: the sea rule (sand, water up to the sea level) | `river` above sea level, `ocean` at or under it |
| herbaceous wetland | one block of water at the surface block over mud | `swamp` |
| mangroves | the same | `mangrove_swamp` |
| moss and lichen | moss block over three dirt | `snowy_plains` (tundra) |
| no data, or no file | the elevation rule (snow line, alpine band, grass, sand under the sea) | `ocean` under the sea level, else `plains` |

Choices worth knowing:

- **Dry below sea level.** The sea fills a column under `seaLevelY` only when its class is water or unknown. Grass,
  bare ground and the rest stay dry there (Death Valley, the Dead Sea shore, polders). With land cover on no box is
  proven all water from the heights, so deep sea costs per-column work like any other terrain.
- **Snow.** Elevation snow stays everywhere except below 2,500 m: the map comes from a few summers of imagery and
  misses seasonal snow on high peaks (Everest is white whatever its pixel says), but at the poles the line falls to
  sea level and lowland the map calls bare or moss is not under permanent snow. A lake above the snow line is frozen
  white.
- **Lakes.** The elevation data gives a lake's surface, so the water ends at the surface block and the bed is carved
  below it: the shore stays level with the bank. No flowing water is started; the blocks are placed as they are.
  The surface height (`surface_y`, the spawn point) is the water surface.
- **Slope.** A bare column reads four neighbours, 30 m of ground away (at least one sampling step), at its own zoom,
  so near and far measure the same distance. Missing neighbours make the read "not ready" like any missing tile.
- **Biomes** depend on the 4 by 4 block column only, read at a fixed 500 m of ground (a coarse overview and a
  coarse elevation zoom), so they are the same in every cube and at every detail level, and a far view needs only a
  few tiles for them.
- **The far view** never waits for land cover. A node whose class is not held yet is "later", never built from the
  elevation rule and kept: far nodes are cached on disk, so one built before its class arrived would have to be
  rebuilt when it did, and the whole distance would change colour at once. What is missing is recorded
  (`EarthGenerator::take_land_requests`, with the same waker as tile requests) for the downloader, which hands it
  to `LandCoverSource::satisfy` and has the nodes asked for again. A place with no file is no data at once.
- **The cache key.** `terrain_key()` ends with `#landcover<data rules>.<surface rules>` while land cover is on, and
  is exactly what it was while off, so worlds made either way never share a far-view cache.

### Still missing

- The **downloader and the address** (they wait for the owner's decision on the bucket; an address lives in
  `horizon-addresses`, the downloader next to `earth_net.rs`, with the same consent rules), and the prefetcher
  taking `take_land_requests` and `land_needs`.
- The **switch** in the Earth page (today only the JSON), and attribution in the Data credits screen and
  `THIRD_PARTY_NOTICES.md`.
- **Trees**, buildings, roads, farmland: the map only changes the ground and the biome.
- Not checked against a real file: the half-pixel convention (the reader treats a pixel as the area it covers),
  whether overview levels keep class boundaries well (they are taken as the file gives them), and how the rules
  look on real places (the tests use synthetic data only).

### Trying it once a downloader exists

1. Implement `LandCoverFetcher` over the bucket (HTTP range requests of `WorldCoverFile::path()`, a 404 for a file
   as `Ok(None)`), open a `LandCoverStore` under `.horizon-cache/`, and build the source with
   `EarthSettings::land_cover_source(store, fetcher)`.
2. Hand it to `CustomCubicGenerator::earth_with_land_cover` where the Earth level is built (`cubic_gen.rs`); when
   the generator has land cover, `earth_biomes()` is the biome source (`Generation` already uses it).
3. Make a world, add `"landCover": true` to its `cubicgen.json`, and open it at a place with every class: the
   Netherlands (polders under the sea level), Death Valley, the Okavango delta, the Everglades, Svalbard.

WorldCover's suggested attribution, when this layer is used (the wording from the land-cover research note, which
takes it from ESA's data access page, https://esa-worldcover.org/en/data-access; check it against that page before
it ships, then show it in the Data credits screen and in `THIRD_PARTY_NOTICES.md`):

```
(c) ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium
```

Citation: Zanaga et al. (2022), ESA WorldCover 10 m 2021 v200, doi:10.5281/zenodo.7254221.

## Known limits

- **Elevation only, one biome (the default).** Without land cover (the default, and the only mode with no
  downloader) there are no rivers or roads, no buildings. The surface is grass, bare stone or snow by height and
  latitude; every column has the same biome, and vanilla structures are off (`structures` stays false).
- **Land below sea level floods** unless land cover says otherwise. The sea is at `seaLevelY`: with elevation
  alone any ground lower than 0 m has water over it, so a dry depression such as the Dead Sea shore, or the
  Netherlands, is under water. With land cover on, ground whose class is known and not water stays dry.
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
- **`crates/horizon-world/src/cubic/earth/`**: `EarthSettings` and the generator that turns heights (and, when
  on, land cover) into columns of soil, stone, sand and water, with blocking and batched (far view) forms, and the
  biome table (`biome.rs`).
- **The downloader** (`earth_net.rs`): the only code that contacts a tile server: https only, only the world's
  address, only when allowed and approved (`earth_consent.rs`).
- **The prefetcher** (`earth_prefetch.rs`): works ahead of the players and fetches the tiles the generator and
  the far view will need, with gaps between requests and back-off for tiles that fail.
- **The far view** (`cubic_lod/earth.rs`): far nodes are built from the elevation tiles alone, never waiting for a
  download; a node whose data is missing is built later.
- **The page** (`title/cubic_customize/earth.rs`) and its default address: the engine holds no built-in address
  (docs/layers.md), so the address comes from `horizon-addresses` through the hook `Hooks::earth_tile_address`,
  which the official executable registers (`crates/horizon-earth-source`).

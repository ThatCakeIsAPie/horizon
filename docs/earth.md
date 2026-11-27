# Earth: the real world as terrain

Earth is a world type of Cubic Chunks. Its land is the real one: the height of every column comes from public
elevation data for a place on the globe you choose, at a scale you choose (1 block = 1 metre by default). A world
at the summit of Mount Everest at 1 block per metre opens on a surface at about y 8,850 and the far view shows the
valleys around it.

Earth is Horizon's own design, in code under Horizon's licence (LGPL-3.0-or-later). It was **inspired by the
Terrarium mod by gegy; Earth shares no code with it.**

## Create an Earth world

1. Singleplayer, Create New World.
2. Press the Cubic Chunks button until it reads Earth.
3. Press Customize. If you have not yet allowed Earth to download from its hosts, a question comes first (see
   "Who decides about downloads"); Yes opens the Earth page, Cancel goes back.
4. Pick a place (Mount Everest, Grand Canyon, Matterhorn, Mount Fuji, Yosemite Valley, Monaco coast, Uluru, Machu
   Picchu), or type a latitude and longitude. Pick the scale.
5. Press Done. (Changed the tile server address to another host? Done asks the same question for that host.)
6. Create New World. If Earth is still not allowed, Create asks the question too; Cancel there puts the world type
   back to what it was before Earth.

The page keeps its fields and buttons reachable by mouse and keyboard like the other Customize pages: click a box to
type into it, Escape lets go of it, and Done and Cancel are in the footer.

## The settings

| Setting | Meaning |
|---|---|
| Origin latitude, longitude | The place at world block (0, 0), in degrees (latitude -85.0511 to 85.0511, the limit of web maps; longitude -180 to 180). Places fill both. |
| Scale | Metres of ground per block, in width and height: "1 block = N metres". The page shows the heights the whole Earth would need at that scale (the deepest trench to the highest summit) and warns when they do not fit the world's height range. |
| Tile server address | A template with `{z}`, `{x}` and `{y}` for the tile. With Horizon's official game it starts as AWS Terrain Tiles (below); the game built alone starts with the box empty. `https://` only (`http://` only to this computer, for testing). |

The Presets overlay shows the same settings as JSON, with four that have no box: `seaLevelY` (the block height of
sea level, 0 by default), `verticalExaggeration` (1), `maxZoom` (the finest tile zoom, 15) and `structures` (on by default, see [Structures](#structures)).
Everything is saved in the world's `cubicgen.json`. There are no switches for downloads or land cover: a new Earth
world is always saved with `allowDownloads: true` and, when the game has a land cover source, `landCover: true` and
`vegetation: true` (plants and trees, see [Plants and trees](#plants-and-trees)). A world made before this keeps
what its file says (a world with no `landCover` stays elevation-only, and one with no `vegetation` grows nothing,
because switching its terrain mid-world would leave seams).

## Where tiles are kept

Tiles are cached under `.horizon-cache/earth-tiles/<hash of the tile address>/{z}/{x}/{y}.png` in the game folder.
Worlds share what they download, and two tile servers never mix. A world with downloads off (or an unapproved
server) runs on whatever is in its folder: copy that folder to move or share the data.

Terrain near a player never waits for a tile that can never come: a tile the game will not fetch (downloads off,
the server not approved, a refused request) or that does not exist at the server reads as flat sea floor (30 m
down) for the session, from memory only, and the disk cache stays as it was. The far view still draws nothing for
it, and the spawn at the origin still says why when its tile is missing. A failure that may pass (a timeout, no
connection, a busy server) is waited out: terrain waits for the tile and retries with back-off.

## Who decides about downloads: the player

A world is a file, and a file from someone else could name any server. So **downloading is the player's consent, not
the world's**. The game downloads tiles only when both are true:

- the world allows downloads (`allowDownloads` in its `cubicgen.json`), and
- the host of its tile address is on the player's approved list.

The list is the file `horizon-earth-servers.txt` in the player's folder (the game folder, or `--player-dir`), one
host a line; lines starting with `#` are ignored, and deleting a line takes the approval back.

- **Approving.** Earth's Customize page (Create World) and Edit World on an Earth world ask one question before
  they open, while a host they need is not on the list: "Allow Earth downloads?", naming the tile server's host
  and the land cover host, with the data credit line, and **Yes** or **Cancel**. Yes adds the hosts and opens the
  page. Cancel (or Escape) goes back to where you were: Earth can't be customized, edited or created without the
  answer. Choosing the Earth world type and pressing Create asks it too; Cancel there leaves the world type you
  had before Earth. When everything is already approved there is no question. Changing the tile address to a new
  host makes Done ask for that host.
- **A world that arrives with downloads on.** When you open it, the game runs on cached tiles, says in chat which
  host the world wants and that you have not approved it, and logs it. Singleplayer, select the world, Edit: the
  question asks for the hosts, and Yes approves them. To take an approval back, delete the host's line from the
  file.
- **Redirects.** A server may redirect a tile request only to another approved host.
- A dedicated server reads the list from its own folder: its owner edits the file.

A request carries only `User-Agent: Horizon/<version>`: no cookies, no identifier of the player, no referrer (a land
cover request adds a `Range` header and nothing else).

**Land cover has its own host**, approved the same way and on the same list: `esa-worldcover.s3.eu-central-1.amazonaws.com`.
The question names both hosts and Yes approves both. Approving only the tile server (an older approval) never
approves the land cover server; the question then names the land cover host alone. A world with land cover on whose host is not
approved (or with downloads off, or in a game with no land cover address) **runs exactly as if land cover were off**
(its terrain and its far-view cache are the elevation-only ones), says so in the log, and, for an unapproved host,
once in chat; it never waits for land cover. Cached land cover is not used while it is off.

**Climate has its own host too**: `os.unil.cloud.switch.ch` (CHELSA). The question names every host not yet approved
(tiles, land cover, climate) and one Yes approves them all. A world with climate on whose host is not approved runs
as if climate were off (biomes by latitude, see [Climate](#climate)), logs it and never waits.

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

Land cover comes from **ESA WorldCover 2021 v200** (CC BY 4.0, Zanaga et al. 2022, doi:10.5281/zenodo.7254221). ESA
asks for this credit line (https://esa-worldcover.org/en/data-access, "Attribution text"), which the "Data credits..."
screen and `THIRD_PARTY_NOTICES.md` show:

```
© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium
```

Rivers and lakes come from a small atlas **shipped inside the game** (nothing is downloaded for them; see "Water
atlas" below), a filtered and modified extract of two CC BY 4.0 datasets from HydroSHEDS
(https://www.hydrosheds.org): **RiverATLAS v1.0** (Linke et al. 2019, doi:10.1038/s41597-019-0300-6), the river
reaches with long-term discharge from WaterGAP v2.2 and catchment elevation from EarthEnv-DEM90, and **HydroLAKES
v1.0** (Messager et al. 2016, doi:10.1038/ncomms13603). RiverATLAS rather than HydroRIVERS because HydroRIVERS
alone is under the HydroSHEDS v1 licence, which asks for an end-user licence agreement that an open-source game
cannot give. The river lines still derive from HydroSHEDS v1, so the "Data credits..." screen and
`THIRD_PARTY_NOTICES.md` also carry its publisher's statement word for word (`water_attribution.txt`), with both
citations and the licence.

## Land cover

The next data layer is land cover from **ESA WorldCover 2021 v200**: 10 m classes (tree cover, shrubland, grassland,
cropland, built-up, bare, snow and ice, water, wetland, mangroves, moss and lichen) under CC BY 4.0, in 3° by 3°
GeoTIFF files on a public bucket, read with HTTP range requests. The game downloads only the pieces it reads (a
file's first 32 KiB, then each compressed tile), never a whole file (they are 100 MB or more). In
`crates/horizon-earth`:

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
- **`landfield.rs`**: `LandCoverStore`, a bounded memory cache of decoded tiles over a disk cache (see "Where land
  cover is kept"), with no fetcher.
  A lookup that lacks data answers with the `Fetch` to make (file, byte range, index or tile); the caller fetches
  the bytes and hands them to `provide`, which refuses anything that does not decode and stores nothing then.
  `LandCoverField` places it on world blocks through a `Projection`, like the elevation field, with `needs` to list
  what a rectangle will want for prefetching.
- **`landsource.rs`**: the seam a downloader plugs into. `LandCoverFetcher::fetch(file, range)` gives the bytes of
  a byte range of a file (cut short at the end of the file, as an HTTP range request does), `Ok(None)` when the
  file does not exist at the source (open sea has none), or an error. `LandCoverSource` is a field plus a fetcher:
  its blocking reads fetch what the store names, hand it over and ask again; a file the fetcher says is absent is
  marked absent (kept on disk) and counts as no data from then on, so it is not an error and nothing retries it. A
  fetcher that refuses (`FetchError::Refused`) is answered as no data for the rest of the run (memory only).
- **`testing.rs`**: a writer of synthetic tiled TIFFs and `MemoryLandCover`, a fetcher over them, so the tests need
  no real data.

### Downloading and consent

- **`earth_landnet.rs`** (`crates/horizon-client`) is the `LandCoverFetcher` over the bucket
  (`https://esa-worldcover.s3.eu-central-1.amazonaws.com/<path>`, with `WorldCoverFile::path()` for the path): an
  HTTP range request per piece on the same HTTP stack as the tile downloader (`earth_net::Http`: neutral
  user agent, the same timeouts, redirects only to approved hosts, the same status classification: 404 is no file,
  403 is a refusal, 5xx and timeouts are temporary). It checks the answer's `Content-Range`, caps the bytes of
  a response, and has at most 3 requests in flight. Checked against the real bucket: a land file answers 206, an
  open-sea file 404.
- **The address** is not in the world file. The official game offers it through `Hooks::earth_land_cover_address`
  (`horizon-addresses::EARTH_LAND_COVER`, registered by `horizon-earth-source`); `HORIZON_EARTH_LAND_COVER_URL`
  overrides it (and is the only source in the game built alone). Without either, there is no land cover source:
  a new world is saved without land cover, and a world that asks for it runs without.
- **No switch.** `landCover` in `cubicgen.json` is written `true` for every new Earth world when the game has a
  source (the Create World page writes it even if Customize was never opened); a world file without the key stays
  off.
- **The prefetcher** (`earth_prefetch.rs`) fetches land cover on its thread: the pieces the far view recorded
  (`take_land_requests`), then rounds of `land_needs` for the base-zoom area around the players; stale requests
  are skipped (`LandCoverStore::is_missing`). A far node that waited for a class is asked again by the far view's
  own retry (about once a second for nodes that answered "later") and is built once the bytes are in; nothing is
  rebuilt that was not waiting. The generator's blocking reads (near cubes) fetch what they need themselves.

### Where land cover is kept

`.horizon-cache/earth-landcover/<hash of the address>/worldcover-2021-v200-rules1/<file name>/` in the game folder:
`index-<offset>.bin` (the file's first 32 KiB and any further index piece), `length-<n>` (the file's size once
known), `tile-<level>-<n>.bin` (each compressed tile as fetched, decoded again on load) and an empty `absent` for
a file the bucket does not have. A second session at the same place reads everything from there and makes no
request (a test serves a file over HTTP, drops the server, and reads it again). Delete the folder to start over;
a new rule version starts a new folder.

### What the generator does with it

With `"landCover": true` in the world's `cubicgen.json` (a new world's file always has it; an old world's file leaves
the name out while it is off) **and** a source handed to the generator (`EarthGenerator::with_land_cover`, built by
`EarthSettings::land_cover_source`; `CustomCubicGenerator::earth_with_land_cover` for the game's cubic level, which
`cubic_gen.rs` calls with the world's `EarthTiles::land`). With either missing the terrain is the elevation-only
terrain, block for block (a test compares them). Where a column's
class is known it replaces the elevation rule for the top of the column (`surface.rs`); no data keeps the old rule.
The class is read at the zoom of the sampling step, so a far voxel uses a coarser overview level than a block.
The biome column below is the one without climate; with `climate` on see [Climate](#climate), and
[Dry places](#dry-places) for what a dry climate does to shrubland, grassland and bare ground.

| Class | Top of the column (over stone) | Biome |
|---|---|---|
| snow and ice, or a column at or above the snow line (the latitude line, never below 2,500 m) | on ground 0.84 steep or more (about 40°) bare stone; snow and ice class, gentle (under 0.6): a glacier, one snow block over six packed ice, then stone; snow and ice class, 0.6 to 0.84: an icefall, packed ice over blue ice; other classes above the snow line: two snow blocks | `frozen_peaks` from 3,000 m, `snowy_slopes` from 1,500 m, else `snowy_plains` |
| tree cover | grass over three dirt | `jungle` under 12° latitude, `forest` under 55°, `taiga` under 66°, else `snowy_taiga` |
| shrubland | grass over three dirt | `savanna` under 35°, `plains`, `snowy_plains` from 66° |
| grassland | grass over three dirt | `meadow` from 1,500 m, `savanna` under 20°, `plains`, `snowy_plains` from 66° |
| cropland | grass over three dirt (farmland dries out without water, and nothing else fits) | `plains` |
| built-up | the land around it (see "Built-up land" under Choices worth knowing below): never a class of its own once decoded, so it is whichever row above the most common natural land around the town is | that row's |
| bare | stone inside the 1,500 m alpine band under the snow line; else stone where steep (a neighbour 30 m away differs by 0.6 of the distance or more); else gravel from 50° latitude; else in a known wet climate basalt rock (70 % basalt, 14 % smooth basalt, 9 % tuff, 7 % blackstone, over stone); else sand | `stony_peaks` from 2,000 m, `desert` under 40°, else `windswept_gravelly_hills` |
| water | the water rules (see Water): a lake, a river's wide bed, or the sea when at or under the sea level plus 3 m | `river` above sea level, `ocean` at or under it |
| herbaceous wetland | one block of water at the surface block over mud; in a frozen climate (climate on, mean annual temperature under 0 C) grass over three dirt (tundra: its open water is mapped as water) | `swamp`; `snowy_plains` in a frozen climate |
| mangroves | the same | `mangrove_swamp` |
| moss and lichen | a mix by column (`MOSS_MIX`): 55 % moss block, 15 % grass block, 12 % stone, 10 % gravel, 8 % coarse dirt; dirt under moss, grass and coarse dirt, stone under stone and gravel | `snowy_plains` (tundra) |
| no data, or no file | the elevation rule (snow line, alpine band, grass, sand under the sea) | `ocean` under the sea level, else `plains` |

Choices worth knowing:

- **Dry below sea level.** The sea fills a column under `seaLevelY` only when its class is water or unknown. Grass,
  bare ground and the rest stay dry there (Death Valley, the Dead Sea shore, polders). With land cover on no box is
  proven all water from the heights, so deep sea costs per-column work like any other terrain.
- **Built-up land** is natural land: as if no one ever lived there. Each decoded land-cover tile has its built-up
  pixels (code 50) given, town by town (a patch of joined built-up pixels), the most common natural class (tree,
  shrub, grass, crop, bare, moss) within 32 pixels around the town, so a town in a forest is forest with its trees,
  not the lawns at its edge; a town with none around it takes the tile's most common natural class, else grassland
  (`landcover::unbuild`, called as each tile is decoded in `landfield.rs`).
  Water, wetland, mangrove and snow never spread into a town: a town on a shore is dry ground. Every read (near
  ground, far view, biome, vegetation, water rules, far trees, the probe) gets its classes from those decoded tiles,
  so none of them ever sees a built-up class: the ground, biome and tree density of a town are those of the land
  around it, near and far alike, and there are no parcels, grey tops, roads or street trees. The class stays in
  `LandClass` for the decoder only; a `match` in `horizon-world` that must name it treats it as grassland. The raw
  tiles on disk are unchanged (the rule runs on decode), and `surface::LAND_RULES` went up so far-view nodes built
  with grey patches are not reused.
- **Snow.** Elevation snow stays everywhere except below 2,500 m: the map comes from a few summers of imagery and
  misses seasonal snow on high peaks (Everest is white whatever its pixel says), but at the poles the line falls to
  sea level and lowland the map calls bare or moss is not under permanent snow. A lake above the snow line is frozen
  white.
- **Lakes.** A lake is flat (see Water): at its recorded level when the water atlas has it, else one block under
  the lowest elevation of its columns nearby. No flowing water is started; the blocks are placed as they are. The
  surface height (`surface_y`, the spawn point) is still the elevation surface.
- **Slope.** A bare column reads four neighbours, 30 m of ground away (at least one sampling step), at its own zoom,
  so near and far measure the same distance. Missing neighbours make the read "not ready" like any missing tile.
- **Biomes** depend on the 4 by 4 block column only, read at a fixed 500 m of ground (a coarse overview and a
  coarse elevation zoom), so they are the same in every cube and at every detail level, and a far view needs only a
  few tiles for them.
- **The far view** never waits for land cover. A node whose class is not held yet is "later", never built from the
  elevation rule and kept: far nodes are cached on disk, so one built before its class arrived would have to be
  rebuilt when it did, and the whole distance would change colour at once. What is missing is recorded
  (`EarthGenerator::take_land_requests`, with the same waker as tile requests) for the prefetcher, which hands it
  to `LandCoverSource::satisfy`; the far view asks the nodes again. A place with no file is no data at once.
- **The cache key.** `terrain_key()` ends with `#landcover<data rules>.<surface rules>` while land cover is on, and
  is exactly what it was while off, so worlds made either way never share a far-view cache.

### Plants and trees

With `"vegetation": true` and land cover on, an Earth world grows 26.3's own vegetation: the vegetal-step features of
each biome (trees, grass, ferns, flowers, bushes, cacti, dead bushes, kelp and seagrass in water, swamp and mangrove
trees), run by the same code as Custom Cubic (`custom/vegetation.rs`) in each cube's population. Earth's population
also places ores and structures (see [Structures](#structures)), but no lakes or springs. Without land cover the
world has one biome and grows nothing, and the settings in force say `vegetation` is off (`cubic_gen.rs`).

The biomes are read at 500 m, too coarse to place a forest's edge, so the density follows the 10 m land class of
the exact column (`earth/vegetation.rs`, `EarthVegetationMask`). Each feature is sorted by its name into a kind:
**Tree** (`trees_*`, `bamboo_*`, `birch_tall`, `dark_forest_vegetation`, `mushroom_island_vegetation`,
`pale_garden_vegetation`, `fallen_*`), **Water** (seagrass, kelp, sea pickles, `warm_ocean_vegetation`, coral),
**Dry** (cactus, dead bush, dry grass) and **Plant** (everything else: grass, ferns, flowers, bushes, berries, sugar
cane, pumpkins, mushrooms, lily pads, leaf litter...).

| Class | What grows |
|---|---|
| tree cover | everything the biome has (the forest's density is the biome's own) |
| shrubland | plants and dry plants; trees on a fixed 20 % of the columns |
| grassland | plants and dry plants; trees on a fixed 2 % of the columns |
| cropland, moss and lichen | plants only |
| bare | dry plants only (cactus and dead bush in a desert, as vanilla) |
| snow and ice | nothing |
| water | water plants only |
| wetland, mangroves | everything (the `swamp` and `mangrove_swamp` features) |
| no data, or no file | everything the biome has |

In a world with climate, where the biome of a shrub or grass column is desert, badlands, eroded badlands, savanna or
savanna plateau, plant features (not dry ones, not trees) grow on only 30 % of the columns, again a fixed column
hash (see [Dry places](#dry-places)). Dry plants on grassland only show in biomes that have them, so wet places are
unchanged.

The share of columns that may hold a tree is a fixed hash of the column, so the vanilla feature still decides where
a tree stands and the share thins its density. Lily pads count as plants, so they float on swamps but not on lakes.

- **The cache key.** `terrain_key()` also ends with `#vegetation<rules>` (now `#vegetation2`) while land cover and
  vegetation are on, so old far-view caches are rebuilt. The far view draws its own simple trees
  (see Far view: trees and snow); the key also ends with `#fartrees2` then.

### Structures

With `"structures": true` (the default, also for a file that leaves it out) an Earth world builds Cubic's vanilla
structures: villages, temples, mineshafts, strongholds, ocean monuments, woodland mansions, shipwrecks, ruins and the
rest of 26.3's overworld sets, all on as in Custom Cubic's defaults. The client runs the same code as Custom Cubic
(`cubic/custom/structures.rs`, built in `cubic_gen.rs`) in each cube's population, so Earth populates a cube whenever
it has structures, vegetation or ores.

- **The real ground.** A structure start reads the generator's own surface (`earth/structure_ground.rs`,
  `chunk_ground`): the topmost solid block of each column and the water over it. A village or temple stands on the
  ground top, wherever it is (a plain, a pass at 1,500 m, a coast); shipwrecks and ocean ruins lie on the sea floor;
  dry land below `seaLevelY` (land cover on) is ground, not sea; a lake above sea level is water.
- **The terrain is adapted** around a structure as in vanilla (the beard: ground filled under a house, cut away over
  it); away from structures every block is the plain generator's.
- **Biomes** come from Earth's biome source, so a structure's biome rules follow the land cover and climate; without
  land cover the world has one biome and every structure that biome allows can start anywhere.
- **Needs the generator data.** Without 26.3's generator data the world has no structures and its settings in
  force say `structures` is false (`cubic_gen.rs`). The world option "generate structures" is not read for Cubic
  worlds (Custom Cubic ignores it too); `structures` is the switch.
- **Far view shows none.** The distant terrain has no structures (Custom Cubic's far villages are Custom-only), so
  they appear as the cubes around the player are made.
- **Underground structures keep vanilla's depths.** Mineshafts, strongholds, ancient cities and trial chambers are
  placed against vanilla's bottom (y -64), not against the local ground: under a high mountain they are deep below
  the surface, and nothing moves them for the sea or for a trench.
- **Ground moves only where it joins.** The adaptation moves each column's top up or down through the blocks joined
  to the ground; a lump the beard would make under a piece floating far over the ground is left out, and a buried
  piece's carving under the ground does not open it.
- A world made before structures were made keeps what its terrain was: a cube populated without them is never
  given any later, and its starts are left out of `/locate`.

### Climate

With `"climate": true` (new worlds get it whenever the game has a climate address and land cover is on; an old
world's file leaves it out and keeps its biomes) the biomes follow the real climate instead of latitude bands. The
land cover table above still decides the ground; the climate only picks the biome.

- **The data.** CHELSA V2.1 (Karger et al. 2017, Scientific Data 4, 170122; data doi:10.16904/envidat.228.v2.1),
  the 1981-2010 bioclim layers `bio01` (mean annual air temperature) and `bio12` (annual precipitation), 30 arc
  seconds (about 1 km), land and sea, 84° N to 90° S. Licence CC0 1.0. Two cloud-optimised GeoTIFFs (BigTIFF,
  16-bit, DEFLATE with predictor 2, 512 by 512 tiles, 7 overview levels) at
  `https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim/` (`horizon_addresses::EARTH_CLIMATE`, or
  `HORIZON_EARTH_CLIMATE_URL`). The game reads the first 64 KiB of each (the whole tile index) and then single tiles
  by range request (`horizon-earth`: `climate.rs`, `climatefield.rs`, `climatesource.rs`; client:
  `earth_climatenet.rs`). A world needs a few tiles in all.
- **Why CHELSA.** Free for any use (CC0), global over land and sea at 1 km, and served as COGs that can be read piece
  by piece like the land cover. WorldCover's elevation tiles hold no climate. WorldClim 2.1 is only offered as zip
  archives (nothing to range-read) and RESOLVE Ecoregions 2017 is vector data (CC BY 4.0) that would need a tiling
  step and a host of our own.
- **Where it is kept.** `.horizon-cache/earth-climate/<hash of the address>/chelsa-v2.1-1981-2010-rules1/` in the game
  folder.
- **The mapping** (`biome.rs`, `Biome::with_climate`): vanilla's own `OverworldBiomeBuilder` tables. Mean annual
  temperature gives the row (frozen under 0 °C, cold under 8, temperate under 18, warm; warm and dry is vanilla's
  hot row), an aridity ratio `P / (20 (T + 7))` gives the humidity column (0 to 4). Vanilla's pick is used when it
  fits the land class, else the class's own choice for that row: tree cover becomes forest, birch, dark or flower
  forest, taiga, old growth taiga, jungle or savanna (a very wet temperate forest is old growth spruce taiga, a
  frozen one snowy taiga); grass plains, sunflower plains, meadow (from 1,500 m), savanna or snowy plains; dry
  shrubland desert; bare ground desert or badlands where dry, else stony peaks or windswept gravelly hills; the sea
  warm, lukewarm, normal, cold or frozen ocean by temperature (deep under 200 m); rivers freeze below 0 °C. About a
  third of 2 km cells take vanilla's variant (sunflower plains, old growth birch, bamboo jungle, sparse jungle, ice
  spikes, old growth pine taiga). Snow and the snow line, wetlands and mangroves stay as in the table above. In a dry climate, open ground follows the [Dry places](#dry-places) rules instead (badlands, desert, savanna).
- **Fallback.** Where the climate has no data (no source, a refused host, a no-data pixel) land uses the table above
  exactly, and seas and rivers take a temperature from latitude and height (`27 - 0.0074 lat²` °C, 6.5 °C colder a
  kilometre up).
- **The far view** waits for the climate as it does for land cover: a missing piece is "later" and recorded
  (`EarthGenerator::take_climate_requests`) for the prefetcher.
- **The cache key.** `terrain_key()` also ends with `#climate<data rules>.<biome rules>` while land cover and climate
  are on. Dry climates change the ground, the biome and the plants: see [Dry places](#dry-places).

### Dry places

With land cover, vegetation as wanted, and `climate` on, a **dry climate** changes the ground, the biome and the plants of
open land (bare ground, shrubland, grassland). Without climate (the setting off, no source, or a column the climate has
no data for) every rule is exactly what it was without it: a test compares the kinds and the biomes. All thresholds
are named constants in one place, `earth/dry.rs`; the rules are in `surface.rs`, `biome.rs` and `vegetation.rs`.

**What is dry.** The Koppen-style aridity ratio `r = P / (20 (T + 7))` (`T` mean annual temperature in °C, `P` annual
rain in mm, the divisor at least 20), read at the column as the biome reads it (500 m of ground):

| Constant | Value | Meaning |
|---|---|---|
| `DRY_ARIDITY` | 1.3 | `r` below it, and `T` of 0 °C or more: a dry climate (steppe and desert); about 440 mm a year at 10 °C, 700 mm at 20 °C |
| `DESERT_ARIDITY` | 0.5 | `r` below it: desert, sparser still; about 170 mm a year at 10 °C, 270 mm at 20 °C |
| `HOT_C` | 18 °C | from here gentle dry bare ground is sand dunes rather than barren red ground |
| `SAVANNA_FROM_C` | 8 °C | from here (vanilla's temperate row) dry shrub and grass are desert or savanna |
| `RUGGED_M`, `ERODED_M` | 150 m, 400 m | relief from which dry ground is badlands, eroded badlands |
| `BAND_BLOCKS`, `DRY_ROCK_DEPTH` | 4, 32 | blocks one terracotta band spans; blocks of bands under a dry top |
| `PLANT_SHARE` | 300 of 1,000 | columns that keep plant features in a dry biome |

Why 1.3 and 0.5: Koppen's dry boundary with this divisor (rain spread over the year) is `r` = 1 and its desert
boundary 0.5; with summer rain the dry boundary is about 1.3 to 1.4. A mean annual climate alone cannot tell the
seasons apart, so 1.3 takes in the semi-arid steppes with summer rain without taking in wet places. CHELSA as the
game reads it (500 m of ground):

| Place | T (°C) | P (mm) | r | Dry? |
|---|---|---|---|---|
| Grand Canyon, South Rim (Grand Canyon Village, 2,100 m) | 10.0 | 429 | 1.26 | dry scrub |
| Grand Canyon, upper walls (36.07 N, 112.12 W) | 12.5 | 363 | 0.93 | dry scrub |
| Grand Canyon, Indian Garden | 15.3 | 323 | 0.72 | dry scrub |
| Grand Canyon, Phantom Ranch (inner canyon, 750 m) | 17.6 | 281 | 0.57 | dry scrub, near desert |
| Grand Canyon, North Rim | 8.5 | 527 | 1.70 | no |
| Flagstaff (pine forest) | 9.5 | 556 | 1.68 | no |
| Denver / Lubbock (semi-arid steppe) | 9.9 / 15.8 | 433 / 515 | 1.28 / 1.13 | dry scrub |
| Phoenix / Las Vegas / Death Valley | 23.2 / 20.9 / 26.1 | 218 / 137 / 79 | 0.36 / 0.25 / 0.12 | desert |
| Sahara edge, Aïr mountains | 29.0 | 131 | 0.18 | desert |
| Serengeti | 22.3 | 646 | 1.10 | dry scrub (savanna) |
| Rome / Paris | 15.8 / 11.3 | 769 / 657 | 1.69 / 1.80 | no |
| Alps, Bernese Oberland | 6.4 | 1,662 | 6.20 | no |
| Amazon | 26.2 | 2,376 | 3.57 | no |
| Pacific Northwest, Olympic Peninsula | 9.4 | 2,871 | 8.77 | no |

**Surface** (`SurfaceKind::on_classified`). Snow and the alpine band still win first (steep snow is rock, snow and ice class is a glacier; see the class table); bare gentle ground in a wet climate is basalt, sand stays for the hot dry places and for ground without climate data; tree cover (rim forest),
cropland, water, wetland, mangroves and moss are unchanged.

| Class in a dry climate | Steep (`STEEP_GRADIENT`) | Gentle |
|---|---|---|
| bare | `Banded`: terracotta bands | `T` of 18 °C or more: sand (dunes); else `Barren`: red sand 55 %, coarse dirt 45 % |
| shrubland | `Banded` | `Scrub`: a mix by a fixed hash of the block column (so block by block, never in patches): desert (`r` < 0.5) 15 % grass block, 45 % coarse dirt, 40 % red sand; dry scrub (0.5 to 1.3) 40 %, 40 %, 20 % |
| grassland | `Banded` | grass over three dirt, as in a wet climate: the map says the ground is covered |

Bare ground from 50° latitude stays gravel. Grassland and tree cover never take dry tops: the red sand and coarse
dirt of the dry places are only where the map says the cover is sparse (shrub) or bare, so a grass or forest pixel
next to a shrub one does not turn into a 30 m orange square. The mix is carried in the kind (`Scrub(DryTop)`). A far
voxel stands for many columns, so it does not show one column's pick: the top voxel of `Scrub` is coarse dirt and
of `Barren` red sand, each mix's main block (`dry::SCRUB_FAR`, `dry::BARREN_FAR`); far trees may stand on `Scrub`.

**Bands.** Under the top of a `Banded`, `Scrub` or `Barren` column the blocks are terracotta down to 32 blocks under
the surface, stone below (the section fill shortcut uses the same depth while climate is on, and today's depth
with it off). The band at height `y` is entry `(y div 4) mod 192` of Horizon's port of vanilla's clay bands
(`worldgen/material_surface.rs`, `ClayBands::generate`: terracotta, orange, yellow, brown, red, white and light gray)
made once from a fixed seed (not the world's, so every world and every far cache agree) and indexed by **absolute**
height with no noise offset, so the strata run level across canyon walls whatever the surface. A far voxel of
banded rock (`step` blocks tall, from `lo`) is the band at its middle height, `lo + step / 2`, for every column,
the surface voxel too: its colour depends on the height alone, so a row of far voxels is one level band at every
far level, and where a voxel lies within one band it is the near band. One band spans 4
blocks, not vanilla's 1: real strata are tens of metres thick, and a 1,000 m wall of 1-block bands would be
pin-stripes. The pattern repeats every 768 blocks.

**Biomes** (`Biome::with_relief`; `with_climate` is the same without relief). Open ground in a dry climate:

| Ground | Biome |
|---|---|
| relief 150 m or more | `badlands`; from 400 m `eroded_badlands` (a new biome, last in the list, ids of the others stay) |
| bare, flat | `desert`; `badlands` from 1,000 m when temperate or warmer (as before) |
| shrub or grass, flat, 8 °C or more | `desert` when `r` < 0.5; else `savanna` (`savanna_plateau` from 1,500 m) |
| shrub or grass, flat, under 8 °C | as without the dry rule |

Relief is the elevation range of the 500 m cell's middle and its four neighbours 250 m away, read at the biome's
own zoom (so the far view's `biomes_ready` and the blocking read ask for the tiles around the cell too; it is read
only for dry open ground). Wet places do not change.

**Plants.** Grassland also allows dry plants (dead bush, cactus, dry grass), which only dry biomes have. In desert,
badlands, eroded badlands, savanna and savanna plateau, shrub and grass columns keep plant features (grass, ferns,
flowers, bushes) on only 30 % of the columns, by a fixed column hash, via the biome the mask asks the generator for
(`biome_at_quart_blocking`, remembered per 4 by 4 column). It needs a climate source: without one the biomes do not
say which places are dry.

- **The cache key.** `surface::LAND_RULES` is 8 (3: steep snow, glaciers, wet bare rock; 4: moss mix; 7: built-up land is natural land, no built-up kind; 8: dry grassland stays grass, far dry voxels level), `biome::CLIMATE_BIOME_RULES` and
  `vegetation::VEGETATION_RULES` are 2 (the key ends `#landcover<data>.8 ... #vegetation2 ... #climate<data>.2`), so
  caches made before these rules are rebuilt.

### Ore richness

Where an Earth world has ores, some places are richer in an ore than others, following real geology: the Pilbara is
rich in iron, Chile's copper belt in copper, the Witwatersrand in gold, coal basins in coal, kimberlite fields in
diamond. The map is an Earth setting (`richnessMap` in `oreGen`, on by default) with a strength slider
(`richnessStrength`) on the Ores page; ore meteors ignore it (they bring ores a place lacks). See
[cubic-ores.md](cubic-ores.md) for the ores themselves.

- **What it is.** `horizon_earth::ore_richness::richness(ore, latitude, longitude)`: a multiplier per ore, 1 for an
  average piece of land, clamped to 0.3 ..= 5. It is 1 over the ocean, for ores the map does not know (anything but
  `iron`, `copper`, `coal`, `gold`, `redstone`, `lapis`, `diamond`, `emerald`) and for a place that is not a number.
  The engine raises it to the strength (`richness^strength`) and clamps again to 0.3 ..= 5, so strength 0 turns the
  map off and 1 uses it as it is.
- **The asset.** `crates/horizon-earth/data/ore_richness.bin` (600,646 bytes), built into the game with
  `include_bytes!`: no download, no extra host, no question. A 0.25° grid (1440 by 720 cells, about 28 km at the
  equator) per ore, one byte per cell holding `32 log2(richness) + 128`, rows delta coded and zlib compressed. It is
  decoded once, on first use (about 8 MB in memory), and a lookup is a Catmull-Rom bicubic mix of the 4 by 4 nearest
  cells in log space over a field that was already smoothed, so there are no grid edges. `ORE_RICHNESS_RULES` (1)
  goes into the cache key of a world whose ores follow the map.
- **The data** (all public domain except GLiM, CC BY 3.0; `THIRD_PARTY_NOTICES.md` and the "Data credits..."
  screen credit them):
  - USGS Mineral Resources Data System (MRDS, about 300,000 worldwide records, https://mrdata.usgs.gov/mrds/);
  - USGS Major mineral deposits of the world (OFR 2005-1294, about 3,200 deposits,
    https://mrdata.usgs.gov/major-deposits/);
  - USGS Mineral operations outside the United States (mines only, https://mrdata.usgs.gov/mineral-operations/);
  - USGS World Coal Quality Inventory (OFR 2010-1196, https://pubs.usgs.gov/of/2010/1196/) and Coal fields of the
    conterminous United States (OFR 2012-1205, https://pubs.usgs.gov/of/2012/1205/);
  - GLiM, the Global Lithological Map at 0.5° (Hartmann & Moosdorf 2012, doi:10.1594/PANGAEA.788537), for the rock
    type where deposit records are thin;
  - Natural Earth 1:50m land (public domain) for the coast.
- **Which ore is which commodity.** Iron: iron ore, hematite, magnetite, siderite. Copper, gold, diamond: the same.
  Coal: coal, lignite, anthracite, coal fields and coal sample sites. Emerald: emerald, beryl, aquamarine (beryllium
  at a quarter weight). Redstone has no real ore, so it follows mercury and cinnabar (rare earths at 0.3 weight).
  Lapis: lapis lazuli, with other semi-precious gemstones at 0.15 weight as a proxy for the metamorphic terrains it
  comes from.
- **Survey bias.** Some countries are surveyed far more closely than others (MRDS is mostly the United States), so
  the raw count of deposits would make them look rich in everything. Instead a place's richness in an ore is the
  ore's share of all the records near it against its share worldwide (a 1.25° kernel for the ore, 2.5° for all
  records). Where there are few records of anything the share falls back to a prior learned from the rock type
  (0.6 to 1.8), not to barren. Records count by weight (a major deposit 10, a mine 4, a producer 3, a prospect 1.5,
  an occurrence 1), and the ratio from the globally compiled major deposits alone is blended in at 35%, so
  world-class districts stand out everywhere. The result is centred so the median piece of land is 1.
- **What it looks like** (observations, not rules; the code knows no places): Pilbara iron 3.2, Mesabi iron 3.5,
  Escondida copper 2.8, Witwatersrand gold 2.7, Hunter Valley coal 3.2, Powder River Basin coal 1.8, Kimberley diamond
  3.8, Muzo emerald 3.6, Almadén redstone 2.5, Sar-e-Sang lapis 2.7; the US Great Plains iron, copper and gold about
  0.4; the Greenland ice and the open sea about 1.
- **Rebuilding it.** `tools/earth_ore_richness.py` downloads the sources into a cache outside the repository, builds
  the grid and writes the asset; the run is deterministic. From the repository root, with a venv in the cache (its docstring lists the
  packages: numpy, scipy, pyshp, xlrd, pillow):

  ```
  ../horizon-ores-data-cache/venv/bin/python tools/earth_ore_richness.py \
      --cache ../horizon-ores-data-cache --out crates/horizon-earth/data/ore_richness.bin
  ```

  A new asset changes the size the tests expect and should bump `ORE_RICHNESS_RULES`.
- **Thin.** Coal outside the United States rests on mine records and coal sample sites, not basin outlines. Lapis
  and emerald have few records, so they are mostly flat; some places come out rich in an ore for a neighbour's sake
  (beryl pegmatites a couple of degrees from a coal basin). The deposit data is not current (MRDS stopped being
  updated in 2011).

### Snow

With `"snow": true` (new worlds get it with land cover; an old world's file leaves it out and keeps its look) the
vegetation's population ends with vanilla's top layer step: each biome's own `freeze_top_layer` feature
(`custom/vegetation.rs`, `Vegetation::with_snow`, run after the vegetal features). It puts a snow layer (`layers=1`)
on the top block of every column whose biome is cold enough, on tree leaves too, turns the grass under it snowy,
and freezes still water to ice. It needs `vegetation` (it runs in the same population) and is left out of the
settings in force when there is none. Land cover has no say on it: the biome alone decides.

- **No height drop.** Vanilla lowers a biome's temperature above sea level + 17. Earth's biomes already come from
  the real temperature at the real ground height (or the snow line), so the pass is shown a sea level far above any
  block and the drop never applies. Snow falls where the biome itself is cold: snowy taiga, snowy plains, snowy
  slopes, frozen peaks, ice spikes, frozen river and frozen ocean (vanilla's own patchy rule); nothing falls in a
  warm biome, however high.
- **The order.** The pass takes each column's top from the whole 32-block region of the population, so a canopy
  is snowed by the population whose region holds its top. A tree that a population grows after that pass (the cube
  above populated first, or a neighbour's tree reaching over) keeps bare leaves.
- **The far view** draws the same rule (see Far view: trees and snow).
- **The cache key.** `terrain_key()` also ends with `#snow<rules>` while land cover, vegetation and `snow` are on.

### Water

With land cover on, the generator runs water rules over every column (`cubic/earth/water.rs` holds the rules as
plain functions, `water_gen.rs` feeds them). The rivers and the lakes' levels come from the water atlas
(`horizon_earth::WaterAtlas`, handed in with `with_water` when the build embeds one); without it the sea, the
shelf, the shores and lakes at a level of their own still run.

- **The top of water.** Water at a level of `L` metres has its top block at `block_y(L) - 1`: the sea's top is
  `seaLevelY - 1`, and a lake whose elevation is at its level sits one block under its shore.
- **The sea.** Land-cover water, or no data, whose elevation is at most 3 m over the sea level (real coastal water
  often reads 0 to a few metres) is sea. Its bed is a shelf: 1 block deep next to land, 0.75 block deeper for every
  block away, at most 6 blocks, never higher than the elevation.
- **Rivers.** Each atlas reach is a line with a water level at each point that never rises downstream, and a
  discharge. Width is `7.2 Q^0.5` metres and depth `0.27 Q^0.39` metres (hydraulic geometry, Andreadis et al. 2013),
  at least 2 m wide and one block deep. Columns within the half width are the channel: water at the level of the
  nearest point of the line, deepest in the middle, over gravel. Land-cover water within 30 m (or one half width)
  beyond the channel is the river's wider bed, half as deep. Beyond the channel the bank blends from one block over
  the water to the elevation over 2 to 16 blocks, so a river over low ground gets raised banks and a river in a
  canyon sits on its floor. A river more than 32 blocks under the ground is left out.
- **Lakes.** Land-cover water above the sea is a lake: flat at the level of the atlas lake whose box holds it and
  whose level is within 20 m of the elevation, else one block under the lowest elevation of its columns within 8
  samples. It deepens 0.75 block per block from its shore, to its mean depth (2 to 16 blocks), or 3 without a
  record.
- **Bounds.** A column's ground is never more than 32 blocks under its elevation and a river or lake top never more
  than 8 over it, so box classification stays exact with margins of that much.
- **Nothing hangs.** A dry column next to water whose top is over its ground is raised to that top (at most 8
  blocks); then water next to lower dry ground is lowered to it, its bed carved to keep a block of water, or it
  becomes dry. Steps between two water columns stay (a river stepping down).
- **Shores.** Dry ground within 3 blocks of sea or lake water, and at most 2 blocks over it, and the ground right
  beside a river channel, is sand, gravel where it is cold (a mean annual temperature under 2 C, or 50 degrees of
  latitude and more without climate), stone where it is steep. Snow and marsh keep their own top. A
  cold sea or lake bed is gravel.
- **Near and far agree.** The rules read a window of 11 samples on every side of what is asked, at the same spacing
  and zoom, and a column's answer depends only on the samples within 11 of it, so cubes, single blocks and far nodes
  agree. In the far view a river at least a quarter of a sample wide shows as one sample column at least.
- **Biomes and plants.** A 4 by 4 column whose middle block is in a river channel is `river` (`frozen_river` when
  cold), and a channel grows water plants only.
- **Sky cover.** The far view's sky cover uses the final tops once the land cover is held, so a carved channel is
  not shaded by the ground it was carved from.
- **The cache key.** `terrain_key()` also has `#water<atlas rules>.<generator rules>` while land cover is on, with
  `.noatlas` when the build has no atlas.
- **Checking a place without the game.** `cargo run --profile fast -p horizon-world --example earth_probe -- <lat>
  <lon> --cache <game>/.horizon-cache --size 4000 --cell 50` builds the generator a new world at that place gets
  (land cover, climate, the water atlas; it downloads with `curl` what the cache lacks) and prints, north up, the
  surface kinds of the near rules and of the far rules and the WorldCover class at every cell, and a few columns
  in detail (height, kind, class, biome, temperature).

### Far view: trees and snow

With land cover and vegetation on, the far view draws trees from data it already holds, without running vanilla's
features (Custom Cubic's far trees, `cubic_far_structures.rs`, run those and read blocks with blocking calls, which
would hang the far view while a tile cannot be had). Each far node also needs the land class and the Earth biome of
its columns; when one is missing the node is *not ready* and the request is recorded, like a missing tile. Nothing
waits. Style, shapes and sizes are in the table of `earth/far_trees.rs`.

- **Which tree.** Each Earth biome has one style (oak, birch, spruce, dark oak, jungle, acacia, mangrove; round,
  cone or flat; height, canopy radius, spacing) or none: deserts, badlands, peaks, slopes, oceans, rivers and ice
  spikes have no trees. Trees stand on grass, moss and marsh tops only, and the land class thins them as
  near does (tree cover, wetland, mangrove and no data all, shrub a fifth, grass a fiftieth, the rest none).
- **Level 0 and 1.** Level 0 nodes are exact cubes: trees of a jittered grid (one candidate per cell of the
  style's spacing at a fixed hash of the cell, no world seed) are drawn as real logs and leaves, only into air, and
  a tree in a neighbouring cube still draws its canopy here (the columns are read 3 blocks beyond the cube). Level 1
  is made of level 0 and gets the trees with it.
- **Level 2 and up.** One canopy per column, no per-tree loop: a column has leaves when a fixed hash of it is under
  the style's cover (class share times the share of ground the trees' canopies cover). From 16 blocks a step a cover
  of 0.4 or more fills the column, so a forest reads as solid green. Voxels whose middle the canopy covers are
  leaves; when none does (the canopy is thinner than a voxel) the ground's top voxel is leaves. No logs.
- **Air above trees.** A node is called sky only from 32 blocks over the ground when trees are on, so canopies over a
  node boundary are not cut. Sky cover (light) stays terrain only.
- **Snow.** With snow on, a column is cold when its biome snows (snowy taiga, snowy plains, snowy slopes, frozen
  peaks, ice spikes, frozen river and frozen ocean; the same rule as near, no height drop). Level 0 puts a snow
  layer on top of the ground or a tree, makes the grass under it snowy and freezes the top water block. Level 2 and
  up make the top voxel a snow block, the top water voxel ice, and the top of a canopy a snow layer on about half
  the columns (speckled like snowy spruce; a layer lets the light through as leaves do, so the ground under it stays
  lit, see Far view below).
- **Off.** With trees and snow off (an old world, or no land cover) the far output is what it was.
- **The cache key.** `terrain_key()` ends with `#fartrees2` (`far_trees::FAR_TREE_RULES`) while land cover and
  vegetation are on, and `#snow1` when snow is on too.

### Far view

The standing rule for all far-view work, Cubic and Earth:

- **Coarse first, refine later.** The distance is never empty. A far node is first drawn coarse, as fast as the
  far view was before the detail; the detail comes later on the same low-priority background builder, nearest and in
  view first, and each refined node replaces its coarse one in place (the old one stays drawn until then). Every
  final far node (all levels, generated ones too) is kept in the far disk cache (`lod/cube_cache.rs`, under the
  world's `horizon_cube_lod/`), refined ones as final and coarse ones marked coarse, so a revisit or a reopened world
  draws at once (refined where it was refined) and only refines what is left.
- **As much as possible on the GPU.** Generation, meshing, culling and drawing already run there where they can
  (`gfx/cube_gen`, `cube_mesh`, `cube_far`); new far work goes through the same material stage and inputs shader
  packs hook (real block states, normals and light), never a private pass.
- **Earth now.** With far trees on, a node is made first without its trees (ground, colour and snow) and again whole
  once every planned node is drawn (`cubic_lod/refine.rs`, a `LateContent`: up to 48 nodes at a time, nearest first,
  those behind the camera counted at three times their distance). `HORIZON_FAR_TWO_PASS=0` makes nodes whole at
  once, to compare.
- **Frame cost.** Whole far tiles are culled and drawn on the GPU (`gfx/far_cull.rs`: frustum and a two-phase
  Hi-Z test on the previous frame's depth, indirect draws through the same terrain pipelines); the CPU lists only
  tiles that fade or carry water. Far threads run below the game (nice +10, `HORIZON_FAR_PRIORITY=2` leaves them as
  spawned), there are at most `threads - 2` far workers, and results are handed over within a per-frame budget.
  Knobs, for tuning: `HORIZON_FAR_WORKERS` (default threads - 2, max 8), `HORIZON_FAR_DELIVER_MAX` (256 nodes),
  `HORIZON_FAR_DELIVER_BYTES` (4 MiB), `HORIZON_FAR_DELIVER_US` (1000 us), `HORIZON_FAR_CACHE_MB` (in-memory node
  cache, 192), `HORIZON_FAR_DISK_MB` (disk cache per world, 1024), `HORIZON_FAR_GPU_CULL=0` and `HORIZON_FAR_HIZ=0`
  (CPU culling, no occlusion; to compare), `HORIZON_FAR_WHOLE_WHEN_COVERED=1` (skip the coarse pass where something
  is drawn; measured slower, off).
- **GPU mesher.** With `gpuFarTerrain` on (and compute shaders, and no shader pack drawing terrain), Earth's far
  nodes, still sampled on the CPU, are uploaded into a node pool on the renderer's device
  (`GpuBackend::pool_only`) and meshed and drawn by `gfx/cube_mesh` and `gfx/cube_far`, as a Custom world's.
  Each column's fractional water top (`CubeNode::fluid_tops`) travels with the mesher job and rides in the quad
  (`Quad::top`, 1/256 voxel), so water draws at its real height, as the CPU mesher draws it. A world whose block
  states or biome ids do not fit the quad (17 and 8 bits) stays on the CPU mesher. The log says
  `Earth far terrain on the GPU far path`; `horizon-ctl state` shows `cube_lod.gpu_pool_only` and `gpu_uploaded`.
  `HORIZON_FAR_POOL_ROOM_CPU` (32): free pool slots a worker waits for before sampling a node.
- **Grass sides.** The far mesher (`lod::mesh::far_style`, which the GPU material table shares) draws the side faces
  of a grass block with its top sprite, grass-tinted: on a far terraced slope a voxel's side stands for slope covered
  in grass, and the dirt side with its thin green strip made forested and grassy hills read brown. A snowy grass
  block's side stays snowy; near (section) meshing is unchanged.
- **Light.** Far terrain has no light propagation: a voxel's sky light is 15 in the open, 4 under a roof, and 13
  under foliage, an opaque voxel light passes (`light_permeable`: leaves, snow layers), about what ground under near
  trees gets. Before this a canopy was a roof and the ground under far trees was drawn near black.

### Water atlas

Rivers and lakes come from one small file shipped inside the game, `crates/horizon-earth/data/water_atlas.bin`
(15.5 MB, embedded with `include_bytes!`; nothing is downloaded for it). `WaterAtlas::embedded()` reads its
header and index once; each 1-degree cell is inflated and decoded the first time a query meets it, then kept.

- **Rivers.** The RiverATLAS reaches (the HydroRIVERS network) whose long-term average discharge is at least
  **15 m3/s**: 711,038 of 8,477,883 reaches, with the id of the reach each flows into. The points are the data's
  own (they step between the cells of a 15-arc-second grid, about 460 m), upstream first, stored to about 1e-4
  degree (11.6 m). Discharge is kept to 0.07 %.
- **River levels.** A water surface level for every point, in whole decimetres, from RiverATLAS's `ele_mt_cmn`
  (the lowest ground of each reach's catchment, from EarthEnv-DEM90), worked over the whole network before any
  reach is dropped: a median along the main stem takes off one- and two-reach spikes; pits that are left are
  filled from the outlets up (so a reach is never below the one it flows into); then, from the sources down, a
  reach starts at the lowest end of its inflows and ends at its own estimate if that is lower. So levels never rise
  downstream, a reach ends exactly where the next begins, rivers that reach the sea stay at or above 0 m, and
  inland basins keep their own levels (the Dead Sea's below 0). Between a reach's ends the level is linear in
  distance.
- **Lakes.** HydroLAKES lakes and reservoirs of at least **2 km2**: 90,681 of 1,427,688, each as its bounding box
  (rounded outwards), surface level (`Elevation`), area, mean depth and whether it is a reservoir. Smaller lakes are
  left to the generator's own rule.
- **Limits.** The size budget (16 MB for the whole file) sets both thresholds; at 10 m3/s the file would be
  18.3 MB. The levels come from a 90 m DEM's catchment minima: on the Colorado at Phantom Ranch they say about
  770 m (the river is at about 735 m), on the Amazon at Manaus about 7 m (its low water is nearer 15 m). The lines
  can sit a couple of kilometres off a narrow canyon's floor. Lake boxes are boxes, not shores.
- **Rebuild.** `python3 tools/earth_water.py build --cache DIR` (numpy, pyogrio, pyarrow and shapely; about 12 GB
  of memory and 25 minutes once the 3.3 GB of downloads are in `DIR`; `--min-discharge` and `--min-lake-area`
  fix the thresholds). `python3 tools/earth_water.py check FILE` decodes a file and checks every rule; the script's
  docstring lists the inputs, the rules and the format, which `crates/horizon-earth/src/waterfile.rs` reads.

### Still missing

- Farmland as farmland (cropland is grass ground). Far trees are simple stand-ins (no mangrove roots, no vines, no thick trunks).
- Climate is the yearly mean only: no seasons, no snow that comes and goes.
- Not checked against a real file by the tests (they use synthetic data only): the half-pixel convention (the reader
  treats a pixel as the area it covers), and whether overview levels keep class boundaries well (they are taken as
  the file gives them). See the real-data check in the pull request for how the rules look at the Grand Canyon.

## Known limits

- **Elevation only, one biome (land cover off).** Without land cover there are no rivers. The surface is grass, bare stone or snow by height and
  latitude; every column has the same biome. Vanilla structures are on (see [Structures](#structures)); their biome rules
  see that one biome everywhere.
- **No towns.** Earth has no buildings, roads or towns, with or without land cover: where the map says built-up the
  world is the natural land around it (see "Built-up land" under Land cover), so cities look like the
  countryside around them. Vanilla structures are separate from this.
- **Land below sea level floods** unless land cover says otherwise. The sea is at `seaLevelY`: with elevation
  alone any ground lower than 0 m has water over it, so a dry depression such as the Dead Sea shore, or the
  Netherlands, is under water. With land cover on, ground whose class is known and not water stays dry.
- **Water is a snapshot.** Rivers and lakes are placed, not simulated: water never flows. A marsh is one block of
  water at its ground and its edges are not held in by the water rules. A river whose atlas line is off from the
  elevation's valley is still carved where the line says, within 32 blocks of depth.
- **The data is coarse.** The tiles have about one point per 30 metres in most places (finer in a few). Closer
  than that at 1 block = 1 metre, terrain is smooth.
- **The projection is local.** Blocks map to the globe by a projection tangent at the origin, so far from it
  (hundreds of kilometres at a large scale) distances and directions drift from the real ones.
- Seas are not charted in detail: ETOPO1 is coarse.
- **Dry ground follows the map's pixels.** Shrubland in a dry climate is a block-by-block mix, but where it borders
  grassland or forest the edge is the land-cover pixel's (about 10 m) square edge. Far away a dry top is one block
  per voxel (coarse dirt for scrub, red sand for barren ground), not the mix.
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
- **The downloaders** (`earth_net.rs` for tiles, `earth_landnet.rs` for land cover): the only code that contacts
  a server: https only, only the world's tile address or the game's land cover address, only when allowed and
  approved (`earth_consent.rs`); one shared HTTP stack.
- **The prefetcher** (`earth_prefetch.rs`): works ahead of the players and fetches the tiles the generator and
  the far view will need, with gaps between requests and back-off for tiles that fail.
- **The far view** (`cubic_lod/earth.rs`): far nodes are built from the elevation tiles alone, never waiting for a
  download; a node whose data is missing is built later.
- **The page** (`title/cubic_customize/earth.rs`) and its default address: the engine holds no built-in address
  (docs/layers.md), so the address comes from `horizon-addresses` through the hook `Hooks::earth_tile_address`,
  which the official executable registers (`crates/horizon-earth-source`).

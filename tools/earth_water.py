#!/usr/bin/env python3
"""Build the Earth water atlas: crates/horizon-earth/data/water_atlas.bin.

The atlas is the river network and the lakes the Earth world type draws water from. It is one preprocessed file
embedded in the game (`include_bytes!`), read by `crates/horizon-earth/src/water.rs`; the game downloads nothing
for it. This script is rerunnable: it downloads into a cache directory (skipping what is there), keeps the
expensive intermediate arrays there as `.npz`, and writes the asset.

Inputs (both CC BY 4.0; see THIRD_PARTY_NOTICES.md and docs/earth.md, "The data and its attribution"):

* RiverATLAS v1.0, part of HydroATLAS v1.0 (Linke et al. 2019, Scientific Data 6: 283,
  doi:10.1038/s41597-019-0300-6; https://www.hydrosheds.org/hydroatlas). File `RiverATLAS_Data_v10.gdb.zip`
  (2,506,480,742 bytes) from figshare, https://ndownloader.figshare.com/files/20087321 (article 9890531,
  doi:10.6084/m9.figshare.9890531). That host answers plain HTTP clients (figshare.com/ndownloader/... sits behind
  a bot check) but slowly, about 0.5 MB/s per connection; if the download fails or is refused, fetch the file any
  other way (a browser, or `curl -r` ranges in parallel) and put it in the cache directory under that name. The
  unzip checks every member's CRC. Used: the reach lines (the HydroRIVERS v1.0 network), `HYRIV_ID`,
  `NEXT_DOWN`, `ENDORHEIC`, `dis_m3_pyr` (long-term average discharge, m3/s) and `ele_mt_cmn` (the minimum
  elevation of the reach's own catchment, metres, from EarthEnv-DEM90).
* HydroLAKES v1.0 (Messager et al. 2016, Nature Communications 7: 13603, doi:10.1038/ncomms13603;
  https://www.hydrosheds.org/products/hydrolakes). File `HydroLAKES_polys_v10_shp.zip` (820 MB) from
  https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip. Used: `Hylak_id`, `Lake_type`
  (2 = reservoir), `Lake_area` (km2), `Depth_avg` (m), `Elevation` (m) and each polygon's bounding box.

Rules (all from the data; no place is named here):

* River LEVELS (the water surface, in whole decimetres) are made over the WHOLE network, before any reach is
  dropped. Each reach's own estimate is `ele_mt_cmn` (its catchment's lowest ground, which lies at its outlet).
  It is noisy from reach to reach, both ways, so each estimate first becomes the MEDIAN of itself and the two
  reaches below and above it on its main stem (the inflow with the most discharge). Some estimates are still far
  too low (DEM pits and voids), so they are then FILLED: walking from the outlets upstream, each is raised to at
  least its downstream reach's (ocean outlets to >= 0 m).
  Then, walking from the sources to the outlets in NEXT_DOWN order: a source reach starts at its estimate; any other reach
  starts at the lowest end level of the reaches flowing into it, and every one of those inflows then ends exactly at
  that junction level; a reach ends at min(start, own estimate). Levels never rise downstream and are continuous at
  every junction. Reaches whose basin drains to the ocean (`ENDORHEIC` = 0) are clamped to >= 0 m; inland basins
  keep their own levels (which may be below sea level). Per-point levels are linear in distance along the
  (simplified) line from the start level to the end level, rounded to decimetres (rounding never makes a level
  rise).
* RIVERS kept: discharge >= `--min-discharge` (m3/s). Without the option the smallest candidate in
  `DISCHARGE_STEPS` whose atlas fits the budget is used (the 2026-11 build: 15 m3/s, 711,038 of 8,477,883
  reaches; 10 m3/s would take 18.3 MB). Lines are simplified (Douglas-Peucker) with a tolerance of
  `SIMPLIFY_WIDTH_FRACTION` of the river's width, width_m = `WIDTH_COEF` * Q ** `WIDTH_EXP` (a common hydraulic
  geometry fit), clamped to `SIMPLIFY_MIN_M`..`SIMPLIFY_MAX_M`; the first and last points always stay. (The lines
  step between the pixel centres of a 15" grid, about 460 m apart, so a tolerance of at most 50 m removes almost
  nothing; the points are kept as the data has them.) A reach crossing the date line is split there.
* LAKES kept: `Lake_area` >= `--min-lake-area` (km2); without the option, the smallest candidate in
  `LAKE_AREA_STEPS` whose lakes alone fit in `LAKE_SHARE` of the budget (the 2026-11 build: 2 km2, 90,681 of
  1,427,688 lakes, 1.9 MB). Smaller lakes are left to the game's local rule from land cover and elevation.
* Budget: the whole file <= `--budget` bytes (default 16,000,000).

Output format (the reader, crates/horizon-earth/src/waterfile.rs, documents the same): a header (magic `HZWA`,
version = WATER_RULES, units per degree = 9600, reach and lake counts, cell count), an index of non-empty 1-degree
cells (lat i16, lon i16, offset u32, compressed length u32, raw length u32; sorted), then one raw DEFLATE block per
cell holding every reach and lake whose bounding box meets the cell, each kind in rising id order: ids as steps,
next_down as a step from the id, quantities as u16 log codes (2^((c - 32768) / 1024)), the first point from the
cell's corner and then steps (units), levels as a first value and decimetre drops; lakes as their box (corner from
the cell's corner, size) and level. See `encode_reach`, `encode_lake` and `assemble`.

Usage:
  python3 tools/earth_water.py build [--cache DIR] [--min-discharge Q] [--min-lake-area A] [--budget BYTES]
  python3 tools/earth_water.py check PATH         # decode a file and verify every rule
  python3 tools/earth_water.py fixture PATH       # the tiny file the Rust test decodes (stdlib only)

`build` needs numpy, pyogrio, pyarrow and shapely (a venv in the cache directory is fine) and about 12 GB of
memory and 10 GB of disk in the cache; it takes about 25 minutes once the downloads are there. `check` and
`fixture` need only the standard library.
"""

import argparse
import math
import os
import struct
import sys
import time
import urllib.request
import zlib

# The format.
MAGIC = b"HZWA"
WATER_RULES = 1  # crates/horizon-earth/src/water.rs, `WATER_RULES`
UNITS = 9_600  # coordinate units per degree (about 1.04e-4 degree, 11.6 m; HydroSHEDS's 15" grid is 40 units)
CELL_DEG = 1
MAX_CELL_RAW = 64 << 20
FLAG_RESERVOIR = 1

# The build.
DEFAULT_BUDGET = 16_000_000
DISCHARGE_STEPS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 50.0]
LAKE_AREA_STEPS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0]
LAKE_SHARE = 0.2  # the lakes' share of the budget when their threshold is chosen automatically
WIDTH_COEF = 2.7  # river width in metres = WIDTH_COEF * Q ** WIDTH_EXP
WIDTH_EXP = 0.5
SIMPLIFY_WIDTH_FRACTION = 0.5
SIMPLIFY_MIN_M = 5.0
SIMPLIFY_MAX_M = 50.0
OCEAN_FLOOR_DM = 0  # rivers draining to the ocean never go below sea level
METRES_PER_DEG = 111_320.0

RIVERATLAS_ZIP = "RiverATLAS_Data_v10.gdb.zip"
RIVERATLAS_URL = "https://ndownloader.figshare.com/files/20087321"  # figshare.com/ndownloader/... hits a bot check
HYDROLAKES_ZIP = "HydroLAKES_polys_v10_shp.zip"
HYDROLAKES_URL = "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "crates", "horizon-earth", "data", "water_atlas.bin")
DEFAULT_CACHE = os.environ.get("HORIZON_EARTH_WATER_CACHE", os.path.expanduser("~/.cache/horizon-earth-water"))


# ---------------------------------------------------------------------------------------------------------------
# Writing and reading the format (standard library only).


def varint(n):
    if n < 0:
        raise ValueError("varint of a negative number")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def zvarint(n):
    return varint((n << 1) ^ (n >> 63) if n < 0 else n << 1)


def cell(v_units):
    """The index cell of a coordinate in units (floor of degrees), like `cells_of` in water.rs."""
    return v_units // (UNITS * CELL_DEG)


def cells_of_box(s, w, n, e):
    """The cells a box in units (south, west, north, east; west <= east) meets, like `cells_of` in water.rs."""
    s = max(min(s, 90 * UNITS), -90 * UNITS)
    n = max(min(n, 90 * UNITS), -90 * UNITS)
    w = max(w, -180 * UNITS)
    e = min(e, 180 * UNITS)
    return [(la, lo) for la in range(cell(s), cell(n) + 1) for lo in range(cell(w), cell(e) + 1)]


def log_code(x):
    """A positive quantity as a u16: 0 for zero (or less, or not a number), else round(1024 * log2(x)) + 32768
    clamped to 1..=65535 (0.07 % steps, 2**-32 to 2**32)."""
    if not (x > 0) or math.isinf(x):
        return 0
    return max(1, min(65535, round(math.log2(x) * 1024) + 32768))


def log_value(code):
    """The value of a `log_code` (as f64; the reader rounds it to f32)."""
    return 0.0 if code == 0 else 2.0 ** ((code - 32768) / 1024)


def encode_reach(rid, next_down, discharge, lats, lons, levels_dm):
    """One reach: ids, discharge, points (units, upstream first) and levels (decimetres, never rising). Returned
    as (head, first point, tail): a cell's block writes the id step, head, the first point from the cell's corner,
    tail (see `assemble`)."""
    n = len(lats)
    assert n >= 2 and len(lons) == n and len(levels_dm) == n
    if next_down == rid:
        raise ValueError(f"reach {rid} flows into itself")
    head = bytearray()
    head += zvarint(next_down - rid) if next_down else zvarint(0)
    head += struct.pack("<H", log_code(discharge))
    head += varint(n)
    tail = bytearray()
    for i in range(1, n):
        tail += zvarint(lats[i] - lats[i - 1])
        tail += zvarint(lons[i] - lons[i - 1])
    tail += zvarint(levels_dm[0])
    for i in range(1, n):
        drop = levels_dm[i - 1] - levels_dm[i]
        if drop < 0:
            raise ValueError(f"reach {rid}: level rises at point {i}")
        tail += varint(drop)
    return bytes(head), (lats[0], lons[0]), bytes(tail)


def encode_lake(lid, s, w, n, e, level_dm, area_km2, depth_m, reservoir):
    """One lake: box in units (floor south/west, ceil north/east), level (dm), area, depth, flags. Returned as
    (south-west corner, tail): a cell's block writes the id step, the corner from the cell's corner, tail."""
    tail = bytearray()
    tail += varint(n - s)
    tail += varint(e - w)
    tail += zvarint(level_dm)
    tail += struct.pack("<HH", log_code(area_km2), log_code(depth_m))
    tail.append(FLAG_RESERVOIR if reservoir else 0)
    return None, (s, w), bytes(tail)


def assemble(reaches, lakes, level=9):
    """The file from encoded features: `reaches` and `lakes` are lists of (id, (s, w, n, e) in units, encoded),
    `encoded` as `encode_reach` / `encode_lake` return it. Ids must be unique per kind."""
    cells = {}
    for rid, box, data in reaches:
        for key in cells_of_box(*box):
            cells.setdefault(key, ([], []))[0].append((rid, data))
    for lid, box, data in lakes:
        for key in cells_of_box(*box):
            cells.setdefault(key, ([], []))[1].append((lid, data))
    keys = sorted(cells)
    blocks = []
    for key in keys:
        rs, ls = cells[key]
        corner = (key[0] * UNITS * CELL_DEG, key[1] * UNITS * CELL_DEG)
        raw = bytearray(varint(len(rs)) + varint(len(ls)))
        for part in (sorted(rs, key=lambda t: t[0]), sorted(ls, key=lambda t: t[0])):
            prev = 0
            for fid, (head, first, tail) in part:
                if fid <= prev and prev:
                    raise ValueError(f"id {fid} repeated")
                raw += varint(fid - prev)
                if head is not None:
                    raw += head
                raw += zvarint(first[0] - corner[0])
                raw += zvarint(first[1] - corner[1])
                raw += tail
                prev = fid
        if len(raw) > MAX_CELL_RAW:
            raise ValueError(f"cell {key} is {len(raw)} bytes raw")
        c = zlib.compressobj(level, zlib.DEFLATED, -15, 9)
        blocks.append((key, c.compress(bytes(raw)) + c.flush(), len(raw)))
    header = MAGIC + struct.pack("<IIIII", WATER_RULES, UNITS, len(reaches), len(lakes), len(keys))
    offset = len(header) + 16 * len(keys)
    index = bytearray()
    for (la, lo), comp, raw_len in blocks:
        index += struct.pack("<hhIII", la, lo, offset, len(comp), raw_len)
        offset += len(comp)
    return header + bytes(index) + b"".join(comp for _, comp, _ in blocks)


class Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def take(self, n):
        if self.p + n > len(self.d):
            raise ValueError("truncated")
        b = self.d[self.p : self.p + n]
        self.p += n
        return b

    def varint(self):
        n = shift = 0
        while True:
            b = self.take(1)[0]
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return n
            if shift >= 70:
                raise ValueError("varint too long")

    def zvarint(self):
        u = self.varint()
        return (u >> 1) ^ -(u & 1)


def decode(data):
    """Every cell of a file: {(lat, lon): (reaches, lakes)} plus the header counts. Raises on any error."""
    if data[:4] != MAGIC:
        raise ValueError("bad magic")
    version, units, n_reaches, n_lakes, n_cells = struct.unpack_from("<IIIII", data, 4)
    if version != WATER_RULES:
        raise ValueError(f"version {version}")
    cells = {}
    prev = None
    for i in range(n_cells):
        la, lo, off, ln, raw_len = struct.unpack_from("<hhIII", data, 24 + 16 * i)
        if prev is not None and (la, lo) <= prev:
            raise ValueError("index not sorted")
        prev = (la, lo)
        if off + ln > len(data) or raw_len > MAX_CELL_RAW:
            raise ValueError("block out of range")
        raw = zlib.decompress(data[off : off + ln], -15)
        if len(raw) != raw_len:
            raise ValueError("raw length")
        r = Reader(raw)
        nr, nl = r.varint(), r.varint()
        corner = (la * units, lo * units)
        reaches, lakes = [], []
        rid = 0
        for _ in range(nr):
            rid += r.varint()
            step = r.zvarint()
            nd = rid + step if step else None
            (qc,) = struct.unpack("<H", r.take(2))
            n = r.varint()
            lat, lon = corner[0] + r.zvarint(), corner[1] + r.zvarint()
            pts = [(lat / units, lon / units)]
            for _ in range(n - 1):
                lat += r.zvarint()
                lon += r.zvarint()
                pts.append((lat / units, lon / units))
            lev = r.zvarint()
            levels = [lev]
            for _ in range(n - 1):
                lev -= r.varint()
                levels.append(lev)
            reaches.append(dict(id=rid, next_down=nd, discharge=log_value(qc), points=pts, levels_dm=levels))
        lid = 0
        for _ in range(nl):
            lid += r.varint()
            s, w = corner[0] + r.zvarint(), corner[1] + r.zvarint()
            dn, de = r.varint(), r.varint()
            lev = r.zvarint()
            ac, dc = struct.unpack("<HH", r.take(4))
            flags = r.take(1)[0]
            lakes.append(
                dict(
                    id=lid,
                    south=s / units,
                    west=w / units,
                    north=(s + dn) / units,
                    east=(w + de) / units,
                    level_dm=lev,
                    area=log_value(ac),
                    depth=log_value(dc),
                    reservoir=bool(flags & FLAG_RESERVOIR),
                )
            )
        if r.p != len(raw):
            raise ValueError("trailing bytes in block")
        cells[(la, lo)] = (reaches, lakes)
    return dict(units=units, reaches=n_reaches, lakes=n_lakes, cells=cells)


def check(path):
    """Decode `path` and verify the rules; prints a summary; returns the decoded atlas."""
    with open(path, "rb") as f:
        atlas = decode(f.read())
    reaches, lakes = {}, {}
    for (la, lo), (rs, ls) in atlas["cells"].items():
        for r in rs:
            reaches[r["id"]] = r
            lats = [p[0] for p in r["points"]]
            lons = [p[1] for p in r["points"]]
            if not all(-90 <= v <= 90 for v in lats) or not all(-180 <= v <= 180 for v in lons):
                raise ValueError(f"reach {r['id']}: point out of range")
            if max(lons) - min(lons) > 180:
                raise ValueError(f"reach {r['id']} crosses the date line")
            if (la, lo) not in [(a, b) for a in range(math.floor(min(lats)), math.floor(max(lats)) + 1)
                                for b in range(math.floor(min(lons)), math.floor(max(lons)) + 1)]:
                raise ValueError(f"reach {r['id']} listed in a cell its box does not meet")
            if any(b > a for a, b in zip(r["levels_dm"], r["levels_dm"][1:])):
                raise ValueError(f"reach {r['id']}: level rises")
        for lk in ls:
            lakes[lk["id"]] = lk
    if len(reaches) != atlas["reaches"] or len(lakes) != atlas["lakes"]:
        raise ValueError(f"counts: header {atlas['reaches']}/{atlas['lakes']}, found {len(reaches)}/{len(lakes)}")
    joins = breaks = 0
    for r in reaches.values():
        d = reaches.get(r["next_down"]) if r["next_down"] else None
        if d is not None:
            joins += 1
            if d["levels_dm"][0] != r["levels_dm"][-1]:
                breaks += 1
    if breaks:
        raise ValueError(f"{breaks} of {joins} junctions are not continuous")
    print(f"{path}: ok: {len(reaches)} reaches, {len(lakes)} lakes, {len(atlas['cells'])} cells, "
          f"{joins} junctions continuous, {os.path.getsize(path)} bytes")
    return atlas


def fixture(path):
    """The tiny atlas the Rust test decodes (contents listed in water_tests.rs and format.md)."""

    def u(deg):
        return round(deg * UNITS)

    def reach(rid, nd, q, pts, levels):
        lats = [u(p[0]) for p in pts]
        lons = [u(p[1]) for p in pts]
        box = (min(lats), min(lons), max(lats), max(lons))
        return rid, box, encode_reach(rid, nd, q, lats, lons, [round(v * 10) for v in levels])

    def lake(lid, s, w, n, e, level, area, depth, res):
        box = (math.floor(s * UNITS), math.floor(w * UNITS), math.ceil(n * UNITS), math.ceil(e * UNITS))
        return lid, box, encode_lake(lid, *box, round(level * 10), area, depth, res)

    reaches = [
        reach(1, 2, 12.5, [(10.5, 20.5), (10.6, 20.7), (11.2, 21.3)], [120.5, 110.0, 110.0]),
        reach(2, None, 30.25, [(11.2, 21.3), (11.25, 21.4)], [110.0, 3.2]),
        reach(3, None, 1.0, [(0.5, -179.5), (0.6, -179.4)], [5.0, -1.5]),
    ]
    lakes = [
        lake(7, 10.1, 20.1, 10.2, 20.2, 50.0, 1.5, 4.25, True),
        lake(8, -33.5, -70.8, -33.4, -70.6, -12.3, 0.75, 2.0, False),
    ]
    data = assemble(reaches, lakes)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    check(path)


# ---------------------------------------------------------------------------------------------------------------
# Downloads.


def download(cache, name, url):
    path = os.path.join(cache, name)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(cache, exist_ok=True)
    print(f"downloading {url} -> {path}", flush=True)
    tmp = path + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "horizon-earth-water/1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            while True:
                b = resp.read(1 << 20)
                if not b:
                    break
                f.write(b)
    except Exception as err:  # noqa: BLE001 (any failure means: fetch it by hand)
        sys.exit(f"download of {url} failed ({err}); fetch it with a web browser and save it as {path}")
    if os.path.getsize(tmp) < 1_000_000:
        sys.exit(f"download of {url} returned a small file (a bot check?); fetch it with a web browser into {path}")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------------------------------------------
# Rivers.


def river_attributes(cache, gdb):
    """id, next_down, endorheic, discharge and catchment-min elevation of every reach (cached .npz)."""
    import numpy as np
    import pyogrio

    npz = os.path.join(cache, "rivers_attrs.npz")
    if os.path.exists(npz):
        return dict(np.load(npz))
    t = time.time()
    layer = pyogrio.list_layers(gdb)[0][0]
    cols = ["HYRIV_ID", "NEXT_DOWN", "ENDORHEIC", "dis_m3_pyr", "ele_mt_cmn"]
    _, table = pyogrio.read_arrow(gdb, layer=layer, columns=cols, read_geometry=False)
    out = {
        "id": table["HYRIV_ID"].to_numpy().astype(np.int64),
        "down": table["NEXT_DOWN"].to_numpy().astype(np.int64),
        "endo": table["ENDORHEIC"].to_numpy().astype(np.int8),
        "dis": table["dis_m3_pyr"].to_numpy().astype(np.float32),
        "ele": table["ele_mt_cmn"].to_numpy().astype(np.float64),
    }
    np.savez(npz, **out)
    print(f"read {len(out['id'])} reach attributes in {time.time() - t:.0f} s", flush=True)
    return out


def network_levels(a):
    """The start and end level (decimetres) of every reach, monotone over the whole network (see the docstring)."""
    import numpy as np

    ids = a["id"]
    order = np.argsort(ids)
    sid = ids[order]
    n = len(ids)
    pos = np.searchsorted(sid, a["down"])
    pos = np.clip(pos, 0, n - 1)
    has = (a["down"] != 0) & (sid[pos] == a["down"])
    down = np.where(has, order[pos], -1)
    print(f"{n} reaches, {int((~has).sum())} outlets, {int(((a['down'] != 0) & ~has).sum())} dangling next_down")

    # The own estimate; a missing one (no data) takes the downstream reach's estimate, else 0.
    est = np.round(a["ele"] * 10.0)
    bad = ~np.isfinite(est) | (a["ele"] < -1000)
    for _ in range(100):
        if not bad.any():
            break
        idx = np.nonzero(bad)[0]
        d = down[idx]
        ok = (d >= 0) & ~bad[np.maximum(d, 0)]
        est[idx[ok]] = est[d[ok]]
        bad[idx[ok]] = False
        if not ok.any():
            break
    print(f"{int(bad.sum())} reaches without an elevation estimate (set to 0)")
    est[bad] = 0
    est = est.astype(np.int64)

    # Topological generations, sources first (Kahn): a reach comes after every reach that flows into it.
    indeg = np.bincount(down[down >= 0], minlength=n)
    left = indeg.copy()
    frontier = np.nonzero(indeg == 0)[0]
    generations = []
    while len(frontier):
        generations.append(frontier)
        d = down[frontier]
        d = d[d >= 0]
        np.subtract.at(left, d, 1)
        cand = np.unique(d)
        frontier = cand[left[cand] == 0]

    # Spikes off first. The DEM-based estimate is noisy from one reach to the next, both ways (in the Grand Canyon
    # a run of 723, 798, 708, 777, 697 m; on a Missouri tributary 94..99 m around one -17 m). Each estimate becomes
    # the median of itself, the two reaches below it and the two above it on its main stem (the inflow with the
    # most discharge), as far as they exist, which removes any spike one or two reaches long.
    has_in = down >= 0
    src = np.nonzero(has_in)[0]
    by = np.lexsort((a["dis"][src], down[src]))  # grouped by downstream reach, most discharge last
    src = src[by]
    last = np.ones(len(src), dtype=bool)
    last[:-1] = down[src][1:] != down[src][:-1]
    up = np.full(n, -1, dtype=np.int64)
    up[down[src[last]]] = src[last]

    def hop(link, idx):
        return np.where(idx >= 0, link[np.maximum(idx, 0)], -1)

    d1 = down
    d2 = hop(down, d1)
    u1 = up
    u2 = hop(up, u1)
    window = np.stack([est] + [np.where(k >= 0, est[np.maximum(k, 0)], np.nan) for k in (d1, d2, u1, u2)], axis=1)
    smooth = np.round(np.nanmedian(window.astype(np.float64), axis=1)).astype(np.int64)
    print(f"{int((smooth != est).sum())} reach estimates changed by the median along the stem", flush=True)
    est = smooth

    # Then fill what pits are left. `ele_mt_cmn` is a catchment MINIMUM, and DEM voids and pits make some far too
    # low (4,282 reaches below -100 m); carved down the network as they are, one such pit would set every reach
    # below it to sea level (the Mississippi at Memphis came out at 0 m).
    # So each estimate is first raised to at least its downstream reach's (filled) estimate, outlets first; ocean
    # outlets not below sea level. After that a reach is never above the reaches that flow into it.
    ocean = a["endo"] == 0
    filled = est.copy()
    outlet = down < 0
    filled[outlet & ocean] = np.maximum(filled[outlet & ocean], OCEAN_FLOOR_DM)
    for frontier in reversed(generations):
        d = down[frontier]
        m = d >= 0
        filled[frontier[m]] = np.maximum(filled[frontier[m]], filled[d[m]])
    raised = int((filled > est).sum())
    print(f"{raised} reach estimates raised to their downstream level (pits filled)")
    est = filled

    big = np.iinfo(np.int64).max
    inflow_min = np.full(n, big, dtype=np.int64)
    start = np.zeros(n, dtype=np.int64)
    end = np.zeros(n, dtype=np.int64)
    done = 0
    gens = len(generations)
    for frontier in generations:
        s = np.where(inflow_min[frontier] == big, est[frontier], inflow_min[frontier])
        start[frontier] = s
        end[frontier] = np.minimum(s, est[frontier])
        done += len(frontier)
        d = down[frontier]
        m = d >= 0
        np.minimum.at(inflow_min, d[m], end[frontier][m])
    if done != n:
        print(f"warning: {n - done} reaches in cycles; they keep their own estimate")
        rest = left > 0
        start[rest] = est[rest]
        end[rest] = est[rest]
    # Every inflow ends at its junction: the downstream reach's start (<= its own end already).
    m = down >= 0
    end[m] = start[down[m]]
    # Basins that reach the ocean stay at or above sea level.
    start[ocean] = np.maximum(start[ocean], OCEAN_FLOOR_DM)
    end[ocean] = np.maximum(end[ocean], OCEAN_FLOOR_DM)
    assert (end <= start).all()
    assert (end[m] == start[down[m]]).all()
    print(f"levels over {gens} generations")
    return start, end


def river_geometry(cache, gdb, q_min):
    """Points (degrees, upstream first) of every reach with discharge >= q_min (cached .npz per q_min)."""
    import numpy as np
    import pyogrio
    import shapely

    npz = os.path.join(cache, f"rivers_geom_q{q_min:g}.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        return z["id"], z["offsets"], z["xy"]
    t = time.time()
    layer = pyogrio.list_layers(gdb)[0][0]
    # Streamed in batches and filtered here: GDAL's OpenFileGDB driver returns no rows for a full read with a
    # `where` on this table (it does with a feature limit), and the whole table's lines do not fit in memory.
    ids_parts, xy_parts, count_parts = [], [], []
    with pyogrio.open_arrow(
        gdb, layer=layer, columns=["HYRIV_ID", "dis_m3_pyr"], batch_size=250_000, use_pyarrow=True
    ) as (meta, reader):
        geom_col = meta.get("geometry_name") or "Shape"
        for batch in reader:
            q = batch.column("dis_m3_pyr").to_numpy(zero_copy_only=False)
            keep = np.nonzero(q >= q_min)[0]
            if len(keep) == 0:
                continue
            ids_b = batch.column("HYRIV_ID").to_numpy(zero_copy_only=False)[keep].astype(np.int64)
            names = batch.schema.names
            col = batch.column(geom_col if geom_col in names else names[-1])
            wkb = col.take(keep).to_numpy(zero_copy_only=False)
            geom = shapely.line_merge(shapely.from_wkb(wkb))
            # A multi-part line that does not merge keeps its parts in order.
            xy_b, idx = shapely.get_coordinates(geom, return_index=True)
            ids_parts.append(ids_b)
            xy_parts.append(xy_b)
            count_parts.append(np.bincount(idx, minlength=len(ids_b)))
    ids = np.concatenate(ids_parts) if ids_parts else np.zeros(0, np.int64)
    xy = np.concatenate(xy_parts) if xy_parts else np.zeros((0, 2))
    counts = np.concatenate(count_parts) if count_parts else np.zeros(0, np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if len(ids) == 0:
        sys.exit(f"no reach lines with discharge >= {q_min} read from {gdb}")
    np.savez(npz, id=ids, offsets=offsets, xy=xy)
    print(f"read {len(ids)} reach lines (Q >= {q_min}) in {time.time() - t:.0f} s", flush=True)
    return ids, offsets, xy


def split_dateline(lats, lons, levels):
    """Pieces of a line (units) that never cross the date line: [(lats, lons, levels)]. The line is cut where a
    step spans more than 180 degrees, at a point on the line (+-180) shared by both pieces' levels."""
    half = 180 * UNITS
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > half:
            a_lon, b_lon = lons[i - 1], lons[i]
            edge_a = half if a_lon > 0 else -half
            edge_b = -edge_a
            # The crossing's latitude, linear in the unwrapped longitude.
            b_unwrapped = b_lon + (2 * half if a_lon > 0 else -2 * half)
            f = (edge_a - a_lon) / (b_unwrapped - a_lon) if b_unwrapped != a_lon else 0.0
            lat_x = round(lats[i - 1] + f * (lats[i] - lats[i - 1]))
            lev_x = levels[i - 1]
            first = (lats[:i] + [lat_x], lons[:i] + [edge_a], levels[:i] + [lev_x])
            rest = split_dateline([lat_x] + lats[i:], [edge_b] + lons[i:], [lev_x] + levels[i:])
            return [first] + rest
    return [(lats, lons, levels)]


def build_rivers(cache, gdb, q_min):
    """Encoded reaches (id, box, bytes, discharge) with discharge >= q_min, and the attribute arrays."""
    import numpy as np

    a = river_attributes(cache, gdb)
    lv_npz = os.path.join(cache, "rivers_levels.npz")
    if os.path.exists(lv_npz):
        z = np.load(lv_npz)
        start, end = z["start"], z["end"]
    else:
        start, end = network_levels(a)
        np.savez(lv_npz, start=start, end=end)
    order = np.argsort(a["id"])
    sid = a["id"][order]
    gids, offsets, xy = river_geometry(cache, gdb, q_min)
    gi = order[np.searchsorted(sid, gids)]
    assert (a["id"][gi] == gids).all()

    # Direction check: the last point of a reach should touch the first point of its downstream reach.
    first = {int(r): xy[offsets[k]] for k, r in enumerate(gids[:200_000])}
    near = far = 0
    for k in range(min(len(gids), 200_000)):
        d = int(a["down"][gi[k]])
        if d in first:
            last = xy[offsets[k + 1] - 1]
            if abs(last[0] - first[d][0]) + abs(last[1] - first[d][1]) < 1e-3:
                near += 1
            else:
                far += 1
    print(f"direction check: {near} reaches end at their downstream reach's start, {far} do not")
    if far > near:
        sys.exit("lines are not digitised upstream to downstream; reverse them before going on")

    import shapely

    t = time.time()
    # Douglas-Peucker on every line at once, the tolerance in degrees of latitude (stricter east-west by cos(lat)).
    q_all = a["dis"][gi].astype(np.float64)
    width = WIDTH_COEF * q_all**WIDTH_EXP
    tol_deg = np.clip(SIMPLIFY_WIDTH_FRACTION * width, SIMPLIFY_MIN_M, SIMPLIFY_MAX_M) / METRES_PER_DEG
    counts = np.diff(offsets)
    lines = shapely.linestrings(xy, indices=np.repeat(np.arange(len(gids)), counts))
    simple = shapely.simplify(lines, tol_deg, preserve_topology=False)
    sxy, sidx = shapely.get_coordinates(simple, return_index=True)
    soff = np.concatenate([[0], np.cumsum(np.bincount(sidx, minlength=len(gids)))]).astype(np.int64)
    lat_u = np.round(sxy[:, 1] * UNITS).astype(np.int64)
    lon_u = np.round(sxy[:, 0] * UNITS).astype(np.int64)
    print(f"simplified {len(xy)} points to {len(sxy)} ({time.time() - t:.0f} s)", flush=True)

    out = []
    splits = 0
    t = time.time()
    for k in range(len(gids)):
        rid = int(gids[k])
        i = gi[k]
        q = float(a["dis"][i])
        lats_all = lat_u[soff[k] : soff[k + 1]].tolist()
        lons_all = lon_u[soff[k] : soff[k + 1]].tolist()
        if len(lats_all) < 2:  # a degenerate line: its own original ends
            lo = xy[offsets[k]], xy[offsets[k + 1] - 1]
            lats_all = [round(lo[0][1] * UNITS), round(lo[1][1] * UNITS)]
            lons_all = [round(lo[0][0] * UNITS), round(lo[1][0] * UNITS)]
        # Drop repeated points (after rounding).
        lats, lons = [lats_all[0]], [lons_all[0]]
        for la, lo in zip(lats_all[1:], lons_all[1:]):
            if la != lats[-1] or lo != lons[-1]:
                lats.append(la)
                lons.append(lo)
        if len(lats) < 2:
            lats.append(lats[0])
            lons.append(lons[0])
        # Levels: linear in distance from start to end, rounded (never rising).
        s_dm, e_dm = int(start[i]), int(end[i])
        cl = math.cos(math.radians(lats[0] / UNITS))
        dist = [0.0]
        for j in range(1, len(lats)):
            dist.append(dist[-1] + math.hypot(lats[j] - lats[j - 1], (lons[j] - lons[j - 1]) * cl))
        total = dist[-1]
        levels = [s_dm if total == 0 else round(s_dm + (e_dm - s_dm) * (dj / total)) for dj in dist]
        levels[0], levels[-1] = s_dm, e_dm
        nd = int(a["down"][i])
        pieces = split_dateline(lats, lons, levels)
        if len(pieces) > 1:
            splits += 1
        for p, (plat, plon, plev) in enumerate(pieces):
            pid = rid if p == 0 else (0x8000_0000 | rid) + (p - 1)
            pnd = nd if p == len(pieces) - 1 else ((0x8000_0000 | rid) + p)
            box = (min(plat), min(plon), max(plat), max(plon))
            out.append((pid, box, encode_reach(pid, pnd, q, plat, plon, plev), q))
        if k % 500_000 == 0 and k:
            print(f"  encoded {k}/{len(gids)} reaches ({time.time() - t:.0f} s)", flush=True)
    print(f"encoded {len(out)} reaches (Q >= {q_min}; {splits} split at the date line) in {time.time() - t:.0f} s")
    return out


# ---------------------------------------------------------------------------------------------------------------
# Lakes.


def lake_table(cache, shp_zip):
    """id, type, area, depth, elevation and bounding box of every lake (cached .npz)."""
    import numpy as np
    import pyogrio
    import shapely

    npz = os.path.join(cache, "lakes.npz")
    if os.path.exists(npz):
        return dict(np.load(npz))
    t = time.time()
    import zipfile

    with zipfile.ZipFile(shp_zip) as z:
        inner = next(n for n in z.namelist() if n.endswith(".shp"))
    path = f"/vsizip/{shp_zip}/{inner}"
    layers = pyogrio.list_layers(path)
    layer = layers[0][0]
    cols = ["Hylak_id", "Lake_type", "Lake_area", "Depth_avg", "Elevation"]
    _, table = pyogrio.read_arrow(path, layer=layer, columns=cols)
    gcol = [c for c in table.column_names if c not in cols][0]
    geom = shapely.from_wkb(table[gcol].to_numpy(zero_copy_only=False))
    b = shapely.bounds(geom)
    out = {
        "id": table["Hylak_id"].to_numpy().astype(np.int64),
        "type": table["Lake_type"].to_numpy().astype(np.int8),
        "area": table["Lake_area"].to_numpy().astype(np.float32),
        "depth": table["Depth_avg"].to_numpy().astype(np.float32),
        "ele": table["Elevation"].to_numpy().astype(np.float64),
        "west": b[:, 0],
        "south": b[:, 1],
        "east": b[:, 2],
        "north": b[:, 3],
    }
    np.savez(npz, **out)
    print(f"read {len(out['id'])} lakes in {time.time() - t:.0f} s", flush=True)
    return out


def build_lakes(t, a_min):
    import numpy as np

    keep = np.nonzero(t["area"] >= a_min)[0]
    out = []
    for i in keep:
        s = math.floor(float(t["south"][i]) * UNITS)
        w = math.floor(float(t["west"][i]) * UNITS)
        n = math.ceil(float(t["north"][i]) * UNITS)
        e = math.ceil(float(t["east"][i]) * UNITS)
        lid = int(t["id"][i])
        level = round(float(t["ele"][i]) * 10)
        data = encode_lake(lid, s, w, n, e, level, float(t["area"][i]), float(t["depth"][i]), int(t["type"][i]) == 2)
        out.append((lid, (s, w, n, e), data))
    return out


# ---------------------------------------------------------------------------------------------------------------
# The sanity check on real places (validation only: no build rule depends on these).

VALIDATION = [
    # The brief's (35.0, -90.1) is 9.7 km from the river's line; this is Memphis's riverfront.
    ("Mississippi at Memphis", "reach", (35.15, -90.07), 2.0, "discharge > 5000, level about 55..75 m"),
    # The 15" line runs about 2.1 km north of Phantom Ranch.
    ("Colorado, Grand Canyon floor", "reach", (36.10, -112.09), 2.5, "level 700..800 m"),
    ("Amazon near Manaus", "reach", (-3.1, -59.9), 10.0, "discharge > 50000"),
    ("Lake Tahoe", "lake", (39.09, -120.03), 0.0, "level 1890..1905 m"),
    ("Lake Mead", "lake", (36.15, -114.45), 0.0, "reservoir"),
]


def seg_dist_km(lat, lon, a, b):
    cl = math.cos(math.radians(lat))
    ax, ay = (a[1] - lon) * cl, a[0] - lat
    bx, by = (b[1] - lon) * cl, b[0] - lat
    dx, dy = bx - ax, by - ay
    ln = dx * dx + dy * dy
    f = 0.0 if ln == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / ln))
    return math.hypot(ax + f * dx, ay + f * dy) * 111.32, f


def validate(atlas):
    for name, kind, (lat, lon), radius_km, what in VALIDATION:
        cell_rows = [atlas["cells"].get((math.floor(lat) + a, math.floor(lon) + b), ([], [])) for a in (-1, 0, 1)
                     for b in (-1, 0, 1)]
        if kind == "reach":
            best = []
            for rs, _ in cell_rows:
                for r in rs:
                    pts, lev = r["points"], r["levels_dm"]
                    for j in range(len(pts) - 1):
                        d, f = seg_dist_km(lat, lon, pts[j], pts[j + 1])
                        if d <= radius_km:
                            best.append((d, r["id"], r["discharge"], (lev[j] + f * (lev[j + 1] - lev[j])) / 10))
            best.sort()
            top = sorted(set(best), key=lambda t: -t[2])[:3]
            print(f"  {name}: {len(best)} segments within {radius_km} km; biggest: "
                  + ", ".join(f"id {i} Q {q:.0f} level {lv:.1f} m at {d:.2f} km" for d, i, q, lv in top)
                  + f" (want {what})")
        else:
            hits = {lk["id"]: lk for _, ls in cell_rows for lk in ls
                    if lk["south"] - 0.02 <= lat <= lk["north"] + 0.02 and lk["west"] - 0.02 <= lon <= lk["east"] + 0.02}
            top = sorted(hits.values(), key=lambda lk: -lk["area"])[:3]
            print(f"  {name}: " + ", ".join(
                f"id {lk['id']} area {lk['area']:.1f} level {lk['level_dm'] / 10:.1f} m reservoir {lk['reservoir']}"
                for lk in top) + f" (want {what})")


# ---------------------------------------------------------------------------------------------------------------


def build(args):
    cache = args.cache
    os.makedirs(cache, exist_ok=True)
    gdb_zip = download(cache, RIVERATLAS_ZIP, RIVERATLAS_URL)
    gdb = os.path.join(cache, "RiverATLAS_v10.gdb")
    if not os.path.isdir(gdb):
        import zipfile

        print(f"unzipping {gdb_zip}", flush=True)
        with zipfile.ZipFile(gdb_zip) as z:
            z.extractall(cache)
        found = [d for d in os.listdir(cache) if d.endswith(".gdb")]
        if not found:
            sys.exit("no .gdb in the RiverATLAS zip")
        gdb = os.path.join(cache, found[0])
    lakes_zip = download(cache, HYDROLAKES_ZIP, HYDROLAKES_URL)

    t = lake_table(cache, lakes_zip)
    budget = args.budget
    if args.min_lake_area is not None:
        a_min = args.min_lake_area
        lakes = build_lakes(t, a_min)
    else:
        for a_min in LAKE_AREA_STEPS:
            lakes = build_lakes(t, a_min)
            size = len(assemble([], lakes))
            print(f"lakes >= {a_min} km2: {len(lakes)} lakes, {size} bytes alone", flush=True)
            if size <= LAKE_SHARE * budget:
                break
    print(f"lakes: {len(lakes)} of {len(t['id'])} kept (area >= {a_min} km2)", flush=True)

    steps = [args.min_discharge] if args.min_discharge is not None else DISCHARGE_STEPS
    data = None
    every = build_rivers(cache, gdb, min(steps))
    for q_min in steps:
        reaches = [r for r in every if r[3] >= q_min]
        data = assemble([(r[0], r[1], r[2]) for r in reaches], lakes)
        print(f"Q >= {q_min}: {len(reaches)} reaches, {len(lakes)} lakes: {len(data)} bytes", flush=True)
        if len(data) <= budget:
            break
    if data is None or len(data) > budget:
        sys.exit(f"no threshold fits the budget of {budget} bytes")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(data)
    n_all = len(river_attributes(cache, gdb)["id"])
    print(f"wrote {args.out}: {len(data)} bytes; rivers: {len(reaches)} of {n_all} reaches (discharge >= {q_min} "
          f"m3/s); lakes: {len(lakes)} of {len(t['id'])} (area >= {a_min} km2)")
    validate(check(args.out))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download, build and write the atlas")
    b.add_argument("--cache", default=DEFAULT_CACHE, help="download and scratch directory (outside the repo)")
    b.add_argument("--out", default=DEFAULT_OUT)
    b.add_argument("--min-discharge", type=float, help="keep reaches with at least this discharge, m3/s")
    b.add_argument("--min-lake-area", type=float, help="keep lakes with at least this area, km2")
    b.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="the largest file, bytes")
    c = sub.add_parser("check", help="decode a file and verify every rule")
    c.add_argument("path")
    f = sub.add_parser("fixture", help="write the tiny test atlas")
    f.add_argument("path")
    args = p.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "check":
        validate(check(args.path))
    else:
        fixture(args.path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build crates/horizon-earth/data/ore_richness.bin: a smooth global "ore richness" grid per game ore,
derived from real open data. No place names, coordinates or region boxes are used anywhere in this
script: everything comes from the downloaded data sets.

Rerun from the repo root (venv + cache live outside the repo, by default in a folder next to the checkout;
downloads are skipped when already cached):

    python3 -m venv ../horizon-ores-data-cache/venv
    ../horizon-ores-data-cache/venv/bin/pip install numpy scipy pyshp xlrd pillow
    ../horizon-ores-data-cache/venv/bin/python tools/earth_ore_richness.py \
        --cache ../horizon-ores-data-cache --out crates/horizon-earth/data/ore_richness.bin

(Those are the defaults: the cache is <repo>/../horizon-ores-data-cache, --out is resolved relative to the repo.)

Output format (see the part C format note): magic HZOR, u32 version 1, u16 cols, u16 rows, f32 lon_west,
f32 lat_north, f32 cell_deg, u8 n_ores + (u8 len + name) per ore, u32 raw_len, zlib(level 9) of
ore-major / north->south / west->east u8 samples, each row horizontally delta coded mod 256.
q: log2(richness) = (q-128)/32; ocean and no-information = 128; richness clamped 0.3..5.

SOURCES (all public domain or CC BY, all allow redistribution of a derived grid)
  1. USGS Mineral Resources Data System (MRDS), ~305k worldwide records (US heavy), commodity lists,
     development status.  https://mrdata.usgs.gov/mrds/   (mrds-csv.zip)
     Licence: US Government work, public domain; metadata "Use_Constraints: none".
     Cite: U.S. Geological Survey, 2005, Mineral Resources Data System: USGS, Reston, Virginia.
  2. USGS "Major mineral deposits of the world" (OFR 2005-1294), 3168 globally compiled deposits incl.
     gemstones (emerald, lapis lazuli, diamond).  https://mrdata.usgs.gov/major-deposits/
     (ofr20051294-csv.zip)  Licence: public domain, "Use_Constraints: none".
     Cite: Schulz, K.J., and Briskey, J.A., Major mineral deposits of the world (data from USGS OFR 2005-1294
     A-E, Zientek & Orris; Cunningham et al.; Peters et al.; Nokleberg et al.; Taylor et al.).
  3. USGS "Mineral operations outside the United States" (minfac, Minerals Yearbook facility reports; only
     mines/quarries are used).  https://mrdata.usgs.gov/mineral-operations/ (minfac-csv.zip)
     Licence: public domain.  Cite: USGS OFR 2006-1135, 2006-1375, 2010-1254, 2010-1255, 2010-1257.
  4. USGS World Coal Quality Inventory (WoCQI), ~840 distinct coal sample/mine locations in 50+ countries.
     https://pubs.usgs.gov/of/2010/1196/  (WoCQI_v1_1.xls, WoCQI_ADD_v1.xls)  Licence: public domain.
     Cite: Tewalt, S. et al., 2010, Chemical analyses in the World Coal Quality Inventory, USGS OFR 2010-1196,
     doi:10.3133/ofr20101196.
  5. USGS "Coal fields of the conterminous United States" (OFR 2012-1205; Coal_Fields polygons only, fetched
     with HTTP range reads out of the 220 MB GIS.zip).  https://pubs.usgs.gov/of/2012/1205/
     Licence: public domain, "Use_Constraints: None".  Cite: East, J.A., 2012, Coal fields of the conterminous
     United States - National Coal Resource Assessment updated version: USGS OFR 2012-1205.
  6. GLiM, Global Lithological Map database v1.0 gridded to 0.5 degrees (first-level lithology classes).
     https://doi.org/10.1594/PANGAEA.788537  Licence: CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/).
     Cite: Hartmann, J. and Moosdorf, N., 2012, Global Lithological Map Database v1.0 (gridded to 0.5 degrees
     spatial resolution), PANGAEA, doi:10.1594/PANGAEA.788537; supplement to Hartmann & Moosdorf (2012),
     The new global lithological map database GLiM, Geochem. Geophys. Geosyst. 13, Q12004,
     doi:10.1029/2012GC004370.
  7. Natural Earth 1:50m land polygons (land/ocean mask).  https://www.naturalearthdata.com/
     Licence: public domain.
  Not used: Global Energy Monitor Global Coal Mine Tracker (CC BY 4.0 but behind a request form, not
  scriptable); the USGS former-Soviet-Union coal basin shapefiles (URLs are gone).

ORE -> COMMODITY MAPPING (matched case-insensitively on commodity / ore-mineral text)
  iron      = iron, iron ore, hematite, magnetite (MRDS ore minerals count half), pig iron; Fe
  copper    = copper; Cu
  gold      = gold; Au
  coal      = coal, lignite, brown coal, anthracite; coal-field polygons (area); WoCQI coal sample sites
  diamond   = diamond
  emerald   = emerald, beryl, aquamarine (weight 1), beryllium (weight 0.25, pegmatite proxy); Be
  redstone  = mercury, cinnabar (weight 1) plus rare-earth elements (weight 0.3); Hg, REE
  lapis     = lapis lazuli (weight 1) plus other coloured/semi-precious gemstones (weight 0.15, a
              proxy for metamorphic / contact-metamorphic gem terrains, and the lithology prior is
              learned from the same pool)

METHOD
  1. Every record becomes a weighted point: major deposits x10, mineral operations (mines) x4, WoCQI
     coal sites x3, MRDS producer / past producer x3, prospect x1.5, occurrence / unknown x1, plants
     ignored.  Commodity weights: first commodity 1, second 0.5, third 0.25.  Coal-field polygon area
     adds weight 4 per fully covered 0.25 degree cell.
  2. Points are binned on the 0.25 degree grid and Gaussian smoothed (longitude sigma widened by
     1/cos(lat), wrapped): fine sigma for the ore, broad sigma for "all records" (a measure of how
     well surveyed a place is).
  3. SURVEY BIAS: richness is the ore's share of local records relative to its global share,
         ratio = (d_ore + a * prior) / (d_all * g_ore + a)
     d_all is the all-commodity density, g_ore the ore's global share, a a pseudocount, and `prior`
     the lithology prior.  Where there are few records at all (unsurveyed land) the ratio falls back
     to the lithology prior instead of "barren"; densely surveyed areas are judged only by share.
     The same ratio computed from the 3168 globally consistent major deposits alone is blended in
     (log space) so world-class districts stand out even where MRDS is thin or over-detailed.
  4. LITHOLOGY PRIOR: for each ore, over 0.5 degree cells that contain any record, the fraction that
     also contain the ore, per GLiM class, shrunk toward the global fraction and clamped to 0.6..1.8.
  5. Log-space combine, recentre so the land median is 1, soft-saturate toward 0.3 and 5, light Gaussian
     pre-smooth of the log field on land, feather the coast, ocean = 1 (q=128).
"""
import argparse
import io
import os
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO.parent / "horizon-ores-data-cache"
DEFAULT_OUT = REPO / "crates/horizon-earth/data/ore_richness.bin"

COLS, ROWS = 1440, 720
CELL = 0.25
LON_W, LAT_N = -180.0, 90.0
ORES = ["iron", "copper", "coal", "gold", "redstone", "lapis", "diamond", "emerald"]
R_MIN, R_MAX = 0.3, 5.0
UA = {"User-Agent": "Mozilla/5.0 (horizon earth_ore_richness.py; public data build script)"}

URLS = {
    "mrds-csv.zip": "https://mrdata.usgs.gov/mrds/mrds-csv.zip",
    "ofr20051294-csv.zip": "https://mrdata.usgs.gov/major-deposits/ofr20051294-csv.zip",
    "minfac-csv.zip": "https://mrdata.usgs.gov/mineral-operations/minfac-csv.zip",
    "WoCQI_v1_1.xls": "https://pubs.usgs.gov/of/2010/1196/data/WoCQI_v1_1.xls",
    "WoCQI_ADD_v1.xls": "https://pubs.usgs.gov/of/2010/1196/data/WoCQI_ADD_v1.xls",
    "glim.zip": "https://hdl.handle.net/10013/epic.39939.d001",
    "ne_50m_land.zip": "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip",
}
COAL_FIELDS_ZIP = "https://pubs.usgs.gov/of/2012/1205/Downloads/GIS.zip"
COAL_FIELDS_MEMBER = "GIS/Updated Coal Fields/Coal_Fields."

# Tunables
SRC_W = {"major": 10.0, "minfac": 4.0, "wocqi": 3.0, "coalfield_cell": 4.0}
MRDS_STATUS_W = {"producer": 3.0, "past producer": 3.0, "prospect": 1.5, "occurrence": 1.0, "unknown": 1.0}
COMMOD_W = (1.0, 0.5, 0.25)
SIGMA_ORE = 1.25      # degrees, fine kernel for the ore density
SIGMA_ALL = 2.5       # degrees, broad kernel for the all-record (survey effort) density
SIGMA_MAJOR = 1.75    # degrees, kernel for the major-deposits-only ratio
PSEUDO_A = 0.35       # per square degree, in record weight
PSEUDO_A_MAJOR = 0.12
MAJOR_BLEND = 0.35    # share of the major-only log ratio in the final log ratio
PRIOR_LO, PRIOR_HI = 0.6, 1.8
LOG_SCALE = 0.75      # scale applied to positive log2(ratio) before saturation
NEG_SCALE = 0.55      # gentler scale for poor places, so well-surveyed barren land is not all pinned at the floor
FINAL_SIGMA = 0.6     # degrees, final smoothing of the log field


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------------------------
def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log("download", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    tmp.rename(dest)
    return dest


class RangeFile(io.RawIOBase):
    """Seekable file over HTTP range requests, so one member can be read out of a huge remote zip."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = r.read()
        b[: len(d)] = d
        self.pos += len(d)
        return len(d)


def prepare_sources(cache):
    dl = cache / "dl"
    paths = {}
    for name, url in URLS.items():
        paths[name] = fetch(url, dl / name)
    # unpack
    def unzip(name, sub):
        d = dl / sub
        if not d.exists() or not any(d.iterdir()):
            d.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(paths[name]) as z:
                z.extractall(d)
        return d

    out = {
        "mrds": next(unzip("mrds-csv.zip", "mrds").glob("mrds.csv")),
        "major": unzip("ofr20051294-csv.zip", "major"),
        "minfac": next(unzip("minfac-csv.zip", "minfac").rglob("minfac.csv")),
        "glim": next(unzip("glim.zip", "glim").glob("*.asc")),
        "land": next(unzip("ne_50m_land.zip", "ne_land").glob("*.shp")),
        "wocqi": [paths["WoCQI_v1_1.xls"], paths["WoCQI_ADD_v1.xls"]],
    }
    cf = dl / "coalfields"
    if not (cf / "usgs_coal_fields.shp").exists():
        cf.mkdir(parents=True, exist_ok=True)
        log("range-reading coal fields from", COAL_FIELDS_ZIP)
        z = zipfile.ZipFile(io.BufferedReader(RangeFile(COAL_FIELDS_ZIP), buffer_size=1 << 16))
        for info in z.infolist():
            if info.filename.startswith(COAL_FIELDS_MEMBER) and info.filename[-4:] in (".shp", ".shx", ".dbf", ".prj"):
                (cf / ("usgs_coal_fields" + info.filename[-4:])).write_bytes(z.read(info))
    out["coalfields"] = cf / "usgs_coal_fields.shp"
    return out


# ----------------------------------------------------------------------------------------------
# Commodity tagging
# ----------------------------------------------------------------------------------------------
import re

TAG_PATTERNS = [
    ("iron", re.compile(r"\biron\b|\bfe\b|hematite|magnetite|siderite|limonite|goethite|taconite")),
    ("copper", re.compile(r"copper|\bcu\b|chalcopyrite|chalcocite|bornite|malachite|azurite|cuprite")),
    ("gold", re.compile(r"\bgold\b|\bau\b")),
    ("coal", re.compile(r"\bcoal\b|lignite|anthracite|bituminous|coking")),
    ("diamond", re.compile(r"diamond|kimberlite")),
    ("beryl", re.compile(r"emerald|beryl\b|aquamarine|morganite|heliodor")),
    ("beryllium", re.compile(r"beryllium|\bbe\b")),
    ("mercury", re.compile(r"mercury|cinnabar|\bhg\b")),
    ("ree", re.compile(r"\bree\b|rare earth|monazite|bastnaesite|xenotime")),
    ("lapis", re.compile(r"lapis|lazurite")),
    ("gem", re.compile(r"gemstone|garnet|\bjade|ruby|sapphire|opal\b|tourmaline|topaz|kunzite|turquoise|"
                       r"amethyst|peridot|spinel|corundum|agate|jasper|onyx|tanzanite")),
]
# ore -> {tag: weight}
ORE_TAGS = {
    "iron": {"iron": 1.0},
    "copper": {"copper": 1.0},
    "gold": {"gold": 1.0},
    "coal": {"coal": 1.0},
    "diamond": {"diamond": 1.0},
    "emerald": {"beryl": 1.0, "beryllium": 0.25},
    "redstone": {"mercury": 1.0, "ree": 0.3},
    "lapis": {"lapis": 1.0, "gem": 0.15},
}
# ore minerals in MRDS `ore` column count half (only for commodities the commodity fields missed)
MINERAL_TAGS = {"iron", "copper", "beryl", "mercury", "lapis", "coal"}


def tags_of(text):
    text = text.lower()
    return {t for t, p in TAG_PATTERNS if p.search(text)}


class Points:
    """Record store: parallel lists of lat, lon, all-weight and per-tag weight dicts."""

    def __init__(self):
        self.lat, self.lon, self.w_all, self.tags, self.src = [], [], [], [], []

    def add(self, lat, lon, w_all, tagw, src):
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return
        self.lat.append(lat)
        self.lon.append(lon)
        self.w_all.append(w_all)
        self.tags.append(tagw)
        self.src.append(src)

    def __len__(self):
        return len(self.lat)


def fnum(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_mrds(path, pts):
    import csv
    csv.field_size_limit(10**9)
    n = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            lat, lon = fnum(r["latitude"]), fnum(r["longitude"])
            if lat is None or lon is None or (lat == 0 and lon == 0):
                continue
            st = r["dev_stat"].strip().lower()
            if st == "plant":
                continue
            w = MRDS_STATUS_W.get(st, 1.0)
            tagw = {}
            for k, cw in zip(("commod1", "commod2", "commod3"), COMMOD_W):
                for t in tags_of(r[k]):
                    tagw[t] = max(tagw.get(t, 0.0), cw)
            for t in tags_of(r["ore"]) & MINERAL_TAGS:
                tagw.setdefault(t, 0.5)
            pts.add(lat, lon, w, {t: w * v for t, v in tagw.items()}, "mrds")
            n += 1
    log("MRDS records", n)


def load_major(d, pts):
    import csv
    base = next(d.rglob("deposit.csv")).parent
    comm = {}
    with open(base / "commodity.csv", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            comm.setdefault(r["gid"], []).append(r["value"])
    n = 0
    with open(base / "deposit.csv", encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            lat, lon = fnum(r["latitude"]), fnum(r["longitude"])
            if lat is None or lon is None:
                continue
            text = " ; ".join(comm.get(r["gid"], []) + [r["commodity"]])
            w = SRC_W["major"]
            pts.add(lat, lon, w, {t: w for t in tags_of(text)}, "major")
            n += 1
    log("major deposits", n)


def load_minfac(path, pts):
    import csv
    n = 0
    with open(path, encoding="latin-1", newline="") as f:
        for r in csv.DictReader(f):
            ft = r["fac_type"].lower()
            if not ("mine" in ft or "quarry" in ft or "field" in ft and "coal" in r["commodity"].lower()):
                continue
            lat, lon = fnum(r["latitude"]), fnum(r["longitude"])
            if lat is None or lon is None:
                continue
            com = r["commodity"].lower()
            if "steel" in com and "iron ore" not in com:
                continue
            w = SRC_W["minfac"]
            pts.add(lat, lon, w, {t: w for t in tags_of(com)}, "minfac")
            n += 1
    log("mineral operations (mines)", n)


def load_wocqi(paths, pts):
    import xlrd
    seen = set()
    for p in paths:
        wb = xlrd.open_workbook(str(p))
        for s in wb.sheets():
            if s.nrows < 3 or "COUNTRY" not in str(s.cell_value(1, 0)).upper():
                continue
            hdr = [str(h).upper() for h in s.row_values(1)]
            ilon = next(i for i, h in enumerate(hdr) if h.startswith("LONGITUDE"))
            ilat = next(i for i, h in enumerate(hdr) if h.startswith("LATITUDE"))
            for r in range(2, s.nrows):
                v = s.row_values(r)
                lat, lon = fnum(v[ilat]), fnum(v[ilon])
                if lat is None or lon is None:
                    continue
                key = (round(lat, 2), round(lon, 2))
                if key in seen:
                    continue
                seen.add(key)
                w = SRC_W["wocqi"]
                pts.add(lat, lon, w, {"coal": w}, "wocqi")
    log("WoCQI distinct coal sites", len(seen))


# ----------------------------------------------------------------------------------------------
# Rasters
# ----------------------------------------------------------------------------------------------
def lat_centres():
    return LAT_N - (np.arange(ROWS) + 0.5) * CELL


def bin_points(lat, lon, w):
    r = np.clip(((LAT_N - np.asarray(lat)) / CELL).astype(int), 0, ROWS - 1)
    c = np.clip(((np.asarray(lon) - LON_W) / CELL).astype(int), 0, COLS - 1)
    g = np.bincount(r * COLS + c, weights=np.asarray(w, dtype=float), minlength=ROWS * COLS)
    return g.reshape(ROWS, COLS)


def smooth(grid, sigma_deg):
    """Gaussian smooth, longitude kernel widened by 1/cos(lat), wrapped; returns mass per cell."""
    out = np.empty_like(grid, dtype=float)
    lat = lat_centres()
    for i in range(ROWS):
        s = sigma_deg / (CELL * max(np.cos(np.radians(lat[i])), 0.12))
        out[i] = ndi.gaussian_filter1d(grid[i].astype(float), min(s, 90.0), mode="wrap", truncate=3.5)
    return ndi.gaussian_filter1d(out, sigma_deg / CELL, axis=0, mode="constant", truncate=3.5)


def density(grid, sigma_deg):
    """Smoothed weight per square degree."""
    area = (CELL * CELL) * np.maximum(np.cos(np.radians(lat_centres())), 0.05)[:, None]
    return smooth(grid, sigma_deg) / area


def masked_smooth(field, mask, sigma_cells):
    """Gaussian smooth `field` using only `mask` cells (normalised convolution)."""
    m = mask.astype(float)
    num = ndi.gaussian_filter(field * m, sigma_cells, mode=("nearest", "wrap"))
    den = ndi.gaussian_filter(m, sigma_cells, mode=("nearest", "wrap"))
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def rasterise_polys(shp_path, ss=4, only_part_fn=None):
    """Fraction of each grid cell covered by the polygons of a shapefile (supersampled)."""
    import shapefile
    from PIL import Image, ImageDraw
    W, H = COLS * ss, ROWS * ss
    img = Image.new("1", (W, H), 0)
    dr = ImageDraw.Draw(img)
    rd = shapefile.Reader(str(shp_path))
    for shp in rd.shapes():
        pts = shp.points
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ring = pts[a:b]
            if len(ring) < 3:
                continue
            xy = [((x - LON_W) / CELL * ss, (LAT_N - y) / CELL * ss) for x, y in ring]
            area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(xy, xy[1:] + xy[:1]))
            # shapefile: outer rings clockwise (image coords flip sign), holes counter-clockwise
            hole = area < 0
            dr.polygon(xy, fill=0 if hole else 1)
    a = np.asarray(img, dtype=np.float32)
    return a.reshape(ROWS, ss, COLS, ss).mean(axis=(1, 3))


def glim_neutral_classes(path):
    """Values of GLiM classes that carry no lithological information: water, ice, no data."""
    names = Path(path).parent / "Classnames.txt"
    out = set()
    for line in names.read_text(errors="replace").splitlines()[1:]:
        f = line.strip().split(";")
        if len(f) >= 4 and f[3].strip('"') in ("wb", "ig", "nd"):
            out.add(int(f[1]))
    return out


def load_glim(path):
    a = np.loadtxt(path, skiprows=6)
    a = np.where(a < 0, 0, a).astype(int)  # 0 = no data
    # fill no-data with nearest classified cell
    idx = ndi.distance_transform_edt(a == 0, return_distances=False, return_indices=True)
    filled = a[idx[0], idx[1]]
    return a, filled


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def ore_weights(pts, ore, sources=None):
    tw = ORE_TAGS[ore]
    w = np.zeros(len(pts))
    for i, tg in enumerate(pts.tags):
        if sources is not None and pts.src[i] not in sources:
            continue
        s = 0.0
        for t, v in tw.items():
            if t in tg:
                s += tg[t] * v
        w[i] = s
    return w


def lithology_prior(pts, ore, cls05, neutral):
    """Per GLiM class: shrunk fraction of record-bearing 0.5 deg cells that carry the ore."""
    lat = np.asarray(pts.lat)
    lon = np.asarray(pts.lon)
    r = np.clip(((90.0 - lat) / 0.5).astype(int), 0, 359)
    c = np.clip(((lon + 180.0) / 0.5).astype(int), 0, 719)
    any_cell = np.zeros((360, 720), bool)
    any_cell[r, c] = True
    w = ore_weights(pts, ore)
    ore_cell = np.zeros((360, 720), bool)
    sel = w > 0
    ore_cell[r[sel], c[sel]] = True
    ncls = int(cls05.max()) + 1
    n_any = np.bincount(cls05[any_cell], minlength=ncls).astype(float)
    n_ore = np.bincount(cls05[ore_cell & any_cell], minlength=ncls).astype(float)
    g = n_ore.sum() / max(n_any.sum(), 1.0)
    b = 25.0
    ratio = ((n_ore + b * g) / (n_any + b)) / max(g, 1e-9)
    ratio = np.clip(ratio, PRIOR_LO, PRIOR_HI)
    ratio[0] = 1.0
    for v in neutral:
        ratio[v] = 1.0  # ice / water / no data: no lithological information
    return ratio


def build(paths):
    pts = Points()
    load_mrds(paths["mrds"], pts)
    load_major(paths["major"], pts)
    load_minfac(paths["minfac"], pts)
    load_wocqi(paths["wocqi"], pts)
    log("total records", len(pts))
    lat = np.asarray(pts.lat)
    lon = np.asarray(pts.lon)
    w_all = np.asarray(pts.w_all, dtype=float)
    src = np.asarray(pts.src)

    # land mask and coal fields
    land_frac = rasterise_polys(paths["land"])
    land = land_frac >= 0.5
    cf = rasterise_polys(paths["coalfields"])  # coverage fraction 0..1
    coal_cell_w = cf * SRC_W["coalfield_cell"]
    cls05_raw, cls05 = load_glim(paths["glim"])
    neutral = glim_neutral_classes(paths["glim"])
    cls025 = np.repeat(np.repeat(cls05, 2, axis=0), 2, axis=1)

    g_all = bin_points(lat, lon, w_all) + coal_cell_w  # coal field area also counts as survey knowledge
    major_sel = src == "major"
    g_all_major = bin_points(lat[major_sel], lon[major_sel], w_all[major_sel])
    d_all = density(g_all, SIGMA_ALL)
    d_all_major = density(g_all_major, SIGMA_MAJOR)
    landm = ndi.binary_dilation(land, iterations=1)

    log_field = {}
    for ore in ORES:
        w = ore_weights(pts, ore)
        grid = bin_points(lat, lon, w)
        if ore == "coal":
            grid = grid + coal_cell_w
        grid_major = bin_points(lat[major_sel], lon[major_sel], w[major_sel])
        # lithology prior, smoothed over land
        pr = lithology_prior(pts, ore, cls05, neutral)
        prior = np.log2(pr[cls025])
        prior = masked_smooth(prior, land, 2.0)
        pri_lin = 2.0 ** prior

        d_ore = density(grid, SIGMA_ORE)
        d_ore_major = density(grid_major, SIGMA_MAJOR)
        g_share = d_ore[landm].sum() / max(d_all[landm].sum(), 1e-9)
        g_share_major = d_ore_major[landm].sum() / max(d_all_major[landm].sum(), 1e-9)
        ratio = (d_ore + PSEUDO_A * pri_lin) / (d_all * g_share + PSEUDO_A)
        ratio_major = (d_ore_major + PSEUDO_A_MAJOR * pri_lin) / (d_all_major * g_share_major + PSEUDO_A_MAJOR)
        L = (1.0 - MAJOR_BLEND) * np.log2(ratio) + MAJOR_BLEND * np.log2(ratio_major)
        log_field[ore] = (L, g_share, grid.sum())
    return log_field, land, land_frac


def finish(L, land):
    """Recentre, scale, saturate, smooth, mask -> richness log2 field (ocean 0)."""
    med = np.median(L[land])
    L = L - med
    L = np.where(L > 0, L * LOG_SCALE, L * NEG_SCALE)
    hi, lo = np.log2(R_MAX), np.log2(R_MIN)
    L = np.where(L > 0, hi * np.tanh(L / hi), lo * np.tanh(L / lo))  # lo < 0: soft floor
    L = masked_smooth(L, land, FINAL_SIGMA / CELL)
    # feather coast: weight from smoothed land fraction (a straight coast ~0.5 -> weight 1)
    lf = ndi.gaussian_filter(land.astype(float), 1.5, mode=("nearest", "wrap"))
    L = L * np.clip(2.0 * lf, 0.0, 1.0)
    L = np.where(land, L, 0.0)
    return np.clip(L, lo, hi)


def encode(qs):
    """qs: dict ore -> uint8 [ROWS, COLS]. Returns file bytes."""
    head = b"HZOR" + struct.pack("<I", 1) + struct.pack("<HH", COLS, ROWS)
    head += struct.pack("<fff", LON_W, LAT_N, CELL)
    head += struct.pack("<B", len(ORES))
    for o in ORES:
        nb = o.encode("ascii")
        head += struct.pack("<B", len(nb)) + nb
    raw = bytearray()
    for o in ORES:
        q = qs[o].astype(np.uint8)
        d = np.empty_like(q)
        d[:, 0] = q[:, 0]
        d[:, 1:] = (q[:, 1:].astype(np.int16) - q[:, :-1].astype(np.int16)) % 256
        raw += d.tobytes()
    head += struct.pack("<I", len(raw))
    return head + zlib.compress(bytes(raw), 9)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    paths = prepare_sources(args.cache)
    log_field, land, land_frac = build(paths)

    qs = {}
    stats = []
    for ore in ORES:
        L, g_share, nw = log_field[ore]
        Lf = finish(L, land)
        q = np.where(land, np.rint(128 + 32 * Lf), 128).astype(np.int64)
        q = np.clip(q, 72, 202)
        q[~land] = 128
        qs[ore] = q.astype(np.uint8)
        rich = 2.0 ** ((q[land] - 128) / 32.0)
        stats.append((ore, g_share, rich.min(), np.percentile(rich, 5), np.median(rich),
                      np.percentile(rich, 95), rich.max(), float((rich > 1.5).mean())))
    blob = encode(qs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)

    print(f"wrote {args.out} ({len(blob)} bytes), grid {COLS}x{ROWS} at {CELL} deg, land cells {int(land.sum())}")
    print(f"{'ore':9s} {'g_share':>8s} {'min':>6s} {'p5':>6s} {'median':>7s} {'p95':>6s} {'max':>6s} {'frac>1.5':>9s}")
    for s in stats:
        print(f"{s[0]:9s} {s[1]:8.4f} {s[2]:6.2f} {s[3]:6.2f} {s[4]:7.2f} {s[5]:6.2f} {s[6]:6.2f} {s[7]:9.3f}")


if __name__ == "__main__":
    main()

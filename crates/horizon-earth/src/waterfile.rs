//! The water atlas file: how [`WaterAtlas::encode`](crate::water::WaterAtlas::encode) writes it and
//! [`WaterAtlas::decode`](crate::water::WaterAtlas::decode) reads it. `tools/earth_water.py` writes the same format
//! (the full spec, byte by byte, is in that script's docstring and in the notes' `format.md`).
//!
//! The file is a header, an index of the non-empty 1-degree cells, and one raw DEFLATE block per cell:
//!
//! * Header (24 bytes, little-endian): the magic `HZWA`; the version, equal to [`WATER_RULES`]; the coordinate
//!   units per degree ([`UNITS`] = 9,600: about 11.6 m, and HydroSHEDS's 15-arc-second grid is exactly 40 units);
//!   the number of distinct reaches and of distinct lakes; the number of cells.
//! * Index: one 16-byte entry per cell, sorted by `(lat, lon)` with no repeat: the cell's `lat` and `lon` (`i16`,
//!   the floor of the degrees, as [`cells_of`](crate::water) computes them), the block's offset from the start of
//!   the file, its compressed length and its inflated length (`u32` each; at most [`MAX_CELL_RAW`]).
//! * A block holds every reach and every lake whose bounding box meets the cell (so a feature over several cells
//!   is in each), reaches then lakes, each in strictly ascending id order: the two counts, then per reach the step
//!   from the previous reach's id (from 0 for the first), the step from its id to the id it flows into (zigzag; 0
//!   for none), its discharge (a [log code](log_value)), the point count (at least 2), the first point as a step
//!   from the cell's south-west corner and then the step to each next point (units), the first level and then the
//!   drop to each next level (whole decimetres; a drop is never negative, so a level never rises); per lake the
//!   step from the previous lake's id, its box's south-west corner as a step from the cell's corner and the box's
//!   size (units, rounded outwards), its level (decimetres), its area (km2) and mean depth (m) as log codes, and a
//!   flag byte (bit 0: a reservoir). Integers are LEB128 varints, signed ones zigzag-coded; log codes are `u16`.
//!
//! Reading checks everything: every length and offset against the bytes there are, every count against the bytes
//! left, every id and coordinate against its range; a bad file or block is an error, never a panic.

use std::collections::HashSet;

use crate::projection::LatLon;
use crate::water::{Lake, RiverReach, WATER_RULES, WaterAtlasError, cells_of};

/// The first four bytes of an atlas file.
pub(crate) const MAGIC: &[u8; 4] = b"HZWA";
/// The coordinate units per degree the encoder writes (about 1.04e-4 degree, 11.6 m).
pub const UNITS: u32 = 9_600;
/// The largest inflated cell block accepted, in bytes.
pub const MAX_CELL_RAW: usize = 64 << 20;
/// The largest units-per-degree a file may declare.
const MAX_UNITS: u32 = 10_000_000;
/// The header's length in bytes.
const HEADER_LEN: usize = 24;
/// One index entry's length in bytes.
const ENTRY_LEN: usize = 16;
/// The smallest encoded reach (id, next, discharge, count, two points, two levels: a byte each but the u16).
const MIN_REACH_BYTES: usize = 11;
/// The smallest encoded lake (id, corner, size, level: a byte each; two u16; the flags).
const MIN_LAKE_BYTES: usize = 11;
/// The log code of 1: a code `c` > 0 stands for `2^((c - LOG_ONE) / LOG_STEPS)`, 0 for 0.
const LOG_ONE: f64 = 32_768.0;
/// Log codes per doubling.
const LOG_STEPS: f64 = 1_024.0;
/// The flag bit of a reservoir.
const FLAG_RESERVOIR: u8 = 1;
/// Decimetres in a metre: levels are stored in whole decimetres.
const DM_PER_M: f32 = 10.0;

/// Where one cell's block lies in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub key: (i32, i32),
    pub offset: usize,
    pub len: usize,
    pub raw_len: usize,
}

/// A file's header and index.
#[derive(Debug)]
pub(crate) struct Header {
    pub units: f64,
    pub reaches: u32,
    pub lakes: u32,
    pub index: Vec<Entry>,
}

fn err(what: impl Into<String>) -> WaterAtlasError {
    WaterAtlasError(what.into())
}

/// A bounds-checked reader over bytes.
struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }

    fn left(&self) -> usize {
        self.b.len() - self.p
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], WaterAtlasError> {
        if n > self.left() {
            return Err(err("truncated"));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], WaterAtlasError> {
        let mut a = [0; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn u8(&mut self) -> Result<u8, WaterAtlasError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, WaterAtlasError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn i16(&mut self) -> Result<i16, WaterAtlasError> {
        Ok(i16::from_le_bytes(self.array()?))
    }

    fn u16(&mut self) -> Result<u16, WaterAtlasError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn varint(&mut self) -> Result<u64, WaterAtlasError> {
        let mut n: u64 = 0;
        for shift in (0..64).step_by(7) {
            let b = self.u8()?;
            let bits = u64::from(b & 0x7f);
            if shift == 63 && bits > 1 {
                return Err(err("varint overflows"));
            }
            n |= bits << shift;
            if b & 0x80 == 0 {
                return Ok(n);
            }
        }
        Err(err("varint too long"))
    }

    fn zvarint(&mut self) -> Result<i64, WaterAtlasError> {
        let u = self.varint()?;
        Ok((u >> 1) as i64 ^ -((u & 1) as i64))
    }

    /// A count of items each at least `min_bytes` long, checked against the bytes left.
    fn count(&mut self, min_bytes: usize) -> Result<usize, WaterAtlasError> {
        let n = usize::try_from(self.varint()?).map_err(|_| err("count too large"))?;
        if n.saturating_mul(min_bytes) > self.left() {
            return Err(err("count larger than the data"));
        }
        Ok(n)
    }
}

/// Reads and checks the header and the index of `bytes`.
pub(crate) fn read_header(bytes: &[u8]) -> Result<Header, WaterAtlasError> {
    let mut c = Cursor::new(bytes);
    if c.take(4).map_err(|_| err("shorter than the header"))? != MAGIC {
        return Err(err("not a water atlas (bad magic)"));
    }
    let version = c.u32()?;
    if version != WATER_RULES {
        return Err(err(format!(
            "version {version}, this game reads {WATER_RULES}"
        )));
    }
    let units = c.u32()?;
    if units == 0 || units > MAX_UNITS {
        return Err(err(format!("bad units per degree {units}")));
    }
    let reaches = c.u32()?;
    let lakes = c.u32()?;
    let cells = c.u32()? as usize;
    let data_start = cells
        .checked_mul(ENTRY_LEN)
        .and_then(|n| n.checked_add(HEADER_LEN))
        .filter(|&n| n <= bytes.len())
        .ok_or_else(|| err("index larger than the file"))?;
    let mut index = Vec::with_capacity(cells);
    for _ in 0..cells {
        let key = (i32::from(c.i16()?), i32::from(c.i16()?));
        let offset = c.u32()? as usize;
        let len = c.u32()? as usize;
        let raw_len = c.u32()? as usize;
        if !(-90..=90).contains(&key.0) || !(-180..=180).contains(&key.1) {
            return Err(err(format!("cell {key:?} off the globe")));
        }
        if index.last().is_some_and(|e: &Entry| e.key >= key) {
            return Err(err("index not sorted"));
        }
        if offset < data_start || offset.checked_add(len).is_none_or(|end| end > bytes.len()) {
            return Err(err(format!("cell {key:?} block outside the file")));
        }
        if raw_len > MAX_CELL_RAW {
            return Err(err(format!("cell {key:?} block too large")));
        }
        index.push(Entry {
            key,
            offset,
            len,
            raw_len,
        });
    }
    Ok(Header {
        units: f64::from(units),
        reaches,
        lakes,
        index,
    })
}

/// The reaches and lakes of one cell's block.
pub(crate) fn read_cell(
    bytes: &[u8],
    entry: &Entry,
    units: f64,
) -> Result<(Vec<RiverReach>, Vec<Lake>), WaterAtlasError> {
    let block = bytes
        .get(entry.offset..entry.offset.saturating_add(entry.len))
        .ok_or_else(|| err("block outside the file"))?;
    let raw = miniz_oxide::inflate::decompress_to_vec_with_limit(block, entry.raw_len)
        .map_err(|e| err(format!("block does not inflate: {e:?}")))?;
    if raw.len() != entry.raw_len {
        return Err(err("block inflates to the wrong length"));
    }
    let mut c = Cursor::new(&raw);
    let n_reaches = c.count(1)?;
    let n_lakes = c.count(1)?;
    if n_reaches.saturating_mul(MIN_REACH_BYTES) + n_lakes.saturating_mul(MIN_LAKE_BYTES) > c.left()
    {
        return Err(err("counts larger than the block"));
    }
    // Units are whole numbers up to MAX_UNITS, so the corner is exact.
    let corner = (
        i64::from(entry.key.0) * units as i64,
        i64::from(entry.key.1) * units as i64,
    );
    let degrees = |v: i64| v as f64 / units;
    let on_globe =
        |lat: f64, lon: f64| (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon);
    let mut reaches = Vec::with_capacity(n_reaches);
    let mut id = 0u32;
    for _ in 0..n_reaches {
        id = next_id(id, c.varint()?)?;
        let step = c.zvarint()?;
        let next_down = if step == 0 {
            None
        } else {
            Some(
                i64::from(id)
                    .checked_add(step)
                    .and_then(|d| u32::try_from(d).ok())
                    .ok_or_else(|| err(format!("reach {id} flows into a bad id")))?,
            )
        };
        let discharge_m3s = log_value(c.u16()?);
        let n = c.count(3)?;
        if n < 2 {
            return Err(err(format!("reach {id} has fewer than two points")));
        }
        let mut points = Vec::with_capacity(n);
        let (mut lat, mut lon) = corner;
        for _ in 0..n {
            let (dlat, dlon) = (c.zvarint()?, c.zvarint()?);
            lat = lat
                .checked_add(dlat)
                .ok_or_else(|| err("point overflows"))?;
            lon = lon
                .checked_add(dlon)
                .ok_or_else(|| err("point overflows"))?;
            let p = LatLon {
                lat: degrees(lat),
                lon: degrees(lon),
            };
            if !on_globe(p.lat, p.lon) {
                return Err(err(format!("reach {id} has a point off the globe")));
            }
            points.push(p);
        }
        let mut levels_m = Vec::with_capacity(n);
        let mut level = c.zvarint()?;
        levels_m.push(dm_to_m(level)?);
        for _ in 1..n {
            let drop = i64::try_from(c.varint()?).map_err(|_| err("level drop too large"))?;
            level = level
                .checked_sub(drop)
                .ok_or_else(|| err("level underflows"))?;
            levels_m.push(dm_to_m(level)?);
        }
        reaches.push(RiverReach {
            id,
            points,
            levels_m,
            discharge_m3s,
            next_down,
        });
    }
    let mut lakes = Vec::with_capacity(n_lakes);
    let mut id = 0u32;
    for _ in 0..n_lakes {
        id = next_id(id, c.varint()?)?;
        let s = corner
            .0
            .checked_add(c.zvarint()?)
            .ok_or_else(|| err("lake overflows"))?;
        let w = corner
            .1
            .checked_add(c.zvarint()?)
            .ok_or_else(|| err("lake overflows"))?;
        let dn = i64::try_from(c.varint()?).map_err(|_| err("lake too large"))?;
        let de = i64::try_from(c.varint()?).map_err(|_| err("lake too large"))?;
        let n = s.checked_add(dn).ok_or_else(|| err("lake overflows"))?;
        let e = w.checked_add(de).ok_or_else(|| err("lake overflows"))?;
        let level_m = dm_to_m(c.zvarint()?)?;
        let area_km2 = log_value(c.u16()?);
        let depth_avg_m = log_value(c.u16()?);
        let flags = c.u8()?;
        if flags & !FLAG_RESERVOIR != 0 {
            return Err(err(format!("lake {id} has unknown flags")));
        }
        let lake = Lake {
            id,
            south: degrees(s),
            west: degrees(w),
            north: degrees(n),
            east: degrees(e),
            level_m,
            area_km2,
            depth_avg_m,
            reservoir: flags & FLAG_RESERVOIR != 0,
        };
        if !on_globe(lake.south, lake.west) || !on_globe(lake.north, lake.east) {
            return Err(err(format!("lake {id} is off the globe")));
        }
        lakes.push(lake);
    }
    if c.left() != 0 {
        return Err(err("trailing bytes in a block"));
    }
    Ok((reaches, lakes))
}

/// The id after `prev` by a step that must be positive (ids rise strictly in a block; the first steps from 0).
fn next_id(prev: u32, step: u64) -> Result<u32, WaterAtlasError> {
    if step == 0 {
        return Err(err("ids not in strictly rising order"));
    }
    u32::try_from(step)
        .ok()
        .and_then(|s| prev.checked_add(s))
        .ok_or_else(|| err("id out of range"))
}

/// The value a log code stands for: 0 for 0, else `2^((code - 32768) / 1024)` (steps of 0.07 %).
pub fn log_value(code: u16) -> f32 {
    if code == 0 {
        0.0
    } else {
        ((f64::from(code) - LOG_ONE) / LOG_STEPS).exp2() as f32
    }
}

/// The log code nearest a value (0 for zero, less, or not finite; clamped to 1..=65535).
pub fn log_code(x: f32) -> u16 {
    if x.is_nan() || x <= 0.0 || x.is_infinite() {
        return 0;
    }
    (f64::from(x).log2() * LOG_STEPS + LOG_ONE)
        .round()
        .clamp(1.0, 65_535.0) as u16
}

/// A level in whole decimetres as metres (the same function for every reader, so equal decimetres give equal
/// metres).
fn dm_to_m(dm: i64) -> Result<f32, WaterAtlasError> {
    let dm = i32::try_from(dm).map_err(|_| err("level out of range"))?;
    Ok(dm as f32 / DM_PER_M)
}

/// Metres as whole decimetres.
fn m_to_dm(m: f32) -> i64 {
    (f64::from(m) * f64::from(DM_PER_M)).round() as i64
}

fn put_varint(out: &mut Vec<u8>, mut n: u64) {
    loop {
        let b = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn put_zvarint(out: &mut Vec<u8>, n: i64) {
    put_varint(out, ((n << 1) ^ (n >> 63)) as u64);
}

fn units_of(deg: f64) -> i64 {
    (deg * f64::from(UNITS)).round() as i64
}

/// One feature's bytes in a block, but for the id step and the first point (which depend on the cell).
struct Encoded {
    id: u32,
    head: Vec<u8>,
    first: (i64, i64),
    tail: Vec<u8>,
}

/// The file holding `reaches` and `lakes` (see the module doc). Coordinates are rounded to [`UNITS`], levels to
/// decimetres and quantities to log codes; a level that rises from one point to the next is written flat (a drop of
/// 0). Each feature goes to every cell its rounded bounding box meets. Ids must be unique per kind (a repeated id is
/// written once); a reach that flows into itself is written as flowing nowhere.
pub(crate) fn write(reaches: &[RiverReach], lakes: &[Lake]) -> Vec<u8> {
    use std::collections::BTreeMap;
    let units = f64::from(UNITS);
    let deg = |v: i64| v as f64 / units;
    let mut cells: BTreeMap<(i32, i32), (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    let mut reach_enc: Vec<Encoded> = Vec::new();
    let mut seen = HashSet::new();
    for r in reaches
        .iter()
        .filter(|r| r.points.len() >= 2 && seen.insert(r.id))
    {
        let mut head = Vec::new();
        let step = r
            .next_down
            .filter(|&d| d != r.id)
            .map_or(0, |d| i64::from(d) - i64::from(r.id));
        put_zvarint(&mut head, step);
        head.extend_from_slice(&log_code(r.discharge_m3s).to_le_bytes());
        put_varint(&mut head, r.points.len() as u64);
        let q: Vec<(i64, i64)> = r
            .points
            .iter()
            .map(|p| (units_of(p.lat), units_of(p.lon)))
            .collect();
        let mut tail = Vec::new();
        for w in q.windows(2) {
            put_zvarint(&mut tail, w[1].0 - w[0].0);
            put_zvarint(&mut tail, w[1].1 - w[0].1);
        }
        let mut level = m_to_dm(r.levels_m.first().copied().unwrap_or(0.0));
        put_zvarint(&mut tail, level);
        for i in 1..r.points.len() {
            let next = r.levels_m.get(i).map_or(level, |&l| m_to_dm(l).min(level));
            put_varint(&mut tail, (level - next) as u64);
            level = next;
        }
        let bounds = (
            deg(q.iter().map(|p| p.0).min().unwrap_or(0)),
            deg(q.iter().map(|p| p.1).min().unwrap_or(0)),
            deg(q.iter().map(|p| p.0).max().unwrap_or(0)),
            deg(q.iter().map(|p| p.1).max().unwrap_or(0)),
        );
        for key in cells_of(bounds) {
            cells.entry(key).or_default().0.push(reach_enc.len());
        }
        reach_enc.push(Encoded {
            id: r.id,
            head,
            first: q[0],
            tail,
        });
    }
    let mut lake_enc: Vec<Encoded> = Vec::new();
    let mut seen = HashSet::new();
    for l in lakes.iter().filter(|l| seen.insert(l.id)) {
        let s = (l.south * units).floor() as i64;
        let w = (l.west * units).floor() as i64;
        let n = ((l.north * units).ceil() as i64).max(s);
        let e = ((l.east * units).ceil() as i64).max(w);
        let mut tail = Vec::new();
        put_varint(&mut tail, (n - s) as u64);
        put_varint(&mut tail, (e - w) as u64);
        put_zvarint(&mut tail, m_to_dm(l.level_m));
        tail.extend_from_slice(&log_code(l.area_km2).to_le_bytes());
        tail.extend_from_slice(&log_code(l.depth_avg_m).to_le_bytes());
        tail.push(if l.reservoir { FLAG_RESERVOIR } else { 0 });
        for key in cells_of((deg(s), deg(w), deg(n), deg(e))) {
            cells.entry(key).or_default().1.push(lake_enc.len());
        }
        lake_enc.push(Encoded {
            id: l.id,
            head: Vec::new(),
            first: (s, w),
            tail,
        });
    }
    let mut blocks = Vec::with_capacity(cells.len());
    for (key, (mut rs, mut ls)) in cells {
        rs.sort_unstable_by_key(|&i| reach_enc[i].id);
        ls.sort_unstable_by_key(|&i| lake_enc[i].id);
        let corner = (
            i64::from(key.0) * i64::from(UNITS),
            i64::from(key.1) * i64::from(UNITS),
        );
        let mut raw = Vec::new();
        put_varint(&mut raw, rs.len() as u64);
        put_varint(&mut raw, ls.len() as u64);
        for (list, enc) in [(&rs, &reach_enc), (&ls, &lake_enc)] {
            let mut prev = 0u32;
            for &i in list {
                let f = &enc[i];
                put_varint(&mut raw, u64::from(f.id - prev));
                raw.extend_from_slice(&f.head);
                put_zvarint(&mut raw, f.first.0 - corner.0);
                put_zvarint(&mut raw, f.first.1 - corner.1);
                raw.extend_from_slice(&f.tail);
                prev = f.id;
            }
        }
        blocks.push((
            key,
            miniz_oxide::deflate::compress_to_vec(&raw, 9),
            raw.len(),
        ));
    }
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    for v in [
        WATER_RULES,
        UNITS,
        reach_enc.len() as u32,
        lake_enc.len() as u32,
        blocks.len() as u32,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut offset = HEADER_LEN + ENTRY_LEN * blocks.len();
    for ((lat, lon), comp, raw_len) in &blocks {
        out.extend_from_slice(&(*lat as i16).to_le_bytes());
        out.extend_from_slice(&(*lon as i16).to_le_bytes());
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        out.extend_from_slice(&(*raw_len as u32).to_le_bytes());
        offset += comp.len();
    }
    for (_, comp, _) in blocks {
        out.extend_from_slice(&comp);
    }
    out
}

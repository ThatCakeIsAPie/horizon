//! Timing helpers: the stages of a window's land class read, one at a time (warp, projection, store lookups,
//! speck and vote), for the cost breakdown in horizon-world's `earth_class_profile` example.

#[allow(unused_imports)]
use super::super::*;

use std::cell::RefCell;
use std::hint::black_box;

/// What a window read keeps of one tile lookup, so the same file / tile is not looked up again for each sample.
#[derive(Default)]
struct Cache {
    file: Option<(WorldCoverFile, bool, Option<Arc<TiffIndex>>)>,
    tiles: Vec<((WorldCoverFile, usize, usize), Option<Arc<LandTile>>)>,
}

impl Cache {
    /// `Ok(None)` when the file is absent.
    fn index(
        &mut self,
        store: &LandCoverStore,
        file: WorldCoverFile,
    ) -> Result<Option<Arc<TiffIndex>>, Fetch> {
        if let Some((f, absent, index)) = &self.file {
            if *f == file {
                return Ok(if *absent { None } else { index.clone() });
            }
        }
        if store.is_absent(file) {
            self.file = Some((file, true, None));
            return Ok(None);
        }
        let index = store.index_of(file)?;
        self.file = Some((file, false, Some(index.clone())));
        Ok(Some(index))
    }

    fn tile(
        &mut self,
        store: &LandCoverStore,
        file: WorldCoverFile,
        level_no: usize,
        level: &Level,
        tile_no: usize,
    ) -> Result<Option<Arc<LandTile>>, Fetch> {
        let key = (file, level_no, tile_no);
        if let Some((_, t)) = self.tiles.iter().find(|(k, _)| *k == key) {
            return Ok(t.clone());
        }
        let t = store.tile_of(file, level_no, level, tile_no)?;
        if self.tiles.len() >= 8 {
            self.tiles.remove(0);
        }
        self.tiles.push((key, t.clone()));
        Ok(t)
    }
}

fn fold(acc: u64, v: u64) -> u64 {
    (acc ^ v).wrapping_mul(0x0000_0100_0000_01B3)
}

/// A copy of `LandCoverStore::read(.., smooth = true)`: up to and including the own tile (`full` false), or
/// all of it (`full` true), as a checksum. `cache` resolves file / index / tile once per window instead of per
/// sample. `Err` is the fetch of a missing piece.
fn read_stage(
    store: &LandCoverStore,
    lat: f64,
    lon: f64,
    metres_per_sample: f64,
    full: bool,
    cache: Option<&RefCell<Cache>>,
) -> Result<u64, Fetch> {
    let Some(file) = WorldCoverFile::containing(lat, lon) else {
        return Ok(1);
    };
    let index = match cache {
        None => {
            if store.is_absent(file) {
                return Ok(2);
            }
            store.index_of(file)?
        }
        Some(c) => match c.borrow_mut().index(store, file)? {
            None => return Ok(2),
            Some(index) => index,
        },
    };
    let level_no = level_for(&index.levels, metres_per_sample);
    let Some(level) = index.levels.get(level_no) else {
        return Ok(3);
    };
    let (px, py) = file.pixel(lat, lon, level.width, level.height);
    let inside = file.pixel_fraction(lat, lon, level.width, level.height);
    let (col, row) = (px / level.tile_width, py / level.tile_height);
    let Some(tile_no) = level.tile_index(col, row) else {
        return Ok(4);
    };
    let salt =
        ((u64::from(file.south as u16) << 32) | (u64::from(file.west as u16) << 16)) ^ (level_no as u64);
    let tile_of = |no: usize| match cache {
        None => store.tile_of(file, level_no, level, no),
        Some(c) => c.borrow_mut().tile(store, file, level_no, level, no),
    };
    let tile = match tile_of(tile_no) {
        Ok(None) => return Ok(5),
        Ok(Some(tile)) => tile,
        Err(fetch) => return Err(fetch),
    };
    let mut acc = fold(fold(u64::from(px), u64::from(py)), tile_no as u64);
    acc = fold(acc, inside.0.to_bits() ^ inside.1.to_bits().rotate_left(7));
    acc = fold(acc, salt);
    if !full {
        return Ok(acc);
    }
    let (lx, ly) = (px % level.tile_width, py % level.tile_height);
    let place = (i64::from(px), i64::from(py), salt);
    if let Some(kept) = tile.speck_kept(lx, ly, inside, place) {
        return Ok(fold(acc, kept as u64));
    }
    let (u, v) = (inside.0 - 0.5, inside.1 - 0.5);
    let (step_x, tx) = (if u >= 0.0 { 1_i64 } else { -1 }, u.abs());
    let (step_y, ty) = (if v >= 0.0 { 1_i64 } else { -1 }, v.abs());
    let (own, here) = ((i64::from(px), i64::from(py)), (col, row));
    let code_at = |x: i64, y: i64| -> Result<u8, Fetch> {
        let x = x.clamp(0, i64::from(level.width) - 1) as u32;
        let y = y.clamp(0, i64::from(level.height) - 1) as u32;
        let there = (x / level.tile_width, y / level.tile_height);
        let (lx, ly) = (x % level.tile_width, y % level.tile_height);
        if there == here {
            return Ok(tile.code_at(lx, ly).unwrap_or(0));
        }
        let Some(no) = level.tile_index(there.0, there.1) else {
            return Ok(0);
        };
        match tile_of(no as usize)? {
            Some(other) => Ok(other.code_at(lx, ly).unwrap_or(0)),
            None => Ok(0),
        }
    };
    let samples = [
        (own.0, own.1, (1.0 - tx) * (1.0 - ty)),
        (own.0 + step_x, own.1, tx * (1.0 - ty)),
        (own.0, own.1 + step_y, (1.0 - tx) * ty),
        (own.0 + step_x, own.1 + step_y, tx * ty),
    ];
    let mut votes: [(u8, f64); 4] = [(0, 0.0); 4];
    let mut used = 0;
    for (x, y, weight) in samples {
        let code = code_at(x, y)?;
        match votes[..used].iter_mut().find(|(c, _)| *c == code) {
            Some((_, total)) => *total += weight,
            None => {
                votes[used] = (code, weight);
                used += 1;
            }
        }
    }
    let mut best = votes[0];
    for &(code, total) in &votes[1..used] {
        if total > best.1 + 1e-12 {
            best = (code, total);
        }
    }
    let class = LandClass::from_code(best.0).unwrap_or(LandClass::NoData);
    Ok(fold(acc, class as u64))
}

impl LandCoverField {
    /// The loop of `class_grid_reference` (j outer, i inner), stopped after `stage`, folded into a checksum so
    /// nothing is optimised out: 0 the loop only; 1 the warp; 2 + the projection; 3 + the store's lookup up to
    /// the own tile (no speck, no vote); 4 the whole read (the work of `class_grid_reference`); 5 = 3 with the
    /// file, index and tiles resolved once per window (a small cache) instead of per sample; 6 = 4 likewise.
    /// Returns `u64::MAX` as soon as a read needs bytes the store does not hold.
    #[doc(hidden)]
    pub fn class_grid_stage(
        &self,
        x0: i32,
        z0: i32,
        width: usize,
        depth: usize,
        stride: i32,
        metres_per_sample: f64,
        stage: u8,
    ) -> u64 {
        let cache = RefCell::new(Cache::default());
        let mut acc = 0xCBF2_9CE4_8422_2325_u64;
        for j in 0..depth {
            for i in 0..width {
                let x = f64::from(x0) + (i as f64) * f64::from(stride) + 0.5;
                let z = f64::from(z0) + (j as f64) * f64::from(stride) + 0.5;
                if stage == 0 {
                    acc = fold(acc, black_box(x).to_bits() ^ black_box(z).to_bits());
                    continue;
                }
                let (wx, wz) = self.warped(x, z, metres_per_sample);
                if stage == 1 {
                    acc = fold(acc, black_box(wx).to_bits() ^ black_box(wz).to_bits().rotate_left(1));
                    continue;
                }
                let at = self.projection.to_lat_lon(wx, wz);
                if stage == 2 {
                    acc = fold(acc, black_box(at.lat).to_bits() ^ black_box(at.lon).to_bits().rotate_left(1));
                    continue;
                }
                let got = match stage {
                    3 => read_stage(&self.store, at.lat, at.lon, metres_per_sample, false, None),
                    4 => read_stage(&self.store, at.lat, at.lon, metres_per_sample, true, None),
                    5 => read_stage(&self.store, at.lat, at.lon, metres_per_sample, false, Some(&cache)),
                    _ => read_stage(&self.store, at.lat, at.lon, metres_per_sample, true, Some(&cache)),
                };
                match got {
                    Ok(v) => acc = fold(acc, black_box(v)),
                    Err(_) => return u64::MAX,
                }
            }
        }
        black_box(acc)
    }
}

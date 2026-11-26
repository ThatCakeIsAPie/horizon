//! Cube far terrain kept on disk, beside a dimension's save
//! (`<dimension>/horizon_cube_lod/`), so a world opened again shows what was
//! seen before: every final far node of every level (generated Earth and
//! Custom terrain, the player's edits and vanilla's decorated terrain built
//! from saved cubes, and the coarser levels made of them).
//!
//! One append-only file per bucket of 8 x 8 x 8 nodes of one level. A record is
//! `[u32 length][kind u8][level u8][x i32][y i32][z i32][body][u32 checksum]`
//! (`length` counts everything after itself): kind 0 stores a node, kind 1
//! erases it. A newer record of a key replaces the older (a node made whole
//! later, a refreshed node). A record cut short by a crash ends the file there
//! when it is opened; a damaged body fails its checksum and reads as missing.
//! A file mostly made of dead records is rewritten on open. The folder holds a
//! fingerprint of the format, the build, the seed and the block registry; any
//! change empties it.
//!
//! Levels 0 to [`MAX_CACHED_LEVEL`] are stored. A level 0 node read from a
//! saved cube is never stored by the sampler (the save is the source); a level
//! 0 node the generator made is. When a cube is saved the owner of the save
//! calls [`CubeLodCache::invalidate`], which erases every stored node over it
//! at every level, level 0 included.
//!
//! Writes are batched ([`CubeLodCache::put_many`]: one append per bucket file)
//! and meant for one writer thread; nothing here is called on a far worker for
//! a write.
//!
//! Size: the folder is bounded ([`CubeLodCache::open_with_cap`], default
//! [`DEFAULT_CAP_MB`], `HORIZON_FAR_CACHE_MB`, 0 = unlimited). A bucket file's
//! modification time is set to now when it is opened in a session and when it
//! is written; when the files together pass the cap, whole bucket files with
//! the oldest modification time are deleted until they take 90% of it.
use super::cube::{
    Changes, CoverMask, CubeKey, CubeNode, FluidTops, MAX_CUBE_LEVEL, NO_FLUID_TOP, VOLUME, Voxel,
};
use crate::fast_hash::FxHashMap;
use flate2::{Compression, read::ZlibDecoder, write::ZlibEncoder};
use horizon_core::position::SectionPos;
use horizon_world::registry::BlockStateId;
use std::{
    collections::VecDeque,
    fs,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::SystemTime,
};

/// The coarsest level kept on disk (nodes 262,144 blocks wide): above the
/// root of any far view (`lodDistance` 1024 chunks = 16,384 blocks needs
/// level 10).
pub const MAX_CACHED_LEVEL: u8 = 14;
const _: () = assert!(MAX_CACHED_LEVEL <= MAX_CUBE_LEVEL);
/// A world's cache size when `HORIZON_FAR_CACHE_MB` does not say.
pub const DEFAULT_CAP_MB: u64 = 1024;
const FORMAT: u32 = 3;
const FINGERPRINT: &str = "fingerprint.txt";
/// Nodes along one side of a bucket file.
const BUCKET: i32 = 8;
/// Kind, level and the three coordinates, after the length.
const HEAD: usize = 14;
const TAIL: usize = 4;
const NODE: u8 = 0;
const ERASE: u8 = 1;
/// Cubes remembered for [`CubeLodCache::changes_since`] before it says "all".
const LOG: usize = 4096;
/// A bucket file this large with mostly dead records is rewritten on open.
const COMPACT_ABOVE: u64 = 1 << 20;
/// A node's raw voxels: a state and a biome each.
const RAW: usize = VOLUME * 6;
/// Bucket files kept open at once: the rest keep their index and open the
/// file again when it is used (a far view of 1,024 chunks touches thousands).
const MAX_OPEN: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct BucketId(u8, i32, i32, i32);

impl BucketId {
    fn of(key: CubeKey) -> Self {
        Self(
            key.level,
            key.x.div_euclid(BUCKET),
            key.y.div_euclid(BUCKET),
            key.z.div_euclid(BUCKET),
        )
    }
    fn file(self) -> String {
        format!("n{}.{}.{}.{}.bin", self.0, self.1, self.2, self.3)
    }
    /// The bucket a file name (`n<level>.<x>.<y>.<z>.bin`) is of.
    fn parse(name: &str) -> Option<Self> {
        let mut parts = name.strip_prefix('n')?.strip_suffix(".bin")?.split('.');
        let id = Self(
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
        );
        parts.next().is_none().then_some(id)
    }
}

fn checksum(bytes: &[u8]) -> u32 {
    // FNV-1a.
    bytes.iter().fold(0x811c_9dc5_u32, |h, &b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    })
}

fn open_file(path: &Path) -> std::io::Result<fs::File> {
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

struct Bucket {
    path: PathBuf,
    /// `None` once closed to keep the open files few: opened again on use.
    file: Option<fs::File>,
    /// Each key's latest record: where its content (after the length) starts
    /// and how long it is. `None`: erased.
    index: FxHashMap<CubeKey, Option<(u64, u32)>>,
    end: u64,
    /// When it was last used (the cache's tick), to close the least recent.
    used: u64,
}

impl Bucket {
    fn open(path: &Path) -> std::io::Result<Self> {
        let mut bucket = Self::scan(path)?;
        let live: u64 = bucket
            .index
            .values()
            .flatten()
            .map(|&(_, len)| 4 + u64::from(len))
            .sum();
        if bucket.end > COMPACT_ABOVE && live * 2 < bucket.end {
            // Rewritten beside it and swapped in with the file closed.
            let records = bucket.live_records()?;
            drop(bucket);
            let temporary = path.with_extension("tmp");
            fs::write(&temporary, &records)?;
            fs::rename(&temporary, path)?;
            return Self::scan(path);
        }
        Ok(bucket)
    }

    fn scan(path: &Path) -> std::io::Result<Self> {
        let mut file = open_file(path)?;
        let len = file.metadata()?.len();
        let mut index = FxHashMap::default();
        let mut end = 0u64;
        {
            let mut reader = BufReader::new(&mut file);
            loop {
                let mut length = [0; 4];
                if reader.read_exact(&mut length).is_err() {
                    break;
                }
                let length = u32::from_le_bytes(length);
                let mut head = [0; HEAD];
                if (length as usize) < HEAD + TAIL || reader.read_exact(&mut head).is_err() {
                    break;
                }
                let content = end + 4;
                // A record cut short by a crash ends the file there.
                if content + u64::from(length) > len
                    || reader
                        .seek_relative(i64::from(length) - HEAD as i64)
                        .is_err()
                {
                    break;
                }
                let int = |at: usize| i32::from_le_bytes(head[at..at + 4].try_into().unwrap());
                let key = CubeKey::new(head[1], int(2), int(6), int(10));
                match head[0] {
                    NODE => index.insert(key, Some((content, length))),
                    ERASE => index.insert(key, None),
                    _ => break,
                };
                end = content + u64::from(length);
            }
        }
        if len != end {
            file.set_len(end)?;
        }
        // Opened in this session: the last to go when the cache is full.
        let _ = file.set_modified(SystemTime::now());
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(file),
            index,
            end,
            used: 0,
        })
    }

    /// The file, opened again if it was closed.
    fn handle(&mut self) -> std::io::Result<&mut fs::File> {
        if self.file.is_none() {
            self.file = Some(open_file(&self.path)?);
        }
        Ok(self.file.as_mut().expect("just opened"))
    }

    /// The file's live records only, as a new file's bytes.
    fn live_records(&mut self) -> std::io::Result<Vec<u8>> {
        let mut keys: Vec<_> = self
            .index
            .iter()
            .filter_map(|(&key, at)| at.map(|at| (at.0, key, at.1)))
            .collect();
        keys.sort_unstable();
        let mut out = Vec::new();
        for (offset, _, length) in keys {
            let mut content = vec![0; length as usize];
            let file = self.handle()?;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut content)?;
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&content);
        }
        Ok(out)
    }

    /// The body of `key`'s latest record, if it is intact.
    fn read(&mut self, key: CubeKey) -> Option<Vec<u8>> {
        let (offset, length) = (*self.index.get(&key)?)?;
        let mut content = vec![0; length as usize];
        let file = self.handle().ok()?;
        file.seek(SeekFrom::Start(offset)).ok()?;
        file.read_exact(&mut content).ok()?;
        let (rest, tail) = content.split_at(content.len() - TAIL);
        if checksum(rest) != u32::from_le_bytes(tail.try_into().ok()?) {
            return None;
        }
        Some(rest[HEAD..].to_vec())
    }

    /// Appends `records` (kind, key, body) in one write, so a crash leaves at
    /// most a torn last record, and the latest of a key wins.
    fn append_many(&mut self, records: &[(u8, CubeKey, &[u8])]) -> std::io::Result<()> {
        let mut out = Vec::with_capacity(
            records
                .iter()
                .map(|r| 4 + HEAD + r.2.len() + TAIL)
                .sum::<usize>(),
        );
        // Where each record's content starts and its length.
        let mut placed = Vec::with_capacity(records.len());
        for &(kind, key, body) in records {
            let start = out.len();
            out.extend_from_slice(&[0; 4]);
            out.push(kind);
            out.push(key.level);
            for c in [key.x, key.y, key.z] {
                out.extend_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(body);
            let sum = checksum(&out[start + 4..]);
            out.extend_from_slice(&sum.to_le_bytes());
            let length = (out.len() - start - 4) as u32;
            out[start..start + 4].copy_from_slice(&length.to_le_bytes());
            placed.push((self.end + start as u64 + 4, length));
        }
        let end = self.end;
        let file = self.handle()?;
        file.seek(SeekFrom::Start(end))?;
        if let Err(e) = file.write_all(&out) {
            let _ = file.set_len(end);
            return Err(e);
        }
        // Writing is using: the file goes to the back of the eviction line.
        let _ = file.set_modified(SystemTime::now());
        for (&(kind, key, _), at) in records.iter().zip(placed) {
            self.index.insert(key, (kind == NODE).then_some(at));
        }
        self.end += out.len() as u64;
        Ok(())
    }
}

#[derive(Default)]
struct ChangeLog {
    /// Sequence number of the next change.
    next: u64,
    entries: VecDeque<(u64, SectionPos)>,
}

#[derive(Default)]
struct Buckets {
    /// Open buckets; `None`: the file does not exist (or did not open).
    map: FxHashMap<BucketId, Option<Bucket>>,
    tick: u64,
}

impl Buckets {
    /// Closes the files of the least recently used buckets beyond
    /// [`MAX_OPEN`] (down to half of it); their indexes stay.
    fn trim(&mut self) {
        let mut open: Vec<(u64, BucketId)> = self
            .map
            .iter()
            .filter_map(|(&id, b)| {
                b.as_ref()
                    .filter(|b| b.file.is_some())
                    .map(|b| (b.used, id))
            })
            .collect();
        if open.len() <= MAX_OPEN {
            return;
        }
        open.sort_unstable();
        let close = open.len() - MAX_OPEN / 2;
        for (_, id) in open.into_iter().take(close) {
            if let Some(Some(bucket)) = self.map.get_mut(&id) {
                bucket.file = None;
            }
        }
    }
}

/// A dimension's cube far terrain on disk, shared by the sampling threads,
/// the writer thread and the save that invalidates it.
pub struct CubeLodCache {
    dir: PathBuf,
    buckets: Mutex<Buckets>,
    changes: Mutex<ChangeLog>,
    /// Most bytes the bucket files may take together; 0: no limit.
    cap: u64,
    /// The bucket files' bytes together (kept up to date; recounted when it
    /// passes `cap`).
    total: AtomicU64,
    /// Appends made: one per bucket and batch.
    writes: AtomicU64,
}

/// The cache size `HORIZON_FAR_CACHE_MB` asks for, in bytes (0: no limit).
pub fn cap_from_env() -> u64 {
    std::env::var("HORIZON_FAR_CACHE_MB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_CAP_MB)
        .saturating_mul(1 << 20)
}

impl CubeLodCache {
    /// Opens `dir`, emptying it when its fingerprint is not `fingerprint`
    /// (see [`fingerprint`]), bounded as [`cap_from_env`] says. `None` (no
    /// caching) on an error.
    pub fn open(dir: &Path, fingerprint: &str) -> Option<Arc<Self>> {
        Self::open_with_cap(dir, fingerprint, cap_from_env())
    }

    /// [`Self::open`] with a size cap of `cap` bytes (0: no limit): the
    /// oldest bucket files are deleted at once when the folder is over it.
    pub fn open_with_cap(dir: &Path, fingerprint: &str, cap: u64) -> Option<Arc<Self>> {
        let current = fs::read_to_string(dir.join(FINGERPRINT)).ok();
        if current.as_deref() != Some(fingerprint) {
            if current.is_some() {
                log::info!("Horizon: cube far terrain cache is out of date; building it again");
            }
            let reset = || -> std::io::Result<()> {
                if dir.exists() {
                    fs::remove_dir_all(dir)?;
                }
                fs::create_dir_all(dir)?;
                fs::write(dir.join(FINGERPRINT), fingerprint)
            };
            if let Err(e) = reset() {
                log::warn!(
                    "Horizon: cube far terrain will not be kept ({}): {e}",
                    dir.display()
                );
                return None;
            }
        }
        let cache = Arc::new(Self {
            dir: dir.to_path_buf(),
            buckets: Mutex::default(),
            changes: Mutex::default(),
            cap,
            total: AtomicU64::new(0),
            writes: AtomicU64::new(0),
        });
        // The files already there count, and the oldest go when over the cap.
        let total: u64 = cache.files().iter().map(|f| f.2).sum();
        cache.total.store(total, Relaxed);
        cache.enforce();
        Some(cache)
    }

    /// Appends made to bucket files so far (one per bucket and batch).
    pub fn writes(&self) -> u64 {
        self.writes.load(Relaxed)
    }

    /// The bucket files' bytes together, as far as this cache knows.
    pub fn bytes(&self) -> u64 {
        self.total.load(Relaxed)
    }

    fn add_bytes(&self, before: u64, after: u64) {
        if after >= before {
            self.total.fetch_add(after - before, Relaxed);
        } else {
            let _ = self
                .total
                .try_update(Relaxed, Relaxed, |t| Some(t.saturating_sub(before - after)));
        }
    }

    fn with_id<T>(
        &self,
        id: BucketId,
        create: bool,
        f: impl FnOnce(&mut Bucket) -> Option<T>,
    ) -> Option<T> {
        let mut guard = self.buckets.lock().ok()?;
        let buckets = &mut *guard;
        buckets.tick += 1;
        let tick = buckets.tick;
        let open = match buckets.map.get(&id) {
            Some(Some(_)) => false,
            Some(None) => create,
            None => true,
        };
        let mut opened = false;
        if open {
            let path = self.dir.join(id.file());
            let bucket = if create || path.exists() {
                let before = fs::metadata(&path).map_or(0, |m| m.len());
                match Bucket::open(&path) {
                    Ok(bucket) => {
                        self.add_bytes(before, bucket.end);
                        opened = true;
                        Some(bucket)
                    }
                    Err(e) => {
                        log::warn!("Horizon: cube far terrain {}: {e}", path.display());
                        None
                    }
                }
            } else {
                None
            };
            buckets.map.insert(id, bucket);
        }
        let bucket = buckets.map.get_mut(&id)?.as_mut()?;
        bucket.used = tick;
        let reopened = bucket.file.is_none();
        let before = bucket.end;
        let out = f(bucket);
        self.add_bytes(before, bucket.end);
        if opened || reopened {
            buckets.trim();
        }
        out
    }

    fn with_bucket<T>(
        &self,
        key: CubeKey,
        create: bool,
        f: impl FnOnce(&mut Bucket) -> Option<T>,
    ) -> Option<T> {
        self.with_id(BucketId::of(key), create, f)
    }

    /// Whether `key` is stored.
    pub fn contains(&self, key: CubeKey) -> bool {
        key.level <= MAX_CACHED_LEVEL
            && self
                .with_bucket(key, false, |b| {
                    Some(matches!(b.index.get(&key), Some(Some(_))))
                })
                .unwrap_or(false)
    }

    /// The stored node `key`, if there is one and it is intact.
    pub fn get(&self, key: CubeKey) -> Option<CubeNode> {
        if key.level > MAX_CACHED_LEVEL {
            return None;
        }
        let body = self.with_bucket(key, false, |b| b.read(key))?;
        decode(key, &body)
    }

    /// Stores `node` (levels 0 to [`MAX_CACHED_LEVEL`]), replacing any older.
    pub fn put(&self, node: &CubeNode) {
        self.put_many(std::slice::from_ref(node));
    }

    /// Stores `nodes`, a later one of a key replacing an earlier: grouped by
    /// bucket, each bucket's records appended in one write. Encoding is done
    /// before the cache is locked. Then the folder is brought under its cap.
    pub fn put_many(&self, nodes: &[CubeNode]) {
        let mut groups: FxHashMap<BucketId, Vec<(CubeKey, Vec<u8>)>> = FxHashMap::default();
        for node in nodes {
            if node.key.level <= MAX_CACHED_LEVEL {
                groups
                    .entry(BucketId::of(node.key))
                    .or_default()
                    .push((node.key, encode(node)));
            }
        }
        for (id, items) in groups {
            let records: Vec<(u8, CubeKey, &[u8])> = items
                .iter()
                .map(|(k, b)| (NODE, *k, b.as_slice()))
                .collect();
            self.with_id(id, true, |b| {
                match b.append_many(&records) {
                    Ok(()) => {
                        self.writes.fetch_add(1, Relaxed);
                    }
                    Err(e) => log::warn!("Horizon: keeping cube far terrain: {e}"),
                }
                Some(())
            });
        }
        self.enforce();
    }

    /// Every bucket file on disk: modification time, bucket, bytes.
    fn files(&self) -> Vec<(SystemTime, BucketId, u64)> {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        entries
            .flatten()
            .filter_map(|e| {
                let id = BucketId::parse(e.file_name().to_str()?)?;
                let meta = e.metadata().ok()?;
                let when = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                Some((when, id, meta.len()))
            })
            .collect()
    }

    /// Brings the folder under its cap: when the bucket files together take
    /// more, whole files with the oldest modification time (the least
    /// recently opened or written) are deleted until they take 90% of it.
    fn enforce(&self) {
        if self.cap == 0 || self.total.load(Relaxed) <= self.cap {
            return;
        }
        let Ok(mut guard) = self.buckets.lock() else {
            return;
        };
        let mut files = self.files();
        let mut total: u64 = files.iter().map(|f| f.2).sum();
        if total > self.cap {
            files.sort_unstable();
            let target = self.cap - self.cap / 10;
            let mut dropped = 0usize;
            for (_, id, len) in files {
                if total <= target {
                    break;
                }
                if fs::remove_file(self.dir.join(id.file())).is_ok() {
                    guard.map.remove(&id);
                    total -= len;
                    dropped += 1;
                }
            }
            log::info!(
                "Horizon: cube far terrain cache over its {} MB: dropped {dropped} least recently used files",
                self.cap >> 20
            );
        }
        self.total.store(total, Relaxed);
    }

    /// Cube `cube` was saved again (or for the first time): erases every
    /// stored node over it, at every level from 0, and tells the samplers to
    /// drop theirs ([`Self::changes_since`]). Cheap: at most one lookup per
    /// level, and an appended record only for a node that is stored. The save
    /// calls this from any thread, when it queues the cube's write.
    pub fn invalidate(&self, cube: SectionPos) {
        for level in 0..=MAX_CACHED_LEVEL {
            let key = CubeKey::containing(level, cube);
            self.with_bucket(key, false, |b| {
                if matches!(b.index.get(&key), Some(Some(_)))
                    && let Err(e) = b.append_many(&[(ERASE, key, &[])])
                {
                    log::warn!("Horizon: dropping cube far terrain: {e}");
                }
                Some(())
            });
        }
        let Ok(mut log) = self.changes.lock() else {
            return;
        };
        let at = log.next;
        log.next += 1;
        log.entries.push_back((at, cube));
        if log.entries.len() > LOG {
            log.entries.pop_front();
        }
    }

    /// Where [`Self::changes_since`] starts for a new reader: now.
    pub fn cursor(&self) -> u64 {
        self.changes.lock().map_or(0, |log| log.next)
    }

    /// The cubes invalidated since `cursor`, which then moves to now.
    pub fn changes_since(&self, cursor: &mut u64) -> Changes {
        let Ok(log) = self.changes.lock() else {
            return Changes::All;
        };
        if *cursor == log.next {
            return Changes::None;
        }
        let first = log.entries.front().map_or(log.next, |e| e.0);
        let out = if *cursor < first {
            Changes::All
        } else {
            Changes::Cubes(
                log.entries
                    .iter()
                    .filter(|e| e.0 >= *cursor)
                    .map(|e| e.1)
                    .collect(),
            )
        };
        *cursor = log.next;
        out
    }
}

/// What the cache depends on besides the node data: the format, this build
/// (the generator it samples), the world's seed and the block registry.
pub fn fingerprint(seed: i64, block_states: usize) -> String {
    format!(
        "horizon-cube-lod {FORMAT}\nversion {}\nexe {}\nseed {seed}\nblock_states {block_states}\n",
        env!("CARGO_PKG_VERSION"),
        super::bake::exe_stamp(),
    )
}

fn encode(node: &CubeNode) -> Vec<u8> {
    let mut out = Vec::new();
    let uniform = node.as_uniform();
    let cover = node.covered_mask();
    let tops = node.fluid_tops();
    out.push(
        u8::from(uniform.is_some())
            | u8::from(cover.is_some()) << 1
            | u8::from(tops.is_some()) << 2,
    );
    if let Some(mask) = cover {
        for word in mask {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    if let Some(tops) = tops {
        for h in tops {
            out.extend_from_slice(&h.to_le_bytes());
        }
    }
    match uniform {
        Some(v) => {
            out.extend_from_slice(&v.state.0.to_le_bytes());
            out.extend_from_slice(&v.biome.to_le_bytes());
        }
        None => {
            let mut raw = Vec::with_capacity(RAW);
            for i in 0..VOLUME {
                raw.extend_from_slice(&node.at(i).state.0.to_le_bytes());
            }
            for i in 0..VOLUME {
                raw.extend_from_slice(&node.at(i).biome.to_le_bytes());
            }
            let mut encoder = ZlibEncoder::new(out, Compression::fast());
            let _ = encoder.write_all(&raw);
            return encoder.finish().unwrap_or_default();
        }
    }
    out
}

fn decode(key: CubeKey, body: &[u8]) -> Option<CubeNode> {
    let (&flags, mut rest) = body.split_first()?;
    let mut cover = None;
    if flags & 2 != 0 {
        let mut mask: CoverMask = [0; 4];
        let (words, tail) = rest.split_at_checked(32)?;
        for (word, bytes) in mask.iter_mut().zip(words.as_chunks::<8>().0) {
            *word = u64::from_le_bytes(*bytes);
        }
        cover = Some(Box::new(mask));
        rest = tail;
    }
    let mut tops = None;
    if flags & 4 != 0 {
        let mut heights: FluidTops = [NO_FLUID_TOP; 256];
        let (bytes, tail) = rest.split_at_checked(512)?;
        for (h, pair) in heights.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *h = u16::from_le_bytes(*pair);
        }
        tops = Some(Box::new(heights));
        rest = tail;
    }
    let node = if flags & 1 != 0 {
        let state = u32::from_le_bytes(rest.get(..4)?.try_into().ok()?);
        let biome = u16::from_le_bytes(rest.get(4..6)?.try_into().ok()?);
        CubeNode::uniform(
            key,
            Voxel {
                state: BlockStateId(state),
                biome,
            },
        )
    } else {
        let mut raw = Vec::with_capacity(RAW);
        ZlibDecoder::new(rest).read_to_end(&mut raw).ok()?;
        if raw.len() != RAW {
            return None;
        }
        let (states, biomes) = raw.split_at(VOLUME * 4);
        let mut dense = Box::new(
            [Voxel {
                state: BlockStateId(0),
                biome: 0,
            }; VOLUME],
        );
        for (i, voxel) in dense.iter_mut().enumerate() {
            voxel.state = BlockStateId(u32::from_le_bytes(
                states[i * 4..i * 4 + 4].try_into().ok()?,
            ));
            voxel.biome = u16::from_le_bytes(biomes[i * 2..i * 2 + 2].try_into().ok()?);
        }
        CubeNode::from_dense(key, dense)
    };
    Some(node.with_covered(cover).with_fluid_tops(tops))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod::cube::{Kind, downsample};
    use std::time::Duration;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("horizon-cube-lod-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }
    fn dense(key: CubeKey, salt: u32) -> CubeNode {
        CubeNode::from_fn(key, |x, y, z| Voxel {
            state: BlockStateId(((x + 2 * y + 3 * z) as u32 + salt) % 5),
            biome: (x ^ z) as u16,
        })
    }
    fn same(a: &CubeNode, b: &CubeNode) {
        assert_eq!(a.key, b.key);
        assert_eq!(a.covered_mask(), b.covered_mask());
        assert_eq!(a.fluid_tops(), b.fluid_tops());
        for i in 0..VOLUME {
            assert_eq!(a.at(i), b.at(i), "voxel {i}");
        }
    }
    fn file_of(dir: &Path, key: CubeKey) -> PathBuf {
        dir.join(BucketId::of(key).file())
    }
    fn bytes_of(dir: &Path) -> u64 {
        fs::read_dir(dir)
            .unwrap()
            .flatten()
            .filter(|e| BucketId::parse(e.file_name().to_str().unwrap()).is_some())
            .map(|e| e.metadata().unwrap().len())
            .sum()
    }
    /// Sets the modification time of `key`'s bucket file to `secs` ago.
    fn age(dir: &Path, key: CubeKey, secs: u64) {
        let file = fs::OpenOptions::new()
            .write(true)
            .open(file_of(dir, key))
            .unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(secs))
            .unwrap();
    }

    #[test]
    fn nodes_round_trip_across_sessions_at_every_level_with_cover_and_fluid_tops() {
        let dir = scratch("round");
        let mut mask = Box::new([0u64; 4]);
        mask[1] = 0xdead_beef;
        let mut tops = Box::new([NO_FLUID_TOP; 256]);
        tops[17] = 4096;
        tops[255] = 17;
        // Level 0 (a generated cube), a middle level and a high one (a far view
        // of 1,024 chunks is level 10), each with cover and fluid tops.
        let keys = [
            CubeKey::new(0, -7, 3, 9),
            CubeKey::new(2, -3, 5, 9),
            CubeKey::new(12, 1, -1, 0),
            CubeKey::new(MAX_CACHED_LEVEL, 0, 0, -1),
        ];
        let nodes: Vec<CubeNode> = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                dense(k, i as u32)
                    .with_covered(Some(mask.clone()))
                    .with_fluid_tops(Some(tops.clone()))
            })
            .collect();
        let uniform_key = CubeKey::new(3, 0, -1, 0);
        let uniform = CubeNode::uniform(
            uniform_key,
            Voxel {
                state: BlockStateId(7),
                biome: 3,
            },
        );
        {
            let cache = CubeLodCache::open(&dir, "a").unwrap();
            assert!(cache.get(keys[0]).is_none());
            for node in &nodes {
                cache.put(node);
            }
            cache.put(&uniform);
            for node in &nodes {
                assert!(cache.contains(node.key));
                same(&cache.get(node.key).unwrap(), node);
            }
            // Deeper than the cache goes: refused.
            let too_deep = CubeKey::new(MAX_CACHED_LEVEL + 1, 0, 0, 0);
            cache.put(&dense(too_deep, 0));
            assert!(!cache.contains(too_deep) && cache.get(too_deep).is_none());
        }
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        for node in &nodes {
            let got = cache.get(node.key).unwrap();
            same(&got, node);
            assert!(got.covered_mask().is_some() && got.fluid_tops().is_some());
        }
        same(&cache.get(uniform_key).unwrap(), &uniform);
        assert!(cache.get(CubeKey::new(2, -3, 5, 8)).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    /// An Earth sea node (all water, its tops on the node's top) round-trips, and a parent of eight stored
    /// children keeps the tops, as `of_cached_children` / the writer's `keep_into` build it.
    #[test]
    fn a_sea_node_keeps_its_water_tops_in_the_cache_and_so_does_a_parent_of_cached_children() {
        let dir = scratch("sea-tops");
        let water = Voxel {
            state: BlockStateId(9),
            biome: 4,
        };
        let sea_top = |key: CubeKey, wet: bool| {
            let mut tops = Box::new([NO_FLUID_TOP; 256]);
            if wet {
                tops.iter_mut().for_each(|t| *t = 4096);
            }
            CubeNode::uniform(key, water).with_fluid_tops(Some(tops))
        };
        let parent = CubeKey::new(3, -4, -1, 2);
        // The upper four end the water on their tops, the lower four are water going on.
        let children: Vec<CubeNode> = parent
            .children()
            .iter()
            .map(|&c| sea_top(c, c.y > 2 * parent.y))
            .collect();
        {
            let cache = CubeLodCache::open(&dir, "a").unwrap();
            for child in &children {
                cache.put(child);
            }
        }
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        let mut got = Vec::new();
        for child in &children {
            let node = cache.get(child.key).unwrap();
            same(&node, child);
            assert_eq!(node.fluid_tops().is_some(), child.key.y > 2 * parent.y);
            got.push(node);
        }
        let refs: [&CubeNode; 8] = std::array::from_fn(|i| &got[i]);
        let air = BlockStateId(0);
        let merged = downsample(refs, air, |s| {
            if s == air { Kind::Air } else { Kind::Fluid }
        });
        let tops = merged.fluid_tops().expect("the parent of cached children has tops");
        assert!(tops.iter().all(|&h| h == 4096));
        // And the parent put and read back keeps them.
        cache.put(&merged);
        same(&cache.get(parent).unwrap(), &merged);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_saved_cube_erases_the_nodes_over_it_at_every_level_and_only_those() {
        let dir = scratch("invalidate");
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        let cube = SectionPos::new(-5, 3, 20);
        let over: Vec<CubeKey> = (0..=MAX_CACHED_LEVEL)
            .map(|l| CubeKey::containing(l, cube))
            .collect();
        assert_eq!(over[0], CubeKey::new(0, -5, 3, 20));
        // Beside it: a neighbour cube, a level 1 node away, a coarse node
        // that does not hold the cube.
        let beside = [
            CubeKey::new(0, -5, 3, 21),
            CubeKey::new(1, 100, 100, 100),
            CubeKey::new(5, 9, 9, 9),
        ];
        for &key in over.iter().chain(&beside) {
            cache.put(&dense(key, 0));
        }
        let mut cursor = cache.cursor();
        cache.invalidate(cube);
        for &key in &over {
            assert!(!cache.contains(key), "{key:?} survived");
            assert!(cache.get(key).is_none());
        }
        for &key in &beside {
            assert!(cache.contains(key), "{key:?} was erased");
        }
        assert_eq!(cache.changes_since(&mut cursor), Changes::Cubes(vec![cube]));
        assert_eq!(cache.changes_since(&mut cursor), Changes::None);
        drop(cache);
        // The erasure is on disk too.
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        assert!(over.iter().all(|&k| !cache.contains(k)));
        assert!(beside.iter().all(|&k| cache.contains(k)));
        // A node stored after the save is kept.
        cache.put(&dense(over[0], 4));
        assert!(cache.contains(over[0]));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_changed_fingerprint_empties_the_folder() {
        let dir = scratch("fingerprint");
        let key = CubeKey::new(1, 0, 0, 0);
        let cache = CubeLodCache::open(&dir, "build 1").unwrap();
        cache.put(&dense(key, 0));
        drop(cache);
        let cache = CubeLodCache::open(&dir, "build 1").unwrap();
        assert!(cache.contains(key));
        drop(cache);
        let cache = CubeLodCache::open(&dir, "build 2").unwrap();
        assert!(!cache.contains(key));
        assert!(cache.get(key).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_last_record_is_cut_off_and_damage_reads_as_missing() {
        let dir = scratch("torn");
        let (a, b) = (CubeKey::new(1, 0, 0, 0), CubeKey::new(1, 1, 0, 0));
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        cache.put(&dense(a, 0));
        cache.put(&dense(b, 1));
        drop(cache);
        let file = dir.join(BucketId::of(a).file());
        let bytes = fs::read(&file).unwrap();
        // The last record loses its end.
        fs::write(&file, &bytes[..bytes.len() - 10]).unwrap();
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        assert!(cache.get(a).is_some());
        assert!(cache.get(b).is_none());
        // Appending after the cut works.
        cache.put(&dense(b, 1));
        assert!(cache.get(b).is_some());
        drop(cache);
        // A flipped byte inside a body fails the checksum.
        let mut bytes = fs::read(&file).unwrap();
        bytes[30] ^= 0xff;
        fs::write(&file, &bytes).unwrap();
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        assert!(cache.get(a).is_none());
        assert!(cache.get(b).is_some());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn so_many_changes_that_the_log_overflows_say_all() {
        let dir = scratch("log");
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        let mut cursor = cache.cursor();
        for i in 0..(LOG as i32 + 10) {
            cache.invalidate(SectionPos::new(i, 0, 0));
        }
        assert_eq!(cache.changes_since(&mut cursor), Changes::All);
        cache.invalidate(SectionPos::new(1, 2, 3));
        assert_eq!(
            cache.changes_since(&mut cursor),
            Changes::Cubes(vec![SectionPos::new(1, 2, 3)])
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_batch_is_the_same_records_as_single_puts_in_one_write_per_bucket() {
        let (batched, single) = (scratch("batch-a"), scratch("batch-b"));
        // Ten nodes in one bucket, three in another, and a newer one of a key
        // in the same batch (it wins).
        let mut nodes: Vec<CubeNode> = (0..10)
            .map(|x| dense(CubeKey::new(3, x % 8, 0, 0), x as u32))
            .collect();
        nodes.extend((0..3).map(|x| dense(CubeKey::new(3, 8 + x, 0, 0), 1)));
        nodes.push(dense(CubeKey::new(3, 0, 0, 0), 99));
        let a = CubeLodCache::open(&batched, "a").unwrap();
        a.put_many(&nodes);
        assert_eq!(a.writes(), 2, "one write per bucket");
        let b = CubeLodCache::open(&single, "a").unwrap();
        for node in &nodes {
            b.put(node);
        }
        assert_eq!(b.writes(), nodes.len() as u64);
        assert_eq!(a.bytes(), b.bytes());
        // Same files, byte for byte (the records of a bucket in the same order).
        for key in [CubeKey::new(3, 0, 0, 0), CubeKey::new(3, 8, 0, 0)] {
            assert_eq!(
                fs::read(file_of(&batched, key)).unwrap(),
                fs::read(file_of(&single, key)).unwrap()
            );
        }
        drop((a, b));
        let a = CubeLodCache::open(&batched, "a").unwrap();
        let mut latest = FxHashMap::default();
        for node in &nodes {
            latest.insert(node.key, node);
        }
        assert_eq!(latest.len(), 11);
        for (key, node) in latest {
            same(&a.get(key).unwrap(), node);
        }
        // An empty batch is nothing.
        a.put_many(&[]);
        let _ = fs::remove_dir_all(&batched);
        let _ = fs::remove_dir_all(&single);
    }

    #[test]
    fn a_newer_record_replaces_the_older_across_sessions() {
        let dir = scratch("late");
        let key = CubeKey::new(4, 2, 0, -3);
        let (coarse, whole) = (dense(key, 1), dense(key, 3));
        {
            let cache = CubeLodCache::open(&dir, "a").unwrap();
            cache.put(&coarse);
            same(&cache.get(key).unwrap(), &coarse);
            cache.put(&whole);
            same(&cache.get(key).unwrap(), &whole);
        }
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        same(&cache.get(key).unwrap(), &whole);
        // And again, after a reopen, in a batch.
        let later = dense(key, 4);
        cache.put_many(&[dense(key, 2), later.clone()]);
        drop(cache);
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        same(&cache.get(key).unwrap(), &later);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_least_recently_used_bucket_files_go_first_when_over_the_cap() {
        let dir = scratch("evict");
        let keys: Vec<CubeKey> = (0..4).map(|i| CubeKey::new(2, i * 8, 0, 0)).collect();
        {
            let cache = CubeLodCache::open_with_cap(&dir, "a", 0).unwrap();
            for &key in &keys[..3] {
                cache.put(&dense(key, 1));
            }
        }
        // Equal records: equal files.
        let size = fs::metadata(file_of(&dir, keys[0])).unwrap().len();
        assert!(
            keys[..3]
                .iter()
                .all(|&k| fs::metadata(file_of(&dir, k)).unwrap().len() == size)
        );
        age(&dir, keys[0], 3000);
        age(&dir, keys[1], 2000);
        age(&dir, keys[2], 1000);
        // Room for the three and half of a fourth: nothing goes on open.
        let cap = 3 * size + size / 2;
        let cache = CubeLodCache::open_with_cap(&dir, "a", cap).unwrap();
        assert_eq!(bytes_of(&dir), 3 * size);
        assert_eq!(cache.bytes(), 3 * size);
        // The oldest file is opened in this session: now the newest.
        assert!(cache.get(keys[0]).is_some());
        // A fourth file takes the folder over the cap: the oldest left (the
        // second) goes, whole, and the folder is under 90% of the cap.
        cache.put(&dense(keys[3], 1));
        assert!(!file_of(&dir, keys[1]).exists());
        for &k in &[keys[0], keys[2], keys[3]] {
            assert!(file_of(&dir, k).exists(), "{k:?}");
        }
        assert_eq!(bytes_of(&dir), 3 * size);
        assert!(bytes_of(&dir) <= cap - cap / 10);
        assert_eq!(cache.bytes(), bytes_of(&dir));
        assert!(cache.get(keys[1]).is_none());
        same(&cache.get(keys[0]).unwrap(), &dense(keys[0], 1));
        // The deleted bucket takes new records again (and the folder is over
        // its cap again: the third file, the oldest now, goes).
        cache.put(&dense(keys[1], 1));
        same(&cache.get(keys[1]).unwrap(), &dense(keys[1], 1));
        assert!(!file_of(&dir, keys[2]).exists());
        assert_eq!(bytes_of(&dir), 3 * size);
        drop(cache);
        // Opened with a smaller cap, the oldest go at once.
        age(&dir, keys[0], 500);
        age(&dir, keys[1], 400);
        age(&dir, keys[3], 300);
        let cache = CubeLodCache::open_with_cap(&dir, "a", 2 * size + size / 2).unwrap();
        assert_eq!(bytes_of(&dir), 2 * size);
        assert!(!file_of(&dir, keys[0]).exists());
        assert!(cache.contains(keys[1]) && cache.contains(keys[3]));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn many_batches_stay_under_the_cap_and_no_cap_keeps_everything() {
        let dir = scratch("cap");
        let one = {
            let cache = CubeLodCache::open_with_cap(&dir, "probe", 0).unwrap();
            cache.put(&dense(CubeKey::new(1, 0, 0, 0), 0));
            cache.bytes()
        };
        let cap = 10 * one;
        let cache = CubeLodCache::open_with_cap(&dir, "cap", cap).unwrap();
        for batch in 0..20 {
            let nodes: Vec<CubeNode> = (0..5)
                .map(|i| dense(CubeKey::new(1, (batch * 5 + i) * 8, 0, 0), 0))
                .collect();
            cache.put_many(&nodes);
            assert!(bytes_of(&dir) <= cap, "batch {batch}: {}", bytes_of(&dir));
            assert_eq!(cache.bytes(), bytes_of(&dir));
        }
        // The newest batch is there.
        assert!(cache.contains(CubeKey::new(1, 99 * 8, 0, 0)));
        drop(cache);
        let dir2 = scratch("nocap");
        let cache = CubeLodCache::open_with_cap(&dir2, "a", 0).unwrap();
        for i in 0..30 {
            cache.put(&dense(CubeKey::new(1, i * 8, 0, 0), 0));
        }
        assert_eq!(bytes_of(&dir2), 30 * one);
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&dir2);
    }

    #[test]
    fn bucket_file_names_parse_back_and_nothing_else_does() {
        assert_eq!(DEFAULT_CAP_MB, 1024);
        assert_eq!(
            BucketId::parse("n3.-1.0.12.bin")
                .map(|b| b.file())
                .as_deref(),
            Some("n3.-1.0.12.bin")
        );
        assert!(BucketId::parse("fingerprint.txt").is_none());
        assert!(BucketId::parse("n3.1.2.bin").is_none());
        assert!(BucketId::parse("n3.1.2.3.4.bin").is_none());
        assert!(BucketId::parse("n3.1.2.3.tmp").is_none());
    }

    #[test]
    fn far_more_buckets_than_open_files_still_read_and_write() {
        let dir = scratch("open");
        let cache = CubeLodCache::open_with_cap(&dir, "a", 0).unwrap();
        let count = MAX_OPEN as i32 * 2 + 20;
        let keys: Vec<CubeKey> = (0..count).map(|i| CubeKey::new(2, i * 8, 0, 0)).collect();
        for &key in &keys {
            cache.put(&dense(key, 1));
        }
        let open = |c: &CubeLodCache| {
            let b = c.buckets.lock().unwrap();
            b.map
                .values()
                .flatten()
                .filter(|b| b.file.is_some())
                .count()
        };
        assert!(open(&cache) <= MAX_OPEN, "{}", open(&cache));
        // Closed ones are opened again on use, and take appends.
        for &key in &keys {
            same(&cache.get(key).unwrap(), &dense(key, 1));
        }
        cache.put(&dense(keys[0], 2));
        same(&cache.get(keys[0]).unwrap(), &dense(keys[0], 2));
        assert!(open(&cache) <= MAX_OPEN);
        drop(cache);
        let cache = CubeLodCache::open_with_cap(&dir, "a", 0).unwrap();
        same(&cache.get(keys[0]).unwrap(), &dense(keys[0], 2));
        same(&cache.get(keys[1]).unwrap(), &dense(keys[1], 1));
        let _ = fs::remove_dir_all(&dir);
    }
}

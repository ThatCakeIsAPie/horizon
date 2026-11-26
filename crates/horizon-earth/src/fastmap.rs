//! A fast hasher for the maps the far view reads hundreds of times a node (tile stores, river bins, biome memo).
//!
//! The std default (SipHash, randomly keyed) costs more than the lookups it guards when the keys are a few
//! integers. The keys here are never attacker-chosen, so a multiplicative hash does. Only lookups ever depend on
//! it: no code iterates these maps in a way that decides an output.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0x517c_c1b7_2722_0a95;

/// A multiplicative hasher (the one rustc uses) with the entropy rotated into the low bits.
#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl FastHasher {
    #[inline]
    fn add(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(K);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf));
        }
    }
    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.add(u64::from(v));
    }
    #[inline]
    fn write_u16(&mut self, v: u16) {
        self.add(u64::from(v));
    }
    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.add(u64::from(v));
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }
    #[inline]
    fn write_usize(&mut self, v: usize) {
        self.add(v as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

/// `BuildHasher` of [`FastHasher`].
pub type FastBuild = BuildHasherDefault<FastHasher>;
/// A `HashMap` over [`FastHasher`].
pub type FastMap<K, V> = HashMap<K, V, FastBuild>;
/// A `HashSet` over [`FastHasher`].
pub type FastSet<K> = HashSet<K, FastBuild>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn behaves_as_a_map() {
        let mut m: FastMap<(i32, i32), u32> = FastMap::default();
        for x in -50..50 {
            for z in -50..50 {
                m.insert((x, z), (x * 1000 + z) as u32);
            }
        }
        assert_eq!(m.len(), 10_000);
        for x in -50..50 {
            for z in -50..50 {
                assert_eq!(m[&(x, z)], (x * 1000 + z) as u32);
            }
        }
    }
}

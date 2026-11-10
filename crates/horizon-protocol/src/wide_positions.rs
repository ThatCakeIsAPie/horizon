//! Wide positions (arismoko/horizon#1903): a level marked Horizon-only may one
//! day hold block positions that vanilla's packed forms cannot carry, so only a
//! connection that agreed on them may enter it. The agreement is made per
//! connection in the configuration phase, beside [`crate::mod_list`].
//!
//! A Horizon client sends its [`VERSION`] on [`CHANNEL`] once the server's brand
//! says Horizon; the server answers with the same payload only when it
//! [`accepts`] that version, and the connection has the capability from then
//! on. A vanilla client never hears the channel, and a client whose version the
//! server does not accept is treated as one.
use crate::buf::{Reader, Writer};

/// The custom payload channel, both ways.
pub const CHANNEL: &str = "horizon:wide_positions";
/// The forms this build reads and writes. Raised whenever they change.
pub const VERSION: i32 = 1;

/// `version` as a payload.
pub fn encode(version: i32) -> Vec<u8> {
    let mut w = Writer::new();
    w.varint(version);
    w.data
}

/// A payload's version; None for one that isn't exactly a version.
pub fn decode(data: &[u8]) -> Option<i32> {
    let mut r = Reader::new(data);
    let version = r.varint().ok()?;
    (r.remaining() == 0).then_some(version)
}

/// Whether this build agrees on the forms of a peer that sent `version`.
pub fn accepts(version: i32) -> bool {
    version == VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_lone_version_is_read() {
        assert_eq!(decode(&encode(VERSION)), Some(VERSION));
        assert_eq!(decode(&[]), None);
        let mut trailing = encode(VERSION);
        trailing.push(0);
        assert_eq!(decode(&trailing), None);
    }
}

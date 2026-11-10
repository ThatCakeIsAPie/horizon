//! The default tile address of Earth worlds. The engine has no built-in address of an outside
//! service (docs/layers.md#built-in-addresses), so the official executable registers this hook and
//! Create World's Earth page asks for the address through it. Without it (horizon-client's own
//! executable) the page's address box starts empty.
#![forbid(unsafe_code)]

use horizon_platform::hooks::Hooks;

/// Offers [`horizon_addresses::EARTH_TILES`].
pub static HOOKS: Source = Source;

pub struct Source;

impl Hooks for Source {
    fn earth_tile_address(&self) -> Option<&'static str> {
        Some(horizon_addresses::EARTH_TILES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_offered_address_is_a_template_over_https() {
        let address = HOOKS.earth_tile_address().unwrap();
        assert!(address.starts_with("https://"));
        for part in ["{z}", "{x}", "{y}"] {
            assert!(address.contains(part));
        }
    }
}

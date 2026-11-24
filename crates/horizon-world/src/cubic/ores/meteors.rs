//! Ore meteors. (CONTRACT STUB: Part D replaces the bodies; the public
//! signatures stay.)

use horizon_core::position::SectionPos;

use super::settings::{MeteorKind, MeteorSettings};
use crate::cubic::custom::populate::CubeWorld;
use crate::cubic::custom::settings::UserFunction;
use crate::registry::BlockRegistry;

impl Default for MeteorSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            per_square_km: 4.0,
            min_diameter: 5,
            max_diameter: 25,
            size_curve: UserFunction::new(&[(5.0, 0.6), (7.0, 1.0), (12.0, 0.35), (25.0, 0.03)]),
            size_floor: 0.02,
            surface_share: 0.6,
            sunk_share: 0.25,
            buried_share: 0.15,
            buried_depth: 50.0,
            mix_share: 0.2,
            kinds: vec![MeteorKind::default()],
        }
    }
}

impl MeteorSettings {
    /// Checks the numbers, naming the setting that is wrong.
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

/// The compiled meteors of one world.
pub struct Meteors {
    _private: (),
}

impl Meteors {
    pub fn new(
        _world_seed: i64,
        _s: &MeteorSettings,
        _reg: &BlockRegistry,
    ) -> Result<Self, String> {
        Ok(Self { _private: () })
    }

    /// Whether any meteor can generate (enabled, rate above 0, some kind).
    pub fn is_active(&self) -> bool {
        false
    }

    /// Places, whole, every meteor `cube` owns. All writes stay inside the
    /// 2 x 2 x 2 cubes from `cube` up (the population window). `ground(x, z)`
    /// is the generator's estimate of the Y of the top solid block of a
    /// column (the real surface on Earth). Uses its own seeded random, never
    /// the cube's population random. Call after the surface decoration.
    pub fn place(
        &self,
        _cube: SectionPos,
        _world: &mut dyn CubeWorld,
        _ground: &dyn Fn(i32, i32) -> i32,
    ) {
    }
}

//! Horizon-only levels (arismoko/horizon#1903): a level a mod registers with
//! `DimensionSettings::horizon_only` takes only the players whose client agreed
//! on wide positions (`horizon_protocol::wide_positions`). Anyone else is
//! refused, told why: a player saved there is disconnected as they join, a
//! portal, a teleport or a mod's respawn there leaves them where they are, and
//! a respawn point there sends them to the world spawn.
use super::{
    data::component_nbt,
    hub::{Hub, TickCtx},
};
use crate::{lang::Language, text::Text};

/// The refusal's translation key.
const KEY: &str = "horizon.level.horizon_only";

/// Why the level `id` refused a player: translated by a Horizon client, in
/// English for a vanilla one, which has no Horizon keys.
pub fn refusal(id: &str) -> Text {
    let english = Language::load(&[], crate::lang::DEFAULT);
    let fallback = english.get(KEY).expect("the English refusal");
    Text::translate_with_fallback(KEY, fallback, &[Text::literal(id)])
}

impl Hub {
    /// Places the guest `conn` as any player is (`PlayerList.placeNewPlayer`),
    /// then disconnects them when the level they were saved in refuses their
    /// client. Their record keeps that level: they come back to it with a
    /// client that may enter.
    pub fn place_joining_player(&mut self, conn: u32, ctx: &mut TickCtx) {
        self.place_new_player(conn, ctx);
        let Some(dimension) = self
            .guest(conn)
            .filter(|g| !g.may_enter(&g.dimension))
            .map(|g| g.dimension.clone())
        else {
            return;
        };
        self.kick(conn, component_nbt(&refusal(&dimension).component()));
    }
}

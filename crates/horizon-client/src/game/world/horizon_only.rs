//! Refusing a player a Horizon-only level at travel time. What makes a level
//! Horizon-only, the refusal text and the join-time check are in
//! `horizon_client::lan::horizon_only`.
use super::{Dim, Game, native_server::system_chat};
use horizon_client::lan::horizon_only::refusal;

impl Game<'_> {
    /// Whether the server player whose entity is `id` is refused the level
    /// `to`; a refused player is told why in their chat. False for any other
    /// entity.
    pub(super) fn refuse_level(&mut self, id: u32, to: Dim) -> bool {
        let Some(uuid) = self.lan.as_ref().and_then(|host| {
            host.hub
                .online_players()
                .find(|g| g.entity_id as u32 == id && !g.may_enter(to.id()))
                .map(|g| g.uuid)
        }) else {
            return false;
        };
        self.native_send(uuid, system_chat(&refusal(to.id())));
        true
    }
}

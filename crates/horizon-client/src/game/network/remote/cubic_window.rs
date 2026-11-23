//! A cubic level's window and view over a server connection (docs/cubic-chunks.md): the
//! `horizon:cubic_window` the server sets and moves, and the `horizon:cubic_view` the client asks.
use super::*;

impl Game<'_> {
    /// `horizon:cubic_window` (docs/cubic-chunks.md): the level the player is in is now the
    /// window of cubes from `window.min_y`. The level is laid out again in place
    /// (`World::relayout`): every loaded column keeps the sections both windows hold, with
    /// their meshes, and the sections that enter are air placeholders until the server sends
    /// the columns again (nearest first). A placeholder is not loaded for the cube far
    /// terrain, which draws there meanwhile. Entities, the player and the far terrain are
    /// untouched, and so are the attachments (they are keyed by column and block position).
    /// The chunk format changes, so a column still decoding for the old window is decoded
    /// again. Until the player's column is lit again the local player does not move
    /// (`tick_remote_player` waits for `Streaming::is_lit`).
    pub(in crate::game) fn remote_cubic_window(&mut self, window: horizon_protocol::cubic::Window) {
        let shape = horizon_client::chunk::LevelShape::cubic_window(
            self.world.shape,
            window.min_y,
            window.height,
        );
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        r.chunk_format = Arc::new(r.chunk_format.with_window(shape));
        let (old, new) = (self.world.layout(), shape.column());
        let mut air = [Arc::new(horizon_client::chunk::Section::default())];
        horizon_client::chunk::share_uniform(&mut air);
        let [air] = air;
        self.world.relayout(shape, |_, _, _| air.clone());
        self.streaming.shift_layout(old, new);
        // The sections that entered, per column, wait for the server's columns.
        use horizon_client::section_mask::SectionMask;
        let delta = (old.min_y() - new.min_y()) >> 4;
        let old_low = old.min_y() >> 4;
        let old_high = old_low + old.sections() as i32;
        let entering: SectionMask = (0..new.sections())
            .filter(|&i| {
                let y = (new.min_y() >> 4) + i as i32;
                !(old_low..old_high).contains(&y)
            })
            .collect();
        let keep = SectionMask::below(new.sections());
        r.cubic_placeholders.retain(|_, mask| {
            *mask = mask.shifted(delta) & keep;
            !mask.is_empty()
        });
        for &column in self.world.chunks.keys() {
            *r.cubic_placeholders.entry(column).or_default() |= entering;
        }
        r.cubic_placeholders.retain(|_, mask| !mask.is_empty());
        // Meshes waiting to upload are still right for their sections: moved with them.
        self.pending_uploads.retain_mut(|(key, _)| {
            key.1 += delta;
            (0..new.sections() as i32).contains(&key.1)
        });
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.shift_layout(new);
        }
        self.apply_settings();
    }

    /// `horizon:cubic_view`: once the client is in a cubic level, and whenever the Vertical
    /// Distance setting changes, the server is told how many cubes above and below the
    /// player it wants loaded.
    pub(super) fn send_cubic_view(&mut self) {
        if !self.world.shape.cubic {
            return;
        }
        let wanted = self.settings.horizon.cubic_vertical.clamp(
            horizon_client::graphics::CUBIC_VERTICAL_MIN,
            horizon_client::graphics::CUBIC_VERTICAL_MAX,
        ) as u8;
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        if r.cubic_vertical_sent == Some(wanted) {
            return;
        }
        r.cubic_vertical_sent = Some(wanted);
        self.send(S::CustomPayload(
            horizon_protocol::packets::configuration::CustomPayload {
                channel: horizon_protocol::cubic::VIEW_CHANNEL.into(),
                data: horizon_protocol::cubic::View { vertical: wanted }.encode(),
            },
        ));
    }
}

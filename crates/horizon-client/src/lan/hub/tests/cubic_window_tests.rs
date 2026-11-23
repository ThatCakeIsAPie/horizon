//! A cubic level's window and cubes on the wire: the window follows `Login` and precedes the
//! chunks, a slide sends only the entering cubes, and each vertical shard is sent its own.
use super::*;

/// What a player was sent, in order: `Login`, a cubic window as (min_y, height), or a
/// chunk column's position.
#[derive(Debug, PartialEq)]
enum Seen {
    Login,
    Window(i32, i32),
    Chunk(i32, i32),
    /// A `horizon:cubes` payload: its column and its cube Ys.
    Cubes(i32, i32, Vec<i32>),
}

fn seen(rx: &std::sync::mpsc::Receiver<crate::lan::Out>) -> Vec<Seen> {
    let mut out = Vec::new();
    let mut one = |body: &[u8]| {
        let mut r = horizon_protocol::buf::Reader::new(body);
        let id = r.varint().unwrap();
        match C::decode(id, r.rest()) {
            Ok(C::Login(_)) => out.push(Seen::Login),
            Ok(C::LevelChunkWithLight(p)) => out.push(Seen::Chunk(p.x, p.z)),
            Ok(C::CustomPayload(p)) if p.channel == horizon_protocol::cubic::WINDOW_CHANNEL => {
                let w = horizon_protocol::cubic::Window::decode(&p.data).unwrap();
                out.push(Seen::Window(w.min_y, w.height));
            }
            Ok(C::CustomPayload(p)) if p.channel == horizon_protocol::cubic::CUBES_CHANNEL => {
                let c = horizon_protocol::cubic::Cubes::decode(&p.data).unwrap();
                out.push(Seen::Cubes(
                    c.x,
                    c.z,
                    c.cubes.iter().map(|cube| cube.cy).collect(),
                ));
            }
            _ => {}
        }
    };
    while let Ok(o) = rx.try_recv() {
        match o {
            crate::lan::Out::Packet(b) => one(&b),
            crate::lan::Out::Packets(bs) => bs.iter().for_each(|b| one(b)),
            _ => {}
        }
    }
    out
}

/// A lit cubic world of the window's shape, with the columns around the origin.
fn cubic_world(shape: crate::chunk::LevelShape) -> World {
    let mut world = World::new(horizon_world::registry::BlockStateId(0), 0);
    world.shape = shape;
    for cz in -3..=3 {
        for cx in -3..=3 {
            let mut chunk = crate::chunk::Chunk::with_layout(cx, cz, shape.column());
            chunk.lit = true;
            world.chunks.insert((cx, cz), std::sync::Arc::new(chunk));
        }
    }
    world
}

/// A tick context over `world`, the player standing at `y`.
fn cubic_ctx<'a>(
    world: &'a mut World,
    next: &'a mut u32,
    border: &'a crate::worldborder::WorldBorder,
    y: f64,
) -> TickCtx<'a> {
    TickCtx {
        sea_level: 63,
        world,
        entities: &[],
        next_entity: next,
        spawn: [0.5, y, 0.5],
        spawn_angle: [0.0; 2],
        spawn_dimension: "minecraft:overworld",
        difficulty: 2,
        difficulty_locked: false,
        clock_rate: 1.0,
        tick_rate: 20.0,
        tick_frozen: false,
        tick_steps: Default::default(),
        collision: None,
        border,
        streaming: None,
    }
}

/// A player in a cubic level is told its window after the Login packet and before any
/// chunk; when the server slides the window it is told the new one and sent only the
/// cubes that entered, for the columns it holds, nearest first, and the chunks are not
/// sent again.
#[test]
fn a_cubic_levels_window_comes_before_its_chunks_and_then_only_the_entering_cubes() {
    use crate::chunk::LevelShape;
    let (mut hub, rx) = hub_with_a_guest();
    let first = LevelShape::cubic_window(LevelShape::OVERWORLD, 1_000_000, 512);
    let second = LevelShape::cubic_window(LevelShape::OVERWORLD, 1_000_256, 512);
    hub.cubic_window_changed("minecraft:overworld", first);
    assert!(hub.lan.shared.cubic() && horizon_protocol::buf::wide_positions());
    // The window of the (already placed) guest at that change; then a fresh placement.
    assert_eq!(seen(&rx), vec![Seen::Window(1_000_000, 512)]);
    let mut world = cubic_world(first);
    let border = crate::worldborder::WorldBorder::default();
    let mut next = 100;
    let mut ctx = cubic_ctx(&mut world, &mut next, &border, 1_000_100.0);
    hub.guests
        .update(0, |g| g.position = [0.5, 1_000_100.0, 0.5]);
    hub.place_new_player(1, &mut ctx);
    let mut stream = seen(&rx);
    for _ in 0..40 {
        hub.ticks += 1;
        hub.update_chunks(&ctx);
        stream.extend(seen(&rx));
        hub.guests.update(0, |g| g.unacknowledged = 0);
    }
    assert_eq!(
        &stream[..2],
        [Seen::Login, Seen::Window(1_000_000, 512)],
        "the window follows Login"
    );
    let chunks: Vec<_> = stream
        .iter()
        .filter_map(|s| match s {
            Seen::Chunk(x, z) => Some((*x, *z)),
            _ => None,
        })
        .collect();
    assert!(chunks.len() > 20, "the view is sent: {}", chunks.len());
    assert_eq!(chunks[0], (0, 0), "nearest first");
    assert!(
        !stream[2..].iter().any(|s| matches!(s, Seen::Window(..))),
        "one window, before the chunks"
    );
    assert!(
        !stream.iter().any(|s| matches!(s, Seen::Cubes(..))),
        "no cubes before a slide"
    );
    // The server slides the window by 16 cubes (the same level, in place).
    let mut moved = cubic_world(second);
    let ctx = cubic_ctx(&mut moved, &mut next, &border, 1_000_100.0);
    hub.cubic_window_changed("minecraft:overworld", second);
    let mut again = seen(&rx);
    assert_eq!(again, vec![Seen::Window(1_000_256, 512)]);
    for _ in 0..40 {
        hub.ticks += 1;
        hub.update_chunks(&ctx);
        again.extend(seen(&rx));
        hub.guests.update(0, |g| g.unacknowledged = 0);
    }
    assert!(
        !again.iter().any(|s| matches!(s, Seen::Chunk(..))),
        "the chunks are not sent again"
    );
    // Window 1 held cubes 62500..=62531, window 2 holds 62516..=62547: 62532..=62547
    // entered, in every column the player holds.
    let entering: Vec<i32> = (62_532..=62_547).collect();
    let cubes: Vec<_> = again
        .iter()
        .filter_map(|s| match s {
            Seen::Cubes(x, z, cys) => Some((*x, *z, cys.clone())),
            _ => None,
        })
        .collect();
    assert!(cubes.iter().all(|(_, _, cys)| *cys == entering));
    let mut sent: Vec<_> = cubes.iter().map(|(x, z, _)| (*x, *z)).collect();
    assert_eq!(sent[0], (0, 0), "nearest first");
    let near = |&(x, z): &(i32, i32)| x * x + z * z;
    assert!(
        sent.windows(2).all(|w| near(&w[0]) <= near(&w[1])),
        "by distance"
    );
    sent.sort_unstable();
    let mut held = chunks.clone();
    held.sort_unstable();
    assert_eq!(sent, held, "a payload for every column the player holds");
    // The encoder follows the new window: its sections start at the new floor.
    assert_eq!(hub.encoder.window_min_y(), 1_000_256);
    horizon_protocol::buf::set_wide_positions(false);
}

/// Two players of one cubic dimension, 20,000 blocks apart in height, are in two
/// vertical shards: each is sent its own shard's window, columns and entering cubes,
/// and nothing of the other's.
#[test]
fn each_player_of_a_cubic_dimension_is_sent_only_its_shards_window_and_cubes() {
    use crate::chunk::LevelShape;
    let overworld = "minecraft:overworld";
    let (mut hub, rx_low) = hub_with_a_guest();
    let (tx, rx_high) = std::sync::mpsc::channel();
    let mut carol = Guest::new(
        crate::lan::Joined {
            conn: 2,
            uuid: 2,
            name: "Carol".into(),
            properties: Vec::new(),
            info: horizon_protocol::packets::configuration::ClientInformation::new(8),
            brand: "vanilla".into(),
            sender: crate::lan::PlayerSender::new(tx, std::sync::Arc::default()),
            latency: std::sync::Arc::new(std::sync::atomic::AtomicI32::new(0)),
            address: "127.0.0.1:25566".parse().unwrap(),
            expected: false,
        },
        2,
        [0.5, 20_000.0, 0.5],
        8,
    );
    carol.placed = true;
    carol.dimension = overworld.into();
    carol.shard = 1;
    hub.guests.push(carol);
    hub.guests.update(0, |g| g.position = [0.5, 100.0, 0.5]);
    let low = LevelShape::cubic_window(LevelShape::OVERWORLD, 0, 304);
    let high = LevelShape::cubic_window(LevelShape::OVERWORLD, 19_856, 304);
    let border = crate::worldborder::WorldBorder::default();
    let mut next = 100;

    // The primary shard is in place: only Bob is in it.
    hub.set_level(overworld, 0, Some(low));
    hub.cubic_window_changed(overworld, low);
    assert_eq!(seen(&rx_low), vec![Seen::Window(0, 304)]);
    assert!(seen(&rx_high).is_empty(), "Carol is in another shard");
    let mut world = cubic_world(low);
    let mut ctx = cubic_ctx(&mut world, &mut next, &border, 100.0);
    hub.place_new_player(1, &mut ctx);
    let mut bob = seen(&rx_low);
    for _ in 0..40 {
        hub.ticks += 1;
        hub.update_chunks(&ctx);
        bob.extend(seen(&rx_low));
        hub.guests.update(0, |g| g.unacknowledged = 0);
    }
    assert!(bob.iter().any(|s| matches!(s, Seen::Chunk(..))));
    assert!(
        seen(&rx_high).is_empty(),
        "nothing of Bob's level reaches Carol"
    );

    // The upper shard in place: Carol's window, login and chunks; Bob hears none of it.
    hub.set_level(overworld, 1, Some(high));
    hub.cubic_window_changed(overworld, high);
    assert_eq!(seen(&rx_high), vec![Seen::Window(19_856, 304)]);
    let mut up = cubic_world(high);
    let mut ctx = cubic_ctx(&mut up, &mut next, &border, 20_000.0);
    hub.guests.update(1, |g| g.position = [0.5, 20_000.0, 0.5]);
    hub.place_new_player(2, &mut ctx);
    let mut carol_stream = seen(&rx_high);
    for _ in 0..40 {
        hub.ticks += 1;
        hub.update_chunks(&ctx);
        carol_stream.extend(seen(&rx_high));
        hub.guests.update(1, |g| g.unacknowledged = 0);
    }
    assert!(carol_stream.iter().any(|s| matches!(s, Seen::Chunk(..))));
    assert!(
        seen(&rx_low).is_empty(),
        "Bob is sent nothing of the other shard"
    );

    // The upper shard slides up a layer: Carol is told and sent the entering cube of
    // each column she holds; Bob, in the primary shard, is sent nothing.
    let up_one = LevelShape::cubic_window(LevelShape::OVERWORLD, 19_872, 304);
    let mut slid = cubic_world(up_one);
    let ctx = cubic_ctx(&mut slid, &mut next, &border, 20_000.0);
    hub.cubic_window_changed(overworld, up_one);
    let mut after = seen(&rx_high);
    assert_eq!(after, vec![Seen::Window(19_872, 304)]);
    for _ in 0..40 {
        hub.ticks += 1;
        hub.update_chunks(&ctx);
        after.extend(seen(&rx_high));
        hub.guests.update(1, |g| g.unacknowledged = 0);
    }
    let entering = 19_872 / 16 + 18;
    let cubes: Vec<_> = after
        .iter()
        .filter_map(|s| match s {
            Seen::Cubes(_, _, cys) => Some(cys.clone()),
            _ => None,
        })
        .collect();
    assert!(!cubes.is_empty());
    assert!(cubes.iter().all(|cys| *cys == [entering]), "{cubes:?}");
    assert!(
        seen(&rx_low).is_empty(),
        "a slide of the other shard reaches nobody in Bob's"
    );
    // Back in Bob's shard, a slide of the primary reaches Bob and not Carol.
    hub.set_level(overworld, 0, Some(low));
    let down = LevelShape::cubic_window(LevelShape::OVERWORLD, 16, 304);
    hub.cubic_window_changed(overworld, down);
    assert_eq!(seen(&rx_low), vec![Seen::Window(16, 304)]);
    assert!(seen(&rx_high).is_empty());
    horizon_protocol::buf::set_wide_positions(false);
}

/// The window sliding by a layer sends the layer; sliding the other way sends the one
/// that entered there; a column whose light has not settled waits for it.
#[test]
fn entering_cubes_wait_for_light_and_follow_each_slide() {
    use crate::chunk::LevelShape;
    let (mut hub, rx) = hub_with_a_guest();
    let first = LevelShape::cubic_window(LevelShape::OVERWORLD, 0, 304);
    hub.cubic_window_changed("minecraft:overworld", first);
    let _ = seen(&rx);
    let mut world = cubic_world(first);
    let border = crate::worldborder::WorldBorder::default();
    let mut next = 100;
    {
        let mut ctx = cubic_ctx(&mut world, &mut next, &border, 100.0);
        hub.guests.update(0, |g| g.position = [0.5, 100.0, 0.5]);
        hub.place_new_player(1, &mut ctx);
    }
    let ctx = cubic_ctx(&mut world, &mut next, &border, 100.0);
    for _ in 0..40 {
        hub.ticks += 1;
        hub.update_chunks(&ctx);
        hub.guests.update(0, |g| g.unacknowledged = 0);
    }
    let _ = seen(&rx);
    let columns = hub.guests[0].sent_chunks.len();
    assert!(columns > 20);
    // Up one layer.
    let up = LevelShape::cubic_window(LevelShape::OVERWORLD, 16, 304);
    let mut slid = cubic_world(up);
    slid.dirty_light.insert((0, 0));
    let ctx = cubic_ctx(&mut slid, &mut next, &border, 100.0);
    hub.cubic_window_changed("minecraft:overworld", up);
    let mut stream = seen(&rx);
    for _ in 0..60 {
        hub.ticks += 1;
        hub.update_chunks(&ctx);
        stream.extend(seen(&rx));
        hub.guests.update(0, |g| g.unacknowledged = 0);
    }
    let cubes: Vec<_> = stream
        .iter()
        .filter_map(|s| match s {
            Seen::Cubes(x, z, cys) => Some(((*x, *z), cys.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(cubes.len(), columns - 1, "the unlit column waits");
    assert!(
        cubes
            .iter()
            .all(|(column, cys)| *column != (0, 0) && *cys == [19])
    );
    // Its light settles: the column's cubes follow.
    slid.dirty_light.clear();
    let ctx = cubic_ctx(&mut slid, &mut next, &border, 100.0);
    hub.ticks += 1;
    hub.update_chunks(&ctx);
    assert_eq!(seen(&rx), vec![Seen::Cubes(0, 0, vec![19])]);
    // Down two layers: the cubes that entered below, nearest the player first.
    let down = LevelShape::cubic_window(LevelShape::OVERWORLD, -16, 304);
    let mut slid = cubic_world(down);
    let ctx = cubic_ctx(&mut slid, &mut next, &border, 100.0);
    hub.cubic_window_changed("minecraft:overworld", down);
    let mut stream = seen(&rx);
    for _ in 0..60 {
        hub.ticks += 1;
        hub.update_chunks(&ctx);
        stream.extend(seen(&rx));
        hub.guests.update(0, |g| g.unacknowledged = 0);
    }
    assert!(
        stream
            .iter()
            .any(|s| matches!(s, Seen::Cubes(0, 0, cys) if *cys == [0, -1]))
    );
    horizon_protocol::buf::set_wide_positions(false);
}

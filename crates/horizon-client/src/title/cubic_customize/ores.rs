//! The Ores page of the Customize screen: Cubic's two-tier ore generation
//! ([`OreGenSettings`], `"oreGen"` in the settings JSON) for Custom Cubic and
//! Hybrid, and for Earth, whose second page it is.
//!
//! The page lists the world's ore entries (block, vein size, biomes, tier 1's
//! rich zone and tier 2's background, each with its own switch), then the ore
//! meteors. Earth also has the richness map's switch and strength. A row is a
//! label and a control; there is no helper text.
//!
//! **What the page reads and writes.**
//! * Custom Cubic / Hybrid: `ore_gen` when the settings have it, else the old
//!   `standardOres` / `periodicGaussianOres` lists converted
//!   ([`OreGenSettings::from_legacy`]). Building the page never writes. The
//!   first edit writes `oreGen` and clears the two legacy lists, so the world
//!   keeps generating what the page showed.
//! * Earth: `ore_gen`; none means ores are off, and the page is the one switch.
//!   Turning it on fills [`OreGenSettings::earth_defaults`].
//!
//! The size spline of the meteors is edited as one row of sliders a point, not
//! a drag-the-curve editor.
use super::{
    Builder, Item, Mapping, State, Target, WIDTH, block_name, button, disabled, int, labelled,
    number, number_shown, on_off, show, toggle, tr,
};
use crate::cubic_settings::CubicKind;
use horizon_world::cubic::custom::CustomGeneratorSettings;
use horizon_world::cubic::custom::settings::{BlockStateDesc, UserFunctionPoint};
use horizon_world::cubic::ores::{
    Background, MeteorKind, MeteorSettings, OreEntry, OreGenSettings, RangeMode, RichMode,
    RichZone, Shape,
};

/// Blocks per vein.
const VEIN: (i32, i32) = (1, 64);
/// The block Y track of a static band: its low end is no lower limit (for a
/// minimum), its high end no upper one (for a maximum).
const Y_TRACK: (i32, i32) = (-512, 1024);
/// The depth track of a band under the ground, in blocks.
const DEPTH_TRACK: (i32, i32) = (0, 512);
/// The most veins per cube the intensity slider reaches; the settings take more.
const INTENSITY_MAX: f64 = 32.0;
/// The most veins per cube the background slider reaches.
const RATE_MAX: f64 = 4.0;
/// The size of a noise pocket: 2^2 to 2^9 blocks.
const NOISE_SCALE: Mapping = Mapping::PosExp { lo: 2.0, hi: 9.0 };
/// The seed offset slider's track.
const SEED: (i32, i32) = (-1000, 1000);
/// The richness strength slider's top.
const STRENGTH_MAX: f64 = 2.0;
/// The meteor diameters the sliders take (the settings allow 3 to 25).
const DIAMETER: (i32, i32) = (3, 25);
/// The most meteors per square kilometre the slider reaches.
const FREQUENCY_MAX: f64 = 32.0;
const BURIED_DEPTH: (i32, i32) = (0, 128);
const MIX_MAX: f64 = 0.5;
const KIND_WEIGHT_MAX: f64 = 10.0;
/// The ores a meteor kind picks from (short names, as `MeteorKind` has them).
const METEOR_ORES: [&str; 8] = [
    "iron", "copper", "coal", "gold", "redstone", "lapis", "diamond", "emerald",
];

fn key(name: &str) -> String {
    format!("horizon.cubic.oregen.{name}")
}
fn label(name: &str) -> String {
    tr(&key(name))
}

// ---------------------------------------------------------------------------
// Reading and writing the settings
// ---------------------------------------------------------------------------

/// The ore settings the Ores page shows: Custom Cubic's and Hybrid's own, or
/// their old lists converted; Earth's, none when it has none (and Flat Cubic
/// has no ores).
pub(super) fn view(state: &State) -> Option<OreGenSettings> {
    match state.kind {
        CubicKind::Earth => state.earth.ore_gen().clone(),
        CubicKind::Custom | CubicKind::Hybrid => Some(settings_view(&state.settings)),
        CubicKind::Flat => None,
    }
}

fn settings_view(c: &CustomGeneratorSettings) -> OreGenSettings {
    c.ore_gen.clone().unwrap_or_else(|| {
        OreGenSettings::from_legacy(
            &c.standard_ores,
            &c.periodic_gaussian_ores,
            c.expected_base_height,
            c.expected_height_variation,
        )
    })
}

/// Changes the ore settings. Custom Cubic and Hybrid write `oreGen` (from the
/// old lists when it was not there) and clear the lists; Earth starts from
/// [`OreGenSettings::earth_defaults`] when it had none.
pub(super) fn edit(state: &mut State, f: impl FnOnce(&mut OreGenSettings)) {
    match state.kind {
        CubicKind::Earth => {
            let ores = state
                .earth
                .ore_gen_mut()
                .get_or_insert_with(OreGenSettings::earth_defaults);
            f(ores);
        }
        CubicKind::Custom | CubicKind::Hybrid => state.edit(|c| {
            let mut ores = settings_view(c);
            f(&mut ores);
            c.ore_gen = Some(ores);
            c.standard_ores.clear();
            c.periodic_gaussian_ores.clear();
        }),
        CubicKind::Flat => {}
    }
}

/// Changes ore entry `i`, if there is one.
fn edit_ore(state: &mut State, i: usize, f: impl FnOnce(&mut OreEntry)) {
    edit(state, |o| {
        if let Some(entry) = o.ores.get_mut(i) {
            f(entry);
        }
    });
}

/// Changes the meteors.
fn edit_meteors(state: &mut State, f: impl FnOnce(&mut MeteorSettings)) {
    edit(state, |o| f(&mut o.meteors));
}

/// Changes meteor kind `i`, if there is one.
fn edit_kind(state: &mut State, i: usize, f: impl FnOnce(&mut MeteorKind)) {
    edit_meteors(state, |m| {
        if let Some(kind) = m.kinds.get_mut(i) {
            f(kind);
        }
    });
}

/// The block of entry `i`, for the block picker.
pub(super) fn block_of(state: &State, i: usize) -> Option<String> {
    view(state)?.ores.get(i).map(|o| o.blockstate.name.clone())
}

/// The biomes entry `i` is restricted to, for the biome picker; `None` for
/// every biome.
pub(super) fn biomes_of(state: &State, i: usize) -> Option<Vec<String>> {
    let ores = view(state)?;
    let list = ores.ores.get(i)?.biomes.as_ref()?;
    Some(list.iter().map(|b| b.name().to_owned()).collect())
}

/// The block picker's choice goes into entry `i`.
pub(super) fn set_block(state: &mut State, i: usize, id: &str) {
    edit_ore(state, i, |o| o.blockstate = BlockStateDesc::new(id));
}

/// The biome picker's choice goes into entry `i`: `None` for every biome.
pub(super) fn set_biomes(state: &mut State, i: usize, biomes: Option<Vec<String>>) {
    edit_ore(state, i, |o| {
        o.biomes = biomes.map(|list| {
            list.iter()
                .map(|b| horizon_world::cubic::custom::settings::BiomeDesc::new(b))
                .collect()
        });
    });
}

/// A new entry: iron veins in a flat band, background off.
fn new_ore() -> OreEntry {
    OreEntry {
        blockstate: BlockStateDesc::new("iron_ore"),
        spawn_size: 9,
        rich: RichZone {
            min: 0.0,
            max: 64.0,
            ..RichZone::default()
        },
        ..OreEntry::default()
    }
}

/// Reset to defaults: the ore list, the richness settings and the switch go
/// back to the world type's defaults; the meteors keep their own.
fn reset_ores(state: &mut State) {
    let defaults = if state.kind == CubicKind::Earth {
        OreGenSettings::earth_defaults()
    } else {
        OreGenSettings::custom_defaults()
    };
    edit(state, move |o| {
        o.enabled = defaults.enabled;
        o.ores = defaults.ores;
        o.richness_map = defaults.richness_map;
        o.richness_strength = defaults.richness_strength;
    });
}

// ---------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------

pub(super) fn page(b: &mut Builder, state: &State) {
    // An Earth world with no `oreGen` has no ores: the page is the one switch.
    let Some(o) = view(state) else {
        if state.kind == CubicKind::Earth {
            b.row(vec![toggle(&key("enabled"), false, |st, _| {
                edit(st, |o| o.enabled = true);
            })]);
        }
        return;
    };
    let earth = state.kind == CubicKind::Earth;
    b.row(vec![
        toggle(&key("enabled"), o.enabled, |st, on| {
            edit(st, |o| o.enabled = on);
        }),
        button(label("reset"), reset_ores),
    ]);
    b.row(vec![button(label("add"), |st| {
        edit(st, |o| o.ores.push(new_ore()));
    })]);
    if earth {
        b.row(vec![
            toggle(&key("richness"), o.richness_map, |st, on| {
                edit(st, |o| o.richness_map = on);
            }),
            number(
                &key("strength"),
                Mapping::Linear {
                    min: 0.0,
                    max: STRENGTH_MAX,
                },
                o.richness_strength,
                |st, v| edit(st, |o| o.richness_strength = (v * 20.0).round() / 20.0),
            ),
        ]);
    }
    let count = o.ores.len();
    for (i, entry) in o.ores.iter().enumerate() {
        ore_rows(b, i, entry, count);
    }
    meteor_rows(b, &o.meteors);
}

/// A block Y or a depth under the ground, as a label shows it.
fn bound_text(value: f64) -> String {
    if value.is_finite() {
        show(value)
    } else {
        tr("horizon.cubic.unbounded")
    }
}

/// A slider of one end of a band over `track`: the track's low end is no
/// limit for a minimum (`-inf`) and its high end for a maximum (`+inf`).
/// `set` gets the value to write.
fn bound(
    name: &str,
    track: (i32, i32),
    lower: bool,
    value: f64,
    set: impl Fn(&mut State, f64) + 'static,
) -> Item {
    let (lo, hi) = (f64::from(track.0), f64::from(track.1));
    number_shown(
        name,
        Mapping::Int {
            min: track.0,
            max: track.1,
        },
        value.clamp(lo, hi),
        bound_text(value),
        move |st, v| {
            set(
                st,
                if lower && v <= lo {
                    f64::NEG_INFINITY
                } else if !lower && v >= hi {
                    f64::INFINITY
                } else {
                    v
                },
            );
        },
    )
}

/// The band's two ends keep `min <= max`: the end being dragged stops at the
/// other.
fn keep_order(min: &mut f64, max: &mut f64, lower: bool) {
    if lower {
        *min = min.min(*max);
    } else {
        *max = max.max(*min);
    }
}

fn mode_key(mode: RichMode) -> &'static str {
    match mode {
        RichMode::StaticHeight => "mode.static",
        RichMode::BelowGround => "mode.below",
        RichMode::Noise => "mode.noise",
    }
}
fn next_mode(mode: RichMode) -> RichMode {
    match mode {
        RichMode::StaticHeight => RichMode::BelowGround,
        RichMode::BelowGround => RichMode::Noise,
        RichMode::Noise => RichMode::StaticHeight,
    }
}
fn shape_key(shape: Shape) -> &'static str {
    match shape {
        Shape::Flat => "shape.flat",
        Shape::Triangle => "shape.triangle",
        Shape::Bell => "shape.bell",
    }
}
fn next_shape(shape: Shape) -> Shape {
    match shape {
        Shape::Flat => Shape::Triangle,
        Shape::Triangle => Shape::Bell,
        Shape::Bell => Shape::Flat,
    }
}
fn range_key(range: RangeMode) -> &'static str {
    match range {
        RangeMode::Everywhere => "range.everywhere",
        RangeMode::StaticHeight => "mode.static",
        RangeMode::BelowGround => "mode.below",
    }
}
fn next_range(range: RangeMode) -> RangeMode {
    match range {
        RangeMode::Everywhere => RangeMode::StaticHeight,
        RangeMode::StaticHeight => RangeMode::BelowGround,
        RangeMode::BelowGround => RangeMode::Everywhere,
    }
}

/// Entry `i`'s rows: its heading, block and size, the move buttons, biomes,
/// tier 1 and tier 2. A tier that is off has its controls greyed.
fn ore_rows(b: &mut Builder, i: usize, e: &OreEntry, count: usize) {
    let block = block_name(&e.blockstate.name);
    b.heading(format!("{}. {block}", i + 1));
    b.row(vec![
        button(labelled("horizon.cubic.block", block), move |st| {
            st.open_picker(Target::OreBlock(i));
        }),
        int(
            &key("size"),
            VEIN.0,
            VEIN.1,
            e.spawn_size.clamp(VEIN.0, VEIN.1),
            move |st, v| edit_ore(st, i, |o| o.spawn_size = v),
        ),
    ]);
    let small = (WIDTH - 3.0 * 4.0) / 4.0;
    b.sized(
        vec![
            (
                small,
                disabled(
                    button(label("up"), move |st| {
                        edit(st, |o| {
                            if i > 0 && i < o.ores.len() {
                                o.ores.swap(i - 1, i);
                            }
                        });
                    }),
                    i == 0,
                ),
            ),
            (
                small,
                disabled(
                    button(label("down"), move |st| {
                        edit(st, |o| {
                            if i + 1 < o.ores.len() {
                                o.ores.swap(i, i + 1);
                            }
                        });
                    }),
                    i + 1 >= count,
                ),
            ),
            (
                small,
                button(label("duplicate"), move |st| {
                    edit(st, |o| {
                        if let Some(copy) = o.ores.get(i).cloned() {
                            o.ores.insert(i + 1, copy);
                        }
                    });
                }),
            ),
            (
                small,
                button(tr("selectServer.delete"), move |st| {
                    edit(st, |o| {
                        if i < o.ores.len() {
                            o.ores.remove(i);
                        }
                    });
                }),
            ),
        ],
        4.0,
    );
    let biomes = match &e.biomes {
        None => tr("horizon.cubic.biome.all"),
        Some(list) => list.len().to_string(),
    };
    let rich = &e.rich;
    b.row(vec![
        button(labelled(&key("biomes"), biomes), move |st| {
            st.open_picker(Target::OreBiomes(i));
        }),
        toggle(&key("tier1"), rich.enabled, move |st, on| {
            edit_ore(st, i, |o| o.rich.enabled = on);
        }),
    ]);
    tier_one(b, i, rich);
    tier_two(b, i, &e.background);
}

fn tier_one(b: &mut Builder, i: usize, r: &RichZone) {
    let off = !r.enabled;
    let mode = r.mode;
    let shape = r.shape;
    b.row(vec![
        disabled(
            button(labelled(&key("mode"), label(mode_key(mode))), move |st| {
                edit_ore(st, i, |o| {
                    o.rich.mode = next_mode(mode);
                    o.rich.min = f64::NEG_INFINITY;
                    o.rich.max = f64::INFINITY;
                });
            }),
            off,
        ),
        disabled(
            button(
                labelled(&key("shape"), label(shape_key(shape))),
                move |st| edit_ore(st, i, |o| o.rich.shape = next_shape(shape)),
            ),
            off,
        ),
    ]);
    match mode {
        RichMode::StaticHeight | RichMode::BelowGround => {
            let (track, low, high) = if mode == RichMode::StaticHeight {
                (Y_TRACK, "min_y", "max_y")
            } else {
                (DEPTH_TRACK, "min_depth", "max_depth")
            };
            b.row(vec![
                disabled(
                    bound(&key(low), track, true, r.min, move |st, v| {
                        edit_ore(st, i, |o| {
                            o.rich.min = v;
                            keep_order(&mut o.rich.min, &mut o.rich.max, true);
                        });
                    }),
                    off,
                ),
                disabled(
                    bound(&key(high), track, false, r.max, move |st, v| {
                        edit_ore(st, i, |o| {
                            o.rich.max = v;
                            keep_order(&mut o.rich.min, &mut o.rich.max, false);
                        });
                    }),
                    off,
                ),
            ]);
        }
        RichMode::Noise => {
            b.row(vec![
                disabled(
                    number(
                        &key("noise_scale"),
                        NOISE_SCALE,
                        r.noise_scale,
                        move |st, v| edit_ore(st, i, |o| o.rich.noise_scale = v.round()),
                    ),
                    off,
                ),
                disabled(
                    number(
                        &key("threshold"),
                        Mapping::Linear {
                            min: -1.0,
                            max: 1.0,
                        },
                        r.noise_threshold,
                        move |st, v| {
                            edit_ore(st, i, |o| {
                                o.rich.noise_threshold = (v * 100.0).round() / 100.0;
                            });
                        },
                    ),
                    off,
                ),
            ]);
            b.row(vec![disabled(
                int(
                    &key("seed"),
                    SEED.0,
                    SEED.1,
                    i32::try_from(r.noise_seed)
                        .unwrap_or(0)
                        .clamp(SEED.0, SEED.1),
                    move |st, v| edit_ore(st, i, |o| o.rich.noise_seed = i64::from(v)),
                ),
                off,
            )]);
        }
    }
    b.row(vec![disabled(
        number_shown(
            &key("intensity"),
            Mapping::Cubic { max: INTENSITY_MAX },
            r.intensity,
            format!("{:.2}", r.intensity),
            move |st, v| edit_ore(st, i, |o| o.rich.intensity = (v * 100.0).round() / 100.0),
        ),
        off,
    )]);
}

fn tier_two(b: &mut Builder, i: usize, bg: &Background) {
    let off = !bg.enabled;
    let range = bg.range;
    b.row(vec![
        toggle(&key("tier2"), bg.enabled, move |st, on| {
            edit_ore(st, i, |o| o.background.enabled = on);
        }),
        disabled(
            number_shown(
                &key("rate"),
                Mapping::Cubic { max: RATE_MAX },
                bg.rate,
                format!("{:.2}", bg.rate),
                move |st, v| edit_ore(st, i, |o| o.background.rate = (v * 100.0).round() / 100.0),
            ),
            off,
        ),
    ]);
    b.row(vec![disabled(
        button(
            labelled(&key("range"), label(range_key(range))),
            move |st| {
                edit_ore(st, i, |o| {
                    o.background.range = next_range(range);
                    o.background.min = f64::NEG_INFINITY;
                    o.background.max = f64::INFINITY;
                });
            },
        ),
        off,
    )]);
    if range != RangeMode::Everywhere {
        let (track, low, high) = if range == RangeMode::StaticHeight {
            (Y_TRACK, "min_y", "max_y")
        } else {
            (DEPTH_TRACK, "min_depth", "max_depth")
        };
        b.row(vec![
            disabled(
                bound(&key(low), track, true, bg.min, move |st, v| {
                    edit_ore(st, i, |o| {
                        o.background.min = v;
                        keep_order(&mut o.background.min, &mut o.background.max, true);
                    });
                }),
                off,
            ),
            disabled(
                bound(&key(high), track, false, bg.max, move |st, v| {
                    edit_ore(st, i, |o| {
                        o.background.max = v;
                        keep_order(&mut o.background.min, &mut o.background.max, false);
                    });
                }),
                off,
            ),
        ]);
    }
}

// ---------------------------------------------------------------------------
// Meteors
// ---------------------------------------------------------------------------

/// Hundredths, as the share and weight sliders step.
fn hundredths(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn meteor_rows(b: &mut Builder, m: &MeteorSettings) {
    let off = !m.enabled;
    b.heading(label("meteors"));
    let frequency = if m.per_square_km <= 0.0 {
        on_off(false)
    } else {
        show(m.per_square_km)
    };
    b.row(vec![
        toggle(&key("meteors"), m.enabled, |st, on| {
            edit_meteors(st, |m| m.enabled = on);
        }),
        disabled(
            number_shown(
                &key("meteor.rate"),
                Mapping::Cubic { max: FREQUENCY_MAX },
                m.per_square_km,
                frequency,
                |st, v| edit_meteors(st, |m| m.per_square_km = hundredths(v)),
            ),
            off,
        ),
    ]);
    let diameter = |name: &str, value: i32, lower: bool| {
        disabled(
            int(
                &key(name),
                DIAMETER.0,
                DIAMETER.1,
                value.clamp(DIAMETER.0, DIAMETER.1),
                move |st, v| {
                    edit_meteors(st, |m| {
                        if lower {
                            m.min_diameter = v.min(m.max_diameter);
                        } else {
                            m.max_diameter = v.max(m.min_diameter);
                        }
                    });
                },
            ),
            off,
        )
    };
    b.row(vec![
        diameter("meteor.min", m.min_diameter, true),
        diameter("meteor.max", m.max_diameter, false),
    ]);
    spline_rows(b, m, off);
    let share = |name: &str, value: f64, f: fn(&mut MeteorSettings, f64)| {
        disabled(
            number(
                &key(name),
                Mapping::Linear { min: 0.0, max: 1.0 },
                value,
                move |st, v| edit_meteors(st, |m| f(m, hundredths(v))),
            ),
            off,
        )
    };
    b.row(vec![
        share("meteor.surface", m.surface_share, |m, v| {
            m.surface_share = v
        }),
        share("meteor.sunk", m.sunk_share, |m, v| m.sunk_share = v),
    ]);
    b.row(vec![
        share("meteor.buried", m.buried_share, |m, v| m.buried_share = v),
        disabled(
            int(
                &key("meteor.depth"),
                BURIED_DEPTH.0,
                BURIED_DEPTH.1,
                (m.buried_depth.round() as i32).clamp(BURIED_DEPTH.0, BURIED_DEPTH.1),
                |st, v| edit_meteors(st, |m| m.buried_depth = f64::from(v)),
            ),
            off,
        ),
    ]);
    b.row(vec![disabled(
        number(
            &key("meteor.mix"),
            Mapping::Linear {
                min: 0.0,
                max: MIX_MAX,
            },
            m.mix_share,
            |st, v| edit_meteors(st, |m| m.mix_share = hundredths(v)),
        ),
        off,
    )]);
    kind_rows(b, m, off);
}

/// The size spline: a row a point (its diameter, its weight and a delete),
/// then Add point and the floor. A point's diameter stops at its neighbours',
/// so the points stay sorted while one is dragged.
fn spline_rows(b: &mut Builder, m: &MeteorSettings, off: bool) {
    b.heading(label("meteor.spline"));
    let points = &m.size_curve.0;
    let count = points.len();
    for (p, point) in points.iter().enumerate() {
        b.sized(
            vec![
                (
                    138.0,
                    disabled(
                        number(
                            &key("meteor.size"),
                            Mapping::Int {
                                min: DIAMETER.0,
                                max: DIAMETER.1,
                            },
                            point.y,
                            move |st, v| {
                                edit_meteors(st, |m| {
                                    let pts = &mut m.size_curve.0;
                                    let below =
                                        p.checked_sub(1).and_then(|n| pts.get(n)).map(|q| q.y);
                                    let above = pts.get(p + 1).map(|q| q.y);
                                    let mut y = v;
                                    if let Some(a) = above {
                                        y = y.min(a);
                                    }
                                    if let Some(l) = below {
                                        y = y.max(l);
                                    }
                                    if let Some(q) = pts.get_mut(p) {
                                        q.y = y;
                                    }
                                });
                            },
                        ),
                        off,
                    ),
                ),
                (
                    138.0,
                    disabled(
                        number(
                            &key("meteor.weight"),
                            Mapping::Linear { min: 0.0, max: 1.0 },
                            point.v,
                            move |st, v| {
                                edit_meteors(st, |m| {
                                    if let Some(q) = m.size_curve.0.get_mut(p) {
                                        q.v = hundredths(v);
                                    }
                                });
                            },
                        ),
                        off,
                    ),
                ),
                (
                    24.0,
                    disabled(
                        button("x".into(), move |st| {
                            edit_meteors(st, |m| {
                                if p < m.size_curve.0.len() {
                                    m.size_curve.0.remove(p);
                                }
                            });
                        }),
                        off || count <= 1,
                    ),
                ),
            ],
            5.0,
        );
    }
    b.row(vec![
        disabled(
            button(tr("horizon.cubic.point.add"), |st| {
                edit_meteors(st, |m| {
                    let last = m.size_curve.0.last().copied().unwrap_or(UserFunctionPoint {
                        y: f64::from(DIAMETER.0),
                        v: 0.5,
                    });
                    m.size_curve.0.push(UserFunctionPoint {
                        y: (last.y + 3.0).min(f64::from(DIAMETER.1)).max(last.y),
                        v: last.v,
                    });
                    m.size_curve.0.sort_by(|a, b| a.y.total_cmp(&b.y));
                });
            }),
            off,
        ),
        disabled(
            number_shown(
                &key("meteor.floor"),
                Mapping::Cubic { max: 1.0 },
                m.size_floor,
                show(hundredths(m.size_floor)),
                |st, v| edit_meteors(st, |m| m.size_floor = hundredths(v)),
            ),
            off,
        ),
    ]);
}

fn ore_name(short: &str) -> String {
    if short.is_empty() {
        label("meteor.none")
    } else if METEOR_ORES.contains(&short) {
        label(&format!("meteor.ore.{short}"))
    } else {
        short.to_owned()
    }
}

/// The ore after `current` in the cycle of [`METEOR_ORES`], and for a second
/// ore the empty name (none) after the last.
fn next_ore(current: &str, allow_none: bool) -> String {
    let next = match METEOR_ORES.iter().position(|o| *o == current) {
        Some(at) if at + 1 < METEOR_ORES.len() => METEOR_ORES[at + 1],
        Some(_) if allow_none => "",
        Some(_) | None => METEOR_ORES[0],
    };
    next.to_owned()
}

/// The kinds, a row each (ore, second ore, weight, delete), then Add kind and
/// Reset meteors.
fn kind_rows(b: &mut Builder, m: &MeteorSettings, off: bool) {
    b.heading(label("meteor.kinds"));
    for (k, kind) in m.kinds.iter().enumerate() {
        let (ore, second) = (kind.ore.clone(), kind.second.clone());
        b.sized(
            vec![
                (
                    88.0,
                    disabled(
                        button(
                            labelled(&key("meteor.ore"), ore_name(&kind.ore)),
                            move |st| edit_kind(st, k, |c| c.ore = next_ore(&ore, false)),
                        ),
                        off,
                    ),
                ),
                (
                    88.0,
                    disabled(
                        button(
                            labelled(&key("meteor.second"), ore_name(&kind.second)),
                            move |st| edit_kind(st, k, |c| c.second = next_ore(&second, true)),
                        ),
                        off,
                    ),
                ),
                (
                    88.0,
                    disabled(
                        number(
                            &key("meteor.weight"),
                            Mapping::Linear {
                                min: 0.0,
                                max: KIND_WEIGHT_MAX,
                            },
                            kind.weight,
                            move |st, v| {
                                edit_kind(st, k, |c| c.weight = (v * 10.0).round() / 10.0);
                            },
                        ),
                        off,
                    ),
                ),
                (
                    24.0,
                    disabled(
                        button("x".into(), move |st| {
                            edit_meteors(st, |m| {
                                if k < m.kinds.len() {
                                    m.kinds.remove(k);
                                }
                            });
                        }),
                        off,
                    ),
                ),
            ],
            6.0,
        );
    }
    b.row(vec![
        disabled(
            button(label("meteor.add_kind"), |st| {
                edit_meteors(st, |m| {
                    // The first ore no kind has yet, else iron again.
                    let ore = METEOR_ORES
                        .iter()
                        .find(|o| m.kinds.iter().all(|k| k.ore != **o))
                        .unwrap_or(&METEOR_ORES[0]);
                    m.kinds.push(MeteorKind {
                        ore: (*ore).to_owned(),
                        ..MeteorKind::default()
                    });
                });
            }),
            off,
        ),
        button(label("meteor.reset"), |st| {
            edit_meteors(st, |m| *m = MeteorSettings::default());
        }),
    ]);
}

#[cfg(test)]
#[path = "ores_tests.rs"]
mod tests;

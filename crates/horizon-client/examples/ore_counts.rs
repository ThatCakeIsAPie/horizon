//! Ore counts by depth for Cubic's two-tier ores, headless: generates and
//! populates a column of cubes through the client's own generation path
//! ([`horizon_client_lib::cubic_gen::Generation`]) and prints, per band of Y
//! (Custom Cubic) or of depth under the local ground (Earth), how many blocks
//! of each ore it holds.
//!
//! ore_counts custom [--x X] [--z Z] [--top 300] [--bottom -2100] [--band 32] [--size 4] [--seed 1]
//! ore_counts earth --lat LAT --lon LON --tiles <cache dir> --url <tile address template>
//!     [--deep 400] [--up 32] [--band 16] [--size 4] [--seed 1]
//!
//! `--size N` populates N x N chunk columns and counts the inner (N - 1) x (N - 1), whose every block has had all
//! the populations that reach it. Earth downloads missing tiles from `--url` (approved for this run only, in a list
//! kept in the cache dir) into `--tiles`.
use horizon_client_lib::cubic_gen::{CubicGeneration, CubicKind, Generation};
use horizon_client_lib::earth_consent::Approved;
use horizon_core::position::SectionPos;
use horizon_world::cubic::custom::populate::CubeWorld;
use horizon_world::cubic::custom::testing::{MapWorld, default_block_names, registry_of};
use horizon_world::cubic::earth::EarthSettings;
use horizon_world::cubic::ores::OreGenSettings;
use horizon_world::registry::{BlockRegistry, BlockStateId};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

type Error = Box<dyn std::error::Error + Send + Sync>;

const ORES: [&str; 8] = [
    "coal", "iron", "copper", "gold", "redstone", "diamond", "lapis", "emerald",
];

struct Args {
    rest: Vec<String>,
}

impl Args {
    fn get<T: std::str::FromStr>(&self, name: &str, default: Option<T>) -> Result<T, Error>
    where
        T::Err: std::error::Error + Send + Sync + 'static,
    {
        match self.rest.iter().position(|a| a == name) {
            Some(i) => Ok(self
                .rest
                .get(i + 1)
                .ok_or(format!("{name} needs a value"))?
                .parse()?),
            None => default.ok_or_else(|| format!("{name} is required").into()),
        }
    }
}

fn id(reg: &BlockRegistry, name: &str) -> Option<u32> {
    reg.default_state(&format!("minecraft:{name}"))
        .map(|BlockStateId(i)| i)
}

fn main() -> Result<(), Error> {
    let mut argv = std::env::args().skip(1);
    let usage = "usage: ore_counts custom|earth [options] (see the file's header)";
    let mode = argv.next().ok_or(usage)?;
    let args = Args {
        rest: argv.collect(),
    };
    let registry = registry_of(&default_block_names());
    let seed: i64 = args.get("--seed", Some(1))?;
    let size: i32 = args.get("--size", Some(4))?;
    let (generation, label) = match mode.as_str() {
        "custom" => {
            let choice = CubicGeneration {
                kind: CubicKind::Custom,
                settings_json: String::new(), // the defaults: oreGen = custom_defaults()
            };
            (
                Generation::new(&choice, seed, registry, None)?,
                "Y".to_owned(),
            )
        }
        "earth" => {
            let lat: f64 = args.get("--lat", None)?;
            let lon: f64 = args.get("--lon", None)?;
            let tiles: PathBuf = args.get("--tiles", None)?;
            let url: String = args.get("--url", None)?;
            std::fs::create_dir_all(&tiles)?;
            let mut settings = EarthSettings::at(lat, lon);
            settings.tile_url = url.clone();
            settings.allow_downloads = true;
            settings.land_cover = false;
            settings.climate = false;
            settings.vegetation = false;
            settings.ore_gen = Some(OreGenSettings::earth_defaults());
            let mut approved = Approved::load(&tiles);
            approved.approve(&horizon_client_lib::earth_net::host_of(&url)?)?;
            let parts = horizon_client_lib::earth_net::open_with(&settings, &tiles, &approved);
            if !parts.downloads {
                return Err("tile downloads are refused (see the log)".into());
            }
            let choice = CubicGeneration {
                kind: CubicKind::Earth,
                settings_json: settings.to_json_pretty(),
            };
            (
                Generation::with_tiles(&choice, seed, registry, None, Some(Arc::new(parts)))?,
                "depth".to_owned(),
            )
        }
        _ => return Err(usage.into()),
    };
    let generator = generation.generator.clone();
    let earth = mode == "earth";
    let (x0, z0) = if earth {
        (0, 0)
    } else {
        (
            args.get("--x", Some(0))? >> 4,
            args.get("--z", Some(0))? >> 4,
        )
    };
    let band: i32 = args.get("--band", Some(if earth { 16 } else { 32 }))?;

    // `--find-meteors R`: list the default meteors within R blocks of the column instead of counting.
    let find: i32 = args.get("--find-meteors", Some(0))?;
    if find > 0 {
        let ores = if earth {
            OreGenSettings::earth_defaults()
        } else {
            OreGenSettings::custom_defaults()
        };
        let meteors =
            horizon_world::cubic::ores::meteors::Meteors::new(seed, &ores.meteors, registry)?;
        let (cx, cz) = (x0 * 16, z0 * 16);
        let ground = |x: i32, z: i32| generator.surface_y(x, z, 512).unwrap_or(64);
        let mut spots = meteors.find(cx - find, cz - find, cx + find, cz + find, &ground);
        spots.sort_by_key(|s| (s.x - cx).pow(2) + (s.z - cz).pow(2));
        println!(
            "meteors within {find} of {cx} {cz} (seed {seed}): {}",
            spots.len()
        );
        for s in spots {
            println!(
                "{:>7} {:>6} {:>7}  d {:>2}  {:<8} {}",
                s.x, s.y, s.z, s.diameter, ores.meteors.kinds[s.kind].ore, s.placement
            );
        }
        return Ok(());
    }

    // The counted columns and their ground (Earth: the real surface, the reference of the depths).
    let mut ground = BTreeMap::new();
    for cx in x0 + 1..x0 + size {
        for cz in z0 + 1..z0 + size {
            for x in cx * 16..cx * 16 + 16 {
                for z in cz * 16..cz * 16 + 16 {
                    let g = if earth {
                        generator
                            .surface_y(x, z, i32::MAX)
                            .ok_or("no ground (tiles missing?)")?
                    } else {
                        0
                    };
                    ground.insert((x, z), g);
                }
            }
        }
    }
    let (lo_y, hi_y) = if earth {
        let deep: i32 = args.get("--deep", Some(400))?;
        let up: i32 = args.get("--up", Some(32))?;
        let min = *ground.values().min().unwrap();
        let max = *ground.values().max().unwrap();
        (min - deep, max + up)
    } else {
        (
            args.get("--bottom", Some(-2100))?,
            args.get("--top", Some(300))?,
        )
    };
    // Cubes lo..=hi are populated; lo + 1 ..= hi are complete.
    let (lo, hi) = ((lo_y >> 4) - 1, hi_y >> 4);

    let start = std::time::Instant::now();
    let mut world = MapWorld::new(0);
    for cx in x0..=x0 + size {
        for cz in z0..=z0 + size {
            for cy in lo..=hi + 1 {
                let mut blocks = [0_u32; 4096];
                generator.generate(SectionPos::new(cx, cy, cz), &mut blocks);
                world.insert([cx, cy, cz], blocks);
            }
        }
    }
    let generated = start.elapsed();
    for cy in (lo..=hi).rev() {
        for cx in x0..x0 + size {
            for cz in z0..z0 + size {
                let p = SectionPos::new(cx, cy, cz);
                if generation.needs_population(p) {
                    generator.populate(p, &mut world);
                }
            }
        }
    }
    let populated = start.elapsed() - generated;

    // Count.
    let ore_ids: Vec<[Option<u32>; 2]> = ORES
        .iter()
        .map(|o| {
            [
                id(registry, &format!("{o}_ore")),
                id(registry, &format!("deepslate_{o}_ore")),
            ]
        })
        .collect();
    let (stone, deepslate) = (id(registry, "stone"), id(registry, "deepslate"));
    // band index -> (ore counts, deepslate, stone + deepslate)
    let mut bands: BTreeMap<i32, ([u64; 8], u64, u64)> = BTreeMap::new();
    for (&(x, z), &g) in &ground {
        for y in (lo + 1) * 16..(hi + 1) * 16 {
            let key = if earth { g - y } else { y };
            let row = bands.entry(key.div_euclid(band)).or_default();
            let b = world.block([x, y, z]);
            for (i, ids) in ore_ids.iter().enumerate() {
                if ids.contains(&Some(b)) {
                    row.0[i] += 1;
                }
            }
            if Some(b) == deepslate {
                row.1 += 1;
            }
            if Some(b) == stone || Some(b) == deepslate {
                row.2 += 1;
            }
        }
    }
    let columns = ground.len();
    let g_min = ground.values().min().unwrap();
    let g_max = ground.values().max().unwrap();
    if earth {
        println!(
            "Earth at {} {} seed {seed}: {columns} columns, ground Y {g_min}..{g_max}; depth = ground - Y",
            args.get::<f64>("--lat", None)?,
            args.get::<f64>("--lon", None)?
        );
    } else {
        println!(
            "Custom Cubic defaults seed {seed}: {columns} columns at chunks x {}..{}, z {}..{}",
            x0 + 1,
            x0 + size - 1,
            z0 + 1,
            z0 + size - 1
        );
    }
    println!(
        "generate {:.1} s, populate {:.1} s; ore blocks per band (both stone and deepslate variants)",
        generated.as_secs_f64(),
        populated.as_secs_f64()
    );
    let mut header = format!("{:>13}", label);
    for o in ORES {
        header += &format!(" {o:>8}");
    }
    header += if earth { "  deepsl%" } else { "" };
    println!("{header}");
    let order: Vec<i32> = if earth {
        bands.keys().copied().collect()
    } else {
        bands.keys().rev().copied().collect()
    };
    let mut total = [0_u64; 8];
    for k in order {
        let (counts, deep, rock) = bands[&k];
        let (a, b) = (k * band, k * band + band - 1);
        let mut line = format!("{a:>6}..{b:<6}");
        for (i, c) in counts.iter().enumerate() {
            line += &format!(" {c:>8}");
            total[i] += c;
        }
        if earth {
            let pct = if rock == 0 {
                0.0
            } else {
                100.0 * deep as f64 / rock as f64
            };
            line += &format!("  {pct:>6.1}");
        }
        println!("{line}");
    }
    let mut line = format!("{:>13}", "total");
    for c in total {
        line += &format!(" {c:>8}");
    }
    println!("{line}");
    Ok(())
}

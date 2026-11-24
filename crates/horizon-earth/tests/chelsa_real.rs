//! Reads the real CHELSA V2.1 files over HTTPS with `curl` and checks five places against values read by hand.
//! Ignored by default (it needs the network and fetches a few MiB of the 150 MB and 350 MB files):
//!
//! `cargo test -p horizon-earth --test chelsa_real -- --ignored --nocapture`

use std::process::Command;
use std::sync::Mutex;

use horizon_earth::climate::{ClimateVar, cache_key};
use horizon_earth::tiff::ByteRange;
use horizon_earth::{ClimateFetch, ClimateFetcher, ClimateLookup, ClimateStore, FetchError};

const BASE: &str = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim/";

/// A fetcher that asks the server with the system's `curl`, one range request a fetch.
struct CurlFetcher {
    requests: Mutex<Vec<(ClimateVar, ByteRange, usize)>>,
}

impl ClimateFetcher for CurlFetcher {
    fn fetch(&self, var: ClimateVar, range: ByteRange) -> Result<Option<Vec<u8>>, FetchError> {
        let url = format!("{BASE}{}", var.path());
        let last = range.end() - 1;
        let out = Command::new("curl")
            .args(["-sS", "--fail", "--max-time", "60", "-r"])
            .arg(format!("{}-{last}", range.offset))
            .arg(&url)
            .output()
            .map_err(|e| FetchError::Other(format!("curl did not start: {e}")))?;
        if out.status.success() {
            self.requests
                .lock()
                .unwrap()
                .push((var, range, out.stdout.len()));
            return Ok(Some(out.stdout));
        }
        let why = String::from_utf8_lossy(&out.stderr).into_owned();
        // `--fail` reports an HTTP error as exit code 22 and "The requested URL returned error: 404".
        if out.status.code() == Some(22) && why.contains("404") {
            return Ok(None);
        }
        Err(FetchError::Temporary(why))
    }
}

/// Fetches what a lookup lacks until it answers.
fn resolve(
    store: &ClimateStore,
    fetcher: &CurlFetcher,
    lookup: impl Fn(&ClimateStore) -> ClimateLookup,
) -> ClimateLookup {
    for _ in 0..64 {
        match lookup(store) {
            ClimateLookup::Need(fetch) => satisfy(store, fetcher, &fetch),
            done => return done,
        }
    }
    panic!("a lookup that never finishes");
}

fn satisfy(store: &ClimateStore, fetcher: &CurlFetcher, fetch: &ClimateFetch) {
    match fetcher.fetch(fetch.var, fetch.range).expect("curl") {
        None => store.mark_absent(fetch.var),
        Some(bytes) => store.provide(fetch, &bytes).expect("the store takes it"),
    }
}

/// (name, lat, lon, temperature C, precipitation mm), worked out by hand from the raw pixels.
const SPOTS: [(&str, f64, f64, f64, f64); 5] = [
    ("Amazon", -3.1, -60.0, 26.65, 2449.0),
    ("Sahara", 23.0, 10.0, 24.05, 6.5),
    ("Canada", 55.0, -100.0, -0.85, 505.5),
    ("Alps", 46.5, 8.0, -5.75, 2513.0),
    ("Atlantic", 0.0, -30.0, 25.85, 1578.0),
];

#[test]
#[ignore = "needs the network"]
fn five_places_read_from_the_real_files_match_the_values_read_by_hand() {
    let scratch = std::env::temp_dir().join(format!("chelsa-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let store = ClimateStore::new(&scratch, 16);
    let fetcher = CurlFetcher {
        requests: Mutex::new(Vec::new()),
    };
    println!("cache key {} in {}", cache_key(), scratch.display());
    let mut wrong: Vec<String> = Vec::new();
    for (name, lat, lon, want_t, want_p) in SPOTS {
        let nearest = resolve(&store, &fetcher, |s| s.lookup_nearest(lat, lon, 1.0));
        let bilinear = resolve(&store, &fetcher, |s| s.lookup(lat, lon, 1.0));
        let (ClimateLookup::Climate(n), ClimateLookup::Climate(b)) = (nearest, bilinear) else {
            panic!("{name}: no climate: {nearest:?} {bilinear:?}");
        };
        println!(
            "{name:9} ({lat:6.1}, {lon:6.1})  nearest {:7.2} C {:8.1} mm   bilinear {:7.2} C {:8.1} mm   hand {want_t:7.2} C {want_p:8.1} mm",
            n.temperature_c, n.precipitation_mm, b.temperature_c, b.precipitation_mm
        );
        if (f64::from(n.temperature_c) - want_t).abs() >= 0.01 {
            wrong.push(format!(
                "{name}: temperature {} vs {want_t}",
                n.temperature_c
            ));
        }
        // Precipitation is whole millimetres in the file; the hand values were rounded to half ones.
        if (f64::from(n.precipitation_mm) - want_p).abs() > 1.0 {
            wrong.push(format!(
                "{name}: precipitation {} vs {want_p}",
                n.precipitation_mm
            ));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
    // A coarse read of the same place comes from an overview and is not far off.
    let coarse = resolve(&store, &fetcher, |s| s.lookup(46.5, 8.0, 50_000.0));
    println!("Alps at 50 km a sample: {coarse:?}");
    assert!(matches!(coarse, ClimateLookup::Climate(_)));
    // Off the data in the far north: no data, and no request.
    assert_eq!(store.lookup(86.0, 0.0, 1.0), ClimateLookup::NoData);
    let requests = fetcher.requests.lock().unwrap();
    let bytes: usize = requests.iter().map(|r| r.2).sum();
    println!(
        "{} range requests, {} KiB in all",
        requests.len(),
        bytes / 1024
    );
    assert!(bytes < 16 << 20, "far less than the 500 MB of the files");
    drop(requests);
    let _ = std::fs::remove_dir_all(&scratch);
}

//! Tests of the land cover downloader against a tiny server on 127.0.0.1: nothing here touches the internet.
use super::*;
use crate::earth_net::tests::{Reply, Server};
use horizon_earth::testing::{MemoryLandCover, TempDir};
use horizon_earth::{LandClass, Lookup};

const FILE: WorldCoverFile = WorldCoverFile { south: 45, west: 6 };

fn settings(land: bool, allow: bool) -> EarthSettings {
    let mut s = EarthSettings::at(46.0, 7.5);
    s.land_cover = land;
    s.allow_downloads = allow;
    s
}

fn local() -> Approved {
    Approved::of(&["127.0.0.1"])
}

/// The first and last byte of a `Range: bytes=a-b` header.
fn range_of(headers: &[String]) -> Option<(usize, usize)> {
    let h = headers
        .iter()
        .find(|h| h.to_ascii_lowercase().starts_with("range:"))?;
    let (a, b) = h.split_once("bytes=")?.1.split_once('-')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// A server of one file's bytes at the dataset's path, honouring ranges like S3.
fn serve(file: &'static [u8]) -> Server {
    Server::start_seeing(move |path, headers| {
        if !path.ends_with(&FILE.path()) {
            return Reply::Status(404);
        }
        let Some((a, b)) = range_of(headers) else {
            return Reply::Ok(file.to_vec());
        };
        if a >= file.len() {
            return Reply::Status(416);
        }
        let b = b.min(file.len() - 1);
        Reply::Partial(
            file[a..=b].to_vec(),
            format!("bytes {a}-{b}/{}", file.len()),
        )
    })
}

fn fetcher(server: &Server) -> HttpLandCover {
    HttpLandCover::new(
        &settings(true, true),
        Some(&server.template("/")),
        &local(),
    )
}

fn range(offset: u64, len: u64) -> ByteRange {
    ByteRange { offset, len }
}

static BYTES: [u8; 100] = {
    let mut b = [0u8; 100];
    let mut i = 0;
    while i < 100 {
        b[i] = i as u8;
        i += 1;
    }
    b
};

#[test]
fn a_range_is_asked_for_with_the_range_header_and_the_dataset_path_and_nothing_else() {
    let server = serve(&BYTES);
    let got = fetcher(&server).fetch(FILE, range(10, 20)).unwrap();
    assert_eq!(got.as_deref(), Some(&BYTES[10..30]));
    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let (line, headers) = &requests[0];
    assert_eq!(
        line,
        &format!("GET /v200/2021/map/{} HTTP/1.1", {
            let p = FILE.path();
            p.rsplit('/').next().unwrap().to_owned()
        })
    );
    let lower: Vec<String> = headers.iter().map(|h| h.to_ascii_lowercase()).collect();
    assert!(lower.contains(&"range: bytes=10-29".to_owned()), "{lower:?}");
    assert!(
        lower.contains(&format!("user-agent: horizon/{}", horizon_core::release::VERSION).to_owned()),
        "{lower:?}"
    );
    for h in &lower {
        assert!(
            !["cookie", "referer", "authorization", "x-"]
                .iter()
                .any(|bad| h.starts_with(bad)),
            "{h}"
        );
    }
}

#[test]
fn a_range_past_the_end_of_the_file_is_cut_short_like_the_store_expects() {
    let server = serve(&BYTES);
    let got = fetcher(&server).fetch(FILE, range(90, 50)).unwrap();
    assert_eq!(got.as_deref(), Some(&BYTES[90..]));
}

#[test]
fn a_server_that_ignores_range_is_sliced_not_trusted() {
    let server = Server::start(|_| Reply::Ok(BYTES.to_vec()));
    let got = fetcher(&server).fetch(FILE, range(10, 5)).unwrap();
    assert_eq!(got.as_deref(), Some(&BYTES[10..15]));
}

#[test]
fn bytes_from_another_offset_than_asked_are_an_error() {
    let server = Server::start(|_| Reply::Partial(BYTES[..10].to_vec(), "bytes 50-59/100".into()));
    let err = fetcher(&server).fetch(FILE, range(10, 10)).unwrap_err();
    assert!(matches!(err, FetchError::Other(_)), "{err:?}");
}

#[test]
fn a_404_is_a_file_that_does_not_exist_and_other_failures_are_classified_like_tiles() {
    let server = Server::start(|_| Reply::Status(404));
    assert_eq!(fetcher(&server).fetch(FILE, range(0, 10)), Ok(None));
    for (code, want) in [(403, "refused"), (503, "temporary"), (418, "other")] {
        let server = Server::start(move |_| Reply::Status(code));
        let err = fetcher(&server).fetch(FILE, range(0, 10)).unwrap_err();
        let got = match err {
            FetchError::Refused(_) => "refused",
            FetchError::Temporary(_) => "temporary",
            FetchError::Other(_) => "other",
            FetchError::NotFound => "not found",
        };
        assert_eq!(got, want, "HTTP {code}");
    }
}

#[test]
fn nothing_is_requested_unless_the_world_allows_downloads_and_the_host_is_approved() {
    let server = serve(&BYTES);
    let base = server.template("/");
    let cases = [
        ("land cover off", settings(false, true), Some(base.as_str()), local()),
        ("downloads off", settings(true, false), Some(base.as_str()), local()),
        ("no address", settings(true, true), None, local()),
        ("not approved", settings(true, true), Some(base.as_str()), Approved::of(&[])),
        (
            "http to another host",
            settings(true, true),
            Some("http://example.com/"),
            Approved::of(&["example.com"]),
        ),
    ];
    for (name, settings, address, approved) in cases {
        let f = HttpLandCover::new(&settings, address, &approved);
        assert!(!f.downloads(), "{name}");
        let why = f.refusal().unwrap().to_owned();
        assert!(!why.is_empty());
        match f.fetch(FILE, range(0, 10)) {
            Err(FetchError::Refused(w)) => assert_eq!(w, why, "{name}"),
            other => panic!("{name}: {other:?}"),
        }
    }
    assert!(server.requests().is_empty());
}

#[test]
fn a_redirect_leads_only_to_an_approved_host() {
    let target = serve(&BYTES);
    let there = target.template("/");
    // The same machine under another name is another host: `localhost` is not approved.
    let to = format!("{}v200/2021/map/x.tif", there.replace("127.0.0.1", "localhost"));
    let server = Server::start(move |_| Reply::Redirect(to.clone()));
    let err = fetcher(&server).fetch(FILE, range(0, 10)).unwrap_err();
    assert!(matches!(err, FetchError::Refused(_)), "{err:?}");
    assert!(target.requests().is_empty());
}

#[test]
fn only_a_few_requests_are_in_flight_at_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (now, most) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let (n, m) = (now.clone(), most.clone());
    let server = Server::start(move |_| {
        let k = n.fetch_add(1, Ordering::SeqCst) + 1;
        m.fetch_max(k, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(80));
        n.fetch_sub(1, Ordering::SeqCst);
        Reply::Partial(BYTES[..10].to_vec(), "bytes 0-9/100".into())
    });
    let f = Arc::new(HttpLandCover::over(
        Ok(server.template("/")),
        local(),
        Duration::from_secs(5),
        Duration::from_secs(5),
        Duration::from_secs(5),
        2,
    ));
    let threads: Vec<_> = (0..6)
        .map(|_| {
            let f = f.clone();
            std::thread::spawn(move || f.fetch(FILE, range(0, 10)).unwrap())
        })
        .collect();
    for t in threads {
        assert!(t.join().unwrap().is_some());
    }
    assert!(most.load(Ordering::SeqCst) <= 2, "{}", most.load(Ordering::SeqCst));
}

#[test]
fn a_host_is_only_named_for_a_world_that_wants_land_cover_and_downloads() {
    let a = Some("https://Bucket.Example.org/");
    assert_eq!(
        host_for(&settings(true, true), a).as_deref(),
        Some("bucket.example.org")
    );
    assert_eq!(host_for(&settings(false, true), a), None);
    assert_eq!(host_for(&settings(true, false), a), None);
    assert_eq!(host_for(&settings(true, true), None), None);
    assert_eq!(host_for(&settings(true, true), Some("ftp://x.org/")), None);
}

#[test]
fn a_world_whose_host_is_not_approved_has_no_land_source_and_says_why() {
    let dir = TempDir::new();
    let open = |approved: &Approved| {
        open_with(
            &settings(true, true),
            dir.path(),
            Some("https://bucket.example.org/"),
            approved,
        )
    };
    let refused = open(&Approved::of(&["tiles.example.com"]));
    assert!(refused.source.is_none());
    assert!(refused.why_not.unwrap().contains("bucket.example.org"));
    assert!(open(&Approved::of(&["bucket.example.org"])).source.is_some());
    // A world that does not ask for land cover gets none without a word.
    let off = open_with(&settings(false, true), dir.path(), Some("https://bucket.example.org/"), &Approved::of(&["bucket.example.org"]));
    assert!(off.source.is_none());
}

/// The whole path over HTTP: a miniature file served by range, read through the store, kept on disk, and read
/// again by a second store with the server gone.
#[test]
fn a_second_run_at_the_same_place_makes_no_request() {
    let file: &'static [u8] = Box::leak(
        MemoryLandCover::new()
            .with_classes(FILE, 256, 64, 2, |_, _| 40)
            .file_bytes(FILE)
            .unwrap()
            .into_boxed_slice(),
    );
    let server = serve(file);
    let dir = TempDir::new();
    let base = server.template("/");
    let run = || {
        open_with(
            &settings(true, true),
            dir.path(),
            Some(&base),
            &local(),
        )
        .source
        .unwrap()
    };
    let first = run();
    let class = first.class_at_blocking(0.5, 0.5, 1.0).unwrap();
    // A different projection origin than the file: whatever it reads, it is the file's class or no data.
    assert!(matches!(class, LandClass::Crop | LandClass::NoData));
    let asked = server.requests().len();
    assert!(asked >= 2, "an index and a tile, got {asked}");
    drop(first);
    // The server is gone; a new store over the same folder answers from disk.
    drop(server);
    let second = run();
    assert_eq!(second.class_at_blocking(0.5, 0.5, 1.0).unwrap(), class);
    assert!(matches!(
        second.field().class_at_block(0, 0),
        Lookup::Class(_)
    ));
}

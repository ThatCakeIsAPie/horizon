//! Tests of the tile downloader against a tiny server on 127.0.0.1: nothing here touches the internet.
use super::*;
use horizon_earth::testing::{TempDir, encode_tile};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// What the test server answers a path with.
enum Reply {
    /// 200 with these bytes (and their length).
    Ok(Vec<u8>),
    /// 200 with these bytes and no length: the body ends when the connection does.
    Streamed(Vec<u8>),
    Status(u16),
    Redirect(String),
    /// No answer for this long.
    Stall(Duration),
}

struct Server {
    port: u16,
    /// Each request as its request line and its headers.
    seen: Arc<Mutex<Vec<(String, Vec<String>)>>>,
    stop: Arc<AtomicBool>,
}

impl Server {
    fn start(reply: impl Fn(&str) -> Reply + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let reply = Arc::new(reply);
        let (their_seen, their_stop) = (seen.clone(), stop.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if their_stop.load(Ordering::SeqCst) {
                    return;
                }
                let Ok(stream) = stream else { continue };
                let (reply, seen) = (reply.clone(), their_seen.clone());
                std::thread::spawn(move || answer(stream, &*reply, &seen));
            }
        });
        Self { port, seen, stop }
    }

    fn template(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    fn requests(&self) -> Vec<(String, Vec<String>)> {
        self.seen.lock().unwrap().clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Wake the accept loop so it ends.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
    }
}

fn answer(
    mut stream: TcpStream,
    reply: &dyn Fn(&str) -> Reply,
    seen: &Mutex<Vec<(String, Vec<String>)>>,
) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let mut headers = Vec::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap_or(0) == 0 || header.trim().is_empty() {
            break;
        }
        headers.push(header.trim().to_owned());
    }
    let target = line.split_whitespace().nth(1).unwrap_or("/").to_owned();
    // The handler decides by the path; the query holds the tile numbers.
    let path = target.split('?').next().unwrap_or("/").to_owned();
    seen.lock().unwrap().push((line.trim().to_owned(), headers));
    let head =
        |status: &str, extra: &str| format!("HTTP/1.1 {status}\r\nConnection: close\r\n{extra}");
    let _ = match reply(&path) {
        Reply::Ok(body) => stream
            .write_all(
                head("200 OK", &format!("Content-Length: {}\r\n\r\n", body.len())).as_bytes(),
            )
            .and_then(|()| stream.write_all(&body)),
        Reply::Streamed(body) => stream
            .write_all(head("200 OK", "\r\n").as_bytes())
            .and_then(|()| stream.write_all(&body)),
        Reply::Status(code) => stream
            .write_all(head(&format!("{code} Whatever"), "Content-Length: 0\r\n\r\n").as_bytes()),
        Reply::Redirect(to) => stream.write_all(
            head(
                "302 Found",
                &format!("Location: {to}\r\nContent-Length: 0\r\n\r\n"),
            )
            .as_bytes(),
        ),
        Reply::Stall(time) => {
            std::thread::sleep(time);
            Ok(())
        }
    };
}

fn settings(template: &str, allow: bool) -> EarthSettings {
    let mut s = EarthSettings::at(46.0, 7.5);
    s.tile_url = template.to_owned();
    s.allow_downloads = allow;
    s
}

const ID: TileId = TileId { z: 5, x: 17, y: 11 };

#[test]
fn a_tile_is_downloaded_from_the_address_with_its_numbers_in_place() {
    let server = Server::start(|_| Reply::Ok(b"png bytes".to_vec()));
    let fetcher = HttpTileFetcher::new(&settings(&server.template("/tiles/{z}/{x}/{y}.png"), true));
    assert_eq!(fetcher.fetch(ID).unwrap(), b"png bytes");
    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0].0.starts_with("GET /tiles/5/17/11.png "),
        "{}",
        requests[0].0
    );
}

#[test]
fn the_address_template_replaces_every_placeholder_wherever_it_stands() {
    let id = TileId { z: 3, x: 4, y: 2 };
    assert_eq!(
        tile_url("https://example.com/{z}/{x}/{y}.png", id),
        "https://example.com/3/4/2.png"
    );
    assert_eq!(
        tile_url(
            "https://example.com/t?row={y}&col={x}&level={z}&again={z}",
            id
        ),
        "https://example.com/t?row=2&col=4&level=3&again=3"
    );
}

#[test]
fn a_request_says_only_that_it_is_horizon() {
    let server = Server::start(|_| Reply::Ok(vec![1, 2, 3]));
    let fetcher = HttpTileFetcher::new(&settings(&server.template("/{z}/{x}/{y}"), true));
    fetcher.fetch(ID).unwrap();
    let headers: Vec<String> = server.requests()[0]
        .1
        .iter()
        .map(|h| h.to_lowercase())
        .collect();
    let agent = headers
        .iter()
        .find(|h| h.starts_with("user-agent:"))
        .unwrap();
    assert!(agent.starts_with("user-agent: horizon/"), "{agent}");
    for name in ["cookie", "authorization", "referer", "x-"] {
        assert!(
            !headers.iter().any(|h| h.starts_with(name)),
            "a {name} header was sent: {headers:?}"
        );
    }
}

#[test]
fn a_missing_tile_is_not_found_and_a_busy_server_is_temporary() {
    let server = Server::start(|path| match path {
        "/404" | "/410" => Reply::Status(if path == "/404" { 404 } else { 410 }),
        "/429" => Reply::Status(429),
        "/503" => Reply::Status(503),
        "/500" => Reply::Status(500),
        "/403" => Reply::Status(403),
        _ => Reply::Status(418),
    });
    let get = |path: &str| {
        HttpTileFetcher::new(&settings(
            &server.template(&format!("{path}?{{z}}{{x}}{{y}}")),
            true,
        ))
        .fetch(ID)
    };
    assert_eq!(get("/404"), Err(FetchError::NotFound));
    assert_eq!(get("/410"), Err(FetchError::NotFound));
    for path in ["/429", "/503", "/500"] {
        assert!(matches!(get(path), Err(FetchError::Temporary(_))), "{path}");
    }
    for path in ["/403", "/418"] {
        assert!(matches!(get(path), Err(FetchError::Other(_))), "{path}");
    }
}

#[test]
fn a_server_that_does_not_answer_is_a_timeout_that_can_be_retried() {
    let server = Server::start(|_| Reply::Stall(Duration::from_secs(5)));
    let fetcher = HttpTileFetcher::new(&settings(&server.template("/{z}/{x}/{y}"), true))
        .limits(Duration::from_millis(250), MAX_TILE_BYTES);
    let started = Instant::now();
    let error = fetcher.fetch(ID).unwrap_err();
    assert!(matches!(error, FetchError::Temporary(_)), "{error:?}");
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
}

#[test]
fn nothing_listening_is_temporary_too() {
    // A port that was just free: nobody answers on it.
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let fetcher = HttpTileFetcher::new(&settings(
        &format!("http://127.0.0.1:{port}/{{z}}/{{x}}/{{y}}"),
        true,
    ));
    assert!(matches!(fetcher.fetch(ID), Err(FetchError::Temporary(_))));
}

#[test]
fn a_response_over_the_size_cap_is_refused_however_it_is_sent() {
    let big = vec![7u8; 5_000];
    let sent = big.clone();
    let server = Server::start(move |path| {
        if path.starts_with("/streamed") {
            Reply::Streamed(sent.clone())
        } else {
            Reply::Ok(sent.clone())
        }
    });
    let capped = |path: &str, cap: usize| {
        HttpTileFetcher::new(&settings(
            &server.template(&format!("{path}/{{z}}/{{x}}/{{y}}")),
            true,
        ))
        .limits(Duration::from_secs(5), cap)
        .fetch(ID)
    };
    // By its declared length, and by counting what arrives when no length is declared.
    for path in ["/sized", "/streamed"] {
        let error = capped(path, 1_000).unwrap_err();
        assert!(
            matches!(&error, FetchError::Other(why) if why.contains("larger")),
            "{path}: {error:?}"
        );
    }
    // Exactly at the cap is fine.
    assert_eq!(capped("/sized", 5_000).unwrap(), big);
    assert_eq!(capped("/streamed", 5_000).unwrap(), big);
}

#[test]
fn nothing_is_downloaded_unless_the_world_allows_it() {
    let server = Server::start(|_| Reply::Ok(vec![1]));
    let template = server.template("/{z}/{x}/{y}");
    let off = HttpTileFetcher::new(&settings(&template, false));
    assert!(!off.downloads());
    assert!(matches!(off.fetch(ID), Err(FetchError::Other(why)) if why.contains("allowDownloads")));
    let none = HttpTileFetcher::new(&settings("", true));
    assert!(!none.downloads());
    assert!(matches!(none.fetch(ID), Err(FetchError::Other(why)) if why.contains("tileUrl")));
    assert!(
        server.requests().is_empty(),
        "a refused fetch reached the network"
    );
}

#[test]
fn addresses_the_game_may_not_download_from_are_refused() {
    for (template, why) in [
        ("http://example.com/{z}/{x}/{y}.png", "https"),
        ("ftp://example.com/{z}/{x}/{y}.png", "https"),
        ("example.com/{z}/{x}/{y}.png", "https"),
        ("https://example.com/{z}/{x}.png", "{y}"),
        ("https://example.com/tiles.png", "{z}"),
        ("https://example.com/{z}/{x}/{y}/{s}.png", "other than"),
        ("https://tiles-{x}.example.com/{z}/{y}.png", "host"),
        (
            "https://user:secret@example.com/{z}/{x}/{y}.png",
            "user name",
        ),
        ("https:///{z}/{x}/{y}.png", "no host"),
        ("https://exa mple.com/{z}/{x}/{y}.png", "spaces"),
        ("http://127.0.0.1.example.com/{z}/{x}/{y}.png", "https"),
        ("http://localhost.example.com/{z}/{x}/{y}.png", "https"),
    ] {
        let error = check_template(template).expect_err(template);
        assert!(error.contains(why), "{template}: {error}");
        let fetcher = HttpTileFetcher::new(&settings(template, true));
        assert!(!fetcher.downloads(), "{template}");
        assert!(
            matches!(fetcher.fetch(ID), Err(FetchError::Other(_))),
            "{template}"
        );
    }
    for template in [
        "https://example.com/{z}/{x}/{y}.png",
        "https://example.com:8443/a/{z}/{x}/{y}?k=1",
        "http://127.0.0.1:8080/{z}/{x}/{y}.png",
        "http://localhost/{z}/{x}/{y}.png",
        "http://[::1]:9/{z}/{x}/{y}.png",
    ] {
        assert_eq!(check_template(template), Ok(()), "{template}");
    }
}

#[test]
fn a_redirect_is_followed_only_to_an_address_that_is_allowed() {
    let server = Server::start(|path| match path {
        "/start" => Reply::Redirect("/moved".into()),
        "/moved" => Reply::Ok(b"arrived".to_vec()),
        "/leaves" => Reply::Redirect("http://example.com/tile.png".into()),
        _ => Reply::Redirect("/loop".into()),
    });
    let get = |path: &str| {
        HttpTileFetcher::new(&settings(
            &server.template(&format!("{path}?{{z}}{{x}}{{y}}")),
            true,
        ))
        .fetch(ID)
    };
    assert_eq!(get("/start").unwrap(), b"arrived");
    let error = get("/leaves").unwrap_err();
    assert!(
        matches!(&error, FetchError::Other(why) if why.contains("not allowed")),
        "{error:?}"
    );
    // A loop ends.
    let error = get("/loop").unwrap_err();
    assert!(
        matches!(&error, FetchError::Other(why) if why.contains("redirects")),
        "{error:?}"
    );
}

#[test]
fn tiles_of_different_addresses_are_cached_apart() {
    let a = address_key("https://one.example.com/{z}/{x}/{y}.png");
    let b = address_key("https://two.example.com/{z}/{x}/{y}.png");
    assert_ne!(a, b);
    assert_eq!(a, address_key("  https://one.example.com/{z}/{x}/{y}.png "));
    assert_eq!(address_key(""), "none");
    assert!(
        a.len() >= 16 && a.chars().all(|c| c.is_ascii_hexdigit()),
        "{a}"
    );
    let root = TempDir::new();
    let one = open(
        &settings("https://one.example.com/{z}/{x}/{y}.png", false),
        root.path(),
    );
    let two = open(
        &settings("https://two.example.com/{z}/{x}/{y}.png", false),
        root.path(),
    );
    assert_eq!(one.dir, root.path().join(&a));
    assert_ne!(one.dir, two.dir);
    assert!(!one.downloads && !two.downloads);
    assert_eq!(cache_root().file_name().unwrap(), "earth-tiles");
    assert!(cache_root().ends_with(CACHE_DIR));
}

#[test]
fn a_store_over_the_downloader_saves_what_it_gets_where_the_address_says() {
    let hills = |lat: f64, lon: f64| 100.0 + lat + lon;
    let server = Server::start(move |_| Reply::Ok(encode_tile(ID, &hills)));
    let root = TempDir::new();
    let template = server.template("/dem/{z}/{x}/{y}.png");
    let tiles = open(&settings(&template, true), root.path());
    assert!(tiles.downloads);
    let tile = tiles.store.load(ID).unwrap();
    assert!(tile.max() > 100.0);
    assert!(tiles.dir.join("5/17/11.png").is_file());
    // The second load is local.
    tiles.store.load(ID).unwrap();
    assert_eq!(server.requests().len(), 1);
    // Another world with downloads off, over the same address, finds the cached tile without the server.
    let offline = open(&settings(&template, false), root.path());
    assert!(!offline.downloads);
    assert!(offline.store.cached(ID).is_some());
    assert!(offline.store.load(TileId { z: 5, x: 18, y: 11 }).is_err());
    assert_eq!(server.requests().len(), 1);
}

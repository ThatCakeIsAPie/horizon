use super::*;
use horizon_earth::testing::TempDir;

#[test]
fn an_approved_host_survives_a_restart() {
    let dir = TempDir::new();
    let mut list = Approved::load(dir.path());
    assert!(!list.contains("tiles.example.com"));
    list.approve("Tiles.Example.com").unwrap();
    list.approve("127.0.0.1").unwrap();
    // A new game reads the file again.
    let again = Approved::load(dir.path());
    assert!(again.contains("tiles.example.com") && again.contains("TILES.example.com"));
    assert!(again.contains("127.0.0.1"));
    assert!(!again.contains("other.example.com"));
    // The approval is of the host only.
    assert!(!again.contains("example.com") && !again.contains("evil-tiles.example.com"));
}

#[test]
fn revoking_takes_the_approval_back_for_good() {
    let dir = TempDir::new();
    let mut list = Approved::load(dir.path());
    list.approve("a.example.com").unwrap();
    list.approve("b.example.com").unwrap();
    list.revoke("a.example.com").unwrap();
    let again = Approved::load(dir.path());
    assert!(!again.contains("a.example.com") && again.contains("b.example.com"));
}

#[test]
fn the_file_may_be_edited_by_hand_and_bad_lines_are_ignored() {
    let dir = TempDir::new();
    std::fs::write(
        dir.path().join(FILE),
        "# a comment\n  Tiles.Example.COM  \n\nnot a host!\nhttps://x.example.com/path\n[::1]\n",
    )
    .unwrap();
    let list = Approved::load(dir.path());
    assert_eq!(
        list.hosts().collect::<Vec<_>>(),
        vec!["[::1]", "tiles.example.com"]
    );
}

#[test]
fn another_games_approval_is_not_lost_by_this_ones() {
    let dir = TempDir::new();
    let mut mine = Approved::load(dir.path());
    let mut theirs = Approved::load(dir.path());
    theirs.approve("theirs.example.com").unwrap();
    mine.approve("mine.example.com").unwrap();
    let both = Approved::load(dir.path());
    assert!(both.contains("theirs.example.com") && both.contains("mine.example.com"));
}

#[test]
fn what_is_not_a_host_is_refused() {
    let mut list = Approved::of(&[]);
    for bad in ["", "a b", "a/b", "x@y.com", "https://x.com", "a:80"] {
        assert!(list.approve(bad).is_err(), "{bad:?}");
    }
    assert!(list.hosts().next().is_none());
}

fn earth_world(root: &Path, json: &str) {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("cubicgen.json"), json).unwrap();
    std::fs::write(root.join("horizon_cubic.json"), r#"{"type":"earth"}"#).unwrap();
}

#[test]
fn a_world_wanting_an_unapproved_host_is_reported_with_that_host() {
    let dir = TempDir::new();
    let world = dir.path().join("w");
    let wants = |allow: bool| {
        format!(
            r#"{{"originLatitude":1,"originLongitude":2,"tileUrl":"https://Tiles.Example.com:8443/{{z}}/{{x}}/{{y}}.png","allowDownloads":{allow}}}"#
        )
    };
    earth_world(&world, &wants(true));
    let none = Approved::of(&[]);
    if crate::cubic_settings::load_world_generation(&world).is_none() {
        // The world files are recognised by `cubic_settings`; this test needs them to be.
        panic!("test world is not read as an Earth world");
    }
    assert_eq!(
        pending_host(&world, &none).as_deref(),
        Some("tiles.example.com")
    );
    assert_eq!(
        pending_host(&world, &Approved::of(&["tiles.example.com"])),
        None
    );
    // A world that downloads nothing has nothing to approve.
    earth_world(&world, &wants(false));
    assert_eq!(pending_host(&world, &none), None);
}

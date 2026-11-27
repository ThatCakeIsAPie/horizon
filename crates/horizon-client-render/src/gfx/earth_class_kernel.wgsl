// The Earth far view's land-class kernel: horizon_earth::ClassWindowPlan::evaluate on the GPU, bit for bit.
// Appended after earth_class.wgsl (the exact f64 / u64 helpers). Every f64 expression copies the evaluator's
// (crates/horizon-earth/src/landfield_gpu.rs) in the same order; every product that feeds an addition or a
// subtraction goes through mul_nc so the driver cannot fuse it (it does, otherwise).
//
// One invocation per sample; workgroup (x: samples of a window in 64s, y: window of the batch).
//
// Buffers (all u64 words unless said):
//   params  : 0 nc_flag (never 12345), 1 nc_zero (always 0), 2 vote epsilon (1e-12) bits, 3 SPECK_KEEP bits,
//             4 (1 - f64::EPSILON) bits, 5 u32 words per pool slot, 8..264 class table
//             (LandClass::from_code(c).unwrap_or(NoData).code() for c in 0..256)
//   windows : 2 per window: job offset (word index in `jobs`), out base | samples << 32
//   jobs    : per window, see JOB LAYOUT below
//   pool    : u32 words; slot s holds a tile's pixels from word s * slot_words, 2 pixels a word, each pixel
//             16 bits: code | speck code << 8 (speck code 0: not a speck)
//   out     : u32 per sample: the class code, 255 when the plan does not hold the sample's file or tile
//
// JOB LAYOUT (word offsets from the job's start):
//   0  width | depth << 32
//   1  files | tiles << 32
//   2  coarse amplitude_x   3 coarse amplitude_z   4 fine amplitude_x   5 fine amplitude_z      (f64 bits)
//   6  coarse salt_x        7 coarse salt_z        8 fine salt_x        9 fine salt_z
//   10 metres_per_block  11 lat0  12 lon0  13 metres_per_degree_lon  14 earth_radius_m  15 radians_to_degrees
//   16 columns: 7 words each: pos, coarse (fade, term0, term1), fine (fade, term0, term1); then rows, alike
//   then files, 7 words each:
//      0 south (i32 bits) | west (i32 bits) << 32
//      1 no-data flag (absent / no level) | level_no << 32
//      2 width | height << 32     3 tile_width | tile_height << 32     4 tiles_across | tiles_down << 32
//      5 speck salt (already xored with 0x5EC45BEC00000001)
//      6 first tile entry | tile entries << 32
//   then tile entries, 2 words each: 0 col | row << 32, 1 pool slot (0xFFFFFFFF: blank tile, reads code 0)

@group(0) @binding(0) var<storage, read> params: array<u64>;
@group(0) @binding(1) var<storage, read> windows: array<u64>;
@group(0) @binding(2) var<storage, read> jobs: array<u64>;
@group(0) @binding(3) var<storage, read> pool: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<u32>;

fn nc_flag() -> u32 { return u32(params[0] & 0xFFFFFFFFlu); }
fn nc_zero() -> u32 { return u32(params[1] & 0xFFFFFFFFlu); }

// Replaced by the Rust side: true routes every division through div_cr (fma-corrected) instead of the native `/`.
const USE_DIV_CR: bool = false;

fn fdiv(a: f64, b: f64) -> f64 {
    if USE_DIV_CR { return div_cr(a, b); }
    return a / b;
}

const OUTSIDE: u32 = 255u;
const BLANK_SLOT: u32 = 0xFFFFFFFFu;
const AXIS_WORDS: u32 = 7u;
const FILE_WORDS: u32 = 7u;
const HEADER_WORDS: u32 = 16u;
const LATTICE_COLUMN_MUL: u64 = 0x9E3779B97F4A7C15lu;
const LATTICE_ROW_ADD: u64 = 0xD1B54A32D192ED03lu;
const CELL_MUL: u64 = 0xA24BAED4963EE407lu;
const CELL_XOR: u64 = 0x165667B19E3779F9lu;

fn lo32(x: u64) -> u32 { return u32(x & 0xFFFFFFFFlu); }
fn hi32(x: u64) -> u32 { return u32(x >> 32u); }
fn jf(i: u32) -> f64 { return bitcast<f64>(jobs[i]); }
fn pf(i: u32) -> f64 { return bitcast<f64>(params[i]); }
fn class_code(code: u32) -> u32 { return u32(params[8u + (code & 0xFFu)]); }

fn is_finite64(x: f64) -> bool { return (bitcast<u64>(x) & ~SIGN64) < 0x7FF0000000000000lu; }

// Rust f64::clamp for non-NaN x.
fn clamp64(x: f64, lo: f64, hi: f64) -> f64 {
    var r = x;
    if r < lo { r = lo; }
    if r > hi { r = hi; }
    return r;
}

// octave_noise: c and r index the octave's (fade, term0, term1) of the column and of the row.
fn noise(c: u32, r: u32, salt: u64) -> f64 {
    let cfade = jf(c);
    let ct0 = jobs[c + 1u];
    let ct1 = jobs[c + 2u];
    let rfade = jf(r);
    let rt0 = jobs[r + 1u];
    let rt1 = jobs[r + 2u];
    let a = unit53(mix64(salt ^ ct0 ^ rt0));
    let b = unit53(mix64(salt ^ ct1 ^ rt0));
    let cc = unit53(mix64(salt ^ ct0 ^ rt1));
    let d = unit53(mix64(salt ^ ct1 ^ rt1));
    let top = a + mul_nc(b - a, cfade);
    let bottom = cc + mul_nc(d - cc, cfade);
    return mul_nc(2.0lf, top + mul_nc(bottom - top, rfade)) - 1.0lf;
}

// crate::warp::cell_hash.
fn cell_hash(a: i64, b: i64, c: i64, salt: u64) -> f64 {
    let h = mix64(
        salt ^ (bitcast<u64>(a) * LATTICE_COLUMN_MUL)
            ^ mix64(bitcast<u64>(b) + LATTICE_ROW_ADD)
            ^ mix64((bitcast<u64>(c) * CELL_MUL) ^ CELL_XOR)
    );
    return unit53(h);
}

// Rust `v - v.floor()` clamped to [0, 1 - EPSILON], 0 when not finite.
fn fraction(v: f64) -> f64 {
    let f = v - floor(v);
    if is_finite64(f) {
        return clamp64(f, 0.0lf, pf(4u));
    }
    return 0.0lf;
}

// The pool slot of tile (col, row) of the file whose entries start at `first`; OUTSIDE when not in the plan.
fn find_slot(tiles: u32, first: u32, count: u32, col: u32, row: u32) -> u32 {
    let want = u64(col) | (u64(row) << 32u);
    for (var k = 0u; k < count; k++) {
        let e = tiles + (first + k) * 2u;
        if jobs[e] == want {
            return lo32(jobs[e + 1u]);
        }
    }
    return OUTSIDE;
}

fn pixel16(slot: u32, index: u32) -> u32 {
    let word = pool[slot * u32(params[5]) + index / 2u];
    return (word >> ((index & 1u) * 16u)) & 0xFFFFu;
}

struct FileView {
    lw: u32,
    lh: u32,
    tw: u32,
    th: u32,
    across: u32,
    down: u32,
    first: u32,
    count: u32,
}

// The code of pixel (x, y) of the level, clamped to it; 0 in a tile the level has not or that is blank;
// OUTSIDE for a tile the plan does not hold.
fn neighbour_code(f: FileView, tiles: u32, x: i64, y: i64) -> u32 {
    let cx = u32(clamp(x, 0li, i64(f.lw) - 1li));
    let cy = u32(clamp(y, 0li, i64(f.lh) - 1li));
    let col = cx / f.tw;
    let row = cy / f.th;
    let lx = cx % f.tw;
    let ly = cy % f.th;
    if col >= f.across || row >= f.down {
        return 0u;
    }
    let slot = find_slot(tiles, f.first, f.count, col, row);
    if slot == OUTSIDE {
        return OUTSIDE;
    }
    if slot == BLANK_SLOT {
        return 0u;
    }
    return pixel16(slot, ly * f.tw + lx) & 0xFFu;
}

fn sample(job: u32, width: u32, depth: u32, i: u32, j: u32) -> u32 {
    let col = job + HEADER_WORDS + i * AXIS_WORDS;
    let row = job + HEADER_WORDS + width * AXIS_WORDS + j * AXIS_WORDS;
    let files = job + HEADER_WORDS + (width + depth) * AXIS_WORDS;
    let nf = lo32(jobs[job + 1u]);
    let tiles = files + nf * FILE_WORDS;

    // The warp.
    let cx = mul_nc(jf(job + 2u), noise(col + 1u, row + 1u, jobs[job + 6u]));
    let cz = mul_nc(jf(job + 3u), noise(col + 1u, row + 1u, jobs[job + 7u]));
    let fx = mul_nc(jf(job + 4u), noise(col + 4u, row + 4u, jobs[job + 8u]));
    let fz = mul_nc(jf(job + 5u), noise(col + 4u, row + 4u, jobs[job + 9u]));
    let x = (jf(col) + cx) + fx;
    let z = (jf(row) + cz) + fz;

    // The projection (Tangent::to_lat_lon).
    let mpb = jf(job + 10u);
    let east = mul_nc(x, mpb);
    let north = mul_nc(-z, mpb);
    let lat = jf(job + 11u) + mul_nc(fdiv(north, jf(job + 14u)), jf(job + 15u));
    let lon = wrap_longitude(jf(job + 12u) + fdiv(east, jf(job + 13u)));

    // WorldCoverFile::containing.
    if !(is_finite64(lat) && is_finite64(lon) && abs(lat) <= 90.0lf) {
        return 0u;
    }
    let wrapped = wrap_longitude(lon);
    let fs = i32(min(mul_nc(floor(fdiv(lat, 3.0lf)), 3.0lf), 87.0lf));
    let fw = i32(clamp64(mul_nc(floor(fdiv(wrapped, 3.0lf)), 3.0lf), -180.0lf, 177.0lf));
    var file = OUTSIDE;
    for (var k = 0u; k < nf; k++) {
        let w = jobs[files + k * FILE_WORDS];
        if bitcast<i32>(lo32(w)) == fs && bitcast<i32>(hi32(w)) == fw {
            file = files + k * FILE_WORDS;
            break;
        }
    }
    if file == OUTSIDE {
        return OUTSIDE;
    }
    if lo32(jobs[file + 1u]) != 0u {
        return 0u;
    }
    var f: FileView;
    f.lw = lo32(jobs[file + 2u]);
    f.lh = hi32(jobs[file + 2u]);
    f.tw = lo32(jobs[file + 3u]);
    f.th = hi32(jobs[file + 3u]);
    f.across = lo32(jobs[file + 4u]);
    f.down = hi32(jobs[file + 4u]);
    let speck_salt = jobs[file + 5u];
    f.first = lo32(jobs[file + 6u]);
    f.count = hi32(jobs[file + 6u]);

    // WorldCoverFile::pixel and pixel_fraction.
    let east_off = wrap_longitude(lon - f64(fw));
    let fx_pixels = mul_nc(fdiv(east_off, 3.0lf), f64(f.lw));
    let north_edge = f64(fs) + 3.0lf;
    let fy_pixels = mul_nc(fdiv(north_edge - lat, 3.0lf), f64(f.lh));
    let pix_x = clamp_px(floor(fx_pixels), f.lw);
    let pix_y = clamp_px(floor(fy_pixels), f.lh);
    let inside_x = fraction(fx_pixels);
    let inside_y = fraction(fy_pixels);

    let tile_col = pix_x / f.tw;
    let tile_row = pix_y / f.th;
    if tile_col >= f.across || tile_row >= f.down {
        return 0u;
    }
    let slot = find_slot(tiles, f.first, f.count, tile_col, tile_row);
    if slot == OUTSIDE {
        return OUTSIDE;
    }
    if slot == BLANK_SLOT {
        return 0u;
    }
    let lx = pix_x % f.tw;
    let ly = pix_y % f.th;

    // LandTile::speck_kept.
    let speck = (pixel16(slot, ly * f.tw + lx) >> 8u) & 0xFFu;
    if speck != 0u {
        let cell_x = clamp(f64_to_i64_sat(mul_nc(inside_x, 8.0lf)), 0li, 7li);
        let cell_y = clamp(f64_to_i64_sat(mul_nc(inside_y, 8.0lf)), 0li, 7li);
        let key = cell_y * 8li + cell_x;
        let h = cell_hash(i64(pix_x), i64(pix_y), key, speck_salt);
        if h < pf(3u) {
            return class_code(speck);
        }
    }

    // The vote of the four pixels whose centres are nearest.
    let u = inside_x - 0.5lf;
    let v = inside_y - 0.5lf;
    let step_x = select(-1li, 1li, u >= 0.0lf);
    let step_y = select(-1li, 1li, v >= 0.0lf);
    let tx = abs(u);
    let ty = abs(v);
    let own_x = i64(pix_x);
    let own_y = i64(pix_y);
    var weights: array<f64, 4>;
    weights[0] = mul_nc(1.0lf - tx, 1.0lf - ty);
    weights[1] = mul_nc(tx, 1.0lf - ty);
    weights[2] = mul_nc(1.0lf - tx, ty);
    weights[3] = mul_nc(tx, ty);
    var codes: array<u32, 4>;
    codes[0] = neighbour_code(f, tiles, own_x, own_y);
    codes[1] = neighbour_code(f, tiles, own_x + step_x, own_y);
    codes[2] = neighbour_code(f, tiles, own_x, own_y + step_y);
    codes[3] = neighbour_code(f, tiles, own_x + step_x, own_y + step_y);
    var vote_code: array<u32, 4>;
    var vote_total: array<f64, 4>;
    var used = 0u;
    for (var k = 0u; k < 4u; k++) {
        let code = codes[k];
        if code == OUTSIDE {
            return OUTSIDE;
        }
        var found = false;
        for (var m = 0u; m < used; m++) {
            if vote_code[m] == code {
                vote_total[m] = vote_total[m] + weights[k];
                found = true;
                break;
            }
        }
        if !found {
            vote_code[used] = code;
            vote_total[used] = weights[k];
            used = used + 1u;
        }
    }
    var best_code = vote_code[0];
    var best_total = vote_total[0];
    let eps = pf(2u);
    for (var m = 1u; m < used; m++) {
        if vote_total[m] > best_total + eps {
            best_code = vote_code[m];
            best_total = vote_total[m];
        }
    }
    return class_code(best_code);
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.y;
    let job = u32(windows[2u * w]);
    let ob = windows[2u * w + 1u];
    let n = hi32(ob);
    let s = gid.x;
    if s >= n {
        return;
    }
    let dims = jobs[job];
    let width = lo32(dims);
    let depth = hi32(dims);
    out[lo32(ob) + s] = sample(job, width, depth, s % width, s / width);
}

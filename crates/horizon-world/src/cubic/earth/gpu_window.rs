//! A hook for computing the padded window's elevation samples somewhere else (a GPU), without this crate knowing
//! how: [`WindowGpu`] takes a [`WindowJob`] (the separable plan with the tiles it reads) and answers later with the
//! heights, so a far worker can queue the job, do its land cover and river work on the CPU, and only then wait.
//!
//! The generator holds the hook in a slot that can be filled after it is built ([`EarthGenerator::set_window_gpu`]),
//! and whether it is used is a flag ([`EarthGenerator::set_window_gpu_enabled`]) the environment variable
//! `HORIZON_EARTH_GPU_WINDOW` (`0` or `1`) overrides. Whatever goes wrong (no hook, switched off, the hook declines
//! the job, the wait gives nothing or a list of the wrong length) the generator reads the window on the CPU from
//! the same plan; the CPU answer is the reference, so the hook only ever saves time.
//!
//! This crate has no GPU dependency: the implementation lives with the renderer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, PoisonError, RwLock};

use horizon_earth::WindowPlan;

use super::generator::EarthGenerator;

/// The environment variable that forces the hook off (`0`) or on (`1`), whatever the setting says.
pub const WINDOW_GPU_ENV: &str = "HORIZON_EARTH_GPU_WINDOW";

/// One padded window to compute.
///
/// Today it is the elevation only. **Extending later:** a land cover part adds an optional field here (for example
/// `class: Option<ClassJob>`) and [`WindowResult`] gains `classes: Option<Vec<LandClass>>`; a hook that does not
/// do classes leaves it `None` and the generator keeps computing the classes on the CPU.
#[derive(Clone, Debug)]
pub struct WindowJob {
    /// The samples and the tiles they read.
    pub plan: Arc<WindowPlan>,
}

/// What a [`WindowGpu`] made of a [`WindowJob`].
#[derive(Clone, Debug)]
pub struct WindowResult {
    /// The elevations in metres, row by row (`plan.width * plan.depth` of them): what
    /// [`WindowPlan::heights_m`] gives, to within whatever precision the hook documents. The generator turns them
    /// into block heights itself.
    pub heights_m: Vec<f32>,
    // Extending later: `pub classes: Option<Vec<LandClass>>`, present when the job asked for the classes.
}

/// A job in flight.
pub trait WindowTicket: Send {
    /// Blocks until the answer is ready. `None` (the device is lost, the batch failed) sends the caller to the
    /// CPU path. Dropping a ticket without waiting abandons the job.
    fn wait(self: Box<Self>) -> Option<WindowResult>;
}

/// Computes windows on another device. Called from many far workers at once.
pub trait WindowGpu: Send + Sync {
    /// Queues the job and returns at once; `None` means the hook cannot take it now (the caller uses the CPU).
    fn submit(&self, job: WindowJob) -> Option<Box<dyn WindowTicket>>;
}

/// The generator's hook slot and switch.
#[derive(Default)]
pub(super) struct WindowGpuSlot {
    hook: RwLock<Option<Arc<dyn WindowGpu>>>,
    on: AtomicBool,
}

impl WindowGpuSlot {
    /// The hook, when one is installed and the switch is on.
    pub(super) fn active(&self) -> Option<Arc<dyn WindowGpu>> {
        if !self.enabled() {
            return None;
        }
        self.hook
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn enabled(&self) -> bool {
        env_override().unwrap_or_else(|| self.on.load(Ordering::Relaxed))
    }
}

/// Whether the game offers windows to a GPU hook when the environment does not say. Off: measured on a GTX 1070 Ti
/// (examples/earth_window_bench in horizon-client-render, warm tiles, ms per node), the separable CPU read costs
/// 0.016 of thread CPU while the GPU costs 0.014 plus 0.22-0.30 blocked in the round trip; the whole `columns_at`
/// with 6 workers took 1.53 CPU / 1.67 wall with the hook off and 1.65 / 1.89 with it on.
pub const WINDOW_GPU_DEFAULT_ON: bool = false;

/// What `HORIZON_EARTH_GPU_WINDOW` says, read once: `Some(true)` for `1`, `Some(false)` for `0`, else `None`.
pub fn window_gpu_env_override() -> Option<bool> {
    env_override()
}

/// Whether the hook should run, given the game's GPU far terrain setting: the environment variable when set, else
/// [`WINDOW_GPU_DEFAULT_ON`] and the setting.
pub fn window_gpu_wanted(setting_on: bool) -> bool {
    env_override().unwrap_or(WINDOW_GPU_DEFAULT_ON && setting_on)
}

/// Whether a hook could ever run in this process ([`window_gpu_wanted`] for some setting), so a caller can skip
/// making one.
pub fn window_gpu_possible() -> bool {
    env_override().unwrap_or(WINDOW_GPU_DEFAULT_ON)
}

/// `HORIZON_EARTH_GPU_WINDOW`: `Some(false)` for `0`, `Some(true)` for `1`, read once.
fn env_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| match std::env::var(WINDOW_GPU_ENV).ok()?.trim() {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    })
}

impl EarthGenerator {
    /// Installs (or removes, with `None`) the hook that computes padded windows. It takes effect while
    /// [`EarthGenerator::window_gpu_enabled`]; the generator is shared, so this works through `&self`.
    pub fn set_window_gpu(&self, gpu: Option<Arc<dyn WindowGpu>>) {
        *self
            .window_gpu
            .hook
            .write()
            .unwrap_or_else(PoisonError::into_inner) = gpu;
    }

    /// Switches the hook on or off (off until told otherwise). The environment variable `HORIZON_EARTH_GPU_WINDOW`
    /// (`0` or `1`), when set, wins over this.
    pub fn set_window_gpu_enabled(&self, on: bool) {
        self.window_gpu.on.store(on, Ordering::Relaxed);
    }

    /// Whether windows are offered to the hook: the switch (or the environment variable) is on. It says nothing
    /// about whether a hook is installed.
    pub fn window_gpu_enabled(&self) -> bool {
        self.window_gpu.enabled()
    }
}

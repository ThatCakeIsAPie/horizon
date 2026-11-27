//! Earth's far-view padded windows (elevation and land cover classes, both exact) on the renderer's device
//! (`gfx::earth_window`), behind the GPU far terrain setting and `HORIZON_EARTH_GPU_WINDOW`
//! ([`horizon_world::cubic::earth::window_gpu_wanted`]). Off by default (slower in game); with `=1`, only when the device can do the classes too
//! (`SHADER_F64 | SHADER_INT64`): the elevation alone is measured slower than the exact CPU read.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use horizon_world::cubic::earth::{
    EarthGenerator, WindowGpu, WindowJob, WindowTicket, window_gpu_env_override, window_gpu_possible,
    window_gpu_wanted,
};

use crate::gfx::earth_window::EarthWindowGpu;

/// The GPU far terrain setting as the game last passed it ([`set_setting`]).
static SETTING: AtomicBool = AtomicBool::new(false);

/// Tells the hooks the GPU far terrain setting (each update; a relaxed store).
pub fn set_setting(gpu_far: bool) {
    SETTING.store(gpu_far, Ordering::Relaxed);
}

/// The device's sampler, taking jobs only while [`window_gpu_wanted`] says so.
struct Gated(Arc<EarthWindowGpu>);

impl WindowGpu for Gated {
    fn submit(&self, job: WindowJob) -> Option<Box<dyn WindowTicket>> {
        if !window_gpu_wanted(SETTING.load(Ordering::Relaxed)) {
            return None;
        }
        self.0.submit(job)
    }
}

/// Installs the window sampler on `earth` when it could ever be used (nothing is made otherwise).
pub(super) fn install(earth: &EarthGenerator, device: &wgpu::Device, queue: &wgpu::Queue) {
    if !window_gpu_possible() {
        return;
    }
    let Some(gpu) = EarthWindowGpu::new(device.clone(), queue.clone()) else {
        log::info!("Horizon: Earth window sampler not on the GPU (no compute)");
        return;
    };
    if !gpu.does_classes() && window_gpu_env_override() != Some(true) {
        log::info!(
            "Horizon: Earth windows stay on the CPU (the device cannot do the land cover classes exactly; \
             the elevation alone is slower on the GPU)"
        );
        return;
    }
    log::info!(
        "Horizon: Earth window elevation on the GPU ({}; land cover classes {})",
        if gpu.uses_f64() { "f64, exact" } else { "f32" },
        if gpu.does_classes() { "too, exact" } else { "on the CPU" }
    );
    earth.set_window_gpu(Some(Arc::new(Gated(gpu))));
    earth.set_window_gpu_enabled(true);
}

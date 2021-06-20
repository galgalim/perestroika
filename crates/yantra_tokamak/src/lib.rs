use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use ash::vk;
use ash::vk::EntryFnV1_0;
use ash::vk::InstanceFnV1_0;
pub use ash::{Device, EntryCustom, Instance};
use std::borrow::Cow;
use std::cell::RefCell;
use std::default::Default;
use std::ffi::{CStr, CString};
use std::ops::Drop;
use std::os::raw::c_char;
use std::ptr;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

pub const APPLICATION_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
pub const ENGINE_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
pub const API_VERSION: u32 = vk::make_api_version(0, 1, 0, 92);

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub const IS_PAINT_FPS_COUNTER: bool = false;

pub struct Tokamak {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl Tokamak {
    pub fn init(window: &Window) -> Tokamak {
        unsafe {
            let entry = ash::Entry::new().unwrap();
            let instance = Self::create_instance(&entry, &window);
            Tokamak { entry, instance }
        }
    }

    fn create_instance(entry: &ash::Entry, window: &Window) -> ash::Instance {
        let app_name = CString::new("Yantra").unwrap();
        let engine_name = CString::new("Yantra").unwrap();

        let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
        let layers_names_raw: Vec<*const i8> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();
        let mut extension_names_raw = surface_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();
        extension_names_raw.push(DebugUtils::name().as_ptr());

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(APPLICATION_VERSION)
            .engine_name(&engine_name)
            .engine_version(ENGINE_VERSION)
            .api_version(API_VERSION)
            .build();

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names_raw)
            .build();

        let instance: ash::Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance!")
        };

        instance
    }

    pub fn draw_frame(&mut self) {
        // Drawing will be here
    }

    pub fn recreate_swapchain<P: winit::dpi::Pixel>(
        &mut self,
        dims: Option<winit::dpi::PhysicalSize<P>>,
    ) {
    }
}

impl Drop for Tokamak {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

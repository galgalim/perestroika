#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(not(any(feature = "dx12", feature = "metal")))]
extern crate gfx_backend_vulkan as back;

use hal::Instance;
use image::RgbaImage;
use log::{debug, info, trace, warn};
use nalgebra_glm as glm;
use std::io::{stdin, Cursor, Read, Write};
use std::thread;
use std::time::Instant;

pub use logging::setup_logging;

pub mod audio_render;
pub mod camera;
pub mod experimental;
pub mod game;
pub mod input;
pub mod logging;
pub mod ui;
pub mod video_render;
pub mod window;

pub use game::{TileType, World};

pub fn run_perestroika() {
    std::env::set_var("RUST_BACKTRACE", "1");
    //let audio_thread = std::thread::spawn(|| audio_render::run_audio_system());

    let event_loop = winit::event_loop::EventLoop::new();

    let wb = winit::window::WindowBuilder::new()
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
            64.0, 64.0,
        )))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            1920, 1080,
        )))
        .with_title("perestroika".to_string());

    // instantiate backend
    let (_window, instance, mut adapters, surface) = {
        let window = wb.build(&event_loop).unwrap();
        let instance =
            back::Instance::create("perestroika", 1).expect("Failed to create an instance!");
        let adapters = instance.enumerate_adapters();
        let surface = unsafe {
            instance
                .create_surface(&window)
                .expect("Failed to create a surface!")
        };
        // Return `window` so it is not dropped: dropping it invalidates `surface`.
        (window, Some(instance), adapters, surface)
    };

    for adapter in &adapters {
        debug!("{:?}", adapter.info);
    }

    let adapter = adapters.remove(0);

    info!("Generating World...");
    let mut world = World::new(32);
    world.map[0] = TileType::Dirt;
    let (world_width, world_length) = world.dimensions();

    let mut map_img = RgbaImage::new(world_width, world_length);
    for (x, y, pixel) in map_img.enumerate_pixels_mut() {
        let tile = &world.map[((y * world_width) + x) as usize];
        match tile {
            TileType::Grass => {
                *pixel = image::Rgba([0, 125, 0, 255]);
            }
            TileType::Dirt => {
                *pixel = image::Rgba([139, 69, 19, 255]);
            }
            TileType::Desert => {
                *pixel = image::Rgba([255, 222, 173, 255]);
            }
            TileType::Water => {
                *pixel = image::Rgba([0, 0, 255, 255]);
            }
        }
    }

    let mut renderer = video_render::Renderer::new(instance, surface, adapter, map_img)
        .expect("Failed to create Renderer!");

    renderer.render();

    event_loop.run(move |event, _, control_flow| match event {
        winit::event::Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::CloseRequested => {
                *control_flow = winit::event_loop::ControlFlow::Exit
            }
            winit::event::WindowEvent::KeyboardInput {
                input:
                    winit::event::KeyboardInput {
                        virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = winit::event_loop::ControlFlow::Exit,
            winit::event::WindowEvent::Resized(dims) => {
                debug!("resized to {:?}", dims);
                renderer.dimensions = hal::window::Extent2D {
                    width: dims.width,
                    height: dims.height,
                };
                renderer.recreate_swapchain();
            }
            _ => {}
        },
        winit::event::Event::RedrawEventsCleared => {
            renderer.render();
        }
        _ => {}
    });
}

#[cfg(test)]
mod test {
    #[test]
    fn it_works() {
        assert_eq!(1 + 1, 2);
    }
}

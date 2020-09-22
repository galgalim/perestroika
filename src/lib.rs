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
use std::collections::HashSet;
use std::io::{stdin, Cursor, Read, Write};
use std::thread;
use std::time::Instant;
use winit::event::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::ControlFlow;

pub use logging::setup_logging;

pub mod audio_render;
pub mod camera;
pub mod experimental;
pub mod game;
pub mod logging;
pub mod ui;
pub mod video_render;

use camera::Camera;
pub use game::{TileType, World};
use video_render::view_projection;

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
    let (window, instance, mut adapters, surface) = {
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

    let view_projection = view_projection();
    let models = vec![
        glm::identity(),
        glm::translate(&glm::identity(), &glm::make_vec3(&[1.5, 0.1, 0.0])),
        glm::translate(&glm::identity(), &glm::make_vec3(&[-3.0, 2.0, 3.0])),
        glm::translate(&glm::identity(), &glm::make_vec3(&[0.5, -4.0, 4.0])),
        glm::translate(&glm::identity(), &glm::make_vec3(&[-3.4, -2.3, 1.0])),
        glm::translate(&glm::identity(), &glm::make_vec3(&[-2.8, -0.7, 5.0])),
    ];
    let (mut frame_width, mut frame_height): (f32, f32) = window.inner_size().into();
    let camera = Camera::at_position(glm::make_vec3(&[0.0, 0.0, -5.0]));
    let mut keys_held: HashSet<VirtualKeyCode> = HashSet::new();
    let mut grabbed = false;
    let mut orientation_change: (f32, f32) = (0.0, 0.0);
    let mut new_frame_size: Option<(f64, f64)> = None;
    let mut seconds: f32 = 0.0;
    let mut last_timestamp = Instant::now();
    let mut spare_time = 0.0;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = winit::event_loop::ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = winit::event_loop::ControlFlow::Exit,
                WindowEvent::Resized(dims) => {
                    debug!("resized to {:?}", dims);
                    renderer.dimensions = hal::window::Extent2D {
                        width: dims.width,
                        height: dims.height,
                    };
                    new_frame_size = Some((dims.width as f64, dims.height as f64));
                    renderer.recreate_swapchain();
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state,
                            virtual_keycode: Some(code),
                            ..
                        },
                    ..
                } => {
                    #[cfg(feature = "metal")]
                    {
                        match state {
                            ElementState::Pressed => keys_held.insert(code),
                            ElementState::Released => keys_held.remove(&code),
                        }
                    };
                    if state == ElementState::Pressed {
                        match code {
                            VirtualKeyCode::Escape => {
                                if grabbed {
                                    debug!("Escape pressed while grabbed, releasing the mouse!");
                                    window
                                        .set_cursor_grab(false)
                                        .expect("Failed to release the mouse grab!");
                                    window.set_cursor_visible(true);
                                    grabbed = false;
                                }
                            }
                            _ => (),
                        }
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    if grabbed {
                        debug!("Click! We already have the mouse grabbed.");
                    } else {
                        debug!("Click! Grabbing the mouse.");
                        window
                            .set_cursor_grab(true)
                            .expect("Failed to grab the mouse!");
                        window.set_cursor_visible(false);
                        grabbed = true;
                    }
                }
                WindowEvent::Focused(false) => {
                    if grabbed {
                        debug!("Lost Focus, releasing the mouse grab...");
                        window
                            .set_cursor_grab(false)
                            .expect("Failed to release the mouse grab!");
                        window.set_cursor_visible(true);
                        grabbed = false;
                    } else {
                        debug!("Lost Focus when mouse wasn't grabbed.");
                    }
                }
                _ => {}
            },
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                ..
            } => {
                if grabbed {
                    orientation_change.0 -= dx as f32;
                    orientation_change.1 -= dy as f32;
                }
            }
            Event::RedrawEventsCleared => {
                renderer.render(&view_projection, &models).unwrap();
            }
            _ => {}
        }
        seconds = {
            let now = Instant::now();
            let duration = now.duration_since(last_timestamp);
            last_timestamp = now;
            duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9
        };
        keys_held = if grabbed {
            keys_held.clone()
        } else {
            HashSet::new()
        }
    });
}

#[cfg(test)]
mod test {
    #[test]
    fn it_works() {
        assert_eq!(1 + 1, 2);
    }
}

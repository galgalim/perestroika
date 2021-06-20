use image::RgbaImage;
use log::{debug, error, info, trace};
use nalgebra_glm as glm;
use std::collections::HashSet;
use std::time::Instant;
use winit::event::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::ControlFlow;
use yantra::{TileType, World};
use yantra_log::setup_logging;
use yantra_tokamak::Tokamak;

fn main() {
    setup_logging(1).expect("failed to initialize logging.");
    info!("Starting Yantra...");
    std::env::set_var("RUST_BACKTRACE", "full");

    let event_loop = winit::event_loop::EventLoop::new();

    let wb = winit::window::WindowBuilder::new()
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
            64.0, 64.0,
        )))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            1920, 1080,
        )))
        .with_title("Yantra".to_string());

    let window = wb.build(&event_loop).unwrap();

    let mut tokamak = Tokamak::init(&window);

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
                    tokamak.recreate_swapchain(Some(dims));
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
                tokamak.draw_frame();
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

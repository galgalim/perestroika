use crate::window::WinitState;
use log::debug;
use std::collections::HashSet;
use std::time::Instant;
use winit::event::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent,
};

#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub end_requested: bool,
    pub new_frame_size: Option<(f64, f64)>,
    pub swap_projection: bool,
    pub keys_held: HashSet<VirtualKeyCode>,
    pub orientation_change: (f32, f32),
    pub seconds: f32,
}

impl UserInput {
    /*
    pub fn poll_events_loop(winit_state: &mut WinitState, last_timestamp: &mut Instant) -> Self {
        let mut output = UserInput::default();
        // We have to manually split the borrow here. rustc, why you so dumb sometimes?
        let events_loop = &mut winit_state.events_loop;
        let window = &mut winit_state.window;
        let keys_held = &mut winit_state.keys_held;
        let grabbed = &mut winit_state.grabbed;
        // now we actually poll those events
        events_loop.poll_events(|event| match event {
            // Close when asked
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => output.end_requested = true,

            // Track all keys, all the time. Note that because of key rollover details
            // it's possible to get key released events for keys we don't think are
            // pressed. This is a hardware limit, not something you can evade.
            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        virtual_keycode: Some(code),
                        state,
                        ..
                    }),
                ..
            } => drop(match state {
                ElementState::Pressed => keys_held.insert(code),
                ElementState::Released => keys_held.remove(&code),
            }),

            // We want to respond to some of the keys specially when they're also
            // window events too (meaning that the window was focused when the event
            // happened).
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(code),
                                ..
                            },
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
                        VirtualKeyCode::Tab => output.swap_projection = !output.swap_projection,
                        VirtualKeyCode::Escape => {
                            if *grabbed {
                                debug!("Escape pressed while grabbed, releasing the mouse!");
                                window
                                    .set_cursor_grab(false)
                                    .expect("Failed to release the mouse grab!");
                                window.set_cursor_visible(true);
                                *grabbed = false;
                            }
                        }
                        _ => (),
                    }
                }
            }

            // Always track the mouse motion, but only update the orientation if
            // we're "grabbed".
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                ..
            } => {
                if *grabbed {
                    output.orientation_change.0 -= dx as f32;
                    output.orientation_change.1 -= dy as f32;
                }
            }

            // Left clicking in the window causes the mouse to get grabbed
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    },
                ..
            } => {
                if *grabbed {
                    debug!("Click! We already have the mouse grabbed.");
                } else {
                    debug!("Click! Grabbing the mouse.");
                    window
                        .set_cursor_grab(true)
                        .expect("Failed to grab the mouse!");
                    window.set_cursor_visible(false);
                    *grabbed = true;
                }
            }

            // Automatically release the mouse when focus is lost
            Event::WindowEvent {
                event: WindowEvent::Focused(false),
                ..
            } => {
                if *grabbed {
                    debug!("Lost Focus, releasing the mouse grab...");
                    window
                        .set_cursor_grab(false)
                        .expect("Failed to release the mouse grab!");
                    window.set_cursor_visible(true);
                    *grabbed = false;
                } else {
                    debug!("Lost Focus when mouse wasn't grabbed.");
                }
            }

            // Update our size info if the window changes size.
            Event::WindowEvent {
                event: WindowEvent::Resized(logical),
                ..
            } => {
                output.new_frame_size = Some((logical.width as f64, logical.height as f64));
            }

            _ => (),
        });
        output.seconds = {
            let now = Instant::now();
            let duration = now.duration_since(*last_timestamp);
            *last_timestamp = now;
            duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9
        };
        output.keys_held = if *grabbed {
            keys_held.clone()
        } else {
            HashSet::new()
        };
        output
    }
    */
}

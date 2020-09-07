use std::collections::HashSet;
use winit::dpi::{LogicalSize, Size};
use winit::error::OsError;
use winit::event::VirtualKeyCode;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

pub const WINDOW_NAME: &str = "Instanced Drawing";

#[derive(Debug)]
pub struct WinitState {
    pub event_loop: EventLoop<()>,
    pub window: Window,
    pub keys_held: HashSet<VirtualKeyCode>,
    pub grabbed: bool,
}

impl WinitState {
    pub fn new<T: Into<String>, S: Into<Size>>(title: T, size: S) -> Result<Self, OsError> {
        let event_loop = EventLoop::new();
        let output = WindowBuilder::new()
            .with_inner_size(size)
            .with_title(title)
            .build(&event_loop);
        output.map(|window| Self {
            event_loop,
            window,
            grabbed: false,
            keys_held: HashSet::new(),
        })
    }
}

impl Default for WinitState {
    fn default() -> Self {
        Self::new(
            WINDOW_NAME,
            LogicalSize {
                width: 800.0,
                height: 600.0,
            },
        )
        .expect("Could not create a window!")
    }
}

use nalgebra_glm as glm;
use std::collections::HashSet;
use winit::dpi::{LogicalSize, Size};
use winit::error::OsError;
use winit::event::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: glm::TVec3<f32>,
    pitch_deg: f32,
    yaw_deg: f32,
}

impl Camera {
    const UP: [f32; 3] = [0.0, 1.0, 0.0];

    fn make_front(&self) -> glm::TVec3<f32> {
        let pitch_rad = f32::to_radians(self.pitch_deg);
        let yaw_rad = f32::to_radians(self.yaw_deg);
        glm::make_vec3(&[
            yaw_rad.sin() * pitch_rad.cos(),
            pitch_rad.sin(),
            yaw_rad.cos() * pitch_rad.cos(),
        ])
    }

    pub fn update_orientation(&mut self, d_pitch_deg: f32, d_yaw_deg: f32) {
        self.pitch_deg = (self.pitch_deg + d_pitch_deg).max(-89.0).min(89.0);
        self.yaw_deg = (self.yaw_deg + d_yaw_deg) % 360.0;
    }

    pub fn update_position(&mut self, keys: &HashSet<VirtualKeyCode>, distance: f32) {
        let up = glm::make_vec3(&Self::UP);
        let forward = self.make_front();
        let cross_normalized = glm::cross::<f32, glm::U3>(&forward, &up).normalize();
        let mut move_vector = keys
            .iter()
            .fold(glm::make_vec3(&[0.0, 0.0, 0.0]), |vec, key| match *key {
                VirtualKeyCode::W => vec + forward,
                VirtualKeyCode::S => vec - forward,
                VirtualKeyCode::A => vec + cross_normalized,
                VirtualKeyCode::D => vec - cross_normalized,
                VirtualKeyCode::E => vec + up,
                VirtualKeyCode::Q => vec - up,
                _ => vec,
            });
        if move_vector != glm::zero() {
            move_vector = move_vector.normalize();
            self.position += move_vector * distance;
        }
    }

    pub fn make_view_matrix(&self) -> glm::TMat4<f32> {
        glm::look_at_lh(
            &self.position,
            &(self.position + self.make_front()),
            &glm::make_vec3(&Self::UP),
        )
    }

    pub const fn at_position(position: glm::TVec3<f32>) -> Self {
        Self {
            position,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
        }
    }
}

use crate::camera::Camera;
use crate::input::UserInput;
use nalgebra_glm as glm;
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct LocalState {
    pub frame_width: f64,
    pub frame_height: f64,
    pub cubes: Vec<glm::TMat4<f32>>,
    pub camera: Camera,
    pub perspective_projection: glm::TMat4<f32>,
    pub spare_time: f32,
}

impl LocalState {
    pub fn update_from_input(&mut self, input: UserInput) {
        if let Some(frame_size) = input.new_frame_size {
            self.frame_width = frame_size.0;
            self.frame_height = frame_size.1;
        }
        assert!(self.frame_width != 0.0 && self.frame_height != 0.0);
        self.spare_time += input.seconds;
        const ONE_SIXTIETH: f32 = 1.0 / 60.0;
        while self.spare_time > 0.0 {
            for (i, cube_mut) in self.cubes.iter_mut().enumerate() {
                let r = ONE_SIXTIETH * 30.0 * (i as f32 + 1.0) / 1 as f32;
                *cube_mut = glm::rotate(
                    &cube_mut,
                    f32::to_radians(r),
                    &glm::make_vec3(&[0.3, 0.4, 0.5]).normalize(),
                );
            }
            self.spare_time -= ONE_SIXTIETH;
        }
        const MOUSE_SENSITIVITY: f32 = 0.05;
        let d_pitch_deg = input.orientation_change.1 * MOUSE_SENSITIVITY;
        let d_yaw_deg = -input.orientation_change.0 * MOUSE_SENSITIVITY;
        self.camera.update_orientation(d_pitch_deg, d_yaw_deg);
        self.camera
            .update_position(&input.keys_held, 5.0 * input.seconds);
    }
}

#[derive(Debug, Clone)]
pub enum TileType {
    Grass,
    Dirt,
    Water,
    Desert,
}

#[derive(Debug)]
pub struct World {
    rng: ThreadRng,
    size: u32,
    pub map: Vec<TileType>,
    people: Vec<Person>,
    entity_count: u64,
}

impl World {
    pub fn new(size: u32) -> World {
        let rng = thread_rng();
        let mut map = vec![TileType::Grass; (size * size) as usize];
        for (i, tile) in map.iter_mut().enumerate() {
            let y = (i as f32 / size as f32).floor() as usize;
            let x = i - (y * size as usize);
            if y % 2 == 0 {
                if x % 2 == 0 {
                    *tile = TileType::Dirt;
                } else {
                    *tile = TileType::Grass;
                }
            } else {
                if x % 2 == 0 {
                    *tile = TileType::Grass;
                } else {
                    *tile = TileType::Dirt;
                }
            }
        }
        World {
            rng,
            size,
            people: vec![],
            map,
            entity_count: 0,
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.size, self.size)
    }

    pub fn spawn_person(&mut self, x: i32, y: i32) -> u64 {
        let id = self.entity_count + 1;
        self.people.push(Person::new(id, x, y));
        self.entity_count = id;
        return id;
    }

    pub fn get_person(&self, id: u64) -> Option<Person> {
        for person in self.people.iter() {
            if person.id == id {
                return Some(person.clone());
            }
        }

        None
    }

    pub fn despawn_person(&mut self, id: u64) {
        let mut id_to_delete = None;
        for (i, person) in self.people.iter_mut().enumerate() {
            if person.id == id {
                id_to_delete = Some(i);
            }
        }

        if let Some(i) = id_to_delete {
            self.people.remove(i);
        }
    }

    pub fn send_action(&mut self, action: Action) {}

    pub fn update(&mut self) {}
}

#[derive(Debug, Clone)]
pub struct Person {
    id: u64,
    x: i32,
    y: i32,
    needs: Needs,
    schedule: Vec<Action>,
}

impl Person {
    pub fn new(id: u64, x: i32, y: i32) -> Person {
        Person {
            id,
            x,
            y,
            needs: Needs::default(),
            schedule: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub enum Action {
    MoveTo(Position),
}

#[derive(Debug, Clone)]
pub struct Position {
    x: u32,
    y: u32,
}

#[derive(Debug, Clone)]
pub enum ActionType {
    MoveTo,
}

#[derive(Debug, Clone, Default)]
pub struct Needs {
    pub subsistence: u8,
    pub protection: u8,
    pub affection: u8,
    pub understanding: u8,
    pub participation: u8,
    pub idleness: u8,
    pub creation: u8,
    pub identity: u8,
    pub freedom: u8,
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn it_works() {
        let mut world = World::new(64);
        let person_1 = world.spawn_person(0, 0);
        let person_2 = world.spawn_person(0, 0);
        let person_3 = world.spawn_person(0, 0);
        assert_eq!(world.people.len(), 3);
        world.despawn_person(person_1);
        assert_eq!(world.people.len(), 2);
        world.despawn_person(person_2);
        assert_eq!(world.people.len(), 1);
        world.despawn_person(person_3);
        assert_eq!(world.people.len(), 0);
    }
}

use rand::prelude::*;

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
            map,
            entity_count: 0,
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.size, self.size)
    }

    pub fn update(&mut self) {}
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

use super::DIMS;
use core::{marker::PhantomData, mem::ManuallyDrop};
use gfx_hal::{
    adapter::Adapter, format as f, image as i, memory as m, prelude::*, window::Extent2D,
    MemoryTypeId,
};

pub struct DepthImage<B: gfx_hal::Backend, D: gfx_hal::device::Device<B>> {
    pub image: ManuallyDrop<B::Image>,
    pub requirements: m::Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub image_view: ManuallyDrop<B::ImageView>,
    pub phantom: PhantomData<D>,
}

impl<B: gfx_hal::Backend, D: gfx_hal::device::Device<B>> DepthImage<B, D> {
    pub fn new(adapter: &Adapter<B>, device: &D, extent: Extent2D) -> Result<Self, &'static str> {
        unsafe {
            let mut the_image = device
                .create_image(
                    gfx_hal::image::Kind::D2(extent.width, extent.height, 1, 1),
                    1,
                    f::Format::D32Sfloat,
                    gfx_hal::image::Tiling::Optimal,
                    gfx_hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
                    m::SparseFlags::empty(),
                    gfx_hal::image::ViewCapabilities::empty(),
                )
                .map_err(|_| "Couldn't crate the image!")?;
            let requirements = device.get_image_requirements(&the_image);
            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(id, memory_type)| {
                    // BIG NOTE: THIS IS DEVICE LOCAL NOT CPU VISIBLE
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(m::Properties::DEVICE_LOCAL)
                })
                .map(|(id, _)| MemoryTypeId(id))
                .ok_or("Couldn't find a memory type to support the image!")?;
            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Couldn't allocate image memory!")?;
            device
                .bind_image_memory(&memory, 0, &mut the_image)
                .map_err(|_| "Couldn't bind the image memory!")?;
            let image_view = device
                .create_image_view(
                    &the_image,
                    gfx_hal::image::ViewKind::D2,
                    f::Format::D32Sfloat,
                    gfx_hal::format::Swizzle::NO,
                    gfx_hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
                    i::SubresourceRange {
                        aspects: f::Aspects::DEPTH,
                        ..Default::default()
                    },
                )
                .map_err(|_| "Couldn't create the image view!")?;
            Ok(Self {
                image: ManuallyDrop::new(the_image),
                requirements,
                memory: ManuallyDrop::new(memory),
                image_view: ManuallyDrop::new(image_view),
                phantom: PhantomData,
            })
        }
    }
    pub unsafe fn manually_drop(&self, device: &D) {
        use core::ptr::read;
        device.destroy_image_view(ManuallyDrop::into_inner(read(&self.image_view)));
        device.destroy_image(ManuallyDrop::into_inner(read(&self.image)));
        device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
    }
}

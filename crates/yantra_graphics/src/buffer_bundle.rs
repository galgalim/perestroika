use core::{marker::PhantomData, mem::ManuallyDrop};
use gfx_hal::{buffer, command, memory as m, prelude::*, queue::QueueGroup};
use std::iter;

pub struct BufferBundle<B: gfx_hal::Backend, D: gfx_hal::device::Device<B>> {
    pub buffer: ManuallyDrop<B::Buffer>,
    pub requirements: m::Requirements,
    pub memory_type_id: gfx_hal::MemoryTypeId,
    pub memory: ManuallyDrop<B::Memory>,
    pub phantom: PhantomData<D>,
}

impl<B: gfx_hal::Backend, D: gfx_hal::device::Device<B>> BufferBundle<B, D> {
    pub fn new(
        device: &D,
        size: u64,
        usage: buffer::Usage,
        memory_types: &[gfx_hal::adapter::MemoryType],
        memory_props: m::Properties,
    ) -> Result<Self, &'static str> {
        unsafe {
            let mut buffer = device
                .create_buffer(size, usage, m::SparseFlags::empty())
                .map_err(|_| "Couldn't create a buffer!")?;
            let requirements = device.get_buffer_requirements(&buffer);
            let memory_type_id: gfx_hal::MemoryTypeId = memory_types
                .iter()
                .enumerate()
                .position(|(id, mem_type)| {
                    requirements.type_mask & (1 << id) != 0
                        && mem_type.properties.contains(memory_props)
                })
                .unwrap()
                .into();

            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Couldn't allocate buffer memory!")?;
            device
                .bind_buffer_memory(&memory, 0, &mut buffer)
                .map_err(|_| "Couldn't bind the buffer memory!")?;
            Ok(Self {
                buffer: ManuallyDrop::new(buffer),
                requirements,
                memory_type_id,
                memory: ManuallyDrop::new(memory),
                phantom: PhantomData,
            })
        }
    }

    pub unsafe fn manually_drop(&self, device: &D) {
        use core::ptr::read;
        device.destroy_buffer(ManuallyDrop::into_inner(read(&self.buffer)));
        device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
    }
}

pub fn copy_buffer<B: gfx_hal::Backend, D: gfx_hal::device::Device<B>>(
    device: &B::Device,
    queue_group: &mut QueueGroup<B>,
    command_pool: &mut B::CommandPool,
    src_buffer: &mut BufferBundle<B, D>,
    dst_buffer: &mut BufferBundle<B, D>,
    region: command::BufferCopy,
) {
    let mut copy_fence = device.create_fence(false).expect("Could not create fence");
    unsafe {
        let mut cmd_buffer = command_pool.allocate_one(command::Level::Primary);
        cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

        cmd_buffer.copy_buffer(&src_buffer.buffer, &dst_buffer.buffer, iter::once(region));

        cmd_buffer.finish();

        queue_group.queues[0].submit(
            iter::once(&cmd_buffer),
            iter::empty(),
            iter::empty(),
            Some(&mut copy_fence),
        );

        device
            .wait_for_fence(&copy_fence, !0)
            .expect("Can't wait for fence");
    }

    unsafe {
        device.destroy_fence(copy_fence);
    }
}

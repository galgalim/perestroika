extern crate gfx_backend_vulkan as back;

use core::{
    marker::PhantomData,
    mem::{size_of, size_of_val, ManuallyDrop},
    ops::Deref,
};
use gfx_hal::{
    adapter::Adapter,
    buffer::{self, SubRange},
    command, format as f,
    format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle},
    image as i, memory as m,
    memory::Dependencies,
    pass,
    pass::{
        Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDependency,
    },
    pool,
    prelude::*,
    pso,
    pso::{
        BakedStates, DepthStencilDesc, DepthTest, ElemOffset, Face, FrontFace, InputAssemblerDesc,
        PipelineStage, PolygonMode, Rasterizer, ShaderStageFlags, StencilTest, VertexInputRate,
        Viewport,
    },
    queue::QueueGroup,
    window::{self, Extent2D},
    IndexType, MemoryTypeId,
};
use glm::TMat4;
use image::RgbaImage;
use log::{debug, error, info, warn};
use nalgebra_glm as glm;
use std::time::Instant;
use std::{
    borrow::Borrow,
    collections::HashSet,
    io::Cursor,
    iter,
    mem::{self},
    ptr,
};
use winit::dpi::{LogicalSize, Size};
use winit::error::OsError;
use winit::event::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

mod buffer_bundle;
mod camera;
mod depth_image;
mod graphics_pipeline;
mod render_pass;

use buffer_bundle::{copy_buffer, BufferBundle};
pub use camera::Camera;
use depth_image::DepthImage;
use graphics_pipeline::create_create_graphics_pipeline;
use render_pass::create_render_pass;
use std::{
    ffi::CString,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

pub fn cast_slice(ts: &[f32]) -> Option<&[u32]> {
    use core::mem::align_of;
    // Handle ZST (this all const folds)
    if size_of::<f32>() == 0 || size_of::<u32>() == 0 {
        if size_of::<f32>() == size_of::<u32>() {
            unsafe {
                return Some(core::slice::from_raw_parts(
                    ts.as_ptr() as *const u32,
                    ts.len(),
                ));
            }
        } else {
            return None;
        }
    }
    // Handle alignments (this const folds)
    if align_of::<u32>() > align_of::<f32>() {
        // possible mis-alignment at the new type (this is a real runtime check)
        if (ts.as_ptr() as usize) % align_of::<u32>() != 0 {
            return None;
        }
    }
    if size_of::<f32>() == size_of::<u32>() {
        // same size, so we direct cast, keeping the old length
        unsafe {
            Some(core::slice::from_raw_parts(
                ts.as_ptr() as *const u32,
                ts.len(),
            ))
        }
    } else {
        // we might have slop, which would cause us to fail
        let byte_size = size_of::<f32>() * ts.len();
        let (new_count, new_overflow) =
            (byte_size / size_of::<u32>(), byte_size % size_of::<u32>());
        if new_overflow > 0 {
            return None;
        } else {
            unsafe {
                Some(core::slice::from_raw_parts(
                    ts.as_ptr() as *const u32,
                    new_count,
                ))
            }
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const DIMS: window::Extent2D = window::Extent2D { width: 1024, height: 768 };

const ENTRY_NAME: &str = "main";

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
#[repr(C)]
struct Vertex {
    pos: [f32; 4],
    uv: [f32; 2],
}

#[cfg_attr(rustfmt, rustfmt_skip)]
const CUBE_VERTEXES: [Vertex; 24] = [
  // Face 1 (front)
  Vertex { pos: [0.0, 0.0, 0.0, 1.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { pos: [0.0, 1.0, 0.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { pos: [1.0, 0.0, 0.0, 1.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { pos: [1.0, 1.0, 0.0, 1.0], uv: [1.0, 0.0] }, /* top right */
  // Face 2 (top)
  Vertex { pos: [0.0, 1.0, 0.0, 1.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { pos: [0.0, 1.0, 1.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { pos: [1.0, 1.0, 0.0, 1.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { pos: [1.0, 1.0, 1.0, 1.0], uv: [1.0, 0.0] }, /* top right */
  // Face 3 (back)
  Vertex { pos: [0.0, 0.0, 1.0, 1.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { pos: [0.0, 1.0, 1.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { pos: [1.0, 0.0, 1.0, 1.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { pos: [1.0, 1.0, 1.0, 1.0], uv: [1.0, 0.0] }, /* top right */
  // Face 4 (bottom)
  Vertex { pos: [0.0, 0.0, 0.0, 1.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { pos: [0.0, 0.0, 1.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { pos: [1.0, 0.0, 0.0, 1.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { pos: [1.0, 0.0, 1.0, 1.0], uv: [1.0, 0.0] }, /* top right */
  // Face 5 (left)
  Vertex { pos: [0.0, 0.0, 1.0, 1.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { pos: [0.0, 1.0, 1.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { pos: [0.0, 0.0, 0.0, 1.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { pos: [0.0, 1.0, 0.0, 1.0], uv: [1.0, 0.0] }, /* top right */
  // Face 6 (right)
  Vertex { pos: [1.0, 0.0, 0.0, 1.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { pos: [1.0, 1.0, 0.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { pos: [1.0, 0.0, 1.0, 1.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { pos: [1.0, 1.0, 1.0, 1.0], uv: [1.0, 0.0] }, /* top right */
];

#[cfg_attr(rustfmt, rustfmt_skip)]
const CUBE_INDEXES: [u16; 36] = [
   0,  1,  2,  2,  1,  3, // front
   4,  5,  6,  7,  6,  5, // top
  10,  9,  8,  9, 10, 11, // back
  12, 14, 13, 15, 13, 14, // bottom
  16, 17, 18, 19, 18, 17, // left
  20, 21, 22, 23, 22, 21, // right
];

#[derive(Debug, Clone, Copy)]
struct UniformBlock {
    projection: [[f32; 4]; 4],
}

pub struct Renderer<B: gfx_hal::Backend> {
    instance: Option<B::Instance>,
    device: B::Device,
    queue_group: QueueGroup<B>,
    desc_pool: ManuallyDrop<B::DescriptorPool>,
    surface: ManuallyDrop<B::Surface>,
    adapter: gfx_hal::adapter::Adapter<B>,
    format: gfx_hal::format::Format,
    pub dimensions: window::Extent2D,
    viewport: pso::Viewport,
    render_pass: ManuallyDrop<B::RenderPass>,
    pipeline: ManuallyDrop<B::GraphicsPipeline>,
    pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    desc_set: B::DescriptorSet,
    set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    submission_complete_semaphores: Vec<B::Semaphore>,
    submission_complete_fences: Vec<B::Fence>,
    cmd_pools: Vec<B::CommandPool>,
    cmd_buffers: Vec<B::CommandBuffer>,
    vertex_buffer_bundle: BufferBundle<B, B::Device>,
    index_buffer_bundle: BufferBundle<B, B::Device>,
    image_upload_buffer: ManuallyDrop<B::Buffer>,
    depth_image: DepthImage<B, B::Device>,
    map_image: ManuallyDrop<B::Image>,
    image_srv: ManuallyDrop<B::ImageView>,
    image_memory: ManuallyDrop<B::Memory>,
    image_upload_memory: ManuallyDrop<B::Memory>,
    sampler: ManuallyDrop<B::Sampler>,
    frames_in_flight: usize,
    frame: u64,
}

impl<B> Renderer<B>
where
    B: gfx_hal::Backend,
{
    pub fn new(window: &Window, map_img: RgbaImage) -> Result<Renderer<B>, &'static str> {
        let (window, instance, mut adapters, mut surface) = {
            let instance = B::Instance::create("yantra", 1).expect("Failed to create an instance!");
            let adapters = instance.enumerate_adapters();
            let surface = unsafe {
                instance
                    .create_surface(window)
                    .expect("Failed to create a surface!")
            };
            // Return `window` so it is not dropped: dropping it invalidates `surface`.
            (window, Some(instance), adapters, surface)
        };
        for adapter in &adapters {
            debug!("{:?}", adapter.info);
        }

        let adapter = adapters.remove(0);

        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.properties().limits;

        // Build a new device and associated command queues
        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .ok_or("Couldn't find a QueueFamily with graphics!")?;
        let mut gpu = unsafe {
            adapter
                .physical_device
                .open(&[(family, &[1.0])], gfx_hal::Features::empty())
                .map_err(|_| "Couldn't open the PhysicalDevice!")?
        };
        let mut queue_group = gpu.queue_groups.pop().unwrap();
        let device = gpu.device;
        let mut command_pool = unsafe {
            device
                .create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())
                .map_err(|_| "Could not create the raw command pool!")?
        };

        // Setup renderpass and pipeline
        let set_layout = ManuallyDrop::new(
            unsafe {
                device.create_descriptor_set_layout(
                    vec![
                        pso::DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: pso::DescriptorType::Image {
                                ty: pso::ImageDescriptorType::Sampled {
                                    with_sampler: false,
                                },
                            },
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        pso::DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: pso::DescriptorType::Sampler,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                    ]
                    .into_iter(),
                    iter::empty(),
                )
            }
            .map_err(|_| "Can't create descriptor set layout")?,
        );

        // Descriptors
        let mut desc_pool = ManuallyDrop::new(
            unsafe {
                device.create_descriptor_pool(
                    1, // sets
                    vec![
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Image {
                                ty: pso::ImageDescriptorType::Sampled {
                                    with_sampler: false,
                                },
                            },
                            count: 1,
                        },
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Sampler,
                            count: 1,
                        },
                    ]
                    .into_iter(),
                    pso::DescriptorPoolCreateFlags::empty(),
                )
            }
            .map_err(|_| "Can't create descriptor pool")?,
        );
        let mut desc_set = unsafe { desc_pool.allocate_one(&set_layout) }
            .map_err(|_| "Couldn't make a Descriptor Set!")?;

        // Buffer allocations
        debug!("Memory types: {:?}", memory_types);
        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let (mut vertex_buffer_bundle, mut vertex_transfer_buffer_bundle, vertex_padded_buffer_len) = {
            let (vertex_buffer_len, vertex_padded_buffer_len) =
                get_buffer_lens::<Vertex>(&CUBE_VERTEXES, non_coherent_alignment);
            let mut vertex_transfer_buffer_bundle = BufferBundle::new(
                &device,
                vertex_padded_buffer_len,
                buffer::Usage::VERTEX | buffer::Usage::TRANSFER_SRC,
                &memory_types,
                m::Properties::CPU_VISIBLE | m::Properties::COHERENT,
            )
            .unwrap();

            let mut vertex_buffer_bundle = BufferBundle::new(
                &device,
                vertex_padded_buffer_len,
                buffer::Usage::VERTEX | buffer::Usage::TRANSFER_DST,
                &memory_types,
                m::Properties::DEVICE_LOCAL,
            )
            .unwrap();

            vertex_transfer_buffer_bundle = unsafe {
                let mut memory = ManuallyDrop::into_inner(vertex_transfer_buffer_bundle.memory);
                let requirements = vertex_transfer_buffer_bundle.requirements;
                let vertex_buffer_mapping = device
                    .map_memory(
                        &mut memory,
                        m::Segment {
                            offset: 0,
                            size: Some(requirements.size),
                        },
                    )
                    .unwrap();
                ptr::copy_nonoverlapping(
                    CUBE_VERTEXES.as_ptr() as *const u8,
                    vertex_buffer_mapping,
                    vertex_buffer_len as usize,
                );
                device
                    .flush_mapped_memory_ranges(iter::once((&memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&mut memory);
                vertex_transfer_buffer_bundle.memory = ManuallyDrop::new(memory);
                vertex_transfer_buffer_bundle
            };
            (
                vertex_buffer_bundle,
                vertex_transfer_buffer_bundle,
                vertex_padded_buffer_len,
            )
        };

        copy_buffer(
            &device,
            &mut queue_group,
            &mut command_pool,
            &mut vertex_transfer_buffer_bundle,
            &mut vertex_buffer_bundle,
            command::BufferCopy {
                src: 0,
                dst: 0,
                size: vertex_padded_buffer_len,
            },
        );

        let (mut index_buffer_bundle, mut index_transfer_buffer_bundle, index_padded_buffer_len) = {
            let (index_buffer_len, index_padded_buffer_len) =
                get_buffer_lens::<u16>(&CUBE_INDEXES, non_coherent_alignment);
            let mut index_transfer_buffer_bundle = BufferBundle::new(
                &device,
                index_padded_buffer_len,
                buffer::Usage::INDEX | buffer::Usage::TRANSFER_SRC,
                &memory_types,
                m::Properties::CPU_VISIBLE | m::Properties::COHERENT,
            )
            .unwrap();

            let mut index_buffer_bundle = BufferBundle::new(
                &device,
                index_padded_buffer_len,
                buffer::Usage::INDEX | buffer::Usage::TRANSFER_DST,
                &memory_types,
                m::Properties::DEVICE_LOCAL,
            )
            .unwrap();

            index_transfer_buffer_bundle = unsafe {
                let mut memory = ManuallyDrop::into_inner(index_transfer_buffer_bundle.memory);
                let requirements = index_transfer_buffer_bundle.requirements;
                let index_buffer_mapping = device
                    .map_memory(
                        &mut memory,
                        m::Segment {
                            offset: 0,
                            size: Some(requirements.size),
                        },
                    )
                    .unwrap();
                ptr::copy_nonoverlapping(
                    CUBE_INDEXES.as_ptr() as *const u8,
                    index_buffer_mapping,
                    index_buffer_len as usize,
                );
                device
                    .flush_mapped_memory_ranges(iter::once((&memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&mut memory);
                index_transfer_buffer_bundle.memory = ManuallyDrop::new(memory);
                index_transfer_buffer_bundle
            };

            (
                index_buffer_bundle,
                index_transfer_buffer_bundle,
                index_padded_buffer_len,
            )
        };

        copy_buffer(
            &device,
            &mut queue_group,
            &mut command_pool,
            &mut index_transfer_buffer_bundle,
            &mut index_buffer_bundle,
            command::BufferCopy {
                src: 0,
                dst: 0,
                size: index_padded_buffer_len,
            },
        );

        // Image Upload
        let upload_type = vertex_transfer_buffer_bundle.memory_type_id;

        let (width, height) = map_img.dimensions();

        let kind = i::Kind::D2(width as i::Size, height as i::Size, 1, 1);
        let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let image_stride = 4usize;
        let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
        let upload_size = (height * row_pitch) as u64;
        let padded_upload_size = ((upload_size + non_coherent_alignment - 1)
            / non_coherent_alignment)
            * non_coherent_alignment;

        let mut image_upload_buffer = ManuallyDrop::new(
            unsafe {
                device.create_buffer(
                    padded_upload_size,
                    buffer::Usage::TRANSFER_SRC,
                    m::SparseFlags::empty(),
                )
            }
            .unwrap(),
        );
        let image_mem_reqs = unsafe { device.get_buffer_requirements(&image_upload_buffer) };

        // copy image data into staging buffer
        let image_upload_memory = unsafe {
            let mut memory = device
                .allocate_memory(upload_type, image_mem_reqs.size)
                .map_err(|_| "Couldn't allocate buffer memory!")?;
            device
                .bind_buffer_memory(&memory, 0, &mut image_upload_buffer)
                .map_err(|_| "Couldn't bind the buffer memory!")?;
            let mapping = device.map_memory(&mut memory, m::Segment::ALL).unwrap();
            for y in 0..height as usize {
                let row = &(*map_img)[y * (width as usize) * image_stride
                    ..(y + 1) * (width as usize) * image_stride];
                ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    mapping.offset(y as isize * row_pitch as isize),
                    width as usize * image_stride,
                );
            }
            device
                .flush_mapped_memory_ranges(iter::once((&memory, m::Segment::ALL)))
                .unwrap();
            device.unmap_memory(&mut memory);
            ManuallyDrop::new(memory)
        };

        let mut map_image = ManuallyDrop::new(
            unsafe {
                device.create_image(
                    kind,
                    1,
                    ColorFormat::SELF,
                    i::Tiling::Optimal,
                    i::Usage::TRANSFER_DST | i::Usage::SAMPLED,
                    m::SparseFlags::empty(),
                    i::ViewCapabilities::empty(),
                )
            }
            .unwrap(),
        );
        let image_req = unsafe { device.get_image_requirements(&map_image) };

        let device_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                image_req.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(m::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();
        let image_memory = ManuallyDrop::new(
            unsafe { device.allocate_memory(device_type, image_req.size) }.unwrap(),
        );

        unsafe { device.bind_image_memory(&image_memory, 0, &mut map_image) }.unwrap();
        let image_srv = ManuallyDrop::new(
            unsafe {
                device.create_image_view(
                    &map_image,
                    i::ViewKind::D2,
                    ColorFormat::SELF,
                    Swizzle::NO,
                    i::Usage::TRANSFER_DST | i::Usage::SAMPLED,
                    i::SubresourceRange {
                        aspects: f::Aspects::COLOR,
                        ..Default::default()
                    },
                )
            }
            .unwrap(),
        );

        let sampler = ManuallyDrop::new(
            unsafe {
                device.create_sampler(&i::SamplerDesc::new(
                    i::Filter::Nearest,
                    i::WrapMode::Mirror,
                ))
            }
            .expect("Can't create sampler"),
        );

        unsafe {
            device.write_descriptor_set(pso::DescriptorSetWrite {
                set: &mut desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: vec![
                    pso::Descriptor::Image(&*image_srv, i::Layout::ShaderReadOnlyOptimal),
                    pso::Descriptor::Sampler(&*sampler),
                ]
                .into_iter(),
            });
        }

        // copy buffer to texture
        let mut image_copy_fence = device.create_fence(false).expect("Could not create fence");
        unsafe {
            let mut cmd_buffer = command_pool.allocate_one(command::Level::Primary);
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            let image_barrier = m::Barrier::Image {
                states: (i::Access::empty(), i::Layout::Undefined)
                    ..(i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal),
                target: &*map_image,
                families: None,
                range: i::SubresourceRange {
                    aspects: f::Aspects::COLOR,
                    ..Default::default()
                },
            };

            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                m::Dependencies::empty(),
                iter::once(image_barrier),
            );

            info!("Image Extent: ({},{})", width, height);

            cmd_buffer.copy_buffer_to_image(
                &image_upload_buffer,
                &map_image,
                i::Layout::TransferDstOptimal,
                iter::once(command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: row_pitch / (image_stride as u32),
                    buffer_height: height as u32,
                    image_layers: i::SubresourceLayers {
                        aspects: f::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: i::Offset { x: 0, y: 0, z: 0 },
                    image_extent: i::Extent {
                        width,
                        height,
                        depth: 1,
                    },
                }),
            );

            let image_barrier = m::Barrier::Image {
                states: (i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal)
                    ..(i::Access::SHADER_READ, i::Layout::ShaderReadOnlyOptimal),
                target: &*map_image,
                families: None,
                range: i::SubresourceRange {
                    aspects: f::Aspects::COLOR,
                    ..Default::default()
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                m::Dependencies::empty(),
                iter::once(image_barrier),
            );

            cmd_buffer.finish();

            queue_group.queues[0].submit(
                iter::once(&cmd_buffer),
                iter::empty(),
                iter::empty(),
                Some(&mut image_copy_fence),
            );

            device
                .wait_for_fence(&image_copy_fence, !0)
                .expect("Can't wait for fence");
        }

        unsafe {
            device.destroy_fence(image_copy_fence);
        }

        let caps = surface.capabilities(&adapter.physical_device);
        let formats = surface.supported_formats(&adapter.physical_device);
        debug!("formats: {:?}", formats);
        let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = window::SwapchainConfig::from_caps(&caps, format, DIMS);
        debug!("{:?}", swap_config);
        let extent = swap_config.extent;
        debug!("Extent: {:?}", extent);
        unsafe {
            surface
                .configure_swapchain(&device, swap_config)
                .expect("Can't configure swapchain");
        };

        // Depth Buffer
        let depth_image = DepthImage::new(&adapter, &device, extent).unwrap();

        let render_pass = create_render_pass::<B>(&device, format);

        // Define maximum number of frames we want to be able to be "in flight" (being computed
        // simultaneously) at once
        let frames_in_flight = 3;

        // The number of the rest of the resources is based on the frames in flight.
        let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
        let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);
        // Note: We don't really need a different command pool per frame in such a simple demo like this,
        // but in a more 'real' application, it's generally seen as optimal to have one command pool per
        // thread per frame. There is a flag that lets a command pool reset individual command buffers
        // which are created from it, but by default the whole pool (and therefore all buffers in it)
        // must be reset at once. Furthermore, it is often the case that resetting a whole pool is actually
        // faster and more efficient for the hardware than resetting individual command buffers, so it's
        // usually best to just make a command pool for each set of buffers which need to be reset at the
        // same time (each frame). In our case, each pool will only have one command buffer created from it,
        // though.
        let mut cmd_pools = Vec::with_capacity(frames_in_flight);
        let mut cmd_buffers = Vec::with_capacity(frames_in_flight);

        cmd_pools.push(command_pool);
        for _ in 1..frames_in_flight {
            unsafe {
                cmd_pools.push(
                    device
                        .create_command_pool(
                            queue_group.family,
                            pool::CommandPoolCreateFlags::empty(),
                        )
                        .expect("Can't create command pool"),
                );
            }
        }

        for i in 0..frames_in_flight {
            submission_complete_semaphores.push(
                device
                    .create_semaphore()
                    .expect("Could not create semaphore"),
            );
            submission_complete_fences
                .push(device.create_fence(true).expect("Could not create fence"));
            cmd_buffers.push(unsafe { cmd_pools[i].allocate_one(command::Level::Primary) });
        }

        let push_constants = vec![(ShaderStageFlags::VERTEX, 0..64)];
        let pipeline_layout = ManuallyDrop::new(
            unsafe {
                device.create_pipeline_layout(iter::once(&*set_layout), push_constants.into_iter())
            }
            .expect("Can't create pipeline layout"),
        );

        let mut shader_compiler = shaderc::Compiler::new().unwrap();
        let shader_compiler_options = shaderc::CompileOptions::new().unwrap();
        let vs_module = {
            let spirv = shader_compiler
                .compile_into_spirv(
                    &include_str!("../../../shaders/cube.vert"),
                    shaderc::ShaderKind::Vertex,
                    "cube.vert",
                    "main",
                    Some(&shader_compiler_options),
                )
                .unwrap();
            unsafe { device.create_shader_module(&spirv.as_binary()) }.unwrap()
        };
        let fs_module = {
            let spirv = shader_compiler
                .compile_into_spirv(
                    &include_str!("../../../shaders/cube.frag"),
                    shaderc::ShaderKind::Fragment,
                    "cube.frag",
                    "main",
                    Some(&shader_compiler_options),
                )
                .unwrap();
            unsafe { device.create_shader_module(&spirv.as_binary()) }.unwrap()
        };

        let pipeline = create_create_graphics_pipeline::<B>(
            &device,
            &render_pass,
            &pipeline_layout,
            &vs_module,
            &fs_module,
        );

        unsafe {
            device.destroy_shader_module(vs_module);
        }
        unsafe {
            device.destroy_shader_module(fs_module);
        }
        // Rendering setup
        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0..1.0,
        };

        Ok(Renderer {
            instance,
            device,
            queue_group,
            desc_pool,
            surface: ManuallyDrop::new(surface),
            adapter,
            format,
            dimensions: DIMS,
            viewport,
            render_pass,
            pipeline,
            pipeline_layout,
            desc_set,
            set_layout,
            submission_complete_semaphores,
            submission_complete_fences,
            cmd_pools,
            cmd_buffers,
            vertex_buffer_bundle,
            index_buffer_bundle,
            image_upload_buffer,
            depth_image,
            map_image,
            image_srv,
            image_memory,
            image_upload_memory,
            sampler,
            frames_in_flight,
            frame: 0,
        })
    }

    pub fn recreate_swapchain(&mut self) {
        let caps = self.surface.capabilities(&self.adapter.physical_device);
        let swap_config = window::SwapchainConfig::from_caps(&caps, self.format, self.dimensions);
        debug!("{:?}", swap_config);
        let extent = swap_config.extent.to_extent();

        unsafe {
            self.surface
                .configure_swapchain(&self.device, swap_config)
                .expect("Can't create swapchain");
        }

        self.viewport.rect.w = extent.width as _;
        self.viewport.rect.h = extent.height as _;
    }

    pub fn render(
        &mut self,
        view_projection: &glm::TMat4<f32>,
        models: &[glm::TMat4<f32>],
    ) -> Result<(), &'static str> {
        let surface_image = unsafe {
            match self.surface.acquire_image(!0) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    return Ok(());
                }
            }
        };

        let caps = self.surface.capabilities(&self.adapter.physical_device);
        let swap_config = window::SwapchainConfig::from_caps(&caps, self.format, self.dimensions);
        let fat = swap_config.framebuffer_attachment();
        debug!("Attachments: {:?}", fat);

        let framebuffer = unsafe {
            self.device
                .create_framebuffer(
                    &self.render_pass,
                    iter::once(fat),
                    i::Extent {
                        width: self.dimensions.width,
                        height: self.dimensions.height,
                        depth: 1,
                    },
                )
                .unwrap()
        };

        // Compute index into our resource ring buffers based on the frame number
        // and number of frames in flight. Pay close attention to where this index is needed
        // versus when the swapchain image index we got from acquire_image is needed.
        let frame_idx = self.frame as usize % self.frames_in_flight;

        // Wait for the fence of the previous submission of this frame and reset it; ensures we are
        // submitting only up to maximum number of frames_in_flight if we are submitting faster than
        // the gpu can keep up with. This would also guarantee that any resources which need to be
        // updated with a CPU->GPU data copy are not in use by the GPU, so we can perform those updates.
        // In this case there are none to be done, however.
        unsafe {
            let fence = &mut self.submission_complete_fences[frame_idx];
            self.device
                .wait_for_fence(fence, !0)
                .expect("Failed to wait for fence");
            self.device
                .reset_fence(fence)
                .expect("Failed to reset fence");
            self.cmd_pools[frame_idx].reset(false);
        }

        // Rendering
        let cmd_buffer = &mut self.cmd_buffers[frame_idx];
        unsafe {
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            cmd_buffer.set_viewports(0, iter::once(self.viewport.clone()));
            cmd_buffer.set_scissors(0, iter::once(self.viewport.rect));
            cmd_buffer.bind_graphics_pipeline(&self.pipeline);
            cmd_buffer.bind_vertex_buffers(
                0,
                iter::once((&*self.vertex_buffer_bundle.buffer, buffer::SubRange::WHOLE)),
            );
            cmd_buffer.bind_index_buffer(
                &self.index_buffer_bundle.buffer,
                SubRange::WHOLE,
                IndexType::U16,
            );
            cmd_buffer.bind_graphics_descriptor_sets(
                &self.pipeline_layout,
                0,
                iter::once(&self.desc_set),
                iter::empty(),
            );

            cmd_buffer.begin_render_pass(
                &self.render_pass,
                &framebuffer,
                self.viewport.rect,
                iter::once(command::RenderAttachmentInfo {
                    image_view: surface_image.borrow(),
                    clear_value: command::ClearValue {
                        color: command::ClearColor {
                            float32: [0.8, 0.8, 0.8, 1.0],
                        },
                    },
                }),
                command::SubpassContents::Inline,
            );
            for model in models.iter() {
                let mvp = view_projection * model;
                cmd_buffer.push_graphics_constants(
                    &self.pipeline_layout,
                    ShaderStageFlags::VERTEX,
                    0,
                    cast_slice(&mvp.data).unwrap(),
                );
                cmd_buffer.draw_indexed(0..CUBE_INDEXES.len() as u32, 0, 0..1);
            }
            cmd_buffer.end_render_pass();
            cmd_buffer.finish();
            self.queue_group.queues[0].submit(
                iter::once(&*cmd_buffer),
                iter::empty(),
                iter::once(&self.submission_complete_semaphores[frame_idx]),
                Some(&mut self.submission_complete_fences[frame_idx]),
            );

            // present frame
            let result = self.queue_group.queues[0].present(
                &mut self.surface,
                surface_image,
                Some(&mut self.submission_complete_semaphores[frame_idx]),
            );

            self.device.destroy_framebuffer(framebuffer);

            if result.is_err() {
                self.recreate_swapchain();
            }
        }

        // Increment our frame
        self.frame += 1;

        Ok(())
    }
}

impl<B> Drop for Renderer<B>
where
    B: gfx_hal::Backend,
{
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
        unsafe {
            // TODO: When ManuallyDrop::take (soon to be renamed to ManuallyDrop::read) is stabilized we should use that instead.
            self.device
                .destroy_descriptor_pool(ManuallyDrop::into_inner(ptr::read(&self.desc_pool)));
            self.device
                .destroy_descriptor_set_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.set_layout,
                )));

            self.vertex_buffer_bundle.manually_drop(&self.device);
            self.index_buffer_bundle.manually_drop(&self.device);

            self.depth_image.manually_drop(&self.device);

            self.device
                .destroy_buffer(ManuallyDrop::into_inner(ptr::read(
                    &self.image_upload_buffer,
                )));
            self.device
                .destroy_image(ManuallyDrop::into_inner(ptr::read(&self.map_image)));
            self.device
                .destroy_image_view(ManuallyDrop::into_inner(ptr::read(&self.image_srv)));
            self.device
                .destroy_sampler(ManuallyDrop::into_inner(ptr::read(&self.sampler)));
            for p in self.cmd_pools.drain(..) {
                self.device.destroy_command_pool(p);
            }
            for s in self.submission_complete_semaphores.drain(..) {
                self.device.destroy_semaphore(s);
            }
            for f in self.submission_complete_fences.drain(..) {
                self.device.destroy_fence(f);
            }
            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&self.render_pass)));
            self.surface.unconfigure_swapchain(&self.device);
            self.device
                .free_memory(ManuallyDrop::into_inner(ptr::read(&self.image_memory)));
            self.device.free_memory(ManuallyDrop::into_inner(ptr::read(
                &self.image_upload_memory,
            )));
            self.device
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.pipeline)));
            self.device
                .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.pipeline_layout,
                )));
            if let Some(instance) = &self.instance {
                let surface = ManuallyDrop::into_inner(ptr::read(&self.surface));
                instance.destroy_surface(surface);
            }
        }
        debug!("DROPPED!");
    }
}

pub fn view() -> TMat4<f32> {
    glm::look_at_lh(
        &glm::vec3(7.0f32, 3.0f32, -4.0f32),
        &glm::vec3(0.0f32, 0.0f32, 0.0f32),
        &glm::vec3(0.0f32, 1.0f32, 0.0f32),
    )
}

pub fn projection() -> TMat4<f32> {
    let mut tmp = glm::perspective_lh_zo(1920.0 / 1080.0, f32::to_radians(50.0), 0.1f32, 100.0f32);
    tmp[(1, 1)] *= -1.0;
    tmp
}

pub fn view_projection() -> TMat4<f32> {
    let proj = projection();
    let view = view();
    proj * view
}

pub fn get_buffer_lens<T>(buffer: &[T], non_coherent_alignment: u64) -> (u64, u64) {
    let buffer_stride = mem::size_of::<T>() as u64;

    let buffer_len = buffer.len() as u64 * buffer_stride;
    assert_ne!(buffer_len, 0);

    let padded_buffer_len = ((buffer_len + non_coherent_alignment - 1) / non_coherent_alignment)
        * non_coherent_alignment;
    (buffer_len, padded_buffer_len)
}

#[cfg(test)]
mod test {
    #[test]
    fn it_works() {
        assert_eq!(1 + 1, 2);
    }
}

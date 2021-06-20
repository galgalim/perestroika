use super::{Vertex, ENTRY_NAME};
use core::mem::{self, size_of, ManuallyDrop};
use gfx_hal::{
    format as f,
    pass::Subpass,
    prelude::*,
    pso,
    pso::{
        DepthStencilDesc, DepthTest, ElemOffset, Face, FrontFace, PolygonMode, Rasterizer,
        VertexInputRate,
    },
};

pub fn create_create_graphics_pipeline<B: gfx_hal::Backend>(
    device: &B::Device,
    render_pass: &B::RenderPass,
    pipeline_layout: &B::PipelineLayout,
    vs_module: &B::ShaderModule,
    fs_module: &B::ShaderModule,
) -> ManuallyDrop<B::GraphicsPipeline> {
    let (vs_entry, fs_entry) = (
        pso::EntryPoint {
            entry: ENTRY_NAME,
            module: vs_module,
            specialization: pso::Specialization::default(),
        },
        pso::EntryPoint {
            entry: ENTRY_NAME,
            module: fs_module,
            specialization: pso::Specialization::default(),
        },
    );

    let subpass = Subpass {
        index: 0,
        main_pass: &*render_pass,
    };

    let depth_stencil = DepthStencilDesc {
        depth: Some(DepthTest {
            fun: gfx_hal::pso::Comparison::LessEqual,
            write: true,
        }),
        depth_bounds: false,
        stencil: None,
    };

    let rasterizer = Rasterizer {
        depth_clamping: false,
        polygon_mode: PolygonMode::Fill,
        cull_face: Face::BACK,
        front_face: FrontFace::Clockwise,
        depth_bias: None,
        conservative: false,
        line_width: pso::State::Dynamic,
    };

    let vertex_buffers = vec![pso::VertexBufferDesc {
        binding: 0,
        stride: mem::size_of::<Vertex>() as u32,
        rate: VertexInputRate::Vertex,
    }];

    let attributes = vec![
        pso::AttributeDesc {
            location: 0,
            binding: 0,
            element: pso::Element {
                format: f::Format::Rgba32Sfloat,
                offset: 0,
            },
        },
        pso::AttributeDesc {
            location: 1,
            binding: 0,
            element: pso::Element {
                format: f::Format::Rg32Sfloat,
                offset: size_of::<[f32; 4]>() as ElemOffset,
            },
        },
    ];

    let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
        pso::PrimitiveAssemblerDesc::Vertex {
            buffers: &vertex_buffers,
            attributes: &attributes,
            input_assembler: pso::InputAssemblerDesc {
                primitive: pso::Primitive::TriangleList,
                with_adjacency: false,
                restart_index: None,
            },
            vertex: vs_entry,
            geometry: None,
            tessellation: None,
        },
        rasterizer,
        Some(fs_entry),
        &*pipeline_layout,
        subpass,
    );
    pipeline_desc.depth_stencil = depth_stencil;
    pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
        mask: pso::ColorMask::ALL,
        blend: Some(pso::BlendState::ALPHA),
    });

    ManuallyDrop::new(unsafe {
        device
            .create_graphics_pipeline(&pipeline_desc, None)
            .unwrap()
    })
}

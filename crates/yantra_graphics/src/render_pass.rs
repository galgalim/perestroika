use core::mem::ManuallyDrop;
use gfx_hal::{
    format as f, image as i, memory as m, pass,
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDependency},
    prelude::*,
    pso::PipelineStage,
};
use std::iter;

pub fn create_render_pass<B: gfx_hal::Backend>(
    device: &B::Device,
    format: f::Format,
) -> ManuallyDrop<B::RenderPass> {
    let color_attachment = pass::Attachment {
        format: Some(format),
        samples: 1,
        ops: pass::AttachmentOps::new(
            pass::AttachmentLoadOp::Clear,
            pass::AttachmentStoreOp::Store,
        ),
        stencil_ops: pass::AttachmentOps::DONT_CARE,
        layouts: i::Layout::Undefined..i::Layout::Present,
    };

    let depth_attachment = Attachment {
        format: Some(f::Format::D32Sfloat),
        samples: 1,
        ops: pass::AttachmentOps::new(
            pass::AttachmentLoadOp::Clear,
            pass::AttachmentStoreOp::Store,
        ),
        stencil_ops: AttachmentOps::DONT_CARE,
        layouts: i::Layout::Undefined..i::Layout::DepthStencilAttachmentOptimal,
    };

    let subpass = pass::SubpassDesc {
        colors: &[(0, i::Layout::ColorAttachmentOptimal)],
        depth_stencil: Some(&(1, i::Layout::DepthStencilAttachmentOptimal)),
        //depth_stencil: None,
        inputs: &[],
        resolves: &[],
        preserves: &[],
    };

    let dependency = SubpassDependency {
        passes: None..Some(0),
        stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT
            ..PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS,
        accesses: i::Access::empty()
            ..(i::Access::COLOR_ATTACHMENT_READ | i::Access::COLOR_ATTACHMENT_WRITE),
        flags: m::Dependencies::empty(),
    };

    let in_dependency = SubpassDependency {
        passes: None..Some(0),
        stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT
            ..PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS,
        accesses: i::Access::empty()
            ..(i::Access::COLOR_ATTACHMENT_READ
                | i::Access::COLOR_ATTACHMENT_WRITE
                | i::Access::DEPTH_STENCIL_ATTACHMENT_READ
                | i::Access::DEPTH_STENCIL_ATTACHMENT_WRITE),
        flags: m::Dependencies::empty(),
    };

    let out_dependency = SubpassDependency {
        passes: Some(0)..None,
        stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS
            ..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
        accesses: (i::Access::COLOR_ATTACHMENT_READ
            | i::Access::COLOR_ATTACHMENT_WRITE
            | i::Access::DEPTH_STENCIL_ATTACHMENT_READ
            | i::Access::DEPTH_STENCIL_ATTACHMENT_WRITE)..i::Access::empty(),
        flags: m::Dependencies::empty(),
    };
    ManuallyDrop::new(
        unsafe {
            device.create_render_pass(
                vec![color_attachment, depth_attachment].into_iter(),
                iter::once(subpass),
                iter::empty(),
            )
        }
        .expect("Can't create render pass"),
    )
}

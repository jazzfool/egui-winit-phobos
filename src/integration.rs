use std::alloc::alloc;
use std::sync::{Arc, Mutex};
use egui::{Context, epaint, ImageData, TextureId};

use phobos as ph;
use phobos::{GraphicsCmdBuffer, IncompleteCmdBuffer, TransferCmdBuffer, vk};
use winit::event_loop::EventLoop;

use anyhow::Result;
use egui::epaint::ahash::AHashMap;
use egui::epaint::{ImageDelta, Primitive};
use egui_winit::EventResponse;
use winit::window::Window;

use log::trace;

pub struct Integration {
    context: Context,
    egui_winit: egui_winit::State,
    device: Arc<ph::Device>,
    allocator: Arc<Mutex<ph::Allocator>>,
    exec: Arc<ph::ExecutionManager>,
    sampler: ph::Sampler,
    width: u32,
    height: u32,
    scale_factor: f32,
    textures: AHashMap<TextureId, (ph::Image, ph::ImageView)>,
    user_textures: AHashMap<TextureId, ph::ImageView>,
    pipelines: Arc<Mutex<ph::PipelineCache>>,
    descriptors: Arc<Mutex<ph::DescriptorCache>>,
}

// TODO: Phobos: Create push constants

impl Integration {
    fn bytes_to_spirv(buffer: &[u8]) -> Vec<u32> {
        let (_, binary, _) = unsafe { buffer.align_to::<u32>() };
        Vec::from(binary)
    }

    pub fn new<T>(
        width: u32, height: u32,
        scale_factor: f32,
        event_loop: &EventLoop<T>,
        font_definitions: egui::FontDefinitions,
        style: egui::Style,
        device: Arc<ph::Device>,
        allocator: Arc<Mutex<ph::Allocator>>,
        exec: Arc<ph::ExecutionManager>,
        pipelines: Arc<Mutex<ph::PipelineCache>>,
        descriptors: Arc<Mutex<ph::DescriptorCache>>,) -> Result<Self> {

        let context = Context::default();
        context.set_fonts(font_definitions);
        context.set_style(style);

        let egui_winit = egui_winit::State::new(&event_loop);

        let vtx_code = include_bytes!("shaders/spv/vert.spv");
        let frag_code = include_bytes!("shaders/spv/frag.spv");

        let vtx = ph::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, Self::bytes_to_spirv(vtx_code));
        let frag = ph::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, Self::bytes_to_spirv(frag_code));

        let pci = ph::PipelineBuilder::new("egui_pipeline")
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 2, vk::Format::R8G8B8A8_UNORM)?
            .attach_shader(vtx)
            .attach_shader(frag)
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .cull_mask(vk::CullModeFlags::NONE)
            .blend_additive_unmasked(vk::BlendFactor::SRC_ALPHA, vk::BlendFactor::ONE_MINUS_SRC_ALPHA, vk::BlendFactor::ZERO, vk::BlendFactor::ONE)
            .build();

        {
            pipelines.lock().unwrap().create_named_pipeline(pci)?;
        }

        let sampler = ph::Sampler::new(device.clone(), vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::FALSE,
            max_anisotropy: 0.0,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE,
            border_color: Default::default(),
            unnormalized_coordinates: vk::FALSE,
        })?;

        Ok(Self {
            context,
            egui_winit,
            device,
            allocator,
            exec,
            sampler,
            width,
            height,
            scale_factor,
            textures: Default::default(),
            user_textures: Default::default(),
            pipelines,
            descriptors,
        })
    }

    /// handling winit event.
    pub fn handle_event(&mut self, winit_event: &egui_winit::winit::event::WindowEvent<'_>) -> EventResponse {
        self.egui_winit.on_event(&self.context, winit_event)
    }

    /// begin frame.
    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.egui_winit.take_egui_input(window);
        self.context.begin_frame(raw_input);
    }

    /// end frame.
    pub fn end_frame(&mut self, window: &Window) -> egui::FullOutput {
        let output = self.context.end_frame();
        self.egui_winit.handle_platform_output(window, &self.context, output.platform_output.clone());
        output
    }

    pub fn context(&self) -> Context {
        self.context.clone()
    }

    pub fn register_user_texture(&mut self, image_view: &ph::ImageView) -> TextureId {
        self.user_textures.insert(TextureId::User(image_view.id), image_view.clone());
        TextureId::User(image_view.id)
    }

    pub fn unregister_user_texture(&mut self, texture_id: TextureId) {
        if let TextureId::User(id) = texture_id {
            self.user_textures.remove(&texture_id);
        } else {
            eprintln!("The internal texture cannot be unregistered; please pass the texture ID of UserTexture.");
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    pub async fn paint<'s: 'e, 'e, 'q>(
        &'s mut self,
        inputs: &[ph::VirtualResource], // Sampled images
        output: ph::VirtualResource,
        load_op: vk::AttachmentLoadOp,
        clear_value: Option<vk::ClearColorValue>,
        clipped_meshes: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
    ) -> Result<ph::Pass<'e, 'q, ph::domain::All>> {

        for (id, delta) in &textures_delta.set {
            trace!("Updating texture id: {:?}", id);
            self.update_texture(*id, &delta).await?;
        }

        // Free textures
        self.textures.retain(|tex, _| !textures_delta.free.contains(tex));
        // Create pass
        self.get_pass(clipped_meshes, inputs, output, load_op, clear_value)
    }

    fn get_pass<'s: 'e, 'e, 'q>(&'s mut self,
                                clipped_meshes: Vec<egui::ClippedPrimitive>,
                                inputs: &[ph::VirtualResource], // Sampled images
                                output: ph::VirtualResource,
                                load_op: vk::AttachmentLoadOp,
                                clear_value: Option<vk::ClearColorValue>) -> Result<ph::Pass<'e, 'q, ph::domain::All>> {
        let mut builder = ph::PassBuilder::render("egui_render")
            .color_attachment(output, load_op, clear_value)?;
        for input in inputs {
            builder = builder.sample_image(input.clone(), ph::PipelineStage::FRAGMENT_SHADER);
        }

        builder = builder.execute(move |mut cmd, mut ifc, bindings| {
            let vtx_size = Self::vertex_buffer_size(&clipped_meshes);
            let idx_size = Self::index_buffer_size(&clipped_meshes);
            let mut vertex_buffer = ifc.allocate_scratch_vbo(vtx_size)?;
            let mut idx_buffer = ifc.allocate_scratch_ibo(idx_size)?;
            let mut vtx_offset = 0 as usize;
            let mut idx_offset = 0 as usize;
            let mut vertex_base = 0;
            let mut index_base = 0;

            let mut cmd = cmd.bind_graphics_pipeline("egui_pipeline", self.pipelines.clone())?;
            cmd = cmd.bind_vertex_buffer(0, vertex_buffer);
            cmd = cmd.bind_index_buffer(idx_buffer, vk::IndexType::UINT32);

            let vtx_slice = vertex_buffer.mapped_slice::<u8>()?;
            let idx_slice = idx_buffer.mapped_slice::<u8>()?;

            cmd = cmd.viewport(vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.width as f32,
                height: self.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            });
            let width_points = self.width as f32 / self.scale_factor as f32;
            let height_points = self.height as f32 / self.scale_factor as f32;
            let pc = [width_points, height_points];
            cmd = cmd.push_constants(vk::ShaderStageFlags::VERTEX, 0, &pc);

            for egui::ClippedPrimitive { clip_rect, primitive } in &clipped_meshes {
                let mesh = match primitive {
                    Primitive::Mesh(mesh) => { mesh }
                    Primitive::Callback(_) => { todo!() }
                };
                if mesh.vertices.is_empty() || mesh.indices.is_empty() { continue; }
                let view = if let TextureId::User(id) = mesh.texture_id {
                    self.user_textures.get(&TextureId::User(id)).unwrap().clone()
                } else {
                    self.textures.get(&mesh.texture_id).unwrap().1.clone()
                };
                cmd = cmd.bind_new_descriptor_set(0, self.descriptors.clone(),
                                                  ph::DescriptorSetBuilder::new()
                                                      .bind_sampled_image(0, view.clone(), &self.sampler)
                                                      .build()
                )?;

                let v_slice = &mesh.vertices;
                let v_size = std::mem::size_of_val(&v_slice[0]);
                let v_copy_size = v_slice.len() * v_size;

                let i_slice = &mesh.indices;
                let i_size = std::mem::size_of_val(&i_slice[0]);
                let i_copy_size = i_slice.len() * i_size;

                unsafe {
                    vtx_slice.as_mut_ptr().offset(vtx_offset as isize).copy_from(v_slice.as_ptr() as *const u8, v_copy_size);
                    idx_slice.as_mut_ptr().offset(idx_offset as isize).copy_from(i_slice.as_ptr() as *const u8, i_copy_size);
                    vtx_offset += v_copy_size;
                    idx_offset += i_copy_size;
                }

                let min = clip_rect.min;
                let min = egui::Pos2 {
                    x: min.x * self.scale_factor as f32,
                    y: min.y * self.scale_factor as f32,
                };
                let min = egui::Pos2 {
                    x: f32::clamp(min.x, 0.0, self.width as f32),
                    y: f32::clamp(min.y, 0.0, self.height as f32),
                };
                let max = clip_rect.max;
                let max = egui::Pos2 {
                    x: max.x * self.scale_factor as f32,
                    y: max.y * self.scale_factor as f32,
                };
                let max = egui::Pos2 {
                    x: f32::clamp(max.x, min.x, self.width as f32),
                    y: f32::clamp(max.y, min.y, self.height as f32),
                };

                cmd = cmd.scissor(vk::Rect2D {
                    offset: vk::Offset2D {
                        x: min.x.round() as i32,
                        y: min.y.round() as i32,
                    },
                    extent: vk::Extent2D {
                        width: (max.x.round() - min.x) as u32,
                        height: (max.y.round() - min.y) as u32,
                    },
                });
                cmd = cmd.draw_indexed(mesh.indices.len() as u32, 1, index_base, vertex_base, 0);

                vertex_base += mesh.vertices.len() as i32;
                index_base += mesh.indices.len() as u32;
            }

            Ok(cmd)
        });

        Ok(builder.build())
    }

    fn vertex_buffer_size(_prims: &[egui::ClippedPrimitive]) -> vk::DeviceSize {
        4 * 1024 * 1024
    }

    fn index_buffer_size(_prims: &[egui::ClippedPrimitive]) -> vk::DeviceSize {
        4 * 1024 * 1024
    }

    fn get_pixel_data(delta: &ImageDelta) -> Result<Vec<u8>> {
        Ok(match &delta.image {
            ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                image.pixels.iter().flat_map(|color| color.to_array()).collect()
            }
            ImageData::Font(image) => {
                image.srgba_pixels(None).flat_map(|color| color.to_array()).collect()
            }
        })
    }

    async fn update_texture(&mut self, texture: TextureId, delta: &ImageDelta) -> Result<()> {
        let (image, view) = self.upload_image(texture, &delta)?.await;
        // We now have a texture in GPU memory. If delta pos exists, we need to update an existing texture.
        // Otherwise, we need to register it as a new texture
        if let Some(pos) = delta.pos {
            let existing_texture = self.textures.get(& texture);
            if let Some((existing_image, existing_view)) = existing_texture {
                self.update_image(texture, pos, &delta, &view, existing_view).await?;
            }
        } else {
            self.textures.insert(texture, (image, view));
        }

        Ok(())
    }

    async fn update_image(&self, texture: TextureId, pos: [usize; 2], delta: &ImageDelta, src: &ph::ImageView, dst: &ph::ImageView) -> Result<()> {
        let top_left = vk::Offset3D { x: pos[0] as i32, y: pos[1] as i32, z: 0 };
        let bottom_right = vk::Offset3D { x: pos[0] as i32 + delta.image.width() as i32, y: pos[1] as i32 + delta.image.height() as i32, z: 1 };
        let src_offsets = [ vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: dst.size.width as i32, y: dst.size.height as i32, z: dst.size.depth as i32 } ];
        let cmd = self.exec.on_domain::<ph::domain::Graphics>()?
            .transition_image(
                dst,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::NONE,
                vk::AccessFlags::TRANSFER_WRITE)
            .transition_image(
                src,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::NONE,
                vk::AccessFlags::TRANSFER_WRITE)
            .blit_image(src, dst, &src_offsets, &[top_left, bottom_right], vk::Filter::NEAREST)
            .transition_image(
                dst,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::NONE)
            .finish()?;
        ph::ExecutionManager::submit(self.exec.clone(), cmd)?.await;
        Ok(())
    }

    fn upload_image(&mut self, texture: TextureId, delta: &ImageDelta) -> Result<ph::GpuFuture<(ph::Image, ph::ImageView)>> {
        // Extract pixel data from egui
        let data = Self::get_pixel_data(&delta)?;
        let staging = ph::Buffer::new(
            self.device.clone(),
            self.allocator.clone(),
            data.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            ph::MemoryLocation::CpuToGpu
        )?;
        // Allocate staging buffer and copy pixel data to it
        let mut staging_view = staging.view_full();
        staging_view.mapped_slice::<u8>()?.copy_from_slice(&data);

        // Create a new image and an image view for it.
        let image = ph::Image::new(
            self.device.clone(),
            self.allocator.clone(),
            delta.image.width() as u32,
            delta.image.height() as u32,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::Format::R8G8B8A8_UNORM,
            vk::SampleCountFlags::TYPE_1
        )?;
        let view = image.view(vk::ImageAspectFlags::COLOR)?;

        let cmd = self.exec.on_domain::<ph::domain::Transfer>()?;
        let cmd = cmd.transition_image(
            &view,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::NONE,
            vk::AccessFlags::TRANSFER_WRITE);
        let cmd = cmd.copy_buffer_to_image(&staging_view, &view)?;
        let cmd = cmd.transition_image(
            &view,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::NONE
        );

        let cmd = cmd.finish()?;
        let fence = ph::ExecutionManager::submit(self.exec.clone(), cmd)?;
        // WOOO GPU FUTURES
        Ok(fence.with_cleanup(move || {
            drop(staging);
        }).attach_value((image, view)))
    }
}
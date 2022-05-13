
use rand::Rng;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::swapchain::Surface;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use std::process::CommandArgs;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::device::{Device, Features, DeviceCreateInfo, QueueCreateInfo, Queue};
use bytemuck::{Pod, Zeroable};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::sync:: {self, GpuFuture};
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::Pipeline;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::format::{Format, ClearValue};
use image::{ImageBuffer, Rgba};
use vulkano::image::view::ImageView;
use vulkano::device::DeviceExtensions;

use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Subpass, Framebuffer, FramebufferCreateInfo};
use winit::event::{Event, WindowEvent};


fn select_physical_device<'a>(
    instance: &'a Arc<Instance>,
    surface: Arc<Surface<Window>>,
    device_extensions: &DeviceExtensions,
) -> (PhysicalDevice<'a>, QueueFamily<'a>) {
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .expect("no device available");

    (physical_device, queue_family)
}

fn main() {
    //initialize vulkan and get the physical device (GPU)
    let instance = Instance::new(InstanceCreateInfo::default()).expect("failed to create instance");


    //only needed when using swapchains, e.g. rendering to the viewport
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };


    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

    //check the selected physical device for queue families. These are groupings of queues, which are kind of like threads on the GPU
    //every queue family is responsible for specific actions (overlap possible), like graphics processing
    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }
    
    //get all queue families from the physical device and chose the one that supports graphics processing
    let queue_family = physical.queue_families()
        .find(|&q| q.supports_graphics())
    .expect("couldn't find a graphical queue family");


    //create a new device (communication channel to physical device) <- left
    //for the chosen GPU
    //and queues based on the device -> right
    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            // here we pass the desired queue families that we want to use
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .expect("failed to create device");
    

    //chose the graphics queue. In this case, it is expected that we only got 1 queue, even tho multiple queues are theoretically possible
    let v_q :Vec<Arc<Queue>> = queues.collect();
    if v_q.len() > 1 {
        println!("Something went wrong, queues result unexpected!");
    }

    let queue = v_q.first().unwrap();

    //Now the actual initialization which needs to be done pretty much everytime you need to do something in vulkan is done. Everything below
    //are just a bunch of use-cases and examples on how to go from here to get vulkan to do something!






// here we derive all these traits to ensure the data behaves as simple as possible
#[repr(C)]      //organize the struct in memory as if it was a C struct
#[derive(Default, Copy, Clone, Zeroable, Pod)]      //derive Traits Default value, Copy, Clone, init with all 0 Bytes and plain old data (no functions or fancies) for the struct
struct MyStruct {
    a: u32,
    b: u32, 
}                   //only types that implement Send and Sync or are static can be used fully in a buffer context



    let data = MyStruct { a: 5, b: 69 };
    let buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, data)
    .expect("failed to create buffer"); //expect will unpack a Result (or Option) and throw an unrecoverable, willing Error with the given message on fail



 /*    let iter = (0..128).map(|_| 5u8);                
    let buffer =                        
    CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, iter).unwrap();  //create a buffer with 128 elements containing the number 5 in u8
    */            


    let mut content = buffer.write().unwrap();
    // `content` implements `DerefMut` whose target is of type `MyStruct` (the content of the buffer)
    content.a *= 2;
    content.b = 9;



    //NEXT: COPYING CONTENT FROM ONE BUFFER TO ANOTHER

    let source_content: Vec<i32> = (0..64).collect();       //Vector (List) with elements (i32) of 0 to 63
    let source = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, source_content.into_iter())
    .expect("failed to create buffer");

    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();   //Vector with all 0s
    let destination = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, destination_content)
    .expect("failed to create buffer");


    //create a primary command buffer. A buffer that stores commands for the GPU (as opposed to the previous buffers, which store data)
    //these buffered commands are transferred to the GPU in unison, because these transfers take time. (performance optimization)
    
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(), 
        CommandBufferUsage::OneTimeSubmit,          //the buffer(s) can only be submitted once
    )
    .unwrap();

    builder.copy_buffer(source.clone(), destination.clone()).unwrap();          //clones are needed or the original ownership gets transferred, since the source and destination variables are Arc<>, these clones are cheap ond lead to a copied reference

    let command_buffer = builder.build().unwrap();    //create the actual buffer


    let future = sync::now(device.clone())
    .then_execute(queue.clone(), command_buffer)
    .unwrap()
    .then_signal_fence_and_flush()        //results in the execution of the commandBuffers on the selected GPU that is associated with the device and sends a fence (action completed) signal back, which will be stored in the future variable as soon as it is completed (async)
    .unwrap();

    future.wait(None).unwrap();  //stops this CPU thread until the future has the result
    let src_content = source.read().unwrap();                   //read the content from the buffers. The vectors  were converted to slices 
    let destination_content = destination.read().unwrap();
    assert_eq!(&*src_content, &*destination_content);




    //NEXT: CONCURRENCY IN GPU COMMANDS
    let data_iter = 0..65536;               //create a buffer with 65536 entries (x 32Bytes)
    let data_buffer =
    CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
        .expect("failed to create buffer");


    //shading logic is written in GLSL, which looks a bit like C. This needs to be imported and will be compiled aswell

    mod cs {

        vulkano_shaders::shader!{
            ty: "compute",                  //use version 450, create workgroups (1024) with a x size of 64, y size of 1 and z size of 1 (one dimensional data structure, otherwise use y for 2 and y and z for 3 dimensions)  -> define a buffer data structure and call the function. Layout buffer Data is a slot for a descriptor set
            src: "
    #version 450                                
    
    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
    
    layout(set = 0, binding = 0) buffer Data {
        uint data[];
    } buf;
    
    void main() {
        uint idx = gl_GlobalInvocationID.x;
        buf.data[idx] *= 12;
    }"
        }
    }
    let shader = cs::load(device.clone())       //create shader for the current device based on our definition in cs
    .expect("failed to create shader module");

    //compute pipeline to actually execute the shader
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("failed to create compute pipeline");

    //create descriptor set, in order to do that, we need to get the general layout of the descriptors that we put in th GSLS code before (buffer Data)
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    //create a DescriptorSet with the Layout, use the corresponding binding and put the data_buffer (65536 integers from 0 to 65536) in
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
    )
    .unwrap();


    //next: create a command buffer similiar to the one that copies the buffer seen above, just for the shader we created
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    
    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0, // 0 is the index of our set
            set,
        )
        .dispatch([1024, 1, 1])     //spawn 1024 working groups
        .unwrap();
    
    let command_buffer = builder.build().unwrap();

    //submit the command buffer
    let future = sync::now(device.clone())
    .then_execute(queue.clone(), command_buffer)
    .unwrap()
    .then_signal_fence_and_flush()
    .unwrap();

        //wait for the execution to complete
    future.wait(None).unwrap();
    //check the buffer for changed values (we could have changed these values our Rust code aswell, but we used the GPUs Ability to perform concurrent actions on data structures)
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Everything succeeded!");


    //NEXT: Image creation

    //create a new image for the device, 1024 * 1024 pixels, with RGBA, used for the queue family
    let image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1, // images can be arrays of layers
        },
        Format::R8G8B8A8_UNORM,    //RED -> GREEN -> BLUE -> ALPHA
        Some(queue.family()),
    )
    .unwrap();

    //create a commandBuffer to clear the image we just created
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    


    //create a Buffer to write to. 1024 * 1024, but each pixel has 4 values (RGBA), so 1024* 1024 * 4. 
    //Images cannot be directly accessed by the CPU, only the GPU, so this is needed for further actions
    let buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),  //initialize every value with 0
    )
    .expect("failed to create buffer");
    
    builder
    .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0]))
    .unwrap()
    .copy_image_to_buffer(image.clone(), buf.clone())
    .unwrap();

    let command_buffer = builder.build().unwrap();
    //execute the copy of the image to the buffer
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    //create image from the contents in the buffer
    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap(); 

    //save the image to a .png file 
    image.save("image.png").unwrap();

    println!("Everything succeeded!");



    mod mandelbrot_shader {

        vulkano_shaders::shader!{
            ty: "compute",                  //calculate for a given pixel the mandelbrot set. The closer the pixel is to being in the set, the brighter it will be. In the set = complete white
            src: "
            #version 450

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
            
            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
            
            void main() {
                vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
                vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);
            
                vec2 z = vec2(0.0, 0.0);
                float i;
                for (i = 0.0; i < 1.0; i += 0.005) {
                    z = vec2(
                        z.x * z.x - z.y * z.y + c.x,
                        z.y * z.x + z.x * z.y + c.y
                    );
            
                    if (length(z) > 4.0) {
                        break;
                    }
                }
            
                vec4 to_write = vec4(vec3(i), 1.0);
                imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
            }
            "
        }
    }

    //create a new image to draw to
    let image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.family()),
    )
    .unwrap();
    
    //create an ImageView, which is needed in order to pass the image to the GPU shader
    let view = ImageView::new_default(image.clone()).unwrap();


    let shader = mandelbrot_shader::load(device.clone()).expect("failed to create ShaderModule for Mandelbrot");
        //compute pipeline to actually execute the shader
        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("failed to create compute pipeline");


    //create a descriptor set that includes the ImageView we just created
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())], // 0 is the binding
    )
    .unwrap();

    //create a buffer to store the output in
    let buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");


    //create a CommandBufferBuilder for the actual commandBuffer
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    //create a CommandBuffer with the correct commands to submit
    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone())
        .unwrap();
    
    let command_buffer = builder.build().unwrap();
    
    println!("Starting Mandelbrot calculation ..." );
    //submit the commands to the GPU
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    //wait for the execution to be finished
    future.wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("Mandelbrot.png").unwrap();

    println!("Everything succeeded!");




    //NEXT UP: GRAPHICS PIPELINE and DRAWING TRIANGLES


    //create a struct to represent a vertex, a single corner of a triangle. The only shape the GPU can actually draw without tessellation
    #[repr(C)]
    #[derive(Default, Copy, Clone, Zeroable, Pod)]
    struct Vertex {
        position: [f32; 2],
    }
    //tell vulkano, that our vertex struct (name does not matter, aswell as the name for position) represents a vertex
    vulkano::impl_vertex!(Vertex, position);

    //create 3 vertices, which make up our triangle
    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [ 0.0,  0.5] };
    let vertex3 = Vertex { position: [ 0.5, -0.25] };


    //create a buffer that contains these 3 vertices (called a vertex buffer)
    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),           //vertex buffer indicates how the buffer is meant to be used and has nothing to do with its properties
        false,
        vec![vertex1, vertex2, vertex3].into_iter(),
    )
    .unwrap();

    //creating a vertex shader is similiar to a compute shader. Here we set the gl position to the position of the element in the vector, ergo our vertex positions!
    mod my_vertex_shader {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: "
    #version 450
    
    layout(location = 0) in vec2 position;
    
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
    }"
        }
    }
    
    mod my_fragment_shader {
        vulkano_shaders::shader!{
            ty: "fragment",
            src: "
    #version 450
    
    layout(location = 0) out vec4 f_color;
    
    void main() {
        f_color = vec4(1.0, 0.0, 0.0, 1.0);
    }"
        }
    }

    //get objects that represent our shaders
    let my_vertex_shader = my_vertex_shader::load(device.clone()).expect("failed to create shader module");
    let my_fragment_shader = my_fragment_shader::load(device.clone()).expect("failed to create shader module");



    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [1024.0, 1024.0],
        depth_range: 0.0..1.0,
    };

    //create a render pass, a time period, during which the GPU is allowed to draw on the source
    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: Format::R8G8B8A8_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();


    let image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1, // images can be arrays of layers
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.family()),
    )
    .unwrap();
    

    let view = ImageView::new_default(image.clone()).unwrap();
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap();


    let pipeline = GraphicsPipeline::start()
        // Describes the layout of the vertex input and how should it behave
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify
        // which one.
        .vertex_shader(my_vertex_shader.entry_point("main").unwrap(), ())
        // Indicate the type of the primitives (the default is a list of triangles)
        .input_assembly_state(InputAssemblyState::new())
        // Set the fixed viewport
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        // Same as the vertex input, but this for the fragment input
        .fragment_shader(my_fragment_shader.entry_point("main").unwrap(), ())
        // This graphics pipeline object concerns the first pass of the render pass.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        // Now that everything is specified, we call `build`.
        .build(device.clone())
        .unwrap();
    



        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        
                        //create the target buffer to draw to, this is not needed for the actual computation, just to see the output
                        let buf = CpuAccessibleBuffer::from_iter(
                            device.clone(),
                            BufferUsage::all(),
                            false,
                            (0..1024 * 1024 * 4).map(|_| 0u8),
                        )
                        .expect("failed to create buffer");

    

    println!("Beginning drawing of triangle!");


    //once again, create a commandBufferBuilder, to submit commands to the GPU
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    
    //tell the build what we want
    builder
        .begin_render_pass(                     //enter the render pass, which means tell the GPU, we want to draw on the source
            framebuffer.clone(),                //the framebuffer contains elements which are needed by the draw calls
            SubpassContents::Inline,  
            vec![[0.0, 0.0, 1.0, 1.0].into()], //clear the view with all blue
        )
        .unwrap()
    
        // new stuff
        .bind_pipeline_graphics(pipeline.clone())           //tell the GPU we want to execute the commands defined in the graphics pipeline and subsequent shaders
        .bind_vertex_buffers(0, vertex_buffer.clone()) //add our vertices as data to be drawn at location 0
        .draw(
            3, 1, 0, 0, // 3 is the number of vertices, 1 is the number of instances, start at vertex 0, instance 0
        )
        
        .unwrap()
        .end_render_pass()
        .unwrap()    
        .copy_image_to_buffer(image, buf.clone()) //copy our result to our image buffer
        .unwrap();
    
    // build the commandBuffer 
    let command_buffer = builder.build().unwrap();
    
    //execute the commands defined in the command buffer structure on the GPU and ask for a signal fence (done signal)
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    
    //read the content from the buffer we copied our result to
    let buffer_content = buf.read().unwrap();
    //create an image from the buffer data
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    //save the image to a .png file
    image.save("triangle.png").unwrap();
    
    println!("Everything succeeded!");




    //NEXT UP: WINDOWING!

    //tell vulkan that we need to load extensions in order to render to the viewport
    let required_extensions = vulkano_win::required_extensions();
    //this is required, too. But I have no idea why. Something about the need to forward the instance to the window. This essentially loads a different configuration for vulkan
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .expect("failed to create instance");

    //create an EventLoop and a surface that correspons to it. Thus we will be able to handle events (changed sizes, mouse clicks, button pressed, refreshs, etc)
    let event_loop = EventLoop::new(); 
    let surface = WindowBuilder::new()  //abstraction of object that can be drawn to. Get the actual window by calling surface.window()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();


    event_loop.run(|event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            },
             _ => ()
        }
    });
}

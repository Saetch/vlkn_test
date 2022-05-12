
use rand::Rng;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use std::process::CommandArgs;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::device::{Device, Features, DeviceCreateInfo, QueueCreateInfo, Queue};
use bytemuck::{Pod, Zeroable};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::sync:: {self, GpuFuture};
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::Pipeline;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::format::{Format, ClearValue};
use image::{ImageBuffer, Rgba};
use vulkano::image::view::ImageView;

fn main() {
    //initialize vulkan and get the physical device (GPU)
    let instance = Instance::new(InstanceCreateInfo::default()).expect("failed to create instance");
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

}

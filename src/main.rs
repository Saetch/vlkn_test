
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
            ty: "compute",                  //use version 450, create workgroups (1024) with a x size of 64, y size of 1 and z size of 1 (one dimensional data structure, otherwise use y for 2 and y and z for 3 dimensions)  -> define a buffer data structure and call the function
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
}

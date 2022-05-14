
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::swapchain::Surface;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use std::sync::Arc;
use vulkano::device::{Device,  DeviceCreateInfo, QueueCreateInfo, Queue};
use vulkano::device::DeviceExtensions;
use winit::event::{Event, WindowEvent};

//this 'a is a lifetime parameter, it indicates in this case that instance aswell as the returned values have the same lifetime
//this is needed for Rust's borrow checker (the thing that lets Rust not need memory management or garbage collection), it indicates how long a value or object is valid. In this case the phys_device and queue family is only valid as long as the device(loaded instance of vulkan) is
fn select_physical_device<'a>(
    instance: &'a Arc<Instance>,
    surface: Arc<Surface<Window>>,
    device_extensions: &DeviceExtensions,
) -> (PhysicalDevice<'a>, QueueFamily<'a>) {


    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)      //get a list of all physical devices that support vulkan
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))               //filter the iterator for all devices that have a set of supported extensions which includes the provided device_extensions
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))  //further filter only for devices which queue families support graphics and surface (and look for which queue family it is). Unwrap_or() maps the unwrap error to the provided default value (in this case false)
                .map(|q| (p, q))                                                                          //create a tuple from the physical device and the queue family found
        })
        .min_by_key(|(p, _)| match p.properties().device_type {                                        //provide the priority in which to chose 0->1->2->3->4, based on the type, dedicated GPU scores best, so this is chosen first, if it exists, otherwise integrated GPU and so on ...
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

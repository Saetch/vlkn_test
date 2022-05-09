
use rand::Rng;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use std::time::{Duration, SystemTime};
fn main() {
    let instance = Instance::new(InstanceCreateInfo::default()).expect("failed to create instance");
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
    let mut x:i64 = 0;
    let mut rng = rand::thread_rng();
    let now = SystemTime::now();
    for i in 0 as i64 ..100000000  {
        x+= rng.gen::<i32>() as i64 % 5 ;
    }

    println!("Interesting! {}, this took {} ms", &x, now.elapsed().unwrap().as_millis());
    println!("Hey, I've finished successfully!" );
}

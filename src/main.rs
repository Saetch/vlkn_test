
use rand::Rng;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use std::time::{Duration, SystemTime};
fn main() {
    let instance = Instance::new(InstanceCreateInfo::default()).expect("failed to create instance");
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

}

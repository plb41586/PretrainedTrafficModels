use std::path::Path;

use pcap::{Capture, Device};

pub fn setup_capture_from_file(file_name: &Path) -> Capture<pcap::Offline> {
    let cap = Capture::from_file(file_name)
        .unwrap_or_else(|e| panic!("Failed to open pcap file {}: {e}", file_name.display()));
    println!("Capture from pcap file started: {}", file_name.display());
    cap
}

/// Open a live capture on the named interface.
pub fn setup_capture(interface_name: &str) -> Capture<pcap::Active> {
    let device_list = Device::list().expect("Failed to list capture devices");
    let available: Vec<String> = device_list.iter().map(|d| d.name.clone()).collect();

    let device = device_list
        .into_iter()
        .find(|d| d.name == interface_name)
        .unwrap_or_else(|| {
            panic!(
                "Capture device {:?} not found. Available devices: {}",
                interface_name,
                available.join(", ")
            )
        });

    println!("Capture from network interface started: {}", device.name);

    Capture::from_device(device)
        .expect("Failed to open capture device")
        .promisc(true)
        .snaplen(65535)
        .open()
        .expect("Failed to activate capture")
}

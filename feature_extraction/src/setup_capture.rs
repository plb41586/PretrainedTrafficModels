use pcap::{Capture, Device};

pub fn setup_capture_from_file(file_name: &str) -> Capture<pcap::Offline> {
    // Setup Capture from File
    let cap = Capture::from_file(file_name).expect("Failed to open pcap file");
    println!("Capture from Pcap file started!");
    println!("Using file {}", file_name);
    cap
}

pub fn setup_capture(interface_name: &str) -> Capture<pcap::Active> {
    // Get the Device List
    let device_list = Device::list().expect("device list failed");
    // Select device with given name
    let device = device_list
        .into_iter()
        .find(|d| d.name == interface_name)
        .expect("device not found");

    println!("Using device {}", device.name);

    // Setup Capture from Network Interface
    let cap = Capture::from_device(device)
        .unwrap()
        .promisc(true)
        .open()
        .unwrap();
    println!("Capture from Network Interface started!");
    cap
}

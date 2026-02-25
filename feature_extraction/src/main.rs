use pcap::{Capture, Device};
// mod csv_handler;
mod feature_parser;
use feature_parser::{parse_packet, ParsedPacketSet};
mod error_handling_tools;
use error_handling_tools::print_error_chain;
mod setup_capture;
use setup_capture::{
    setup_capture_from_file,
    setup_capture,
};
mod falkor_integration;
mod topology_loader;
use topology_loader::load_network_topology;
mod flow_tracker;
use flow_tracker::FlowTracker;
use flow_tracker::FlowKey;
use std::net::IpAddr;
mod redis_integration;
use redis_integration::connect_to_redis;
use redis::{Client, Commands};
use std::env;
// use std::{fs::File, thread, time};

// Performance measurement
use std::time::{Duration, Instant};

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    
    let file_path = if args.len() > 1 {
        // Use the provided file path from command line
        &args[1]
    } else {
        // Use default file path if no argument provided
        // "/workspace/data/Network_Traffic/Phase2/2ndPhase-timed-MergedV2.pcap"
        "/workspace/data/Network_Traffic/Phase1/1stPhase-timed-Merged.pcap"
        // "/workspace/data/Network_Traffic/filtered/Phase2--65_128to65_2.pcapng"
    };

    let cache_payloads: bool = if args.len() > 2 {
        args[2].to_lowercase() == "true"
    } else {
        false
    };

    let graph_name: &str = if args.len() > 3 {
        &args[3]
    }else {
       "Phase2"
    };
    
    println!("Caching payloads to Redis Queue: {}", cache_payloads);

    println!("Using file: {}", file_path);

    // Setup connection to redis server
    let mut conn = connect_to_redis();
    println!("Connected to Redis");
    println!("Connecting to Falkor");
    let FalkorClient = falkor_integration::connect_to_falkor();
    // let mut FalkorGraph = FalkorClient.select_graph("IDS_Node_1");
    let mut graph = FalkorClient.select_graph(graph_name);
    load_network_topology(&mut graph, "/workspace/data/Network_Traffic/Devices.toml")
        .expect("Failed to load network topology");
    println!("Loaded network topology into FalkorDB");
    // Setup capture from file or network interface
    let mut cap = setup_capture_from_file(file_path);
    
    // Capture and parse packets
    // Push features to FalkorDB to construct Communication Graph
    let mut pak_num = 0;
    let mut issues: i32 = 0;
    
    // Initialize flow tracker
    let mut tracker = FlowTracker::default();
    println!("Flow tracker initialized!");
    
    while let Ok(packet) = cap.next_packet() {
        match parse_packet(&packet) {
            Ok(parsed_packet) => {
                let key: FlowKey = FlowKey::from_parsed_packet(&parsed_packet).unwrap();
                let _is_new_flow = tracker.process_packet(&key, parsed_packet.payload_set.data.len() as u16, parsed_packet.timevalue);
                let normalized_key = key.normalize();
                let stats = tracker.get_flow_stats(&normalized_key).unwrap();
                
                if cache_payloads {
                    // Push PayloadSet Redis Queue for further analysis in Python
                    let flow_identifier = key.to_string();
                    let payloadset = parsed_packet.payload_set.clone();
                    redis_integration::push_payloadset_to_redis_queue(&mut conn, &flow_identifier, &payloadset);
                }
            }
            Err(e) => {
                println!("Error while handling packet {:?}", pak_num);
                print_error_chain(&e);
                issues += 1;
            }
        }
        pak_num = pak_num + 1;
        // if pak_num == 10000 {
        //     println!("Handled 1000 packets, exiting for testing purposes");
        //     break;
        // }
    }
    
    println!("Estimated memory usage: {} bytes", tracker.estimated_memory_usage());
    // tracker.print_flows();
    println!("Pushing flows to FalkorDB");
    let now = Instant::now();
    tracker.push_flows_to_falkor(graph_name);
    println!("Pushed all flows to FalkorDB in {:?}", now.elapsed());
    println!("Finished capturing packets with {} issues", issues);
    println!("Finished capturing packets: {:?}", pak_num);
    println!("Exiting");
}
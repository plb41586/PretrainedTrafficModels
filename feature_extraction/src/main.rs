use falkordb::FalkorSyncClient;
use libc::connect;
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
// CLI interface
use clap::Parser;
use std::path::PathBuf;

/// PCAP network traffic analyzer
#[derive(Parser, Debug)]
#[command(name = "pcap-analyzer", version, about)]
struct Cli {
    /// Path to the PCAP file to analyze
    #[arg(short, long, default_value = "/home/plb41586/workspace/data/CICAPT-IIoT/Network_Traffic/Phase1/1stPhase-timed-Merged.pcap")]
    file: PathBuf,

    /// Cache packet payloads in memory during processing
    #[arg(short, long, default_value_t = true)]
    cache_payloads: bool,

    /// FalkorDB graph name. If provided, results are written to FalkorDB.
    /// If omitted, FalkorDB integration is disabled.
    #[arg(short, long)]
    graph_name: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let use_falkor = cli.graph_name.is_some();
    let graph_name = cli.graph_name.as_deref().unwrap_or("");

    println!("File:           {}", cli.file.display());
    println!("Cache payloads: {}", cli.cache_payloads);
    println!("Use FalkorDB:   {}", use_falkor);
    if use_falkor {
        println!("Graph name:     {}", graph_name);
    }

    // Setup connection to redis server
    let mut redis_conn: Option<redis::Connection> = if cli.cache_payloads {
        Some(connect_to_redis())
    } else {
        None
    };
    // Setup connection to FalkorDB
    let FalkorClient: Option<FalkorSyncClient> = if use_falkor {
        Some(falkor_integration::connect_to_falkor())
    } else {
        None
    };
    if let Some(client) = &FalkorClient {
        let mut graph = client.select_graph(graph_name);
        load_network_topology(&mut graph, "/workspace/data/Network_Traffic/Devices.toml")
            .expect("Failed to load network topology");
        println!("Loaded network topology into FalkorDB");
        // Setup capture from file or network interface
    }
    let mut cap = setup_capture_from_file(&cli.file);
    
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
                if let Some(conn) = &mut redis_conn {
                    let flow_identifier = key.to_string();
                    let payloadset = parsed_packet.payload_set.clone();
                    redis_integration::push_payloadset_to_redis_queue(conn, &flow_identifier, &payloadset);
                }
            }
            Err(e) => {
                println!("Error while handling packet {:?}", pak_num);
                print_error_chain(&e);
                issues += 1;
            }
        }
        pak_num = pak_num + 1;
        if pak_num == 10000 {
            println!("Handled 1000 packets, exiting for testing purposes");
            break;
        }
    }
    
    println!("Estimated memory usage: {} bytes", tracker.estimated_memory_usage());
    if let Some(client) = &FalkorClient {
        println!("Pushing flows to FalkorDB");
        let now = Instant::now();
        tracker.push_flows_to_falkor(graph_name, &client);
        println!("Pushed all flows to FalkorDB in {:?}", now.elapsed());
    }
    println!("Finished capturing packets with {} issues", issues);
    println!("Finished capturing packets: {:?}", pak_num);
    println!("Exiting");
}
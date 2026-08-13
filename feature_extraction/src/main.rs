mod error_handling_tools;
mod falkor_integration;
mod feature_parser;
mod flow_tracker;
mod redis_integration;
mod setup_capture;
mod topology_loader;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use falkordb::FalkorSyncClient;
use pcap::Activated;
use polars::prelude::*;

use error_handling_tools::print_error_chain;
use feature_parser::{parse_packet, ParsedPacketSet};
use flow_tracker::{FlowKey, FlowTracker};
use redis_integration::connect_to_redis;
use setup_capture::{setup_capture, setup_capture_from_file};
use topology_loader::load_network_topology;

/// PCAP network traffic analyzer.
///
/// Reads packets from a capture file or a live interface, tracks bidirectional
/// flows, and optionally writes them to FalkorDB, streams payloads to Redis,
/// and exports parsed packets to Parquet. Every backend is opt-in: with no
/// backend flags the tool parses and reports without touching the network.
#[derive(Parser, Debug)]
#[command(name = "feature_extractor", version, about, long_about = None)]
struct Cli {
    /// Path to the pcap file to analyze. Mutually exclusive with --interface.
    #[arg(short, long, value_name = "PATH", conflicts_with = "interface")]
    file: Option<PathBuf>,

    /// Capture live from this network interface instead of a file.
    #[arg(short, long, value_name = "NAME")]
    interface: Option<String>,

    /// FalkorDB graph name. If given, flows are written to FalkorDB;
    /// if omitted, FalkorDB is not contacted at all.
    #[arg(short, long, value_name = "NAME")]
    graph_name: Option<String>,

    /// Stream packet payloads and feature sets to a Redis queue.
    #[arg(short, long, default_value_t = false)]
    cache_payloads: bool,

    /// Write parsed packets to this Parquet file. If omitted, nothing is exported.
    #[arg(short, long, value_name = "PATH")]
    pl_outfile: Option<PathBuf>,

    /// Network topology TOML to load into the graph before capture.
    /// Only used when --graph-name is set.
    #[arg(short, long, value_name = "PATH")]
    topology: Option<PathBuf>,

    /// FalkorDB host:port.
    #[arg(long, value_name = "HOST:PORT", default_value = "falkordb:6379")]
    falkor_host: String,

    /// Redis host:port.
    #[arg(long, value_name = "HOST:PORT", default_value = "redis:6379")]
    redis_host: String,

    /// Rows buffered in memory before each Parquet row group is flushed.
    #[arg(long, value_name = "N", default_value_t = 50_000)]
    pl_chunk_size: usize,

    /// Stop after this many packets. 0 means no limit.
    #[arg(long, value_name = "N", default_value_t = 0)]
    limit: u64,
}

/// Tallies for the run, reported at the end.
#[derive(Default)]
struct RunStats {
    packets: u64,
    parse_errors: u64,
    /// Parsed fine, but carried no ARP/ICMP/TCP/UDP tuple to key a flow on.
    unkeyed: u64,
    redis_errors: u64,
}

fn main() {
    let cli = Cli::parse();

    if cli.file.is_none() && cli.interface.is_none() {
        eprintln!("Error: provide either --file <PATH> or --interface <NAME>.");
        std::process::exit(2);
    }

    let use_falkor = cli.graph_name.is_some();
    let graph_name = cli.graph_name.clone().unwrap_or_default();

    println!("Source:         {}", match (&cli.file, &cli.interface) {
        (Some(f), _) => format!("file {}", f.display()),
        (_, Some(i)) => format!("interface {}", i),
        _ => unreachable!(),
    });
    println!("Cache payloads: {}", cli.cache_payloads);
    println!("Use FalkorDB:   {}", use_falkor);
    if use_falkor {
        println!("Graph name:     {}", graph_name);
        println!("FalkorDB host:  {}", cli.falkor_host);
    }
    if cli.cache_payloads {
        println!("Redis host:     {}", cli.redis_host);
    }
    if let Some(out) = &cli.pl_outfile {
        println!("Parquet out:    {}", out.display());
    }

    // Redis is opened only when payload caching is requested.
    let mut redis_conn = if cli.cache_payloads {
        println!("Connecting to Redis...");
        Some(connect_to_redis(&cli.redis_host))
    } else {
        None
    };

    // FalkorDB is opened only when a graph name is given. One client, shared
    // between topology loading and the final flow write.
    let falkor_client: Option<FalkorSyncClient> = if use_falkor {
        println!("Connecting to FalkorDB...");
        Some(falkor_integration::connect_to_falkor(&cli.falkor_host))
    } else {
        None
    };

    if let (Some(client), Some(topology)) = (&falkor_client, &cli.topology) {
        let mut graph = client.select_graph(&graph_name);
        load_network_topology(&mut graph, &topology.to_string_lossy())
            .expect("Failed to load network topology");
        println!("Loaded network topology from {}", topology.display());
    }

    let mut parquet = match cli.pl_outfile.as_ref() {
        Some(path) => Some(ParquetSink::create(path, cli.pl_chunk_size)),
        None => None,
    };

    let mut tracker = FlowTracker::default();
    println!("Flow tracker initialized. Starting capture...");

    let now = Instant::now();
    let stats = match (&cli.file, &cli.interface) {
        (Some(path), _) => {
            let mut cap = setup_capture_from_file(path);
            run_capture(&mut cap, &mut tracker, &mut redis_conn, &mut parquet, cli.limit)
        }
        (_, Some(name)) => {
            let mut cap = setup_capture(name);
            run_capture(&mut cap, &mut tracker, &mut redis_conn, &mut parquet, cli.limit)
        }
        _ => unreachable!(),
    };
    let elapsed = now.elapsed();

    if let Some(sink) = parquet.take() {
        let rows = sink.finish();
        println!("Wrote {} rows to Parquet", rows);
    }

    println!();
    println!("Capture finished in {:?}", elapsed);
    println!("  packets read:        {}", stats.packets);
    println!("  parse errors:        {}", stats.parse_errors);
    println!("  packets without key: {}", stats.unkeyed);
    if stats.redis_errors > 0 {
        println!("  redis push errors:   {}", stats.redis_errors);
    }
    println!("  flows tracked:       {}", tracker.flow_count());
    println!("  unique IPs:          {}", tracker.unique_ip_count());
    println!("  flow table size:     ~{} bytes", tracker.estimated_memory_usage());

    if let Some(client) = &falkor_client {
        println!("\nPushing flows to FalkorDB...");
        let now = Instant::now();
        let (written, failed) = tracker.push_flows_to_falkor(&graph_name, client);
        println!("Pushed {} flows in {:?} ({} failed)", written, now.elapsed(), failed);
    }
}

/// Drive the capture loop. Generic over offline and live captures.
fn run_capture<T: Activated + ?Sized>(
    cap: &mut pcap::Capture<T>,
    tracker: &mut FlowTracker,
    redis_conn: &mut Option<redis::Connection>,
    parquet: &mut Option<ParquetSink>,
    limit: u64,
) -> RunStats {
    let mut stats = RunStats::default();

    while let Ok(packet) = cap.next_packet() {
        stats.packets += 1;

        match parse_packet(&packet) {
            Ok(mut parsed_packet) => {
                // Not every parsable packet yields a flow tuple — IGMP, bare
                // Ethernet control frames and anything below the transport
                // layer land here. Skip and count them; unwrapping killed the
                // whole run on the first one.
                let Some(key) = FlowKey::from_parsed_packet(&parsed_packet) else {
                    stats.unkeyed += 1;
                    continue;
                };

                // The canonical key is what identifies the bidirectional flow,
                // so it is also what downstream consumers should group on.
                let flow_identifier = key.normalize().to_string();
                parsed_packet.flow_key = flow_identifier.clone();

                tracker.process_packet(&key, &parsed_packet);

                if let Some(conn) = redis_conn.as_mut() {
                    if let Err(e) = redis_integration::push_payloadset_to_redis_queue(
                        conn,
                        &flow_identifier,
                        &parsed_packet.payload_set,
                    ) {
                        stats.redis_errors += 1;
                        eprintln!("Redis payload push failed: {}", e);
                    }
                    if let Err(e) = redis_integration::push_featureset_to_redis_queue(
                        conn,
                        &flow_identifier,
                        &parsed_packet.features,
                    ) {
                        stats.redis_errors += 1;
                        eprintln!("Redis feature push failed: {}", e);
                    }
                }

                if let Some(sink) = parquet.as_mut() {
                    sink.push(&parsed_packet);
                }
            }
            Err(e) => {
                stats.parse_errors += 1;
                eprintln!("Error while handling packet {}", stats.packets);
                print_error_chain(&e);
            }
        }

        if limit > 0 && stats.packets >= limit {
            println!("Reached --limit of {} packets, stopping.", limit);
            break;
        }
    }

    stats
}

/// Streams parsed packets to Parquet in row-group sized chunks.
///
/// The previous implementation buffered every packet — full frame bytes plus a
/// same-length mask — until the capture ended, which exhausts memory on any
/// large capture. This flushes a row group every `chunk_size` packets so
/// resident memory stays bounded regardless of capture size.
struct ParquetSink {
    writer: polars::io::parquet::write::BatchedWriter<std::fs::File>,
    chunk_size: usize,
    rows_written: usize,

    proto_hierarchy: Vec<String>,
    flow_key: Vec<String>,
    timestamp_s: Vec<i64>,
    timestamp_us: Vec<i64>,
    data: Vec<Vec<u8>>,
    mask: Vec<Vec<u8>>,
    header_len: Vec<u32>,
}

impl ParquetSink {
    fn schema() -> Schema {
        Schema::from_iter([
            Field::new("proto_hierarchy".into(), DataType::String),
            Field::new("flow_key".into(), DataType::String),
            Field::new("timestamp_s".into(), DataType::Int64),
            Field::new("timestamp_us".into(), DataType::Int64),
            Field::new("data".into(), DataType::Binary),
            Field::new("mask".into(), DataType::Binary),
            Field::new("header_len".into(), DataType::UInt32),
        ])
    }

    fn create(path: &std::path::Path, chunk_size: usize) -> Self {
        let file = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("Could not create {}: {e}", path.display()));
        let writer = ParquetWriter::new(file)
            .batched(&Self::schema())
            .expect("Failed to open Parquet writer");

        let chunk_size = chunk_size.max(1);
        Self {
            writer,
            chunk_size,
            rows_written: 0,
            proto_hierarchy: Vec::with_capacity(chunk_size),
            flow_key: Vec::with_capacity(chunk_size),
            timestamp_s: Vec::with_capacity(chunk_size),
            timestamp_us: Vec::with_capacity(chunk_size),
            data: Vec::with_capacity(chunk_size),
            mask: Vec::with_capacity(chunk_size),
            header_len: Vec::with_capacity(chunk_size),
        }
    }

    fn push(&mut self, packet: &ParsedPacketSet) {
        self.proto_hierarchy.push(packet.payload_set.proto_hierarchy.clone());
        self.flow_key.push(packet.flow_key.clone());
        self.timestamp_s.push(packet.timevalue.seconds);
        self.timestamp_us.push(packet.timevalue.microseconds);
        self.data.push(packet.payload_set.data.clone());
        // The mask is one bool per byte; pack it to bytes so Parquet stores it
        // as a binary blob rather than a list of 8-bit integers.
        self.mask.push(
            packet
                .payload_set
                .mask
                .iter()
                .map(|&b| if b { 1u8 } else { 0u8 })
                .collect(),
        );
        self.header_len.push(packet.payload_set.header_len);

        if self.proto_hierarchy.len() >= self.chunk_size {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.proto_hierarchy.is_empty() {
            return;
        }

        let n = self.proto_hierarchy.len();
        let df = DataFrame::new(n, vec![
            Column::new("proto_hierarchy".into(), std::mem::take(&mut self.proto_hierarchy)),
            Column::new("flow_key".into(), std::mem::take(&mut self.flow_key)),
            Column::new("timestamp_s".into(), std::mem::take(&mut self.timestamp_s)),
            Column::new("timestamp_us".into(), std::mem::take(&mut self.timestamp_us)),
            BinaryChunked::from_iter_values(
                "data".into(),
                std::mem::take(&mut self.data).into_iter(),
            )
            .into_column(),
            BinaryChunked::from_iter_values(
                "mask".into(),
                std::mem::take(&mut self.mask).into_iter(),
            )
            .into_column(),
            Column::new("header_len".into(), std::mem::take(&mut self.header_len)),
        ])
        .expect("Failed to build Parquet row group");

        self.writer
            .write_batch(&df)
            .expect("Failed to write Parquet row group");
        self.rows_written += n;
    }

    fn finish(mut self) -> usize {
        self.flush();
        self.writer.finish().expect("Failed to finalize Parquet file");
        self.rows_written
    }
}

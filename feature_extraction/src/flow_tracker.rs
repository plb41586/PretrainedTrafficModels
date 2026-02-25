use std::net::{IpAddr, SocketAddr};
use falkordb::{FalkorSyncClient, GraphSchema};
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::{Duration, SystemTime};

use crate::feature_parser::{ParsedPacketSet, TimeStamp};

use falkordb::{SyncGraph};
use crate::falkor_integration;

type FlowTable = FxHashMap<FlowKey, FlowStats>;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct FlowKey {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: String,
}

impl FlowKey {
    pub fn to_string(&self) -> String {
        format!("{}:{} -> {}:{} ({})", self.src_ip, self.src_port, self.dst_ip, self.dst_port, self.protocol)
    }

    pub fn is_normalized(&self) -> bool {
        self.src_ip < self.dst_ip || (self.src_ip == self.dst_ip && self.src_port <= self.dst_port)
    }

    pub fn normalized_key(&self) -> FlowKey {
        if self.src_ip < self.dst_ip || (self.src_ip == self.dst_ip && self.src_port < self.dst_port) {
            println!("Check for normalization before use!");
            self.clone()
        } else {
            FlowKey {
                src_ip: self.dst_ip,
                dst_ip: self.src_ip,
                src_port: self.dst_port,
                dst_port: self.src_port,
                protocol: self.protocol.clone(),
            }
        }
    }
}
pub struct FlowStats {
    // Directional counters
    pub fwd_packet_count: u64,
    pub bwd_packet_count: u64,
    pub fwd_byte_count: u64,
    pub bwd_byte_count: u64,
    
    // Timing
    pub first_seen: TimeStamp,
    pub last_seen: TimeStamp,
    pub duration: Duration,
    
    // Advanced statistics
    pub min_packet_size: u16,
    pub max_packet_size: u16,
    pub avg_packet_size: f32,
    
    pub avg_pps: f32,
    pub avg_bps: f32,
}

impl FlowStats {
    /// Create new flow statistics with initial packet data
    pub fn new(key: &FlowKey,packet_size: u16, first_seen: TimeStamp) -> Self {
        if key.src_ip < key.dst_ip || (key.src_ip == key.dst_ip && key.src_port < key.dst_port) {
            Self {
                fwd_byte_count: packet_size as u64,
                bwd_byte_count: 0,
                fwd_packet_count: 1,
                bwd_packet_count: 0,
                first_seen: first_seen,
                last_seen: first_seen,
                min_packet_size: packet_size,
                max_packet_size: packet_size,
                avg_bps: packet_size as f32,
                avg_pps: 1.0,
                avg_packet_size: packet_size as f32,
                duration: Duration::new(0, 0),
            }
        } else {
            Self {
                fwd_byte_count: 0,
                bwd_byte_count: packet_size as u64,
                fwd_packet_count: 0,
                bwd_packet_count: 1,
                first_seen: first_seen,
                last_seen: first_seen,
                min_packet_size: packet_size,
                max_packet_size: packet_size,
                avg_bps: packet_size as f32,
                avg_pps: 1.0,
                avg_packet_size: packet_size as f32,
                duration: Duration::new(0, 0),
            }
        }
    }

    /// Update statistics with new packet information
    pub fn update(&mut self, key: &FlowKey, packet_size: u16, packet_timestamp: TimeStamp) {
        let mut fwd: bool;
        if key.src_ip < key.dst_ip || (key.src_ip == key.dst_ip && key.src_port < key.dst_port) {
            fwd = true;
        } else {
            fwd = false;
        }
        if fwd {
            self.fwd_packet_count += 1;
            self.fwd_byte_count += packet_size as u64;
        } else {
            self.bwd_packet_count += 1;
            self.bwd_byte_count += packet_size as u64;
        }
        self.last_seen = packet_timestamp;
        self.duration = self.last_seen.duration_since(&self.first_seen);
        self.min_packet_size = self.min_packet_size.min(packet_size);
        self.max_packet_size = self.max_packet_size.max(packet_size);
        self.avg_packet_size = self.avg_packet_size();
        if self.duration.as_secs() > 0 {
            self.avg_pps = (self.fwd_packet_count + self.bwd_packet_count) as f32 / self.duration.as_secs_f32();
            self.avg_bps = (self.fwd_byte_count + self.bwd_byte_count) as f32 / self.duration.as_secs_f32()
        }
    }

    /// Calculate average packet size
    pub fn avg_packet_size(&self) -> f32 {
        let total_packet_count = self.fwd_packet_count + self.bwd_packet_count;
        let total_byte_count = self.fwd_byte_count + self.bwd_byte_count;
        if total_packet_count > 0 {
            total_byte_count as f32 / total_packet_count as f32
        } else {
            0.0
        }
    }

    /// Get flow duration
    pub fn duration(&self) -> Duration {
        self.last_seen.duration_since(&self.first_seen)
    }
}

/// Main flow tracking structure
pub struct FlowTracker {
    flows: FxHashMap<FlowKey, FlowStats>,
    flow_timeout: Duration,
    falkor_client: FalkorSyncClient,
    seen_ips: FxHashSet<IpAddr>,
}

impl FlowTracker {
    /// Initialize a new FlowTracker with specified capacity and timeout
    pub fn new(initial_capacity: usize, flow_timeout_seconds: u64) -> Self {
        Self {
            flows: FxHashMap::with_capacity_and_hasher(initial_capacity, Default::default()),
            flow_timeout: Duration::from_secs(flow_timeout_seconds),
            falkor_client: falkor_integration::connect_to_falkor(),
            seen_ips: FxHashSet::with_capacity_and_hasher(10_000, Default::default()),
        }
    }

    /// Initialize with default settings (1M flows, 300 second timeout)
    pub fn default() -> Self {
        Self::new(1_000_000, 300)
    }

    /// Process a packet and update flow statistics
    /// Returns true if this was a new flow, false if existing flow was updated
    pub fn process_packet(&mut self, flow_key: &FlowKey, packet_size: u16, packet_timestamp: TimeStamp) -> bool {
        let flow_key_clone = match flow_key.is_normalized() {
            true => flow_key.clone(),
            false => flow_key.normalized_key(),
        };

        match self.flows.get_mut(&flow_key_clone) {
            Some(flow_stats) => {
                // Update existing flow
                flow_stats.update(&flow_key, packet_size, packet_timestamp);
                false // Existing flow
            }
            None => {
                // Create new flow
                self.seen_ips.insert(flow_key.src_ip);
                self.seen_ips.insert(flow_key.dst_ip);
                let new_stats = FlowStats::new(&flow_key, packet_size, packet_timestamp);
                self.flows.insert(flow_key_clone, new_stats);
                return true; // New flow
            }
        }
    }

    /// Get statistics for a specific flow
    pub fn get_flow_stats(&self, flow_key: &FlowKey) -> Option<&FlowStats> {
        self.flows.get(flow_key)
    }

    /// Get mutable reference to flow statistics (for advanced operations)
    pub fn get_flow_stats_mut(&mut self, flow_key: &FlowKey) -> Option<&mut FlowStats> {
        self.flows.get_mut(flow_key)
    }

    /// Check if a flow exists
    pub fn contains_flow(&self, flow_key: &FlowKey) -> bool {
        self.flows.contains_key(flow_key)
    }

    /// Get current number of active flows
    pub fn flow_count(&self) -> usize {
        self.flows.len()
    }

    /// Check if flow table is empty
    pub fn is_empty(&self) -> bool {
        self.flows.is_empty()
    }

    /// Get memory usage estimation in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        // Rough estimation: each entry is approximately 100-150 bytes
        self.flows.len() * 128
    }

    pub fn print_flows(&self) {
        for (key, stats) in &self.flows {
            println!("{:?} => fwd_packets: {}, bwd_packets: {}, fwd_bytes: {}, bwd_bytes: {}, duration: {:?}, avg_pkt_size: {:.2}",
                key, stats.fwd_packet_count, stats.bwd_packet_count, stats.fwd_byte_count, stats.bwd_byte_count, stats.duration(), stats.avg_packet_size());
        }
    }

    pub fn has_seen_ip(&self, ip: &IpAddr) -> bool {
        self.seen_ips.contains(ip)
    }

    pub fn unique_ip_count(&self) -> usize {
        self.seen_ips.len()
    }

    pub fn push_flows_to_falkor(&self, graph_name: &str) {
        let mut graph = self.falkor_client.select_graph(graph_name);
        for ip in self.seen_ips.iter() {
            let _ = falkor_integration::merge_ip_address(&mut graph, ip);
        }
        for (key, stats) in &self.flows {
            let ret = falkor_integration::insert_edge_only(&mut graph, key, stats);
            match ret {
                Ok(_) => {
                    // Successfully inserted
                }
                Err(e) => {
                    println!("Error inserting flow into FalkorDB: {:?}", e);
                    println!("{:?}", key.to_string())
                }
            }
        }
    }
}

impl FlowKey {
    /// Create a new FlowKey
    pub fn new(src_ip: IpAddr, dst_ip: IpAddr, src_port: u16, dst_port: u16, protocol: String) -> Self {
        Self {
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol,
        }
    }
    
    pub fn from_parsed_packet(packet: &ParsedPacketSet) -> Option<Self> {
        let (src_ip, dst_ip, src_port, dst_port, protocol) = if packet.features.arp_set {
            (
                IpAddr::from(packet.features.arp_features.src_proto_ipv4),
                IpAddr::from(packet.features.arp_features.dst_proto_ipv4),
                0, // ARP doesn't have ports, use 0
                0, // ARP doesn't have ports, use 0
                packet.payload_set.proto_hierarchy.clone(),
            )
        } else if packet.features.icmp_set {
            (
                packet.features.ip_addresses.src_host,
                packet.features.ip_addresses.dst_host,
                0, // ICMP doesn't have ports, use 0
                0, // ICMP doesn't have ports, use 0
                packet.payload_set.proto_hierarchy.clone(),
            )
        } else if packet.features.tcp_set {
            (
                packet.features.ip_addresses.src_host,
                packet.features.ip_addresses.dst_host,
                packet.features.tcp_features.tcp_srcport,
                packet.features.tcp_features.tcp_dstport,
                packet.payload_set.proto_hierarchy.clone(),
            )
        }else if packet.features.udp_set {
            (
                packet.features.ip_addresses.src_host,
                packet.features.ip_addresses.dst_host,
                packet.features.udp_features.udp_port_src,
                packet.features.udp_features.udp_port_dst,
                packet.payload_set.proto_hierarchy.clone(),
            )
        } else {
            // Unsupported protocol for flow key
            return None;
        };

        Some(Self::new(src_ip, dst_ip, src_port, dst_port, protocol))
    }

    /// Create a normalized FlowKey for bidirectional flow tracking
    /// Ensures consistent representation regardless of packet direction
    pub fn new_normalized(src_ip: IpAddr, dst_ip: IpAddr, src_port: u16, dst_port: u16, protocol: String) -> Self {
        if src_ip < dst_ip || (src_ip == dst_ip && src_port < dst_port) {
            Self::new(src_ip, dst_ip, src_port, dst_port, protocol)
        } else {
            Self::new(dst_ip, src_ip, dst_port, src_port, protocol)
        }
    }

    pub fn normalize(&self) -> FlowKey {
        if self.src_ip < self.dst_ip || (self.src_ip == self.dst_ip && self.src_port < self.dst_port) {
            Self::new(self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol.clone())
        } else {
            Self::new(self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol.clone())
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration() {

        let mut tracker = FlowTracker::default();

        let src = IpAddr::from([192, 168, 1, 1]);
        let dst = IpAddr::from([192, 168, 1, 2]);
        let flow_key1 = FlowKey::new(src, dst, 12345, 80, "TCP".to_string());
        // Create initial timestamp: 10 seconds, 500000 microseconds (0.5s)
        let first_seen = TimeStamp::new(10, 500_000);
        tracker.process_packet(&flow_key1, 125, first_seen);

        let stats = tracker.get_flow_stats(&flow_key1).unwrap();
        let flow_count = tracker.flow_count();
        // Check initial stats
        assert_eq!(flow_count, 1);
        assert_eq!(stats.duration().as_secs(), 0);
        assert_eq!(stats.duration().as_nanos(), 0);
        assert_eq!(stats.fwd_byte_count, 125);
        assert_eq!(stats.bwd_byte_count, 0);
        assert_eq!(stats.fwd_packet_count, 1);
        assert_eq!(stats.bwd_packet_count, 0);
        assert_eq!(stats.min_packet_size, 125);
        assert_eq!(stats.max_packet_size, 125);

        // Update with packet 2 seconds later: 12 seconds, 500000 microseconds
        let second_packet = TimeStamp::new(12, 500_000);
        let flow_key2 = FlowKey::new(dst, src, 80, 12345, "TCP".to_string());
        tracker.process_packet(&flow_key2, 150, second_packet);
        let stats = tracker.get_flow_stats(&flow_key1).unwrap();
        
        println!("Duration after second packet: {:?}", stats.duration().as_secs());
        // Check updated stats
        assert_eq!(tracker.flow_count(), 1);
        assert_eq!(stats.duration().as_secs(), 2);
        assert_eq!(stats.fwd_byte_count, 125);
        assert_eq!(stats.bwd_byte_count, 150);
        assert_eq!(stats.fwd_packet_count, 1);
        assert_eq!(stats.bwd_packet_count, 1);
        assert_eq!(stats.min_packet_size, 125);
        assert_eq!(stats.max_packet_size, 150);
        
        // Update with another packet 1.5 seconds later: 14 seconds, 5 microseconds
        let third_packet = TimeStamp::new(14, 5);
        tracker.process_packet(&flow_key1, 80, third_packet);
        let stats = tracker.get_flow_stats(&flow_key1).unwrap();
        
        // Check updated stats
        assert_eq!(tracker.flow_count(), 1);
        assert_eq!(stats.fwd_byte_count, 205);
        assert_eq!(stats.bwd_byte_count, 150);
        assert_eq!(stats.fwd_packet_count, 2);
        assert_eq!(stats.bwd_packet_count, 1);
        assert_eq!(stats.min_packet_size, 80);
        assert_eq!(stats.max_packet_size, 150);
        let duration = stats.duration();
        assert_eq!(duration.as_secs(), 3);
        assert_eq!(duration.as_millis(), 3_500);
    }
}
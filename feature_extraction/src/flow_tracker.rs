use std::mem::size_of;
use std::net::IpAddr;
use std::time::Duration;

use entropy::shannon_entropy;
use falkordb::FalkorSyncClient;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::falkor_integration;
use crate::feature_parser::{ParsedPacketSet, TimeStamp, TransportProtocol};

/// The transport-level protocol that identifies a flow.
///
/// This is deliberately a closed set rather than the free-form protocol
/// hierarchy string. Two things depend on that:
///
/// 1. Flow identity. The hierarchy string is rebuilt per packet from whatever
///    parsed successfully, so a bare TCP ACK and an MQTT publish on the *same*
///    connection produce different strings. Keying on them split one connection
///    into several flows.
/// 2. Cypher safety. The relationship type is the one part of a query that
///    cannot be parameterised, so it has to be interpolated. Drawing it from a
///    closed set of ASCII identifiers makes that provably safe.
///
/// The full hierarchy still travels with the flow — see `FlowStats::proto_hierarchy`,
/// which is written to the graph as an ordinary (parameterised) edge property.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum FlowProtocol {
    Arp,
    Icmpv4,
    Icmpv6,
    Tcp,
    Udp,
}

impl FlowProtocol {
    pub fn as_str(&self) -> &'static str {
        match self {
            FlowProtocol::Arp => "ARP",
            FlowProtocol::Icmpv4 => "ICMPv4",
            FlowProtocol::Icmpv6 => "ICMPv6",
            FlowProtocol::Tcp => "TCP",
            FlowProtocol::Udp => "UDP",
        }
    }
}

impl std::fmt::Display for FlowProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct FlowKey {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: FlowProtocol,
}

impl FlowKey {
    pub fn new(
        src_ip: IpAddr,
        dst_ip: IpAddr,
        src_port: u16,
        dst_port: u16,
        protocol: FlowProtocol,
    ) -> Self {
        Self { src_ip, dst_ip, src_port, dst_port, protocol }
    }

    /// Build a flow key from a parsed packet.
    ///
    /// Returns `None` for anything that is not ARP, ICMP, TCP or UDP — the
    /// caller must skip those packets rather than unwrap.
    pub fn from_parsed_packet(packet: &ParsedPacketSet) -> Option<Self> {
        let (src_ip, dst_ip, src_port, dst_port, protocol) = if packet.features.arp_set {
            (
                IpAddr::from(packet.features.arp_features.src_proto_ipv4),
                IpAddr::from(packet.features.arp_features.dst_proto_ipv4),
                0,
                0,
                FlowProtocol::Arp,
            )
        } else if packet.features.icmp_set {
            // ICMPv4 and ICMPv6 are distinct protocols and must not share a
            // flow. `transport_protocol` is populated during parsing.
            let protocol = match packet.features.transport_protocol {
                TransportProtocol::ICMPv6 => FlowProtocol::Icmpv6,
                _ => FlowProtocol::Icmpv4,
            };
            (
                packet.features.ip_addresses.src_host,
                packet.features.ip_addresses.dst_host,
                0,
                0,
                protocol,
            )
        } else if packet.features.tcp_set {
            (
                packet.features.ip_addresses.src_host,
                packet.features.ip_addresses.dst_host,
                packet.features.tcp_features.tcp_srcport,
                packet.features.tcp_features.tcp_dstport,
                FlowProtocol::Tcp,
            )
        } else if packet.features.udp_set {
            (
                packet.features.ip_addresses.src_host,
                packet.features.ip_addresses.dst_host,
                packet.features.udp_features.udp_port_src,
                packet.features.udp_features.udp_port_dst,
                FlowProtocol::Udp,
            )
        } else {
            return None;
        };

        Some(Self::new(src_ip, dst_ip, src_port, dst_port, protocol))
    }

    /// Single source of truth for flow direction.
    /// Returns true when (src_ip, src_port) sorts before (dst_ip, dst_port).
    /// The equal-on-both-sides case returns true — it's a self-loop on the same
    /// port, which is degenerate but deterministic.
    #[inline]
    fn is_forward_tuple(src_ip: IpAddr, dst_ip: IpAddr, src_port: u16, dst_port: u16) -> bool {
        (src_ip, src_port) <= (dst_ip, dst_port)
    }

    /// True if this key is already in its canonical (forward) orientation.
    #[inline]
    pub fn is_forward(&self) -> bool {
        Self::is_forward_tuple(self.src_ip, self.dst_ip, self.src_port, self.dst_port)
    }

    /// Return a canonical copy of this key. Cheap no-op clone when already forward.
    pub fn normalize(&self) -> FlowKey {
        if self.is_forward() {
            self.clone()
        } else {
            FlowKey {
                src_ip: self.dst_ip,
                dst_ip: self.src_ip,
                src_port: self.dst_port,
                dst_port: self.src_port,
                protocol: self.protocol,
            }
        }
    }

    /// Heap bytes owned by this key, for memory accounting.
    #[inline]
    fn heap_size(&self) -> usize {
        0
    }
}

impl std::fmt::Display for FlowKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{} -> {}:{} ({})",
            self.src_ip, self.src_port, self.dst_ip, self.dst_port, self.protocol
        )
    }
}

pub struct FlowStats {
    pub fwd_packet_count: u64,
    pub bwd_packet_count: u64,
    pub fwd_byte_count: u64,
    pub bwd_byte_count: u64,

    pub first_seen: TimeStamp,
    pub last_seen: TimeStamp,

    pub min_packet_size: u32,
    pub max_packet_size: u32,

    /// Richest protocol hierarchy observed on this flow, e.g.
    /// `Ethernet->IPv4->TCP->MQTT`. Descriptive only — it is not part of the
    /// flow's identity, because it varies packet to packet within one connection.
    pub proto_hierarchy: String,

    /// Entropy stats
    pub fwd_packet_entropy_agg: f32,
    pub fwd_min_packet_entropy: f32,
    pub fwd_max_packet_entropy: f32,

    pub bwd_packet_entropy_agg: f32,
    pub bwd_min_packet_entropy: f32,
    pub bwd_max_packet_entropy: f32,
}

impl FlowStats {
    /// Create new flow statistics from the first packet.
    /// `is_forward` indicates whether the packet travels in the canonical
    /// direction of the (normalized) flow key.
    pub fn new(
        is_forward: bool,
        packet_size: u32,
        first_seen: TimeStamp,
        h_packet: f32,
        proto_hierarchy: String,
    ) -> Self {
        let (fwd_packet_count, bwd_packet_count, fwd_byte_count, bwd_byte_count) = if is_forward {
            (1, 0, packet_size as u64, 0)
        } else {
            (0, 1, 0, packet_size as u64)
        };

        // Seed the entropy extrema only for the direction this packet travelled.
        // The other direction stays at 0.0 with a zero packet count, and
        // `update` seeds it on its first packet. Never treat 0.0 itself as
        // "unset" — an all-identical payload has a genuine entropy of 0.0.
        let (fwd_h, bwd_h) = if is_forward { (h_packet, 0.0) } else { (0.0, h_packet) };

        Self {
            fwd_packet_count,
            bwd_packet_count,
            fwd_byte_count,
            bwd_byte_count,
            first_seen,
            last_seen: first_seen,
            min_packet_size: packet_size,
            max_packet_size: packet_size,
            proto_hierarchy,
            fwd_packet_entropy_agg: fwd_h,
            fwd_min_packet_entropy: fwd_h,
            fwd_max_packet_entropy: fwd_h,
            bwd_packet_entropy_agg: bwd_h,
            bwd_min_packet_entropy: bwd_h,
            bwd_max_packet_entropy: bwd_h,
        }
    }

    /// Update statistics with a new packet.
    /// `is_forward` indicates the direction of *this* packet relative to the
    /// flow's canonical orientation.
    pub fn update(
        &mut self,
        is_forward: bool,
        packet_size: u32,
        packet_timestamp: TimeStamp,
        h_packet: f32,
        proto_hierarchy: &str,
    ) {
        if is_forward {
            if self.fwd_packet_count == 0 {
                self.fwd_min_packet_entropy = h_packet;
                self.fwd_max_packet_entropy = h_packet;
            } else {
                self.fwd_min_packet_entropy = self.fwd_min_packet_entropy.min(h_packet);
                self.fwd_max_packet_entropy = self.fwd_max_packet_entropy.max(h_packet);
            }
            self.fwd_packet_count += 1;
            self.fwd_byte_count += packet_size as u64;
            self.fwd_packet_entropy_agg += h_packet;
        } else {
            if self.bwd_packet_count == 0 {
                self.bwd_min_packet_entropy = h_packet;
                self.bwd_max_packet_entropy = h_packet;
            } else {
                self.bwd_min_packet_entropy = self.bwd_min_packet_entropy.min(h_packet);
                self.bwd_max_packet_entropy = self.bwd_max_packet_entropy.max(h_packet);
            }
            self.bwd_packet_count += 1;
            self.bwd_byte_count += packet_size as u64;
            self.bwd_packet_entropy_agg += h_packet;
        }

        // Keep the most specific hierarchy seen. Layers are only appended when
        // that layer actually parsed, so a longer string is a richer observation.
        if proto_hierarchy.len() > self.proto_hierarchy.len() {
            self.proto_hierarchy.clear();
            self.proto_hierarchy.push_str(proto_hierarchy);
        }

        self.last_seen = packet_timestamp;
        self.min_packet_size = self.min_packet_size.min(packet_size);
        self.max_packet_size = self.max_packet_size.max(packet_size);
    }

    #[inline]
    pub fn total_packet_count(&self) -> u64 {
        self.fwd_packet_count + self.bwd_packet_count
    }

    #[inline]
    pub fn total_byte_count(&self) -> u64 {
        self.fwd_byte_count + self.bwd_byte_count
    }

    pub fn duration(&self) -> Duration {
        self.last_seen.duration_since(&self.first_seen)
    }

    pub fn avg_packet_size(&self) -> f32 {
        let packets = self.total_packet_count();
        if packets > 0 {
            self.total_byte_count() as f32 / packets as f32
        } else {
            0.0
        }
    }

    /// Average packets per second over the flow's lifetime.
    /// Returns 0.0 for instantaneous flows (single packet or identical timestamps).
    pub fn avg_pps(&self) -> f32 {
        let secs = self.duration().as_secs_f32();
        if secs > 0.0 {
            self.total_packet_count() as f32 / secs
        } else {
            0.0
        }
    }

    /// Average bytes per second over the flow's lifetime.
    /// Returns 0.0 for instantaneous flows (single packet or identical timestamps).
    pub fn avg_bps(&self) -> f32 {
        let secs = self.duration().as_secs_f32();
        if secs > 0.0 {
            self.total_byte_count() as f32 / secs
        } else {
            0.0
        }
    }

    /// Heap bytes owned by these stats, for memory accounting.
    #[inline]
    fn heap_size(&self) -> usize {
        self.proto_hierarchy.capacity()
    }
}

pub struct FlowTracker {
    flows: FxHashMap<FlowKey, FlowStats>,
    seen_ips: FxHashSet<IpAddr>,
}

#[allow(dead_code)] // inspection helpers, used by tests and available to callers
impl FlowTracker {
    /// Create a tracker with room for `initial_capacity` flows.
    ///
    /// Note this performs no I/O — the tracker holds no database handle. The
    /// Falkor client is passed to `push_flows_to_falkor` at the end of the run,
    /// so parsing and flow accounting work with no database reachable at all.
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            flows: FxHashMap::with_capacity_and_hasher(initial_capacity, Default::default()),
            seen_ips: FxHashSet::with_capacity_and_hasher(10_000, Default::default()),
        }
    }

    pub fn default() -> Self {
        Self::new(1_000_000)
    }

    /// Process a packet and update flow statistics.
    /// Returns true if a new flow was created, false if an existing flow was updated.
    pub fn process_packet(&mut self, flow_key: &FlowKey, parsed_packet: &ParsedPacketSet) -> bool {
        let packet_size = parsed_packet.payload_set.data.len() as u32;
        // Decide direction once, from the original key.
        let is_forward = flow_key.is_forward();

        // Normalize the key we'll use for storage.
        let canonical_key = if is_forward { flow_key.clone() } else { flow_key.normalize() };

        // Shannon entropy over the captured frame.
        let h_packet = shannon_entropy(&parsed_packet.payload_set.data);
        let hierarchy = &parsed_packet.payload_set.proto_hierarchy;

        match self.flows.get_mut(&canonical_key) {
            Some(flow_stats) => {
                flow_stats.update(
                    is_forward,
                    packet_size,
                    parsed_packet.timevalue,
                    h_packet,
                    hierarchy,
                );
                false
            }
            None => {
                self.seen_ips.insert(flow_key.src_ip);
                self.seen_ips.insert(flow_key.dst_ip);
                let new_stats = FlowStats::new(
                    is_forward,
                    packet_size,
                    parsed_packet.timevalue,
                    h_packet,
                    hierarchy.clone(),
                );
                self.flows.insert(canonical_key, new_stats);
                true
            }
        }
    }

    pub fn get_flow_stats(&self, flow_key: &FlowKey) -> Option<&FlowStats> {
        self.flows.get(flow_key)
    }

    pub fn contains_flow(&self, flow_key: &FlowKey) -> bool {
        self.flows.contains_key(flow_key)
    }

    pub fn flow_count(&self) -> usize {
        self.flows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.flows.is_empty()
    }

    /// Approximate resident size of the flow table in bytes.
    ///
    /// Counts the actual key and value types plus the heap each entry owns, and
    /// the hash table's own allocation. Still an estimate — it does not account
    /// for allocator overhead — but it tracks reality rather than assuming a
    /// fixed 128 bytes per flow.
    pub fn estimated_memory_usage(&self) -> usize {
        let entry_size = size_of::<FlowKey>() + size_of::<FlowStats>();
        let heap: usize = self
            .flows
            .iter()
            .map(|(k, v)| k.heap_size() + v.heap_size())
            .sum();
        let table = self.flows.capacity() * (entry_size + 1);
        let ips = self.seen_ips.capacity() * size_of::<IpAddr>();
        table + heap + ips
    }

    pub fn print_flows(&self) {
        for (key, stats) in &self.flows {
            println!(
                "{} => fwd_packets: {}, bwd_packets: {}, fwd_bytes: {}, bwd_bytes: {}, duration: {:?}, avg_pkt_size: {:.2}",
                key,
                stats.fwd_packet_count,
                stats.bwd_packet_count,
                stats.fwd_byte_count,
                stats.bwd_byte_count,
                stats.duration(),
                stats.avg_packet_size()
            );
        }
    }

    pub fn has_seen_ip(&self, ip: &IpAddr) -> bool {
        self.seen_ips.contains(ip)
    }

    pub fn unique_ip_count(&self) -> usize {
        self.seen_ips.len()
    }

    /// Write every tracked flow to the graph. Returns (written, failed).
    pub fn push_flows_to_falkor(
        &self,
        graph_name: &str,
        falkor_client: &FalkorSyncClient,
    ) -> (usize, usize) {
        let mut graph = falkor_client.select_graph(graph_name);
        for ip in self.seen_ips.iter() {
            if let Err(e) = falkor_integration::merge_ip_address(&mut graph, ip) {
                eprintln!("Error merging IP {} into FalkorDB: {}", ip, e);
            }
        }

        let mut written = 0usize;
        let mut failed = 0usize;
        for (key, stats) in &self.flows {
            match falkor_integration::insert_flow_edge(&mut graph, key, stats) {
                Ok(_) => written += 1,
                Err(e) => {
                    failed += 1;
                    eprintln!("Error inserting flow into FalkorDB: {}", e);
                    eprintln!("  flow: {}", key);
                }
            }
        }
        (written, failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(len: usize, tv_sec: i64, tv_usec: i64, hierarchy: &str) -> ParsedPacketSet {
        let ts = libc::timeval { tv_sec, tv_usec };
        let mut p = ParsedPacketSet::new(vec![0u8; len], ts);
        p.payload_set.proto_hierarchy = hierarchy.to_string();
        p
    }

    fn packet_with_data(data: Vec<u8>, tv_sec: i64, tv_usec: i64) -> ParsedPacketSet {
        let ts = libc::timeval { tv_sec, tv_usec };
        ParsedPacketSet::new(data, ts)
    }

    #[test]
    fn tracks_bidirectional_counters_and_duration() {
        let mut tracker = FlowTracker::new(16);

        let src = IpAddr::from([192, 168, 1, 1]);
        let dst = IpAddr::from([192, 168, 1, 2]);
        let fwd = FlowKey::new(src, dst, 12345, 80, FlowProtocol::Tcp);
        let bwd = FlowKey::new(dst, src, 80, 12345, FlowProtocol::Tcp);

        tracker.process_packet(&fwd, &packet(125, 10, 500_000, "Ethernet->IPv4->TCP"));

        let stats = tracker.get_flow_stats(&fwd).unwrap();
        assert_eq!(tracker.flow_count(), 1);
        assert_eq!(stats.duration().as_secs(), 0);
        assert_eq!(stats.fwd_byte_count, 125);
        assert_eq!(stats.bwd_byte_count, 0);
        assert_eq!(stats.fwd_packet_count, 1);
        assert_eq!(stats.bwd_packet_count, 0);
        assert_eq!(stats.min_packet_size, 125);
        assert_eq!(stats.max_packet_size, 125);

        // Reverse direction, two seconds later, folds into the same flow.
        tracker.process_packet(&bwd, &packet(150, 12, 500_000, "Ethernet->IPv4->TCP"));

        let stats = tracker.get_flow_stats(&fwd).unwrap();
        assert_eq!(tracker.flow_count(), 1);
        assert_eq!(stats.duration().as_secs(), 2);
        assert_eq!(stats.fwd_byte_count, 125);
        assert_eq!(stats.bwd_byte_count, 150);
        assert_eq!(stats.fwd_packet_count, 1);
        assert_eq!(stats.bwd_packet_count, 1);
        assert_eq!(stats.min_packet_size, 125);
        assert_eq!(stats.max_packet_size, 150);

        // Forward again, four seconds after the first packet.
        tracker.process_packet(&fwd, &packet(80, 14, 500_000, "Ethernet->IPv4->TCP"));

        let stats = tracker.get_flow_stats(&fwd).unwrap();
        assert_eq!(tracker.flow_count(), 1);
        assert_eq!(stats.fwd_byte_count, 205);
        assert_eq!(stats.bwd_byte_count, 150);
        assert_eq!(stats.fwd_packet_count, 2);
        assert_eq!(stats.bwd_packet_count, 1);
        assert_eq!(stats.min_packet_size, 80);
        assert_eq!(stats.max_packet_size, 150);

        let duration = stats.duration();
        assert_eq!(duration.as_secs(), 4);
        assert_eq!(duration.as_millis(), 4_000);

        // Derived rates are computed on demand, so a sub-second flow is not
        // stuck reporting the single-packet seed values.
        assert_eq!(stats.total_packet_count(), 3);
        assert!((stats.avg_pps() - 0.75).abs() < 1e-6);
        assert!((stats.avg_bps() - 88.75).abs() < 1e-4);
        assert!((stats.avg_packet_size() - (355.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn sub_second_flows_report_real_rates() {
        let mut tracker = FlowTracker::new(4);
        let a = IpAddr::from([10, 0, 0, 1]);
        let b = IpAddr::from([10, 0, 0, 2]);
        let key = FlowKey::new(a, b, 1000, 2000, FlowProtocol::Udp);

        // Two packets 100ms apart — well under one whole second.
        tracker.process_packet(&key, &packet(100, 5, 0, "Ethernet->IPv4->UDP"));
        tracker.process_packet(&key, &packet(100, 5, 100_000, "Ethernet->IPv4->UDP"));

        let stats = tracker.get_flow_stats(&key).unwrap();
        assert_eq!(stats.duration().as_secs(), 0);
        assert_eq!(stats.duration().as_millis(), 100);
        // 2 packets / 0.1s = 20 pps, 200 bytes / 0.1s = 2000 bps.
        assert!((stats.avg_pps() - 20.0).abs() < 1e-3, "pps was {}", stats.avg_pps());
        assert!((stats.avg_bps() - 2000.0).abs() < 1e-1, "bps was {}", stats.avg_bps());
    }

    #[test]
    fn protocol_hierarchy_does_not_split_a_connection() {
        let mut tracker = FlowTracker::new(4);
        let a = IpAddr::from([10, 0, 0, 1]);
        let b = IpAddr::from([10, 0, 0, 2]);
        let key = FlowKey::new(a, b, 40000, 1883, FlowProtocol::Tcp);

        // A bare ACK parses only as far as TCP; the next packet on the same
        // connection carries an MQTT frame and parses one layer deeper.
        tracker.process_packet(&key, &packet(60, 1, 0, "Ethernet->IPv4->TCP"));
        tracker.process_packet(&key, &packet(200, 1, 5_000, "Ethernet->IPv4->TCP->MQTT"));

        assert_eq!(tracker.flow_count(), 1, "one connection must be one flow");

        let stats = tracker.get_flow_stats(&key).unwrap();
        assert_eq!(stats.total_packet_count(), 2);
        // The richest observation is retained as a descriptive property.
        assert_eq!(stats.proto_hierarchy, "Ethernet->IPv4->TCP->MQTT");
    }

    #[test]
    fn icmpv4_and_icmpv6_are_separate_flows() {
        let mut tracker = FlowTracker::new(4);
        let a = IpAddr::from([10, 0, 0, 1]);
        let b = IpAddr::from([10, 0, 0, 2]);

        let v4 = FlowKey::new(a, b, 0, 0, FlowProtocol::Icmpv4);
        let v6 = FlowKey::new(a, b, 0, 0, FlowProtocol::Icmpv6);

        tracker.process_packet(&v4, &packet(64, 1, 0, "Ethernet->IPv4->ICMPv4"));
        tracker.process_packet(&v6, &packet(64, 1, 0, "Ethernet->IPv6->ICMPv6"));

        assert_eq!(tracker.flow_count(), 2);
    }

    #[test]
    fn zero_entropy_packet_is_a_real_minimum() {
        let mut tracker = FlowTracker::new(4);
        let a = IpAddr::from([10, 0, 0, 1]);
        let b = IpAddr::from([10, 0, 0, 2]);
        let key = FlowKey::new(a, b, 1234, 5678, FlowProtocol::Tcp);

        // All-identical payload: genuine Shannon entropy of exactly 0.0.
        tracker.process_packet(&key, &packet_with_data(vec![0u8; 256], 1, 0));
        // High-entropy payload.
        let varied: Vec<u8> = (0..=255u8).collect();
        tracker.process_packet(&key, &packet_with_data(varied, 1, 1_000));

        let stats = tracker.get_flow_stats(&key).unwrap();
        assert_eq!(stats.fwd_packet_count, 2);
        // The old sentinel treated 0.0 as "unset" and let the second packet
        // overwrite the minimum. The minimum must stay at the true 0.0.
        assert_eq!(stats.fwd_min_packet_entropy, 0.0);
        assert!(
            stats.fwd_max_packet_entropy > 7.0,
            "max entropy was {}",
            stats.fwd_max_packet_entropy
        );
    }

    #[test]
    fn entropy_seeds_each_direction_independently() {
        let mut tracker = FlowTracker::new(4);
        let a = IpAddr::from([10, 0, 0, 1]);
        let b = IpAddr::from([10, 0, 0, 2]);
        let fwd = FlowKey::new(a, b, 1000, 2000, FlowProtocol::Tcp);
        let bwd = FlowKey::new(b, a, 2000, 1000, FlowProtocol::Tcp);

        // Forward packet is high entropy, backward packet is zero entropy.
        let varied: Vec<u8> = (0..=255u8).collect();
        tracker.process_packet(&fwd, &packet_with_data(varied, 1, 0));
        tracker.process_packet(&bwd, &packet_with_data(vec![7u8; 128], 1, 1_000));

        let stats = tracker.get_flow_stats(&fwd).unwrap();
        assert!(stats.fwd_min_packet_entropy > 7.0);
        assert_eq!(stats.bwd_min_packet_entropy, 0.0);
        assert_eq!(stats.bwd_packet_count, 1);
    }

    #[test]
    fn normalization_is_symmetric() {
        let a = IpAddr::from([10, 0, 0, 9]);
        let b = IpAddr::from([10, 0, 0, 1]);
        let fwd = FlowKey::new(a, b, 100, 200, FlowProtocol::Tcp);
        let bwd = FlowKey::new(b, a, 200, 100, FlowProtocol::Tcp);

        assert_eq!(fwd.normalize(), bwd.normalize());
        assert!(bwd.is_forward());
        assert!(!fwd.is_forward());
        // Normalizing an already-canonical key is a no-op.
        assert_eq!(bwd.normalize(), bwd);
    }

    #[test]
    fn memory_estimate_grows_with_the_table() {
        let mut tracker = FlowTracker::new(4);
        let empty = tracker.estimated_memory_usage();

        let a = IpAddr::from([10, 0, 0, 1]);
        for port in 0..64u16 {
            let key = FlowKey::new(a, IpAddr::from([10, 0, 1, 1]), port, 80, FlowProtocol::Tcp);
            tracker.process_packet(&key, &packet(100, 1, 0, "Ethernet->IPv4->TCP"));
        }

        assert_eq!(tracker.flow_count(), 64);
        assert!(tracker.estimated_memory_usage() > empty);
    }
}

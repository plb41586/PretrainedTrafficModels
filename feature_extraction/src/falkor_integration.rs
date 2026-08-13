use falkordb::{FalkorClientBuilder, FalkorConnectionInfo, FalkorSyncClient, FalkorValue, SyncGraph};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv6Addr};
use std::time::Duration;

use crate::flow_tracker::{FlowKey, FlowStats};

// --- Connection ---

/// Connect to FalkorDB. `host` is a `host:port` pair, e.g. `falkordb:6379`.
pub fn connect_to_falkor(host: &str) -> FalkorSyncClient {
    let url = format!("falkor://{host}");
    let connection_info: FalkorConnectionInfo = url
        .as_str()
        .try_into()
        .unwrap_or_else(|e| panic!("Invalid FalkorDB connection info {url:?}: {e}"));

    FalkorClientBuilder::new()
        .with_connection_info(connection_info)
        // Keepalive for a long-lived flow-tracking client sitting behind NAT /
        // stateful firewalls. TCP-only; no effect on Unix-socket/embedded paths.
        .with_tcp_keepalive(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|e| panic!("Failed to connect to FalkorDB at {host}: {e}"))
}

// --- Shared helpers ---

fn ip_label(ip: &IpAddr) -> &'static str {
    match ip {
        IpAddr::V4(_) => "IPv4",
        IpAddr::V6(_) => "IPv6",
    }
}

fn ip_scope(ip: &IpAddr) -> &'static str {
    let is_private = match ip {
        IpAddr::V4(v4) => v4.is_private() || v4.is_loopback() || v4.is_link_local(),
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unicast_link_local()
                || is_ipv6_unique_local(v6)
                || v6.is_multicast()
        }
    };
    if is_private { "internal" } else { "external" }
}

/// True for the unique-local range fc00::/7 — the top 7 bits masked by 0xfe00.
fn is_ipv6_unique_local(ipv6: &Ipv6Addr) -> bool {
    (ipv6.segments()[0] & 0xfe00) == 0xfc00
}

/// Single source of truth for flow-edge properties. Emits native typed
/// `FalkorValue`s — ints stay ints, floats stay floats. No stringification.
fn flow_edge_params(key: &FlowKey, stats: &FlowStats) -> HashMap<String, FalkorValue> {
    let mut params: HashMap<String, FalkorValue> = HashMap::with_capacity(32);

    // Endpoint identity — plain strings; client escapes them.
    params.insert("src_ip".into(), key.src_ip.to_string().into());
    params.insert("dst_ip".into(), key.dst_ip.to_string().into());

    // Edge identity.
    params.insert("src_port".into(), (key.src_port as i64).into());
    params.insert("dst_port".into(), (key.dst_port as i64).into());

    // The full protocol stack, e.g. "Ethernet->VLAN (802.1Q)->IPv4->TCP->MQTT".
    // This travels as a parameter precisely because it can contain spaces,
    // parentheses and other characters that are unsafe to interpolate.
    params.insert("proto_hierarchy".into(), stats.proto_hierarchy.clone().into());

    // Counters.
    params.insert("fwd_packet_count".into(), (stats.fwd_packet_count as i64).into());
    params.insert("bwd_packet_count".into(), (stats.bwd_packet_count as i64).into());
    params.insert("fwd_byte_count".into(), (stats.fwd_byte_count as i64).into());
    params.insert("bwd_byte_count".into(), (stats.bwd_byte_count as i64).into());

    // Timing.
    params.insert("first_seen".into(), (stats.first_seen.as_secs() as i64).into());
    params.insert("last_seen".into(), (stats.last_seen.as_secs() as i64).into());
    params.insert("duration_ms".into(), (stats.duration().as_millis() as i64).into());

    // Size extrema.
    params.insert("min_packet_size".into(), (stats.min_packet_size as i64).into());
    params.insert("max_packet_size".into(), (stats.max_packet_size as i64).into());

    // Derived values consumed by downstream ML — real floats, computed on
    // demand from the counters rather than read from a cached field.
    params.insert("avg_packet_size".into(), (stats.avg_packet_size() as f64).into());
    params.insert("avg_pps".into(), (stats.avg_pps() as f64).into());
    params.insert("avg_bps".into(), (stats.avg_bps() as f64).into());

    // Entropy stats — f32 widened to f64 so they land as native float
    // FalkorValues (FalkorValue has no f32 path).
    params.insert("fwd_packet_entropy_agg".into(), (stats.fwd_packet_entropy_agg as f64).into());
    params.insert("fwd_min_packet_entropy".into(), (stats.fwd_min_packet_entropy as f64).into());
    params.insert("fwd_max_packet_entropy".into(), (stats.fwd_max_packet_entropy as f64).into());
    params.insert("bwd_packet_entropy_agg".into(), (stats.bwd_packet_entropy_agg as f64).into());
    params.insert("bwd_min_packet_entropy".into(), (stats.bwd_min_packet_entropy as f64).into());
    params.insert("bwd_max_packet_entropy".into(), (stats.bwd_max_packet_entropy as f64).into());

    params
}

/// Write a flow edge between two existing IP nodes, creating it if absent and
/// refreshing its properties if present. Assumes the nodes were already created
/// via `merge_ip_address`.
///
/// `MERGE` rather than `CREATE`: re-running the extractor over the same capture
/// and graph name updates the existing edges instead of duplicating every one
/// of them. The edge is identified by its endpoints, relationship type and port
/// pair; everything else is set on each write.
pub fn insert_flow_edge(
    graph: &mut SyncGraph,
    key: &FlowKey,
    stats: &FlowStats,
) -> Result<(), String> {
    let protocol = validate_protocol(key.protocol.as_str())?;
    let src_label = ip_label(&key.src_ip);
    let dst_label = ip_label(&key.dst_ip);

    let query = format!(
        "MATCH (s:{src_label} {{ address: $src_ip }})
         MATCH (d:{dst_label} {{ address: $dst_ip }})
         MERGE (s)-[r:`{protocol}` {{ src_port: $src_port, dst_port: $dst_port }}]->(d)
         {set_block}",
        set_block = EDGE_SET_BLOCK,
    );

    let params = flow_edge_params(key, stats);

    graph
        .query(&query)
        .with_params(params)
        .with_timeout(5000)
        .execute()
        .map_err(|e| format!("Failed to write edge: {}", e))?;

    Ok(())
}

/// Ensure an IP node exists with the right label and `type` property.
pub fn merge_ip_address(graph: &mut SyncGraph, ip_addr: &IpAddr) -> Result<(), String> {
    let label = ip_label(ip_addr);
    let scope = ip_scope(ip_addr);

    let query = format!("MERGE (ip:{label} {{ address: $address, type: $type }})");

    let mut params: HashMap<String, FalkorValue> = HashMap::with_capacity(2);
    params.insert("address".into(), ip_addr.to_string().into());
    params.insert("type".into(), scope.into());

    graph
        .query(&query)
        .with_params(params)
        .with_timeout(5000)
        .execute()
        .map_err(|e| format!("Failed to merge IP address: {}", e))?;

    Ok(())
}

/// The `SET` block applied on every edge write. Kept as a `const` so the
/// property list cannot drift from `flow_edge_params`.
const EDGE_SET_BLOCK: &str = "SET r.proto_hierarchy = $proto_hierarchy,
        r.fwd_packet_count = $fwd_packet_count,
        r.bwd_packet_count = $bwd_packet_count,
        r.fwd_byte_count = $fwd_byte_count,
        r.bwd_byte_count = $bwd_byte_count,
        r.first_seen = $first_seen,
        r.last_seen = $last_seen,
        r.duration_ms = $duration_ms,
        r.min_packet_size = $min_packet_size,
        r.max_packet_size = $max_packet_size,
        r.avg_packet_size = $avg_packet_size,
        r.avg_pps = $avg_pps,
        r.avg_bps = $avg_bps,
        r.fwd_packet_entropy_agg = $fwd_packet_entropy_agg,
        r.fwd_min_packet_entropy = $fwd_min_packet_entropy,
        r.fwd_max_packet_entropy = $fwd_max_packet_entropy,
        r.bwd_packet_entropy_agg = $bwd_packet_entropy_agg,
        r.bwd_min_packet_entropy = $bwd_min_packet_entropy,
        r.bwd_max_packet_entropy = $bwd_max_packet_entropy";

/// Cypher cannot parameterise a relationship type, so it must be interpolated.
/// `FlowKey::protocol` is a closed enum whose `as_str` values are all plain
/// ASCII identifiers, which makes that safe — this check enforces the invariant
/// so a future variant cannot quietly reintroduce injectable text.
///
/// Note the *protocol hierarchy* — the free-form string that can contain spaces
/// and parentheses, e.g. "Ethernet->VLAN (802.1Q)->TCP" — never reaches this
/// function. It is written as a parameter, so packets carrying VLAN, MPLS,
/// PPPoE or Profinet tags are stored rather than rejected.
fn validate_protocol(protocol: &str) -> Result<&str, String> {
    if protocol.is_empty() || protocol.len() > 64 {
        return Err(format!("Invalid relationship type length: {:?}", protocol));
    }
    if !protocol.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(format!(
            "Relationship type contains disallowed characters: {:?}",
            protocol
        ));
    }
    Ok(protocol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_tracker::FlowProtocol;

    #[test]
    fn every_flow_protocol_is_a_valid_relationship_type() {
        for p in [
            FlowProtocol::Arp,
            FlowProtocol::Icmpv4,
            FlowProtocol::Icmpv6,
            FlowProtocol::Tcp,
            FlowProtocol::Udp,
        ] {
            assert!(
                validate_protocol(p.as_str()).is_ok(),
                "{} must be usable as a relationship type",
                p
            );
        }
    }

    #[test]
    fn protocol_validator_rejects_garbage() {
        assert!(validate_protocol("").is_err());
        assert!(validate_protocol("foo`bar").is_err());
        assert!(validate_protocol("foo}) CREATE (x)").is_err());
        assert!(validate_protocol("has spaces").is_err());
        assert!(validate_protocol(&"a".repeat(200)).is_err());
    }

    #[test]
    fn ip_label_is_per_address() {
        let v4: IpAddr = "1.2.3.4".parse().unwrap();
        let v6: IpAddr = "::1".parse().unwrap();
        assert_eq!(ip_label(&v4), "IPv4");
        assert_eq!(ip_label(&v6), "IPv6");
    }

    #[test]
    fn mixed_family_flows_label_each_endpoint_separately() {
        // The bug this guards: taking the label from the source address and
        // applying it to both nodes, so the destination MATCH never matches and
        // the edge is silently dropped.
        let v4: IpAddr = "192.168.1.1".parse().unwrap();
        let v6: IpAddr = "2001:db8::1".parse().unwrap();
        assert_ne!(ip_label(&v4), ip_label(&v6));
    }

    #[test]
    fn ip_scope_classification() {
        let private: IpAddr = "192.168.4.4".parse().unwrap();
        let public: IpAddr = "8.8.8.8".parse().unwrap();
        let ula: IpAddr = "fd00::1".parse().unwrap();
        let global6: IpAddr = "2001:db8::1".parse().unwrap();

        assert_eq!(ip_scope(&private), "internal");
        assert_eq!(ip_scope(&public), "external");
        assert_eq!(ip_scope(&ula), "internal");
        assert_eq!(ip_scope(&global6), "external");
    }

    #[test]
    fn edge_set_block_covers_every_parameter_it_names() {
        // Guards drift between EDGE_SET_BLOCK and flow_edge_params: every
        // $placeholder in the SET block must be produced by the params builder.
        use crate::feature_parser::TimeStamp;

        let key = FlowKey::new(
            "10.0.0.1".parse().unwrap(),
            "10.0.0.2".parse().unwrap(),
            1234,
            80,
            FlowProtocol::Tcp,
        );
        let stats = FlowStats::new(
            true,
            100,
            TimeStamp::new(1, 0),
            0.5,
            "Ethernet->IPv4->TCP".to_string(),
        );
        let params = flow_edge_params(&key, &stats);

        for token in EDGE_SET_BLOCK.split('$').skip(1) {
            let name: String = token
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            assert!(
                params.contains_key(&name),
                "SET block references ${} but flow_edge_params does not produce it",
                name
            );
        }
    }
}

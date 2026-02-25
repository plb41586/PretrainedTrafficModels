use etherparse::err::ip;
use falkordb::{FalkorClientBuilder, FalkorConnectionInfo, FalkorSyncClient, SyncGraph};
use crate::feature_parser::{ParsedPacketSet};
use crate::flow_tracker::{FlowKey, FlowStats};
use std::net::IpAddr;

pub fn connect_to_falkor() -> FalkorSyncClient {
    // Initialize the Falkor client
    let connection_info: FalkorConnectionInfo = "falkor://falkordb:6379"
        .try_into()
        .expect("Invalid connection info");

    let client = FalkorClientBuilder::new()
        .with_connection_info(connection_info)
        .build()
        .expect("Failed to build client");
    
    return client;
}

pub fn insert_from_flow_tracker(
    graph: &mut SyncGraph,
    key: &crate::flow_tracker::FlowKey,
    stats: &crate::flow_tracker::FlowStats
) -> Result<(), String> {
    let mut ip_version: &str;
    if key.src_ip.is_ipv4() {
        ip_version = "IPv4";
    } else if key.src_ip.is_ipv6() {
        ip_version = "IPv6";
    } else {
        ip_version = "Unknown";
        return Err("Source IP is neither IPv4 nor IPv6".to_string());
    }
    
    // Convert duration to milliseconds for storage
    let duration_ms = stats.duration.as_millis() as u64;
    
    // Merge Nodes and Relationship from FlowKey with FlowStats properties
    let merge_node_query = format!(
        r#"MERGE (s:{} {{ address: '{}'}})
        MERGE (d:{} {{ address: '{}' }})
        MERGE (s)-[r:`{}` {{ 
            src_port: {}, 
            dst_port: {},
            fwd_packet_count: {},
            bwd_packet_count: {},
            fwd_byte_count: {},
            bwd_byte_count: {},
            first_seen: {},
            last_seen: {},
            duration_ms: {},
            min_packet_size: {},
            max_packet_size: {},
            avg_packet_size: {},
            avg_pps: {},
            avg_bps: {}
        }}]->(d)"#,
        ip_version,
        key.src_ip,
        ip_version,
        key.dst_ip,
        key.protocol,
        key.src_port,
        key.dst_port,
        stats.fwd_packet_count,
        stats.bwd_packet_count,
        stats.fwd_byte_count,
        stats.bwd_byte_count,
        stats.first_seen.as_secs(),
        stats.last_seen.as_secs(),
        duration_ms,
        stats.min_packet_size,
        stats.max_packet_size,
        stats.avg_packet_size,
        stats.avg_pps,
        stats.avg_bps
    );
    
    graph
        .query(merge_node_query)
        .with_timeout(5000)
        .execute()
        .expect("Failed to create nodes");
    
    Ok(())
}

pub fn insert_edge_only(
    graph: &mut SyncGraph,
    key: &crate::flow_tracker::FlowKey,
    stats: &crate::flow_tracker::FlowStats
) -> Result<(), String> {
    let ip_version: &str;
    if key.src_ip.is_ipv4() {
        ip_version = "IPv4";
    } else if key.src_ip.is_ipv6() {
        ip_version = "IPv6";
    } else {
        return Err("Source IP is neither IPv4 nor IPv6".to_string());
    }
    
    // Convert duration to milliseconds for storage
    let duration_ms = stats.duration.as_millis() as u64;
    
    // Create relationship between existing nodes
    let create_edge_query = format!(
        r#"MATCH (s:{} {{ address: '{}'}})
        MATCH (d:{} {{ address: '{}' }})
        CREATE (s)-[r:`{}` {{ 
            src_port: {}, 
            dst_port: {},
            fwd_packet_count: {},
            bwd_packet_count: {},
            fwd_byte_count: {},
            bwd_byte_count: {},
            first_seen: {},
            last_seen: {},
            duration_ms: {},
            min_packet_size: {},
            max_packet_size: {},
            avg_packet_size: {},
            avg_pps: {},
            avg_bps: {}
        }}]->(d)"#,
        ip_version,
        key.src_ip,
        ip_version,
        key.dst_ip,
        key.protocol,
        key.src_port,
        key.dst_port,
        stats.fwd_packet_count,
        stats.bwd_packet_count,
        stats.fwd_byte_count,
        stats.bwd_byte_count,
        stats.first_seen.as_secs(),
        stats.last_seen.as_secs(),
        duration_ms,
        stats.min_packet_size,
        stats.max_packet_size,
        stats.avg_packet_size,
        stats.avg_pps,
        stats.avg_bps
    );
    
    graph
        .query(create_edge_query)
        .with_timeout(5000)
        .execute()
        .map_err(|e| format!("Failed to create edge: {}", e))?;
    
    Ok(())
}

pub fn update_existing_edge(
    graph: &mut SyncGraph,
    key: &crate::flow_tracker::FlowKey,
    stats: &crate::flow_tracker::FlowStats
) -> Result<(), String> {
    let ip_version: &str;
    if key.src_ip.is_ipv4() {
        ip_version = "IPv4";
    } else if key.src_ip.is_ipv6() {
        ip_version = "IPv6";
    } else {
        return Err("Source IP is neither IPv4 nor IPv6".to_string());
    }
    
    // Convert duration to milliseconds for storage
    let duration_ms = stats.duration.as_millis() as u64;
    
    // Update existing relationship properties
    let update_edge_query = format!(
        r#"MATCH (s:{} {{ address: '{}'}})
        -[r:`{}` {{ src_port: {}, dst_port: {} }}]->
        (d:{} {{ address: '{}' }})
        SET r.fwd_packet_count = {},
            r.bwd_packet_count = {},
            r.fwd_byte_count = {},
            r.bwd_byte_count = {},
            r.first_seen = {},
            r.last_seen = {},
            r.duration_ms = {},
            r.min_packet_size = {},
            r.max_packet_size = {},
            r.avg_packet_size = {},
            r.avg_pps = {},
            r.avg_bps = {}"#,
        ip_version,
        key.src_ip,
        key.protocol,
        key.src_port,
        key.dst_port,
        ip_version,
        key.dst_ip,
        stats.fwd_packet_count,
        stats.bwd_packet_count,
        stats.fwd_byte_count,
        stats.bwd_byte_count,
        stats.first_seen.as_secs(),
        stats.last_seen.as_secs(),
        duration_ms,
        stats.min_packet_size,
        stats.max_packet_size,
        stats.avg_packet_size,
        stats.avg_pps,
        stats.avg_bps
    );
    
    graph
        .query(update_edge_query)
        .with_timeout(5000)
        .execute()
        .map_err(|e| format!("Failed to update edge: {}", e))?;
    
    Ok(())
}

pub fn merge_ip_address(
    graph: &mut SyncGraph,
    ip_addr: &IpAddr
) -> Result<(), String> {
    let (ip_version, ip_type) = match ip_addr {
        IpAddr::V4(ipv4) => {
            let is_private = ipv4.is_private() || ipv4.is_loopback() || ipv4.is_link_local();
            ("IPv4", if is_private { "internal" } else { "external" })
        },
        IpAddr::V6(ipv6) => {
            let is_private = ipv6.is_loopback() ||           
                           ipv6.is_unicast_link_local() ||   
                           is_ipv6_unique_local(ipv6) ||
                           ipv6.is_multicast();              // Multicast addresses
            ("IPv6", if is_private { "internal" } else { "external" })
        }
    };
    
    let merge_ip_query = format!(
        r#"MERGE (ip:{} {{ address: '{}', type: '{}' }})"#,
        ip_version,
        ip_addr,
        ip_type
    );
    
    graph
        .query(merge_ip_query)
        .with_timeout(5000)
        .execute()
        .map_err(|e| format!("Failed to merge IP address: {}", e))?;
    
    Ok(())
}

fn is_ipv6_unique_local(ipv6: &std::net::Ipv6Addr) -> bool {
    let segments = ipv6.segments();
    (segments[0] & 0xfe00) == 0xfc00
}

pub fn basic_falkor_test() {
    // Initialize the Falkor client
    let connection_info: FalkorConnectionInfo = "falkor://falkordb:6379"
        .try_into()
        .expect("Invalid connection info");

    let client = FalkorClientBuilder::new()
        .with_connection_info(connection_info)
        .build()
        .expect("Failed to build client");

    // // Select the social graph
    // let mut graph = client.select_graph("test1");

    // // Create 100 nodes and return a handful
    // let mut nodes = graph.query("UNWIND range(0, 100) AS i CREATE (n { v:1 }) RETURN n LIMIT 10")
    //             .with_timeout(5000)
    //             .execute()
    //             .expect("Failed executing query");

    // // Can also be collected, like any other iterator
    // while let Some(node) = nodes.data.next() {
    //    println ! ("{:?}", node);
    // }
    // Select the test graph
    let mut graph = client.select_graph("test1");

    // Create 10 nodes (fixed: no concat function)
    let create_nodes_query = r#"
       UNWIND range(1, 10) AS i
       CREATE (n:Person { 
           id: i, 
           name: 'Person_' + toString(i), 
           age: 20 + i, 
           active: i % 2 = 0 
       })
   "#;

    graph
        .query(create_nodes_query)
        .with_timeout(5000)
        .execute()
        .expect("Failed to create nodes");

    // Create edges connecting each node to the next
    let create_edges_query = r#"
       UNWIND range(1, 9) AS i
       MATCH (a:Person {id: i}), (b:Person {id: i + 1})
       CREATE (a)-[:KNOWS]->(b)
   "#;

    graph
        .query(create_edges_query)
        .with_timeout(5000)
        .execute()
        .expect("Failed to create edges");

    // Return all nodes and their outgoing relationships
    let return_query = r#"
       MATCH (n:Person)
       OPTIONAL MATCH (n)-[r:KNOWS]->(m)
       RETURN n, r, m
       LIMIT 20
   "#;

    let mut results = graph
        .query(return_query)
        .with_timeout(5000)
        .execute()
        .expect("Failed executing return query");

    while let Some(record) = results.data.next() {
        println!("{:?}", record);
    }
}

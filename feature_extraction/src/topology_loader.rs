use serde::Deserialize;
use std::fs;
use falkordb::{FalkorClientBuilder, FalkorConnectionInfo, FalkorSyncClient, SyncGraph};
use std::net::Ipv4Addr;
use std::str::FromStr;

#[derive(Debug, Deserialize)]
struct NetworkTopology {
    devices: Vec<Device>,
}

#[derive(Debug, Deserialize)]
struct Device {
    name: String,
    roles: Vec<String>,
    ip_addresses: Vec<String>,
}

/// Load network topology from TOML file and insert into Falkor graph
pub fn load_network_topology(
    graph: &mut SyncGraph,
    topology_file: &str,
) -> Result<(), String> {
    // Read and parse the TOML file
    let contents = fs::read_to_string(topology_file)
        .map_err(|e| format!("Failed to read topology file: {}", e))?;
    
    let topology: NetworkTopology = toml::from_str(&contents)
        .map_err(|e| format!("Failed to parse topology file: {}", e))?;

    // Process each device
    for device in topology.devices {
        // Skip devices without IP addresses
        if device.ip_addresses.is_empty() {
            println!("Skipping device '{}' - no IP addresses", device.name);
            continue;
        }

        // Insert device into graph
        insert_device(graph, &device)?;
    }

    Ok(())
}

/// Check if an IPv4 address is private (internal) or public (external)
fn is_private_ip(ip_str: &str) -> Result<bool, String> {
    let ip = Ipv4Addr::from_str(ip_str)
        .map_err(|e| format!("Invalid IPv4 address '{}': {}", ip_str, e))?;
    
    // Check RFC 1918 private address ranges:
    // 10.0.0.0/8
    // 172.16.0.0/12
    // 192.168.0.0/16
    // Also check localhost (127.0.0.0/8) and link-local (169.254.0.0/16)
    Ok(ip.is_private() || ip.is_loopback() || ip.is_link_local())
}

// /// Insert a single device and its IP relationships into the graph
// fn insert_device(graph: &mut SyncGraph, device: &Device) -> Result<(), String> {
//     // Escape single quotes in device name and roles for Cypher query
//     let device_name = escape_cypher_string(&device.name);
//     let roles_str = device.roles
//         .iter()
//         .map(|r| escape_cypher_string(r))
//         .collect::<Vec<_>>()
//         .join("', '");

//     // Create the device node
//     let create_device_query = format!(
//         "MERGE (d:Device {{ name: '{}', roles: ['{}'] }})",
//         device_name, roles_str
//     );

//     graph
//         .query(create_device_query)
//         .with_timeout(5000)
//         .execute()
//         .map_err(|e| format!("Failed to create device node: {}", e))?;

//     // Create IP nodes and relationships
//     for ip_address in &device.ip_addresses {
//         let ip_escaped = escape_cypher_string(ip_address);
        
//         let create_relationship_query = format!(
//             "MERGE (d:Device {{ name: '{}' }})
//             MERGE (ip:IPv4 {{ address: '{}' }})
//             MERGE (d)-[:CONTROLS]->(ip)",
//             device_name, ip_escaped
//         );

//         graph
//             .query(create_relationship_query)
//             .with_timeout(5000)
//             .execute()
//             .map_err(|e| format!("Failed to create IP relationship for {}: {}", ip_address, e))?;
//     }

//     println!(
//         "Successfully inserted device '{}' with {} IP address(es)",
//         device.name,
//         device.ip_addresses.len()
//     );

//     Ok(())
// }

/// Insert a single device and its IP relationships into the graph
/// with IP classification (internal/external)
fn insert_device(graph: &mut SyncGraph, device: &Device) -> Result<(), String> {
    // Escape single quotes in device name and roles for Cypher query
    let device_name = escape_cypher_string(&device.name);
    let roles_str = device.roles
        .iter()
        .map(|r| escape_cypher_string(r))
        .collect::<Vec<_>>()
        .join("', '");
    
    // Create the device node
    let create_device_query = format!(
        "MERGE (d:Device {{ name: '{}', roles: ['{}'] }})",
        device_name, roles_str
    );
    graph
        .query(create_device_query)
        .with_timeout(5000)
        .execute()
        .map_err(|e| format!("Failed to create device node: {}", e))?;
    
    // Create IP nodes and relationships with classification
    for ip_address in &device.ip_addresses {
        let ip_escaped = escape_cypher_string(ip_address);
        
        // Determine if IP is private or public
        let is_private = is_private_ip(ip_address)?;
        let ip_type = if is_private { "internal" } else { "external" };
        
        let create_relationship_query = format!(
            "MERGE (d:Device {{ name: '{}' }})
            MERGE (ip:IPv4 {{ address: '{}', type: '{}' }})
            MERGE (d)-[:CONTROLS]->(ip)",
            device_name, ip_escaped, ip_type
        );
        
        graph
            .query(create_relationship_query)
            .with_timeout(5000)
            .execute()
            .map_err(|e| format!("Failed to create IP relationship for {}: {}", ip_address, e))?;
    }
    
    println!(
        "Successfully inserted device '{}' with {} IP address(es)",
        device.name,
        device.ip_addresses.len()
    );
    
    Ok(())
}

/// Escape single quotes in strings for Cypher queries
fn escape_cypher_string(s: &str) -> String {
    s.replace("'", "\\'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_cypher_string() {
        assert_eq!(escape_cypher_string("test"), "test");
        assert_eq!(escape_cypher_string("test's"), "test\\'s");
    }
}
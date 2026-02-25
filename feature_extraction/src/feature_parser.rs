use byteorder::{BigEndian, ReadBytesExt};
use simple_dns::{Packet as DnsPacket, Question as DNSQuestion};
use etherparse::{
    err::{self, ip},
    Icmpv4Slice, Icmpv6Header, InternetSlice, PacketHeaders, SlicedPacket, TcpHeader,
    TransportHeader, UdpHeader, TransportSlice, TcpSlice, UdpSlice,
};
use pcap::Packet;
use redis::{FromRedisValue, RedisError, RedisResult, RedisWrite, ToRedisArgs, Value};
use rmp_serde::{from_slice, to_vec};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::str;
use std::{
    io::Cursor,
    net::{IpAddr, Ipv4Addr},
};
use thiserror::Error;
use libc::timeval;
use std::collections::HashMap;
use std::time::Duration;

use crate::feature_parser;

#[derive(Error, Debug)]
pub enum ParserError {
    #[error("Failed to find EtherType")]
    EtherTypeError(#[from] EtherTypeError),
    #[error("Failed to slice packet")]
    SliceError(#[from] err::packet::SliceError),
    #[error("Ethertype not implemented")]
    EthertypeNotImplemented,
    #[error("Failed to find TransportHeader")]
    TransportHeaderError,
    #[error("Failed to find TCP features")]
    TcpFeatureError(#[from] TcpFeaturesError),
    #[error("Failed to find UDP features")]
    UdpFeatureError(#[from] UdpFeaturesError),
}

// Parse Ethernet package
pub fn parse_packet(packet: &Packet) -> Result<ParsedPacketSet, ParserError> {
    let timevalue = packet.header.ts;
    //Construct the ParsedPacketSet struct
    let mut parsed_packet = ParsedPacketSet::new(packet.data.to_vec(), timevalue);

    // Begin parsing the packet
    let ether_type = parse_ether_type(packet)?;
    let eth_slice = SlicedPacket::from_ethernet(packet.data)?;
    let eth_header = PacketHeaders::from_ethernet_slice(packet.data)?;

    // Set payload set information for Ethernet header
    parsed_packet.payload_set.header_len = 14; // Ethernet header length is 14 bytes
    for i in 0..12 {
        parsed_packet.payload_set.mask[i] = true;
    } // Set first 12 bits of mask to 1 (Ethernet addresses)
    parsed_packet
        .payload_set
        .proto_hierarchy
        .push_str("Ethernet");

    // Add EtherType to protocol hierarchy
    parsed_packet
        .payload_set
        .proto_hierarchy
        .push_str(&format!("->{}", ether_type));

    // Continue parsing
    if ether_type == "ARP".to_string() {
        if let Ok(arp_features) = get_arp_features(eth_slice) {
            // Set ARP features
            parsed_packet.features.arp_features = arp_features;
            parsed_packet.features.arp_set = true;
            // adapt payload set
            parsed_packet.payload_set.header_len = parsed_packet.payload_set.header_len + 28;
            for i in 22..42 {
                parsed_packet.payload_set.mask[i] = true;
            }
        }
        return Ok(parsed_packet);
    } else if ether_type == "IPv4".to_string() || ether_type == "IPv6".to_string() {
        if let Ok(()) = parse_ip_layer(&eth_slice, &mut parsed_packet) {
            parsed_packet.features.ip_set = true;
        }
    }
    else {
        return Err(ParserError::EthertypeNotImplemented);
    }
    let transport = eth_slice
        .transport
        .as_ref()
        .ok_or(ParserError::TransportHeaderError)?;

    //initiate dest_port
    let mut dest_port: u16 = 0;
    let mut src_port: u16 = 0;
    //initiate transport payload
    let mut transport_payload: Vec<u8> = Vec::new();
    let transport_protocol = match transport {
        TransportSlice::Icmpv4(_) => {
            if let Ok(transport_features) = get_icmp_v4_features(packet) {
                parsed_packet.features.icmp_features = transport_features;
                parsed_packet.features.icmp_set = true;
                // adapt payload set
                parsed_packet.payload_set.header_len = parsed_packet.payload_set.header_len + 8;
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->ICMPv4");
            }
            TransportProtocol::ICMPv4
        }
        TransportSlice::Icmpv6(_) => {
            if let Ok(transport_features) = get_icmp_v6_features(packet) {
                parsed_packet.features.icmp_features = transport_features;
                parsed_packet.features.icmp_set = true;
                // adapt payload set
                parsed_packet.payload_set.header_len = parsed_packet.payload_set.header_len + 8;
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->ICMPv6");
            }
            TransportProtocol::ICMPv6
        }
        TransportSlice::Tcp(tcp_slice) => {
            let tcp_header = tcp_slice.to_header();
            if let Ok(()) = parse_tcp_layer(&tcp_header, &eth_slice, &mut parsed_packet) {
                parsed_packet.features.tcp_set = true;
                // adapt protocol hierarchy
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->TCP");
                dest_port = parsed_packet.features.tcp_features.tcp_dstport;
                src_port = parsed_packet.features.tcp_features.tcp_srcport;
                transport_payload = tcp_slice.payload().to_vec();
            }
            TransportProtocol::TCP
        }
        TransportSlice::Udp(udp_slice) => {
            let udp_header = udp_slice.to_header();
            if let Ok(()) = get_udp_features(&udp_header, &mut parsed_packet) {
                parsed_packet.features.udp_set = true;
                dest_port = parsed_packet.features.udp_features.udp_port_dst;
                src_port = parsed_packet.features.udp_features.udp_port_src;
                // adapt protocol hierarchy
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->UDP");
                transport_payload = udp_slice
                    .payload()
                    .to_vec();
            }
            TransportProtocol::UDP
        }
    };

    let application_protocol_candidate = detect_protocol_candidate(&transport_protocol, src_port, dest_port);

    match application_protocol_candidate {
        ApplicationProtocol::DNS => {
            // Dummy Timestamp for DNS context
            let mut dns_context = DnsContext::new();
            if let Ok(DNS_features) = parse_dns(&transport_payload, 1000, &mut dns_context) {
                parsed_packet.features.DNS_features = DNS_features;
                // adapt payload set
                parsed_packet.payload_set.header_len = parsed_packet.payload_set.header_len + 12; // DNS header length is 12 bytes
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->DNS");
            }
        }
        ApplicationProtocol::MDNS => {
            // Dummy Timestamp for DNS context
            let mut dns_context = DnsContext::new();
            if let Ok(DNS_features) = parse_dns(&transport_payload, 1000, &mut dns_context) {
                parsed_packet.features.DNS_features = DNS_features;
                // adapt payload set
                parsed_packet.payload_set.header_len = parsed_packet.payload_set.header_len + 12; // DNS header length is 12 bytes
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->MDNS");
                parsed_packet.features.DNS_features.mdns = true;
            }
        }
        ApplicationProtocol::MQTT => {
            if let Ok(()) = parse_mqtt(transport_payload, &mut parsed_packet) {
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->MQTT");
            }
        }
        ApplicationProtocol::MBTCP => {
            if let Ok(mbtcp_features) = parse_modbus_tcp(transport_payload) {
                parsed_packet.features.modbus_tcp_features = mbtcp_features;
                if parsed_packet.payload_set.data.len()
                    < (7 + parsed_packet.payload_set.header_len as usize)
                {
                    return Ok(parsed_packet);
                }
                parsed_packet.payload_set.header_len += 7; // Modbus TCP header length is 7 bytes
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->ModbusTCP");
            }
        }
        ApplicationProtocol::HTTP => {
            if let Ok(()) = parse_http(transport_payload, &mut parsed_packet) {
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->HTTP");
            }
        }
        ApplicationProtocol::HTTPS => {
            if let Ok(()) = parse_http(transport_payload, &mut parsed_packet) {
                parsed_packet
                    .payload_set
                    .proto_hierarchy
                    .push_str("->HTTPS");
            }
        }
        ApplicationProtocol::SSH => {
            parsed_packet
                .payload_set
                .proto_hierarchy
                .push_str("->SSH"); 
        }
        _ => {}
    }
    // if parsed_packet.payload_set.header_len
    //     > parsed_packet.payload_set.data.len() as u32
    // {
    //     println!("Header length exceeds data length, resetting header length");
    //     print!("packet: {:02x?}", parsed_packet.payload_set.data);
    // }
    return Ok(parsed_packet);
}

// // Feature Structs and Functions

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum TransportProtocol {
    ICMPv6,
    ICMPv4,
    TCP,
    UDP,
    Unknown,
}

#[derive(Error, Debug)]
pub enum EtherTypeError {
    #[error("Unknown EtherType: 0x{0:04X}")]
    UnknownEtherType(u16),
    #[error("Data too Short to Contain Ethernet")]
    EtherLenError,
}

fn parse_ether_type(packet: &Packet) -> Result<String, EtherTypeError> {
    let data = packet.data;
    if data.len() >= 14 {
        // EtherType is at bytes 12 and 13
        let ethertype = u16::from_be_bytes([data[12], data[13]]);

        let name = match ethertype {
            0x0800 => "IPv4",
            0x86DD => "IPv6",
            0x0806 => "ARP",
            0x8100 => "VLAN (802.1Q)",
            0x88A8 => "Provider Bridging (802.1ad/Q-in-Q)",
            0x8847 => "MPLS Unicast",
            0x8848 => "MPLS Multicast",
            0x8863 => "PPPoE Discovery",
            0x8864 => "PPPoE Session",
            0x8808 => "IEEE Std 802.3 - Ethernet Passive Optical Network (EPON)",
            0x88CC => "LLDP (Link Layer Discovery Protocol)",
            0x88E5 => "MACsec (IEEE 802.1AE)",
            0x8915 => "ROCEv2 (RDMA over Converged Ethernet v2)",
            0x8906 => "Fibre Channel over Ethernet (FCoE)",
            0x880B => "PPP",
            0x8137 => "IPX",
            0x88E3 => "MRP (medium redundancy protocol)",
            0x8892 => "Profinet RT",
            _ => return Err(EtherTypeError::UnknownEtherType(ethertype)),
        };

        Ok(name.to_string())
    } else {
        Err(EtherTypeError::EtherLenError)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArpFeatures {
    pub dst_proto_ipv4: Ipv4Addr,
    pub src_proto_ipv4: Ipv4Addr,
    pub opcode: u32,
    pub hw_size: u32,
}

#[derive(Error, Debug)]
pub enum ArpFeaturesError {
    #[error("Failed to slice packet")]
    SliceError(#[from] err::packet::SliceError),
    #[error("Failed to extract Ether Payload")]
    EtherPayloadError,
    #[error("Ether payload too short")]
    EtherPayloadLengthError,
}

fn get_arp_features(ethernet_slice: SlicedPacket) -> Result<ArpFeatures, ArpFeaturesError> {
    let data = ethernet_slice
        .ether_payload()
        .ok_or(ArpFeaturesError::EtherPayloadError)?
        .payload;

    // Ensure the packet is long enough to contain an ARP packet (28 bytes for IPv4-over-Ethernet)
    if data.len() < 28 {
        return Err(ArpFeaturesError::EtherPayloadLengthError);
    }

    // Extract the ARP fields
    let hw_size = data[4] as u32;
    let opcode = u16::from_be_bytes([data[6], data[7]]) as u32;

    // Source Protocol Address (Sender IPv4 Address)
    let src_proto_ipv4 = Ipv4Addr::new(data[14], data[15], data[16], data[17]);

    // Destination Protocol Address (Target IPv4 Address)
    let dst_proto_ipv4 = Ipv4Addr::new(data[24], data[25], data[26], data[27]);

    // Construct the ArpFeatures struct
    Ok(ArpFeatures {
        dst_proto_ipv4,
        src_proto_ipv4,
        opcode,
        hw_size,
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IpAddresses {
    pub src_host: IpAddr,
    pub dst_host: IpAddr,
}

#[derive(Error, Debug)]
pub enum IpFeaturesError {
    #[error("Failed to slice packet")]
    SliceError(#[from] err::packet::SliceError),
    #[error("Failed to parse net")]
    NoNetSlice,
}

fn parse_ip_layer(
    ethernet_slice: &SlicedPacket,
    parsed_packet_set: &mut ParsedPacketSet,
) -> Result<(), IpFeaturesError> {
    let net_slice = ethernet_slice
        .net
        .as_ref()
        .ok_or(IpFeaturesError::NoNetSlice)?;
    match net_slice {
        etherparse::NetSlice::Ipv4(ipv4_slice) => {
            let src_host = ipv4_slice.header().source_addr();
            let dst_host = ipv4_slice.header().destination_addr();
            parsed_packet_set.features.ip_addresses = IpAddresses {
                src_host: IpAddr::from(src_host),
                dst_host: IpAddr::from(dst_host),
            };
            let ip_header_len: u32 = ipv4_slice.header().slice().len() as u32;
            parsed_packet_set.payload_set.header_len =
                parsed_packet_set.payload_set.header_len + ip_header_len as u32;
            for i in 26..34 {
                parsed_packet_set.payload_set.mask[i] = true;
            }
            return Ok(());
        }
        etherparse::NetSlice::Ipv6(ipv6_slice) => {
            let src_host = ipv6_slice.header().source_addr();
            let dst_host = ipv6_slice.header().destination_addr();
            parsed_packet_set.features.ip_addresses = IpAddresses {
                src_host: IpAddr::from(src_host),
                dst_host: IpAddr::from(dst_host),
            };
            parsed_packet_set.payload_set.header_len =
                parsed_packet_set.payload_set.header_len + 40;
            for i in 22..54 {
                parsed_packet_set.payload_set.mask[i] = true;
            }
            return Ok(());
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpFeatures {
    pub tcp_ack_raw: u32,
    pub tcp_checksum: u16,
    pub tcp_connection_fin: bool,
    pub tcp_connection_rst: bool,
    pub tcp_connection_syn: bool,
    pub tcp_connection_synack: bool,
    pub tcp_dstport: u16,
    pub tcp_flags: u32,
    pub tcp_len: u16,
    pub tcp_options: Vec<u8>,
    pub tcp_seq: u32,
    pub tcp_srcport: u16,
}

#[derive(Error, Debug)]
pub enum TcpFeaturesError {
    #[error("Failed to slice packet")]
    SliceError(#[from] err::packet::SliceError),
    #[error("Failed to retrieve TCP Payload")]
    TcpPayloadError,
}
fn parse_tcp_layer<'a>(
    tcp_header: &'a TcpHeader,
    eth_slice: &'a SlicedPacket,
    parsed_packet: &mut ParsedPacketSet,
) -> Result<(), TcpFeaturesError> {
    // Initialize TcpFeatures with data from the TCP header
    let mut tcp_features = TcpFeatures {
        tcp_ack_raw: tcp_header.acknowledgment_number,
        tcp_checksum: tcp_header.checksum,
        tcp_connection_fin: tcp_header.fin,
        tcp_connection_rst: tcp_header.rst,
        tcp_connection_syn: tcp_header.syn,
        tcp_connection_synack: tcp_header.syn && tcp_header.ack,
        tcp_dstport: tcp_header.destination_port,
        tcp_flags: 0,
        tcp_len: 0,
        tcp_options: Vec::new(),
        tcp_seq: tcp_header.sequence_number,
        tcp_srcport: tcp_header.source_port,
    };

    // Calculate tcp_flags
    tcp_features.tcp_flags = (tcp_header.ns as u32) << 8
        | (tcp_header.cwr as u32) << 7
        | (tcp_header.ece as u32) << 6
        | (tcp_header.urg as u32) << 5
        | (tcp_header.ack as u32) << 4
        | (tcp_header.psh as u32) << 3
        | (tcp_header.rst as u32) << 2
        | (tcp_header.syn as u32) << 1
        | (tcp_header.fin as u32);

    // Set tcp_options if available
    if !tcp_header.options.is_empty() {
        tcp_features.tcp_options = tcp_header.options.to_vec();
    }

    // Set tcp_payload if the length is non-zero
    let transport_slice = eth_slice
        .transport
        .as_ref()
        .ok_or(TcpFeaturesError::TcpPayloadError)?;

    match transport_slice {
        etherparse::TransportSlice::Tcp(tcp_slice) => {
            let payload = tcp_slice.payload();
            let tcp_len = payload.len() as usize;
            tcp_features.tcp_len = tcp_len as u16;
        }
        _ => {
            let tcp_len = tcp_header.header_len() as usize;
            tcp_features.tcp_len = tcp_len as u16;
            //leave payload as empty vec
        }
    }
    parsed_packet.features.tcp_features = tcp_features;

    // Update the payload set with TCP header length
    let data_offset: u8 = tcp_header.data_offset();
    let tcp_header_len: u32 = data_offset as u32 * 4;
    parsed_packet.payload_set.header_len = parsed_packet.payload_set.header_len + tcp_header_len;
    // Check if header len is consistent with data
    if parsed_packet.payload_set.header_len > parsed_packet.payload_set.data.len() as u32 {
        return Err(TcpFeaturesError::TcpPayloadError);
    }
    return Ok(());
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UdpFeatures {
    pub udp_port_dst: u16,
    pub udp_port_src: u16,
    pub udp_stream_id: u32,
    pub udp_time_delta: f32, // Time delta between packets not shure how and when to implement this best tbd
}

#[derive(Error, Debug)]
pub enum UdpFeaturesError {
    #[error("Failed to retrieve UDP Payload")]
    UdpPayloadError,
}
// Revisit udp_stream_id and udp_time_delta later!!
fn get_udp_features<'a>(
    udp_header: &UdpHeader,
    parsed_packet: &mut ParsedPacketSet,
) -> Result<(), UdpFeaturesError> {
    let udp_features = UdpFeatures {
        udp_port_dst: udp_header.destination_port,
        udp_port_src: udp_header.source_port,
        udp_stream_id: 0,    // Placeholder, calculate or assign as needed
        udp_time_delta: 0.0, // Placeholder, calculate or assign as needed
    };
    parsed_packet.features.udp_features = udp_features;
    // Update the payload set with UDP header length
    parsed_packet.payload_set.header_len = parsed_packet.payload_set.header_len + 8; // UDP header length is 8 bytes
    if parsed_packet.payload_set.header_len > parsed_packet.payload_set.data.len() as u32 {
        return Err(UdpFeaturesError::UdpPayloadError);
    }
    return Ok(());
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IcmpFeatures {
    pub icmp_checksum: u16,
    pub icmp_seq_le: u16,
    pub icmp_transmit_time: u32,
    pub icmp_unused: Vec<u8>,
}

#[derive(Error, Debug)]
pub enum IcmpError {
    #[error("failed to slice packet")]
    SliceError(#[from] err::packet::SliceError),
    #[error("failed to get ip Payload")]
    IpPayloadError,
    #[error("failed due to payload length")]
    LenError(#[from] err::LenError),
}

// Correct implementation of transmit time is yet to be confirmed!!
fn get_icmp_v4_features<'a>(packet: &'a Packet) -> Result<IcmpFeatures, IcmpError> {
    let eth_slice = SlicedPacket::from_ethernet(packet)?;
    let ip_payload = eth_slice
        .ip_payload()
        .ok_or(IcmpError::IpPayloadError)?
        .payload;
    let icmpslice = Icmpv4Slice::from_slice(ip_payload)?;
    let packet_type = icmpslice.type_u8();
    let packet_code = icmpslice.code_u8();
    let five2eight = icmpslice.bytes5to8();
    let seq_num = match packet_type {
        0 | 8 => u16::from_be_bytes([five2eight[2], five2eight[3]]),
        _ => 0,
    };
    let transmit_time: u32 = match [packet_type, packet_code] {
        [13, 0] | [14, 0] => u32::from_be_bytes([
            ip_payload[16],
            ip_payload[17],
            ip_payload[18],
            ip_payload[19],
        ]),
        _ => 0,
    };

    Ok(IcmpFeatures {
        icmp_checksum: icmpslice.checksum(),
        icmp_seq_le: seq_num,
        icmp_transmit_time: transmit_time,
        icmp_unused: icmpslice.payload().to_vec(),
    })
}

#[derive(Error, Debug)]
enum Icmpv6Error {
    #[error("Failed to slice packet")]
    SliceError(#[from] err::packet::SliceError),
    #[error("Could not extract Ethernet Payload")]
    NoEtherPayload,
    #[error("Len Error when extracting ip payload")]
    LenError(#[from] err::LenError),
}

fn get_icmp_v6_features<'a>(packet: &'a Packet) -> Result<IcmpFeatures, Icmpv6Error> {
    let data = SlicedPacket::from_ethernet(packet)?
        .ether_payload()
        .ok_or(Icmpv6Error::NoEtherPayload)?
        .payload;
    let (icmp6_header, unused_b) = Icmpv6Header::from_slice(data)?;
    let data = icmp6_header.to_bytes();
    let packet_type: u8 = data[0];
    let packet_code: u8 = data[1];

    // Default sequence number for non-relevant ICMPv6 types
    let seq_num: u16 = match packet_type {
        128 | 129 => u16::from_be_bytes([data[4], data[5]]), // ICMPv6 Echo Request/Reply
        _ => 0,
    };

    let transmit_time: u32;
    // Use ICMPv6 time-related message types and extract relevant bytes for the timestamp
    if (packet_type == 133 || packet_type == 134) && (packet_code == 0) {
        // Use appropriate byte offsets for the transmit time in ICMPv6 packets if needed
        transmit_time = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    } else {
        transmit_time = 0;
    }

    let icmp_features = IcmpFeatures {
        icmp_checksum: icmp6_header.checksum,
        icmp_seq_le: seq_num, // bytes 4 and 5 for ICMPv6
        icmp_transmit_time: transmit_time,
        icmp_unused: unused_b.to_vec(),
    };

    return Ok(icmp_features);
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum ApplicationProtocol {
    HTTP,
    HTTPS,
    DNS,
    MDNS,
    MQTT,
    MBTCP,
    SSH,
    Unknown,
}

fn detect_protocol_candidate(
    transport: &TransportProtocol,
    src_port: u16,
    dest_port: u16,
) -> ApplicationProtocol {
    match transport {
        TransportProtocol::TCP => match_tcp_ports(src_port, dest_port),
        TransportProtocol::UDP => match_udp_ports(src_port, dest_port),
        _ => ApplicationProtocol::Unknown,
    }
}

fn match_tcp_ports(src_port: u16, dest_port: u16) -> ApplicationProtocol {
    // Check both directions: client->server (dest) and server->client (src)
    match (src_port, dest_port) {
        (80, _) | (_, 80) | (8080, _) | (_, 8080) | (8888, _) | (_, 8888) | (3000, _) | (_, 3000)  | (8000, _) | (_, 8000) | (8008, _) | (_, 8008) => ApplicationProtocol::HTTP,
        (443, _) | (_, 443) => ApplicationProtocol::HTTPS,
        (1883, _) | (_, 1883) | (8883, _) | (_, 8883) => ApplicationProtocol::MQTT,
        (502, _) | (_, 502) => ApplicationProtocol::MBTCP,
        (53, _) | (_, 53) => ApplicationProtocol::DNS,
        (5353, _) | (_, 5353) => ApplicationProtocol::MDNS,
        (22, _) | (_, 22) => ApplicationProtocol::SSH,
        _ => ApplicationProtocol::Unknown,
    }
}

fn match_udp_ports(src_port: u16, dest_port: u16) -> ApplicationProtocol {
    match (src_port, dest_port) {
        (53, _) | (_, 53) => ApplicationProtocol::DNS,
        (5353, _) | (_, 5353) => ApplicationProtocol::MDNS,
        _ => ApplicationProtocol::Unknown,
    }
}
#[derive(Error, Debug)]
pub enum DnsMdnsError {
    #[error("Failed to Parse DNS/MDNS Packet")]
    DNSParsingError(#[from] Box<dyn std::error::Error>),
    #[error("Simple DNS parsing error: {0}")]
    SimpleDnsError(#[from] simple_dns::SimpleDnsError),
    #[error("No questions found in packet")]
    NoQuestions,
    #[error("Not a query packet")]
    NotQuery,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DnsFeatures {
    pub rfc_compatible: bool,
    pub mdns: bool,
    pub dns_qry_name: String,
    pub dns_qry_name_len: u16,
    pub dns_qry_qu: bool,
    pub dns_qry_type: u16,
    pub dns_retransmit: bool,
    pub dns_retransmit_request_in: Option<u32>,
}

// Context for tracking retransmissions
pub struct DnsContext {
    previous_queries: HashMap<String, (u32, u16)>, // (timestamp, transaction_id)
}

impl DnsContext {
    pub fn new() -> Self {
        Self {
            previous_queries: HashMap::new(),
        }
    }
    
    /// Clear old entries to prevent unbounded memory growth
    pub fn cleanup_old_entries(&mut self, current_timestamp: u32, max_age_ms: u32) {
        self.previous_queries.retain(|_, (timestamp, _)| {
            current_timestamp.saturating_sub(*timestamp) <= max_age_ms
        });
    }
}

pub fn parse_dns(
    packet: &[u8],
    timestamp: u32,
    context: &mut DnsContext,
) -> Result<DnsFeatures, DnsMdnsError> {
    // Parse the packet using simple_dns
    // let dns_packet = DnsPacket::parse(packet)?;
    if let Ok(dns_packet) = DnsPacket::parse(packet) {
        // Proceed with processing
        // Extract transaction ID for later
        let id_c = dns_packet.id();

        // Get the first question
        let questions = dns_packet.questions;
        if questions.is_empty() {
            return Err(DnsMdnsError::NoQuestions);
        }
        
        let question = &questions[0];

        // Only process query packets (not responses)
        if question.unicast_response {
            return Err(DnsMdnsError::NotQuery);
        }

        // Extract features from the question
        let qname_str = question.qname.to_string();
        let qtype_value = question.qtype.into();
        let is_unicast = question.unicast_response;

        // Create query key for retransmission detection
        let query_key = format!("{}:{}:{}", qname_str, qtype_value, u16::from(question.qclass));

        // Check for retransmission
        let (is_retransmit, retransmit_delta) = check_retransmission(
            &query_key,
            timestamp,
            id_c,
            context,
        );

        let features = DnsFeatures {
            rfc_compatible: true,
            mdns: false,
            dns_qry_name: qname_str.clone(),
            dns_qry_name_len: qname_str.len() as u16,
            dns_qry_qu: is_unicast,
            dns_qry_type: qtype_value,
            dns_retransmit: is_retransmit,
            dns_retransmit_request_in: retransmit_delta,
        };

        // Update context with current query
        context.previous_queries.insert(query_key, (timestamp, id_c));
        
        // Clean up old entries periodically to prevent memory bloat
        if context.previous_queries.len() > 1000 {
            context.cleanup_old_entries(timestamp, 300_000); // 5 minutes
        }

        Ok(features)
    } else {
        let features = DnsFeatures{
                rfc_compatible: false,
                mdns: false,
                dns_qry_name: "".to_string(),
                dns_qry_name_len: 0,
                dns_qry_qu: false,
                dns_qry_type: 0,
                dns_retransmit: false,
                dns_retransmit_request_in: None,
        };
        Ok(features)
}
}

fn check_retransmission(
    query_key: &str,
    current_timestamp: u32,
    current_id: u16,
    context: &DnsContext,
) -> (bool, Option<u32>) {
    if let Some(&(prev_timestamp, _prev_id)) = context.previous_queries.get(query_key) {
        // Consider it a retransmission if it's the same query within a reasonable time window
        let time_diff = current_timestamp.saturating_sub(prev_timestamp);
        
        // Retransmission window: typically DNS clients retry within 1-60 seconds
        if time_diff <= 60_000 && time_diff > 0 { // 60 seconds in milliseconds, but not identical timestamp
            return (true, Some(time_diff));
        }
    }
    
    (false, None)
}

#[derive(Error, Debug)]
enum MqttError {
    #[error("Failed to Parse Mqtt Packet")]
    MqttErrorParsingError(#[from] Box<dyn std::error::Error>),
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MqttFeatures {
    pub mqtt_conack_flags: u8,
    pub mqtt_conflag_cleansess: bool,
    pub mqtt_conflags: u8,
    pub mqtt_hdrflags: u8,
    pub mqtt_len: u32,
    pub mqtt_msgtype: u8,
    pub mqtt_proto_len: u16,
    pub mqtt_protoname: String,
    pub mqtt_topic: String,
    pub mqtt_topic_len: u16,
    pub mqtt_ver: u8,
}
fn parse_mqtt(payload: Vec<u8>, parsed_packet: &mut ParsedPacketSet) -> Result<(), MqttError> {
    let mut cursor = Cursor::new(&payload);
    let header_len_fallback = payload.len();
    
    if payload.is_empty() {
        parsed_packet.payload_set.header_len = parsed_packet.payload_set.data.len() as u32;
        return Ok(());
    }
    
    // Parse fixed header
    let first_byte = cursor
        .read_u8()
        .map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    let msgtype = (first_byte >> 4) & 0x0F;
    let mqtt_hdrflags = first_byte & 0x0F;
    
    parsed_packet.features.mqtt_features.mqtt_msgtype = msgtype;
    parsed_packet.features.mqtt_features.mqtt_hdrflags = mqtt_hdrflags;
    
    // Parse remaining length
    let mqtt_len = parse_remaining_length(&mut cursor)?;
    parsed_packet.features.mqtt_features.mqtt_len = mqtt_len;
    
    // Save cursor position after fixed header
    let fixed_header_end = cursor.position() as usize;
    
    // Parse variable header and payload based on message type
    match msgtype {
        1 => parse_connect(&mut cursor, &mut parsed_packet.features.mqtt_features)?,
        2 => parse_connack(&mut cursor, &mut parsed_packet.features.mqtt_features)?,
        3 => parse_publish(&mut cursor, &mut parsed_packet.features.mqtt_features)?,
        _ => (),
    };
    
    // Calculate total MQTT header length (fixed header + variable header)
    // The variable header ends where the payload begins
    let total_header_len = cursor.position() as usize;
    
    parsed_packet.payload_set.header_len += total_header_len as u32;
    
    Ok(())
}

fn mqtt_fixed_header_length(packet: &[u8]) -> Option<usize> {
    if packet.is_empty() {
        return None; // Empty packet
    }

    let mut length_bytes = 1; // 1 byte for control type
    let mut _multiplier = 1;
    // let mut value = 0;

    for i in 1..packet.len().min(5) {
        // Remaining Length field is at most 4 bytes
        let byte = packet[i];
        // value += (byte as usize & 0x7F) * multiplier;
        _multiplier *= 128;
        length_bytes += 1;
        if byte & 0x80 == 0 {
            break;
        }
    }

    Some(length_bytes) // Fixed header length = 1 (control byte) + length of Remaining Length field
}

fn parse_remaining_length(
    cursor: &mut Cursor<&Vec<u8>>,
) -> Result<u32, Box<dyn std::error::Error>> {
    let mut multiplier = 1;
    let mut value = 0;
    let mut byte_count = 0;
    
    loop {
        // MQTT spec: remaining length is encoded in at most 4 bytes
        if byte_count >= 4 {
            return Err("Remaining length exceeds maximum of 4 bytes".into());
        }
        
        let encoded_byte = cursor.read_u8()? as u32;
        value += (encoded_byte & 127) * multiplier;
        
        byte_count += 1;
        
        // Check if this is the last byte (bit 7 is 0)
        if encoded_byte & 128 == 0 {
            break;
        }
        
        // Only multiply if we're going to loop again
        multiplier *= 128;
    }
    
    Ok(value)
}

fn parse_connect(
    cursor: &mut Cursor<&Vec<u8>>,
    features: &mut MqttFeatures,
) -> Result<(), MqttError> {
    let proto_name_len = cursor
        .read_u16::<BigEndian>()
        .map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    let mut proto_name = vec![0u8; proto_name_len as usize];
    cursor
        .read_exact(&mut proto_name)
        .map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    features.mqtt_protoname =
        String::from_utf8(proto_name).map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    features.mqtt_proto_len = proto_name_len;

    features.mqtt_ver = cursor
        .read_u8()
        .map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    features.mqtt_conflags = cursor
        .read_u8()
        .map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    features.mqtt_conflag_cleansess = (features.mqtt_conflags & 0x02) != 0;

    Ok(())
}

fn parse_connack(
    cursor: &mut Cursor<&Vec<u8>>,
    features: &mut MqttFeatures,
) -> Result<(), MqttError> {
    features.mqtt_conack_flags = cursor
        .read_u8()
        .map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    Ok(())
}

fn parse_publish(
    cursor: &mut Cursor<&Vec<u8>>,
    features: &mut MqttFeatures,
) -> Result<(), MqttError> {
    let topic_len = cursor
        .read_u16::<BigEndian>()
        .map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    let mut topic = vec![0u8; topic_len as usize];
    cursor
        .read_exact(&mut topic)
        .map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    features.mqtt_topic =
        String::from_utf8(topic).map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    features.mqtt_topic_len = topic_len;

    // Read the rest of the payload as the message
    let mut msg = Vec::new();
    cursor
        .read_to_end(&mut msg)
        .map_err(|e| MqttError::MqttErrorParsingError(Box::new(e)))?;
    Ok(())
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ModbusTcpFeatures {
    pub mbtcp_len: u16,
    pub mbtcp_trans_id: u16,
    pub mbtcp_unit_id: u8,
}

fn parse_modbus_tcp(payload: Vec<u8>) -> Result<ModbusTcpFeatures, Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(&payload);
    let mut features = ModbusTcpFeatures::default();

    // Parse Transaction Identifier (2 bytes)
    features.mbtcp_trans_id = cursor.read_u16::<BigEndian>()?;

    // Skip Protocol Identifier (2 bytes)
    cursor.set_position(cursor.position() + 2);

    // Parse Length (2 bytes)
    features.mbtcp_len = cursor.read_u16::<BigEndian>()?;

    // Parse Unit Identifier (1 byte)
    features.mbtcp_unit_id = cursor.read_u8()?;

    Ok(features)
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct HttpFeatures {
    pub http_content_length: u64,
    pub http_request_uri_query: String,
    pub http_request_method: String,
    pub http_referer: String,
    pub http_request_full_uri: String,
    pub http_request_version: String,
    pub http_response: bool,
    pub parse_error: Option<String>,
}

fn parse_http(
    payload: Vec<u8>,
    parsed_packet: &mut ParsedPacketSet,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut features = HttpFeatures::default();
    
    // Safeguard 1: Check minimum payload size
    if payload.is_empty() || payload.len() < 16 {
        features.parse_error = Some("HTTP payload too small".to_string());
        parsed_packet.features.http_features = features;
        return Ok(());
    }

    // Safeguard 2: Limit maximum payload size to prevent memory issues
    const MAX_HTTP_PAYLOAD: usize = 10 * 1024 * 1024; // 10MB
    if payload.len() > MAX_HTTP_PAYLOAD {
        features.parse_error = Some("HTTP payload too large".to_string());
        parsed_packet.features.http_features = features;
        return Ok(());
    }

    let payload_str = String::from_utf8_lossy(&payload);
    
    // Safeguard 3: Split with limit to prevent excessive allocation
    let lines: Vec<&str> = payload_str.split("\r\n").take(500).collect();
    
    if lines.is_empty() {
        features.parse_error = Some("Empty HTTP payload".to_string());
        parsed_packet.features.http_features = features;
        return Ok(());
    }

    // Parse the request/status line
    let first_line = lines[0].trim();
    
    // Safeguard 4: Check first line length
    if first_line.is_empty() || first_line.len() > 8192 {
        features.parse_error = Some("Invalid HTTP first line length".to_string());
        parsed_packet.features.http_features = features;
        return Ok(());
    }

    let parts: Vec<&str> = first_line.split_whitespace().collect();
    
    if parts.len() < 3 {
        features.parse_error = Some("Invalid HTTP first line format".to_string());
        parsed_packet.features.http_features = features;
        return Ok(());
    }

    // Safeguard 5: Validate HTTP version format
    let is_valid_http_version = |v: &str| {
        v.starts_with("HTTP/") && v.len() >= 8 && v.len() <= 12
    };

    if parts[0].starts_with("HTTP/") {
        // This is a response
        if !is_valid_http_version(parts[0]) {
            features.parse_error = Some("Invalid HTTP version format".to_string());
            parsed_packet.features.http_features = features;
            return Ok(());
        }
        features.http_response = true;
        features.http_request_version = parts[0].to_string();
    } else {
        // This is a request
        // Safeguard 6: Validate HTTP method (common methods)
        const VALID_METHODS: &[&str] = &[
            "GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", 
            "PATCH", "TRACE", "CONNECT"
        ];
        
        if !VALID_METHODS.contains(&parts[0]) && parts[0].len() > 20 {
            features.parse_error = Some("Invalid HTTP method".to_string());
            parsed_packet.features.http_features = features;
            return Ok(());
        }

        features.http_request_method = parts[0].to_string();
        
        // Safeguard 7: Validate URI length
        if parts[1].len() > 8192 {
            features.parse_error = Some("HTTP URI too long".to_string());
            parsed_packet.features.http_features = features;
            return Ok(());
        }
        
        features.http_request_full_uri = parts[1].to_string();
        
        // Safeguard 8: Validate HTTP version in request
        if !is_valid_http_version(parts[2]) {
            features.parse_error = Some("Invalid HTTP version format".to_string());
            parsed_packet.features.http_features = features;
            return Ok(());
        }
        
        features.http_request_version = parts[2].to_string();

        // Parse query from full URI
        if let Some((_, query)) = features.http_request_full_uri.split_once('?') {
            // Safeguard 9: Limit query string length
            if query.len() <= 4096 {
                features.http_request_uri_query = query.to_string();
            } else {
                features.http_request_uri_query = query[..4096].to_string();
            }
        }
    }

    parsed_packet.features.http_features = features;

    // Calculate header length with safeguards
    let mut headers_end = 0;
    let mut byte_offset = 0;
    
    // Safeguard 10: Limit header parsing iterations
    const MAX_HEADER_LINES: usize = 200;
    let mut line_count = 0;
    
    for line in payload_str.lines() {
        line_count += 1;
        if line_count > MAX_HEADER_LINES {
            break;
        }
        
        byte_offset += line.len() + 2; // Account for "\r\n"
        
        // Safeguard 11: Prevent excessive header size
        if byte_offset > 65536 { // 64KB header limit
            break;
        }
        
        if line.is_empty() {
            headers_end = byte_offset;
            break;
        }
    }

    // Safeguard 12: Ensure header_len doesn't overflow
    parsed_packet.payload_set.header_len = parsed_packet.payload_set.header_len
        .saturating_add(headers_end as u32);

    Ok(())
}

// Combined feature struct plus placeholder initialization
#[derive(Debug, Serialize, Deserialize)]
pub struct ProtocolFeatureSet {
    pub arp_set: bool,
    pub arp_features: ArpFeatures,
    pub ip_set: bool,
    pub ip_addresses: IpAddresses,
    pub tcp_set: bool,
    pub tcp_features: TcpFeatures,
    pub udp_set: bool,
    pub udp_features: UdpFeatures,
    pub icmp_set: bool,
    pub icmp_features: IcmpFeatures,
    pub transport_protocol: TransportProtocol,
    pub application_protocol: ApplicationProtocol,
    pub DNS_features: DnsFeatures,
    pub mqtt_features: MqttFeatures,
    pub modbus_tcp_features: ModbusTcpFeatures,
    pub http_features: HttpFeatures,
}

impl ProtocolFeatureSet {
    pub fn new() -> Self {
        ProtocolFeatureSet {
            arp_set: false,
            arp_features: ArpFeatures {
                dst_proto_ipv4: Ipv4Addr::new(0, 0, 0, 0),
                src_proto_ipv4: Ipv4Addr::new(0, 0, 0, 0),
                opcode: 0,
                hw_size: 0,
            },
            ip_set: false,
            ip_addresses: IpAddresses {
                src_host: IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
                dst_host: IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            },
            tcp_set: false,
            tcp_features: TcpFeatures {
                tcp_ack_raw: 0,
                tcp_checksum: 0,
                tcp_connection_fin: false,
                tcp_connection_rst: false,
                tcp_connection_syn: false,
                tcp_connection_synack: false,
                tcp_dstport: 0,
                tcp_flags: 0,
                tcp_len: 0,
                tcp_options: Vec::new(),
                tcp_seq: 0,
                tcp_srcport: 0,
            },
            udp_set: false,
            udp_features: UdpFeatures {
                udp_port_dst: 0,
                udp_port_src: 0,
                udp_stream_id: 0,
                udp_time_delta: 0.0,
            },
            icmp_set: false,
            icmp_features: IcmpFeatures {
                icmp_checksum: 0,
                icmp_seq_le: 0,
                icmp_transmit_time: 0,
                icmp_unused: Vec::new(),
            },
            transport_protocol: TransportProtocol::Unknown,
            application_protocol: ApplicationProtocol::Unknown,
            DNS_features: DnsFeatures {
                rfc_compatible: false,
                mdns: false,
                dns_qry_name: "".to_string(),
                dns_qry_name_len: 0,
                dns_qry_qu: false,
                dns_qry_type: 0,
                dns_retransmit: false,
                dns_retransmit_request_in: None,
            },
            mqtt_features: MqttFeatures {
                mqtt_conack_flags: 0,
                mqtt_conflag_cleansess: false,
                mqtt_conflags: 0,
                mqtt_hdrflags: 0,
                mqtt_len: 0,
                mqtt_msgtype: 0,
                mqtt_proto_len: 0,
                mqtt_protoname: "".to_string(),
                mqtt_topic: "".to_string(),
                mqtt_topic_len: 0,
                mqtt_ver: 0,
            },
            modbus_tcp_features: ModbusTcpFeatures {
                mbtcp_len: 0,
                mbtcp_trans_id: 0,
                mbtcp_unit_id: 0,
            },
            http_features: HttpFeatures {
                http_content_length: 0,
                http_request_uri_query: "".to_string(),
                http_request_method: "".to_string(),
                http_referer: "".to_string(),
                http_request_full_uri: "".to_string(),
                http_request_version: "".to_string(),
                http_response: false,
                parse_error: None,
            },
        }
    }
}

/// Represents a timestamp with second and microsecond precision.
/// This is a wrapper around the `libc::timeval` structure used by libpcap,
/// providing serialization support and a more idiomatic Rust interface.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeStamp {
    /// Seconds since Unix epoch (January 1, 1970)
    pub seconds: i64,
    // Seconds since Unix epoch (January 1, 1970) 
    pub microseconds: i64,
}

impl TimeStamp {
    pub fn new(seconds: i64, microseconds: i64) -> Self {
        TimeStamp {
            seconds,
            microseconds,
        }
    }
    
    /// Returns the seconds component of the timestamp
    pub fn as_secs(&self) -> i64 {
        self.seconds
    }
    
    /// Returns the total timestamp as microseconds
    pub fn as_micros(&self) -> i64 {
        self.seconds * 1_000_000 + self.microseconds
    }
    /// Calculates the duration in microseconds since the given timestamp.
    /// Returns the difference as `std::time::Duration`.
    pub fn duration_since(&self, earlier: &TimeStamp) -> Duration {
        let seconds_diff = (self.seconds - earlier.seconds) as u64;
        let microseconds_diff = self.microseconds - earlier.microseconds;
        
        if microseconds_diff >= 0 {
            Duration::new(seconds_diff, (microseconds_diff * 1000) as u32)
        } else {
            Duration::new(seconds_diff - 1, ((1_000_000 + microseconds_diff) * 1000) as u32)
        }
    }
}

impl ToRedisArgs for ParsedPacketSet {
    fn write_redis_args<W>(&self, out: &mut W)
    where
        W: ?Sized + RedisWrite,
    {
        let serialized = to_vec(self).expect("Serialization failed");
        out.write_arg(&serialized);
    }
}

impl FromRedisValue for ParsedPacketSet {
    fn from_redis_value(v: &Value) -> RedisResult<Self> {
        match *v {
            Value::BulkString(ref bytes) => from_slice(bytes).map_err(|e| {
                RedisError::from((
                    redis::ErrorKind::ParseError,
                    "Failed to deserialize CombinedFeatures",
                    e.to_string(),
                ))
            }),
            _ => Err(RedisError::from((
                redis::ErrorKind::TypeError,
                "Expected binary data for CombinedFeatures",
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayloadSet {
    pub data: Vec<u8>,
    pub mask: Vec<bool>,
    pub header_len: u32,
    pub proto_hierarchy: String,
}

impl PayloadSet {
    pub fn new(data: Vec<u8>) -> Self {
        PayloadSet {
            mask: vec![false; data.len()],
            data: data,
            header_len: 0,
            proto_hierarchy: String::new(),
        }
    }
}

impl ToRedisArgs for PayloadSet {
    fn write_redis_args<W>(&self, out: &mut W)
    where
        W: ?Sized + RedisWrite,
    {
        let serialized = to_vec(self).expect("Serialization failed");
        out.write_arg(&serialized);
    }
}

impl FromRedisValue for PayloadSet {
    fn from_redis_value(v: &Value) -> RedisResult<Self> {
        match *v {
            Value::BulkString(ref bytes) => from_slice(bytes).map_err(|e| {
                RedisError::from((
                    redis::ErrorKind::ParseError,
                    "Failed to deserialize PayloadSet",
                    e.to_string(),
                ))
            }),
            _ => Err(RedisError::from((
                redis::ErrorKind::TypeError,
                "Expected binary data for PayloadSet",
            ))),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ParsedPacketSet {
    pub timevalue: TimeStamp, 
    pub features: ProtocolFeatureSet,
    pub payload_set: PayloadSet,
}

impl ParsedPacketSet {
    pub fn new(data: Vec<u8>, timevalue: timeval) -> Self {
        ParsedPacketSet {
            timevalue: TimeStamp::new(timevalue.tv_sec, timevalue.tv_usec),
            features: ProtocolFeatureSet::new(),
            payload_set: PayloadSet::new(data),
        }
    }
}
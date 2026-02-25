use byteorder::{BigEndian, ReadBytesExt};
// use dns_parser::{Packet as DnsPacket, QueryType};
use etherparse::{err::{self}, InternetSlice, PacketHeaders, SlicedPacket, TcpHeader,
    TransportHeader};
use pcap::Packet;
use std::io::Cursor;
use std::str;
use thiserror::Error;
// Redis imports
use rmp_serde::{from_slice, to_vec};
use redis::{FromRedisValue, RedisError, RedisResult, RedisWrite, ToRedisArgs, Value};
use serde::{Deserialize, Serialize};
use log::{warn, error};

#[derive(Error, Debug)]
pub enum ParserError {
    #[error("Failed to find EtherType")]
    EtherTypeError(#[from] EtherTypeError),
    #[error("Failed to slice packet")]
    SliceError(#[from] err::packet::SliceError),
    #[error("Failed to find TransportHeader")]
    TransportHeaderError,
    #[error("Failed to find TCP features")]
    TcpFeatureError(#[from] TcpFeaturesError),
    #[error("Failed to find UDP features")]
    UdpFeatureError(#[from] UdpFeaturesError),
    #[error("Failed to parse mqtt")]
    MqttError(#[from] MqttError),
}

// Parse Ethernet package
pub fn parse_packet(packet: &Packet) -> Result<MaskedHeaderPayload, ParserError> {
    // create masked_header_payload struct
    let header_len: u32 = 0;
    // create deep copy of packet.data for feature masking
    let packet_data = packet.data.to_vec();
    // create mask for packet_data filled with 0
    let mask: Vec<u8> = vec![0; packet_data.len()];
    let mut parsed_packet = MaskedHeaderPayload {
        header_len: header_len,
        data: packet_data,
        mask: mask,
        proto_hierarchy: String::new(),
    };
    let ether_type = parse_ether_type(packet)?;
    // println!("EtherType: {:?}", ether_type);
    let eth_slice = SlicedPacket::from_ethernet(packet.data)?;
    let eth_header = PacketHeaders::from_ethernet_slice(packet.data)?;
    parsed_packet.header_len = 14;
    for i in 0..12 {parsed_packet.mask[i] = 1;}
    parsed_packet.proto_hierarchy.push_str("Ethernet -> ");
    if ether_type == "ARP".to_string() {
        if let Ok(_arp_parsed) = parse_arp_header(eth_slice) {
            parsed_packet.header_len = parsed_packet.header_len + 28;
            parsed_packet.proto_hierarchy.push_str("ARP");
            for i in 22..42 {parsed_packet.mask[i] = 1;}
            return Ok(parsed_packet);
        }
        return Ok(parsed_packet);
    } else if ether_type == "IPv4".to_string() || ether_type == "IPv6".to_string() {
        if let Ok(_ip_parsed) = parse_ip_layer(&eth_slice, &mut parsed_packet) {
            parsed_packet.proto_hierarchy.push_str(&ether_type.as_str());
            parsed_packet.proto_hierarchy.push_str(" -> ");
        } else {
            // println!("Error parsing IP layer");
            return Ok(parsed_packet);
        }
    }
    let transport_header = eth_header
        .transport
        .as_ref()
        .ok_or(ParserError::TransportHeaderError)?;

    let mut dest_port: u16 = 0;

    let transport_protocol = match transport_header {
        TransportHeader::Icmpv4(_) => {
            parsed_packet.header_len = parsed_packet.header_len + 8;
            parsed_packet.proto_hierarchy.push_str("ICMPv4");
            return Ok(parsed_packet);
        }
        TransportHeader::Icmpv6(_) => {
            parsed_packet.header_len = parsed_packet.header_len + 8;
            parsed_packet.proto_hierarchy.push_str("ICMPv6");
            return Ok(parsed_packet);
        }
        TransportHeader::Tcp(tcp_header) => {
            if let Ok(_tcp_parsed) = parse_tcp_layer(&tcp_header, &mut parsed_packet) {
                dest_port = tcp_header.destination_port;
                parsed_packet.proto_hierarchy.push_str("TCP -> ");
            }
            TransportProtocol::TCP
        }
        TransportHeader::Udp(udp_header) => {
            if let Ok(_transport_features) = parse_udp_layer(&mut parsed_packet) {
                dest_port = udp_header.destination_port;
                parsed_packet.proto_hierarchy.push_str("UDP -> ");
            }
            TransportProtocol::UDP
        }
    };

    let application_protocol_candidate = detect_protocol_candidate(&transport_protocol, dest_port);

    match application_protocol_candidate {
        ApplicationProtocol::DNS | ApplicationProtocol::MDNS => {
            if parsed_packet.data.len() < 12 + parsed_packet.header_len as usize {
                return Ok(parsed_packet);
            }
            parsed_packet.header_len += 12;
            parsed_packet.proto_hierarchy.push_str("DNS");
        }
        ApplicationProtocol::MQTT => {
            parse_mqtt(&mut parsed_packet)?;
        }
        ApplicationProtocol::MBTCP => {
            if parsed_packet.data.len() < 8 + parsed_packet.header_len as usize {
                return Ok(parsed_packet);
            }
            parsed_packet.header_len += 8;
            parsed_packet.proto_hierarchy.push_str("MBTCP");
        }
        ApplicationProtocol::HTTP => {
            if parse_http(&mut parsed_packet).is_ok() {
                parsed_packet.proto_hierarchy.push_str("HTTP");
            }
        }
        ApplicationProtocol::HTTPS => {
            if parse_http(&mut parsed_packet).is_ok() {
                parsed_packet.proto_hierarchy.push_str("HTTPS");
            }
        }
        _ => {}
    }

    return Ok(parsed_packet)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedHeaderPayload {
    pub data: Vec<u8>,
    pub mask: Vec<u8>,
    pub header_len: u32,
    pub proto_hierarchy: String,
}

impl MaskedHeaderPayload {
    pub fn new(data: Vec<u8>, mask: Vec<u8>, header_len: u32) -> Self {
        MaskedHeaderPayload {
            data: data,
            mask: mask,
            header_len: header_len,
            proto_hierarchy: String::new(),
        }
    }
}

#[derive(Error, Debug)]
pub enum EtherTypeError {
    #[error("Unknown EtherType")]
    EtherTypeError,
    #[error("Data too Short to Contain Ethernet")]
    EtherLenError,
}

fn parse_ether_type(packet: &Packet) -> Result<String, EtherTypeError> {
    let data = packet.data;
    if data.len() >= 14 {
        // The EtherType is located at bytes 12 and 13 in the Ethernet header
        let ethertype = u16::from_be_bytes([data[12], data[13]]);

        match ethertype {
            0x0800 => return Ok("IPv4".to_string()),
            0x86DD => return Ok("IPv6".to_string()),
            0x0806 => return Ok("ARP".to_string()),
            _ => return Err(EtherTypeError::EtherTypeError),
        }
    } else {
        return Err(EtherTypeError::EtherLenError);
    }
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

fn parse_arp_header(ethernet_slice: SlicedPacket) -> Result<(), ArpFeaturesError> {
    let data = ethernet_slice
        .ether_payload()
        .ok_or(ArpFeaturesError::EtherPayloadError)?
        .payload;

    // Ensure the packet is long enough to contain an ARP packet (28 bytes for IPv4-over-Ethernet)
    if data.len() < 28 {
        return Err(ArpFeaturesError::EtherPayloadLengthError);
    }

    Ok(())
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
    parsed_packet: &mut MaskedHeaderPayload,
) -> Result<(), IpFeaturesError> {
    let net_slice = ethernet_slice
        .net
        .as_ref()
        .ok_or(IpFeaturesError::NoNetSlice)?;
    match net_slice {
        InternetSlice::Ipv4(ipv4_slice) => {
            let ip_header_len: u32 = ipv4_slice.header().slice().len() as u32;
            parsed_packet.header_len = parsed_packet.header_len + ip_header_len as u32;
            for i in 26..34 {parsed_packet.mask[i] = 1;}
            return Ok(());
        }
        InternetSlice::Ipv6(_ipv6_slice) => {
            parsed_packet.header_len = parsed_packet.header_len + 40;
            for i in 22..54 {parsed_packet.mask[i] = 1;}
            return Ok(());
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum TransportProtocol {
    ICMPv6,
    ICMPv4,
    TCP,
    UDP,
    Unknown,
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
    parsed_packet: &mut MaskedHeaderPayload,
) -> Result<(), TcpFeaturesError> {
    // find data offset
    let data_offset: u8 = tcp_header.data_offset();
    let tcp_header_len: u32 = data_offset as u32 * 4;
    parsed_packet.header_len = parsed_packet.header_len + tcp_header_len;
    // Check if header len is consistent with data
    if parsed_packet.header_len > parsed_packet.data.len() as u32 {
        return Err(TcpFeaturesError::TcpPayloadError);
    }
    return Ok(());
}

#[derive(Error, Debug)]
pub enum UdpFeaturesError {
    #[error("Failed to retrieve TCP Payload")]
    UdpPayloadError,
}
// Revisit udp_stream_id and udp_time_delta later!!
fn parse_udp_layer<'a>(parsed_packet: &mut MaskedHeaderPayload) -> Result<(), UdpFeaturesError> {
    parsed_packet.header_len = parsed_packet.header_len + 8;
    if parsed_packet.header_len > parsed_packet.data.len() as u32 {
        return Err(UdpFeaturesError::UdpPayloadError);
    }
    return Ok(());
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum ApplicationProtocol {
    HTTP,
    HTTPS,
    DNS,
    MDNS,
    MQTT,
    MBTCP,
    Unknown,
}

fn detect_protocol_candidate(transport: &TransportProtocol, dest_port: u16) -> ApplicationProtocol {
    if transport == &TransportProtocol::TCP {
        match dest_port {
            80 | 8080 => ApplicationProtocol::HTTP,
            443 => ApplicationProtocol::HTTPS,
            1883 | 8883 => ApplicationProtocol::MQTT,
            502 => ApplicationProtocol::MBTCP,
            _ => ApplicationProtocol::Unknown,
        }
    } else if transport == &TransportProtocol::UDP {
        match dest_port {
            53 => ApplicationProtocol::DNS,
            5353 => ApplicationProtocol::MDNS,
            _ => ApplicationProtocol::Unknown,
        }
    } else {
        return ApplicationProtocol::Unknown;
    }
}

#[derive(Error, Debug)]
pub enum MqttError {
    #[error("Failed to parse MQTT message")]
    BoxedError(#[from] Box<dyn std::error::Error>),
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("String error: {0}")]
    Utf8Error(#[from] str::Utf8Error),
    #[error("Error while parsing Fixed Header")]
    FixedHeaderError,
    #[error("Unknown MQTT message type")]
    UnknownMessageType,
}

fn parse_mqtt(parsed_packet: &mut MaskedHeaderPayload) -> Result<(), MqttError> {
    let payload = &parsed_packet.data[parsed_packet.header_len as usize..];
    let mut cursor = Cursor::new(&payload);

    // Default to setting the header length to the full packet length in case of errors
    let header_len_fallback = payload.len();

    if payload.is_empty() {
        log::warn!("Empty MQTT packet or Ack Syn Synack packet");
        parsed_packet.header_len = parsed_packet.data.len() as u32;
        return Ok(());
    }

    // Parse fixed header
    let mqtt_hdrflags = match cursor.read_u8() {
        Ok(byte) => byte,
        Err(e) => {
            log::warn!("Failed to read MQTT fixed header: {:?}", e);
            parsed_packet.header_len += header_len_fallback as u32;
            return Ok(());
        }
    };
    let mqtt_msgtype = (mqtt_hdrflags & 0xF0) >> 4;

    // Parse remaining length
    let mqtt_len = match parse_remaining_length(&mut cursor) {
        Ok(len) => len,
        Err(e) => {
            log::warn!("Failed to parse remaining length: {:?}", e);
            parsed_packet.header_len += header_len_fallback as u32;
            return Ok(());
        }
    };

    let header_len = match mqtt_fixed_header_length(payload) {
        Some(len) => len,
        None => {
            log::warn!("Invalid fixed header length");
            parsed_packet.header_len += header_len_fallback as u32;
            return Ok(());
        }
    };

    // Determine variable header length safely
    let variable_header_len = match mqtt_msgtype {
        1 => 10,  // CONNECT (Fixed 10 bytes)
        2 => 2,   // CONNACK (Fixed 2 bytes)
        3 => {
            if cursor.get_ref().len() < 2 {
                log::warn!("Packet too short for PUBLISH topic length");
                parsed_packet.header_len += header_len_fallback as u32;
                return Ok(());
            }
            let topic_len = match cursor.read_u16::<BigEndian>() {
                Ok(len) => len as usize,
                Err(e) => {
                    log::warn!("Failed to read PUBLISH topic length: {:?}", e);
                    parsed_packet.header_len += header_len_fallback as u32;
                    return Ok(());
                }
            };
            let packet_id_bytes = if (mqtt_hdrflags & 0x06) >> 1 > 0 { 2 } else { 0 };
            2 + topic_len + packet_id_bytes
        }
        4..=11 => 2,  // PUBACK, PUBREC, PUBREL, PUBCOMP, SUBSCRIBE, SUBACK, UNSUBSCRIBE, UNSUBACK
        12..=14 => 0, // PINGREQ, PINGRESP, DISCONNECT
        15 => mqtt_len, // AUTH (MQTT 5.0, variable length)
        _ => {
            log::warn!("Unknown MQTT message type: {}", mqtt_msgtype);
            parsed_packet.header_len += header_len_fallback as u32;
            return Ok(());
        }
    };

    // Update header length safely
    parsed_packet.header_len += (header_len + variable_header_len) as u32;
    parsed_packet.proto_hierarchy.push_str("MQTT");
    
    Ok(())
}

fn mqtt_fixed_header_length(packet: &[u8]) -> Option<usize> {
    if packet.is_empty() {
        return None; // Empty packet
    }
    
    let mut length_bytes = 1; // 1 byte for control type
    let mut _multiplier = 1;
    // let mut value = 0;

    for i in 1..packet.len().min(5) { // Remaining Length field is at most 4 bytes
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
    cursor: &mut Cursor<&&[u8]>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut multiplier = 1;
    let mut value= 0;
    loop {
        let encoded_byte = cursor.read_u8()? as u32;
        value += (encoded_byte & 127) * multiplier;
        multiplier *= 128;
        if encoded_byte & 128 == 0 {
            break;
        }
    }
    Ok(value as usize)
}

fn parse_http(parsed_packet: &mut MaskedHeaderPayload) -> Result<(), Box<dyn std::error::Error>> {

    let payload = &parsed_packet.data[parsed_packet.header_len as usize..];
    let payload_str = String::from_utf8_lossy(&payload);

    let mut headers_end = 0;
    let mut byte_offset = 0;

    for line in payload_str.lines() {
        byte_offset += line.len() + 2; // Account for "\r\n"
        if line.is_empty() {
            headers_end = byte_offset;
            break;
        }
    }

    if headers_end == 0 {
        return Err("Failed to determine HTTP header length".into());
    }

    parsed_packet.header_len += headers_end as u32;
    Ok(())
}

impl ToRedisArgs for MaskedHeaderPayload {
    fn write_redis_args<W>(&self, out: &mut W)
    where
        W: ?Sized + RedisWrite,
    {
        let serialized = to_vec(self).expect("Serialization failed");
        out.write_arg(&serialized);
    }
}

impl FromRedisValue for MaskedHeaderPayload {
    fn from_redis_value(v: &Value) -> RedisResult<Self> {
        match *v {
            Value::BulkString(ref bytes) => from_slice(bytes).map_err(|e| {
                RedisError::from((
                    redis::ErrorKind::ParseError,
                    "Failed to deserialize MaskedHeaderPayload",
                    e.to_string(),
                ))
            }),
            _ => Err(RedisError::from((
                redis::ErrorKind::TypeError,
                "Expected binary data for MaskedHeaderPayload",
            ))),
        }
    }
}
// Csv handler mainly intended to write caputured features to disk
// line wise representation can also be used for simple inter process
// communication.
// Usage:
// Import with:
// use csv_handler::CsvHandler;
// Initiate the handler:
// let mut csv_handler = CsvHandler::new();
// Write the CSV to a file
// let _writer_ret = csv_handler.write_to_file("output.csv");
// add combined features struct (features) to in-memory Csv representation:
// let _return_handle = csv_handler.add_features(&features);

use crate::feature_parser::ProtocolFeatureSet;
use std::io::{Cursor, Write};

#[allow(dead_code)]
pub struct CsvHandler {
    buffer: Cursor<Vec<u8>>,
}

#[allow(dead_code)]
impl CsvHandler {
    pub fn new() -> Self {
        let mut handler = CsvHandler {
            buffer: Cursor::new(Vec::new()),
        };
        handler.write_header().expect("Failed to write CSV header");
        handler
    }

    fn write_header(&mut self) -> std::io::Result<()> {
        writeln!(self.buffer, "frame.time,ip.src_host,ip.dst_host,arp.dst.proto_ipv4,arp.opcode,arp.hw.size,arp.src.proto_ipv4,icmp.checksum,icmp.seq_le,icmp.transmit_timestamp,icmp.unused,http.file_data,http.content_length,http.request.uri.query,http.request.method,http.referer,http.request.full_uri,http.request.version,http.response,http.tls_port,tcp.ack,tcp.ack_raw,tcp.checksum,tcp.connection.fin,tcp.connection.rst,tcp.connection.syn,tcp.connection.synack,tcp.dstport,tcp.flags,tcp.flags.ack,tcp.len,tcp.options,tcp.payload,tcp.seq,tcp.srcport,udp.port,udp.stream,udp.time_delta,dns.qry.name,dns.qry.name.len,dns.qry.qu,dns.qry.type,dns.retransmission,dns.retransmit_request,dns.retransmit_request_in,mqtt.conack.flags,mqtt.conflag.cleansess,mqtt.conflags,mqtt.hdrflags,mqtt.len,mqtt.msg_decoded_as,mqtt.msg,mqtt.msgtype,mqtt.proto_len,mqtt.protoname,mqtt.topic,mqtt.topic_len,mqtt.ver,mbtcp.len,mbtcp.trans_id,mbtcp.unit_id,Attack_label,Attack_type")
    }

    pub fn add_features(&mut self, features: &ProtocolFeatureSet) -> std::io::Result<()> {
        writeln!(
            self.buffer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            "Not_in_Data",
            features.ip_features.src_host,
            features.ip_features.dst_host,
            features.arp_features.dst_proto_ipv4,
            features.arp_features.opcode,
            features.arp_features.hw_size,
            features.arp_features.src_proto_ipv4,
            features.icmp_features.icmp_checksum,
            features.icmp_features.icmp_seq_le,
            features.icmp_features.icmp_transmit_time,
            features.icmp_features.icmp_unused.iter().map(|&b| format!("{:02x}", b)).collect::<Vec<String>>().join(","),
            features.http_features.http_file_data,
            features.http_features.http_content_length,
            features.http_features.http_request_uri_query,
            features.http_features.http_request_method,
            features.http_features.http_referer,
            features.http_features.http_request_full_uri,
            features.http_features.http_request_version,
            features.http_features.http_response,
            0, // http.tls_port is not in the struct
            1, // tcp.ack is not in the struct, assuming 1 as default
            features.tcp_features.tcp_ack_raw,
            features.tcp_features.tcp_checksum,
            features.tcp_features.tcp_connection_fin,
            features.tcp_features.tcp_connection_rst,
            features.tcp_features.tcp_connection_syn,
            features.tcp_features.tcp_connection_synack,
            features.tcp_features.tcp_dstport,
            features.tcp_features.tcp_flags,
            1, // tcp.flags.ack is not in the struct, assuming 1 as default
            features.tcp_features.tcp_len,
            features.tcp_features.tcp_options.iter().map(|&b| format!("{:02x}", b)).collect::<Vec<String>>().join(","),
            features.tcp_features.tcp_payload.iter().map(|&b| format!("{:02x}", b)).collect::<Vec<String>>().join(""),
            features.tcp_features.tcp_seq,
            features.tcp_features.tcp_srcport,
            features.udp_features.udp_port_dst, // Assuming this is the same as udp.port
            features.udp_features.udp_stream_id,
            features.udp_features.udp_time_delta,
            features.DNS_features.dns_qry_name,
            features.DNS_features.dns_qry_name_len,
            features.DNS_features.dns_qry_qu,
            features.DNS_features.dns_qry_type,
            features.DNS_features.dns_retransmit,
            false, // dns.retransmit_request is not in the struct
            features.DNS_features.dns_retransmit_request_in.unwrap_or(0),
            features.mqtt_features.mqtt_conack_flags,
            features.mqtt_features.mqtt_conflag_cleansess,
            features.mqtt_features.mqtt_conflags,
            features.mqtt_features.mqtt_hdrflags,
            features.mqtt_features.mqtt_len,
            features.mqtt_features.mqtt_msg_decoded_as,
            features.mqtt_features.mqtt_msg.iter().map(|&b| format!("{:02x}", b)).collect::<Vec<String>>().join(""),
            features.mqtt_features.mqtt_msgtype,
            features.mqtt_features.mqtt_proto_len,
            features.mqtt_features.mqtt_protoname,
            features.mqtt_features.mqtt_topic,
            features.mqtt_features.mqtt_topic_len,
            features.mqtt_features.mqtt_ver,
            features.modbus_tcp_features.mbtcp_len,
            features.modbus_tcp_features.mbtcp_trans_id,
            features.modbus_tcp_features.mbtcp_unit_id,
            "", // Attack_label is not in the struct
            ""  // Attack_type is not in the struct
        )
    }

    pub fn write_to_file(&self, filename: &str) -> std::io::Result<()> {
        std::fs::write(filename, self.buffer.get_ref())
    }

    pub fn get_csv_content(&self) -> &[u8] {
        self.buffer.get_ref()
    }
}

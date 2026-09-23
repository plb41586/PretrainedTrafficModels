# 01 · Feature extraction (pcap → packet parquet)

**What it does:** parses every frame of a capture, assigns it to a bidirectional flow,
marks the bytes that identify endpoints, and writes one parquet row per packet.

**Entry point:** the Rust CLI in `feature_extraction/` (binary `feature_extractor`).

```
cargo build --release                                   # in feature_extraction/
cargo run --release -- (--file <pcap> | --interface <name>) \
    [--pl-outfile <path>] [--pl-chunk-size <n>] [--limit <n>] \
    [--graph-name <name>] [--topology <path>] [--falkor-host <host:port>] \
    [--cache-payloads] [--redis-host <host:port>]
```

This pipeline only ever runs `--file <pcap> --pl-outfile <parquet>`. `--limit N` stops
after N packets and makes a cheap smoke test. Without either `--file` or `--interface`
the CLI exits with code 2, because there is no default input.

**Consumes:** a `.pcap` file (or a live `--interface`).
**Produces:** a packet parquet ([artefacts § Packet parquet](../reference/artefacts.md#packet-parquet-stage-1--stage-2))
and a summary on stdout (packets, parse errors, unkeyed packets, flows).

## Where the logic lives

| Concern | File · symbol |
|---|---|
| CLI flags, capture loop, run statistics | `src/main.rs` · `Cli`, `run_capture`, `RunStats` |
| Parquet writer (bounded memory, one row group per `--pl-chunk-size` rows) | `src/main.rs` · `ParquetSink` |
| Layer-by-layer parsing, `proto_hierarchy`, `header_len`, endpoint `mask` | `src/feature_parser.rs` · `parse_packet`, `parse_ip_layer`, `parse_tcp_layer`, `get_udp_features`, `parse_mqtt`, `parse_modbus_tcp`, `parse_http`, `parse_dns` |
| Packet record | `src/feature_parser.rs` · `ParsedPacketSet`, `PayloadSet`, `ProtocolFeatureSet` |
| Flow identity and direction normalisation | `src/flow_tracker.rs` · `FlowKey::{from_parsed_packet, normalize}`, `FlowProtocol` |
| Per-flow statistics (only used by the graph backend) | `src/flow_tracker.rs` · `FlowTracker`, `FlowStats` |
| Optional backends | `src/redis_integration.rs`, `src/falkor_integration.rs`, `src/topology_loader.rs` |
| Python reader for the Redis payloads | `feature_extraction/pythonclient/client.py` |
| Batch job used for the current artefacts (not tracked in git) | `temp/extract_all.sh`: loops the CLI over every capture and writes `logs/summary.tsv` |

## How it works

- **Flow key.** A packet with an ARP, ICMPv4/v6, TCP or UDP tuple gets
  `FlowKey(src_ip, dst_ip, src_port, dst_port, protocol)`, which is then *normalised* so
  that the lower `(ip, port)` comes first. Both directions of a connection therefore share
  one key. The key uses only the transport protocol, never the full hierarchy, so
  packets on one connection that parse to different depths still end up in one flow. A
  packet without such a tuple is counted as `unkeyed` and skipped.
- **Mask.** See [data-representation § Packet → tokens](../reference/data-representation.md#packet--tokens).
  The Rust side marks the bytes, and the Python side replaces them.
- **Backends are opt-in.** FalkorDB is contacted only with `--graph-name`, Redis only with
  `--cache-payloads`, and parquet is written only with `--pl-outfile`. This pipeline uses
  the parquet path alone.

`feature_extraction/MIGRATION.md` covers the full CLI and output changes from the older
extractors. `cargo test` runs the extractor's unit tests; they need no database.

## Invariants

- The stored frame is the whole frame, starting at the Ethernet header. Nothing is
  truncated at this stage.
- The `flow_key` string format is part of the lineage: re-extracting a capture changes
  it, and everything downstream has to be regenerated
  ([artefacts § Lineage](../reference/artefacts.md#lineage-and-provenance-checks)).

## Reuse

The packet parquet is independent of the models. Any byte-level or flow-level model can
read it: group by `flow_key`, order by `timestamp_s, timestamp_us`, and use `mask`
to hide endpoint identity. For flow statistics instead of bytes, the FalkorDB path
exports per-flow counters (`FlowStats`).

## Known gaps

These are tracked in `TODO.md`. Check them first when a new capture looks wrong:

- **VLAN, MPLS, PPPoE and Profinet frames are dropped** as parse errors (only IPv4, IPv6
  and ARP EtherTypes are handled). On a VLAN-tagged capture this can silently remove a
  large share of the traffic, so compare `parse errors` against `packets read` in the log.
- **Frames of 1520 bytes or more** are written to the parquet, but downstream they cannot
  be tokenised ([data-representation](../reference/data-representation.md#constants)).
  Captures made with TSO/GRO offload or jumbo frames contain such frames.
- **Flows never expire.** The flow table grows for the whole capture. This does not
  matter for pcaps, but it limits long `--interface` runs.

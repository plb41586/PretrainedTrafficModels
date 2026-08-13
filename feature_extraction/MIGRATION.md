# Migrating downstream code from version1 / version2 to the merged extractor

This describes every change that is **visible outside the binary** — command line,
Redis queues, FalkorDB graph, Parquet files — so that the code and scripts consuming
this extractor can be updated before `version1/` and `version2/` are retired.

Purely internal refactoring is listed only in the last section, for anyone who
vendors or imports the source rather than running the binary.

Throughout: **v1** = `version1/`, **v2** = `version2/`, **merged** = this tree.

---

## 0. TL;DR — what will break

| Consumer | Breaks? | Why |
|---|---|---|
| Shell scripts / compose files invoking the binary | **Yes** | Positional args gone (v1); topology and hosts are now explicit flags |
| Python reading Redis payload queues | **Yes** | Queue *names* changed format and content; queue count drops |
| Python decoding the msgpack payloads | No | `PayloadSet` and `ProtocolFeatureSet` field sets are unchanged |
| Cypher queries against the graph | **Yes** | Relationship type changed from the protocol hierarchy to the transport token |
| Parquet readers | Mostly no | Schema identical; `flow_key` contents changed |
| Anything comparing flow counts across runs | **Yes** | One connection is now one flow |

---

## 1. Command line

### v1 (positional)

```
feature_extractor [FILE] [CACHE_PAYLOADS] [GRAPH_NAME]
```

Defaults: file `/workspace/data/Network_Traffic/Phase1/1stPhase-timed-Merged.pcap`,
cache_payloads `false`, graph name `Phase2`. Redis and FalkorDB were **always**
contacted, and the topology TOML was **always** loaded from a hardcoded path.

### v2 (flags)

```
feature_extractor --file <PATH> [--cache-payloads] [--graph-name <NAME>] [--pl-outfile <PATH>]
```

Default file `/home/plb41586/workspace/data/CICAPT-IIoT/Network_Traffic/Phase1/1stPhase-timed-Merged.pcap`.
Topology loaded from the same hardcoded path whenever `--graph-name` was set.

### merged

```
feature_extractor (--file <PATH> | --interface <NAME>)
                  [--graph-name <NAME>] [--cache-payloads] [--pl-outfile <PATH>]
                  [--topology <PATH>]
                  [--falkor-host <HOST:PORT>] [--redis-host <HOST:PORT>]
                  [--pl-chunk-size <N>] [--limit <N>]
```

### Rewriting invocations

| Old | New |
|---|---|
| `feature_extractor cap.pcap` | `feature_extractor --file cap.pcap --graph-name Phase2 --topology /workspace/data/Network_Traffic/Devices.toml --falkor-host falkordbLMdefender:6379` |
| `feature_extractor cap.pcap true MyGraph` | `feature_extractor --file cap.pcap --cache-payloads --graph-name MyGraph --topology /workspace/data/Network_Traffic/Devices.toml --falkor-host falkordbLMdefender:6379 --redis-host falkordbLMdefender:6379` |
| `feature_extractor --file cap.pcap --graph-name G` (v2) | `feature_extractor --file cap.pcap --graph-name G --topology /workspace/data/Network_Traffic/Devices.toml` |

### Four behaviour changes to watch for

1. **No default input.** Omitting both `--file` and `--interface` exits with status 2
   instead of silently reading a hardcoded path. Any script that relied on the default
   must now name the file.

2. **No default graph name.** v1 defaulted to `Phase2` and always wrote to FalkorDB.
   In merged, **omitting `--graph-name` means nothing is written to FalkorDB at all**
   — the run parses and reports, then exits. This is the change most likely to look
   like "the tool silently stopped working". Pass `--graph-name` explicitly.

3. **Topology is opt-in.** v1 and v2 loaded `Devices.toml` from a hardcoded path.
   merged loads it only when `--topology` is given. If your graph depends on the
   topology nodes existing, add the flag.

4. **Hosts are flags, and the defaults changed.** They default to `falkordb:6379`
   and `redis:6379` (v2's names). If you are on v1's single-host deployment, pass
   `--falkor-host falkordbLMdefender:6379 --redis-host falkordbLMdefender:6379`,
   or set the defaults in `src/main.rs` to match your environment.

`--limit N` stops after N packets (0 = no limit) and is the cheapest smoke test.
`--interface NAME` captures live instead of from a file.

---

## 2. Redis queues

### Queue names changed — this is the biggest downstream break

Both forks used the **as-observed, un-normalized** flow key, with the **full protocol
hierarchy** in the parentheses:

```
192.168.1.10:40000 -> 192.168.1.20:1883 (Ethernet->IPv4->TCP)
192.168.1.20:1883 -> 192.168.1.10:40000 (Ethernet->IPv4->TCP->MQTT)
```

merged uses the **normalized** key with the **transport token**:

```
192.168.1.10:40000 -> 192.168.1.20:1883 (TCP)
```

The format string is unchanged — `"{src_ip}:{src_port} -> {dst_ip}:{dst_port} ({protocol})"` —
but two things about its contents changed:

- **Direction is normalized.** Packets travelling in either direction of one flow now
  land in the *same* list. Previously they were split across two differently-named lists.
- **The parenthesised field is now one of** `ARP`, `ICMPv4`, `ICMPv6`, `TCP`, `UDP`,
  never a `->`-joined hierarchy.

Consequence: a consumer that enumerates queues (`SCAN`/`KEYS`) and parses the names must
be updated. The number of distinct queues drops substantially — both because the two
directions merge and because app-layer variation no longer forks the name. Per-flow
message volume rises correspondingly.

The full hierarchy is still available per packet, as `proto_hierarchy` inside the
`PayloadSet` payload.

### Feature queues are back

v1 pushed a second list per flow, `"{flow_identifier}features"`, carrying
`ProtocolFeatureSet`. **v2 removed this.** merged restores it, so a v2-based
consumer will start seeing `...features` lists appear. If you do not want them,
the push is a single call in `run_capture` (`src/main.rs`).

Note the suffix is appended with no separator, so the feature queue for the example
above is `192.168.1.10:40000 -> 192.168.1.20:1883 (TCP)features`.

### Message payloads are unchanged

Both payload types are still msgpack (`rmp-serde`) with identical field sets:

- `PayloadSet` — `data` (bytes), `mask` (array of bool, one per byte of `data`),
  `header_len` (u32), `proto_hierarchy` (string). **No change.**
- `ProtocolFeatureSet` — all fields as before, including the non-snake-case
  `DNS_features` key, which is deliberately preserved because renaming it would
  break the wire format.

**One value-level change:** `transport_protocol` and `application_protocol` were
never assigned in v2 and always decoded as `Unknown`. They are populated correctly
in merged (as they were in v1). If your v2 consumer works around those fields being
useless, it can now trust them.

### Redis failures no longer abort the run

The pushes returned `()` and panicked on error in both forks. They now return
`RedisResult<()>`; failures are counted, logged to stderr, and the capture continues.
The count is reported at the end as `redis push errors`. A consumer should no longer
assume that a completed run means every message was delivered — check that line.

---

## 3. FalkorDB graph

### Relationship type changed

Both forks used the full protocol hierarchy as the relationship type:

```cypher
MATCH ()-[r:`Ethernet->IPv4->TCP->MQTT`]->() RETURN r
```

merged uses the transport token, with the hierarchy demoted to a property:

```cypher
MATCH ()-[r:TCP]->() WHERE r.proto_hierarchy ENDS WITH '->MQTT' RETURN r
```

Rewrite rules for existing queries:

| Old pattern | New pattern |
|---|---|
| `` -[r:`Ethernet->IPv4->TCP`]-> `` | `-[r:TCP]->` |
| `` -[r:`Ethernet->IPv4->UDP->DNS`]-> `` | `-[r:UDP]-> WHERE r.proto_hierarchy ENDS WITH '->DNS'` |
| `` -[r:`Ethernet->ARP`]-> `` | `-[r:ARP]->` |
| `type(r) STARTS WITH 'Ethernet'` | `r.proto_hierarchy STARTS WITH 'Ethernet'` |

There are now exactly five relationship types: `ARP`, `ICMPv4`, `ICMPv6`, `TCP`, `UDP`.

### New edge properties

- `proto_hierarchy` (string) — new on every edge. Holds the richest hierarchy observed
  on that flow, e.g. `Ethernet->IPv4->TCP->MQTT`. Where packets on one flow parsed to
  different depths, the deepest wins.
- Six entropy properties, written by v1 but **not by v2**:
  `fwd_packet_entropy_agg`, `fwd_min_packet_entropy`, `fwd_max_packet_entropy`,
  `bwd_packet_entropy_agg`, `bwd_min_packet_entropy`, `bwd_max_packet_entropy`.
  The `_agg` values are running **sums**, not means — divide by the matching
  direction's packet count to get a mean. Entropy is measured over the whole captured
  frame, headers included.

Everything else (`src_port`, `dst_port`, `fwd_/bwd_packet_count`, `fwd_/bwd_byte_count`,
`first_seen`, `last_seen`, `duration_ms`, `min_/max_packet_size`, `avg_packet_size`,
`avg_pps`, `avg_bps`) keeps its name and meaning.

### Node schema unchanged, but v2 wrote fewer edges than it should have

`:IPv4` / `:IPv6` nodes with `address` and `type` (`internal`/`external`) are unchanged.

However v2 derived the node label from the *source* address and applied it to both
endpoints, so any flow between an IPv4 and an IPv6 host failed to match and its edge
was never created. Those edges now appear. A v2-derived graph is missing them; a
re-run against a fresh graph is the only way to recover them.

### Writes are now idempotent — but do not mix conventions

Both forks used `CREATE` for edges, so re-running over the same capture and graph
duplicated every relationship. merged uses `MERGE` on
(source, destination, relationship type, `src_port`, `dst_port`) followed by `SET`.

**Use a fresh graph name for the first merged run.** Pointing merged at a graph that
already contains fork-written edges will not clean up:

- old edges carry hierarchy-shaped relationship types that merged never matches, so
  they linger as stale duplicates alongside the new `:TCP` edges;
- if a previous `CREATE` run left genuine duplicates, `MERGE` binds one arbitrarily
  and updates it, leaving the rest stale.

### Flow counts will change

Independent of any bug fix, the *definition* of a flow changed:

- The protocol hierarchy left the flow key, so a connection whose packets parsed to
  different depths (a bare TCP ACK vs. an MQTT publish on the same socket) is now
  **one** flow rather than several. Expect **fewer edges with higher per-edge counters**.
- ICMPv4 and ICMPv6 previously shared a key when their hierarchies matched; they are
  now always distinct.

Do not compare edge counts or per-edge statistics across the version boundary. Any
downstream model trained on fork-era flow features should be re-fitted.

---

## 4. Parquet export

### Schema is unchanged from v2

| Column | Type |
|---|---|
| `proto_hierarchy` | String |
| `flow_key` | String |
| `timestamp_s` | Int64 |
| `timestamp_us` | Int64 |
| `data` | Binary |
| `mask` | Binary |
| `header_len` | UInt32 |

Existing readers keep working without change. (`data` and `mask` were already written
as `Binary` in v2 — polars maps `Vec<Vec<u8>>` to `Binary`, not to a list of integers.)

`mask` is still one byte per byte of `data` — `0x00` or `0x01`, not a packed bitmap —
so `len(mask) == len(data)` still holds.

### What did change

- **`flow_key` contents.** Same change as the Redis queue names: normalized direction,
  transport token instead of the hierarchy. This is now the same string used for the
  Redis queue and derivable from the graph edge, so the three can finally be joined.
  Code that parsed the old hierarchy out of `flow_key` should read the
  `proto_hierarchy` column instead.
- **Row groups.** The file is written incrementally, one row group per
  `--pl-chunk-size` packets (default 50,000), instead of one giant row group at the end.
  `pl.read_parquet` / `pd.read_parquet` are unaffected. This is what makes the export
  usable on large captures at all — the fork buffered every packet in memory until EOF.
- **The file is valid only after a clean exit.** As before, but worth restating now
  that runs can be long: a killed process leaves an unfinalized file.

---

## 5. Environment and build

- **rustc ≥ 1.88** is required and pinned via `rust-version` in `Cargo.toml`.
  Neither fork could resolve dependencies on rustc 1.87.
- **`Cargo.lock` is committed.** Neither fork had one, which is why their builds
  drifted onto incompatible crate versions. Do not delete it; update deliberately.
- **`falkordb` is `0.10.1`** (v1's pin). v2's `0.1.10` was assumed to be a
  transposition — confirm if that assumption is wrong.
- **New dependencies relative to v1:** `clap` 4, `polars` 0.53.
  **New relative to v2:** `entropy` 0.4.3.
- CI or images that pinned an older toolchain need bumping.

---

## 6. Internal API changes

Only relevant if you vendor, fork, or import this source. The binary's behaviour is
covered above.

### Removed files

- `feature_parser_light.rs` (v2) — was never declared as a module and imported `log`,
  which was never a dependency, so it had never compiled. Deleted.
- `csv_handler.rs` (both) — byte-identical in both forks and commented out of the
  module tree in both. Deleted.

### `flow_tracker`

| Before | After |
|---|---|
| `FlowKey.protocol: String` | `FlowKey.protocol: FlowProtocol` (new enum) |
| `FlowKey::new(.., protocol: String)` | `FlowKey::new(.., protocol: FlowProtocol)` |
| `FlowKey::to_string()` (v2 inherent method) | `Display` impl (v1 already had this) |
| `is_normalized()`, `normalized_key()`, `new_normalized()` (v2) | removed — use `is_forward()` / `normalize()` |
| `FlowStats::new(is_forward, u16, ts, f32)` | `FlowStats::new(is_forward, u32, ts, f32, proto_hierarchy: String)` |
| `FlowStats::update(is_forward, u16, ts, f32)` | `FlowStats::update(is_forward, u32, ts, f32, proto_hierarchy: &str)` |
| `FlowStats.duration/avg_packet_size/avg_pps/avg_bps` fields (v2) | methods, computed on demand |
| `min_/max_packet_size: u16` | `u32` |
| `FlowTracker::new(capacity, timeout_secs)` | `FlowTracker::new(capacity)` |
| `FlowTracker` held a `FalkorSyncClient` (v1) | holds none — constructing it does no I/O |
| `push_flows_to_falkor(&self, graph_name)` | `push_flows_to_falkor(&self, graph_name, &FalkorSyncClient) -> (usize, usize)` |
| `flow_timeout` field | removed — it was never read in either fork |
| `get_flow_stats_mut()` | removed — unused |

`FlowStats` gained `proto_hierarchy: String`.

### `falkor_integration`

| Before | After |
|---|---|
| `connect_to_falkor()` | `connect_to_falkor(host: &str)` |
| `insert_edge_only()` / `update_existing_edge()` / `insert_from_flow_tracker()` | one `insert_flow_edge()` using `MERGE` + `SET` |
| `basic_falkor_test()` (v2) | removed |
| `EDGE_PROPERTY_BLOCK` const | removed — `EDGE_SET_BLOCK` covers both paths |

`validate_protocol` now guards only the relationship type, which comes from the
`FlowProtocol` enum, so its accepted charset is narrower (`[A-Za-z0-9_]`). This is
safe because the free-form hierarchy no longer passes through it — it travels as a
query parameter, where spaces and parentheses are harmless.

### `redis_integration`

| Before | After |
|---|---|
| `connect_to_redis()` | `connect_to_redis(host: &str)` |
| pushes return `()` and panic on error | return `redis::RedisResult<()>` |
| `push_featureset_to_redis_queue` absent in v2 | present |

### `setup_capture`

`setup_capture_from_file(&str)` → `setup_capture_from_file(&Path)`.
`setup_capture(interface_name)` is now reachable from the CLI via `--interface`;
it was dead code in both forks.

### `feature_parser`

- `ParsedPacketSet` gained `flow_key: String`, populated by the capture loop with the
  normalized key. This changes the msgpack shape of `ParsedPacketSet` itself — nothing
  in-tree serializes that type (only `PayloadSet` and `ProtocolFeatureSet` are pushed),
  but check any out-of-tree consumer that does.
- `transport_protocol` / `application_protocol` assignments restored (v2 had dropped
  them, leaving both permanently `Unknown`).
- No other struct or field changes; the parsing logic itself is untouched.

---

## 7. Suggested rollout

1. Build and run the tests: `cargo test` — 15 tests, no database needed.
2. Smoke-test on a real capture with **no backend flags**:
   `feature_extractor --file <cap>.pcap --limit 10000`.
   This exercises the parser and flow tracker without touching Redis or FalkorDB.
   Check the reported `parse errors` and `packets without key` counts look sane.
3. Point it at a **fresh** graph name and compare against a fork-written graph on the
   same capture. Expect fewer edges with larger counters — see §3.
4. Update Cypher queries per §3, then the Redis consumer per §2.
5. Re-run the Parquet export and confirm the reader still loads it (§4); update any
   code that parsed the old `flow_key` format.
6. Retire `version1/` and `version2/`.

## 8. Known gaps

- **Not yet run against a real capture.** The merged tree compiles warning-free and
  its unit tests pass, but it has not been executed end to end on a pcap. Step 2 above
  is the outstanding verification.
- **VLAN, MPLS, PPPoE and Profinet frames are still dropped**, as they were in both
  forks: `parse_packet` returns `EthertypeNotImplemented` for any EtherType other than
  IPv4, IPv6 and ARP, and the frame is counted as a parse error. On VLAN-segmented
  industrial captures this can remove a large share of traffic. Fixing it means
  continuing past the tag to the inner header — a parser change, deliberately not
  bundled into this merge.
- **Flows never expire.** The unused `flow_timeout` was removed rather than
  implemented; the flow table grows for the length of the capture. Fine for
  batch pcap processing, a real constraint for long-running `--interface` capture.

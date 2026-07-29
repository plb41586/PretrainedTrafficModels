# TODO

Cross-cutting work items that don't belong to a single script.

## Rework the feature extractor's flow key (`feature_extraction/`)

The `flow_key` string written into the packet parquets is not usable as a flow identity as-is.
`data_tools/SplitFlowsDF.py` currently works around this by deriving a canonical conversation
key in Python; fixing the items below makes that workaround redundant (it collapses to a plain
`group_by("flow_key")`), but it requires re-extracting every pcap.

- [ ] **Write the normalized key.** `src/main.rs:104-105` builds the key with
      `FlowKey::from_parsed_packet(...)` — which ends in `Self::new(...)`
      (`src/flow_tracker.rs:309`), not `Self::new_normalized(...)` — and stringifies it *before*
      `key.normalize()` on line 107. That `normalize()` only feeds the `FlowTracker` HashMap, so
      the dumped key stays directional: `A:59573 -> B:1883 (…)` and `B:1883 -> A:59573 (…)` are
      two keys for one conversation. Normalize before `to_string()`.
- [ ] **Stop putting the whole proto_hierarchy in the key.** `FlowKey::to_string`
      (`src/flow_tracker.rs:24`) prints `self.protocol`, and `from_parsed_packet` sets that to
      `payload_set.proto_hierarchy.clone()`. So one TCP connection fragments into
      `… (Ethernet->IPv4->TCP)` for bare ACKs/handshake and `… (Ethernet->IPv4->TCP->MQTT)` for
      payload-bearing packets. Use the L4 protocol only — the full hierarchy is already its own
      `proto_hierarchy` column.
- [ ] **Fix or delete `FlowKey::normalized_key()`** (`src/flow_tracker.rs:31`). Its branch is
      inverted relative to `normalize()` (`:322`) — it returns `self` unchanged on the
      *already-normalized* path while printing `"Check for normalization before use!"`, and
      swaps otherwise. Two functions, same job, disagreeing.
- [ ] **`src/main.rs:104` `.unwrap()`s `from_parsed_packet`**, which returns `None` for anything
      that is not ARP/ICMP/TCP/UDP — a single such packet panics the whole extraction run.

## Documentation

- [ ] `CLAUDE.md` documents `AttackLabel` and `FlowID` columns under "Data conventions", but no
      parquet in `data_artefacts/` has either. Attack labelling is by file name
      (`attacks/<Class>.parquet`) and the flow identifier is the `flow_key` string. The
      `TrainingDatasetHandler` / `ValidationDatasetHandler` classes in `DataUtils.py` that rely
      on those columns are dead against the current artefacts.

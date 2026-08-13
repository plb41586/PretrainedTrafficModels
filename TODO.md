# TODO

Cross-cutting work items that don't belong to a single script.

## Re-extract the packet parquets with the merged extractor (`feature_extraction/`)

The flow-key defects listed here previously — directional keys, the whole `proto_hierarchy`
baked into the key, the disagreeing `normalized_key()`, and the `.unwrap()` that panicked the
run on the first non-ARP/ICMP/TCP/UDP packet — are all **fixed** by the merged extractor
(see `feature_extraction/MIGRATION.md`). What remains is the data-side follow-through.

- [ ] **Re-extract the pcaps.** Everything under `data_artefacts/` was written by the pre-merge
      extractor and still carries old-format keys. The files are internally consistent, so this
      is not urgent — `SplitFlowsDF`'s Python canonicalization already recovers the right
      conversations from them.
- [ ] **Regenerate splits and latent caches per capture, together.** Re-extracting changes the
      `flow_key` strings, so a re-extracted parquet must not be paired with a split or a
      `latents_*` cache built from the old one.
- [ ] **Then simplify `SplitFlowsDF.py`.** Once no pre-merge parquet is still in use,
      `conversation_key_map` / `transport_prefix` / `FLOW_KEY_RE` collapse to a plain
      `group_by("flow_key")`. Until then they are a correct no-op on new keys, so leaving them
      in costs only a little work per run.

## Known gaps carried over from the merge

- [ ] **VLAN, MPLS, PPPoE and Profinet frames are dropped** — `parse_packet` returns
      `EthertypeNotImplemented` for any EtherType but IPv4, IPv6 and ARP, and counts the frame
      as a parse error. On VLAN-segmented captures this can silently remove a large share of
      traffic; watch the reported `parse errors` count.
- [ ] **Flows never expire** — the flow table grows for the length of the capture. Fine for
      batch pcap processing, a real constraint for long-running `--interface` capture.

## Documentation

- [ ] `CLAUDE.md` documents `AttackLabel` and `FlowID` columns under "Data conventions", but no
      parquet in `data_artefacts/` has either. Attack labelling is by file name
      (`attacks/<Class>.parquet`) and the flow identifier is the `flow_key` string. The
      `TrainingDatasetHandler` / `ValidationDatasetHandler` classes in `DataUtils.py` that rely
      on those columns are dead against the current artefacts.

# TODO

Cross-cutting work items that don't belong to a single script.

## Re-extract the packet parquets with the merged extractor (`feature_extraction/`)

The flow-key defects listed here previously — directional keys, the whole `proto_hierarchy`
baked into the key, the disagreeing `normalized_key()`, and the `.unwrap()` that panicked the
run on the first non-ARP/ICMP/TCP/UDP packet — are all **fixed** by the merged extractor
(see `feature_extraction/MIGRATION.md`). What remains is the data-side follow-through.

- [x] **Re-extract the pcaps.** Everything under `data_artefacts/` was written by the pre-merge
      extractor and still carries old-format keys. The files are internally consistent, so this
      is not urgent — `SplitFlowsDF`'s Python canonicalization already recovers the right
      conversations from them.
- [x] **Retrain on the Packet level.** Both packet-level models were retrained on the new
      `flow_split` at `EncoderDim=128`, 2 epochs each, one per GPU concurrently:
      `TrainingOutputs/PacketAE_IIoTset_d128` (test CE 0.1298, non-pad byte acc 0.5956 — clears
      the old run's 0.1379 / 0.5539 plateau) and `TrainingOutputs/PacketMLM_IIoTset_d128`
      (masked-token acc 0.3886; `_best.ckpt` is epoch 0, epoch 1 overfits). Shared config now
      lives in `RawByteTrafficModelling/PreTraining/RunConfig.py`; `PacketLevelMLM.py` moved off
      the deleted CICAPT `Phase1_split` onto the same IIoTset `flow_split` as the AE.
      Open follow-ups from those runs: the AE is *underfitting* (train non-pad 0.604 vs test
      0.596, still improving when the schedule ran out) so the next lever is capacity —
      `dim=256` and/or more layers; the MLM is the opposite and wants regularisation, not
      capacity. The proto-hierarchy aux head ends at loss 1.8e-5 / accuracy 1.0000, i.e. it
      contributes no gradient and shapes nothing — drop it or replace it with a non-trivial task.
- [x] **Regenerate splits and latent caches per capture, together.** Re-extracting changes the
      `flow_key` strings, so a re-extracted parquet must not be paired with a split or a
      `latents_*` cache built from the old one. Done for IIoTset-Ferrag:
      `flow_split/latents_PacketAE_d128_best/{train,test}`, 128-dim, built from
      `PacketAE_IIoTset_d128_best` (sha256 pinned in each `meta.json`) and verified end to end
      with `CheckLatentCache.py` on both splits. Every downstream script was repointed at the
      merged-extractor tree in the same pass.
- [ ] **Then simplify `SplitFlowsDF.py`.** Once no pre-merge parquet is still in use,
      `conversation_key_map` / `transport_prefix` / `FLOW_KEY_RE` collapse to a plain
      `group_by("flow_key")`. Until then they are a correct no-op on new keys, so leaving them
      in costs only a little work per run.
      **Now unblocked:** as of the repointing above, no script references
      `data_artefacts/deprecated_*` any more — every consumer reads the merged-extractor tree.
      The canonicalization is a pure no-op on these keys, so it can be collapsed whenever the
      deprecated parquets are deleted for good.

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

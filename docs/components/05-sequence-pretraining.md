# 05 · Sequence-level pretraining (packet latents → flow encoder)

**What it does:** trains a `Sequence_Encoder` that compresses a window of up to 64 packets
into one flow vector `z`, the bottleneck of a `SequenceAutoencoder`. Stage 6 exports `z`
as the flow embedding.

**Entry points** (in `RawByteTrafficModelling/PreTraining/`):

| Script | Stage | Starts from |
|---|---|---|
| `SequenceLevelAutoEncoder.py` | 1: reconstruction only | random init + frozen packet encoder |
| `InspectFlowGroups.py` | before stage 2: read-only group profile | latent caches |
| `SequenceAEStructureFinetune.py` | 2 (optional): reconstruction + supervised contrastive | stage-1 `_best` |

**Consumes:** the `train`/`test` latent caches and the packet-AE checkpoint that built them.
The byte-level eval also reads `DATASET.test` (raw bytes).
**Produces:** `TrainingOutputs/<RUN_NAME>/SequenceLevelAutoEncoder_<RUN_NAME>_{untrained,E<n>,best}.ckpt`.
Stage 2 keeps the same file prefix, so stage 6 only needs `RUN_NAME` changed to use it.

## Where the logic lives

| Concern | File · symbol |
|---|---|
| Models | `ModelDefinitions.py::Sequence_Encoder`, `SequenceDecoder`, `SequenceAutoencoder` ([model-library § Sequence level](../reference/model-library.md#sequence-level)) |
| Batches | `DataUtils.py::CachedLatentSequenceHandler` ([data-representation § Flow → windows](../reference/data-representation.md#flow--windows)) |
| Loss | `SequenceAutoencoder.loss`: masked MSE (normalised) + `length_loss_weight` × length CE |
| Baselines | `ModelDefinitions.py::baseline_mses` |
| Byte-level eval set (alignment checks + packet-AE ceiling) | `SeqByteEval.py::build_byte_eval_set`, used by stage 2. Stage 1 still has the same code inline |
| Byte accuracy | `ModelDefinitions.py::byte_level_reconstruction`, `decoder_byte_accuracy` |
| Grouping for stage 2 | `DataUtils.py::parse_flow_key`, `flow_group_labels`, `flow_group_ids`, `group_support`; `CachedLatentSequenceHandler.epoch_grouped_flow_batches` |
| Contrastive loss and diagnostics | `StructureLosses.py::supcon`, `effective_rank`, `group_distance_stats`, `knn_group_agreement` |

## Stage 1: reconstruction

- **Setup:** the packet encoder comes from `PACKET_AE_CKPT` and stays frozen. The sequence
  encoder and decoder have width `SEQ_DIM` (the size of `z`, which is what the width sweep
  varies). `P = PACKETS_PER_SEQUENCE = 65`.
- **Normalisation:** target mean and std are computed once from a large random sample of
  train latents and stored in the checkpoint (`set_target_stats`). Every MSE is measured
  in that normalised space.
- **Each epoch:** one random window per train flow, followed by a deterministic pass over
  every `test` window.
- **Per-epoch checks** (all written to `metrics.csv` / `curves.png`):
  - Reconstruction MSE against the two learn-nothing baselines.
  - Length-head accuracy.
  - **Byte accuracy:** the reconstructed latents are decoded by the frozen packet decoder
    and compared against the packet AE's own accuracy on the same packets (the ceiling).
    Unlike the normalised MSE, this number is interpretable.
- **Selection:** `_best` = lowest test total loss.
- **Step budget:** an epoch has `ceil(#train flows / batch_size)` steps, so the number of
  steps depends on the capture. Adjust `Epochs` for each dataset, as the script's own
  comment does for CICAPT.

## Stage 2: structure fine-tune

Stage 1 says nothing about where a flow should lie relative to other flows, and in
practice `z` is shaped mostly by flow length. Stage 2 adds a supervised contrastive term
(`supcon`, on L2-normalised `z`) that pulls together flows sharing an endpoint group.
Endpoint bytes are masked at the input, so this supervision is information the model
cannot read off the bytes: it has to learn a behavioural fingerprint of the device.

- **Before running:** use `InspectFlowGroups.py` to choose `COARSE_KEY` (`endpoint_pair` /
  `host_lo` / `host_hi`) and to check group count, the largest group, the share of groups
  with at least 8 flows, and train→test coverage. The grouping has only one level because
  IIoTset has too little protocol variety for a second one. A two-level version is a
  small change for a capture that has the variety (see the `StructureLosses.py`
  docstring).
- **Collapse prevention:** reconstruction and the length head stay in the loss at full
  weight, and `struct_eff_rank` in the metrics is the collapse alarm. `LAMBDA_STRUCT` is
  the knob to sweep.
- **Batches:** built with `MAX_GROUP_RUN` so that each batch contains enough distinct groups
  to provide negatives. `BALANCE_GROUPS` gives each group equal weight instead of each
  anchor.
- **Metrics:** reported pooled, for supported groups only (`*_sup`), and per `seq_len` bin.
- **Current configuration:** IIoTset-specific. `STAGE1_RUN`, `DATASET`, `CACHE_TAG` and
  `PACKET_AE_CKPT` all point at IIoTset runs.

## Reuse

- **Flow embeddings:** `load_SeqAE_checkpoint` → `SequenceAutoencoder` →
  `.encode(seq_lens, latents=…)` or `.encode(seq_lens, tokens=…)`. The token path needs
  no cache, since the packet encoder is inside the checkpoint.
- **Supervised flow classification / fine-tuning:** use `model.encoder` (a
  `Sequence_Encoder`) as the backbone, add a head on `z`, and call
  `unfreeze_packet_encoder()` to fine-tune end to end.
- **Other structure objectives:** `supcon` works on any `(B, D)` embeddings and integer
  labels, so a different grouping (a device type, an application protocol) only needs a
  different label vector.

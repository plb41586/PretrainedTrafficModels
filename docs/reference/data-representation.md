# Data representation

How a packet row becomes a token sequence, and how a flow becomes a batch of packet
windows. Stages 3–6 all depend on this, and none of them may change it on its own.

## Packet → tokens

| Step | Where | What happens |
|---|---|---|
| Endpoint mask | Rust: `feature_parser.rs::parse_packet`, `parse_ip_layer` | Marks the bytes that identify an endpoint: Ethernet MACs `[0,12)`, IPv4 addresses `[26,34)`, IPv6 addresses `[22,54)`, ARP sender/target hardware and protocol addresses `[22,42)`. Ports are not masked |
| Mask application | `DataUtils.py::PreTrainingDatasetHandler.apply_mask` (via `get_pretraining_data`) | Replaces each masked byte with `<EndPointMasking>`, so no model ever sees an address |
| Token IDs | `DataUtils.py::ID_Encoder.construct_input_ids` | Writes bytes 0..255 as they are, then `<CLS>` directly after the last byte, then `<pad>` up to 1520 positions |
| Pooling | `ModelDefinitions.py::DynamicCLSPooling` | Finds the `<CLS>` position in each sequence and takes its hidden state as the packet latent |

The offsets are hardcoded and assume an untagged Ethernet frame. That assumption holds
for everything the extractor currently writes, because VLAN, MPLS and PPPoE frames are
dropped ([01 § Known gaps](../components/01-feature-extraction.md#known-gaps)).

### Constants

These are defined once, in `PreTraining/RunConfig.py`. Build encoders and params through
`make_id_encoder()` and `packet_encoder_params()` rather than hand-copying the constants.

| Constant | Value | Note |
|---|---|---|
| `SPECIAL_IDS` | `<pad>`=256, `</s>`=257, `<CLS>`=258, `<mask>`=259, `<EndPointMasking>`=260, `<BOS>`=261 | `</s>` is emitted only with SOS placement |
| `VOCAB_SIZE` | 262 | |
| `PACKET_ID_LEN` | 1520 | also hardcoded inside `ID_Encoder` |
| `CLS_PLACEMENT` | `"EOS"` | Every cache and checkpoint depends on this. A latent cache built with one placement cannot be read by a model trained on the other |

Packets of 1520 bytes or more do not fit (CLS needs one slot after the data) and
`construct_input_ids` raises on them. Untagged Ethernet frames without FCS are at most
1514 bytes, but captures taken with segmentation offload (TSO/GRO) or jumbo frames can
contain larger ones. See [01 § Known gaps](../components/01-feature-extraction.md#known-gaps).

## Flow index

`PreTrainingDatasetHandler.build_flow_index()` groups a split parquet by `flow_key` and
returns one row per flow, with the parquet row indices of its packets in timestamp order.
Flows are sorted by `flow_key`.

This index defines **cache row order** ([artefacts § latent cache](artefacts.md#packet-latent-cache-stage-4--stages-5-6)).
It is deterministic for a given parquet, which is why the token path (encoding from
bytes) and the cache path produce identical windows. The alignment checks in
[artefacts § Lineage](artefacts.md#lineage-and-provenance-checks) depend on this.

## Flow → windows

Constant: `PACKETS_PER_SEQUENCE = P = 65` in each sequence-level script, which gives
`seq_len = P − 1 = 64` real packets per window. The extra slot holds the sequence-level
CLS token that `Sequence_Encoder` writes at index `seq_len`
([model-library](model-library.md#sequence-level)).

| Use | Where | Windowing |
|---|---|---|
| Training | `CachedLatentSequenceHandler.epoch_flow_batches` / `draw_latent_batch` (`epoch_grouped_flow_batches` in stage 2) | one random window per flow per epoch, so an epoch has one step per `batch_size` flows |
| Evaluation / export | `CachedLatentSequenceHandler.enumerate_windows`; `SequenceEmbeddingAD.py::enumerate_windows` copies the same rule | non-overlapping 64-packet windows. The remainder is dropped, and a flow shorter than 64 packets gives one short window |
| Batch tensor | `latent_batch_from_windows` / `draw_latent_batch` | `(B, P, D)` latents with real packets at the front and exact zeros in the padding slots, plus `seq_lens` `(B,)` |
| Padding mask | `ModelDefinitions.py::build_padding_mask(seq_lens, P)` | `True` at real packets. Use this and do not rebuild the mask elsewhere |

`seq_len` (1..64) is recorded for each exported window. Short flows cluster strongly in
embedding space, which is why the AD stage stratifies its results by length
([07 § Length confound](../components/07-anomaly-detection.md#length-confound)).

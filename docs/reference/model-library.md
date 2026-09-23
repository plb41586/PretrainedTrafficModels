# Model library (`RawByteTrafficModelling/ModelComponents/`)

This is the reusable part of the code: the training scripts build their models from it,
and other downstream tasks can too. Stage docs say which of these classes they train;
this file says what each class is.

| File | Contents |
|---|---|
| `BackBones.py` | `MambaBackbone`, `TransformerBackbone` and their `*BackboneParams` dataclasses |
| `ModelDefinitions.py` | backbone factory, params dataclasses, `load_*`/`save_checkpoint`, every model class, eval helpers |
| `DataUtils.py` | `ID_Encoder`, dataset handlers, latent-cache loader, flow-key parsing and grouping |
| `StructureLosses.py` | `supcon` and embedding-geometry diagnostics (stage 2 fine-tune) |
| `FlowID.py` | older address-hash `FlowID`, superseded by `flow_key` and unused |

## Backbones

Every backbone maps `(B, L, dim) → (B, L, dim)`, which is why each model level can use
either one.

- `ModelDefinitions.py::BACKBONES` maps a name to a `(ParamsClass, ModuleClass)` pair, and
  `build_backbone(kind, params)` constructs from it.
- `unpack_backbone_params(kind, dict)` rebuilds the params dataclass from a checkpoint dict.
- **To add a backbone**, add its params dataclass in `BackBones.py`, register it in
  `BACKBONES`, and extend `unpack_backbone_params`. If the two registrations drift apart,
  loading a checkpoint that uses the new backbone fails.
- `TransformerBackbone` takes `causal=True` in `forward`. At packet level `max_len` must
  be at least 1520 (`RunConfig.packet_encoder_params` sets this).

## Params and checkpoints

Each model has a params dataclass. It nests the params of its sub-models, and those are
stored as plain dicts once saved to disk:

```
SeqAutoEncoderParams
├── SeqEncParams: SeqEncoderParams
│   ├── EncoderParams: EncoderParams ── BackboneParams
│   └── SeqBackboneParams
└── SeqDecBackbone
AutoEncoderParams ── ENC_Params: EncoderParams, DecBackbone
MLM_Params        ── EncoderParams
```

After `torch.load`, always use the matching loader: `load_AE_Checkpoint`,
`load_MLM_checkpoint` or `load_SeqAE_checkpoint` (these call the `unpack_*_params`
functions). Do not construct a dataclass directly from `ckpt["config"]`, because its
nested members would stay dicts. See [artefacts § Checkpoints](artefacts.md#checkpoints-stages-3-5--later-stages)
for the file format.

## Packet level

| Class | Role | Trained by |
|---|---|---|
| `Packet_Encoder` | embedding → backbone → `DynamicCLSPooling` → one `EncoderDim` latent per packet | inside `PacketAutoencoder` |
| `AutoregressiveDecoder` | latent projected into position 0, then teacher-forced byte decoding (`forward`) or sampling (`generate`) | inside `PacketAutoencoder` |
| `PacketAutoencoder` | encoder + decoder. `forward(tokens) → (logits, latent)`; `reconstruct` | `PacketLevelAutoEncoder.py` |
| `Packet_MLM` | its own embedding and backbone, a per-token head and a CLS proto-hierarchy head | `PacketLevelMLM.py` |
| `Packet_Classifier` | `Packet_Encoder` + a linear head, for supervised packet tasks | not used by any current script |

`Packet_MLM` does not wrap a `Packet_Encoder`, so its weights cannot be loaded into the
sequence level as they are, even though the shapes match.

## Sequence level

| Class | Role |
|---|---|
| `Sequence_Encoder` | Uses a (normally frozen) `Packet_Encoder` as the "embedding layer" for whole packets. It projects the latents to `SeqEncoderDim`, writes a learned CLS at index `seq_len` (right after the last real packet, so this also works with a causal backbone), runs the sequence backbone and gathers the CLS output as `z`. It takes either `tokens (B,P,1520)` or cached `latents (B,P,D)` |
| `SequenceDecoder` | Rebuilds all P packet latents from `z` in parallel: `[proj(z) | learned position queries]` → backbone → linear head |
| `SequenceAutoencoder` | encoder + decoder + optional length head (`P+1` classes). It normalises targets with stored `target_mean`/`target_std` (`set_target_stats`), and its loss is padding-masked MSE plus `length_loss_weight` × length CE. `encode()` returns `z` only |
| `SequenceClassifier` | older self-contained Mamba classifier that does not use the factory. Kept for comparison only |

`freeze_packet_encoder()` / `unfreeze_packet_encoder()` switch the packet encoder
between frozen and trainable. Its `train()` override keeps a frozen encoder in eval mode.

## Evaluation helpers (`ModelDefinitions.py`, all `@torch.no_grad`)

| Function | Measures |
|---|---|
| `baseline_mses` | the MSE of predicting the global mean, or the per-position mean, in normalised latent space. A model has learned something only if it beats these |
| `decoder_byte_accuracy` | teacher-forced byte accuracy of a frozen packet decoder on a set of latents, reported as `all` and `nonpad` |
| `byte_level_reconstruction` | sequence AE → denormalised latents → packet decoder → byte accuracy. This puts the sequence level on the same scale as the packet AE's own accuracy (the ceiling) |
| `precompute_latents`, `compute_target_stats` | older helpers from before the latent cache existed |

There was also a nearest-neighbour `retrieval_accuracy` metric, which asked whether a
reconstructed latent's nearest neighbour among the batch's true latents was its own
target. It was removed on purpose. Near-duplicate packets are everywhere in this
traffic, and a validation batch has about 9.6k candidates, so the metric stayed near
chance while reconstruction MSE improved 16x. It measured how dense the target space
is, not how good the model is. Byte accuracy against the packet-AE ceiling replaced it.

## Data handlers (`DataUtils.py`)

| Symbol | Status |
|---|---|
| `ID_Encoder` | in use. See [data-representation](data-representation.md) |
| `PreTrainingDatasetHandler` | in use. Handles row-level packet batches (`sample_epoch_packet_indices`, `get_pretraining_data`) and the flow index (`build_flow_index`) |
| `CachedLatentSequenceHandler` | in use. Builds windowed batches from a latent cache |
| `load_latent_cache`, `checkpoint_fingerprint` | in use. Load a cache and check its provenance |
| `parse_flow_key`, `flow_group_labels`, `flow_group_ids`, `group_support` | in use (stage 2 fine-tune, `InspectFlowGroups.py`) |
| `TrainingDatasetHandler`, `ValidationDatasetHandler` | dead code: they need `AttackLabel`/`FlowID` columns that no current parquet has |

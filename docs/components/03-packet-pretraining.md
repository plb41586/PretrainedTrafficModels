# 03 · Packet-level pretraining (packets → packet encoder)

**What it does:** trains a `Packet_Encoder` that compresses one packet (1520 tokens) into
a single D-dimensional latent vector. The sequence level uses this encoder, frozen, as
its per-packet embedding, so its reconstruction quality sets the upper limit for
everything after it.

**Entry points** (in `RawByteTrafficModelling/PreTraining/`):

| Script | Objective | Used downstream? |
|---|---|---|
| `PacketLevelAutoEncoder.py` | autoregressive byte reconstruction from the latent | **yes**: its `_best` checkpoint feeds stages 4–6 |
| `PacketLevelMLM.py` | masked-token reconstruction + proto-hierarchy classification on CLS | not currently (see [Reuse](#reuse)) |

**Consumes:** `DATASET.train` and `DATASET.test` packet parquets. Rows are sampled
individually, and flows play no role at this level.
**Produces:** `TrainingOutputs/<RUN_NAME>/…{_untrained,_latest,_E<n>,_best}.ckpt`, plus
`metrics.csv` and `curves.png` ([artefacts § Checkpoints](../reference/artefacts.md#checkpoints-stages-3-5--later-stages)).

## Where the logic lives

| Concern | File · symbol |
|---|---|
| Encoder spec (shared by AE and MLM) | `RunConfig.py::packet_encoder_params` |
| Tokenisation and masking | see [data-representation](../reference/data-representation.md) |
| Batching | `DataUtils.py::PreTrainingDatasetHandler.sample_epoch_packet_indices`, `get_pretraining_data` |
| Models | `ModelDefinitions.py::Packet_Encoder`, `AutoregressiveDecoder`, `PacketAutoencoder`, `Packet_MLM` ([model-library](../reference/model-library.md#packet-level)) |
| AE loss and metrics | `PacketLevelAutoEncoder.py::reconstruction_losses`, `evaluate` |
| MLM masking | `keras_hub.layers.MaskedLMMaskGenerator`, configured by `MASK_*` constants in `PacketLevelMLM.py` |

## How it works (autoencoder)

- Pipeline: `Packet_Encoder` → latent projected into decoder position 0 → Mamba decoder,
  teacher-forced, predicting all 1520 positions.
- Loss: per-token cross-entropy, with the `<pad>` class weighted by `PAD_LOSS_WEIGHT`.
  Most positions are padding, and without the weight the loss is dominated by a token the
  model learns in the first few hundred steps. The unweighted CE is also logged, so runs
  with different weights can be compared.
- Metrics: accuracy over all positions and **non-pad** accuracy. Only non-pad accuracy is
  informative, because the all-position number is high just from predicting padding.
- Evaluation: a fixed random subset of `test` (`fixed_eval_batches`), identical at every
  eval, so the curve is not noisy from resampling.
- Selection: `_best` = lowest **unweighted** test CE at epoch end.
- Every eval interval writes a `_latest` checkpoint, so a multi-hour epoch can be resumed
  (`RESUME_FROM`).

## Configuration

`RUN_NAME`, `DATASET`, `DEVICE_INDEX`, `ENCODER_DIM`, `ENCODER_LAYERS`/`DECODER_LAYERS`,
the optimisation constants and `SMOKE`. The AE and the MLM can run at the same time, one
per GPU ([conventions](../reference/conventions.md#running-anything)).

## Reuse

- **Packet embeddings for any task:** load with `load_AE_Checkpoint`, build a
  `PacketAutoencoder`, and use `.encoder`.
- **Supervised packet classification:** `Packet_Classifier` wraps a `Packet_Encoder` with
  a linear head (`PacketClassifierParams`). No current script trains it.
- **MLM encoder at sequence level:** `Packet_MLM` builds its own embedding and backbone
  instead of wrapping `Packet_Encoder`. Its state-dict keys therefore need remapping
  before the sequence level can load it, even though the shapes match (both come from
  `packet_encoder_params`).
- **Packet-level AD (legacy):** `AnomalyDetection/AutoEncoderAD.py` (per-packet
  reconstruction loss) and `EncoderEmbeddingAD.py` (packet embedding export). Both are
  older scripts; see [07 § Other scripts](07-anomaly-detection.md#other-scripts).
